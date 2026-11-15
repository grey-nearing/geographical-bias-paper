"""Download TDX-Hydro / GEOGLOWS v2 tables, global streams, and 1940-2024 mean annual discharge.

All files are 100% public anonymous HTTPS downloads from AWS S3:
  https://geoglows-v2.s3.us-west-2.amazonaws.com/

Outputs in data/raw/tdx_geoglows/:
  - v2-model-table.parquet        (6.84M TDX-Hydro reaches: topology, lengths, areas, VPUs)
  - v2-countries-table.parquet    (reach -> country mapping)
  - vpu-boundaries.gpkg           (VPU boundary polygons)
  - global_streams_simplified.gpkg (global 6.84M stream geometries for snapping)
  - geoglows_v2_mean_annual_q.parquet (exact 1940-2024 85-yr mean annual Q for all 6,838,900 reaches)
"""

from __future__ import annotations

import concurrent.futures
import json
import struct
import subprocess
import sys
import urllib.request
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.benchmark_config import RAW_DIR

OUT_DIR = RAW_DIR / "tdx_geoglows"
S3_BASE = "https://geoglows-v2.s3.us-west-2.amazonaws.com"


def download_file(rel_path: str, dst_name: str | None = None) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUT_DIR / (dst_name or Path(rel_path).name)
    if dst.exists() and dst.stat().st_size > 1024:
        print(f"[skip] {dst.name} already exists ({dst.stat().st_size / 1e6:.1f} MB)", flush=True)
        return dst
    tmp = dst.with_suffix(dst.suffix + ".part")
    url = f"{S3_BASE}/{rel_path}"
    print(f"Downloading {url} -> {dst}", flush=True)
    subprocess.run(["curl", "-fL", "--retry", "3", "-o", str(tmp), url], check=True)
    tmp.rename(dst)
    print(f"  saved {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)", flush=True)
    return dst


def _fetch_bytes(url: str, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "global-ungaugedness/0.1"})
            with urllib.request.urlopen(req, timeout=45) as resp:
                return resp.read()
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Failed to fetch {url}: {e}") from e


def _decode_zarr_chunk(raw: bytes, zarray: dict) -> np.ndarray:
    """Decompress a single Zarr v2 chunk (handles blosc/zlib/gzip/raw)."""
    comp = zarray.get("compressor")
    dtype = np.dtype(zarray["dtype"])
    if comp is None:
        buf = raw
    elif comp["id"] == "zlib":
        buf = zlib.decompress(raw)
    elif comp["id"] == "gzip":
        import gzip
        buf = gzip.decompress(raw)
    else:
        import numcodecs
        buf = numcodecs.get_codec(comp).decode(raw)
    arr = np.frombuffer(buf, dtype=dtype)
    return arr


def download_geoglows_mean_q(max_workers: int = 12) -> Path:
    """Stream all 1,368 chunks of retrospective/yearly-timeseries.zarr and compute 1940-2024 mean Q."""
    out_pq = OUT_DIR / "geoglows_v2_mean_annual_q.parquet"
    if out_pq.exists() and out_pq.stat().st_size > 10_000_000:
        print(f"[skip] {out_pq.name} already exists", flush=True)
        return out_pq

    zarr_base = f"{S3_BASE}/retrospective/yearly-timeseries.zarr"
    q_meta = json.loads(_fetch_bytes(f"{zarr_base}/Q/.zarray"))
    rid_meta = json.loads(_fetch_bytes(f"{zarr_base}/river_id/.zarray"))
    print("GEOGLOWS yearly-timeseries Q metadata:", q_meta, flush=True)
    print("GEOGLOWS river_id metadata:", rid_meta, flush=True)

    n_years, n_rivers = q_meta["shape"]
    c_years, c_rivers = q_meta["chunks"]
    assert c_years == n_years, f"Expected single time chunk, got {c_years} vs {n_years}"
    n_q_chunks = (n_rivers + c_rivers - 1) // c_rivers

    # Fetch river_id array
    rid_c = rid_meta["chunks"][0]
    n_rid_chunks = (n_rivers + rid_c - 1) // rid_c
    river_ids = np.empty(n_rivers, dtype=np.int64)

    def fetch_rid_chunk(ci: int) -> tuple[int, np.ndarray]:
        raw = _fetch_bytes(f"{zarr_base}/river_id/{ci}")
        arr = _decode_zarr_chunk(raw, rid_meta)
        return ci, arr

    print(f"Fetching {n_rid_chunks} river_id chunks...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for ci, arr in ex.map(fetch_rid_chunk, range(n_rid_chunks)):
            s = ci * rid_c
            e = min(s + rid_c, n_rivers)
            river_ids[s:e] = arr[: e - s]

    # Fetch Q chunks and immediately compute 85-year mean & median per river_id
    q_mean = np.empty(n_rivers, dtype=np.float32)
    q_median = np.empty(n_rivers, dtype=np.float32)
    q_nvalid = np.empty(n_rivers, dtype=np.int16)

    def fetch_q_chunk(ci: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        raw = _fetch_bytes(f"{zarr_base}/Q/0.{ci}")
        arr = _decode_zarr_chunk(raw, q_meta)
        s = ci * c_rivers
        e = min(s + c_rivers, n_rivers)
        width = e - s
        # Zarr stores full chunk padded or exact edge chunk
        order = q_meta.get("order", "C")
        if arr.size == c_years * c_rivers:
            mat = arr.reshape((c_years, c_rivers), order=order)[:, :width]
        else:
            mat = arr.reshape((c_years, width), order=order)
        mat64 = mat.astype(np.float64)
        valid = np.isfinite(mat64) & (mat64 >= 0.0)
        nval = np.sum(valid, axis=0).astype(np.int16)
        mat_nan = np.where(valid, mat64, np.nan)
        with np.errstate(all="ignore"):
            mn = np.nanmean(mat_nan, axis=0).astype(np.float32)
            md = np.nanmedian(mat_nan, axis=0).astype(np.float32)
        return ci, mn, md, nval

    print(f"Streaming {n_q_chunks} Q chunks (shape {n_years} x {n_rivers:,})...", flush=True)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for ci, mn, md, nval in ex.map(fetch_q_chunk, range(n_q_chunks)):
            s = ci * c_rivers
            e = min(s + c_rivers, n_rivers)
            q_mean[s:e] = mn
            q_median[s:e] = md
            q_nvalid[s:e] = nval
            done += 1
            if done % 200 == 0 or done == n_q_chunks:
                print(f"  {done}/{n_q_chunks} chunks done ({100.0 * done / n_q_chunks:.1f}%)", flush=True)

    df = pd.DataFrame({
        "LINKNO": river_ids,
        "q_mean_m3s": q_mean,
        "q_median_m3s": q_median,
        "n_valid_years": q_nvalid,
    })
    df.to_parquet(out_pq, index=False)
    print(f"Wrote {out_pq} ({len(df):,} reaches, median Q={np.nanmedian(q_mean):.3f} m3/s)", flush=True)
    return out_pq


if __name__ == "__main__":
    download_file("tables/v2-model-table.parquet")
    download_file("tables/v2-countries-table.parquet")
    download_file("hydrography-global/vpu-boundaries.gpkg")
    download_file("hydrography-global/global_streams_simplified.gpkg")
    download_geoglows_mean_q(max_workers=10)
    print("Done TDX-Hydro / GEOGLOWS v2 download!", flush=True)
