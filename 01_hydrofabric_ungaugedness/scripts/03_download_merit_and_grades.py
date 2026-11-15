"""Download public MERIT-Basins v0.1 (all 9 Pfafstetter L1 regions + coastal hillslopes)
and stage/compute the 1979-2013 35-year mean annual runoff grid from GRADES v0.1.

Official public ReachHydro Google Drive sources (Lin et al., 2019, WRR):
  - MERIT-Basins zip/pfaf_level_01: folder 1qYRqhnBVk7n4UTeJBOGwebso8iu_KOLr
  - MERIT-Basins coastal_hillslopes: folder 1ypWH78DK4p3EonFoTLzW9hakQgFYhUHO
    (manifest of all 300 public files in configs/merit_coastal_hillslopes_drive_ids.json)
  - GRADES v0.1 daily runoff (1979-2013, 0.25 deg, mm/day):
    folder 13BCvmcC4ln2pSW4kILm4r2l7aRugoihQ

If the pre-aggregated 6.7 MB `grades_mean_annual_runoff_1979_2013.nc` from the companion
Zenodo package is present in `$GEO_BIAS_DATA_DIR/flow_models/`, it is copied directly to
avoid downloading ~11.8 GB of daily GRADES NetCDFs.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.benchmark_config import GEO_BIAS_DATA_DIR, RAW_DIR

MERIT_DIR = RAW_DIR / "merit"
FLOW_DIR = RAW_DIR / "flow_models"
HILLSLOPES_MANIFEST = REPO_ROOT / "configs" / "merit_coastal_hillslopes_drive_ids.json"

MERIT_PFAF_ZIPS = [
    ("1GTz5sItFBpMFHeLhBWqwyVlRQgTUeieM", "pfaf_1_MERIT_Hydro_v07_Basins_v01.zip"),
    ("1Ygj2de11ZDs6l6p90l2WBMZKXbd2dCEr", "pfaf_2_MERIT_Hydro_v07_Basins_v01.zip"),
    ("13OLqyc1QtNJs0itJTCfuz3l-AB7aztD_", "pfaf_3_MERIT_Hydro_v07_Basins_v01.zip"),
    ("1HhzYKcQ_2WKObvp2xL4cO1aS-sJ3q8G-", "pfaf_4_MERIT_Hydro_v07_Basins_v01.zip"),
    ("1udPibrTbyUku3ZIQtjPMTMzOwyvdQS2K", "pfaf_5_MERIT_Hydro_v07_Basins_v01.zip"),
    ("1vZKI9V1_kGj9mPD1aoVgN6oCBYlR_bgs", "pfaf_6_MERIT_Hydro_v07_Basins_v01.zip"),
    ("1zaJH8r6dFCJL6TmCya12Oq6XKRlNJHMz", "pfaf_7_MERIT_Hydro_v07_Basins_v01.zip"),
    ("1Lt78g4ouTRc4j28HB9O5YMltE-N0sJdw", "pfaf_8_MERIT_Hydro_v07_Basins_v01.zip"),
    ("1sPcZVKlyABUN41tmCj6LbC0s2-NwVC4P", "pfaf_9_MERIT_Hydro_v07_Basins_v01.zip"),
]

GRADES_RUNOFF_FILES = [
    (1979, "1CmoBDRzn-KuNAshhnJ7DVQ_VAHyDQv1T"),
    (1980, "1ilx07s_aZNxXzSwD_BqAtyCr-lDNYG2-"),
    (1981, "1SEgaKnGujykkuRDia7LVVCwSpkevJM77"),
    (1982, "1NlieB6xaaOTCedmkgzwZU3PbX9ncpTrC"),
    (1983, "18mQDKwEy04KpKm59cPj1o64V1s3phCDU"),
    (1984, "15bETfEzepwnXnrRqh1pJ9_pJKx4IZfa7"),
    (1985, "17Tuq5Yymvhy-MVOb3ISM5ukUVvQlxpu2"),
    (1986, "1fSdK7AqKOxDibfEIQVwd8qZTfIbOt_-U"),
    (1987, "1Uiin59WLZvDFk17Df2OeWbvU0ci2I1j_"),
    (1988, "1X1qtawnCBSsD8HORwcZe-En0HvDfsrou"),
    (1989, "1RNfg5TtJVDDzATp8vcaFTAerJXznnp1u"),
    (1990, "1MSwygnHM7sJj8qbm7VVGBmArnpvrK7hp"),
    (1991, "1oggww8h9-t5bmx_n8tksQHU0eav1Y5mU"),
    (1992, "10vBUMO0gl8H8m2_rISepOfJ8PrdruCUr"),
    (1993, "1327lfaDoD6oW4JPxr23CpiSguVYYgUro"),
    (1994, "12RpPm-EYf9c4tIkUySfLWU827kBCd2sW"),
    (1995, "1y0VT_QnB4RvwVM3Z1-I9TwgKThEE6A6_"),
    (1996, "12hSKwVpe64TpCkRCdxBZlfHvkl-DGwOk"),
    (1997, "1UkWYtNNblN8Dro8ncnTjEBuRZ8oRopIQ"),
    (1998, "1M7HftB6F3SBlHtFRKrJKTTFqVc9aMKQp"),
    (1999, "1BMajhEJbVWZgAK6cLKBOHrPlYUwvncoy"),
    (2000, "1eeBgLHjL9PgLoZSn6ak5A8a0Xzvsrj1J"),
    (2001, "15I0-u3ifErbOUV4BmPSXiRr3nGKL_37r"),
    (2002, "1j34IToEGYWQyFfJkxURXtN_wFxykTkBM"),
    (2003, "1sgCXhCBhue3PCIRoHInCar5PuFiS5ui1"),
    (2004, "1BCRvL9ec6lUHbDze-hugX-SREU8F8ZHU"),
    (2005, "1lUDPXjKbTWmSBoNnE_pAaPVuARft2Mup"),
    (2006, "1uXPCYCV4NUqfwLUSOsRjj08MbtKboU0K"),
    (2007, "19URWmODb9qMJ5gQ3NkzTt4FDJ0vlS2i_"),
    (2008, "1ILu0rOSwWG49SAa6EUJgX1BdQmtBb5E3"),
    (2009, "1fLh5d5N-zux1t0Cg0zkUl42mAd7fBxPx"),
    (2010, "1XNpkzsJSaa2gB7bPv10cI_14TNS6FgsC"),
    (2011, "1zrr0SfpaZKGzN7TqtmTa9FdDcEcgGQrY"),
    (2012, "1NlpOsMUipddTqbXSTiZ8JXerLeQ8WZaO"),
    (2013, "1-w2hl3FS8FTZ5oEDtQUnVCboZTL5O4UC"),
]


def _is_html_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return True
    head = path.read_bytes()[:256].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html")


def _gdrive_download_public(file_id: str, dst: Path, expected_size: int | None = None, use_direct_first: bool = False) -> None:
    """Download a public Google Drive file via direct HTTPS and/or gdown."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and (expected_size is None or dst.stat().st_size == expected_size) and not _is_html_file(dst):
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(4):
        try:
            if use_direct_first and attempt == 0:
                url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
                subprocess.run(
                    ["curl", "-fsSL", "--retry", "3", "-o", str(tmp), url],
                    check=True,
                )
                if not _is_html_file(tmp) and (expected_size is None or tmp.stat().st_size == expected_size):
                    tmp.rename(dst)
                    return
                tmp.unlink(missing_ok=True)

            import gdown

            out = gdown.download(id=file_id, output=str(tmp), quiet=(expected_size is not None and expected_size < 50_000_000))
            if out and tmp.exists() and not _is_html_file(tmp):
                if expected_size is None or tmp.stat().st_size == expected_size:
                    tmp.rename(dst)
                    return
            tmp.unlink(missing_ok=True)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            print(f"  [retry {attempt+1}/4] {dst.name}: {exc}", flush=True)
        time.sleep(2 * (attempt + 1))

    raise RuntimeError(
        f"Failed to download Google Drive file {file_id} -> {dst}. "
        "Note: ReachHydro's public Google Drive folder enforces a 24-hour anonymous IP download quota. "
        "If rate-limited, download MERIT-Basins v0.1 from ReachHydro's official Globus mirror "
        "(https://www.reachhydro.org/home/params/merit-basins, Globus endpoint "
        "origin_id=f349971a-b759-4e38-8bc4-f6ccb057c499) and point MERIT_BASINS_LOCAL_DIR "
        "to the directory containing pfaf_*_MERIT_Hydro_v07_Basins_v01.zip (or pfaf_level_01/) "
        "and coastal_hillslopes/."
    )


def _stage_from_local_mirror() -> None:
    """Optionally stage pre-downloaded raw ReachHydro MERIT-Basins files from MERIT_BASINS_LOCAL_DIR."""
    local_dir_env = os.environ.get("MERIT_BASINS_LOCAL_DIR", "").strip()
    if not local_dir_env:
        return
    src_root = Path(local_dir_env)
    if not src_root.exists():
        raise FileNotFoundError(f"MERIT_BASINS_LOCAL_DIR does not exist: {src_root}")
    print(f"[local-mirror] Staging raw ReachHydro MERIT-Basins from {src_root} -> {MERIT_DIR}", flush=True)
    MERIT_DIR.mkdir(parents=True, exist_ok=True)
    extract_dir = MERIT_DIR / "pfaf_level_01"
    extract_dir.mkdir(parents=True, exist_ok=True)
    hs_dir = MERIT_DIR / "coastal_hillslopes"
    hs_dir.mkdir(parents=True, exist_ok=True)

    # Link/copy any pfaf zip files
    for _, name in MERIT_PFAF_ZIPS:
        src_z = src_root / name
        dst_z = MERIT_DIR / name
        if src_z.exists() and not dst_z.exists():
            os.symlink(src_z.resolve(), dst_z)

    # Link/copy any pre-extracted pfaf_level_01 shapefiles
    src_pfaf = src_root / "pfaf_level_01"
    if src_pfaf.is_dir():
        for f in src_pfaf.iterdir():
            dst_f = extract_dir / f.name
            if f.is_file() and not dst_f.exists():
                os.symlink(f.resolve(), dst_f)

    # Link/copy coastal_hillslopes files
    src_hs = src_root / "coastal_hillslopes"
    if src_hs.is_dir():
        for f in src_hs.iterdir():
            dst_f = hs_dir / f.name
            if f.is_file() and not dst_f.exists():
                os.symlink(f.resolve(), dst_f)


def download_merit_basins() -> None:
    MERIT_DIR.mkdir(parents=True, exist_ok=True)
    _stage_from_local_mirror()
    extract_dir = MERIT_DIR / "pfaf_level_01"
    extract_dir.mkdir(parents=True, exist_ok=True)

    for fid, name in MERIT_PFAF_ZIPS:
        pfaf_num = name.split("_")[1]
        riv_hits = list(extract_dir.glob(f"riv_pfaf_{pfaf_num}_*.shp"))
        cat_hits = list(extract_dir.glob(f"cat_pfaf_{pfaf_num}_*.shp"))
        if riv_hits and cat_hits:
            print(f"[skip] MERIT pfaf_{pfaf_num} already extracted", flush=True)
            continue

        zpath = MERIT_DIR / name
        if not zpath.exists():
            print(f"Downloading {name} from public ReachHydro Drive...", flush=True)
            _gdrive_download_public(fid, zpath, use_direct_first=False)

        print(f"Extracting {name} -> {extract_dir}...", flush=True)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(extract_dir)
        for sub in list(extract_dir.rglob("*.shp")):
            if sub.parent != extract_dir:
                for ext in [".shp", ".dbf", ".shx", ".prj", ".cpg"]:
                    cand = sub.with_suffix(ext)
                    if cand.exists():
                        shutil.move(str(cand), str(extract_dir / cand.name))

    # Download all 300 public coastal hillslope shapefile components
    hs_dir = MERIT_DIR / "coastal_hillslopes"
    hs_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(HILLSLOPES_MANIFEST.read_text())
    missing = [
        item for item in manifest
        if not ((hs_dir / item["name"]).exists() and (hs_dir / item["name"]).stat().st_size == item["size"])
    ]
    if not missing:
        print(f"[skip] coastal_hillslopes already complete ({len(list(hs_dir.glob('*.shp')))} shps)", flush=True)
    else:
        print(f"Downloading {len(missing)}/{len(manifest)} MERIT coastal_hillslopes files...", flush=True)

        def _fetch_hs(item: dict) -> str:
            _gdrive_download_public(
                item["id"],
                hs_dir / item["name"],
                expected_size=int(item["size"]),
                use_direct_first=True,
            )
            return item["name"]

        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            for _ in ex.map(_fetch_hs, missing):
                done += 1
                if done % 50 == 0 or done == len(missing):
                    print(f"  coastal_hillslopes: {done}/{len(missing)} downloaded", flush=True)

    print(
        f"MERIT-Basins ready: {len(list(extract_dir.glob('riv_*.shp')))} river shps, "
        f"{len(list(extract_dir.glob('cat_*.shp')))} cat shps, "
        f"{len(list(hs_dir.glob('*.shp')))} hillslope shps",
        flush=True,
    )


def compute_grades_mean_runoff() -> Path:
    """Stage or compute exact 1979-2013 mean annual runoff (mm/day) on 0.25-deg global grid."""
    FLOW_DIR.mkdir(parents=True, exist_ok=True)
    out_nc = FLOW_DIR / "grades_mean_annual_runoff_1979_2013.nc"
    if out_nc.exists() and out_nc.stat().st_size > 1_000_000:
        print(f"[skip] {out_nc.name} already exists", flush=True)
        return out_nc

    zenodo_preagg = GEO_BIAS_DATA_DIR / "flow_models" / "grades_mean_annual_runoff_1979_2013.nc"
    if zenodo_preagg.exists() and zenodo_preagg.stat().st_size > 1_000_000:
        print(f"Copying pre-aggregated GRADES mean runoff from {zenodo_preagg} -> {out_nc}", flush=True)
        shutil.copy2(zenodo_preagg, out_nc)
        return out_nc

    sum_grid = None
    cnt_grid = None
    lats = None
    lons = None

    for yr, fid in GRADES_RUNOFF_FILES:
        nc_path = FLOW_DIR / f"grades_runoff_{yr}.nc"
        if not nc_path.exists():
            print(f"Downloading GRADES runoff_{yr}.nc...", flush=True)
            _gdrive_download_public(fid, nc_path, use_direct_first=False)
        else:
            print(f"Using existing {nc_path.name}...", flush=True)

        with xr.open_dataset(nc_path) as ds:
            if lats is None:
                lats = ds["lat"].values.copy()
                lons = ds["lon"].values.copy()
                sum_grid = np.zeros((lats.size, lons.size), dtype=np.float64)
                cnt_grid = np.zeros((lats.size, lons.size), dtype=np.int64)
            arr = ds["runoff"].values
            valid = np.isfinite(arr) & (arr >= 0.0)
            sum_grid += np.where(valid, arr.astype(np.float64), 0.0).sum(axis=0)
            cnt_grid += valid.sum(axis=0)

        nc_path.unlink(missing_ok=True)
        print(f"  accumulated {yr} (valid grid cells: {(cnt_grid > 0).sum():,})", flush=True)

    mean_mm_day = np.where(cnt_grid > 0, sum_grid / np.maximum(cnt_grid, 1), np.nan).astype(np.float32)
    out_ds = xr.Dataset(
        {
            "runoff_mean_mm_day": (("lat", "lon"), mean_mm_day, {
                "units": "mm/day",
                "long_name": "GRADES v0.1 1979-2013 mean daily total runoff",
            }),
            "valid_days": (("lat", "lon"), cnt_grid.astype(np.int32)),
        },
        coords={"lat": lats, "lon": lons},
    )
    out_ds.to_netcdf(out_nc)
    print(f"Saved {out_nc} (global land mean runoff = {np.nanmean(mean_mm_day):.3f} mm/day)", flush=True)
    return out_nc


if __name__ == "__main__":
    download_merit_basins()
    compute_grades_mean_runoff()
    print("Done MERIT-Basins + GRADES download!", flush=True)
