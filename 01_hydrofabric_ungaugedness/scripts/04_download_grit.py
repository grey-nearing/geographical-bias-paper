"""Download GRIT v1.0 (30m FABDEM multi-threaded river network & component catchments).

Public Zenodo API source (Wortmann et al., 2025, Water Resources Research; v1.0 release):
  https://zenodo.org/api/records/17435232 (DOI: 10.5281/zenodo.17435232)

Downloads all 7 regional segment networks (AF, AS, EU, NA, SA, SI, SP) and global catchments in EPSG:4326:
  - GRITv1.0_segments_{REG}_EPSG4326.gpkg.zip               (~2.83 GB total)
  - GRITv1.0_component_catchments_GLOBAL_EPSG4326.gpkg.zip  (808.1 MB)
  - GRITv1.0_domain_GLOBAL_EPSG4326.gpkg.zip                (18.2 MB)

Note: Flow-related metrics are skipped on GRIT v1.0 by design (no operational
global flow model runs on GRIT).
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.benchmark_config import RAW_DIR

OUT_DIR = RAW_DIR / "grit_v10"

ZENODO_RECORD_API = "https://zenodo.org/api/records/17435232"
REGIONS = ["AF", "AS", "EU", "NA", "SA", "SI", "SP"]

TARGET_FILES = (
    [f"GRITv1.0_segments_{r}_EPSG4326.gpkg.zip" for r in REGIONS]
    + [
        "GRITv1.0_component_catchments_GLOBAL_EPSG4326.gpkg.zip",
        "GRITv1.0_domain_GLOBAL_EPSG4326.gpkg.zip",
    ]
)


def download_grit() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(ZENODO_RECORD_API, headers={"User-Agent": "python-urllib/3.13"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        meta = json.loads(resp.read().decode("utf-8"))

    file_map = {f["key"]: f for f in meta["files"]}

    for key in TARGET_FILES:
        info = file_map[key]
        expected_size = int(info["size"])
        url = info["links"]["self"]
        dst = OUT_DIR / key

        if dst.exists() and dst.stat().st_size == expected_size:
            print(f"[skip] {key} ({expected_size / 1e6:.1f} MB)", flush=True)
        else:
            tmp = dst.with_suffix(dst.suffix + ".part")
            print(f"Downloading {key} ({expected_size / 1e6:.1f} MB)...", flush=True)
            for attempt in range(4):
                try:
                    r = urllib.request.Request(url, headers={"User-Agent": "python-urllib/3.13"})
                    with urllib.request.urlopen(r, timeout=300) as resp, open(tmp, "wb") as f:
                        while True:
                            chunk = resp.read(1024 * 1024)
                            if not chunk:
                                break
                            f.write(chunk)
                    if tmp.stat().st_size == expected_size:
                        tmp.rename(dst)
                        break
                    else:
                        print(f"  size mismatch {tmp.stat().st_size} != {expected_size}, retrying...", flush=True)
                except Exception as e:
                    print(f"  attempt {attempt+1} failed: {e}", flush=True)
                    time.sleep(3 * (attempt + 1))
            else:
                raise RuntimeError(f"Failed to download {key}")
            time.sleep(1.0)

        gpkg_name = key.replace(".zip", "")
        gpkg_path = OUT_DIR / gpkg_name
        if not gpkg_path.exists():
            print(f"  extracting {key}...", flush=True)
            with zipfile.ZipFile(dst, "r") as zf:
                zf.extractall(OUT_DIR)

    print("Done GRIT v1.0 download and extraction!", flush=True)


if __name__ == "__main__":
    download_grit()
