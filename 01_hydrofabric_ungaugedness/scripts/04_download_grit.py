"""Download GRIT v0.6 (30m FABDEM multi-threaded river network & component catchments).

Public Zenodo API source (Wortmann et al., 2025, Water Resources Research):
  https://zenodo.org/api/records/11219313

Downloads all 7 regions (AF, AS, EU, NA, SA, SI, SP) in EPSG:4326:
  - GRITv06_segments_{REG}_EPSG4326.gpkg.zip            (~2.80 GB total)
  - GRITv06_component_catchments_{REG}_EPSG4326.gpkg.zip (~2.92 GB total)
  - GRITv06_domain_GLOBAL_EPSG4326.gpkg.zip             (17.9 MB)
  - GRITv06_sinks_GLOBAL_EPSG4326.gpkg.zip              (55.1 MB)

Note: Flow-related metrics are skipped on GRIT v0.6 by design (no operational
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

OUT_DIR = RAW_DIR / "grit"

ZENODO_RECORD_API = "https://zenodo.org/api/records/11219313"
REGIONS = ["AF", "AS", "EU", "NA", "SA", "SI", "SP"]

TARGET_FILES = (
    [f"GRITv06_segments_{r}_EPSG4326.gpkg.zip" for r in REGIONS]
    + [f"GRITv06_component_catchments_{r}_EPSG4326.gpkg.zip" for r in REGIONS]
    + [
        "GRITv06_domain_GLOBAL_EPSG4326.gpkg.zip",
        "GRITv06_sinks_GLOBAL_EPSG4326.gpkg.zip",
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
                    with urllib.request.urlopen(r, timeout=180) as resp, open(tmp, "wb") as f:
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
            # Polite delay for Zenodo API
            time.sleep(1.0)

        # Extract gpkg if not yet extracted
        gpkg_name = key.replace(".zip", "")
        gpkg_path = OUT_DIR / gpkg_name
        if not gpkg_path.exists():
            print(f"  extracting {key}...", flush=True)
            with zipfile.ZipFile(dst, "r") as zf:
                zf.extractall(OUT_DIR)

    print("Done GRIT v0.6 download and extraction!", flush=True)


if __name__ == "__main__":
    download_grit()
