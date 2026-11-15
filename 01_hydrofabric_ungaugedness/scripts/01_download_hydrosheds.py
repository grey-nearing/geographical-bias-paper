"""Download raw public HydroSHEDS datasets (BasinATLAS v10 L1-L12 + HydroRIVERS v10).

Public upstream sources:
  - BasinATLAS v10 GDB (all L1-L12 + WaterGAP v2.2 runoff/discharge/climate):
    https://ndownloader.figshare.com/files/20082137
  - HydroRIVERS v10 Global Shapefile (8.5M reaches + WaterGAP discharge):
    https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_shp.zip
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.benchmark_config import RAW_DIR

OUT_DIR = RAW_DIR / "hydrosheds"

BASINATLAS_URL = "https://ndownloader.figshare.com/files/20082137"
HYDRORIVERS_URL = "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_shp.zip"


def _download_http(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"Downloading {url} -> {dst}", flush=True)
    subprocess.run(["curl", "-fL", "--retry", "5", "--retry-delay", "5", "-o", str(tmp), url], check=True)
    tmp.rename(dst)


def stage_basinatlas() -> Path:
    gdb_dir = OUT_DIR / "BasinATLAS_v10.gdb"
    if gdb_dir.exists() and any(gdb_dir.glob("*.gdbtable")):
        print(f"[skip] BasinATLAS_v10.gdb already present at {gdb_dir}", flush=True)
        return gdb_dir

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = OUT_DIR / "BasinATLAS_Data_v10.gdb.zip"
    if not zip_path.exists():
        _download_http(BASINATLAS_URL, zip_path)
    print(f"Extracting {zip_path}...", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(OUT_DIR)
    for p in OUT_DIR.rglob("BasinATLAS_v10.gdb"):
        if p != gdb_dir:
            shutil.move(str(p), str(gdb_dir))
            break
    return gdb_dir


def stage_hydrorivers() -> Path:
    shp_dir = OUT_DIR / "HydroRIVERS_v10_shp"
    if shp_dir.exists() and any(shp_dir.glob("*.shp")):
        print(f"[skip] HydroRIVERS_v10_shp already present at {shp_dir}", flush=True)
        return shp_dir

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = OUT_DIR / "HydroRIVERS_v10_shp.zip"
    if not zip_path.exists():
        _download_http(HYDRORIVERS_URL, zip_path)

    print(f"Extracting {zip_path}...", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(OUT_DIR)
    return shp_dir


if __name__ == "__main__":
    g = stage_basinatlas()
    r = stage_hydrorivers()
    print(f"Done HydroSHEDS download:\n  BasinATLAS: {g}\n  HydroRIVERS: {r}", flush=True)
