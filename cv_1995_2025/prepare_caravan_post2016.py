"""Scan 17,011 public Caravan MultiMet basins, filter for post-2016 data, and build Zarr stores."""

import concurrent.futures
import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = Path("./data")
CARAVAN_NC_DIR = BASE_DIR / "data" / "Caravan-nc"
SPLITS_DIR = BASE_DIR / "splits"


def _scan_one_basin(basin_id: str) -> dict:
  sub = basin_id.split("_")[0]
  nc_path = CARAVAN_NC_DIR / "timeseries" / "netcdf" / sub / f"{basin_id}.nc"
  if not nc_path.exists():
    return {"gauge_id": basin_id, "subdataset": sub, "exists": False}
  try:
    with xr.open_dataset(nc_path) as ds:
      if "streamflow" not in ds:
        return {"gauge_id": basin_id, "subdataset": sub, "exists": False}
      q = ds["streamflow"].load()
      valid_mask = ~np.isnan(q.values)
      total_valid = int(valid_mask.sum())
      if total_valid == 0:
        return {
            "gauge_id": basin_id,
            "subdataset": sub,
            "exists": True,
            "record_length_total": 0,
            "n_valid_post2016": 0,
            "min_date": None,
            "max_date": None,
        }
      dates = pd.to_datetime(q.date.values)
      valid_dates = dates[valid_mask]
      post2016_mask = valid_mask & (dates >= pd.Timestamp("2016-01-01"))
      n_post2016 = int(post2016_mask.sum())
      res = {
          "gauge_id": basin_id,
          "subdataset": sub,
          "exists": True,
          "record_length_total": total_valid,
          "n_valid_post2016": n_post2016,
          "min_date": str(valid_dates[0])[:10],
          "max_date": str(valid_dates[-1])[:10],
      }
      for yr in range(2016, 2025):
        yr_mask = (
            valid_mask
            & (dates >= pd.Timestamp(f"{yr}-01-01"))
            & (dates <= pd.Timestamp(f"{yr}-12-31"))
        )
        res[f"days_{yr}"] = int(yr_mask.sum())
      return res
  except Exception as e:
    return {
        "gauge_id": basin_id,
        "subdataset": sub,
        "exists": False,
        "error": str(e),
    }


def main():
  all_list_path = SPLITS_DIR / "all-caravan-multimet-basins.txt"
  with open(all_list_path) as f:
    raw_basins = [line.strip() for line in f if line.strip()]

  public_basins = [
      b for b in raw_basins if not b.lower().startswith("grdc_")
  ]
  print(f"Total public MultiMet basins to scan: {len(public_basins)}")

  with concurrent.futures.ProcessPoolExecutor(max_workers=16) as ex:
    rows = list(ex.map(_scan_one_basin, public_basins, chunksize=64))

  df = pd.DataFrame(rows)
  out_csv = BASE_DIR / "data" / "caravan_basin_streamflow_coverage.csv"
  df.to_csv(out_csv, index=False)
  print(f"Saved coverage report to {out_csv}")

  print("\n=== Summary by Subdataset ===")
  for sub, g in df.groupby("subdataset"):
    n_tot = len(g)
    n_16_any = int((g["n_valid_post2016"] > 0).sum())
    n_16_30d = int((g["n_valid_post2016"] >= 30).sum())
    n_16_365d = int((g["n_valid_post2016"] >= 365).sum())
    print(
        f"{sub:12s}: total={n_tot:5d} | post2016 (>0d)={n_16_any:5d} |"
        f" (>=30d)={n_16_30d:5d} | (>=365d)={n_16_365d:5d}"
    )

  print("\n=== Total Across All Public MultiMet Basins ===")
  print(f"Total public MultiMet basins : {len(df)}")
  print(f"With >0 valid days >= 2016   : {int((df['n_valid_post2016'] > 0).sum())}")
  print(f"With >=30 valid days >= 2016 : {int((df['n_valid_post2016'] >= 30).sum())}")
  print(f"With >=365 valid days >= 2016: {int((df['n_valid_post2016'] >= 365).sum())}")

  print("\n=== Basins with >= 180 valid days in each individual year ===")
  for yr in range(2016, 2025):
    col = f"days_{yr}"
    if col in df.columns:
      print(
          f"  Year {yr}: >0d = {int((df[col] > 0).sum()):5d} | >=180d ="
          f" {int((df[col] >= 180).sum()):5d} | >=330d ="
          f" {int((df[col] >= 330).sum()):5d}"
      )


if __name__ == "__main__":
  main()
