# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Grey Nearing
# Created: 2026-07-14


"""Calculate Observed Streamflow Record Lengths.

This script calculates the length of the observed streamflow data record (number
of valid daily observations) for each catchment used in the study.

Main Steps:
1. Open consolidated nature data zarr.
2. Extract lead_time=0 observations.
3. Calculate non-null observation counts per basin.
4. Save valid streamflow observation lengths.

Required Input Files:
- Consolidated Zarr:
file_paths.CONSOLIDATED_ZARR_PATH

Saved Output Files:
- GRDC Record Lengths: file_paths.GRDC_RECORD_LENGTHS_PATH
"""

import time
import numpy as np
import pandas as pd
import xarray as xr
from utils import file_paths
from utils import gfile_compat as gfile


# ==============================================================================
# Main Calculation Pipeline
# ==============================================================================
# This section orchestrates the loading of daily streamflow observations, 
# computes the total number of valid data points (excluding missing values) 
# for every individual catchment, formats the catchment identifiers correctly, 
# and finally writes the summarized record lengths out to Google's file system.

def _load_zarr_dataset(zarr_path):
  """Opens a Zarr dataset stored on Google's file system.

  Args:
    zarr_path: A string specifying the file path to the Zarr dataset on disk.

  Returns:
    An xarray.Dataset containing the streamflow observations.
  """
  start = time.time()
  print(f"[{time.strftime('%H:%M:%S')}] Opening Zarr at: {zarr_path}")
  ds = xr.open_zarr(zarr_path)
  print(
      f"[{time.strftime('%H:%M:%S')}] Opened Zarr dataset in"
      f" {time.time() - start:.2f} seconds. Variables: {len(ds)}."
  )
  return ds


def _compute_valid_counts(ds):
  """Calculates the number of valid (non-null) daily observations per catchment.

  Args:
    ds: An xarray.Dataset containing daily streamflow observations and a 
      dimension for forecast lead times.

  Returns:
    An xarray.DataArray containing the total count of valid daily streamflow 
    observations for each basin.
  """
  start = time.time()
  print(
      f"[{time.strftime('%H:%M:%S')}] Extracting observations and calculating"
      " valid daily counts..."
  )
  # Extract observations where the forecast lead time is zero. 
  # Lead time of zero isolates the actual daily observed streamflow values,
  # keeping us from accidentally including any forecasted data.
  obs_daily = ds["observation"].sel(lead_time=0)
  print(
      f"[{time.strftime('%H:%M:%S')}] Extracted observations in"
      f" {time.time() - start:.2f} seconds."
  )

  # Calculate the total number of non-null (valid) observations for each catchment 
  # across the entire time record. Calling .compute() triggers the actual parallel 
  # calculation across the Dask-backed xarray dataset in memory.
  start_time = time.time()
  valid_counts = obs_daily.notnull().sum(dim="time").compute()
  print(
      f"[{time.strftime('%H:%M:%S')}] Computation completed in"
      f" {time.time() - start_time:.2f} seconds."
  )
  return valid_counts


def _convert_and_decode_df(valid_counts):
  """Converts validity counts to a Pandas DataFrame and cleans up gauge IDs.

  Args:
    valid_counts: An xarray.DataArray containing observation counts keyed by 
      gauge/basin ID.

  Returns:
    A pandas.DataFrame where the index represents string basin IDs, and the 
    single column contains their respective 'record_length' values.
  """
  # Transform the xarray DataArray into a Pandas DataFrame for simpler 
  # tabular manipulation and easier export to CSV formats in the next steps.
  start = time.time()
  df = valid_counts.to_dataframe(name="record_length")
  print(
      f"[{time.strftime('%H:%M:%S')}] Created record length DataFrame in"
      f" {time.time() - start:.2f} seconds, shape: {df.shape}"
  )

  # Gauge IDs may be stored as byte strings in the underlying data format.
  # This converts them into standard UTF-8 strings (e.g. from b'123' to '123') 
  # so they are clean and readable when written out to our required formats.
  start = time.time()
  print(f"[{time.strftime('%H:%M:%S')}] Decoding gauge IDs...")
  decoded_ids = [
      gid.decode("utf-8") if isinstance(gid, bytes) else str(gid)
      for gid in df.index
  ]
  df.index = decoded_ids
  df.index.name = "basin_id"
  print(
      f"[{time.strftime('%H:%M:%S')}] Decoded IDs in {time.time() - start:.2f}"
      f" seconds. df shape: {df.shape}"
  )
  return df


def _save_to_csv(df, out_path):
  """Writes a DataFrame to a CSV file in Google's internal file system.

  Args:
    df: A pandas.DataFrame to be saved.
    out_path: A string specifying the destination file path.
  """
  # Write the resulting DataFrame holding basin IDs and their observation counts 
  # into a CSV file, which will be provided as an input in downstream analyses.
  start = time.time()
  print(f"[{time.strftime('%H:%M:%S')}] Saving to {out_path}...")
  file_paths.ensure_directories_exist()
  with gfile.Open(out_path, "wt") as f:
    df.to_csv(f)
  print(
      f"[{time.strftime('%H:%M:%S')}] Saved to CSV in {time.time() - start:.2f}"
      " seconds."
  )


def main(unused_argv=None):
  """Entry point for the record lengths script.

  Args:
    unused_argv: A list of command-line arguments passed to the script. This 
      is required by the absl.app framework but unused in this script.
  """
  zarr_path = file_paths.CONSOLIDATED_ZARR_PATH
  out_path = file_paths.GRDC_RECORD_LENGTHS_PATH

  if not gfile.Exists(zarr_path) and gfile.Exists(out_path):
    with gfile.Open(out_path, "rt") as f:
      df = pd.read_csv(f, index_col=0)
    print(
        f"[{time.strftime('%H:%M:%S')}] Raw streamflow Zarr not present at"
        f" {zarr_path}; verified existing record lengths CSV at {out_path} with"
        f" {len(df)} basins."
    )
    return

  ds = _load_zarr_dataset(zarr_path)
  valid_counts = _compute_valid_counts(ds)
  df = _convert_and_decode_df(valid_counts)
  _save_to_csv(df, out_path)

  print(
      f"[{time.strftime('%H:%M:%S')}] Process complete. Saved {len(df)} basin"
      " record lengths."
  )


if __name__ == "__main__":
  from absl import app

  app.run(main)
