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


"""Create Generic Training and Inference Datasets.

This script creates generic datasets (without target metrics) for Gauged models,
Ungauged models, and Global HyBAS Inference.

Main Steps:
1. Load generic attributes, overlaps, anomaly scores, and record lengths.
2. Create Gauged Generic Dataset by concatenating relevant features.
3. Create Ungauged Generic Dataset (combining Random and Separated).
4. Create HyBAS Generic Inference Dataset.
5. Save final compiled generic datasets to disk.

Required Input Files:
- GRDC Inputs: file_paths.GRDC_ATTRIBUTES_PATH,
file_paths.GRDC_RECORD_LENGTHS_PATH
- Overlaps and Anomalies: Path endpoints from generated anomaly and overlap
outputs.

Saved Output Files:
- Gauged Dataset: file_paths.GAUGED_GENERIC_DATA_PATH
- Ungauged Dataset: file_paths.UNGAUGED_GENERIC_DATA_PATH
- HyBAS Inference Dataset: file_paths.HYBAS_GENERIC_DATA_PATH
"""

import os
import time
import numpy as np
import pandas as pd
from utils import file_paths
from utils import io_utils
from utils.io_utils import load_csv
from utils import gfile_compat as gfile


def load_csv_optimized(path, usecols=None, index_col=None):
  """Loads a CSV file into a pandas DataFrame optimally.

  This function loads CSV files using the internal PyGlib library to support
  internal file systems. It also allows selection of specific columns to reduce 
  memory usage and handles setting the DataFrame index based on known ID 
  columns if an explicit index is not provided.

  Args:
    path: A string representing the file path of the CSV to load.
    usecols: An optional list of strings specifying which column names to load.
    index_col: An optional string representing the column name to use as the
      DataFrame index.

  Returns:
    A pandas DataFrame containing the loaded CSV data, with an appropriately 
    set index.
  """
  with gfile.Open(path, "rt") as f:
    df = pd.read_csv(f, usecols=usecols)
  if index_col and index_col in df.columns:
    df.set_index(index_col, inplace=True)
  elif "Unnamed: 0" in df.columns:
    df.set_index("Unnamed: 0", inplace=True)
    if index_col:
      df.index.name = index_col
  elif "gauge_id" in df.columns:
    df.set_index("gauge_id", inplace=True)
  elif "hybas_id" in df.columns:
    df.set_index("hybas_id", inplace=True)
  return df


def load_grdc_attributes() -> pd.DataFrame:
  """Loads and preprocesses catchment attributes for GRDC basins.

  This function reads the generic attributes corresponding to the GRDC gauged
  basins. It drops any spatial columns (like coordinates) that are explicitly
  banned to prevent models from learning geographic coordinates. Finally, it 
  returns only numeric features with no missing data.

  Returns:
    A pandas DataFrame containing the filtered, numeric generic attributes 
    for GRDC basins.
  """
  print(f"[{time.strftime('%H:%M:%S')}] Loading GRDC attributes...")
  start_time = time.time()
  grdc_attr_raw = load_csv(file_paths.GRDC_ATTRIBUTES_PATH)
  print(
      f"   -> grdc_attr_raw shape: {grdc_attr_raw.shape}, len:"
      f" {len(grdc_attr_raw)}"
  )
  grdc_features = grdc_attr_raw.drop(
      columns=[
          c
          for c in grdc_attr_raw.columns
          if c.lower() in file_paths.BANNED_SPATIAL_COLS
      ]
  )
  grdc_features = grdc_features.select_dtypes(include=[np.number]).dropna()
  elapsed = time.time() - start_time
  print(f"   -> Completed in {elapsed:.2f}s")
  print(
      f"   -> grdc_features shape: {grdc_features.shape}, len:"
      f" {len(grdc_features)}"
  )
  return grdc_features


def load_grdc_record_lengths() -> pd.DataFrame:
  """Loads and standardizes GRDC record length data.

  This function loads the record lengths associated with GRDC gauges, which 
  indicate how many years of data each gauge has. It ensures the index is appropriately
  named 'basin_id' and the data column is standardized to 'record_length'.

  Returns:
    A pandas DataFrame indexed by gauge ID containing the 'record_length' for 
    each basin.
  """
  print(f"[{time.strftime('%H:%M:%S')}] Loading GRDC record lengths...")
  start_time = time.time()
  grdc_record_lengths = load_csv(file_paths.GRDC_RECORD_LENGTHS_PATH)
  if "Gauge ID" in grdc_record_lengths.columns:
    grdc_record_lengths = grdc_record_lengths.set_index("Gauge ID")
  grdc_record_lengths.index.name = "basin_id"
  grdc_record_lengths = grdc_record_lengths.rename(
      columns={"Record Length": "record_length"}
  )
  elapsed = time.time() - start_time
  print(f"   -> Completed in {elapsed:.2f}s")
  print(
      f"   -> grdc_record_lengths shape: {grdc_record_lengths.shape}, len:"
      f" {len(grdc_record_lengths)}"
  )
  return grdc_record_lengths


def load_anomalies() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Loads anomaly scores for the Gauged, Random, and Separated splits.

  Anomaly scores correspond to isolation forest or nearest-neighbor distance 
  metrics that describe how unusual a catchment is. This function reads 
  and returns the precomputed anomaly datasets for all three subsets of GRDC gauges.

  Returns:
    A tuple containing three pandas DataFrames:
      - gauged_anomalies: Anomaly scores for the Gauged split.
      - random_anomalies: Anomaly scores for the Random validation split.
      - separated_anomalies: Anomaly scores for the Spatially Separated split.
  """
  print(f"[{time.strftime('%H:%M:%S')}] Loading GRDC anomaly scores...")
  start_time = time.time()
  gauged_anomalies = load_csv(file_paths.GAUGED_ANOMALIES_PATH)
  random_anomalies = load_csv(file_paths.RANDOM_ANOMALIES_PATH)
  separated_anomalies = load_csv(file_paths.SEPARATED_ANOMALIES_PATH)
  elapsed = time.time() - start_time
  print(f"   -> Completed in {elapsed:.2f}s")
  print(
      f"   -> gauged_anomalies shape: {gauged_anomalies.shape}, len:"
      f" {len(gauged_anomalies)}"
  )
  print(
      f"   -> random_anomalies shape: {random_anomalies.shape}, len:"
      f" {len(random_anomalies)}"
  )
  print(
      f"   -> separated_anomalies shape: {separated_anomalies.shape}, len:"
      f" {len(separated_anomalies)}"
  )
  return gauged_anomalies, random_anomalies, separated_anomalies


def load_grdc_overlaps() -> pd.DataFrame:
  """Loads GRDC spatial overlaps data.

  This data contains features relating to the hierarchical overlap of basins, 
  such as the fraction of a basin's area comprised by upstream gauged basins. 

  Returns:
    A pandas DataFrame containing spatial overlap characteristics for GRDC basins.
  """
  print(f"[{time.strftime('%H:%M:%S')}] Loading GRDC spatial overlaps...")
  start_time = time.time()
  grdc_overlaps = load_csv(file_paths.TRAIN_ML_FEATURES_PATH)
  elapsed = time.time() - start_time
  print(f"   -> Completed in {elapsed:.2f}s")
  print(
      f"   -> grdc_overlaps shape: {grdc_overlaps.shape}, len:"
      f" {len(grdc_overlaps)}"
  )
  return grdc_overlaps


def create_gauged_dataset(
    grdc_features: pd.DataFrame,
    gauged_anomalies: pd.DataFrame,
    anomaly_cols: list[str],
    grdc_record_lengths: pd.DataFrame,
) -> pd.DataFrame:
  """Creates the generic dataset for Gauged models.

  This dataset combines general attributes, anomaly scores, and record lengths. 
  Gauged models are intended to be trained on basins where streamflow is measured,
  so they do not require spatial overlap features that simulate nested predictions.

  Args:
    grdc_features: A DataFrame of general GRDC attributes.
    gauged_anomalies: A DataFrame of anomaly scores for the gauged split.
    anomaly_cols: A list of column names for the anomaly scores to use.
    grdc_record_lengths: A DataFrame describing the record length corresponding
      to each GRDC gauge.

  Returns:
    A pandas DataFrame containing the merged dataset for the Gauged model, 
    dropping any basins with missing data.
  """
  # We combine the primary features, the selected anomaly columns, and the record lengths.
  # We perform an inner join (taking the intersection of basins) and drop any rows with missing data.
  # Note: Gauged models do not receive spatial coverage/overlap information as inputs.
  print(f"[{time.strftime('%H:%M:%S')}] Creating Gauged Generic Dataset...")
  start_time = time.time()
  df_gauged = pd.concat(
      [
          grdc_features,
          gauged_anomalies[anomaly_cols],
          grdc_record_lengths,
      ],
      axis=1,
      join="inner",
  ).dropna()
  elapsed = time.time() - start_time
  print(f"   -> Completed in {elapsed:.2f}s")
  print(f"   -> df_gauged shape: {df_gauged.shape}, len: {len(df_gauged)}")
  return df_gauged


def create_ungauged_dataset(
    grdc_features: pd.DataFrame,
    random_anomalies: pd.DataFrame,
    separated_anomalies: pd.DataFrame,
    grdc_overlaps: pd.DataFrame,
    grdc_record_lengths: pd.DataFrame,
    anomaly_cols: list[str],
    overlap_cols: list[str],
) -> pd.DataFrame:
  """Creates the generic dataset for Ungauged models (Random + Separated splits).

  Ungauged models must learn how to simulate nested catchments (i.e. predicting
  for an ungauged area inside of a larger gauged basin, or vice versa). Hence, 
  they receive overlap features. We construct two partitions: "random" which
  has actual overlap data representing typical spatial proximity, and "separated"
  where overlaps are forced to zero to simulate a completely disconnected catchment.

  Args:
    grdc_features: A DataFrame of general GRDC attributes.
    random_anomalies: Anomaly scores for basins in the random split.
    separated_anomalies: Anomaly scores for basins in the separated split.
    grdc_overlaps: Spatial overlap fractions amongst basins.
    grdc_record_lengths: Observation record lengths.
    anomaly_cols: Which columns to select from the anomaly DataFrames.
    overlap_cols: Which columns to select from the overlaps DataFrames.

  Returns:
    A pandas DataFrame comprising both the random and separated partitions 
    vertically stacked, representing unmeasured catchments.
  """
  print(f"[{time.strftime('%H:%M:%S')}] Creating Ungauged Random partition...")
  start_time = time.time()
  df_random = pd.concat(
      [
          grdc_features,
          random_anomalies[anomaly_cols],
          grdc_overlaps[overlap_cols],
          grdc_record_lengths,
      ],
      axis=1,
      join="inner",
  ).dropna()
  elapsed = time.time() - start_time
  print(f"   -> Completed in {elapsed:.2f}s")
  print(f"   -> df_random shape: {df_random.shape}, len: {len(df_random)}")

  print(
      f"[{time.strftime('%H:%M:%S')}] Creating Ungauged Separated partition..."
  )
  start_time = time.time()
  # For the separated partition, we forcefully simulate that these test basins 
  # are entirely disconnected from any training basins by zeroing out the overlap features.
  separated_overlaps = grdc_overlaps[overlap_cols].copy()
  separated_overlaps[:] = 0
  df_separated = pd.concat(
      [
          grdc_features,
          separated_anomalies[anomaly_cols],
          separated_overlaps,
          grdc_record_lengths,
      ],
      axis=1,
      join="inner",
  ).dropna()
  elapsed = time.time() - start_time
  print(f"   -> Completed in {elapsed:.2f}s")
  print(
      f"   -> df_separated shape: {df_separated.shape}, len:"
      f" {len(df_separated)}"
  )

  print(
      f"[{time.strftime('%H:%M:%S')}] Combining random and separated"
      " partitions..."
  )
  start_time = time.time()
  df_ungauged = pd.concat([df_random, df_separated], axis=0)
  elapsed = time.time() - start_time
  print(f"   -> Completed in {elapsed:.2f}s")
  print(
      f"   -> df_ungauged shape (combined): {df_ungauged.shape}, len:"
      f" {len(df_ungauged)}"
  )

  return df_ungauged


def save_grdc_datasets(df_gauged: pd.DataFrame, df_ungauged: pd.DataFrame):
  """Prints a summary report and saves the generic GRDC datasets to disk.

  This outputs debugging statistics (shapes, column subsets) helping developers 
  ensure the resulting datasets look correct before writing them out to CSV format.

  Args:
    df_gauged: The compiled gauged model dataset.
    df_ungauged: The compiled ungauged model dataset.
  """
  print("=" * 80)
  print("GRDC DATASETS REPORT")
  print("=" * 80)
  print(f"1. Gauged Generic Dataset:")
  print(f"   - Shape: {df_gauged.shape}")
  print(
      f"   - Columns: {list(df_gauged.columns[:5])} ..."
      f" {list(df_gauged.columns[-5:])}"
  )
  print(f"   - Index Name: {df_gauged.index.name}")
  print(f"   - Save Path: {file_paths.GAUGED_GENERIC_DATA_PATH}")
  print("-" * 80)
  print(f"2. Ungauged Generic Dataset (Random + Separated):")
  print(f"   - Shape: {df_ungauged.shape}")
  print(
      f"   - Columns: {list(df_ungauged.columns[:5])} ..."
      f" {list(df_ungauged.columns[-5:])}"
  )
  print(f"   - Index Name: {df_ungauged.index.name}")
  print(f"   - Save Path: {file_paths.UNGAUGED_GENERIC_DATA_PATH}")
  print("=" * 80)

  # Create the target output directory if it doesn't already exist, then save 
  # the merged DataFrames as CSVs to be consumed by the next stage of the pipeline.
  file_paths.ensure_directories_exist()
  print(f"[{time.strftime('%H:%M:%S')}] Saving Gauged Generic Dataset...")
  start_time = time.time()
  with gfile.Open(file_paths.GAUGED_GENERIC_DATA_PATH, "wt") as f:
    df_gauged.to_csv(f, index=True)
  elapsed = time.time() - start_time
  print(f"   -> Saved Gauged in {elapsed:.2f}s")

  print(f"[{time.strftime('%H:%M:%S')}] Saving Ungauged Generic Dataset...")
  start_time = time.time()
  with gfile.Open(file_paths.UNGAUGED_GENERIC_DATA_PATH, "wt") as f:
    df_ungauged.to_csv(f, index=True)
  elapsed = time.time() - start_time
  print(f"   -> Saved Ungauged in {elapsed:.2f}s")


def load_hybas_attributes(grdc_features: pd.DataFrame) -> pd.DataFrame:
  """Loads and filters global HydroBASINS (HyBAS) attributes.

  This systematically loads the massive global HyBAS attribute table but optimizes
  memory usage by pre-filtering columns to only include those identical to the 
  features used during the GRDC models' training.

  Args:
    grdc_features: The base GRDC properties DataFrame used to determine which 
      columns should be retained in the global dataset.

  Returns:
    A pandas DataFrame containing global HyBAS attributes matching the training 
    features.
  """
  print(
      f"[{time.strftime('%H:%M:%S')}] Loading HyBAS attributes (optimized)..."
  )
  start_time = time.time()
  
  # Read just the column headers from the massive global file to plan which to load.
  with gfile.Open(file_paths.HYBAS_ATTRIBUTES_PATH, "rt") as f:
    header = pd.read_csv(f, nrows=0)
  original_cols = list(header.columns)
  cleaned_cols = [c.replace("HYDRO_ATLAS:", "") for c in original_cols]

  # The very first column provides the unique identifier for each HyBAS basin.
  # We must explicitly keep it to use as the row index.
  index_col_name = original_cols[0]
  cols_to_keep = [index_col_name]

  grdc_cols_set = set(grdc_features.columns)
  for orig_c, clean_c in zip(original_cols[1:], cleaned_cols[1:]):
    if clean_c in grdc_cols_set:
      cols_to_keep.append(orig_c)

  print(
      f"   -> Keeping {len(cols_to_keep)} columns out of {len(original_cols)}"
  )
  hybas_attr_raw = load_csv_optimized(
      file_paths.HYBAS_ATTRIBUTES_PATH,
      usecols=cols_to_keep,
      index_col="hybas_id",
  )
  print(
      f"   -> hybas_attr_raw shape: {hybas_attr_raw.shape}, len:"
      f" {len(hybas_attr_raw)}"
  )
  hybas_attr_raw.columns = hybas_attr_raw.columns.str.replace(
      "HYDRO_ATLAS:", "", regex=False
  )
  hybas_features = hybas_attr_raw.drop(
      columns=[
          c
          for c in hybas_attr_raw.columns
          if c.lower() in file_paths.BANNED_SPATIAL_COLS
      ]
  )
  hybas_features = hybas_features.select_dtypes(include=[np.number]).dropna()
  elapsed = time.time() - start_time
  print(f"   -> Completed in {elapsed:.2f}s")
  print(
      f"   -> hybas_features shape: {hybas_features.shape}, len:"
      f" {len(hybas_features)}"
  )
  return hybas_features


def load_hybas_anomalies() -> pd.DataFrame:
  """Loads anomaly scores calculated for all global HyBAS catchments.

  Returns:
    A pandas DataFrame containing anomaly scores indexed by HyBAS IDs.
  """
  print(f"[{time.strftime('%H:%M:%S')}] Loading HyBAS anomalies...")
  start_time = time.time()
  hybas_anomalies = load_csv(
      file_paths.INFER_HYBAS_CARAVAN_OUT_PATH, index_col="hybas_id"
  )
  elapsed = time.time() - start_time
  print(f"   -> Completed in {elapsed:.2f}s")
  print(
      f"   -> hybas_anomalies shape: {hybas_anomalies.shape}, len:"
      f" {len(hybas_anomalies)}"
  )
  return hybas_anomalies


def load_hybas_overlaps() -> pd.DataFrame:
  """Loads spatial overlap fractions for the global HyBAS dataset.

  Because the global inference requires overlap calculations for millions of 
  basins, the data is partitioned into multiple chunks across a directory. 
  This function globs and concatenates all chunks together.

  Returns:
    A pandas DataFrame of overlap characteristics for all global HyBAS basins.
  """
  print(f"[{time.strftime('%H:%M:%S')}] Loading HyBAS overlaps...")
  start_time = time.time()
  overlap_files = gfile.Glob(
      os.path.join(file_paths.HYBAS_COVERAGE_DIR, "infer_ml_features_*.csv")
  )
  hybas_overlaps_list = []
  for f in overlap_files:
    df_cont = load_csv(f, index_col="hybas_id")
    hybas_overlaps_list.append(df_cont)
  hybas_overlaps = (
      pd.concat(hybas_overlaps_list) if hybas_overlaps_list else pd.DataFrame()
  )
  elapsed = time.time() - start_time
  print(f"   -> Completed in {elapsed:.2f}s")
  print(
      f"   -> hybas_overlaps shape: {hybas_overlaps.shape}, len:"
      f" {len(hybas_overlaps)}"
  )
  return hybas_overlaps


def create_hybas_dataset(
    hybas_features: pd.DataFrame,
    hybas_anomalies: pd.DataFrame,
    hybas_overlaps: pd.DataFrame,
    grdc_record_lengths: pd.DataFrame,
    anomaly_cols: list[str],
    overlap_cols: list[str],
) -> pd.DataFrame:
  """Combines attributes, anomalies, and overlaps into the final inference dataset.

  Args:
    hybas_features: Filtered general attributes for HyBAS basins.
    hybas_anomalies: Anomaly scores mapped to HyBAS basins.
    hybas_overlaps: Spatial overlap tracking for the global basins.
    grdc_record_lengths: The record lengths from the training GRDC set, used to 
      derive the standardized control constant for inference.
    anomaly_cols: List of column names specifying which anomaly metrics to include.
    overlap_cols: List of column names specifying which overlap metrics to include.

  Returns:
    A final, combined pandas DataFrame configured to feed directly into the 
    global ML inference pipeline.
  """
  print(f"[{time.strftime('%H:%M:%S')}] Combining HyBAS features...")
  start_time = time.time()
  df_hybas = pd.concat(
      [
          hybas_features,
          hybas_anomalies[anomaly_cols],
          hybas_overlaps[overlap_cols],
      ],
      axis=1,
      join="inner",
  ).dropna()

  # The AI models expect a 'record_length' feature even during inference. We feed 
  # the maximum observed record length from training as a constant to eliminate it 
  # as a source of variable bias when predicting broadly across unmonitored sites.
  max_record_length = grdc_record_lengths["record_length"].max()
  df_hybas["record_length"] = max_record_length
  elapsed = time.time() - start_time
  print(f"   -> Completed in {elapsed:.2f}s")
  print(
      f"   -> df_hybas shape (with record_length): {df_hybas.shape}, len:"
      f" {len(df_hybas)}"
  )
  return df_hybas


def save_hybas_dataset(df_hybas: pd.DataFrame):
  """Prints terminal report and writes the final HyBAS dataset to disk.

  Args:
    df_hybas: The finalized inference dataset to be saved out.
  """
  print("=" * 80)
  print("HyBAS DATASET REPORT")
  print("=" * 80)
  print(f"1. HyBAS Generic Inference Dataset:")
  print(f"   - Shape: {df_hybas.shape}")
  print(
      f"   - Columns: {list(df_hybas.columns[:5])} ..."
      f" {list(df_hybas.columns[-5:])}"
  )
  print(f"   - Index Name: {df_hybas.index.name}")
  print(f"   - Save Path (CSV): {file_paths.HYBAS_GENERIC_DATA_PATH}")
  print(f"   - Save Path (Zarr): {file_paths.HYBAS_GENERIC_DATA_ZARR_PATH}")
  print("=" * 80)

  # Ensure the destination folder exists to prevent I/O errors and dump the output
  file_paths.ensure_directories_exist()
  
  # Save to Zarr
  print(f"[{time.strftime('%H:%M:%S')}] Saving HyBAS Generic Dataset to Zarr...")
  start_time = time.time()
  zarr_path = file_paths.HYBAS_GENERIC_DATA_ZARR_PATH
  io_utils.save_hybas_zarr(df_hybas, zarr_path)
  print(f"   -> Saved HyBAS Zarr dataset in {time.time() - start_time:.2f}s")

  # Save to CSV
  print(f"[{time.strftime('%H:%M:%S')}] Saving HyBAS Generic Dataset to CSV (legacy)...")
  start_time = time.time()
  with gfile.Open(file_paths.HYBAS_GENERIC_DATA_PATH, "wt") as f:
    df_hybas.to_csv(f, index=True)
  print(f"   -> Saved HyBAS CSV dataset in {time.time() - start_time:.2f}s")


def main(argv):
  """Main entry point. Coordinates dataset building across subsets.

  Args:
    argv: List of command line arguments (mostly unused by this script).
  """
  del argv

  # Define the required anomaly metrics array and overlap statistics 
  # needed when merging various tables into our final ML ready-framework.
  anomaly_cols = [
      "anomaly_score_if",
      "anomaly_score_ocsvm",
      "anomaly_score_lof",
  ]
  overlap_cols = [
      "total_upstream_coverage",
      "max_upstream_coverage",
      "max_downstream_coverage",
      "upstream_gauge_count",
  ]

  #
  # STAGE 1: Generating Training Datasets (GRDC)
  # 
  # Before predicting on uncharted global territory, two distinct datasets are 
  # built using gauges from GRDC: A completely gauged dataset acting as a baseline, 
  # and a simulated 'ungauged' set designed to teach the network spatial cross-prediction.
  # Note: The "Load" sections pull from numerous pre-computed CSV files generated previously.

  grdc_features = load_grdc_attributes()
  grdc_record_lengths = load_grdc_record_lengths()
  gauged_anomalies, random_anomalies, separated_anomalies = load_anomalies()
  grdc_overlaps = load_grdc_overlaps()

  # Construct the dataframe exclusively intended to train models mimicking gauged areas.
  df_gauged = create_gauged_dataset(
      grdc_features, gauged_anomalies, anomaly_cols, grdc_record_lengths
  )

  # Construct the complex split-dataframe for teaching ungauged geographic generalization.
  df_ungauged = create_ungauged_dataset(
      grdc_features,
      random_anomalies,
      separated_anomalies,
      grdc_overlaps,
      grdc_record_lengths,
      anomaly_cols,
      overlap_cols,
  )

  # Visualize standard parameters and flush them out to storage.
  save_grdc_datasets(df_gauged, df_ungauged)

  #
  # STAGE 2: Generating Global Inference Dataset (HyBAS)
  #
  # The ultimate test is predicting over millions of unmonitored global watersheds.
  # This part loads comparable characteristics from the massive HydroBASIN catalogs 
  # and produces a mirrored data structure the trained model can ingest directly.

  hybas_features = load_hybas_attributes(grdc_features)
  hybas_anomalies = load_hybas_anomalies()
  hybas_overlaps = load_hybas_overlaps()

  # Re-apply dataset alignments specific to the millions of global features
  df_hybas = create_hybas_dataset(
      hybas_features,
      hybas_anomalies,
      hybas_overlaps,
      grdc_record_lengths,
      anomaly_cols,
      overlap_cols,
  )

  # Report status and write the global file to storage for future steps.
  save_hybas_dataset(df_hybas)

  print(f"[{time.strftime('%H:%M:%S')}] HyBAS dataset creation complete.")


if __name__ == "__main__":
  from absl import app

  app.run(main)
