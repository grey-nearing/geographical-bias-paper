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

"""Input/Output utility functions for the Geographical Bias project.

This module provides helpers for loading CSV files, reading experimental
metrics, and loading geospatial data from disk.
"""

import os
import time
import geopandas as gpd
import numpy as np
import pandas as pd
from utils import file_paths
from utils import gfile_compat as gfile

def load_csv(path, index_col=None):
  """Loads a CSV file from a given path using gfile.

  Automatically handles setting standard index columns if present.

  Args:
      path (str): The path to the CSV file.
      index_col (str, optional): The column to set as the index.

  Returns:
      pd.DataFrame: The loaded DataFrame.
  """
  with gfile.Open(path, 'rt') as f:
    df = pd.read_csv(f)
  if index_col and index_col in df.columns:
    df.set_index(index_col, inplace=True)
  elif 'Unnamed: 0' in df.columns:
    df.set_index('Unnamed: 0', inplace=True)
  elif 'gauge_id' in df.columns:
    df.set_index('gauge_id', inplace=True)
  return df

def load_experiment_metrics(
    base_path: str, experiment_type: str, metric: str, rename_to: str = 'skill'
) -> pd.Series:
  """Loads target skill metrics for a specific experiment and metric.

  Handles index standardization (prepending 'GRDC_' if necessary) and replaces
  infinities with NaN.

  Args:
      base_path (str): The base directory containing metrics.
      experiment_type (str): The experiment directory (e.g. 'random', 'gauged').
      metric (str): The target performance metric (e.g. 'NSE').
      rename_to (str): The name to assign to the resulting Series.

  Returns:
      pd.Series: The standardized skill metrics Series indexed by basin/gauge ID.
  """
  file_path = os.path.join(base_path, experiment_type, f'{metric}.csv')
  df = load_csv(file_path)
  series = df['0'] if '0' in df.columns else df.iloc[:, 0]
  
  # Standardize index prefix to 'GRDC_' for legacy numeric GRDC IDs,
  # while preserving Public Caravan IDs (e.g. 'camels_...', 'hysets_...').
  if experiment_type in ['random', 'separated', 'gauged']:
    if len(series.index) > 0 and str(series.index[0]).isdigit():
      series.index = [f'GRDC_{idx}' for idx in series.index]
      
  series = series.replace([np.inf, -np.inf], np.nan)
  return series.rename(rename_to)

def load_gdf_from_cns(
    cns_path: str, index_col: str = 'unique_gauge_id'
) -> gpd.GeoDataFrame:
  """Loads a GeoJSON file from the given CNS path into a GeoPandas DataFrame.

  This function performs two critical preprocessing steps:
  1. Reprojects the data to an Equal-Area CRS (EPSG:6933) to ensure area
  calculations are physically accurate in square meters.
  2. Applies a buffer(0) trick to clean topological errors (like self-intersecting
     polygons) that would otherwise crash the GEOS engine during intersections.

  Args:
      cns_path (str): The file path to the GeoJSON on disk.
      index_col (str): The column to set as the dataframe index.

  Returns:
      gpd.GeoDataFrame: The loaded, reprojected, and geometrically cleaned
      dataframe.
  """
  if not gfile.Exists(cns_path):
    raise FileNotFoundError(f'Python gfile cannot find: {cns_path}')

  print(f"[{time.strftime('%H:%M:%S')}] Loading file {cns_path} ... ")
  load_start = time.time()

  with gfile.Open(cns_path, 'rb') as cns_f:
    gdf = gpd.read_file(cns_f)

  # Dynamically set the index based on the dataset type (GRDC vs HyBAS)
  if index_col in gdf.columns:
    gdf.set_index(index_col, inplace=True)
  else:
    print(
        f"[{time.strftime('%H:%M:%S')}] Warning: '{index_col}' not found."
        ' Leaving default integer index.'
    )

  # Reproject to an Equal-Area CRS (EPSG:6933)
  if gdf.crs and gdf.crs.to_epsg() == 4326:
    print(
        f"[{time.strftime('%H:%M:%S')}] Reprojecting to Equal Earth (EPSG:6933)"
        ' for accurate areas...'
    )
    gdf = gdf.to_crs('EPSG:6933')

  # Clean geometry to force valid mathematical shapes
  print(
      f"[{time.strftime('%H:%M:%S')}] Cleaning geometry to resolve topological errors..."
  )
  gdf.geometry = gdf.geometry.buffer(0)

  print(
      f"[{time.strftime('%H:%M:%S')}] File successfully loaded/projected in {time.time() - load_start:.2f} seconds."
  )
  return gdf


def load_and_preprocess_data(
    model_type: str, metric: str, sample_size: int = None
) -> tuple[pd.DataFrame, pd.Series]:
  """Loads and preprocesses features and target skills for training.

  Args:
    model_type: 'gauged' or 'ungauged'.
    metric: The target performance metric.
    sample_size: Optional limit on the number of samples.

  Returns:
    A tuple of (X, y) where X is the feature DataFrame and y is the target
    Series.
  """
  start_time = time.time()
  wall_start = time.strftime('%H:%M:%S')

  if model_type == 'gauged':
    path_generic = file_paths.GAUGED_GENERIC_DATA_PATH
    print(
        f"[{wall_start}] Loading Gauged Generic Dataset from"
        f" {path_generic}..."
    )
    load_start = time.time()
    with gfile.Open(path_generic, 'rt') as f:
      df_generic = pd.read_csv(f, index_col=0)
    print(
        f"   -> Loaded in {time.time() - load_start:.2f}s | Shape:"
        f" {df_generic.shape}"
    )

    path_target = file_paths.METRICS_DIR
    print(
        f"[{time.strftime('%H:%M:%S')}] Loading Gauged target skill from"
        f" {path_target} for metric {metric}..."
    )
    load_start = time.time()
    skill_target = load_experiment_metrics(path_target, 'gauged', metric)
    print(
        f"   -> Loaded in {time.time() - load_start:.2f}s | Shape:"
        f" {skill_target.shape}"
    )

    print(f"[{time.strftime('%H:%M:%S')}] Joining datasets...")
    join_start = time.time()
    df_model = pd.concat([df_generic, skill_target], axis=1, join='inner').dropna()
    print(
        f"   -> Joined and cleaned in {time.time() - join_start:.2f}s | Shape:"
        f" {df_model.shape}"
    )

    if sample_size and len(df_model) > sample_size:
      df_model = df_model.sample(n=sample_size, random_state=42)
      print(f"   -> Subsampled to shape: {df_model.shape}")

  else:  # ungauged
    path_generic = file_paths.UNGAUGED_GENERIC_DATA_PATH
    print(
        f"[{wall_start}] Loading Ungauged Generic Dataset from"
        f" {path_generic}..."
    )
    load_start = time.time()
    with gfile.Open(path_generic, 'rt') as f:
      df_generic_combined = pd.read_csv(f, index_col=0)
    print(
        f"   -> Loaded in {time.time() - load_start:.2f}s | Shape:"
        f" {df_generic_combined.shape}"
    )

    print(f"[{time.strftime('%H:%M:%S')}] Splitting combined datasets...")
    n_samples = len(df_generic_combined) // 2
    df_random_generic = df_generic_combined.iloc[:n_samples]
    df_separated_generic = df_generic_combined.iloc[n_samples:]

    path_target = file_paths.METRICS_DIR
    print(
        f"[{time.strftime('%H:%M:%S')}] Loading Ungauged target skills from"
        f" {path_target} for metric {metric}..."
    )
    load_start = time.time()
    random_skill = load_experiment_metrics(path_target, 'random', metric)
    separated_skill = load_experiment_metrics(path_target, 'separated', metric)
    print(
        f"   -> Loaded in {time.time() - load_start:.2f}s | Random Shape:"
        f" {random_skill.shape} | Separated Shape: {separated_skill.shape}"
    )

    print(f"[{time.strftime('%H:%M:%S')}] Joining random partition...")
    join_start = time.time()
    df_random = pd.concat(
        [df_random_generic, random_skill], axis=1, join='inner'
    ).dropna()
    print(
        f"   -> Joined in {time.time() - join_start:.2f}s | Shape:"
        f" {df_random.shape}"
    )

    print(f"[{time.strftime('%H:%M:%S')}] Joining separated partition...")
    join_start = time.time()
    df_separated = pd.concat(
        [df_separated_generic, separated_skill], axis=1, join='inner'
    ).dropna()
    print(
        f"   -> Joined in {time.time() - join_start:.2f}s | Shape:"
        f" {df_separated.shape}"
    )

    if sample_size:
      if len(df_random) > sample_size:
        df_random = df_random.sample(n=sample_size, random_state=42)
      if len(df_separated) > sample_size:
        df_separated = df_separated.sample(n=sample_size, random_state=42)
      print(
          f"   -> Subsampled shapes | Random: {df_random.shape} | Separated:"
          f" {df_separated.shape}"
      )

    print(f"[{time.strftime('%H:%M:%S')}] Combining partitions...")
    combine_start = time.time()
    df_model = pd.concat([df_random, df_separated], axis=0)
    print(
        f"   -> Combined in {time.time() - combine_start:.2f}s | Shape:"
        f" {df_model.shape}"
    )

  # Apply outlier truncation
  print(f"[{time.strftime('%H:%M:%S')}] Applying outlier truncation...")
  trunc_start = time.time()
  if file_paths.OUTLIER_MIN is not None:
    df_model.loc[df_model['skill'] < file_paths.OUTLIER_MIN, 'skill'] = (
        file_paths.OUTLIER_MIN
    )
  if file_paths.OUTLIER_MAX is not None:
    df_model.loc[df_model['skill'] > file_paths.OUTLIER_MAX, 'skill'] = (
        file_paths.OUTLIER_MAX
    )
  print(f"   -> Applied in {time.time() - trunc_start:.2f}s")

  x = df_model.drop(columns=['skill'])
  y = df_model['skill']

  print(
      f"[{time.strftime('%H:%M:%S')}] Data preparation complete in"
      f" {time.time() - start_time:.2f}s (Wall time: {wall_start} to"
      f" {time.strftime('%H:%M:%S')})"
  )
  return x, y


def open_zarr_group(zarr_path: str, mode: str = 'r'):
  """Opens a Zarr group from a local or cloud filesystem."""
  import zarr
  store = zarr_path
  try:
    return zarr.open_group(store=store, mode=mode, zarr_format=2)
  except TypeError:
    return zarr.open_group(store=store, mode=mode)


def save_hybas_zarr(df_hybas: pd.DataFrame, zarr_path: str = None):
  """Saves the HyBAS inference DataFrame to a Zarr store (compatible with Zarr v2 and v3)."""
  if zarr_path is None:
    zarr_path = file_paths.HYBAS_GENERIC_DATA_ZARR_PATH
  root = open_zarr_group(zarr_path, mode='w')
  features_arr = np.asarray(df_hybas.values, dtype=np.float32)
  hybas_id_arr = np.asarray(df_hybas.index.astype(str), dtype='U32')
  columns_arr = np.asarray(df_hybas.columns.astype(str), dtype='U64')
  if hasattr(root, 'create_dataset'):
    root.create_dataset(
        'features',
        shape=features_arr.shape,
        dtype=features_arr.dtype,
        data=features_arr,
        chunks=(100000, df_hybas.shape[1]),
        overwrite=True,
    )
    root.create_dataset(
        'hybas_id',
        shape=hybas_id_arr.shape,
        dtype=hybas_id_arr.dtype,
        data=hybas_id_arr,
        chunks=(100000,),
        overwrite=True,
    )
    root.create_dataset(
        'columns',
        shape=columns_arr.shape,
        dtype=columns_arr.dtype,
        data=columns_arr,
        chunks=(df_hybas.shape[1],),
        overwrite=True,
    )
  else:
    root.create_array(
        'features',
        data=features_arr,
        chunks=(100000, df_hybas.shape[1]),
        overwrite=True,
    )
    root.create_array(
        'hybas_id',
        data=hybas_id_arr,
        chunks=(100000,),
        overwrite=True,
    )
    root.create_array(
        'columns',
        data=columns_arr,
        chunks=(df_hybas.shape[1],),
        overwrite=True,
    )


def load_hybas_zarr(
    zarr_path: str = None, csv_fallback_path: str = None
) -> pd.DataFrame:
  """Loads the HyBAS inference DataFrame from Zarr, with automatic CSV fallback."""
  if zarr_path is None:
    zarr_path = file_paths.HYBAS_GENERIC_DATA_ZARR_PATH
  if csv_fallback_path is None:
    csv_fallback_path = file_paths.HYBAS_GENERIC_DATA_PATH
  try:
    root = open_zarr_group(zarr_path, mode='r')
    features = root['features'][:]
    hybas_ids = [str(x) for x in root['hybas_id'][:]]
    columns = [str(x) for x in root['columns'][:]]
    return pd.DataFrame(features, index=hybas_ids, columns=columns)
  except Exception as e:
    print(
        f"   -> Zarr load from {zarr_path} failed ({e}); falling back to CSV"
        f" {csv_fallback_path}..."
    )
    with gfile.Open(csv_fallback_path, 'rt') as f:
      df = pd.read_csv(f, index_col=0)
    df.index = [str(x) for x in df.index]
    return df

