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


def load_csv(path, index_col=None):
  """Loads a CSV file from a given path.

  Automatically handles setting standard index columns if present.

  Args:
      path (str): The path to the CSV file.
      index_col (str, optional): The column to set as the index.

  Returns:
      pd.DataFrame: The loaded DataFrame.
  """
  df = pd.read_csv(path)
  if index_col and index_col in df.columns:
    df.set_index(index_col, inplace=True)
  elif 'Unnamed: 0' in df.columns:
    df.set_index('Unnamed: 0', inplace=True)
  elif 'gauge_id' in df.columns:
    df.set_index('gauge_id', inplace=True)
  return df

def load_experiment_metrics(
    base_path: str,
    experiment_type: str,
    metric: str,
    rename_to: str = 'skill',
    lead_time: str = '1',
) -> pd.Series:
  """Loads target skill metrics for a specific experiment and metric.

  Handles index standardization (prepending 'GRDC_' if necessary) and replaces
  infinities with NaN.

  Args:
      base_path (str): The base directory containing metrics.
      experiment_type (str): The experiment directory (e.g. 'random', 'gauged').
      metric (str): The target performance metric (e.g. 'NSE').
      rename_to (str): The name to assign to the resulting Series.
      lead_time (str): Forecast lead-time column to extract (default '1').

  Returns:
      pd.Series: The standardized skill metrics Series indexed by basin/gauge ID.
  """
  file_path = os.path.join(base_path, experiment_type, f'{metric}.csv')
  df = load_csv(file_path)
  lead_col = str(lead_time)
  if lead_col in df.columns:
    series = df[lead_col]
  elif '1' in df.columns:
    series = df['1']
  elif '0' in df.columns:
    series = df['0']
  else:
    series = df.iloc[:, 0]

  # Standardize index prefix to 'GRDC_' for legacy numeric GRDC IDs,
  # while preserving Public Caravan IDs (e.g. 'camels_...', 'hysets_...').
  base_split = experiment_type.split('_')[0]
  if base_split in ['random', 'separated', 'gauged']:
    if len(series.index) > 0 and str(series.index[0]).isdigit():
      series.index = [f'GRDC_{idx}' for idx in series.index]

  series = series.replace([np.inf, -np.inf], np.nan)
  return series.rename(rename_to)

def load_gdf(
    filepath: str, index_col: str = 'unique_gauge_id'
) -> gpd.GeoDataFrame:
  """Loads a GeoJSON file from disk into a GeoPandas DataFrame.

  Reprojects to Equal-Area CRS (EPSG:6933) and applies buffer(0) geometry cleanup.
  """
  if not os.path.exists(filepath):
    raise FileNotFoundError(f'File not found: {filepath}')

  print(f"[{time.strftime('%H:%M:%S')}] Loading file {filepath} ... ")
  load_start = time.time()
  gdf = gpd.read_file(filepath)

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
    model_type: str,
    metric: str,
    sample_size: int = None,
    exp_suffix: str = '',
    lead_time: str = '1',
    drop_record_length: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
  """Loads and preprocesses features and target skills for training.

  Supports both unified 3-split ('gauged', 'random', 'separated') and legacy
  2-split ('random', 'separated') ungauged feature matrices, optional adaptation
  experiment suffixes ('', '_ft', '_da', '_ft_da'), and lead-time column selection.

  Args:
    model_type: 'gauged' or 'ungauged'.
    metric: The target performance metric.
    sample_size: Optional limit on the number of samples.
    exp_suffix: Optional experiment suffix ('', '_ft', '_da', '_ft_da').
    lead_time: Lead-time column name in hydro metric CSVs (default '1').
    drop_record_length: If True, drops synthetic 'record_length' (95-feature spec).

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
    pq_generic = os.path.splitext(path_generic)[0] + '.parquet'
    if os.path.exists(pq_generic):
      df_generic = pd.read_parquet(pq_generic)
    else:
      df_generic = pd.read_csv(path_generic, index_col=0)
    print(
        f"   -> Loaded in {time.time() - load_start:.2f}s | Shape:"
        f" {df_generic.shape}"
    )

    path_target = file_paths.METRICS_DIR
    print(
        f"[{time.strftime('%H:%M:%S')}] Loading Gauged target skill from"
        f" {path_target} for metric {metric} (suffix='{exp_suffix}', lead={lead_time})..."
    )
    load_start = time.time()
    skill_target = load_experiment_metrics(
        path_target, f'gauged{exp_suffix}', metric, lead_time=lead_time
    )
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

  else:  # ungauged (supports unified 3-split or legacy 2-split generic data)
    path_generic = file_paths.UNGAUGED_GENERIC_DATA_PATH
    print(
        f"[{wall_start}] Loading Ungauged Generic Dataset from"
        f" {path_generic}..."
    )
    load_start = time.time()
    pq_generic = os.path.splitext(path_generic)[0] + '.parquet'
    if os.path.exists(pq_generic):
      df_generic_combined = pd.read_parquet(pq_generic)
    else:
      df_generic_combined = pd.read_csv(path_generic, index_col=0)
    print(
        f"   -> Loaded in {time.time() - load_start:.2f}s | Shape:"
        f" {df_generic_combined.shape}"
    )

    path_target = file_paths.METRICS_DIR
    print(
        f"[{time.strftime('%H:%M:%S')}] Splitting combined dataset and loading"
        f" target skills for {metric} (suffix='{exp_suffix}', lead={lead_time})..."
    )
    is_three_split = (
        len(df_generic_combined) % 3 == 0
        and len(df_generic_combined) > 0
        and df_generic_combined.index[: len(df_generic_combined) // 3].equals(
            df_generic_combined.index[
                len(df_generic_combined) // 3 : 2 * (len(df_generic_combined) // 3)
            ]
        )
    )
    if is_three_split:
      n_samples = len(df_generic_combined) // 3
      split_frames = [
          ('gauged', df_generic_combined.iloc[:n_samples]),
          ('random', df_generic_combined.iloc[n_samples : 2 * n_samples]),
          ('separated', df_generic_combined.iloc[2 * n_samples :]),
      ]
    else:
      n_samples = len(df_generic_combined) // 2
      split_frames = [
          ('random', df_generic_combined.iloc[:n_samples]),
          ('separated', df_generic_combined.iloc[n_samples:]),
      ]

    joined_parts = []
    for split_name, df_part_generic in split_frames:
      skill_part = load_experiment_metrics(
          path_target,
          f'{split_name}{exp_suffix}',
          metric,
          lead_time=lead_time,
      )
      df_part = pd.concat(
          [df_part_generic, skill_part], axis=1, join='inner'
      ).dropna()
      if sample_size and len(df_part) > sample_size:
        df_part = df_part.sample(n=sample_size, random_state=42)
      print(f"   -> Joined {split_name}{exp_suffix} partition | Shape: {df_part.shape}")
      joined_parts.append(df_part)

    df_model = pd.concat(joined_parts, axis=0)
    print(f"   -> Combined partitions | Shape: {df_model.shape}")

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
  if drop_record_length and 'record_length' in x.columns:
    x = x.drop(columns=['record_length'])
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
    pq_fallback = os.path.splitext(csv_fallback_path)[0] + '.parquet'
    if os.path.exists(pq_fallback):
      df = pd.read_parquet(pq_fallback)
    else:
      df = pd.read_csv(csv_fallback_path, index_col=0)
    df.index = [str(x) for x in df.index]
    return df


_SHAP_EXPLAINER_CACHE = {}


def eval_shap_chunk_from_pkl(args):
  """Worker helper for multi-process TreeSHAP evaluation via cached /tmp pickle."""
  import pickle as _pickle  # pylint: disable=g-import-not-at-top
  pkl_path, chunk_idx, chunk_df = args
  explainer = _SHAP_EXPLAINER_CACHE.get(pkl_path)
  if explainer is None:
    _SHAP_EXPLAINER_CACHE.clear()
    with open(pkl_path, 'rb') as fp:
      explainer = _pickle.load(fp)
    _SHAP_EXPLAINER_CACHE[pkl_path] = explainer
  return chunk_idx, explainer.shap_values(chunk_df, check_additivity=False)


