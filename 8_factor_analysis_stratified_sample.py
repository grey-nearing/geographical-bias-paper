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

"""Stratified SHAP analysis script for the Geographical Bias project.

This script clusters global catchments to select representative samples,
computes
exact KernelSHAP attributions using the trained skill predictor model, and
aggregates results to evaluate feature importance across metrics.
"""

import importlib
import os
import pickle
import sys
import time
import warnings
from absl import app
from absl import flags
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from utils import file_paths
from utils import io_utils
from utils.feature_utils import FEATURE_CLASSIFICATION
from utils.feature_utils import FEATURE_PALETTE
from utils.io_utils import load_csv
from utils.io_utils import load_experiment_metrics
from utils.viz_utils import save_figure
from utils import gfile_compat as gfile

# Core configuration flags
flags.DEFINE_string(
    'metric',
    'NSE',
    'The metric to analyze (e.g. NSE, 2.0-year-2.0-window-recall).',
)
flags.DEFINE_integer('shard_id', 0, 'Shard ID (0 to num_shards-1).')
flags.DEFINE_integer('num_shards', 1, 'Number of parallel shards.')
flags.DEFINE_boolean(
    'recalculate_shap', True, 'Whether to calculate SHAP values.'
)
flags.DEFINE_boolean(
    'skip_existing',
    True,
    'Whether to skip calculation if shard output already exists.',
)
flags.DEFINE_boolean(
    'aggregate_only',
    False,
    'Whether to skip calculation and only aggregate/plot results.',
)
flags.DEFINE_string(
    'shap_sampling_method',
    'stratified_1k',
    'SHAP sampling method (directory name): e.g. stratified_1k',
)
flags.DEFINE_integer(
    'sample_size',
    1000,
    'Number of stratified representative basins to select via K-Means.',
)
FLAGS = flags.FLAGS


def load_data(metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Loads training background features and country mapping."""
  print(f"[{time.strftime('%H:%M:%S')}] Loading Ungauged Generic Dataset...")
  with gfile.Open(file_paths.UNGAUGED_GENERIC_DATA_PATH, 'rt') as f:
    df_generic_combined = pd.read_csv(f, index_col=0)
  n_samples = len(df_generic_combined) // 2
  df_random_generic = df_generic_combined.iloc[:n_samples]
  df_separated_generic = df_generic_combined.iloc[n_samples:]

  print(f"[{time.strftime('%H:%M:%S')}] Loading Ungauged target skills...")
  random_skill = load_experiment_metrics(
      file_paths.METRICS_DIR, 'random', metric
  )
  separated_skill = load_experiment_metrics(
      file_paths.METRICS_DIR, 'separated', metric
  )

  df_random = pd.concat(
      [df_random_generic, random_skill], axis=1, join='inner'
  ).dropna()
  df_separated = pd.concat(
      [df_separated_generic, separated_skill], axis=1, join='inner'
  ).dropna()

  df_model = pd.concat([df_random, df_separated], axis=0)
  X_ungauged = df_model.drop(columns=['skill'])
  print(
      f"[{time.strftime('%H:%M:%S')}] Training data loaded:"
      f' {X_ungauged.shape[0]} samples, {X_ungauged.shape[1]} features'
  )

  hybas_mapping_file = file_paths.HYBAS_MAPPING_PATH
  if not gfile.Exists(hybas_mapping_file):
    raise FileNotFoundError(
        f'HyBAS country mapping file not found at {hybas_mapping_file}.'
    )
  print(f"[{time.strftime('%H:%M:%S')}] Loading saved HyBAS country mapping...")
  mapping_df = load_csv(hybas_mapping_file)

  return X_ungauged, mapping_df


def load_stratified_inference_chunk(
    shard_id: int, num_shards: int, columns: list[str], sample_size: int
) -> pd.DataFrame:
  """Clusters full features to find K representative basins and loads the shard's subset."""
  hybas_zarr_path = file_paths.HYBAS_GENERIC_DATA_ZARR_PATH
  print(
      f"[{time.strftime('%H:%M:%S')}] Loading HyBAS features from Zarr for"
      f' clustering: {hybas_zarr_path}...'
  )
  load_start = time.time()
  df_hybas_all = io_utils.load_hybas_zarr(hybas_zarr_path)
  features = df_hybas_all.values
  print(
      f'   -> Loaded all {features.shape[0]} basins in'
      f' {time.time() - load_start:.2f}s'
  )

  # Standardize and Cluster
  print(
      f"[{time.strftime('%H:%M:%S')}] Standardizing and running"
      f' MiniBatchKMeans K={sample_size}...'
  )
  t0 = time.time()
  scaler = StandardScaler()
  features_scaled = scaler.fit_transform(features)
  kmeans = MiniBatchKMeans(
      n_clusters=sample_size, batch_size=10000, random_state=42, n_init=3
  )
  kmeans.fit(features_scaled)
  print(f'   -> Clustering complete in {time.time() - t0:.2f}s')

  # Find closest basin to each cluster center
  print(f"[{time.strftime('%H:%M:%S')}] Finding representative medoids...")
  t0 = time.time()
  labels = kmeans.labels_
  centers = kmeans.cluster_centers_

  selected_indices = []
  for i in range(sample_size):
    cluster_mask = labels == i
    cluster_points = features_scaled[cluster_mask]
    cluster_indices = np.where(cluster_mask)[0]
    if len(cluster_indices) == 0:
      continue
    dists = np.linalg.norm(cluster_points - centers[i], axis=1)
    min_idx = dists.argmin()
    selected_indices.append(cluster_indices[min_idx])

  selected_indices = sorted(selected_indices)

  # Slice selected indices for this shard
  chunk_size = int(np.ceil(sample_size / num_shards))
  start_idx = shard_id * chunk_size
  end_idx = min(sample_size, (shard_id + 1) * chunk_size)
  selected_indices_shard = selected_indices[start_idx:end_idx]

  print(
      f"[{time.strftime('%H:%M:%S')}] Shard {shard_id + 1}/{num_shards} loading"
      f' {len(selected_indices_shard)} selected basins...'
  )

  df_hybas_chunk = df_hybas_all.iloc[selected_indices_shard].copy()
  df_hybas_chunk = df_hybas_chunk.reindex(columns=columns, fill_value=0)
  return df_hybas_chunk


def run_shap_calculation(
    metric: str,
    shard_id: int,
    num_shards: int,
    recalculate_shap: bool,
    skip_existing: bool,
    sampling_method: str,
    X_ungauged: pd.DataFrame,
    X_infer_hybas_chunk: pd.DataFrame,
) -> pd.DataFrame:
  """Runs exact KernelSHAP on the sliced inference data."""
  save_dir = os.path.join(file_paths.SHAP_BASE_DIR, sampling_method)
  gfile.MakeDirs(save_dir)
  save_path = os.path.join(
      save_dir, f'shard_{shard_id}_exact_kernelshap_scores.csv'
  )

  if skip_existing and gfile.Exists(save_path):
    print(
        f"[{time.strftime('%H:%M:%S')}] Shard results already exist at"
        f' {save_path}. Loading...'
    )
    with gfile.Open(save_path, 'rt') as f:
      return pd.read_csv(f, index_col=0)

  if not recalculate_shap:
    print(
        f"[{time.strftime('%H:%M:%S')}] RECALCULATE_SHAP is False. Skipping"
        ' calculation.'
    )
    return None

  # Load the pre-trained skill predictor model
  model_load_start_time = time.time()
  ungauged_model_file = file_paths.FINAL_UNGAUGED_MODEL
  with gfile.Open(ungauged_model_file, 'rb') as f:
    final_ungauged_model = pickle.load(f)
  print(
      f"[{time.strftime('%H:%M:%S')}] Loaded ungauged model in"
      f' {time.time() - model_load_start_time:.2f}s.'
  )

  X_infer_hybas_shard = X_infer_hybas_chunk
  print(
      f"[{time.strftime('%H:%M:%S')}] Shard {shard_id + 1}/{num_shards}:"
      f' Processing {len(X_infer_hybas_shard)} basins'
  )

  print(
      f"[{time.strftime('%H:%M:%S')}] Generating 100 k-means background"
      ' samples from X_ungauged...'
  )
  background = shap.kmeans(X_ungauged, 100)

  def predict_wrapper(X):
    """Helper wrapper to ensure prediction input format matches pandas DataFrame."""
    if isinstance(X, np.ndarray):
      return final_ungauged_model.predict(
          pd.DataFrame(X, columns=X_ungauged.columns)
      )
    return final_ungauged_model.predict(X)

  explainer = shap.KernelExplainer(predict_wrapper, background)

  print(
      f"[{time.strftime('%H:%M:%S')}] Calculating exact KernelSHAP for"
      f' {len(X_infer_hybas_shard)} basins...'
  )
  t0 = time.time()

  # Run calculation
  raw_shap_values = explainer.shap_values(X_infer_hybas_shard, njobs=-1)

  elapsed = time.time() - t0
  print(
      f"[{time.strftime('%H:%M:%S')}] Shard {shard_id} calculation complete in"
      f' {elapsed:.2f}s ({elapsed / max(1, len(X_infer_hybas_shard)):.2f}s per'
      ' basin).'
  )

  df_shap_scores = pd.DataFrame(
      raw_shap_values,
      index=X_infer_hybas_shard.index,
      columns=X_infer_hybas_shard.columns,
  )

  print(f"[{time.strftime('%H:%M:%S')}] Saving shard results to {save_path}...")
  with gfile.Open(save_path, 'wt') as f:
    df_shap_scores.to_csv(f, index=True)

  return df_shap_scores


def aggregate_shap_scores(
    metric: str, num_shards: int, sampling_method: str
) -> pd.DataFrame:
  """Aggregates all shard CSV files into a single DataFrame."""
  save_dir = os.path.join(file_paths.SHAP_BASE_DIR, sampling_method)
  print(
      f"\n[{time.strftime('%H:%M:%S')}] Aggregating SHAP scores from"
      f' {save_dir}...'
  )

  all_shards = []
  missing_shards = []

  for shard_id in range(num_shards):
    shard_file = os.path.join(
        save_dir, f'shard_{shard_id}_exact_kernelshap_scores.csv'
    )
    if not gfile.Exists(shard_file):
      missing_shards.append(shard_id)
      continue
    with gfile.Open(shard_file, 'rt') as f:
      all_shards.append(pd.read_csv(f, index_col=0))

  if missing_shards:
    raise FileNotFoundError(
        f'Could not aggregate SHAP scores. The following {len(missing_shards)}'
        f' shards are missing: {missing_shards}'
    )

  df_combined = pd.concat(all_shards, axis=0)
  print(
      f"[{time.strftime('%H:%M:%S')}] Successfully aggregated"
      f' {len(df_combined)} basins.'
  )

  # Save the consolidated aggregate
  aggregate_path = os.path.join(
      save_dir, f'consolidated_exact_kernelshap_scores_{metric}.csv'
  )
  with gfile.Open(aggregate_path, 'wt') as f:
    df_combined.to_csv(f, index=True)
  print(f'   -> Saved consolidated file to: {aggregate_path}')

  return df_combined


def prepare_feature_map(feature_names: list[str]) -> dict[str, str]:
  """Maps individual feature names to their corresponding categories."""
  full_feature_map = {}
  for feat in feature_names:
    if feat in FEATURE_CLASSIFICATION:
      full_feature_map[feat] = FEATURE_CLASSIFICATION[feat]
    elif (
        'p_minus_pet' in feat
        or 'p_div_pet' in feat
        or 'et_deficit' in feat
        or 'precip' in feat
    ):
      full_feature_map[feat] = 'Climate & Hydrology'
    elif 'soil_porosity' in feat:
      full_feature_map[feat] = 'Soil and Geology'
    elif 'twi_proxy' in feat or 'log_calculated_drain_area' in feat:
      full_feature_map[feat] = 'Topography'
    elif (
        'total_forest_cover' in feat
        or 'total_arid_cover' in feat
        or 'forest_aridity' in feat
    ):
      full_feature_map[feat] = 'Land Cover'
    elif (
        'human_intensity' in feat
        or 'log_pop' in feat
        or 'log_rev' in feat
        or 'log_ria' in feat
        or 'log_riv' in feat
    ):
      full_feature_map[feat] = 'Anthropogenic Factors'
    else:
      matched = False
      for base_feat in FEATURE_CLASSIFICATION:
        if base_feat in feat:
          full_feature_map[feat] = FEATURE_CLASSIFICATION[base_feat]
          matched = True
          break
      if not matched:
        full_feature_map[feat] = 'Other'
  return full_feature_map


def plot_global_importance(df_shap_scores: pd.DataFrame, save_dir: str):
  """Plots global feature importance (top 15 features)."""
  print(f"[{time.strftime('%H:%M:%S')}] Plotting global feature importance...")
  feature_map = prepare_feature_map(df_shap_scores.columns)

  mean_abs_shap = df_shap_scores.abs().mean().sort_values(ascending=False)
  df_plot = pd.DataFrame(
      {'feature': mean_abs_shap.index, 'shap': mean_abs_shap.values}
  )
  df_plot['category'] = df_plot['feature'].map(feature_map)

  plt.figure(figsize=(10, 8))
  sns.barplot(
      x='shap',
      y='feature',
      hue='category',
      data=df_plot.head(15),
      palette=FEATURE_PALETTE,
      dodge=False,
  )
  plt.title('Global Feature Importance (Mean Absolute SHAP Value)')
  plt.xlabel('Mean |SHAP Value|')
  plt.ylabel('Feature')
  plt.tight_layout()
  save_figure(plt.gcf(), save_dir, 'global_feature_importance.png')
  plt.close()


def main(argv):
  """Main execution function for sharded SHAP analysis."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  os.environ['GEOGRAPHIC_BIAS_METRIC'] = FLAGS.metric
  importlib.reload(file_paths)
  file_paths.ensure_directories_exist()

  warnings.filterwarnings('ignore', category=DeprecationWarning)
  warnings.filterwarnings('ignore', category=FutureWarning)
  sns.set_theme(style='whitegrid', palette='muted')

  X_ungauged = None
  X_infer_hybas_chunk = None
  mapping_df = None

  if not FLAGS.aggregate_only:
    # 1. Load Data
    X_ungauged, mapping_df = load_data(FLAGS.metric)
    X_infer_hybas_chunk = load_stratified_inference_chunk(
        FLAGS.shard_id, FLAGS.num_shards, X_ungauged.columns, FLAGS.sample_size
    )

    # 2. Run SHAP Calculation (for this shard)
    _ = run_shap_calculation(
        metric=FLAGS.metric,
        shard_id=FLAGS.shard_id,
        num_shards=FLAGS.num_shards,
        recalculate_shap=FLAGS.recalculate_shap,
        skip_existing=FLAGS.skip_existing,
        sampling_method=FLAGS.shap_sampling_method,
        X_ungauged=X_ungauged,
        X_infer_hybas_chunk=X_infer_hybas_chunk,
    )

    if FLAGS.num_shards > 1:
      print(
          f"[{time.strftime('%H:%M:%S')}] Shard {FLAGS.shard_id} complete."
          ' Exiting.'
      )
      sys.exit(0)

  # 3. Aggregate all SHAP scores (runs on shard 0 or when aggregate_only=True)
  _ = aggregate_shap_scores(
      metric=FLAGS.metric,
      num_shards=FLAGS.num_shards,
      sampling_method=FLAGS.shap_sampling_method,
  )

  print('Stratified SHAP evaluation complete!')


if __name__ == '__main__':
  app.run(main)
