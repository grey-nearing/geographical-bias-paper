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

# Author: Jetski AI Agent
# Created: 2026-07-22

"""Pre-filtering logic to select candidate streamflow gauge locations.

This script implements two selection strategies to identify a subset of global
Level 12 HydroBASINS that are the most promising candidates for gauge placement
evaluations:
1. Representativeness Bucket: Identifies physically dense clusters of ungauged
   basins that are poorly represented in Caravan (high anomaly scores).
2. Exposure Buckets: Identifies the top 100 highest exposure basins globally,
   and the top 20 highest exposure basins per country.
"""

import os
import re
import time
from absl import app
from absl import flags
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import zarr

from utils import file_paths
from utils import io_utils
from utils import stats_utils
from utils import gfile_compat as gfile

FLAGS = flags.FLAGS

def _define_flags():
  if 'output_dir' not in FLAGS:
    flags.DEFINE_string(
        'output_dir',
        os.path.join(file_paths.V2_RESULTS_DIR, 'candidates'),
        'Directory to write candidate lists.',
    )
  if 'n_candidates' not in FLAGS:
    flags.DEFINE_integer(
        'n_candidates',
        2000,
        'Target number of total candidates (approximately).',
    )
  if 'n_clusters' not in FLAGS:
    flags.DEFINE_integer(
        'n_clusters',
        2000,
        'Number of KMeans clusters to group the global basin physical space into.',
    )
  if 'metric' not in FLAGS:
    flags.DEFINE_string(
        'metric',
        '5.0-year-2.0-window-recall',
        'Metric used to pull prediction anomalies and fallback exposure.',
    )


if __name__ == '__main__':
  _define_flags()

NON_PHYSICAL_MODEL_COLS = {
    'anomaly_score_if',
    'anomaly_score_ocsvm',
    'anomaly_score_lof',
    'total_upstream_coverage',
    'max_upstream_coverage',
    'max_downstream_coverage',
    'upstream_gauge_count',
    'record_length',
}


def load_global_data():
  print(f"[{time.strftime('%H:%M:%S')}] Loading base physical data...")
  df_base = io_utils.load_hybas_zarr(file_paths.HYBAS_GENERIC_DATA_ZARR_PATH)
  df_base.index = [
      idx if str(idx).startswith('hybas_') else 'hybas_' + str(idx)
      for idx in df_base.index
  ]
  physical_cols = [
      col
      for col in df_base.columns
      if not col.startswith('candidate_')
      and not col.startswith('eval_')
      and col.lower() not in file_paths.BANNED_SPATIAL_COLS
      and col not in NON_PHYSICAL_MODEL_COLS
  ]
  return df_base, physical_cols


def generate_representativeness_bucket(
    df_base, physical_cols, n_candidates, n_clusters
):
  """Selects physically out-of-distribution (OOD) ungauged cluster centroids.

  Note on sign convention (see `1_anomaly_scores.py`): scikit-learn's
  `decision_function` returns LOWER (more negative) values for MORE anomalous
  (out-of-distribution) basins and HIGHER (positive) values for in-distribution
  basins. We therefore sort ASCENDING and select `idxmin()` per cluster.
  """
  print(f"[{time.strftime('%H:%M:%S')}] Generating OOD Representativeness Bucket...")
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(df_base[physical_cols].fillna(0.0))

  kmeans = MiniBatchKMeans(
      n_clusters=n_clusters,
      random_state=42,
      batch_size=10000,
      max_no_improvement=20,
  )
  cluster_labels = kmeans.fit_predict(X_scaled)

  df_clusters = pd.DataFrame({'cluster': cluster_labels}, index=df_base.index)
  if 'anomaly_score_ocsvm' in df_base.columns:
    # Standardized composite OOD score across OCSVM, IF, and LOF (lower = more OOD)
    s_oc = (
        df_base['anomaly_score_ocsvm'] - df_base['anomaly_score_ocsvm'].mean()
    ) / (df_base['anomaly_score_ocsvm'].std() + 1e-8)
    s_if = (
        df_base['anomaly_score_if'] - df_base['anomaly_score_if'].mean()
    ) / (df_base['anomaly_score_if'].std() + 1e-8)
    df_clusters['anomaly_score'] = 0.7 * s_oc + 0.3 * s_if
  else:
    anomaly_csv = file_paths.INFER_HYBAS_CARAVAN_OUT_PATH
    if gfile.Exists(anomaly_csv):
      with gfile.Open(anomaly_csv, 'rt') as f:
        df_scores = pd.read_csv(f, index_col='hybas_id')
      df_scores.index = [
          idx if idx.startswith('hybas_') else 'hybas_' + idx
          for idx in df_scores.index
      ]
      df_clusters['anomaly_score'] = df_scores['anomaly_score_ocsvm'].reindex(
          df_clusters.index, fill_value=0.0
      )
    else:
      df_clusters['anomaly_score'] = 0.0

  # Sort ASCENDING: lowest decision_function = highest multivariate OOD anomaly
  cluster_stats = (
      df_clusters.groupby('cluster')['anomaly_score']
      .mean()
      .sort_values(ascending=True)
  )
  target_clusters = cluster_stats.head(n_candidates).index

  candidates = []
  for cluster_id in target_clusters:
    cluster_basins = df_clusters[df_clusters['cluster'] == cluster_id]
    best_candidate = cluster_basins['anomaly_score'].idxmin()
    if pd.notna(best_candidate):
      candidates.append(best_candidate)

  return candidates


def generate_exposure_buckets(df_base):
  print(f"[{time.strftime('%H:%M:%S')}] Generating Exposure & Exposure-OOD Buckets...")
  hybas_ids = list(df_base.index)
  hybas_ids_set = set(hybas_ids)

  # Load mapping for countries
  with gfile.Open(file_paths.HYBAS_MAPPING_PATH, 'rt') as f:
    mapping_df = pd.read_csv(f, index_col='hybas_id')
  mapping_df.index = [
      idx if idx.startswith('hybas_') else 'hybas_' + idx
      for idx in mapping_df.index
  ]

  # Load Exposure
  return_periods = [5, 10, 20, 50, 100]
  exposures_df = pd.DataFrame(index=hybas_ids)
  with gfile.Open(file_paths.GLOBAL_BASIN_EXPOSURE_BY_RP_PATH, 'rt') as f:
    exp_df = pd.read_csv(f, index_col='hybas_id')
  idx_series = exp_df.index.astype(str).str.replace(r'\.0$', '', regex=True)
  exp_df.index = np.where(
      idx_series.str.startswith('hybas_'), idx_series, 'hybas_' + idx_series
  )
  for T in return_periods:
    col_name = f'exposed_pop_{T}'
    if col_name in exp_df.columns:
      exposures_df[T] = exp_df.loc[
          exp_df.index.isin(hybas_ids_set), col_name
      ].reindex(hybas_ids, fill_value=0.0)
    else:
      exposures_df[T] = 0.0

  annual_exposure = stats_utils.calculate_trapezoidal_expected_annual_risk(
      exposures_df, return_periods=return_periods
  )
  annual_exposure = pd.Series(annual_exposure, index=hybas_ids)

  # Compute OOD weight (excess OOD distance relative to 90th-percentile in-distribution Caravan reference)
  if 'anomaly_score_ocsvm' in df_base.columns:
    oc = df_base['anomaly_score_ocsvm']
    q90_in_dist = float(oc.quantile(0.90))
    ood_deficit = (q90_in_dist - oc).clip(lower=0.0)
  else:
    ood_deficit = pd.Series(1.0, index=hybas_ids)

  df_comb = pd.DataFrame({
      'exposure': annual_exposure,
      'ood_weighted_exposure': annual_exposure * ood_deficit,
  })
  df_comb = df_comb.join(mapping_df, how='left')

  df_by_exp = df_comb.sort_values(by='exposure', ascending=False)
  global_1000 = list(df_by_exp.head(1000).index)

  df_by_exp_ood = df_comb.sort_values(
      by='ood_weighted_exposure', ascending=False
  )
  global_exp_ood_1000 = list(df_by_exp_ood.head(1000).index)

  country_top_20 = []
  for _, group in df_by_exp.groupby('country_name'):
    country_top_20.extend(list(group.head(20).index))

  return global_1000, global_exp_ood_1000, country_top_20


def main(argv):
  del argv
  if not gfile.Exists(FLAGS.output_dir):
    gfile.MakeDirs(FLAGS.output_dir)

  df_base, physical_cols = load_global_data()
  half_candidates = FLAGS.n_candidates // 2

  rep_candidates = generate_representativeness_bucket(
      df_base=df_base,
      physical_cols=physical_cols,
      n_candidates=half_candidates,
      n_clusters=FLAGS.n_clusters,
  )

  (
      global_1000_candidates,
      global_exp_ood_1000_candidates,
      country_20_candidates,
  ) = generate_exposure_buckets(df_base=df_base)

  combined = sorted(
      set(
          rep_candidates
          + global_1000_candidates
          + global_exp_ood_1000_candidates
          + country_20_candidates
      )
  )

  # Save buckets cleanly isolated in v2_exposure_blockers/candidates/
  with gfile.Open(
      os.path.join(FLAGS.output_dir, 'representativeness_candidates.csv'), 'w'
  ) as f:
    pd.DataFrame({'hybas_id': rep_candidates}).to_csv(f, index=False)

  with gfile.Open(
      os.path.join(FLAGS.output_dir, 'exposure_global_1000_candidates.csv'), 'w'
  ) as f:
    pd.DataFrame({'hybas_id': global_1000_candidates}).to_csv(f, index=False)

  with gfile.Open(
      os.path.join(FLAGS.output_dir, 'exposure_ood_1000_candidates.csv'), 'w'
  ) as f:
    pd.DataFrame({'hybas_id': global_exp_ood_1000_candidates}).to_csv(
        f, index=False
    )

  with gfile.Open(
      os.path.join(FLAGS.output_dir, 'exposure_country_top_20_candidates.csv'),
      'w',
  ) as f:
    pd.DataFrame({'hybas_id': country_20_candidates}).to_csv(f, index=False)

  combined_path = os.path.join(FLAGS.output_dir, 'combined_candidates.csv')
  with gfile.Open(combined_path, 'w') as f:
    pd.DataFrame({'hybas_id': combined}).to_csv(f, index=False)

  print(
      f"[{time.strftime('%H:%M:%S')}] Saved {len(combined)} unique v2"
      f' candidates to {combined_path}'
  )


if __name__ == '__main__':
  app.run(main)
