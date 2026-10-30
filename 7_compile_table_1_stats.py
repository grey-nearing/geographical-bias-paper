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


# %%
"""Script to pre-compile summary statistics for Table 1 across all metrics.

This is run after all cross-validation and global inference models have finished.
It saves the compiled CSV to disk so notebooks can load it instantly.
"""

import os
import time
import warnings
from absl import app
from absl import flags
import pandas as pd
from sklearn.metrics import r2_score

from utils import cap_utils
from utils import file_paths
from utils.io_utils import load_experiment_metrics
from utils import gfile_compat as gfile

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

FLAGS = flags.FLAGS


def main(argv):
  """Main execution function to compile and save summary statistics."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  script_start = time.time()
  print(f"[{time.strftime('%H:%M:%S')}] Starting Table 1 pre-compilation...")
  print(f"Processing {len(file_paths.ALL_METRICS)} metrics...")

  # Optional: Load HyBAS-to-country mapping to stratify extrapolation skill by CAP system status
  hybas_has_system = None
  if gfile.Exists(file_paths.HYBAS_MAPPING_PATH):
    try:
      with gfile.Open(file_paths.HYBAS_MAPPING_PATH, 'rt') as f:
        df_map = pd.read_csv(f)
      if 'Unnamed: 0' in df_map.columns and 'hybas_id' not in df_map.columns:
        df_map = df_map.rename(columns={'Unnamed: 0': 'hybas_id'})
      if 'hybas_id' in df_map.columns and 'country_name' in df_map.columns:
        df_map['hybas_id'] = df_map['hybas_id'].astype(str)
        df_map['country_name'] = df_map['country_name'].replace(
            file_paths.country_name_map
        )
        df_map['has_system'] = df_map['country_name'].map(
            cap_utils.has_effective_flood_warning_system
        )
        hybas_has_system = df_map.drop_duplicates('hybas_id').set_index(
            'hybas_id'
        )['has_system']
    except Exception as e:
      print(f"Warning: Could not load HYBAS_MAPPING_PATH for CAP stratification: {e}")

  records = []
  processed = 0

  for metric in file_paths.ALL_METRICS:
    name = file_paths.METRIC_NAME_MAP.get(metric, metric)
    short_name = file_paths.METRIC_SHORT_NAME_MAP.get(metric, metric)

    # 1. Load training data & cross-validation predictions
    g_cv_path = os.path.join(file_paths.CV_PREDS_DIR, f'gauged_regression_cv_preds_{metric}.csv')
    u_cv_path = os.path.join(file_paths.CV_PREDS_DIR, f'ungauged_regression_cv_preds_{metric}.csv')

    n_g, r2_g, train_g_mean, train_g_med = None, None, None, None
    if gfile.Exists(g_cv_path):
      try:
        with gfile.Open(g_cv_path, 'rt') as f:
          df_g_cv = pd.read_csv(f)
        if 'true' in df_g_cv.columns and 'pred' in df_g_cv.columns:
          valid_g = df_g_cv.dropna(subset=['true', 'pred'])
          n_g = len(valid_g)
          if n_g > 0:
            train_g_mean = valid_g['true'].mean()
            train_g_med = valid_g['true'].median()
          if n_g > 1:
            r2_g = r2_score(valid_g['true'], valid_g['pred'])
      except Exception as e:
        print(f"Error loading gauged CV for {metric}: {e}")

    n_u, r2_u, train_u_mean, train_u_med = None, None, None, None
    train_u_q25, train_u_q75 = None, None
    if gfile.Exists(u_cv_path):
      try:
        with gfile.Open(u_cv_path, 'rt') as f:
          df_u_cv = pd.read_csv(f)
        if 'true' in df_u_cv.columns and 'pred' in df_u_cv.columns:
          valid_u = df_u_cv.dropna(subset=['true', 'pred'])
          n_u = len(valid_u)
          if n_u > 0:
            train_u_mean = valid_u['true'].mean()
            train_u_med = valid_u['true'].median()
            train_u_q25 = valid_u['true'].quantile(0.25)
            train_u_q75 = valid_u['true'].quantile(0.75)
          if n_u > 1:
            r2_u = r2_score(valid_u['true'], valid_u['pred'])
      except Exception as e:
        print(f"Error loading ungauged CV for {metric}: {e}")

    # 2. Load global extrapolation predictions across HyBAS level 12 basins
    g_hybas_path = os.path.join(file_paths.HYBAS_PREDS_DIR, f'global_hybas_predictions_gauged_{metric}.csv')
    u_hybas_path = os.path.join(file_paths.HYBAS_PREDS_DIR, f'global_hybas_predictions_ungauged_{metric}.csv')

    g_mean, g_med, u_mean, u_med = None, None, None, None
    u_q25, u_q75 = None, None
    delta_mean, delta_med, pct_mean, pct_med = None, None, None, None
    g_med_with_sys, u_med_with_sys, delta_med_with_sys = None, None, None
    g_med_no_sys, u_med_no_sys, delta_med_no_sys = None, None, None

    if gfile.Exists(g_hybas_path) and gfile.Exists(u_hybas_path):
      try:
        with gfile.Open(g_hybas_path, 'rt') as f:
          s_g = pd.read_csv(f, index_col=0).iloc[:, 0].dropna()
        with gfile.Open(u_hybas_path, 'rt') as f:
          s_u = pd.read_csv(f, index_col=0).iloc[:, 0].dropna()

        common_idx = s_g.index.intersection(s_u.index)
        if len(common_idx) > 0:
          sg_com = s_g.loc[common_idx]
          su_com = s_u.loc[common_idx]

          g_mean, g_med = sg_com.mean(), sg_com.median()
          u_mean, u_med = su_com.mean(), su_com.median()
          u_q25, u_q75 = su_com.quantile(0.25), su_com.quantile(0.75)

          delta = sg_com - su_com
          delta_mean, delta_med = delta.mean(), delta.median()

          valid_mask = sg_com.abs() > 1e-4
          pct = (delta[valid_mask] / sg_com[valid_mask]) * 100.0
          pct = pct.clip(lower=-1000.0, upper=1000.0)
          pct_mean, pct_med = pct.mean(), pct.median()

          if hybas_has_system is not None:
            sys_flags = hybas_has_system.reindex(
                common_idx.astype(str)
            )
            with_mask = (sys_flags == True).values
            no_mask = (sys_flags == False).values
            if with_mask.sum() > 0:
              g_med_with_sys = sg_com[with_mask].median()
              u_med_with_sys = su_com[with_mask].median()
              delta_med_with_sys = delta[with_mask].median()
            if no_mask.sum() > 0:
              g_med_no_sys = sg_com[no_mask].median()
              u_med_no_sys = su_com[no_mask].median()
              delta_med_no_sys = delta[no_mask].median()
      except Exception as e:
        print(f"Error processing global hybas for {metric}: {e}")

    # Fallback to direct evaluation metrics if global arrays are absent
    if g_mean is None:
      try:
        s_g = load_experiment_metrics(file_paths.METRICS_DIR, 'gauged', metric).dropna()
        s_u = load_experiment_metrics(file_paths.METRICS_DIR, 'random', metric).dropna()
        common_idx = s_g.index.intersection(s_u.index)
        if len(common_idx) > 0:
          sg_com = s_g.loc[common_idx]
          su_com = s_u.loc[common_idx]
          g_mean, g_med = sg_com.mean(), sg_com.median()
          u_mean, u_med = su_com.mean(), su_com.median()
          delta = sg_com - su_com
          delta_mean, delta_med = delta.mean(), delta.median()
          valid_mask = sg_com.abs() > 1e-4
          pct = (delta[valid_mask] / sg_com[valid_mask]) * 100.0
          pct = pct.clip(lower=-1000.0, upper=1000.0)
          pct_mean, pct_med = pct.mean(), pct.median()
      except Exception as e:
        print(f"Error loading direct metrics fallback for {metric}: {e}")

    records.append({
        'Metric ID': metric,
        'Metric Name': name,
        'Short Name': short_name,
        'N (Gauged)': f"{n_g:,}" if n_g is not None else 'N/A',
        'N (Ungauged)': f"{n_u:,}" if n_u is not None else 'N/A',
        'Train Mean (Gauged)': f"{train_g_mean:.3f}" if train_g_mean is not None else 'N/A',
        'Train Median (Gauged)': f"{train_g_med:.3f}" if train_g_med is not None else 'N/A',
        'Train Mean (Ungauged)': f"{train_u_mean:.3f}" if train_u_mean is not None else 'N/A',
        'Train Median (Ungauged)': f"{train_u_med:.3f}" if train_u_med is not None else 'N/A',
        'Train Q25 (Ungauged)': f"{train_u_q25:.3f}" if train_u_q25 is not None else 'N/A',
        'Train Q75 (Ungauged)': f"{train_u_q75:.3f}" if train_u_q75 is not None else 'N/A',
        'CV R2 (Gauged)': f"{r2_g:.3f}" if r2_g is not None else 'N/A',
        'CV R2 (Ungauged)': f"{r2_u:.3f}" if r2_u is not None else 'N/A',
        'Extrap Mean (Gauged)': f"{g_mean:.3f}" if g_mean is not None else 'N/A',
        'Extrap Median (Gauged)': f"{g_med:.3f}" if g_med is not None else 'N/A',
        'Extrap Mean (Ungauged)': f"{u_mean:.3f}" if u_mean is not None else 'N/A',
        'Extrap Median (Ungauged)': f"{u_med:.3f}" if u_med is not None else 'N/A',
        'Extrap Q25 (Ungauged)': f"{u_q25:.3f}" if u_q25 is not None else 'N/A',
        'Extrap Q75 (Ungauged)': f"{u_q75:.3f}" if u_q75 is not None else 'N/A',
        'Delta Skill (Mean)': f"{delta_mean:.3f}" if delta_mean is not None else 'N/A',
        'Delta Skill (Median)': f"{delta_med:.3f}" if delta_med is not None else 'N/A',
        'Delta Skill % (Mean)': f"{pct_mean:.1f}%" if pct_mean is not None else 'N/A',
        'Delta Skill % (Median)': f"{pct_med:.1f}%" if pct_med is not None else 'N/A',
        'Extrap Median Gauged (With System)': f"{g_med_with_sys:.3f}" if g_med_with_sys is not None else 'N/A',
        'Extrap Median Ungauged (With System)': f"{u_med_with_sys:.3f}" if u_med_with_sys is not None else 'N/A',
        'Delta Skill Median (With System)': f"{delta_med_with_sys:.3f}" if delta_med_with_sys is not None else 'N/A',
        'Extrap Median Gauged (Without Effective System)': f"{g_med_no_sys:.3f}" if g_med_no_sys is not None else 'N/A',
        'Extrap Median Ungauged (Without Effective System)': f"{u_med_no_sys:.3f}" if u_med_no_sys is not None else 'N/A',
        'Delta Skill Median (Without Effective System)': f"{delta_med_no_sys:.3f}" if delta_med_no_sys is not None else 'N/A',
    })

    processed += 1
    if processed % 10 == 0 or processed == len(file_paths.ALL_METRICS):
      print(f"  - [{time.strftime('%H:%M:%S')}] Processed {processed}/{len(file_paths.ALL_METRICS)} metrics...")

  df_table_1 = pd.DataFrame(records)

  # Ensure v2 statistics directory exists and write CSV
  for d in [file_paths.STATS_DIR, file_paths.FIGURES_DIR]:
    if not gfile.Exists(d):
      gfile.MakeDirs(d)
  out_path = file_paths.TABLE_1_PRECOMPUTED_STATS_PATH
  with gfile.Open(out_path, 'wt') as f:
    df_table_1.to_csv(f, index=False)

  print(f"[{time.strftime('%H:%M:%S')}] Pre-compiled summary stats saved to: {out_path}")
  print(f"Total time elapsed: {time.time() - script_start:.2f} seconds.")


if __name__ == '__main__':
  app.run(main)
