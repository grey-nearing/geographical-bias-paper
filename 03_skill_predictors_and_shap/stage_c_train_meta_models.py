#!/usr/bin/env python3
"""Stage C Round-2 clean meta-model training, gauge-grouped CV, and global inference.

Fixes all Round-2 statistical & modeling issues:
1. Drops `record_length` completely -> exact 95 features (`features_95.json`).
2. Strict gauge-grouped outer 10-fold CV + gauge-grouped inner 5-fold StackingRegressor CV.
3. Metric-aware bounds clipping (fixes 1.0 upper-clip bug on Beta-KGE, Alpha-NSE, Beta-NSE).
4. Harmonized 2016-2023 continuous metrics & 7-day causal RP metrics (Lead Time 1, col '1').
5. Models paired within-gauge adaptation lifts (delta_ft, delta_da, delta_ft_da) to eliminate
   cross-model subtraction artifacts (such as Bangladesh M_FT_DA < M_base).
6. Computes BOTH 3-split unified gauge-grouped CV R^2 AND 2-split ungauged-only CV R^2 for
   every metric in Table S1.
7. Generates clean baseline global predictions across all 1,034,083 HydroBASINS L12 watersheds.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
import time
import warnings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
  sys.path.insert(0, SCRIPT_DIR)

# Cap BLAS/OpenMP threads per worker BEFORE importing numpy/sklearn/torch
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '4')

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from stage_c_skill_predictor import (
    CLASSIC_METRICS,
    SkillPredictorR2,
    get_metric_bounds,
)

warnings.filterwarnings('ignore')

CANON_DIR = os.environ.get('GEO_BIAS_DATA_DIR', './data')
ROUND2_DIR = os.path.join(CANON_DIR, 'round2')
METRICS_DIR = os.path.join(ROUND2_DIR, 'hydro_model_metrics')
TRAIN_PARQUET_SRC = os.path.join(CANON_DIR, 'generic_training_data', 'ungauged_generic_training_data.parquet')
INFER_PARQUET_SRC = os.path.join(CANON_DIR, 'generic_training_data', 'hybas_generic_inference_data.parquet')
TRAIN_CSV = os.path.join(CANON_DIR, 'generic_training_data', 'ungauged_generic_training_data.csv')
INFER_CSV = os.path.join(CANON_DIR, 'generic_training_data', 'hybas_generic_inference_data.csv')

OUT_DIR = os.path.join(ROUND2_DIR, 'meta_models')
MODELS_DIR = os.path.join(OUT_DIR, 'models')
CV_DIR = os.path.join(OUT_DIR, 'cv_preds')
GLOBAL_DIR = os.path.join(OUT_DIR, 'global_preds')
INFER_PARQUET = os.path.join(OUT_DIR, 'hybas_inference_95_float32.parquet')
TRAIN_PARQUET = os.path.join(OUT_DIR, 'train_95_float32.parquet')
FEATURES_JSON = os.path.join(OUT_DIR, 'features_95.json')

SPLITS = ['gauged', 'random', 'separated']
N_GAUGES = 8811

PRIORITY1_METRICS = [
    'NSE',
    'KGE',
    'log-NSE',
    'log-KGE',
    'Beta-NSE',
    'Beta-KGE',
    'Alpha-NSE',
    'pearson-r',
    '1.01-year-2.0-window-recall',
    '2.0-year-2.0-window-recall',
    '5.0-year-2.0-window-recall',
    '10.0-year-2.0-window-recall',
    '20.0-year-2.0-window-recall',
    '50.0-year-2.0-window-recall',
    '1.01-year-2.0-window-precision',
    '2.0-year-2.0-window-precision',
    '5.0-year-2.0-window-precision',
    '10.0-year-2.0-window-precision',
    '20.0-year-2.0-window-precision',
    '50.0-year-2.0-window-precision',
    '5.0-year-2.0-window-f1',
    '10.0-year-2.0-window-f1',
    '20.0-year-2.0-window-f1',
]

PRIORITY2_METRICS = [
    '1.01-year-1.0-window-recall',
    '2.0-year-1.0-window-recall',
    '5.0-year-1.0-window-recall',
    '10.0-year-1.0-window-recall',
    '20.0-year-1.0-window-recall',
    '1.01-year-1.0-window-precision',
    '2.0-year-1.0-window-precision',
    '5.0-year-1.0-window-precision',
    '10.0-year-1.0-window-precision',
    '20.0-year-1.0-window-precision',
    '1.01-year-0.0-window-recall',
    '2.0-year-0.0-window-recall',
    '5.0-year-0.0-window-recall',
    '10.0-year-0.0-window-recall',
    '20.0-year-0.0-window-recall',
    '1.01-year-0.0-window-precision',
    '2.0-year-0.0-window-precision',
    '5.0-year-0.0-window-precision',
    '10.0-year-0.0-window-precision',
    '20.0-year-0.0-window-precision',
]

# Metrics that need full counterfactual regimes (base + delta_ft + delta_da + delta_ft_da)
FULL_REGIME_METRICS = {
    'NSE',
    'KGE',
    'log-NSE',
    'log-KGE',
    'Beta-NSE',
    'Beta-KGE',
    'Alpha-NSE',
    'pearson-r',
    '1.01-year-2.0-window-recall',
    '2.0-year-2.0-window-recall',
    '5.0-year-2.0-window-recall',
    '10.0-year-2.0-window-recall',
    '20.0-year-2.0-window-recall',
    '50.0-year-2.0-window-recall',
    '5.0-year-2.0-window-precision',
    '10.0-year-2.0-window-precision',
    '20.0-year-2.0-window-precision',
    '50.0-year-2.0-window-precision',
    '5.0-year-2.0-window-f1',
    '10.0-year-2.0-window-f1',
    '20.0-year-2.0-window-f1',
}


def prepare_canonical_feature_parquets() -> list[str]:
  """Builds cached float32 training & global inference parquets with exact 95 columns."""
  for d in [OUT_DIR, MODELS_DIR, CV_DIR, GLOBAL_DIR]:
    os.makedirs(d, exist_ok=True)

  if os.path.exists(FEATURES_JSON) and os.path.exists(TRAIN_PARQUET) and os.path.exists(INFER_PARQUET):
    with open(FEATURES_JSON) as f:
      return json.load(f)

  print('Preparing canonical 95-feature float32 parquets...', flush=True)
  df_u = (
      pd.read_parquet(TRAIN_PARQUET_SRC)
      if os.path.exists(TRAIN_PARQUET_SRC)
      else pd.read_csv(TRAIN_CSV, index_col=0)
  )
  assert len(df_u) == 3 * N_GAUGES, f'Unexpected len(df_u)={len(df_u)}'
  assert (df_u.index[:N_GAUGES] == df_u.index[N_GAUGES : 2 * N_GAUGES]).all()
  assert (df_u.index[:N_GAUGES] == df_u.index[2 * N_GAUGES :]).all()

  features_95 = [c for c in df_u.columns if c != 'record_length']
  assert len(features_95) == 95, f'Expected 95 features, got {len(features_95)}'
  df_u95 = df_u[features_95].astype(np.float32)
  assert df_u95.isna().sum().sum() == 0, 'NaNs in training features!'

  with open(FEATURES_JSON, 'w') as f:
    json.dump(features_95, f, indent=2)
  df_u95.to_parquet(TRAIN_PARQUET)
  print(f'  Saved {TRAIN_PARQUET} {df_u95.shape}', flush=True)

  df_inf = (
      pd.read_parquet(INFER_PARQUET_SRC)
      if os.path.exists(INFER_PARQUET_SRC)
      else pd.read_csv(INFER_CSV, index_col=0)
  )
  df_inf95 = df_inf[features_95].astype(np.float32)
  assert len(df_inf95) == 1034083, f'Unexpected len(df_inf95)={len(df_inf95)}'
  assert df_inf95.isna().sum().sum() == 0, 'NaNs in global inference features!'
  df_inf95.to_parquet(INFER_PARQUET)
  print(f'  Saved {INFER_PARQUET} {df_inf95.shape}', flush=True)
  return features_95


def load_3split_target(metric: str, sfx: str, slice_ids: pd.Index) -> np.ndarray:
  """Loads stacked 3-split Lead-Time-1 target array (length 26,433) aligned to slice_ids."""
  parts = []
  for split in SPLITS:
    p = os.path.join(METRICS_DIR, f'{split}{sfx}', f'{metric}.csv')
    dm = pd.read_csv(p, index_col=0)
    s = pd.to_numeric(dm['1'], errors='coerce').reindex(slice_ids)
    parts.append(s.values.astype(np.float64))
  return np.concatenate(parts, axis=0)


def outer_grouped_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    split_labels: np.ndarray,
    metric: str,
    is_delta: bool,
    n_threads: int,
    n_splits: int = 10,
) -> pd.DataFrame:
  """Runs 10-fold shuffled gauge-grouped CV and returns aligned OOF dataframe."""
  valid_mask = np.isfinite(y)
  Xv = X.iloc[valid_mask]
  yv = y[valid_mask]
  gv = groups[valid_mask]
  sv = split_labels[valid_mask]
  orig_idx = np.where(valid_mask)[0]

  uniq_g = pd.Index(groups).unique()  # Deterministic across all targets/regimes!
  kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

  oof_pred = np.full(len(yv), np.nan, dtype=np.float64)
  fold_id = np.full(len(yv), -1, dtype=np.int32)

  lo, hi = get_metric_bounds(metric)
  span = hi - lo
  yv_clip = np.clip(yv, -span, span) if is_delta else np.clip(yv, lo, hi)

  for fold_i, (_, va_u) in enumerate(kf.split(uniq_g)):
    va_set = set(uniq_g[va_u])
    is_va = np.fromiter((g in va_set for g in gv), dtype=bool, count=len(gv))
    is_tr = ~is_va
    model = SkillPredictorR2(metric=metric, is_delta=is_delta, n_threads=n_threads)
    model.fit(Xv.iloc[is_tr], pd.Series(yv_clip[is_tr], index=Xv.index[is_tr]), groups=gv[is_tr])
    preds = model.predict(Xv.iloc[is_va]).values
    oof_pred[is_va] = preds
    fold_id[is_va] = fold_i

  return pd.DataFrame({
      'row_idx': orig_idx,
      'gauge_id': gv,
      'split': sv,
      'fold': fold_id,
      'true': yv_clip,
      'true_raw': yv,
      'pred': oof_pred,
  })


def train_one_metric_task(args: tuple[str, str, int, bool]) -> dict:
  """Trains one (metric, regime) task: CV + final model + global predictions."""
  metric, regime, n_threads, overwrite = args
  t0 = time.time()

  # Ensure thread caps inside worker process
  os.environ['OMP_NUM_THREADS'] = str(n_threads)
  os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
  os.environ['MKL_NUM_THREADS'] = str(n_threads)

  summary_path = os.path.join(OUT_DIR, 'summaries', f'{regime}__{metric}.json')
  os.makedirs(os.path.dirname(summary_path), exist_ok=True)
  if (not overwrite) and os.path.exists(summary_path):
    with open(summary_path) as f:
      return json.load(f)

  df_train = pd.read_parquet(TRAIN_PARQUET)
  slice_ids = df_train.index[:N_GAUGES]
  groups = df_train.index.values
  split_labels = np.array(
      ['gauged'] * N_GAUGES + ['random'] * N_GAUGES + ['separated'] * N_GAUGES
  )

  lo, hi = get_metric_bounds(metric)
  y_base_raw = load_3split_target(metric, '', slice_ids)
  y_base_clip = np.where(np.isfinite(y_base_raw), np.clip(y_base_raw, lo, hi), np.nan)

  is_delta = regime.startswith('delta_')
  if regime == 'base':
    X_use = df_train
    y_use = y_base_raw
    g_use = groups
    s_use = split_labels
    do_global = True
    save_model = True
  elif regime == 'base_2split':
    # Pure ungauged 2-split (random + separated) CV benchmark for Table S1
    mask_u = split_labels != 'gauged'
    X_use = df_train.iloc[mask_u]
    y_use = y_base_raw[mask_u]
    g_use = groups[mask_u]
    s_use = split_labels[mask_u]
    do_global = False
    save_model = False
  elif regime in ('delta_ft', 'delta_da', 'delta_ft_da'):
    sfx = regime[len('delta'):]  # '_ft', '_da', '_ft_da'
    y_m_raw = load_3split_target(metric, sfx, slice_ids)
    y_m_clip = np.where(np.isfinite(y_m_raw), np.clip(y_m_raw, lo, hi), np.nan)
    both_ok = np.isfinite(y_base_clip) & np.isfinite(y_m_clip)
    y_use = np.where(both_ok, y_m_clip - y_base_clip, np.nan)
    X_use = df_train
    g_use = groups
    s_use = split_labels
    do_global = True
    save_model = True
  else:
    raise ValueError(f'Unknown regime: {regime}')

  # 1. Outer 10-fold gauge-grouped CV
  df_cv = outer_grouped_cv(
      X_use, y_use, g_use, s_use, metric=metric, is_delta=is_delta, n_threads=n_threads
  )
  cv_csv = os.path.join(CV_DIR, f'{regime}__{metric}.csv')
  df_cv.to_csv(cv_csv, index=False)

  r2_all = float(r2_score(df_cv['true'], df_cv['pred']))
  mae_all = float(mean_absolute_error(df_cv['true'], df_cv['pred']))
  rmse_all = float(np.sqrt(mean_squared_error(df_cv['true'], df_cv['pred'])))

  df_cv_u = df_cv[df_cv['split'] != 'gauged']
  r2_ung = float(r2_score(df_cv_u['true'], df_cv_u['pred'])) if len(df_cv_u) > 0 else np.nan
  df_cv_g = df_cv[df_cv['split'] == 'gauged']
  r2_gau = float(r2_score(df_cv_g['true'], df_cv_g['pred'])) if len(df_cv_g) > 0 else np.nan

  # 2. Fit final full model & predict all 1,034,083 global watersheds
  global_med = np.nan
  global_q25 = np.nan
  global_q75 = np.nan
  if save_model:
    valid_mask = np.isfinite(y_use)
    Xv = X_use.iloc[valid_mask]
    yv = y_use[valid_mask]
    gv = g_use[valid_mask]
    final_model = SkillPredictorR2(metric=metric, is_delta=is_delta, n_threads=n_threads)
    final_model.fit(Xv, pd.Series(yv, index=Xv.index), groups=gv)
    model_path = os.path.join(MODELS_DIR, f'{regime}__{metric}.joblib')
    joblib.dump(final_model, model_path, compress=3)

    if do_global:
      df_inf = pd.read_parquet(INFER_PARQUET)
      preds = final_model.predict(df_inf)
      pred_path = os.path.join(GLOBAL_DIR, f'{regime}__{metric}.parquet')
      pd.DataFrame({'pred': preds.values.astype(np.float32)}, index=df_inf.index).to_parquet(
          pred_path
      )
      global_q25, global_med, global_q75 = [
          float(v) for v in np.nanpercentile(preds.values, [25, 50, 75])
      ]

  res = {
      'metric': metric,
      'regime': regime,
      'is_delta': is_delta,
      'n_cv': int(len(df_cv)),
      'n_cv_ung': int(len(df_cv_u)),
      'r2_all': r2_all,
      'r2_ung': r2_ung,
      'r2_gau': r2_gau,
      'mae_all': mae_all,
      'rmse_all': rmse_all,
      'obs_mean': float(df_cv['true'].mean()),
      'obs_median': float(df_cv['true'].median()),
      'obs_ung_median': float(df_cv_u['true'].median()) if len(df_cv_u) > 0 else np.nan,
      'global_pred_median': global_med,
      'global_pred_q25': global_q25,
      'global_pred_q75': global_q75,
      'elapsed_s': float(time.time() - t0),
  }
  with open(summary_path, 'w') as f:
    json.dump(res, f, indent=2)
  return res


def compute_reconstructed_cv_summary():
  """Computes CV R^2 of reconstructed y_hat_m = clip(y_hat_base + delta_hat_m) for all deltas."""
  summary_dir = os.path.join(OUT_DIR, 'summaries')
  rows = []
  for fname in sorted(os.listdir(summary_dir)):
    if fname.endswith('.json'):
      with open(os.path.join(summary_dir, fname)) as f:
        rows.append(json.load(f))
  df_sum = pd.DataFrame(rows)

  # Add reconstructed y_m CV R^2 for every delta task
  recon_r2_all = []
  recon_r2_ung = []
  recon_neg_delta_pct = []
  for _, r in df_sum.iterrows():
    if not r['is_delta']:
      recon_r2_all.append(np.nan)
      recon_r2_ung.append(np.nan)
      recon_neg_delta_pct.append(np.nan)
      continue
    m = r['metric']
    reg = r['regime']
    base_cv_p = os.path.join(CV_DIR, f'base__{m}.csv')
    delta_cv_p = os.path.join(CV_DIR, f'{reg}__{m}.csv')
    if not (os.path.exists(base_cv_p) and os.path.exists(delta_cv_p)):
      recon_r2_all.append(np.nan)
      recon_r2_ung.append(np.nan)
      recon_neg_delta_pct.append(np.nan)
      continue
    dfb = pd.read_csv(base_cv_p)[['row_idx', 'split', 'true', 'pred']].rename(
        columns={'true': 'y_base', 'pred': 'p_base'}
    )
    dfd = pd.read_csv(delta_cv_p)[['row_idx', 'true', 'pred']].rename(
        columns={'true': 'd_true', 'pred': 'd_pred'}
    )
    mrg = dfb.merge(dfd, on='row_idx', how='inner')
    lo, hi = get_metric_bounds(m)
    y_m_true = np.clip(mrg['y_base'] + mrg['d_true'], lo, hi)
    y_m_pred = np.clip(mrg['p_base'] + mrg['d_pred'], lo, hi)
    recon_r2_all.append(float(r2_score(y_m_true, y_m_pred)))
    mu = mrg['split'] != 'gauged'
    recon_r2_ung.append(float(r2_score(y_m_true[mu], y_m_pred[mu])))
    recon_neg_delta_pct.append(float((mrg['d_pred'] < 0).mean() * 100.0))

  df_sum['recon_r2_all'] = recon_r2_all
  df_sum['recon_r2_ung'] = recon_r2_ung
  df_sum['pred_delta_neg_pct'] = recon_neg_delta_pct
  out_csv = os.path.join(OUT_DIR, 'stage_c_cv_summary.csv')
  df_sum.sort_values(['metric', 'regime']).to_csv(out_csv, index=False)
  print(f'Saved master CV summary ({len(df_sum)} rows) to {out_csv}', flush=True)
  return df_sum


def build_task_list(phase: str, n_threads: int, overwrite: bool) -> list[tuple[str, str, int, bool]]:
  tasks = []
  metrics = []
  if phase in ('1', 'all'):
    metrics.extend(PRIORITY1_METRICS)
  if phase in ('2', 'all'):
    metrics.extend(PRIORITY2_METRICS)

  for m in metrics:
    tasks.append((m, 'base', n_threads, overwrite))
    tasks.append((m, 'base_2split', n_threads, overwrite))
    tasks.append((m, 'delta_da', n_threads, overwrite))
    if m in FULL_REGIME_METRICS:
      tasks.append((m, 'delta_ft', n_threads, overwrite))
      tasks.append((m, 'delta_ft_da', n_threads, overwrite))
  return tasks


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--phase', choices=['1', '2', 'all'], default='all')
  parser.add_argument('--workers', type=int, default=12)
  parser.add_argument('--threads', type=int, default=4)
  parser.add_argument('--overwrite', action='store_true')
  args = parser.parse_args()

  t0 = time.time()
  prepare_canonical_feature_parquets()
  tasks = build_task_list(args.phase, args.threads, args.overwrite)
  print(
      f'Launching Stage C phase={args.phase}: {len(tasks)} tasks across {args.workers} workers '
      f'({args.threads} threads/worker)...',
      flush=True,
  )

  completed = 0
  with ProcessPoolExecutor(max_workers=args.workers) as ex:
    fut_map = {ex.submit(train_one_metric_task, t): t for t in tasks}
    for fut in as_completed(fut_map):
      t = fut_map[fut]
      completed += 1
      res = fut.result()
      print(
          f"[{completed:3d}/{len(tasks)}] {res['regime']:11s} | {res['metric']:32s} | "
          f"R2_all={res['r2_all']:+.4f} R2_ung={res['r2_ung']:+.4f} "
          f"g_med={res['global_pred_median'] if np.isfinite(res['global_pred_median']) else -999:.4f} "
          f"({res['elapsed_s']:.1f}s)",
          flush=True,
      )

  compute_reconstructed_cv_summary()
  print(f'Stage C phase={args.phase} ALL DONE in {(time.time() - t0) / 60:.1f} min!', flush=True)


if __name__ == '__main__':
  main()
