#!/usr/bin/env python3
"""Stage C Round-3 meta-model training, cross-domain spatial CV, and global inference.

Implements A3 methodological upgrades:
1. Deduplicates the 8,811 Caravan station records to 8,072 unique physical stations
   (`canonical_gauge_id`, 24,216 3-split rows) so collocated records cannot leak
   across folds.
2. Scale-normalizes `anomaly_score_ocsvm` per split by `|min|` so training and
   global inference share the exact [-1, ~+0.45] scale with Stage B1 tiers.
3. Replaces non-extrapolating 15-NN and unregularized ExtraTrees with a unified
   regularized 4-learner stack (HGB + LightGBM + XGBoost + WeightedRidgePipeline)
   with MAIN_BAS-grouped inner CV (`GroupKFold`).
4. Evaluates multi-level spatial & domain-shift CV:
   - Primary: `MAIN_BAS_5f` (5-fold major river basin GroupKFold) on all tasks.
   - Station: `Station_5f` (5-fold physical station GroupKFold) on all full tasks.
   - Cross-domain: `Climate_8f` (8-cluster hydroclimate GroupKFold) and `LOCO_5cont`
     (5-continent leave-one-continent-out CV) on core continuous & flood metrics,
     persisting full OOF residuals for Stage D cross-domain UQ propagation.
5. Generates global baseline & paired-lift predictions across all 1,034,083
   HydroBASINS L12 watersheds.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
import sys
import time
import warnings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
  sys.path.insert(0, SCRIPT_DIR)

# Cap BLAS/OpenMP threads per worker BEFORE importing numpy/sklearn
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '2')

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from stage_c_skill_predictor import (
    CLASSIC_METRICS,
    DEFAULT_REG_PRESET,
    SkillPredictorR2,
    get_metric_bounds,
)

warnings.filterwarnings('ignore')

CANON_DIR = os.environ.get('GEO_BIAS_DATA_DIR', './data')
ROUND_SUBDIR = os.environ.get('GEO_BIAS_ROUND_SUBDIR', 'round2')
ROUND2_DIR = os.path.join(CANON_DIR, ROUND_SUBDIR)
_METRICS_CAND = os.path.join(ROUND2_DIR, 'hydro_model_metrics')
METRICS_DIR = (
    _METRICS_CAND
    if os.path.exists(_METRICS_CAND)
    else os.path.join(CANON_DIR, 'round2', 'hydro_model_metrics')
)
TRAIN_PARQUET_SRC = os.path.join(
    CANON_DIR, 'generic_training_data', 'ungauged_generic_training_data.parquet'
)
INFER_PARQUET_SRC = os.path.join(
    CANON_DIR, 'generic_training_data', 'hybas_generic_inference_data.parquet'
)
TRAIN_CSV = os.path.join(
    CANON_DIR, 'generic_training_data', 'ungauged_generic_training_data.csv'
)
INFER_CSV = os.path.join(
    CANON_DIR, 'generic_training_data', 'hybas_generic_inference_data.csv'
)

OUT_DIR = os.path.join(ROUND2_DIR, 'meta_models')
MODELS_DIR = os.path.join(OUT_DIR, 'models')
CV_DIR = os.path.join(OUT_DIR, 'cv_preds')
CV_STATION_DIR = os.path.join(OUT_DIR, 'cv_preds_station')
CV_CLIMATE_DIR = os.path.join(OUT_DIR, 'cv_preds_climate')
CV_LOCO_DIR = os.path.join(OUT_DIR, 'cv_preds_loco')
GLOBAL_DIR = os.path.join(OUT_DIR, 'global_preds')
INFER_PARQUET = os.path.join(OUT_DIR, 'hybas_inference_95_float32.parquet')
TRAIN_PARQUET = os.path.join(OUT_DIR, 'train_95_float32.parquet')
META_PARQUET = os.path.join(OUT_DIR, 'train_station_metadata.parquet')
FEATURES_JSON = os.path.join(OUT_DIR, 'features_95.json')

SPLITS = ['gauged', 'random', 'separated']
RAW_N_GAUGES = 8811
N_GAUGES = 8072  # Physical deduplicated station count (verified dynamically via META_PARQUET)

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

# Core tasks that receive full Climate_8f & LOCO_5cont cross-domain CV
CROSS_DOMAIN_BASE_METRICS = {
    'NSE',
    'KGE',
    '5.0-year-2.0-window-recall',
    '10.0-year-2.0-window-recall',
    '20.0-year-2.0-window-recall',
    '50.0-year-2.0-window-recall',
    '5.0-year-2.0-window-precision',
    '10.0-year-2.0-window-precision',
    '20.0-year-2.0-window-precision',
    '50.0-year-2.0-window-precision',
}
CROSS_DOMAIN_DELTA_METRICS = {
    '5.0-year-2.0-window-recall',
    '10.0-year-2.0-window-recall',
    '20.0-year-2.0-window-recall',
    '5.0-year-2.0-window-precision',
    '10.0-year-2.0-window-precision',
    '20.0-year-2.0-window-precision',
}


def _needs_cross_domain(metric: str, regime: str) -> bool:
  if regime == 'base':
    return metric in CROSS_DOMAIN_BASE_METRICS
  if regime in ('delta_ft', 'delta_da', 'delta_ft_da'):
    if metric in CROSS_DOMAIN_DELTA_METRICS:
      return True
    if metric == 'NSE' and regime == 'delta_ft_da':
      return True
  return False


def normalize_ocsvm_series(vals: np.ndarray) -> np.ndarray:
  """Scale-normalizes OneClassSVM decision_function by |rho| = |min| into [-1, ~+0.45]."""
  arr = np.asarray(vals, dtype=np.float64)
  mn = float(np.nanmin(arr))
  if mn < -2.0:
    return (arr / abs(mn)).astype(np.float32)
  return arr.astype(np.float32)


def load_train_station_metadata() -> tuple[pd.DataFrame, int]:
  """Loads deduplicated physical station metadata (indexed by native_station_id)."""
  meta = pd.read_parquet(META_PARQUET)
  return meta, len(meta)


def prepare_canonical_feature_parquets() -> list[str]:
  """Builds deduplicated 8,072-station (24,216-row) float32 training & global inference parquets."""
  for d in [
      OUT_DIR,
      MODELS_DIR,
      CV_DIR,
      CV_STATION_DIR,
      CV_CLIMATE_DIR,
      CV_LOCO_DIR,
      GLOBAL_DIR,
  ]:
    os.makedirs(d, exist_ok=True)

  if (
      os.path.exists(FEATURES_JSON)
      and os.path.exists(TRAIN_PARQUET)
      and os.path.exists(INFER_PARQUET)
      and os.path.exists(META_PARQUET)
  ):
    with open(FEATURES_JSON) as f:
      return json.load(f)

  print(
      'Preparing canonical 95-feature float32 parquets (physical deduplication + OCSVM scale normalization)...',
      flush=True,
  )
  df_u = (
      pd.read_parquet(TRAIN_PARQUET_SRC)
      if os.path.exists(TRAIN_PARQUET_SRC)
      else pd.read_csv(TRAIN_CSV, index_col=0)
  )
  assert len(df_u) == 3 * RAW_N_GAUGES, f'Unexpected len(df_u)={len(df_u)}'
  assert (df_u.index[:RAW_N_GAUGES] == df_u.index[RAW_N_GAUGES : 2 * RAW_N_GAUGES]).all()
  assert (df_u.index[:RAW_N_GAUGES] == df_u.index[2 * RAW_N_GAUGES :]).all()

  features_95 = [c for c in df_u.columns if c != 'record_length']
  assert len(features_95) == 95, f'Expected 95 features, got {len(features_95)}'
  df_u95 = df_u[features_95].astype(np.float32).copy()
  assert df_u95.isna().sum().sum() == 0, 'NaNs in training features!'

  # 1. Scale-normalize anomaly_score_ocsvm per split (matching Stage B1 [-1, ~+0.45])
  oc_loc = df_u95.columns.get_loc('anomaly_score_ocsvm')
  for i in range(3):
    s = slice(i * RAW_N_GAUGES, (i + 1) * RAW_N_GAUGES)
    df_u95.iloc[s, oc_loc] = normalize_ocsvm_series(df_u95.iloc[s, oc_loc].values)

  # 2. Map 8,811 Caravan native_station_ids to deduplicated physical stations + MAIN_BAS + continent
  slice_ids_raw = pd.Index(df_u95.index[:RAW_N_GAUGES])
  gd = pd.read_parquet(
      os.path.join(CANON_DIR, 'attributes', 'global_gauges_deduplicated.parquet')
  )
  gs = pd.read_parquet(
      os.path.join(CANON_DIR, 'attributes', 'global_gauges_by_source_archive.parquet')
  )
  sub_car = (
      gs[
          gs['native_station_id'].isin(slice_ids_raw)
          & gs['source_dataset'].str.contains('Caravan', na=False)
      ]
      .drop_duplicates('native_station_id')
      .set_index('native_station_id')
      .reindex(slice_ids_raw)
  )
  assert sub_car['canonical_gauge_id'].notna().all(), 'Unmapped Caravan station IDs!'
  meta_raw = sub_car[['canonical_gauge_id']].merge(
      gd[['canonical_gauge_id', 'continent', 'snapped_main_bas', 'latitude', 'longitude']],
      on='canonical_gauge_id',
      how='left',
  )
  meta_raw.index = slice_ids_raw
  assert meta_raw['snapped_main_bas'].notna().all(), 'Missing snapped_main_bas!'
  assert meta_raw['continent'].notna().all(), 'Missing continent!'

  keep_first_mask = ~meta_raw['canonical_gauge_id'].duplicated(keep='first').values
  meta_kept = meta_raw.iloc[keep_first_mask].copy()
  keep_3split = np.tile(keep_first_mask, 3)
  df_u95_dedup = df_u95.iloc[keep_3split].copy()
  n_phys = len(meta_kept)

  # 3. Deterministic 8-cluster hydroclimate stratification for Climate_8f CV & Stage D bootstrap
  clim_cols = ['pre_mm_uyr', 'pet_mm_uyr', 'tmp_dc_uyr', 'ari_ix_uav', 'snw_pc_uyr']
  X_clim = StandardScaler().fit_transform(df_u95_dedup.iloc[:n_phys][clim_cols].fillna(0))
  meta_kept['climate_cluster'] = KMeans(
      n_clusters=8, random_state=42, n_init=20
  ).fit_predict(X_clim).astype(np.int32)
  meta_kept['snapped_main_bas'] = meta_kept['snapped_main_bas'].astype(np.int64)

  with open(FEATURES_JSON, 'w') as f:
    json.dump(features_95, f, indent=2)
  meta_kept.to_parquet(META_PARQUET)
  df_u95_dedup.to_parquet(TRAIN_PARQUET)
  print(
      f'  Saved {TRAIN_PARQUET} {df_u95_dedup.shape} ({n_phys:,} unique physical stations, '
      f'{meta_kept["snapped_main_bas"].nunique():,} MAIN_BAS, '
      f'{meta_kept["continent"].nunique()} continents, 8 climate clusters)',
      flush=True,
  )

  # 4. Global 1,034,083 inference matrix with normalized anomaly_score_ocsvm
  df_inf = (
      pd.read_parquet(INFER_PARQUET_SRC)
      if os.path.exists(INFER_PARQUET_SRC)
      else pd.read_csv(INFER_CSV, index_col=0)
  )
  df_inf95 = df_inf[features_95].astype(np.float32).copy()
  assert len(df_inf95) == 1034083, f'Unexpected len(df_inf95)={len(df_inf95)}'
  assert df_inf95.isna().sum().sum() == 0, 'NaNs in global inference features!'
  df_inf95['anomaly_score_ocsvm'] = normalize_ocsvm_series(
      df_inf95['anomaly_score_ocsvm'].values
  )
  df_inf95.to_parquet(INFER_PARQUET)
  print(
      f'  Saved {INFER_PARQUET} {df_inf95.shape} '
      f'(ocsvm range=[{df_inf95["anomaly_score_ocsvm"].min():.3f}, '
      f'{df_inf95["anomaly_score_ocsvm"].max():.3f}])',
      flush=True,
  )
  return features_95


def load_3split_target(metric: str, sfx: str, slice_ids: pd.Index) -> np.ndarray:
  """Loads stacked 3-split Lead-Time-1 target array aligned to deduplicated slice_ids."""
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
    outer_groups: np.ndarray,
    inner_groups: np.ndarray,
    meta_3: dict[str, np.ndarray],
    split_labels: np.ndarray,
    metric: str,
    is_delta: bool,
    n_threads: int,
    n_splits: int = 5,
    preset: str = DEFAULT_REG_PRESET,
) -> pd.DataFrame:
  """Runs deterministic GroupKFold outer CV with MAIN_BAS-grouped inner Stacking CV."""
  valid_mask = np.isfinite(y)
  Xv = X.iloc[valid_mask]
  yv = y[valid_mask]
  ogv = outer_groups[valid_mask]
  igv = inner_groups[valid_mask]
  sv = split_labels[valid_mask]
  orig_idx = np.where(valid_mask)[0]

  lo, hi = get_metric_bounds(metric)
  span = hi - lo
  yv_clip = np.clip(yv, -span, span) if is_delta else np.clip(yv, lo, hi)

  n_uniq = len(pd.Index(ogv).unique())
  k_eff = min(n_splits, max(2, n_uniq))
  gkf = GroupKFold(n_splits=k_eff)

  oof_pred = np.full(len(yv), np.nan, dtype=np.float64)
  fold_id = np.full(len(yv), -1, dtype=np.int32)

  for fold_i, (tr_i, va_i) in enumerate(gkf.split(Xv, yv_clip, ogv)):
    model = SkillPredictorR2(
        metric=metric,
        is_delta=is_delta,
        n_threads=n_threads,
        preset=preset,
        inner_splits=3,
    )
    model.fit(
        Xv.iloc[tr_i],
        pd.Series(yv_clip[tr_i], index=Xv.index[tr_i]),
        groups=igv[tr_i],
    )
    oof_pred[va_i] = model.predict(Xv.iloc[va_i]).values
    fold_id[va_i] = fold_i

  return pd.DataFrame({
      'row_idx': orig_idx,
      'gauge_id': Xv.index.values,
      'canonical_gauge_id': meta_3['canonical_gauge_id'][valid_mask],
      'main_bas': meta_3['snapped_main_bas'][valid_mask],
      'continent': meta_3['continent'][valid_mask],
      'climate_cluster': meta_3['climate_cluster'][valid_mask],
      'split': sv,
      'fold': fold_id,
      'true': yv_clip,
      'true_raw': yv,
      'pred': oof_pred,
  })


def _cv_metrics(df_cv: pd.DataFrame) -> tuple[float, float, float, float, float]:
  r2_all = float(r2_score(df_cv['true'], df_cv['pred']))
  mae_all = float(mean_absolute_error(df_cv['true'], df_cv['pred']))
  rmse_all = float(np.sqrt(mean_squared_error(df_cv['true'], df_cv['pred'])))
  df_u = df_cv[df_cv['split'] != 'gauged']
  r2_ung = float(r2_score(df_u['true'], df_u['pred'])) if len(df_u) > 0 else np.nan
  df_g = df_cv[df_cv['split'] == 'gauged']
  r2_gau = float(r2_score(df_g['true'], df_g['pred'])) if len(df_g) > 0 else np.nan
  return r2_all, r2_ung, r2_gau, mae_all, rmse_all


def train_one_metric_task(args: tuple[str, str, int, bool, str]) -> dict:
  """Trains one (metric, regime) task: multi-level CV + final model + global predictions."""
  metric, regime, n_threads, overwrite, preset = args
  t0 = time.time()

  os.environ['OMP_NUM_THREADS'] = str(n_threads)
  os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
  os.environ['MKL_NUM_THREADS'] = str(n_threads)
  os.environ['NUMEXPR_NUM_THREADS'] = str(n_threads)

  summary_path = os.path.join(OUT_DIR, 'summaries', f'{regime}__{metric}.json')
  os.makedirs(os.path.dirname(summary_path), exist_ok=True)
  if (not overwrite) and os.path.exists(summary_path):
    with open(summary_path) as f:
      return json.load(f)

  df_train = pd.read_parquet(TRAIN_PARQUET)
  meta, n_g = load_train_station_metadata()
  slice_ids = meta.index
  assert len(df_train) == 3 * n_g
  assert (df_train.index[:n_g] == slice_ids).all()

  meta_3 = {
      'canonical_gauge_id': np.tile(meta['canonical_gauge_id'].values, 3),
      'snapped_main_bas': np.tile(meta['snapped_main_bas'].values, 3),
      'continent': np.tile(meta['continent'].values, 3),
      'climate_cluster': np.tile(meta['climate_cluster'].values, 3),
  }
  split_labels = np.array(['gauged'] * n_g + ['random'] * n_g + ['separated'] * n_g)

  lo, hi = get_metric_bounds(metric)
  y_base_raw = load_3split_target(metric, '', slice_ids)
  y_base_clip = np.where(np.isfinite(y_base_raw), np.clip(y_base_raw, lo, hi), np.nan)

  is_delta = regime.startswith('delta_')
  if regime == 'base':
    mask_rows = np.ones(3 * n_g, dtype=bool)
    y_use = y_base_raw
    do_global = True
    save_model = True
  elif regime == 'base_2split':
    mask_rows = split_labels != 'gauged'
    y_use = y_base_raw[mask_rows]
    do_global = False
    save_model = False
  elif regime in ('delta_ft', 'delta_da', 'delta_ft_da'):
    sfx = regime[len('delta') :]
    y_m_raw = load_3split_target(metric, sfx, slice_ids)
    y_m_clip = np.where(np.isfinite(y_m_raw), np.clip(y_m_raw, lo, hi), np.nan)
    both_ok = np.isfinite(y_base_clip) & np.isfinite(y_m_clip)
    y_use = np.where(both_ok, y_m_clip - y_base_clip, np.nan)
    mask_rows = np.ones(3 * n_g, dtype=bool)
    do_global = True
    save_model = True
  else:
    raise ValueError(f'Unknown regime: {regime}')

  X_use = df_train.iloc[mask_rows]
  s_use = split_labels[mask_rows]
  m_use = {k: v[mask_rows] for k, v in meta_3.items()}
  mbas_use = m_use['snapped_main_bas']
  cid_use = m_use['canonical_gauge_id']

  # 1. Primary CV: 5-fold MAIN_BAS GroupKFold
  df_cv_mb = outer_grouped_cv(
      X_use,
      y_use,
      outer_groups=mbas_use,
      inner_groups=mbas_use,
      meta_3=m_use,
      split_labels=s_use,
      metric=metric,
      is_delta=is_delta,
      n_threads=n_threads,
      n_splits=5,
      preset=preset,
  )
  df_cv_mb.to_csv(os.path.join(CV_DIR, f'{regime}__{metric}.csv'), index=False)
  r2_mb_all, r2_mb_ung, r2_mb_gau, mae_mb, rmse_mb = _cv_metrics(df_cv_mb)

  # 2. Secondary CV: 5-fold Station GroupKFold (all full models)
  r2_st_all, r2_st_ung = np.nan, np.nan
  if regime != 'base_2split':
    df_cv_st = outer_grouped_cv(
        X_use,
        y_use,
        outer_groups=cid_use,
        inner_groups=mbas_use,
        meta_3=m_use,
        split_labels=s_use,
        metric=metric,
        is_delta=is_delta,
        n_threads=n_threads,
        n_splits=5,
        preset=preset,
    )
    df_cv_st.to_csv(os.path.join(CV_STATION_DIR, f'{regime}__{metric}.csv'), index=False)
    r2_st_all, r2_st_ung, _, _, _ = _cv_metrics(df_cv_st)

  # 3. Cross-domain CV: 8-fold Hydroclimate + 5-Continent LOCO (core metrics)
  r2_cl_all, r2_cl_ung = np.nan, np.nan
  r2_lo_all, r2_lo_ung = np.nan, np.nan
  if _needs_cross_domain(metric, regime):
    df_cv_cl = outer_grouped_cv(
        X_use,
        y_use,
        outer_groups=m_use['climate_cluster'],
        inner_groups=mbas_use,
        meta_3=m_use,
        split_labels=s_use,
        metric=metric,
        is_delta=is_delta,
        n_threads=n_threads,
        n_splits=8,
        preset=preset,
    )
    df_cv_cl.to_csv(os.path.join(CV_CLIMATE_DIR, f'{regime}__{metric}.csv'), index=False)
    r2_cl_all, r2_cl_ung, _, _, _ = _cv_metrics(df_cv_cl)

    df_cv_lo = outer_grouped_cv(
        X_use,
        y_use,
        outer_groups=m_use['continent'],
        inner_groups=mbas_use,
        meta_3=m_use,
        split_labels=s_use,
        metric=metric,
        is_delta=is_delta,
        n_threads=n_threads,
        n_splits=5,
        preset=preset,
    )
    df_cv_lo.to_csv(os.path.join(CV_LOCO_DIR, f'{regime}__{metric}.csv'), index=False)
    r2_lo_all, r2_lo_ung, _, _, _ = _cv_metrics(df_cv_lo)

  # 4. Fit final full model (MAIN_BAS inner CV) & predict all 1,034,083 global watersheds
  global_med = np.nan
  global_q25 = np.nan
  global_q75 = np.nan
  if save_model:
    valid_mask = np.isfinite(y_use)
    Xv = X_use.iloc[valid_mask]
    yv = y_use[valid_mask]
    gv = mbas_use[valid_mask]
    final_model = SkillPredictorR2(
        metric=metric,
        is_delta=is_delta,
        n_threads=n_threads,
        preset=preset,
        inner_splits=5,
    )
    final_model.fit(Xv, pd.Series(yv, index=Xv.index), groups=gv)
    model_path = os.path.join(MODELS_DIR, f'{regime}__{metric}.joblib')
    joblib.dump(final_model, model_path, compress=3)

    if do_global:
      df_inf = pd.read_parquet(INFER_PARQUET)
      preds = final_model.predict(df_inf)
      pred_path = os.path.join(GLOBAL_DIR, f'{regime}__{metric}.parquet')
      pd.DataFrame(
          {'pred': preds.values.astype(np.float32)}, index=df_inf.index
      ).to_parquet(pred_path)
      global_q25, global_med, global_q75 = [
          float(v) for v in np.nanpercentile(preds.values, [25, 50, 75])
      ]

  df_cv_u = df_cv_mb[df_cv_mb['split'] != 'gauged']
  res = {
      'metric': metric,
      'regime': regime,
      'preset': preset,
      'is_delta': is_delta,
      'n_phys_stations': int(n_g),
      'n_cv': int(len(df_cv_mb)),
      'n_cv_ung': int(len(df_cv_u)),
      'r2_all': r2_mb_all,
      'r2_ung': r2_mb_ung,
      'r2_gau': r2_mb_gau,
      'r2_mainbas_all': r2_mb_all,
      'r2_mainbas_ung': r2_mb_ung,
      'r2_station_all': r2_st_all,
      'r2_station_ung': r2_st_ung,
      'r2_climate_all': r2_cl_all,
      'r2_climate_ung': r2_cl_ung,
      'r2_loco_all': r2_lo_all,
      'r2_loco_ung': r2_lo_ung,
      'mae_all': mae_mb,
      'rmse_all': rmse_mb,
      'obs_mean': float(df_cv_mb['true'].mean()),
      'obs_median': float(df_cv_mb['true'].median()),
      'obs_ung_median': float(df_cv_u['true'].median()) if len(df_cv_u) > 0 else np.nan,
      'global_pred_median': global_med,
      'global_pred_q25': global_q25,
      'global_pred_q75': global_q75,
      'elapsed_s': float(time.time() - t0),
  }
  with open(summary_path, 'w') as f:
    json.dump(res, f, indent=2)
  return res


def _recon_pair_r2(cv_dir: str, m: str, reg: str) -> tuple[float, float, float]:
  base_p = os.path.join(cv_dir, f'base__{m}.csv')
  delta_p = os.path.join(cv_dir, f'{reg}__{m}.csv')
  if not (os.path.exists(base_p) and os.path.exists(delta_p)):
    return np.nan, np.nan, np.nan
  dfb = pd.read_csv(base_p)[['row_idx', 'split', 'true', 'pred']].rename(
      columns={'true': 'y_base', 'pred': 'p_base'}
  )
  dfd = pd.read_csv(delta_p)[['row_idx', 'true', 'pred']].rename(
      columns={'true': 'd_true', 'pred': 'd_pred'}
  )
  mrg = dfb.merge(dfd, on='row_idx', how='inner')
  lo, hi = get_metric_bounds(m)
  y_m_true = np.clip(mrg['y_base'] + mrg['d_true'], lo, hi)
  y_m_pred = np.clip(mrg['p_base'] + mrg['d_pred'], lo, hi)
  r2_a = float(r2_score(y_m_true, y_m_pred))
  mu = mrg['split'] != 'gauged'
  r2_u = float(r2_score(y_m_true[mu], y_m_pred[mu])) if mu.any() else np.nan
  neg_pct = float((mrg['d_pred'] < 0).mean() * 100.0)
  return r2_a, r2_u, neg_pct


def compute_reconstructed_cv_summary():
  """Computes CV R^2 of reconstructed y_hat_m = clip(y_hat_base + delta_hat_m) across all CV levels."""
  summary_dir = os.path.join(OUT_DIR, 'summaries')
  rows = []
  for fname in sorted(os.listdir(summary_dir)):
    if fname.endswith('.json'):
      with open(os.path.join(summary_dir, fname)) as f:
        rows.append(json.load(f))
  df_sum = pd.DataFrame(rows)

  recon_mb_all, recon_mb_ung, neg_pcts = [], [], []
  recon_st_all, recon_cl_all, recon_lo_all = [], [], []
  for _, r in df_sum.iterrows():
    if not r['is_delta']:
      recon_mb_all.append(np.nan)
      recon_mb_ung.append(np.nan)
      neg_pcts.append(np.nan)
      recon_st_all.append(np.nan)
      recon_cl_all.append(np.nan)
      recon_lo_all.append(np.nan)
      continue
    m, reg = r['metric'], r['regime']
    r_ma, r_mu, npct = _recon_pair_r2(CV_DIR, m, reg)
    r_sa, _, _ = _recon_pair_r2(CV_STATION_DIR, m, reg)
    r_ca, _, _ = _recon_pair_r2(CV_CLIMATE_DIR, m, reg)
    r_la, _, _ = _recon_pair_r2(CV_LOCO_DIR, m, reg)
    recon_mb_all.append(r_ma)
    recon_mb_ung.append(r_mu)
    neg_pcts.append(npct)
    recon_st_all.append(r_sa)
    recon_cl_all.append(r_ca)
    recon_lo_all.append(r_la)

  df_sum['recon_r2_all'] = recon_mb_all
  df_sum['recon_r2_ung'] = recon_mb_ung
  df_sum['recon_r2_mainbas_all'] = recon_mb_all
  df_sum['recon_r2_station_all'] = recon_st_all
  df_sum['recon_r2_climate_all'] = recon_cl_all
  df_sum['recon_r2_loco_all'] = recon_lo_all
  df_sum['pred_delta_neg_pct'] = neg_pcts
  out_csv = os.path.join(OUT_DIR, 'stage_c_cv_summary.csv')
  df_sum.sort_values(['metric', 'regime']).to_csv(out_csv, index=False)
  print(f'Saved master CV summary ({len(df_sum)} rows) to {out_csv}', flush=True)
  return df_sum


def build_task_list(
    phase: str, n_threads: int, overwrite: bool, preset: str
) -> list[tuple[str, str, int, bool, str]]:
  tasks = []
  metrics = []
  if phase in ('1', 'all'):
    metrics.extend(PRIORITY1_METRICS)
  if phase in ('2', 'all'):
    metrics.extend(PRIORITY2_METRICS)

  for m in metrics:
    tasks.append((m, 'base', n_threads, overwrite, preset))
    tasks.append((m, 'base_2split', n_threads, overwrite, preset))
    tasks.append((m, 'delta_da', n_threads, overwrite, preset))
    if m in FULL_REGIME_METRICS:
      tasks.append((m, 'delta_ft', n_threads, overwrite, preset))
      tasks.append((m, 'delta_ft_da', n_threads, overwrite, preset))

  # Sort heaviest tasks FIRST (cross-domain CV -> full models -> base_2split) to avoid tail stragglers
  def _priority(t):
    m, reg = t[0], t[1]
    if _needs_cross_domain(m, reg):
      return 0
    if reg != 'base_2split':
      return 1
    return 2

  tasks.sort(key=_priority)
  return tasks


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--phase', choices=['1', '2', 'all'], default='all')
  parser.add_argument('--workers', type=int, default=16)
  parser.add_argument('--threads', type=int, default=2)
  parser.add_argument('--preset', choices=['v1', 'v2_strongreg'], default=DEFAULT_REG_PRESET)
  parser.add_argument('--overwrite', action='store_true')
  args = parser.parse_args()

  t0 = time.time()
  prepare_canonical_feature_parquets()
  tasks = build_task_list(args.phase, args.threads, args.overwrite, args.preset)
  print(
      f'Launching Stage C phase={args.phase} (round={ROUND_SUBDIR}, preset={args.preset}): '
      f'{len(tasks)} tasks across {args.workers} workers ({args.threads} threads/worker)...',
      flush=True,
  )

  os.environ['OMP_NUM_THREADS'] = str(args.threads)
  os.environ['OPENBLAS_NUM_THREADS'] = str(args.threads)
  os.environ['MKL_NUM_THREADS'] = str(args.threads)
  os.environ['NUMEXPR_NUM_THREADS'] = str(args.threads)

  completed = 0
  with ProcessPoolExecutor(
      max_workers=args.workers, mp_context=mp.get_context('spawn')
  ) as ex:
    fut_map = {ex.submit(train_one_metric_task, t): t for t in tasks}
    for fut in as_completed(fut_map):
      completed += 1
      res = fut.result()
      cl_s = f" Cl={res['r2_climate_all']:+.3f}" if np.isfinite(res.get('r2_climate_all', np.nan)) else ''
      lo_s = f" LO={res['r2_loco_all']:+.3f}" if np.isfinite(res.get('r2_loco_all', np.nan)) else ''
      st_s = f" St={res['r2_station_all']:+.3f}" if np.isfinite(res.get('r2_station_all', np.nan)) else ''
      print(
          f"[{completed:3d}/{len(tasks)}] {res['regime']:11s} | {res['metric']:32s} | "
          f"MB={res['r2_mainbas_all']:+.4f}{st_s}{cl_s}{lo_s} ({res['elapsed_s']:.1f}s)",
          flush=True,
      )

  compute_reconstructed_cv_summary()
  print(
      f'Stage C phase={args.phase} ALL DONE in {(time.time() - t0) / 60:.1f} min!',
      flush=True,
  )


if __name__ == '__main__':
  main()
