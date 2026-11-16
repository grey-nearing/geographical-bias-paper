#!/usr/bin/env python3
"""Stage D: Round-2 Counterfactuals, Table 1/S1/S3, Precision/F1/CSI/alpha Sensitivity,
B=20 Gauge-Clustered Bootstrap 95% CIs, and Unified S1.6 (Fig. S8) Gauge Siting.

Operates on ALL 1,034,083 global HydroBASINS L12 watersheds (9 continents), fixed
Round-2 Fathom v3.1 x WorldPop 2020 exposure (w_100 = 0.015, sum w = 0.200),
95-feature Round-2 paired-lift SkillPredictorR2 models, pure feature-space Q75
Universal OOD, and strict physical G-set (`snap_active` / `snap_all`) as primary
alongside `union` and `ge95` sensitivity columns.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
import sys
import time
import warnings

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '4')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.environ.get('GEO_BIAS_REPO_DIR', os.path.dirname(SCRIPT_DIR))
for d in (SCRIPT_DIR, REPO_DIR):
  if d not in sys.path:
    sys.path.insert(0, d)

CANON_DIR = os.environ.get('GEO_BIAS_DATA_DIR', './data')
os.environ.setdefault('GEO_BIAS_DATA_DIR', CANON_DIR)
os.environ.setdefault('GEO_BIAS_THIRD_PARTY_DIR', CANON_DIR)
os.environ.setdefault('GEO_BIAS_OUTPUT_DIR', CANON_DIR)

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from utils import cap_utils, file_paths
from stage_c_skill_predictor import (
    CLASSIC_METRICS,
    SkillPredictorR2,
    get_metric_bounds,
)
from stage_c_train_meta_models import (
    FULL_REGIME_METRICS,
    N_GAUGES,
    PRIORITY1_METRICS,
    PRIORITY2_METRICS,
    SPLITS,
    load_3split_target,
)

warnings.filterwarnings('ignore')

ROUND2_DIR = os.path.join(CANON_DIR, 'round2')
META_DIR = os.path.join(ROUND2_DIR, 'meta_models')
MODELS_DIR = os.path.join(META_DIR, 'models')
GLOBAL_PREDS_DIR = os.path.join(META_DIR, 'global_preds')
CV_DIR = os.path.join(META_DIR, 'cv_preds')
TIER_DIR = os.path.join(ROUND2_DIR, 'tier_features')
EXP_PARQUET = os.path.join(ROUND2_DIR, 'exposure', 'global_basin_exposure_defended_r2.parquet')
INFER_PARQUET = os.path.join(META_DIR, 'hybas_inference_95_float32.parquet')
TRAIN_PARQUET = os.path.join(META_DIR, 'train_95_float32.parquet')
FEATURES_JSON = os.path.join(META_DIR, 'features_95.json')

OUT_DIR = os.path.join(ROUND2_DIR, 'counterfactuals')
PREDS_CACHE_DIR = os.path.join(OUT_DIR, 'tier_preds')
BOOT_DIR = os.path.join(OUT_DIR, 'bootstrap')
S16_DIR = os.path.join(ROUND2_DIR, 's16_gauge_placement')

# Exact trapezoidal exceedance weights for T in {5, 10, 20, 50, 100} yr (sum = 0.200)
# Including tail p < p_5 = 0.01 yr^-1: w_100 = 0.5*(0.02 - 0.01) + 0.01 = 0.015!
RP_WEIGHTS = {5: 0.05, 10: 0.075, 20: 0.04, 50: 0.02, 100: 0.015}
assert abs(sum(RP_WEIGHTS.values()) - 0.200) < 1e-12

COV_COLS = [
    'total_upstream_coverage',
    'max_upstream_coverage',
    'max_downstream_coverage',
    'upstream_gauge_count',
]
OOD_COLS = [
    'anomaly_score_if',
    'anomaly_score_ocsvm',
    'anomaly_score_lof',
]

TIER_ORDER = [
    'tier0_zero_gauges',
    'tier1a_active_post2016_ccby_8811',
    'tier1b_strict_ccby_19491',
    'tier1b_full_caravan_24847',
    'tier2a_caravan_estreams_32713',
    'tier2_academic_compilations_38069',
    'tier_caravan_estreams_34577',
    'tier2_all_open_academic_34630',
    'tier_public_discharge_93827',
    'tier3_open_public_apis_and_bulk_106410',
    'tier4_all_publicly_knowable_121819',
    'tier5_universal_idealized_ceiling',
    'tier5_dense_corridor_ceiling',
]

TIER_LABELS = {
    'tier0_zero_gauges': 'Zero-Gauge Counterfactual (0)',
    'tier1a_active_post2016_ccby_8811': 'Baseline: Active Post-2016 Caravan (8,811 / 8,075 phys)',
    'tier1b_strict_ccby_19491': 'Strict CC-BY-4.0 Regional Caravan (19,491 / 18,037 phys)',
    'tier1b_full_caravan_24847': 'Full Caravan Compendium (24,847 / 22,638 phys)',
    'tier2a_caravan_estreams_32713': 'Caravan + EStreams Archive (32,713 / 27,876 phys)',
    'tier2_academic_compilations_38069': 'Caravan + EStreams + GRDC (38,069 / 32,174 phys)',
    'tier_caravan_estreams_34577': 'Tier 1: Standardized Open Compilations (34,577 phys)',
    'tier2_all_open_academic_34630': 'All Open Academic Compilations (34,630 phys)',
    'tier_public_discharge_93827': 'Tier 2: Publicly Accessible Discharge Records (93,827 phys)',
    'tier3_open_public_apis_and_bulk_106410': 'Open Public APIs & Bulk Portals incl. Stage (106,410 phys)',
    'tier4_all_publicly_knowable_121819': 'Tier 3: All Publicly Knowable In Situ Gauges (121,819 phys)',
    'tier5_universal_idealized_ceiling': 'Universal Gauging (1,034,083 watersheds + Q75 Climate)',
    'tier5_dense_corridor_ceiling': 'Dense Corridor Ceiling (Q75 Multi-Gauge + Q75 Climate)',
}

EVAL_RP_METRICS = [
    ('r5', '5.0-year-2.0-window-recall'),
    ('r10', '10.0-year-2.0-window-recall'),
    ('r20', '20.0-year-2.0-window-recall'),
    ('r50', '50.0-year-2.0-window-recall'),
    ('p5', '5.0-year-2.0-window-precision'),
    ('p10', '10.0-year-2.0-window-precision'),
    ('p20', '20.0-year-2.0-window-precision'),
    ('f1_5', '5.0-year-2.0-window-f1'),
    ('f1_10', '10.0-year-2.0-window-f1'),
    ('f1_20', '20.0-year-2.0-window-f1'),
]
EVAL_CLASSIC_METRICS = [
    ('NSE', 'NSE'),
    ('KGE', 'KGE'),
]


def _norm_hybas(s: pd.Series) -> pd.Series:
  s2 = s.astype(str).str.replace(r'\.0$', '', regex=True)
  return np.where(s2.str.startswith('hybas_'), s2, 'hybas_' + s2)


def load_global_metadata_non_fathom(all_ids: pd.Index) -> pd.DataFrame:
  """Loads public 1,034,083 basin metadata (country, CAP status, lat/lon, sub_area) without Fathom data."""
  file_paths.init('5.0-year-2.0-window-recall')
  df = pd.DataFrame(index=all_ids)

  df_map = pd.read_csv(file_paths.HYBAS_MAPPING_PATH)
  idc = 'hybas_id' if 'hybas_id' in df_map.columns else df_map.columns[0]
  df_map['hid'] = _norm_hybas(df_map[idc])
  df_map['country_name_clean'] = df_map['country_name'].replace(file_paths.country_name_map)
  df_map = df_map.drop_duplicates('hid').set_index('hid')

  df['country_name'] = df_map['country_name_clean'].reindex(all_ids)
  df['has_system'] = (
      df['country_name'].map(cap_utils.has_effective_flood_warning_system).fillna(False).astype(bool)
  )
  df['cap_tier'] = df['country_name'].map(cap_utils.assign_cap_tier).fillna(cap_utils.TIER_4_LABEL)

  df_ll = pd.read_csv(file_paths.HYBAS_LATLON_PATH, index_col=0)
  lat_c = next(c for c in df_ll.columns if 'lat' in c.lower())
  lon_c = next(c for c in df_ll.columns if 'lon' in c.lower())
  df_ll.index = _norm_hybas(pd.Series(df_ll.index))
  df_ll = df_ll[~df_ll.index.duplicated(keep='first')].reindex(all_ids)
  df['lat'] = pd.to_numeric(df_ll[lat_c], errors='coerce')
  df['lon'] = pd.to_numeric(df_ll[lon_c], errors='coerce')

  topo_pq = os.path.join(CANON_DIR, 'attributes', 'hybas_l12_topology_and_runoff.parquet')
  if os.path.exists(topo_pq):
    df_att = pd.read_parquet(topo_pq, columns=['HYBAS_ID', 'SUB_AREA'])
    df_att.index = _norm_hybas(df_att['HYBAS_ID'])
    df_att = df_att[~df_att.index.duplicated(keep='first')].reindex(all_ids)
    df['sub_area'] = pd.to_numeric(df_att['SUB_AREA'], errors='coerce').fillna(0.0)
  else:
    df_att = pd.read_csv(
        file_paths.HYBAS_ATTRIBUTES_PATH,
        usecols=['Unnamed: 0', 'HYDRO_ATLAS:SUB_AREA'],
        index_col='Unnamed: 0',
    )
    df_att.index = _norm_hybas(pd.Series(df_att.index))
    df_att = df_att[~df_att.index.duplicated(keep='first')].reindex(all_ids)
    df['sub_area'] = pd.to_numeric(df_att['HYDRO_ATLAS:SUB_AREA'], errors='coerce').fillna(0.0)
  assert 1.2e8 < df['sub_area'].sum() < 1.5e8, f"Unexpected global land area: {df['sub_area'].sum()}"
  return df


def load_global_metadata_and_exposure(all_ids: pd.Index) -> pd.DataFrame:
  """Loads fixed Round-2 1,034,083 exposure (>0.1m & >0.0m), country, CAP, lat/lon."""
  df = load_global_metadata_non_fathom(all_ids)
  df_e = pd.read_parquet(EXP_PARQUET)
  if 'hybas_id' in df_e.columns:
    df_e['hybas_id'] = _norm_hybas(df_e['hybas_id'])
    df_e = df_e.drop_duplicates('hybas_id').set_index('hybas_id')
  else:
    df_e.index = _norm_hybas(pd.Series(df_e.index))
  df_e = df_e.reindex(all_ids).fillna(0.0)

  df['total_pop'] = df_e['total_population'].astype(np.float64)
  for T in (5, 10, 20, 50, 100):
    c10 = f'exposed_pop_gt10cm_{T}' if f'exposed_pop_gt10cm_{T}' in df_e.columns else f'exposed_pop_{T}'
    c0 = f'exposed_pop_gt0cm_{T}' if f'exposed_pop_gt0cm_{T}' in df_e.columns else f'exposed_pop_gt0_{T}'
    df[f'pop_{T}'] = df_e[c10].astype(np.float64)
    df[f'pop_gt0_{T}'] = df_e[c0].astype(np.float64) if c0 in df_e.columns else df[f'pop_{T}']

  # Exact trapezoidal EAE weights (sum = 0.200)
  df['E_5'] = RP_WEIGHTS[5] * df['pop_5']
  df['E_10'] = RP_WEIGHTS[10] * df['pop_10']
  df['E_20'] = RP_WEIGHTS[20] * df['pop_20']
  df['E_50'] = RP_WEIGHTS[50] * df['pop_50']
  df['E_100'] = RP_WEIGHTS[100] * df['pop_100']
  df['E_20_50_100'] = df['E_20'] + df['E_50'] + df['E_100']
  df['E_total'] = df['E_5'] + df['E_10'] + df['E_20_50_100']

  df['E_gt0_5'] = RP_WEIGHTS[5] * df['pop_gt0_5']
  df['E_gt0_10'] = RP_WEIGHTS[10] * df['pop_gt0_10']
  df['E_gt0_20_50_100'] = (
      RP_WEIGHTS[20] * df['pop_gt0_20']
      + RP_WEIGHTS[50] * df['pop_gt0_50']
      + RP_WEIGHTS[100] * df['pop_gt0_100']
  )
  df['E_gt0_total'] = df['E_gt0_5'] + df['E_gt0_10'] + df['E_gt0_20_50_100']
  return df


def _fast_predict_rp_family(
    models_dict: dict[str, SkillPredictorR2],
    X_rp: pd.DataFrame,
    knn_cache: dict,
    n_threads: int = 32,
) -> dict[str, np.ndarray]:
  """Predicts all RP metrics in a family, sharing PyTorch 15-NN lookups across metrics."""
  torch.set_num_threads(n_threads)
  out = {}
  for short_k, mod in models_dict.items():
    impl = mod.impl
    X_al = X_rp.reindex(columns=impl.feature_columns, fill_value=0)
    hgbr, et, knn = impl.ensemble.estimators_
    p_hgbr = hgbr.predict(X_al)
    et.named_steps['extratreesregressor'].n_jobs = n_threads
    with joblib.parallel_backend('threading', n_jobs=n_threads):
      p_et = et.predict(X_al)

    knn_reg = knn.named_steps['kneighborsregressor']
    fit_X = knn_reg._fit_X
    fit_y = np.asarray(knn_reg._y, dtype=np.float64).ravel()
    cache_key = (X_al.shape, fit_X.shape, float(fit_X[0, 0]), float(fit_X[-1, -1]), float(fit_X.sum()))
    if cache_key not in knn_cache:
      X_knn = np.ascontiguousarray(knn[:-1].transform(X_al), dtype=np.float32)
      t_ref = torch.from_numpy(np.ascontiguousarray(fit_X, dtype=np.float32))
      k_n = int(knn_reg.n_neighbors)
      n_q = len(X_knn)
      dists = np.empty((n_q, k_n), dtype=np.float64)
      nbr_idx = np.empty((n_q, k_n), dtype=np.int64)
      with torch.no_grad():
        for s in range(0, n_q, 16384):
          e = min(n_q, s + 16384)
          d_mat = torch.cdist(torch.from_numpy(X_knn[s:e]), t_ref)
          vals, idx = torch.topk(d_mat, k=k_n, dim=1, largest=False, sorted=True)
          dists[s:e] = vals.numpy().astype(np.float64)
          nbr_idx[s:e] = idx.numpy()
      zero_mask = dists <= 1e-15
      any_zero = zero_mask.any(axis=1)
      with np.errstate(divide='ignore'):
        nbr_w = np.where(any_zero[:, None], zero_mask.astype(np.float64), 1.0 / dists)
      nbr_w /= nbr_w.sum(axis=1, keepdims=True)
      knn_cache[cache_key] = (nbr_idx, nbr_w)
    else:
      nbr_idx, nbr_w = knn_cache[cache_key]

    p_knn = np.einsum('ij,ij->i', fit_y[nbr_idx], nbr_w)
    preds = impl.ensemble.final_estimator_.predict(np.column_stack([p_hgbr, p_et, p_knn]))
    out[short_k] = np.clip(preds, *impl.clip_bounds).astype(np.float32)
  return out


def _fast_predict_classic_family(
    models_dict: dict[str, SkillPredictorR2],
    X_cl: pd.DataFrame,
    n_threads: int = 32,
) -> dict[str, np.ndarray]:
  out = {}
  for short_k, mod in models_dict.items():
    impl = mod.impl
    for est in impl.model.estimators_:
      if hasattr(est, 'n_jobs'):
        try:
          est.n_jobs = n_threads
        except Exception:
          pass
    X_al = X_cl.reindex(columns=impl.feature_names, fill_value=0)
    out[short_k] = impl.predict_from_engineered(X_al).astype(np.float32)
  return out


def build_table_s1(df_meta: pd.DataFrame):
  """Builds clean Round-2 Table S1 across all 38 metrics + 50y + F1 using paired lifts."""
  print('\n--- Building Round-2 Table S1 (38 metrics + 50y + F1; 3-split & 2-split CV R^2) ---', flush=True)
  df_train = pd.read_parquet(TRAIN_PARQUET, columns=['pre_mm_uyr'])
  slice_ids = df_train.index[:N_GAUGES]
  df_cv_sum = pd.read_csv(os.path.join(META_DIR, 'stage_c_cv_summary.csv'))
  cv_idx = df_cv_sum.set_index(['metric', 'regime'])

  has_sys = df_meta['has_system'].values
  is_mapped = df_meta['country_name'].notna().values
  m_sys = has_sys & is_mapped
  m_nosys = (~has_sys) & is_mapped

  all_metrics = PRIORITY1_METRICS + PRIORITY2_METRICS
  rows = []
  for m in all_metrics:
    lo, hi = get_metric_bounds(m)
    # Observed skills across splits (raw clipped cleanly to natural bounds ONLY for CV R^2, raw unclipped for observed medians!)
    y_g = load_3split_target(m, '', slice_ids)[:N_GAUGES]
    y_r = load_3split_target(m, '', slice_ids)[N_GAUGES : 2 * N_GAUGES]
    y_s = load_3split_target(m, '', slice_ids)[2 * N_GAUGES :]
    y_u = np.concatenate([y_r, y_s])

    p_base_g = pd.read_parquet(os.path.join(GLOBAL_PREDS_DIR, f'base__{m}.parquet'))['pred'].reindex(df_meta.index).values
    r2_3sp = float(cv_idx.loc[(m, 'base'), 'r2_all']) if (m, 'base') in cv_idx.index else np.nan
    r2_2sp = float(cv_idx.loc[(m, 'base_2split'), 'r2_all']) if (m, 'base_2split') in cv_idx.index else np.nan

    def _med_iqr(arr):
      v = arr[np.isfinite(arr)]
      if len(v) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0
      q10, q25, q50, q75, q90 = np.percentile(v, [10, 25, 50, 75, 90])
      return float(q50), float(q10), float(q25), float(q75), float(q90), float(q75 - q25), int(len(v))

    obs_u_med, _, obs_u_q25, obs_u_q75, _, obs_u_iqr, n_u = _med_iqr(y_u)
    obs_g_med, _, _, _, _, _, n_g = _med_iqr(y_g)
    obs_r_med, _, _, _, _, _, _ = _med_iqr(y_r)
    obs_s_med, _, _, _, _, _, _ = _med_iqr(y_s)
    gp_med, gp_q10, gp_q25, gp_q75, gp_q90, gp_iqr, _ = _med_iqr(p_base_g)
    sys_med, _, _, _, _, _, _ = _med_iqr(p_base_g[m_sys])
    nosys_med, _, _, _, _, _, _ = _med_iqr(p_base_g[m_nosys])

    rows.append({
        'metric': m,
        'regime': 'base',
        'n_ungauged_obs': n_u,
        'n_3split_cv': int(cv_idx.loc[(m, 'base'), 'n_cv']) if (m, 'base') in cv_idx.index else 0,
        'obs_gauged_median': obs_g_med,
        'obs_random_median': obs_r_med,
        'obs_separated_median': obs_s_med,
        'obs_ungauged_median': obs_u_med,
        'obs_ungauged_q25': obs_u_q25,
        'obs_ungauged_q75': obs_u_q75,
        'obs_ungauged_iqr': obs_u_iqr,
        'meta_cv_r2_3split': r2_3sp,
        'meta_cv_r2_2split_ung': r2_2sp,
        'meta_cv_r2_delta_direct': np.nan,
        'global_pred_median': gp_med,
        'global_pred_q10': gp_q10,
        'global_pred_q25': gp_q25,
        'global_pred_q75': gp_q75,
        'global_pred_q90': gp_q90,
        'global_pred_iqr': gp_iqr,
        'pred_median_sys60': sys_med,
        'pred_median_nosys115': nosys_med,
    })

    for reg in ('delta_ft', 'delta_da', 'delta_ft_da'):
      if (m, reg) not in cv_idx.index:
        continue
      sfx = reg[len('delta'):]
      ym_all = load_3split_target(m, sfx, slice_ids)
      ym_g, ym_r, ym_s = ym_all[:N_GAUGES], ym_all[N_GAUGES : 2 * N_GAUGES], ym_all[2 * N_GAUGES :]
      ym_u = np.concatenate([ym_r, ym_s])
      d_g = pd.read_parquet(os.path.join(GLOBAL_PREDS_DIR, f'{reg}__{m}.parquet'))['pred'].reindex(df_meta.index).values
      p_m_g = np.clip(p_base_g + d_g, lo, hi)

      obs_u_med, _, obs_u_q25, obs_u_q75, _, obs_u_iqr, n_u = _med_iqr(ym_u)
      obs_g_med, _, _, _, _, _, _ = _med_iqr(ym_g)
      obs_r_med, _, _, _, _, _, _ = _med_iqr(ym_r)
      obs_s_med, _, _, _, _, _, _ = _med_iqr(ym_s)
      gp_med, gp_q10, gp_q25, gp_q75, gp_q90, gp_iqr, _ = _med_iqr(p_m_g)
      sys_med, _, _, _, _, _, _ = _med_iqr(p_m_g[m_sys])
      nosys_med, _, _, _, _, _, _ = _med_iqr(p_m_g[m_nosys])

      rows.append({
          'metric': m,
          'regime': reg.replace('delta_', ''),
          'n_ungauged_obs': n_u,
          'n_3split_cv': int(cv_idx.loc[(m, reg), 'n_cv']),
          'obs_gauged_median': obs_g_med,
          'obs_random_median': obs_r_med,
          'obs_separated_median': obs_s_med,
          'obs_ungauged_median': obs_u_med,
          'obs_ungauged_q25': obs_u_q25,
          'obs_ungauged_q75': obs_u_q75,
          'obs_ungauged_iqr': obs_u_iqr,
          'meta_cv_r2_3split': float(cv_idx.loc[(m, reg), 'recon_r2_all']),
          'meta_cv_r2_2split_ung': float(cv_idx.loc[(m, reg), 'recon_r2_ung']),
          'meta_cv_r2_delta_direct': float(cv_idx.loc[(m, reg), 'r2_all']),
          'global_pred_median': gp_med,
          'global_pred_q10': gp_q10,
          'global_pred_q25': gp_q25,
          'global_pred_q75': gp_q75,
          'global_pred_q90': gp_q90,
          'global_pred_iqr': gp_iqr,
          'pred_median_sys60': sys_med,
          'pred_median_nosys115': nosys_med,
      })

  os.makedirs(OUT_DIR, exist_ok=True)
  df_ts1 = pd.DataFrame(rows)
  out_p = os.path.join(OUT_DIR, 'Table_S1_4regimes_r2.csv')
  out_p3 = os.path.join(OUT_DIR, 'Table_S3_4regimes_r2.csv')
  df_ts1.to_csv(out_p, index=False)
  df_ts1.to_csv(out_p3, index=False)
  print(f'Saved {out_p3} ({len(df_ts1)} rows)!', flush=True)
  return df_ts1


def run_part_table_s3():
  """Builds Table S3 (43 metrics x 4 regimes) using ONLY public data (zero Fathom dependency)."""
  os.makedirs(OUT_DIR, exist_ok=True)
  all_ids = pd.read_parquet(INFER_PARQUET, columns=['pre_mm_uyr']).index
  assert len(all_ids) == 1034083
  df_meta = load_global_metadata_non_fathom(all_ids)
  return build_table_s1(df_meta)


def run_part_counterfactuals(n_threads: int = 32, overwrite: bool = False):
  os.makedirs(OUT_DIR, exist_ok=True)
  os.makedirs(PREDS_CACHE_DIR, exist_ok=True)
  t0 = time.time()

  with open(FEATURES_JSON) as f:
    features_95 = json.load(f)
  df_inf = pd.read_parquet(INFER_PARQUET)[features_95]
  all_ids = df_inf.index
  assert len(all_ids) == 1034083

  print('1. Loading fixed Round-2 exposure & global metadata (1,034,083 basins)...', flush=True)
  df_meta = load_global_metadata_and_exposure(all_ids)
  print(
      f"   Global total_pop={df_meta['total_pop'].sum()/1e9:.3f}B | "
      f"E_total(>0.1m)={df_meta['E_total'].sum()/1e6:.2f}M/yr | "
      f"E_total(>0.0m)={df_meta['E_gt0_total'].sum()/1e6:.2f}M/yr | "
      f"Mapped={df_meta['country_name'].notna().sum():,} basins, "
      f"{df_meta['country_name'].nunique()} territories "
      f"({df_meta.loc[df_meta['has_system'], 'country_name'].nunique()} sys / "
      f"{df_meta.loc[~df_meta['has_system'] & df_meta['country_name'].notna(), 'country_name'].nunique()} nosys)",
      flush=True,
  )

  build_table_s1(df_meta)

  print('\n2. Loading Stage C paired-lift models & Stage B1 tier features/masks...', flush=True)
  tier_feats = {}
  for tk in TIER_ORDER:
    tier_feats[tk] = pd.read_parquet(os.path.join(TIER_DIR, f'{tk}.parquet')).reindex(all_ids)
  df_masks = pd.read_parquet(os.path.join(TIER_DIR, 'tier_active_and_all_masks_r2.parquet')).reindex(all_ids)

  # Load base + delta_ft + delta_da + delta_ft_da models
  rp_models = defaultdict(dict)
  cl_models = defaultdict(dict)
  for fam in ('base', 'delta_ft', 'delta_da', 'delta_ft_da'):
    for sk, m_full in EVAL_RP_METRICS:
      p = os.path.join(MODELS_DIR, f'{fam}__{m_full}.joblib')
      if os.path.exists(p):
        rp_models[fam][sk] = joblib.load(p)
    for sk, m_full in EVAL_CLASSIC_METRICS:
      p = os.path.join(MODELS_DIR, f'{fam}__{m_full}.joblib')
      if os.path.exists(p):
        cl_models[fam][sk] = joblib.load(p)

  # Precompute transformed base matrices ONCE
  X_rp_base = rp_models['base']['r5'].impl._preprocess_features(df_inf)
  X_cl_base = cl_models['base']['NSE'].impl._engineer_features(df_inf)

  cov_t1a = tier_feats['tier1a_active_post2016_ccby_8811'][COV_COLS]
  ood_t1a = tier_feats['tier1a_active_post2016_ccby_8811'][OOD_COLS]

  def eval_combo(
      fam: str,
      combo_key: str,
      cov_df: pd.DataFrame,
      ood_df: pd.DataFrame,
      knn_cache: dict,
      eval_mask: np.ndarray | None = None,
  ) -> dict[str, np.ndarray]:
    pq = os.path.join(PREDS_CACHE_DIR, f'{fam}__{combo_key}.parquet')
    if (not overwrite) and os.path.exists(pq):
      dfc = pd.read_parquet(pq).reindex(all_ids)
      return {c: dfc[c].values for c in dfc.columns}

    ts = time.time()
    n_all = len(all_ids)
    all_keys = list(rp_models[fam].keys()) + list(cl_models[fam].keys())
    if eval_mask is not None:
      idx_sub = np.flatnonzero(eval_mask)
      if len(idx_sub) == 0:
        res = {k: np.zeros(n_all, dtype=np.float32) for k in all_keys}
        pd.DataFrame(res, index=all_ids).to_parquet(pq)
        print(f'   [{fam}__{combo_key}] mask=0 rows -> instant zeros ({time.time() - ts:.2f}s)', flush=True)
        return res
      X_rp = X_rp_base.iloc[idx_sub].copy()
      for c in COV_COLS:
        X_rp[c] = cov_df[c].values[idx_sub]
      for c in OOD_COLS:
        X_rp[c] = ood_df[c].values[idx_sub]
      sub_res = _fast_predict_rp_family(rp_models[fam], X_rp, knn_cache=knn_cache, n_threads=n_threads)

      X_cl = X_cl_base.iloc[idx_sub].copy()
      for c in COV_COLS:
        X_cl[c] = cov_df[c].values[idx_sub]
      for c in OOD_COLS:
        X_cl[c] = ood_df[c].values[idx_sub]
      sub_res.update(_fast_predict_classic_family(cl_models[fam], X_cl, n_threads=n_threads))

      res = {}
      for k, arr_sub in sub_res.items():
        full_arr = np.zeros(n_all, dtype=np.float32)
        full_arr[idx_sub] = arr_sub
        res[k] = full_arr
      pd.DataFrame(res, index=all_ids).to_parquet(pq)
      print(
          f'   [{fam}__{combo_key}] masked={len(idx_sub):,} done in {time.time() - ts:.1f}s '
          f'(med_mask r5={np.median(sub_res["r5"]):+.4f})',
          flush=True,
      )
      return res

    X_rp = X_rp_base.copy()
    for c in COV_COLS:
      X_rp[c] = cov_df[c].values
    for c in OOD_COLS:
      X_rp[c] = ood_df[c].values
    res = _fast_predict_rp_family(rp_models[fam], X_rp, knn_cache=knn_cache, n_threads=n_threads)

    X_cl = X_cl_base.copy()
    for c in COV_COLS:
      X_cl[c] = cov_df[c].values
    for c in OOD_COLS:
      X_cl[c] = ood_df[c].values
    res.update(_fast_predict_classic_family(cl_models[fam], X_cl, n_threads=n_threads))

    pd.DataFrame(res, index=all_ids).to_parquet(pq)
    print(f'   [{fam}__{combo_key}] full done in {time.time() - ts:.1f}s (med r5={np.median(res["r5"]):+.4f})', flush=True)
    return res

  # Evaluate all unique (cov_df, ood_df) pairs per family (exact masked eval for delta_* on Tiers 0..4!)
  FULL_DELTA_TIERS = {
      'tier1a_active_post2016_ccby_8811',
      'tier5_universal_idealized_ceiling',
      'tier5_dense_corridor_ceiling',
  }
  F2F3_TIERS = {
      'tier_caravan_estreams_34577',
      'tier_public_discharge_93827',
      'tier4_all_publicly_knowable_121819',
      'tier5_universal_idealized_ceiling',
      'tier5_dense_corridor_ceiling',
  }
  print('\n3. Evaluating all 13 tiers across paired-lift model families (exact masked delta_* optimization)...', flush=True)
  evals = {}
  for tk in TIER_ORDER:
    cov_k = tier_feats[tk][COV_COLS]
    ood_k = tier_feats[tk][OOD_COLS]
    knn_cache_joint = {}
    evals[('base', tk, 'joint')] = eval_combo('base', f'{tk}__joint', cov_k, ood_k, knn_cache_joint, eval_mask=None)
    for fam in ('delta_ft', 'delta_da', 'delta_ft_da'):
      if tk in FULL_DELTA_TIERS:
        em = None
      elif fam == 'delta_ft':
        em = df_masks[f'{tk}__union_all'].values.astype(bool)
      else:
        em = df_masks[f'{tk}__union_active'].values.astype(bool)
      evals[(fam, tk, 'joint')] = eval_combo(fam, f'{tk}__joint', cov_k, ood_k, knn_cache_joint, eval_mask=em)

    if tk in F2F3_TIERS:
      evals[('base', tk, 'f2_cov_only_base1a')] = eval_combo(
          'base', f'{tk}__f2_cov_only_base1a', cov_k, ood_t1a, {}, eval_mask=None
      )
      if tk == 'tier5_dense_corridor_ceiling':
        evals[('base', tk, 'f3_ood_only_base1a')] = evals[('base', 'tier5_universal_idealized_ceiling', 'f3_ood_only_base1a')]
      else:
        evals[('base', tk, 'f3_ood_only_base1a')] = eval_combo(
            'base', f'{tk}__f3_ood_only_base1a', cov_t1a, ood_k, {}, eval_mask=None
        )
    elif tk == 'tier0_zero_gauges':
      evals[('base', tk, 'f2_cov_only_base1a')] = evals[('base', tk, 'joint')]
    elif tk == 'tier1a_active_post2016_ccby_8811':
      evals[('base', tk, 'f2_cov_only_base1a')] = evals[('base', tk, 'joint')]
      evals[('base', tk, 'f3_ood_only_base1a')] = evals[('base', tk, 'joint')]
  evals[('base', 'tier0_zero_gauges', 'f3_ood_only_base1a')] = evals[('base', 'tier1a_active_post2016_ccby_8811', 'joint')]

  def reconstruct(p_base: dict, p_delta: dict) -> dict[str, np.ndarray]:
    out = {}
    for k in p_base:
      if k in p_delta:
        m_full = dict(EVAL_RP_METRICS + EVAL_CLASSIC_METRICS)[k]
        lo, hi = get_metric_bounds(m_full)
        out[k] = np.clip(p_base[k] + p_delta[k], lo, hi).astype(np.float32)
      else:
        out[k] = p_base[k].copy()
    return out

  def blend(p_global: dict, p_local: dict, mask: np.ndarray) -> dict[str, np.ndarray]:
    return {k: np.where(mask, p_local[k], p_global[k]) for k in p_global}

  def blend3(p_global: dict, p_ft: dict, p_ftda: dict, mask_all: np.ndarray, mask_act: np.ndarray) -> dict[str, np.ndarray]:
    return {k: np.where(mask_act, p_ftda[k], np.where(mask_all, p_ft[k], p_global[k])) for k in p_global}

  e5 = df_meta['E_5'].values
  e10 = df_meta['E_10'].values
  e20 = df_meta['E_20'].values
  e50 = df_meta['E_50'].values
  e100 = df_meta['E_100'].values
  e20p = df_meta['E_20_50_100'].values
  etot = df_meta['E_total'].values

  e5_0 = df_meta['E_gt0_5'].values
  e10_0 = df_meta['E_gt0_10'].values
  e20p_0 = df_meta['E_gt0_20_50_100'].values
  etot_0 = df_meta['E_gt0_total'].values

  def calc_warned(p: dict, mode: str = 'recall') -> np.ndarray:
    if mode == 'recall':
      return e5 * p['r5'] + e10 * p['r10'] + e20p * p['r20']
    if mode == 'recall_gt0':
      return e5_0 * p['r5'] + e10_0 * p['r10'] + e20p_0 * p['r20']
    if mode == 'recall_r50direct':
      return e5 * p['r5'] + e10 * p['r10'] + e20 * p['r20'] + (e50 + e100) * p['r50']
    if mode == 'recall_decay_mild':  # R50 = 0.85*R20, R100 = 0.70*R20
      return e5 * p['r5'] + e10 * p['r10'] + (e20 + 0.85 * e50 + 0.70 * e100) * p['r20']
    if mode == 'recall_decay_steep':  # R50 = 0.50*R20, R100 = 0.25*R20
      return e5 * p['r5'] + e10 * p['r10'] + (e20 + 0.50 * e50 + 0.25 * e100) * p['r20']
    if mode == 'f1':
      return e5 * p['f1_5'] + e10 * p['f1_10'] + e20p * p['f1_20']
    if mode == 'csi':
      csi5 = np.where((p['r5'] > 1e-6) & (p['p5'] > 1e-6), 1.0 / (1.0 / np.maximum(p['r5'], 1e-6) + 1.0 / np.maximum(p['p5'], 1e-6) - 1.0), 0.0)
      csi10 = np.where((p['r10'] > 1e-6) & (p['p10'] > 1e-6), 1.0 / (1.0 / np.maximum(p['r10'], 1e-6) + 1.0 / np.maximum(p['p10'], 1e-6) - 1.0), 0.0)
      csi20 = np.where((p['r20'] > 1e-6) & (p['p20'] > 1e-6), 1.0 / (1.0 / np.maximum(p['r20'], 1e-6) + 1.0 / np.maximum(p['p20'], 1e-6) - 1.0), 0.0)
      return e5 * csi5 + e10 * csi10 + e20p * csi20
    raise ValueError(mode)

  def calc_rp_clipped(p_new: dict, p_ref: dict) -> np.ndarray:
    return (
        e5 * np.maximum(0.0, p_new['r5'] - p_ref['r5'])
        + e10 * np.maximum(0.0, p_new['r10'] - p_ref['r10'])
        + e20p * np.maximum(0.0, p_new['r20'] - p_ref['r20'])
    )

  p_b0 = evals[('base', 'tier0_zero_gauges', 'joint')]
  p_b1a = evals[('base', 'tier1a_active_post2016_ccby_8811', 'joint')]
  p_ft1a = reconstruct(p_b1a, evals[('delta_ft', 'tier1a_active_post2016_ccby_8811', 'joint')])
  p_da1a = reconstruct(p_b1a, evals[('delta_da', 'tier1a_active_post2016_ccby_8811', 'joint')])
  p_ftda1a = reconstruct(p_b1a, evals[('delta_ft_da', 'tier1a_active_post2016_ccby_8811', 'joint')])
  p_b4 = evals[('base', 'tier4_all_publicly_knowable_121819', 'joint')]

  w_b0 = calc_warned(p_b0)
  w_b1a = calc_warned(p_b1a)
  w_b4 = calc_warned(p_b4)

  is_mapped = df_meta['country_name'].notna().values
  m_sys60 = df_meta['has_system'].values & is_mapped
  m_nosys115 = (~df_meta['has_system'].values) & is_mapped
  m_unmapped = ~is_mapped

  print('\n4. Computing multi-mechanism summary across all 13 tiers (physical G-set primary + union/ge95 sensitivity)...', flush=True)
  summary_rows = []
  sens_rows = []
  country_vec_store = {}

  for tk in TIER_ORDER:
    p_bk = evals[('base', tk, 'joint')]
    p_ftk = reconstruct(p_bk, evals[('delta_ft', tk, 'joint')])
    p_dak = reconstruct(p_bk, evals[('delta_da', tk, 'joint')])
    p_ftdak = reconstruct(p_bk, evals[('delta_ft_da', tk, 'joint')])

    mechs = [('000_global_train_joint', p_bk)]
    if ('base', tk, 'f2_cov_only_base1a') in evals:
      mechs.append(('000_global_train_f2_cov_only_base1a', evals[('base', tk, 'f2_cov_only_base1a')]))
    if ('base', tk, 'f3_ood_only_base1a') in evals:
      mechs.append(('000_global_train_f3_ood_only_base1a', evals[('base', tk, 'f3_ood_only_base1a')]))
    for mask_type, all_col, act_col in [
        ('snap', f'{tk}__snap_all', f'{tk}__snap_active'),
        ('snap_main', f'{tk}__snap_main_all', f'{tk}__snap_main_active'),
        ('union', f'{tk}__union_all', f'{tk}__union_active'),
        ('ge95', f'{tk}__ge95_all', f'{tk}__ge95_active'),
    ]:
      m_all = df_masks[all_col].values.astype(bool)
      m_act = df_masks[act_col].values.astype(bool)
      mechs.extend([
          (f'001_da_only_{mask_type}', blend(p_b1a, p_da1a, m_act)),
          (f'101_train_plus_da_{mask_type}', blend(p_bk, p_dak, m_act)),
          (f'010_ft_only_{mask_type}', blend(p_b1a, p_ft1a, m_all)),
          (f'110_train_plus_ft_{mask_type}', blend(p_bk, p_ftk, m_all)),
          (f'011_ft_plus_da_only_{mask_type}', blend3(p_b1a, p_ft1a, p_ftda1a, m_all, m_act)),
          (f'111_all_three_combined_{mask_type}', blend3(p_bk, p_ftk, p_ftdak, m_all, m_act)),
      ])

    for mech_name, p in mechs:
      w_vec = calc_warned(p, 'recall')
      unw_vec = etot - w_vec
      d_net = w_vec - w_b1a
      d_clip = calc_rp_clipped(p, p_b1a)
      d_net_t0 = w_vec - w_b0
      d_net_t4 = w_vec - w_b4

      summary_rows.append({
          'tier_key': tk,
          'tier_label': TIER_LABELS[tk],
          'mechanism': mech_name,
          'n_snap_all': int(df_masks[f'{tk}__snap_all'].sum()),
          'n_snap_active': int(df_masks[f'{tk}__snap_active'].sum()),
          'n_ge95_all': int(df_masks[f'{tk}__ge95_all'].sum()),
          'n_ge95_active': int(df_masks[f'{tk}__ge95_active'].sum()),
          'r5_med_pct': float(np.median(p['r5']) * 100.0),
          'r10_med_pct': float(np.median(p['r10']) * 100.0),
          'r20_med_pct': float(np.median(p['r20']) * 100.0),
          'p5_med_pct': float(np.median(p['p5']) * 100.0),
          'p10_med_pct': float(np.median(p['p10']) * 100.0),
          'p20_med_pct': float(np.median(p['p20']) * 100.0),
          'f1_5_med_pct': float(np.median(p['f1_5']) * 100.0),
          'NSE_med': float(np.median(p['NSE'])),
          'KGE_med': float(np.median(p['KGE'])),
          'E_total_global': float(etot.sum()),
          'E_total_mapped': float(etot[is_mapped].sum()),
          'E_total_sys60': float(etot[m_sys60].sum()),
          'E_total_nosys115': float(etot[m_nosys115].sum()),
          'E_total_unmapped': float(etot[m_unmapped].sum()),
          'warned_global': float(w_vec.sum()),
          'unwarned_global': float(unw_vec.sum()),
          'pct_warned_global': float(100.0 * w_vec.sum() / etot.sum()),
          'delta_net_global': float(d_net.sum()),
          'delta_clip_global': float(d_clip.sum()),
          'pct_unw_closed_net_global': float(100.0 * d_net.sum() / (etot.sum() - w_b1a.sum())),
          'warned_mapped': float(w_vec[is_mapped].sum()),
          'unwarned_mapped': float(unw_vec[is_mapped].sum()),
          'delta_net_mapped': float(d_net[is_mapped].sum()),
          'delta_clip_mapped': float(d_clip[is_mapped].sum()),
          'warned_sys60': float(w_vec[m_sys60].sum()),
          'unwarned_sys60': float(unw_vec[m_sys60].sum()),
          'delta_net_sys60': float(d_net[m_sys60].sum()),
          'delta_clip_sys60': float(d_clip[m_sys60].sum()),
          'warned_nosys115': float(w_vec[m_nosys115].sum()),
          'unwarned_nosys115': float(unw_vec[m_nosys115].sum()),
          'delta_net_nosys115': float(d_net[m_nosys115].sum()),
          'delta_clip_nosys115': float(d_clip[m_nosys115].sum()),
          'pct_unw_closed_net_nosys115': float(100.0 * d_net[m_nosys115].sum() / (etot[m_nosys115].sum() - w_b1a[m_nosys115].sum())),
          'delta_net_vs_t0_global': float(d_net_t0.sum()),
          'delta_net_vs_t4_global': float(d_net_t4.sum()),
      })

      # Store sensitivity metrics on primary anchored tiers
      if tk in (
          'tier1a_active_post2016_ccby_8811',
          'tier_caravan_estreams_34577',
          'tier_public_discharge_93827',
          'tier4_all_publicly_knowable_121819',
          'tier5_universal_idealized_ceiling',
      ) and mech_name in ('000_global_train_joint', '111_all_three_combined_snap', '111_all_three_combined_union'):
        w_gt0 = calc_warned(p, 'recall_gt0')
        w_r50 = calc_warned(p, 'recall_r50direct')
        w_dm = calc_warned(p, 'recall_decay_mild')
        w_ds = calc_warned(p, 'recall_decay_steep')
        w_f1 = calc_warned(p, 'f1')
        w_csi = calc_warned(p, 'csi')
        # Exposure-weighted average false alarm rate (1 - P)
        far5 = 1.0 - p['p5']
        far10 = 1.0 - p['p10']
        far20 = 1.0 - p['p20']
        ew_far = (e5 * far5 + e10 * far10 + e20p * far20).sum() / etot.sum()
        ew_far_nosys = (e5[m_nosys115] * far5[m_nosys115] + e10[m_nosys115] * far10[m_nosys115] + e20p[m_nosys115] * far20[m_nosys115]).sum() / etot[m_nosys115].sum()
        for alpha in (0.25, 0.50, 0.70, 1.00):
          sens_rows.append({
              'tier_key': tk,
              'mechanism': mech_name,
              'alpha': alpha,
              'warned_recall_global': float(alpha * w_vec.sum()),
              'warned_recall_nosys115': float(alpha * w_vec[m_nosys115].sum()),
              'delta_net_recall_global': float(alpha * (w_vec - w_b1a).sum()),
              'delta_net_recall_nosys115': float(alpha * (w_vec[m_nosys115] - w_b1a[m_nosys115]).sum()),
              'warned_f1_global': float(alpha * w_f1.sum()),
              'warned_f1_nosys115': float(alpha * w_f1[m_nosys115].sum()),
              'warned_csi_global': float(alpha * w_csi.sum()),
              'warned_csi_nosys115': float(alpha * w_csi[m_nosys115].sum()),
              'warned_gt0_global': float(alpha * w_gt0.sum()),
              'warned_r50direct_global': float(alpha * w_r50.sum()),
              'warned_decay_mild_global': float(alpha * w_dm.sum()),
              'warned_decay_steep_global': float(alpha * w_ds.sum()),
              'exp_weighted_far_global_pct': float(ew_far * 100.0),
              'exp_weighted_far_nosys115_pct': float(ew_far_nosys * 100.0),
              'med_far5_global_pct': float(np.median(far5) * 100.0),
              'med_far5_nosys115_pct': float(np.median(far5[m_nosys115]) * 100.0),
          })

      # Keep vectors needed for Table S3
      if (tk, mech_name) in {
          ('tier_caravan_estreams_34577', '111_all_three_combined_snap'),
          ('tier_public_discharge_93827', '111_all_three_combined_snap'),
          ('tier4_all_publicly_knowable_121819', '000_global_train_joint'),
          ('tier4_all_publicly_knowable_121819', '010_ft_only_snap'),
          ('tier4_all_publicly_knowable_121819', '001_da_only_snap'),
          ('tier4_all_publicly_knowable_121819', '101_train_plus_da_snap'),
          ('tier4_all_publicly_knowable_121819', '111_all_three_combined_snap'),
          ('tier4_all_publicly_knowable_121819', '111_all_three_combined_union'),
          ('tier5_universal_idealized_ceiling', '000_global_train_joint'),
          ('tier5_universal_idealized_ceiling', '000_global_train_f2_cov_only_base1a'),
          ('tier5_universal_idealized_ceiling', '000_global_train_f3_ood_only_base1a'),
          ('tier5_universal_idealized_ceiling', '010_ft_only_snap'),
          ('tier5_universal_idealized_ceiling', '001_da_only_snap'),
          ('tier5_universal_idealized_ceiling', '101_train_plus_da_snap'),
          ('tier5_universal_idealized_ceiling', '111_all_three_combined_snap'),
      }:
        country_vec_store[(tk, mech_name)] = (w_vec, d_net, d_clip)

  df_sum = pd.DataFrame(summary_rows)
  df_sum.to_csv(os.path.join(OUT_DIR, 'multi_mechanism_tiered_exposure_r2.csv'), index=False)
  df_sens = pd.DataFrame(sens_rows)
  df_sens.to_csv(os.path.join(OUT_DIR, 'precision_f1_csi_alpha_sensitivity_r2.csv'), index=False)

  # Build Basin-level & Country-level Table S3 across ALL 175 territories
  print('\n5. Building clean Round-2 Table S3 (175 territories) & checking mega-delta inversions...', flush=True)
  df_b = df_meta[['country_name', 'has_system', 'cap_tier', 'lat', 'lon', 'sub_area', 'total_pop', 'E_total', 'E_gt0_total']].copy()
  df_b['zero_up_t1a'] = (tier_feats['tier1a_active_post2016_ccby_8811']['total_upstream_coverage'].values <= 0.0).astype(float) * 100.0
  df_b['zero_up_t4'] = (tier_feats['tier4_all_publicly_knowable_121819']['total_upstream_coverage'].values <= 0.0).astype(float) * 100.0
  df_b['warned_t1a'] = w_b1a
  df_b['unwarned_t1a'] = etot - w_b1a
  df_b['warned_t4_base'] = w_b4
  df_b['unwarned_t4_base'] = etot - w_b4

  for (tk, mech_name), (w_v, dn_v, dc_v) in country_vec_store.items():
    tag = f'{tk}__{mech_name}'
    df_b[f'dnet__{tag}'] = dn_v
    df_b[f'dclip__{tag}'] = dc_v

  df_b.to_parquet(os.path.join(OUT_DIR, 'basin_level_exposure_counterfactuals_r2.parquet'))

  # Country table (mapped 175 territories)
  df_mapped = df_b[df_b['country_name'].notna()].copy()
  sum_cols = ['sub_area', 'total_pop', 'E_total', 'E_gt0_total', 'warned_t1a', 'unwarned_t1a', 'warned_t4_base', 'unwarned_t4_base'] + [
      c for c in df_b.columns if c.startswith('dnet__') or c.startswith('dclip__')
  ]
  grp = df_mapped.groupby('country_name', as_index=False)
  df_c = grp[sum_cols].sum()
  df_c['n_basins'] = grp.size()['size'].values
  df_c['has_system'] = grp['has_system'].first()['has_system'].values.astype(bool)
  df_c['cap_tier'] = grp['cap_tier'].first()['cap_tier'].values
  df_c['zero_up_pct_t1a'] = grp['zero_up_t1a'].mean()['zero_up_t1a'].values
  df_c['zero_up_pct_t4'] = grp['zero_up_t4'].mean()['zero_up_t4'].values

  # Verify mega-delta inversions (d_111 - d_000 < 0)!
  col_t4_000 = 'dnet__tier4_all_publicly_knowable_121819__000_global_train_joint'
  col_t4_111 = 'dnet__tier4_all_publicly_knowable_121819__111_all_three_combined_snap'
  col_t5_000 = 'dnet__tier5_universal_idealized_ceiling__000_global_train_joint'
  col_t5_111 = 'dnet__tier5_universal_idealized_ceiling__111_all_three_combined_snap'
  inv_t4 = df_c[df_c[col_t4_111] < df_c[col_t4_000]].copy()
  inv_t5 = df_c[df_c[col_t5_111] < df_c[col_t5_000]].copy()
  print(f'   Inversion audit (Combined 111 < Train Alone 000): Tier 3 = {len(inv_t4)} countries, Universal = {len(inv_t5)} countries', flush=True)
  if len(inv_t5) > 0:
    inv_t5['diff_111_000'] = inv_t5[col_t5_111] - inv_t5[col_t5_000]
    print(inv_t5.sort_values('diff_111_000')[['country_name', 'E_total', col_t5_000, col_t5_111, 'diff_111_000']].head(10).to_string(index=False))
  for cname in ('Bangladesh', 'Viet Nam', 'Myanmar', 'India', 'China', 'Pakistan', 'Nigeria', 'Sudan'):
    sub = df_c[df_c['country_name'] == cname]
    if len(sub) > 0:
      r = sub.iloc[0]
      print(
          f"   {cname:12s}: E={r['E_total']/1e6:6.2f}M | Unw={r['unwarned_t1a']/1e6:6.2f}M | "
          f"T3_000={r[col_t4_000]/1e6:+.3f}M T3_111={r[col_t4_111]/1e6:+.3f}M | "
          f"U_000={r[col_t5_000]/1e6:+.3f}M U_111={r[col_t5_111]/1e6:+.3f}M",
          flush=True,
      )

  df_c = df_c.sort_values('E_total', ascending=False).reset_index(drop=True)
  df_c.to_csv(os.path.join(OUT_DIR, 'Table_S3_175territories_r2.csv'), index=False)

  # Print clean headline Table 1 preview
  print('\n=== HEADLINE TABLE 1 PREVIEW (Round 2 Fixed Exposure, 1,034,083 basins, Paired Lifts, Physical G-set, Unclipped Net) ===')
  for tk in [
      'tier1a_active_post2016_ccby_8811',
      'tier_caravan_estreams_34577',
      'tier_public_discharge_93827',
      'tier4_all_publicly_knowable_121819',
      'tier5_universal_idealized_ceiling',
  ]:
    for mech in ('000_global_train_joint', '010_ft_only_snap', '001_da_only_snap', '111_all_three_combined_snap', '111_all_three_combined_union'):
      sub = df_sum[(df_sum['tier_key'] == tk) & (df_sum['mechanism'] == mech)]
      if len(sub) > 0:
        r = sub.iloc[0]
        print(
            f"  {tk[:22]:22s} | {mech:28s} | GlbNet={r['delta_net_global']/1e6:+.2f}M ({r['pct_unw_closed_net_global']:+.1f}%) | "
            f"CAPGapNet={r['delta_net_nosys115']/1e6:+.2f}M ({r['pct_unw_closed_net_nosys115']:+.1f}%) | Sys60={r['delta_net_sys60']/1e6:+.2f}M"
        )
  print(f'\nPart 1 (Counterfactuals + Tables S1/S3) complete in {(time.time() - t0)/60:.1f} min!', flush=True)


# ==============================================================================
# PART 2: B=20 GAUGE-CLUSTERED BOOTSTRAP FOR 95% CIs ON TABLE 1, FIG. 3, TABLE S3
# ==============================================================================

BOOT_TIERS = [
    'tier0_zero_gauges',
    'tier1a_active_post2016_ccby_8811',
    'tier_caravan_estreams_34577',
    'tier_public_discharge_93827',
    'tier4_all_publicly_knowable_121819',
    'tier5_universal_idealized_ceiling',
]


def _run_one_bootstrap_rep(b_idx: int, n_threads: int) -> dict:
  """Fits one gauge-clustered bootstrap replicate across (r5, r10, r20) x 4 families."""
  os.environ['OMP_NUM_THREADS'] = str(n_threads)
  os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
  os.environ['MKL_NUM_THREADS'] = str(n_threads)
  torch.set_num_threads(n_threads)

  rep_json = os.path.join(BOOT_DIR, f'rep_{b_idx:03d}.json')
  rep_ctry = os.path.join(BOOT_DIR, f'rep_{b_idx:03d}_countries.parquet')
  if os.path.exists(rep_json) and os.path.exists(rep_ctry):
    with open(rep_json) as f:
      return json.load(f)

  t0 = time.time()
  with open(FEATURES_JSON) as f:
    features_95 = json.load(f)
  df_train = pd.read_parquet(TRAIN_PARQUET)[features_95]
  slice_ids = df_train.index[:N_GAUGES]
  df_inf = pd.read_parquet(INFER_PARQUET)[features_95]
  all_ids = df_inf.index
  df_meta = load_global_metadata_and_exposure(all_ids)
  df_masks = pd.read_parquet(os.path.join(TIER_DIR, 'tier_active_and_all_masks_r2.parquet')).reindex(all_ids)

  tier_feats = {tk: pd.read_parquet(os.path.join(TIER_DIR, f'{tk}.parquet')).reindex(all_ids) for tk in BOOT_TIERS}

  # Resample 8,811 Caravan gauges WITH REPLACEMENT, preserving cluster identity across all 3 splits!
  rng = np.random.default_rng(10000 + b_idx)
  sampled_g_idx = rng.integers(0, N_GAUGES, size=N_GAUGES)
  row_indices = np.concatenate([sampled_g_idx, sampled_g_idx + N_GAUGES, sampled_g_idx + 2 * N_GAUGES])
  boot_groups = np.tile(np.arange(N_GAUGES), 3)  # cluster by draw index (or original gauge ID)
  orig_groups = np.tile(sampled_g_idx, 3)  # exact gauge-ID clustering inside StackingRegressor!
  X_boot = df_train.iloc[row_indices].copy()

  rp_metrics = [('r5', '5.0-year-2.0-window-recall'), ('r10', '10.0-year-2.0-window-recall'), ('r20', '20.0-year-2.0-window-recall')]
  models = defaultdict(dict)
  for sk, m_full in rp_metrics:
    y_base_raw = load_3split_target(m_full, '', slice_ids)
    y_base_clip = np.where(np.isfinite(y_base_raw), np.clip(y_base_raw, 0.0, 1.0), np.nan)
    y_b = y_base_clip[row_indices]
    ok_b = np.isfinite(y_b)
    m_base = SkillPredictorR2(m_full, is_delta=False, n_threads=n_threads)
    m_base.fit(X_boot.iloc[ok_b], pd.Series(y_b[ok_b], index=X_boot.index[ok_b]), groups=orig_groups[ok_b])
    models['base'][sk] = m_base

    for fam, sfx in [('delta_ft', '_ft'), ('delta_da', '_da'), ('delta_ft_da', '_ft_da')]:
      ym_raw = load_3split_target(m_full, sfx, slice_ids)
      ym_clip = np.where(np.isfinite(ym_raw), np.clip(ym_raw, 0.0, 1.0), np.nan)
      d_full = np.where(np.isfinite(y_base_clip) & np.isfinite(ym_clip), ym_clip - y_base_clip, np.nan)
      d_b = d_full[row_indices]
      ok_d = np.isfinite(d_b)
      m_d = SkillPredictorR2(m_full, is_delta=True, n_threads=n_threads)
      m_d.fit(X_boot.iloc[ok_d], pd.Series(d_b[ok_d], index=X_boot.index[ok_d]), groups=orig_groups[ok_d])
      models[fam][sk] = m_d

  X_rp_base = models['base']['r5'].impl._preprocess_features(df_inf)
  cov_t1a = tier_feats['tier1a_active_post2016_ccby_8811'][COV_COLS]
  ood_t1a = tier_feats['tier1a_active_post2016_ccby_8811'][OOD_COLS]

  n_all = len(all_ids)
  preds = {}
  for tk in BOOT_TIERS:
    cov_k = tier_feats[tk][COV_COLS]
    ood_k = tier_feats[tk][OOD_COLS]
    X_rp = X_rp_base.copy()
    for c in COV_COLS:
      X_rp[c] = cov_k[c].values
    for c in OOD_COLS:
      X_rp[c] = ood_k[c].values
    knn_c = {}
    preds[('base', tk, 'joint')] = _fast_predict_rp_family(models['base'], X_rp, knn_cache=knn_c, n_threads=n_threads)

    for fam in ('delta_ft', 'delta_da', 'delta_ft_da'):
      need_full = (tk == 'tier5_universal_idealized_ceiling') or (
          tk == 'tier1a_active_post2016_ccby_8811' and fam in ('delta_ft', 'delta_da')
      )
      if need_full:
        preds[(fam, tk, 'joint')] = _fast_predict_rp_family(models[fam], X_rp, knn_cache=knn_c, n_threads=n_threads)
      else:
        mask_k = df_masks[f'{tk}__snap_all' if fam == 'delta_ft' else f'{tk}__snap_active'].values.astype(bool)
        idx_sub = np.flatnonzero(mask_k)
        if len(idx_sub) == 0:
          preds[(fam, tk, 'joint')] = {k: np.zeros(n_all, dtype=np.float32) for k in ('r5', 'r10', 'r20')}
        else:
          sub_p = _fast_predict_rp_family(models[fam], X_rp.iloc[idx_sub], knn_cache=knn_c, n_threads=n_threads)
          full_p = {}
          for k, arr_s in sub_p.items():
            fa = np.zeros(n_all, dtype=np.float32)
            fa[idx_sub] = arr_s
            full_p[k] = fa
          preds[(fam, tk, 'joint')] = full_p

    if tk in ('tier4_all_publicly_knowable_121819', 'tier5_universal_idealized_ceiling'):
      X_f2 = X_rp_base.copy()
      for c in COV_COLS:
        X_f2[c] = cov_k[c].values
      for c in OOD_COLS:
        X_f2[c] = ood_t1a[c].values
      preds[('base', tk, 'f2_cov_only')] = _fast_predict_rp_family(models['base'], X_f2, knn_cache={}, n_threads=n_threads)

      X_f3 = X_rp_base.copy()
      for c in COV_COLS:
        X_f3[c] = cov_t1a[c].values
      for c in OOD_COLS:
        X_f3[c] = ood_k[c].values
      preds[('base', tk, 'f3_ood_only')] = _fast_predict_rp_family(models['base'], X_f3, knn_cache={}, n_threads=n_threads)

  e5 = df_meta['E_5'].values
  e10 = df_meta['E_10'].values
  e20p = df_meta['E_20_50_100'].values
  etot = df_meta['E_total'].values
  is_mapped = df_meta['country_name'].notna().values
  m_sys60 = df_meta['has_system'].values & is_mapped
  m_nosys115 = (~df_meta['has_system'].values) & is_mapped

  def recon(pb, pd_m):
    return {k: np.clip(pb[k] + pd_m[k], 0.0, 1.0) for k in ('r5', 'r10', 'r20')}

  def warned(p):
    return e5 * p['r5'] + e10 * p['r10'] + e20p * p['r20']

  p_b1a = preds[('base', 'tier1a_active_post2016_ccby_8811', 'joint')]
  p_ft1a = recon(p_b1a, preds[('delta_ft', 'tier1a_active_post2016_ccby_8811', 'joint')])
  p_da1a = recon(p_b1a, preds[('delta_da', 'tier1a_active_post2016_ccby_8811', 'joint')])
  w_b1a = warned(p_b1a)

  rep_stats = {
      'rep': b_idx,
      'elapsed_s': float(time.time() - t0),
      'baseline_warned_global': float(w_b1a.sum()),
      'baseline_warned_mapped': float(w_b1a[is_mapped].sum()),
      'baseline_unwarned_global': float((etot - w_b1a).sum()),
      'baseline_warned_sys60': float(w_b1a[m_sys60].sum()),
      'baseline_warned_nosys115': float(w_b1a[m_nosys115].sum()),
      'baseline_unwarned_nosys115': float((etot - w_b1a)[m_nosys115].sum()),
      'scenarios': {},
  }

  ctry_cols = {'warned_t1a': w_b1a, 'unwarned_t1a': etot - w_b1a}
  for tk in BOOT_TIERS:
    p_bk = preds[('base', tk, 'joint')]
    p_ftk = recon(p_bk, preds[('delta_ft', tk, 'joint')])
    p_dak = recon(p_bk, preds[('delta_da', tk, 'joint')])
    p_ftdak = recon(p_bk, preds[('delta_ft_da', tk, 'joint')])
    m_all = df_masks[f'{tk}__snap_all'].values.astype(bool)
    m_act = df_masks[f'{tk}__snap_active'].values.astype(bool)

    scen_map = {
        '000_train': p_bk,
        '010_ft_only': {k: np.where(m_all, p_ft1a[k], p_b1a[k]) for k in ('r5', 'r10', 'r20')},
        '001_da_only': {k: np.where(m_act, p_da1a[k], p_b1a[k]) for k in ('r5', 'r10', 'r20')},
        '101_train_da': {k: np.where(m_act, p_dak[k], p_bk[k]) for k in ('r5', 'r10', 'r20')},
        '111_combined': {k: np.where(m_act, p_ftdak[k], np.where(m_all, p_ftk[k], p_bk[k])) for k in ('r5', 'r10', 'r20')},
    }
    if tk in ('tier4_all_publicly_knowable_121819', 'tier5_universal_idealized_ceiling'):
      scen_map['000_f2_cov'] = preds[('base', tk, 'f2_cov_only')]
      scen_map['000_f3_ood'] = preds[('base', tk, 'f3_ood_only')]

    for sk_name, p_s in scen_map.items():
      w_s = warned(p_s)
      d_s = w_s - w_b1a
      rep_stats['scenarios'][f'{tk}__{sk_name}'] = {
          'delta_global': float(d_s.sum()),
          'delta_sys60': float(d_s[m_sys60].sum()),
          'delta_nosys115': float(d_s[m_nosys115].sum()),
          'warned_global': float(w_s.sum()),
          'warned_mapped': float(w_s[is_mapped].sum()),
          'warned_nosys115': float(w_s[m_nosys115].sum()),
      }
      if tk in ('tier4_all_publicly_knowable_121819', 'tier5_universal_idealized_ceiling') and sk_name in ('000_train', '111_combined'):
        ctry_cols[f'{tk}__{sk_name}'] = d_s

  df_c_rep = pd.DataFrame(ctry_cols, index=all_ids)
  df_c_rep['country_name'] = df_meta['country_name'].values
  df_c_rep = df_c_rep[df_c_rep['country_name'].notna()].groupby('country_name').sum()
  df_c_rep.to_parquet(rep_ctry)

  with open(rep_json, 'w') as f:
    json.dump(rep_stats, f, indent=2)
  return rep_stats


def run_part_bootstrap(n_boot: int = 20, workers: int = 7, threads: int = 8):
  os.makedirs(BOOT_DIR, exist_ok=True)
  t0 = time.time()
  print(f'\n--- Running B={n_boot} Gauge-Clustered Bootstrap Replicates ({workers} workers x {threads} thr) ---', flush=True)
  os.environ['OMP_NUM_THREADS'] = str(threads)
  os.environ['OPENBLAS_NUM_THREADS'] = str(threads)
  os.environ['MKL_NUM_THREADS'] = str(threads)
  os.environ['NUMEXPR_NUM_THREADS'] = str(threads)
  reps = []
  with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as ex:
    futs = [ex.submit(_run_one_bootstrap_rep, b, threads) for b in range(n_boot)]
    for fut in as_completed(futs):
      r = fut.result()
      reps.append(r)
      print(
          f"  [Boot {len(reps):2d}/{n_boot}] rep={r['rep']:2d} ({r['elapsed_s']:.1f}s) | "
          f"BaseUnw={r['baseline_unwarned_global']/1e6:.2f}M | "
          f"T3_111_Glb={r['scenarios']['tier4_all_publicly_knowable_121819__111_combined']['delta_global']/1e6:.2f}M | "
          f"U_111_Glb={r['scenarios']['tier5_universal_idealized_ceiling__111_combined']['delta_global']/1e6:.2f}M",
          flush=True,
      )

  # Aggregate Table 1 / Fig 3 95% CIs
  ci_rows = []
  for base_k in ('baseline_warned_global', 'baseline_warned_mapped', 'baseline_unwarned_global', 'baseline_warned_sys60', 'baseline_warned_nosys115', 'baseline_unwarned_nosys115'):
    vals = np.array([r[base_k] for r in reps])
    ci_rows.append({
        'scenario': base_k,
        'metric': 'level',
        'boot_mean': float(vals.mean()),
        'boot_sd': float(vals.std(ddof=1)),
        'boot_p025': float(np.percentile(vals, 2.5)),
        'boot_p500': float(np.percentile(vals, 50.0)),
        'boot_p975': float(np.percentile(vals, 97.5)),
    })

  scen_keys = sorted(reps[0]['scenarios'].keys())
  for sk in scen_keys:
    for mk in ('delta_global', 'delta_sys60', 'delta_nosys115', 'warned_global', 'warned_mapped', 'warned_nosys115'):
      vals = np.array([r['scenarios'][sk][mk] for r in reps])
      ci_rows.append({
          'scenario': sk,
          'metric': mk,
          'boot_mean': float(vals.mean()),
          'boot_sd': float(vals.std(ddof=1)),
          'boot_p025': float(np.percentile(vals, 2.5)),
          'boot_p500': float(np.percentile(vals, 50.0)),
          'boot_p975': float(np.percentile(vals, 97.5)),
      })

  df_ci = pd.DataFrame(ci_rows)
  df_ci.to_csv(os.path.join(OUT_DIR, 'bootstrap_table1_fig3_ci95_r2.csv'), index=False)

  # Aggregate per-country 95% CIs for Table S3
  ctry_dfs = [pd.read_parquet(os.path.join(BOOT_DIR, f'rep_{b:03d}_countries.parquet')) for b in range(n_boot)]
  c_names = ctry_dfs[0].index
  c_cols = ctry_dfs[0].columns
  c_ci_data = {}
  for col in c_cols:
    mat = np.column_stack([d[col].reindex(c_names).values for d in ctry_dfs])
    c_ci_data[f'{col}__p025'] = np.percentile(mat, 2.5, axis=1)
    c_ci_data[f'{col}__p500'] = np.percentile(mat, 50.0, axis=1)
    c_ci_data[f'{col}__p975'] = np.percentile(mat, 97.5, axis=1)
    c_ci_data[f'{col}__sd'] = np.std(mat, axis=1, ddof=1)
  df_c_ci = pd.DataFrame(c_ci_data, index=c_names)
  df_c_ci.to_csv(os.path.join(OUT_DIR, 'bootstrap_table_s3_countries_ci95_r2.csv'))
  print(f'Bootstrap B={n_boot} complete in {(time.time() - t0)/60:.1f} min!', flush=True)


# ==============================================================================
# PART 3: UNIFIED S1.6 (FIG. S8) NATIONAL N=1,5,10,20 GAUGE PLACEMENT ON 1,034,083
# ==============================================================================

def run_part_s16(n_threads: int = 32):
  """Clean unified S1.6 national gauge siting across all 1,034,083 basins (9 continents),
  95 features, fixed Round-2 exposure (w_100 = 0.015), evaluating BOTH Train Alone (000)
  and Combined Train+FT+DA (111) under paired-lift modeling with exact river-tree updates.
  """
  os.makedirs(S16_DIR, exist_ok=True)
  t0 = time.time()
  print('\n--- Running Unified Round-2 S1.6 (National N=1,5,10,20 Siting across 1,034,083 basins) ---', flush=True)

  with open(FEATURES_JSON) as f:
    features_95 = json.load(f)
  df_inf = pd.read_parquet(INFER_PARQUET)[features_95]
  all_ids = df_inf.index
  id_to_pos = {h: i for i, h in enumerate(all_ids)}
  df_meta = load_global_metadata_and_exposure(all_ids)
  df_t1a = pd.read_parquet(os.path.join(TIER_DIR, 'tier1a_active_post2016_ccby_8811.parquet')).reindex(all_ids)
  df_masks = pd.read_parquet(os.path.join(TIER_DIR, 'tier_active_and_all_masks_r2.parquet')).reindex(all_ids)
  base_snap_all = df_masks['tier1a_active_post2016_ccby_8811__snap_all'].values.astype(bool)

  # Load baseline Tier 1a predictions & delta_ft_da models
  p_b1a = pd.read_parquet(os.path.join(PREDS_CACHE_DIR, 'base__tier1a_active_post2016_ccby_8811__joint.parquet')).reindex(all_ids)
  p_d1a = pd.read_parquet(os.path.join(PREDS_CACHE_DIR, 'delta_ft_da__tier1a_active_post2016_ccby_8811__joint.parquet')).reindex(all_ids)

  rp_base_mods = {sk: joblib.load(os.path.join(MODELS_DIR, f'base__{mf}.joblib')) for sk, mf in [('r5', '5.0-year-2.0-window-recall'), ('r10', '10.0-year-2.0-window-recall'), ('r20', '20.0-year-2.0-window-recall')]}
  rp_ftda_mods = {sk: joblib.load(os.path.join(MODELS_DIR, f'delta_ft_da__{mf}.joblib')) for sk, mf in [('r5', '5.0-year-2.0-window-recall'), ('r10', '10.0-year-2.0-window-recall'), ('r20', '20.0-year-2.0-window-recall')]}

  # Load HydroBASINS L12 global routing topology (all 1,034,083 basins)
  topo = pd.read_csv(file_paths.HYBAS_ROUTING_TOPOLOGY_PATH, dtype={'HYBAS_ID': str, 'NEXT_DOWN': str})
  topo['hid'] = _norm_hybas(topo['HYBAS_ID'])
  topo['nd'] = np.where(topo['NEXT_DOWN'].astype(str).isin(['0', '0.0', 'nan']), '', _norm_hybas(topo['NEXT_DOWN']))
  topo = topo.drop_duplicates('hid').set_index('hid').reindex(all_ids)

  next_down = np.array([id_to_pos.get(nd, -1) for nd in topo['nd'].fillna('').values], dtype=np.int32)
  sub_area = df_meta['sub_area'].values.astype(np.float64)
  if 'UP_AREA' in topo.columns:
    up_area = np.maximum(pd.to_numeric(topo['UP_AREA'], errors='coerce').fillna(0.0).values.astype(np.float64), sub_area)
  else:
    # Compute exact upstream drainage area topologically from sub_area along next_down
    from collections import deque
    n_b = len(all_ids)
    in_deg = np.zeros(n_b, dtype=np.int32)
    for nd_i in next_down:
      if nd_i >= 0:
        in_deg[nd_i] += 1
    up_area = sub_area.copy()
    q_up = deque(np.where(in_deg == 0)[0].tolist())
    while q_up:
      u = q_up.popleft()
      v = int(next_down[u])
      if v >= 0:
        up_area[v] += up_area[u]
        in_deg[v] -= 1
        if in_deg[v] == 0:
          q_up.append(v)

  children = defaultdict(list)
  for idx_i, nd_i in enumerate(next_down):
    if nd_i >= 0:
      children[nd_i].append(idx_i)

  def get_ancestors_and_descendants(cand_list: list[int]) -> tuple[np.ndarray, dict[int, list[int]], dict[int, list[int]]]:
    """Finds all upstream & downstream connected basins for a candidate set S."""
    anc_map = {}
    desc_map = {}
    affected = set()
    for c in cand_list:
      # Upstream (ancestors) via DFS on children
      anc = []
      stack = [c]
      seen = {c}
      while stack:
        u = stack.pop()
        anc.append(u)
        for ch in children.get(u, ()):
          if ch not in seen:
            seen.add(ch)
            stack.append(ch)
      anc_map[c] = anc
      affected.update(anc)

      # Downstream (descendants) along next_down
      desc = []
      cur = c
      while cur >= 0:
        desc.append(cur)
        affected.add(cur)
        cur = next_down[cur]
      desc_map[c] = desc
    return np.array(sorted(affected), dtype=np.int32), anc_map, desc_map

  def compute_updated_cov(cand_list: list[int], aff_arr: np.ndarray, anc_map: dict, desc_map: dict) -> pd.DataFrame:
    """Computes EXACT updated COV_COLS on affected basins when adding new active gauges `cand_list`
    to baseline active+historical Tier 1a gauges (`base_snap_all`), without double-counting either
    existing upstream/downstream baseline gauges or other candidates in `cand_list`.
    """
    aff_ids = all_ids[aff_arr]
    cov_new = df_t1a.loc[aff_ids, COV_COLS].copy()
    pos_in_aff = {int(gpos): li for li, gpos in enumerate(aff_arr)}
    tuc_old_all = df_t1a['total_upstream_coverage'].values

    cand_set = set(cand_list)
    # Downstream path stopping BEFORE any existing baseline gauge (which already covers 100% of its upstream area)
    unblocked_desc = {}
    for c in cand_list:
      path = []
      for u in desc_map[c]:
        if base_snap_all[u]:
          break
        path.append(u)
      unblocked_desc[c] = path
    unblocked_desc_sets = {c: set(p) for c, p in unblocked_desc.items()}

    # Marginal newly covered area added by candidate c (excluding baseline gauges AND maximal unblocked upstream cands)
    raw_new = {c: max(0.0, up_area[c] * (1.0 - float(tuc_old_all[c]))) for c in cand_list}
    eff_new_area = {}
    for c in cand_list:
      # Upstream candidates whose unblocked downstream path reaches c (so no baseline gauge between y and c)
      up_cands = [y for y in cand_list if y != c and c in unblocked_desc_sets[y]]
      # Maximal among up_cands (not shadowed by another y2 strictly between y and c)
      max_up = [y for y in up_cands if not any((y2 != y) and (y2 in unblocked_desc_sets[y]) for y2 in up_cands)]
      eff_new_area[c] = max(0.0, raw_new[c] - sum(raw_new[y] for y in max_up))

    up_add_area = defaultdict(float)
    up_max_new = defaultdict(float)
    up_cnt_add = defaultdict(int)
    for c in cand_list:
      a_c = up_area[c]
      a_eff = eff_new_area[c]
      for u in unblocked_desc[c]:
        if up_area[u] > 0:
          up_add_area[u] += a_eff / up_area[u]
      for u in desc_map[c]:
        if up_area[u] > 0:
          up_max_new[u] = max(up_max_new[u], a_c / up_area[u])
        up_cnt_add[u] += 1

    # Upstream basins of candidate c: max_downstream_coverage = max(old, up_area[u] / up_area[c])
    down_max_new = defaultdict(float)
    for c in cand_list:
      a_c = up_area[c]
      if a_c <= 0:
        continue
      for u in anc_map[c]:
        down_max_new[u] = max(down_max_new[u], up_area[u] / a_c)

    tuc = cov_new['total_upstream_coverage'].values.copy()
    muc = cov_new['max_upstream_coverage'].values.copy()
    mdc = cov_new['max_downstream_coverage'].values.copy()
    ugc = cov_new['upstream_gauge_count'].values.copy()

    for u, add_f in up_add_area.items():
      li = pos_in_aff[u]
      tuc[li] = min(1.0, tuc[li] + add_f)
    for u, mx_f in up_max_new.items():
      li = pos_in_aff[u]
      muc[li] = min(1.0, max(muc[li], mx_f))
    for u, cnt_f in up_cnt_add.items():
      li = pos_in_aff[u]
      ugc[li] = ugc[li] + cnt_f
    for u, dm_f in down_max_new.items():
      li = pos_in_aff[u]
      mdc[li] = min(1.0, max(mdc[li], dm_f))
    for c in cand_list:
      li = pos_in_aff[c]
      tuc[li] = 1.0
      muc[li] = 1.0
      mdc[li] = 1.0

    cov_new['total_upstream_coverage'] = tuc
    cov_new['max_upstream_coverage'] = muc
    cov_new['max_downstream_coverage'] = mdc
    cov_new['upstream_gauge_count'] = ugc
    return cov_new

  # Select top-20 exposed UNGAUGED candidate basins per country (strictly E_total > 0, not in tier1a physical gauges)
  df_cand_pool = df_meta[(df_meta['country_name'].notna()) & (df_meta['E_total'] > 0) & (~base_snap_all)].copy()
  df_cand_pool['gpos'] = [id_to_pos[h] for h in df_cand_pool.index]

  country_top20_raw = {}
  for cname, grp in df_cand_pool.groupby('country_name'):
    top20 = grp.sort_values('E_total', ascending=False).head(20)['gpos'].tolist()
    if top20:
      country_top20_raw[cname] = top20

  print(f'   Evaluating single candidates across {len(country_top20_raw)} flood-exposed countries...', flush=True)
  e5 = df_meta['E_5'].values
  e10 = df_meta['E_10'].values
  e20p = df_meta['E_20_50_100'].values
  ctry_arr = df_meta['country_name'].values
  w_b1a = e5 * p_b1a['r5'].values + e10 * p_b1a['r10'].values + e20p * p_b1a['r20'].values

  # Precompute transformed base features ONCE
  X_rp_global = rp_base_mods['r5'].impl._preprocess_features(df_inf)
  for c in COV_COLS:
    X_rp_global[c] = df_t1a[c].values
  for c in OOD_COLS:
    X_rp_global[c] = df_t1a[c].values

  def eval_sets_batch(tasks: list[tuple[list[int], str]]) -> list[tuple[float, float, float, float]]:
    """Evaluates many candidate sets in large vectorized chunks (~50k affected rows/chunk)."""
    results = [None] * len(tasks)
    chunk_items = []
    chunk_rows = 0

    def flush_chunk():
      if not chunk_items:
        return
      X_cat = pd.concat([it['X'] for it in chunk_items], axis=0, ignore_index=True)
      knn_c = {}
      pb_cat = _fast_predict_rp_family(rp_base_mods, X_cat, knn_cache=knn_c, n_threads=n_threads)
      pd_cat = _fast_predict_rp_family(rp_ftda_mods, X_cat, knn_cache=knn_c, n_threads=n_threads)
      offset = 0
      for it in chunk_items:
        n = len(it['aff'])
        aff = it['aff']
        sl = slice(offset, offset + n)
        offset += n
        r5_b, r10_b, r20_b = pb_cat['r5'][sl], pb_cat['r10'][sl], pb_cat['r20'][sl]
        r5_d, r10_d, r20_d = pd_cat['r5'][sl], pd_cat['r10'][sl], pd_cat['r20'][sl]
        w_tr = e5[aff] * r5_b + e10[aff] * r10_b + e20p[aff] * r20_b
        is_new = np.isin(aff, it['cands'])
        r5_c = np.where(is_new, np.clip(r5_b + r5_d, 0.0, 1.0), r5_b)
        r10_c = np.where(is_new, np.clip(r10_b + r10_d, 0.0, 1.0), r10_b)
        r20_c = np.where(is_new, np.clip(r20_b + r20_d, 0.0, 1.0), r20_b)
        w_co = e5[aff] * r5_c + e10[aff] * r10_c + e20p[aff] * r20_c
        d_tr = w_tr - w_b1a[aff]
        d_co = w_co - w_b1a[aff]
        in_c = (ctry_arr[aff] == it['cname'])
        results[it['tidx']] = (float(d_tr[in_c].sum()), float(d_co[in_c].sum()), float(d_tr.sum()), float(d_co.sum()))
      chunk_items.clear()

    for tidx, (cand_list, target_country) in enumerate(tasks):
      aff_arr, anc_map, desc_map = get_ancestors_and_descendants(cand_list)
      cov_new = compute_updated_cov(cand_list, aff_arr, anc_map, desc_map)
      X_sub = X_rp_global.iloc[aff_arr].copy()
      for c in COV_COLS:
        X_sub[c] = cov_new[c].values
      chunk_items.append({'tidx': tidx, 'cands': cand_list, 'cname': target_country, 'aff': aff_arr, 'X': X_sub})
      chunk_rows += len(aff_arr)
      if chunk_rows >= 80000:
        flush_chunk()
        chunk_rows = 0
    flush_chunk()
    return results

  country_base_exp = df_meta[df_meta['country_name'].notna()].groupby('country_name')['E_total'].sum().to_dict()
  country_unw_exp = pd.Series(df_meta['E_total'].values - w_b1a, index=all_ids).groupby(df_meta['country_name']).sum().to_dict()

  # Phase 1: Batch-evaluate ALL single candidates globally
  single_tasks = []
  single_meta = []
  for cname, raw_cands in sorted(country_top20_raw.items()):
    for c in raw_cands:
      single_tasks.append(([c], cname))
      single_meta.append((cname, c))
  print(f'   Phase 1: Batch-evaluating {len(single_tasks)} single candidates...', flush=True)
  single_res = eval_sets_batch(single_tasks)
  scored_by_country = defaultdict(list)
  for (cname, c), (dt_c, dc_c, dt_g, dc_g) in zip(single_meta, single_res):
    scored_by_country[cname].append((c, dt_c, dc_c, dt_g, dc_g))

  # Phase 2: Batch-evaluate ALL joint N in {1, 5, 10, 20} sets for both Train Alone and Combined
  joint_tasks = []
  joint_meta = []
  for cname in sorted(country_top20_raw.keys()):
    scored = scored_by_country[cname]
    ranked_train = [x[0] for x in sorted(scored, key=lambda x: x[1], reverse=True)]
    ranked_comb = [x[0] for x in sorted(scored, key=lambda x: x[2], reverse=True)]
    for N in (1, 5, 10, 20):
      joint_tasks.append((ranked_train[: min(N, len(ranked_train))], cname))
      joint_meta.append((cname, N, 'train'))
      joint_tasks.append((ranked_comb[: min(N, len(ranked_comb))], cname))
      joint_meta.append((cname, N, 'comb'))
  print(f'   Phase 2: Batch-evaluating {len(joint_tasks)} joint N=1,5,10,20 sets...', flush=True)
  joint_res = eval_sets_batch(joint_tasks)

  nat_rows_map = {}
  for cname in sorted(country_top20_raw.keys()):
    base_e = float(country_base_exp.get(cname, 0.0))
    unw_e = float(country_unw_exp.get(cname, 0.0))
    nat_rows_map[cname] = {
        'Country': cname,
        'Baseline Exposure (people/yr)': base_e,
        'Unwarned Exposure (people/yr)': unw_e,
        'Max_Evaluated_Candidates': len(country_top20_raw[cname]),
        'has_system': cap_utils.has_effective_flood_warning_system(cname),
        'CAP_Tier': cap_utils.assign_cap_tier(cname),
    }
  for (cname, N, kind), (dt_c, dc_c, dt_g, dc_g) in zip(joint_meta, joint_res):
    row = nat_rows_map[cname]
    base_e = row['Baseline Exposure (people/yr)']
    if kind == 'train':
      row[f'N={N}_Train_Dom'] = dt_c
      row[f'N={N}_Train_Dom_PctBase'] = 100.0 * dt_c / base_e if base_e > 0 else 0.0
      row[f'N={N}_Train_Glb'] = dt_g
    else:
      row[f'N={N}_Combined_Dom'] = dc_c
      row[f'N={N}_Combined_Dom_PctBase'] = 100.0 * dc_c / base_e if base_e > 0 else 0.0
      row[f'N={N}_Combined_Glb'] = dc_g
  nat_rows = list(nat_rows_map.values())

  df_nat = pd.DataFrame(nat_rows).sort_values('N=20_Combined_Dom', ascending=False).reset_index(drop=True)
  out_nat = os.path.join(S16_DIR, 'country_joint_n_gauge_r2.csv')
  df_nat.to_csv(out_nat, index=False)
  print(f'Saved unified Round-2 S1.6 national table ({len(df_nat)} countries) to {out_nat} in {(time.time() - t0)/60:.1f} min!', flush=True)
  for N in (1, 5, 10, 20):
    print(
        f"   N={N:2d}: TrainAlone med={df_nat[f'N={N}_Train_Dom_PctBase'].median():.2f}% sum={df_nat[f'N={N}_Train_Dom'].sum()/1e6:.2f}M/yr | "
        f"Combined(Train+FT+DA) med={df_nat[f'N={N}_Combined_Dom_PctBase'].median():.2f}% sum={df_nat[f'N={N}_Combined_Dom'].sum()/1e6:.2f}M/yr",
        flush=True,
    )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--part',
      choices=['table_s3', 'counterfactuals', 'bootstrap', 's16', 'all'],
      default='all',
  )
  parser.add_argument('--threads', type=int, default=min(32, os.cpu_count() or 8))
  parser.add_argument('--n-boot', type=int, default=20)
  parser.add_argument('--boot-workers', type=int, default=7)
  parser.add_argument('--boot-threads', type=int, default=8)
  parser.add_argument('--overwrite', action='store_true')
  args = parser.parse_args()

  if args.part == 'table_s3':
    run_part_table_s3()
    return

  if not os.path.exists(EXP_PARQUET):
    print(
        f'[Notice] Licensed Fathom exposure file not found at {EXP_PARQUET}.\n'
        '         Running public non-Fathom Table S3 build and skipping Fathom-dependent '
        'exposure counterfactuals / bootstrap / S1.6 placement.\n'
        '         To run exposure counterfactuals, license Fathom v3 hazard maps independently '
        'and stage global_basin_exposure_defended_r2.parquet via stage_b0_merge_and_audit_exposure.py.',
        flush=True,
    )
    if args.part in ('counterfactuals', 'all'):
      run_part_table_s3()
    return

  if args.part in ('counterfactuals', 'all'):
    run_part_counterfactuals(n_threads=args.threads, overwrite=args.overwrite)
  if args.part in ('bootstrap', 'all'):
    run_part_bootstrap(n_boot=args.n_boot, workers=args.boot_workers, threads=args.boot_threads)
  if args.part in ('s16', 'all'):
    run_part_s16(n_threads=args.threads)


if __name__ == '__main__':
  main()
