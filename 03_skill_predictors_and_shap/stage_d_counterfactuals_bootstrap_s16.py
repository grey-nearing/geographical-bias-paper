#!/usr/bin/env python3
"""Stage D: Round-3 Counterfactuals, Table S3, A1 Warning-to-Action Bounds (Recall/CSI/Bayesian/Coping),
A2 Universal Bounds (No-OOD-Clamp vs. Idealized OOD->Q75 Ceiling), A3 B=200 Stratified MAIN_BAS x Climate
Block Bootstrap + OOD-Scaled Cross-Domain Residual Propagation, A5 Spatial N_eff Stats, A6 Strict >=2022
Daily-Discharge DA Sensitivity, and Unified S1.6 (Fig. S8) National Gauge Siting.

Operates on ALL 1,034,083 global HydroBASINS L12 watersheds (9 continents), fixed
Fathom v3.1 x WorldPop 2020 exposure (w_100 = 0.015, sum w = 0.200) when locally
staged, 95-feature regularized cross-domain paired-lift SkillPredictorR2 models,
scale-normalized OOD tiers, and strict physical G-set (`snap_active` / `snap_all`)
as primary alongside `snap_strict2022`, `union`, and `ge95` sensitivity columns.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
import sys
import time
import warnings

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '2')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.environ.get('GEO_BIAS_REPO_DIR', os.path.dirname(SCRIPT_DIR))
for d in (SCRIPT_DIR, REPO_DIR):
  if d not in sys.path:
    sys.path.insert(0, d)

CANON_DIR = os.environ.get('GEO_BIAS_DATA_DIR', './data')
os.environ.setdefault('GEO_BIAS_DATA_DIR', CANON_DIR)
os.environ.setdefault('GEO_BIAS_THIRD_PARTY_DIR', CANON_DIR)
os.environ.setdefault('GEO_BIAS_OUTPUT_DIR', CANON_DIR)

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils import cap_utils, file_paths, spatial_stats
from stage_c_skill_predictor import (
    CLASSIC_METRICS,
    DEFAULT_REG_PRESET,
    SkillPredictorR2,
    get_metric_bounds,
)
from stage_c_train_meta_models import (
    PRIORITY1_METRICS,
    PRIORITY2_METRICS,
    load_3split_target,
    load_train_station_metadata,
)

warnings.filterwarnings('ignore')

ROUND_SUBDIR = os.environ.get('GEO_BIAS_ROUND_SUBDIR', 'round2')
ROUND_DIR = os.path.join(CANON_DIR, ROUND_SUBDIR)
META_DIR = os.path.join(ROUND_DIR, 'meta_models')
MODELS_DIR = os.path.join(META_DIR, 'models')
GLOBAL_PREDS_DIR = os.path.join(META_DIR, 'global_preds')
CV_DIR = os.path.join(META_DIR, 'cv_preds')
CV_CLIMATE_DIR = os.path.join(META_DIR, 'cv_preds_climate')
CV_LOCO_DIR = os.path.join(META_DIR, 'cv_preds_loco')
TIER_DIR = os.path.join(ROUND_DIR, 'tier_features')

_EXP_CAND = os.path.join(ROUND_DIR, 'exposure', 'global_basin_exposure_defended_r2.parquet')
EXP_PARQUET = (
    _EXP_CAND
    if os.path.exists(_EXP_CAND)
    else os.path.join(CANON_DIR, 'round2', 'exposure', 'global_basin_exposure_defended_r2.parquet')
)
INFER_PARQUET = os.path.join(META_DIR, 'hybas_inference_95_float32.parquet')
TRAIN_PARQUET = os.path.join(META_DIR, 'train_95_float32.parquet')
FEATURES_JSON = os.path.join(META_DIR, 'features_95.json')

OUT_DIR = os.path.join(ROUND_DIR, 'counterfactuals')
PREDS_CACHE_DIR = os.path.join(OUT_DIR, 'tier_preds')
BOOT_DIR = os.path.join(OUT_DIR, 'bootstrap')
S16_DIR = os.path.join(ROUND_DIR, 's16_gauge_placement')

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
CLIM_COLS = ['pre_mm_uyr', 'pet_mm_uyr', 'tmp_dc_uyr', 'ari_ix_uav', 'snw_pc_uyr']

TIER_ORDER = [
    'tier0_zero_gauges',
    'tier1a_active_post2016_ccby_8811',
    'tier1_open_compilations',
    'tier2_public_discharge',
    'tier4_all_discoverable',
    'tier5_universal_no_ood_clamp',
    'tier5_universal_idealized_ceiling',
    'tier5_dense_corridor_ceiling',
]

TIER_LABELS = {
    'tier0_zero_gauges': 'Zero-Gauge Counterfactual (0)',
    'tier1a_active_post2016_ccby_8811': 'Baseline: Active Post-2016 Caravan (8,811 / 8,075 phys)',
    'tier1_open_compilations': 'Tier 1: Standardized, Open-License Discharge Datasets (32,369 phys)',
    'tier2_public_discharge': 'Tier 2: Gauges with Discharge Records (82,596 phys)',
    'tier4_all_discoverable': 'Tier 3: All Publicly Discoverable Stream Gauges (119,169 phys)',
    'tier5_universal_no_ood_clamp': 'Universal Gauging — Direct Local FT+DA (no OOD clamp)',
    'tier5_universal_idealized_ceiling': 'Universal Gauging — Idealized Global-Training Ceiling (OOD -> Q75)',
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
    ('p50', '50.0-year-2.0-window-precision'),
    ('f1_5', '5.0-year-2.0-window-f1'),
    ('f1_10', '10.0-year-2.0-window-f1'),
    ('f1_20', '20.0-year-2.0-window-f1'),
]
EVAL_CLASSIC_METRICS = [
    ('NSE', 'NSE'),
    ('KGE', 'KGE'),
]

BOOT_RP_METRICS = [
    ('r5', '5.0-year-2.0-window-recall'),
    ('r10', '10.0-year-2.0-window-recall'),
    ('r20', '20.0-year-2.0-window-recall'),
    ('p5', '5.0-year-2.0-window-precision'),
    ('p10', '10.0-year-2.0-window-precision'),
    ('p20', '20.0-year-2.0-window-precision'),
]

BOOT_TIERS = [
    'tier1a_active_post2016_ccby_8811',
    'tier1_open_compilations',
    'tier2_public_discharge',
    'tier4_all_discoverable',
    'tier5_universal_no_ood_clamp',
    'tier5_universal_idealized_ceiling',
]


def _norm_hybas(s: pd.Series | pd.Index) -> np.ndarray:
  s2 = pd.Series(s).astype(str).str.replace(r'\.0$', '', regex=True)
  return np.where(s2.str.startswith('hybas_'), s2, 'hybas_' + s2)


def _csi_vec(r: np.ndarray, p: np.ndarray) -> np.ndarray:
  """Exact Critical Success Index = 1 / (1/R + 1/P - 1) = (CSI/R) * R."""
  r_pos = np.maximum(r, 1e-6)
  p_pos = np.maximum(p, 1e-6)
  return np.where((r > 1e-6) & (p > 1e-6), 1.0 / (1.0 / r_pos + 1.0 / p_pos - 1.0), 0.0)


def _load_inform_and_centroids() -> tuple[dict, dict, dict, dict]:
  """Loads INFORM 2026 Vulnerability & Lack of Coping Capacity + Natural Earth centroids."""
  file_paths.init('5.0-year-2.0-window-recall')
  world = gpd.read_file(file_paths.NATURAL_EARTH_PATH)
  world = world.dropna(subset=[world.geometry.name])
  world = world[world.is_valid & (~world.is_empty)].copy()
  world['map_name'] = world['name'].replace(file_paths.country_name_map)
  world['iso_clean'] = world['iso_a3'].astype(str).str.strip()
  for nm, iso in [('France', 'FRA'), ('Norway', 'NOR'), ('Kosovo', 'XKX'), ('Somaliland', 'SOL')]:
    world.loc[world['map_name'] == nm, 'iso_clean'] = iso
  c2iso = dict(zip(world['map_name'], world['iso_clean']))

  w_diss = world[['map_name', 'geometry']].dissolve(by='map_name')
  w_rep = w_diss.geometry.representative_point()
  c2lat = dict(zip(w_diss.index, w_rep.y))
  c2lon = dict(zip(w_diss.index, w_rep.x))

  df_inf = pd.read_csv(file_paths.RISK_FILE, skiprows=[0, 2])
  df_inf['ISO3'] = df_inf['ISO3'].astype(str).str.strip()
  df_inf['COUNTRY_CLEAN'] = df_inf['COUNTRY'].astype(str).str.strip().replace(file_paths.country_name_map)

  vuln_map, cop_map = {}, {}
  for src_col, dst_dict in [
      ('VULNERABILITY', vuln_map),
      ('LACK OF COPING CAPACITY', cop_map),
  ]:
    df_inf[src_col] = pd.to_numeric(df_inf[src_col], errors='coerce')
    iso_m = df_inf.dropna(subset=['ISO3', src_col]).drop_duplicates('ISO3').set_index('ISO3')[src_col]
    nam_m = df_inf.dropna(subset=['COUNTRY_CLEAN', src_col]).drop_duplicates('COUNTRY_CLEAN').set_index('COUNTRY_CLEAN')[src_col]
    for cname in set(world['map_name']).union(set(nam_m.index)):
      val = iso_m.get(c2iso.get(cname), nam_m.get(cname, np.nan))
      if pd.notna(val):
        dst_dict[cname] = float(val)
  return vuln_map, cop_map, c2lat, c2lon


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
  """Loads 1,034,083 exposure (>0.1m & >0.0m), country, CAP, lat/lon, and INFORM coping capacity."""
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

  _, cop_map, _, _ = _load_inform_and_centroids()
  cop_gap = df['country_name'].map(cop_map)
  med_cop = float(np.nanmedian(list(cop_map.values()))) if cop_map else 5.0
  df['inform_coping_cap'] = (1.0 - cop_gap.fillna(med_cop).values / 10.0).clip(0.0, 1.0)
  return df


def _fast_predict_rp_family(
    models_dict: dict[str, SkillPredictorR2],
    X_rp: pd.DataFrame,
    n_threads: int = 24,
) -> dict[str, np.ndarray]:
  """Predicts all return-period metrics in a family using regularized HGB+LGB+XGB+Ridge stack."""
  out = {}
  x_cols = list(X_rp.columns)
  for short_k, mod in models_dict.items():
    mod.set_n_threads(n_threads)
    impl = mod.impl
    if x_cols == impl.feature_columns:
      X_al = X_rp
    else:
      X_al = X_rp.reindex(columns=impl.feature_columns, fill_value=0)
    out[short_k] = impl.predict_from_preprocessed(X_al).astype(np.float32)
  return out


def _fast_predict_classic_family(
    models_dict: dict[str, SkillPredictorR2],
    X_cl: pd.DataFrame,
    n_threads: int = 24,
) -> dict[str, np.ndarray]:
  out = {}
  for short_k, mod in models_dict.items():
    mod.set_n_threads(n_threads)
    impl = mod.impl
    X_al = X_cl.reindex(columns=impl.feature_names, fill_value=0)
    out[short_k] = impl.predict_from_engineered(X_al).astype(np.float32)
  return out


def build_table_s1(df_meta: pd.DataFrame):
  """Builds Table S3 across all completed Stage C metrics using paired lifts and multi-level CV R^2."""
  print('\n--- Building Table S3 (multi-level spatial & cross-domain CV R^2 across 4 regimes) ---', flush=True)
  meta, n_g = load_train_station_metadata()
  slice_ids = meta.index
  df_cv_sum = pd.read_csv(os.path.join(META_DIR, 'stage_c_cv_summary.csv'))
  cv_idx = df_cv_sum.set_index(['metric', 'regime'])

  has_sys = df_meta['has_system'].values
  is_mapped = df_meta['country_name'].notna().values
  m_sys = has_sys & is_mapped
  m_nosys = (~has_sys) & is_mapped

  all_metrics = PRIORITY1_METRICS + PRIORITY2_METRICS
  rows = []
  skipped = 0

  def _cv_col(m_name: str, reg_name: str, col: str) -> float:
    if (m_name, reg_name) in cv_idx.index and col in cv_idx.columns:
      val = cv_idx.loc[(m_name, reg_name), col]
      return float(val) if pd.notna(val) else np.nan
    return np.nan

  def _med_iqr(arr):
    v = arr[np.isfinite(arr)]
    if len(v) == 0:
      return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0
    q10, q25, q50, q75, q90 = np.percentile(v, [10, 25, 50, 75, 90])
    return float(q50), float(q10), float(q25), float(q75), float(q90), float(q75 - q25), int(len(v))

  for m in all_metrics:
    base_pq = os.path.join(GLOBAL_PREDS_DIR, f'base__{m}.parquet')
    if not os.path.exists(base_pq):
      skipped += 1
      continue
    lo, hi = get_metric_bounds(m)
    y_all = load_3split_target(m, '', slice_ids)
    y_g, y_r, y_s = y_all[:n_g], y_all[n_g : 2 * n_g], y_all[2 * n_g :]
    y_u = np.concatenate([y_r, y_s])

    p_base_g = pd.read_parquet(base_pq)['pred'].reindex(df_meta.index).values
    obs_u_med, _, obs_u_q25, obs_u_q75, _, obs_u_iqr, n_u = _med_iqr(y_u)
    obs_g_med, _, _, _, _, _, _ = _med_iqr(y_g)
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
        'meta_cv_r2_3split': _cv_col(m, 'base', 'r2_all'),
        'meta_cv_r2_2split_ung': _cv_col(m, 'base_2split', 'r2_all'),
        'meta_cv_r2_mainbas_all': _cv_col(m, 'base', 'r2_mainbas_all'),
        'meta_cv_r2_station_all': _cv_col(m, 'base', 'r2_station_all'),
        'meta_cv_r2_climate_all': _cv_col(m, 'base', 'r2_climate_all'),
        'meta_cv_r2_loco_all': _cv_col(m, 'base', 'r2_loco_all'),
        'meta_cv_r2_delta_direct': np.nan,
        'pred_delta_neg_pct': np.nan,
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
      d_pq = os.path.join(GLOBAL_PREDS_DIR, f'{reg}__{m}.parquet')
      if (m, reg) not in cv_idx.index or not os.path.exists(d_pq):
        continue
      sfx = reg[len('delta') :]
      ym_all = load_3split_target(m, sfx, slice_ids)
      ym_g, ym_r, ym_s = ym_all[:n_g], ym_all[n_g : 2 * n_g], ym_all[2 * n_g :]
      ym_u = np.concatenate([ym_r, ym_s])
      d_g = pd.read_parquet(d_pq)['pred'].reindex(df_meta.index).values
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
          'meta_cv_r2_3split': _cv_col(m, reg, 'recon_r2_all'),
          'meta_cv_r2_2split_ung': _cv_col(m, reg, 'recon_r2_ung'),
          'meta_cv_r2_mainbas_all': _cv_col(m, reg, 'recon_r2_mainbas_all'),
          'meta_cv_r2_station_all': _cv_col(m, reg, 'recon_r2_station_all'),
          'meta_cv_r2_climate_all': _cv_col(m, reg, 'recon_r2_climate_all'),
          'meta_cv_r2_loco_all': _cv_col(m, reg, 'recon_r2_loco_all'),
          'meta_cv_r2_delta_direct': _cv_col(m, reg, 'r2_all'),
          'pred_delta_neg_pct': _cv_col(m, reg, 'pred_delta_neg_pct'),
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
  print(f'Saved {out_p3} ({len(df_ts1)} rows across {df_ts1["metric"].nunique()} metrics; skipped={skipped} pending Track B)!', flush=True)
  return df_ts1


def run_part_table_s3():
  """Builds Table S3 using ONLY public data (zero Fathom dependency)."""
  os.makedirs(OUT_DIR, exist_ok=True)
  all_ids = pd.read_parquet(INFER_PARQUET, columns=['pre_mm_uyr']).index
  assert len(all_ids) == 1034083
  df_meta = load_global_metadata_non_fathom(all_ids)
  return build_table_s1(df_meta)


def run_part_counterfactuals(n_threads: int = 24, overwrite: bool = False):
  os.makedirs(OUT_DIR, exist_ok=True)
  os.makedirs(PREDS_CACHE_DIR, exist_ok=True)
  t0 = time.time()

  with open(FEATURES_JSON) as f:
    features_95 = json.load(f)
  df_inf = pd.read_parquet(INFER_PARQUET)[features_95]
  all_ids = df_inf.index
  assert len(all_ids) == 1034083

  print(f'1. Loading exposure & global metadata (round={ROUND_SUBDIR}, 1,034,083 basins)...', flush=True)
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

  X_rp_base = rp_models['base']['r5'].impl._preprocess_features(df_inf)
  X_cl_base = cl_models['base']['NSE'].impl._engineer_features(df_inf)

  cov_t1a = tier_feats['tier1a_active_post2016_ccby_8811'][COV_COLS]
  ood_t1a = tier_feats['tier1a_active_post2016_ccby_8811'][OOD_COLS]

  def eval_combo(
      fam: str,
      combo_key: str,
      cov_df: pd.DataFrame,
      ood_df: pd.DataFrame,
      eval_mask: np.ndarray | None = None,
  ) -> dict[str, np.ndarray]:
    pq = os.path.join(PREDS_CACHE_DIR, f'{fam}__{combo_key}.parquet')
    if (not overwrite) and os.path.exists(pq):
      dfc = pd.read_parquet(pq).reindex(all_ids)
      expected_keys = set(rp_models[fam].keys()) | set(cl_models[fam].keys())
      if expected_keys.issubset(set(dfc.columns)):
        return {c: dfc[c].values for c in dfc.columns}

    ts = time.time()
    n_all = len(all_ids)
    all_keys = list(rp_models[fam].keys()) + list(cl_models[fam].keys())
    if eval_mask is not None:
      idx_sub = np.flatnonzero(eval_mask)
      if len(idx_sub) == 0:
        res = {k: np.zeros(n_all, dtype=np.float32) for k in all_keys}
        pd.DataFrame(res, index=all_ids).to_parquet(pq)
        return res
      X_rp = X_rp_base.iloc[idx_sub].copy()
      for c in COV_COLS:
        X_rp[c] = cov_df[c].values[idx_sub]
      for c in OOD_COLS:
        X_rp[c] = ood_df[c].values[idx_sub]
      sub_res = _fast_predict_rp_family(rp_models[fam], X_rp, n_threads=n_threads)

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
    res = _fast_predict_rp_family(rp_models[fam], X_rp, n_threads=n_threads)

    X_cl = X_cl_base.copy()
    for c in COV_COLS:
      X_cl[c] = cov_df[c].values
    for c in OOD_COLS:
      X_cl[c] = ood_df[c].values
    res.update(_fast_predict_classic_family(cl_models[fam], X_cl, n_threads=n_threads))

    pd.DataFrame(res, index=all_ids).to_parquet(pq)
    print(f'   [{fam}__{combo_key}] full done in {time.time() - ts:.1f}s (med r5={np.median(res["r5"]):+.4f})', flush=True)
    return res

  FULL_DELTA_TIERS = {
      'tier1a_active_post2016_ccby_8811',
      'tier5_universal_no_ood_clamp',
      'tier5_universal_idealized_ceiling',
      'tier5_dense_corridor_ceiling',
  }
  F2F3_TIERS = {
      'tier1_open_compilations',
      'tier2_public_discharge',
      'tier4_all_discoverable',
      'tier5_universal_no_ood_clamp',
      'tier5_universal_idealized_ceiling',
      'tier5_dense_corridor_ceiling',
  }
  print(f'\n3. Evaluating all {len(TIER_ORDER)} tiers across regularized paired-lift families...', flush=True)
  evals = {}
  for tk in TIER_ORDER:
    cov_k = tier_feats[tk][COV_COLS]
    ood_k = tier_feats[tk][OOD_COLS]
    evals[('base', tk, 'joint')] = eval_combo('base', f'{tk}__joint', cov_k, ood_k, eval_mask=None)
    for fam in ('delta_ft', 'delta_da', 'delta_ft_da'):
      if tk in FULL_DELTA_TIERS:
        em = None
      elif fam == 'delta_ft':
        em = df_masks[f'{tk}__union_all'].values.astype(bool)
      else:
        em = df_masks[f'{tk}__union_active'].values.astype(bool)
      evals[(fam, tk, 'joint')] = eval_combo(fam, f'{tk}__joint', cov_k, ood_k, eval_mask=em)

    if tk in F2F3_TIERS:
      if tk == 'tier5_universal_idealized_ceiling' and ('base', 'tier5_universal_no_ood_clamp', 'f2_cov_only_base1a') in evals:
        evals[('base', tk, 'f2_cov_only_base1a')] = evals[('base', 'tier5_universal_no_ood_clamp', 'f2_cov_only_base1a')]
      else:
        evals[('base', tk, 'f2_cov_only_base1a')] = eval_combo(
            'base', f'{tk}__f2_cov_only_base1a', cov_k, ood_t1a, eval_mask=None
        )
      if tk == 'tier5_dense_corridor_ceiling':
        evals[('base', tk, 'f3_ood_only_base1a')] = evals[('base', 'tier5_universal_idealized_ceiling', 'f3_ood_only_base1a')]
      elif tk == 'tier5_universal_no_ood_clamp':
        evals[('base', tk, 'f3_ood_only_base1a')] = evals[('base', 'tier4_all_discoverable', 'f3_ood_only_base1a')]
      else:
        evals[('base', tk, 'f3_ood_only_base1a')] = eval_combo(
            'base', f'{tk}__f3_ood_only_base1a', cov_t1a, ood_k, eval_mask=None
        )
    elif tk == 'tier0_zero_gauges':
      evals[('base', tk, 'f2_cov_only_base1a')] = evals[('base', tk, 'joint')]
    elif tk == 'tier1a_active_post2016_ccby_8811':
      evals[('base', tk, 'f2_cov_only_base1a')] = evals[('base', tk, 'joint')]
      evals[('base', tk, 'f3_ood_only_base1a')] = evals[('base', tk, 'joint')]
  evals[('base', 'tier0_zero_gauges', 'f3_ood_only_base1a')] = evals[('base', 'tier1a_active_post2016_ccby_8811', 'joint')]

  metric_lookup = dict(EVAL_RP_METRICS + EVAL_CLASSIC_METRICS)

  def reconstruct(p_base: dict, p_delta: dict) -> dict[str, np.ndarray]:
    out = {}
    for k in p_base:
      if k in p_delta:
        lo, hi = get_metric_bounds(metric_lookup[k])
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
  cop_cap = df_meta['inform_coping_cap'].values

  e5_0 = df_meta['E_gt0_5'].values
  e10_0 = df_meta['E_gt0_10'].values
  e20p_0 = df_meta['E_gt0_20_50_100'].values

  def calc_warned(p: dict, mode: str = 'recall') -> np.ndarray:
    if mode == 'recall':
      return e5 * p['r5'] + e10 * p['r10'] + e20p * p['r20']
    if mode == 'recall_gt0':
      return e5_0 * p['r5'] + e10_0 * p['r10'] + e20p_0 * p['r20']
    if mode == 'recall_r50direct':
      return e5 * p['r5'] + e10 * p['r10'] + e20 * p['r20'] + (e50 + e100) * p['r50']
    if mode == 'recall_decay_mild':
      return e5 * p['r5'] + e10 * p['r10'] + (e20 + 0.85 * e50 + 0.70 * e100) * p['r20']
    if mode == 'recall_decay_steep':
      return e5 * p['r5'] + e10 * p['r10'] + (e20 + 0.50 * e50 + 0.25 * e100) * p['r20']
    if mode == 'f1':
      return e5 * p['f1_5'] + e10 * p['f1_10'] + e20p * p['f1_20']
    if mode == 'csi':
      return e5 * _csi_vec(p['r5'], p['p5']) + e10 * _csi_vec(p['r10'], p['p10']) + e20p * _csi_vec(p['r20'], p['p20'])
    if mode == 'csi_p50direct':
      p50_use = p['p50'] if 'p50' in p else p['p20']
      return (
          e5 * _csi_vec(p['r5'], p['p5'])
          + e10 * _csi_vec(p['r10'], p['p10'])
          + e20 * _csi_vec(p['r20'], p['p20'])
          + (e50 + e100) * _csi_vec(p['r50'], p50_use)
      )
    if mode == 'bayesian_pr':
      return e5 * p['p5'] * p['r5'] + e10 * p['p10'] * p['r10'] + e20p * p['p20'] * p['r20']
    if mode == 'false_alarms':
      # Expected person-false-alarms/yr = sum_k w_k E_k R_k (1 - P_k) / max(P_k, 0.05)
      fp5 = e5 * p['r5'] * (1.0 - p['p5']) / np.maximum(p['p5'], 0.05)
      fp10 = e10 * p['r10'] * (1.0 - p['p10']) / np.maximum(p['p10'], 0.05)
      fp20 = e20p * p['r20'] * (1.0 - p['p20']) / np.maximum(p['p20'], 0.05)
      return fp5 + fp10 + fp20
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
  p_b4 = evals[('base', 'tier4_all_discoverable', 'joint')]

  w_b0 = calc_warned(p_b0, 'recall')
  w_b1a = calc_warned(p_b1a, 'recall')
  w_b4 = calc_warned(p_b4, 'recall')
  w_csi_b1a = calc_warned(p_b1a, 'csi')
  w_bpr_b1a = calc_warned(p_b1a, 'bayesian_pr')

  is_mapped = df_meta['country_name'].notna().values
  m_sys60 = df_meta['has_system'].values & is_mapped
  m_nosys115 = (~df_meta['has_system'].values) & is_mapped
  m_unmapped = ~is_mapped

  print('\n4. Computing multi-mechanism summary across all tiers (Recall + CSI + Bayesian + strict2022)...', flush=True)
  summary_rows = []
  sens_rows = []
  country_vec_store = {}
  country_csi_store = {}

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

    mask_specs = [
        ('snap', f'{tk}__snap_all', f'{tk}__snap_active'),
        ('snap_main', f'{tk}__snap_main_all', f'{tk}__snap_main_active'),
        ('union', f'{tk}__union_all', f'{tk}__union_active'),
        ('ge95', f'{tk}__ge95_all', f'{tk}__ge95_active'),
    ]
    strict_col = f'{tk}__snap_active_strict2022'
    if strict_col in df_masks.columns:
      mask_specs.append(('snap_strict2022', f'{tk}__snap_all', strict_col))

    for mask_type, all_col, act_col in mask_specs:
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

      w_csi = calc_warned(p, 'csi')
      unw_csi = etot - w_csi
      d_csi = w_csi - w_csi_b1a

      w_bpr = calc_warned(p, 'bayesian_pr')
      d_bpr = w_bpr - w_bpr_b1a
      fp_vec = calc_warned(p, 'false_alarms')

      summary_rows.append({
          'tier_key': tk,
          'tier_label': TIER_LABELS[tk],
          'mechanism': mech_name,
          'n_snap_all': int(df_masks[f'{tk}__snap_all'].sum()),
          'n_snap_active': int(df_masks[f'{tk}__snap_active'].sum()),
          'n_snap_active_strict2022': int(df_masks[strict_col].sum()) if strict_col in df_masks.columns else 0,
          'n_ge95_all': int(df_masks[f'{tk}__ge95_all'].sum()),
          'n_ge95_active': int(df_masks[f'{tk}__ge95_active'].sum()),
          'r5_med_pct': float(np.median(p['r5']) * 100.0),
          'r10_med_pct': float(np.median(p['r10']) * 100.0),
          'r20_med_pct': float(np.median(p['r20']) * 100.0),
          'r50_med_pct': float(np.median(p['r50']) * 100.0) if 'r50' in p else np.nan,
          'p5_med_pct': float(np.median(p['p5']) * 100.0),
          'p10_med_pct': float(np.median(p['p10']) * 100.0),
          'p20_med_pct': float(np.median(p['p20']) * 100.0),
          'p50_med_pct': float(np.median(p['p50']) * 100.0) if 'p50' in p else np.nan,
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
          # A1: False-alarm discounted CSI (alpha = CSI/R)
          'warned_csi_global': float(w_csi.sum()),
          'unwarned_csi_global': float(unw_csi.sum()),
          'pct_warned_csi_global': float(100.0 * w_csi.sum() / etot.sum()),
          'delta_net_csi_global': float(d_csi.sum()),
          'pct_unw_closed_csi_global': float(100.0 * d_csi.sum() / (etot.sum() - w_csi_b1a.sum())),
          'warned_csi_sys60': float(w_csi[m_sys60].sum()),
          'delta_net_csi_sys60': float(d_csi[m_sys60].sum()),
          'warned_csi_nosys115': float(w_csi[m_nosys115].sum()),
          'unwarned_csi_nosys115': float(unw_csi[m_nosys115].sum()),
          'delta_net_csi_nosys115': float(d_csi[m_nosys115].sum()),
          'pct_unw_closed_csi_nosys115': float(100.0 * d_csi[m_nosys115].sum() / (etot[m_nosys115].sum() - w_csi_b1a[m_nosys115].sum())),
          # A1 SM: Bayesian credibility alpha = P and annual false-alarm volume
          'warned_bayesian_pr_global': float(w_bpr.sum()),
          'warned_bayesian_pr_nosys115': float(w_bpr[m_nosys115].sum()),
          'delta_net_bayesian_pr_global': float(d_bpr.sum()),
          'delta_net_bayesian_pr_nosys115': float(d_bpr[m_nosys115].sum()),
          'false_alarms_global': float(fp_vec.sum()),
          'false_alarms_nosys115': float(fp_vec[m_nosys115].sum()),
          'false_alarms_sys60': float(fp_vec[m_sys60].sum()),
      })

      if tk in (
          'tier1a_active_post2016_ccby_8811',
          'tier1_open_compilations',
          'tier2_public_discharge',
          'tier4_all_discoverable',
          'tier5_universal_no_ood_clamp',
          'tier5_universal_idealized_ceiling',
      ) and mech_name in (
          '000_global_train_joint',
          '111_all_three_combined_snap',
          '111_all_three_combined_union',
          '111_all_three_combined_snap_strict2022',
      ):
        w_gt0 = calc_warned(p, 'recall_gt0')
        w_r50 = calc_warned(p, 'recall_r50direct')
        w_dm = calc_warned(p, 'recall_decay_mild')
        w_ds = calc_warned(p, 'recall_decay_steep')
        w_f1 = calc_warned(p, 'f1')
        w_csi_p50 = calc_warned(p, 'csi_p50direct')
        w_cop_rec = cop_cap * w_vec
        w_cop_csi = cop_cap * w_csi
        w_cop_rec_b1a = cop_cap * w_b1a
        w_cop_csi_b1a = cop_cap * w_csi_b1a

        far5 = 1.0 - p['p5']
        far10 = 1.0 - p['p10']
        far20 = 1.0 - p['p20']
        far50 = 1.0 - (p['p50'] if 'p50' in p else p['p20'])
        ew_far = (e5 * far5 + e10 * far10 + e20p * far20).sum() / etot.sum()
        ew_far_nosys = (e5[m_nosys115] * far5[m_nosys115] + e10[m_nosys115] * far10[m_nosys115] + e20p[m_nosys115] * far20[m_nosys115]).sum() / etot[m_nosys115].sum()
        ew_far_p50 = (e5 * far5 + e10 * far10 + e20 * far20 + (e50 + e100) * far50).sum() / etot.sum()
        ew_far_p50_nosys = (
            e5[m_nosys115] * far5[m_nosys115]
            + e10[m_nosys115] * far10[m_nosys115]
            + e20[m_nosys115] * far20[m_nosys115]
            + (e50[m_nosys115] + e100[m_nosys115]) * far50[m_nosys115]
        ).sum() / etot[m_nosys115].sum()

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
              'delta_net_csi_global': float(alpha * d_csi.sum()),
              'delta_net_csi_nosys115': float(alpha * d_csi[m_nosys115].sum()),
              'warned_csi_p50direct_global': float(alpha * w_csi_p50.sum()),
              'warned_csi_p50direct_nosys115': float(alpha * w_csi_p50[m_nosys115].sum()),
              'warned_bayesian_pr_global': float(alpha * w_bpr.sum()),
              'warned_bayesian_pr_nosys115': float(alpha * w_bpr[m_nosys115].sum()),
              'delta_net_bayesian_pr_global': float(alpha * d_bpr.sum()),
              'delta_net_bayesian_pr_nosys115': float(alpha * d_bpr[m_nosys115].sum()),
              'warned_coping_recall_global': float(alpha * w_cop_rec.sum()),
              'warned_coping_recall_nosys115': float(alpha * w_cop_rec[m_nosys115].sum()),
              'delta_net_coping_recall_nosys115': float(alpha * (w_cop_rec[m_nosys115] - w_cop_rec_b1a[m_nosys115]).sum()),
              'warned_coping_csi_global': float(alpha * w_cop_csi.sum()),
              'warned_coping_csi_nosys115': float(alpha * w_cop_csi[m_nosys115].sum()),
              'delta_net_coping_csi_nosys115': float(alpha * (w_cop_csi[m_nosys115] - w_cop_csi_b1a[m_nosys115]).sum()),
              'false_alarms_global': float(fp_vec.sum()),
              'false_alarms_nosys115': float(fp_vec[m_nosys115].sum()),
              'warned_gt0_global': float(alpha * w_gt0.sum()),
              'warned_r50direct_global': float(alpha * w_r50.sum()),
              'warned_decay_mild_global': float(alpha * w_dm.sum()),
              'warned_decay_steep_global': float(alpha * w_ds.sum()),
              'exp_weighted_far_global_pct': float(ew_far * 100.0),
              'exp_weighted_far_nosys115_pct': float(ew_far_nosys * 100.0),
              'exp_weighted_far_p50direct_global_pct': float(ew_far_p50 * 100.0),
              'exp_weighted_far_p50direct_nosys115_pct': float(ew_far_p50_nosys * 100.0),
              'med_far5_global_pct': float(np.median(far5) * 100.0),
              'med_far5_nosys115_pct': float(np.median(far5[m_nosys115]) * 100.0),
          })

      if (tk, mech_name) in {
          ('tier1_open_compilations', '111_all_three_combined_snap'),
          ('tier2_public_discharge', '111_all_three_combined_snap'),
          ('tier4_all_discoverable', '000_global_train_joint'),
          ('tier4_all_discoverable', '010_ft_only_snap'),
          ('tier4_all_discoverable', '001_da_only_snap'),
          ('tier4_all_discoverable', '101_train_plus_da_snap'),
          ('tier4_all_discoverable', '111_all_three_combined_snap'),
          ('tier4_all_discoverable', '111_all_three_combined_union'),
          ('tier4_all_discoverable', '111_all_three_combined_snap_strict2022'),
          ('tier5_universal_no_ood_clamp', '000_global_train_joint'),
          ('tier5_universal_no_ood_clamp', '111_all_three_combined_snap'),
          ('tier5_universal_idealized_ceiling', '000_global_train_joint'),
          ('tier5_universal_idealized_ceiling', '000_global_train_f2_cov_only_base1a'),
          ('tier5_universal_idealized_ceiling', '000_global_train_f3_ood_only_base1a'),
          ('tier5_universal_idealized_ceiling', '010_ft_only_snap'),
          ('tier5_universal_idealized_ceiling', '001_da_only_snap'),
          ('tier5_universal_idealized_ceiling', '101_train_plus_da_snap'),
          ('tier5_universal_idealized_ceiling', '111_all_three_combined_snap'),
      }:
        country_vec_store[(tk, mech_name)] = (w_vec, d_net, d_clip)
        country_csi_store[(tk, mech_name)] = (w_csi, d_csi)

  df_sum = pd.DataFrame(summary_rows)
  df_sum.to_csv(os.path.join(OUT_DIR, 'multi_mechanism_tiered_exposure_r2.csv'), index=False)
  df_sens = pd.DataFrame(sens_rows)
  df_sens.to_csv(os.path.join(OUT_DIR, 'precision_f1_csi_alpha_sensitivity_r2.csv'), index=False)

  # Build Basin-level & Country-level Table S2/S3 across ALL 175 territories
  print('\n5. Building Basin & 175-Territory Tables + A5 Spatial N_eff Correlations...', flush=True)
  df_b = df_meta[['country_name', 'has_system', 'cap_tier', 'lat', 'lon', 'sub_area', 'total_pop', 'E_total', 'E_gt0_total']].copy()
  df_b['zero_up_t1a'] = (tier_feats['tier1a_active_post2016_ccby_8811']['total_upstream_coverage'].values <= 0.0).astype(float) * 100.0
  df_b['zero_up_t4'] = (tier_feats['tier4_all_discoverable']['total_upstream_coverage'].values <= 0.0).astype(float) * 100.0
  df_b['u4_exp'] = tier_feats['tier4_all_discoverable']['U_net_tree_phys'].values * etot
  df_b['warned_t1a'] = w_b1a
  df_b['unwarned_t1a'] = etot - w_b1a
  df_b['warned_t4_base'] = w_b4
  df_b['unwarned_t4_base'] = etot - w_b4
  df_b['warned_csi_t1a'] = w_csi_b1a
  df_b['unwarned_csi_t1a'] = etot - w_csi_b1a

  for (tk, mech_name), (w_v, dn_v, dc_v) in country_vec_store.items():
    tag = f'{tk}__{mech_name}'
    df_b[f'dnet__{tag}'] = dn_v
    df_b[f'dclip__{tag}'] = dc_v
  for (tk, mech_name), (wc_v, dnc_v) in country_csi_store.items():
    tag = f'{tk}__{mech_name}'
    df_b[f'dnet_csi__{tag}'] = dnc_v

  df_b.to_parquet(os.path.join(OUT_DIR, 'basin_level_exposure_counterfactuals_r2.parquet'))

  df_mapped = df_b[df_b['country_name'].notna()].copy()
  sum_cols = [
      'sub_area', 'total_pop', 'E_total', 'E_gt0_total', 'u4_exp',
      'warned_t1a', 'unwarned_t1a', 'warned_t4_base', 'unwarned_t4_base',
      'warned_csi_t1a', 'unwarned_csi_t1a',
  ] + [c for c in df_b.columns if c.startswith('dnet__') or c.startswith('dclip__') or c.startswith('dnet_csi__')]
  grp = df_mapped.groupby('country_name', as_index=False)
  df_c = grp[sum_cols].sum()
  df_c['n_basins'] = grp.size()['size'].values
  df_c['has_system'] = grp['has_system'].first()['has_system'].values.astype(bool)
  df_c['cap_tier'] = grp['cap_tier'].first()['cap_tier'].values
  df_c['zero_up_pct_t1a'] = grp['zero_up_t1a'].mean()['zero_up_t1a'].values
  df_c['zero_up_pct_t4'] = grp['zero_up_t4'].mean()['zero_up_t4'].values
  df_c['flood_exp_ung_pct_t4'] = np.where(df_c['E_total'] > 0, 100.0 * df_c['u4_exp'] / df_c['E_total'], np.nan)

  col_t4_000 = 'dnet__tier4_all_discoverable__000_global_train_joint'
  col_t4_111 = 'dnet__tier4_all_discoverable__111_all_three_combined_snap'
  col_t5n_111 = 'dnet__tier5_universal_no_ood_clamp__111_all_three_combined_snap'
  col_t5_000 = 'dnet__tier5_universal_idealized_ceiling__000_global_train_joint'
  col_t5_111 = 'dnet__tier5_universal_idealized_ceiling__111_all_three_combined_snap'
  col_t5n_csi = 'dnet_csi__tier5_universal_no_ood_clamp__111_all_three_combined_snap'
  col_t5_csi = 'dnet_csi__tier5_universal_idealized_ceiling__111_all_three_combined_snap'

  df_c = df_c.sort_values('E_total', ascending=False).reset_index(drop=True)
  df_c.to_csv(os.path.join(OUT_DIR, 'Table_S3_175territories_r2.csv'), index=False)

  # A5: Spatially adjusted Clifford-Richardson-Hemon / Dutilleul N_eff & p_spatial
  vuln_map, cop_map, c2lat, c2lon = _load_inform_and_centroids()
  sc = df_c[(df_c['E_total'] > 0) & (df_c['country_name'] != 'Greenland')].copy()
  sc['vuln'] = sc['country_name'].map(vuln_map)
  sc['cop'] = sc['country_name'].map(cop_map)
  sc['clat'] = sc['country_name'].map(c2lat)
  sc['clon'] = sc['country_name'].map(c2lon)
  sc['percap_idealized'] = np.where(sc['total_pop'] > 0, sc[col_t5_111] / sc['total_pop'] * 1e6, np.nan)
  sc['frac_idealized'] = np.where(sc['E_total'] > 0, sc[col_t5_111] / sc['E_total'] * 100.0, np.nan)
  sc['percap_no_ood'] = np.where(sc['total_pop'] > 0, sc[col_t5n_111] / sc['total_pop'] * 1e6, np.nan)
  sc['frac_no_ood'] = np.where(sc['E_total'] > 0, sc[col_t5n_111] / sc['E_total'] * 100.0, np.nan)
  sc['percap_csi_idealized'] = np.where(sc['total_pop'] > 0, sc[col_t5_csi] / sc['total_pop'] * 1e6, np.nan)
  sc['percap_csi_no_ood'] = np.where(sc['total_pop'] > 0, sc[col_t5n_csi] / sc['total_pop'] * 1e6, np.nan)

  sp_out = {}
  for label, xcol, ycol in [
      ('inform_coping_vs_flood_exp_ung', 'cop', 'flood_exp_ung_pct_t4'),
      ('inform_vuln_vs_flood_exp_ung', 'vuln', 'flood_exp_ung_pct_t4'),
      ('inform_vuln_vs_unforecasted_total_idealized', 'vuln', col_t5_111),
      ('inform_vuln_vs_unforecasted_per_capita_idealized', 'vuln', 'percap_idealized'),
      ('inform_vuln_vs_unforecasted_frac_idealized', 'vuln', 'frac_idealized'),
      ('inform_vuln_vs_unforecasted_total_no_ood', 'vuln', col_t5n_111),
      ('inform_vuln_vs_unforecasted_per_capita_no_ood', 'vuln', 'percap_no_ood'),
      ('inform_vuln_vs_unforecasted_frac_no_ood', 'vuln', 'frac_no_ood'),
      ('inform_vuln_vs_unforecasted_csi_total_idealized', 'vuln', col_t5_csi),
      ('inform_vuln_vs_unforecasted_csi_per_capita_idealized', 'vuln', 'percap_csi_idealized'),
      ('inform_vuln_vs_unforecasted_csi_per_capita_no_ood', 'vuln', 'percap_csi_no_ood'),
  ]:
    sp_out[label] = spatial_stats.spatial_spearman_neff(
        sc[xcol].values, sc[ycol].values, sc['clat'].values, sc['clon'].values, k=5
    )
    r_s = sp_out[label]
    print(
        f"   [A5 Spatial] {label:50s}: N={r_s['n']:3d}, rs={r_s['spearman_r']:+.3f}, "
        f"p_naive={r_s['p_naive']:.2e}, Neff={r_s['n_eff']:.1f}, p_spatial={r_s['p_spatial']:.2e}",
        flush=True,
    )
  sp_path = os.path.join(OUT_DIR, 'spatial_neff_stats_r3.json')
  with open(sp_path, 'w') as f:
    json.dump(sp_out, f, indent=2)

  print('\n=== HEADLINE PREVIEW (Round 3 Regularized Cross-Domain, Recall vs. CSI, No-OOD vs. Ceiling) ===')
  for tk in [
      'tier1a_active_post2016_ccby_8811',
      'tier1_open_compilations',
      'tier2_public_discharge',
      'tier4_all_discoverable',
      'tier5_universal_no_ood_clamp',
      'tier5_universal_idealized_ceiling',
  ]:
    for mech in (
        '000_global_train_joint',
        '010_ft_only_snap',
        '001_da_only_snap',
        '111_all_three_combined_snap',
        '111_all_three_combined_snap_strict2022',
    ):
      if 'strict2022' in mech and tk != 'tier4_all_discoverable':
        continue
      sub = df_sum[(df_sum['tier_key'] == tk) & (df_sum['mechanism'] == mech)]
      if len(sub) > 0:
        r = sub.iloc[0]
        print(
            f"  {tk[:26]:26s} | {mech:36s} | "
            f"Rec Glb={r['delta_net_global']/1e6:+.2f}M CAP={r['delta_net_nosys115']/1e6:+.2f}M | "
            f"CSI Glb={r['delta_net_csi_global']/1e6:+.2f}M CAP={r['delta_net_csi_nosys115']/1e6:+.2f}M",
            flush=True,
        )
  print(f'\nPart 1 (Counterfactuals + Tables + A5) complete in {(time.time() - t0)/60:.1f} min!', flush=True)


# ==============================================================================
# PART 2: A3 B=200 STRATIFIED MAIN_BAS x CLIMATE BLOCK BOOTSTRAP +
#         OOD-SCALED CROSS-DOMAIN (LOCO / CLIMATE) RESIDUAL PROPAGATION
# ==============================================================================

def _build_bootstrap_context(all_ids: pd.Index, df_train: pd.DataFrame, df_inf: pd.DataFrame, meta: pd.DataFrame) -> dict:
  """Precomputes block structure, global macro-region IDs, OOD excess weights, and empirical
  zero-centered cross-domain (LOCO + Climate) group-mean residual matrix."""
  n_g = len(meta)
  # 1. Assign every global basin to its nearest hydroclimate cluster (0..7) & continent code
  scaler = StandardScaler().fit(df_train.iloc[:n_g][CLIM_COLS].fillna(0).values)
  km = KMeans(n_clusters=8, random_state=42, n_init=20).fit(scaler.transform(df_train.iloc[:n_g][CLIM_COLS].fillna(0).values))
  global_clim = km.predict(scaler.transform(df_inf[CLIM_COLS].fillna(0).values)).astype(np.int32)

  # HydroBASINS Pfafstetter first-digit continent mapping
  digit_to_cont = {'1': 'af', '2': 'eu', '3': 'as', '4': 'as', '5': 'au', '6': 'sa', '7': 'na', '8': 'na', '9': 'eu'}
  first_dig = pd.Series(all_ids).astype(str).str.replace('hybas_', '', regex=False).str[0].values
  global_cont = np.array([digit_to_cont.get(d, 'as') for d in first_dig])
  global_region_key = np.array([f'{c}_{cl}' for c, cl in zip(global_cont, global_clim)])
  uniq_regions, global_region_idx = np.unique(global_region_key, return_inverse=True)

  # 2. Compute normalized OOD excess weight omega_i in [0, 1] per basin relative to Caravan gauges
  #    omega = 0 at in-distribution median (<=50th pct), ramping linearly to 1 at >=90th pct OOD
  t1a = pd.read_parquet(os.path.join(TIER_DIR, 'tier1a_active_post2016_ccby_8811.parquet')).reindex(all_ids)
  t1a_mask = pd.read_parquet(os.path.join(TIER_DIR, 'tier_active_and_all_masks_r2.parquet')).reindex(all_ids)[
      'tier1a_active_post2016_ccby_8811__snap_all'
  ].values.astype(bool)
  pcts_ood = []
  for oc in OOD_COLS:
    ref_vals = np.sort(t1a.loc[t1a_mask, oc].dropna().values)
    # Lower anomaly score = MORE OOD -> percentile of dissimilarity = 1 - CDF_ref(x)
    cdf = np.searchsorted(ref_vals, t1a[oc].values, side='right') / max(len(ref_vals), 1)
    pcts_ood.append(1.0 - cdf)
  mean_ood_pct = np.mean(pcts_ood, axis=0)
  omega_t1a = np.clip((mean_ood_pct - 0.50) / 0.40, 0.0, 1.0).astype(np.float32)

  # Also OOD excess for tier5_universal_idealized_ceiling (shift magnitude = max(0, omega_t1a - omega_t5))
  t5 = pd.read_parquet(os.path.join(TIER_DIR, 'tier5_universal_idealized_ceiling.parquet')).reindex(all_ids)
  pcts_ood_t5 = []
  for oc in OOD_COLS:
    ref_vals = np.sort(t1a.loc[t1a_mask, oc].dropna().values)
    cdf = np.searchsorted(ref_vals, t5[oc].values, side='right') / max(len(ref_vals), 1)
    pcts_ood_t5.append(1.0 - cdf)
  omega_t5_shift = np.clip(mean_ood_pct - np.mean(pcts_ood_t5, axis=0), 0.0, 1.0).astype(np.float32)

  # 3. Empirical zero-centered cross-domain group-mean residual pool across all 24 (fam, metric) pairs
  #    Rows = held-out LOCO continent groups (5) + held-out Climate cluster groups (8) = 13 macro-domain shifts!
  fam_metric_keys = [(fam, sk, mf) for fam in ('base', 'delta_ft', 'delta_da', 'delta_ft_da') for sk, mf in BOOT_RP_METRICS]
  group_res_rows = []
  for cv_dir, grp_col in [(CV_LOCO_DIR, 'continent'), (CV_CLIMATE_DIR, 'climate_cluster')]:
    # Collect available group names from base__r5
    p_ref = os.path.join(cv_dir, 'base__5.0-year-2.0-window-recall.csv')
    if not os.path.exists(p_ref):
      continue
    df_ref = pd.read_csv(p_ref)
    if grp_col not in df_ref.columns:
      continue
    grps = sorted(df_ref[grp_col].dropna().unique().tolist())
    # Pre-load all 24 CV files for this cv_dir
    cv_tables = {}
    for fam, sk, mf in fam_metric_keys:
      p_f = os.path.join(cv_dir, f'{fam}__{mf}.csv')
      if os.path.exists(p_f):
        dfc = pd.read_csv(p_f)
        dfc['err'] = dfc['true'] - dfc['pred']
        cv_tables[(fam, sk)] = dfc.groupby(grp_col)['err'].mean().to_dict()
      else:
        cv_tables[(fam, sk)] = {}
    mat_sub = np.zeros((len(grps), len(fam_metric_keys)), dtype=np.float64)
    for gi, gval in enumerate(grps):
      for ci, (fam, sk, _) in enumerate(fam_metric_keys):
        mat_sub[gi, ci] = float(cv_tables[(fam, sk)].get(gval, 0.0))
    # Zero-center within each CV scheme so expected bias is identically 0
    mat_sub -= mat_sub.mean(axis=0, keepdims=True)
    group_res_rows.append(mat_sub)

  if group_res_rows:
    domain_err_pool = np.vstack(group_res_rows)
  else:
    domain_err_pool = np.zeros((1, len(fam_metric_keys)), dtype=np.float64)

  # 4. Build intact (snapped_main_bas, climate_cluster) blocks stratified by (continent, climate_cluster)
  blk_key = np.array([f'{mb}_{cl}' for mb, cl in zip(meta['snapped_main_bas'].values, meta['climate_cluster'].values)])
  uniq_blks, station_blk_idx = np.unique(blk_key, return_inverse=True)
  n_blks = len(uniq_blks)
  blk_strat_raw = {}
  for si in range(n_g):
    bi = station_blk_idx[si]
    if bi not in blk_strat_raw:
      blk_strat_raw[bi] = f"{meta['continent'].iloc[si]}_{meta['climate_cluster'].iloc[si]}"
  strat_counts = defaultdict(int)
  for bi in range(n_blks):
    strat_counts[blk_strat_raw[bi]] += 1
  # Merge strata with < 3 blocks into their climate_cluster so every stratum has clean Dirichlet variance
  strata_to_blks = defaultdict(list)
  for bi in range(n_blks):
    sk_str = blk_strat_raw[bi]
    if strat_counts[sk_str] < 3:
      sk_str = 'clim_' + sk_str.split('_')[-1]
    strata_to_blks[sk_str].append(bi)

  blk_sizes = np.bincount(station_blk_idx, minlength=n_blks).astype(np.float64)

  return {
      'n_regions': int(len(uniq_regions)),
      'global_region_idx': global_region_idx.astype(np.int32),
      'omega_t1a': omega_t1a,
      'omega_t5_shift': omega_t5_shift,
      'fam_metric_keys': [(f, sk) for f, sk, _ in fam_metric_keys],
      'domain_err_pool': domain_err_pool,
      'station_blk_idx': station_blk_idx.astype(np.int32),
      'blk_sizes': blk_sizes,
      'strata_to_blks': {k: np.array(v, dtype=np.int32) for k, v in strata_to_blks.items()},
  }


_BOOT_SHARED: dict = {}


def _select_boot_core_mask(
    df_meta_all: pd.DataFrame,
    min_exp: float = 5.0,
    country_cov: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
  """Partitions positive-exposure basins into (core_mask, tail_mask).

  When min_exp <= 0, core_mask covers 100% of positive-exposure basins (590,652)
  and tail_mask is empty. When min_exp > 0 (e.g. 5.0 people/yr with 95% per-territory
  safeguard), core_mask covers >99.2% of global exposure, >98.9% of CAP-gap exposure,
  and >=95% of every mapped territory, while tail_mask captures the remaining <0.8%
  micro-exposure basins for exact full-sample Stage-D tail closure.
  """
  etot = df_meta_all['E_total'].values
  pos_mask = etot > 0.0
  if min_exp <= 0.0:
    return pos_mask, np.zeros(len(df_meta_all), dtype=bool)

  core = etot >= float(min_exp)
  is_mapped = df_meta_all['country_name'].notna().values
  extra = np.zeros(len(df_meta_all), dtype=bool)
  if country_cov > 0.0:
    sub = df_meta_all[is_mapped & pos_mask]
    for _, grp in sub.groupby('country_name', sort=False):
      ctot = float(grp['E_total'].sum())
      if ctot <= 0.0:
        continue
      pos_idx = df_meta_all.index.get_indexer(grp.index)
      ccore = float(etot[pos_idx][core[pos_idx]].sum())
      if ccore / ctot < country_cov:
        order = np.argsort(-etot[pos_idx])
        csum = np.cumsum(etot[pos_idx[order]])
        k = int(np.searchsorted(csum, country_cov * ctot)) + 1
        extra[pos_idx[order[:k]]] = True
  core_mask = (core | extra) & pos_mask
  tail_mask = pos_mask & (~core_mask)
  return core_mask, tail_mask


def _compute_boot_tail_closure(
    tail_ids: pd.Index,
    df_meta_tail: pd.DataFrame,
    df_masks_tail: pd.DataFrame,
) -> tuple[dict | None, pd.DataFrame | None]:
  """Computes exact full-sample Stage-D scenario totals & country sums on tail_ids."""
  if len(tail_ids) == 0:
    return None, None
  metric_keys = [sk for sk, _ in BOOT_RP_METRICS]

  def _load_p(fam: str, tk: str) -> dict[str, np.ndarray]:
    pq = os.path.join(PREDS_CACHE_DIR, f'{fam}__{tk}__joint.parquet')
    dfc = pd.read_parquet(pq, columns=metric_keys).reindex(tail_ids)
    return {k: dfc[k].values.astype(np.float32) for k in metric_keys}

  def _recon(pb: dict[str, np.ndarray], pd_m: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.clip(pb[k] + pd_m[k], 0.0, 1.0) for k in metric_keys}

  e5 = df_meta_tail['E_5'].values
  e10 = df_meta_tail['E_10'].values
  e20p = df_meta_tail['E_20_50_100'].values
  etot = df_meta_tail['E_total'].values
  is_mapped = df_meta_tail['country_name'].notna().values
  m_sys60 = df_meta_tail['has_system'].values & is_mapped
  m_nosys115 = (~df_meta_tail['has_system'].values) & is_mapped

  def warned_rec(p: dict) -> np.ndarray:
    return e5 * p['r5'] + e10 * p['r10'] + e20p * p['r20']

  def warned_csi(p: dict) -> np.ndarray:
    return (
        e5 * _csi_vec(p['r5'], p['p5'])
        + e10 * _csi_vec(p['r10'], p['p10'])
        + e20p * _csi_vec(p['r20'], p['p20'])
    )

  p_b1a = _load_p('base', 'tier1a_active_post2016_ccby_8811')
  p_ft1a = _recon(p_b1a, _load_p('delta_ft', 'tier1a_active_post2016_ccby_8811'))
  p_da1a = _recon(p_b1a, _load_p('delta_da', 'tier1a_active_post2016_ccby_8811'))
  w_b1a = warned_rec(p_b1a)
  wc_b1a = warned_csi(p_b1a)

  tail_suite = {
      'baseline_warned_global': float(w_b1a.sum()),
      'baseline_warned_mapped': float(w_b1a[is_mapped].sum()),
      'baseline_unwarned_global': float((etot - w_b1a).sum()),
      'baseline_warned_sys60': float(w_b1a[m_sys60].sum()),
      'baseline_warned_nosys115': float(w_b1a[m_nosys115].sum()),
      'baseline_unwarned_nosys115': float((etot - w_b1a)[m_nosys115].sum()),
      'baseline_warned_csi_global': float(wc_b1a.sum()),
      'baseline_warned_csi_sys60': float(wc_b1a[m_sys60].sum()),
      'baseline_warned_csi_nosys115': float(wc_b1a[m_nosys115].sum()),
      'scenarios': {},
  }
  ctry_vecs = {'warned_t1a': w_b1a, 'unwarned_t1a': etot - w_b1a, 'warned_csi_t1a': wc_b1a}

  for tk in BOOT_TIERS:
    p_bk = p_b1a if tk == 'tier1a_active_post2016_ccby_8811' else _load_p('base', tk)
    p_ftdak = _recon(p_bk, _load_p('delta_ft_da', tk))
    m_all = df_masks_tail[f'{tk}__snap_all'].values.astype(bool)
    m_act = df_masks_tail[f'{tk}__snap_active'].values.astype(bool)
    is_univ = tk in ('tier5_universal_no_ood_clamp', 'tier5_universal_idealized_ceiling')
    if is_univ:
      p_comb = p_ftdak
    else:
      p_ftk = p_ft1a if tk == 'tier1a_active_post2016_ccby_8811' else _recon(p_bk, _load_p('delta_ft', tk))
      p_comb = {k: np.where(m_act, p_ftdak[k], np.where(m_all, p_ftk[k], p_bk[k])) for k in metric_keys}

    scen_map = {
        '000_train': p_bk,
        '010_ft_only': {k: np.where(m_all, p_ft1a[k], p_b1a[k]) for k in metric_keys},
        '001_da_only': {k: np.where(m_act, p_da1a[k], p_b1a[k]) for k in metric_keys},
        '111_combined': p_comb,
    }
    if tk == 'tier4_all_discoverable' and f'{tk}__snap_active_strict2022' in df_masks_tail.columns:
      m_act22 = df_masks_tail[f'{tk}__snap_active_strict2022'].values.astype(bool)
      scen_map['001_da_only_strict2022'] = {k: np.where(m_act22, p_da1a[k], p_b1a[k]) for k in metric_keys}
      scen_map['111_combined_strict2022'] = {k: np.where(m_act22, p_ftdak[k], np.where(m_all, p_ftk[k], p_bk[k])) for k in metric_keys}

    for sk_name, p_s in scen_map.items():
      w_s = warned_rec(p_s)
      d_s = w_s - w_b1a
      wc_s = warned_csi(p_s)
      dc_s = wc_s - wc_b1a
      tail_suite['scenarios'][f'{tk}__{sk_name}'] = {
          'delta_global': float(d_s.sum()),
          'delta_sys60': float(d_s[m_sys60].sum()),
          'delta_nosys115': float(d_s[m_nosys115].sum()),
          'warned_global': float(w_s.sum()),
          'warned_mapped': float(w_s[is_mapped].sum()),
          'warned_nosys115': float(w_s[m_nosys115].sum()),
          'delta_csi_global': float(dc_s.sum()),
          'delta_csi_sys60': float(dc_s[m_sys60].sum()),
          'delta_csi_nosys115': float(dc_s[m_nosys115].sum()),
          'warned_csi_global': float(wc_s.sum()),
          'warned_csi_nosys115': float(wc_s[m_nosys115].sum()),
      }
      if tk in (
          'tier4_all_discoverable',
          'tier5_universal_no_ood_clamp',
          'tier5_universal_idealized_ceiling',
      ) and sk_name in ('000_train', '111_combined'):
        ctry_vecs[f'{tk}__{sk_name}'] = d_s
        ctry_vecs[f'csi__{tk}__{sk_name}'] = dc_s

  df_c_tail = pd.DataFrame(ctry_vecs, index=tail_ids)
  df_c_tail['country_name'] = df_meta_tail['country_name'].values
  df_c_tail = df_c_tail[df_c_tail['country_name'].notna()].groupby('country_name').sum()
  return tail_suite, df_c_tail


def _add_tail_suite(suite: dict, tail_suite: dict | None) -> dict:
  if tail_suite is None:
    return suite
  for k, v in tail_suite.items():
    if k == 'scenarios':
      for sk, smap in v.items():
        dst = suite['scenarios'][sk]
        for mk, mv in smap.items():
          dst[mk] += mv
    else:
      suite[k] += v
  return suite


def _init_boot_spawn_worker(ctx_pkl_path: str, n_threads: int) -> None:
  os.environ['OMP_NUM_THREADS'] = str(n_threads)
  os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
  os.environ['MKL_NUM_THREADS'] = str(n_threads)
  os.environ['NUMEXPR_NUM_THREADS'] = str(n_threads)
  try:
    import threadpoolctl  # pylint: disable=g-import-not-at-top
    threadpoolctl.threadpool_limits(limits=n_threads)
  except Exception:
    pass
  _BOOT_SHARED.clear()
  _BOOT_SHARED.update(joblib.load(ctx_pkl_path, mmap_mode='r'))


def _run_one_bootstrap_rep(args: tuple) -> dict:
  """Fits one stratified MAIN_BAS x Climate Bayesian-bootstrap replicate across
  (r5, r10, r20, p5, p10, p20) x 4 families (24 fixed-stacker refits) +
  OOD-scaled cross-domain residual propagation using memory-mapped shared context."""
  b_idx, n_threads = args
  os.environ['OMP_NUM_THREADS'] = str(n_threads)
  os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
  os.environ['MKL_NUM_THREADS'] = str(n_threads)
  os.environ['NUMEXPR_NUM_THREADS'] = str(n_threads)

  rep_json = os.path.join(BOOT_DIR, f'rep_{b_idx:03d}.json')
  rep_ctry = os.path.join(BOOT_DIR, f'rep_{b_idx:03d}_countries.parquet')
  if os.path.exists(rep_json) and os.path.exists(rep_ctry):
    with open(rep_json) as f:
      return json.load(f)

  t0 = time.time()
  S = _BOOT_SHARED
  ctx = S['ctx']
  df_train = S['df_train']
  y_targets = S['y_targets']
  final_estimators = S['final_estimators']
  X_rp_base = S['X_rp_base']
  tier_cov = S['tier_cov']
  tier_ood = S['tier_ood']
  df_masks = S['df_masks']
  core_ids = S['core_ids']
  e5 = S['e5']
  e10 = S['e10']
  e20p = S['e20p']
  etot = S['etot']
  is_mapped = S['is_mapped']
  m_sys60 = S['m_sys60']
  m_nosys115 = S['m_nosys115']
  country_name = S['country_name']
  reg_idx_pos = S['reg_idx_core']
  omega_t1a_pos = S['omega_t1a_core']
  omega_t5_shift_pos = S['omega_t5_shift_core']
  tail_suite = S['tail_suite']
  tail_ctry_df = S['tail_ctry_df']

  # 1. Draw continuous stratified Bayesian-bootstrap Dirichlet weights over intact MAIN_BAS x Climate blocks
  rng = np.random.default_rng(20000 + b_idx)
  n_blks = len(ctx['blk_sizes'])
  blk_w = np.ones(n_blks, dtype=np.float64)
  for _, blk_arr in ctx['strata_to_blks'].items():
    g_draw = rng.exponential(1.0, size=len(blk_arr))
    n_strat_stations = float(ctx['blk_sizes'][blk_arr].sum())
    denom = float(np.dot(g_draw, ctx['blk_sizes'][blk_arr]))
    if denom > 0:
      blk_w[blk_arr] = g_draw * (n_strat_stations / denom)
  station_w = blk_w[ctx['station_blk_idx']]
  row_w = np.tile(station_w, 3)

  # 2. Fit all 24 (fam, metric) models via fast fit_with_fixed_stacker reusing Stage C's RidgeCV stacker
  models = defaultdict(dict)
  for sk, m_full in BOOT_RP_METRICS:
    for fam in ('base', 'delta_ft', 'delta_da', 'delta_ft_da'):
      ok_m, y_ok = y_targets[(fam, sk)]
      m_obj = SkillPredictorR2(
          m_full,
          is_delta=(fam != 'base'),
          n_threads=n_threads,
          preset=DEFAULT_REG_PRESET,
      )
      m_obj.fit_with_fixed_stacker(
          df_train.iloc[ok_m],
          y_ok,
          sample_weight=row_w[ok_m],
          final_estimator=final_estimators[(fam, sk)],
      )
      models[fam][sk] = m_obj

  # 3. Predict ONLY non-dead (family, tier) combinations on core exposure basins
  n_pos = len(core_ids)
  metric_keys = [sk for sk, _ in BOOT_RP_METRICS]
  preds = {}
  for tk in BOOT_TIERS:
    X_rp = X_rp_base.copy()
    for c, vals in tier_cov[tk].items():
      X_rp[c] = vals
    for c, vals in tier_ood[tk].items():
      X_rp[c] = vals
    preds[('base', tk)] = _fast_predict_rp_family(models['base'], X_rp, n_threads=n_threads)

    is_t1a = tk == 'tier1a_active_post2016_ccby_8811'
    is_univ = tk in ('tier5_universal_no_ood_clamp', 'tier5_universal_idealized_ceiling')
    m_all = df_masks[f'{tk}__snap_all'].values.astype(bool)
    m_act = df_masks[f'{tk}__snap_active'].values.astype(bool)
    has_act22 = tk == 'tier4_all_discoverable' and f'{tk}__snap_active_strict2022' in df_masks.columns
    m_act22 = df_masks[f'{tk}__snap_active_strict2022'].values.astype(bool) if has_act22 else m_act

    # delta_ft: full on t1a (used for 010_ft_only across all tiers), skipped on universal (m_act=all True),
    # strictly masked to (m_all & ~m_act) [or (m_all & (~m_act | ~m_act22)) on tier4] elsewhere!
    if is_t1a:
      preds[('delta_ft', tk)] = _fast_predict_rp_family(models['delta_ft'], X_rp, n_threads=n_threads)
    elif not is_univ:
      ft_mask = m_all & ((~m_act) | (~m_act22))
      idx_sub = np.flatnonzero(ft_mask)
      if len(idx_sub) == 0:
        preds[('delta_ft', tk)] = {k: np.zeros(n_pos, dtype=np.float32) for k in metric_keys}
      else:
        sub_p = _fast_predict_rp_family(models['delta_ft'], X_rp.iloc[idx_sub], n_threads=n_threads)
        full_p = {}
        for k, arr_s in sub_p.items():
          fa = np.zeros(n_pos, dtype=np.float32)
          fa[idx_sub] = arr_s
          full_p[k] = fa
        preds[('delta_ft', tk)] = full_p

    # delta_da: needed ONLY on t1a (used for 001_da_only & 001_da_only_strict2022 across all tiers)
    if is_t1a:
      preds[('delta_da', tk)] = _fast_predict_rp_family(models['delta_da'], X_rp, n_threads=n_threads)

    # delta_ft_da: full on universal, strictly masked to active snaps elsewhere
    if is_univ:
      preds[('delta_ft_da', tk)] = _fast_predict_rp_family(models['delta_ft_da'], X_rp, n_threads=n_threads)
    else:
      idx_sub = np.flatnonzero(m_act | m_act22)
      if len(idx_sub) == 0:
        preds[('delta_ft_da', tk)] = {k: np.zeros(n_pos, dtype=np.float32) for k in metric_keys}
      else:
        sub_p = _fast_predict_rp_family(models['delta_ft_da'], X_rp.iloc[idx_sub], n_threads=n_threads)
        full_p = {}
        for k, arr_s in sub_p.items():
          fa = np.zeros(n_pos, dtype=np.float32)
          fa[idx_sub] = arr_s
          full_p[k] = fa
        preds[('delta_ft_da', tk)] = full_p

  # 4. Draw coherent per-(continent, climate_cluster) OOD-scaled cross-domain residual vectors
  #    Rademacher (+/-1) wild draw from zero-centered LOCO + Climate empirical group-mean pool -> E[err] = 0 exactly!
  pool = ctx['domain_err_pool']
  n_pool = len(pool)
  n_reg = ctx['n_regions']
  fmk_to_col = {k: ci for ci, k in enumerate(ctx['fam_metric_keys'])}

  draw_idx = rng.integers(0, n_pool, size=n_reg)
  draw_sgn = rng.choice([-1.0, 1.0], size=(n_reg, 1))
  reg_err_mat = pool[draw_idx] * draw_sgn  # (n_reg, 24)

  # Separate independent wild draw for the OOD->Q75 extrapolation shift itself on idealized ceiling
  draw_idx_ood = rng.integers(0, n_pool, size=n_reg)
  draw_sgn_ood = rng.choice([-1.0, 1.0], size=(n_reg, 1))
  reg_err_ood_shift = pool[draw_idx_ood] * draw_sgn_ood

  def _domain_err(fam: str, sk: str, scale_arr: np.ndarray, use_ood_shift: bool = False) -> np.ndarray:
    ci = fmk_to_col[(fam, sk)]
    mat = reg_err_ood_shift if use_ood_shift else reg_err_mat
    return (mat[reg_idx_pos, ci] * scale_arr).astype(np.float32)

  def warned_rec(p: dict) -> np.ndarray:
    return e5 * p['r5'] + e10 * p['r10'] + e20p * p['r20']

  def warned_csi(p: dict) -> np.ndarray:
    return e5 * _csi_vec(p['r5'], p['p5']) + e10 * _csi_vec(p['r10'], p['p10']) + e20p * _csi_vec(p['r20'], p['p20'])

  def eval_suite(add_domain_err: bool) -> tuple[dict, dict]:
    """Computes full suite of scenarios either with OOD-scaled cross-domain error (total UQ)
    or pure block bootstrap only, closed with exact full-sample Stage-D micro-exposure tail."""
    def _base_p(tk: str) -> dict[str, np.ndarray]:
      pb = preds[('base', tk)]
      if not add_domain_err:
        return pb
      out = {}
      for k in metric_keys:
        v = pb[k] + _domain_err('base', k, omega_t1a_pos, use_ood_shift=False)
        if tk == 'tier5_universal_idealized_ceiling':
          v = v + _domain_err('base', k, omega_t5_shift_pos, use_ood_shift=True)
        out[k] = np.clip(v, 0.0, 1.0)
      return out

    def _recon_p(pb: dict, tk: str, fam: str) -> dict[str, np.ndarray]:
      pd_m = preds[(fam, tk)]
      out = {}
      for k in metric_keys:
        d_val = pd_m[k]
        if add_domain_err:
          d_val = d_val + _domain_err(fam, k, omega_t1a_pos, use_ood_shift=False)
        out[k] = np.clip(pb[k] + d_val, 0.0, 1.0)
      return out

    p_b1a = _base_p('tier1a_active_post2016_ccby_8811')
    p_ft1a = _recon_p(p_b1a, 'tier1a_active_post2016_ccby_8811', 'delta_ft')
    p_da1a = _recon_p(p_b1a, 'tier1a_active_post2016_ccby_8811', 'delta_da')
    w_b1a = warned_rec(p_b1a)
    wc_b1a = warned_csi(p_b1a)

    suite = {
        'baseline_warned_global': float(w_b1a.sum()),
        'baseline_warned_mapped': float(w_b1a[is_mapped].sum()),
        'baseline_unwarned_global': float((etot - w_b1a).sum()),
        'baseline_warned_sys60': float(w_b1a[m_sys60].sum()),
        'baseline_warned_nosys115': float(w_b1a[m_nosys115].sum()),
        'baseline_unwarned_nosys115': float((etot - w_b1a)[m_nosys115].sum()),
        'baseline_warned_csi_global': float(wc_b1a.sum()),
        'baseline_warned_csi_sys60': float(wc_b1a[m_sys60].sum()),
        'baseline_warned_csi_nosys115': float(wc_b1a[m_nosys115].sum()),
        'scenarios': {},
    }
    ctry_vecs = {'warned_t1a': w_b1a, 'unwarned_t1a': etot - w_b1a, 'warned_csi_t1a': wc_b1a}

    for tk in BOOT_TIERS:
      p_bk = p_b1a if tk == 'tier1a_active_post2016_ccby_8811' else _base_p(tk)
      p_ftdak = _recon_p(p_bk, tk, 'delta_ft_da')
      m_all = df_masks[f'{tk}__snap_all'].values.astype(bool)
      m_act = df_masks[f'{tk}__snap_active'].values.astype(bool)
      is_univ = tk in ('tier5_universal_no_ood_clamp', 'tier5_universal_idealized_ceiling')
      if is_univ:
        p_comb = p_ftdak
      else:
        p_ftk = p_ft1a if tk == 'tier1a_active_post2016_ccby_8811' else _recon_p(p_bk, tk, 'delta_ft')
        p_comb = {k: np.where(m_act, p_ftdak[k], np.where(m_all, p_ftk[k], p_bk[k])) for k in metric_keys}

      scen_map = {
          '000_train': p_bk,
          '010_ft_only': {k: np.where(m_all, p_ft1a[k], p_b1a[k]) for k in metric_keys},
          '001_da_only': {k: np.where(m_act, p_da1a[k], p_b1a[k]) for k in metric_keys},
          '111_combined': p_comb,
      }
      if tk == 'tier4_all_discoverable' and f'{tk}__snap_active_strict2022' in df_masks.columns:
        m_act22 = df_masks[f'{tk}__snap_active_strict2022'].values.astype(bool)
        scen_map['001_da_only_strict2022'] = {k: np.where(m_act22, p_da1a[k], p_b1a[k]) for k in metric_keys}
        scen_map['111_combined_strict2022'] = {k: np.where(m_act22, p_ftdak[k], np.where(m_all, p_ftk[k], p_bk[k])) for k in metric_keys}

      for sk_name, p_s in scen_map.items():
        w_s = warned_rec(p_s)
        d_s = w_s - w_b1a
        wc_s = warned_csi(p_s)
        dc_s = wc_s - wc_b1a
        suite['scenarios'][f'{tk}__{sk_name}'] = {
            'delta_global': float(d_s.sum()),
            'delta_sys60': float(d_s[m_sys60].sum()),
            'delta_nosys115': float(d_s[m_nosys115].sum()),
            'warned_global': float(w_s.sum()),
            'warned_mapped': float(w_s[is_mapped].sum()),
            'warned_nosys115': float(w_s[m_nosys115].sum()),
            'delta_csi_global': float(dc_s.sum()),
            'delta_csi_sys60': float(dc_s[m_sys60].sum()),
            'delta_csi_nosys115': float(dc_s[m_nosys115].sum()),
            'warned_csi_global': float(wc_s.sum()),
            'warned_csi_nosys115': float(wc_s[m_nosys115].sum()),
        }
        if add_domain_err and tk in (
            'tier4_all_discoverable',
            'tier5_universal_no_ood_clamp',
            'tier5_universal_idealized_ceiling',
        ) and sk_name in ('000_train', '111_combined'):
          ctry_vecs[f'{tk}__{sk_name}'] = d_s
          ctry_vecs[f'csi__{tk}__{sk_name}'] = dc_s
    return _add_tail_suite(suite, tail_suite), ctry_vecs

  total_uq, ctry_cols = eval_suite(add_domain_err=True)
  pure_uq, _ = eval_suite(add_domain_err=False)

  rep_stats = {
      'rep': b_idx,
      'elapsed_s': float(time.time() - t0),
      **total_uq,
      'pure_block_boot': pure_uq,
  }

  df_c_rep = pd.DataFrame(ctry_cols, index=core_ids)
  df_c_rep['country_name'] = country_name
  df_c_rep = df_c_rep[df_c_rep['country_name'].notna()].groupby('country_name').sum()
  if tail_ctry_df is not None:
    df_c_rep = df_c_rep.add(tail_ctry_df, fill_value=0.0)
  df_c_rep.to_parquet(rep_ctry)

  with open(rep_json, 'w') as f:
    json.dump(rep_stats, f, indent=2)
  return rep_stats


def run_part_bootstrap(
    n_boot: int = 200,
    workers: int = 48,
    threads: int = 1,
    min_exp: float = 5.0,
    country_cov: float = 0.95,
):
  os.makedirs(BOOT_DIR, exist_ok=True)
  t0 = time.time()
  os.environ['OMP_NUM_THREADS'] = str(threads)
  os.environ['OPENBLAS_NUM_THREADS'] = str(threads)
  os.environ['MKL_NUM_THREADS'] = str(threads)
  os.environ['NUMEXPR_NUM_THREADS'] = str(threads)

  print(
      f'\n--- Running A3 B={n_boot} Stratified MAIN_BAS x Climate Block Bootstrap + '
      f'OOD-Scaled Cross-Domain UQ ({workers} workers x {threads} thr, min_exp={min_exp}, ccov={country_cov}) ---',
      flush=True,
  )

  with open(FEATURES_JSON) as f:
    features_95 = json.load(f)
  meta, _ = load_train_station_metadata()
  slice_ids = meta.index
  df_train = pd.read_parquet(TRAIN_PARQUET)[features_95]
  df_inf_all = pd.read_parquet(INFER_PARQUET)[features_95]
  all_ids = df_inf_all.index
  df_meta_all = load_global_metadata_and_exposure(all_ids)

  ctx = _build_bootstrap_context(all_ids, df_train, df_inf_all, meta)
  ctx_path = os.path.join(BOOT_DIR, 'bootstrap_context.joblib')
  joblib.dump(ctx, ctx_path, compress=1)

  core_mask, tail_mask = _select_boot_core_mask(df_meta_all, min_exp=min_exp, country_cov=country_cov)
  core_ids = all_ids[core_mask]
  tail_ids = all_ids[tail_mask]
  etot_all = df_meta_all['E_total'].values
  is_map_all = df_meta_all['country_name'].notna().values
  m_cap_all = (~df_meta_all['has_system'].values) & is_map_all
  print(
      f"   Context ready: {len(ctx['blk_sizes']):,} MAIN_BAS x climate blocks across "
      f"{len(ctx['strata_to_blks'])} strata | {ctx['n_regions']} global macro-regions | "
      f"domain_err_pool={ctx['domain_err_pool'].shape}",
      flush=True,
  )
  print(
      f"   Core basins: {len(core_ids):,} ({100*etot_all[core_mask].sum()/etot_all.sum():.3f}% Glb, "
      f"{100*etot_all[core_mask & m_cap_all].sum()/etot_all[m_cap_all].sum():.3f}% CAP) | "
      f"Exact Stage-D tail basins: {len(tail_ids):,}",
      flush=True,
  )

  df_masks_all = pd.read_parquet(os.path.join(TIER_DIR, 'tier_active_and_all_masks_r2.parquet')).reindex(all_ids)
  tail_suite, tail_ctry_df = _compute_boot_tail_closure(
      tail_ids,
      df_meta_all.iloc[tail_mask],
      df_masks_all.iloc[tail_mask],
  )

  # Preload all 24 targets and Stage-C fixed RidgeCV final estimators ONCE before fork CoW
  y_targets = {}
  final_estimators = {}
  for sk, m_full in BOOT_RP_METRICS:
    sc_base = joblib.load(os.path.join(MODELS_DIR, f'base__{m_full}.joblib'))
    final_estimators[('base', sk)] = sc_base.impl.final_estimator_
    y_base_raw = load_3split_target(m_full, '', slice_ids)
    y_base_clip = np.where(np.isfinite(y_base_raw), np.clip(y_base_raw, 0.0, 1.0), np.nan)
    ok_b = np.isfinite(y_base_clip)
    y_targets[('base', sk)] = (ok_b, pd.Series(y_base_clip[ok_b], index=df_train.index[ok_b]))
    for fam, sfx in [('delta_ft', '_ft'), ('delta_da', '_da'), ('delta_ft_da', '_ft_da')]:
      sc_d = joblib.load(os.path.join(MODELS_DIR, f'{fam}__{m_full}.joblib'))
      final_estimators[(fam, sk)] = sc_d.impl.final_estimator_
      ym_raw = load_3split_target(m_full, sfx, slice_ids)
      ym_clip = np.where(np.isfinite(ym_raw), np.clip(ym_raw, 0.0, 1.0), np.nan)
      d_full = np.where(np.isfinite(y_base_clip) & np.isfinite(ym_clip), ym_clip - y_base_clip, np.nan)
      ok_d = np.isfinite(d_full)
      y_targets[(fam, sk)] = (ok_d, pd.Series(d_full[ok_d], index=df_train.index[ok_d]))

  # Precompute float32 preprocessed inference DataFrame & float32 tier columns ONCE
  tmp_mod = SkillPredictorR2('5.0-year-2.0-window-recall', is_delta=False, n_threads=threads, preset=DEFAULT_REG_PRESET)
  X_rp_base = tmp_mod.impl._preprocess_features(df_inf_all.iloc[core_mask]).astype(np.float32)
  tier_cov, tier_ood = {}, {}
  for tk in BOOT_TIERS:
    tf = pd.read_parquet(os.path.join(TIER_DIR, f'{tk}.parquet'), columns=COV_COLS + OOD_COLS).reindex(core_ids)
    tier_cov[tk] = {c: tf[c].values.astype(np.float32) for c in COV_COLS}
    tier_ood[tk] = {c: tf[c].values.astype(np.float32) for c in OOD_COLS}

  df_meta_core = df_meta_all.iloc[core_mask]
  is_mapped_core = df_meta_core['country_name'].notna().values
  _BOOT_SHARED.clear()
  _BOOT_SHARED.update({
      'ctx': ctx,
      'df_train': df_train,
      'y_targets': y_targets,
      'final_estimators': final_estimators,
      'X_rp_base': X_rp_base,
      'tier_cov': tier_cov,
      'tier_ood': tier_ood,
      'df_masks': df_masks_all.iloc[core_mask],
      'core_ids': core_ids,
      'e5': df_meta_core['E_5'].values.copy(),
      'e10': df_meta_core['E_10'].values.copy(),
      'e20p': df_meta_core['E_20_50_100'].values.copy(),
      'etot': df_meta_core['E_total'].values.copy(),
      'is_mapped': is_mapped_core.copy(),
      'm_sys60': (df_meta_core['has_system'].values & is_mapped_core).copy(),
      'm_nosys115': ((~df_meta_core['has_system'].values) & is_mapped_core).copy(),
      'country_name': df_meta_core['country_name'].values.copy(),
      'reg_idx_core': ctx['global_region_idx'][core_mask].copy(),
      'omega_t1a_core': ctx['omega_t1a'][core_mask].copy(),
      'omega_t5_shift_core': ctx['omega_t5_shift'][core_mask].copy(),
      'tail_suite': tail_suite,
      'tail_ctry_df': tail_ctry_df,
  })
  del df_inf_all, df_meta_all, df_masks_all

  os.environ['OMP_NUM_THREADS'] = str(threads)
  os.environ['OPENBLAS_NUM_THREADS'] = str(threads)
  os.environ['MKL_NUM_THREADS'] = str(threads)
  os.environ['NUMEXPR_NUM_THREADS'] = str(threads)

  ctx_pkl_path = os.path.join(BOOT_DIR, f'_boot_shared_ctx_{os.getpid()}.joblib')
  joblib.dump(dict(_BOOT_SHARED), ctx_pkl_path, compress=0)
  _BOOT_SHARED.clear()

  reps = []
  try:
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context('spawn'),
        initializer=_init_boot_spawn_worker,
        initargs=(ctx_pkl_path, threads),
    ) as ex:
      futs = [ex.submit(_run_one_bootstrap_rep, (b, threads)) for b in range(n_boot)]
      for fut in as_completed(futs):
        r = fut.result()
        reps.append(r)
        if len(reps) <= 5 or len(reps) % 20 == 0 or len(reps) == n_boot:
          print(
              f"  [Boot {len(reps):3d}/{n_boot}] rep={r['rep']:3d} ({r['elapsed_s']:.1f}s) | "
              f"BaseW={r['baseline_warned_global']/1e6:.2f}M | "
              f"T3_111={r['scenarios']['tier4_all_discoverable__111_combined']['delta_global']/1e6:+.2f}M | "
              f"U_noOOD={r['scenarios']['tier5_universal_no_ood_clamp__111_combined']['delta_global']/1e6:+.2f}M | "
              f"U_ceil={r['scenarios']['tier5_universal_idealized_ceiling__111_combined']['delta_global']/1e6:+.2f}M",
              flush=True,
          )
  finally:
    if os.path.exists(ctx_pkl_path):
      try:
        os.remove(ctx_pkl_path)
      except OSError:
        pass

  base_keys = (
      'baseline_warned_global', 'baseline_warned_mapped', 'baseline_unwarned_global',
      'baseline_warned_sys60', 'baseline_warned_nosys115', 'baseline_unwarned_nosys115',
      'baseline_warned_csi_global', 'baseline_warned_csi_sys60', 'baseline_warned_csi_nosys115',
  )
  scen_metrics = (
      'delta_global', 'delta_sys60', 'delta_nosys115', 'warned_global', 'warned_mapped', 'warned_nosys115',
      'delta_csi_global', 'delta_csi_sys60', 'delta_csi_nosys115', 'warned_csi_global', 'warned_csi_nosys115',
  )

  ci_rows = []
  for base_k in base_keys:
    vals = np.array([r[base_k] for r in reps])
    vals_p = np.array([r['pure_block_boot'][base_k] for r in reps])
    ci_rows.append({
        'scenario': base_k,
        'metric': 'level',
        'boot_mean': float(vals.mean()),
        'boot_sd': float(vals.std(ddof=1)),
        'boot_p025': float(np.percentile(vals, 2.5)),
        'boot_p500': float(np.percentile(vals, 50.0)),
        'boot_p975': float(np.percentile(vals, 97.5)),
        'pure_p025': float(np.percentile(vals_p, 2.5)),
        'pure_p500': float(np.percentile(vals_p, 50.0)),
        'pure_p975': float(np.percentile(vals_p, 97.5)),
    })

  scen_keys = sorted(reps[0]['scenarios'].keys())
  for sk in scen_keys:
    for mk in scen_metrics:
      vals = np.array([r['scenarios'][sk][mk] for r in reps])
      vals_p = np.array([r['pure_block_boot']['scenarios'][sk][mk] for r in reps])
      ci_rows.append({
          'scenario': sk,
          'metric': mk,
          'boot_mean': float(vals.mean()),
          'boot_sd': float(vals.std(ddof=1)),
          'boot_p025': float(np.percentile(vals, 2.5)),
          'boot_p500': float(np.percentile(vals, 50.0)),
          'boot_p975': float(np.percentile(vals, 97.5)),
          'pure_p025': float(np.percentile(vals_p, 2.5)),
          'pure_p500': float(np.percentile(vals_p, 50.0)),
          'pure_p975': float(np.percentile(vals_p, 97.5)),
      })

  df_ci = pd.DataFrame(ci_rows)
  df_ci.to_csv(os.path.join(OUT_DIR, 'bootstrap_table1_fig3_ci95_r2.csv'), index=False)

  # Calibration audit: verify bootstrap median vs. full-sample point estimate
  sum_csv = os.path.join(OUT_DIR, 'multi_mechanism_tiered_exposure_r2.csv')
  if os.path.exists(sum_csv):
    df_pt = pd.read_csv(sum_csv).set_index(['tier_key', 'mechanism'])
    ci_idx = df_ci.set_index(['scenario', 'metric'])
    cal = {}
    checks = [
        ('tier1a_active_post2016_ccby_8811', '000_global_train_joint', 'baseline_warned_global', 'level', 'warned_global'),
        ('tier1a_active_post2016_ccby_8811', '000_global_train_joint', 'baseline_warned_nosys115', 'level', 'warned_nosys115'),
        ('tier4_all_discoverable', '111_all_three_combined_snap', 'tier4_all_discoverable__111_combined', 'delta_global', 'delta_net_global'),
        ('tier4_all_discoverable', '111_all_three_combined_snap', 'tier4_all_discoverable__111_combined', 'delta_nosys115', 'delta_net_nosys115'),
        ('tier5_universal_no_ood_clamp', '111_all_three_combined_snap', 'tier5_universal_no_ood_clamp__111_combined', 'delta_global', 'delta_net_global'),
        ('tier5_universal_no_ood_clamp', '111_all_three_combined_snap', 'tier5_universal_no_ood_clamp__111_combined', 'delta_nosys115', 'delta_net_nosys115'),
        ('tier5_universal_idealized_ceiling', '111_all_three_combined_snap', 'tier5_universal_idealized_ceiling__111_combined', 'delta_global', 'delta_net_global'),
        ('tier5_universal_idealized_ceiling', '111_all_three_combined_snap', 'tier5_universal_idealized_ceiling__111_combined', 'delta_nosys115', 'delta_net_nosys115'),
    ]
    print('\n=== A3 BOOTSTRAP CALIBRATION AUDIT (Median vs. Full-Sample Point Estimate) ===', flush=True)
    for tk, mk, ci_s, ci_m, pt_col in checks:
      if (tk, mk) in df_pt.index and (ci_s, ci_m) in ci_idx.index:
        pt = float(df_pt.loc[(tk, mk), pt_col])
        r_ci = ci_idx.loc[(ci_s, ci_m)]
        med = float(r_ci['boot_p500'])
        p_med = float(r_ci['pure_p500'])
        bias_pct = 100.0 * (med - pt) / pt if abs(pt) > 1e-6 else 0.0
        cal[f'{ci_s}__{ci_m}'] = {
            'point_M': pt / 1e6,
            'boot_med_M': med / 1e6,
            'pure_med_M': p_med / 1e6,
            'rel_diff_pct': bias_pct,
            'ci95_total_M': [float(r_ci['boot_p025']) / 1e6, float(r_ci['boot_p975']) / 1e6],
            'ci95_pure_M': [float(r_ci['pure_p025']) / 1e6, float(r_ci['pure_p975']) / 1e6],
        }
        print(
            f"  {ci_s[:36]:36s} {ci_m:14s}: pt={pt/1e6:+.3f}M | med={med/1e6:+.3f}M ({bias_pct:+.2f}%) | "
            f"total95=[{r_ci['boot_p025']/1e6:+.2f}, {r_ci['boot_p975']/1e6:+.2f}]M | "
            f"pure95=[{r_ci['pure_p025']/1e6:+.2f}, {r_ci['pure_p975']/1e6:+.2f}]M",
            flush=True,
        )
    with open(os.path.join(OUT_DIR, 'bootstrap_calibration_audit_r3.json'), 'w') as f:
      json.dump(cal, f, indent=2)

  # Aggregate per-country 95% CIs for Table S2
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

def run_part_s16(n_threads: int = 24):
  """Clean unified S1.6 national gauge siting across all 1,034,083 basins (9 continents),
  95 features, starting from the full 119,169 publicly discoverable gauge catalog
  (tier4_all_discoverable) and placing N = 1, 5, 10, 20 truly new active gauges on
  unmonitored flooded sub-basins (~tier4_all_discoverable__snap_all)."""
  os.makedirs(S16_DIR, exist_ok=True)
  t0 = time.time()
  print(f'\n--- Running Unified S1.6 (round={ROUND_SUBDIR}, National N=1,5,10,20 Siting beyond 119,169 catalog across 1,034,083 basins) ---', flush=True)

  with open(FEATURES_JSON) as f:
    features_95 = json.load(f)
  df_inf = pd.read_parquet(INFER_PARQUET)[features_95]
  all_ids = df_inf.index
  id_to_pos = {h: i for i, h in enumerate(all_ids)}
  df_meta = load_global_metadata_and_exposure(all_ids)
  df_t4 = pd.read_parquet(os.path.join(TIER_DIR, 'tier4_all_discoverable.parquet')).reindex(all_ids)
  df_masks = pd.read_parquet(os.path.join(TIER_DIR, 'tier_active_and_all_masks_r2.parquet')).reindex(all_ids)
  base_snap_all = df_masks['tier4_all_discoverable__snap_all'].values.astype(bool)
  base_snap_active = df_masks['tier4_all_discoverable__snap_active'].values.astype(bool)
  base_snap_hist_only = base_snap_all & (~base_snap_active)

  p_b4 = pd.read_parquet(os.path.join(PREDS_CACHE_DIR, 'base__tier4_all_discoverable__joint.parquet')).reindex(all_ids)
  p_ft4 = pd.read_parquet(os.path.join(PREDS_CACHE_DIR, 'delta_ft__tier4_all_discoverable__joint.parquet')).reindex(all_ids)
  p_ftda4 = pd.read_parquet(os.path.join(PREDS_CACHE_DIR, 'delta_ft_da__tier4_all_discoverable__joint.parquet')).reindex(all_ids)
  rp_base_mods = {sk: joblib.load(os.path.join(MODELS_DIR, f'base__{mf}.joblib')) for sk, mf in [('r5', '5.0-year-2.0-window-recall'), ('r10', '10.0-year-2.0-window-recall'), ('r20', '20.0-year-2.0-window-recall')]}
  rp_ft_mods = {sk: joblib.load(os.path.join(MODELS_DIR, f'delta_ft__{mf}.joblib')) for sk, mf in [('r5', '5.0-year-2.0-window-recall'), ('r10', '10.0-year-2.0-window-recall'), ('r20', '20.0-year-2.0-window-recall')]}
  rp_ftda_mods = {sk: joblib.load(os.path.join(MODELS_DIR, f'delta_ft_da__{mf}.joblib')) for sk, mf in [('r5', '5.0-year-2.0-window-recall'), ('r10', '10.0-year-2.0-window-recall'), ('r20', '20.0-year-2.0-window-recall')]}

  topo = pd.read_csv(file_paths.HYBAS_ROUTING_TOPOLOGY_PATH, dtype={'HYBAS_ID': str, 'NEXT_DOWN': str})
  topo['hid'] = _norm_hybas(topo['HYBAS_ID'])
  topo['nd'] = np.where(topo['NEXT_DOWN'].astype(str).isin(['0', '0.0', 'nan']), '', _norm_hybas(topo['NEXT_DOWN']))
  topo = topo.drop_duplicates('hid').set_index('hid').reindex(all_ids)

  next_down = np.array([id_to_pos.get(nd, -1) for nd in topo['nd'].fillna('').values], dtype=np.int32)
  sub_area = df_meta['sub_area'].values.astype(np.float64)
  if 'UP_AREA' in topo.columns:
    up_area = np.maximum(pd.to_numeric(topo['UP_AREA'], errors='coerce').fillna(0.0).values.astype(np.float64), sub_area)
  else:
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
    anc_map, desc_map, affected = {}, {}, set()
    for c in cand_list:
      anc, stack, seen = [], [c], {c}
      while stack:
        u = stack.pop()
        anc.append(u)
        for ch in children.get(u, ()):
          if ch not in seen:
            seen.add(ch)
            stack.append(ch)
      anc_map[c] = anc
      affected.update(anc)

      desc, cur = [], c
      while cur >= 0:
        desc.append(cur)
        affected.add(cur)
        cur = next_down[cur]
      desc_map[c] = desc
    return np.array(sorted(affected), dtype=np.int32), anc_map, desc_map

  def compute_updated_cov(cand_list: list[int], aff_arr: np.ndarray, anc_map: dict, desc_map: dict) -> pd.DataFrame:
    aff_ids = all_ids[aff_arr]
    cov_new = df_t4.loc[aff_ids, COV_COLS].copy()
    pos_in_aff = {int(gpos): li for li, gpos in enumerate(aff_arr)}
    tuc_old_all = df_t4['total_upstream_coverage'].values

    unblocked_desc = {}
    for c in cand_list:
      path = []
      for u in desc_map[c]:
        if base_snap_all[u]:
          break
        path.append(u)
      unblocked_desc[c] = path
    unblocked_desc_sets = {c: set(p) for c, p in unblocked_desc.items()}

    raw_new = {c: max(0.0, up_area[c] * (1.0 - float(tuc_old_all[c]))) for c in cand_list}
    eff_new_area = {}
    for c in cand_list:
      up_cands = [y for y in cand_list if y != c and c in unblocked_desc_sets[y]]
      max_up = [y for y in up_cands if not any((y2 != y) and (y2 in unblocked_desc_sets[y]) for y2 in up_cands)]
      eff_new_area[c] = max(0.0, raw_new[c] - sum(raw_new[y] for y in max_up))

    up_add_area, up_max_new, up_cnt_add, down_max_new = defaultdict(float), defaultdict(float), defaultdict(int), defaultdict(float)
    for c in cand_list:
      a_c, a_eff = up_area[c], eff_new_area[c]
      for u in unblocked_desc[c]:
        if up_area[u] > 0:
          up_add_area[u] += a_eff / up_area[u]
      for u in desc_map[c]:
        if up_area[u] > 0:
          up_max_new[u] = max(up_max_new[u], a_c / up_area[u])
        up_cnt_add[u] += 1
      if a_c > 0:
        for u in anc_map[c]:
          down_max_new[u] = max(down_max_new[u], up_area[u] / a_c)

    tuc = cov_new['total_upstream_coverage'].values.copy()
    muc = cov_new['max_upstream_coverage'].values.copy()
    mdc = cov_new['max_downstream_coverage'].values.copy()
    ugc = cov_new['upstream_gauge_count'].values.copy()

    for u, add_f in up_add_area.items():
      tuc[pos_in_aff[u]] = min(1.0, tuc[pos_in_aff[u]] + add_f)
    for u, mx_f in up_max_new.items():
      muc[pos_in_aff[u]] = min(1.0, max(muc[pos_in_aff[u]], mx_f))
    for u, cnt_f in up_cnt_add.items():
      ugc[pos_in_aff[u]] = ugc[pos_in_aff[u]] + cnt_f
    for u, dm_f in down_max_new.items():
      mdc[pos_in_aff[u]] = min(1.0, max(mdc[pos_in_aff[u]], dm_f))
    for c in cand_list:
      li = pos_in_aff[c]
      tuc[li] = muc[li] = mdc[li] = 1.0

    cov_new['total_upstream_coverage'] = tuc
    cov_new['max_upstream_coverage'] = muc
    cov_new['max_downstream_coverage'] = mdc
    cov_new['upstream_gauge_count'] = ugc
    return cov_new

  df_cand_pool = df_meta[(df_meta['country_name'].notna()) & (df_meta['E_total'] > 0) & (~base_snap_all)].copy()
  df_cand_pool['gpos'] = [id_to_pos[h] for h in df_cand_pool.index]
  country_top20_raw = {}
  for cname, grp in df_cand_pool.groupby('country_name'):
    top20 = grp.sort_values('E_total', ascending=False).head(20)['gpos'].tolist()
    if top20:
      country_top20_raw[cname] = top20

  e5 = df_meta['E_5'].values
  e10 = df_meta['E_10'].values
  e20p = df_meta['E_20_50_100'].values
  ctry_arr = df_meta['country_name'].values

  # Baseline forecasted exposure using all 119,169 publicly discoverable gauges
  # (global training everywhere + FT+DA on operational gauges + FT on inactive historical gauges)
  r_b4_comb = {}
  for sk in ('r5', 'r10', 'r20'):
    pb = p_b4[sk].values
    r_b4_comb[sk] = np.where(
        base_snap_active,
        np.clip(pb + p_ftda4[sk].values, 0.0, 1.0),
        np.where(
            base_snap_hist_only,
            np.clip(pb + p_ft4[sk].values, 0.0, 1.0),
            pb,
        ),
    )
  w_b4 = e5 * r_b4_comb['r5'] + e10 * r_b4_comb['r10'] + e20p * r_b4_comb['r20']
  print(f'   Verified 119,169-gauge combined baseline global forecasted exposure = {w_b4.sum()/1e6:.3f}M/yr (Table S6 = 7.75+1.52 = 9.27M/yr)', flush=True)

  X_rp_global = rp_base_mods['r5'].impl._preprocess_features(df_inf)
  for c in COV_COLS:
    X_rp_global[c] = df_t4[c].values
  for c in OOD_COLS:
    X_rp_global[c] = df_t4[c].values

  def eval_sets_batch(tasks: list[tuple[list[int], str]]) -> list[tuple[float, float, float, float]]:
    results = [None] * len(tasks)
    chunk_items = []
    chunk_rows = 0

    def flush_chunk():
      if not chunk_items:
        return
      X_cat = pd.concat([it['X'] for it in chunk_items], axis=0, ignore_index=True)
      pb_cat = _fast_predict_rp_family(rp_base_mods, X_cat, n_threads=n_threads)
      pft_cat = _fast_predict_rp_family(rp_ft_mods, X_cat, n_threads=n_threads)
      pftda_cat = _fast_predict_rp_family(rp_ftda_mods, X_cat, n_threads=n_threads)
      offset = 0
      for it in chunk_items:
        n = len(it['aff'])
        aff = it['aff']
        sl = slice(offset, offset + n)
        offset += n
        r5_b, r10_b, r20_b = pb_cat['r5'][sl], pb_cat['r10'][sl], pb_cat['r20'][sl]
        r5_f, r10_f, r20_f = pft_cat['r5'][sl], pft_cat['r10'][sl], pft_cat['r20'][sl]
        r5_d, r10_d, r20_d = pftda_cat['r5'][sl], pftda_cat['r10'][sl], pftda_cat['r20'][sl]
        is_act_old = base_snap_active[aff]
        is_hist_old = base_snap_hist_only[aff]
        is_new = np.isin(aff, it['cands'])

        # New gauges used for global training only (existing 119k gauges keep existing FT / FT+DA)
        r5_tr = np.where(is_act_old, np.clip(r5_b + r5_d, 0.0, 1.0), np.where(is_hist_old, np.clip(r5_b + r5_f, 0.0, 1.0), r5_b))
        r10_tr = np.where(is_act_old, np.clip(r10_b + r10_d, 0.0, 1.0), np.where(is_hist_old, np.clip(r10_b + r10_f, 0.0, 1.0), r10_b))
        r20_tr = np.where(is_act_old, np.clip(r20_b + r20_d, 0.0, 1.0), np.where(is_hist_old, np.clip(r20_b + r20_f, 0.0, 1.0), r20_b))
        w_tr = e5[aff] * r5_tr + e10[aff] * r10_tr + e20p[aff] * r20_tr

        # New active gauges used for global training + local fine-tuning + real-time DA
        is_act_all = is_act_old | is_new
        r5_co = np.where(is_act_all, np.clip(r5_b + r5_d, 0.0, 1.0), np.where(is_hist_old, np.clip(r5_b + r5_f, 0.0, 1.0), r5_b))
        r10_co = np.where(is_act_all, np.clip(r10_b + r10_d, 0.0, 1.0), np.where(is_hist_old, np.clip(r10_b + r10_f, 0.0, 1.0), r10_b))
        r20_co = np.where(is_act_all, np.clip(r20_b + r20_d, 0.0, 1.0), np.where(is_hist_old, np.clip(r20_b + r20_f, 0.0, 1.0), r20_b))
        w_co = e5[aff] * r5_co + e10[aff] * r10_co + e20p[aff] * r20_co

        d_tr = w_tr - w_b4[aff]
        d_co = w_co - w_b4[aff]
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
  country_unw_exp = pd.Series(df_meta['E_total'].values - w_b4, index=all_ids).groupby(df_meta['country_name']).sum().to_dict()

  # Ensure all flood-exposed countries appear in output even if 0 unmonitored flooded sub-basins remain
  all_flooded_countries = sorted(
      df_meta[(df_meta['country_name'].notna()) & (df_meta['E_total'] > 0)]['country_name'].unique()
  )
  for cname in all_flooded_countries:
    country_top20_raw.setdefault(cname, [])

  single_tasks, single_meta = [], []
  for cname, raw_cands in sorted(country_top20_raw.items()):
    for c in raw_cands:
      single_tasks.append(([c], cname))
      single_meta.append((cname, c))
  print(f'   Phase 1: Batch-evaluating {len(single_tasks)} single candidates across {len(all_flooded_countries)} flood-exposed countries...', flush=True)
  single_res = eval_sets_batch(single_tasks)
  scored_by_country = defaultdict(list)
  for (cname, c), (dt_c, dc_c, dt_g, dc_g) in zip(single_meta, single_res):
    scored_by_country[cname].append((c, dt_c, dc_c, dt_g, dc_g))

  joint_tasks, joint_meta = [], []
  for cname in all_flooded_countries:
    scored = scored_by_country[cname]
    if not scored:
      continue
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
  for cname in all_flooded_countries:
    row = {
        'Country': cname,
        'Baseline Exposure (people/yr)': float(country_base_exp.get(cname, 0.0)),
        'Unwarned Exposure (people/yr)': float(country_unw_exp.get(cname, 0.0)),
        'Max_Evaluated_Candidates': len(country_top20_raw[cname]),
        'has_system': cap_utils.has_effective_flood_warning_system(cname),
        'CAP_Tier': cap_utils.assign_cap_tier(cname),
    }
    for N in (1, 5, 10, 20):
      row[f'N={N}_Train_Dom'] = 0.0
      row[f'N={N}_Train_Dom_PctBase'] = 0.0
      row[f'N={N}_Train_Glb'] = 0.0
      row[f'N={N}_Combined_Dom'] = 0.0
      row[f'N={N}_Combined_Dom_PctBase'] = 0.0
      row[f'N={N}_Combined_Glb'] = 0.0
    nat_rows_map[cname] = row
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

  df_nat = pd.DataFrame(list(nat_rows_map.values())).sort_values('N=20_Combined_Dom', ascending=False).reset_index(drop=True)
  out_nat = os.path.join(S16_DIR, 'country_joint_n_gauge_r2.csv')
  df_nat.to_csv(out_nat, index=False)
  print(f'Saved unified S1.6 national table ({len(df_nat)} countries) to {out_nat} in {(time.time() - t0)/60:.1f} min!', flush=True)
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
  parser.add_argument('--threads', type=int, default=min(24, os.cpu_count() or 8))
  parser.add_argument('--n-boot', type=int, default=200)
  parser.add_argument('--boot-workers', type=int, default=48)
  parser.add_argument('--boot-threads', type=int, default=1)
  parser.add_argument(
      '--boot-min-exp',
      type=float,
      default=5.0,
      help='Core basin E_total threshold (people/yr) for bootstrap prediction; micro-exposure tail closed exactly from Stage-D full-sample preds (0 = all 590,652 positive basins).',
  )
  parser.add_argument(
      '--boot-country-cov',
      type=float,
      default=0.95,
      help='Per-territory minimum exposure fraction guaranteed inside the core bootstrap set.',
  )
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
    run_part_bootstrap(
        n_boot=args.n_boot,
        workers=args.boot_workers,
        threads=args.boot_threads,
        min_exp=args.boot_min_exp,
        country_cov=args.boot_country_cov,
    )
  if args.part in ('s16', 'all'):
    run_part_s16(n_threads=args.threads)


if __name__ == '__main__':
  main()
