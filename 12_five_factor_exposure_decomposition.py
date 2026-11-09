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

"""Unified 5-Factor Global Flood Exposure Decomposition on the Real-World Predictor.

Evaluates:
1. Baseline warned and unwarned riverine flood exposure across all 964,084 global
   HydroBASINS Level 12 sub-basins on the real-world (`ungauged`) skill predictor
   (`M_ungauged`), including open-access gauge coverage statistics.
2. Observational counterfactual ceilings across all exposed sub-basins for
   Factor 2 (universal 100% river-network gauge coverage, alongside the
   independent temporal-split `M_gauged` confirmation cross-check), Factor 3
   (median similarity of top-quartile training basins), Factors 2 + 3 jointly
   (defining the Factor 5 residual meteorological/hydraulic ceiling), and
   Factor 4 (50-year record length).
3. Unified In-Manifold TreeSHAP-Targeted Analysis ("Way B") across ALL factors
   (Factor 3 OOD, Factor 2 Gauge Coverage, Factor 4 Record Length, and cleanly
   separated Factor 1 physical/anthropogenic sub-groups 1a-1g): identifying
   real watersheds where `M_ungauged`'s net group TreeSHAP attribution is
   negative and quantifying both parameter-free 100% SHAP penalty elimination
   and common-scale (P_star = 0.10) nonlinear missed-event reduction curves
   R_new(gamma) = 1 - (1 - R) * (1 - gamma)^w across gamma in {0.10, 0.25, 0.50}.
"""

import concurrent.futures
import copy
import glob
import json
import os
import pickle
import sys
import time
import types
import warnings

os.environ.setdefault(
    'GEO_BIAS_DATA_DIR', 'data'
)
os.environ.setdefault(
    'GEO_BIAS_THIRD_PARTY_DIR',
    'data',
)
os.environ.setdefault(
    'GEO_BIAS_OUTPUT_DIR', 'data'
)

from utils import gfile_compat as gfile

from absl import app
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import r2_score

from utils import cap_utils
from utils import feature_utils
from utils import file_paths
from utils import io_utils
from utils import skill_predictor as sp
import multiprocessing

sys.modules['utils.skill_predictor'] = sp

warnings.filterwarnings('ignore')

# Trapezoidal exceedance-frequency quadrature weights across [5, 10, 20, 50, 100]-yr floods
RP_WEIGHTS = {
    5: 0.05,
    10: 0.075,
    20: 0.04,
    50: 0.02,
    100: 0.005,
}
UNIQUE_METRICS = [
    '5.0-year-2.0-window-recall',
    '10.0-year-2.0-window-recall',
    '20.0-year-2.0-window-recall',
]

# Universal cross-factor reference recall penalty (10 percentage points) for Way B nonlinear curves
P_STAR_COMMON = 0.10

# Clean separation of all 21 surface-water, inundation, wetland, lake, and cryosphere storage features (1c)
# from terrestrial vegetation, agricultural, and biome land cover features (1f), regulation (1b), and urban (1g).
FLOODPLAIN_WATER_FEATS = {
    'inu_pc_umn', 'inu_pc_umx', 'inu_pc_ult',
    'lka_pc_use', 'lkv_mc_usu',
    'wet_pc_ug2', 'wet_pc_u01', 'wet_pc_u03', 'wet_pc_u04', 'wet_pc_u05',
    'wet_pc_u06', 'wet_pc_u07', 'wet_pc_u08', 'wet_pc_u09',
    'prm_pc_use', 'gla_pc_use',
    'glc_pc_u11', 'glc_pc_u17', 'pnv_pc_u09', 'pnv_pc_u14', 'pnv_pc_u15',
}
REGULATION_FEATS = {
    'rev_mc_usu', 'ria_ha_usu', 'ire_pc_use', 'wet_pc_u02', 'wet_pc_ug1',
    'log_rev_mc_usu', 'log_ria_ha_usu', 'dor_pc_pva', 'dam_ct_usu',
}
URBAN_FEATS = {
    'urb_pc_use', 'glc_pc_u13', 'gdp_ud_usu', 'hft_ix_u09', 'hft_ix_u93',
    'nli_ix_uav', 'pop_ct_usu', 'ppd_pk_uav', 'rdd_mk_uav',
    'human_impact_density', 'human_intensity', 'log_pop_ct_usu',
}

GROUP_SHORT_SLUGS = {
    'Factor 3: Out-of-Distribution Representation Bias': 'f3_ood',
    'Factor 2: River-Network Gauge Coverage': 'f2_gauge',
    'Factor 4: Historical Record Length': 'f4_reclen',
    '1a: Arid & Convective Hydroclimate': '1a_climate',
    '1b: Dam, Reservoir & Irrigation Regulation': '1b_regulation',
    '1c: Floodplain, Wetland & Lake Storage': '1c_floodplain',
    '1d: Steep & Small-Scale Channel Routing': '1d_routing',
    '1e: Subsurface Soil & Karst Hydrogeology': '1e_soil',
    '1f: Sparse Canopy, Agricultural & Biome Land Cover': '1f_landcover',
    '1g: Urban & Human Footprint': '1g_urban',
}


def map_feature_to_5_factors(feat_name: str) -> tuple[str, str, str]:
  """Maps any raw or engineered feature to (Factor, GroupKey, PrimaryRemedy)."""
  if feat_name in [
      'total_upstream_coverage',
      'max_upstream_coverage',
      'max_downstream_coverage',
      'upstream_gauge_count',
  ]:
    return (
        'Factor 2: River-Network Gauge Coverage',
        'Factor 2: River-Network Gauge Coverage',
        'Targeted In Situ Gauging & Assimilation',
    )

  if feat_name in [
      'anomaly_score_if',
      'anomaly_score_ocsvm',
      'anomaly_score_lof',
  ]:
    return (
        'Factor 3: Out-of-Distribution Representation Bias',
        'Factor 3: Out-of-Distribution Representation Bias',
        'Under-Sampled Gauging + Training Re-Weighting / Local Fine-Tuning',
    )

  if feat_name == 'record_length':
    return (
        'Factor 4: Historical Record Length',
        'Factor 4: Historical Record Length',
        'Archive Extension',
    )

  if feat_name in FLOODPLAIN_WATER_FEATS:
    return (
        'Factor 1: Hydrological & Physical Process Difficulty',
        '1c: Floodplain, Wetland & Lake Storage',
        'Model Architecture & Assimilation (2D Floodplain/Lake Routing & Satellite Water)',
    )

  if feat_name in REGULATION_FEATS:
    return (
        'Factor 1: Hydrological & Physical Process Difficulty',
        '1b: Dam, Reservoir & Irrigation Regulation',
        'Dynamic Input Data (Reservoir Releases, SWOT Storage & Irrigation)',
    )

  if feat_name in URBAN_FEATS:
    return (
        'Factor 1: Hydrological & Physical Process Difficulty',
        '1g: Urban & Human Footprint',
        'Urban Stormwater & Impervious Drainage Representation',
    )

  subcat = feature_utils.FEATURE_CLASSIFICATION.get(feat_name, None)
  if subcat == 'Climate & Hydrology' or feat_name in [
      'water_balance',
      'aridity_ratio',
      'p_minus_pet',
      'p_div_pet',
      'et_deficit',
      'precip_temp_inter',
      'forest_aridity_inter',
  ]:
    return (
        'Factor 1: Hydrological & Physical Process Difficulty',
        '1a: Arid & Convective Hydroclimate',
        'Atmospheric State Coupling & High-Resolution Dryland Runoff Modeling',
    )

  if subcat == 'Topography' or feat_name in [
      'twi_proxy',
      'log_calculated_drain_area',
      'log_riv_tc_usu',
  ]:
    return (
        'Factor 1: Hydrological & Physical Process Difficulty',
        '1d: Steep & Small-Scale Channel Routing',
        'Model Architecture (Sub-Daily & High-Resolution Channel Routing)',
    )

  if subcat == 'Soil and Geology' or feat_name == 'soil_porosity':
    return (
        'Factor 1: Hydrological & Physical Process Difficulty',
        '1e: Subsurface Soil & Karst Hydrogeology',
        'Subsurface Representation & Soil-Moisture Assimilation',
    )

  if subcat == 'Land Cover' or feat_name in [
      'total_forest_pc',
      'total_forest_cover',
      'total_arid_cover',
  ]:
    return (
        'Factor 1: Hydrological & Physical Process Difficulty',
        '1f: Sparse Canopy, Agricultural & Biome Land Cover',
        'Sparse-Canopy Land-Surface Buffering & Biome-Aware Regionalization',
    )

  raise ValueError(f'Unclassified feature in 5-factor mapping: {feat_name}')


def _set_parallel(mod):
  for _, est in mod.impl.ensemble.estimators:
    if hasattr(est, 'n_jobs'):
      est.n_jobs = -1
  for est in mod.impl.ensemble.estimators_:
    if hasattr(est, 'n_jobs'):
      est.n_jobs = -1
    elif hasattr(est, 'named_steps'):
      for step in est.named_steps.values():
        if hasattr(step, 'n_jobs'):
          step.n_jobs = -1


def _shap_chunk(explainer, chunk):
  return explainer.shap_values(chunk, check_additivity=False)


def main(argv=None):
  del argv
  t0 = time.time()
  file_paths.init('5.0-year-2.0-window-recall')
  file_paths.ensure_directories_exist()

  print('1. Loading global basin exposure (defended, Fathom v3 + WorldPop)...', flush=True)
  df_exp = pd.read_csv(file_paths.GLOBAL_BASIN_EXPOSURE_DEFENDED_BY_RP_PATH)
  id_col = (
      'hybas_id'
      if 'hybas_id' in df_exp.columns
      else ('HYBAS_ID' if 'HYBAS_ID' in df_exp.columns else df_exp.columns[0])
  )
  df_exp[id_col] = (
      df_exp[id_col]
      .astype(str)
      .str.replace(r'\.0$', '', regex=True)
      .apply(lambda x: x if x.startswith('hybas_') else f'hybas_{x}')
  )
  df_exp = df_exp.drop_duplicates(subset=[id_col]).set_index(id_col)

  def _get_exp_col(df, t):
    for c in [f'exposed_pop_{t}', f'exposure_{t}yr', str(t)]:
      if c in df.columns:
        return df[c].fillna(0.0)
    raise KeyError(f'Could not find exposure column for T={t}')

  df_exp['E_5'] = RP_WEIGHTS[5] * _get_exp_col(df_exp, 5)
  df_exp['E_10'] = RP_WEIGHTS[10] * _get_exp_col(df_exp, 10)
  df_exp['E_20_50_100'] = (
      RP_WEIGHTS[20] * _get_exp_col(df_exp, 20)
      + RP_WEIGHTS[50] * _get_exp_col(df_exp, 50)
      + RP_WEIGHTS[100] * _get_exp_col(df_exp, 100)
  )
  df_exp['E_total'] = df_exp['E_5'] + df_exp['E_10'] + df_exp['E_20_50_100']

  print('2. Loading HyBAS country / CAP early warning system mapping...', flush=True)
  df_map = pd.read_csv(file_paths.HYBAS_MAPPING_PATH)
  if 'Unnamed: 0' in df_map.columns and 'hybas_id' not in df_map.columns:
    df_map = df_map.rename(columns={'Unnamed: 0': 'hybas_id'})
  df_map['hybas_id'] = (
      df_map['hybas_id']
      .astype(str)
      .str.replace(r'\.0$', '', regex=True)
      .apply(lambda x: x if x.startswith('hybas_') else f'hybas_{x}')
  )
  df_map['country_name_clean'] = df_map['country_name'].replace(file_paths.country_name_map)
  df_map['has_system'] = df_map['country_name_clean'].map(
      cap_utils.has_effective_flood_warning_system
  )
  df_map['cap_tier'] = df_map['country_name_clean'].map(cap_utils.assign_cap_tier)
  df_map_dedup = df_map.drop_duplicates('hybas_id').set_index('hybas_id')

  df_exp['country_name'] = df_map_dedup['country_name_clean'].reindex(df_exp.index)
  df_exp['has_system'] = (
      df_map_dedup['has_system'].reindex(df_exp.index).fillna(False).astype(bool)
  )
  df_exp['cap_tier'] = (
      df_map_dedup['cap_tier'].reindex(df_exp.index).fillna('Tier 4: Complete Systemic Gap (No CAP)')
  )

  print('\n3. Loading primary real-world skill predictors (M_ungauged) and confirmation M_gauged...', flush=True)
  models_u, models_g = {}, {}
  preds_u, preds_g = {}, {}
  cv_r2_u, cv_r2_g = {}, {}

  for m in UNIQUE_METRICS:
    with open(os.path.join(file_paths.MODELS_DIR, f'final_ungauged_model_{m}.pkl'), 'rb') as f:
      models_u[m] = sp.load_skill_predictor(f)
    with open(os.path.join(file_paths.MODELS_DIR, f'final_gauged_model_{m}.pkl'), 'rb') as f:
      models_g[m] = sp.load_skill_predictor(f)
    _set_parallel(models_u[m])
    _set_parallel(models_g[m])

    su = pd.read_csv(
        os.path.join(file_paths.HYBAS_PREDS_DIR, f'global_hybas_predictions_ungauged_{m}.csv'),
        index_col=0,
    ).iloc[:, 0]
    su.index = (
        su.index.astype(str)
        .str.replace(r'\.0$', '', regex=True)
        .map(lambda x: x if x.startswith('hybas_') else f'hybas_{x}')
    )
    preds_u[m] = su.reindex(df_exp.index).fillna(su.median()).clip(0.0, 1.0)

    sg = pd.read_csv(
        os.path.join(file_paths.HYBAS_PREDS_DIR, f'global_hybas_predictions_gauged_{m}.csv'),
        index_col=0,
    ).iloc[:, 0]
    sg.index = (
        sg.index.astype(str)
        .str.replace(r'\.0$', '', regex=True)
        .map(lambda x: x if x.startswith('hybas_') else f'hybas_{x}')
    )
    preds_g[m] = sg.reindex(df_exp.index).fillna(sg.median()).clip(0.0, 1.0)

    cv_u_path = os.path.join(file_paths.CV_PREDS_DIR, f'ungauged_regression_cv_preds_{m}.csv')
    cv_g_path = os.path.join(file_paths.CV_PREDS_DIR, f'gauged_regression_cv_preds_{m}.csv')
    if os.path.exists(cv_u_path):
      dfc = pd.read_csv(cv_u_path).dropna(subset=['true', 'pred'])
      cv_r2_u[m] = float(r2_score(dfc['true'], dfc['pred']))
    if os.path.exists(cv_g_path):
      dfc = pd.read_csv(cv_g_path).dropna(subset=['true', 'pred'])
      cv_r2_g[m] = float(r2_score(dfc['true'], dfc['pred']))

  # Baseline warned and unwarned exposure purely from M_ungauged
  df_exp['warned_ungauged'] = (
      df_exp['E_5'] * preds_u['5.0-year-2.0-window-recall']
      + df_exp['E_10'] * preds_u['10.0-year-2.0-window-recall']
      + df_exp['E_20_50_100'] * preds_u['20.0-year-2.0-window-recall']
  )
  df_exp['unwarned_ungauged'] = df_exp['E_total'] - df_exp['warned_ungauged']
  df_exp['recall_5y_u'] = preds_u['5.0-year-2.0-window-recall']
  df_exp['recall_10y_u'] = preds_u['10.0-year-2.0-window-recall']
  df_exp['recall_20y_u'] = preds_u['20.0-year-2.0-window-recall']

  # Confirmation-only M_gauged delta
  df_exp['delta_mgauged_confirm_rp_clipped'] = (
      df_exp['E_5'] * (preds_g['5.0-year-2.0-window-recall'] - preds_u['5.0-year-2.0-window-recall']).clip(lower=0.0)
      + df_exp['E_10'] * (preds_g['10.0-year-2.0-window-recall'] - preds_u['10.0-year-2.0-window-recall']).clip(lower=0.0)
      + df_exp['E_20_50_100'] * (preds_g['20.0-year-2.0-window-recall'] - preds_u['20.0-year-2.0-window-recall']).clip(lower=0.0)
  )

  print('\n4. Loading HyBAS Zarr features & M_ungauged training data...', flush=True)
  df_hybas = io_utils.load_hybas_zarr(file_paths.HYBAS_GENERIC_DATA_ZARR_PATH)
  df_hybas.index = (
      df_hybas.index.astype(str)
      .str.replace(r'\.0$', '', regex=True)
      .map(lambda x: x if x.startswith('hybas_') else f'hybas_{x}')
  )

  x_train_u, y_train_u_5 = io_utils.load_and_preprocess_data(
      'ungauged', '5.0-year-2.0-window-recall', None
  )

  # Top-quartile reference benchmark computed strictly from M_ungauged training distribution
  top_q_thresh_u = y_train_u_5.quantile(0.75)
  top_q_basins_u = x_train_u.loc[y_train_u_5 >= top_q_thresh_u]
  ref_medians_u = top_q_basins_u.median()

  common_ids = df_exp.index.intersection(df_hybas.index)
  df_exp_all = df_exp.loc[common_ids].copy()

  # Open-access gauge coverage audit across all 964,084 global sub-basins
  cov_cols = [
      'total_upstream_coverage',
      'max_upstream_coverage',
      'max_downstream_coverage',
      'upstream_gauge_count',
  ]
  zero_cov_mask = (df_hybas.loc[df_exp_all.index, cov_cols].fillna(0.0).max(axis=1) <= 0.0)
  df_exp_all['zero_open_gauge_cov'] = zero_cov_mask
  zero_cov_summary = {
      'n_basins_zero_cov': int(zero_cov_mask.sum()),
      'pct_basins_zero_cov': float(100.0 * zero_cov_mask.mean()),
      'exposure_zero_cov': float(df_exp_all.loc[zero_cov_mask, 'E_total'].sum()),
      'pct_exposure_zero_cov': float(
          100.0 * df_exp_all.loc[zero_cov_mask, 'E_total'].sum() / df_exp_all['E_total'].sum()
      ),
      'unwarned_zero_cov': float(df_exp_all.loc[zero_cov_mask, 'unwarned_ungauged'].sum()),
      'pct_unwarned_zero_cov': float(
          100.0 * df_exp_all.loc[zero_cov_mask, 'unwarned_ungauged'].sum() / df_exp_all['unwarned_ungauged'].sum()
      ),
  }
  print(
      f"   Zero open-access gauge coverage: {zero_cov_summary['n_basins_zero_cov']:,} / {len(df_exp_all):,} sub-basins "
      f"({zero_cov_summary['pct_basins_zero_cov']:.1f}%), containing {zero_cov_summary['exposure_zero_cov']:,.0f} / "
      f"{df_exp_all['E_total'].sum():,.0f} people/yr ({zero_cov_summary['pct_exposure_zero_cov']:.1f}%) of global flood exposure!",
      flush=True,
  )

  df_exp_pos_all = df_exp_all[df_exp_all['E_total'] > 0].sort_values('E_total', ascending=False)
  print(
      f'   Evaluating observational counterfactual ceilings across ALL {len(df_exp_pos_all):,} exposed HyBAS sub-basins '
      f'(100.0% of {df_exp_all["E_total"].sum():,.0f} people/yr exposure)...',
      flush=True,
  )
  cf_ids = df_exp_pos_all.index
  df_hybas_cf = df_hybas.loc[cf_ids]
  df_exp_cf = df_exp_all.loc[cf_ids].copy()

  def eval_counterfactual_u(df_mod, store_col=None):
    """Evaluates M_ungauged observational counterfactual across ALL exposed sub-basins."""
    x_inf = df_mod.reindex(columns=x_train_u.columns, fill_value=0)
    p5 = pd.Series(models_u['5.0-year-2.0-window-recall'].predict(x_inf).clip(0.0, 1.0), index=cf_ids)
    p10 = pd.Series(models_u['10.0-year-2.0-window-recall'].predict(x_inf).clip(0.0, 1.0), index=cf_ids)
    p20 = pd.Series(models_u['20.0-year-2.0-window-recall'].predict(x_inf).clip(0.0, 1.0), index=cf_ids)
    b5 = preds_u['5.0-year-2.0-window-recall'].loc[cf_ids]
    b10 = preds_u['10.0-year-2.0-window-recall'].loc[cf_ids]
    b20 = preds_u['20.0-year-2.0-window-recall'].loc[cf_ids]

    warned_new = df_exp_cf['E_5'] * p5 + df_exp_cf['E_10'] * p10 + df_exp_cf['E_20_50_100'] * p20
    warned_base = df_exp_cf['E_5'] * b5 + df_exp_cf['E_10'] * b10 + df_exp_cf['E_20_50_100'] * b20
    delta = warned_new - warned_base
    delta_pos = delta.clip(lower=0.0)
    delta_rp = (
        df_exp_cf['E_5'] * (p5 - b5).clip(lower=0.0)
        + df_exp_cf['E_10'] * (p10 - b10).clip(lower=0.0)
        + df_exp_cf['E_20_50_100'] * (p20 - b20).clip(lower=0.0)
    )
    if store_col is not None:
      df_exp_all[store_col] = 0.0
      df_exp_all.loc[cf_ids, store_col] = delta_rp

    return {
        'net_delta_global': float(delta.sum()),
        'net_delta_with_sys': float(delta.loc[df_exp_cf['has_system']].sum()),
        'net_delta_no_sys': float(delta.loc[~df_exp_cf['has_system']].sum()),
        'pos_delta_global': float(delta_pos.sum()),
        'pos_delta_with_sys': float(delta_pos.loc[df_exp_cf['has_system']].sum()),
        'pos_delta_no_sys': float(delta_pos.loc[~df_exp_cf['has_system']].sum()),
        'rp_clipped_delta_global': float(delta_rp.sum()),
        'rp_clipped_delta_with_sys': float(delta_rp.loc[df_exp_cf['has_system']].sum()),
        'rp_clipped_delta_no_sys': float(delta_rp.loc[~df_exp_cf['has_system']].sum()),
        'unwarned_remaining_global': float((df_exp_cf['E_total'] - np.maximum(warned_base, warned_new)).sum()),
        'unwarned_remaining_with_sys': float((df_exp_cf.loc[df_exp_cf['has_system'], 'E_total'] - np.maximum(warned_base, warned_new).loc[df_exp_cf['has_system']]).sum()),
        'unwarned_remaining_no_sys': float((df_exp_cf.loc[~df_exp_cf['has_system'], 'E_total'] - np.maximum(warned_base, warned_new).loc[~df_exp_cf['has_system']]).sum()),
        'median_skill_delta_5y': float((p5 - b5).median()),
        'mean_skill_delta_5y': float((p5 - b5).mean()),
    }

  interventions = {}

  # Factor 2 (M_ungauged): Universal Local & River-Network Coverage (100% coverage, >=1 gauge)
  df_mod_cov_local = df_hybas_cf.copy()
  for c in ['total_upstream_coverage', 'max_upstream_coverage', 'max_downstream_coverage']:
    if c in df_mod_cov_local.columns:
      df_mod_cov_local[c] = 1.0
  if 'upstream_gauge_count' in df_mod_cov_local.columns:
    df_mod_cov_local['upstream_gauge_count'] = np.maximum(df_mod_cov_local['upstream_gauge_count'], 1.0)
  interventions['Factor 2 (M_ungauged Primary): Universal Local & Network Gauge Coverage (100% Coverage)'] = eval_counterfactual_u(
      df_mod_cov_local, store_col='cf_factor2_gauge_cov'
  )

  # Factor 2 sensitivity (M_ungauged): Dense Multi-Gauge River Corridor (75th pct of covered training basins)
  df_mod_cov_dense = df_hybas_cf.copy()
  for c in cov_cols:
    if c in df_mod_cov_dense.columns and c in x_train_u.columns:
      q75 = x_train_u[x_train_u[c] > 0][c].quantile(0.75) if (x_train_u[c] > 0).any() else 1.0
      df_mod_cov_dense[c] = np.maximum(df_mod_cov_dense[c], q75)
  interventions['Factor 2 (M_ungauged Dense Corridor): Multi-Gauge Nested Corridor Coverage (75th pct)'] = eval_counterfactual_u(
      df_mod_cov_dense, store_col='cf_factor2_dense_corridor'
  )

  # Factor 3 (M_ungauged): Eliminating Out-of-Distribution Representation Bias (median of top-quartile training basins)
  ood_cols = ['anomaly_score_if', 'anomaly_score_ocsvm', 'anomaly_score_lof']
  df_mod_ood = df_hybas_cf.copy()
  for c in ood_cols:
    if c in df_mod_ood.columns and c in ref_medians_u.index:
      df_mod_ood[c] = np.maximum(df_mod_ood[c], ref_medians_u[c]) if c != 'anomaly_score_lof' else ref_medians_u[c]
  interventions['Factor 3 (M_ungauged): Eliminating Out-of-Distribution Representation Bias'] = eval_counterfactual_u(
      df_mod_ood, store_col='cf_factor3_ood'
  )

  # Combined Observational Ceiling (Factor 2 + Factor 3 together in M_ungauged) -> defines Factor 5 Residual Ceiling
  df_mod_obs_ceil = df_mod_ood.copy()
  for c in ['total_upstream_coverage', 'max_upstream_coverage', 'max_downstream_coverage']:
    if c in df_mod_obs_ceil.columns:
      df_mod_obs_ceil[c] = 1.0
  if 'upstream_gauge_count' in df_mod_obs_ceil.columns:
    df_mod_obs_ceil['upstream_gauge_count'] = np.maximum(df_mod_obs_ceil['upstream_gauge_count'], 1.0)
  interventions['Combined Observational Ceiling (Factor 2 + Factor 3 in M_ungauged)'] = eval_counterfactual_u(
      df_mod_obs_ceil, store_col='cf_obs_ceiling'
  )

  # Factor 4 (M_ungauged): Extending Historical Record Length to 50 Years
  df_mod_rec_u = df_hybas_cf.copy()
  df_mod_rec_u['record_length'] = 50.0
  interventions['Factor 4 (M_ungauged): Extending Historical Record Length to 50 Years'] = eval_counterfactual_u(
      df_mod_rec_u, store_col='cf_factor4_reclen'
  )

  # Secondary Confirmation Check ONLY: Independent M_gauged ceiling for Factor 2
  d_mg_rp = df_exp_all['delta_mgauged_confirm_rp_clipped']
  interventions['Confirmation Check ONLY (M_gauged): Independent In-Sample Local Gauging'] = {
      'rp_clipped_delta_global': float(d_mg_rp.sum()),
      'rp_clipped_delta_with_sys': float(d_mg_rp.loc[df_exp_all['has_system']].sum()),
      'rp_clipped_delta_no_sys': float(d_mg_rp.loc[~df_exp_all['has_system']].sum()),
  }

  print('\n=== BASELINE GLOBAL FLOOD EXPOSURE ACCOUNTING (M_ungauged, people/yr) ===', flush=True)
  baseline_summary = {}
  for label, mask in [
      ('Global Total (All 964,084 Sub-Basins)', pd.Series(True, index=df_exp_all.index)),
      ('Mapped Countries Total', df_exp_all['country_name'].notna()),
      ('Countries WITH National Warning Capacity', df_exp_all['has_system'] & df_exp_all['country_name'].notna()),
      ('Countries WITHOUT Effective Warning Systems', (~df_exp_all['has_system']) & df_exp_all['country_name'].notna()),
  ]:
    sub = df_exp_all.loc[mask]
    baseline_summary[label] = {
        'n_basins': int(len(sub)),
        'E_total': float(sub['E_total'].sum()),
        'warned_ungauged': float(sub['warned_ungauged'].sum()),
        'unwarned_ungauged': float(sub['unwarned_ungauged'].sum()),
        'pct_warned': float(100.0 * sub['warned_ungauged'].sum() / sub['E_total'].sum()),
        'pct_unwarned': float(100.0 * sub['unwarned_ungauged'].sum() / sub['E_total'].sum()),
    }
    print(
        f'  {label:46s} | Total={sub["E_total"].sum():13,.0f} | '
        f'Warned={sub["warned_ungauged"].sum():12,.0f} ({baseline_summary[label]["pct_warned"]:4.1f}%) | '
        f'Unwarned={sub["unwarned_ungauged"].sum():13,.0f} ({baseline_summary[label]["pct_unwarned"]:4.1f}%)'
    )

  print('\n=== OBSERVATIONAL COUNTERFACTUAL CEILINGS (M_ungauged, people/year) ===', flush=True)
  for k, v in interventions.items():
    print(
        f'  {k:78s} | RP-Clip dE+={v["rp_clipped_delta_global"]:12,.0f} '
        f'(With Sys: {v["rp_clipped_delta_with_sys"]:11,.0f}, No Sys: {v["rp_clipped_delta_no_sys"]:10,.0f})'
    )

  # ============================================================================
  # 5. Unified Way B In-Manifold TreeSHAP-Targeted Analysis Across ALL Factors
  # ============================================================================
  print('\n5. Computing Parallel Interventional TreeSHAP & Way B In-Manifold Targeting Across ALL Factors...', flush=True)
  top_n = 8000
  top_ids = df_exp_pos_all.iloc[:top_n].index
  rest_df = df_exp_pos_all.iloc[top_n:].copy()
  rest_df['stratum'] = rest_df['country_name'].fillna('__UNMAPPED__')

  sampled_rest_parts = []
  weights = pd.Series(1.0, index=top_ids)
  rng = np.random.RandomState(42)

  for _, grp_s in rest_df.groupby('stratum', sort=True):
    if len(grp_s) <= 30:
      sampled_rest_parts.extend(grp_s.index.tolist())
      for cid in grp_s.index:
        weights.loc[cid] = 1.0
    else:
      top_c = grp_s.iloc[:15].index
      tail_df = grp_s.iloc[15:]
      chosen_tail = tail_df.sample(n=15, random_state=rng).index
      samp_sum = float(tail_df.loc[chosen_tail, 'E_total'].sum())
      w_val = float(tail_df['E_total'].sum() / samp_sum) if samp_sum > 0 else 1.0
      sampled_rest_parts.extend(top_c.tolist())
      sampled_rest_parts.extend(chosen_tail.tolist())
      for cid in top_c:
        weights.loc[cid] = 1.0
      for cid in chosen_tail:
        weights.loc[cid] = w_val

  shap_ids = top_ids.append(pd.Index(sampled_rest_parts))
  weights = weights.reindex(shap_ids)
  direct_cov_pct = 100.0 * df_exp_all.loc[shap_ids, 'E_total'].sum() / df_exp_all['E_total'].sum()
  weighted_tot_check = float((df_exp_all.loc[shap_ids, 'E_total'] * weights).sum())
  print(
      f'   Evaluating exact TreeSHAP on {len(shap_ids):,} exposed sub-basins '
      f'(top {top_n:,} + {len(sampled_rest_parts):,} country-stratified across all 160 exposed countries; '
      f'direct exposure share = {direct_cov_pct:.1f}%, post-stratified sum = {weighted_tot_check:,.0f} vs exact {df_exp_all["E_total"].sum():,.0f})!',
      flush=True,
  )

  X_shap_raw_u = df_hybas.loc[shap_ids].reindex(columns=x_train_u.columns, fill_value=0)
  X_bg_raw_u = x_train_u.sample(n=100, random_state=42)

  feat_cols = models_u['5.0-year-2.0-window-recall'].impl.feature_columns
  feat_to_group = {c: map_feature_to_5_factors(c)[1] for c in feat_cols}
  group_list_per_col = [feat_to_group[c] for c in feat_cols]

  shap_abs_by_feat = pd.Series(0.0, index=feat_cols)
  shap_def_by_feat = pd.Series(0.0, index=feat_cols)
  shap_def_with_sys = pd.Series(0.0, index=feat_cols)
  shap_def_no_sys = pd.Series(0.0, index=feat_cols)
  shap_net_by_feat = pd.Series(0.0, index=feat_cols)

  has_sys_vec = df_exp_all.loc[shap_ids, 'has_system'].values
  group_nets_by_rp = {}

  mp_ctx = multiprocessing.get_context('spawn')
  with concurrent.futures.ProcessPoolExecutor(
      max_workers=56, mp_context=mp_ctx
  ) as shap_pool:
    for m_name, e_col in [
        ('5.0-year-2.0-window-recall', 'E_5'),
        ('10.0-year-2.0-window-recall', 'E_10'),
        ('20.0-year-2.0-window-recall', 'E_20_50_100'),
    ]:
      tm = time.time()
      sp_impl = models_u[m_name].impl
      X_shap_trans = sp_impl._preprocess_features(X_shap_raw_u).reindex(
          columns=sp_impl.feature_columns, fill_value=0
      )
      X_bg_trans = sp_impl._preprocess_features(X_bg_raw_u).reindex(
          columns=sp_impl.feature_columns, fill_value=0
      )

      est_dict = dict(
          zip(
              [n for n, _ in sp_impl.ensemble.estimators],
              sp_impl.ensemble.estimators_,
          )
      )
      coef_map = dict(
          zip(
              [n for n, _ in sp_impl.ensemble.estimators],
              sp_impl.ensemble.final_estimator_.coef_,
          )
      )

      exp_h = shap.TreeExplainer(est_dict['hgbr'], X_bg_trans)
      pkl_h = f'/tmp/shap_exp_{os.getpid()}_{m_name}_hgbr.pkl'
      with open(pkl_h, 'wb') as fp_h:
        pickle.dump(exp_h, fp_h)
      chunks_h = np.array_split(X_shap_trans, 56)
      res_h = sorted(
          shap_pool.map(
              io_utils.eval_shap_chunk_from_pkl,
              [(pkl_h, idx, ch) for idx, ch in enumerate(chunks_h)],
          ),
          key=lambda x: x[0],
      )
      if os.path.exists(pkl_h):
        os.unlink(pkl_h)
      sv_hgbr = np.vstack([arr for _, arr in res_h])

      et_pipe = est_dict['et']
      et_imp = et_pipe.named_steps['simpleimputer']
      et_reg = et_pipe.named_steps['extratreesregressor']
      X_shap_imp = pd.DataFrame(
          et_imp.transform(X_shap_trans),
          index=X_shap_trans.index,
          columns=X_shap_trans.columns,
      )
      X_bg_imp = pd.DataFrame(
          et_imp.transform(X_bg_trans),
          index=X_bg_trans.index,
          columns=X_bg_trans.columns,
      )
      et_sub = copy.copy(et_reg)
      et_sub.estimators_ = et_reg.estimators_[:60]
      exp_e = shap.TreeExplainer(et_sub, X_bg_imp)
      pkl_e = f'/tmp/shap_exp_{os.getpid()}_{m_name}_et.pkl'
      with open(pkl_e, 'wb') as fp_e:
        pickle.dump(exp_e, fp_e)
      chunks_e = np.array_split(X_shap_imp, 56)
      res_e = sorted(
          shap_pool.map(
              io_utils.eval_shap_chunk_from_pkl,
              [(pkl_e, idx, ch) for idx, ch in enumerate(chunks_e)],
          ),
          key=lambda x: x[0],
      )
      if os.path.exists(pkl_e):
        os.unlink(pkl_e)
      sv_et = np.vstack([arr for _, arr in res_e])

      w_hgbr = max(0.0, float(coef_map.get('hgbr', 0.5)))
      w_et = max(0.0, float(coef_map.get('et', 0.5)))
      w_sum = w_hgbr + w_et if (w_hgbr + w_et) > 0 else 1.0
      sv_combined = (w_hgbr * sv_hgbr + w_et * sv_et) / w_sum

      sv_df = pd.DataFrame(
          sv_combined, index=shap_ids, columns=sp_impl.feature_columns
      )
      group_nets_by_rp[m_name] = sv_df.T.groupby(group_list_per_col).sum().T

      basin_exp_vec = (
          df_exp_all.loc[shap_ids, e_col] * weights
      ).values[:, None]
      def_mat = np.maximum(0.0, -sv_combined * basin_exp_vec)

      shap_abs_by_feat += pd.Series(
          np.abs(sv_combined * basin_exp_vec).sum(axis=0),
          index=sp_impl.feature_columns,
      )
      shap_def_by_feat += pd.Series(
          def_mat.sum(axis=0), index=sp_impl.feature_columns
      )
      shap_def_with_sys += pd.Series(
          def_mat[has_sys_vec].sum(axis=0), index=sp_impl.feature_columns
      )
      shap_def_no_sys += pd.Series(
          def_mat[~has_sys_vec].sum(axis=0), index=sp_impl.feature_columns
      )
      shap_net_by_feat += pd.Series(
          (sv_combined * basin_exp_vec).sum(axis=0),
          index=sp_impl.feature_columns,
      )
      print(
          f'   Finished 56-proc TreeSHAP for {m_name} in {time.time() - tm:.1f}s',
          flush=True,
      )

  shap_records = []
  for feat in shap_abs_by_feat.index:
    factor, subcat, remedy = map_feature_to_5_factors(feat)
    shap_records.append({
        'feature': feat,
        'factor': factor,
        'subcategory': subcat,
        'remedy': remedy,
        'abs_exposure_people_yr': float(shap_abs_by_feat[feat]),
        'deficit_exposure_people_yr': float(shap_def_by_feat[feat]),
        'deficit_with_sys_people_yr': float(shap_def_with_sys[feat]),
        'deficit_no_sys_people_yr': float(shap_def_no_sys[feat]),
        'net_exposure_people_yr': float(shap_net_by_feat[feat]),
    })
  df_shap_exp = pd.DataFrame(shap_records)
  df_shap_exp.to_csv(
      os.path.join(file_paths.STATS_DIR, 'five_factor_exposure_shap_features_v2.csv'),
      index=False,
  )

  # Evaluate unified Way B in-manifold metrics for every group + combined Factor 1 + combined Factors 2+3
  b5_s = df_exp_all.loc[shap_ids, 'recall_5y_u']
  b10_s = df_exp_all.loc[shap_ids, 'recall_10y_u']
  b20_s = df_exp_all.loc[shap_ids, 'recall_20y_u']
  e5_s = df_exp_all.loc[shap_ids, 'E_5']
  e10_s = df_exp_all.loc[shap_ids, 'E_10']
  e20_s = df_exp_all.loc[shap_ids, 'E_20_50_100']
  etot_s = df_exp_all.loc[shap_ids, 'E_total']
  country_s = df_exp_all.loc[shap_ids, 'country_name']
  sys_s = df_exp_all.loc[shap_ids, 'has_system']

  feat_def_by_subcat = df_shap_exp.groupby('subcategory')[
      ['deficit_exposure_people_yr', 'deficit_with_sys_people_yr', 'deficit_no_sys_people_yr']
  ].sum()

  def _eval_way_b_penalties(p5_pen, p10_pen, p20_pen, feat_def_row=None):
    """Computes parameter-free SHAP penalty elimination and common-scale nonlinear gamma curves."""
    def _agg(series_basin):
      w_ser = weights * series_basin
      return {
          'global': float(w_ser.sum()),
          'with_sys': float(w_ser.loc[sys_s].sum()),
          'no_sys': float(w_ser.loc[~sys_s].sum()),
      }, w_ser

    parity_b = (
        e5_s * ((b5_s + p5_pen).clip(0.0, 1.0) - b5_s)
        + e10_s * ((b10_s + p10_pen).clip(0.0, 1.0) - b10_s)
        + e20_s * ((b20_s + p20_pen).clip(0.0, 1.0) - b20_s)
    )
    pen50_b = (
        e5_s * ((b5_s + 0.50 * p5_pen).clip(0.0, 1.0) - b5_s)
        + e10_s * ((b10_s + 0.50 * p10_pen).clip(0.0, 1.0) - b10_s)
        + e20_s * ((b20_s + 0.50 * p20_pen).clip(0.0, 1.0) - b20_s)
    )
    parity_tot, w_parity = _agg(parity_b)
    pen50_tot, _ = _agg(pen50_b)

    w5 = (p5_pen / P_STAR_COMMON).clip(0.0, 1.0)
    w10 = (p10_pen / P_STAR_COMMON).clip(0.0, 1.0)
    w20 = (p20_pen / P_STAR_COMMON).clip(0.0, 1.0)

    nl_out = {}
    w_nl_series = {}
    for gamma_val, g_key in [(0.10, 'gamma_10pct'), (0.25, 'gamma_25pct'), (0.50, 'gamma_50pct')]:
      r5_new = 1.0 - (1.0 - b5_s) * np.power(1.0 - gamma_val, w5)
      r10_new = 1.0 - (1.0 - b10_s) * np.power(1.0 - gamma_val, w10)
      r20_new = 1.0 - (1.0 - b20_s) * np.power(1.0 - gamma_val, w20)
      nl_b = e5_s * (r5_new - b5_s) + e10_s * (r10_new - b10_s) + e20_s * (r20_new - b20_s)
      nl_tot, w_nl = _agg(nl_b)
      nl_out[g_key] = nl_tot
      w_nl_series[g_key] = w_nl

    exp_pen_share = float(
        100.0 * (weights * etot_s).loc[p5_pen > 0.005].sum() / (weights * etot_s).sum()
    )
    q90_p5 = float(p5_pen[p5_pen > 0].quantile(0.90)) if (p5_pen > 0).any() else 0.0

    rec = {
        'parity_100pct': parity_tot,
        'penalty_50pct': pen50_tot,
        'nonlinear_gamma_10pct': nl_out['gamma_10pct'],
        'nonlinear_gamma_25pct': nl_out['gamma_25pct'],
        'nonlinear_gamma_50pct': nl_out['gamma_50pct'],
        'pct_exposure_penalized_gt_0p5pp': exp_pen_share,
        'q90_penalty_5y': q90_p5,
    }
    if feat_def_row is not None:
      rec['feature_sum_shap_deficit'] = {
          'global': float(feat_def_row['deficit_exposure_people_yr']),
          'with_sys': float(feat_def_row['deficit_with_sys_people_yr']),
          'no_sys': float(feat_def_row['deficit_no_sys_people_yr']),
      }
    return rec, w_parity, w_nl_series['gamma_25pct']

  way_b_results = {}
  way_b_csv_rows = []
  country_wayb_cols = {}

  ordered_groups = [
      'Factor 3: Out-of-Distribution Representation Bias',
      '1f: Sparse Canopy, Agricultural & Biome Land Cover',
      'Factor 2: River-Network Gauge Coverage',
      '1a: Arid & Convective Hydroclimate',
      '1c: Floodplain, Wetland & Lake Storage',
      '1d: Steep & Small-Scale Channel Routing',
      '1b: Dam, Reservoir & Irrigation Regulation',
      '1g: Urban & Human Footprint',
      '1e: Subsurface Soil & Karst Hydrogeology',
      'Factor 4: Historical Record Length',
  ]

  print(f'\n=== UNIFIED WAY B IN-MANIFOLD TARGETED EXPOSURE REDUCTION (P_star = {P_STAR_COMMON:.2f}, people/yr) ===', flush=True)
  for g_name in ordered_groups:
    p5_g = (-group_nets_by_rp['5.0-year-2.0-window-recall'][g_name]).clip(lower=0.0)
    p10_g = (-group_nets_by_rp['10.0-year-2.0-window-recall'][g_name]).clip(lower=0.0)
    p20_g = (-group_nets_by_rp['20.0-year-2.0-window-recall'][g_name]).clip(lower=0.0)
    f_row = feat_def_by_subcat.loc[g_name] if g_name in feat_def_by_subcat.index else None
    rec_g, w_par, w_g25 = _eval_way_b_penalties(p5_g, p10_g, p20_g, feat_def_row=f_row)
    way_b_results[g_name] = rec_g

    slug = GROUP_SHORT_SLUGS[g_name]
    country_wayb_cols[f'wayb_parity_{slug}'] = w_par.groupby(country_s).sum()
    country_wayb_cols[f'wayb_g25_{slug}'] = w_g25.groupby(country_s).sum()

    way_b_csv_rows.append({
        'group': g_name,
        'slug': slug,
        'feature_sum_shap_deficit_global': rec_g['feature_sum_shap_deficit']['global'] if f_row is not None else np.nan,
        'parity_100pct_global': rec_g['parity_100pct']['global'],
        'parity_100pct_with_sys': rec_g['parity_100pct']['with_sys'],
        'parity_100pct_no_sys': rec_g['parity_100pct']['no_sys'],
        'nonlinear_g10_global': rec_g['nonlinear_gamma_10pct']['global'],
        'nonlinear_g25_global': rec_g['nonlinear_gamma_25pct']['global'],
        'nonlinear_g25_with_sys': rec_g['nonlinear_gamma_25pct']['with_sys'],
        'nonlinear_g25_no_sys': rec_g['nonlinear_gamma_25pct']['no_sys'],
        'nonlinear_g50_global': rec_g['nonlinear_gamma_50pct']['global'],
        'nonlinear_g50_with_sys': rec_g['nonlinear_gamma_50pct']['with_sys'],
        'nonlinear_g50_no_sys': rec_g['nonlinear_gamma_50pct']['no_sys'],
        'pct_exposure_penalized_gt_0p5pp': rec_g['pct_exposure_penalized_gt_0p5pp'],
        'q90_penalty_5y': rec_g['q90_penalty_5y'],
    })
    print(
        f"  {g_name:50s} | Parity:+{rec_g['parity_100pct']['global']:10,.0f} "
        f"| g=10%:+{rec_g['nonlinear_gamma_10pct']['global']:9,.0f} "
        f"| g=25%:+{rec_g['nonlinear_gamma_25pct']['global']:10,.0f} "
        f"| g=50%:+{rec_g['nonlinear_gamma_50pct']['global']:10,.0f} "
        f"| ExpPen(>0.5pp)={rec_g['pct_exposure_penalized_gt_0p5pp']:4.1f}%"
    )

  # Combined Factor 1 (all physical & anthropogenic catchment attributes 1a-1g together)
  f1_groups = [g for g in ordered_groups if g[:2] in {'1a', '1b', '1c', '1d', '1e', '1f', '1g'}]
  p5_f1 = (-group_nets_by_rp['5.0-year-2.0-window-recall'][f1_groups].sum(axis=1)).clip(lower=0.0)
  p10_f1 = (-group_nets_by_rp['10.0-year-2.0-window-recall'][f1_groups].sum(axis=1)).clip(lower=0.0)
  p20_f1 = (-group_nets_by_rp['20.0-year-2.0-window-recall'][f1_groups].sum(axis=1)).clip(lower=0.0)
  rec_f1, _, _ = _eval_way_b_penalties(p5_f1, p10_f1, p20_f1, feat_def_row=feat_def_by_subcat.loc[f1_groups].sum())
  way_b_results['Combined Factor 1: All Physical & Anthropogenic Catchment Attributes (1a-1g Net)'] = rec_f1
  print(
      f"  {'Combined Factor 1 (1a-1g Net)':50s} | Parity:+{rec_f1['parity_100pct']['global']:10,.0f} "
      f"| g=10%:+{rec_f1['nonlinear_gamma_10pct']['global']:9,.0f} "
      f"| g=25%:+{rec_f1['nonlinear_gamma_25pct']['global']:10,.0f} "
      f"| g=50%:+{rec_f1['nonlinear_gamma_50pct']['global']:10,.0f}"
  )

  # Combined Factors 2 + 3 (Gauge Coverage + OOD Representation Net)
  obs_groups = [
      'Factor 2: River-Network Gauge Coverage',
      'Factor 3: Out-of-Distribution Representation Bias',
  ]
  p5_obs = (-group_nets_by_rp['5.0-year-2.0-window-recall'][obs_groups].sum(axis=1)).clip(lower=0.0)
  p10_obs = (-group_nets_by_rp['10.0-year-2.0-window-recall'][obs_groups].sum(axis=1)).clip(lower=0.0)
  p20_obs = (-group_nets_by_rp['20.0-year-2.0-window-recall'][obs_groups].sum(axis=1)).clip(lower=0.0)
  rec_obs, w_par_obs, w_g25_obs = _eval_way_b_penalties(
      p5_obs, p10_obs, p20_obs, feat_def_row=feat_def_by_subcat.loc[obs_groups].sum()
  )
  way_b_results['Combined Observational Factors (Factors 2 + 3 Net)'] = rec_obs
  country_wayb_cols['wayb_parity_f2_f3_combined'] = w_par_obs.groupby(country_s).sum()
  country_wayb_cols['wayb_g25_f2_f3_combined'] = w_g25_obs.groupby(country_s).sum()
  print(
      f"  {'Combined Factors 2 + 3 (Gauge + OOD Net)':50s} | Parity:+{rec_obs['parity_100pct']['global']:10,.0f} "
      f"| g=10%:+{rec_obs['nonlinear_gamma_10pct']['global']:9,.0f} "
      f"| g=25%:+{rec_obs['nonlinear_gamma_25pct']['global']:10,.0f} "
      f"| g=50%:+{rec_obs['nonlinear_gamma_50pct']['global']:10,.0f}"
  )

  pd.DataFrame(way_b_csv_rows).to_csv(
      os.path.join(file_paths.STATS_DIR, 'five_factor_way_b_in_manifold_v2.csv'),
      index=False,
  )

  # 6. Country-level multi-lever aggregation across all 173 mapped countries
  df_c = (
      df_exp_all.dropna(subset=['country_name'])
      .groupby('country_name')
      .agg(
          n_basins=('E_total', 'count'),
          n_basins_zero_cov=('zero_open_gauge_cov', 'sum'),
          has_system=('has_system', 'first'),
          cap_tier=('cap_tier', 'first'),
          E_total=('E_total', 'sum'),
          warned_ungauged=('warned_ungauged', 'sum'),
          unwarned_ungauged=('unwarned_ungauged', 'sum'),
          median_recall_5y=('recall_5y_u', 'median'),
          median_recall_10y=('recall_10y_u', 'median'),
          median_recall_20y=('recall_20y_u', 'median'),
          cf_factor3_ood=('cf_factor3_ood', 'sum'),
          cf_factor2_gauge_cov=('cf_factor2_gauge_cov', 'sum'),
          cf_factor2_dense_corridor=('cf_factor2_dense_corridor', 'sum'),
          cf_factor4_reclen=('cf_factor4_reclen', 'sum'),
          cf_obs_ceiling=('cf_obs_ceiling', 'sum'),
          delta_mgauged_confirm=('delta_mgauged_confirm_rp_clipped', 'sum'),
      )
      .sort_values('unwarned_ungauged', ascending=False)
  )
  df_c['pct_unwarned'] = np.where(df_c['E_total'] > 0, 100.0 * df_c['unwarned_ungauged'] / df_c['E_total'], 0.0)
  for col_k, ser_k in country_wayb_cols.items():
    df_c[col_k] = ser_k.reindex(df_c.index).fillna(0.0)
  df_c.to_csv(os.path.join(file_paths.STATS_DIR, 'country_exposure_and_blockers_v2.csv'))

  # Save basin-level exposure & observational blocker columns for fast map rendering in Script 9
  basin_cols = [
      'country_name', 'has_system', 'zero_open_gauge_cov',
      'E_total', 'warned_ungauged', 'unwarned_ungauged',
      'recall_5y_u', 'recall_10y_u', 'recall_20y_u',
      'cf_factor3_ood', 'cf_factor2_gauge_cov', 'cf_factor2_dense_corridor',
      'cf_factor4_reclen', 'cf_obs_ceiling', 'delta_mgauged_confirm_rp_clipped',
  ]
  df_exp_all[basin_cols].to_pickle(os.path.join(file_paths.STATS_DIR, 'basin_exposure_and_blockers_v2.pkl'))
  try:
    df_exp_all[basin_cols].to_parquet(os.path.join(file_paths.STATS_DIR, 'basin_exposure_and_blockers_v2.parquet'))
  except Exception:  # pylint: disable=broad-except
    pass

  with open(os.path.join(file_paths.STATS_DIR, 'five_factor_exposure_interventions_v2.json'), 'w') as f:
    json.dump(
        {
            'cv_r2_ungauged': cv_r2_u,
            'cv_r2_gauged_confirm': cv_r2_g,
            'zero_open_gauge_coverage_summary': zero_cov_summary,
            'baseline_summary': baseline_summary,
            'interventions': interventions,
            'way_b_p_star_common': P_STAR_COMMON,
            'way_b_sample_size': int(len(shap_ids)),
            'way_b_direct_exposure_coverage_pct': float(direct_cov_pct),
            'way_b_in_manifold': way_b_results,
        },
        f,
        indent=2,
    )

  print('\n=== FEATURE-LEVEL SHAP EXPOSURE DEFICIT BY FACTOR & SUBCATEGORY (people/year) ===', flush=True)
  subcat_summary = df_shap_exp.groupby(['factor', 'subcategory', 'remedy'])[
      ['deficit_exposure_people_yr', 'deficit_with_sys_people_yr', 'deficit_no_sys_people_yr']
  ].sum()
  subcat_summary['pct_deficit'] = (
      100.0 * subcat_summary['deficit_exposure_people_yr'] / subcat_summary['deficit_exposure_people_yr'].sum()
  )
  print(subcat_summary.to_string())
  print(f'\nCompleted unified v2 5-factor exposure decomposition in {time.time() - t0:.1f}s!', flush=True)


if __name__ == '__main__':
  app.run(main)
