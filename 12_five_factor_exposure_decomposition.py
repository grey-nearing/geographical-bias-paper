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

Evaluates baseline warned and unwarned riverine flood exposure, counterfactual
exposure reductions across all five predictability factors, country-level
multi-lever exposure accounting, and exact TreeSHAP exposure-deficit
attributions directly on the real-world (`ungauged`) skill predictor
(`M_ungauged`). The independent temporal-split `M_gauged` predictor is loaded
solely as a secondary confirmation cross-check on Factor 2 (gauge network
density).
"""

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
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import r2_score

from utils import cap_utils
from utils import feature_utils
from utils import file_paths
from utils import io_utils
from utils import skill_predictor as sp

sys.modules['utils.skill_predictor'] = sp

warnings.filterwarnings('ignore')

# Trapezoidal exceedance-frequency quadrature weights across [5, 10, 20, 50, 100]-yr floods
# matching stats_utils.calculate_trapezoidal_expected_annual_risk (np.trapezoid over p = 1/T in [0.01, 0.20])
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


def map_feature_to_5_factors(feat_name: str) -> tuple[str, str, str]:
  """Maps any raw or engineered feature to (Factor, Subcategory, Primary Remedy)."""
  if feat_name in [
      'total_upstream_coverage',
      'max_upstream_coverage',
      'max_downstream_coverage',
      'upstream_gauge_count',
  ]:
    return (
        'Factor 2: River-Network Gauge Coverage',
        'Topological Gauge Coverage',
        'Targeted In Situ Gauging & Assimilation',
    )

  if feat_name in [
      'anomaly_score_if',
      'anomaly_score_ocsvm',
      'anomaly_score_lof',
  ]:
    return (
        'Factor 3: Out-of-Distribution Representation Bias',
        'Multivariate OOD Anomaly Scores',
        'Under-Sampled Gauging + Training Re-Weighting / Local Fine-Tuning',
    )

  if feat_name == 'record_length':
    return (
        'Factor 4: Historical Record Length',
        'Record Length',
        'Archive Extension',
    )

  subcat = feature_utils.FEATURE_CLASSIFICATION.get(feat_name, None)
  if subcat is None:
    if feat_name in [
        'water_balance',
        'aridity_ratio',
        'p_minus_pet',
        'p_div_pet',
        'et_deficit',
        'precip_temp_inter',
        'forest_aridity_inter',
    ]:
      subcat = 'Climate & Hydrology'
    elif feat_name in ['total_forest_pc', 'total_forest_cover', 'total_arid_cover']:
      subcat = 'Land Cover'
    elif feat_name in [
        'human_impact_density',
        'human_intensity',
        'log_pop_ct_usu',
        'log_rev_mc_usu',
        'log_ria_ha_usu',
    ]:
      subcat = 'Anthropogenic Factors'
    elif feat_name in ['twi_proxy', 'log_calculated_drain_area', 'log_riv_tc_usu']:
      subcat = 'Topography'
    elif feat_name in ['soil_porosity']:
      subcat = 'Soil and Geology'
    else:
      subcat = 'Other Physical'

  remedy_map = {
      'Land Cover': 'Model Architecture & Assimilation (Floodplain/Lake Routing & Satellite Water)',
      'Climate & Hydrology': 'Dynamic Input Data (High-Resolution Convective/Satellite Precipitation)',
      'Anthropogenic Factors': 'Dynamic Input Data (Reservoir Releases, SWOT Storage & Irrigation)',
      'Topography': 'Model Architecture (Sub-Daily & High-Resolution Channel Routing)',
      'Soil and Geology': 'Subsurface Representation & Soil-Moisture Assimilation',
  }
  remedy = remedy_map.get(subcat, 'Model & Input Improvements')
  return ('Factor 1: Hydrological & Physical Process Difficulty', subcat, remedy)


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
      models_u[m] = pickle.load(f)
    with open(os.path.join(file_paths.MODELS_DIR, f'final_gauged_model_{m}.pkl'), 'rb') as f:
      models_g[m] = pickle.load(f)
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

  print('\n4. Loading HyBAS Zarr features & M_ungauged training data (ZERO M_gauged dependency for benchmarks)...', flush=True)
  df_hybas = io_utils.load_hybas_zarr(file_paths.HYBAS_GENERIC_DATA_ZARR_PATH)
  df_hybas.index = (
      df_hybas.index.astype(str)
      .str.replace(r'\.0$', '', regex=True)
      .map(lambda x: x if x.startswith('hybas_') else f'hybas_{x}')
  )

  x_train_u, y_train_u_5 = io_utils.load_and_preprocess_data(
      'ungauged', '5.0-year-2.0-window-recall', None
  )

  # Top-quartile reference benchmark computed PURELY from M_ungauged training distribution!
  top_q_thresh_u = y_train_u_5.quantile(0.75)
  top_q_basins_u = x_train_u.loc[y_train_u_5 >= top_q_thresh_u]
  ref_medians_u = top_q_basins_u.median()

  common_ids = df_exp.index.intersection(df_hybas.index)
  df_exp_all = df_exp.loc[common_ids].copy()
  df_exp_pos_all = df_exp_all[df_exp_all['E_total'] > 0].sort_values('E_total', ascending=False)

  print(
      f'   Evaluating exact M_ungauged counterfactuals across ALL {len(df_exp_pos_all):,} exposed HyBAS sub-basins '
      f'(100.0% of {df_exp_all["E_total"].sum():,.0f} people/yr exposure)!',
      flush=True,
  )
  cf_ids = df_exp_pos_all.index
  df_hybas_cf = df_hybas.loc[cf_ids]
  df_exp_cf = df_exp_all.loc[cf_ids].copy()

  def eval_counterfactual_u(df_mod, store_col=None):
    """Evaluates M_ungauged counterfactual across ALL exposed basins and optionally stores per-basin gain."""
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

  # Factor 2 (M_ungauged): Full Local + River-Network Topological Coverage (100% coverage, >=1 gauge)
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
  cov_cols = ['total_upstream_coverage', 'max_upstream_coverage', 'max_downstream_coverage', 'upstream_gauge_count']
  df_mod_cov_dense = df_hybas_cf.copy()
  for c in cov_cols:
    if c in df_mod_cov_dense.columns and c in x_train_u.columns:
      q75 = x_train_u[x_train_u[c] > 0][c].quantile(0.75) if (x_train_u[c] > 0).any() else 1.0
      df_mod_cov_dense[c] = np.maximum(df_mod_cov_dense[c], q75)
  interventions['Factor 2 (M_ungauged Dense Corridor): Multi-Gauge Nested Corridor Coverage (75th pct)'] = eval_counterfactual_u(
      df_mod_cov_dense, store_col='cf_factor2_dense_corridor'
  )

  # Factor 3 (M_ungauged): Eliminating Out-of-Distribution Representation Bias
  ood_cols = ['anomaly_score_if', 'anomaly_score_ocsvm', 'anomaly_score_lof']
  df_mod_ood = df_hybas_cf.copy()
  for c in ood_cols:
    if c in df_mod_ood.columns and c in ref_medians_u.index:
      df_mod_ood[c] = np.maximum(df_mod_ood[c], ref_medians_u[c]) if c != 'anomaly_score_lof' else ref_medians_u[c]
  interventions['Factor 3 (M_ungauged): Eliminating Out-of-Distribution Representation Bias'] = eval_counterfactual_u(
      df_mod_ood, store_col='cf_factor3_ood'
  )

  # Combined Observational Ceiling (Factor 2 + Factor 3 together in M_ungauged) -> defines Factor 5 Residual Ceiling!
  df_mod_obs_ceil = df_mod_ood.copy()
  for c in ['total_upstream_coverage', 'max_upstream_coverage', 'max_downstream_coverage']:
    if c in df_mod_obs_ceil.columns:
      df_mod_obs_ceil[c] = 1.0
  if 'upstream_gauge_count' in df_mod_obs_ceil.columns:
    df_mod_obs_ceil['upstream_gauge_count'] = np.maximum(df_mod_obs_ceil['upstream_gauge_count'], 1.0)
  interventions['Combined Observational Ceiling (Factor 2 + Factor 3 in M_ungauged)'] = eval_counterfactual_u(
      df_mod_obs_ceil, store_col='cf_obs_ceiling'
  )

  # Factor 4 (M_ungauged!): Extending Historical Record Length to 50 Years directly in M_ungauged
  df_mod_rec_u = df_hybas_cf.copy()
  df_mod_rec_u['record_length'] = 50.0
  interventions['Factor 4 (M_ungauged): Extending Historical Record Length to 50 Years'] = eval_counterfactual_u(
      df_mod_rec_u, store_col='cf_factor4_reclen'
  )

  # Factor 1a: Climate & Hydrology (Aridity / Water Balance -> Favorable Humid Reference Regime from M_ungauged)
  clim_cols = [c for c, cat in feature_utils.FEATURE_CLASSIFICATION.items() if cat == 'Climate & Hydrology']
  df_mod_clim = df_hybas_cf.copy()
  for c in clim_cols:
    if c in df_mod_clim.columns and c in ref_medians_u.index:
      df_mod_clim[c] = ref_medians_u[c]
  interventions['Factor 1a (M_ungauged): Climate & Aridity Regime -> Humid Reference'] = eval_counterfactual_u(
      df_mod_clim, store_col='cf_factor1a_climate'
  )

  # Factor 1b: Anthropogenic Regulation (Zero Reservoirs/Dams/Irrigation)
  df_mod_no_dams = df_hybas_cf.copy()
  for c in ['rev_mc_usu', 'ria_ha_usu', 'dor_pc_pva', 'ire_pc_use', 'dam_ct_usu']:
    if c in df_mod_no_dams.columns:
      df_mod_no_dams[c] = 0.0
  interventions['Factor 1b (M_ungauged): Removing Anthropogenic Reservoir & Irrigation Regulation'] = eval_counterfactual_u(
      df_mod_no_dams, store_col='cf_factor1b_regulation'
  )

  # Factor 1c: Land Cover, Floodplains, Wetlands & Lakes
  lc_cols = [c for c, cat in feature_utils.FEATURE_CLASSIFICATION.items() if cat == 'Land Cover']
  df_mod_lc = df_hybas_cf.copy()
  for c in lc_cols:
    if c in df_mod_lc.columns and c in ref_medians_u.index:
      df_mod_lc[c] = ref_medians_u[c]
  interventions['Factor 1c (M_ungauged): Land Cover, Floodplains, Wetlands & Lakes -> Reference'] = eval_counterfactual_u(
      df_mod_lc, store_col='cf_factor1c_landcover'
  )

  # Factor 1d: Topography & Drainage Scale
  topo_cols = [c for c, cat in feature_utils.FEATURE_CLASSIFICATION.items() if cat == 'Topography']
  df_mod_topo = df_hybas_cf.copy()
  for c in topo_cols:
    if c in df_mod_topo.columns and c in ref_medians_u.index:
      df_mod_topo[c] = ref_medians_u[c]
  interventions['Factor 1d (M_ungauged): Topography & Drainage Scale -> Reference'] = eval_counterfactual_u(
      df_mod_topo, store_col='cf_factor1d_topography'
  )

  # Factor 1e: Soil & Hydrogeology
  soil_cols = [c for c, cat in feature_utils.FEATURE_CLASSIFICATION.items() if cat == 'Soil and Geology']
  df_mod_soil = df_hybas_cf.copy()
  for c in soil_cols:
    if c in df_mod_soil.columns and c in ref_medians_u.index:
      df_mod_soil[c] = ref_medians_u[c]
  interventions['Factor 1e (M_ungauged): Soil & Hydrogeology -> Reference'] = eval_counterfactual_u(
      df_mod_soil, store_col='cf_factor1e_soil'
  )

  # Full Catchment Ceiling (Factors 1 + 2 + 3 + 4 all together in M_ungauged)
  anthro_cols = [c for c, cat in feature_utils.FEATURE_CLASSIFICATION.items() if cat == 'Anthropogenic Factors']
  phys_cols = clim_cols + anthro_cols + lc_cols + topo_cols + soil_cols
  df_mod_all = df_mod_obs_ceil.copy()
  for c in phys_cols:
    if c in df_mod_all.columns and c in ref_medians_u.index:
      df_mod_all[c] = ref_medians_u[c]
  df_mod_all['record_length'] = 50.0
  interventions['Full Catchment Ceiling (Factors 1+2+3+4 in M_ungauged)'] = eval_counterfactual_u(
      df_mod_all, store_col='cf_all_factors_ceiling'
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

  print('\n=== COUNTERFACTUAL EXPOSURE REDUCTION BY FACTOR (100% M_ungauged, people/year) ===', flush=True)
  for k, v in interventions.items():
    print(
        f'  {k:74s} | RP-Clip dE+={v["rp_clipped_delta_global"]:12,.0f} '
        f'(With Sys: {v["rp_clipped_delta_with_sys"]:11,.0f}, No Sys: {v["rp_clipped_delta_no_sys"]:10,.0f})'
    )

  # 5. Country-level multi-lever aggregation across all mapped countries
  df_c = (
      df_exp_all.dropna(subset=['country_name'])
      .groupby('country_name')
      .agg(
          n_basins=('E_total', 'count'),
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
          cf_factor1a_climate=('cf_factor1a_climate', 'sum'),
          cf_factor1b_regulation=('cf_factor1b_regulation', 'sum'),
          cf_factor1c_landcover=('cf_factor1c_landcover', 'sum'),
          cf_factor1d_topography=('cf_factor1d_topography', 'sum'),
          cf_factor1e_soil=('cf_factor1e_soil', 'sum'),
          cf_factor4_reclen=('cf_factor4_reclen', 'sum'),
          cf_obs_ceiling=('cf_obs_ceiling', 'sum'),
          delta_mgauged_confirm=('delta_mgauged_confirm_rp_clipped', 'sum'),
      )
      .sort_values('unwarned_ungauged', ascending=False)
  )
  df_c['pct_unwarned'] = np.where(df_c['E_total'] > 0, 100.0 * df_c['unwarned_ungauged'] / df_c['E_total'], 0.0)
  df_c.to_csv(os.path.join(file_paths.STATS_DIR, 'country_exposure_and_blockers_v2.csv'))

  # Also save basin-level exposure & blocker columns for fast map rendering in Script 9!
  basin_cols = [
      'country_name', 'has_system', 'E_total', 'warned_ungauged', 'unwarned_ungauged',
      'recall_5y_u', 'recall_10y_u', 'recall_20y_u',
      'cf_factor3_ood', 'cf_factor2_gauge_cov', 'cf_factor2_dense_corridor',
      'cf_factor1a_climate', 'cf_factor1b_regulation', 'cf_factor1c_landcover',
      'cf_obs_ceiling', 'delta_mgauged_confirm_rp_clipped',
  ]
  df_exp_all[basin_cols].to_pickle(os.path.join(file_paths.STATS_DIR, 'basin_exposure_and_blockers_v2.pkl'))
  try:
    df_exp_all[basin_cols].to_parquet(os.path.join(file_paths.STATS_DIR, 'basin_exposure_and_blockers_v2.parquet'))
  except Exception:  # pylint: disable=broad-except
    pass

  # 6. Exact TreeSHAP Exposure Attribution (in people/year) on Exposure-Weighted Global Sample (M_ungauged)
  print('\n5. Computing Exact TreeSHAP Exposure Attribution (people/year) on M_ungauged...', flush=True)
  top_n = 3000
  sample_n = 2000
  top_ids = df_exp_pos_all.iloc[:top_n].index
  rest_df = df_exp_pos_all.iloc[top_n:]
  rest_sample_ids = rest_df.sample(n=min(sample_n, len(rest_df)), random_state=42).index
  shap_ids = top_ids.append(rest_sample_ids)

  weights = pd.Series(1.0, index=shap_ids)
  for sys_flag in [True, False]:
    rest_sys = rest_df[rest_df['has_system'] == sys_flag]
    samp_sys = rest_sample_ids[rest_df.loc[rest_sample_ids, 'has_system'] == sys_flag]
    if len(samp_sys) > 0 and rest_sys.loc[samp_sys, 'E_total'].sum() > 0:
      weights.loc[samp_sys] = rest_sys['E_total'].sum() / rest_sys.loc[samp_sys, 'E_total'].sum()

  X_shap_raw_u = df_hybas.loc[shap_ids].reindex(columns=x_train_u.columns, fill_value=0)
  X_bg_raw_u = x_train_u.sample(n=200, random_state=42)

  feat_cols = models_u['5.0-year-2.0-window-recall'].impl.feature_columns
  shap_abs_by_feat = pd.Series(0.0, index=feat_cols)
  shap_def_by_feat = pd.Series(0.0, index=feat_cols)
  shap_def_with_sys = pd.Series(0.0, index=feat_cols)
  shap_def_no_sys = pd.Series(0.0, index=feat_cols)
  shap_net_by_feat = pd.Series(0.0, index=feat_cols)

  has_sys_vec = df_exp_all.loc[shap_ids, 'has_system'].values

  for m_name, e_col in [
      ('5.0-year-2.0-window-recall', 'E_5'),
      ('10.0-year-2.0-window-recall', 'E_10'),
      ('20.0-year-2.0-window-recall', 'E_20_50_100'),
  ]:
    sp_impl = models_u[m_name].impl
    X_shap_trans = sp_impl._preprocess_features(X_shap_raw_u).reindex(columns=sp_impl.feature_columns, fill_value=0)
    X_bg_trans = sp_impl._preprocess_features(X_bg_raw_u).reindex(columns=sp_impl.feature_columns, fill_value=0)

    est_dict = dict(zip([n for n, _ in sp_impl.ensemble.estimators], sp_impl.ensemble.estimators_))
    coef_map = dict(zip([n for n, _ in sp_impl.ensemble.estimators], sp_impl.ensemble.final_estimator_.coef_))

    sv_hgbr = shap.TreeExplainer(est_dict['hgbr'], X_bg_trans).shap_values(X_shap_trans, check_additivity=False)

    et_pipe = est_dict['et']
    et_imp = et_pipe.named_steps['simpleimputer']
    et_reg = et_pipe.named_steps['extratreesregressor']
    X_shap_imp = pd.DataFrame(et_imp.transform(X_shap_trans), index=X_shap_trans.index, columns=X_shap_trans.columns)
    X_bg_imp = pd.DataFrame(et_imp.transform(X_bg_trans), index=X_bg_trans.index, columns=X_bg_trans.columns)
    et_sub = copy.copy(et_reg)
    et_sub.estimators_ = et_reg.estimators_[:60]
    sv_et = shap.TreeExplainer(et_sub, X_bg_imp).shap_values(X_shap_imp, check_additivity=False)

    w_hgbr = coef_map.get('hgbr', 0.5)
    w_et = coef_map.get('et', 0.5)
    w_sum = w_hgbr + w_et if (w_hgbr + w_et) > 0 else 1.0
    sv_combined = (w_hgbr * sv_hgbr + w_et * sv_et) / w_sum

    basin_exp_vec = (df_exp_all.loc[shap_ids, e_col] * weights).values[:, None]
    def_mat = np.maximum(0.0, -sv_combined * basin_exp_vec)

    shap_abs_by_feat += pd.Series(np.abs(sv_combined * basin_exp_vec).sum(axis=0), index=sp_impl.feature_columns)
    shap_def_by_feat += pd.Series(def_mat.sum(axis=0), index=sp_impl.feature_columns)
    shap_def_with_sys += pd.Series(def_mat[has_sys_vec].sum(axis=0), index=sp_impl.feature_columns)
    shap_def_no_sys += pd.Series(def_mat[~has_sys_vec].sum(axis=0), index=sp_impl.feature_columns)
    shap_net_by_feat += pd.Series((sv_combined * basin_exp_vec).sum(axis=0), index=sp_impl.feature_columns)

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

  with open(os.path.join(file_paths.STATS_DIR, 'five_factor_exposure_interventions_v2.json'), 'w') as f:
    json.dump(
        {
            'cv_r2_ungauged': cv_r2_u,
            'cv_r2_gauged_confirm': cv_r2_g,
            'baseline_summary': baseline_summary,
            'interventions': interventions,
        },
        f,
        indent=2,
    )

  print('\n=== SHAP EXPOSURE DEFICIT BY FACTOR & REMEDY (people/year) ===', flush=True)
  subcat_summary = df_shap_exp.groupby(['factor', 'subcategory', 'remedy'])[['deficit_exposure_people_yr', 'deficit_with_sys_people_yr', 'deficit_no_sys_people_yr']].sum()
  subcat_summary['pct_deficit'] = 100.0 * subcat_summary['deficit_exposure_people_yr'] / subcat_summary['deficit_exposure_people_yr'].sum()
  print(subcat_summary.to_string())
  print(f'\nCompleted unified v2 5-factor exposure decomposition in {time.time() - t0:.1f}s!', flush=True)


if __name__ == '__main__':
  app.run(main)
