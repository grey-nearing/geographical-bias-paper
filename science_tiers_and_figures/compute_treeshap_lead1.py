"""Compute Lead-Time-1 TreeSHAP feature attributions across all 12 metrics on N=1,000 representative global watersheds, and render Fig. 5 and Fig. S6."""

from concurrent.futures import ProcessPoolExecutor
import glob
import json
import os
import pickle
import sys
import time
import types
import warnings

os.environ['GEO_BIAS_DATA_DIR'] = './data'
os.environ['GEO_BIAS_THIRD_PARTY_DIR'] = './data'
os.environ['GEO_BIAS_OUTPUT_DIR'] = './data'

sys.path = [p for p in sys.path if not p.endswith('/local_pkg')]

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb

gfile_mod = types.ModuleType('local_pkg.pyglib.gfile')
gfile_mod.Open = open
gfile_mod.Exists = os.path.exists
gfile_mod.MakeDirs = lambda p: os.makedirs(p, exist_ok=True)
gfile_mod.Glob = glob.glob
gfile_mod.ListDir = os.listdir
pyglib_mod = types.ModuleType('local_pkg.pyglib')
pyglib_mod.gfile = gfile_mod
sys.modules['local_pkg.pyglib'] = pyglib_mod
sys.modules['local_pkg.pyglib.gfile'] = gfile_mod
gfile = gfile_mod

if './data' not in sys.path:
  sys.path.insert(0, './data')

from local_pkg.intelligence.gsnearing.geographical_bias_paper_v2.utils import file_paths
from local_pkg.intelligence.gsnearing.geographical_bias_paper_v2.utils import io_utils
from local_pkg.intelligence.gsnearing.geographical_bias_paper_v2.utils import skill_predictor
from local_pkg.intelligence.gsnearing.geographical_bias_paper_v2.utils.feature_utils import (
    FEATURE_CLASSIFICATION,
    FEATURE_DESCRIPTIONS,
    FEATURE_PALETTE,
)

warnings.filterwarnings('ignore')

OUT_DIR = './data/science_results'
SHAP_OUT_DIR = os.path.join(OUT_DIR, 'shap_scores_lead1')
os.makedirs(SHAP_OUT_DIR, exist_ok=True)

NATURE_DOUBLE_COL_IN = 183.0 / 25.4
SAVE_DPI = 350

METRICS = [
    'NSE',
    'KGE',
    '1.01-year-2.0-window-precision',
    '2.0-year-2.0-window-precision',
    '5.0-year-2.0-window-precision',
    '10.0-year-2.0-window-precision',
    '20.0-year-2.0-window-precision',
    '1.01-year-2.0-window-recall',
    '2.0-year-2.0-window-recall',
    '5.0-year-2.0-window-recall',
    '10.0-year-2.0-window-recall',
    '20.0-year-2.0-window-recall',
]

RECALL_METRICS = [
    '1.01-year-2.0-window-recall',
    '2.0-year-2.0-window-recall',
    '5.0-year-2.0-window-recall',
    '10.0-year-2.0-window-recall',
    '20.0-year-2.0-window-recall',
]

PRECISION_METRICS = [
    '1.01-year-2.0-window-precision',
    '2.0-year-2.0-window-precision',
    '5.0-year-2.0-window-precision',
    '10.0-year-2.0-window-precision',
    '20.0-year-2.0-window-precision',
]


def configure_rc():
  sns.set_theme(style='whitegrid', palette='colorblind')
  plt.rcParams.update({
      'figure.dpi': SAVE_DPI,
      'savefig.dpi': SAVE_DPI,
      'font.family': 'sans-serif',
      'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
      'font.size': 7.5,
      'axes.titlesize': 8.2,
      'axes.titleweight': 'bold',
      'axes.labelsize': 7.5,
      'axes.labelweight': 'bold',
      'xtick.labelsize': 6.8,
      'ytick.labelsize': 6.8,
      'legend.fontsize': 6.8,
      'legend.title_fontsize': 7.2,
      'axes.linewidth': 0.6,
      'grid.linewidth': 0.45,
      'lines.linewidth': 1.0,
      'patch.linewidth': 0.6,
  })


def prepare_feature_map(feature_names: list[str]) -> dict[str, str]:
  full_feature_map = {}
  for feat in feature_names:
    if feat in FEATURE_CLASSIFICATION:
      full_feature_map[feat] = FEATURE_CLASSIFICATION[feat]
    elif (
        'p_minus_pet' in feat
        or 'p_div_pet' in feat
        or 'et_deficit' in feat
        or 'precip' in feat
        or 'water_balance' in feat
        or 'aridity_ratio' in feat
    ):
      full_feature_map[feat] = 'Climate & Hydrology'
    elif 'soil_porosity' in feat:
      full_feature_map[feat] = 'Soil and Geology'
    elif 'twi_proxy' in feat or 'log_calculated_drain_area' in feat:
      full_feature_map[feat] = 'Topography'
    elif (
        'total_forest' in feat
        or 'total_arid_cover' in feat
        or 'forest_aridity' in feat
    ):
      full_feature_map[feat] = 'Land Cover'
    elif (
        'human_' in feat
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


def short_feature_label(feat: str, max_len: int = 34) -> str:
  custom = {
      'anomaly_score_if': 'Climate OOD (Isolation Forest)',
      'anomaly_score_ocsvm': 'Climate OOD (One-Class SVM)',
      'anomaly_score_lof': 'Climate OOD (Local Outlier Factor)',
      'total_upstream_coverage': 'Upstream gauge coverage (%)',
      'max_upstream_coverage': 'Max single-gauge upstream cov.',
      'max_downstream_coverage': 'Max downstream gauge cov.',
      'upstream_gauge_count': 'Upstream gauge count',
      'record_length': 'Gauge record length (yr)',
      'calculated_drain_area': 'Drainage area',
      'ele_mt_uav': 'Mean elevation (m)',
      'slp_dg_uav': 'Mean terrain slope (deg)',
      'pre_mm_uyr': 'Mean annual precipitation',
      'pet_mm_uyr': 'Potential evapotranspiration',
      'aet_mm_uyr': 'Actual evapotranspiration',
      'cmi_ix_uyr': 'Climate moisture index',
      'ari_ix_uav': 'Aridity index',
      'tmp_dc_uyr': 'Mean annual air temperature',
      'snw_pc_uyr': 'Snow cover extent (%)',
      'swc_pc_uyr': 'Soil water content (%)',
      'snd_pc_uav': 'Soil sand fraction (%)',
      'slt_pc_uav': 'Soil silt fraction (%)',
      'cly_pc_uav': 'Soil clay fraction (%)',
      'soc_th_uav': 'Soil organic carbon (t/ha)',
      'ero_kh_uav': 'Soil erosion rate (kg/ha/yr)',
      'kar_pc_use': 'Karst geology extent (%)',
      'for_pc_use': 'Forest cover extent (%)',
      'crp_pc_use': 'Cropland extent (%)',
      'pst_pc_use': 'Pasture extent (%)',
      'urb_pc_use': 'Urban extent (%)',
      'gla_pc_use': 'Glacier cover (%)',
      'prm_pc_use': 'Permafrost extent (%)',
      'pac_pc_use': 'Protected area extent (%)',
      'lka_pc_use': 'Lake area extent (%)',
      'lkv_mc_usu': 'Total lake volume',
      'rev_mc_usu': 'Total reservoir volume',
      'ria_ha_usu': 'Total reservoir area',
      'riv_tc_usu': 'Total river channel volume',
      'ire_pc_use': 'Irrigated area extent (%)',
      'pop_ct_usu': 'Watershed population count',
      'ppd_pk_uav': 'Population density (persons/km²)',
      'gdp_ud_usu': 'Watershed GDP (USD)',
      'rdd_mk_uav': 'Road network density',
      'nli_ix_uav': 'Nighttime lights index',
      'hft_ix_u09': 'Human footprint index (2009)',
      'hft_ix_u93': 'Human footprint index (1993)',
      'inu_pc_ult': 'Long-term inundation extent (%)',
      'inu_pc_umn': 'Minimum inundation extent (%)',
      'inu_pc_umx': 'Maximum inundation extent (%)',
  }
  if feat in custom:
    return custom[feat]
  desc = FEATURE_DESCRIPTIONS.get(feat, feat)
  desc = desc.split(';')[0].rstrip('.')
  desc = (
      desc.replace('Potential Natural Veg:', 'PNV:')
      .replace('Land cover:', 'LC:')
      .replace('Wetlands:', 'Wetland:')
  )
  if len(desc) > max_len:
    desc = desc[: max_len - 1].rstrip() + '…'
  return desc


def compute_single_metric_treeshap(args):
  metric, X_strat, raw_cols = args
  t0 = time.time()
  out_csv = os.path.join(
      SHAP_OUT_DIR, f'consolidated_treeshap_scores_{metric}_lead1.csv'
  )
  if os.path.exists(out_csv) and os.path.getsize(out_csv) > 100000:
    df_sv_raw = pd.read_csv(out_csv, index_col=0)
    df_sv_raw.index = df_sv_raw.index.astype(str)
    return metric, df_sv_raw, time.time() - t0

  pkl_path = f'./data/saved_models/final_ungauged_model_{metric}.pkl'
  with open(pkl_path, 'rb') as f:
    sp = pickle.load(f)

  impl = sp.impl
  if hasattr(impl, 'ensemble'):
    # SkillPredictorReturnPeriod: ('hgbr', hgbr), ('et', et_pipe), ('knn', knn_pipe)
    X_trans = impl._preprocess_features(X_strat).reindex(
        columns=impl.feature_columns, fill_value=0.0
    )
    est_hgbr, est_et_pipe, _ = impl.ensemble.estimators_
    ridge = impl.ensemble.final_estimator_
    w_hgbr, w_et, _ = ridge.coef_
    w_sum = w_hgbr + w_et
    w_hgbr /= w_sum
    w_et /= w_sum

    exp_hgbr = shap.TreeExplainer(
        est_hgbr, data=X_trans.iloc[::10], feature_perturbation='interventional'
    )
    sv_hgbr = exp_hgbr.shap_values(X_trans, check_additivity=False)

    imp = est_et_pipe.named_steps['simpleimputer']
    et_reg = est_et_pipe.named_steps['extratreesregressor']
    et_reg.estimators_ = et_reg.estimators_[:120]
    X_imp = pd.DataFrame(
        imp.transform(X_trans), index=X_trans.index, columns=X_trans.columns
    )
    exp_et = shap.TreeExplainer(
        et_reg, data=X_imp.iloc[::10], feature_perturbation='interventional'
    )
    sv_et = exp_et.shap_values(X_imp, check_additivity=False)

    sv_trans = w_hgbr * sv_hgbr + w_et * sv_et
    df_sv_trans = pd.DataFrame(
        sv_trans, index=X_strat.index, columns=X_trans.columns
    )

    df_sv_raw = pd.DataFrame(0.0, index=X_strat.index, columns=raw_cols)
    for c in raw_cols:
      if c in df_sv_trans.columns:
        df_sv_raw[c] += df_sv_trans[c]

    if 'water_balance' in df_sv_trans.columns:
      df_sv_raw['pre_mm_uyr'] += 0.5 * df_sv_trans['water_balance']
      df_sv_raw['pet_mm_uyr'] += 0.5 * df_sv_trans['water_balance']
    if 'aridity_ratio' in df_sv_trans.columns:
      df_sv_raw['pre_mm_uyr'] += 0.5 * df_sv_trans['aridity_ratio']
      df_sv_raw['pet_mm_uyr'] += 0.5 * df_sv_trans['aridity_ratio']
    if 'total_forest_pc' in df_sv_trans.columns:
      fcols = [
          f'glc_pc_u0{i}'
          for i in range(1, 6)
          if f'glc_pc_u0{i}' in df_sv_raw.columns
      ]
      for fc in fcols:
        df_sv_raw[fc] += df_sv_trans['total_forest_pc'] / len(fcols)
    if 'human_impact_density' in df_sv_trans.columns:
      df_sv_raw['hft_ix_u09'] += 0.5 * df_sv_trans['human_impact_density']
      df_sv_raw['calculated_drain_area'] += (
          0.5 * df_sv_trans['human_impact_density']
      )
  else:
    # SkillPredictorClassic: ('xgb', xgb), ('lgb', lgb), ('hgb', hgb)
    X_trans = impl._engineer_features(X_strat).reindex(
        columns=impl.feature_names, fill_value=0.0
    )
    est_xgb, est_lgb, est_hgb = impl.model.estimators_
    scaler = impl.model.final_estimator_.named_steps['robustscaler']
    ridge = impl.model.final_estimator_.named_steps['ridgecv']
    eff_w = ridge.coef_ / scaler.scale_
    eff_w = eff_w / np.sum(eff_w)
    w_xgb, w_lgb, w_hgb = eff_w[0], eff_w[1], eff_w[2]

    # 1. XGBoost native C++ exact TreeSHAP via pred_contribs
    dmat = xgb.DMatrix(X_trans)
    sv_xgb = est_xgb.get_booster().predict(dmat, pred_contribs=True)[:, :-1]

    # 2. LightGBM native C++ exact TreeSHAP via pred_contrib
    sv_lgb = est_lgb.predict(X_trans, pred_contrib=True)[:, :-1]

    # 3. HistGradientBoosting TreeExplainer
    exp_hgb = shap.TreeExplainer(est_hgb)
    sv_hgb = exp_hgb.shap_values(X_trans, check_additivity=False)

    sv_trans = w_xgb * sv_xgb + w_lgb * sv_lgb + w_hgb * sv_hgb
    df_sv_trans = pd.DataFrame(
        sv_trans, index=X_strat.index, columns=X_trans.columns
    )

    df_sv_raw = pd.DataFrame(0.0, index=X_strat.index, columns=raw_cols)
    for c in raw_cols:
      if c in df_sv_trans.columns:
        df_sv_raw[c] += df_sv_trans[c]

    for eng_col, pairs in [
        ('p_minus_pet', [('pre_mm_uyr', 0.5), ('pet_mm_uyr', 0.5)]),
        ('p_div_pet', [('pre_mm_uyr', 0.5), ('pet_mm_uyr', 0.5)]),
        ('et_deficit', [('pet_mm_uyr', 0.5), ('aet_mm_uyr', 0.5)]),
        ('soil_porosity', [('snd_pc_uav', 0.5), ('cly_pc_uav', 0.5)]),
        ('twi_proxy', [('calculated_drain_area', 0.5), ('slp_dg_uav', 0.5)]),
        ('total_arid_cover', [('glc_pc_u16', 1.0/3), ('glc_pc_u20', 1.0/3), ('glc_pc_u21', 1.0/3)]),
        ('human_intensity', [('gdp_ud_usu', 1.0/3), ('pop_ct_usu', 1.0/3), ('rdd_mk_uav', 1.0/3)]),
        ('log_calculated_drain_area', [('calculated_drain_area', 1.0)]),
        ('log_rev_mc_usu', [('rev_mc_usu', 1.0)]),
        ('log_ria_ha_usu', [('ria_ha_usu', 1.0)]),
        ('log_riv_tc_usu', [('riv_tc_usu', 1.0)]),
        ('log_pop_ct_usu', [('pop_ct_usu', 1.0)]),
        ('precip_temp_inter', [('pre_mm_uyr', 0.5), ('tmp_dc_uyr', 0.5)]),
    ]:
      if eng_col in df_sv_trans.columns:
        for rc, wt in pairs:
          if rc in df_sv_raw.columns:
            df_sv_raw[rc] += wt * df_sv_trans[eng_col]

    fcols = [
        f'glc_pc_u0{i}'
        for i in range(1, 6)
        if f'glc_pc_u0{i}' in df_sv_raw.columns
    ]
    if 'total_forest_cover' in df_sv_trans.columns and fcols:
      for fc in fcols:
        df_sv_raw[fc] += df_sv_trans['total_forest_cover'] / len(fcols)
    if 'forest_aridity_inter' in df_sv_trans.columns:
      if 'ari_ix_uav' in df_sv_raw.columns:
        df_sv_raw['ari_ix_uav'] += 0.5 * df_sv_trans['forest_aridity_inter']
      if fcols:
        for fc in fcols:
          df_sv_raw[fc] += 0.5 * df_sv_trans['forest_aridity_inter'] / len(fcols)

  out_csv = os.path.join(
      SHAP_OUT_DIR, f'consolidated_treeshap_scores_{metric}_lead1.csv'
  )
  df_sv_raw.to_csv(out_csv)
  return metric, df_sv_raw, time.time() - t0


def main():
  t0 = time.time()
  configure_rc()

  print('1. Loading N=1,000 stratified representative global watersheds and hydroatlas attributes...', flush=True)
  nse_ref_path = './data/saved_predictions/shap/NSE/ungauged/stratified_1000/consolidated_exact_kernelshap_scores_NSE.csv'
  df_nse_ref = pd.read_csv(nse_ref_path, index_col=0)
  strat_ids = df_nse_ref.index.astype(str).tolist()
  raw_cols = df_nse_ref.columns.tolist()

  file_paths.init('5.0-year-2.0-window-recall')
  df_global_X = io_utils.load_hybas_zarr()
  df_global_X.index = df_global_X.index.astype(str)
  X_strat = df_global_X.reindex(strat_ids)
  print(f'Loaded X_strat shape: {X_strat.shape} across {len(raw_cols)} raw attributes.', flush=True)

  print('2. Computing Lead-Time-1 TreeSHAP across all 12 core metrics in parallel...', flush=True)
  tasks = [(m, X_strat, raw_cols) for m in METRICS]
  feature_importances = {}
  category_importances = {}
  category_std = {}

  with ProcessPoolExecutor(max_workers=12) as ex:
    for metric, df_shap, dt in ex.map(compute_single_metric_treeshap, tasks):
      print(f'  Finished {metric:32s} in {dt:.1f}s (mean |SHAP| sum = {df_shap.abs().mean().sum():.4f})', flush=True)
      feature_importances[metric] = df_shap.abs().mean()
      feature_map = prepare_feature_map(df_shap.columns.tolist())
      cat_shap = df_shap.T.groupby(feature_map).sum().T
      category_importances[metric] = cat_shap.abs().mean()
      category_std[metric] = cat_shap.abs().std()

  # 2b. Build long-format basin x category DataFrame for errorbar='sd'
  shap_raw_list = []
  for metric in METRICS:
    out_csv = os.path.join(
        SHAP_OUT_DIR, f'consolidated_treeshap_scores_{metric}_lead1.csv'
    )
    df_shap = pd.read_csv(out_csv, index_col=0)
    fmap = prepare_feature_map(df_shap.columns.tolist())
    cat_shap_abs = df_shap.T.groupby(fmap).sum().T.abs()
    df_m_raw = (
        cat_shap_abs.reset_index()
        .rename(columns={'index': 'basin_id'})
        .melt(id_vars='basin_id', var_name='category', value_name='importance')
    )
    df_m_raw['metric'] = metric
    shap_raw_list.append(df_m_raw)
  df_raw_all_f5 = pd.concat(shap_raw_list, ignore_index=True)

  df_features = pd.DataFrame(feature_importances).fillna(0.0)
  df_categories = pd.DataFrame(category_importances).fillna(0.0)
  df_categories_pct = df_categories.div(df_categories.sum(axis=0), axis=1) * 100.0

  # Save category importances & top-35 features CSVs
  cat_csv = os.path.join(OUT_DIR, 'treeshap_category_importances_lead1.csv')
  df_categories.to_csv(cat_csv)
  cat_pct_csv = os.path.join(OUT_DIR, 'treeshap_category_importances_pct_lead1.csv')
  df_categories_pct.to_csv(cat_pct_csv)

  df_features_norm = df_features.div(df_features.max(axis=0), axis=1)
  top35_feats_max = df_features.max(axis=1).sort_values(ascending=False).head(35).index.tolist()
  top35_feats_norm = df_features_norm.mean(axis=1).sort_values(ascending=False).head(35).index.tolist()
  top35_feats = top35_feats_max
  fmap_all = prepare_feature_map(df_features.index.tolist())
  df_top35 = df_features.loc[top35_feats].copy()
  df_top35.insert(0, 'category', [fmap_all[f] for f in top35_feats])
  df_top35.insert(1, 'description', [short_feature_label(f, max_len=60) for f in top35_feats])
  df_top35['max_importance'] = df_features.loc[top35_feats].max(axis=1)
  df_top35['mean_normalized_importance'] = df_features_norm.loc[top35_feats].mean(axis=1)
  top35_csv = os.path.join(OUT_DIR, 'treeshap_top35_features_lead1.csv')
  df_top35.to_csv(top35_csv)
  print(f'Saved {cat_csv} and {top35_csv}', flush=True)

  # 3. Render Fig5_TreeSHAP_lead1.png (and .pdf) — matching Figure 4/5 in 9_figures_and_tables_for_paper.py
  print('3. Rendering Fig5_TreeSHAP_lead1.png...', flush=True)
  metric_palette = {
      'NSE': '#0d47a1',
      'KGE': '#42a5f5',
      '1.01-year-2.0-window-precision': '#ffcc80',
      '2.0-year-2.0-window-precision': '#ffa726',
      '5.0-year-2.0-window-precision': '#fb8c00',
      '10.0-year-2.0-window-precision': '#e65100',
      '20.0-year-2.0-window-precision': '#b71c1c',
      '1.01-year-2.0-window-recall': '#c8e6c9',
      '2.0-year-2.0-window-recall': '#81c784',
      '5.0-year-2.0-window-recall': '#4caf50',
      '10.0-year-2.0-window-recall': '#2e7d32',
      '20.0-year-2.0-window-recall': '#1b5e20',
  }
  metric_labels_map = {
      'NSE': 'NSE',
      'KGE': 'KGE',
      '1.01-year-2.0-window-precision': 'Precision (1y)',
      '2.0-year-2.0-window-precision': 'Precision (2y)',
      '5.0-year-2.0-window-precision': 'Precision (5y)',
      '10.0-year-2.0-window-precision': 'Precision (10y)',
      '20.0-year-2.0-window-precision': 'Precision (20y)',
      '1.01-year-2.0-window-recall': 'Recall (1y)',
      '2.0-year-2.0-window-recall': 'Recall (2y)',
      '5.0-year-2.0-window-recall': 'Recall (5y)',
      '10.0-year-2.0-window-recall': 'Recall (10y)',
      '20.0-year-2.0-window-recall': 'Recall (20y)',
  }
  plot_palette = {**FEATURE_PALETTE, 'Other': 'gray'}
  short_cat_label = {
      'Climate & Hydrology': 'Climate/Hydrol.',
      'Land Cover': 'Land Cover',
      'Anomaly Score': 'Anomaly Score',
      'Topography': 'Topography',
      'Soil and Geology': 'Soil & Geology',
      'Anthropogenic Factors': 'Anthropogenic',
      'Coverage/Overlap': 'Gauge Coverage',
      'Record Length': 'Record Length',
  }

  fig5 = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 6.4))
  gs5 = gridspec.GridSpec(
      3, 5, height_ratios=[1.28, 1.0, 1.0], hspace=0.52, wspace=1.28
  )

  ax_comp = fig5.add_subplot(gs5[0, 0:3])
  cat_order = df_categories.mean(axis=1).sort_values(ascending=False).index.tolist()
  sns.barplot(
      x='category',
      y='importance',
      hue='metric',
      data=df_raw_all_f5,
      order=cat_order,
      hue_order=METRICS,
      palette=metric_palette,
      ax=ax_comp,
      edgecolor='black',
      linewidth=0.35,
      errorbar='sd',
      err_kws={'linewidth': 0.55},
  )
  ax_comp.set_title(
      '(a) Feature category importance across all 12 metrics',
      fontsize=7.2,
      fontweight='bold',
      loc='left',
      pad=4,
  )
  ax_comp.set_xlabel('')
  ax_comp.set_ylabel('Mean |SHAP|', fontsize=6.6, fontweight='bold')
  ax_comp.set_xticklabels(
      [short_cat_label.get(c, c) for c in cat_order],
      rotation=22,
      ha='right',
      fontsize=5.8,
  )
  ax_comp.tick_params(axis='y', labelsize=5.8)
  handles_f5, labels_f5 = ax_comp.get_legend_handles_labels()
  ax_comp.legend(
      handles_f5,
      [metric_labels_map.get(lbl, lbl) for lbl in labels_f5],
      title='Metric',
      loc='upper right',
      fontsize=4.9,
      title_fontsize=5.4,
      ncol=2,
      frameon=True,
      handlelength=1.0,
      columnspacing=0.7,
  )
  sns.despine(ax=ax_comp)
  ax_comp.grid(axis='y', linestyle='--', alpha=0.45)

  def plot_cat_panel(ax, metric, title_str, show_xlabel=True):
    df_m = df_categories[metric].sort_values(ascending=False)
    df_mr = df_raw_all_f5[df_raw_all_f5['metric'] == metric]
    sns.barplot(
        x='importance',
        y='category',
        data=df_mr,
        order=df_m.index,
        palette=plot_palette,
        hue='category',
        legend=False,
        ax=ax,
        edgecolor='black',
        linewidth=0.35,
        errorbar='sd',
        err_kws={'linewidth': 0.55},
    )
    ax.set_yticklabels(
        [short_cat_label.get(c, c) for c in df_m.index], fontsize=4.8
    )
    ax.set_title(title_str, fontsize=6.2, fontweight='bold', loc='left', pad=3)
    ax.set_xlabel('Mean |SHAP|' if show_xlabel else '', fontsize=5.5, fontweight='bold')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=4.8)
    sns.despine(ax=ax)
    ax.grid(axis='x', linestyle='--', alpha=0.45)

  for idx, (metric, pl_ltr) in enumerate([('NSE', '(b)'), ('KGE', '(c)')]):
    ax = fig5.add_subplot(gs5[0, 3 + idx])
    plot_cat_panel(ax, metric, f'{pl_ltr} {metric}', show_xlabel=True)

  sub_letters_rows = [
      ['(d)', '(e)', '(f)', '(g)', '(h)'],
      ['(i)', '(j)', '(k)', '(l)', '(m)'],
  ]
  for row_i, m_group, pfx, ltrs, show_x in [
      (1, RECALL_METRICS, 'Recall', sub_letters_rows[0], False),
      (2, PRECISION_METRICS, 'Precision', sub_letters_rows[1], True),
  ]:
    for idx, metric in enumerate(m_group):
      ax = fig5.add_subplot(gs5[row_i, idx])
      yr_str = '1y' if metric.startswith('1.01') else f"{metric.split('.')[0]}y"
      plot_cat_panel(ax, metric, f'{ltrs[idx]} {pfx} ({yr_str})', show_xlabel=show_x)

  fig5_png = os.path.join(OUT_DIR, 'Fig5_TreeSHAP_lead1.png')
  fig5_pdf = os.path.join(OUT_DIR, 'Fig5_TreeSHAP_lead1.pdf')
  fig5.savefig(fig5_png, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.03)
  fig5.savefig(fig5_pdf, bbox_inches='tight', pad_inches=0.03)
  plt.close(fig5)
  print(f'Saved {fig5_png}', flush=True)

  # 4. Render FigS6_TreeSHAP_top35_lead1.png (and .pdf) — matching Extended_Data_Figure_1.png
  print('4. Rendering FigS6_TreeSHAP_top35_lead1.png...', flush=True)
  short_col_labels = [
      'NSE', 'KGE',
      'Prec (1y)', 'Prec (2y)', 'Prec (5y)', 'Prec (10y)', 'Prec (20y)',
      'Rec (1y)', 'Rec (2y)', 'Rec (5y)', 'Rec (10y)', 'Rec (20y)',
  ]
  df_s6_raw = df_features.loc[top35_feats, METRICS].copy()
  df_s6_raw.columns = short_col_labels
  fmap_s6 = prepare_feature_map(top35_feats)
  y_labels_s6 = [
      f'{short_feature_label(f, max_len=38)} ({fmap_s6[f]})' for f in top35_feats
  ]

  fig_s6, ax_s6 = plt.subplots(figsize=(NATURE_DOUBLE_COL_IN, 7.6))
  sns.heatmap(
      df_s6_raw,
      ax=ax_s6,
      cmap='viridis',
      annot=True,
      fmt='.4f',
      annot_kws={'size': 4.8},
      linewidths=0.30,
      cbar_kws={'label': 'Mean Absolute Shapley (TreeSHAP) Value', 'shrink': 0.82},
  )
  ax_s6.set_yticks(np.arange(len(top35_feats)) + 0.5)
  ax_s6.set_yticklabels(y_labels_s6, rotation=0, fontsize=5.6)
  ax_s6.set_xticklabels(short_col_labels, rotation=32, ha='right', fontsize=6.2, fontweight='bold')
  ax_s6.set_title(
      'Global Feature Importance Heatmap (Top 35 Features Across 12 Metrics, Lead Time 1)',
      fontsize=8.0, fontweight='bold', pad=8,
  )
  ax_s6.set_xlabel('Hydrological Evaluation Metric', fontsize=7.0, fontweight='bold')
  ax_s6.set_ylabel('Watershed Predictor Feature (Category)', fontsize=7.0, fontweight='bold')

  s6_png = os.path.join(OUT_DIR, 'FigS6_TreeSHAP_top35_lead1.png')
  s6_pdf = os.path.join(OUT_DIR, 'FigS6_TreeSHAP_top35_lead1.pdf')
  fig_s6.savefig(s6_png, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.03)
  fig_s6.savefig(s6_pdf, bbox_inches='tight', pad_inches=0.03)
  plt.close(fig_s6)
  print(f'Saved {s6_png}', flush=True)

  # 5. Write sec4_treeshap_summary_lead1.md
  print('5. Writing sec4_treeshap_summary_lead1.md...', flush=True)
  cat_order_pct = df_categories_pct.mean(axis=1).sort_values(ascending=False).index.tolist()
  classic_cols = ['NSE', 'KGE']
  rp_cols = RECALL_METRICS + PRECISION_METRICS

  md = [
      '# Section 4: TreeSHAP Feature Attribution Summary at Lead Time 1 ($t_0 + 1$ Day)',
      '',
      'Computed on $N = 1,000$ stratified representative global watersheds across all 12 core Lead-Time-1 `M_base` (`ungauged`) `SkillPredictor` meta-models (`NSE`, `KGE`, and `1.01, 2.0, 5.0, 10.0, 20.0`-year `2d` flood recall and precision).',
      '',
      '## 1. Category-Level Relative Importance (% of Total Category |SHAP|)',
      '',
      '| Feature Category | Mean Across All 12 Metrics (%) | Classic (`NSE`, `KGE`) Mean (%) | Return-Period (`Recall` & `Precision`) Mean (%) | `NSE` (%) | `KGE` (%) | `5y Recall (2d)` (%) | `5y Precision (2d)` (%) | `20y Recall (2d)` (%) |',
      '| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |',
  ]
  for cat in cat_order_pct:
    m_all = float(df_categories_pct.loc[cat, METRICS].mean())
    m_cls = float(df_categories_pct.loc[cat, classic_cols].mean())
    m_rp = float(df_categories_pct.loc[cat, rp_cols].mean())
    v_nse = float(df_categories_pct.loc[cat, 'NSE'])
    v_kge = float(df_categories_pct.loc[cat, 'KGE'])
    v_r5 = float(df_categories_pct.loc[cat, '5.0-year-2.0-window-recall'])
    v_p5 = float(df_categories_pct.loc[cat, '5.0-year-2.0-window-precision'])
    v_r20 = float(df_categories_pct.loc[cat, '20.0-year-2.0-window-recall'])
    md.append(
        f'| **{cat}** | **{m_all:.2f}%** | {m_cls:.2f}% | {m_rp:.2f}% | {v_nse:.2f}% | {v_kge:.2f}% | {v_r5:.2f}% | {v_p5:.2f}% | {v_r20:.2f}% |'
    )

  md.extend([
      '',
      '## 2. Category-Level Raw Mean |SHAP| Across All 12 Metrics',
      '',
      '| Feature Category | `NSE` | `KGE` | `Rec 1y` | `Rec 2y` | `Rec 5y` | `Rec 10y` | `Rec 20y` | `Prec 1y` | `Prec 2y` | `Prec 5y` | `Prec 10y` | `Prec 20y` |',
      '| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |',
  ])
  for cat in cat_order_pct:
    row_vals = [
        f"{df_categories.loc[cat, m]:.4f}"
        for m in ['NSE', 'KGE'] + RECALL_METRICS + PRECISION_METRICS
    ]
    md.append(f"| **{cat}** | " + ' | '.join(row_vals) + ' |')

  md.extend([
      '',
      '## 3. Top-8 Features Per Metric (Raw Mean |SHAP| & Category, Matching `Fig. 5`)',
      '',
  ])
  for m in ['NSE', 'KGE', '5.0-year-2.0-window-recall', '10.0-year-2.0-window-recall', '20.0-year-2.0-window-recall', '5.0-year-2.0-window-precision']:
    top8 = df_features[m].sort_values(ascending=False).head(8)
    md.append(f'### `{m}`')
    for rank, (f, val) in enumerate(top8.items(), 1):
      md.append(f'{rank}. **`{f}`** ({short_feature_label(f, max_len=50)}, *{fmap_all[f]}*): Mean |SHAP| = **{val:.4f}**')
    md.append('')

  md.extend([
      '## 4. Top-35 Features Across All 12 Metrics (Sorted by Mean Normalized Importance, Matching `Fig. S6`)',
      '',
      '| Rank | Feature Code | Description | Category | Mean Norm. |SHAP| | `NSE` | `KGE` | `5y Recall (2d)` | `10y Recall (2d)` | `20y Recall (2d)` | `5y Precision (2d)` |',
      '| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |',
  ])
  for idx, f in enumerate(top35_feats, 1):
    r = df_top35.loc[f]
    md.append(
        f"| {idx} | `{f}` | {r['description']} | {r['category']} | **{r['mean_normalized_importance']:.3f}** | "
        f"{r['NSE']:.4f} | {r['KGE']:.4f} | {r['5.0-year-2.0-window-recall']:.4f} | "
        f"{r['10.0-year-2.0-window-recall']:.4f} | {r['20.0-year-2.0-window-recall']:.4f} | "
        f"{r['5.0-year-2.0-window-precision']:.4f} |"
    )

  md_path = os.path.join(OUT_DIR, 'sec4_treeshap_summary_lead1.md')
  with open(md_path, 'w') as f:
    f.write('\n'.join(md) + '\n')
  print(f'Saved {md_path} in {time.time() - t0:.1f}s!', flush=True)


if __name__ == '__main__':
  main()
