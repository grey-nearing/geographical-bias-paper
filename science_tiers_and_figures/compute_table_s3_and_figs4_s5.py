"""Compute Lead-Time-1 Table S3 (173 countries), country-level correlations, and render Fig. S4 and Fig. S5."""

import glob
import io
import json
import os
import sys
import time
import types
import warnings

os.environ['GEO_BIAS_DATA_DIR'] = './data'
os.environ['GEO_BIAS_THIRD_PARTY_DIR'] = './data'
os.environ['GEO_BIAS_OUTPUT_DIR'] = './data'

sys.path = [p for p in sys.path if not p.endswith('/local_pkg')]

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns

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

warnings.filterwarnings('ignore')

OUT_DIR = './data/science_results'
os.makedirs(OUT_DIR, exist_ok=True)

CANON_V2_DIR = './data'
PRED_DIR = './data/science_results_lead1/predictions'
FEAT_DIR = './data/coverage_fractions'

NATURE_DOUBLE_COL_IN = 183.0 / 25.4
SAVE_DPI = 350


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


def _normalize_hybas_id(series: pd.Series) -> pd.Series:
  nums = pd.to_numeric(
      series.astype(str)
      .str.replace('hybas_', '', regex=False)
      .str.replace(r'\.0$', '', regex=True),
      errors='coerce',
  )
  return 'hybas_' + nums.astype('Int64').astype(str)


def _load_world_gdf() -> gpd.GeoDataFrame:
  with gfile.Open(file_paths.NATURAL_EARTH_PATH, 'rb') as f:
    world = gpd.read_file(f)
  if 'name' in world.columns:
    world = world[world['name'] != 'Antarctica'].copy()
  world = world.dropna(subset=[world.geometry.name])
  world = world[world.is_valid & (~world.is_empty)].copy()
  world['map_name'] = world['name'].replace(file_paths.country_name_map)
  iso_fix = {'France': 'FRA', 'Norway': 'NOR'}
  world['iso_clean'] = [
      iso_fix.get(mname, str(iso).strip())
      for mname, iso in zip(world['map_name'], world.get('iso_a3', ''))
  ]
  return world


def _load_inform_vulnerability(world_gdf: gpd.GeoDataFrame) -> pd.Series:
  country_to_iso = dict(zip(world_gdf['map_name'], world_gdf['iso_clean']))
  with gfile.Open(file_paths.RISK_FILE, 'rt') as f:
    df_inform = pd.read_csv(io.StringIO(f.read()), skiprows=[0, 2])
  df_inform['ISO3'] = df_inform['ISO3'].astype(str).str.strip()
  df_inform['COUNTRY_CLEAN'] = (
      df_inform['COUNTRY']
      .astype(str)
      .str.strip()
      .replace(file_paths.country_name_map)
  )
  df_inform['VULN_NUM'] = pd.to_numeric(df_inform['VULNERABILITY'], errors='coerce')
  iso_vuln = (
      df_inform.dropna(subset=['ISO3', 'VULN_NUM'])
      .drop_duplicates('ISO3')
      .set_index('ISO3')['VULN_NUM']
  )
  name_vuln = (
      df_inform.dropna(subset=['COUNTRY_CLEAN', 'VULN_NUM'])
      .drop_duplicates('COUNTRY_CLEAN')
      .set_index('COUNTRY_CLEAN')['VULN_NUM']
  )
  all_countries = sorted(set(country_to_iso.keys()).union(set(name_vuln.index)))
  out = {}
  for c in all_countries:
    iso = country_to_iso.get(c)
    val = iso_vuln.get(iso, np.nan) if iso else np.nan
    if pd.isna(val):
      val = name_vuln.get(c, np.nan)
    out[c] = val
  return pd.Series(out, name='Vulnerability')


def _load_hybas_latlon() -> pd.DataFrame:
  with gfile.Open(file_paths.HYBAS_LATLON_PATH, 'rt') as f:
    df_ll = pd.read_csv(f, index_col=0)
  lat_col = next(c for c in df_ll.columns if 'lat' in c.lower())
  lon_col = next(c for c in df_ll.columns if 'lon' in c.lower())
  out = pd.DataFrame({
      'hybas_id': _normalize_hybas_id(pd.Series(df_ll.index)).values,
      'lat': pd.to_numeric(df_ll[lat_col], errors='coerce').values,
      'lon': pd.to_numeric(df_ll[lon_col], errors='coerce').values,
  })
  return out.dropna(subset=['lat', 'lon']).drop_duplicates('hybas_id').set_index('hybas_id')


def _fmt_sci_tex(p: float) -> str:
  if p <= 0 or np.isnan(p):
    return '0'
  if p >= 0.001:
    return f'{p:.3f}'
  exp = int(np.floor(np.log10(p)))
  mant = p / (10 ** exp)
  return f'{mant:.1f} \\times 10^{{{exp}}}'


def fmt_p_plain(p: float) -> str:
  if p <= 0 or np.isnan(p):
    return '0'
  if p >= 0.001:
    return f'{p:.4f}'
  return f'{p:.2e}'


def compute_clipped_gain(E_5, E_10, E_20, r5, r10, r20, r5_b, r10_b, r20_b):
  return (
      E_5 * np.maximum(0.0, r5 - r5_b)
      + E_10 * np.maximum(0.0, r10 - r10_b)
      + E_20 * np.maximum(0.0, r20 - r20_b)
  )


def compute_warned(E_5, E_10, E_20, r5, r10, r20):
  return E_5 * r5 + E_10 * r10 + E_20 * r20


def main():
  t0 = time.time()
  configure_rc()

  print('1. Loading basin exposure, country mappings, lat/lon, and INFORM 2026 Vulnerability...', flush=True)
  world_gdf = _load_world_gdf()
  vuln_series = _load_inform_vulnerability(world_gdf)
  df_latlon = _load_hybas_latlon()

  exp_df = pd.read_parquet(
      f'{CANON_V2_DIR}/v2_exposure_blockers/statistics/basin_exposure_and_blockers_v2.parquet',
      columns=['country_name', 'has_system', 'E_total'],
  )
  exp_df.index = _normalize_hybas_id(pd.Series(exp_df.index)).values
  exp_df['country_name'] = exp_df['country_name'].replace(file_paths.country_name_map)
  exp_df = exp_df.join(df_latlon, how='left')

  rp_raw = pd.read_csv(f'{CANON_V2_DIR}/exposure/global_basin_exposure_defended_by_return_period.csv')
  id_col = 'hybas_id' if 'hybas_id' in rp_raw.columns else rp_raw.columns[0]
  rp_raw[id_col] = _normalize_hybas_id(rp_raw[id_col]).values
  rp_raw = rp_raw.drop_duplicates(subset=[id_col]).set_index(id_col).reindex(exp_df.index).fillna(0.0)

  def _get_exp_col(df, t):
    for c in [f'exposed_pop_{t}', f'exposure_{t}yr', str(t)]:
      if c in df.columns:
        return df[c].values
    raise KeyError(f'Missing exposure column for T={t}')

  E_5 = 0.05 * _get_exp_col(rp_raw, 5)
  E_10 = 0.075 * _get_exp_col(rp_raw, 10)
  E_20 = 0.04 * _get_exp_col(rp_raw, 20) + 0.02 * _get_exp_col(rp_raw, 50) + 0.005 * _get_exp_col(rp_raw, 100)
  E_tot = E_5 + E_10 + E_20
  exp_df['E_5'] = E_5
  exp_df['E_10'] = E_10
  exp_df['E_20_50_100'] = E_20
  exp_df['E_total'] = E_tot
  has_sys = exp_df['has_system'].values.astype(bool)

  # Load coverage features for Tier 1a and Tier 4
  f_t1a = pd.read_parquet(f'{FEAT_DIR}/tier1a_active_post2016_ccby_8811.parquet', columns=['total_upstream_coverage'])
  f_t4 = pd.read_parquet(f'{FEAT_DIR}/tier4_all_publicly_knowable_121819.parquet', columns=['total_upstream_coverage'])
  out_t1a = f_t1a['total_upstream_coverage'].values >= 0.95
  out_t4 = f_t4['total_upstream_coverage'].values >= 0.95
  out_t5 = np.ones(len(exp_df), dtype=bool)

  zero_up_t4 = (f_t4['total_upstream_coverage'].values <= 0.0).astype(float) * 100.0
  zero_up_t1a = (f_t1a['total_upstream_coverage'].values <= 0.0).astype(float) * 100.0
  exp_df['zero_up_t4'] = zero_up_t4
  exp_df['zero_up_t1a'] = zero_up_t1a

  print('2. Loading cached Lead-Time-1 counterfactual predictions across 964,084 watersheds...', flush=True)
  def load_p(m_tag, tier, mode):
    return pd.read_parquet(f'{PRED_DIR}/{m_tag}__{tier}__{mode}.parquet')

  p_base_t1a_base = load_p('base', 'tier1a_active_post2016_ccby_8811', 'joint')
  p_da_t1a_base = load_p('da', 'tier1a_active_post2016_ccby_8811', 'joint')

  p_base_t4_base = load_p('base', 'tier4_all_publicly_knowable_121819', 'joint')
  p_base_t4_joint = p_base_t4_base
  p_da_t4_base = load_p('da', 'tier4_all_publicly_knowable_121819', 'joint')
  p_da_t4_joint = p_da_t4_base

  p_base_t5_f2 = load_p('base', 'tier5_universal_idealized_ceiling', 'f2_cov_only_base1a')
  p_base_t5_f3 = load_p('base', 'tier5_universal_idealized_ceiling', 'f3_ood_only_base1a')
  p_base_t5_joint = load_p('base', 'tier5_universal_idealized_ceiling', 'joint')

  p_da_t5_joint = load_p('da', 'tier5_universal_idealized_ceiling', 'joint')

  # Baseline 1: Tier 1a (8,811 Post-2016 Caravan Gauges, 000 No DA)
  r5_b1, r10_b1, r20_b1 = p_base_t1a_base['r5_u'].values, p_base_t1a_base['r10_u'].values, p_base_t1a_base['r20_u'].values
  W_b1 = compute_warned(E_5, E_10, E_20, r5_b1, r10_b1, r20_b1)
  U_b1 = E_tot - W_b1

  # Baseline 2: Tier 4 (121,819 All Publicly Knowable Gauges, 000 No DA)
  r5_b4, r10_b4, r20_b4 = p_base_t4_base['r5_u'].values, p_base_t4_base['r10_u'].values, p_base_t4_base['r20_u'].values
  W_b4 = compute_warned(E_5, E_10, E_20, r5_b4, r10_b4, r20_b4)
  U_b4 = E_tot - W_b4

  exp_df['warned_t1a'] = W_b1
  exp_df['unwarned_t1a'] = U_b1
  exp_df['warned_t4'] = W_b4
  exp_df['unwarned_t4'] = U_b4

  # --- Counterfactuals relative to Tier 1a Baseline (matching multi_mechanism_tiered_skill_and_exposure_lead1.csv) ---
  # Tier 4 (121,819 Publicly Knowable Gauges):
  # 000: Global Retraining Only (M_base at tier4__joint)
  exp_df['d_t4_000_clip_from_t1a'] = compute_clipped_gain(
      E_5, E_10, E_20,
      p_base_t4_joint['r5_u'].values, p_base_t4_joint['r10_u'].values, p_base_t4_joint['r20_u'].values,
      r5_b1, r10_b1, r20_b1,
  )
  exp_df['d_t4_000_net_from_t1a'] = compute_warned(
      E_5, E_10, E_20,
      p_base_t4_joint['r5_u'].values, p_base_t4_joint['r10_u'].values, p_base_t4_joint['r20_u'].values,
  ) - W_b1

  # 001: 30-Day Variational DA Only at Tier 4 outlets (M_base fixed at Tier 1a)
  r5_t4_001 = np.where(out_t4, p_da_t1a_base['r5_u'].values, r5_b1)
  r10_t4_001 = np.where(out_t4, p_da_t1a_base['r10_u'].values, r10_b1)
  r20_t4_001 = np.where(out_t4, p_da_t1a_base['r20_u'].values, r20_b1)
  exp_df['d_t4_001_clip_from_t1a'] = compute_clipped_gain(E_5, E_10, E_20, r5_t4_001, r10_t4_001, r20_t4_001, r5_b1, r10_b1, r20_b1)
  exp_df['d_t4_001_net_from_t1a'] = compute_warned(E_5, E_10, E_20, r5_t4_001, r10_t4_001, r20_t4_001) - W_b1

  # 101: Combined (Global Retraining + 30-Day DA at Tier 4 outlets)
  r5_t4_101 = np.where(out_t4, p_da_t4_joint['r5_u'].values, p_base_t4_joint['r5_u'].values)
  r10_t4_101 = np.where(out_t4, p_da_t4_joint['r10_u'].values, p_base_t4_joint['r10_u'].values)
  r20_t4_101 = np.where(out_t4, p_da_t4_joint['r20_u'].values, p_base_t4_joint['r20_u'].values)
  exp_df['d_t4_101_clip_from_t1a'] = compute_clipped_gain(E_5, E_10, E_20, r5_t4_101, r10_t4_101, r20_t4_101, r5_b1, r10_b1, r20_b1)
  exp_df['d_t4_101_net_from_t1a'] = compute_warned(E_5, E_10, E_20, r5_t4_101, r10_t4_101, r20_t4_101) - W_b1

  # Tier 5 (Universal Gauging, 964,084 Watersheds) from Tier 1a:
  # 000: Global Retraining Only (M_base at tier5__joint)
  exp_df['d_t5_000_clip_from_t1a'] = compute_clipped_gain(
      E_5, E_10, E_20,
      p_base_t5_joint['r5_u'].values, p_base_t5_joint['r10_u'].values, p_base_t5_joint['r20_u'].values,
      r5_b1, r10_b1, r20_b1,
  )
  exp_df['d_t5_000_net_from_t1a'] = compute_warned(
      E_5, E_10, E_20,
      p_base_t5_joint['r5_u'].values, p_base_t5_joint['r10_u'].values, p_base_t5_joint['r20_u'].values,
  ) - W_b1

  # 001: 30-Day Variational DA Only at Tier 5 outlets (all 964,084 watersheds)
  exp_df['d_t5_001_clip_from_t1a'] = compute_clipped_gain(
      E_5, E_10, E_20,
      p_da_t1a_base['r5_u'].values, p_da_t1a_base['r10_u'].values, p_da_t1a_base['r20_u'].values,
      r5_b1, r10_b1, r20_b1,
  )
  exp_df['d_t5_001_net_from_t1a'] = compute_warned(
      E_5, E_10, E_20,
      p_da_t1a_base['r5_u'].values, p_da_t1a_base['r10_u'].values, p_da_t1a_base['r20_u'].values,
  ) - W_b1

  # 101: Combined (Global Retraining + 30-Day DA at Tier 5 = da__tier5_universal__joint)
  exp_df['d_t5_101_clip_from_t1a'] = compute_clipped_gain(
      E_5, E_10, E_20,
      p_da_t5_joint['r5_u'].values, p_da_t5_joint['r10_u'].values, p_da_t5_joint['r20_u'].values,
      r5_b1, r10_b1, r20_b1,
  )
  exp_df['d_t5_101_net_from_t1a'] = compute_warned(
      E_5, E_10, E_20,
      p_da_t5_joint['r5_u'].values, p_da_t5_joint['r10_u'].values, p_da_t5_joint['r20_u'].values,
  ) - W_b1

  # Single-factor disaggregations at Tier 5 from Tier 1a (000 and 101)
  exp_df['d_t5_000_f2_clip_from_t1a'] = compute_clipped_gain(
      E_5, E_10, E_20, p_base_t5_f2['r5_u'].values, p_base_t5_f2['r10_u'].values, p_base_t5_f2['r20_u'].values, r5_b1, r10_b1, r20_b1
  )
  exp_df['d_t5_000_f3_clip_from_t1a'] = compute_clipped_gain(
      E_5, E_10, E_20, p_base_t5_f3['r5_u'].values, p_base_t5_f3['r10_u'].values, p_base_t5_f3['r20_u'].values, r5_b1, r10_b1, r20_b1
  )
  exp_df['d_t5_000_f4_clip_from_t1a'] = np.zeros(len(exp_df), dtype=float)

  # --- Counterfactuals relative to Tier 4 (121,819-Gauge) Baseline (matching legacy Table S3 format) ---
  exp_df['d_t5_000_f2_clip_from_t4'] = compute_clipped_gain(
      E_5, E_10, E_20, p_base_t5_f2['r5_u'].values, p_base_t5_f2['r10_u'].values, p_base_t5_f2['r20_u'].values, r5_b4, r10_b4, r20_b4
  )
  exp_df['d_t5_000_f3_clip_from_t4'] = compute_clipped_gain(
      E_5, E_10, E_20, p_base_t5_f3['r5_u'].values, p_base_t5_f3['r10_u'].values, p_base_t5_f3['r20_u'].values, r5_b4, r10_b4, r20_b4
  )
  exp_df['d_t5_000_f4_clip_from_t4'] = np.zeros(len(exp_df), dtype=float)
  exp_df['d_t5_000_joint_clip_from_t4'] = compute_clipped_gain(
      E_5, E_10, E_20, p_base_t5_joint['r5_u'].values, p_base_t5_joint['r10_u'].values, p_base_t5_joint['r20_u'].values, r5_b4, r10_b4, r20_b4
  )
  exp_df['d_t5_001_clip_from_t4'] = compute_clipped_gain(
      E_5, E_10, E_20, p_da_t4_base['r5_u'].values, p_da_t4_base['r10_u'].values, p_da_t4_base['r20_u'].values, r5_b4, r10_b4, r20_b4
  )
  exp_df['d_t5_101_joint_clip_from_t4'] = compute_clipped_gain(
      E_5, E_10, E_20, p_da_t5_joint['r5_u'].values, p_da_t5_joint['r10_u'].values, p_da_t5_joint['r20_u'].values, r5_b4, r10_b4, r20_b4
  )

  # Aggregate by Country across all 173 modeled countries
  print('3. Aggregating across all 173 modeled countries...', flush=True)
  sum_cols = [
      'E_total', 'warned_t1a', 'unwarned_t1a', 'warned_t4', 'unwarned_t4',
      'd_t4_000_clip_from_t1a', 'd_t4_000_net_from_t1a',
      'd_t4_001_clip_from_t1a', 'd_t4_001_net_from_t1a',
      'd_t4_101_clip_from_t1a', 'd_t4_101_net_from_t1a',
      'd_t5_000_clip_from_t1a', 'd_t5_000_net_from_t1a',
      'd_t5_001_clip_from_t1a', 'd_t5_001_net_from_t1a',
      'd_t5_101_clip_from_t1a', 'd_t5_101_net_from_t1a',
      'd_t5_000_f2_clip_from_t1a', 'd_t5_000_f3_clip_from_t1a', 'd_t5_000_f4_clip_from_t1a',
      'd_t5_000_f2_clip_from_t4', 'd_t5_000_f3_clip_from_t4', 'd_t5_000_f4_clip_from_t4',
      'd_t5_000_joint_clip_from_t4', 'd_t5_001_clip_from_t4', 'd_t5_101_joint_clip_from_t4',
  ]

  grp = exp_df.groupby('country_name', as_index=False)
  df_c = grp[sum_cols].sum()
  df_c['n_basins'] = grp.size()['size'].values
  df_c['has_system'] = grp['has_system'].first()['has_system'].values.astype(bool)
  df_c['zero_up_pct_t4'] = grp['zero_up_t4'].mean()['zero_up_t4'].values
  df_c['zero_up_pct_t1a'] = grp['zero_up_t1a'].mean()['zero_up_t1a'].values
  df_c['inform_vulnerability'] = df_c['country_name'].map(vuln_series)

  # Compute percentages of unwarned exposure (and baseline exposure)
  df_c['unwarned_pct_t1a'] = np.where(df_c['E_total'] > 0, 100.0 * df_c['unwarned_t1a'] / df_c['E_total'], 0.0)
  df_c['unwarned_pct_t4'] = np.where(df_c['E_total'] > 0, 100.0 * df_c['unwarned_t4'] / df_c['E_total'], 0.0)

  for col_in, col_pct_u1, col_pct_u4 in [
      ('d_t4_000_clip_from_t1a', 'pct_unw_t1a_d_t4_000', None),
      ('d_t4_001_clip_from_t1a', 'pct_unw_t1a_d_t4_001', None),
      ('d_t4_101_clip_from_t1a', 'pct_unw_t1a_d_t4_101', None),
      ('d_t5_000_clip_from_t1a', 'pct_unw_t1a_d_t5_000', None),
      ('d_t5_001_clip_from_t1a', 'pct_unw_t1a_d_t5_001', None),
      ('d_t5_101_clip_from_t1a', 'pct_unw_t1a_d_t5_101', None),
      ('d_t5_000_joint_clip_from_t4', None, 'pct_unw_t4_d_t5_000_joint'),
      ('d_t5_000_f2_clip_from_t4', None, 'pct_unw_t4_d_t5_000_f2'),
      ('d_t5_000_f3_clip_from_t4', None, 'pct_unw_t4_d_t5_000_f3'),
      ('d_t5_101_joint_clip_from_t4', None, 'pct_unw_t4_d_t5_101_joint'),
  ]:
    if col_pct_u1:
      df_c[col_pct_u1] = np.where(df_c['unwarned_t1a'] > 0, 100.0 * df_c[col_in] / df_c['unwarned_t1a'], 0.0)
    if col_pct_u4:
      df_c[col_pct_u4] = np.where(df_c['unwarned_t4'] > 0, 100.0 * df_c[col_in] / df_c['unwarned_t4'], 0.0)

  df_c = df_c.sort_values('E_total', ascending=False).reset_index(drop=True)
  csv_s3 = os.path.join(OUT_DIR, 'Table_S3_lead1.csv')
  df_c.to_csv(csv_s3, index=False)
  print(f'Saved {csv_s3} ({len(df_c)} countries)', flush=True)

  # 4. Compute exact Country-Level Spearman Correlations
  print('4. Computing country-level Spearman correlations...', flush=True)
  df_c_exp = df_c[df_c['E_total'] > 0].copy()
  df_c_vuln = df_c_exp.dropna(subset=['inform_vulnerability']).copy()

  rho_zup_exp_t4, p_zup_exp_t4 = spearmanr(df_c_exp['zero_up_pct_t4'], df_c_exp['E_total'])
  rho_zup_exp_t1a, p_zup_exp_t1a = spearmanr(df_c_exp['zero_up_pct_t1a'], df_c_exp['E_total'])

  corr_records = []
  for regime_lbl, col_abs, col_pct_unw, zup_col in [
      ('000 Global Retraining Only: Tier 4 -> Tier 5 (Joint Data Bias)', 'd_t5_000_joint_clip_from_t4', 'pct_unw_t4_d_t5_000_joint', 'zero_up_pct_t4'),
      ('000 Global Retraining Only: Tier 1a -> Tier 5 (Joint Data Bias)', 'd_t5_000_clip_from_t1a', 'pct_unw_t1a_d_t5_000', 'zero_up_pct_t4'),
      ('000 Global Retraining Only: Tier 1a -> Tier 4 (121,819 Public Gauges)', 'd_t4_000_clip_from_t1a', 'pct_unw_t1a_d_t4_000', 'zero_up_pct_t4'),
      ('001 30-Day Variational DA Only: Tier 1a -> Tier 5 (Universal Gauging)', 'd_t5_001_clip_from_t1a', 'pct_unw_t1a_d_t5_001', 'zero_up_pct_t4'),
      ('001 30-Day Variational DA Only: Tier 1a -> Tier 4 (121,819 Public Gauges)', 'd_t4_001_clip_from_t1a', 'pct_unw_t1a_d_t4_001', 'zero_up_pct_t4'),
      ('101 Combined (Retraining + 30-Day DA): Tier 1a -> Tier 5 (Universal Gauging)', 'd_t5_101_clip_from_t1a', 'pct_unw_t1a_d_t5_101', 'zero_up_pct_t4'),
      ('101 Combined (Retraining + 30-Day DA): Tier 4 -> Tier 5 (Universal Gauging)', 'd_t5_101_joint_clip_from_t4', 'pct_unw_t4_d_t5_101_joint', 'zero_up_pct_t4'),
      ('101 Combined (Retraining + 30-Day DA): Tier 1a -> Tier 4 (121,819 Public Gauges)', 'd_t4_101_clip_from_t1a', 'pct_unw_t1a_d_t4_101', 'zero_up_pct_t4'),
  ]:
    r_zup_pct, p_zup_pct = spearmanr(df_c_exp[zup_col], df_c_exp[col_pct_unw])
    r_zup_abs, p_zup_abs = spearmanr(df_c_exp[zup_col], df_c_exp[col_abs])
    r_vul_pct, p_vul_pct = spearmanr(df_c_vuln['inform_vulnerability'], df_c_vuln[col_pct_unw])
    r_vul_abs, p_vul_abs = spearmanr(df_c_vuln['inform_vulnerability'], df_c_vuln[col_abs])
    corr_records.append({
        'regime': regime_lbl,
        'n_exposed_countries': int(len(df_c_exp)),
        'r_s_pct_vs_zero_up_gauges': float(r_zup_pct),
        'p_pct_vs_zero_up_gauges': float(p_zup_pct),
        'r_s_abs_vs_zero_up_gauges': float(r_zup_abs),
        'p_abs_vs_zero_up_gauges': float(p_zup_abs),
        'n_vuln_countries': int(len(df_c_vuln)),
        'r_s_pct_vs_inform_vuln': float(r_vul_pct),
        'p_pct_vs_inform_vuln': float(p_vul_pct),
        'r_s_abs_vs_inform_vuln': float(r_vul_abs),
        'p_abs_vs_inform_vuln': float(p_vul_abs),
    })

  df_corrs = pd.DataFrame(corr_records)
  df_corrs.to_csv(os.path.join(OUT_DIR, 'country_correlations_lead1.csv'), index=False)

  # 5. Render FigS4_lead1.png (4-Panel Flood Exposure & Baseline Warning Coverage at Lead Time 1)
  print('5. Rendering FigS4_lead1.png...', flush=True)
  global_tot_exp = float(E_tot.sum())
  global_warned_t4 = float(W_b4.sum())
  global_unw_t4 = float(U_b4.sum())
  global_warned_t1a = float(W_b1.sum())
  global_unw_t1a = float(U_b1.sum())

  fig_s4 = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 4.95))
  gs_s4 = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.05], hspace=0.28, wspace=0.16)

  # Panel (a): National Flood Warning Systems
  ax_s4a = fig_s4.add_subplot(gs_s4[0, 0])
  w_s4a = world_gdf.copy()
  cap_map = df_c.set_index('country_name')['has_system'].to_dict()
  w_s4a['has_cap'] = w_s4a['map_name'].map(cap_map)
  world_gdf.plot(ax=ax_s4a, color='#ECEFF1', edgecolor='#B0BEC5', linewidth=0.20)
  w_s4a[w_s4a['has_cap'] == True].plot(ax=ax_s4a, color='#0072B2', edgecolor='#263238', linewidth=0.18, alpha=0.86)
  w_s4a[w_s4a['has_cap'] == False].plot(ax=ax_s4a, color='#D55E00', edgecolor='#263238', linewidth=0.18, alpha=0.84)
  ax_s4a.set_xlim([-180, 180])
  ax_s4a.set_ylim([-58, 84])
  ax_s4a.set_axis_off()
  ax_s4a.set_title('(a) National flood warning systems', fontsize=8.0, fontweight='bold', loc='left', pad=3)
  handles_s4a = [
      Patch(facecolor='#0072B2', edgecolor='#263238', alpha=0.86, label='Established system (60 countries)'),
      Patch(facecolor='#D55E00', edgecolor='#263238', alpha=0.84, label='Lacking system (113 countries)'),
  ]
  ax_s4a.legend(handles=handles_s4a, loc='lower left', frameon=True, framealpha=0.94, fontsize=6.2)

  # Panel (b): Baseline Expected Annual Flood Exposure
  df_ws_pts = exp_df.dropna(subset=['lat', 'lon']).copy()
  df_ws_zero = df_ws_pts[df_ws_pts['E_total'] <= 0]
  df_ws_pos = df_ws_pts[df_ws_pts['E_total'] > 0].sort_values('E_total', ascending=True)
  df_ws_pos['pct_warned_ws'] = (100.0 * df_ws_pos['warned_t4'] / df_ws_pos['E_total']).clip(0.0, 100.0)

  base_ylgnbu = plt.get_cmap('YlGnBu')
  cmap_exp = mcolors.LinearSegmentedColormap.from_list('YlGnBu_trunc', base_ylgnbu(np.linspace(0.24, 1.0, 256)))

  ax_s4b = fig_s4.add_subplot(gs_s4[0, 1])
  world_gdf.plot(ax=ax_s4b, color='#F4F6F8', edgecolor='none', zorder=1)
  ax_s4b.scatter(df_ws_zero['lon'].values, df_ws_zero['lat'].values, s=0.04, marker='.', color='#D8DFE6', linewidths=0, rasterized=True, zorder=2)
  sc_s4b = ax_s4b.scatter(
      df_ws_pos['lon'].values, df_ws_pos['lat'].values,
      c=np.maximum(df_ws_pos['E_total'].values, 0.1), s=0.08, marker='.',
      cmap=cmap_exp, norm=mcolors.LogNorm(vmin=1.0, vmax=1e5),
      linewidths=0, rasterized=True, zorder=3
  )
  world_gdf.boundary.plot(ax=ax_s4b, color='#455A64', linewidth=0.18, zorder=4)
  cbar_s4b = fig_s4.colorbar(sc_s4b, ax=ax_s4b, orientation='horizontal', fraction=0.052, pad=0.02, shrink=0.84, extend='both')
  cbar_s4b.set_label(f'Flood exposure (persons/yr; total {global_tot_exp/1e6:.1f}M/yr)', fontsize=6.6, fontweight='bold')
  cbar_s4b.ax.tick_params(labelsize=6.2)
  ax_s4b.set_xlim([-180, 180])
  ax_s4b.set_ylim([-58, 84])
  ax_s4b.set_axis_off()
  ax_s4b.set_title('(b) Baseline annual flood exposure', fontsize=8.0, fontweight='bold', loc='left', pad=3)

  # Panel (c): Zero Upstream Gauges vs. Flood Exposure
  ax_s4c = fig_s4.add_subplot(gs_s4[1, 0])
  for has_sys_flag, lbl, col, mk in [
      (True, 'Established system', '#0072B2', 'o'),
      (False, 'Lacking system', '#D55E00', '^'),
  ]:
    sub = df_c_exp[df_c_exp['has_system'] == has_sys_flag]
    ax_s4c.scatter(
        sub['zero_up_pct_t4'], sub['E_total'],
        s=16, color=col, marker=mk, alpha=0.78, edgecolor='black', linewidth=0.35, label=lbl
    )

  x_grid = np.linspace(0, 100, 200)
  log_y = np.log10(np.maximum(df_c_exp['E_total'].values, 1.0))
  slope, intercept = np.polyfit(df_c_exp['zero_up_pct_t4'].values, log_y, 1)
  ax_s4c.plot(x_grid, 10**(intercept + slope * x_grid), color='#263238', linestyle='--', linewidth=1.0)

  for cname, label_txt, dx_pt, dy_pt in [
      ('India', 'India', 7, 2),
      ('Bangladesh', 'Bangladesh', -38, -14),
      ('China', 'China', -32, 8),
      ('Brazil', 'Brazil', -42, -11),
      ('Nigeria', 'Nigeria', 6, 6),
      ('United States of America', 'USA', -26, -14),
  ]:
    rc = df_c_exp[df_c_exp['country_name'] == cname]
    if not rc.empty:
      xv = float(rc['zero_up_pct_t4'].iloc[0])
      yv = float(rc['E_total'].iloc[0])
      ax_s4c.annotate(
          label_txt, xy=(xv, yv), xytext=(dx_pt, dy_pt),
          textcoords='offset points',
          fontsize=6.2, fontweight='bold', color='#263238',
          arrowprops=dict(arrowstyle='-', color='#546E7A', lw=0.4, shrinkA=1, shrinkB=1)
      )

  ax_s4c.set_yscale('log')
  ax_s4c.set_xlim(-4, 108)
  ax_s4c.set_ylim(5.0, 4.5e8)
  ax_s4c.text(
      0.03, 0.06,
      f'Spearman $r_s = {rho_zup_exp_t4:+.2f}$ ($p = {p_zup_exp_t4:.3f}$, $N = {len(df_c_exp)}$)',
      transform=ax_s4c.transAxes, fontsize=6.4,
      bbox=dict(boxstyle='round,pad=0.22', facecolor='#FFF8E1', edgecolor='#B0BEC5', alpha=0.94)
  )
  ax_s4c.legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=6.2)
  ax_s4c.set_xlabel('Watersheds with zero upstream gauges (%)', fontsize=7.0, fontweight='bold')
  ax_s4c.set_ylabel('Flood exposure (persons/yr)', fontsize=7.2, fontweight='bold')
  ax_s4c.set_title('(c) Zero upstream gauges vs. flood exposure', fontsize=8.0, fontweight='bold', loc='left', pad=3)
  ax_s4c.grid(True, which='both', linestyle='--', alpha=0.4)
  sns.despine(ax=ax_s4c)

  # Panel (d): Expected Flood Exposure Warned by Model (Lead Time 1)
  ax_s4d = fig_s4.add_subplot(gs_s4[1, 1])
  world_gdf.plot(ax=ax_s4d, color='#F4F6F8', edgecolor='none', zorder=1)
  ax_s4d.scatter(df_ws_zero['lon'].values, df_ws_zero['lat'].values, s=0.04, marker='.', color='#D8DFE6', linewidths=0, rasterized=True, zorder=2)
  sc_s4d = ax_s4d.scatter(
      df_ws_pos['lon'].values, df_ws_pos['lat'].values,
      c=df_ws_pos['pct_warned_ws'].values, s=0.08, marker='.',
      cmap='viridis', vmin=10.0, vmax=50.0,
      linewidths=0, rasterized=True, zorder=3
  )
  world_gdf.boundary.plot(ax=ax_s4d, color='#455A64', linewidth=0.18, zorder=4)
  pct_global_warned_t4 = (global_warned_t4 / global_tot_exp) * 100.0
  cbar_s4d = fig_s4.colorbar(sc_s4d, ax=ax_s4d, orientation='horizontal', fraction=0.052, pad=0.02, shrink=0.84, extend='both')
  cbar_s4d.set_label(
      f'Exposure warned at α = 1.0 (%; global {pct_global_warned_t4:.1f}%)',
      fontsize=6.6, fontweight='bold'
  )
  cbar_s4d.ax.tick_params(labelsize=6.2)
  ax_s4d.set_xlim([-180, 180])
  ax_s4d.set_ylim([-58, 84])
  ax_s4d.set_axis_off()
  ax_s4d.set_title('(d) Flood exposure warned by global model', fontsize=8.0, fontweight='bold', loc='left', pad=3)

  s4_png = os.path.join(OUT_DIR, 'FigS4_lead1.png')
  s4_pdf = os.path.join(OUT_DIR, 'FigS4_lead1.pdf')
  fig_s4.savefig(s4_png, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.03)
  fig_s4.savefig(s4_pdf, bbox_inches='tight', pad_inches=0.03)
  plt.close(fig_s4)
  print(f'Saved {s4_png}', flush=True)

  # 6. Render FigS5_lead1.png (2-Panel Country Data-Bias Gap vs Zero Upstream Gauges & Vulnerability)
  print('6. Rendering FigS5_lead1.png...', flush=True)
  fig_s5 = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 2.95))
  gs_s5 = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.0], wspace=0.22)

  df_c_bias = df_c[df_c['unwarned_t4'] > 0].copy()
  df_c_bias['bias_pct_unw'] = df_c_bias['pct_unw_t4_d_t5_000_joint']
  df_c_bias['bias_abs'] = df_c_bias['d_t5_000_joint_clip_from_t4']

  ax_s5a = fig_s5.add_subplot(gs_s5[0, 0])
  rho_bcov, p_bcov = spearmanr(df_c_bias['zero_up_pct_t4'], df_c_bias['bias_pct_unw'])
  for has_sys_flag, lbl, col, mk in [
      (True, 'Established system', '#0072B2', 'o'),
      (False, 'Lacking system', '#D55E00', '^'),
  ]:
    sub = df_c_bias[df_c_bias['has_system'] == has_sys_flag]
    sizes = np.clip(np.sqrt(sub['E_total']) / 45.0, 10.0, 95.0)
    ax_s5a.scatter(sub['zero_up_pct_t4'], sub['bias_pct_unw'], s=sizes, color=col, marker=mk, alpha=0.75, edgecolor='black', linewidth=0.35, label=lbl)

  sns.regplot(data=df_c_bias, x='zero_up_pct_t4', y='bias_pct_unw', scatter=False, ax=ax_s5a, color='#263238', line_kws={'linestyle': '--', 'linewidth': 1.0}, ci=95)
  for cname, label_txt, dx_pt, dy_pt in [
      ('India', 'India', -34, 16),
      ('Bangladesh', 'Bangladesh', -56, 4),
      ('Pakistan', 'Pakistan', -32, 24),
      ('Somalia', 'Somalia', -34, 28),
      ('China', 'China', -38, 6),
      ('Brazil', 'Brazil', -42, -18),
      ('Nigeria', 'Nigeria', -58, -10),
      ('United States of America', 'USA', -28, 10),
  ]:
    rc = df_c_bias[df_c_bias['country_name'] == cname]
    if not rc.empty:
      xv = float(rc['zero_up_pct_t4'].iloc[0])
      yv = float(rc['bias_pct_unw'].iloc[0])
      ax_s5a.annotate(
          label_txt, xy=(xv, yv), xytext=(dx_pt, dy_pt),
          textcoords='offset points',
          fontsize=6.2, fontweight='bold', color='#263238',
          arrowprops=dict(arrowstyle='-', color='#546E7A', lw=0.4, shrinkA=1, shrinkB=1)
      )

  ax_s5a.text(
      0.03, 0.90,
      f'$r_s = {rho_bcov:+.2f}$ ($p = {_fmt_sci_tex(p_bcov)}$)',
      transform=ax_s5a.transAxes, fontsize=6.3,
      bbox=dict(boxstyle='round,pad=0.20', facecolor='#FFF8E1', edgecolor='#B0BEC5', alpha=0.94)
  )
  ax_s5a.set_xlabel('Watersheds with zero upstream gauges (%)', fontsize=7.0, fontweight='bold')
  ax_s5a.set_ylabel('Unwarned exposure from data bias (%)', fontsize=7.2, fontweight='bold')
  ax_s5a.set_title('(a) Data-bias gap vs. zero upstream gauges', fontsize=8.0, fontweight='bold', loc='left', pad=3)
  ax_s5a.legend(loc='lower right', fontsize=6.1, frameon=True, framealpha=0.94)
  ax_s5a.grid(True, linestyle='--', alpha=0.45)
  sns.despine(ax=ax_s5a)

  ax_s5b = fig_s5.add_subplot(gs_s5[0, 1])
  df_c_bv = df_c_bias.dropna(subset=['inform_vulnerability']).copy()
  rho_bv_pct, p_bv_pct = spearmanr(df_c_bv['inform_vulnerability'], df_c_bv['bias_pct_unw'])
  rho_bv_abs, p_bv_abs = spearmanr(df_c_bv['inform_vulnerability'], df_c_bv['bias_abs'])
  for has_sys_flag, lbl, col, mk in [
      (True, 'Established system', '#0072B2', 'o'),
      (False, 'Lacking system', '#D55E00', '^'),
  ]:
    sub = df_c_bv[df_c_bv['has_system'] == has_sys_flag]
    sizes = np.clip(np.sqrt(sub['E_total']) / 45.0, 10.0, 95.0)
    ax_s5b.scatter(sub['inform_vulnerability'], sub['bias_pct_unw'], s=sizes, color=col, marker=mk, alpha=0.75, edgecolor='black', linewidth=0.35, label=lbl)

  sns.regplot(data=df_c_bv, x='inform_vulnerability', y='bias_pct_unw', scatter=False, ax=ax_s5b, color='#263238', line_kws={'linestyle': '--', 'linewidth': 1.0}, ci=95)
  for cname, label_txt, dx_pt, dy_pt in [
      ('United States of America', 'USA', -26, 16),
      ('China', 'China', -8, 14),
      ('India', 'India', -24, 14),
      ('Bangladesh', 'Bangladesh', -32, 14),
      ('Pakistan', 'Pakistan', -42, -16),
      ('Nigeria', 'Nigeria', 8, -14),
      ('Somalia', 'Somalia', -38, 16),
  ]:
    rc = df_c_bv[df_c_bv['country_name'] == cname]
    if not rc.empty:
      xv = float(rc['inform_vulnerability'].iloc[0])
      yv = float(rc['bias_pct_unw'].iloc[0])
      ax_s5b.annotate(
          label_txt, xy=(xv, yv), xytext=(dx_pt, dy_pt),
          textcoords='offset points',
          fontsize=6.2, fontweight='bold', color='#263238',
          arrowprops=dict(arrowstyle='-', color='#546E7A', lw=0.4, shrinkA=1, shrinkB=1)
      )

  ax_s5b.text(
      0.03, 0.82,
      f'% gap: $r_s = {rho_bv_pct:+.2f}$ ($p = {_fmt_sci_tex(p_bv_pct)}$)\n'
      f'Persons/yr: $r_s = {rho_bv_abs:+.2f}$ ($p = {_fmt_sci_tex(p_bv_abs)}$)',
      transform=ax_s5b.transAxes, fontsize=6.2,
      bbox=dict(boxstyle='round,pad=0.20', facecolor='#FFF8E1', edgecolor='#B0BEC5', alpha=0.94)
  )
  ax_s5b.set_xlabel('INFORM 2026 Vulnerability Index', fontsize=7.2, fontweight='bold')
  ax_s5b.set_ylabel('Unwarned exposure from data bias (%)', fontsize=7.2, fontweight='bold')
  ax_s5b.set_title('(b) Data-bias gap vs. vulnerability', fontsize=8.0, fontweight='bold', loc='left', pad=3)
  ax_s5b.legend(loc='lower right', fontsize=6.2, frameon=True, framealpha=0.94)
  ax_s5b.grid(True, linestyle='--', alpha=0.45)
  sns.despine(ax=ax_s5b)

  s5_png = os.path.join(OUT_DIR, 'FigS5_lead1.png')
  s5_pdf = os.path.join(OUT_DIR, 'FigS5_lead1.pdf')
  fig_s5.savefig(s5_png, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.03)
  fig_s5.savefig(s5_pdf, bbox_inches='tight', pad_inches=0.03)
  plt.close(fig_s5)
  print(f'Saved {s5_png}', flush=True)

  # 7. Write Table_S3_lead1.md
  md_lines = [
      '# Table S3 & Country-Level Correlations at Lead Time 1 ($t_0 + 1$ Day)',
      '',
      '## 1. Global Summary Totals Across 964,084 Watersheds (173 Countries)',
      f'- **Total Expected Annual Riverine Flood Exposure**: **{global_tot_exp:,.0f} persons/yr** (100.0%)',
      f'  - In 60 countries with established warning systems (617,831 watersheds): **{float(exp_df.loc[has_sys, "E_total"].sum()):,.0f} persons/yr** ({100.0*float(exp_df.loc[has_sys, "E_total"].sum())/global_tot_exp:.1f}%)',
      f'  - In 113 countries lacking warning systems (346,253 watersheds): **{float(exp_df.loc[~has_sys, "E_total"].sum()):,.0f} persons/yr** ({100.0*float(exp_df.loc[~has_sys, "E_total"].sum())/global_tot_exp:.1f}%)',
      f'- **Baseline at Tier 1a (8,811 Post-2016 Caravan Gauges, `000` No DA)**:',
      f'  - Warned at $\\alpha = 1.0$: **{global_warned_t1a:,.0f} persons/yr** ({100.0*global_warned_t1a/global_tot_exp:.2f}%)',
      f'  - Unwarned at $\\alpha = 1.0$: **{global_unw_t1a:,.0f} persons/yr** ({100.0*global_unw_t1a/global_tot_exp:.2f}%) [With warning sys: **{float(exp_df.loc[has_sys, "unwarned_t1a"].sum()):,.0f}/yr**; Lacking sys: **{float(exp_df.loc[~has_sys, "unwarned_t1a"].sum()):,.0f}/yr**]',
      f'- **Baseline at Tier 4 (121,819 All Publicly Knowable Gauges, `000` No DA)**:',
      f'  - Warned at $\\alpha = 1.0$: **{global_warned_t4:,.0f} persons/yr** ({100.0*global_warned_t4/global_tot_exp:.2f}%)',
      f'  - Unwarned at $\\alpha = 1.0$: **{global_unw_t4:,.0f} persons/yr** ({100.0*global_unw_t4/global_tot_exp:.2f}%) [With warning sys: **{float(exp_df.loc[has_sys, "unwarned_t4"].sum()):,.0f}/yr**; Lacking sys: **{float(exp_df.loc[~has_sys, "unwarned_t4"].sum()):,.0f}/yr**]',
      '',
      '### Global Exposure Reductions at Tier 4 and Tier 5 (Lead Time 1)',
      '',
      '| Counterfactual Intervention | Global Clipped $\\Delta E$ (persons/yr) | % of Unwarned | Global Net $\\Delta E$ (persons/yr) | With Warning Systems (60) Clipped $\\Delta E$ | Lacking Warning Systems (113) Clipped $\\Delta E$ |',
      '| :--- | :---: | :---: | :---: | :---: | :---: |',
  ]

  for lbl, col_c, col_n, unw_denom in [
      ('**Tier 1a -> Tier 4 (`000` Global Retraining Only)**', 'd_t4_000_clip_from_t1a', 'd_t4_000_net_from_t1a', global_unw_t1a),
      ('**Tier 1a -> Tier 4 (`001` 30-Day DA Only)**', 'd_t4_001_clip_from_t1a', 'd_t4_001_net_from_t1a', global_unw_t1a),
      ('**Tier 1a -> Tier 4 (`101` Combined Retraining + DA)**', 'd_t4_101_clip_from_t1a', 'd_t4_101_net_from_t1a', global_unw_t1a),
      ('**Tier 1a -> Tier 5 (`000` Global Retraining Only: Joint)**', 'd_t5_000_clip_from_t1a', 'd_t5_000_net_from_t1a', global_unw_t1a),
      ('  - Factor 2 Only (100% Gauge Coverage, `000` from Tier 1a)', 'd_t5_000_f2_clip_from_t1a', None, global_unw_t1a),
      ('  - Factor 3 Only (In-Distribution Climate, `000` from Tier 1a)', 'd_t5_000_f3_clip_from_t1a', None, global_unw_t1a),
      ('  - Factor 4 Only (50-Yr Record Length, `000` from Tier 1a)', 'd_t5_000_f4_clip_from_t1a', None, global_unw_t1a),
      ('**Tier 1a -> Tier 5 (`001` 30-Day DA Only)**', 'd_t5_001_clip_from_t1a', 'd_t5_001_net_from_t1a', global_unw_t1a),
      ('**Tier 1a -> Tier 5 (`101` Combined Retraining + DA)**', 'd_t5_101_clip_from_t1a', 'd_t5_101_net_from_t1a', global_unw_t1a),
      ('**Tier 4 -> Tier 5 (`000` Global Retraining Only: Joint)**', 'd_t5_000_joint_clip_from_t4', None, global_unw_t4),
      ('  - Factor 2 Only (100% Gauge Coverage, `000` from Tier 4)', 'd_t5_000_f2_clip_from_t4', None, global_unw_t4),
      ('  - Factor 3 Only (In-Distribution Climate, `000` from Tier 4)', 'd_t5_000_f3_clip_from_t4', None, global_unw_t4),
      ('  - Factor 4 Only (50-Yr Record Length, `000` from Tier 4)', 'd_t5_000_f4_clip_from_t4', None, global_unw_t4),
      ('**Tier 4 -> Tier 5 (`001` 30-Day DA Only)**', 'd_t5_001_clip_from_t4', None, global_unw_t4),
      ('**Tier 4 -> Tier 5 (`101` Combined Retraining + DA)**', 'd_t5_101_joint_clip_from_t4', None, global_unw_t4),
  ]:
    tot_c = float(exp_df[col_c].sum())
    tot_n_str = f"{float(exp_df[col_n].sum()):+,.0f}" if col_n else '—'
    sys_c = float(exp_df.loc[has_sys, col_c].sum())
    nosys_c = float(exp_df.loc[~has_sys, col_c].sum())
    md_lines.append(
        f"| {lbl} | {tot_c:,.0f} | {100.0*tot_c/unw_denom:.2f}% | {tot_n_str} | {sys_c:,.0f} | {nosys_c:,.0f} |"
    )

  # Section 2: Country-Level Correlations
  md_lines.extend([
      '',
      '---',
      '',
      '## 2. Country-Level Spearman Rank Correlations (`Fig. S4(c)` and `Fig. S5(a),(b)`)',
      '',
      f'- **`Fig. S4(c)` (% Watersheds with Zero Upstream Gauges vs. Baseline Annual Flood Exposure, $N = {len(df_c_exp)}$ flood-exposed countries)**:',
      f'  - Using Tier 4 (121,819 gauges) zero-upstream-gauge %: **Spearman $r_s = {rho_zup_exp_t4:+.3f}$ (two-sided $p = {fmt_p_plain(p_zup_exp_t4)}$)**.',
      f'  - Using Tier 1a (8,811 gauges) zero-upstream-gauge %: **Spearman $r_s = {rho_zup_exp_t1a:+.3f}$ (two-sided $p = {fmt_p_plain(p_zup_exp_t1a)}$)**.',
      '',
      '### `Fig. S5(a),(b)` Correlations Across Mechanisms and Baselines',
      '',
      '| Regime / Counterfactual Definition | vs. % Zero Upstream Gauges ($N=160$): % Unwarned $r_s$ ($p$-val) | vs. % Zero Upstream Gauges ($N=160$): Persons/yr $r_s$ ($p$-val) | vs. INFORM 2026 Vulnerability ($N=155$): % Unwarned $r_s$ ($p$-val) | vs. INFORM 2026 Vulnerability ($N=155$): Persons/yr $r_s$ ($p$-val) |',
      '| :--- | :---: | :---: | :---: | :---: |',
  ])

  for _, cr in df_corrs.iterrows():
    md_lines.append(
        f"| {cr['regime']} | "
        f"**{cr['r_s_pct_vs_zero_up_gauges']:+.3f}** ($p={fmt_p_plain(cr['p_pct_vs_zero_up_gauges'])}$) | "
        f"**{cr['r_s_abs_vs_zero_up_gauges']:+.3f}** ($p={fmt_p_plain(cr['p_abs_vs_zero_up_gauges'])}$) | "
        f"**{cr['r_s_pct_vs_inform_vuln']:+.3f}** ($p={fmt_p_plain(cr['p_pct_vs_inform_vuln'])}$) | "
        f"**{cr['r_s_abs_vs_inform_vuln']:+.3f}** ($p={fmt_p_plain(cr['p_abs_vs_inform_vuln'])}$) |"
    )

  # Section 3: Full 173-Country Table S3 (Multi-Mechanism Tier 4 & Tier 5 + Legacy Factor Columns)
  md_lines.extend([
      '',
      '---',
      '',
      '## 3. Table S3 (Part A): Country-Level Flood Exposure and Multi-Mechanism Warning Gains at Tier 4 and Tier 5 (All 173 Countries, Sorted by Baseline Exposure)',
      '',
      '| Country | Watersheds | Flood Warning System | Baseline Exposure (persons/yr) | Unwarned Exposure at $\\alpha=1.0$, Tier 1a (persons/yr) | Unwarned, Tier 1a (%) | Tier 4 (`000` Retrain) Reduction | Tier 4 (`001` DA Only) Reduction | Tier 4 (`101` Combined) Reduction (persons/yr / % unw) | Tier 5 (`000` Retrain) Reduction | Tier 5 (`001` DA Only) Reduction | Tier 5 (`101` Combined) Reduction (persons/yr / % unw) | INFORM 2026 Vulnerability |',
      '| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |',
  ])

  for _, r in df_c.iterrows():
    vul_str = f"{r['inform_vulnerability']:.1f}" if pd.notna(r['inform_vulnerability']) else '—'
    sys_str = 'Yes' if r['has_system'] else 'No'
    md_lines.append(
        f"| {r['country_name']} | {int(r['n_basins']):,} | {sys_str} | "
        f"{r['E_total']:,.0f} | {r['unwarned_t1a']:,.0f} | {r['unwarned_pct_t1a']:.1f}% | "
        f"{r['d_t4_000_clip_from_t1a']:,.0f} ({r['pct_unw_t1a_d_t4_000']:.1f}%) | "
        f"{r['d_t4_001_clip_from_t1a']:,.0f} ({r['pct_unw_t1a_d_t4_001']:.1f}%) | "
        f"{r['d_t4_101_clip_from_t1a']:,.0f} ({r['pct_unw_t1a_d_t4_101']:.1f}%) | "
        f"{r['d_t5_000_clip_from_t1a']:,.0f} ({r['pct_unw_t1a_d_t5_000']:.1f}%) | "
        f"{r['d_t5_001_clip_from_t1a']:,.0f} ({r['pct_unw_t1a_d_t5_001']:.1f}%) | "
        f"{r['d_t5_101_clip_from_t1a']:,.0f} ({r['pct_unw_t1a_d_t5_101']:.1f}%) | "
        f"{vul_str} |"
    )

  md_lines.extend([
      '',
      '---',
      '',
      '## 4. Table S3 (Part B): Country-Level Factor Disaggregation (`Tier 4 -> Tier 5`, Matching Current Manuscript `Table S3` Columns)',
      '',
      '| Country | Watersheds | Flood Warning System | Baseline Exposure (persons/yr) | Unwarned Exposure at $\\alpha = 1.0$ (persons/yr) | Unwarned (%) | Reduction: 100% Gauge Coverage (persons/yr) | Reduction: In-Distribution Climate (persons/yr) | Reduction: Combined `000` (persons/yr) | Reduction: Combined `000` (% of Unwarned) | Reduction: Combined `101` w/ DA (persons/yr / % unw) | INFORM 2026 Vulnerability Index |',
      '| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |',
  ])

  for _, r in df_c.iterrows():
    vul_str = f"{r['inform_vulnerability']:.1f}" if pd.notna(r['inform_vulnerability']) else '—'
    sys_str = 'Yes' if r['has_system'] else 'No'
    md_lines.append(
        f"| {r['country_name']} | {int(r['n_basins']):,} | {sys_str} | "
        f"{r['E_total']:,.0f} | {r['unwarned_t4']:,.0f} | {r['unwarned_pct_t4']:.1f}% | "
        f"{r['d_t5_000_f2_clip_from_t4']:,.0f} | "
        f"{r['d_t5_000_f3_clip_from_t4']:,.0f} | "
        f"{r['d_t5_000_joint_clip_from_t4']:,.0f} | "
        f"{r['pct_unw_t4_d_t5_000_joint']:.1f}% | "
        f"{r['d_t5_101_joint_clip_from_t4']:,.0f} ({r['pct_unw_t4_d_t5_101_joint']:.1f}%) | "
        f"{vul_str} |"
    )

  md_s3 = os.path.join(OUT_DIR, 'Table_S3_lead1.md')
  with open(md_s3, 'w') as f:
    f.write('\n'.join(md_lines) + '\n')
  print(f'Saved {md_s3} in {time.time() - t0:.1f}s!', flush=True)


if __name__ == '__main__':
  main()
