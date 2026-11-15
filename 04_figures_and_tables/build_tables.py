#!/usr/bin/env python3
"""Build publication tables (Table 1 and Tables S1–S5; Fathom-free by default).

Produces strictly non-Fathom outputs in `--out-dir` (default: `$GEO_BIAS_DATA_DIR/round2/tables`):
  - Table_1_ungaugedness_ranges.csv: Multi-hydrofabric min–max ranges across six
    fine-scale global river maps (U_never, U_net_Eq4, and 1D flow-weighted discharge).
    Optional Fathom exposure rows are appended ONLY if `exposure_eae` exists in
    `master_2d_metrics.parquet`.
  - Table_S1_gauge_status.csv: Mutually exclusive operational & temporal status
    breakdown of 34,577 / 93,827 / 121,819 stream gauges under the uniform 2016 cutoff.
  - Table_S2_175territories.csv (+ Table_S2_summary_stats.json): Country/territory-level
    HydroSHEDS L12 basin counts, alerting status, fractional land & population
    ungaugedness (%), standardized open gauges inside borders, and INFORM 2026 scores.
    Optional Fathom exposure columns are merged ONLY if local Stage D exposure outputs exist.
  - Table_S3_4regimes_skill.csv: Observed Caravan CV skill, meta-model CV R², and
    1,034,083-basin predicted skill across all 43 hydrological metrics and 4 regimes.
  - Table_S4_features_95.csv: Complete catalog of the 95 meta-model input features
    grouped into the seven attribution categories.
  - Table_S5_ungaugedness_breakdown.csv: Global, regional, Strahler stream-order
    (1..8+ and 7+), and data-accessibility tier breakdown across 1,034,083 basins
    (135.01M km², 45,633 km³/yr annual runoff).
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys
import warnings

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = Path(os.environ.get('GEO_BIAS_REPO_DIR', str(SCRIPT_DIR.parent)))
if str(REPO_DIR) not in sys.path:
  sys.path.insert(0, str(REPO_DIR))

CANON_DIR = Path(os.environ.get('GEO_BIAS_DATA_DIR', str(REPO_DIR / 'data')))
os.environ.setdefault('GEO_BIAS_DATA_DIR', str(CANON_DIR))
os.environ.setdefault('GEO_BIAS_THIRD_PARTY_DIR', str(CANON_DIR))
os.environ.setdefault('GEO_BIAS_OUTPUT_DIR', str(CANON_DIR))

HYDRO_DATA_DIR = Path(
    os.environ.get(
        'GLOBAL_UNGAUGEDNESS_DATA_DIR',
        str(REPO_DIR / '01_hydrofabric_ungaugedness' / 'data'),
    )
)
HYDRO_RESULTS_DIR = Path(
    os.environ.get(
        'GLOBAL_UNGAUGEDNESS_RESULTS_DIR',
        str(REPO_DIR / '01_hydrofabric_ungaugedness' / 'results'),
    )
)

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from utils import cap_utils, feature_utils, file_paths

warnings.filterwarnings('ignore')

R2_DIR = CANON_DIR / 'round2'
TIER_DIR = R2_DIR / 'tier_features'
CF_DIR = R2_DIR / 'counterfactuals'
META_DIR = R2_DIR / 'meta_models'
DEFAULT_OUT_DIR = R2_DIR / 'tables'

FINE_6_FABRICS = [
    'hydrosheds_l12',
    'hydrosheds_rivers',
    'merit_basins',
    'tdx_hydro',
    'grit_v06_mainstem_tree',
    'grit_v06_bifurcating_dag',
]
FLOW_4_FABRICS = [
    'hydrosheds_l12',
    'hydrosheds_rivers',
    'merit_basins',
    'tdx_hydro',
]
FLOW_3_1D_FABRICS = [
    'hydrosheds_rivers',
    'merit_basins',
    'tdx_hydro',
]
TIERS_HYDRO = [
    ('tier1_open_compilations', 'tier1'),
    ('tier2_public_discharge', 'tier2'),
    ('tier3_all_known', 'tier3'),
]


def _norm_hybas(s: pd.Series | pd.Index) -> np.ndarray:
  s2 = pd.Series(s).astype(str).str.replace(r'\.0$', '', regex=True)
  return np.where(s2.str.startswith('hybas_'), s2, 'hybas_' + s2)


def _load_topology_and_runoff() -> pd.DataFrame:
  topo_pq = CANON_DIR / 'attributes' / 'hybas_l12_topology_and_runoff.parquet'
  if topo_pq.exists():
    df = pd.read_parquet(topo_pq)
    df['HYBAS_ID'] = df['HYBAS_ID'].astype(np.int64)
    df['NEXT_DOWN'] = df['NEXT_DOWN'].astype(np.int64)
    df['continent'] = df['continent'].astype(str).str.lower()
    df['area_m2'] = pd.to_numeric(df['area_m2'], errors='coerce').fillna(0.0)
    if 'SUB_AREA' not in df.columns:
      df['SUB_AREA'] = df['area_m2'] / 1e6
    df['run_mm_syr'] = pd.to_numeric(df['run_mm_syr'], errors='coerce').fillna(0.0).clip(lower=0.0)
    return df

  topo_csv = CANON_DIR / 'attributes' / 'hybas_routing_topology.csv'
  df = pd.read_csv(topo_csv, dtype={'HYBAS_ID': str, 'NEXT_DOWN': str, 'continent': str})
  df['HYBAS_ID'] = df['HYBAS_ID'].astype(np.int64)
  df['NEXT_DOWN'] = df['NEXT_DOWN'].astype(np.int64)
  df['continent'] = df['continent'].astype(str).str.lower()
  df['area_m2'] = pd.to_numeric(df['area_m2'], errors='coerce').fillna(0.0)

  hatt = pd.read_csv(
      CANON_DIR / 'attributes' / 'hybas_attributes.csv',
      usecols=['Unnamed: 0', 'HYDRO_ATLAS:SUB_AREA', 'HYDRO_ATLAS:run_mm_syr'],
      dtype={'Unnamed: 0': str},
  )
  hatt['HYBAS_ID'] = hatt['Unnamed: 0'].str.replace('hybas_', '', regex=False).astype(np.int64)
  hatt = hatt.drop_duplicates('HYBAS_ID').set_index('HYBAS_ID')
  df['SUB_AREA'] = pd.to_numeric(
      pd.Series(hatt['HYDRO_ATLAS:SUB_AREA'].reindex(df['HYBAS_ID']).values, index=df.index),
      errors='coerce',
  ).fillna(df['area_m2'] / 1e6)
  df['run_mm_syr'] = pd.to_numeric(
      pd.Series(hatt['HYDRO_ATLAS:run_mm_syr'].reindex(df['HYBAS_ID']).values, index=df.index),
      errors='coerce',
  ).fillna(0.0).clip(lower=0.0)
  return df


def _load_worldpop_pop(all_ids: list[str]) -> pd.Series:
  candidates = [
      CANON_DIR / 'attributes' / 'hydrosheds_l12_worldpop_population.parquet',
      HYDRO_DATA_DIR / 'exposure' / 'hydrosheds_l12_worldpop_population.parquet',
      REPO_DIR / '01_hydrofabric_ungaugedness' / 'data' / 'exposure' / 'hydrosheds_l12_worldpop_population.parquet',
  ]
  for p in candidates:
    if p.exists():
      wp = pd.read_parquet(p)
      idc = 'HYBAS_ID' if 'HYBAS_ID' in wp.columns else wp.columns[0]
      popc = 'total_population' if 'total_population' in wp.columns else next(c for c in wp.columns if 'pop' in c.lower())
      wp.index = _norm_hybas(wp[idc])
      wp = wp[~wp.index.duplicated(keep='first')].reindex(all_ids)
      return pd.to_numeric(wp[popc], errors='coerce').fillna(0.0)
  raise FileNotFoundError(
      'Missing hydrosheds_l12_worldpop_population.parquet; checked '
      + ', '.join(str(c) for c in candidates)
  )


# ==============================================================================
# TABLE 1: Multi-Hydrofabric Global Hydrological Ungaugedness Ranges
# ==============================================================================
def build_table1(out_dir: Path) -> Path:
  print('[Table 1] Building multi-hydrofabric ungaugedness ranges...', flush=True)
  m2d_p = HYDRO_RESULTS_DIR / 'master' / 'master_2d_metrics.parquet'
  m1d_p = HYDRO_RESULTS_DIR / 'master' / 'master_1d_metrics.parquet'
  if not m2d_p.exists() or not m1d_p.exists():
    raise FileNotFoundError(
        f'Missing master hydrofabric parquets in {HYDRO_RESULTS_DIR / "master"}. '
        'Run 01_hydrofabric_ungaugedness/scripts/20_synthesize_master_results.py first.'
    )

  m2d = pd.read_parquet(m2d_p)
  m1d = pd.read_parquet(m1d_p)

  # Canonical filter over shared non-Greenland / non-Antarctica domain
  m2d_dom = m2d[
      (m2d['domain_name'] == 'common_no_grl_ata')
      & (m2d['hydrofabric'].isin(FINE_6_FABRICS))
  ].copy()
  is_dag = m2d_dom['hydrofabric'] == 'grit_v06_bifurcating_dag'
  canon_2d = m2d_dom[
      (is_dag & (m2d_dom['snapping_mode'] == 'dag_width_weighted') & (m2d_dom['ratio_basis'] == 'area_dag'))
      | ((~is_dag) & (m2d_dom['snapping_mode'] == 'outlet') & (m2d_dom['ratio_basis'] == 'area'))
  ].copy()

  has_exposure = 'exposure_eae' in set(canon_2d['weight'].unique())

  rows: list[dict] = []

  def _add_2d_row(section: str, metric: str, weight: str, label: str, fabrics: list[str]) -> None:
    sub = canon_2d[
        (canon_2d['metric'] == metric)
        & (canon_2d['weight'] == weight)
        & (canon_2d['hydrofabric'].isin(fabrics))
    ]
    rec: dict[str, object] = {
        'section': section,
        'metric': metric,
        'weight': weight,
        'measure': label,
        'n_hydrofabrics': int(sub['hydrofabric'].nunique()),
    }
    for tier_name, prefix in TIERS_HYDRO:
      vals = sub[sub['tier'] == tier_name]['value'].astype(float).values * 100.0
      if len(vals) == 0:
        rec[f'{prefix}_min_pct'] = np.nan
        rec[f'{prefix}_max_pct'] = np.nan
        rec[f'{prefix}_range'] = ''
      else:
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        rec[f'{prefix}_min_pct'] = round(vmin, 4)
        rec[f'{prefix}_max_pct'] = round(vmax, 4)
        rec[f'{prefix}_range'] = f'{vmin:.1f}–{vmax:.1f}%'
    rows.append(rec)

  _add_2d_row('0. Never Passes a Gauge', 'U_never', 'area', 'Global land area', FINE_6_FABRICS)
  _add_2d_row('0. Never Passes a Gauge', 'U_never', 'runoff', 'Global river runoff (3 flow models)', FLOW_4_FABRICS)
  _add_2d_row('0. Never Passes a Gauge', 'U_never', 'population', 'Global population', FINE_6_FABRICS)
  if has_exposure:
    _add_2d_row('0. Never Passes a Gauge', 'U_never', 'exposure_eae', 'Annual flood exposure (optional licensed Fathom)', FINE_6_FABRICS)

  _add_2d_row('1. Fractional Watershed Coverage', 'U_net_Eq4', 'area', 'Global land area', FINE_6_FABRICS)
  _add_2d_row('1. Fractional Watershed Coverage', 'U_net_Eq4', 'runoff', 'Global river runoff (3 flow models)', FLOW_4_FABRICS)
  _add_2d_row('1. Fractional Watershed Coverage', 'U_net_Eq4', 'population', 'Global population', FINE_6_FABRICS)
  if has_exposure:
    _add_2d_row('1. Fractional Watershed Coverage', 'U_net_Eq4', 'exposure_eae', 'Annual flood exposure (optional licensed Fathom)', FINE_6_FABRICS)

  # 1D flow-volume weighted discharge row
  sub_1d = m1d[
      (m1d['domain_name'] == 'common_no_grl_ata')
      & (m1d['hydrofabric'].isin(FLOW_3_1D_FABRICS))
      & (m1d['metric'] == 'U_net_1d')
      & (m1d['weight'] == 'Q_ds')
      & (m1d['ratio_basis'] == 'flow')
      & (m1d['stratum'] == 'all_cutoff_free')
  ]
  rec_1d: dict[str, object] = {
      'section': '1. Fractional Watershed Coverage',
      'metric': 'U_net_1d',
      'weight': 'Q_ds',
      'measure': 'Global river discharge (1D flow-volume weighted, 3 flow models)',
      'n_hydrofabrics': int(sub_1d['hydrofabric'].nunique()),
  }
  for tier_name, prefix in TIERS_HYDRO:
    vals = sub_1d[sub_1d['tier'] == tier_name]['value'].astype(float).values * 100.0
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    rec_1d[f'{prefix}_min_pct'] = round(vmin, 4)
    rec_1d[f'{prefix}_max_pct'] = round(vmax, 4)
    rec_1d[f'{prefix}_range'] = f'{vmin:.1f}–{vmax:.1f}%'
  rows.append(rec_1d)

  df_out = pd.DataFrame(rows)
  out_path = out_dir / 'Table_1_ungaugedness_ranges.csv'
  df_out.to_csv(out_path, index=False)
  print(f'  Saved {out_path} ({len(df_out)} rows; Fathom exposure rows included={has_exposure})', flush=True)
  return out_path


# ==============================================================================
# TABLE S1: Operational & Temporal Status of Global Stream Gauges (2016 Cutoff)
# ==============================================================================
def build_table_s1(out_dir: Path) -> Path:
  print('[Table S1] Building operational & temporal gauge status table...', flush=True)
  g = pd.read_parquet(CANON_DIR / 'attributes' / 'global_gauges_deduplicated.parquet')

  m_t1 = g['all_source_datasets'].fillna('').str.contains('caravan|estream', case=False, regex=True)
  m_t2 = (
      g['best_data_access_policy'].isin({'open_public_api_or_bulk', 'open_academic_compilation'})
      & g['best_observation_type'].isin({'daily_discharge', 'monthly_discharge'})
  )
  m_t3 = pd.Series(True, index=g.index)

  tiers = [
      ('tier1', g[m_t1]),
      ('tier2', g[m_t2]),
      ('tier3', g[m_t3]),
  ]

  def _masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    ar = df['active_reason'].astype(str)
    vy = pd.to_numeric(df['verified_record_end_year'], errors='coerce')
    c_ver_ge16 = ar.isin({'1_cv8811_verified_active', '2_end_year_ge_2016'})
    c_port_act_nodate = (ar == '3_portal_status_active') & vy.isna()
    c_ver_le15 = ar == '6_end_year_lt_2016'
    c_port_inact = ar == '5_portal_status_inactive'
    c_port_act_le15 = (ar == '3_portal_status_active') & vy.notna() & (vy < 2016)
    c_live_undated = ar == '7_option1_live_portal_undated'
    c_static_undated = ar == '8_undated_historical_or_unverified'
    c_active_group = c_ver_ge16 | c_port_act_nodate
    c_hist_group = c_ver_le15 | c_port_inact | c_port_act_le15
    c_unclass_group = c_live_undated | c_static_undated
    # Sanity check: mutually exclusive and collectively exhaustive
    total_check = c_active_group.astype(int) + c_hist_group.astype(int) + c_unclass_group.astype(int)
    assert (total_check == 1).all(), 'Status categories are not mutually exclusive & exhaustive!'
    return {
        'active_total': c_active_group,
        'verified_ge_2016': c_ver_ge16,
        'portal_active_no_end_year': c_port_act_nodate,
        'historical_or_inactive_total': c_hist_group,
        'verified_le_2015': c_ver_le15,
        'portal_inactive': c_port_inact,
        'portal_active_record_le_2015': c_port_act_le15,
        'unclassifiable_total': c_unclass_group,
        'live_portal_undated': c_live_undated,
        'static_archive_undated': c_static_undated,
        'total_distinct_gauges': pd.Series(True, index=df.index),
        'section3_assimilation_total': df['is_active_opt1'].astype(bool),
    }

  row_specs = [
      ('active_total', 'Active or recently active', True),
      ('verified_ge_2016', 'Verified streamflow record from 2016 onward', False),
      ('portal_active_no_end_year', 'Portal-listed active (no verified end year)', False),
      ('historical_or_inactive_total', 'Confirmed historical or inactive', True),
      ('verified_le_2015', 'Verified record ending in 2015 or earlier', False),
      ('portal_inactive', 'Portal-listed inactive (no post-2015 record)', False),
      ('portal_active_record_le_2015', 'Portal-listed active, but verified record ended <= 2015', False),
      ('unclassifiable_total', 'Unclassifiable (no end date or status)', True),
      ('live_portal_undated', 'Listed on live national portal without record end date', False),
      ('static_archive_undated', 'Static inventory or archive without dates or status', False),
      ('total_distinct_gauges', 'Total distinct physical gauges', True),
      ('section3_assimilation_total', 'Active + undated live-portal gauges (Section 3 assimilation)', True),
  ]

  tier_masks = {k: _masks(df) for k, df in tiers}
  tier_totals = {k: len(df) for k, df in tiers}

  rows = []
  for key, label, is_rollup in row_specs:
    rec: dict[str, object] = {'category_key': key, 'status_label': label, 'is_rollup': is_rollup}
    for k, _ in tiers:
      cnt = int(tier_masks[k][key].sum())
      pct = cnt / tier_totals[k] * 100.0
      rec[f'{k}_count'] = cnt
      rec[f'{k}_pct'] = round(pct, 4)
      pct_str = '<0.1%' if (0 < pct < 0.05) else f'{pct:.1f}%'
      rec[f'{k}_display'] = f'{cnt:,} ({pct_str})'
    rows.append(rec)

  df_out = pd.DataFrame(rows)
  out_path = out_dir / 'Table_S1_gauge_status.csv'
  df_out.to_csv(out_path, index=False)
  print(f'  Saved {out_path} (totals: {tier_totals})', flush=True)
  return out_path


# ==============================================================================
# TABLE S2: Country- and Territory-Level Summary (175 Territories; Non-Fathom)
# ==============================================================================
def build_table_s2(out_dir: Path) -> Path:
  print('[Table S2] Building 175-territory ungaugedness, open gauges & INFORM table...', flush=True)
  file_paths.init('5.0-year-2.0-window-recall')

  topo = _load_topology_and_runoff()
  all_ids = _norm_hybas(topo['HYBAS_ID']).tolist()

  df = pd.DataFrame(index=all_ids)
  df['sub_area'] = topo['SUB_AREA'].values.astype(np.float64)
  df['total_pop'] = _load_worldpop_pop(all_ids).values.astype(np.float64)

  df_map = pd.read_csv(file_paths.HYBAS_MAPPING_PATH)
  idc = 'hybas_id' if 'hybas_id' in df_map.columns else df_map.columns[0]
  df_map['hid'] = _norm_hybas(df_map[idc])
  df_map['country_name_clean'] = df_map['country_name'].replace(file_paths.country_name_map)
  df_map = df_map.drop_duplicates('hid').set_index('hid')
  df['country_name'] = df_map['country_name_clean'].reindex(all_ids)

  t4 = pd.read_parquet(TIER_DIR / 'tier4_all_publicly_knowable_121819.parquet', columns=['U_net_tree_phys'])
  t4.index = _norm_hybas(t4.index)
  df['u4'] = t4['U_net_tree_phys'].reindex(all_ids).astype(np.float64)
  df['u4_area'] = df['u4'] * df['sub_area']
  df['u4_pop'] = df['u4'] * df['total_pop']

  mapped = df[df['country_name'].notna()].copy()
  ctry = mapped.groupby('country_name').agg(
      n_basins=('sub_area', 'size'),
      total_area_km2=('sub_area', 'sum'),
      total_pop=('total_pop', 'sum'),
      u4_area=('u4_area', 'sum'),
      u4_pop=('u4_pop', 'sum'),
  ).reset_index()
  ctry['has_system'] = ctry['country_name'].map(cap_utils.has_effective_flood_warning_system).fillna(False).astype(bool)
  ctry['alerting_status'] = np.where(ctry['has_system'], 'National System', 'CAP Gap')
  ctry['land_ung_pct'] = ctry['u4_area'] / ctry['total_area_km2'] * 100.0
  ctry['pop_ung_pct'] = np.where(ctry['total_pop'] > 0, ctry['u4_pop'] / ctry['total_pop'] * 100.0, np.nan)

  # Standardized open gauges (Tier 1: Caravan + EStreams) inside Natural Earth borders
  world = gpd.read_file(file_paths.NATURAL_EARTH_PATH)
  world = world.dropna(subset=[world.geometry.name])
  world = world[world.is_valid & (~world.is_empty)].copy()
  world['map_name'] = world['name'].replace(file_paths.country_name_map)
  world['iso_clean'] = world['iso_a3'].astype(str).str.strip()
  for nm, iso in [('France', 'FRA'), ('Norway', 'NOR'), ('Kosovo', 'XKX'), ('Somaliland', 'SOL')]:
    world.loc[world['map_name'] == nm, 'iso_clean'] = iso
  c2iso = dict(zip(world['map_name'], world['iso_clean']))

  gauges = pd.read_parquet(CANON_DIR / 'attributes' / 'global_gauges_deduplicated.parquet')
  t1 = gauges[gauges['all_source_datasets'].fillna('').str.contains('caravan|estream', case=False, regex=True)].copy()
  pts = gpd.GeoDataFrame(t1, geometry=gpd.points_from_xy(t1['longitude'], t1['latitude']), crs='EPSG:4326')
  joined = gpd.sjoin(pts, world[['map_name', 'geometry']], how='left', predicate='within')
  open_cnt = joined.groupby('map_name').size()
  ctry['open_gauges_in_borders'] = ctry['country_name'].map(open_cnt).fillna(0).astype(int)

  # INFORM 2026 Vulnerability and Lack of Coping Capacity
  df_inf = pd.read_csv(file_paths.RISK_FILE, skiprows=[0, 2])
  df_inf['ISO3'] = df_inf['ISO3'].astype(str).str.strip()
  df_inf['COUNTRY_CLEAN'] = df_inf['COUNTRY'].astype(str).str.strip().replace(file_paths.country_name_map)
  for src_col, dst_col in [
      ('VULNERABILITY', 'inform_vulnerability'),
      ('LACK OF COPING CAPACITY', 'inform_lack_of_coping_capacity'),
  ]:
    df_inf[src_col] = pd.to_numeric(df_inf[src_col], errors='coerce')
    iso_m = df_inf.dropna(subset=['ISO3', src_col]).drop_duplicates('ISO3').set_index('ISO3')[src_col]
    nam_m = df_inf.dropna(subset=['COUNTRY_CLEAN', src_col]).drop_duplicates('COUNTRY_CLEAN').set_index('COUNTRY_CLEAN')[src_col]
    ctry[dst_col] = [iso_m.get(c2iso.get(c), nam_m.get(c, np.nan)) for c in ctry['country_name']]

  out_cols = [
      'country_name',
      'n_basins',
      'alerting_status',
      'has_system',
      'land_ung_pct',
      'pop_ung_pct',
      'open_gauges_in_borders',
      'inform_vulnerability',
      'inform_lack_of_coping_capacity',
  ]
  df_out = ctry[out_cols].sort_values('country_name').reset_index(drop=True)

  # Optional local Fathom exposure columns ONLY if Stage D exposure CSV exists locally
  opt_fathom_p = CF_DIR / 'Table_S3_175territories_r2.csv'
  merged_fathom = False
  if opt_fathom_p.exists():
    df_f = pd.read_csv(opt_fathom_p)
    extra_cols = [c for c in df_f.columns if c not in set(df_out.columns) and c != 'country_name']
    if extra_cols and 'country_name' in df_f.columns:
      df_out = df_out.merge(df_f[['country_name'] + extra_cols], on='country_name', how='left')
      merged_fathom = True

  out_path = out_dir / 'Table_S2_175territories.csv'
  df_out.to_csv(out_path, index=False)

  # Summary statistics matching Fig. 4 & Section 2 (174 non-Greenland countries)
  no_grl = df_out[df_out['country_name'] != 'Greenland'].copy()
  w_df = no_grl[no_grl['has_system']]
  g_df = no_grl[~no_grl['has_system']]
  mw_land = mannwhitneyu(g_df['land_ung_pct'].dropna(), w_df['land_ung_pct'].dropna())
  mw_pop = mannwhitneyu(g_df['pop_ung_pct'].dropna(), w_df['pop_ung_pct'].dropna())
  sub_inf = no_grl.dropna(subset=['inform_vulnerability', 'inform_lack_of_coping_capacity', 'pop_ung_pct'])
  rs_v, p_v = spearmanr(sub_inf['inform_vulnerability'], sub_inf['pop_ung_pct'])
  rs_c, p_c = spearmanr(sub_inf['inform_lack_of_coping_capacity'], sub_inf['pop_ung_pct'])

  summary = {
      'n_territories_total': int(len(df_out)),
      'n_countries_excl_greenland': int(len(no_grl)),
      'n_established_system': int(len(w_df)),
      'n_cap_gap_excl_greenland': int(len(g_df)),
      'land_ung_median_cap_gap_pct': float(g_df['land_ung_pct'].median()),
      'land_ung_median_sys_pct': float(w_df['land_ung_pct'].median()),
      'land_ung_mannwhitney_p': float(mw_land.pvalue),
      'pop_ung_median_cap_gap_pct': float(g_df['pop_ung_pct'].median()),
      'pop_ung_median_sys_pct': float(w_df['pop_ung_pct'].median()),
      'pop_ung_mannwhitney_p': float(mw_pop.pvalue),
      'n_zero_open_gauges_cap_gap': int((g_df['open_gauges_in_borders'] == 0).sum()),
      'n_zero_open_gauges_sys': int((w_df['open_gauges_in_borders'] == 0).sum()),
      'inform_n': int(len(sub_inf)),
      'inform_vuln_spearman_r': float(rs_v),
      'inform_vuln_spearman_p': float(p_v),
      'inform_coping_spearman_r': float(rs_c),
      'inform_coping_spearman_p': float(p_c),
      'optional_fathom_columns_merged': merged_fathom,
  }
  with open(out_dir / 'Table_S2_summary_stats.json', 'w') as f:
    json.dump(summary, f, indent=2)

  print(
      f'  Saved {out_path} ({len(df_out)} territories; 174 non-GRL={len(w_df)} sys vs {len(g_df)} gap; '
      f'INFORM N={len(sub_inf)}; Fathom cols={merged_fathom})',
      flush=True,
  )
  return out_path


# ==============================================================================
# TABLE S3: 43 Hydrological Skill Metrics Across 4 Modeling Regimes
# ==============================================================================
def build_table_s3(out_dir: Path) -> Path:
  print('[Table S3] Building 43-metric x 4-regime skill summary table...', flush=True)
  src_candidates = [
      CF_DIR / 'Table_S3_4regimes_r2.csv',
      CF_DIR / 'Table_S1_4regimes_r2.csv',
  ]
  src = next((p for p in src_candidates if p.exists()), None)
  if src is None:
    raise FileNotFoundError(
        'Missing Stage D 4-regime skill table. Run '
        '03_skill_predictors_and_shap/stage_d_counterfactuals_bootstrap_s16.py --part table_s3 first.'
    )

  df = pd.read_csv(src)
  # Ensure strictly non-Fathom columns only
  safe_cols = [c for c in df.columns if 'eae' not in c.lower() and 'exposure' not in c.lower() and 'fathom' not in c.lower()]
  df = df[safe_cols].copy()

  out_path = out_dir / 'Table_S3_4regimes_skill.csv'
  df.to_csv(out_path, index=False)
  print(f'  Saved {out_path} ({len(df)} rows, {df["metric"].nunique()} unique metrics)', flush=True)
  return out_path


# ==============================================================================
# TABLE S4: Complete 95 Meta-Model Input Features Catalog
# ==============================================================================
def build_table_s4(out_dir: Path) -> Path:
  print('[Table S4] Building 95-feature meta-model catalog...', flush=True)
  feat_candidates = [
      META_DIR / 'features_95.json',
      REPO_DIR / 'results' / 'meta_models' / 'features_95.json',
  ]
  feat_p = next((p for p in feat_candidates if p.exists()), None)
  if feat_p is None:
    raise FileNotFoundError('Missing features_95.json. Run stage_c_train_meta_models.py first.')

  with open(feat_p) as f:
    features_95 = json.load(f)
  assert len(features_95) == 95, f'Expected 95 features, got {len(features_95)}'

  cat_rename = {
      'Anomaly Score': 'Climatic Similarity (OOD)',
      'Coverage/Overlap': 'River-Network Gauge Coverage',
      'Soil and Geology': 'Soil and Geology',
  }
  cat_order = [
      'Climate & Hydrology',
      'Topography',
      'Soil and Geology',
      'Land Cover',
      'Anthropogenic Factors',
      'River-Network Gauge Coverage',
      'Climatic Similarity (OOD)',
  ]

  rows = []
  for feat in features_95:
    raw_cat = feature_utils.get_feature_type(feat)
    disp_cat = cat_rename.get(raw_cat, raw_cat)
    desc = feature_utils.FEATURE_DESCRIPTIONS[feat]
    rows.append({
        'variable_symbol': feat,
        'description': desc,
        'category': disp_cat,
        '_cat_rank': cat_order.index(disp_cat) if disp_cat in cat_order else 99,
    })

  df_out = pd.DataFrame(rows).sort_values(['_cat_rank', 'variable_symbol']).drop(columns=['_cat_rank']).reset_index(drop=True)
  out_path = out_dir / 'Table_S4_features_95.csv'
  df_out.to_csv(out_path, index=False)
  print(f'  Saved {out_path} ({len(df_out)} features across {df_out["category"].nunique()} categories)', flush=True)
  return out_path


# ==============================================================================
# TABLE S5: Global, Regional, Strahler Stream-Order & Tier Ungaugedness Breakdown
# ==============================================================================
def _compute_strahler_order(hybas_ids: np.ndarray, next_down: np.ndarray) -> np.ndarray:
  """Computes exact topological Strahler stream order over the HydroBASINS directed forest."""
  n = len(hybas_ids)
  id_to_idx = {int(h): i for i, h in enumerate(hybas_ids)}
  next_idx = np.full(n, -1, dtype=np.int32)
  indeg = np.zeros(n, dtype=np.int32)
  for i in range(n):
    nd = int(next_down[i])
    if nd != 0 and nd in id_to_idx:
      j = id_to_idx[nd]
      if j != i:
        next_idx[i] = j
        indeg[j] += 1

  q = deque(np.where(indeg == 0)[0].tolist())
  max_child = np.zeros(n, dtype=np.int16)
  cnt_max = np.zeros(n, dtype=np.int16)
  order = np.ones(n, dtype=np.int16)

  while q:
    u = q.popleft()
    if max_child[u] == 0:
      order[u] = 1
    elif cnt_max[u] >= 2:
      order[u] = max_child[u] + 1
    else:
      order[u] = max_child[u]
    v = next_idx[u]
    if v >= 0:
      ou = order[u]
      if ou > max_child[v]:
        max_child[v] = ou
        cnt_max[v] = 1
      elif ou == max_child[v]:
        cnt_max[v] += 1
      indeg[v] -= 1
      if indeg[v] == 0:
        q.append(v)

  return order


def build_table_s5(out_dir: Path) -> Path:
  print('[Table S5] Building global, regional, Strahler-order & tier ungaugedness breakdown...', flush=True)
  topo = _load_topology_and_runoff()
  all_ids = _norm_hybas(topo['HYBAS_ID']).tolist()
  area_km2 = topo['area_m2'].values.astype(np.float64) / 1e6
  runoff_km3_yr = area_km2 * topo['run_mm_syr'].values.astype(np.float64) * 1e-6  # km^2 * mm/yr * 1e-6 = km^3/yr
  global_area = float(area_km2.sum())

  region_map = {
      'na': 'North America',
      'sa': 'South America',
      'eu': 'Europe',
      'as': 'Asia',
      'si': 'Asia',
      'au': 'Australia and Oceania',
      'ar': 'Arctic and Greenland',
      'gr': 'Arctic and Greenland',
      'af': 'Africa',
  }
  regions = topo['continent'].map(region_map).values
  strahler = _compute_strahler_order(topo['HYBAS_ID'].values, topo['NEXT_DOWN'].values)

  masks_df = pd.read_parquet(TIER_DIR / 'tier_active_and_all_masks_r2.parquet')
  masks_df.index = _norm_hybas(masks_df.index)
  masks_df = masks_df.reindex(all_ids)

  def _load_tier(tk: str) -> pd.DataFrame:
    df_t = pd.read_parquet(
        TIER_DIR / f'{tk}.parquet',
        columns=['total_upstream_coverage', 'max_downstream_coverage', 'U_net_tree_phys'],
    )
    df_t.index = _norm_hybas(df_t.index)
    return df_t.reindex(all_ids)

  t4 = _load_tier('tier4_all_publicly_knowable_121819')

  def _summarize(part: str, group: str, idx_mask: np.ndarray, df_t: pd.DataFrame, snap_col: str) -> dict:
    a = area_km2[idx_mask]
    r = runoff_km3_yr[idx_mask]
    a_tot = float(a.sum())
    r_tot = float(r.sum())

    snap_g = masks_df[snap_col].values.astype(bool)[idx_mask]
    cov_up = df_t['total_upstream_coverage'].values[idx_mask]
    cov_dn = df_t['max_downstream_coverage'].values[idx_mask]
    u_net = df_t['U_net_tree_phys'].values.astype(np.float64)[idx_mask]

    outlet_ung = ~snap_g
    zero_up = cov_up <= 0.0
    entire_ung = zero_up & (cov_dn <= 0.0)

    q25, q50, q75 = np.percentile(u_net * 100.0, [25, 50, 75])
    return {
        'part': part,
        'group': group,
        'n_basins': int(idx_mask.sum()),
        'land_area_km2': round(a_tot, 2),
        'land_area_share_pct': round(a_tot / global_area * 100.0, 4),
        'runoff_km3_yr': round(r_tot, 2),
        'outlet_ung_area_pct': round(float(a[outlet_ung].sum()) / a_tot * 100.0, 4),
        'outlet_ung_basin_pct': round(float(outlet_ung.mean()) * 100.0, 4),
        'zero_up_area_pct': round(float(a[zero_up].sum()) / a_tot * 100.0, 4),
        'zero_up_basin_pct': round(float(zero_up.mean()) * 100.0, 4),
        'entire_net_ung_area_pct': round(float(a[entire_ung].sum()) / a_tot * 100.0, 4),
        'entire_net_ung_basin_pct': round(float(entire_ung.mean()) * 100.0, 4),
        'U_net_area_pct': round(float(np.sum(u_net * a)) / a_tot * 100.0, 4),
        'U_net_runoff_pct': round(float(np.sum(u_net * r)) / r_tot * 100.0, 4) if r_tot > 0 else np.nan,
        'U_net_basin_mean_pct': round(float(np.mean(u_net)) * 100.0, 4),
        'U_net_basin_sd_pct': round(float(np.std(u_net, ddof=0)) * 100.0, 4),
        'U_net_basin_median_pct': round(float(q50), 4),
        'U_net_basin_q25_pct': round(float(q25), 4),
        'U_net_basin_q75_pct': round(float(q75), 4),
    }

  rows = []
  all_mask = np.ones(len(all_ids), dtype=bool)
  snap_t4 = 'tier4_all_publicly_knowable_121819__snap_all'

  # Part A: Global & 7 regions
  rows.append(_summarize('Part A: Global & Regional', 'Global', all_mask, t4, snap_t4))
  for reg in [
      'North America',
      'South America',
      'Europe',
      'Asia',
      'Australia and Oceania',
      'Arctic and Greenland',
      'Africa',
  ]:
    rows.append(_summarize('Part A: Global & Regional', reg, regions == reg, t4, snap_t4))

  # Part B: Strahler stream orders 1..7, 8+, and 7+ combined
  for k in range(1, 8):
    lbl = 'Strahler Order 1 (Headwaters)' if k == 1 else f'Strahler Order {k}'
    rows.append(_summarize('Part B: Strahler Stream Order', lbl, strahler == k, t4, snap_t4))
  rows.append(_summarize('Part B: Strahler Stream Order', 'Strahler Order 8+', strahler >= 8, t4, snap_t4))
  rows.append(_summarize('Part B: Strahler Stream Order', 'Strahler Order 7+ (Combined)', strahler >= 7, t4, snap_t4))

  # Part C: 4 data-accessibility tiers
  tier_rows = [
      ('tier4_all_publicly_knowable_121819', 'All Publicly Discoverable Gauges (121,819)'),
      ('tier_public_discharge_93827', 'Confirmed Publicly Accessible Records (93,827)'),
      ('tier_caravan_estreams_34577', 'Standardized Open Compilations (Caravan + EStreams; 34,577)'),
      ('tier1a_active_post2016_ccby_8811', 'Post-2016 Caravan Training Subset (8,811 records; 8,075 stations)'),
  ]
  for tk, lbl in tier_rows:
    df_t = t4 if tk == 'tier4_all_publicly_knowable_121819' else _load_tier(tk)
    rows.append(_summarize('Part C: Data-Accessibility Tier', lbl, all_mask, df_t, f'{tk}__snap_all'))

  df_out = pd.DataFrame(rows)
  out_path = out_dir / 'Table_S5_ungaugedness_breakdown.csv'
  df_out.to_csv(out_path, index=False)
  print(
      f'  Saved {out_path} ({len(df_out)} rows; Global={global_area/1e6:.2f}M km², '
      f'Runoff={runoff_km3_yr.sum():,.0f} km³/yr)',
      flush=True,
  )
  return out_path


def main() -> None:
  parser = argparse.ArgumentParser(description='Build publication tables (Table 1 and Tables S1–S5; Fathom-free).')
  parser.add_argument('--out-dir', type=str, default=str(DEFAULT_OUT_DIR))
  parser.add_argument(
      '--only',
      type=str,
      default='',
      help='Optional comma-separated subset of tables: t1,ts1,ts2,ts3,ts4,ts5',
  )
  args = parser.parse_args()

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  selected = {x.strip().lower() for x in args.only.split(',') if x.strip()} if args.only else {'t1', 'ts1', 'ts2', 'ts3', 'ts4', 'ts5'}

  builders = [
      ('t1', build_table1),
      ('ts1', build_table_s1),
      ('ts2', build_table_s2),
      ('ts3', build_table_s3),
      ('ts4', build_table_s4),
      ('ts5', build_table_s5),
  ]
  for key, fn in builders:
    if key in selected:
      fn(out_dir)

  print(f'[build_tables] Done. Outputs written to {out_dir}', flush=True)


if __name__ == '__main__':
  main()
