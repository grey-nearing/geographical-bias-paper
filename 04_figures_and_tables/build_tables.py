#!/usr/bin/env python3
"""Build publication tables (Supplementary Tables S1–S8; Fathom-free by default).

Produces strictly non-Fathom outputs in `--out-dir` (default: `$GEO_BIAS_DATA_DIR/round2/tables`),
numbered strictly in order of first citation in the Science manuscript (0 standalone main tables):
  - Table_S1_72_gauge_repositories.csv: Complete inventory of the 72 physical stream-gauge
    repositories retained across 4 source categories (361,020 archive rows -> 119,169 gauges).
  - Table_S2_gauge_status.csv: Mutually exclusive operational & temporal status
    breakdown of 32,369 / 82,596 / 119,169 stream gauges under the uniform 2016 cutoff
    plus observation-type breakdowns and strict >=2022 daily-discharge DA counts.
  - Table_S3_ungaugedness_ranges.csv: Multi-hydrofabric min–max ranges across six
    fine-scale global hydrofabrics (U_never, U_net_Eq4, and 1D flow-weighted discharge;
    visualized in Fig. 2b). Optional Fathom exposure rows are appended ONLY if
    `exposure_eae` exists in `master_2d_metrics.parquet` and `--public-only` is not set.
  - Table_S4_175territories.csv (+ Table_S4_summary_stats.json): Country/territory-level
    HydroSHEDS L12 basin counts, alerting status, fractional land & population
    ungaugedness (%), standardized open gauges inside borders, INFORM 2026 scores,
    and spatially adjusted (Moran's I / Clifford–Richardson–Hémon–Dutilleul N_eff) stats.
    Optional Fathom exposure columns are merged ONLY if local Stage D exposure outputs exist
    and `--public-only` is not set.
  - Table_S5_features_95.csv: Complete catalog of the 95 meta-model input features
    grouped into the seven attribution categories (with ria_ha_usu in Topography).
  - Table_S6_forecast_skill_and_exposure.csv (OPTIONAL — licensed Fathom v3): Exposure
    counterfactual breakdown across CAP-gap, alerting, and global domains under alpha = 1
    (Recall) and alpha = CSI/R, direct FT+DA (no OOD clamp) vs. idealized OOD -> Q75 ceiling,
    strict >=2022 daily-discharge DA, and 95% block-bootstrap + cross-domain UQ intervals
    (visualized in Fig. 5; plus Table_S6_action_regime_sensitivity.csv for SM S1.4.1).
  - Table_S7_4regimes_skill.csv: Observed Caravan CV skill, multi-level meta-model CV R²,
    and 1,034,083-basin predicted skill across all 43 hydrological metrics and 4 regimes.
  - Table_S8_ungaugedness_breakdown.csv: Global, regional, Strahler stream-order
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

from utils import cap_utils, feature_utils, file_paths, spatial_stats

warnings.filterwarnings('ignore')

R2_DIR = CANON_DIR / os.environ.get('GEO_BIAS_ROUND_SUBDIR', 'round2')
TIER_DIR = R2_DIR / 'tier_features'
CF_DIR = R2_DIR / 'counterfactuals'
META_DIR = R2_DIR / 'meta_models'
DEFAULT_OUT_DIR = R2_DIR / 'tables'

FINE_6_FABRICS = [
    'hydrosheds_l12',
    'hydrosheds_rivers',
    'merit_basins',
    'tdx_hydro',
    'grit_v10_mainstem_tree',
    'grit_v10_bifurcating_dag',
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
      CANON_DIR / 'round2' / 'exposure' / 'global_basin_exposure_defended_r2.parquet',
  ]
  for p in candidates:
    if p.exists():
      wp = pd.read_parquet(p)
      popc = 'total_population' if 'total_population' in wp.columns else next(c for c in wp.columns if 'pop' in c.lower())
      if 'HYBAS_ID' in wp.columns:
        wp.index = _norm_hybas(wp['HYBAS_ID'])
      elif 'hybas_id' in wp.columns:
        wp.index = _norm_hybas(wp['hybas_id'])
      else:
        wp.index = _norm_hybas(wp.index)
      wp = wp[~wp.index.duplicated(keep='first')].reindex(all_ids)
      return pd.to_numeric(wp[popc], errors='coerce').fillna(0.0)
  raise FileNotFoundError(
      'Missing hydrosheds_l12_worldpop_population.parquet; checked '
      + ', '.join(str(c) for c in candidates)
  )


# ==============================================================================
# TABLE S3: Multi-Hydrofabric Global Hydrological Ungaugedness Ranges
# ==============================================================================
def build_table_s3(out_dir: Path, public_only: bool = False) -> Path:
  print('[Table S3] Building multi-hydrofabric ungaugedness ranges...', flush=True)
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
  is_dag = m2d_dom['hydrofabric'] == 'grit_v10_bifurcating_dag'
  canon_2d = m2d_dom[
      (is_dag & (m2d_dom['snapping_mode'] == 'dag_width_weighted') & (m2d_dom['ratio_basis'] == 'area_dag'))
      | ((~is_dag) & (m2d_dom['snapping_mode'] == 'outlet') & (m2d_dom['ratio_basis'] == 'area'))
  ].copy()

  has_exposure = (not public_only) and ('exposure_eae' in set(canon_2d['weight'].unique()))

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

  _add_2d_row('1. Fractional Watershed Ungaugedness', 'U_net_Eq4', 'area', 'Global land area', FINE_6_FABRICS)
  _add_2d_row('1. Fractional Watershed Ungaugedness', 'U_net_Eq4', 'runoff', 'Global river runoff (3 flow models)', FLOW_4_FABRICS)
  _add_2d_row('1. Fractional Watershed Ungaugedness', 'U_net_Eq4', 'population', 'Global population', FINE_6_FABRICS)
  if has_exposure:
    _add_2d_row('1. Fractional Watershed Ungaugedness', 'U_net_Eq4', 'exposure_eae', 'Annual flood exposure (optional licensed Fathom)', FINE_6_FABRICS)

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
      'section': '1. Fractional Watershed Ungaugedness',
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
  out_path = out_dir / 'Table_S3_ungaugedness_ranges.csv'
  df_out.to_csv(out_path, index=False)
  print(f'  Saved {out_path} ({len(df_out)} rows; Fathom exposure rows included={has_exposure})', flush=True)
  return out_path


# ==============================================================================
# TABLE S6 (OPTIONAL — Requires Licensed Fathom v3 Counterfactual Outputs)
# ==============================================================================
def build_table_s6(out_dir: Path, public_only: bool = False) -> Path | None:
  if public_only:
    print('[Table S6] SKIPPED (--public-only mode; requires licensed Fathom v3).', flush=True)
    return None
  sum_csv = CF_DIR / 'multi_mechanism_tiered_exposure_r2.csv'
  if not sum_csv.exists():
    print(
        f'[Table S6] SKIPPED: Licensed Fathom v3 counterfactual summary not found at {sum_csv}.',
        flush=True,
    )
    return None

  print('[Table S6] Building optional Fathom v3 forecast-skill & exposure counterfactual table...', flush=True)
  df = pd.read_csv(sum_csv)
  boot_csv = CF_DIR / 'bootstrap_table1_fig3_ci95_r2.csv'
  df_ci = pd.read_csv(boot_csv) if boot_csv.exists() else None

  def _ci(scen: str, metric: str) -> tuple[float, float]:
    if df_ci is None:
      return (np.nan, np.nan)
    sub = df_ci[(df_ci['scenario'] == scen) & (df_ci['metric'] == metric)]
    if sub.empty:
      return (np.nan, np.nan)
    return (float(sub.iloc[0]['boot_p025']) / 1e6, float(sub.iloc[0]['boot_p975']) / 1e6)

  tier_specs = [
      ('tier1a_active_post2016_ccby_8811', 'Baseline: Post-2016 Caravan (8,811 records / 8,075 phys stations)'),
      ('tier1_open_compilations', 'Tier 1: Standardized, Open-License Discharge Datasets (32,369)'),
      ('tier2_public_discharge', 'Tier 2: Gauges with Discharge Records (82,596)'),
      ('tier4_all_discoverable', 'Tier 3: All Publicly Discoverable Physical Gauges (119,169)'),
      ('tier5_universal_no_ood_clamp', 'Universal Gauging — Direct Local FT+DA (no OOD clamp)'),
      ('tier5_universal_idealized_ceiling', 'Universal Gauging — Idealized Global-Training Ceiling (OOD -> Q75)'),
  ]
  mech_specs = [
      ('000_global_train_joint', '000_train', 'Global training alone'),
      ('010_ft_only_snap', '010_ft_only', 'Local fine-tuning (FT) alone'),
      ('001_da_only_snap', '001_da_only', 'Real-time data assimilation (DA) alone'),
      ('111_all_three_combined_snap', '111_combined', 'Combined (global training + FT + DA)'),
      ('001_da_only_snap_strict2022', '001_da_only_strict2022', 'Real-time DA alone (strict >=2022 daily discharge)'),
      ('111_all_three_combined_snap_strict2022', '111_combined_strict2022', 'Combined (strict >=2022 daily discharge DA)'),
  ]

  rows = []
  for tk, t_lbl in tier_specs:
    for mk, b_mk, m_lbl in mech_specs:
      if 'strict2022' in mk and tk != 'tier4_all_discoverable':
        continue
      sub = df[(df['tier_key'] == tk) & (df['mechanism'] == mk)]
      if sub.empty:
        continue
      r = sub.iloc[0]
      if tk == 'tier1a_active_post2016_ccby_8811' and mk == '000_global_train_joint':
        dg_lo, dg_hi = (0.0, 0.0)
        dc_lo, dc_hi = (0.0, 0.0)
        wg_lo, wg_hi = _ci('baseline_warned_global', 'level')
        wc_lo, wc_hi = _ci('baseline_warned_nosys115', 'level')
        dg_csi_lo, dg_csi_hi = (0.0, 0.0)
        dc_csi_lo, dc_csi_hi = (0.0, 0.0)
        wg_csi_lo, wg_csi_hi = _ci('baseline_warned_csi_global', 'level')
        wc_csi_lo, wc_csi_hi = _ci('baseline_warned_csi_nosys115', 'level')
      else:
        dg_lo, dg_hi = _ci(f'{tk}__{b_mk}', 'delta_global')
        dc_lo, dc_hi = _ci(f'{tk}__{b_mk}', 'delta_nosys115')
        wg_lo, wg_hi = _ci(f'{tk}__{b_mk}', 'warned_global')
        wc_lo, wc_hi = _ci(f'{tk}__{b_mk}', 'warned_nosys115')
        dg_csi_lo, dg_csi_hi = _ci(f'{tk}__{b_mk}', 'delta_csi_global')
        dc_csi_lo, dc_csi_hi = _ci(f'{tk}__{b_mk}', 'delta_csi_nosys115')
        wg_csi_lo, wg_csi_hi = _ci(f'{tk}__{b_mk}', 'warned_csi_global')
        wc_csi_lo, wc_csi_hi = _ci(f'{tk}__{b_mk}', 'warned_csi_nosys115')

      def _gval(col: str, div: float = 1.0) -> float:
        return round(float(r[col]) / div, 4) if col in r and pd.notna(r[col]) else np.nan

      rows.append({
          'tier_key': tk,
          'tier_label': t_lbl,
          'mechanism': mk,
          'mechanism_label': m_lbl,
          'n_snap_all': int(r['n_snap_all']),
          'n_snap_active': int(r['n_snap_active']),
          'r5_med_pct': round(float(r['r5_med_pct']), 4),
          'p5_med_pct': _gval('p5_med_pct'),
          'NSE_med': round(float(r['NSE_med']), 4),
          'E_total_global_M_yr': round(float(r['E_total_global']) / 1e6, 4),
          'E_total_sys60_M_yr': round(float(r['E_total_sys60']) / 1e6, 4),
          'E_total_nosys115_M_yr': round(float(r['E_total_nosys115']) / 1e6, 4),
          'warned_global_M_yr': round(float(r['warned_global']) / 1e6, 4),
          'warned_global_ci95_lo_M_yr': round(wg_lo, 4) if np.isfinite(wg_lo) else np.nan,
          'warned_global_ci95_hi_M_yr': round(wg_hi, 4) if np.isfinite(wg_hi) else np.nan,
          'pct_warned_global': round(float(r['pct_warned_global']), 4),
          'delta_net_global_M_yr': round(float(r['delta_net_global']) / 1e6, 4),
          'delta_net_global_ci95_lo_M_yr': round(dg_lo, 4) if np.isfinite(dg_lo) else np.nan,
          'delta_net_global_ci95_hi_M_yr': round(dg_hi, 4) if np.isfinite(dg_hi) else np.nan,
          'pct_unw_closed_global': round(float(r['pct_unw_closed_net_global']), 4),
          'warned_nosys115_M_yr': round(float(r['warned_nosys115']) / 1e6, 4),
          'warned_nosys115_ci95_lo_M_yr': round(wc_lo, 4) if np.isfinite(wc_lo) else np.nan,
          'warned_nosys115_ci95_hi_M_yr': round(wc_hi, 4) if np.isfinite(wc_hi) else np.nan,
          'delta_net_nosys115_M_yr': round(float(r['delta_net_nosys115']) / 1e6, 4),
          'delta_net_nosys115_ci95_lo_M_yr': round(dc_lo, 4) if np.isfinite(dc_lo) else np.nan,
          'delta_net_nosys115_ci95_hi_M_yr': round(dc_hi, 4) if np.isfinite(dc_hi) else np.nan,
          'pct_unw_closed_nosys115': round(float(r['pct_unw_closed_net_nosys115']), 4),
          'warned_sys60_M_yr': round(float(r['warned_sys60']) / 1e6, 4),
          'delta_net_sys60_M_yr': round(float(r['delta_net_sys60']) / 1e6, 4),
          'warned_csi_global_M_yr': _gval('warned_csi_global', 1e6),
          'warned_csi_global_ci95_lo_M_yr': round(wg_csi_lo, 4) if np.isfinite(wg_csi_lo) else np.nan,
          'warned_csi_global_ci95_hi_M_yr': round(wg_csi_hi, 4) if np.isfinite(wg_csi_hi) else np.nan,
          'delta_net_csi_global_M_yr': _gval('delta_net_csi_global', 1e6),
          'delta_net_csi_global_ci95_lo_M_yr': round(dg_csi_lo, 4) if np.isfinite(dg_csi_lo) else np.nan,
          'delta_net_csi_global_ci95_hi_M_yr': round(dg_csi_hi, 4) if np.isfinite(dg_csi_hi) else np.nan,
          'warned_csi_nosys115_M_yr': _gval('warned_csi_nosys115', 1e6),
          'warned_csi_nosys115_ci95_lo_M_yr': round(wc_csi_lo, 4) if np.isfinite(wc_csi_lo) else np.nan,
          'warned_csi_nosys115_ci95_hi_M_yr': round(wc_csi_hi, 4) if np.isfinite(wc_csi_hi) else np.nan,
          'delta_net_csi_nosys115_M_yr': _gval('delta_net_csi_nosys115', 1e6),
          'delta_net_csi_nosys115_ci95_lo_M_yr': round(dc_csi_lo, 4) if np.isfinite(dc_csi_lo) else np.nan,
          'delta_net_csi_nosys115_ci95_hi_M_yr': round(dc_csi_hi, 4) if np.isfinite(dc_csi_hi) else np.nan,
          'warned_csi_sys60_M_yr': _gval('warned_csi_sys60', 1e6),
          'delta_net_csi_sys60_M_yr': _gval('delta_net_csi_sys60', 1e6),
      })

  df_out = pd.DataFrame(rows)
  out_path = out_dir / 'Table_S6_forecast_skill_and_exposure.csv'
  df_out.to_csv(out_path, index=False)
  print(f'  Saved {out_path} ({len(df_out)} rows; bootstrap CIs included={df_ci is not None})', flush=True)

  # Also export action-regime sensitivity table (SM S1.4.1) if generated by Stage D
  act_csv = CF_DIR / 'action_regime_sensitivity_r3.csv'
  if not act_csv.exists():
    act_csv = CF_DIR / 'precision_f1_csi_alpha_sensitivity_r2.csv'
  if act_csv.exists():
    df_act = pd.read_csv(act_csv)
    act_out = out_dir / 'Table_S6_action_regime_sensitivity.csv'
    df_act.to_csv(act_out, index=False)
    print(f'  Saved {act_out} ({len(df_act)} rows)', flush=True)
  return out_path


# ==============================================================================
# TABLE S2: Operational & Temporal Status of Global Stream Gauges (2016 Cutoff)
# ==============================================================================
def build_table_s2(out_dir: Path) -> Path:
  print('[Table S2] Building operational & temporal gauge status table...', flush=True)
  g = pd.read_parquet(CANON_DIR / 'attributes' / 'global_gauges_deduplicated.parquet')

  m_t1 = g['in_tier_1'].astype(bool)
  m_t2 = g['in_tier_2'].astype(bool)
  m_t3 = g['in_tier_3'].astype(bool)

  tiers = [
      ('tier1', g[m_t1]),
      ('tier2', g[m_t2]),
      ('tier3', g[m_t3]),
  ]

  def _masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    ar = df['active_reason'].astype(str)
    vy = pd.to_numeric(df['verified_record_end_year'], errors='coerce')
    obs = df['best_observation_type'].astype(str)
    c_ver_ge16 = ar.isin({'1_cv8811_verified_active', '2_end_year_ge_2016'})
    c_ver_ge16_daily = c_ver_ge16 & (obs == 'daily_discharge')
    c_ver_ge16_attr = c_ver_ge16 & obs.isin({'catchment_attributes_and_boundaries', 'flow_indices_only'})
    c_ver_ge16_stage_mo = c_ver_ge16 & obs.isin({'stage_or_fluviometric', 'monthly_discharge'})
    c_port_act_nodate = (ar == '3_portal_status_active') & vy.isna()
    c_ver_le15 = ar == '6_end_year_lt_2016'
    c_port_inact = ar == '5_portal_status_inactive'
    c_port_act_le15 = (ar == '3_portal_status_active') & vy.notna() & (vy < 2016)
    c_live_undated = ar.isin({'7_option1_live_portal_undated', '7_option1_live_operational_portal'})
    c_static_undated = ar.isin({'8_undated_historical_or_unverified', '8_undated_historical_or_metadata'})
    c_active_group = c_ver_ge16 | c_port_act_nodate
    c_hist_group = c_ver_le15 | c_port_inact | c_port_act_le15
    c_unclass_group = c_live_undated | c_static_undated
    c_sec3 = df['is_active_opt1'].astype(bool)
    c_sec3_daily = c_sec3 & (obs == 'daily_discharge')
    c_sec3_stage = c_sec3 & (obs == 'stage_or_fluviometric')
    c_strict_2022 = vy.notna() & (vy >= 2022) & (obs == 'daily_discharge')
    # Sanity check: mutually exclusive and collectively exhaustive
    total_check = c_active_group.astype(int) + c_hist_group.astype(int) + c_unclass_group.astype(int)
    assert (total_check == 1).all(), 'Status categories are not mutually exclusive & exhaustive!'
    return {
        'active_total': c_active_group,
        'verified_ge_2016': c_ver_ge16,
        'verified_ge_2016_daily_discharge': c_ver_ge16_daily,
        'verified_ge_2016_attributes_or_indices': c_ver_ge16_attr,
        'verified_ge_2016_stage_or_monthly': c_ver_ge16_stage_mo,
        'portal_active_no_end_year': c_port_act_nodate,
        'historical_or_inactive_total': c_hist_group,
        'verified_le_2015': c_ver_le15,
        'portal_inactive': c_port_inact,
        'portal_active_record_le_2015': c_port_act_le15,
        'unclassifiable_total': c_unclass_group,
        'live_portal_undated': c_live_undated,
        'static_archive_undated': c_static_undated,
        'total_distinct_gauges': pd.Series(True, index=df.index),
        'section3_assimilation_total': c_sec3,
        'section3_assimilation_daily_discharge': c_sec3_daily,
        'section3_assimilation_stage_only': c_sec3_stage,
        'strict_ge_2022_daily_discharge': c_strict_2022,
    }

  row_specs = [
      ('active_total', 'Active or recently active', True),
      ('verified_ge_2016', 'Verified record from 2016 onward (all observation types)', False),
      ('verified_ge_2016_daily_discharge', '  - Daily streamflow discharge (>=2016)', False),
      ('verified_ge_2016_attributes_or_indices', '  - Catchment attributes or summary flow indices (>=2016)', False),
      ('verified_ge_2016_stage_or_monthly', '  - Stage/fluviometric or monthly discharge (>=2016)', False),
      ('portal_active_no_end_year', 'Portal-listed active (no verified end year)', False),
      ('historical_or_inactive_total', 'Confirmed historical or inactive', True),
      ('verified_le_2015', 'Verified record ending in 2015 or earlier', False),
      ('portal_inactive', 'Portal-listed inactive (no post-2015 record)', False),
      ('portal_active_record_le_2015', 'Portal-listed active, but verified record ended <= 2015', False),
      ('unclassifiable_total', 'Unclassifiable (no end date or status)', True),
      ('live_portal_undated', 'Listed on live national portal without record end date', False),
      ('static_archive_undated', 'Static inventory or archive without dates or status', False),
      ('total_distinct_gauges', 'Total distinct physical gauges', True),
      ('section3_assimilation_total', 'Active + undated live-portal gauges (primary DA upper bound)', True),
      ('section3_assimilation_daily_discharge', '  - Confirmed daily discharge inside DA set', False),
      ('section3_assimilation_stage_only', '  - Stage-only / fluviometric stations inside DA set', False),
      ('strict_ge_2022_daily_discharge', 'Strict daily-discharge >=2022 sensitivity subset (DA lower bound)', True),
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
  out_path = out_dir / 'Table_S2_gauge_status.csv'
  df_out.to_csv(out_path, index=False)
  print(f'  Saved {out_path} (totals: {tier_totals})', flush=True)
  return out_path


# ==============================================================================
# TABLE S4: Country- and Territory-Level Summary (175 Territories; Non-Fathom by Default)
# ==============================================================================
def build_table_s4(out_dir: Path, public_only: bool = False) -> Path:
  print('[Table S4] Building 175-territory ungaugedness, open gauges & INFORM table...', flush=True)
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

  t4 = pd.read_parquet(TIER_DIR / 'tier4_all_discoverable.parquet', columns=['U_net_tree_phys'])
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
  t1 = gauges[gauges['in_tier_1'].astype(bool)].copy()
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

  # Natural Earth centroids for Clifford-Richardson-Hemon / Dutilleul spatial N_eff
  w_diss = world[['map_name', 'geometry']].dissolve(by='map_name')
  w_rep = w_diss.geometry.representative_point()
  c2lat = dict(zip(w_diss.index, w_rep.y))
  c2lon = dict(zip(w_diss.index, w_rep.x))
  ctry['centroid_lat'] = ctry['country_name'].map(c2lat)
  ctry['centroid_lon'] = ctry['country_name'].map(c2lon)

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
      'centroid_lat',
      'centroid_lon',
  ]
  df_out = ctry[out_cols].sort_values('country_name').reset_index(drop=True)

  # Optional local Fathom exposure columns ONLY if Stage D exposure CSV exists locally
  opt_fathom_p = CF_DIR / 'Table_S3_175territories_r2.csv'
  merged_fathom = False
  if (not public_only) and opt_fathom_p.exists():
    df_f = pd.read_csv(opt_fathom_p)
    extra_cols = [c for c in df_f.columns if c not in set(df_out.columns) and c != 'country_name']
    if extra_cols and 'country_name' in df_f.columns:
      df_out = df_out.merge(df_f[['country_name'] + extra_cols], on='country_name', how='left')
      merged_fathom = True

  out_path = out_dir / 'Table_S4_175territories.csv'
  df_out.to_csv(out_path, index=False)

  # Summary statistics matching Fig. 3e-f & Section 2 (174 non-Greenland countries)
  no_grl = df_out[df_out['country_name'] != 'Greenland'].copy()
  w_df = no_grl[no_grl['has_system']]
  g_df = no_grl[~no_grl['has_system']]
  mw_land = mannwhitneyu(g_df['land_ung_pct'].dropna(), w_df['land_ung_pct'].dropna())
  mw_pop = mannwhitneyu(g_df['pop_ung_pct'].dropna(), w_df['pop_ung_pct'].dropna())
  sub_inf = no_grl.dropna(subset=['inform_vulnerability', 'inform_lack_of_coping_capacity', 'pop_ung_pct'])
  rs_v, p_v = spearmanr(sub_inf['inform_vulnerability'], sub_inf['pop_ung_pct'])
  rs_c, p_c = spearmanr(sub_inf['inform_lack_of_coping_capacity'], sub_inf['pop_ung_pct'])

  spatial_neff = {}
  for x_col in ('inform_lack_of_coping_capacity', 'inform_vulnerability'):
    for y_col in ('land_ung_pct', 'pop_ung_pct'):
      spatial_neff[f'{x_col}__vs__{y_col}'] = spatial_stats.spatial_spearman_neff(
          no_grl[x_col].values,
          no_grl[y_col].values,
          no_grl['centroid_lat'].values,
          no_grl['centroid_lon'].values,
          k=5,
      )

  # Merge optional Stage D Fathom spatial N_eff stats if present locally
  opt_sp_p = CF_DIR / 'spatial_neff_stats_r3.json'
  if (not public_only) and opt_sp_p.exists():
    with open(opt_sp_p) as f:
      spatial_neff.update(json.load(f))

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
      'spatial_neff': spatial_neff,
      'optional_fathom_columns_merged': merged_fathom,
  }
  with open(out_dir / 'Table_S4_summary_stats.json', 'w') as f:
    json.dump(summary, f, indent=2)

  print(
      f'  Saved {out_path} ({len(df_out)} territories; 174 non-GRL={len(w_df)} sys vs {len(g_df)} gap; '
      f'INFORM N={len(sub_inf)}; Fathom cols={merged_fathom})',
      flush=True,
  )
  return out_path


# ==============================================================================
# TABLE S7: 43 Hydrological Skill Metrics Across 4 Modeling Regimes (Data S2)
# ==============================================================================
def build_table_s7(out_dir: Path) -> Path:
  print('[Table S7] Building 43-metric x 4-regime skill summary table...', flush=True)
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

  out_path = out_dir / 'Table_S7_4regimes_skill.csv'
  df.to_csv(out_path, index=False)
  print(f'  Saved {out_path} ({len(df)} rows, {df["metric"].nunique()} unique metrics)', flush=True)
  return out_path


# ==============================================================================
# TABLE S5: Complete 95 Meta-Model Input Features Catalog
# ==============================================================================
def build_table_s5(out_dir: Path) -> Path:
  print('[Table S5] Building 95-feature meta-model catalog...', flush=True)
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
  out_path = out_dir / 'Table_S5_features_95.csv'
  df_out.to_csv(out_path, index=False)
  print(f'  Saved {out_path} ({len(df_out)} features across {df_out["category"].nunique()} categories)', flush=True)
  return out_path


# ==============================================================================
# TABLE S8: Global, Regional, Strahler Stream-Order & Tier Ungaugedness Breakdown
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


def build_table_s8(out_dir: Path) -> Path:
  print('[Table S8] Building global, regional, Strahler-order & tier ungaugedness breakdown...', flush=True)
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

  t4 = _load_tier('tier4_all_discoverable')

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
  snap_t4 = 'tier4_all_discoverable__snap_all'

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
      ('tier4_all_discoverable', 'All Publicly Discoverable Physical Gauges (119,169)'),
      ('tier2_public_discharge', 'Gauges with Discharge Records (82,596)'),
      ('tier1_open_compilations', 'Standardized, Open-License Discharge Datasets (32,369)'),
      ('tier1a_active_post2016_ccby_8811', 'Post-2016 Caravan Training Subset (8,811 records; 8,075 stations)'),
  ]
  for tk, lbl in tier_rows:
    df_t = t4 if tk == 'tier4_all_discoverable' else _load_tier(tk)
    rows.append(_summarize('Part C: Data-Accessibility Tier', lbl, all_mask, df_t, f'{tk}__snap_all'))

  df_out = pd.DataFrame(rows)
  out_path = out_dir / 'Table_S8_ungaugedness_breakdown.csv'
  df_out.to_csv(out_path, index=False)
  print(
      f'  Saved {out_path} ({len(df_out)} rows; Global={global_area/1e6:.2f}M km², '
      f'Runoff={runoff_km3_yr.sum():,.0f} km³/yr)',
      flush=True,
  )
  return out_path


def main() -> None:
  parser = argparse.ArgumentParser(description='Build publication tables (Supplementary Tables S1–S8; optional Fathom Table S6).')
  parser.add_argument('--out-dir', type=str, default=str(DEFAULT_OUT_DIR))
  parser.add_argument(
      '--only',
      type=str,
      default='',
      help='Optional comma-separated subset of tables: ts1,ts2,ts3,ts4,ts5,ts6,ts7,ts8',
  )
  parser.add_argument(
      '--public-only',
      action='store_true',
      help='Build strictly non-Fathom public tables (omits Table S6 and Fathom rows/cols in Tables S3–S4).',
  )
  args = parser.parse_args()

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  selected = {x.strip().lower() for x in args.only.split(',') if x.strip()} if args.only else {'ts1', 'ts2', 'ts3', 'ts4', 'ts5', 'ts6', 'ts7', 'ts8'}

  from build_table_s1 import build_table_s1

  builders = [
      ('ts1', lambda d: build_table_s1(d)),
      ('ts2', lambda d: build_table_s2(d)),
      ('ts3', lambda d: build_table_s3(d, public_only=args.public_only)),
      ('ts4', lambda d: build_table_s4(d, public_only=args.public_only)),
      ('ts5', lambda d: build_table_s5(d)),
      ('ts6', lambda d: build_table_s6(d, public_only=args.public_only)),
      ('ts7', lambda d: build_table_s7(d)),
      ('ts8', lambda d: build_table_s8(d)),
  ]
  for key, fn in builders:
    if key in selected:
      fn(out_dir)

  print(f'[build_tables] Done. Outputs written to {out_dir}', flush=True)


if __name__ == '__main__':
  main()
