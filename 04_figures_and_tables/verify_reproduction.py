#!/usr/bin/env python3
"""End-to-end automated verification of manuscript reproduction (v3.3.0).

Checks all public figures (6 main / 8 SM citation-ordered suite: 10 public + 4 optional Fathom),
all 7 public supplementary tables (Tables S1–S5, S7–S8) plus optional licensed Fathom Table S6,
and every headline quantitative finding reported in the manuscript. Writes
`verification_report.json` and `verification_report.md` and exits with code 0 iff all checks pass.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = Path(os.environ.get('GEO_BIAS_REPO_DIR', str(SCRIPT_DIR.parent)))
CANON_DIR = Path(os.environ.get('GEO_BIAS_DATA_DIR', str(REPO_DIR / 'data')))
ROUND_SUBDIR = os.environ.get('GEO_BIAS_ROUND_SUBDIR', 'round2')
HYDRO_RESULTS_DIR = Path(
    os.environ.get(
        'GLOBAL_UNGAUGEDNESS_RESULTS_DIR',
        str(REPO_DIR / '01_hydrofabric_ungaugedness' / 'results'),
    )
)

import numpy as np
import pandas as pd

EXPECTED_FIGURES = [
    'Fig_1_global_overview.png',
    'Fig_2_scale_and_global_summary.png',
    'Fig_3_geography_and_alerting.png',
    'Fig_6_exact_group_shapley.png',
    'Fig_S1_gauge_status_histograms.png',
    'Fig_S2_conceptual_ungaugedness.png',
    'Fig_S3_alerting_coverage_map.png',
    'Fig_S4_delta_nse_network_vs_local.png',
    'Fig_S5_ood_sextiles_recall.png',
    'Fig_S6_global_predicted_skill.png',
]

EXPECTED_FATHOM_FIGURES = [
    'Fig_4_global_flood_maps.png',
    'Fig_5_exposure_synthesis.png',
    'Fig_S7_precision_recall_curves.png',
    'Fig_S8_national_gauge_siting.png',
]

EXPECTED_TABLES = [
    'Table_S1_72_gauge_repositories.csv',
    'Table_S2_gauge_status.csv',
    'Table_S3_ungaugedness_ranges.csv',
    'Table_S4_175territories.csv',
    'Table_S5_features_95.csv',
    'Table_S7_4regimes_skill.csv',
    'Table_S8_ungaugedness_breakdown.csv',
]


class Verifier:
  """Accumulates structured pass/fail checks."""

  def __init__(self) -> None:
    self.checks: list[dict[str, Any]] = []

  def check(self, group: str, name: str, passed: bool, actual: Any, expected: Any, note: str = '') -> None:
    self.checks.append({
        'group': group,
        'name': name,
        'passed': bool(passed),
        'actual': actual,
        'expected': expected,
        'note': note,
    })

  def check_close(
      self,
      group: str,
      name: str,
      actual: float,
      expected: float,
      atol: float,
      note: str = '',
  ) -> None:
    ok = bool(np.isfinite(actual) and abs(float(actual) - float(expected)) <= atol)
    self.check(group, name, ok, round(float(actual), 6), f'{expected} ± {atol}', note)

  def check_equal(self, group: str, name: str, actual: Any, expected: Any, note: str = '') -> None:
    self.check(group, name, actual == expected, actual, expected, note)


def verify_figures(v: Verifier, fig_dir: Path, with_fathom: bool = False) -> None:
  targets = list(EXPECTED_FIGURES) + (list(EXPECTED_FATHOM_FIGURES) if with_fathom else [])
  for fname in targets:
    p = fig_dir / fname
    exists = p.exists()
    size_kb = p.stat().st_size / 1024.0 if exists else 0.0
    v.check(
        'Figures',
        fname,
        exists and size_kb >= 20.0,
        f'{size_kb:.1f} KB' if exists else 'MISSING',
        '>= 20.0 KB',
    )


def verify_table_s1(v: Verifier, table_dir: Path) -> None:
  p = table_dir / 'Table_S1_72_gauge_repositories.csv'
  if not p.exists():
    v.check('Table S1', 'file_exists', False, 'MISSING', str(p))
    return
  df = pd.read_csv(p)
  v.check_equal('Table S1', 'n_repositories', len(df), 72)
  v.check_equal('Table S1', 'total_archive_rows', int(df['archive_rows'].sum()), 361020)


def verify_table_s3(v: Verifier, table_dir: Path, with_fathom: bool = False) -> None:
  p = table_dir / 'Table_S3_ungaugedness_ranges.csv'
  if not p.exists():
    v.check('Table S3', 'file_exists', False, 'MISSING', str(p))
    return
  df = pd.read_csv(p)

  def _row(metric: str, weight: str) -> pd.Series:
    sub = df[(df['metric'] == metric) & (df['weight'] == weight)]
    if len(sub) != 1:
      raise KeyError(f'Missing row metric={metric}, weight={weight}')
    return sub.iloc[0]

  expectations = [
      ('U_never', 'area', (58.5, 60.1), (35.5, 38.6), (31.7, 35.1)),
      ('U_never', 'runoff', (52.4, 59.2), (26.8, 34.6), (23.4, 31.5)),
      ('U_never', 'population', (67.8, 69.0), (39.5, 40.2), (27.9, 30.4)),
      ('U_net_Eq4', 'area', (89.9, 96.2), (83.8, 93.7), (79.8, 92.2)),
      ('U_net_Eq4', 'runoff', (87.6, 94.5), (80.7, 91.2), (76.0, 88.3)),
      ('U_net_Eq4', 'population', (84.5, 93.7), (73.9, 88.9), (61.1, 83.5)),
      ('U_net_1d', 'Q_ds', (31.7, 41.5), (12.6, 15.0), (9.5, 11.8)),
  ]
  if with_fathom:
    expectations.extend([
        ('U_never', 'exposure_eae', (64.2, 64.8), (35.7, 36.3), (23.4, 25.4)),
        ('U_net_Eq4', 'exposure_eae', (80.1, 92.3), (64.6, 86.5), (53.0, 81.5)),
    ])
  for metric, weight, t1, t2, t3 in expectations:
    try:
      r = _row(metric, weight)
    except KeyError as e:
      v.check('Table S3', f'{metric}_{weight}', False, str(e), 'row present')
      continue
    for prefix, (exp_lo, exp_hi) in [('tier1', t1), ('tier2', t2), ('tier3', t3)]:
      v.check_close('Table S3', f'{metric}.{weight}.{prefix}_min', float(r[f'{prefix}_min_pct']), exp_lo, atol=0.15)
      v.check_close('Table S3', f'{metric}.{weight}.{prefix}_max', float(r[f'{prefix}_max_pct']), exp_hi, atol=0.15)


def verify_fathom_outputs(v: Verifier, table_dir: Path, cf_dir: Path, s16_dir: Path) -> None:
  """Verifies optional licensed Fathom v3 outputs (Table S6, Table S4 exposure cols, bootstrap CIs, S1.6)."""
  # Table S6
  t6_p = table_dir / 'Table_S6_forecast_skill_and_exposure.csv'
  if not t6_p.exists():
    v.check('Table S6 (Fathom)', 'file_exists', False, 'MISSING', str(t6_p))
  else:
    df = pd.read_csv(t6_p)

    def _r(tk: str, mk: str) -> pd.Series:
      sub = df[(df['tier_key'] == tk) & (df['mechanism'] == mk)]
      if len(sub) != 1:
        raise KeyError(f'Missing {tk} / {mk}')
      return sub.iloc[0]

    b = _r('tier1a_active_post2016_ccby_8811', '000_global_train_joint')
    v.check_close('Table S6 (Fathom)', 'E_total_global_M_yr', float(b['E_total_global_M_yr']), 29.54, atol=0.05)
    v.check_close('Table S6 (Fathom)', 'E_total_sys60_M_yr', float(b['E_total_sys60_M_yr']), 20.03, atol=0.05)
    v.check_close('Table S6 (Fathom)', 'E_total_nosys115_M_yr', float(b['E_total_nosys115_M_yr']), 9.19, atol=0.05)
    v.check('Table S6 (Fathom)', 'baseline.warned_global_positive', float(b['warned_global_M_yr']) > 5.0, float(b['warned_global_M_yr']), '> 5.0 M/yr')
    v.check('Table S6 (Fathom)', 'baseline.warned_csi_positive', float(b['warned_csi_global_M_yr']) > 2.0, float(b['warned_csi_global_M_yr']), '> 2.0 M/yr')
    for tk, mk in [
        ('tier4_all_discoverable', '111_all_three_combined_snap'),
        ('tier4_all_discoverable', '111_all_three_combined_snap_strict2022'),
        ('tier5_universal_no_ood_clamp', '111_all_three_combined_snap'),
        ('tier5_universal_idealized_ceiling', '111_all_three_combined_snap'),
    ]:
      try:
        row = _r(tk, mk)
        v.check('Table S6 (Fathom)', f'{tk[:20]}.{mk[:16]}.delta_pos', float(row['delta_net_global_M_yr']) > 0.0, round(float(row['delta_net_global_M_yr']), 4), '> 0.0 M/yr')
        v.check('Table S6 (Fathom)', f'{tk[:20]}.{mk[:16]}.csi_pos', float(row['delta_net_csi_global_M_yr']) > 0.0, round(float(row['delta_net_csi_global_M_yr']), 4), '> 0.0 M/yr')
      except KeyError as e:
        v.check('Table S6 (Fathom)', f'{tk}.{mk}', False, str(e), 'row present')

  # Action-regime sensitivity table (SM S1.4.1)
  act_p = table_dir / 'Table_S6_action_regime_sensitivity.csv'
  v.check('Table S6 (Fathom)', 'action_regime_sensitivity_exists', act_p.exists(), act_p.exists(), True)

  # Table S4 Fathom columns
  ts4_p = table_dir / 'Table_S4_175territories.csv'
  if ts4_p.exists():
    df_s4 = pd.read_csv(ts4_p)
    has_etot = 'E_total' in df_s4.columns
    v.check('Table S4 (Fathom)', 'E_total_col_present', has_etot, has_etot, True)
    if has_etot:
      v.check_close('Table S4 (Fathom)', 'mapped_E_total_M_yr', float(df_s4['E_total'].sum()) / 1e6, 29.22, atol=0.05)
      v.check_close('Table S4 (Fathom)', 'sys60_E_total_M_yr', float(df_s4.loc[df_s4['has_system'], 'E_total'].sum()) / 1e6, 20.03, atol=0.05)
      v.check_close('Table S4 (Fathom)', 'nosys115_E_total_M_yr', float(df_s4.loc[~df_s4['has_system'], 'E_total'].sum()) / 1e6, 9.19, atol=0.05)

  # Bootstrap 95% CIs
  boot_p = cf_dir / 'bootstrap_table1_fig3_ci95_r2.csv'
  if not boot_p.exists():
    v.check('Bootstrap (Fathom)', 'ci95_file_exists', False, 'MISSING', str(boot_p))
  else:
    df_b = pd.read_csv(boot_p)
    v.check('Bootstrap (Fathom)', 'ci95_nonempty', len(df_b) >= 40, len(df_b), '>= 40 rows')
    v.check('Bootstrap (Fathom)', 'ci95_ordered', bool((df_b['boot_p025'] <= df_b['boot_p975']).all()), 'all p025<=p975', 'all p025<=p975')

  # S1.6 National N=1..20 Gauge Siting
  s16_p = s16_dir / 'country_joint_n_gauge_r2.csv'
  if not s16_p.exists():
    v.check('S1.6 Siting (Fathom)', 'file_exists', False, 'MISSING', str(s16_p))
  else:
    df_nat = pd.read_csv(s16_p)
    v.check_equal('S1.6 Siting (Fathom)', 'n_exposed_countries', len(df_nat), 173)
    for N in [1, 5, 10, 20]:
      v.check('S1.6 Siting (Fathom)', f'N{N}_combined_sum_positive', float(df_nat[f'N={N}_Combined_Dom'].sum()) > 0.0, round(float(df_nat[f'N={N}_Combined_Dom'].sum()) / 1e6, 4), '> 0.0 M/yr')


def verify_table_s2(v: Verifier, table_dir: Path) -> None:
  p = table_dir / 'Table_S2_gauge_status.csv'
  if not p.exists():
    v.check('Table S2', 'file_exists', False, 'MISSING', str(p))
    return
  df = pd.read_csv(p).set_index('category_key')
  expected = {
      'active_total': (20373, 35052, 49238),
      'verified_ge_2016': (19942, 31860, 45184),
      'portal_active_no_end_year': (431, 3192, 4054),
      'historical_or_inactive_total': (10310, 23444, 29495),
      'verified_le_2015': (9843, 22126, 27501),
      'portal_inactive': (36, 789, 1457),
      'portal_active_record_le_2015': (431, 529, 537),
      'unclassifiable_total': (1686, 24100, 40436),
      'live_portal_undated': (1349, 23758, 38518),
      'static_archive_undated': (337, 342, 1918),
      'total_distinct_gauges': (32369, 82596, 119169),
      'section3_assimilation_total': (22153, 59339, 88293),
  }
  for key, (e1, e2, e3) in expected.items():
    for col, exp in [('tier1_count', e1), ('tier2_count', e2), ('tier3_count', e3)]:
      v.check_equal('Table S2', f'{key}.{col}', int(df.loc[key, col]), exp)


def verify_table_s4(v: Verifier, table_dir: Path) -> None:
  p = table_dir / 'Table_S4_175territories.csv'
  sp = table_dir / 'Table_S4_summary_stats.json'
  if not p.exists() or not sp.exists():
    v.check('Table S4', 'files_exist', False, 'MISSING', f'{p} and {sp}')
    return
  df = pd.read_csv(p)
  stats = json.loads(sp.read_text())
  v.check_equal('Table S4 / Fig. 3', 'n_territories', len(df), 175)
  v.check_equal('Table S4 / Fig. 3', 'n_established_system', stats['n_established_system'], 60)
  v.check_equal('Table S4 / Fig. 3', 'n_cap_gap_excl_greenland', stats['n_cap_gap_excl_greenland'], 114)
  v.check_equal('Table S4 / Fig. 3', 'n_zero_open_gauges_sys', stats['n_zero_open_gauges_sys'], 17)
  v.check_equal('Table S4 / Fig. 3', 'n_zero_open_gauges_cap_gap', stats['n_zero_open_gauges_cap_gap'], 86)
  v.check_close('Table S4 / Fig. 3', 'land_ung_median_cap_gap_pct', stats['land_ung_median_cap_gap_pct'], 88.6, atol=0.15)
  v.check_close('Table S4 / Fig. 3', 'land_ung_median_sys_pct', stats['land_ung_median_sys_pct'], 61.3, atol=0.15)
  v.check('Table S4 / Fig. 3', 'land_ung_mannwhitney_p', stats['land_ung_mannwhitney_p'] < 1e-8, stats['land_ung_mannwhitney_p'], '< 1e-8')
  v.check_close('Table S4 / Fig. 3', 'pop_ung_median_cap_gap_pct', stats['pop_ung_median_cap_gap_pct'], 81.8, atol=0.15)
  v.check_close('Table S4 / Fig. 3', 'pop_ung_median_sys_pct', stats['pop_ung_median_sys_pct'], 39.1, atol=0.15)
  v.check('Table S4 / Fig. 3', 'pop_ung_mannwhitney_p', stats['pop_ung_mannwhitney_p'] < 1e-10, stats['pop_ung_mannwhitney_p'], '< 1e-10')
  v.check_equal('Table S4 / Fig. 3', 'inform_n', stats['inform_n'], 166)
  v.check_close('Table S4 / Fig. 3', 'inform_vuln_spearman_r', stats['inform_vuln_spearman_r'], 0.3179, atol=0.005)
  v.check_close('Table S4 / Fig. 3', 'inform_coping_spearman_r', stats['inform_coping_spearman_r'], 0.5407, atol=0.005)
  sp_neff = stats.get('spatial_neff', {})
  for k_sp in ['inform_lack_of_coping_capacity__vs__pop_ung_pct', 'inform_vulnerability__vs__pop_ung_pct']:
    p_sp = sp_neff.get(k_sp, {}).get('p_spatial', 1.0)
    v.check('Table S4 / Fig. 3 (A5)', f'p_spatial.{k_sp}', p_sp < 0.05, round(float(p_sp), 6), '< 0.05')


def verify_table_s7(v: Verifier, table_dir: Path) -> None:
  p = table_dir / 'Table_S7_4regimes_skill.csv'
  if not p.exists():
    v.check('Table S7', 'file_exists', False, 'MISSING', str(p))
    return
  df = pd.read_csv(p)
  v.check('Table S7', 'n_unique_metrics', int(df['metric'].nunique()) >= 23, int(df['metric'].nunique()), '>= 23 (43 after Phase 2)')

  def _val(metric: str, regime: str, col: str) -> float:
    sub = df[(df['metric'] == metric) & (df['regime'] == regime)]
    return float(sub.iloc[0][col]) if len(sub) > 0 and col in sub.columns else float('nan')

  checks = [
      ('NSE', 'base', 'obs_ungauged_median', 0.48, 0.015),
      ('NSE', 'ft_da', 'obs_ungauged_median', 0.79, 0.015),
      ('KGE', 'base', 'obs_ungauged_median', 0.50, 0.015),
      ('5.0-year-2.0-window-recall', 'base', 'obs_ungauged_median', 0.38, 0.015),
      ('5.0-year-2.0-window-recall', 'ft_da', 'obs_ungauged_median', 0.50, 0.015),
      ('5.0-year-2.0-window-precision', 'base', 'obs_ungauged_median', 0.25, 0.015),
      ('5.0-year-2.0-window-f1', 'base', 'obs_ungauged_median', 0.30, 0.015),
  ]
  for metric, reg, col, exp, tol in checks:
    v.check_close('Table S7', f'{metric}.{reg}.{col}', _val(metric, reg, col), exp, atol=tol)


def verify_table_s5(v: Verifier, table_dir: Path) -> None:
  p = table_dir / 'Table_S5_features_95.csv'
  if not p.exists():
    v.check('Table S5', 'file_exists', False, 'MISSING', str(p))
    return
  df = pd.read_csv(p)
  v.check_equal('Table S5', 'n_features', len(df), 95)
  v.check_equal('Table S5', 'n_categories', int(df['category'].nunique()), 7)
  v.check('Table S5', 'record_length_excluded', 'record_length' not in set(df['variable_symbol']), 'excluded', 'excluded')
  ria_cat = str(df.loc[df['variable_symbol'] == 'ria_ha_usu', 'category'].iloc[0])
  v.check_equal('Table S5 (A10)', 'ria_ha_usu_in_Topography', ria_cat, 'Topography')


def verify_table_s8(v: Verifier, table_dir: Path) -> None:
  p = table_dir / 'Table_S8_ungaugedness_breakdown.csv'
  if not p.exists():
    v.check('Table S8', 'file_exists', False, 'MISSING', str(p))
    return
  df = pd.read_csv(p)
  g = df[(df['part'] == 'Part A: Global & Regional') & (df['group'] == 'Global')].iloc[0]
  v.check_equal('Table S8', 'global_n_basins', int(g['n_basins']), 1034083)
  v.check_close('Table S8', 'global_area_Mkm2', float(g['land_area_km2']) / 1e6, 135.01, atol=0.05)
  v.check_close('Table S8', 'global_runoff_km3_yr', float(g['runoff_km3_yr']), 45633.0, atol=50.0)
  v.check_close('Table S8', 'global_outlet_ung_area_pct', float(g['outlet_ung_area_pct']), 92.0, atol=0.1)
  v.check_close('Table S8', 'global_outlet_ung_basin_pct', float(g['outlet_ung_basin_pct']), 92.6, atol=0.1)
  v.check_close('Table S8', 'global_zero_up_area_pct', float(g['zero_up_area_pct']), 80.5, atol=0.1)
  v.check_close('Table S8', 'global_entire_net_ung_area_pct', float(g['entire_net_ung_area_pct']), 30.9, atol=0.1)
  v.check_close('Table S8', 'global_U_net_area_pct', float(g['U_net_area_pct']), 78.0, atol=0.1)
  v.check_close('Table S8', 'global_U_net_runoff_pct', float(g['U_net_runoff_pct']), 73.7, atol=0.1)
  v.check_close('Table S8', 'global_U_net_basin_mean_pct', float(g['U_net_basin_mean_pct']), 76.5, atol=0.1)
  v.check_close('Table S8', 'global_U_net_basin_sd_pct', float(g['U_net_basin_sd_pct']), 36.4, atol=0.1)
  v.check_close('Table S8', 'global_U_net_basin_median_pct', float(g['U_net_basin_median_pct']), 98.3, atol=0.1)

  # Part B Strahler Order 1 & 7+ checks
  s1 = df[df['group'] == 'Strahler Order 1 (Headwaters)'].iloc[0]
  v.check_equal('Table S8', 'strahler_1_n_basins', int(s1['n_basins']), 593174)
  v.check_close('Table S8', 'strahler_1_U_net_area_pct', float(s1['U_net_area_pct']), 88.2, atol=0.1)
  s7p = df[df['group'] == 'Strahler Order 7+ (Combined)'].iloc[0]
  v.check_equal('Table S8', 'strahler_7plus_n_basins', int(s7p['n_basins']), 4305)
  v.check_close('Table S8', 'strahler_7plus_U_net_area_pct', float(s7p['U_net_area_pct']), 2.4, atol=0.1)

  # Part C tiers
  part_c = df[df['part'] == 'Part C: Data-Accessibility Tier'].set_index('group')
  for lbl, exp_u_area in [
      ('All Publicly Discoverable Physical Gauges (119,169)', 78.0),
      ('Gauges with Discharge Records (82,596)', 82.5),
      ('Standardized, Open-License Discharge Datasets (32,369)', 89.4),
      ('Post-2016 Caravan Training Subset (8,811 records; 8,075 stations)', 96.8),
  ]:
    v.check_close('Table S8', f'part_c.{lbl[:28]}', float(part_c.loc[lbl, 'U_net_area_pct']), exp_u_area, atol=0.1)


def verify_shap_and_cv(v: Verifier, shap_dir: Path, fig_dir: Path, cf_dir: Path) -> None:
  pct_p = shap_dir / 'shap_category_pct_r2.csv'
  if not pct_p.exists():
    v.check('Fig. 6 / SHAP', 'shap_category_pct_exists', False, 'MISSING', str(pct_p))
  else:
    df_pct = pd.read_csv(pct_p, index_col=0)
    recall_cols = [
        '1.01-year-2.0-window-recall',
        '2.0-year-2.0-window-recall',
        '5.0-year-2.0-window-recall',
        '10.0-year-2.0-window-recall',
        '20.0-year-2.0-window-recall',
    ]
    cov_row = [r for r in df_pct.index if 'gauge' in str(r).lower() or 'coverage' in str(r).lower()][0]
    for col in recall_cols:
      vals = df_pct[col]
      cov_val = float(vals.loc[cov_row])
      is_last = cov_val <= float(vals.min()) + 1e-9
      v.check('Fig. 6 / SHAP', f'gauge_cov_last_rank.{col}', is_last, f'{cov_val:.2f}% (min={vals.min():.2f}%)', 'ranked last (7th)')
      v.check('Fig. 6 / SHAP', f'gauge_cov_range.{col}', 0.5 <= cov_val <= 10.0, round(cov_val, 2), '0.5%–10.0%')

  diag_candidates = [
      cf_dir / 'sec3_cv_diagnostics_r2.json',
      fig_dir / 'sec3_cv_diagnostics_r2.json',
      CANON_DIR / ROUND_SUBDIR / 'counterfactuals' / 'sec3_cv_diagnostics_r2.json',
  ]
  diag_p = next((p for p in diag_candidates if p.exists()), None)
  if diag_p is None:
    v.check('Fig. S4 / CV', 'sec3_cv_diagnostics_exists', False, 'MISSING', str(diag_candidates[0]))
  else:
    d = json.loads(diag_p.read_text())
    v.check_equal('Fig. S4 / CV', 'nse_2016_2023_full_n', int(d['nse_2016_2023_full']['n']), 7950)
    v.check_close('Fig. S4 / CV', 'full_rand_minus_sep_med', float(d['nse_2016_2023_full']['paired_rand_minus_sep_med']), 0.052, atol=0.002)
    v.check_close('Fig. S4 / CV', 'full_gauged_minus_rand_med', float(d['nse_2016_2023_full']['paired_gauged_minus_rand_med']), 0.017, atol=0.002)
    v.check_equal('Fig. S4 / CV', 'nse_ge_minus_0_5_n', int(d['nse_2016_2023_ge_minus_0_5']['n']), 6586)
    v.check_close('Fig. S4 / CV', 'subset_rand_minus_sep_med', float(d['nse_2016_2023_ge_minus_0_5']['paired_rand_minus_sep_med']), 0.039, atol=0.002)
    v.check_close('Fig. S4 / CV', 'subset_gauged_minus_rand_med', float(d['nse_2016_2023_ge_minus_0_5']['paired_gauged_minus_rand_med']), 0.015, atol=0.002)


def main() -> None:
  parser = argparse.ArgumentParser(description='Verify full reproduction of figures, tables, and headline numbers.')
  parser.add_argument('--fig-dir', type=str, default=str(CANON_DIR / ROUND_SUBDIR / 'figures'))
  parser.add_argument('--table-dir', type=str, default=str(CANON_DIR / ROUND_SUBDIR / 'tables'))
  parser.add_argument('--shap-dir', type=str, default=str(CANON_DIR / ROUND_SUBDIR / 'shap'))
  parser.add_argument('--cf-dir', type=str, default=str(CANON_DIR / ROUND_SUBDIR / 'counterfactuals'))
  parser.add_argument('--s16-dir', type=str, default=str(CANON_DIR / ROUND_SUBDIR / 's16_gauge_placement'))
  parser.add_argument('--out-dir', type=str, default=str(CANON_DIR / ROUND_SUBDIR / 'verification'))
  parser.add_argument('--with-fathom', action='store_true', help='Also verify optional licensed Fathom v3 figures, tables, bootstrap CIs & S1.6 siting.')
  args = parser.parse_args()

  fig_dir = Path(args.fig_dir)
  table_dir = Path(args.table_dir)
  shap_dir = Path(args.shap_dir)
  cf_dir = Path(args.cf_dir)
  s16_dir = Path(args.s16_dir)
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  v = Verifier()
  verify_figures(v, fig_dir, with_fathom=args.with_fathom)
  verify_table_s1(v, table_dir)
  verify_table_s2(v, table_dir)
  verify_table_s3(v, table_dir, with_fathom=args.with_fathom)
  verify_table_s4(v, table_dir)
  verify_table_s5(v, table_dir)
  verify_table_s7(v, table_dir)
  verify_table_s8(v, table_dir)
  verify_shap_and_cv(v, shap_dir, fig_dir, cf_dir)
  if args.with_fathom:
    verify_fathom_outputs(v, table_dir, cf_dir, s16_dir)

  n_total = len(v.checks)
  n_pass = sum(1 for c in v.checks if c['passed'])
  n_fail = n_total - n_pass

  mode_label = 'Full Reproduction (Public + Licensed Fathom v3)' if args.with_fathom else 'Public Non-Fathom Reproduction'
  report = {
      'summary': {
          'mode': mode_label,
          'with_fathom': bool(args.with_fathom),
          'passed': n_fail == 0,
          'n_total': n_total,
          'n_pass': n_pass,
          'n_fail': n_fail,
      },
      'checks': v.checks,
  }
  json_path = out_dir / 'verification_report.json'
  json_path.write_text(json.dumps(report, indent=2))

  md_lines = [
      f'# {mode_label} Verification Report',
      '',
      f'**Overall status:** `{"PASS" if n_fail == 0 else "FAIL"}` ({n_pass}/{n_total} checks passed)',
      '',
      '| Group | Check | Status | Actual | Expected |',
      '| --- | --- | :---: | --- | --- |',
  ]
  for c in v.checks:
    st = 'PASS' if c['passed'] else '**FAIL**'
    md_lines.append(f"| {c['group']} | `{c['name']}` | {st} | `{c['actual']}` | `{c['expected']}` |")
  md_path = out_dir / 'verification_report.md'
  md_path.write_text('\n'.join(md_lines) + '\n')

  print(f'[verify_reproduction] {n_pass}/{n_total} checks passed ({n_fail} failed). Report: {json_path}', flush=True)
  if n_fail > 0:
    print('\nFAILED CHECKS:', flush=True)
    for c in v.checks:
      if not c['passed']:
        print(f"  - [{c['group']}] {c['name']}: actual={c['actual']} vs expected={c['expected']}", flush=True)
    sys.exit(1)


if __name__ == '__main__':
  main()
