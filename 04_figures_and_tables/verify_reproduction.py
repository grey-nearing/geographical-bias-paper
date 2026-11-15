#!/usr/bin/env python3
"""End-to-end automated verification of non-Fathom manuscript reproduction.

Checks all 10 public figures, all 6 public tables (Table 1, Tables S1–S5),
and every headline quantitative finding reported in the manuscript (excluding
proprietary Fathom v3 flood-exposure outputs). Writes `verification_report.json`
and `verification_report.md` and exits with code 0 iff all checks pass.
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
    'Fig_2_polygon_size_dependence.png',
    'Fig_3_regional_aridity_dissimilarity.png',
    'Fig_4_cap_gap_and_inform.png',
    'Fig_6_exact_group_shapley.png',
    'Fig_S1_delta_nse_network_vs_local.png',
    'Fig_S2_top35_permutation_shap.png',
    'Fig_S3_conceptual_ungaugedness.png',
    'Fig_S4_ood_sextiles_recall.png',
    'Fig_S5_global_predicted_skill.png',
]

EXPECTED_TABLES = [
    'Table_1_ungaugedness_ranges.csv',
    'Table_S1_gauge_status.csv',
    'Table_S2_175territories.csv',
    'Table_S3_4regimes_skill.csv',
    'Table_S4_features_95.csv',
    'Table_S5_ungaugedness_breakdown.csv',
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


def verify_figures(v: Verifier, fig_dir: Path) -> None:
  for fname in EXPECTED_FIGURES:
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


def verify_table1(v: Verifier, table_dir: Path) -> None:
  p = table_dir / 'Table_1_ungaugedness_ranges.csv'
  if not p.exists():
    v.check('Table 1', 'file_exists', False, 'MISSING', str(p))
    return
  df = pd.read_csv(p)

  def _row(metric: str, weight: str) -> pd.Series:
    sub = df[(df['metric'] == metric) & (df['weight'] == weight)]
    if len(sub) != 1:
      raise KeyError(f'Missing row metric={metric}, weight={weight}')
    return sub.iloc[0]

  expectations = [
      ('U_never', 'area', (66.5, 68.4), (40.6, 43.5), (31.6, 35.1)),
      ('U_never', 'runoff', (59.8, 65.9), (32.8, 39.4), (23.1, 31.4)),
      ('U_never', 'population', (78.3, 79.2), (42.9, 44.6), (27.9, 30.4)),
      ('U_net_Eq4', 'area', (91.7, 96.8), (83.7, 93.7), (79.7, 92.1)),
      ('U_net_Eq4', 'runoff', (89.8, 95.6), (80.2, 90.7), (75.7, 87.9)),
      ('U_net_Eq4', 'population', (88.1, 95.0), (72.3, 88.2), (61.1, 83.4)),
      ('U_net_1d', 'Q_ds', (44.8, 53.5), (15.7, 21.6), (9.5, 11.7)),
  ]
  for metric, weight, t1, t2, t3 in expectations:
    try:
      r = _row(metric, weight)
    except KeyError as e:
      v.check('Table 1', f'{metric}_{weight}', False, str(e), 'row present')
      continue
    for prefix, (exp_lo, exp_hi) in [('tier1', t1), ('tier2', t2), ('tier3', t3)]:
      v.check_close('Table 1', f'{metric}.{weight}.{prefix}_min', float(r[f'{prefix}_min_pct']), exp_lo, atol=0.15)
      v.check_close('Table 1', f'{metric}.{weight}.{prefix}_max', float(r[f'{prefix}_max_pct']), exp_hi, atol=0.15)


def verify_table_s1(v: Verifier, table_dir: Path) -> None:
  p = table_dir / 'Table_S1_gauge_status.csv'
  if not p.exists():
    v.check('Table S1', 'file_exists', False, 'MISSING', str(p))
    return
  df = pd.read_csv(p).set_index('category_key')
  expected = {
      'active_total': (22292, 39461, 50596),
      'verified_ge_2016': (22289, 34290, 45353),
      'portal_active_no_end_year': (3, 5171, 5243),
      'historical_or_inactive_total': (10780, 24320, 30218),
      'verified_le_2015': (10726, 22138, 28025),
      'portal_inactive': (31, 1469, 1478),
      'portal_active_record_le_2015': (23, 713, 715),
      'unclassifiable_total': (1505, 30046, 41005),
      'live_portal_undated': (756, 28149, 39087),
      'static_archive_undated': (749, 1897, 1918),
      'total_distinct_gauges': (34577, 93827, 121819),
      'section3_assimilation_total': (23071, 68323, 90398),
  }
  for key, (e1, e2, e3) in expected.items():
    for col, exp in [('tier1_count', e1), ('tier2_count', e2), ('tier3_count', e3)]:
      v.check_equal('Table S1', f'{key}.{col}', int(df.loc[key, col]), exp)


def verify_table_s2(v: Verifier, table_dir: Path) -> None:
  p = table_dir / 'Table_S2_175territories.csv'
  sp = table_dir / 'Table_S2_summary_stats.json'
  if not p.exists() or not sp.exists():
    v.check('Table S2', 'files_exist', False, 'MISSING', f'{p} and {sp}')
    return
  df = pd.read_csv(p)
  stats = json.loads(sp.read_text())
  v.check_equal('Table S2 / Fig. 4', 'n_territories', len(df), 175)
  v.check_equal('Table S2 / Fig. 4', 'n_established_system', stats['n_established_system'], 60)
  v.check_equal('Table S2 / Fig. 4', 'n_cap_gap_excl_greenland', stats['n_cap_gap_excl_greenland'], 114)
  v.check_equal('Table S2 / Fig. 4', 'n_zero_open_gauges_sys', stats['n_zero_open_gauges_sys'], 19)
  v.check_equal('Table S2 / Fig. 4', 'n_zero_open_gauges_cap_gap', stats['n_zero_open_gauges_cap_gap'], 86)
  v.check_close('Table S2 / Fig. 4', 'land_ung_median_cap_gap_pct', stats['land_ung_median_cap_gap_pct'], 88.6, atol=0.15)
  v.check_close('Table S2 / Fig. 4', 'land_ung_median_sys_pct', stats['land_ung_median_sys_pct'], 59.4, atol=0.15)
  v.check('Table S2 / Fig. 4', 'land_ung_mannwhitney_p', stats['land_ung_mannwhitney_p'] < 1e-8, stats['land_ung_mannwhitney_p'], '< 1e-8')
  v.check_close('Table S2 / Fig. 4', 'pop_ung_median_cap_gap_pct', stats['pop_ung_median_cap_gap_pct'], 81.8, atol=0.15)
  v.check_close('Table S2 / Fig. 4', 'pop_ung_median_sys_pct', stats['pop_ung_median_sys_pct'], 39.1, atol=0.15)
  v.check('Table S2 / Fig. 4', 'pop_ung_mannwhitney_p', stats['pop_ung_mannwhitney_p'] < 1e-10, stats['pop_ung_mannwhitney_p'], '< 1e-10')
  v.check_equal('Table S2 / Fig. 4', 'inform_n', stats['inform_n'], 166)
  v.check_close('Table S2 / Fig. 4', 'inform_vuln_spearman_r', stats['inform_vuln_spearman_r'], 0.3179, atol=0.005)
  v.check_close('Table S2 / Fig. 4', 'inform_coping_spearman_r', stats['inform_coping_spearman_r'], 0.5407, atol=0.005)


def verify_table_s3(v: Verifier, table_dir: Path) -> None:
  p = table_dir / 'Table_S3_4regimes_skill.csv'
  if not p.exists():
    v.check('Table S3', 'file_exists', False, 'MISSING', str(p))
    return
  df = pd.read_csv(p)
  v.check_equal('Table S3', 'n_unique_metrics', int(df['metric'].nunique()), 43)

  def _val(metric: str, regime: str, col: str) -> float:
    sub = df[(df['metric'] == metric) & (df['regime'] == regime)]
    return float(sub.iloc[0][col]) if len(sub) > 0 else float('nan')

  # Headline Table S3 & Section 3 values (tolerance 0.015 for ML tree randomness/BLAS)
  checks = [
      ('NSE', 'base', 'obs_ungauged_median', 0.48, 0.006),
      ('NSE', 'ft_da', 'obs_ungauged_median', 0.79, 0.006),
      ('NSE', 'base', 'meta_cv_r2_2split_ung', 0.51, 0.015),
      ('NSE', 'ft_da', 'meta_cv_r2_2split_ung', 0.63, 0.015),
      ('NSE', 'base', 'global_pred_median', 0.15, 0.015),
      ('NSE', 'ft', 'global_pred_median', 0.38, 0.015),
      ('NSE', 'da', 'global_pred_median', 0.58, 0.015),
      ('NSE', 'ft_da', 'global_pred_median', 0.65, 0.015),
      ('KGE', 'base', 'obs_ungauged_median', 0.50, 0.006),
      ('KGE', 'base', 'global_pred_median', 0.19, 0.015),
      ('5.0-year-2.0-window-recall', 'base', 'obs_ungauged_median', 0.38, 0.006),
      ('5.0-year-2.0-window-recall', 'ft_da', 'obs_ungauged_median', 0.50, 0.006),
      ('5.0-year-2.0-window-recall', 'base', 'meta_cv_r2_2split_ung', 0.36, 0.015),
      ('5.0-year-2.0-window-recall', 'base', 'global_pred_median', 0.30, 0.015),
      ('5.0-year-2.0-window-recall', 'da', 'global_pred_median', 0.39, 0.015),
      ('5.0-year-2.0-window-recall', 'ft_da', 'global_pred_median', 0.44, 0.015),
      ('5.0-year-2.0-window-precision', 'base', 'obs_ungauged_median', 0.25, 0.006),
      ('5.0-year-2.0-window-precision', 'base', 'global_pred_median', 0.20, 0.015),
      ('5.0-year-2.0-window-f1', 'base', 'obs_ungauged_median', 0.30, 0.006),
      ('5.0-year-2.0-window-f1', 'base', 'global_pred_median', 0.22, 0.015),
  ]
  for metric, reg, col, exp, tol in checks:
    v.check_close('Table S3', f'{metric}.{reg}.{col}', _val(metric, reg, col), exp, atol=tol)


def verify_table_s4(v: Verifier, table_dir: Path) -> None:
  p = table_dir / 'Table_S4_features_95.csv'
  if not p.exists():
    v.check('Table S4', 'file_exists', False, 'MISSING', str(p))
    return
  df = pd.read_csv(p)
  v.check_equal('Table S4', 'n_features', len(df), 95)
  v.check_equal('Table S4', 'n_categories', int(df['category'].nunique()), 7)
  v.check('Table S4', 'record_length_excluded', 'record_length' not in set(df['variable_symbol']), 'excluded', 'excluded')


def verify_table_s5(v: Verifier, table_dir: Path) -> None:
  p = table_dir / 'Table_S5_ungaugedness_breakdown.csv'
  if not p.exists():
    v.check('Table S5', 'file_exists', False, 'MISSING', str(p))
    return
  df = pd.read_csv(p)
  g = df[(df['part'] == 'Part A: Global & Regional') & (df['group'] == 'Global')].iloc[0]
  v.check_equal('Table S5', 'global_n_basins', int(g['n_basins']), 1034083)
  v.check_close('Table S5', 'global_area_Mkm2', float(g['land_area_km2']) / 1e6, 135.01, atol=0.05)
  v.check_close('Table S5', 'global_runoff_km3_yr', float(g['runoff_km3_yr']), 45633.0, atol=50.0)
  v.check_close('Table S5', 'global_outlet_ung_area_pct', float(g['outlet_ung_area_pct']), 91.9, atol=0.1)
  v.check_close('Table S5', 'global_outlet_ung_basin_pct', float(g['outlet_ung_basin_pct']), 92.5, atol=0.1)
  v.check_close('Table S5', 'global_zero_up_area_pct', float(g['zero_up_area_pct']), 80.4, atol=0.1)
  v.check_close('Table S5', 'global_entire_net_ung_area_pct', float(g['entire_net_ung_area_pct']), 30.9, atol=0.1)
  v.check_close('Table S5', 'global_U_net_area_pct', float(g['U_net_area_pct']), 77.9, atol=0.1)
  v.check_close('Table S5', 'global_U_net_runoff_pct', float(g['U_net_runoff_pct']), 73.4, atol=0.1)
  v.check_close('Table S5', 'global_U_net_basin_mean_pct', float(g['U_net_basin_mean_pct']), 76.4, atol=0.1)
  v.check_close('Table S5', 'global_U_net_basin_sd_pct', float(g['U_net_basin_sd_pct']), 36.5, atol=0.1)
  v.check_close('Table S5', 'global_U_net_basin_median_pct', float(g['U_net_basin_median_pct']), 98.3, atol=0.1)

  # Part B Strahler Order 1 & 7+ checks
  s1 = df[df['group'] == 'Strahler Order 1 (Headwaters)'].iloc[0]
  v.check_equal('Table S5', 'strahler_1_n_basins', int(s1['n_basins']), 593174)
  v.check_close('Table S5', 'strahler_1_U_net_area_pct', float(s1['U_net_area_pct']), 88.1, atol=0.1)
  s7p = df[df['group'] == 'Strahler Order 7+ (Combined)'].iloc[0]
  v.check_equal('Table S5', 'strahler_7plus_n_basins', int(s7p['n_basins']), 4305)
  v.check_close('Table S5', 'strahler_7plus_U_net_area_pct', float(s7p['U_net_area_pct']), 2.4, atol=0.1)

  # Part C tiers
  part_c = df[df['part'] == 'Part C: Data-Accessibility Tier'].set_index('group')
  for lbl, exp_u_area in [
      ('All Publicly Discoverable Gauges (121,819)', 77.9),
      ('Confirmed Publicly Accessible Records (93,827)', 82.5),
      ('Standardized Open Compilations (Caravan + EStreams; 34,577)', 91.2),
      ('Post-2016 Caravan Training Subset (8,811 records; 8,075 stations)', 96.8),
  ]:
    v.check_close('Table S5', f'part_c.{lbl[:28]}', float(part_c.loc[lbl, 'U_net_area_pct']), exp_u_area, atol=0.1)


def verify_shap_and_cv(v: Verifier, shap_dir: Path, fig_dir: Path) -> None:
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
      v.check('Fig. 6 / SHAP', f'gauge_cov_range.{col}', 1.0 <= cov_val <= 7.0, round(cov_val, 2), '1.0%–7.0% (manuscript ~2–6%)')

  diag_candidates = [
      fig_dir / 'sec3_cv_diagnostics_r2.json',
      CANON_DIR / 'round2' / 'counterfactuals' / 'sec3_cv_diagnostics_r2.json',
  ]
  diag_p = next((p for p in diag_candidates if p.exists()), None)
  if diag_p is None:
    v.check('Fig. S1 / CV', 'sec3_cv_diagnostics_exists', False, 'MISSING', str(diag_candidates[0]))
  else:
    d = json.loads(diag_p.read_text())
    v.check_equal('Fig. S1 / CV', 'nse_2016_2023_full_n', int(d['nse_2016_2023_full']['n']), 7950)
    v.check_close('Fig. S1 / CV', 'full_rand_minus_sep_med', float(d['nse_2016_2023_full']['paired_rand_minus_sep_med']), 0.052, atol=0.002)
    v.check_close('Fig. S1 / CV', 'full_gauged_minus_rand_med', float(d['nse_2016_2023_full']['paired_gauged_minus_rand_med']), 0.017, atol=0.002)
    v.check_equal('Fig. S1 / CV', 'nse_ge_minus_0_5_n', int(d['nse_2016_2023_ge_minus_0_5']['n']), 6586)
    v.check_close('Fig. S1 / CV', 'subset_rand_minus_sep_med', float(d['nse_2016_2023_ge_minus_0_5']['paired_rand_minus_sep_med']), 0.039, atol=0.002)
    v.check_close('Fig. S1 / CV', 'subset_gauged_minus_rand_med', float(d['nse_2016_2023_ge_minus_0_5']['paired_gauged_minus_rand_med']), 0.015, atol=0.002)


def main() -> None:
  parser = argparse.ArgumentParser(description='Verify full non-Fathom reproduction of figures, tables, and headline numbers.')
  parser.add_argument('--fig-dir', type=str, default=str(CANON_DIR / 'round2' / 'figures'))
  parser.add_argument('--table-dir', type=str, default=str(CANON_DIR / 'round2' / 'tables'))
  parser.add_argument('--shap-dir', type=str, default=str(CANON_DIR / 'round2' / 'shap'))
  parser.add_argument('--out-dir', type=str, default=str(CANON_DIR / 'round2' / 'verification'))
  args = parser.parse_args()

  fig_dir = Path(args.fig_dir)
  table_dir = Path(args.table_dir)
  shap_dir = Path(args.shap_dir)
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  v = Verifier()
  verify_figures(v, fig_dir)
  verify_table1(v, table_dir)
  verify_table_s1(v, table_dir)
  verify_table_s2(v, table_dir)
  verify_table_s3(v, table_dir)
  verify_table_s4(v, table_dir)
  verify_table_s5(v, table_dir)
  verify_shap_and_cv(v, shap_dir, fig_dir)

  n_total = len(v.checks)
  n_pass = sum(1 for c in v.checks if c['passed'])
  n_fail = n_total - n_pass

  report = {
      'summary': {
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
      '# Non-Fathom Reproduction Verification Report',
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
