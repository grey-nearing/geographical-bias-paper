#!/usr/bin/env python3
"""Generate citation-ordered supplementary figures (Figs. S1–S6 public; optional S7–S8 if Fathom present).

Produces 8 supplementary figures natively authored at Science full-page width (184 mm = 7.244 in)
with every rendered font >= 6.0 pt:
  - Fig_S1_gauge_status_histograms.png: Operational & temporal status of 119,169 stream gauges
    across tiers plus exact histograms of record end year, record length, and latitude.
  - Fig_S2_conceptual_ungaugedness.png: Copies pre-rendered conceptual river-network schematic
    from results/figures/Fig_S3_conceptual_ungaugedness.png (or Fig_S2_conceptual_ungaugedness.png).
  - Fig_S3_alerting_coverage_map.png: Global map of national flood-alerting & WMO CAP coverage
    (60 established national alerting systems vs. 114 CAP-gap countries; Greenland excluded).
  - Fig_S4_delta_nse_network_vs_local.png: Harmonized 2016–2023 ΔNSE across upstream/downstream
    coverage bins (plus writes sec3_cv_diagnostics_r2.json).
  - Fig_S5_ood_sextiles_recall.png: 3x3 OOD sextile boxplots of 5-year 2-day causal recall.
  - Fig_S6_global_predicted_skill.png: Global predicted skill distributions across 1,034,083
    watersheds (from Table_S3_4regimes_r2.csv).
  - Fig_S7_precision_recall_curves.png (OPTIONAL — licensed Fathom v3): (a) Flood Precision vs.
    Recall curves across return periods (1y–10y) and 6 gauge-access stages using P_50 directly
    on 50–100 yr exposure; (b) Sensitivity of CAP-gap effectively warned exposure across
    warning-to-action regimes (alpha = 1, CSI/R, Bayesian P, INFORM Coping Capacity, and false-alarm volume).
  - Fig_S8_national_gauge_siting.png (OPTIONAL — licensed Fathom v3): National N = 1..20 strategic
    gauge-siting maps and Top-15 CAP-gap country trajectories (fixed legend placement, >= 10 cm).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
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

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import seaborn as sns

from utils import cap_utils, file_paths

warnings.filterwarnings('ignore')

SCIENCE_W = 184.0 / 25.4
SAVE_DPI = 350

COL_T1 = '#0072B2'   # Blue: Recent data (>=2016) / Alerting system / Standardized open
COL_T2 = '#009E73'   # Green: Portal-listed active (undated) / Public discharge
COL_T3 = '#D55E00'   # Vermilion: Apparently inactive / CAP-gap / All discoverable
COL_UNK = '#64748B'  # Slate: Unknown / undated

ROUND_SUBDIR = os.environ.get('GEO_BIAS_ROUND_SUBDIR', 'round2')
R2_DIR = CANON_DIR / ROUND_SUBDIR
METRICS_R2_DIR = CANON_DIR / 'round2' / 'hydro_model_metrics'
CF_DIR = R2_DIR / 'counterfactuals'
S16_DIR = R2_DIR / 's16_gauge_placement'
DEFAULT_OUT_DIR = R2_DIR / 'figures'


def _set_pub_rc(font_size: float = 6.6) -> None:
  sns.set_theme(style='whitegrid', palette='colorblind')
  plt.rcParams.update({
      'figure.dpi': SAVE_DPI,
      'savefig.dpi': SAVE_DPI,
      'font.family': 'sans-serif',
      'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
      'font.size': font_size,
      'axes.titlesize': font_size + 0.6,
      'axes.titleweight': 'bold',
      'axes.labelsize': font_size,
      'axes.labelweight': 'bold',
      'axes.linewidth': 0.6,
      'axes.unicode_minus': False,
      'figure.facecolor': 'white',
      'axes.facecolor': 'white',
      'savefig.facecolor': 'white',
  })


def load_r2_metric_series(exp_name: str, metric: str, rename_to: str | None = None) -> pd.Series:
  p_r2 = METRICS_R2_DIR / f'{exp_name}__{metric}.csv'
  if not p_r2.exists():
    p_r2 = METRICS_R2_DIR / exp_name / f'{metric}.csv'
  if not p_r2.exists():
    raise FileNotFoundError(f'Missing OHN metric CSV for {exp_name}/{metric} under {METRICS_R2_DIR}')
  df = pd.read_csv(p_r2, index_col=0)
  col = '1' if '1' in df.columns else ('lead_1' if 'lead_1' in df.columns else df.columns[0])
  s = pd.to_numeric(df[col], errors='coerce')
  s.index = s.index.astype(str)
  s.name = rename_to or metric
  return s


def norm_hybas_id(s: pd.Series | pd.Index) -> np.ndarray:
  s2 = pd.Series(s).astype(str).str.replace(r'\.0$', '', regex=True)
  return np.where(s2.str.startswith('hybas_'), s2, 'hybas_' + s2)


# ==============================================================================
# FIGURE S1: Operational & Temporal Status Histograms (119,169 Physical Gauges)
# ==============================================================================
def build_fig_s1(out_dir: Path) -> Path:
  print('[Fig S1] Rendering gauge operational & temporal status histograms...', flush=True)
  _set_pub_rc(6.5)
  g = pd.read_parquet(CANON_DIR / 'attributes' / 'global_gauges_deduplicated.parquet')
  g['tier1'] = g['in_tier_1'].astype(bool)
  g['tier2'] = g['in_tier_2'].astype(bool)

  vr = g['verified_record_end_year']
  ar = g['active_reason']
  g['status_4'] = np.where(
      vr.notna() & (vr >= 2016), 'Recent data (>=2016)',
      np.where(
          (vr.isna()) & (ar == '3_portal_status_active'), 'Portal active (undated)',
          np.where(
              (vr.notna() & (vr < 2016))
              | ar.isin(['5_portal_status_inactive', '6_end_year_lt_2016'])
              | ((ar == '3_portal_status_active') & (vr < 2016)),
              'Apparently inactive',
              'Unknown / undated',
          ),
      ),
  )

  def _step_hist(ax, vals, bins, color, label, lw=1.35, alpha_fill=0.14):
    w = np.full(len(vals), 100.0 / len(vals), dtype=float)
    ax.hist(vals, bins=bins, weights=w, histtype='stepfilled', color=color, alpha=alpha_fill, edgecolor='none', zorder=2)
    ax.hist(vals, bins=bins, weights=w, histtype='step', color=color, linewidth=lw, label=label, zorder=4)

  fig, axes = plt.subplots(
      2, 2, figsize=(SCIENCE_W, 5.45), dpi=SAVE_DPI,
      gridspec_kw={'wspace': 0.28, 'hspace': 0.44, 'left': 0.12, 'right': 0.985, 'top': 0.875, 'bottom': 0.095},
  )

  # (a) Gauge status across data-access tiers
  ax_a = axes[0, 0]
  tiers = [
      (f'All Discoverable\n({len(g):,})', np.ones(len(g), dtype=bool)),
      (f'Open Discharge\n({int(g["tier2"].sum()):,})', g['tier2'].values),
      (f'Open Datasets\n({int(g["tier1"].sum()):,})', g['tier1'].values),
  ]
  cats = [
      ('Recent data (>=2016)', COL_T1, 'Recent data (≥2016)'),
      ('Portal active (undated)', COL_T2, 'Portal-listed active (undated)'),
      ('Apparently inactive', COL_T3, 'Apparently inactive (≤2015 / portal)'),
      ('Unknown / undated', COL_UNK, 'Unknown (no end date / status)'),
  ]
  y_pos = np.arange(len(tiers))
  lefts = np.zeros(len(tiers))
  handles_a = []
  for key, col, lbl in cats:
    pcts = [int((g[mask]['status_4'] == key).sum()) / len(g[mask]) * 100.0 for _, mask in tiers]
    bars = ax_a.barh(y_pos, pcts, left=lefts, height=0.56, color=col, alpha=0.90, edgecolor='white', linewidth=0.6, label=lbl, zorder=3)
    handles_a.append(bars)
    for yi, (l, p) in enumerate(zip(lefts, pcts)):
      if p >= 11.5:
        ax_a.text(l + p / 2.0, yi, f'{p:.1f}%', ha='center', va='center', fontsize=6.1, fontweight='bold', color='white', zorder=4)
    lefts += np.array(pcts)

  ax_a.set_yticks(y_pos)
  ax_a.set_yticklabels([t[0] for t in tiers], fontsize=6.3, fontweight='bold')
  ax_a.set_xlim(0, 100)
  ax_a.tick_params(axis='x', labelsize=6.2)
  ax_a.set_xlabel('Share of gauges in tier (%)', fontsize=6.7, fontweight='bold')
  ax_a.set_title('(a) Gauge status across data-access tiers', fontsize=7.2, fontweight='bold', loc='left', pad=5)
  ax_a.grid(True, axis='x', linestyle='--', alpha=0.35, color='#CBD5E1', linewidth=0.5)
  sns.despine(ax=ax_a)

  # (b) Record end year (1-yr bins, 1940–2026)
  ax_b = axes[0, 1]
  bins_end = np.arange(1939.5, 2027.5, 1.0)
  for mask, col, tier_lbl, lw, af in [
      (np.ones(len(g), dtype=bool), COL_T3, 'All discoverable', 1.35, 0.10),
      (g['tier2'].values, COL_T2, 'Open discharge', 1.25, 0.08),
      (g['tier1'].values, COL_T1, 'Open datasets', 1.25, 0.12),
  ]:
    vals_all = g.loc[mask, 'verified_record_end_year'].dropna().values
    vals_win = vals_all[(vals_all >= 1940) & (vals_all <= 2026)]
    w = np.full(len(vals_win), 100.0 / len(vals_all), dtype=float)
    ax_b.hist(vals_win, bins=bins_end, weights=w, histtype='stepfilled', color=col, alpha=af, edgecolor='none', zorder=2)
    ax_b.hist(vals_win, bins=bins_end, weights=w, histtype='step', color=col, linewidth=lw, label=f'{tier_lbl} ({len(vals_all):,})', zorder=4)

  n_le15 = int((vr.notna() & (vr <= 2015)).sum())
  n_ge16 = int((vr.notna() & (vr >= 2016)).sum())
  n_nodate = int(vr.isna().sum())
  pct_nodate = 100.0 * n_nodate / len(g)
  ax_b.axvline(2015.5, color='#0F172A', linestyle='--', linewidth=0.95, zorder=5)
  ax_b.axvspan(2015.5, 2029.0, color='#E0F2FE', alpha=0.35, zorder=1)
  ax_b.text(2013.5, 0.63, f'≤2015\n({n_le15:,})', transform=ax_b.get_xaxis_transform(), ha='right', va='top', fontsize=6.0, fontweight='bold', color='#9A3412')
  ax_b.text(2016.5, 0.96, f'≥2016\n({n_ge16:,})', transform=ax_b.get_xaxis_transform(), ha='left', va='top', fontsize=6.0, fontweight='bold', color='#075985')
  ax_b.text(
      0.03, 0.35, f'{n_nodate:,} gauges ({pct_nodate:.1f}%)\nhave no verified end date',
      transform=ax_b.transAxes, ha='left', va='center', fontsize=6.0,
      bbox=dict(boxstyle='round,pad=0.20', facecolor='#F8FAFC', edgecolor='#CBD5E1', alpha=0.95),
  )
  ax_b.set_xlim(1939.5, 2029.5)
  ax_b.set_ylim(0, 26.5)
  ax_b.set_xlabel('Verified record end year (1-yr bins)', fontsize=6.7, fontweight='bold')
  ax_b.set_ylabel('Share of dated gauges (%)', fontsize=6.7, fontweight='bold')
  ax_b.set_title('(b) Record end year', fontsize=7.2, fontweight='bold', loc='left', pad=5)
  ax_b.tick_params(labelsize=6.2)
  ax_b.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=6.0, framealpha=0.95, borderpad=0.25)
  ax_b.grid(True, linestyle='--', alpha=0.35, color='#CBD5E1', linewidth=0.5)
  sns.despine(ax=ax_b)

  # (c) Record length by active status (5-yr bins)
  ax_c = axes[1, 0]
  bins_len = np.arange(0, 130, 5)
  rl_act = g.loc[g['status_4'] == 'Recent data (>=2016)', 'record_length_years'].dropna().values
  rl_inact = g.loc[g['status_4'] == 'Apparently inactive', 'record_length_years'].dropna().values
  rl_act_c = rl_act[(rl_act >= 0) & (rl_act <= 125)]
  rl_inact_c = rl_inact[(rl_inact >= 0) & (rl_inact <= 125)]
  w_act = np.full(len(rl_act_c), 100.0 / len(rl_act), dtype=float)
  w_in = np.full(len(rl_inact_c), 100.0 / len(rl_inact), dtype=float)
  ax_c.hist(rl_act_c, bins=bins_len, weights=w_act, histtype='stepfilled', color=COL_T1, alpha=0.16, edgecolor='none', zorder=2)
  ax_c.hist(rl_act_c, bins=bins_len, weights=w_act, histtype='step', color=COL_T1, linewidth=1.4, label=f'≥2016 (N={len(rl_act):,}; med {np.median(rl_act):.0f} yr)', zorder=4)
  ax_c.hist(rl_inact_c, bins=bins_len, weights=w_in, histtype='stepfilled', color=COL_T3, alpha=0.16, edgecolor='none', zorder=2)
  ax_c.hist(rl_inact_c, bins=bins_len, weights=w_in, histtype='step', color=COL_T3, linewidth=1.4, label=f'≤2015 (N={len(rl_inact):,}; med {np.median(rl_inact):.0f} yr)', zorder=4)
  ax_c.axvline(np.median(rl_act), color=COL_T1, linestyle='--', linewidth=1.0, zorder=5)
  ax_c.axvline(np.median(rl_inact), color=COL_T3, linestyle='--', linewidth=1.0, zorder=5)
  ax_c.set_xlim(0, 125)
  ax_c.set_ylim(0, 21.0)
  ax_c.set_xlabel('Verified record length (years; 5-yr bins)', fontsize=6.7, fontweight='bold')
  ax_c.set_ylabel('Share of group (%)', fontsize=6.7, fontweight='bold')
  ax_c.set_title('(c) Record length by active status', fontsize=7.2, fontweight='bold', loc='left', pad=5)
  ax_c.tick_params(labelsize=6.2)
  ax_c.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=6.0, framealpha=0.95, borderpad=0.25)
  ax_c.grid(True, linestyle='--', alpha=0.35, color='#CBD5E1', linewidth=0.5)
  sns.despine(ax=ax_c)

  # (d) Latitudinal histogram of gauge status
  ax_d = axes[1, 1]
  bins_lat = np.arange(-56, 76, 4)
  m_active_all = g['status_4'].isin(['Recent data (>=2016)', 'Portal active (undated)'])
  m_inactive = g['status_4'] == 'Apparently inactive'
  m_unknown = g['status_4'] == 'Unknown / undated'
  for mask, col, lbl in [
      (m_active_all, COL_T1, f'Recent / active ({int(m_active_all.sum()):,}; {100.0 * m_active_all.mean():.1f}%)'),
      (m_inactive, COL_T3, f'Inactive ({int(m_inactive.sum()):,}; {100.0 * m_inactive.mean():.1f}%)'),
      (m_unknown, COL_UNK, f'Unknown ({int(m_unknown.sum()):,}; {100.0 * m_unknown.mean():.1f}%)'),
  ]:
    _step_hist(ax_d, g.loc[mask, 'latitude'].dropna().values, bins_lat, col, lbl, lw=1.35, alpha_fill=0.12)

  ax_d.axvline(0, color='#94A3B8', linestyle=':', linewidth=0.8, zorder=1)
  ax_d.set_xlim(-56, 76)
  ax_d.set_ylim(0, 16.5)
  ax_d.set_xlabel('Station latitude (°N; 4° bins)', fontsize=6.7, fontweight='bold')
  ax_d.set_ylabel('Share of group (%)', fontsize=6.7, fontweight='bold')
  ax_d.set_title('(d) Latitudinal distribution of gauge status', fontsize=7.2, fontweight='bold', loc='left', pad=5)
  ax_d.tick_params(labelsize=6.2)
  ax_d.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=6.0, framealpha=0.95, borderpad=0.25)
  ax_d.grid(True, linestyle='--', alpha=0.35, color='#CBD5E1', linewidth=0.5)
  sns.despine(ax=ax_d)

  fig.legend(
      handles_a, [c[2] for c in cats],
      loc='upper center', bbox_to_anchor=(0.54, 0.990), ncol=2,
      frameon=True, facecolor='white', edgecolor='#CBD5E1', framealpha=0.98, fontsize=6.3, borderpad=0.30,
  )

  out_path = out_dir / 'Fig_S1_gauge_status_histograms.png'
  fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.05)
  plt.close(fig)
  print(f'  Saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)', flush=True)
  return out_path


# ==============================================================================
# FIGURE S2: Conceptual Schematic of River-Network Ungaugedness (Copy)
# ==============================================================================
def copy_fig_s2(out_dir: Path) -> Path:
  candidates = [
      REPO_DIR / 'results' / 'figures' / 'Fig_S2_conceptual_ungaugedness.png',
      REPO_DIR / 'results' / 'figures' / 'Fig_S3_conceptual_ungaugedness.png',
  ]
  src = next((p for p in candidates if p.exists()), None)
  dst = out_dir / 'Fig_S2_conceptual_ungaugedness.png'
  if src is not None:
    if src.resolve() != dst.resolve():
      shutil.copy2(src, dst)
    print(f'  Verified/copied {dst} ({dst.stat().st_size / 1e6:.2f} MB)', flush=True)
  else:
    print(f'  [Fig S2] WARNING: Schematic not found in {candidates}', flush=True)
  return dst


# ==============================================================================
# FIGURE S3: Global Map of National Flood-Alerting & WMO CAP Coverage (60 vs 114)
# ==============================================================================
def build_fig_s3(out_dir: Path) -> Path:
  print('[Fig S3] Rendering global national flood-alerting & CAP-gap coverage map...', flush=True)
  _set_pub_rc(6.8)
  w = gpd.read_file(file_paths.NATURAL_EARTH_PATH)
  w = w.dropna(subset=[w.geometry.name])
  w = w[w.is_valid & (~w.is_empty) & (w['name'] != 'Antarctica')].copy()
  w['map_name'] = w['name'].replace(file_paths.country_name_map)

  df_map = pd.read_csv(file_paths.HYBAS_MAPPING_PATH)
  ctrys = sorted(set(df_map['country_name'].dropna().astype(str).replace(file_paths.country_name_map)))
  ctrys = [c for c in ctrys if c != 'Greenland']
  status_map = {c: bool(cap_utils.has_effective_flood_warning_system(c)) for c in ctrys}
  n_sys = sum(1 for v in status_map.values() if v)
  n_gap = sum(1 for v in status_map.values() if not v)

  w['warn_grp'] = w['map_name'].map(status_map)
  fig, ax = plt.subplots(1, 1, figsize=(SCIENCE_W, 3.65), dpi=SAVE_DPI)
  w[w['warn_grp'].isna()].plot(ax=ax, color='#E2E8F0', edgecolor='#94A3B8', linewidth=0.25, zorder=1)
  w[w['warn_grp'] == True].plot(ax=ax, color=COL_T1, edgecolor='#0F172A', linewidth=0.28, alpha=0.88, zorder=2)
  w[w['warn_grp'] == False].plot(ax=ax, color=COL_T3, edgecolor='#0F172A', linewidth=0.28, alpha=0.88, zorder=2)
  ax.set_xlim(-178, 180)
  ax.set_ylim(-58, 84)
  ax.set_axis_off()
  ax.set_title(f'National flood-alerting and WMO CAP coverage across {len(status_map)} countries (Greenland excluded)', fontsize=7.6, fontweight='bold', loc='left', pad=6)
  ax.legend(
      handles=[
          Patch(facecolor=COL_T1, edgecolor='#0F172A', linewidth=0.45, label=f'Established national alerting system (N = {n_sys})'),
          Patch(facecolor=COL_T3, edgecolor='#0F172A', linewidth=0.45, label=f'CAP-gap country without confirmed system (N = {n_gap})'),
          Patch(facecolor='#E2E8F0', edgecolor='#94A3B8', linewidth=0.45, label='Excluded / unmapped (incl. Greenland)'),
      ],
      loc='lower left', frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=6.4, framealpha=0.96, borderpad=0.35,
  )

  out_path = out_dir / 'Fig_S3_alerting_coverage_map.png'
  fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.05)
  plt.close(fig)
  print(f'  Saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)', flush=True)
  return out_path


# ==============================================================================
# FIGURE S4 & FIGURE S5: ΔNSE (2016–2023) & OOD Sextiles (5-yr 2-day Recall)
# ==============================================================================
def build_fig_s4_and_s5(out_dir: Path, build_s4: bool = True, build_s5: bool = True) -> None:
  print('[Fig S4 / S5] Computing CV diagnostics and rendering Fig. S4 & Fig. S5...', flush=True)
  _set_pub_rc(6.8)
  file_paths.init('NSE')

  cv_json: dict = {}

  if build_s4:
    feat = pd.read_csv(file_paths.TRAIN_ML_FEATURES_PATH).set_index('gauge_id')
    feat.index = feat.index.astype(str)
    g_nse = load_r2_metric_series('gauged', 'NSE', 'gauged_NSE')
    r_nse = load_r2_metric_series('random', 'NSE', 'random_NSE')
    s_nse = load_r2_metric_series('separated', 'NSE', 'separated_NSE')

    df_nse = feat.join(g_nse).join(r_nse).join(s_nse).dropna(subset=['gauged_NSE', 'random_NSE', 'separated_NSE'])
    df_nse['delta_sep_rand'] = df_nse['random_NSE'] - df_nse['separated_NSE']
    df_nse['delta_gauged_rand'] = df_nse['gauged_NSE'] - df_nse['random_NSE']
    df_sub = df_nse[(df_nse['gauged_NSE'] >= -0.5) & (df_nse['random_NSE'] >= -0.5) & (df_nse['separated_NSE'] >= -0.5)].copy()

    explicit_bins = [-0.01, 0.01, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ['0', '0.01–0.2', '0.2–0.4', '0.4–0.6', '0.6–0.8', '0.8–1.0']
    melted = (
        df_sub.rename_axis('gauge_id').reset_index()
        .melt(
            id_vars=['gauge_id', 'delta_sep_rand', 'delta_gauged_rand'],
            value_vars=['total_upstream_coverage', 'max_upstream_coverage', 'max_downstream_coverage'],
            var_name='Coverage_Type', value_name='Coverage_Fraction',
        )
    )
    melted['Coverage_Bin'] = pd.cut(melted['Coverage_Fraction'], bins=explicit_bins, labels=bin_labels)
    melted['Coverage Metric'] = melted['Coverage_Type'].map({
        'total_upstream_coverage': 'Total upstream coverage',
        'max_upstream_coverage': 'Max single-gauge upstream',
        'max_downstream_coverage': 'Max downstream coverage',
    })

    fig_s4, axes = plt.subplots(2, 1, figsize=(SCIENCE_W, 4.75), sharex=True, dpi=SAVE_DPI)
    sns.boxplot(data=melted, x='Coverage_Bin', y='delta_sep_rand', hue='Coverage Metric', palette=['#0072B2', '#E69F00', '#009E73'], showfliers=False, whis=[10, 90], linewidth=0.55, ax=axes[0])
    med_rs = float(df_sub['delta_sep_rand'].median())
    axes[0].axhline(med_rs, color='#882255', ls='--', lw=1.0, label=f'Overall median ΔNSE ({med_rs:+.3f})')
    axes[0].axhline(0.0, color='gray', lw=0.65, zorder=0)
    axes[0].set_ylabel('ΔNSE (Random − Separated)', fontsize=6.9, fontweight='bold')
    axes[0].set_title(f'(a) River-network information transfer in ungauged CV (2016–2023; N = {len(df_sub):,})', fontsize=7.4, fontweight='bold', loc='left', pad=4)
    axes[0].set_ylim(-0.18, 0.52)
    axes[0].tick_params(labelsize=6.3)
    axes[0].legend(loc='upper right', fontsize=6.0, frameon=True, framealpha=0.96, ncol=2)
    axes[0].set_xlabel('')
    sns.despine(ax=axes[0])

    sns.boxplot(data=melted, x='Coverage_Bin', y='delta_gauged_rand', hue='Coverage Metric', palette=['#0072B2', '#E69F00', '#009E73'], showfliers=False, whis=[10, 90], linewidth=0.55, ax=axes[1])
    med_gr = float(df_sub['delta_gauged_rand'].median())
    axes[1].axhline(med_gr, color='#882255', ls='--', lw=1.0, label=f'Overall median ΔNSE ({med_gr:+.3f})')
    axes[1].axhline(0.0, color='gray', lw=0.65, zorder=0)
    axes[1].set_ylabel('ΔNSE (Gauged − Random)', fontsize=6.9, fontweight='bold')
    axes[1].set_title(f'(b) Local at-site gauge calibration gain (2016–2023; N = {len(df_sub):,})', fontsize=7.4, fontweight='bold', loc='left', pad=4)
    axes[1].set_ylim(-0.12, 0.225)
    axes[1].tick_params(labelsize=6.3)
    axes[1].legend(loc='upper right', fontsize=6.0, frameon=True, framealpha=0.96, ncol=2)
    axes[1].set_xlabel('Drainage area coverage fraction bin', fontsize=6.9, fontweight='bold')
    sns.despine(ax=axes[1])

    fig_s4.tight_layout()
    out_s4 = out_dir / 'Fig_S4_delta_nse_network_vs_local.png'
    fig_s4.savefig(out_s4, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig_s4)
    print(f'  Saved {out_s4} ({out_s4.stat().st_size / 1e6:.2f} MB)', flush=True)

    cv_json['nse_2016_2023_full'] = {
        'n': int(len(df_nse)),
        'gauged_med': float(df_nse['gauged_NSE'].median()),
        'random_med': float(df_nse['random_NSE'].median()),
        'separated_med': float(df_nse['separated_NSE'].median()),
        'paired_rand_minus_sep_med': float(df_nse['delta_sep_rand'].median()),
        'paired_gauged_minus_rand_med': float(df_nse['delta_gauged_rand'].median()),
        'wilcoxon_rand_vs_sep_p': float(wilcoxon(df_nse['random_NSE'], df_nse['separated_NSE']).pvalue),
        'wilcoxon_gauged_vs_rand_p': float(wilcoxon(df_nse['gauged_NSE'], df_nse['random_NSE']).pvalue),
    }
    cv_json['nse_2016_2023_ge_minus_0_5'] = {
        'n': int(len(df_sub)),
        'pct': float(100.0 * len(df_sub) / len(df_nse)),
        'paired_rand_minus_sep_med': med_rs,
        'paired_gauged_minus_rand_med': med_gr,
    }

  if build_s5:
    rand_anom = pd.read_csv(file_paths.RANDOM_ANOMALIES_PATH).set_index('gauge_id')
    sep_anom = pd.read_csv(file_paths.SEPARATED_ANOMALIES_PATH).set_index('gauge_id')
    rand_anom.index = rand_anom.index.astype(str)
    sep_anom.index = sep_anom.index.astype(str)
    g_r5 = load_r2_metric_series('gauged', '5.0-year-2.0-window-recall', 'gauged_rec5')
    r_r5 = load_r2_metric_series('random', '5.0-year-2.0-window-recall', 'random_rec5')
    s_r5 = load_r2_metric_series('separated', '5.0-year-2.0-window-recall', 'separated_rec5')

    df_top = rand_anom.join(g_r5, how='inner').join(r_r5, how='inner').dropna(subset=['gauged_rec5', 'random_rec5'])
    df_bot = sep_anom.join(g_r5, how='inner').join(s_r5, how='inner').dropna(subset=['gauged_rec5', 'separated_rec5'])

    anom_cols = ['anomaly_score_if', 'anomaly_score_ocsvm', 'anomaly_score_lof']
    anom_titles = ['Isolation Forest', 'One-Class SVM', 'Local Outlier Factor']
    ltrs = [['(a)', '(b)', '(c)'], ['(d)', '(e)', '(f)'], ['(g)', '(h)', '(i)']]
    sextile_lbls = ['Q1\n(OOD)', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6\n(In)']

    fig_s5, axes_s5 = plt.subplots(3, 3, figsize=(SCIENCE_W, 5.95), sharey=True, dpi=SAVE_DPI)
    ood_stats = {}
    for i, (acol, atitle) in enumerate(zip(anom_cols, anom_titles)):
      df_top[f'{acol}_bin'] = pd.qcut(df_top[acol], q=6, labels=sextile_lbls, duplicates='drop')
      df_bot[f'{acol}_bin'] = pd.qcut(df_bot[acol], q=6, labels=sextile_lbls, duplicates='drop')
      ood_stats[acol] = {
          'n_gauges': int(len(df_top)),
          'random_q1_med': float(df_top.loc[df_top[f'{acol}_bin'] == sextile_lbls[0], 'random_rec5'].median()),
          'random_q6_med': float(df_top.loc[df_top[f'{acol}_bin'] == sextile_lbls[-1], 'random_rec5'].median()),
          'sep_q1_med': float(df_bot.loc[df_bot[f'{acol}_bin'] == sextile_lbls[0], 'separated_rec5'].median()),
          'sep_q6_med': float(df_bot.loc[df_bot[f'{acol}_bin'] == sextile_lbls[-1], 'separated_rec5'].median()),
      }
      for row_i, (df_u, ycol, pal, hcol, ylab, ttl) in enumerate([
          (df_top, 'random_rec5', 'Blues_r', '#0d47a1', '5-yr recall (Random)', f'{ltrs[0][i]} Random: {atitle}'),
          (df_bot, 'separated_rec5', 'Oranges_r', '#e65100', '5-yr recall (Separated)', f'{ltrs[1][i]} Separated: {atitle}'),
          (df_top, 'gauged_rec5', 'Greens_r', '#1b5e20', '5-yr recall (Gauged)', f'{ltrs[2][i]} Gauged: {atitle}'),
      ]):
        ax = axes_s5[row_i, i]
        sns.boxplot(data=df_u, x=f'{acol}_bin', y=ycol, hue=f'{acol}_bin', palette=pal, dodge=False, width=0.58, legend=False, showfliers=False, whis=[10, 90], linewidth=0.55, ax=ax)
        ax.axhline(df_u[ycol].median(), color=hcol, ls='--', lw=0.9)
        ax.set_xlabel(f'{atitle} sextile' if row_i == 2 else '', fontsize=6.5, fontweight='bold')
        ax.set_title(ttl, fontsize=6.9, fontweight='bold', loc='left', pad=2.5)
        ax.set_ylabel(ylab if i == 0 else '', fontsize=6.7, fontweight='bold')
        ax.tick_params(axis='both', labelsize=6.0)
        sns.despine(ax=ax)

    fig_s5.tight_layout()
    out_s5 = out_dir / 'Fig_S5_ood_sextiles_recall.png'
    fig_s5.savefig(out_s5, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig_s5)
    print(f'  Saved {out_s5} ({out_s5.stat().st_size / 1e6:.2f} MB)', flush=True)
    cv_json['ood_sextiles_5yr_recall'] = ood_stats

  if cv_json:
    CF_DIR.mkdir(parents=True, exist_ok=True)
    diag_path = CF_DIR / 'sec3_cv_diagnostics_r2.json'
    if diag_path.exists():
      try:
        prev = json.loads(diag_path.read_text())
        prev.update(cv_json)
        cv_json = prev
      except Exception:
        pass
    diag_path.write_text(json.dumps(cv_json, indent=2))
    print(f'  Saved {diag_path}', flush=True)


# ==============================================================================
# FIGURE S6: Global Predicted Skill Boxplots across 1,034,083 Watersheds
# ==============================================================================
def build_fig_s6(out_dir: Path) -> Path:
  print('[Fig S6] Rendering global predicted skill boxplots (1,034,083 watersheds)...', flush=True)
  _set_pub_rc(6.8)
  p_ts = CF_DIR / 'Table_S3_4regimes_r2.csv'
  if not p_ts.exists():
    p_ts = CF_DIR / 'Table_S1_4regimes_r2.csv'
  df_ts = pd.read_csv(p_ts)
  df_base = df_ts[df_ts['regime'] == 'base'].set_index('metric')

  metrics_f6 = [
      ('NSE', 'NSE'), ('KGE', 'KGE'),
      ('1.01-year-2.0-window-recall', 'Recall (1y)'), ('2.0-year-2.0-window-recall', 'Recall (2y)'),
      ('5.0-year-2.0-window-recall', 'Recall (5y)'), ('10.0-year-2.0-window-recall', 'Recall (10y)'),
      ('20.0-year-2.0-window-recall', 'Recall (20y)'),
      ('1.01-year-2.0-window-precision', 'Precision (1y)'), ('2.0-year-2.0-window-precision', 'Precision (2y)'),
      ('5.0-year-2.0-window-precision', 'Precision (5y)'), ('10.0-year-2.0-window-precision', 'Precision (10y)'),
      ('20.0-year-2.0-window-precision', 'Precision (20y)'),
  ]
  box_stats = []
  for m_id, m_lbl in metrics_f6:
    r = df_base.loc[m_id]
    box_stats.append({
        'label': m_lbl,
        'med': float(r['global_pred_median']),
        'q1': float(r['global_pred_q25']),
        'q3': float(r['global_pred_q75']),
        'whislo': float(r['global_pred_q10']),
        'whishi': float(r['global_pred_q90']),
        'fliers': [],
    })

  fig, ax = plt.subplots(figsize=(SCIENCE_W, 3.95), dpi=SAVE_DPI)
  bp = ax.bxp(list(reversed(box_stats)), vert=False, patch_artist=True, showfliers=False, widths=0.62)
  cols = ['#CC79A7'] * 5 + ['#009E73'] * 5 + ['#0072B2'] * 2
  for p, c in zip(bp['boxes'], cols):
    p.set_facecolor(c)
    p.set_alpha(0.84)
    p.set_edgecolor('black')
    p.set_linewidth(0.55)
  for m_l in bp['medians']:
    m_l.set_color('black')
    m_l.set_linewidth(1.2)
  for w_l in bp['whiskers'] + bp['caps']:
    w_l.set_linewidth(0.6)
  ax.axvline(0.0, color='#616161', ls='--', lw=0.7)
  xmin = min(-0.44, min(b['whislo'] for b in box_stats) - 0.05)
  xmax = max(0.72, max(b['whishi'] for b in box_stats) + 0.05)
  ax.set_xlim([xmin, xmax])
  ax.tick_params(labelsize=6.4)
  ax.set_xlabel('Predicted hydrological skill across 1,034,083 global watersheds', fontsize=7.0, fontweight='bold')
  ax.set_title('Global watershed forecast skill distributions (post-2016 Caravan training set)', fontsize=7.5, fontweight='bold', loc='left', pad=4)
  handles = [
      Patch(facecolor='#0072B2', edgecolor='black', alpha=0.84, label='Continuous (NSE, KGE; 2016–2023)'),
      Patch(facecolor='#009E73', edgecolor='black', alpha=0.84, label='Flood recall (7-day causal)'),
      Patch(facecolor='#CC79A7', edgecolor='black', alpha=0.84, label='Flood precision (7-day causal)'),
  ]
  ax.legend(handles=handles, loc='lower left', frameon=True, framealpha=0.95, fontsize=6.1, handlelength=1.1, borderpad=0.30, labelspacing=0.22)
  ax.grid(axis='x', linestyle='--', alpha=0.45)
  sns.despine(ax=ax)
  fig.tight_layout()

  out_path = out_dir / 'Fig_S6_global_predicted_skill.png'
  fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.04)
  plt.close(fig)
  print(f'  Saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)', flush=True)
  return out_path


# ==============================================================================
# FIGURE S7 (OPTIONAL — Fathom v3): Precision-Recall Curves (P_50 Direct) &
#                                   Warning-to-Action Regime Sensitivity (A1)
# ==============================================================================
def build_fig_s7_optional(out_dir: Path) -> Path | None:
  pf_csv = CF_DIR / 'precision_f1_csi_alpha_sensitivity_r2.csv'
  mm_csv = CF_DIR / 'multi_mechanism_tiered_exposure_r2.csv'
  if not (pf_csv.exists() and mm_csv.exists()):
    print(f'[Fig S7] SKIPPED: Licensed Fathom v3 precision/alpha sensitivity outputs not found in {CF_DIR}.', flush=True)
    return None

  print('[Fig S7] Rendering Precision-Recall curves (P_50 direct) & A1 action-regime sensitivity...', flush=True)
  _set_pub_rc(6.5)
  pf = pd.read_csv(pf_csv)
  mm = pd.read_csv(mm_csv).set_index(['tier_key', 'mechanism'])
  pf1 = pf[pf['alpha'] == 1.0].set_index(['tier_key', 'mechanism'])

  e_cap = float(mm.loc[('tier1a_active_post2016_ccby_8811', '000_global_train_joint'), 'E_total_nosys115'])
  e_glob = float(mm.loc[('tier1a_active_post2016_ccby_8811', '000_global_train_joint'), 'E_total_global'])

  # Empirical Caravan 1y–10y Precision vs. Recall (ungauged vs locally adapted FT+DA)
  caravan_ung, caravan_adp = [], []
  for rp, lbl in [('1.01', '1y'), ('2.0', '2y'), ('5.0', '5y'), ('10.0', '10y')]:
    r_u = pd.concat([load_r2_metric_series(d, f'{rp}-year-2.0-window-recall') for d in ('random', 'separated')], ignore_index=True).dropna().median() * 100.0
    p_u = pd.concat([load_r2_metric_series(d, f'{rp}-year-2.0-window-precision') for d in ('random', 'separated')], ignore_index=True).dropna().median() * 100.0
    r_a = pd.concat([load_r2_metric_series(d, f'{rp}-year-2.0-window-recall') for d in ('random_ft_da', 'separated_ft_da')], ignore_index=True).dropna().median() * 100.0
    p_a = pd.concat([load_r2_metric_series(d, f'{rp}-year-2.0-window-precision') for d in ('random_ft_da', 'separated_ft_da')], ignore_index=True).dropna().median() * 100.0
    caravan_ung.append((lbl, float(r_u), float(p_u)))
    caravan_adp.append((lbl, float(r_a), float(p_a)))

  far_cap_col = 'exp_weighted_far_p50direct_nosys115_pct' if 'exp_weighted_far_p50direct_nosys115_pct' in pf1.columns else 'exp_weighted_far_nosys115_pct'
  far_glb_col = 'exp_weighted_far_p50direct_global_pct' if 'exp_weighted_far_p50direct_global_pct' in pf1.columns else 'exp_weighted_far_global_pct'

  stage_defs = [
      ('Today', 'tier1a_active_post2016_ccby_8811', '000_global_train_joint'),
      ('32.4k', 'tier1_open_compilations', '111_all_three_combined_snap'),
      ('74.1k', 'tier2_public_discharge', '111_all_three_combined_snap'),
      ('119.2k', 'tier4_all_discoverable', '111_all_three_combined_snap'),
  ]
  if ('tier5_universal_no_ood_clamp', '111_all_three_combined_snap') in pf1.index:
    stage_defs.append(('Univ. (local)', 'tier5_universal_no_ood_clamp', '111_all_three_combined_snap'))
  stage_defs.append(('Univ. (upper)', 'tier5_universal_idealized_ceiling', '111_all_three_combined_snap'))

  cap_stages, glob_stages = [], []
  for lbl, tk, mech in stage_defs:
    row = pf1.loc[(tk, mech)]
    cap_stages.append((lbl, float(row['warned_recall_nosys115']) / e_cap * 100.0, 100.0 - float(row[far_cap_col])))
    glob_stages.append((lbl, float(row['warned_recall_global']) / e_glob * 100.0, 100.0 - float(row[far_glb_col])))

  fig = plt.figure(figsize=(SCIENCE_W, 3.85), dpi=SAVE_DPI)
  gs = gridspec.GridSpec(1, 2, width_ratios=[1.12, 1.00], wspace=0.30, left=0.075, right=0.985, top=0.91, bottom=0.14)

  # --- (a) Precision vs. Recall curves ---
  ax_a = fig.add_subplot(gs[0, 0])
  r_grid = np.linspace(15.0, 80.5, 350)
  for f1_val, lbl_pos in [(20.0, (79.0, 11.4)), (30.0, (79.0, 18.6)), (40.0, (79.0, 27.1)), (50.0, (79.0, 37.2))]:
    mask_r = (2.0 * r_grid - f1_val) > 2.0
    p_iso = (f1_val * r_grid[mask_r]) / (2.0 * r_grid[mask_r] - f1_val)
    valid = (p_iso >= 8.0) & (p_iso <= 56.0)
    ax_a.plot(r_grid[mask_r][valid], p_iso[valid], color='#CBD5E1', lw=0.75, ls=':', zorder=1)
    ax_a.text(lbl_pos[0], lbl_pos[1], f'F1={f1_val/100:.1f}', fontsize=6.0, style='italic', color='#64748B', ha='right', va='bottom', zorder=2)

  cu_x, cu_y = [pt[1] for pt in caravan_ung], [pt[2] for pt in caravan_ung]
  ax_a.plot(cu_x, cu_y, color='#64748B', lw=1.2, ls='--', zorder=3)
  ax_a.scatter(cu_x, cu_y, s=22, marker='D', facecolor='white', edgecolor='#334155', lw=0.9, zorder=5)
  for lbl, rx, py in caravan_ung:
    if lbl == '10y':
      ax_a.text(rx - 1.3, py + 1.4, lbl, fontsize=6.1, fontweight='bold', color='#475569', ha='right', va='bottom', zorder=7)
    else:
      dx = +1.4 if lbl == '1y' else -1.4
      ha = 'left' if dx > 0 else 'right'
      ax_a.text(rx + dx, py + (0.6 if dx < 0 else -0.6), lbl, fontsize=6.1, fontweight='bold', color='#475569', ha=ha, va='center', zorder=7)

  ca_x, ca_y = [pt[1] for pt in caravan_adp], [pt[2] for pt in caravan_adp]
  ax_a.plot(ca_x, ca_y, color='#0F172A', lw=1.3, ls='--', zorder=4)
  ax_a.scatter(ca_x, ca_y, s=24, marker='D', color='#0F172A', edgecolor='white', lw=0.5, zorder=6)
  for lbl, rx, py in caravan_adp:
    dx = -1.4 if lbl in ('1y', '2y') else +1.4
    ha = 'left' if dx > 0 else 'right'
    ax_a.text(rx + dx, py + 0.8, lbl, fontsize=6.1, fontweight='bold', color='#0F172A', ha=ha, va='bottom', zorder=7)

  gl_x, gl_y = [pt[1] for pt in glob_stages], [pt[2] for pt in glob_stages]
  ax_a.plot(gl_x, gl_y, color='#94A3B8', lw=1.1, ls='-', zorder=3)
  ax_a.scatter([gl_x[0], gl_x[3], gl_x[-1]], [gl_y[0], gl_y[3], gl_y[-1]], s=18, marker='o', facecolor='white', edgecolor='#64748B', lw=0.9, zorder=4)

  cp_x, cp_y = [pt[1] for pt in cap_stages], [pt[2] for pt in cap_stages]
  ax_a.plot(cp_x, cp_y, color='#B45309', lw=1.7, ls='-', zorder=5)
  ax_a.scatter(cp_x, cp_y, s=24, marker='o', color='#64748B', edgecolor='white', lw=0.45, zorder=6)
  ax_a.scatter([cp_x[0]], [cp_y[0]], s=38, color='#2563EB', edgecolor='white', lw=0.6, zorder=7)
  ax_a.scatter([cp_x[3]], [cp_y[3]], s=38, color='#0D9488', edgecolor='white', lw=0.6, zorder=7)
  ax_a.scatter([cp_x[-1]], [cp_y[-1]], s=42, color='#D97706', edgecolor='white', lw=0.6, zorder=7)

  ax_a.annotate('Today', xy=(cp_x[0], cp_y[0]), xytext=(cp_x[0] - 2.0, cp_y[0] - 5.2), ha='center', va='top', fontsize=6.1, fontweight='bold', color='#1D4ED8', arrowprops=dict(arrowstyle='-', color='#93C5FD', lw=0.7))
  ax_a.annotate('119.2k', xy=(cp_x[3], cp_y[3]), xytext=(cp_x[3] - 6.5, cp_y[3] + 4.8), ha='center', va='bottom', fontsize=6.1, fontweight='bold', color='#0F766E', arrowprops=dict(arrowstyle='-', color='#5EEAD4', lw=0.7))
  ax_a.annotate('Universal', xy=(cp_x[-1], cp_y[-1]), xytext=(cp_x[-1] + 2.0, cp_y[-1] - 7.8), ha='center', va='top', fontsize=6.1, fontweight='bold', color='#B45309', arrowprops=dict(arrowstyle='-', color='#FDBA74', lw=0.7))

  pr_handles = [
      Line2D([0], [0], color='#0F172A', lw=1.2, ls='--', marker='D', markersize=3.8, markerfacecolor='#0F172A', label='Adapted Caravan (1y–10y)'),
      Line2D([0], [0], color='#64748B', lw=1.1, ls='--', marker='D', markersize=3.8, markerfacecolor='white', markeredgecolor='#334155', label='Ungauged Caravan (1y–10y)'),
      Line2D([0], [0], color='#B45309', lw=1.6, ls='-', marker='o', markersize=4.2, markerfacecolor='#D97706', label='CAP-gap exposure (P50-direct)'),
      Line2D([0], [0], color='#94A3B8', lw=1.1, ls='-', marker='o', markersize=3.8, markerfacecolor='white', markeredgecolor='#64748B', label='Global exposure (P50-direct)'),
  ]
  ax_a.legend(handles=pr_handles, loc='upper left', frameon=True, facecolor='white', edgecolor='#CBD5E1', framealpha=0.95, fontsize=6.0, borderpad=0.28)
  ax_a.set_xlim(15.0, 81.0)
  ax_a.set_ylim(7.5, 57.5)
  ax_a.tick_params(labelsize=6.2)
  ax_a.set_xlabel('Flood recall (% real floods caught)', fontsize=6.8, fontweight='bold')
  ax_a.set_ylabel('Precision (% alerts true)', fontsize=6.8, fontweight='bold')
  ax_a.set_title('(a) Precision vs. recall across thresholds & stages', fontsize=7.2, fontweight='bold', loc='left', pad=5)
  ax_a.grid(True, linestyle='--', alpha=0.35)
  sns.despine(ax=ax_a)

  # --- (b) Sensitivity of CAP-gap share closed by 121.8k vs Universal across A1 regimes ---
  ax_b = fig.add_subplot(gs[0, 1])
  regimes = [
      ('α = 1.0\n(Recall)', 'warned_recall_nosys115'),
      ('α = CSI/R\n(FA-disc.)', 'warned_csi_nosys115'),
      ('α = P\n(Bayesian)', 'warned_bayesian_pr_nosys115'),
      ('INFORM\nCoping×Rec', 'warned_coping_recall_nosys115'),
      ('INFORM\nCoping×CSI', 'warned_coping_csi_nosys115'),
  ]
  regimes = [(lbl, col) for lbl, col in regimes if col in pf1.columns]
  yr = np.arange(len(regimes))[::-1]
  t_base = ('tier1a_active_post2016_ccby_8811', '000_global_train_joint')
  t_121 = ('tier4_all_discoverable', '111_all_three_combined_snap')
  t_nood = ('tier5_universal_no_ood_clamp', '111_all_three_combined_snap')
  t_ceil = ('tier5_universal_idealized_ceiling', '111_all_three_combined_snap')

  sh_share_ceil, sh_share_nood = [], []
  for _, col in regimes:
    v0 = float(pf1.loc[t_base, col])
    v121 = float(pf1.loc[t_121, col])
    vc = float(pf1.loc[t_ceil, col])
    vn = float(pf1.loc[t_nood, col]) if t_nood in pf1.index else vc
    sh_share_ceil.append(100.0 * (v121 - v0) / max(vc - v0, 1.0))
    sh_share_nood.append(100.0 * (v121 - v0) / max(vn - v0, 1.0))

  h = 0.34
  ax_b.barh(yr + h / 2, sh_share_ceil, height=h, color='#0D9488', edgecolor='white', lw=0.45, label='Share of upper-bound gap')
  ax_b.barh(yr - h / 2, sh_share_nood, height=h, color='#D97706', edgecolor='white', lw=0.45, label='Share of local-gain gap')
  for i in range(len(regimes)):
    ax_b.text(sh_share_ceil[i] + 1.0, yr[i] + h / 2, f'{sh_share_ceil[i]:.0f}%', va='center', ha='left', fontsize=6.0, fontweight='bold', color='#0F766E')
    ax_b.text(sh_share_nood[i] + 1.0, yr[i] - h / 2, f'{sh_share_nood[i]:.0f}%', va='center', ha='left', fontsize=6.0, fontweight='bold', color='#9A3412')

  ax_b.set_yticks(yr)
  ax_b.set_yticklabels([r[0] for r in regimes], fontsize=6.1, fontweight='bold')
  ax_b.set_xlim(0, max(58.0, max(sh_share_ceil + sh_share_nood) + 14.0))
  ax_b.set_ylim(-0.60, len(regimes) - 0.15)
  ax_b.tick_params(axis='x', labelsize=6.2)
  ax_b.set_xlabel('CAP-gap deficit closed by 119.2k existing gauges (%)', fontsize=6.6, fontweight='bold')
  ax_b.set_title('(b) Directional invariance across action regimes', fontsize=7.2, fontweight='bold', loc='left', pad=5)
  ax_b.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=6.0, borderpad=0.25)
  ax_b.grid(axis='x', linestyle='--', alpha=0.35)
  sns.despine(ax=ax_b)

  out_path = out_dir / 'Fig_S7_precision_recall_curves.png'
  fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.05)
  plt.close(fig)
  print(f'  Saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)', flush=True)
  return out_path


# ==============================================================================
# FIGURE S8 (OPTIONAL — Fathom v3): National N = 1..20 Strategic Gauge Siting
# ==============================================================================
def build_fig_s8_optional(out_dir: Path) -> Path | None:
  s16_csv = S16_DIR / 'country_joint_n_gauge_r2.csv'
  if not s16_csv.exists():
    print(f'[Fig S8] SKIPPED: Licensed Fathom v3 national gauge-siting output not found at {s16_csv}.', flush=True)
    return None

  print('[Fig S8] Rendering national N = 1..20 strategic gauge siting figure (fixed layout)...', flush=True)
  _set_pub_rc(6.3)
  world = gpd.read_file(file_paths.NATURAL_EARTH_PATH)
  world = world[world['name'] != 'Antarctica'].copy()
  world['map_name'] = world['name'].replace(file_paths.country_name_map)
  df_nat = pd.read_csv(s16_csv)
  nat_idx = df_nat.set_index('Country')
  n_gauges = [1, 5, 10, 20]
  vmax_sh = max(2.0, float(np.nanpercentile(df_nat['N=20_Combined_Dom_PctBase'], 95)))

  fig = plt.figure(figsize=(SCIENCE_W, 6.10), dpi=SAVE_DPI, facecolor='#ffffff')
  gs_maps = gridspec.GridSpec(
      2, 2, wspace=0.06, hspace=0.14,
      left=0.040, right=0.985, top=0.930, bottom=0.495,
  )
  gs_bot = gridspec.GridSpec(
      1, 2, wspace=0.24,
      left=0.085, right=0.985, top=0.385, bottom=0.180,
  )
  axes_maps = [fig.add_subplot(gs_maps[0, 0]), fig.add_subplot(gs_maps[0, 1]), fig.add_subplot(gs_maps[1, 0]), fig.add_subplot(gs_maps[1, 1])]
  ltrs = ['(a)', '(b)', '(c)', '(d)']
  for i, N in enumerate(n_gauges):
    ax = axes_maps[i]
    w_p = world.copy()
    col_pct = f'N={N}_Combined_Dom_PctBase'
    w_p['val'] = w_p['map_name'].map(nat_idx[col_pct])
    world.plot(ax=ax, facecolor='#ECEFF1', edgecolor='#B0BEC5', linewidth=0.20, aspect=None)
    w_p.dropna(subset=['val']).plot(
        column='val', cmap='viridis', ax=ax, vmin=0.0, vmax=vmax_sh,
        legend=False,
        edgecolor='#37474F', linewidth=0.15, aspect=None,
    )
    grl = world[world['map_name'] == 'Greenland']
    if len(grl) > 0:
      grl.plot(ax=ax, facecolor='#E2E8F0', edgecolor='#94A3B8', linewidth=0.20, zorder=3, aspect=None)
    med_c = float(df_nat[col_pct].median())
    tot_c = float(df_nat[f'N={N}_Combined_Dom'].sum()) / 1e6
    ax.set_title(f'{ltrs[i]} N = {N} ({tot_c:.2f}M/yr; med {med_c:.1f}%)', fontsize=6.8, fontweight='bold', loc='left', pad=2.5)
    ax.set_xlim([-180, 180])
    ax.set_ylim([-58, 84])
    ax.set_aspect(1.0)
    ax.set_axis_off()

  cax = fig.add_axes([0.27, 0.455, 0.48, 0.015])
  sm = matplotlib.cm.ScalarMappable(cmap='viridis', norm=matplotlib.colors.Normalize(vmin=0.0, vmax=vmax_sh))
  sm.set_array([])
  cb = fig.colorbar(sm, cax=cax, orientation='horizontal', extend='max')
  cb.ax.tick_params(labelsize=6.1, length=2.5, pad=1.5)
  cb.set_label(f'Annual flood exposure reduced (%, capped at {vmax_sh:.0f}%)', fontsize=6.4, fontweight='bold', labelpad=2.0)

  # Bottom row (e) & (f): Top 15 CAP-gap countries by baseline flood exposure (>= 10 cm)
  ax_e = fig.add_subplot(gs_bot[0, 0])
  ax_f = fig.add_subplot(gs_bot[0, 1])
  top15_df = df_nat[~df_nat['has_system']].sort_values('Baseline Exposure (people/yr)', ascending=False).head(15)
  top15_names = top15_df['Country'].tolist()
  cmap_15 = sns.color_palette('husl', len(top15_names))
  ls_cyc = ['-', '--', '-.']
  mk_cyc = ['o', 's', '^']
  handles_15 = []
  for i, cname in enumerate(top15_names):
    r = nat_idx.loc[cname]
    mc = int(r['Max_Evaluated_Candidates'])
    y_abs, y_pct = [], []
    for N in n_gauges:
      if N > mc and N > 1:
        y_abs.append(np.nan); y_pct.append(np.nan)
      else:
        y_abs.append(float(r[f'N={N}_Combined_Dom']))
        y_pct.append(float(r[f'N={N}_Combined_Dom_PctBase']))
    short_c = 'DR Congo' if cname == 'Congo DR' else ('Vietnam' if cname == 'Viet Nam' else cname)
    lbl = f"{short_c} ({r['Baseline Exposure (people/yr)']/1e6:.2f}M)"
    ln, = ax_e.plot(n_gauges, y_abs, marker=mk_cyc[i // 5], ls=ls_cyc[i % 3], color=cmap_15[i], lw=1.25, markersize=3.5, label=lbl)
    ax_f.plot(n_gauges, y_pct, marker=mk_cyc[i // 5], ls=ls_cyc[i % 3], color=cmap_15[i], lw=1.25, markersize=3.5)
    handles_15.append(ln)

  ax_e.set_xticks(n_gauges)
  ax_e.set_yscale('symlog', linthresh=100.0)
  ax_e.set_ylim(bottom=0.0)
  ax_e.tick_params(labelsize=6.1)
  ax_e.set_xlabel('New active gauges installed per country (N)', fontsize=6.5, fontweight='bold')
  ax_e.set_ylabel('Combined gain (persons/yr)', fontsize=6.5, fontweight='bold')
  ax_e.set_title('(e) Top 15 CAP-gap countries (absolute, ≥ 10 cm)', fontsize=6.9, fontweight='bold', loc='left', pad=4)
  ax_e.grid(True, which='both', ls='--', alpha=0.35)
  sns.despine(ax=ax_e)

  ax_f.set_xticks(n_gauges)
  ax_f.set_ylim(bottom=0.0)
  ax_f.tick_params(labelsize=6.1)
  ax_f.set_xlabel('New active gauges installed per country (N)', fontsize=6.5, fontweight='bold')
  ax_f.set_ylabel('% annual exposure reduced', fontsize=6.5, fontweight='bold')
  ax_f.set_title('(f) Top 15 CAP-gap countries (% reduced)', fontsize=6.9, fontweight='bold', loc='left', pad=4)
  ax_f.grid(True, which='both', ls='--', alpha=0.35)
  sns.despine(ax=ax_f)

  # Clean multi-column legend strictly BELOW panels (e)-(f) — zero overlap with panel (f) title!
  fig.legend(
      handles=handles_15, loc='lower center', bbox_to_anchor=(0.53, 0.005), ncol=5,
      frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=6.0,
      columnspacing=0.75, handlelength=1.4, handletextpad=0.30, borderpad=0.30,
      title='Top 15 CAP-gap countries (total annual flood exposure ≥ 10 cm)', title_fontsize=6.2,
  )
  fig.suptitle('National exposure reduction from installing N = 1 to 20 active stream gauges per country', fontsize=7.6, fontweight='bold', y=0.975)

  out_s8 = out_dir / 'Fig_S8_national_gauge_siting.png'
  fig.savefig(out_s8, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.05)
  plt.close(fig)
  print(f'  Saved {out_s8} ({out_s8.stat().st_size / 1e6:.2f} MB)', flush=True)
  return out_s8


def main() -> None:
  parser = argparse.ArgumentParser(description='Generate citation-ordered supplementary figures (Figs. S1–S6; optional S7–S8).')
  parser.add_argument('--only', type=str, default='all', help='Comma-separated subset: figs1,figs2,figs3,figs4,figs5,figs6,figs7,figs8')
  parser.add_argument('--out-dir', type=str, default=str(DEFAULT_OUT_DIR))
  args = parser.parse_args()

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  targets = (
      {'figs1', 'figs2', 'figs3', 'figs4', 'figs5', 'figs6', 'figs7', 'figs8'}
      if args.only.strip().lower() == 'all'
      else {t.strip().lower() for t in args.only.split(',')}
  )

  if 'figs1' in targets:
    build_fig_s1(out_dir)
  if 'figs2' in targets:
    copy_fig_s2(out_dir)
  if 'figs3' in targets:
    build_fig_s3(out_dir)
  if targets & {'figs4', 'figs5'}:
    build_fig_s4_and_s5(out_dir, build_s4=('figs4' in targets), build_s5=('figs5' in targets))
  if 'figs6' in targets:
    build_fig_s6(out_dir)
  if 'figs7' in targets:
    build_fig_s7_optional(out_dir)
  if 'figs8' in targets:
    build_fig_s8_optional(out_dir)

  print('Supplementary figure generation complete!', flush=True)


if __name__ == '__main__':
  main()
