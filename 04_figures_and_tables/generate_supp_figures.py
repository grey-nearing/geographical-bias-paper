#!/usr/bin/env python3
"""Generate supplementary publication figures (Figs. S1–S5 non-Fathom; optional S6–S8 if Fathom present).

Produces:
  - Fig_S1_delta_nse_network_vs_local.png: Harmonized 2016–2023 ΔNSE across coverage bins
    (plus writes sec3_cv_diagnostics_r2.json).
  - Fig_S2_top35_permutation_shap.png: Top-35 unique predictors 200-path antithetic
    permutation SHAP heatmap across 12 metrics.
  - Fig_S3_conceptual_ungaugedness.png: Copies pre-rendered Latvia conceptual schematic
    from results/figures/Fig_S3_conceptual_ungaugedness.png into output directory.
  - Fig_S4_ood_sextiles_recall.png: 3x3 OOD sextile boxplots of 5-year 2-day causal recall.
  - Fig_S5_global_predicted_skill.png: Global predicted skill distributions across
    1,034,083 watersheds (from Table_S3_4regimes_r2.csv / Table_S1_4regimes_r2.csv).
  - Fig_S6_exposure_and_warnings.png, Fig_S7_data_bias_gap.png,
    Fig_S8_national_gauge_siting.png (OPTIONAL — skipped cleanly when licensed Fathom v3
    exposure counterfactual / S1.6 files are absent).
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
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
import seaborn as sns

from utils import file_paths

warnings.filterwarnings('ignore')

NATURE_W = 183.0 / 25.4
SAVE_DPI = 350

R2_DIR = CANON_DIR / 'round2'
METRICS_R2_DIR = R2_DIR / 'hydro_model_metrics'
CF_DIR = R2_DIR / 'counterfactuals'
S16_DIR = R2_DIR / 's16_gauge_placement'
SHAP_DIR = R2_DIR / 'shap'
DEFAULT_OUT_DIR = R2_DIR / 'figures'

ALL_12_METRICS = [
    'NSE', 'KGE',
    '1.01-year-2.0-window-precision', '2.0-year-2.0-window-precision',
    '5.0-year-2.0-window-precision', '10.0-year-2.0-window-precision',
    '20.0-year-2.0-window-precision',
    '1.01-year-2.0-window-recall', '2.0-year-2.0-window-recall',
    '5.0-year-2.0-window-recall', '10.0-year-2.0-window-recall',
    '20.0-year-2.0-window-recall',
]


def _set_pub_rc(font_size: float = 7.4) -> None:
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
      'figure.facecolor': 'white',
      'axes.facecolor': 'white',
      'savefig.facecolor': 'white',
  })


def load_r2_metric_series(exp_name: str, metric: str, rename_to: str | None = None) -> pd.Series:
  p_r2 = METRICS_R2_DIR / f'{exp_name}__{metric}.csv'
  if not p_r2.exists():
    p_r2 = METRICS_R2_DIR / exp_name / f'{metric}.csv'
  if not p_r2.exists():
    raise FileNotFoundError(f'Missing Round-2 metric CSV for {exp_name}/{metric} under {METRICS_R2_DIR}')
  df = pd.read_csv(p_r2, index_col=0)
  col = '1' if '1' in df.columns else ('lead_1' if 'lead_1' in df.columns else df.columns[0])
  s = pd.to_numeric(df[col], errors='coerce')
  s.index = s.index.astype(str)
  s.name = rename_to or metric
  return s


def _fmt_sci_tex(p: float) -> str:
  if p <= 0 or np.isnan(p):
    return '< 10^{-300}'
  if p >= 0.001:
    return f'{p:.3f}'
  exp = int(np.floor(np.log10(p)))
  mant = p / (10 ** exp)
  return f'{mant:.1f} \\times 10^{{{exp}}}'


# ==============================================================================
# FIGURE S1 & FIGURE S4: ΔNSE (2016–2023) & OOD Sextiles (5-yr 2-day Recall)
# ==============================================================================
def build_fig_s1_and_s4(out_dir: Path, build_s1: bool = True, build_s4: bool = True) -> None:
  print('[Fig S1 / S4] Computing CV diagnostics and rendering Fig. S1 & Fig. S4...', flush=True)
  _set_pub_rc(7.4)
  file_paths.init('NSE')

  cv_json: dict = {}

  if build_s1:
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

    fig_s1, axes = plt.subplots(2, 1, figsize=(NATURE_W, NATURE_W * (295.69 / 468.0)), sharex=True, dpi=SAVE_DPI)
    sns.boxplot(data=melted, x='Coverage_Bin', y='delta_sep_rand', hue='Coverage Metric', palette=['#0072B2', '#E69F00', '#009E73'], showfliers=False, whis=[10, 90], linewidth=0.55, ax=axes[0])
    med_rs = float(df_sub['delta_sep_rand'].median())
    axes[0].axhline(med_rs, color='#882255', ls='--', lw=1.0, label=f'Overall median ΔNSE ({med_rs:+.3f})')
    axes[0].axhline(0.0, color='gray', lw=0.65, zorder=0)
    axes[0].set_ylabel('ΔNSE (Random − Separated)', fontsize=7.0, fontweight='bold')
    axes[0].set_title(f'(a) River-network information transfer in ungauged CV (2016–2023; N = {len(df_sub):,})', fontsize=7.8, fontweight='bold', loc='left', pad=3)
    axes[0].set_ylim(-0.18, 0.50)
    axes[0].legend(loc='upper right', fontsize=5.8, frameon=True, framealpha=0.96, ncol=2)
    axes[0].set_xlabel('')
    sns.despine(ax=axes[0])

    sns.boxplot(data=melted, x='Coverage_Bin', y='delta_gauged_rand', hue='Coverage Metric', palette=['#0072B2', '#E69F00', '#009E73'], showfliers=False, whis=[10, 90], linewidth=0.55, ax=axes[1])
    med_gr = float(df_sub['delta_gauged_rand'].median())
    axes[1].axhline(med_gr, color='#882255', ls='--', lw=1.0, label=f'Overall median ΔNSE ({med_gr:+.3f})')
    axes[1].axhline(0.0, color='gray', lw=0.65, zorder=0)
    axes[1].set_ylabel('ΔNSE (Gauged − Random)', fontsize=7.0, fontweight='bold')
    axes[1].set_title(f'(b) Local at-site gauge calibration gain (2016–2023; N = {len(df_sub):,})', fontsize=7.8, fontweight='bold', loc='left', pad=3)
    axes[1].set_ylim(-0.12, 0.215)
    axes[1].legend(loc='upper right', fontsize=5.8, frameon=True, framealpha=0.96, ncol=2)
    axes[1].set_xlabel('Drainage area coverage fraction bin', fontsize=7.0, fontweight='bold')
    sns.despine(ax=axes[1])

    fig_s1.tight_layout()
    out_s1 = out_dir / 'Fig_S1_delta_nse_network_vs_local.png'
    fig_s1.savefig(out_s1, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig_s1)
    print(f'  Saved {out_s1} ({out_s1.stat().st_size / 1e6:.2f} MB)', flush=True)

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

  if build_s4:
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
    sextile_lbls = ['Q1\n(OOD)', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6\n(In-dist)']

    fig_s4, axes_s4 = plt.subplots(3, 3, figsize=(NATURE_W, NATURE_W * (375.99 / 468.0)), sharey=True, dpi=SAVE_DPI)
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
        ax = axes_s4[row_i, i]
        sns.boxplot(data=df_u, x=f'{acol}_bin', y=ycol, hue=f'{acol}_bin', palette=pal, dodge=False, width=0.58, legend=False, showfliers=False, whis=[10, 90], linewidth=0.55, ax=ax)
        ax.axhline(df_u[ycol].median(), color=hcol, ls='--', lw=0.9)
        ax.set_xlabel(f'{atitle} sextile' if row_i == 2 else '', fontsize=6.6, fontweight='bold')
        ax.set_title(ttl, fontsize=7.2, fontweight='bold', loc='left', pad=2)
        ax.set_ylabel(ylab if i == 0 else '', fontsize=6.8, fontweight='bold')
        ax.tick_params(axis='x', labelsize=5.8)
        sns.despine(ax=ax)

    fig_s4.tight_layout()
    out_s4 = out_dir / 'Fig_S4_ood_sextiles_recall.png'
    fig_s4.savefig(out_s4, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig_s4)
    print(f'  Saved {out_s4} ({out_s4.stat().st_size / 1e6:.2f} MB)', flush=True)
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
# FIGURE S2: Top-35 Individual Features Permutation SHAP Heatmap
# ==============================================================================
def build_fig_s2(out_dir: Path) -> Path:
  print('[Fig S2] Rendering Top-35 permutation SHAP heatmap...', flush=True)
  _set_pub_rc(7.0)

  short_cat = {
      'Climate & Hydrology': 'Climate/Hydrol.',
      'Land Cover': 'Land Cover',
      'Climatic Similarity (OOD)': 'Clim. Sim. (OOD)',
      'Topography': 'Topography',
      'Soil & Geology': 'Soil & Geol.',
      'Anthropogenic': 'Anthropogenic',
      'River-Network Gauge Cov.': 'Gauge Cov.',
  }
  metric_labels = {
      'NSE': 'NSE', 'KGE': 'KGE',
      '1.01-year-2.0-window-precision': 'Prec. (1y)', '2.0-year-2.0-window-precision': 'Prec. (2y)',
      '5.0-year-2.0-window-precision': 'Prec. (5y)', '10.0-year-2.0-window-precision': 'Prec. (10y)',
      '20.0-year-2.0-window-precision': 'Prec. (20y)',
      '1.01-year-2.0-window-recall': 'Recall (1y)', '2.0-year-2.0-window-recall': 'Recall (2y)',
      '5.0-year-2.0-window-recall': 'Recall (5y)', '10.0-year-2.0-window-recall': 'Recall (10y)',
      '20.0-year-2.0-window-recall': 'Recall (20y)',
  }

  df_t35 = pd.read_csv(SHAP_DIR / 'shap_top35_features_r2.csv', index_col=0)
  mat_raw = df_t35[ALL_12_METRICS].copy()
  mat_norm = mat_raw.div(mat_raw.max(axis=0), axis=1) * 100.0
  short_cols = [metric_labels[m] for m in ALL_12_METRICS]
  mat_norm.columns = short_cols
  annot_txt = pd.DataFrame(
      np.vectorize(lambda v: f'{v:.3f}' if v >= 0.01 else f'{v:.4f}')(mat_raw.values),
      index=mat_raw.index,
      columns=short_cols,
  )

  fig, ax = plt.subplots(figsize=(NATURE_W, NATURE_W * (417.0 / 468.0)), dpi=SAVE_DPI)
  sns.heatmap(
      mat_norm, ax=ax, cmap='viridis', vmin=0.0, vmax=100.0,
      annot=annot_txt, fmt='', annot_kws={'size': 4.4},
      linewidths=0.30,
      cbar_kws={'label': 'Relative Importance (% of per-metric maximum Mean |SHAP|)', 'shrink': 0.72, 'pad': 0.02},
  )
  y_lbls = [f"{r['label']} ({short_cat.get(r['category'], r['category'])})" for _, r in df_t35.iterrows()]
  ax.set_yticks(np.arange(len(df_t35)) + 0.5)
  ax.set_yticklabels(y_lbls, rotation=0, fontsize=5.1)
  ax.set_xticklabels(short_cols, rotation=32, ha='right', fontsize=5.8, fontweight='bold')
  ax.set_title('Global Permutation SHAP Feature Importance (Top 35 Unique Predictors Across 12 Metrics)', fontsize=7.4, fontweight='bold', pad=7)
  ax.set_xlabel('Hydrological Evaluation Metric (cells show raw Mean |SHAP|)', fontsize=6.6, fontweight='bold')
  ax.set_ylabel('Watershed Predictor [code] (Category)', fontsize=6.6, fontweight='bold')
  fig.tight_layout()

  out_path = out_dir / 'Fig_S2_top35_permutation_shap.png'
  fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.03)
  plt.close(fig)
  print(f'  Saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)', flush=True)
  return out_path


# ==============================================================================
# FIGURE S3: Conceptual Schematic (Copy archived reference PNG)
# ==============================================================================
def copy_fig_s3(out_dir: Path) -> Path:
  src = REPO_DIR / 'results' / 'figures' / 'Fig_S3_conceptual_ungaugedness.png'
  dst = out_dir / 'Fig_S3_conceptual_ungaugedness.png'
  if src.exists():
    if src.resolve() != dst.resolve():
      shutil.copy2(src, dst)
    print(f'  Verified/copied {dst} ({dst.stat().st_size / 1e6:.2f} MB)', flush=True)
  else:
    print(f'  [Fig S3] WARNING: Schematic not found at {src}', flush=True)
  return dst


# ==============================================================================
# FIGURE S5: Global Predicted Skill Boxplots across 1,034,083 Watersheds
# ==============================================================================
def build_fig_s5(out_dir: Path) -> Path:
  print('[Fig S5] Rendering global predicted skill boxplots (1,034,083 watersheds)...', flush=True)
  _set_pub_rc(7.3)
  p_ts = CF_DIR / 'Table_S3_4regimes_r2.csv'
  if not p_ts.exists():
    p_ts = CF_DIR / 'Table_S1_4regimes_r2.csv'
  df_ts = pd.read_csv(p_ts)
  df_base = df_ts[df_ts['regime'] == 'base'].set_index('metric')

  metrics_f5 = [
      ('NSE', 'NSE'), ('KGE', 'KGE'),
      ('1.01-year-2.0-window-recall', 'Recall (1y)'), ('2.0-year-2.0-window-recall', 'Recall (2y)'),
      ('5.0-year-2.0-window-recall', 'Recall (5y)'), ('10.0-year-2.0-window-recall', 'Recall (10y)'),
      ('20.0-year-2.0-window-recall', 'Recall (20y)'),
      ('1.01-year-2.0-window-precision', 'Precision (1y)'), ('2.0-year-2.0-window-precision', 'Precision (2y)'),
      ('5.0-year-2.0-window-precision', 'Precision (5y)'), ('10.0-year-2.0-window-precision', 'Precision (10y)'),
      ('20.0-year-2.0-window-precision', 'Precision (20y)'),
  ]
  box_stats = []
  for m_id, m_lbl in metrics_f5:
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

  fig, ax = plt.subplots(figsize=(NATURE_W, NATURE_W * (245.12 / 468.0)), dpi=SAVE_DPI)
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
  xmax = max(0.70, max(b['whishi'] for b in box_stats) + 0.05)
  ax.set_xlim([xmin, xmax])
  ax.set_xlabel('Predicted hydrological skill across 1,034,083 global watersheds', fontsize=7.4, fontweight='bold')
  ax.set_title('Global watershed forecast skill distributions (post-2016 Caravan training set)', fontsize=8.0, fontweight='bold', loc='left', pad=4)
  handles = [
      Patch(facecolor='#0072B2', edgecolor='black', alpha=0.84, label='Continuous (NSE, KGE; 2016–2023)'),
      Patch(facecolor='#009E73', edgecolor='black', alpha=0.84, label='Flood recall (7-day causal)'),
      Patch(facecolor='#CC79A7', edgecolor='black', alpha=0.84, label='Flood precision (7-day causal)'),
  ]
  ax.legend(handles=handles, loc='lower left', frameon=True, framealpha=0.95, fontsize=5.6, handlelength=1.1, borderpad=0.25, labelspacing=0.25)
  ax.grid(axis='x', linestyle='--', alpha=0.45)
  sns.despine(ax=ax)
  fig.tight_layout()

  out_path = out_dir / 'Fig_S5_global_predicted_skill.png'
  fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.03)
  plt.close(fig)
  print(f'  Saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)', flush=True)
  return out_path


# ==============================================================================
# OPTIONAL FATHOM SUPPLEMENTARY FIGURES (Figs. S6, S7, S8)
# ==============================================================================
def build_optional_fathom_supp_figs(out_dir: Path, targets: set[str]) -> None:
  basin_pq = CF_DIR / 'basin_level_exposure_counterfactuals_r2.parquet'
  t3_csv = CF_DIR / 'Table_S3_175territories_r2.csv'
  s16_csv = S16_DIR / 'country_joint_n_gauge_r2.csv'

  if targets & {'figs6', 'figs7'}:
    if not (basin_pq.exists() and t3_csv.exists()):
      print(
          '[Fig S6 / S7] SKIPPED: Licensed Fathom v3 flood-exposure counterfactuals '
          f'not present under {CF_DIR}.',
          flush=True,
      )
    else:
      world = gpd.read_file(file_paths.NATURAL_EARTH_PATH)
      world = world[world['name'] != 'Antarctica'].copy()
      world = world.dropna(subset=[world.geometry.name])
      world = world[world.is_valid & (~world.is_empty)].copy()
      world['map_name'] = world['name'].replace(file_paths.country_name_map)
      df_b = pd.read_parquet(basin_pq)
      df_c = pd.read_csv(t3_csv)

      if 'figs6' in targets:
        print('[Fig S6] Generating optional exposure & alerting coverage figure...', flush=True)
        _set_pub_rc(7.3)
        df_c_exp = df_c[(df_c['country_name'] != 'Greenland') & (df_c['E_total'] > 0)].copy()
        glb_tot = float(df_b['E_total'].sum())
        glb_warn = float(df_b['warned_t1a'].sum())
        rho_ze, p_ze = spearmanr(df_c_exp['zero_up_pct_t4'], df_c_exp['E_total'])

        fig = plt.figure(figsize=(9.2, 6.0), dpi=SAVE_DPI)
        gs = fig.add_gridspec(2, 2, wspace=0.20, hspace=0.34, left=0.09, right=0.98, top=0.93, bottom=0.09)
        ax_a = fig.add_subplot(gs[0, 0])
        world['sys_grp'] = world['map_name'].map(dict(zip(df_c['country_name'], df_c['has_system'])))
        world[world['sys_grp'].isna()].plot(ax=ax_a, color='#E2E8F0', edgecolor='#CBD5E1', linewidth=0.25)
        world[world['sys_grp'] == True].plot(ax=ax_a, color='#2171B5', edgecolor='#0F172A', linewidth=0.25, alpha=0.92)
        world[world['sys_grp'] == False].plot(ax=ax_a, color='#E67E22', edgecolor='#0F172A', linewidth=0.25, alpha=0.92)
        ax_a.set_xlim(-175, 180)
        ax_a.set_ylim(-58, 84)
        ax_a.set_axis_off()
        ax_a.set_title('(a) National flood-alerting coverage', fontsize=9.5, fontweight='bold', loc='left', pad=6)

        pts = df_b.dropna(subset=['lat', 'lon'])
        b_zero = pts[pts['E_total'] <= 0]
        b_pos = pts[pts['E_total'] > 0].sort_values('E_total').copy()
        ax_b = fig.add_subplot(gs[0, 1])
        world.plot(ax=ax_b, color='#EAECEE', edgecolor='#BFC9CA', linewidth=0.22)
        ax_b.scatter(b_zero['lon'], b_zero['lat'], s=0.12, c='#E5E7EB', alpha=0.55, rasterized=True)
        sc_b = ax_b.scatter(b_pos['lon'], b_pos['lat'], s=0.28, c=b_pos['E_total'].values, cmap='YlGnBu', norm=mcolors.LogNorm(vmin=0.1, vmax=2000.0), alpha=0.90, edgecolors='none', rasterized=True)
        ax_b.set_xlim(-175, 180)
        ax_b.set_ylim(-58, 84)
        ax_b.set_axis_off()
        ax_b.set_title('(b) Expected annual flood exposure (defended)', fontsize=9.5, fontweight='bold', loc='left', pad=6)
        cb_b = fig.colorbar(sc_b, ax=ax_b, orientation='horizontal', fraction=0.055, pad=0.04, shrink=0.70, extend='both')
        cb_b.set_label(f'Flood exposure >10 cm (persons/yr; total {glb_tot/1e6:.2f}M/yr)', fontsize=7.4, fontweight='bold')

        ax_c = fig.add_subplot(gs[1, 0])
        for hs, col, mk, lbl in [(True, '#2171B5', 'o', 'Established system'), (False, '#E67E22', '^', 'CAP-gap territory')]:
          s = df_c_exp[df_c_exp['has_system'] == hs]
          ax_c.scatter(s['zero_up_pct_t4'], s['E_total'], s=20, c=col, marker=mk, edgecolor='#0F172A', linewidth=0.45, alpha=0.88, label=lbl)
        ax_c.set_yscale('log')
        ax_c.text(0.03, 0.05, f'Spearman $r_s = {rho_ze:+.2f}$ ($p = {_fmt_sci_tex(p_ze)}$, N = {len(df_c_exp)})', transform=ax_c.transAxes, fontsize=7.0, bbox=dict(boxstyle='round,pad=0.24', facecolor='#F8FAFC', edgecolor='#CBD5E1'))
        ax_c.set_xlabel('Watersheds with zero upstream gauges (%)', fontsize=8.0, fontweight='bold')
        ax_c.set_ylabel('Flood exposure (persons/yr)', fontsize=8.0, fontweight='bold')
        ax_c.set_title('(c) Zero upstream gauges vs. flood exposure', fontsize=9.5, fontweight='bold', loc='left', pad=6)
        ax_c.legend(loc='upper left', fontsize=6.8)
        sns.despine(ax=ax_c)

        ax_d = fig.add_subplot(gs[1, 1])
        world.plot(ax=ax_d, color='#EAECEE', edgecolor='#BFC9CA', linewidth=0.22)
        ax_d.scatter(b_zero['lon'], b_zero['lat'], s=0.12, c='#E5E7EB', alpha=0.55, rasterized=True)
        b_pos['pct_fc'] = (b_pos['warned_t1a'] / b_pos['E_total']) * 100.0
        sc_d = ax_d.scatter(b_pos['lon'], b_pos['lat'], s=0.28, c=b_pos['pct_fc'].values, cmap='viridis', vmin=5, vmax=55, alpha=0.90, edgecolors='none', rasterized=True)
        ax_d.set_xlim(-175, 180)
        ax_d.set_ylim(-58, 84)
        ax_d.set_axis_off()
        ax_d.set_title('(d) Flood exposure forecasted by global model', fontsize=9.5, fontweight='bold', loc='left', pad=6)
        cb_d = fig.colorbar(sc_d, ax=ax_d, orientation='horizontal', fraction=0.055, pad=0.04, shrink=0.70, extend='both')
        cb_d.set_label(f'Exposure forecasted at α = 1.0 (%; global {100*glb_warn/glb_tot:.1f}%, {glb_warn/1e6:.2f}M/yr)', fontsize=7.4, fontweight='bold')

        out_s6 = out_dir / 'Fig_S6_exposure_and_warnings.png'
        fig.savefig(out_s6, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        print(f'  Saved {out_s6}', flush=True)

      if 'figs7' in targets:
        print('[Fig S7] Generating optional data-bias gap figure...', flush=True)
        _set_pub_rc(7.3)
        exp_c = df_c[(df_c['country_name'] != 'Greenland') & (df_c['E_total'] > 0)].copy()
        exp_c['pct_train_closed'] = (exp_c['dnet__tier5_universal_idealized_ceiling__000_global_train_joint'] / exp_c['unwarned_t1a']) * 100.0
        exp_c['comb_gain'] = exp_c['dnet__tier5_universal_idealized_ceiling__111_all_three_combined_snap']
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.4, 4.4), dpi=SAVE_DPI, gridspec_kw={'wspace': 0.25})
        for hs, col, mk, lbl in [(True, '#2171B5', 'o', 'Established system'), (False, '#E67E22', '^', 'CAP-gap territory')]:
          s = exp_c[exp_c['has_system'] == hs]
          sz = np.clip(np.sqrt(s['E_total'].values / 1e4) * 5.0 + 10.0, 12, 160)
          ax_a.scatter(s['zero_up_pct_t4'], s['pct_train_closed'], s=sz, c=col, marker=mk, edgecolor='#0F172A', linewidth=0.45, alpha=0.82, label=lbl)
        rs_a, pv_a = spearmanr(exp_c['zero_up_pct_t4'], exp_c['pct_train_closed'])
        ax_a.text(0.03, 0.94, f'$r_s = {rs_a:+.2f}$ ($p = {_fmt_sci_tex(pv_a)}$)', transform=ax_a.transAxes, ha='left', va='top', fontsize=7.6)
        ax_a.set_xlabel('Watersheds with zero upstream gauges (%)')
        ax_a.set_ylabel('Unforecasted exposure closed by training (%)')
        ax_a.set_title('(a) Relative data-bias gap vs. ungauged network', loc='left')
        ax_a.legend(loc='lower right', fontsize=7.0)
        sns.despine(ax=ax_a)

        pos_b = exp_c[exp_c['comb_gain'] > 0]
        for hs, col, mk, lbl in [(True, '#2171B5', 'o', 'Established system'), (False, '#E67E22', '^', 'CAP-gap territory')]:
          s = pos_b[pos_b['has_system'] == hs]
          ax_b.scatter(s['E_total'], s['comb_gain'], s=22, c=col, marker=mk, edgecolor='#0F172A', linewidth=0.45, alpha=0.85, label=lbl)
        rs_b, pv_b = spearmanr(exp_c['E_total'], exp_c['comb_gain'])
        ax_b.text(0.03, 0.94, f'$r_s = {rs_b:+.2f}$ ($p = {_fmt_sci_tex(pv_b)}$)', transform=ax_b.transAxes, ha='left', va='top', fontsize=7.6)
        ax_b.set_xscale('log')
        ax_b.set_yscale('log')
        ax_b.set_xlabel('Expected annual flood exposure (persons/yr)')
        ax_b.set_ylabel('Combined exposure gain (persons/yr)')
        ax_b.set_title('(b) Combined universal gain vs. annual exposure', loc='left')
        ax_b.legend(loc='lower right', fontsize=7.0)
        sns.despine(ax=ax_b)

        out_s7 = out_dir / 'Fig_S7_data_bias_gap.png'
        fig.savefig(out_s7, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        print(f'  Saved {out_s7}', flush=True)

  if 'figs8' in targets:
    if not s16_csv.exists():
      print(
          '[Fig S8] SKIPPED: Licensed Fathom v3 national gauge-siting output '
          f'not present at {s16_csv}.',
          flush=True,
      )
    else:
      print('[Fig S8] Generating optional national N=1..20 gauge siting figure...', flush=True)
      _set_pub_rc(6.8)
      world = gpd.read_file(file_paths.NATURAL_EARTH_PATH)
      world = world[world['name'] != 'Antarctica'].copy()
      world['map_name'] = world['name'].replace(file_paths.country_name_map)
      df_nat = pd.read_csv(s16_csv)
      nat_idx = df_nat.set_index('Country')
      n_gauges = [1, 5, 10, 20]
      vmax_sh = max(2.0, float(np.nanpercentile(df_nat['N=20_Combined_Dom_PctBase'], 95)))
      fig = plt.figure(figsize=(NATURE_W * 1.45, NATURE_W * 1.45 * (221.0 / 468.0)), dpi=SAVE_DPI)
      gs = gridspec.GridSpec(2, 3, width_ratios=[1.10, 1.10, 0.96], wspace=0.26, hspace=0.34, left=0.025, right=0.82, top=0.905, bottom=0.105)
      axes_maps = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
      for i, N in enumerate(n_gauges):
        ax = axes_maps[i]
        w_p = world.copy()
        col_pct = f'N={N}_Combined_Dom_PctBase'
        w_p['val'] = w_p['map_name'].map(nat_idx[col_pct])
        world.plot(ax=ax, facecolor='#ECEFF1', edgecolor='#B0BEC5', linewidth=0.25)
        w_p.dropna(subset=['val']).plot(column='val', cmap='viridis', ax=ax, vmin=0.0, vmax=vmax_sh, legend=True, legend_kwds={'label': f'% reduced (N={N})', 'orientation': 'horizontal', 'shrink': 0.72, 'pad': 0.02, 'extend': 'max'}, edgecolor='#37474F', linewidth=0.15)
        ax.set_title(f'({chr(97+i)}) N = {N}', fontsize=7.0, fontweight='bold')
        ax.set_xlim([-180, 180])
        ax.set_ylim([-58, 84])
        ax.set_axis_off()
      out_s8 = out_dir / 'Fig_S8_national_gauge_siting.png'
      fig.savefig(out_s8, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.05)
      plt.close(fig)
      print(f'  Saved {out_s8}', flush=True)


def main() -> None:
  parser = argparse.ArgumentParser(description='Generate supplementary figures (Figs. S1–S5; optional S6–S8).')
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

  if targets & {'figs1', 'figs4'}:
    build_fig_s1_and_s4(out_dir, build_s1=('figs1' in targets), build_s4=('figs4' in targets))
  if 'figs2' in targets:
    build_fig_s2(out_dir)
  if 'figs3' in targets:
    copy_fig_s3(out_dir)
  if 'figs5' in targets:
    build_fig_s5(out_dir)
  if targets & {'figs6', 'figs7', 'figs8'}:
    build_optional_fathom_supp_figs(out_dir, targets)

  print('Supplementary figure generation complete!', flush=True)


if __name__ == '__main__':
  main()
