"""Compute Lead-Time-1 Section 3 CV statistics and render Fig. S1 and Fig. S2."""

import glob
import json
import os
import sys
import time
import types
import warnings

os.environ['GEO_BIAS_DATA_DIR'] = './data'
os.environ['GEO_BIAS_THIRD_PARTY_DIR'] = './data'
os.environ['GEO_BIAS_OUTPUT_DIR'] = './data'

# Ensure local_pkg workspace does not shadow google.protobuf
sys.path = [p for p in sys.path if not p.endswith('/local_pkg')]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
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

if './data' not in sys.path:
  sys.path.insert(0, './data')

from local_pkg.intelligence.gsnearing.geographical_bias_paper_v2.utils import file_paths
from local_pkg.intelligence.gsnearing.geographical_bias_paper_v2.utils.io_utils import load_experiment_metrics

warnings.filterwarnings('ignore')

OUT_DIR = './data/science_results'
os.makedirs(OUT_DIR, exist_ok=True)

NATURE_DOUBLE_COL_IN = 183.0 / 25.4  # 7.2047 in
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


def summarize_series(s: pd.Series) -> dict:
  s_clean = s.dropna()
  if len(s_clean) == 0:
    return {'n': 0, 'median': np.nan, 'q25': np.nan, 'q75': np.nan, 'iqr': np.nan, 'mean': np.nan}
  q25, q50, q75 = np.percentile(s_clean.values, [25, 50, 75])
  return {
      'n': int(len(s_clean)),
      'median': float(q50),
      'q25': float(q25),
      'q75': float(q75),
      'iqr': float(q75 - q25),
      'mean': float(s_clean.mean()),
  }


def paired_wilcoxon_stats(s1: pd.Series, s2: pd.Series) -> dict:
  """Computes paired differences (s1 - s2) and two-sided Wilcoxon signed-rank p-value."""
  df_p = pd.concat([s1.rename('a'), s2.rename('b')], axis=1).dropna()
  if len(df_p) == 0:
    return {
        'n_paired': 0,
        'median_a': np.nan,
        'median_b': np.nan,
        'unpaired_median_diff_a_minus_b': np.nan,
        'paired_median_diff_a_minus_b': np.nan,
        'paired_q25_diff': np.nan,
        'paired_q75_diff': np.nan,
        'paired_iqr_diff': np.nan,
        'paired_mean_diff': np.nan,
        'wilcoxon_stat': np.nan,
        'wilcoxon_p': np.nan,
    }
  diff = df_p['a'] - df_p['b']
  q25, q50, q75 = np.percentile(diff.values, [25, 50, 75])
  nonzero = diff[diff != 0.0]
  if len(nonzero) > 0:
    w_res = wilcoxon(df_p['a'].values, df_p['b'].values, alternative='two-sided')
    w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
  else:
    w_stat, w_p = 0.0, 1.0
  return {
      'n_paired': int(len(df_p)),
      'median_a': float(df_p['a'].median()),
      'median_b': float(df_p['b'].median()),
      'unpaired_median_diff_a_minus_b': float(df_p['a'].median() - df_p['b'].median()),
      'paired_median_diff_a_minus_b': float(q50),
      'paired_q25_diff': float(q25),
      'paired_q75_diff': float(q75),
      'paired_iqr_diff': float(q75 - q25),
      'paired_mean_diff': float(diff.mean()),
      'wilcoxon_stat': w_stat,
      'wilcoxon_p': w_p,
  }


def fmt_p(p: float) -> str:
  if p is None or np.isnan(p):
    return 'NaN'
  if p == 0.0 or p < 1e-300:
    return '< 1e-300'
  if p < 0.001:
    return f'{p:.2e}'
  return f'{p:.4f}'


def main():
  t0 = time.time()
  configure_rc()

  metrics_to_eval = file_paths.ALL_METRICS
  results = {
      'lead_time': 1,
      'description': 'Lead-Time-1 (t0 + 1, 1-day-ahead forecast) cross-validation statistics across 8,811 Caravan basins (evaluated on 2016-2023; return periods fit on 1986-2023).',
      'metrics': {},
      'nse_ge_minus_0_5_subset': {},
      'ood_sextiles': {},
  }

  print('1. Computing Lead-Time-1 CV stats across all 38 metrics for Baseline (000) and 30-Day DA (001)...', flush=True)
  for m in metrics_to_eval:
    g_000 = load_experiment_metrics(file_paths.METRICS_DIR, 'gauged', m, rename_to='gauged_000', lead_time='1')
    r_000 = load_experiment_metrics(file_paths.METRICS_DIR, 'random', m, rename_to='random_000', lead_time='1')
    s_000 = load_experiment_metrics(file_paths.METRICS_DIR, 'separated', m, rename_to='separated_000', lead_time='1')

    g_001 = load_experiment_metrics(file_paths.METRICS_DIR, 'gauged_da', m, rename_to='gauged_001', lead_time='1')
    r_001 = load_experiment_metrics(file_paths.METRICS_DIR, 'random_da', m, rename_to='random_001', lead_time='1')
    s_001 = load_experiment_metrics(file_paths.METRICS_DIR, 'separated_da', m, rename_to='separated_001', lead_time='1')

    # Also compute all-3-valid intersection stats
    df_000_all3 = pd.concat([g_000, r_000, s_000], axis=1).dropna()
    df_001_all3 = pd.concat([g_001, r_001, s_001], axis=1).dropna()

    # Ungauged combined [random, separated] (N = 2 * N_basins)
    u_000_comb = pd.concat([r_000, s_000], axis=0)
    u_001_comb = pd.concat([r_001, s_001], axis=0)

    results['metrics'][m] = {
        'short_name': file_paths.METRIC_SHORT_NAME_MAP.get(m, m),
        'baseline_000': {
            'gauged': summarize_series(g_000),
            'random': summarize_series(r_000),
            'separated': summarize_series(s_000),
            'ungauged_combined_2x': summarize_series(u_000_comb),
            'n_all3_valid': int(len(df_000_all3)),
            'pairs': {
                'gauged_minus_random': paired_wilcoxon_stats(g_000, r_000),
                'random_minus_separated': paired_wilcoxon_stats(r_000, s_000),
                'gauged_minus_separated': paired_wilcoxon_stats(g_000, s_000),
            },
        },
        'da_001': {
            'gauged': summarize_series(g_001),
            'random': summarize_series(r_001),
            'separated': summarize_series(s_001),
            'ungauged_combined_2x': summarize_series(u_001_comb),
            'n_all3_valid': int(len(df_001_all3)),
            'pairs': {
                'gauged_minus_random': paired_wilcoxon_stats(g_001, r_001),
                'random_minus_separated': paired_wilcoxon_stats(r_001, s_001),
                'gauged_minus_separated': paired_wilcoxon_stats(g_001, s_001),
            },
        },
        'da_minus_baseline': {
            'gauged_da_minus_base': paired_wilcoxon_stats(g_001, g_000),
            'random_da_minus_base': paired_wilcoxon_stats(r_001, r_000),
            'separated_da_minus_base': paired_wilcoxon_stats(s_001, s_000),
        },
    }

  # 2. NSE >= -0.5 subset & Coverage Bin Breakdown (Fig. S1)
  print('2. Computing NSE >= -0.5 subset and coverage bin breakdown (Fig. S1)...', flush=True)
  file_paths.init('NSE')
  feat_ed3 = pd.read_csv(file_paths.TRAIN_ML_FEATURES_PATH).set_index('gauge_id')

  for exp_tag, sfx in [('baseline_000', ''), ('da_001', '_da')]:
    g_nse = load_experiment_metrics(file_paths.METRICS_DIR, f'gauged{sfx}', 'NSE', rename_to='gauged_NSE', lead_time='1')
    r_nse = load_experiment_metrics(file_paths.METRICS_DIR, f'random{sfx}', 'NSE', rename_to='random_NSE', lead_time='1')
    s_nse = load_experiment_metrics(file_paths.METRICS_DIR, f'separated{sfx}', 'NSE', rename_to='separated_NSE', lead_time='1')

    df_nse_full = feat_ed3.join(r_nse).join(g_nse).join(s_nse).dropna(subset=['random_NSE', 'gauged_NSE', 'separated_NSE'])
    df_nse_full['delta_sep_rand'] = df_nse_full['random_NSE'] - df_nse_full['separated_NSE']
    df_nse_full['delta_gauged_rand'] = df_nse_full['gauged_NSE'] - df_nse_full['random_NSE']
    df_nse_full['delta_gauged_sep'] = df_nse_full['gauged_NSE'] - df_nse_full['separated_NSE']

    df_sub = df_nse_full[
        (df_nse_full['random_NSE'] >= -0.5)
        & (df_nse_full['gauged_NSE'] >= -0.5)
        & (df_nse_full['separated_NSE'] >= -0.5)
    ].copy()

    explicit_bins = [-0.01, 0.01, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ['0', '0.01–0.2', '0.2–0.4', '0.4–0.6', '0.6–0.8', '0.8–1.0']
    cov_cols = ['total_upstream_coverage', 'max_upstream_coverage', 'max_downstream_coverage']

    bin_stats_sub = {}
    bin_stats_full = {}
    for ccol in cov_cols:
      bin_stats_sub[ccol] = {}
      bin_stats_full[ccol] = {}
      b_sub = pd.cut(df_sub[ccol], bins=explicit_bins, labels=bin_labels)
      b_full = pd.cut(df_nse_full[ccol], bins=explicit_bins, labels=bin_labels)
      for bl in bin_labels:
        grp_s = df_sub[b_sub == bl]
        bin_stats_sub[ccol][bl] = {
            'n': int(len(grp_s)),
            'random_minus_separated': summarize_series(grp_s['delta_sep_rand']),
            'gauged_minus_random': summarize_series(grp_s['delta_gauged_rand']),
            'gauged_NSE': summarize_series(grp_s['gauged_NSE']),
            'random_NSE': summarize_series(grp_s['random_NSE']),
            'separated_NSE': summarize_series(grp_s['separated_NSE']),
        }
        grp_f = df_nse_full[b_full == bl]
        bin_stats_full[ccol][bl] = {
            'n': int(len(grp_f)),
            'random_minus_separated': summarize_series(grp_f['delta_sep_rand']),
            'gauged_minus_random': summarize_series(grp_f['delta_gauged_rand']),
        }

    # Also test high coverage (>0.8) vs zero coverage (0)
    high_up_sub = df_sub[df_sub['total_upstream_coverage'] > 0.8]
    zero_up_sub = df_sub[df_sub['total_upstream_coverage'] <= 0.0]
    high_dn_sub = df_sub[df_sub['max_downstream_coverage'] > 0.8]
    zero_dn_sub = df_sub[df_sub['max_downstream_coverage'] <= 0.0]

    results['nse_ge_minus_0_5_subset'][exp_tag] = {
        'n_full_valid': int(len(df_nse_full)),
        'n_subset_ge_minus_0_5': int(len(df_sub)),
        'pct_retained': float(100.0 * len(df_sub) / len(df_nse_full)),
        'subset_gauged_NSE': summarize_series(df_sub['gauged_NSE']),
        'subset_random_NSE': summarize_series(df_sub['random_NSE']),
        'subset_separated_NSE': summarize_series(df_sub['separated_NSE']),
        'subset_pairs': {
            'gauged_minus_random': paired_wilcoxon_stats(df_sub['gauged_NSE'], df_sub['random_NSE']),
            'random_minus_separated': paired_wilcoxon_stats(df_sub['random_NSE'], df_sub['separated_NSE']),
            'gauged_minus_separated': paired_wilcoxon_stats(df_sub['gauged_NSE'], df_sub['separated_NSE']),
        },
        'high_vs_zero_coverage_subset': {
            'total_upstream_gt_0_8': {
                'n': int(len(high_up_sub)),
                'random_minus_separated': summarize_series(high_up_sub['delta_sep_rand']),
                'gauged_minus_random': summarize_series(high_up_sub['delta_gauged_rand']),
            },
            'total_upstream_eq_0': {
                'n': int(len(zero_up_sub)),
                'random_minus_separated': summarize_series(zero_up_sub['delta_sep_rand']),
                'gauged_minus_random': summarize_series(zero_up_sub['delta_gauged_rand']),
            },
            'max_downstream_gt_0_8': {
                'n': int(len(high_dn_sub)),
                'random_minus_separated': summarize_series(high_dn_sub['delta_sep_rand']),
                'gauged_minus_random': summarize_series(high_dn_sub['delta_gauged_rand']),
            },
            'max_downstream_eq_0': {
                'n': int(len(zero_dn_sub)),
                'random_minus_separated': summarize_series(zero_dn_sub['delta_sep_rand']),
                'gauged_minus_random': summarize_series(zero_dn_sub['delta_gauged_rand']),
            },
        },
        'coverage_bins_subset': bin_stats_sub,
        'coverage_bins_full': bin_stats_full,
    }

    if exp_tag == 'baseline_000':
      # Render FigS1_lead1.png
      print('   -> Rendering FigS1_lead1.png...', flush=True)
      melted_ed3 = (
          df_sub.rename_axis('gauge_id')
          .reset_index()
          .melt(
              id_vars=['gauge_id', 'delta_sep_rand', 'delta_gauged_rand'],
              value_vars=['total_upstream_coverage', 'max_upstream_coverage', 'max_downstream_coverage'],
              var_name='Coverage_Type',
              value_name='Coverage_Fraction',
          )
      )
      melted_ed3['Coverage_Bin'] = pd.cut(melted_ed3['Coverage_Fraction'], bins=explicit_bins, labels=bin_labels)
      cov_name_map = {
          'total_upstream_coverage': 'Total upstream coverage',
          'max_upstream_coverage': 'Max single-gauge upstream',
          'max_downstream_coverage': 'Max downstream coverage',
      }
      melted_ed3['Coverage Metric'] = melted_ed3['Coverage_Type'].map(cov_name_map)

      fig_s1, axes_s1 = plt.subplots(2, 1, figsize=(NATURE_DOUBLE_COL_IN, 4.65), sharex=True)
      sns.boxplot(
          data=melted_ed3, x='Coverage_Bin', y='delta_sep_rand', hue='Coverage Metric',
          palette=['#0072B2', '#E69F00', '#009E73'], showfliers=False, whis=[10, 90],
          linewidth=0.6, ax=axes_s1[0],
      )
      axes_s1[0].axhline(
          df_sub['delta_sep_rand'].median(), color='#882255', linestyle='--', lw=1.0,
          label=f"Overall median ΔNSE (+{df_sub['delta_sep_rand'].median():.3f})",
      )
      axes_s1[0].axhline(0.0, color='gray', lw=0.7, zorder=0)
      axes_s1[0].set_ylabel('ΔNSE (Random − Separated)', fontsize=7.2, fontweight='bold')
      axes_s1[0].set_title(
          f'(a) River-network information transfer in ungauged CV (N = {len(df_sub):,})',
          fontsize=8.0, fontweight='bold', loc='left', pad=3
      )
      axes_s1[0].legend(loc='upper right', fontsize=6.2, frameon=True, framealpha=0.94)
      axes_s1[0].set_xlabel('')
      sns.despine(ax=axes_s1[0])

      sns.boxplot(
          data=melted_ed3, x='Coverage_Bin', y='delta_gauged_rand', hue='Coverage Metric',
          palette=['#0072B2', '#E69F00', '#009E73'], showfliers=False, whis=[10, 90],
          linewidth=0.6, ax=axes_s1[1],
      )
      axes_s1[1].axhline(
          df_sub['delta_gauged_rand'].median(), color='#882255', linestyle='--', lw=1.0,
          label=f"Overall median ΔNSE (+{df_sub['delta_gauged_rand'].median():.3f})",
      )
      axes_s1[1].axhline(0.0, color='gray', lw=0.7, zorder=0)
      axes_s1[1].set_ylabel('ΔNSE (Gauged − Random)', fontsize=7.2, fontweight='bold')
      axes_s1[1].set_title(
          f'(b) Local at-site gauge calibration gain (N = {len(df_sub):,})',
          fontsize=8.0, fontweight='bold', loc='left', pad=3
      )
      axes_s1[1].legend(loc='upper right', fontsize=6.2, frameon=True, framealpha=0.94)
      axes_s1[1].set_xlabel('Drainage area coverage fraction bin', fontsize=7.2, fontweight='bold')
      sns.despine(ax=axes_s1[1])

      fig_s1.tight_layout()
      s1_png = os.path.join(OUT_DIR, 'FigS1_lead1.png')
      s1_pdf = os.path.join(OUT_DIR, 'FigS1_lead1.pdf')
      fig_s1.savefig(s1_png, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.03)
      fig_s1.savefig(s1_pdf, bbox_inches='tight', pad_inches=0.03)
      plt.close(fig_s1)
      print(f'   -> Saved {s1_png}', flush=True)

  # 3. OOD Sextiles Breakdown (Q1 most OOD to Q6 most in-distribution) & Fig. S2
  print('3. Computing OOD sextile breakdowns for NSE and 5y 2d flood recall (Fig. S2)...', flush=True)
  rand_anom_df = pd.read_csv(file_paths.RANDOM_ANOMALIES_PATH).set_index('gauge_id')
  sep_anom_df = pd.read_csv(file_paths.SEPARATED_ANOMALIES_PATH).set_index('gauge_id')

  # Note on sign convention of anomaly_score_if, anomaly_score_ocsvm, anomaly_score_lof:
  # In sklearn / anomaly_scores CSVs, lower (more negative) values = more anomalous (OOD, Q1),
  # higher (more positive) values = more in-distribution (Q6)!
  # Let's compute composite in-distribution percentile = mean of rank(pct=True) of (if, ocsvm, lof)
  for df_a in [rand_anom_df, sep_anom_df]:
    df_a['composite_indist_pct'] = (
        df_a['anomaly_score_if'].rank(pct=True)
        + df_a['anomaly_score_ocsvm'].rank(pct=True)
        + df_a['anomaly_score_lof'].rank(pct=True)
    ) / 3.0

  sextile_keys = ['Q1 (Most OOD)', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6 (Most In-Dist)']
  detector_cols = [
      ('composite_indist_pct', 'Composite Mean Percentile (IF + OCSVM + LOF)'),
      ('anomaly_score_if', 'Isolation Forest'),
      ('anomaly_score_ocsvm', 'One-Class SVM'),
      ('anomaly_score_lof', 'Local Outlier Factor'),
  ]

  for exp_tag, sfx in [('baseline_000', ''), ('da_001', '_da')]:
    results['ood_sextiles'][exp_tag] = {}
    for target_m, m_min in [('NSE', None), ('NSE_ge_minus_0_5', -0.5), ('5.0-year-2.0-window-recall', 0.0), ('KGE', None)]:
      actual_m = 'NSE' if target_m.startswith('NSE') else target_m
      g_sk = load_experiment_metrics(file_paths.METRICS_DIR, f'gauged{sfx}', actual_m, rename_to='gauged', lead_time='1')
      r_sk = load_experiment_metrics(file_paths.METRICS_DIR, f'random{sfx}', actual_m, rename_to='random', lead_time='1')
      s_sk = load_experiment_metrics(file_paths.METRICS_DIR, f'separated{sfx}', actual_m, rename_to='separated', lead_time='1')

      df_rand_j = rand_anom_df.join(g_sk, how='inner').join(r_sk, how='inner').dropna(subset=['gauged', 'random'])
      df_sep_j = sep_anom_df.join(g_sk, how='inner').join(s_sk, how='inner').dropna(subset=['gauged', 'separated'])
      if m_min is not None:
        df_rand_j = df_rand_j[(df_rand_j['gauged'] >= m_min) & (df_rand_j['random'] >= m_min)]
        df_sep_j = df_sep_j[(df_sep_j['gauged'] >= m_min) & (df_sep_j['separated'] >= m_min)]

      results['ood_sextiles'][exp_tag][target_m] = {}
      for acol, albl in detector_cols:
        b_rand = pd.qcut(df_rand_j[acol], q=6, labels=sextile_keys, duplicates='drop')
        b_sep = pd.qcut(df_sep_j[acol], q=6, labels=sextile_keys, duplicates='drop')

        sex_dict = {'detector_label': albl, 'random': {}, 'separated': {}, 'gauged_by_random_anom': {}, 'gauged_by_sep_anom': {}}
        for sk in sextile_keys:
          sex_dict['random'][sk] = summarize_series(df_rand_j.loc[b_rand == sk, 'random'])
          sex_dict['gauged_by_random_anom'][sk] = summarize_series(df_rand_j.loc[b_rand == sk, 'gauged'])
          sex_dict['separated'][sk] = summarize_series(df_sep_j.loc[b_sep == sk, 'separated'])
          sex_dict['gauged_by_sep_anom'][sk] = summarize_series(df_sep_j.loc[b_sep == sk, 'gauged'])

        # Q6 vs Q1 differences and Mann-Whitney U p-values
        for split_name, df_use, b_use, col_use in [
            ('random', df_rand_j, b_rand, 'random'),
            ('separated', df_sep_j, b_sep, 'separated'),
            ('gauged_by_random_anom', df_rand_j, b_rand, 'gauged'),
            ('gauged_by_sep_anom', df_sep_j, b_sep, 'gauged'),
        ]:
          q1_vals = df_use.loc[b_use == sextile_keys[0], col_use].dropna()
          q6_vals = df_use.loc[b_use == sextile_keys[-1], col_use].dropna()
          if len(q1_vals) > 0 and len(q6_vals) > 0:
            mwu_stat, mwu_p = mannwhitneyu(q6_vals.values, q1_vals.values, alternative='two-sided')
            rho_s, p_s = spearmanr(df_use[acol].values, df_use[col_use].values)
            sex_dict[f'{split_name}_summary'] = {
                'q1_median': float(q1_vals.median()),
                'q6_median': float(q6_vals.median()),
                'q6_minus_q1_median_diff': float(q6_vals.median() - q1_vals.median()),
                'mannwhitney_q6_vs_q1_p': float(mwu_p),
                'spearman_r_with_indist_score': float(rho_s),
                'spearman_p': float(p_s),
            }
        results['ood_sextiles'][exp_tag][target_m][acol] = sex_dict

  # Render FigS2_lead1.png (matching Extended Data Figure 4 / Fig. S2 in Science manuscript)
  print('   -> Rendering FigS2_lead1.png...', flush=True)
  file_paths.init('5.0-year-2.0-window-recall')
  g_sk_ed4 = load_experiment_metrics(file_paths.METRICS_DIR, 'gauged', '5.0-year-2.0-window-recall', rename_to='gauged_rec5', lead_time='1')
  r_sk_ed4 = load_experiment_metrics(file_paths.METRICS_DIR, 'random', '5.0-year-2.0-window-recall', rename_to='random_rec5', lead_time='1')
  s_sk_ed4 = load_experiment_metrics(file_paths.METRICS_DIR, 'separated', '5.0-year-2.0-window-recall', rename_to='separated_rec5', lead_time='1')

  df_top_ed4 = rand_anom_df.join(g_sk_ed4, how='inner').join(r_sk_ed4, how='inner').dropna(subset=['gauged_rec5', 'random_rec5'])
  df_top_ed4 = df_top_ed4[(df_top_ed4['gauged_rec5'] >= 0.0) & (df_top_ed4['random_rec5'] >= 0.0)]
  df_bot_ed4 = sep_anom_df.join(g_sk_ed4, how='inner').join(s_sk_ed4, how='inner').dropna(subset=['gauged_rec5', 'separated_rec5'])
  df_bot_ed4 = df_bot_ed4[(df_bot_ed4['gauged_rec5'] >= 0.0) & (df_bot_ed4['separated_rec5'] >= 0.0)]

  anom_cols = ['anomaly_score_if', 'anomaly_score_ocsvm', 'anomaly_score_lof']
  anom_titles = ['Isolation Forest', 'One-Class SVM', 'Local Outlier Factor']
  ed4_ltrs = [['(a)', '(b)', '(c)'], ['(d)', '(e)', '(f)'], ['(g)', '(h)', '(i)']]
  sextile_labels = ['Q1\n(OOD)', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6\n(In-dist)']

  fig_s2, axes_s2 = plt.subplots(3, 3, figsize=(NATURE_DOUBLE_COL_IN, 5.80), sharey=True)
  for i, (acol, atitle) in enumerate(zip(anom_cols, anom_titles)):
    df_top_ed4[f'{acol}_bin'] = pd.qcut(df_top_ed4[acol], q=6, labels=sextile_labels, duplicates='drop')
    df_bot_ed4[f'{acol}_bin'] = pd.qcut(df_bot_ed4[acol], q=6, labels=sextile_labels, duplicates='drop')

    ax0 = axes_s2[0, i]
    sns.boxplot(
        data=df_top_ed4, x=f'{acol}_bin', y='random_rec5', hue=f'{acol}_bin',
        palette='Blues_r', legend=False, showfliers=False, whis=[10, 90], linewidth=0.55, ax=ax0
    )
    ax0.axhline(df_top_ed4['random_rec5'].median(), color='#0d47a1', ls='--', lw=0.9)
    ax0.set_xlabel('')
    ax0.set_title(f'{ed4_ltrs[0][i]} Random: {atitle}', fontsize=7.4, fontweight='bold', loc='left', pad=2)
    ax0.set_ylabel('5-yr recall (Random)' if i == 0 else '', fontsize=7.0, fontweight='bold')
    ax0.tick_params(axis='x', labelsize=6.0)
    sns.despine(ax=ax0)

    ax1 = axes_s2[1, i]
    sns.boxplot(
        data=df_bot_ed4, x=f'{acol}_bin', y='separated_rec5', hue=f'{acol}_bin',
        palette='Oranges_r', legend=False, showfliers=False, whis=[10, 90], linewidth=0.55, ax=ax1
    )
    ax1.axhline(df_bot_ed4['separated_rec5'].median(), color='#e65100', ls='--', lw=0.9)
    ax1.set_xlabel('')
    ax1.set_title(f'{ed4_ltrs[1][i]} Separated: {atitle}', fontsize=7.4, fontweight='bold', loc='left', pad=2)
    ax1.set_ylabel('5-yr recall (Separated)' if i == 0 else '', fontsize=7.0, fontweight='bold')
    ax1.tick_params(axis='x', labelsize=6.0)
    sns.despine(ax=ax1)

    ax2 = axes_s2[2, i]
    sns.boxplot(
        data=df_top_ed4, x=f'{acol}_bin', y='gauged_rec5', hue=f'{acol}_bin',
        palette='Greens_r', legend=False, showfliers=False, whis=[10, 90], linewidth=0.55, ax=ax2
    )
    ax2.axhline(df_top_ed4['gauged_rec5'].median(), color='#1b5e20', ls='--', lw=0.9)
    ax2.set_xlabel(f'{atitle} sextile', fontsize=6.8, fontweight='bold')
    ax2.set_title(f'{ed4_ltrs[2][i]} Gauged: {atitle}', fontsize=7.4, fontweight='bold', loc='left', pad=2)
    ax2.set_ylabel('5-yr recall (Gauged)' if i == 0 else '', fontsize=7.0, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=6.0)
    sns.despine(ax=ax2)

  fig_s2.tight_layout()
  s2_png = os.path.join(OUT_DIR, 'FigS2_lead1.png')
  s2_pdf = os.path.join(OUT_DIR, 'FigS2_lead1.pdf')
  fig_s2.savefig(s2_png, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.03)
  fig_s2.savefig(s2_pdf, bbox_inches='tight', pad_inches=0.03)
  plt.close(fig_s2)
  print(f'   -> Saved {s2_png}', flush=True)

  # Save JSON
  json_path = os.path.join(OUT_DIR, 'sec3_cv_stats_lead1.json')
  with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
  print(f'Saved {json_path}', flush=True)

  # Build comprehensive Markdown summary (`sec3_cv_stats_lead1.md`)
  md_lines = [
      '# Section 3 Cross-Validation Statistics at Lead Time 1 (`t0 + 1`, 1-Day-Ahead Forecast)',
      '',
      'All metrics are evaluated out-of-sample across the 8,811 Caravan basins (`2016-01-01` to `2023-12-31`, with return-period thresholds fit over the 38-year `1986-01-01` to `2023-12-31` record) at **Lead Time 1 (`t0 + 1` day)**.',
      '',
      '---',
      '',
      '## 1. Core Hydrological & Flood Metrics: Baseline (`000`, No DA) vs. 30-Day Variational DA (`001`)',
      '',
      '### 1A. Baseline (`000`) Medians, IQRs, Paired Differences, and Two-Sided Wilcoxon Signed-Rank Tests',
      '',
      '| Metric | $N$ (Gauged / Rand / Sep) | Gauged Median (IQR; Q25–Q75) | Random Ungauged Median (IQR; Q25–Q75) | Separated Ungauged Median (IQR; Q25–Q75) | Paired $\\Delta(\\text{Gauged} - \\text{Random})$ Med ($p$-val) | Paired $\\Delta(\\text{Random} - \\text{Separated})$ Med ($p$-val) | Paired $\\Delta(\\text{Gauged} - \\text{Separated})$ Med ($p$-val) |',
      '| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |',
  ]

  core_order = [
      'NSE', 'KGE', 'pearson-r', 'log-NSE', 'log-KGE', 'Beta-NSE', 'Beta-KGE', 'Alpha-NSE',
      '1.01-year-2.0-window-recall', '2.0-year-2.0-window-recall', '5.0-year-2.0-window-recall', '10.0-year-2.0-window-recall', '20.0-year-2.0-window-recall',
      '1.01-year-2.0-window-precision', '2.0-year-2.0-window-precision', '5.0-year-2.0-window-precision', '10.0-year-2.0-window-precision', '20.0-year-2.0-window-precision',
      '1.01-year-1.0-window-recall', '2.0-year-1.0-window-recall', '5.0-year-1.0-window-recall', '10.0-year-1.0-window-recall', '20.0-year-1.0-window-recall',
      '1.01-year-1.0-window-precision', '2.0-year-1.0-window-precision', '5.0-year-1.0-window-precision', '10.0-year-1.0-window-precision', '20.0-year-1.0-window-precision',
      '1.01-year-0.0-window-recall', '2.0-year-0.0-window-recall', '5.0-year-0.0-window-recall', '10.0-year-0.0-window-recall', '20.0-year-0.0-window-recall',
      '1.01-year-0.0-window-precision', '2.0-year-0.0-window-precision', '5.0-year-0.0-window-precision', '10.0-year-0.0-window-precision', '20.0-year-0.0-window-precision',
  ]

  for m in core_order:
    d = results['metrics'][m]
    sname = d['short_name']
    b = d['baseline_000']
    g, r, s = b['gauged'], b['random'], b['separated']
    p_gr = b['pairs']['gauged_minus_random']
    p_rs = b['pairs']['random_minus_separated']
    p_gs = b['pairs']['gauged_minus_separated']
    md_lines.append(
        f"| **{sname}** (`{m}`) | {g['n']:,} / {r['n']:,} / {s['n']:,} | "
        f"{g['median']:.3f} ({g['iqr']:.3f}; {g['q25']:.3f} to {g['q75']:.3f}) | "
        f"{r['median']:.3f} ({r['iqr']:.3f}; {r['q25']:.3f} to {r['q75']:.3f}) | "
        f"{s['median']:.3f} ({s['iqr']:.3f}; {s['q25']:.3f} to {s['q75']:.3f}) | "
        f"{p_gr['paired_median_diff_a_minus_b']:+.3f} (unpaired {p_gr['unpaired_median_diff_a_minus_b']:+.3f}; $p={fmt_p(p_gr['wilcoxon_p'])}$) | "
        f"{p_rs['paired_median_diff_a_minus_b']:+.3f} (unpaired {p_rs['unpaired_median_diff_a_minus_b']:+.3f}; $p={fmt_p(p_rs['wilcoxon_p'])}$) | "
        f"{p_gs['paired_median_diff_a_minus_b']:+.3f} (unpaired {p_gs['unpaired_median_diff_a_minus_b']:+.3f}; $p={fmt_p(p_gs['wilcoxon_p'])}$) |"
    )

  md_lines.extend([
      '',
      '### 1B. 30-Day Variational Data Assimilation (`001`) Medians, IQRs, Paired Differences, and Gain Over Baseline (`001 - 000`)',
      '',
      '| Metric | Gauged DA Median (IQR) | Random DA Median (IQR) | Separated DA Median (IQR) | Paired $\\Delta(\\text{Gauged DA} - \\text{Base})$ Med ($p$-val) | Paired $\\Delta(\\text{Random DA} - \\text{Base})$ Med ($p$-val) | Paired $\\Delta(\\text{Sep DA} - \\text{Base})$ Med ($p$-val) |',
      '| :--- | :---: | :---: | :---: | :---: | :---: | :---: |',
  ])

  for m in core_order:
    d = results['metrics'][m]
    sname = d['short_name']
    da = d['da_001']
    db = d['da_minus_baseline']
    g, r, s = da['gauged'], da['random'], da['separated']
    dg = db['gauged_da_minus_base']
    dr = db['random_da_minus_base']
    ds = db['separated_da_minus_base']
    md_lines.append(
        f"| **{sname}** (`{m}`) | "
        f"{g['median']:.3f} ({g['iqr']:.3f}; {g['q25']:.3f}–{g['q75']:.3f}) | "
        f"{r['median']:.3f} ({r['iqr']:.3f}; {r['q25']:.3f}–{r['q75']:.3f}) | "
        f"{s['median']:.3f} ({s['iqr']:.3f}; {s['q25']:.3f}–{s['q75']:.3f}) | "
        f"{dg['paired_median_diff_a_minus_b']:+.3f} ($p={fmt_p(dg['wilcoxon_p'])}$) | "
        f"{dr['paired_median_diff_a_minus_b']:+.3f} ($p={fmt_p(dr['wilcoxon_p'])}$) | "
        f"{ds['paired_median_diff_a_minus_b']:+.3f} ($p={fmt_p(ds['wilcoxon_p'])}$) |"
    )

  # Section 2: NSE >= -0.5 subset (Fig. S1)
  sub0 = results['nse_ge_minus_0_5_subset']['baseline_000']
  sub1 = results['nse_ge_minus_0_5_subset']['da_001']
  md_lines.extend([
      '',
      '---',
      '',
      '## 2. `NSE >= -0.5` Subset & River-Network Coverage Fraction Bins (`Fig. S1`)',
      '',
      f"- **Baseline (`000`)**: Out of $N = {sub0['n_full_valid']:,}$ gauges with valid NSE in all three splits, **$N = {sub0['n_subset_ge_minus_0_5']:,}$ gauges ({sub0['pct_retained']:.1f}%)** have $\\text{{NSE}} \\ge -0.5$ across all three cross-validation splits.",
      f"  - Subset Gauged NSE median = **{sub0['subset_gauged_NSE']['median']:.3f}** (IQR {sub0['subset_gauged_NSE']['iqr']:.3f}; {sub0['subset_gauged_NSE']['q25']:.3f}–{sub0['subset_gauged_NSE']['q75']:.3f})",
      f"  - Subset Random NSE median = **{sub0['subset_random_NSE']['median']:.3f}** (IQR {sub0['subset_random_NSE']['iqr']:.3f}; {sub0['subset_random_NSE']['q25']:.3f}–{sub0['subset_random_NSE']['q75']:.3f})",
      f"  - Subset Separated NSE median = **{sub0['subset_separated_NSE']['median']:.3f}** (IQR {sub0['subset_separated_NSE']['iqr']:.3f}; {sub0['subset_separated_NSE']['q25']:.3f}–{sub0['subset_separated_NSE']['q75']:.3f})",
      f"  - **Overall paired $\\Delta\\text{{NSE}}(\\text{{Random}} - \\text{{Separated}})$** (Fig. S1a horizontal line): **+{sub0['subset_pairs']['random_minus_separated']['paired_median_diff_a_minus_b']:.3f}** (IQR {sub0['subset_pairs']['random_minus_separated']['paired_iqr_diff']:.3f}; {sub0['subset_pairs']['random_minus_separated']['paired_q25_diff']:+.3f} to {sub0['subset_pairs']['random_minus_separated']['paired_q75_diff']:+.3f}; Wilcoxon $p = {fmt_p(sub0['subset_pairs']['random_minus_separated']['wilcoxon_p'])}$).",
      f"  - **Overall paired $\\Delta\\text{{NSE}}(\\text{{Gauged}} - \\text{{Random}})$** (Fig. S1b horizontal line): **+{sub0['subset_pairs']['gauged_minus_random']['paired_median_diff_a_minus_b']:.3f}** (IQR {sub0['subset_pairs']['gauged_minus_random']['paired_iqr_diff']:.3f}; {sub0['subset_pairs']['gauged_minus_random']['paired_q25_diff']:+.3f} to {sub0['subset_pairs']['gauged_minus_random']['paired_q75_diff']:+.3f}; Wilcoxon $p = {fmt_p(sub0['subset_pairs']['gauged_minus_random']['wilcoxon_p'])}$).",
      f"  - **High upstream coverage ($>0.8$, $N={sub0['high_vs_zero_coverage_subset']['total_upstream_gt_0_8']['n']:,}$) vs. Zero upstream coverage ($0$, $N={sub0['high_vs_zero_coverage_subset']['total_upstream_eq_0']['n']:,}$)**:",
      f"    - $\\Delta\\text{{NSE}}(\\text{{Random}} - \\text{{Separated}})$: **+{sub0['high_vs_zero_coverage_subset']['total_upstream_gt_0_8']['random_minus_separated']['median']:.3f}** (IQR {sub0['high_vs_zero_coverage_subset']['total_upstream_gt_0_8']['random_minus_separated']['iqr']:.3f}) at $>0.8$ upstream coverage vs. **+{sub0['high_vs_zero_coverage_subset']['total_upstream_eq_0']['random_minus_separated']['median']:.3f}** (IQR {sub0['high_vs_zero_coverage_subset']['total_upstream_eq_0']['random_minus_separated']['iqr']:.3f}) at $0$ upstream coverage.",
      f"    - $\\Delta\\text{{NSE}}(\\text{{Gauged}} - \\text{{Random}})$: **+{sub0['high_vs_zero_coverage_subset']['total_upstream_gt_0_8']['gauged_minus_random']['median']:.3f}** (IQR {sub0['high_vs_zero_coverage_subset']['total_upstream_gt_0_8']['gauged_minus_random']['iqr']:.3f}) at $>0.8$ upstream coverage vs. **+{sub0['high_vs_zero_coverage_subset']['total_upstream_eq_0']['gauged_minus_random']['median']:.3f}** (IQR {sub0['high_vs_zero_coverage_subset']['total_upstream_eq_0']['gauged_minus_random']['iqr']:.3f}) at $0$ upstream coverage.",
      f"  - **High downstream coverage ($>0.8$, $N={sub0['high_vs_zero_coverage_subset']['max_downstream_gt_0_8']['n']:,}$) vs. Zero downstream coverage ($0$, $N={sub0['high_vs_zero_coverage_subset']['max_downstream_eq_0']['n']:,}$)**:",
      f"    - $\\Delta\\text{{NSE}}(\\text{{Random}} - \\text{{Separated}})$: **+{sub0['high_vs_zero_coverage_subset']['max_downstream_gt_0_8']['random_minus_separated']['median']:.3f}** at $>0.8$ downstream coverage vs. **+{sub0['high_vs_zero_coverage_subset']['max_downstream_eq_0']['random_minus_separated']['median']:.3f}** at $0$ downstream coverage.",
      f"    - $\\Delta\\text{{NSE}}(\\text{{Gauged}} - \\text{{Random}})$: **+{sub0['high_vs_zero_coverage_subset']['max_downstream_gt_0_8']['gauged_minus_random']['median']:.3f}** at $>0.8$ downstream coverage vs. **+{sub0['high_vs_zero_coverage_subset']['max_downstream_eq_0']['gauged_minus_random']['median']:.3f}** at $0$ downstream coverage.",
      f"- **30-Day Variational DA (`001`)**: $N = {sub1['n_subset_ge_minus_0_5']:,}$ gauges ({sub1['pct_retained']:.1f}%) have $\\text{{NSE}} \\ge -0.5$ across all three DA splits; overall paired $\\Delta\\text{{NSE}}(\\text{{Random DA}} - \\text{{Separated DA}}) = +{sub1['subset_pairs']['random_minus_separated']['paired_median_diff_a_minus_b']:.3f}$ and $\\Delta\\text{{NSE}}(\\text{{Gauged DA}} - \\text{{Random DA}}) = +{sub1['subset_pairs']['gauged_minus_random']['paired_median_diff_a_minus_b']:.3f}$.",
      '',
      '### Coverage Bin Breakdown (`NSE >= -0.5` Subset, Baseline `000`)',
      '',
      '| Coverage Metric | Bin | $N$ | $\\Delta\\text{NSE}(\\text{Random} - \\text{Separated})$ Med (IQR; Q25–Q75) | $\\Delta\\text{NSE}(\\text{Gauged} - \\text{Random})$ Med (IQR; Q25–Q75) |',
      '| :--- | :---: | :---: | :---: | :---: |',
  ])

  for ccol in ['total_upstream_coverage', 'max_upstream_coverage', 'max_downstream_coverage']:
    for bl, bstats in sub0['coverage_bins_subset'][ccol].items():
      rs = bstats['random_minus_separated']
      gr = bstats['gauged_minus_random']
      md_lines.append(
          f"| `{ccol}` | `{bl}` | {bstats['n']:,} | "
          f"{rs['median']:+.3f} ({rs['iqr']:.3f}; {rs['q25']:+.3f} to {rs['q75']:+.3f}) | "
          f"{gr['median']:+.3f} ({gr['iqr']:.3f}; {gr['q25']:+.3f} to {gr['q75']:+.3f}) |"
      )

  # Section 3: OOD Sextiles Breakdown (Fig. S2)
  md_lines.extend([
      '',
      '---',
      '',
      '## 3. Hydroclimatic Out-of-Distribution (OOD) Sextile Breakdown (`Q1` Most OOD to `Q6` Most In-Distribution, `Fig. S2`)',
      '',
  ])

  for target_m, target_title in [
      ('NSE', 'NSE (All Evaluated Gauges)'),
      ('NSE_ge_minus_0_5', 'NSE (NSE >= -0.5 Subset)'),
      ('5.0-year-2.0-window-recall', '5-Year, 2-Day Flood Recall'),
      ('KGE', 'KGE (All Evaluated Gauges)'),
  ]:
    md_lines.append(f'### 3.{target_m}: {target_title} (Baseline `000`)')
    md_lines.append('')
    md_lines.append('| Anomaly Detector | Split | Q1 (Most OOD) Med (IQR) | Q2 Med (IQR) | Q3 Med (IQR) | Q4 Med (IQR) | Q5 Med (IQR) | Q6 (Most In-Dist) Med (IQR) | $\\Delta(\\text{Q6} - \\text{Q1})$ | Mann-Whitney $p$ (Q6 vs Q1) |')
    md_lines.append('| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |')
    for acol, _ in detector_cols:
      sdict = results['ood_sextiles']['baseline_000'][target_m][acol]
      albl = sdict['detector_label']
      for sp_key, sp_label in [
          ('random', 'Random Ungauged'),
          ('separated', 'Separated Ungauged'),
          ('gauged_by_random_anom', 'Gauged (Random Anom Bins)'),
      ]:
        row_cells = []
        for sk in sextile_keys:
          st = sdict[sp_key][sk]
          row_cells.append(f"{st['median']:.3f} ({st['iqr']:.3f})")
        summ = sdict[f'{sp_key}_summary']
        md_lines.append(
            f"| {albl} | {sp_label} | "
            + ' | '.join(row_cells)
            + f" | **{summ['q6_minus_q1_median_diff']:+.3f}** | {fmt_p(summ['mannwhitney_q6_vs_q1_p'])} |"
        )
    md_lines.append('')

  md_path = os.path.join(OUT_DIR, 'sec3_cv_stats_lead1.md')
  with open(md_path, 'w') as f:
    f.write('\n'.join(md_lines) + '\n')
  print(f'Saved {md_path} in {time.time() - t0:.1f}s!', flush=True)


if __name__ == '__main__':
  main()
