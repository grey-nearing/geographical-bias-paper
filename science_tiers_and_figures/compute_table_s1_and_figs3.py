"""Compute Lead-Time-1 Table S1 across all 38 metrics (M_base and M_DA) and render Fig. S3."""

from concurrent.futures import ProcessPoolExecutor
import glob
import os
import sys
import time
import types
import warnings

os.environ['GEO_BIAS_DATA_DIR'] = './data'
os.environ['GEO_BIAS_THIRD_PARTY_DIR'] = './data'
os.environ['GEO_BIAS_OUTPUT_DIR'] = './data'

sys.path = [p for p in sys.path if not p.endswith('/local_pkg')]

import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score

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

NATURE_DOUBLE_COL_IN = 183.0 / 25.4
SAVE_DPI = 350

_G_STUDY_INDEX = None
_G_HAS_SYS = None


def _init_worker(study_index, has_sys):
  global _G_STUDY_INDEX, _G_HAS_SYS
  _G_STUDY_INDEX = study_index
  _G_HAS_SYS = has_sys


def _process_one_metric(m: str) -> list[dict]:
  sname = file_paths.METRIC_SHORT_NAME_MAP.get(m, m)
  out_min = file_paths.OUTLIER_MIN_MAP.get(m, 0.0)
  out_max = file_paths.OUTLIER_MAX_MAP.get(m, 1.0)
  out_rows = []

  for model_tag, exp_sfx, label_tag in [('ungauged', '', 'M_base (000)'), ('ungauged_da', '_da', 'M_DA (001)')]:
    r_sk = load_experiment_metrics(file_paths.METRICS_DIR, f'random{exp_sfx}', m, lead_time='1').dropna()
    s_sk = load_experiment_metrics(file_paths.METRICS_DIR, f'separated{exp_sfx}', m, lead_time='1').dropna()
    g_sk = load_experiment_metrics(file_paths.METRICS_DIR, f'gauged{exp_sfx}', m, lead_time='1').dropna()

    u_obs_raw = pd.concat([r_sk, s_sk], axis=0)
    all3_obs_raw = pd.concat([g_sk, r_sk, s_sk], axis=0)

    u_obs = u_obs_raw.clip(lower=out_min, upper=out_max)
    all3_obs = all3_obs_raw.clip(lower=out_min, upper=out_max)

    u_q25, u_med, u_q75 = np.percentile(u_obs.values, [25, 50, 75])
    u_iqr = u_q75 - u_q25

    u_raw_q25, u_raw_med, u_raw_q75 = np.percentile(u_obs_raw.values, [25, 50, 75])
    u_raw_iqr = u_raw_q75 - u_raw_q25

    all3_q25, all3_med, all3_q75 = np.percentile(all3_obs.values, [25, 50, 75])
    all3_iqr = all3_q75 - all3_q25

    cv_path = os.path.join(file_paths.CV_PREDS_DIR, f'{model_tag}_regression_cv_preds_{m}.csv')
    df_cv = pd.read_csv(cv_path).dropna(subset=['true', 'pred'])
    cv_r2_all = float(r2_score(df_cv['true'], df_cv['pred']))
    n_cv = len(df_cv)
    df_cv_ung = df_cv.iloc[n_cv // 3 :]
    cv_r2_ung_only = float(r2_score(df_cv_ung['true'], df_cv_ung['pred']))

    hybas_path = os.path.join(file_paths.HYBAS_PREDS_DIR, f'global_hybas_predictions_{model_tag}_{m}.csv')
    df_hybas_pred = pd.read_csv(hybas_path, index_col=0)
    s_hybas = pd.to_numeric(df_hybas_pred.iloc[:, 0], errors='coerce')
    if len(s_hybas) == len(_G_STUDY_INDEX) and str(s_hybas.index[0]) == str(_G_STUDY_INDEX[0]):
      vals = s_hybas.values
    else:
      idx_str = s_hybas.index.astype(str)
      if not str(idx_str[0]).startswith('hybas_'):
        idx_str = 'hybas_' + idx_str
      s_hybas.index = idx_str
      s_hybas = s_hybas.reindex(_G_STUDY_INDEX)
      vals = s_hybas.fillna(s_hybas.median()).values

    vals_clipped = np.clip(vals, out_min, out_max)
    g_q10, g_q25, g_med, g_q75, g_q90 = np.nanpercentile(vals_clipped, [10, 25, 50, 75, 90])
    g_iqr = g_q75 - g_q25

    sys_vals = vals_clipped[_G_HAS_SYS]
    nosys_vals = vals_clipped[~_G_HAS_SYS]
    sys_q25, sys_med, sys_q75 = np.nanpercentile(sys_vals, [25, 50, 75])
    nosys_q25, nosys_med, nosys_q75 = np.nanpercentile(nosys_vals, [25, 50, 75])

    out_rows.append({
        'model_regime': label_tag,
        'model_tag': model_tag,
        'metric_id': m,
        'metric_short': sname,
        'ungauged_cv_n': int(len(u_obs)),
        'unified_3x_cv_n': int(len(all3_obs)),
        'obs_ungauged_median': float(u_med),
        'obs_ungauged_iqr': float(u_iqr),
        'obs_ungauged_q25': float(u_q25),
        'obs_ungauged_q75': float(u_q75),
        'obs_ungauged_unclipped_median': float(u_raw_med),
        'obs_ungauged_unclipped_iqr': float(u_raw_iqr),
        'obs_unified_3x_median': float(all3_med),
        'obs_unified_3x_iqr': float(all3_iqr),
        'skill_predictor_cv_r2': cv_r2_all,
        'skill_predictor_cv_r2_ungauged_subset': cv_r2_ung_only,
        'global_pred_median': float(g_med),
        'global_pred_iqr': float(g_iqr),
        'global_pred_q10': float(g_q10),
        'global_pred_q25': float(g_q25),
        'global_pred_q75': float(g_q75),
        'global_pred_q90': float(g_q90),
        'pred_median_with_warning_sys_60': float(sys_med),
        'pred_iqr_with_warning_sys_60': float(sys_q75 - sys_q25),
        'pred_median_no_warning_sys_113': float(nosys_med),
        'pred_iqr_no_warning_sys_113': float(nosys_q75 - nosys_q25),
    })
  return out_rows


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


def main():
  t0 = time.time()
  configure_rc()

  print('1. Loading 964,084 study watersheds and canonical 60-country CAP warning system status...', flush=True)
  df_basin_v2 = pd.read_parquet(
      os.path.join(file_paths.STATS_DIR, 'basin_exposure_and_blockers_v2.parquet'),
      columns=['has_system'],
  )
  study_index = df_basin_v2.index.astype(str).values
  has_sys = df_basin_v2['has_system'].values.astype(bool)
  n_with_sys = int(has_sys.sum())
  n_no_sys = int((~has_sys).sum())
  print(
      f'   Study watersheds: {len(study_index):,} (60 countries with warning systems: {n_with_sys:,}; '
      f'113 countries lacking warning systems: {n_no_sys:,})',
      flush=True,
  )

  table_s1_metrics = [
      'NSE',
      'KGE',
      'log-NSE',
      'log-KGE',
      'Beta-NSE',
      'Beta-KGE',
      'Alpha-NSE',
      'pearson-r',
      '1.01-year-2.0-window-recall',
      '2.0-year-2.0-window-recall',
      '5.0-year-2.0-window-recall',
      '10.0-year-2.0-window-recall',
      '20.0-year-2.0-window-recall',
      '1.01-year-2.0-window-precision',
      '2.0-year-2.0-window-precision',
      '5.0-year-2.0-window-precision',
      '10.0-year-2.0-window-precision',
      '20.0-year-2.0-window-precision',
      '1.01-year-1.0-window-recall',
      '2.0-year-1.0-window-recall',
      '5.0-year-1.0-window-recall',
      '10.0-year-1.0-window-recall',
      '20.0-year-1.0-window-recall',
      '1.01-year-1.0-window-precision',
      '2.0-year-1.0-window-precision',
      '5.0-year-1.0-window-precision',
      '10.0-year-1.0-window-precision',
      '20.0-year-1.0-window-precision',
      '1.01-year-0.0-window-recall',
      '2.0-year-0.0-window-recall',
      '5.0-year-0.0-window-recall',
      '10.0-year-0.0-window-recall',
      '20.0-year-0.0-window-recall',
      '1.01-year-0.0-window-precision',
      '2.0-year-0.0-window-precision',
      '5.0-year-0.0-window-precision',
      '10.0-year-0.0-window-precision',
      '20.0-year-0.0-window-precision',
  ]

  print('2. Computing Table S1 statistics across all 38 metrics in parallel (12 workers)...', flush=True)
  rows = []
  with ProcessPoolExecutor(max_workers=12, initializer=_init_worker, initargs=(study_index, has_sys)) as ex:
    for m_rows in ex.map(_process_one_metric, table_s1_metrics):
      rows.extend(m_rows)

  df_table_s1 = pd.DataFrame(rows)
  csv_path = os.path.join(OUT_DIR, 'Table_S1_lead1.csv')
  df_table_s1.to_csv(csv_path, index=False)
  print(f'Saved {csv_path} in {time.time() - t0:.1f}s', flush=True)

  # Build Table_S1_lead1.md matching the exact Science manuscript Table S1 format
  md_lines = [
      '# Table S1 (Lead Time 1, $t_0 + 1$ Day): Hydrological Model Cross-Validation Skill, Meta-Model Accuracy, and Global Watershed Predictions Across All 38 Evaluation Metrics',
      '',
      f'- **Study watersheds**: 964,084 total (469,482 in the 60 countries with established flood warning systems; 494,602 in the 113 countries lacking warning systems).',
      '',
      '## Part A: Baseline Global Hydrological Model (`000`, `M_base` / `ungauged`)',
      '',
      '| Metric | Ungauged CV Evaluations ($N$) | Observed Skill at Caravan Gauges: Median (IQR) | Skill Predictor Meta-Model CV $R^2$ | Predicted Global Skill (964,084 Watersheds): Median (IQR) | Predicted Median: Countries with Warning Systems (60) | Predicted Median: Countries Lacking Warning Systems (113) |',
      '| :--- | :---: | :---: | :---: | :---: | :---: | :---: |',
  ]

  df_base = df_table_s1[df_table_s1['model_tag'] == 'ungauged']
  for _, r in df_base.iterrows():
    md_lines.append(
        f"| {r['metric_short']} | {int(r['ungauged_cv_n']):,} | "
        f"{r['obs_ungauged_median']:.2f} ({r['obs_ungauged_iqr']:.2f}) | "
        f"{r['skill_predictor_cv_r2']:.2f} | "
        f"{r['global_pred_median']:.2f} ({r['global_pred_iqr']:.2f}) | "
        f"{r['pred_median_with_warning_sys_60']:.2f} | "
        f"{r['pred_median_no_warning_sys_113']:.2f} |"
    )

  md_lines.extend([
      '',
      '---',
      '',
      '## Part B: 30-Day Variational Data Assimilation Model (`001`, `M_DA` / `ungauged_da`)',
      '',
      '| Metric | Ungauged CV Evaluations ($N$) | Observed Skill at Caravan Gauges: Median (IQR) | Skill Predictor Meta-Model CV $R^2$ | Predicted Global Skill (964,084 Watersheds): Median (IQR) | Predicted Median: Countries with Warning Systems (60) | Predicted Median: Countries Lacking Warning Systems (113) |',
      '| :--- | :---: | :---: | :---: | :---: | :---: | :---: |',
  ])

  df_da = df_table_s1[df_table_s1['model_tag'] == 'ungauged_da']
  for _, r in df_da.iterrows():
    md_lines.append(
        f"| {r['metric_short']} | {int(r['ungauged_cv_n']):,} | "
        f"{r['obs_ungauged_median']:.2f} ({r['obs_ungauged_iqr']:.2f}) | "
        f"{r['skill_predictor_cv_r2']:.2f} | "
        f"{r['global_pred_median']:.2f} ({r['global_pred_iqr']:.2f}) | "
        f"{r['pred_median_with_warning_sys_60']:.2f} | "
        f"{r['pred_median_no_warning_sys_113']:.2f} |"
    )

  md_lines.extend([
      '',
      '---',
      '',
      '## Part C: 3-Decimal Precision Reference Table (`M_base` and `M_DA`)',
      '',
      '| Metric | Regime | Ungauged CV $N$ | Observed Ungauged Med (IQR; Q25–Q75) | Observed Unclipped Med (IQR) | Meta-Model CV $R^2$ (All / Ung-Only) | Global Predicted Med (IQR; Q25–Q75) | With Warning Sys (60) Med (IQR) | Lacking Warning Sys (113) Med (IQR) |',
      '| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |',
  ])
  for _, r in df_table_s1.iterrows():
    md_lines.append(
        f"| {r['metric_short']} | `{r['model_tag']}` | {int(r['ungauged_cv_n']):,} | "
        f"{r['obs_ungauged_median']:.3f} ({r['obs_ungauged_iqr']:.3f}; {r['obs_ungauged_q25']:.3f}–{r['obs_ungauged_q75']:.3f}) | "
        f"{r['obs_ungauged_unclipped_median']:.3f} ({r['obs_ungauged_unclipped_iqr']:.3f}) | "
        f"{r['skill_predictor_cv_r2']:.3f} ({r['skill_predictor_cv_r2_ungauged_subset']:.3f}) | "
        f"{r['global_pred_median']:.3f} ({r['global_pred_iqr']:.3f}; {r['global_pred_q25']:.3f}–{r['global_pred_q75']:.3f}) | "
        f"{r['pred_median_with_warning_sys_60']:.3f} ({r['pred_iqr_with_warning_sys_60']:.3f}) | "
        f"{r['pred_median_no_warning_sys_113']:.3f} ({r['pred_iqr_no_warning_sys_113']:.3f}) |"
    )

  cont_8 = ['NSE', 'KGE', 'log-NSE', 'log-KGE', 'Beta-NSE', 'Beta-KGE', 'Alpha-NSE', 'pearson-r']
  rp_30 = [m for m in table_s1_metrics if m not in cont_8]
  b_all_r2 = df_base['skill_predictor_cv_r2']
  b_cont_r2 = df_base[df_base['metric_id'].isin(cont_8)]['skill_predictor_cv_r2']
  b_rp_r2 = df_base[df_base['metric_id'].isin(rp_30)]['skill_predictor_cv_r2']
  da_all_r2 = df_da['skill_predictor_cv_r2']

  md_lines.extend([
      '',
      '---',
      '',
      '## Summary of `SkillPredictor` 10-Fold Cross-Validation $R^2$ at Lead Time 1',
      f"- **`M_base` (`ungauged`) across all 38 metrics**: median $R^2 = {b_all_r2.median():.3f}$ (mean $= {b_all_r2.mean():.3f}$, range $= {b_all_r2.min():.3f}$ to ${b_all_r2.max():.3f}$).",
      f"  - Across 8 continuous hydrograph metrics: median $R^2 = {b_cont_r2.median():.3f}$ (range $= {b_cont_r2.min():.3f}$ to ${b_cont_r2.max():.3f}$; NSE $R^2 = {float(df_base.loc[df_base['metric_id']=='NSE', 'skill_predictor_cv_r2'].iloc[0]):.3f}$, KGE $R^2 = {float(df_base.loc[df_base['metric_id']=='KGE', 'skill_predictor_cv_r2'].iloc[0]):.3f}$, log-NSE $R^2 = {float(df_base.loc[df_base['metric_id']=='log-NSE', 'skill_predictor_cv_r2'].iloc[0]):.3f}$, $r$ $R^2 = {float(df_base.loc[df_base['metric_id']=='pearson-r', 'skill_predictor_cv_r2'].iloc[0]):.3f}$).",
      f"  - Across 30 flood return-period precision & recall metrics: median $R^2 = {b_rp_r2.median():.3f}$ (range $= {b_rp_r2.min():.3f}$ to ${b_rp_r2.max():.3f}$; 5y 2d recall $R^2 = {float(df_base.loc[df_base['metric_id']=='5.0-year-2.0-window-recall', 'skill_predictor_cv_r2'].iloc[0]):.3f}$, 10y 2d recall $R^2 = {float(df_base.loc[df_base['metric_id']=='10.0-year-2.0-window-recall', 'skill_predictor_cv_r2'].iloc[0]):.3f}$, 20y 2d recall $R^2 = {float(df_base.loc[df_base['metric_id']=='20.0-year-2.0-window-recall', 'skill_predictor_cv_r2'].iloc[0]):.3f}$).",
      f"- **`M_DA` (`ungauged_da`) across all 38 metrics**: median $R^2 = {da_all_r2.median():.3f}$ (mean $= {da_all_r2.mean():.3f}$, range $= {da_all_r2.min():.3f}$ to ${da_all_r2.max():.3f}$; NSE $R^2 = {float(df_da.loc[df_da['metric_id']=='NSE', 'skill_predictor_cv_r2'].iloc[0]):.3f}$, KGE $R^2 = {float(df_da.loc[df_da['metric_id']=='KGE', 'skill_predictor_cv_r2'].iloc[0]):.3f}$, 5y 2d recall $R^2 = {float(df_da.loc[df_da['metric_id']=='5.0-year-2.0-window-recall', 'skill_predictor_cv_r2'].iloc[0]):.3f}$).",
  ])

  md_path = os.path.join(OUT_DIR, 'Table_S1_lead1.md')
  with open(md_path, 'w') as f:
    f.write('\n'.join(md_lines) + '\n')
  print(f'Saved {md_path}', flush=True)

  # 3. Render FigS3_lead1.png using precomputed quantiles from df_base
  print('3. Rendering FigS3_lead1.png...', flush=True)
  metrics_f2 = [
      ('NSE', 'NSE'),
      ('KGE', 'KGE'),
      ('1.01-year-2.0-window-recall', 'Recall (1y)'),
      ('2.0-year-2.0-window-recall', 'Recall (2y)'),
      ('5.0-year-2.0-window-recall', 'Recall (5y)'),
      ('10.0-year-2.0-window-recall', 'Recall (10y)'),
      ('20.0-year-2.0-window-recall', 'Recall (20y)'),
      ('1.01-year-2.0-window-precision', 'Precision (1y)'),
      ('2.0-year-2.0-window-precision', 'Precision (2y)'),
      ('5.0-year-2.0-window-precision', 'Precision (5y)'),
      ('10.0-year-2.0-window-precision', 'Precision (10y)'),
      ('20.0-year-2.0-window-precision', 'Precision (20y)'),
  ]
  df_base_idx = df_base.set_index('metric_id')
  box_stats_f2 = []
  for m_id, m_lbl in metrics_f2:
    r = df_base_idx.loc[m_id]
    box_stats_f2.append({
        'label': m_lbl,
        'med': float(r['global_pred_median']),
        'q1': float(r['global_pred_q25']),
        'q3': float(r['global_pred_q75']),
        'whislo': float(r['global_pred_q10']),
        'whishi': float(r['global_pred_q90']),
        'fliers': [],
    })

  fig_s3, ax_s3 = plt.subplots(figsize=(NATURE_DOUBLE_COL_IN, 3.85))
  bp = ax_s3.bxp(
      list(reversed(box_stats_f2)),
      vert=False, patch_artist=True, showfliers=False, widths=0.62,
  )
  colors_bxp = ['#CC79A7'] * 5 + ['#009E73'] * 5 + ['#0072B2'] * 2
  for patch, col in zip(bp['boxes'], colors_bxp):
    patch.set_facecolor(col)
    patch.set_alpha(0.82)
    patch.set_edgecolor('black')
    patch.set_linewidth(0.55)
  for med_line in bp['medians']:
    med_line.set_color('black')
    med_line.set_linewidth(1.2)
  for w_line in bp['whiskers'] + bp['caps']:
    w_line.set_linewidth(0.6)
  ax_s3.axvline(0.0, color='#616161', linestyle='--', linewidth=0.7)
  ax_s3.set_xlim([-0.22, 0.85])
  ax_s3.set_xlabel('Predicted hydrological skill across 964,084 global watersheds', fontsize=7.5, fontweight='bold')
  ax_s3.set_title('Global watershed forecast skill distributions', fontsize=8.2, fontweight='bold', loc='left', pad=4)
  handles_s3 = [
      Patch(facecolor='#0072B2', edgecolor='black', alpha=0.82, label='Continuous hydrograph (NSE, KGE)'),
      Patch(facecolor='#009E73', edgecolor='black', alpha=0.82, label='Flood recall (2-day window)'),
      Patch(facecolor='#CC79A7', edgecolor='black', alpha=0.82, label='Flood precision (2-day window)'),
  ]
  ax_s3.legend(handles=handles_s3, loc='lower right', frameon=True, framealpha=0.94, fontsize=6.4)
  ax_s3.grid(axis='x', linestyle='--', alpha=0.45)
  sns.despine(ax=ax_s3)
  fig_s3.tight_layout()

  s3_png = os.path.join(OUT_DIR, 'FigS3_lead1.png')
  s3_pdf = os.path.join(OUT_DIR, 'FigS3_lead1.pdf')
  fig_s3.savefig(s3_png, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.03)
  fig_s3.savefig(s3_pdf, bbox_inches='tight', pad_inches=0.03)
  plt.close(fig_s3)
  print(f'Saved {s3_png} in {time.time() - t0:.1f}s!', flush=True)


if __name__ == '__main__':
  main()
