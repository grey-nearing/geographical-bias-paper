# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation Synthesis: Global & National Gauge Placement Results (v2: M_ungauged).

Generates publication-ready Extended Data Figure 5, Extended Data Figure 6, and
country-level N-gauge placement summary statistics strictly from the v2
`M_ungauged`-centered joint topological and OOD evaluation outputs.
"""

from collections import defaultdict
import io
import json
import os
import shutil
import time
import warnings

from absl import app
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import cap_utils
from utils import file_paths
from utils import io_utils
from utils import viz_utils
from utils import gfile_compat as gfile

warnings.filterwarnings('ignore')


def _safe_unlink(path: str):
  if os.path.lexists(path):
    if os.path.isdir(path) and not os.path.islink(path):
      shutil.rmtree(path)
    else:
      os.unlink(path)


def _save_figure_safe(fig: plt.Figure, figures_dir: str, filename: str):
  _safe_unlink(os.path.join(figures_dir, filename))
  viz_utils.save_figure(fig, figures_dir, filename)


def _load_inform_vulnerability(world_gdf: gpd.GeoDataFrame) -> pd.Series:
  """Maps harmonized country names to INFORM 2026 Socioeconomic Vulnerability."""
  iso_fix = {'France': 'FRA', 'Norway': 'NOR'}
  country_to_iso = {}
  for _, row in world_gdf.iterrows():
    cname = row['map_name']
    iso = iso_fix.get(cname, str(row.get('iso_a3', '')).strip())
    country_to_iso[cname] = iso

  with gfile.Open(file_paths.RISK_FILE, 'r') as f:
    df_inform = pd.read_csv(io.StringIO(f.read()), skiprows=[0, 2])

  df_inform['ISO3'] = df_inform['ISO3'].astype(str).str.strip()
  df_inform['COUNTRY_CLEAN'] = (
      df_inform['COUNTRY']
      .astype(str)
      .str.strip()
      .replace(file_paths.country_name_map)
  )
  df_inform['VULN_NUM'] = pd.to_numeric(
      df_inform['VULNERABILITY'], errors='coerce'
  )
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

  all_countries = sorted(
      set(country_to_iso.keys()).union(set(name_vuln.index))
  )
  out = {}
  for c in all_countries:
    iso = country_to_iso.get(c)
    val = iso_vuln.get(iso, np.nan) if iso else np.nan
    if pd.isna(val):
      val = name_vuln.get(c, np.nan)
    out[c] = val
  return pd.Series(out, name='Vulnerability')


def main(argv):
  del argv
  t_start = time.time()
  sns.set_theme(style='whitegrid', palette='colorblind')
  plt.rcParams['figure.dpi'] = 150
  plt.rcParams['font.family'] = 'sans-serif'

  file_paths.init('5.0-year-2.0-window-recall')
  for d in [file_paths.STATS_DIR, file_paths.FIGURES_DIR]:
    if not gfile.Exists(d):
      gfile.MakeDirs(d)

  # 1. Load v2 joint country N-gauge evaluations, single-candidate evaluations, and global N-gauge sets
  nat_csv = os.path.join(
      file_paths.STATS_DIR, 'country_joint_n_gauge_exposure_reduction_v2.csv'
  )
  single_csv = os.path.join(
      file_paths.STATS_DIR, 'single_candidate_evaluations_v2.csv'
  )
  sets_csv = os.path.join(
      file_paths.STATS_DIR, 'n_gauge_sets_evaluation_summary_v2.csv'
  )
  blockers_csv = os.path.join(
      file_paths.STATS_DIR, 'country_exposure_and_blockers_v2.csv'
  )
  interventions_json = os.path.join(
      file_paths.STATS_DIR, 'five_factor_exposure_interventions_v2.json'
  )

  df_nat = io_utils.load_csv(nat_csv)
  df_single = io_utils.load_csv(single_csv)
  df_sets = io_utils.load_csv(sets_csv)
  df_blockers = io_utils.load_csv(blockers_csv)
  with gfile.Open(interventions_json, 'rt') as f:
    v2_interventions = json.load(f)

  f2_ceiling = float(
      v2_interventions['interventions'][
          'Factor 2 (M_ungauged Primary): Universal Local & Network Gauge Coverage (100% Coverage)'
      ]['rp_clipped_delta_global']
  )
  f2_mgauged = float(
      v2_interventions['interventions'][
          'Confirmation Check ONLY (M_gauged): Independent In-Sample Local Gauging'
      ]['rp_clipped_delta_global']
  )
  global_unwarned = float(
      v2_interventions['baseline_summary'][
          'Global Total (All 964,084 Sub-Basins)'
      ]['unwarned_ungauged']
  )

  df_nat['Country'] = df_nat['Country'].replace(file_paths.country_name_map)
  df_single['country_name'] = df_single['country_name'].replace(
      file_paths.country_name_map
  )
  df_blockers['country_name'] = df_blockers['country_name'].replace(
      file_paths.country_name_map
  )
  unwarned_map = df_blockers.set_index('country_name')['unwarned_ungauged']
  df_nat['Unwarned Exposure (people/yr)'] = (
      df_nat['Country'].map(unwarned_map).fillna(0.0)
  )

  N_gauges = [1, 5, 10, 20]
  for N in N_gauges:
    df_nat[f'N={N} Unwarned Reduced (%)'] = np.where(
        df_nat['Unwarned Exposure (people/yr)'] > 0,
        np.minimum(
            100.0,
            100.0
            * df_nat[f'N={N} Exposure Reduced (people/yr)']
            / df_nat['Unwarned Exposure (people/yr)'],
        ),
        0.0,
    )

  # Export clean country placement table for paper artifacts & stats
  export_cols = [
      'Country',
      'Baseline Exposure (people/yr)',
      'Unwarned Exposure (people/yr)',
      'Max_Evaluated_Candidates',
      'N=1 Exposure Reduced (people/yr)',
      'N=1 Exposure Reduced (%)',
      'N=1 Unwarned Reduced (%)',
      'N=5 Exposure Reduced (people/yr)',
      'N=5 Exposure Reduced (%)',
      'N=5 Unwarned Reduced (%)',
      'N=10 Exposure Reduced (people/yr)',
      'N=10 Exposure Reduced (%)',
      'N=10 Unwarned Reduced (%)',
      'N=20 Exposure Reduced (people/yr)',
      'N=20 Exposure Reduced (%)',
      'N=20 Unwarned Reduced (%)',
      'N=20 Exposure Reduced MGauged Confirm (people/yr)',
      'CAP Warning System Tier',
      'Operational System Status',
  ]
  df_export = df_nat[export_cols].sort_values(
      'N=20 Exposure Reduced (people/yr)', ascending=False
  )
  for out_dir in [file_paths.STATS_DIR, file_paths.FIGURES_DIR]:
    out_path = os.path.join(
        out_dir, 'country_gauge_placement_exposure_reduction.csv'
    )
    _safe_unlink(out_path)
    with gfile.Open(out_path, 'wt') as f:
      df_export.to_csv(f, index=False)

  print('\n=== JOINT NATIONAL N-GAUGE EXPOSURE REDUCTION (TOP 20 BY ABSOLUTE GAIN AT N=20) ===')
  print(df_export.head(20).to_string(index=False))

  print('\n=== MEDIAN NATIONAL % EXPOSURE REDUCTION BY WARNING SYSTEM STATUS ===')
  print(
      df_export.groupby('Operational System Status')[
          [
              'N=1 Exposure Reduced (%)',
              'N=5 Exposure Reduced (%)',
              'N=10 Exposure Reduced (%)',
              'N=20 Exposure Reduced (%)',
              'N=20 Unwarned Reduced (%)',
          ]
      ]
      .median()
      .to_string()
  )

  # 2. Load Natural Earth world shapefile and harmonize country names
  with gfile.Open(file_paths.NATURAL_EARTH_PATH, 'rb') as f:
    world = gpd.read_file(f)
  world = world[world['name'] != 'Antarctica'].copy()
  world['map_name'] = world['name'].replace(file_paths.country_name_map)
  vuln_series = _load_inform_vulnerability(world)

  # ============================================================================
  # EXTENDED DATA FIGURE 5: 4-Panel National % Reduction Maps + Top-15 Vulnerable Curves
  # ============================================================================
  print(f"\n[{time.strftime('%H:%M:%S')}] Rendering Extended Data Figure 5...")
  fig = plt.figure(figsize=(26.5, 13.5))
  gs = gridspec.GridSpec(
      2, 3, width_ratios=[1.15, 1.15, 0.98], wspace=0.18, hspace=0.26
  )

  axes_maps = [
      fig.add_subplot(gs[0, 0]),
      fig.add_subplot(gs[0, 1]),
      fig.add_subplot(gs[1, 0]),
      fig.add_subplot(gs[1, 1]),
  ]
  panel_letters = ['(a)', '(b)', '(c)', '(d)']

  nat_indexed = df_nat.set_index('Country')
  # Shared robust upper bound across N=1..20 so color progression across N is immediately visible
  vmax_shared = max(
      float(np.nanpercentile(df_nat['N=20 Exposure Reduced (%)'], 95)), 2.0
  )

  for i, N in enumerate(N_gauges):
    ax = axes_maps[i]
    plot_df = world.copy()
    col_pct = f'N={N} Exposure Reduced (%)'
    plot_df['val_pct'] = plot_df['map_name'].map(nat_indexed[col_pct])

    world.plot(
        ax=ax, facecolor='#ECEFF1', edgecolor='#B0BEC5', linewidth=0.35
    )
    valid_plot = plot_df.dropna(subset=['val_pct'])
    valid_plot.plot(
        column='val_pct',
        cmap='viridis',
        ax=ax,
        vmin=0.0,
        vmax=vmax_shared,
        legend=True,
        legend_kwds={
            'label': f'% Baseline Exposure Reduced (N = {N}; capped at {vmax_shared:.1f}%)',
            'orientation': 'horizontal',
            'shrink': 0.72,
            'pad': 0.03,
            'extend': 'max',
        },
        edgecolor='#37474F',
        linewidth=0.2,
    )
    med_pct = float(df_nat[col_pct].median())
    tot_red = float(df_nat[f'N={N} Exposure Reduced (people/yr)'].sum())
    ax.set_title(
        f'{panel_letters[i]}  N = {N} New Gauges per Country\n'
        f'(Median = {med_pct:.2f}%, Sum = {tot_red/1e6:.2f}M people/yr)',
        fontsize=12.0,
        fontweight='bold',
        loc='left',
    )
    ax.set_xlim([-180, 180])
    ax.set_ylim([-58, 84])
    ax.set_axis_off()

  # Right column: Top 15 Most Socioeconomically Vulnerable Countries (INFORM)
  ax_lines_abs = fig.add_subplot(gs[0, 2])
  ax_lines_pct = fig.add_subplot(gs[1, 2])

  df_vuln_rank = df_nat.set_index('Country').copy()
  df_vuln_rank['Vulnerability'] = vuln_series.reindex(df_vuln_rank.index)
  df_vuln_rank = df_vuln_rank[
      df_vuln_rank['Baseline Exposure (people/yr)'] >= 1000.0
  ].dropna(subset=['Vulnerability'])
  top_15_countries = (
      df_vuln_rank.sort_values('Vulnerability', ascending=False)
      .head(15)
      .index.tolist()
  )

  cmap_15 = sns.color_palette('husl', 15)
  ls_cycle = ['-', '--', '-.']
  marker_cycle = ['o', 's', '^']

  for i, country in enumerate(top_15_countries):
    row_c = df_vuln_rank.loc[country]
    max_cands = int(row_c['Max_Evaluated_Candidates'])
    y_raw, y_pct = [], []
    for N in N_gauges:
      if N > max_cands and N > 1:
        y_raw.append(np.nan)
        y_pct.append(np.nan)
      else:
        y_raw.append(float(row_c[f'N={N} Exposure Reduced (people/yr)']))
        y_pct.append(float(row_c[f'N={N} Exposure Reduced (%)']))

    ls = ls_cycle[i % 3]
    mk = marker_cycle[i // 5]
    vuln_score = float(row_c['Vulnerability'])
    lbl = f'{country} (INFORM {vuln_score:.1f})'
    ax_lines_abs.plot(
        N_gauges,
        y_raw,
        marker=mk,
        linestyle=ls,
        color=cmap_15[i],
        linewidth=1.9,
        markersize=5.5,
        label=lbl,
    )
    ax_lines_pct.plot(
        N_gauges,
        y_pct,
        marker=mk,
        linestyle=ls,
        color=cmap_15[i],
        linewidth=1.9,
        markersize=5.5,
        label=lbl,
    )

  ax_lines_abs.set_xticks(N_gauges)
  ax_lines_abs.set_yscale('symlog', linthresh=100.0)
  ax_lines_abs.set_ylim(bottom=0.0)
  ax_lines_abs.set_ylabel(
      r'Exposure Reduced at $\alpha = 1.0$ (people/yr)', fontsize=11
  )
  ax_lines_abs.set_title(
      '(e)  15 Most Vulnerable Countries (Absolute Gain)',
      fontsize=12.5,
      fontweight='bold',
      loc='left',
  )
  ax_lines_abs.grid(True, which='both', ls='--', alpha=0.35)
  sns.despine(ax=ax_lines_abs)

  ax_lines_pct.set_xticks(N_gauges)
  ax_lines_pct.set_ylim(bottom=0.0)
  ax_lines_pct.set_xlabel('New Gauges Installed per Country (N)', fontsize=11.5)
  ax_lines_pct.set_ylabel('% of National Flood Exposure Reduced', fontsize=11)
  ax_lines_pct.set_title(
      '(f)  15 Most Vulnerable Countries (% of Baseline Exposure)',
      fontsize=12.5,
      fontweight='bold',
      loc='left',
  )
  ax_lines_pct.grid(True, which='both', ls='--', alpha=0.35)
  ax_lines_pct.legend(
      loc='upper left',
      bbox_to_anchor=(1.02, 1.35),
      fontsize=8.8,
      frameon=True,
      title='Country (INFORM Vulnerability)',
      title_fontsize=9.2,
  )
  sns.despine(ax=ax_lines_pct)

  fig.suptitle(
      'National River-Network Exposure Reduction from Installing N = 1 to 20 Streamflow Gauges per Country',
      fontsize=16,
      fontweight='bold',
      y=0.98,
  )
  _save_figure_safe(fig, file_paths.FIGURES_DIR, 'Extended_Data_Figure_5.png')
  plt.close(fig)

  # ============================================================================
  # EXTENDED DATA FIGURE 6: Global N-Gauge Strategies + Mali Top-5 Topology
  # ============================================================================
  print(f"\n[{time.strftime('%H:%M:%S')}] Preparing Mali case study & Extended Data Figure 6...")
  target_country = 'Mali'
  N_target = 5
  debug_dir = os.path.join(
      file_paths.V2_RESULTS_DIR,
      'scratch',
      f"debug_{target_country.lower().replace(' ', '_')}",
  )
  gfile.MakeDirs(debug_dir)

  df_mali = (
      df_single[df_single['country_name'] == target_country]
      .sort_values(
          by=['within_country_risk_2_exp', 'global_risk_2_exp'], ascending=False
      )
      .reset_index(drop=True)
  )
  target_hybas = df_mali.head(N_target)['candidate_hybas_id'].tolist()
  target_gains = dict(
      zip(
          df_mali.head(N_target)['candidate_hybas_id'],
          df_mali.head(N_target)['within_country_risk_2_exp'],
      )
  )
  print(f'Top {N_target} M_ungauged Gauge Candidates in {target_country}: {target_hybas}')

  cache_outlet_path = os.path.join(debug_dir, 'target_outlets_v2.geojson')
  cache_drainage_path = os.path.join(debug_dir, 'target_drainages_v2.geojson')

  need_build_mali = True
  if gfile.Exists(cache_outlet_path) and gfile.Exists(cache_drainage_path):
    target_outlets_gdf = io_utils.load_gdf_from_cns(cache_outlet_path)
    target_drainages_gdf = io_utils.load_gdf_from_cns(cache_drainage_path)
    if target_outlets_gdf.crs != world.crs:
      target_outlets_gdf = target_outlets_gdf.to_crs(world.crs)
    if target_drainages_gdf.crs != world.crs:
      target_drainages_gdf = target_drainages_gdf.to_crs(world.crs)
    if 'unique_gauge_id' in target_outlets_gdf.columns:
      target_outlets_gdf = target_outlets_gdf.set_index('unique_gauge_id', drop=False)
    if 'unique_gauge_id' in target_drainages_gdf.columns:
      target_drainages_gdf = target_drainages_gdf.set_index('unique_gauge_id', drop=False)
    if set(target_hybas).issubset(set(target_outlets_gdf.index)):
      need_build_mali = False

  if need_build_mali:
    print(f"[{time.strftime('%H:%M:%S')}] Building Mali Level-12 outlet & upstream drainage geometries...")
    af_gpq_path = os.path.join(
        file_paths.THIRD_PARTY_DIR, 'geoparquet/af/hybas_af_lev12_v1c.geoparquet'
    )
    if os.path.exists(af_gpq_path):
      hybas_gdf = gpd.read_parquet(af_gpq_path)
    else:
      af_sub_path = os.path.join(
          file_paths.HYBAS_SUBBASINS_DIR, 'hybas_af_lev12_v1c.geojson'
      )
      if not gfile.Exists(af_sub_path):
        af_sub_path = os.path.join(
            file_paths.HYBAS_POLYGONS_DIR, 'hybas_af_lev12_v1c.geojson'
        )
      with gfile.Open(af_sub_path, 'rb') as f:
        hybas_gdf = gpd.read_file(f)

    if 'HYBAS_ID' in hybas_gdf.columns:
      hybas_gdf['HYBAS_ID'] = (
          hybas_gdf['HYBAS_ID']
          .astype(str)
          .str.replace(r'\.0$', '', regex=True)
          .str.replace('hybas_', '', regex=False)
      )
      hybas_gdf.index = ['hybas_' + idx for idx in hybas_gdf['HYBAS_ID']]
    elif 'unique_gauge_id' in hybas_gdf.columns:
      hybas_gdf.index = [
          idx if str(idx).startswith('hybas_') else 'hybas_' + str(idx)
          for idx in hybas_gdf['unique_gauge_id']
      ]
    if hybas_gdf.crs != world.crs:
      hybas_gdf = hybas_gdf.to_crs(world.crs)

    target_outlets_gdf = hybas_gdf.loc[hybas_gdf.index.isin(target_hybas)].copy()
    target_outlets_gdf['unique_gauge_id'] = target_outlets_gdf.index

    adj = defaultdict(list)
    if 'NEXT_DOWN' in hybas_gdf.columns:
      for child, nd in zip(hybas_gdf.index, hybas_gdf['NEXT_DOWN']):
        parent = str(nd).replace('.0', '')
        if parent and parent not in ('0', 'nan'):
          parent_id = parent if parent.startswith('hybas_') else 'hybas_' + parent
          adj[parent_id].append(child)

    def _get_upstream_set(node_id):
      upstreams = [node_id]
      queue = [node_id]
      while queue:
        curr = queue.pop(0)
        for c in adj.get(curr, []):
          upstreams.append(c)
          queue.append(c)
      return upstreams

    drainage_rows = []
    for th_id in target_hybas:
      if th_id in hybas_gdf.index:
        up_ids = _get_upstream_set(th_id)
        poly = hybas_gdf.loc[hybas_gdf.index.isin(up_ids)].geometry.unary_union
        drainage_rows.append({'unique_gauge_id': th_id, 'geometry': poly})
    target_drainages_gdf = gpd.GeoDataFrame(
        drainage_rows, geometry='geometry', crs=world.crs
    ).set_index('unique_gauge_id', drop=False)

    _safe_unlink(cache_outlet_path)
    with gfile.Open(cache_outlet_path, 'w') as f:
      f.write(target_outlets_gdf.reset_index(drop=True).to_json())
    _safe_unlink(cache_drainage_path)
    with gfile.Open(cache_drainage_path, 'w') as f:
      f.write(target_drainages_gdf.reset_index(drop=True).to_json())

  fig_ed6, (ax_bar, ax_mali) = plt.subplots(
      1, 2, figsize=(22, 8.8), gridspec_kw={'width_ratios': [1.08, 1.05]}
  )

  # Panel a: Grouped bar chart of Bucket 2A, Bucket 2B, and Bucket 1 across N=10, 50, 100, 1000
  bucket_order = [
      'Bucket 2A (Global Exposure)',
      'Bucket 2B (Exposure-Weighted OOD)',
      'Bucket 1 (Representativeness)',
  ]
  bucket_labels = {
      'Bucket 2A (Global Exposure)': 'Exposure-Targeted (Bucket 2A)',
      'Bucket 2B (Exposure-Weighted OOD)': 'Exposure x OOD Targeted (Bucket 2B)',
      'Bucket 1 (Representativeness)': 'Unweighted OOD Representativeness (Bucket 1)',
  }
  bucket_colors = {
      'Bucket 2A (Global Exposure)': '#0072B2',
      'Bucket 2B (Exposure-Weighted OOD)': '#009E73',
      'Bucket 1 (Representativeness)': '#E69F00',
  }

  df_sets_plot = df_sets[df_sets['Bucket'].isin(bucket_order)].copy()
  n_vals = sorted(df_sets_plot['N'].unique())
  x_pos = np.arange(len(n_vals))
  width = 0.25

  for b_idx, b_key in enumerate(bucket_order):
    sub_b = (
        df_sets_plot[df_sets_plot['Bucket'] == b_key]
        .set_index('N')
        .reindex(n_vals)
    )
    y_riv = sub_b['Global_Risk_Reduced'].fillna(0.0).values
    y_ood = sub_b['Global_Risk_Reduced_With_OOD_Transfer'].fillna(0.0).values
    offsets = x_pos + (b_idx - 1) * width
    ax_bar.bar(
        offsets,
        np.maximum(y_riv, 1.0),
        width=width,
        color=bucket_colors[b_key],
        edgecolor='black',
        linewidth=0.7,
        label=bucket_labels[b_key],
    )
    # Plot diamonds showing total gain when including global 20-NN OOD transfer
    ax_bar.scatter(
        offsets,
        np.maximum(y_ood, 1.0),
        marker='D',
        s=46,
        facecolor='white',
        edgecolor=bucket_colors[b_key],
        linewidth=1.8,
        zorder=5,
        label='+ Global OOD Transfer' if b_idx == 0 else None,
    )

  # Reference ceiling lines from unified 5-factor decomposition
  ax_bar.axhline(
      f2_ceiling,
      color='#332288',
      linestyle='--',
      linewidth=1.5,
      label=f'Factor 2 Universal Gauge Coverage Ceiling ({f2_ceiling/1e6:.2f}M)',
  )
  ax_bar.axhline(
      f2_mgauged,
      color='#7F7F7F',
      linestyle=':',
      linewidth=1.5,
      label=f'In-Sample Gauged Confirmation Ceiling ({f2_mgauged/1e6:.2f}M)',
  )

  ax_bar.set_xticks(x_pos)
  ax_bar.set_xticklabels([f'N = {n:,}' for n in n_vals], fontsize=11)
  ax_bar.set_yscale('log')
  ax_bar.set_ylim(bottom=1e5, top=4.2e7)
  ax_bar.set_ylabel(
      r'Global Unwarned Exposure Reduced at $\alpha = 1.0$ (people/yr)',
      fontsize=11.5,
  )
  ax_bar.set_xlabel('Total New Streamflow Gauges Installed Globally (N)', fontsize=11.5)
  ax_bar.set_title(
      '(a)  Global Siting Strategies vs. Universal Gauge Coverage Ceiling',
      fontsize=13,
      fontweight='bold',
      loc='left',
  )
  ax_bar.grid(True, which='both', ls='--', alpha=0.35)

  secax = ax_bar.secondary_yaxis(
      'right',
      functions=(
          lambda x: (x / global_unwarned) * 100.0,
          lambda x: (x / 100.0) * global_unwarned,
      ),
  )
  secax.set_ylabel(
      f'% of Global Unwarned Exposure ({global_unwarned/1e6:.1f}M people/yr)',
      fontsize=11,
  )
  ax_bar.legend(loc='upper left', fontsize=8.8, frameon=True, framealpha=0.92)

  # Panel b: Mali Top-5 Gauge Placement Topology
  world.plot(ax=ax_mali, color='#F5F7F8', edgecolor='#CFD8DC', linewidth=0.6)
  country_border = world[world['map_name'] == target_country]
  country_border.plot(
      ax=ax_mali,
      color='#ECEFF1',
      edgecolor='#263238',
      linewidth=2.0,
  )

  mali_palette = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']
  handles = []
  for i, hybas_id in enumerate(target_hybas):
    if hybas_id not in target_outlets_gdf.index:
      continue
    c_color = mali_palette[i % len(mali_palette)]
    if hybas_id in target_drainages_gdf.index:
      target_drainages_gdf.loc[[hybas_id]].plot(
          ax=ax_mali, color=c_color, alpha=0.30, edgecolor=c_color, linewidth=0.9
      )
    target_outlets_gdf.loc[[hybas_id]].plot(
        ax=ax_mali, color=c_color, edgecolor='black', linewidth=1.4, zorder=4
    )
    rep_pt = target_outlets_gdf.loc[[hybas_id]].geometry.representative_point().iloc[0]
    ax_mali.scatter(
        [rep_pt.x],
        [rep_pt.y],
        s=95,
        marker='o',
        facecolor=c_color,
        edgecolor='black',
        linewidth=1.5,
        zorder=6,
    )
    gain_val = target_gains.get(hybas_id, 0.0)
    handles.append(
        mpatches.Patch(
            facecolor=c_color,
            alpha=0.55,
            edgecolor='black',
            label=f'Rank {i+1} ({hybas_id}): +{gain_val:,.0f} people/yr',
        )
    )

  bounds = country_border.total_bounds
  ax_mali.set_xlim(bounds[0] - 1.2, bounds[2] + 1.2)
  ax_mali.set_ylim(bounds[1] - 1.0, bounds[3] + 2.8)
  mali_joint_5 = float(
      nat_indexed.loc[target_country, 'N=5 Exposure Reduced (people/yr)']
      if target_country in nat_indexed.index
      else 0.0
  )
  mali_pct_5 = float(
      nat_indexed.loc[target_country, 'N=5 Exposure Reduced (%)']
      if target_country in nat_indexed.index
      else 0.0
  )
  ax_mali.legend(
      handles=handles,
      loc='upper left',
      title=f'Top 5 Mali Gauges (Joint N=5: +{mali_joint_5:,.0f} people/yr, {mali_pct_5:.1f}%)',
      fontsize=9.2,
      title_fontsize=9.5,
      frameon=True,
      framealpha=0.94,
  )
  ax_mali.set_title(
      '(b)  Optimal River-Network Gauge Siting in Mali (Outlets & Upstream Catchments)',
      fontsize=13,
      fontweight='bold',
      loc='left',
  )
  ax_mali.axis('off')

  plt.tight_layout()
  _save_figure_safe(
      fig_ed6, file_paths.FIGURES_DIR, 'Extended_Data_Figure_6.png'
  )
  plt.close(fig_ed6)
  print(
      f"[{time.strftime('%H:%M:%S')}] Completed 11_analyze_gauge_placement_results in {time.time() - t_start:.1f}s."
  )


if __name__ == '__main__':
  app.run(main)
