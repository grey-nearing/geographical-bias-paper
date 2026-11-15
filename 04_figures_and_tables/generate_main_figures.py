#!/usr/bin/env python3
"""Generate main-text publication figures (Figs. 1–4, 6; optional Fig. 5 if Fathom present).

Produces:
  - Fig_1_global_overview.png: 4-panel global gauge catalog (121,819 stations),
    never-passes-a-gauge map, continuous L12 fractional ungaugedness, and
    hydroclimatic dissimilarity percentile (rasterized directly from BasinATLAS v10).
  - Fig_2_polygon_size_dependence.png: HydroSHEDS Levels 1–12 dotted scale curve
    plus six fine-scale global hydrofabric configurations and fine-scale inset.
  - Fig_3_regional_aridity_dissimilarity.png: 2x2 continental land/population
    ungaugedness, UNEP aridity zones, and cutoff-free dissimilarity percentiles.
  - Fig_4_cap_gap_and_inform.png: 2x2 national CAP-alerting coverage map (60 vs 114),
    country ungaugedness boxplots, and INFORM 2026 Vulnerability / Coping correlations.
  - Fig_5_exposure_counterfactuals.png (OPTIONAL — skipped cleanly when licensed
    Fathom v3 exposure counterfactual outputs are not present locally).
  - Fig_6_exact_group_shapley.png: Exact 2^7=128 Group Shapley attribution across
    7 predictor categories and 12 hydrological metrics (N=1,000 global medoids).
"""

from __future__ import annotations

import argparse
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.stats import mannwhitneyu, spearmanr
import seaborn as sns

from utils import cap_utils, file_paths

warnings.filterwarnings('ignore')

COL_T1 = '#0072B2'   # Standardized open (34,577) / Established national alerting
COL_T2 = '#009E73'   # Confirmed public discharge (93,827)
COL_T3 = '#D55E00'   # All discoverable (121,819) / CAP-gap country

NATURE_W = 183.0 / 25.4  # 7.2047 in
SAVE_DPI = 350

R2_DIR = CANON_DIR / 'round2'
TIER_DIR = R2_DIR / 'tier_features'
CF_DIR = R2_DIR / 'counterfactuals'
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
  plt.rcParams.update({
      'font.family': 'sans-serif',
      'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
      'font.size': font_size,
      'axes.edgecolor': '#334155',
      'axes.linewidth': 0.65,
      'xtick.color': '#1E293B',
      'ytick.color': '#1E293B',
      'text.color': '#0F172A',
      'figure.facecolor': 'white',
      'axes.facecolor': 'white',
      'savefig.facecolor': 'white',
  })


def norm_hybas_id(s: pd.Series | pd.Index) -> np.ndarray:
  s2 = pd.Series(s).astype(str).str.replace(r'\.0$', '', regex=True)
  return np.where(s2.str.startswith('hybas_'), s2, 'hybas_' + s2)


def load_world() -> gpd.GeoDataFrame:
  w = gpd.read_file(file_paths.NATURAL_EARTH_PATH)
  w = w.dropna(subset=[w.geometry.name])
  w = w[w.is_valid & (~w.is_empty)].copy()
  w['map_name'] = w['name'].replace(file_paths.country_name_map)
  iso_fix = {'France': 'FRA', 'Norway': 'NOR'}
  w['iso_clean'] = [
      iso_fix.get(m, str(i).strip())
      for m, i in zip(w['map_name'], w.get('iso_a3', [''] * len(w)))
  ]
  return w


def load_gauges_tiered() -> pd.DataFrame:
  staged = HYDRO_DATA_DIR / 'gauges' / 'global_gauges.parquet'
  if staged.exists():
    return pd.read_parquet(staged)
  g = pd.read_parquet(CANON_DIR / 'attributes' / 'global_gauges_deduplicated.parquet')
  g['in_tier_1'] = g['all_source_datasets'].str.contains('caravan|estream', case=False, na=False)
  g['in_tier_2'] = (
      g['best_data_access_policy'].isin({'open_public_api_or_bulk', 'open_academic_compilation'})
      & g['best_observation_type'].isin({'daily_discharge', 'monthly_discharge'})
  )
  g['in_tier_3'] = True
  return g


def load_non_fathom_basin_meta(all_ids: pd.Index) -> pd.DataFrame:
  """Loads public 1,034,083 basin metadata (country, CAP, sub_area, WorldPop population)."""
  df = pd.DataFrame(index=all_ids)

  df_map = pd.read_csv(file_paths.HYBAS_MAPPING_PATH)
  idc = 'hybas_id' if 'hybas_id' in df_map.columns else df_map.columns[0]
  df_map['hid'] = norm_hybas_id(df_map[idc])
  df_map['country_name_clean'] = df_map['country_name'].replace(file_paths.country_name_map)
  df_map = df_map.drop_duplicates('hid').set_index('hid')
  df['country_name'] = df_map['country_name_clean'].reindex(all_ids)
  df['has_system'] = (
      df['country_name'].map(cap_utils.has_effective_flood_warning_system).fillna(False).astype(bool)
  )
  df['cap_tier'] = df['country_name'].map(cap_utils.assign_cap_tier).fillna(cap_utils.TIER_4_LABEL)

  topo_pq = CANON_DIR / 'attributes' / 'hybas_l12_topology_and_runoff.parquet'
  if topo_pq.exists():
    df_att = pd.read_parquet(topo_pq, columns=['HYBAS_ID', 'SUB_AREA'])
    df_att.index = norm_hybas_id(df_att['HYBAS_ID'])
    df_att = df_att[~df_att.index.duplicated(keep='first')].reindex(all_ids)
    df['sub_area'] = pd.to_numeric(df_att['SUB_AREA'], errors='coerce').fillna(0.0)
  else:
    df_att = pd.read_csv(
        file_paths.HYBAS_ATTRIBUTES_PATH,
        usecols=['Unnamed: 0', 'HYDRO_ATLAS:SUB_AREA'],
        index_col='Unnamed: 0',
    )
    df_att.index = norm_hybas_id(df_att.index)
    df_att = df_att[~df_att.index.duplicated(keep='first')].reindex(all_ids)
    df['sub_area'] = pd.to_numeric(df_att['HYDRO_ATLAS:SUB_AREA'], errors='coerce').fillna(0.0)

  pop_candidates = [
      CANON_DIR / 'attributes' / 'hydrosheds_l12_worldpop_population.parquet',
      HYDRO_DATA_DIR / 'exposure' / 'hydrosheds_l12_worldpop_population.parquet',
  ]
  pop_pq = next((p for p in pop_candidates if p.exists()), None)
  if pop_pq is None:
    raise FileNotFoundError(f'Missing public WorldPop population parquet; checked {pop_candidates}')
  df_pop = pd.read_parquet(pop_pq)
  id_col = 'HYBAS_ID' if 'HYBAS_ID' in df_pop.columns else ('hybas_id' if 'hybas_id' in df_pop.columns else None)
  if id_col is not None:
    df_pop.index = norm_hybas_id(df_pop[id_col])
  else:
    df_pop.index = norm_hybas_id(df_pop.index)
  df_pop = df_pop[~df_pop.index.duplicated(keep='first')].reindex(all_ids)
  df['total_pop'] = pd.to_numeric(df_pop['total_population'], errors='coerce').fillna(0.0).astype(np.float64)
  return df


def filter_canonical_2d(df: pd.DataFrame) -> pd.DataFrame:
  return df[
      ((df['hydrofabric'] == 'grit_v06_bifurcating_dag') & (df['snapping_mode'] == 'dag_width_weighted') & (df['ratio_basis'] == 'area_dag'))
      | ((df['hydrofabric'] != 'grit_v06_bifurcating_dag') & (df['snapping_mode'] == 'outlet') & (df['ratio_basis'] == 'area'))
  ]


def w_quantile(vals: np.ndarray, wts: np.ndarray, q: float) -> float:
  ord_i = np.argsort(vals)
  v, wt = vals[ord_i], wts[ord_i]
  cdf = np.cumsum(wt) / wt.sum()
  return float(np.interp(q, cdf, v))


def gauged_ref_pct(df: pd.DataFrame, valid_idx: pd.Index) -> pd.Series:
  sub = df.loc[valid_idx]
  g = df[df['U_net_tree_phys'] == 0]
  pcts = []
  for col in ['anomaly_score_if', 'anomaly_score_ocsvm', 'anomaly_score_lof']:
    ref = np.sort(-g[col].values)
    pos = np.searchsorted(ref, -sub[col].values, side='right') / len(ref) * 100.0
    pcts.append(pos)
  return pd.Series(np.mean(pcts, axis=0), index=valid_idx)


# ==============================================================================
# FIGURE 1: 4-Panel Global Overview (Gauges, Never-Passes, Fractional, Dissim)
# ==============================================================================
def build_fig1(world_all: gpd.GeoDataFrame, out_dir: Path) -> Path:
  print('[Fig 1] Generating 4-panel global overview map...', flush=True)
  _set_pub_rc(8.5)
  world = world_all[world_all['name'] != 'Antarctica'].copy()
  gauges = load_gauges_tiered()
  t4 = pd.read_parquet(TIER_DIR / 'tier4_all_publicly_knowable_121819.parquet')
  t4.index = norm_hybas_id(t4.index)

  r_if = (-t4['anomaly_score_if']).rank(pct=True) * 100.0
  r_oc = (-t4['anomaly_score_ocsvm']).rank(pct=True) * 100.0
  r_lo = (-t4['anomaly_score_lof']).rank(pct=True) * 100.0
  t4['dissim_pct'] = (r_if + r_oc + r_lo) / 3.0
  t4['never_flag'] = (t4['intergauge_reach_area_km2'] < 0).astype(np.float32)
  t4['frac_pct'] = (t4['U_net_tree_phys'] * 100.0).astype(np.float32)

  xmin, xmax, ymin, ymax = -180.0, 180.0, -58.0, 84.0
  width, height = 3600, 1420
  transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

  never_grid = np.full((height, width), np.nan, dtype=np.float32)
  frac_grid = np.full((height, width), np.nan, dtype=np.float32)
  dissim_grid = np.full((height, width), np.nan, dtype=np.float32)

  lookup_never = t4['never_flag'].to_dict()
  lookup_frac = t4['frac_pct'].to_dict()
  lookup_dissim = t4['dissim_pct'].to_dict()

  def _rasterize_chunk(ids_arr, geoms_arr) -> int:
    s_never, s_frac, s_dissim = [], [], []
    for hid, geom in zip(ids_arr, geoms_arr):
      if geom is None or geom.is_empty:
        continue
      v_f = lookup_frac.get(hid)
      if v_f is None or np.isnan(v_f):
        continue
      s_never.append((geom, float(lookup_never[hid])))
      s_frac.append((geom, float(v_f)))
      s_dissim.append((geom, float(lookup_dissim[hid])))
    if s_never:
      r_n = rasterize(s_never, out_shape=(height, width), transform=transform, fill=np.nan, dtype='float32')
      m = ~np.isnan(r_n)
      never_grid[m] = r_n[m]

      r_f = rasterize(s_frac, out_shape=(height, width), transform=transform, fill=np.nan, dtype='float32')
      m = ~np.isnan(r_f)
      frac_grid[m] = r_f[m]

      r_d = rasterize(s_dissim, out_shape=(height, width), transform=transform, fill=np.nan, dtype='float32')
      m = ~np.isnan(r_d)
      dissim_grid[m] = r_d[m]
    return len(s_frac)

  gdb_path = HYDRO_DATA_DIR / 'raw' / 'hydrosheds' / 'BasinATLAS_v10.gdb'
  geopq_root = CANON_DIR / 'geoparquet'
  matched_total = 0

  if gdb_path.exists():
    info = pyogrio.read_info(gdb_path, layer='BasinATLAS_v10_lev12')
    n_feat = int(info['features'])
    chunk_size = 200_000
    print(f'  Rasterizing {n_feat:,} polygons from BasinATLAS_v10.gdb (chunks of {chunk_size:,})...', flush=True)
    for start in range(0, n_feat, chunk_size):
      gdf = pyogrio.read_dataframe(
          gdb_path,
          layer='BasinATLAS_v10_lev12',
          columns=['HYBAS_ID'],
          skip_features=start,
          max_features=chunk_size,
      )
      ids = norm_hybas_id(gdf['HYBAS_ID'])
      n_m = _rasterize_chunk(ids, gdf.geometry.values)
      matched_total += n_m
      print(f'    chunk {start:,}–{min(start + chunk_size, n_feat):,}: {n_m:,} rasterized', flush=True)
  elif geopq_root.exists():
    for reg in ['af', 'ar', 'as', 'au', 'eu', 'gr', 'na', 'sa', 'si']:
      pq = geopq_root / reg / f'hybas_{reg}_lev12_v1c.geoparquet'
      if not pq.exists():
        continue
      gdf = gpd.read_parquet(pq, columns=['HYBAS_ID', 'geometry'])
      ids = norm_hybas_id(gdf['HYBAS_ID'])
      n_m = _rasterize_chunk(ids, gdf.geometry.values)
      matched_total += n_m
  else:
    raise FileNotFoundError(
        f'Need BasinATLAS_v10.gdb at {gdb_path} (run 01_download_hydrosheds.py) '
        f'or geoparquet archives at {geopq_root} to rasterize Fig. 1.'
    )

  print(f'  Total polygons rasterized: {matched_total:,}', flush=True)

  cmap_frac = mcolors.LinearSegmentedColormap.from_list(
      'ungauged_r2',
      [
          (0.00, '#1D78B4'),
          (0.20, '#6BAED6'),
          (0.45, '#EBF2D8'),
          (0.68, '#F6B15B'),
          (0.86, '#D95F0E'),
          (1.00, '#8C2D04'),
      ],
  )
  cmap_frac.set_bad(alpha=0.0)
  cmap_dissim = plt.get_cmap('YlOrBr').copy()
  cmap_dissim.set_bad(alpha=0.0)
  cmap_never = mcolors.ListedColormap(['#BDD7EB', '#C0392B'])
  cmap_never.set_bad(alpha=0.0)

  extent = [xmin, xmax, ymin, ymax]
  fig = plt.figure(figsize=(16.8, 9.09), dpi=250)
  gs = gridspec.GridSpec(
      2, 2, wspace=0.04, hspace=0.14,
      left=0.012, right=0.988, top=0.958, bottom=0.105,
  )

  # (a) 121,819 deduplicated physical stream gauges
  ax = fig.add_subplot(gs[0, 0])
  world.plot(ax=ax, color='#F1F5F9', edgecolor='#CBD5E1', linewidth=0.35, zorder=1)
  g_t1 = gauges[gauges['in_tier_1'] == True]
  g_t2_only = gauges[(gauges['in_tier_2'] == True) & (gauges['in_tier_1'] != True)]
  g_t3_only = gauges[gauges['in_tier_2'] != True]
  ax.scatter(g_t3_only['longitude'], g_t3_only['latitude'], s=0.55, c=COL_T3, alpha=0.68, linewidths=0, rasterized=True, zorder=2)
  ax.scatter(g_t2_only['longitude'], g_t2_only['latitude'], s=0.65, c=COL_T2, alpha=0.75, linewidths=0, rasterized=True, zorder=3)
  ax.scatter(g_t1['longitude'], g_t1['latitude'], s=0.75, c=COL_T1, alpha=0.82, linewidths=0, rasterized=True, zorder=4)
  ax.set_xlim(xmin, xmax)
  ax.set_ylim(ymin, ymax)
  ax.set_axis_off()
  n_ctry = int(gauges['country'].nunique()) if 'country' in gauges.columns else 163
  ax.set_title(f'(a) Publicly discoverable stream gauges ({len(gauges):,} stations in {n_ctry} countries)', fontsize=10.8, fontweight='bold', loc='left', pad=4)
  handles_a = [
      Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_T1, markersize=5.5, label=f'Standardized open datasets ({int(gauges["in_tier_1"].sum()):,})'),
      Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_T2, markersize=5.5, label=f'Confirmed public discharge ({int(gauges["in_tier_2"].sum()):,} cumulative)'),
      Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_T3, markersize=5.5, label=f'All publicly discoverable ({len(gauges):,} cumulative)'),
  ]
  ax.legend(handles=handles_a, loc='lower left', frameon=True, facecolor='white', edgecolor='#CBD5E1', framealpha=0.95, fontsize=7.8, borderpad=0.45, handletextpad=0.35)

  # (b) Never passes a gauge
  ax = fig.add_subplot(gs[0, 1])
  world.plot(ax=ax, color='#F1F5F9', edgecolor='none', zorder=1)
  ax.imshow(never_grid, origin='upper', extent=extent, cmap=cmap_never, vmin=-0.5, vmax=1.5, interpolation='nearest', zorder=2)
  world.boundary.plot(ax=ax, color='#334155', linewidth=0.28, zorder=3)
  ax.set_xlim(xmin, xmax)
  ax.set_ylim(ymin, ymax)
  ax.set_axis_off()
  ax.set_title('(b) Land whose runoff never passes any discoverable gauge', fontsize=10.8, fontweight='bold', loc='left', pad=4)
  handles_b = [
      Patch(facecolor='#C0392B', edgecolor='none', label='Never drains past a gauge (32–35% across hydrofabrics)'),
      Patch(facecolor='#BDD7EB', edgecolor='none', label='Drains past at least one discoverable gauge'),
  ]
  ax.legend(handles=handles_b, loc='lower left', frameon=True, facecolor='white', edgecolor='#CBD5E1', framealpha=0.95, fontsize=7.8, borderpad=0.45)

  # (c) Continuous fractional ungauged watershed area
  ax = fig.add_subplot(gs[1, 0])
  world.plot(ax=ax, color='#F1F5F9', edgecolor='none', zorder=1)
  im_c = ax.imshow(frac_grid, origin='upper', extent=extent, cmap=cmap_frac, vmin=0, vmax=100, interpolation='nearest', zorder=2)
  world.boundary.plot(ax=ax, color='#475569', linewidth=0.25, zorder=3)
  ax.set_xlim(xmin, xmax)
  ax.set_ylim(ymin, ymax)
  ax.set_axis_off()
  ax.set_title('(c) Fractional ungauged watershed area (HydroSHEDS Level 12)', fontsize=10.8, fontweight='bold', loc='left', pad=4)
  cax_c = fig.add_axes([0.075, 0.058, 0.36, 0.016])
  cb_c = fig.colorbar(im_c, cax=cax_c, orientation='horizontal')
  cb_c.set_ticks([0, 20, 40, 60, 80, 100])
  cb_c.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
  cb_c.ax.tick_params(labelsize=8.2)
  cb_c.set_label('Fractional ungaugedness (0% = fully gauged, 100% = completely ungauged)', fontsize=8.3, fontweight='bold', labelpad=3)

  # (d) Climate & landscape dissimilarity percentile
  ax = fig.add_subplot(gs[1, 1])
  world.plot(ax=ax, color='#F1F5F9', edgecolor='none', zorder=1)
  im_d = ax.imshow(dissim_grid, origin='upper', extent=extent, cmap=cmap_dissim, vmin=5, vmax=95, interpolation='nearest', zorder=2)
  world.boundary.plot(ax=ax, color='#475569', linewidth=0.25, zorder=3)
  ax.set_xlim(xmin, xmax)
  ax.set_ylim(ymin, ymax)
  ax.set_axis_off()
  ax.set_title('(d) Climate and landscape dissimilarity to gauged basins (HydroATLAS attributes)', fontsize=10.8, fontweight='bold', loc='left', pad=4)
  cax_d = fig.add_axes([0.565, 0.058, 0.36, 0.016])
  cb_d = fig.colorbar(im_d, cax=cax_d, orientation='horizontal', extend='both')
  cb_d.set_ticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
  cb_d.ax.tick_params(labelsize=8.2)
  cb_d.set_label('Climate and landscape dissimilarity percentile (0 = most similar, 100 = most dissimilar)', fontsize=8.3, fontweight='bold', labelpad=3)

  out_path = out_dir / 'Fig_1_global_overview.png'
  fig.savefig(out_path, dpi=250, bbox_inches='tight', pad_inches=0.06)
  plt.close(fig)
  print(f'  Saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)', flush=True)
  return out_path


# ==============================================================================
# FIGURE 2: Polygon-Size Dependence Across HydroSHEDS L1–L12 & Fine Hydrofabrics
# ==============================================================================
def build_fig2(out_dir: Path) -> Path:
  print('[Fig 2] Generating polygon-size dependence figure...', flush=True)
  _set_pub_rc(8.5)
  m2 = pd.read_parquet(HYDRO_RESULTS_DIR / 'master' / 'master_2d_metrics.parquet')

  tiers = [
      ('tier1_open_compilations', COL_T1, 'Standardized open datasets (34,577)'),
      ('tier2_public_discharge', COL_T2, 'Confirmed public discharge (93,827)'),
      ('tier3_all_known', COL_T3, 'All publicly discoverable (121,819)'),
  ]
  hy_levels = [f'hydrosheds_l{i:02d}' for i in range(1, 13)]
  fine_fabrics = [
      ('hydrosheds_l12', 'HydroSHEDS Level 12', 'o', 7.5),
      ('grit_v06_mainstem_tree', 'GRIT v0.6 (mainstem tree)', 'D', 7.0),
      ('grit_v06_bifurcating_dag', 'GRIT v0.6 (bifurcating DAG)', 'P', 7.5),
      ('merit_basins', 'MERIT-Basins', 's', 7.2),
      ('tdx_hydro', 'TDX-Hydro', '^', 8.0),
      ('hydrosheds_rivers', 'HydroSHEDS Rivers', '*', 10.5),
  ]

  def get_point(hf: str, tier: str) -> tuple[float, float]:
    sm = 'dag_width_weighted' if hf == 'grit_v06_bifurcating_dag' else 'outlet'
    rb = 'area_dag' if hf == 'grit_v06_bifurcating_dag' else 'area'
    sa = m2[(m2['hydrofabric'] == hf) & (m2['metric'] == 'U_net_Eq4') & (m2['domain_name'] == 'common_no_grl_ata') & (m2['tier'] == tier) & (m2['weight'] == 'area') & (m2['snapping_mode'] == sm) & (m2['ratio_basis'] == rb)]
    sc = m2[(m2['hydrofabric'] == hf) & (m2['metric'] == 'U_net_Eq4') & (m2['domain_name'] == 'common_no_grl_ata') & (m2['tier'] == tier) & (m2['weight'] == 'count') & (m2['snapping_mode'] == sm) & (m2['ratio_basis'] == rb)]
    mean_a = float(sa['total_weight'].iloc[0]) / float(sc['total_weight'].iloc[0])
    val = float(sa['value'].iloc[0]) * 100.0
    return mean_a, val

  fig, ax = plt.subplots(figsize=(10.8, 6.4), dpi=250)
  hy_coords_t3 = {}
  for tier_key, col, _ in tiers:
    xs, ys = [], []
    for hf in hy_levels:
      ma, val = get_point(hf, tier_key)
      xs.append(ma)
      ys.append(val)
      if tier_key == 'tier3_all_known':
        hy_coords_t3[hf] = (ma, val)
    ax.plot(xs, ys, color=col, linestyle=':', linewidth=2.1, alpha=0.92, zorder=3)
    ax.scatter(xs[:-1], ys[:-1], s=30, facecolors='white', edgecolors=col, linewidths=1.35, zorder=4)

  level_offsets = {
      'hydrosheds_l01': (-6, 9), 'hydrosheds_l02': (0, 9),
      'hydrosheds_l03': (4, -14), 'hydrosheds_l04': (4, -14),
      'hydrosheds_l05': (4, -14), 'hydrosheds_l06': (5, -13),
      'hydrosheds_l07': (6, -12), 'hydrosheds_l08': (6, -12),
      'hydrosheds_l09': (7, -11), 'hydrosheds_l10': (7, -11),
  }
  for hf, (dx, dy) in level_offsets.items():
    ma, val = hy_coords_t3[hf]
    lv = hf.replace('hydrosheds_l', 'L').replace('L0', 'L')
    ax.annotate(lv, (ma, val), xytext=(dx, dy), textcoords='offset points', fontsize=8.0, color='#7C2D12', fontweight='bold')

  for tier_key, col, _ in tiers:
    for hf, _, mk, ms in fine_fabrics:
      ma, val = get_point(hf, tier_key)
      ax.plot(ma, val, marker=mk, markersize=ms, color=col, markeredgecolor='#0F172A', markeredgewidth=0.65, linestyle='None', zorder=6)

  rect = plt.Rectangle((11, 77), 165, 22.5, fill=True, facecolor='#F8FAFC', edgecolor='#94A3B8', linestyle='--', linewidth=0.9, zorder=1)
  ax.add_patch(rect)

  ax.set_xscale('log')
  ax.set_xlim(9, 2.8e7)
  ax.set_ylim(-3, 103)
  ax.set_xlabel('Mean watershed polygon area (km², log scale)', fontsize=10.5, fontweight='bold')
  ax.set_ylabel('Fractionally ungauged global land area (%)', fontsize=10.5, fontweight='bold')
  ax.set_title('Polygon-size dependence of fractional watershed ungaugedness across HydroSHEDS Levels 1–12 and four global hydrofabrics', fontsize=10.6, fontweight='bold', loc='left', pad=10)
  ax.grid(True, linestyle='--', alpha=0.35, color='#CBD5E1')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  ax_in = ax.inset_axes([0.075, 0.095, 0.43, 0.44])
  ax_in.set_facecolor('white')
  for tier_key, col, _ in tiers:
    xs_sub, ys_sub = [], []
    for hf in ['hydrosheds_l09', 'hydrosheds_l10', 'hydrosheds_l11', 'hydrosheds_l12']:
      ma, val = get_point(hf, tier_key)
      xs_sub.append(ma)
      ys_sub.append(val)
    ax_in.plot(xs_sub, ys_sub, color=col, linestyle=':', linewidth=1.8, alpha=0.85, zorder=2)
    for hf, _, mk, ms in fine_fabrics:
      ma, val = get_point(hf, tier_key)
      ax_in.plot(ma, val, marker=mk, markersize=ms * 0.92, color=col, markeredgecolor='#0F172A', markeredgewidth=0.65, linestyle='None', zorder=5)

  inset_labels = {
      'hydrosheds_rivers': ('HydroSHEDS\nRivers', (4, -19)),
      'tdx_hydro': ('TDX-Hydro', (6, -11)),
      'merit_basins': ('MERIT-\nBasins', (-6, -24)),
      'grit_v06_mainstem_tree': ('GRIT v0.6\n(tree & DAG)', (-20, -24)),
      'hydrosheds_l12': ('HydroSHEDS\nL12', (-38, -22)),
  }
  for hf, (txt, (dx, dy)) in inset_labels.items():
    ma, val = get_point(hf, 'tier3_all_known')
    ax_in.annotate(txt, (ma, val), xytext=(dx, dy), textcoords='offset points', fontsize=7.3, fontweight='bold', color='#1E293B', arrowprops=dict(arrowstyle='-', color='#64748B', lw=0.65))

  ax_in.set_xscale('log')
  ax_in.set_xlim(12.5, 175)
  ax_in.set_ylim(75.5, 99.0)
  ax_in.set_xticks([15, 20, 40, 70, 130])
  ax_in.set_xticklabels(['15', '20', '40', '70', '130'], fontsize=7.8)
  ax_in.tick_params(axis='y', labelsize=7.8)
  ax_in.set_title('Fine-scale hydrofabrics (zoom: 13–175 km²)', fontsize=8.4, fontweight='bold', loc='left', pad=3)
  ax_in.grid(True, linestyle='--', alpha=0.35, color='#CBD5E1')
  for sp in ['top', 'right', 'left', 'bottom']:
    ax_in.spines[sp].set_color('#64748B')
    ax_in.spines[sp].set_linewidth(0.9)

  tier_handles = [
      Line2D([0], [0], color=col, linestyle=':', linewidth=2.2, marker='o', markerfacecolor='white', markeredgecolor=col, markersize=5.2, label=lbl)
      for _, col, lbl in tiers
  ]
  fab_handles = [
      Line2D([0], [0], color='none', marker=mk, markerfacecolor='#475569', markeredgecolor='#0F172A', markeredgewidth=0.65, markersize=ms * 0.85, label=name)
      for _, name, mk, ms in fine_fabrics
  ]
  leg1 = ax.legend(handles=tier_handles, title='Data-access tier (HydroSHEDS L1–L12 dotted lines)', loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=True, facecolor='white', edgecolor='#CBD5E1', framealpha=0.96, fontsize=8.2, title_fontsize=8.3)
  leg1.get_title().set_fontweight('bold')
  ax.add_artist(leg1)
  leg2 = ax.legend(handles=fab_handles, title='Fine-scale hydrofabrics (symbols)', loc='upper right', bbox_to_anchor=(0.99, 0.72), frameon=True, facecolor='white', edgecolor='#CBD5E1', framealpha=0.96, fontsize=8.1, title_fontsize=8.3, ncol=1)
  leg2.get_title().set_fontweight('bold')

  out_path = out_dir / 'Fig_2_polygon_size_dependence.png'
  fig.savefig(out_path, dpi=250, bbox_inches='tight', pad_inches=0.08)
  plt.close(fig)
  print(f'  Saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)', flush=True)
  return out_path


# ==============================================================================
# FIGURE 3: Regional, Aridity & Dissimilarity Breakdown (Fathom-Free)
# ==============================================================================
def build_fig3(world_all: gpd.GeoDataFrame, basins: pd.DataFrame, out_dir: Path) -> Path:
  print('[Fig 3] Generating regional, aridity & climate/landscape dissimilarity figure...', flush=True)
  _set_pub_rc(7.4)
  c2cont = dict(zip(world_all['map_name'], world_all['continent']))
  no_grl = basins[basins['country_name'] != 'Greenland'].copy()
  no_grl['continent'] = no_grl['country_name'].map(c2cont)

  t1 = pd.read_parquet(TIER_DIR / 'tier_caravan_estreams_34577.parquet')
  t2 = pd.read_parquet(TIER_DIR / 'tier_public_discharge_93827.parquet')
  t4 = pd.read_parquet(TIER_DIR / 'tier4_all_publicly_knowable_121819.parquet')
  for df in (t1, t2, t4):
    df.index = norm_hybas_id(df.index)

  m2 = pd.read_parquet(HYDRO_RESULTS_DIR / 'master' / 'master_2d_metrics.parquet')
  six = ['hydrosheds_l12', 'hydrosheds_rivers', 'merit_basins', 'tdx_hydro', 'grit_v06_mainstem_tree', 'grit_v06_bifurcating_dag']
  sub_frac = filter_canonical_2d(m2[(m2['hydrofabric'].isin(six)) & (m2['metric'] == 'U_net_Eq4')])

  conts = ['Africa', 'Asia', 'Oceania', 'South America', 'Europe', 'North America']
  cont_labels = ['Africa', 'Asia', 'Oceania', 'S.\nAmerica', 'Europe', 'N.\nAmerica']
  arids = ['Hyper-arid', 'Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']
  arid_labels = ['Hyper-\narid', 'Arid', 'Semi-\narid', 'Dry sub-\nhumid', 'Humid']

  tiers3 = [
      ('tier1_open_compilations', t1, COL_T1, 'Standardized open datasets (34,577 gauges)'),
      ('tier2_public_discharge', t2, COL_T2, 'Confirmed public discharge (93,827)'),
      ('tier3_all_known', t4, COL_T3, 'All publicly discoverable (121,819)'),
  ]
  tiers2 = [
      ('tier1_open_compilations', COL_T1, 'Standardized open (34,577)'),
      ('tier3_all_known', COL_T3, 'All discoverable (121,819)'),
  ]

  fig, axes = plt.subplots(
      2, 2, figsize=(7.20, 5.10), dpi=SAVE_DPI,
      gridspec_kw={'wspace': 0.24, 'hspace': 0.42, 'left': 0.08, 'right': 0.985, 'top': 0.875, 'bottom': 0.095},
  )

  top_handles = []
  for ax, w_key, title, ylabel, grab in [
      (axes[0, 0], 'area', '(a) Fractional ungauged land area by continent', 'Fractionally ungauged land (%)', True),
      (axes[0, 1], 'population', '(b) Fractional ungauged population by continent', 'Fractionally ungauged population (%)', False),
  ]:
    x = np.arange(len(conts))
    bw = 0.255
    for i, (tkey, _, col, lbl) in enumerate(tiers3):
      meds, lo, hi = [], [], []
      for c in conts:
        vals = sub_frac[(sub_frac['domain_type'] == 'continent') & (sub_frac['domain_name'] == c) & (sub_frac['tier'] == tkey) & (sub_frac['weight'] == w_key)]['value'].values * 100.0
        med = float(np.median(vals))
        meds.append(med)
        lo.append(med - float(np.min(vals)))
        hi.append(float(np.max(vals)) - med)
      pos = x + (i - 1) * bw
      bar_c = ax.bar(pos, meds, width=bw, color=col, alpha=0.90, label=lbl, edgecolor='white', linewidth=0.4, zorder=3)
      if grab:
        top_handles.append(bar_c)
      ax.errorbar(pos, meds, yerr=[lo, hi], fmt='none', ecolor='#0F172A', elinewidth=0.75, capsize=1.6, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(cont_labels, fontsize=7.0)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis='y', labelsize=7.0)
    ax.set_ylabel(ylabel, fontsize=7.5, fontweight='bold')
    ax.set_title(title, fontsize=8.0, fontweight='bold', loc='left', pad=5)
    ax.grid(True, axis='y', linestyle='--', alpha=0.35, color='#CBD5E1', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

  fig.legend(
      top_handles, [t[3] for t in tiers3],
      loc='upper center', bbox_to_anchor=(0.53, 0.985), ncol=3,
      frameon=True, facecolor='white', edgecolor='#CBD5E1', framealpha=0.98,
      fontsize=6.6, columnspacing=1.2, handletextpad=0.45, borderpad=0.4,
  )

  # (c) Aridity zones
  ax = axes[1, 0]
  xc = np.arange(len(arids))
  bwc = 0.34
  for i, (tkey, col, _) in enumerate(tiers2):
    meds, lo, hi = [], [], []
    for a in arids:
      vals = sub_frac[(sub_frac['domain_type'] == 'aridity_zone') & (sub_frac['domain_name'] == a) & (sub_frac['tier'] == tkey) & (sub_frac['weight'] == 'area')]['value'].values * 100.0
      med = float(np.median(vals))
      meds.append(med)
      lo.append(med - float(np.min(vals)))
      hi.append(float(np.max(vals)) - med)
    pos = xc + (-bwc / 2 if i == 0 else bwc / 2)
    ax.bar(pos, meds, width=bwc, color=col, alpha=0.90, edgecolor='white', linewidth=0.4, zorder=3)
    ax.errorbar(pos, meds, yerr=[lo, hi], fmt='none', ecolor='#0F172A', elinewidth=0.75, capsize=1.8, zorder=4)
  ax.set_xticks(xc)
  ax.set_xticklabels(arid_labels, fontsize=7.0)
  ax.set_ylim(0, 108)
  ax.set_yticks([0, 20, 40, 60, 80, 100])
  ax.tick_params(axis='y', labelsize=7.0)
  ax.set_ylabel('Fractionally ungauged land (%)', fontsize=7.5, fontweight='bold')
  ax.set_title('(c) Fractional ungauged land by aridity zone', fontsize=8.0, fontweight='bold', loc='left', pad=5)
  ax.grid(True, axis='y', linestyle='--', alpha=0.35, color='#CBD5E1', linewidth=0.5)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  # (d) Cutoff-free climate & landscape dissimilarity percentile
  ax = axes[1, 1]
  xd = np.arange(len(conts))
  bwd = 0.255
  for i, (_, tdf, col, _) in enumerate(tiers3):
    s_pct = gauged_ref_pct(tdf, no_grl.index)
    meds, lo, hi = [], [], []
    for c in conts:
      mask = (no_grl['continent'] == c).values
      vals = s_pct.values[mask]
      wts = no_grl['sub_area'].values[mask]
      med = w_quantile(vals, wts, 0.5)
      q25 = w_quantile(vals, wts, 0.25)
      q75 = w_quantile(vals, wts, 0.75)
      meds.append(med)
      lo.append(med - q25)
      hi.append(q75 - med)
    pos = xd + (i - 1) * bwd
    ax.bar(pos, meds, width=bwd, color=col, alpha=0.90, edgecolor='white', linewidth=0.4, zorder=3)
    ax.errorbar(pos, meds, yerr=[lo, hi], fmt='none', ecolor='#0F172A', elinewidth=0.75, capsize=1.6, zorder=4)
  ref_line = ax.axhline(50, color='#475569', linestyle='--', linewidth=0.95, zorder=2, label='Gauged-basin median (50th)')
  ax.legend(handles=[ref_line], loc='upper right', frameon=True, facecolor='white', edgecolor='#CBD5E1', framealpha=0.95, fontsize=6.0, borderpad=0.3)
  ax.set_xticks(xd)
  ax.set_xticklabels(cont_labels, fontsize=7.0)
  ax.set_ylim(0, 114)
  ax.set_yticks([0, 20, 40, 60, 80, 100])
  ax.tick_params(axis='y', labelsize=7.0)
  ax.set_ylabel('Dissimilarity percentile vs. gauged', fontsize=7.5, fontweight='bold')
  ax.set_title('(d) Climate & landscape dissimilarity (0–100)', fontsize=8.0, fontweight='bold', loc='left', pad=5)
  ax.grid(True, axis='y', linestyle='--', alpha=0.35, color='#CBD5E1', linewidth=0.5)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  out_path = out_dir / 'Fig_3_regional_aridity_dissimilarity.png'
  fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.05)
  plt.close(fig)
  print(f'  Saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)', flush=True)
  return out_path


# ==============================================================================
# FIGURE 4: National CAP-Gap Coverage & INFORM 2026 Correlations (Fathom-Free)
# ==============================================================================
def build_fig4(world_all: gpd.GeoDataFrame, basins: pd.DataFrame, out_dir: Path) -> Path:
  print('[Fig 4] Generating CAP-gap alerting coverage & INFORM 2026 figure...', flush=True)
  _set_pub_rc(7.4)
  c2iso = dict(zip(world_all['map_name'], world_all['iso_clean']))

  t4 = pd.read_parquet(TIER_DIR / 'tier4_all_publicly_knowable_121819.parquet', columns=['U_net_tree_phys'])
  t4.index = norm_hybas_id(t4.index)

  no_grl = basins[basins['country_name'].notna() & (basins['country_name'] != 'Greenland')].copy()
  no_grl['u4_area'] = t4.loc[no_grl.index, 'U_net_tree_phys'].values * no_grl['sub_area']
  no_grl['u4_pop'] = t4.loc[no_grl.index, 'U_net_tree_phys'].values * no_grl['total_pop']

  ctry = no_grl.groupby('country_name').agg(
      has_system=('has_system', 'first'),
      area=('sub_area', 'sum'),
      pop=('total_pop', 'sum'),
      u4_area=('u4_area', 'sum'),
      u4_pop=('u4_pop', 'sum'),
  ).reset_index()
  ctry['U4_area_pct'] = ctry['u4_area'] / ctry['area'] * 100.0
  ctry['U4_pop_pct'] = ctry['u4_pop'] / ctry['pop'] * 100.0

  df_inf = pd.read_csv(file_paths.RISK_FILE, skiprows=[0, 2])
  df_inf['ISO3'] = df_inf['ISO3'].astype(str).str.strip()
  df_inf['COUNTRY_CLEAN'] = df_inf['COUNTRY'].astype(str).str.strip().replace(file_paths.country_name_map)
  for col in ['VULNERABILITY', 'LACK OF COPING CAPACITY']:
    df_inf[col] = pd.to_numeric(df_inf[col], errors='coerce')
    iso_m = df_inf.dropna(subset=['ISO3', col]).drop_duplicates('ISO3').set_index('ISO3')[col]
    nam_m = df_inf.dropna(subset=['COUNTRY_CLEAN', col]).drop_duplicates('COUNTRY_CLEAN').set_index('COUNTRY_CLEAN')[col]
    ctry[col] = [iso_m.get(c2iso.get(c), nam_m.get(c, np.nan)) for c in ctry['country_name']]

  fig = plt.figure(figsize=(7.20, 5.20), dpi=SAVE_DPI)
  gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.36, left=0.075, right=0.985, top=0.90, bottom=0.09)

  # (a) Map of national flood-alerting coverage (174 countries excl. Greenland)
  ax_a = fig.add_subplot(gs[0, 0])
  w_plot = world_all[world_all['name'] != 'Antarctica'].copy()
  status_map = dict(zip(ctry['country_name'], ctry['has_system']))
  w_plot['warn_grp'] = w_plot['map_name'].map(status_map)
  w_plot[w_plot['warn_grp'].isna()].plot(ax=ax_a, color='#E2E8F0', edgecolor='#CBD5E1', linewidth=0.25, zorder=1)
  w_plot[w_plot['warn_grp'] == True].plot(ax=ax_a, color=COL_T1, edgecolor='#0F172A', linewidth=0.25, alpha=0.88, zorder=2)
  w_plot[w_plot['warn_grp'] == False].plot(ax=ax_a, color=COL_T3, edgecolor='#0F172A', linewidth=0.25, alpha=0.88, zorder=2)
  ax_a.set_xlim(-175, 180)
  ax_a.set_ylim(-58, 84)
  ax_a.set_axis_off()
  n_sys = int((ctry['has_system'] == True).sum())
  n_gap = int((ctry['has_system'] == False).sum())
  ax_a.set_title(f'(a) National flood-alerting coverage ({len(ctry)} countries)', fontsize=7.8, fontweight='bold', loc='left', pad=6)
  ax_a.legend(
      handles=[
          Patch(facecolor=COL_T1, edgecolor='#0F172A', linewidth=0.4, label=f'Established system (N = {n_sys})'),
          Patch(facecolor=COL_T3, edgecolor='#0F172A', linewidth=0.4, label=f'CAP-gap country (N = {n_gap})'),
      ],
      loc='lower left', frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=5.6, framealpha=0.95,
  )

  # (b) Ungaugedness by alerting status (clean boxplots without overlapping banner)
  ax_b = fig.add_subplot(gs[0, 1])
  w_df = ctry[ctry['has_system'] == True]
  g_df = ctry[ctry['has_system'] == False]
  positions = [0.8, 1.45, 2.55, 3.2]
  box_data = [w_df['U4_area_pct'].values, g_df['U4_area_pct'].values, w_df['U4_pop_pct'].values, g_df['U4_pop_pct'].values]
  box_cols = [COL_T1, COL_T3, COL_T1, COL_T3]
  bp = ax_b.boxplot(
      box_data, positions=positions, widths=0.46, patch_artist=True, showfliers=False,
      medianprops=dict(color='#0F172A', linewidth=1.3),
      whiskerprops=dict(color='#334155', linewidth=0.8),
      capprops=dict(color='#334155', linewidth=0.8),
      boxprops=dict(linewidth=0.75, edgecolor='#0F172A'),
  )
  rng = np.random.default_rng(42)
  for patch, col, pos, vals in zip(bp['boxes'], box_cols, positions, box_data):
    patch.set_facecolor(col)
    patch.set_alpha(0.75)
    jitter = rng.uniform(-0.12, 0.12, size=len(vals))
    ax_b.scatter(np.full_like(vals, pos) + jitter, vals, s=5.5, color=col, alpha=0.42, edgecolors='none', zorder=2)

  mw_a = mannwhitneyu(g_df['U4_area_pct'], w_df['U4_area_pct']).pvalue
  mw_p = mannwhitneyu(g_df['U4_pop_pct'], w_df['U4_pop_pct']).pvalue
  ax_b.text(1.125, 103.5, f'p = {mw_a:.1e}', ha='center', va='bottom', fontsize=6.2, fontweight='bold', color='#1E293B')
  ax_b.plot([0.8, 1.45], [101.8, 101.8], color='#334155', lw=0.7)
  ax_b.text(2.875, 103.5, f'p = {mw_p:.1e}', ha='center', va='bottom', fontsize=6.2, fontweight='bold', color='#1E293B')
  ax_b.plot([2.55, 3.2], [101.8, 101.8], color='#334155', lw=0.7)

  ax_b.set_xticks([1.125, 2.875])
  ax_b.set_xticklabels(['Land area\n(All discoverable)', 'Population\n(All discoverable)'], fontsize=7.0)
  ax_b.set_ylim(-4, 114)
  ax_b.set_yticks([0, 20, 40, 60, 80, 100])
  ax_b.tick_params(axis='y', labelsize=7.0)
  ax_b.set_ylabel('Fractionally ungauged (%)', fontsize=7.5, fontweight='bold')
  ax_b.set_title('(b) Ungaugedness by alerting status (HydroSHEDS L12)', fontsize=7.8, fontweight='bold', loc='left', pad=6)
  ax_b.grid(True, axis='y', linestyle='--', alpha=0.35, color='#CBD5E1', linewidth=0.5)
  ax_b.spines['top'].set_visible(False)
  ax_b.spines['right'].set_visible(False)

  # (c) & (d) INFORM 2026 Vulnerability & Lack of Coping Capacity
  sub_inf = ctry.dropna(subset=['VULNERABILITY', 'LACK OF COPING CAPACITY', 'U4_pop_pct']).copy()
  n_inf_sys = int((sub_inf['has_system'] == True).sum())
  n_inf_gap = int((sub_inf['has_system'] == False).sum())
  shared_handles = []
  for slot, col_x, xlabel, title in [
      (gs[1, 0], 'VULNERABILITY', 'INFORM 2026 Vulnerability Index (0–10)', '(c) Population ungaugedness vs. vulnerability'),
      (gs[1, 1], 'LACK OF COPING CAPACITY', 'INFORM 2026 Lack of Coping Capacity (0–10)', '(d) Population ungaugedness vs. coping gap'),
  ]:
    ax = fig.add_subplot(slot)
    for has_sys, col, mk, lbl in [
        (True, COL_T1, 'o', f'Established alerting system (N = {n_inf_sys})'),
        (False, COL_T3, '^', f'CAP-gap country (N = {n_inf_gap})'),
    ]:
      s = sub_inf[sub_inf['has_system'] == has_sys]
      sc = ax.scatter(s[col_x], s['U4_pop_pct'], s=18, c=col, marker=mk, alpha=0.78, edgecolors='#0F172A', linewidths=0.35, label=lbl, zorder=3)
      if slot == gs[1, 0]:
        shared_handles.append(sc)
    coef = np.polyfit(sub_inf[col_x].values, sub_inf['U4_pop_pct'].values, 1)
    xg = np.linspace(sub_inf[col_x].min(), sub_inf[col_x].max(), 200)
    yg = np.polyval(coef, xg)
    keep = (yg >= 0) & (yg <= 100)
    ax.plot(xg[keep], yg[keep], color='#334155', linestyle='--', linewidth=1.1, zorder=2)

    rs, pv = spearmanr(sub_inf[col_x], sub_inf['U4_pop_pct'])
    ax.text(
        0.97, 0.05, f'N = {len(sub_inf)}\nrₛ = {rs:+.2f}\np = {pv:.1e}',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=6.4, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#F8FAFC', edgecolor='#CBD5E1', alpha=0.94),
    )
    ax.set_xlim(0.2, 9.8)
    ax.set_ylim(-4, 106)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(labelsize=7.0)
    ax.set_xlabel(xlabel, fontsize=7.4, fontweight='bold')
    if slot == gs[1, 0]:
      ax.set_ylabel('Fractionally ungauged pop. (%)', fontsize=7.5, fontweight='bold')
    ax.set_title(title, fontsize=7.8, fontweight='bold', loc='left', pad=5)
    ax.grid(True, linestyle='--', alpha=0.35, color='#CBD5E1', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

  fig.legend(
      shared_handles,
      [f'Established national alerting system (N = {n_sys})', f'CAP-gap country (N = {n_gap}; {n_inf_gap} in INFORM)'],
      loc='upper center', bbox_to_anchor=(0.53, 0.985), ncol=2,
      frameon=True, facecolor='white', edgecolor='#CBD5E1', framealpha=0.98, fontsize=6.7,
  )

  out_path = out_dir / 'Fig_4_cap_gap_and_inform.png'
  fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.05)
  plt.close(fig)
  print(f'  Saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)', flush=True)
  return out_path


# ==============================================================================
# FIGURE 5 (OPTIONAL — Requires Licensed Fathom v3 Counterfactual Outputs)
# ==============================================================================
def build_fig5_optional(world_all: gpd.GeoDataFrame, out_dir: Path) -> Path | None:
  basin_pq = CF_DIR / 'basin_level_exposure_counterfactuals_r2.parquet'
  sum_csv = CF_DIR / 'multi_mechanism_tiered_exposure_r2.csv'
  t3_csv = CF_DIR / 'Table_S3_175territories_r2.csv'
  if not (basin_pq.exists() and sum_csv.exists() and t3_csv.exists()):
    print(
        '[Fig 5] SKIPPED: Licensed Fathom v3 flood-exposure counterfactual outputs '
        f'not found under {CF_DIR}. (Fathom hazard maps are proprietary and excluded '
        'from public archives.)',
        flush=True,
    )
    return None

  print('[Fig 5] Generating optional Fathom v3 exposure counterfactuals figure...', flush=True)
  _set_pub_rc(7.0)
  world = world_all[world_all['name'] != 'Antarctica'].copy()
  df_b = pd.read_parquet(basin_pq)
  df_sum = pd.read_csv(sum_csv)
  df_c = pd.read_csv(t3_csv)
  boot_csv = CF_DIR / 'bootstrap_table1_fig3_ci95_r2.csv'
  df_ci = pd.read_csv(boot_csv) if boot_csv.exists() else None

  def get_sum(tk: str, mech: str):
    return df_sum[(df_sum['tier_key'] == tk) & (df_sum['mechanism'] == mech)].iloc[0]

  def get_ci_bounds(scen: str, metric: str, point_val_m: float) -> tuple[float, float]:
    if df_ci is None:
      return 0.0, 0.0
    sub = df_ci[(df_ci['scenario'] == scen) & (df_ci['metric'] == metric)]
    if len(sub) == 0:
      return 0.0, 0.0
    lo = float(sub.iloc[0]['boot_p025']) / 1e6
    hi = float(sub.iloc[0]['boot_p975']) / 1e6
    return max(0.0, point_val_m - lo), max(0.0, hi - point_val_m)

  r_u111 = get_sum('tier5_universal_idealized_ceiling', '111_all_three_combined_snap')
  cap_m = float(r_u111['delta_net_nosys115']) / 1e6
  cap_pct = float(r_u111['pct_unw_closed_net_nosys115'])
  glb_m = float(r_u111['delta_net_global']) / 1e6
  glb_pct = float(r_u111['pct_unw_closed_net_global'])
  n_sys = int(df_c['has_system'].sum())
  n_nosys = int((~df_c['has_system']).sum())

  fig = plt.figure(figsize=(NATURE_W, NATURE_W * 0.72), dpi=SAVE_DPI, facecolor='#ffffff')
  gs = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[1.00, 0.055, 0.22, 0.92], hspace=0.05, left=0.01, right=0.99, top=0.945, bottom=0.095)

  col_map = 'dnet__tier5_universal_idealized_ceiling__111_all_three_combined_snap'
  pts = df_b.dropna(subset=['lat', 'lon'])
  pos = pts[pts[col_map] > 0].sort_values(col_map, ascending=True)
  zero = pts[pts[col_map] <= 0]
  vmax_map = max(1e3, float(np.nanpercentile(pos[col_map].values, 99.5))) if len(pos) > 0 else 1e3
  vmin_map = max(0.05, vmax_map / 1e4)

  ax_a = fig.add_subplot(gs[0])
  world.plot(ax=ax_a, color='#F2F4F7', edgecolor='none', zorder=1)
  ax_a.scatter(zero['lon'], zero['lat'], s=0.05, marker='.', color='#D8DFE6', linewidths=0, rasterized=True, zorder=2)
  sc_a = ax_a.scatter(pos['lon'], pos['lat'], c=np.maximum(pos[col_map].values, vmin_map), s=0.08, marker='.', cmap='magma_r', norm=mcolors.LogNorm(vmin=vmin_map, vmax=vmax_map), linewidths=0, rasterized=True, zorder=3)
  world.boundary.plot(ax=ax_a, color='#455A64', linewidth=0.22, zorder=4)
  ax_a.set_xlim([-180, 180])
  ax_a.set_ylim([-58, 84])
  ax_a.set_axis_off()
  ax_a.set_title(f'(a) Unforecasted flood exposure reduction under universal gauging ({cap_m:.2f}M/yr CAP-gap; {glb_m:.2f}M/yr global)', fontsize=7.3, fontweight='bold', loc='center', pad=3.0)
  gs_cb = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], width_ratios=[0.12, 0.76, 0.12])
  cb = fig.colorbar(sc_a, cax=fig.add_subplot(gs_cb[0, 1]), orientation='horizontal', extend='both')
  cb.set_label(f'Unforecasted flood exposure reduction at α = 1.0 (persons/yr; {cap_m:.2f}M/yr CAP-gap, {cap_pct:.1f}%; {glb_m:.2f}M/yr global, {glb_pct:.1f}%)', fontsize=6.0, fontweight='bold', labelpad=2.0)
  cb.ax.tick_params(labelsize=5.9, pad=1.2)

  gs_bot = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[3], width_ratios=[0.07, 0.44, 0.08, 0.41], wspace=0.0)
  ax_b = fig.add_subplot(gs_bot[0, 1])
  ax_c = fig.add_subplot(gs_bot[0, 3])

  b_tiers = [
      ('tier_caravan_estreams_34577', 'Tier 1:\nOpen\n(34,577)'),
      ('tier_public_discharge_93827', 'Tier 2:\nPublic Q\n(93,827)'),
      ('tier4_all_publicly_knowable_121819', 'Tier 3:\nDiscoverable\n(121,819)'),
      ('tier5_universal_idealized_ceiling', 'Universal\ngauging\n(1,034,083)'),
  ]
  xb = np.arange(len(b_tiers)) * 1.16
  bar_step, wb = 0.245, 0.215
  mechs_b = [
      ('000_global_train_joint', '000_train', '#0072B2', 'Global training alone', -1.5),
      ('010_ft_only_snap', '010_ft_only', '#009E73', 'Local FT alone', -0.5),
      ('001_da_only_snap', '001_da_only', '#43A2CA', 'Real-time DA alone', 0.5),
      ('111_all_three_combined_snap', '111_combined', '#E66101', 'Combined (train + FT + DA)', 1.5),
  ]
  max_b_top = 0.0
  for mk, b_mk, col, lbl, sh in mechs_b:
    vals = [float(get_sum(tk, mk)['delta_net_nosys115']) / 1e6 for tk, _ in b_tiers]
    lo_e, hi_e = zip(*[get_ci_bounds(f'{tk}__{b_mk}', 'delta_nosys115', v) for (tk, _), v in zip(b_tiers, vals)])
    xs = xb + sh * bar_step
    ax_b.bar(xs, vals, width=wb, color=col, edgecolor='#222222', linewidth=0.4, label=lbl)
    if df_ci is not None and any(x > 0 for x in hi_e):
      ax_b.errorbar(xs, vals, yerr=[lo_e, hi_e], fmt='none', ecolor='#111111', elinewidth=0.65, capsize=1.4, zorder=5)
    for v, he in zip(vals, hi_e):
      max_b_top = max(max_b_top, v + he)
  ax_b.axvline(2.5 * 1.16, color='#78909C', linestyle='--', linewidth=0.7)
  ax_b.set_xticks(xb)
  ax_b.set_xticklabels([lbl for _, lbl in b_tiers], fontsize=5.6, fontweight='bold')
  ax_b.set_ylim(0, max_b_top * 1.28)
  ax_b.set_ylabel('CAP-gap net reduction (M/yr)', fontsize=6.4, fontweight='bold')
  ax_b.set_title('(b) Exposure reduction in CAP-gap countries', fontsize=7.2, fontweight='bold', loc='left', pad=3.5)
  ax_b.legend(loc='upper left', fontsize=4.8, frameon=True, framealpha=0.95, edgecolor='#CFD8DC')
  ax_b.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.4)
  sns.despine(ax=ax_b)

  c_tiers = [
      ('tier1a_active_post2016_ccby_8811', '000_global_train_joint', 'Post-2016\nCaravan'),
      ('tier_caravan_estreams_34577', '111_all_three_combined_snap', 'Tier 1:\nOpen'),
      ('tier_public_discharge_93827', '111_all_three_combined_snap', 'Tier 2:\nPublic Q'),
      ('tier4_all_publicly_knowable_121819', '111_all_three_combined_snap', 'Tier 3:\nDisc.'),
      ('tier5_universal_idealized_ceiling', '111_all_three_combined_snap', 'Universal\ngauging'),
  ]
  xc = np.arange(len(c_tiers))
  w_sys = np.array([float(get_sum(tk, mk)['warned_sys60']) / 1e6 for tk, mk, _ in c_tiers])
  w_nos = np.array([float(get_sum(tk, mk)['warned_nosys115']) / 1e6 for tk, mk, _ in c_tiers])
  w_tot = w_sys + w_nos
  ax_c.bar(xc, w_sys, width=0.54, color='#0072B2', edgecolor='#222222', linewidth=0.45, label=f'National alerting ({n_sys})')
  ax_c.bar(xc, w_nos, width=0.54, bottom=w_sys, color='#D55E00', edgecolor='#222222', linewidth=0.45, label=f'CAP-gap ({n_nosys})')
  ax_c.axvline(3.5, color='#78909C', linestyle='--', linewidth=0.7)
  ax_c.set_xticks(xc)
  ax_c.set_xticklabels([lbl for _, _, lbl in c_tiers], fontsize=5.4, fontweight='bold')
  ax_c.set_ylim(0, w_tot.max() * 1.24)
  ax_c.set_ylabel('Total exposure forecasted (M/yr)', fontsize=6.4, fontweight='bold')
  ax_c.set_title('(c) Total forecasted exposure', fontsize=7.2, fontweight='bold', loc='left', pad=3.5)
  ax_c.legend(loc='upper left', fontsize=5.1, frameon=True, framealpha=0.95, edgecolor='#CFD8DC')
  ax_c.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.4)
  sns.despine(ax=ax_c)

  out_path = out_dir / 'Fig_5_exposure_counterfactuals.png'
  fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.05)
  plt.close(fig)
  print(f'  Saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)', flush=True)
  return out_path


# ==============================================================================
# FIGURE 6: Exact 2^7=128 Group Shapley Factor Attribution (Fathom-Free)
# ==============================================================================
def build_fig6(out_dir: Path) -> Path:
  print('[Fig 6] Generating Exact Group Shapley factor attribution figure...', flush=True)
  _set_pub_rc(7.0)

  df_cat_sum = pd.read_csv(SHAP_DIR / 'shap_category_summary_r2.csv')
  df_cat_pct = pd.read_csv(SHAP_DIR / 'shap_category_pct_r2.csv', index_col=0)
  df_cat_basin = pd.read_parquet(SHAP_DIR / 'shap_category_per_basin_r2.parquet')

  cat_palette = {
      'Climate & Hydrology': '#1f77b4',
      'Land Cover': '#2ca02c',
      'Climatic Similarity (OOD)': '#ff7f0e',
      'Topography': '#e377c2',
      'Soil & Geology': '#8c564b',
      'Anthropogenic': '#d62728',
      'River-Network Gauge Cov.': '#9467bd',
  }
  short_cat = {
      'Climate & Hydrology': 'Climate/Hydrol.',
      'Land Cover': 'Land Cover',
      'Climatic Similarity (OOD)': 'Clim. Sim. (OOD)',
      'Topography': 'Topography',
      'Soil & Geology': 'Soil & Geol.',
      'Anthropogenic': 'Anthropogenic',
      'River-Network Gauge Cov.': 'Gauge Cov.',
  }
  metric_palette = {
      'NSE': '#0d47a1', 'KGE': '#42a5f5',
      '1.01-year-2.0-window-precision': '#ffcc80', '2.0-year-2.0-window-precision': '#ffa726',
      '5.0-year-2.0-window-precision': '#fb8c00', '10.0-year-2.0-window-precision': '#e65100',
      '20.0-year-2.0-window-precision': '#b71c1c',
      '1.01-year-2.0-window-recall': '#c8e6c9', '2.0-year-2.0-window-recall': '#81c784',
      '5.0-year-2.0-window-recall': '#4caf50', '10.0-year-2.0-window-recall': '#2e7d32',
      '20.0-year-2.0-window-recall': '#1b5e20',
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

  cat_order = df_cat_pct[ALL_12_METRICS].mean(axis=1).sort_values(ascending=False).index.tolist()

  fig_h = NATURE_W * (493.16 / 468.0)
  fig = plt.figure(figsize=(NATURE_W, fig_h), dpi=SAVE_DPI, facecolor='#ffffff')
  gs = gridspec.GridSpec(4, 4, figure=fig, height_ratios=[1.25, 1.0, 1.0, 1.0], hspace=0.58, wspace=0.88, left=0.138, right=0.970, top=0.955, bottom=0.055)

  ax_a = fig.add_subplot(gs[0, :])
  x_cats = np.arange(len(cat_order))
  n_m = len(ALL_12_METRICS)
  w_bar = 0.80 / n_m
  for mi, m in enumerate(ALL_12_METRICS):
    sub_b = df_cat_basin[df_cat_basin['metric'] == m]
    meds, q25s, q75s = [], [], []
    for c in cat_order:
      arr = sub_b.loc[sub_b['category'] == c, 'pct_shap'].values
      q25, q50, q75 = np.percentile(arr, [25, 50, 75])
      meds.append(float(q50))
      q25s.append(float(q25))
      q75s.append(float(q75))
    meds, q25s, q75s = np.array(meds), np.array(q25s), np.array(q75s)
    xpos = x_cats + (mi - (n_m - 1) / 2.0) * w_bar
    ax_a.bar(xpos, meds, width=w_bar, color=metric_palette[m], edgecolor='#222222', linewidth=0.28, label=metric_labels[m])
    ax_a.errorbar(xpos, meds, yerr=[np.maximum(0.0, meds - q25s), np.maximum(0.0, q75s - meds)], fmt='none', ecolor='#222222', elinewidth=0.45, zorder=4)

  ax_a.set_xticks(x_cats)
  ax_a.set_xticklabels([short_cat.get(c, c) for c in cat_order], fontsize=6.2, fontweight='bold', rotation=12, ha='right')
  ax_a.set_ylim(0, 78)
  ax_a.set_ylabel('Normalized |SHAP| share (%)', fontsize=6.6, fontweight='bold')
  ax_a.set_title('(a) Normalized Exact Group Shapley attribution across 12 metrics (median & IQR, N = 1,000 medoids)', fontsize=7.1, fontweight='bold', loc='left', pad=4)
  ax_a.legend(loc='upper right', ncol=4, fontsize=5.0, frameon=True, framealpha=0.95, edgecolor='#CFD8DC', handlelength=1.0, columnspacing=0.62)
  ax_a.grid(axis='x', visible=False)
  ax_a.grid(axis='y', linestyle='--', alpha=0.42, linewidth=0.45)
  sns.despine(ax=ax_a)

  panel_order = [
      ('NSE', '(b) NSE'), ('KGE', '(c) KGE'),
      ('1.01-year-2.0-window-recall', '(d) Recall (1y)'), ('2.0-year-2.0-window-recall', '(e) Recall (2y)'),
      ('5.0-year-2.0-window-recall', '(f) Recall (5y)'), ('10.0-year-2.0-window-recall', '(g) Recall (10y)'),
      ('20.0-year-2.0-window-recall', '(h) Recall (20y)'), ('1.01-year-2.0-window-precision', '(i) Prec. (1y)'),
      ('2.0-year-2.0-window-precision', '(j) Prec. (2y)'), ('5.0-year-2.0-window-precision', '(k) Prec. (5y)'),
      ('10.0-year-2.0-window-precision', '(l) Prec. (10y)'), ('20.0-year-2.0-window-precision', '(m) Prec. (20y)'),
  ]
  for idx_p, (m, ptitle) in enumerate(panel_order):
    row = 1 + idx_p // 4
    col = idx_p % 4
    ax = fig.add_subplot(gs[row, col])
    sub = df_cat_sum[df_cat_sum['metric'] == m].sort_values('median_abs_shap', ascending=True)
    y_pos = np.arange(len(sub))
    meds = sub['median_abs_shap'].values
    q25s = sub['q25_abs_shap'].values
    q75s = sub['q75_abs_shap'].values
    cols_p = [cat_palette.get(c, '#888888') for c in sub['category']]
    ax.barh(y_pos, meds, height=0.70, color=cols_p, edgecolor='#222222', linewidth=0.42)
    ax.errorbar(meds, y_pos, xerr=[np.maximum(0.0, meds - q25s), np.maximum(0.0, q75s - meds)], fmt='none', ecolor='#222222', elinewidth=0.65, capsize=1.5, zorder=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([short_cat.get(c, c) for c in sub['category']], fontsize=5.1, fontweight='bold')
    ax.set_xlim(0.0, max(1e-4, float(q75s.max()) * 1.16))
    ax.tick_params(axis='x', labelsize=5.1)
    ax.set_title(ptitle, fontsize=6.8, fontweight='bold', loc='left', pad=2.5)
    if row == 3:
      ax.set_xlabel('Median |Group Shapley| (IQR)', fontsize=5.3, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.4, linewidth=0.4)
    sns.despine(ax=ax)

  out_path = out_dir / 'Fig_6_exact_group_shapley.png'
  fig.savefig(out_path, dpi=SAVE_DPI)
  plt.close(fig)
  print(f'  Saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)', flush=True)
  return out_path


def main() -> None:
  parser = argparse.ArgumentParser(description='Generate main-text figures (Figs. 1–4, 6; optional Fig. 5).')
  parser.add_argument('--only', type=str, default='all', help='Comma-separated subset: fig1,fig2,fig3,fig4,fig5,fig6')
  parser.add_argument('--out-dir', type=str, default=str(DEFAULT_OUT_DIR))
  args = parser.parse_args()

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  targets = (
      {'fig1', 'fig2', 'fig3', 'fig4', 'fig5', 'fig6'}
      if args.only.strip().lower() == 'all'
      else {t.strip().lower() for t in args.only.split(',')}
  )

  world_all = load_world() if targets & {'fig1', 'fig3', 'fig4', 'fig5'} else None
  basins = None
  if targets & {'fig3', 'fig4'}:
    t4_idx = pd.read_parquet(TIER_DIR / 'tier4_all_publicly_knowable_121819.parquet', columns=['U_net_tree_phys']).index
    basins = load_non_fathom_basin_meta(pd.Index(norm_hybas_id(t4_idx)))

  if 'fig1' in targets:
    build_fig1(world_all, out_dir)
  if 'fig2' in targets:
    build_fig2(out_dir)
  if 'fig3' in targets:
    build_fig3(world_all, basins, out_dir)
  if 'fig4' in targets:
    build_fig4(world_all, basins, out_dir)
  if 'fig5' in targets:
    build_fig5_optional(world_all, out_dir)
  if 'fig6' in targets:
    build_fig6(out_dir)

  print('Main figure generation complete!', flush=True)


if __name__ == '__main__':
  main()
