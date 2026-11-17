#!/usr/bin/env python3
"""Generate main-text publication figures (Figs. 1–3, 6 public; optional Figs. 4–5 if Fathom present).

Produces symmetric 6-figure Science layout natively authored at Science full-page width
(184 mm = 7.244 in) with every rendered font >= 6.0 pt and consistent neutral-gray Greenland
masking across all global maps:
  - Fig_1_global_overview.png: 2x2 global gauge catalog (119,169 physical stations), never-passes-a-gauge
    map, continuous L12 fractional ungaugedness (with area-weighted distribution inset), and
    climate/landscape dissimilarity percentile (with area-weighted distribution inset).
  - Fig_2_scale_and_global_summary.png: 1x2 combining HydroSHEDS L1–L12 + 6-hydrofabric
    polygon-size scaling curve (a) with visual Table 1 multi-hydrofabric min–max range bars (b).
  - Fig_3_geography_and_alerting.png: 2x3 combining continental land/population dumbbells (a),
    aridity-zone ribbons (b), continental dissimilarity dumbbells (c), CAP-gap vs. alerting
    boxplots (d), and INFORM 2026 Lack of Coping Capacity (e) & Vulnerability (f) correlations
    with spatially adjusted Clifford–Richardson–Hémon / Dutilleul p_spatial.
  - Fig_4_global_flood_maps.png (OPTIONAL — licensed Fathom v3): 2x2 pure global flood maps
    twinning Fig. 1: (a) Total annual flood exposure E_0, (b) Forecasted today, (c) Cumulative
    gain from 119,169 existing gauges, (d) Cumulative gain from universal gauging on the
    exact same log color scale as (c).
  - Fig_5_exposure_synthesis.png (OPTIONAL — licensed Fathom v3): 2x2 quantitative synthesis
    replacing Table 2 + old Fig. 5b,c: (a) Exposure decomposition across Global / CAP-gap /
    Alerting comparing alpha = 1 (Recall) vs. alpha = CSI/R and direct FT+DA (no OOD clamp)
    vs. idealized OOD -> Q75 ceiling with 95% CIs, (b) Tiered cumulative CAP-gap & Global gains
    with 95% CIs, (c) Gauge mechanism breakdown in CAP-gap countries, and (d) Per-capita
    unforecasted exposure vs. INFORM 2026 Vulnerability with spatially adjusted p_spatial.
  - Fig_6_exact_group_shapley.png: Exact 2^7 = 128 Group Shapley attribution across 7 predictor
    categories (with ria_ha_usu in Topography) and 12 hydrological metrics (N = 1,000 medoids).
"""

from __future__ import annotations

import argparse
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.stats import mannwhitneyu, spearmanr
import seaborn as sns

from utils import cap_utils, file_paths, spatial_stats

warnings.filterwarnings('ignore')

COL_T1 = '#0072B2'   # Open datasets (32,369) / Established national alerting
COL_T2 = '#009E73'   # Confirmed open-license discharge (74,078)
COL_T3 = '#D55E00'   # All discoverable physical gauges (119,169) / CAP-gap country

SCIENCE_W = 184.0 / 25.4  # 7.2441 in (Science full-page width)
SAVE_DPI = 350

ROUND_SUBDIR = os.environ.get('GEO_BIAS_ROUND_SUBDIR', 'round2')
R2_DIR = CANON_DIR / ROUND_SUBDIR
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


def _set_pub_rc(font_size: float = 6.8) -> None:
  plt.rcParams.update({
      'font.family': 'sans-serif',
      'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
      'font.size': font_size,
      'axes.edgecolor': '#334155',
      'axes.linewidth': 0.60,
      'axes.unicode_minus': False,
      'xtick.color': '#1E293B',
      'ytick.color': '#1E293B',
      'text.color': '#0F172A',
      'figure.facecolor': 'white',
      'axes.facecolor': 'white',
      'savefig.facecolor': 'white',
  })


def _fmt_p_short(p: float) -> str:
  if not np.isfinite(p):
    return 'n/a'
  if p >= 0.05:
    return f'{p:.2f}'
  if p >= 0.001:
    return f'{p:.3f}'
  exp = int(np.floor(np.log10(max(p, 1e-300))))
  mant = p / (10.0 ** exp)
  return f'{mant:.1f}×10^{{{exp}}}'


def _fmt_p_tex(p: float) -> str:
  if not np.isfinite(p):
    return 'n/a'
  if p >= 0.05:
    return f'{p:.2f}'
  if p >= 0.001:
    return f'{p:.3f}'
  exp = int(np.floor(np.log10(max(p, 1e-300))))
  mant = p / (10.0 ** exp)
  return f'${mant:.1f} \\times 10^{{{exp}}}$'


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
  if 'in_tier_1' not in g.columns:
    g['in_tier_1'] = g['best_data_access_policy'].eq('open_academic_compilation') & g['best_observation_type'].isin({'daily_discharge', 'monthly_discharge'})
  if 'in_tier_2' not in g.columns:
    g['in_tier_2'] = (
        g['best_data_access_policy'].isin({'open_public_api_or_bulk', 'open_academic_compilation'})
        & g['best_observation_type'].isin({'daily_discharge', 'monthly_discharge'})
    )
  if 'in_tier_3' not in g.columns:
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
      REPO_DIR / '01_hydrofabric_ungaugedness' / 'data' / 'exposure' / 'hydrosheds_l12_worldpop_population.parquet',
      CANON_DIR / 'round2' / 'exposure' / 'global_basin_exposure_defended_r2.parquet',
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


def _overlay_greenland_gray(ax, world_no_ata: gpd.GeoDataFrame, zorder: float = 3.5) -> None:
  """Masks Greenland uniformly in neutral light gray (#E2E8F0) across all global maps."""
  grl = world_no_ata[world_no_ata['map_name'] == 'Greenland']
  if len(grl) > 0:
    grl.plot(ax=ax, color='#E2E8F0', edgecolor='#94A3B8', linewidth=0.25, zorder=zorder, aspect=None)


# ==============================================================================
# FIGURE 1: 2x2 Global Overview (Consistent Gray Greenland, All Fonts >= 6 pt)
# ==============================================================================
def build_fig1(world_all: gpd.GeoDataFrame, out_dir: Path) -> Path:
  print('[Fig 1] Generating 2x2 global overview map (Science width, gray Greenland, Addor insets)...', flush=True)
  _set_pub_rc(6.8)
  world = world_all[world_all['name'] != 'Antarctica'].copy()
  gauges = load_gauges_tiered()
  t4 = pd.read_parquet(TIER_DIR / 'tier4_all_discoverable.parquet')
  t4.index = norm_hybas_id(t4.index)

  # Load L12 basin areas (km^2) for area-weighted distribution insets
  topo_pq = CANON_DIR / 'attributes' / 'hybas_l12_topology_and_runoff.parquet'
  if topo_pq.exists():
    df_att = pd.read_parquet(topo_pq, columns=['HYBAS_ID', 'SUB_AREA'])
    df_att.index = norm_hybas_id(df_att['HYBAS_ID'])
    df_att = df_att[~df_att.index.duplicated(keep='first')].reindex(t4.index)
    sub_area = pd.to_numeric(df_att['SUB_AREA'], errors='coerce').fillna(0.0).values
  else:
    df_att = pd.read_csv(
        file_paths.HYBAS_ATTRIBUTES_PATH,
        usecols=['Unnamed: 0', 'HYDRO_ATLAS:SUB_AREA'],
        index_col='Unnamed: 0',
    )
    df_att.index = norm_hybas_id(df_att.index)
    df_att = df_att[~df_att.index.duplicated(keep='first')].reindex(t4.index)
    sub_area = pd.to_numeric(df_att['HYDRO_ATLAS:SUB_AREA'], errors='coerce').fillna(0.0).values

  # Exclude Greenland (hybas_9*) before computing dissimilarity percentile (vs gauged basins) & rasterizing
  not_grl = ~pd.Series(t4.index).str.startswith('hybas_9').values
  t4['dissim_pct'] = np.nan
  t4.loc[not_grl, 'dissim_pct'] = gauged_ref_pct(t4.loc[not_grl], pd.Index(t4.index[not_grl])).values
  t4['never_flag'] = np.where(not_grl, (t4['intergauge_reach_area_km2'] < 0).astype(np.float32), np.nan)
  t4['frac_pct'] = np.where(not_grl, (t4['U_net_tree_phys'] * 100.0).astype(np.float32), np.nan)

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
      if geom is None or geom.is_empty or str(hid).startswith('hybas_9'):
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
    print(f'  Rasterizing {n_feat:,} polygons from BasinATLAS_v10.gdb (excluding Greenland)...', flush=True)
    for start in range(0, n_feat, chunk_size):
      gdf = pyogrio.read_dataframe(
          gdb_path, layer='BasinATLAS_v10_lev12', columns=['HYBAS_ID'],
          skip_features=start, max_features=chunk_size,
      )
      ids = norm_hybas_id(gdf['HYBAS_ID'])
      matched_total += _rasterize_chunk(ids, gdf.geometry.values)
  elif geopq_root.exists():
    for reg in ['af', 'ar', 'as', 'au', 'eu', 'na', 'sa', 'si']:
      pq = geopq_root / reg / f'hybas_{reg}_lev12_v1c.geoparquet'
      if not pq.exists():
        continue
      gdf = gpd.read_parquet(pq, columns=['HYBAS_ID', 'geometry'])
      ids = norm_hybas_id(gdf['HYBAS_ID'])
      matched_total += _rasterize_chunk(ids, gdf.geometry.values)
  else:
    raise FileNotFoundError(
        f'Need BasinATLAS_v10.gdb at {gdb_path} or geoparquet archives at {geopq_root} to rasterize Fig. 1.'
    )

  print(f'  Total non-Greenland polygons rasterized: {matched_total:,}', flush=True)

  cmap_frac = mcolors.LinearSegmentedColormap.from_list(
      'ungauged_r2',
      [
          (0.00, '#1D78B4'), (0.20, '#6BAED6'), (0.45, '#EBF2D8'),
          (0.68, '#F6B15B'), (0.86, '#D95F0E'), (1.00, '#8C2D04'),
      ],
  )
  cmap_frac.set_bad(alpha=0.0)
  cmap_dissim = plt.get_cmap('YlOrBr').copy()
  cmap_dissim.set_bad(alpha=0.0)
  cmap_never = mcolors.ListedColormap(['#BDD7EB', '#C0392B'])
  cmap_never.set_bad(alpha=0.0)

  extent = [xmin, xmax, ymin, ymax]
  fig = plt.figure(figsize=(SCIENCE_W, 4.45), dpi=SAVE_DPI)
  gs = gridspec.GridSpec(
      2, 2, wspace=0.035, hspace=0.14,
      left=0.010, right=0.990, top=0.945, bottom=0.135,
  )

  # (a) 119,169 deduplicated physical stream gauges
  ax = fig.add_subplot(gs[0, 0])
  world.plot(ax=ax, color='#F1F5F9', edgecolor='#CBD5E1', linewidth=0.22, zorder=1, aspect=None)
  _overlay_greenland_gray(ax, world, zorder=1.5)
  g_t1 = gauges[gauges['in_tier_1'] == True]
  g_t2_only = gauges[(gauges['in_tier_2'] == True) & (gauges['in_tier_1'] != True)]
  g_t3_only = gauges[gauges['in_tier_2'] != True]
  ax.scatter(g_t3_only['longitude'], g_t3_only['latitude'], s=0.22, c=COL_T3, alpha=0.68, linewidths=0, rasterized=True, zorder=2)
  ax.scatter(g_t2_only['longitude'], g_t2_only['latitude'], s=0.28, c=COL_T2, alpha=0.76, linewidths=0, rasterized=True, zorder=3)
  ax.scatter(g_t1['longitude'], g_t1['latitude'], s=0.34, c=COL_T1, alpha=0.84, linewidths=0, rasterized=True, zorder=4)
  ax.set_xlim(xmin, xmax)
  ax.set_ylim(ymin, ymax)
  ax.set_aspect(1.0)
  ax.set_axis_off()
  n_ctry = int(gauges['country'].nunique()) if 'country' in gauges.columns else 163
  ax.set_title(f'(a) Discoverable stream gauges ({len(gauges):,}; {n_ctry} countries)', fontsize=7.2, fontweight='bold', loc='left', pad=3.5)
  handles_a = [
      Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_T1, markersize=4.5, label=f'Open datasets ({int(gauges["in_tier_1"].sum()):,})'),
      Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_T2, markersize=4.5, label=f'Open discharge ({int(gauges["in_tier_2"].sum()):,})'),
      Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_T3, markersize=4.5, label=f'All discoverable ({len(gauges):,})'),
  ]
  ax.legend(
      handles=handles_a, loc='lower left', frameon=True, facecolor='white', edgecolor='#CBD5E1',
      framealpha=0.95, fontsize=6.0, borderpad=0.30, handletextpad=0.28, labelspacing=0.20,
  )

  # Dynamic U_never land range across 6 hydrofabrics if available
  never_lbl = 'Never passes a gauge (32–36%)'
  m2_p = HYDRO_RESULTS_DIR / 'master' / 'master_2d_metrics.parquet'
  if m2_p.exists():
    m2_tmp = pd.read_parquet(m2_p)
    six_hf = ['hydrosheds_l12', 'hydrosheds_rivers', 'merit_basins', 'tdx_hydro', 'grit_v06_mainstem_tree', 'grit_v06_bifurcating_dag']
    c2_tmp = filter_canonical_2d(m2_tmp[(m2_tmp['domain_name'] == 'common_no_grl_ata') & (m2_tmp['hydrofabric'].isin(six_hf))])
    nv_vals = c2_tmp[(c2_tmp['metric'] == 'U_never') & (c2_tmp['weight'] == 'area') & (c2_tmp['tier'] == 'tier3_all_known')]['value'].astype(float).values * 100.0
    if len(nv_vals) > 0:
      never_lbl = f'Never passes a gauge ({nv_vals.min():.0f}–{nv_vals.max():.0f}%)'

  # (b) Never passes a gauge
  ax = fig.add_subplot(gs[0, 1])
  world.plot(ax=ax, color='#F1F5F9', edgecolor='none', zorder=1, aspect=None)
  ax.imshow(never_grid, origin='upper', extent=extent, cmap=cmap_never, vmin=-0.5, vmax=1.5, interpolation='nearest', zorder=2)
  _overlay_greenland_gray(ax, world, zorder=2.6)
  world.boundary.plot(ax=ax, color='#334155', linewidth=0.20, zorder=3, aspect=None)
  ax.set_xlim(xmin, xmax)
  ax.set_ylim(ymin, ymax)
  ax.set_aspect(1.0)
  ax.set_axis_off()
  ax.set_title('(b) Land whose runoff never passes any gauge', fontsize=7.2, fontweight='bold', loc='left', pad=3.5)
  handles_b = [
      Patch(facecolor='#C0392B', edgecolor='none', label=never_lbl),
      Patch(facecolor='#BDD7EB', edgecolor='none', label='Drains past ≥1 gauge'),
      Patch(facecolor='#E2E8F0', edgecolor='#94A3B8', linewidth=0.4, label='Greenland (excluded)'),
  ]
  ax.legend(
      handles=handles_b, loc='lower left', frameon=True, facecolor='white', edgecolor='#CBD5E1',
      framealpha=0.95, fontsize=6.0, borderpad=0.30, handletextpad=0.32, labelspacing=0.20,
  )

  def _add_addor_inset(
      ax_map,
      vals: np.ndarray,
      areas: np.ndarray,
      cmap_bar,
      vmin_c: float,
      vmax_c: float,
      xtick_lbls: list[str],
      vline: float | None = None,
  ) -> None:
    valid = np.isfinite(vals) & np.isfinite(areas) & (areas > 0)
    v_use = np.clip(vals[valid], 0.0, 100.0)
    w_use = areas[valid]
    bins = np.linspace(0.0, 100.0, 11)
    hist_w, _ = np.histogram(v_use, bins=bins, weights=w_use)
    pct_land = hist_w / max(float(w_use.sum()), 1e-9) * 100.0
    mids = 0.5 * (bins[:-1] + bins[1:])

    card = FancyBboxPatch(
        (0.012, 0.025), 0.252, 0.295,
        boxstyle='round,pad=0.006,rounding_size=0.012',
        transform=ax_map.transAxes,
        facecolor='white', edgecolor='#CBD5E1', linewidth=0.45, alpha=0.93, zorder=5.0,
    )
    ax_map.add_patch(card)
    ax_in = ax_map.inset_axes([0.074, 0.105, 0.178, 0.165], zorder=6.0)
    ax_in.set_facecolor('none')
    bar_cols = [cmap_bar(np.clip((m - vmin_c) / max(vmax_c - vmin_c, 1e-6), 0.0, 1.0)) for m in mids]
    ax_in.bar(mids, pct_land, width=8.8, color=bar_cols, edgecolor='#334155', linewidth=0.32, zorder=3)
    if vline is not None:
      ax_in.axvline(vline, color='#0F172A', linestyle='--', linewidth=0.65, zorder=4)
    ymax_in = float(np.ceil(max(pct_land.max() * 1.12, 10.0) / 10.0) * 10.0)
    ax_in.set_xlim(-2.0, 102.0)
    ax_in.set_ylim(0.0, ymax_in)
    ax_in.set_xticks([0, 50, 100])
    ax_in.set_xticklabels(xtick_lbls, fontsize=6.0)
    top_yt = int(round(pct_land.max() / 5.0) * 5)
    top_yt = max(10, min(int(ymax_in), top_yt))
    ax_in.set_yticks([0, top_yt])
    ax_in.set_yticklabels(['0', f'{top_yt}%'], fontsize=6.0)
    ax_in.tick_params(axis='both', length=1.8, pad=1.2, width=0.45)
    ax_in.set_title('% of land', fontsize=6.0, fontweight='bold', loc='left', pad=1.5)
    ax_in.spines['top'].set_visible(False)
    ax_in.spines['right'].set_visible(False)
    ax_in.spines['left'].set_linewidth(0.45)
    ax_in.spines['bottom'].set_linewidth(0.45)

  # (c) Continuous fractional ungauged watershed area + Addor distribution inset
  ax = fig.add_subplot(gs[1, 0])
  world.plot(ax=ax, color='#F1F5F9', edgecolor='none', zorder=1, aspect=None)
  im_c = ax.imshow(frac_grid, origin='upper', extent=extent, cmap=cmap_frac, vmin=0, vmax=100, interpolation='nearest', zorder=2)
  _overlay_greenland_gray(ax, world, zorder=2.6)
  world.boundary.plot(ax=ax, color='#475569', linewidth=0.18, zorder=3, aspect=None)
  ax.set_xlim(xmin, xmax)
  ax.set_ylim(ymin, ymax)
  ax.set_aspect(1.0)
  ax.set_axis_off()
  ax.set_title('(c) Fractional ungauged watershed area (HydroSHEDS L12)', fontsize=7.2, fontweight='bold', loc='left', pad=3.5)
  _add_addor_inset(ax, t4['frac_pct'].values, sub_area, cmap_frac, 0.0, 100.0, ['0%', '50%', '100%'])
  cax_c = fig.add_axes([0.055, 0.080, 0.40, 0.020])
  cb_c = fig.colorbar(im_c, cax=cax_c, orientation='horizontal')
  cb_c.set_ticks([0, 20, 40, 60, 80, 100])
  cb_c.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
  cb_c.ax.tick_params(labelsize=6.2, pad=1.5)
  cb_c.set_label('Fractional ungaugedness (0% gauged, 100% ungauged)', fontsize=6.3, fontweight='bold', labelpad=2.5)

  # (d) Climate & landscape dissimilarity percentile + Addor distribution inset
  ax = fig.add_subplot(gs[1, 1])
  world.plot(ax=ax, color='#F1F5F9', edgecolor='none', zorder=1, aspect=None)
  im_d = ax.imshow(dissim_grid, origin='upper', extent=extent, cmap=cmap_dissim, vmin=5, vmax=95, interpolation='nearest', zorder=2)
  _overlay_greenland_gray(ax, world, zorder=2.6)
  world.boundary.plot(ax=ax, color='#475569', linewidth=0.18, zorder=3, aspect=None)
  ax.set_xlim(xmin, xmax)
  ax.set_ylim(ymin, ymax)
  ax.set_aspect(1.0)
  ax.set_axis_off()
  ax.set_title('(d) Climate and landscape dissimilarity to gauged basins', fontsize=7.2, fontweight='bold', loc='left', pad=3.5)
  _add_addor_inset(ax, t4['dissim_pct'].values, sub_area, cmap_dissim, 5.0, 95.0, ['0', '50', '100'], vline=50.0)
  cax_d = fig.add_axes([0.545, 0.080, 0.40, 0.020])
  cb_d = fig.colorbar(im_d, cax=cax_d, orientation='horizontal', extend='both')
  cb_d.set_ticks([10, 30, 50, 70, 90])
  cb_d.set_ticklabels(['10th', '30th', '50th', '70th', '90th'])
  cb_d.ax.tick_params(labelsize=6.2, pad=1.5)
  cb_d.set_label('Dissimilarity percentile (0 = similar, 100 = dissimilar)', fontsize=6.3, fontweight='bold', labelpad=2.5)

  out_path = out_dir / 'Fig_1_global_overview.png'
  fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.04)
  plt.close(fig)
  print(f'  Saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)', flush=True)
  return out_path


# ==============================================================================
# FIGURE 2: Polygon-Size Scaling Curve (a) + Visual Table 1 Range Summary (b)
# ==============================================================================
def build_fig2(out_dir: Path) -> Path:
  print('[Fig 2] Generating scale dependence (a) + multi-hydrofabric summary (b)...', flush=True)
  _set_pub_rc(6.6)
  m2 = pd.read_parquet(HYDRO_RESULTS_DIR / 'master' / 'master_2d_metrics.parquet')
  m1 = pd.read_parquet(HYDRO_RESULTS_DIR / 'master' / 'master_1d_metrics.parquet')

  tiers = [
      ('tier1_open_compilations', COL_T1, 'Open datasets (32,369)'),
      ('tier2_public_discharge', COL_T2, 'Open discharge (74,078)'),
      ('tier3_all_known', COL_T3, 'All discoverable (119,169)'),
  ]
  hy_levels = [f'hydrosheds_l{i:02d}' for i in range(1, 13)]
  fine_fabrics = [
      ('hydrosheds_l12', 'HydroSHEDS L12', 'o', 4.6),
      ('grit_v06_mainstem_tree', 'GRIT v0.6 (tree)', 'D', 4.2),
      ('grit_v06_bifurcating_dag', 'GRIT v0.6 (DAG)', 'P', 4.6),
      ('merit_basins', 'MERIT-Basins', 's', 4.3),
      ('tdx_hydro', 'TDX-Hydro', '^', 4.9),
      ('hydrosheds_rivers', 'HydroSHEDS Rivers', '*', 6.4),
  ]

  def get_point(hf: str, tier: str) -> tuple[float, float]:
    sm = 'dag_width_weighted' if hf == 'grit_v06_bifurcating_dag' else 'outlet'
    rb = 'area_dag' if hf == 'grit_v06_bifurcating_dag' else 'area'
    sa = m2[(m2['hydrofabric'] == hf) & (m2['metric'] == 'U_net_Eq4') & (m2['domain_name'] == 'common_no_grl_ata') & (m2['tier'] == tier) & (m2['weight'] == 'area') & (m2['snapping_mode'] == sm) & (m2['ratio_basis'] == rb)]
    sc = m2[(m2['hydrofabric'] == hf) & (m2['metric'] == 'U_net_Eq4') & (m2['domain_name'] == 'common_no_grl_ata') & (m2['tier'] == tier) & (m2['weight'] == 'count') & (m2['snapping_mode'] == sm) & (m2['ratio_basis'] == rb)]
    mean_a = float(sa['total_weight'].iloc[0]) / float(sc['total_weight'].iloc[0])
    val = float(sa['value'].iloc[0]) * 100.0
    return mean_a, val

  fig = plt.figure(figsize=(SCIENCE_W, 4.15), dpi=SAVE_DPI)
  gs = gridspec.GridSpec(
      1, 2, width_ratios=[1.12, 1.00], wspace=0.38,
      left=0.075, right=0.985, top=0.845, bottom=0.125,
  )

  # --- Panel (a): Polygon-size scaling curve + fine-scale zoom inset ---
  ax = fig.add_subplot(gs[0, 0])
  hy_coords_t1 = {}
  for tier_key, col, _ in tiers:
    xs, ys = [], []
    for hf in hy_levels:
      ma, val = get_point(hf, tier_key)
      xs.append(ma)
      ys.append(val)
      if tier_key == 'tier1_open_compilations':
        hy_coords_t1[hf] = (ma, val)
    ax.plot(xs, ys, color=col, linestyle=':', linewidth=1.55, alpha=0.92, zorder=3)
    ax.scatter(xs[:-1], ys[:-1], s=14, facecolors='white', edgecolors=col, linewidths=0.95, zorder=4)

  level_offsets = {
      'hydrosheds_l01': (-4, 6), 'hydrosheds_l03': (5, 4),
      'hydrosheds_l05': (5, 4), 'hydrosheds_l07': (5, 4), 'hydrosheds_l09': (4, 4),
  }
  for hf, (dx, dy) in level_offsets.items():
    ma, val = hy_coords_t1[hf]
    lv = hf.replace('hydrosheds_l', 'L').replace('L0', 'L')
    ax.annotate(
        lv, (ma, val), xytext=(dx, dy), textcoords='offset points',
        fontsize=6.0, color='#0F172A', fontweight='bold', zorder=5,
        bbox=dict(boxstyle='round,pad=0.08', facecolor='white', edgecolor='none', alpha=0.88),
    )

  for tier_key, col, _ in tiers:
    for hf, _, mk, ms in fine_fabrics:
      ma, val = get_point(hf, tier_key)
      ax.plot(ma, val, marker=mk, markersize=ms, color=col, markeredgecolor='#0F172A', markeredgewidth=0.5, linestyle='None', zorder=6)

  rect = plt.Rectangle((11, 77), 165, 22.5, fill=True, facecolor='#F8FAFC', edgecolor='#94A3B8', linestyle='--', linewidth=0.75, zorder=1)
  ax.add_patch(rect)

  ax.set_xscale('log')
  ax.set_xlim(9, 2.8e7)
  ax.set_ylim(-2, 103)
  ax.tick_params(labelsize=6.2)
  ax.set_xlabel('Mean watershed polygon area (km², log scale)', fontsize=6.8, fontweight='bold')
  ax.set_ylabel('Fractionally ungauged land area (%)', fontsize=6.8, fontweight='bold')
  ax.set_title('(a) Polygon-size dependence (L1–L12 & 6 hydrofabrics)', fontsize=7.3, fontweight='bold', loc='left', pad=6)
  ax.grid(True, linestyle='--', alpha=0.35, color='#CBD5E1', linewidth=0.45)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  # Fine-scale zoom inset
  ax_in = ax.inset_axes([0.09, 0.11, 0.49, 0.45])
  ax_in.set_facecolor('white')
  for tier_key, col, _ in tiers:
    xs_sub, ys_sub = [], []
    for hf in ['hydrosheds_l09', 'hydrosheds_l10', 'hydrosheds_l11', 'hydrosheds_l12']:
      ma, val = get_point(hf, tier_key)
      xs_sub.append(ma)
      ys_sub.append(val)
    ax_in.plot(xs_sub, ys_sub, color=col, linestyle=':', linewidth=1.25, alpha=0.85, zorder=2)
    for hf, _, mk, ms in fine_fabrics:
      ma, val = get_point(hf, tier_key)
      ax_in.plot(ma, val, marker=mk, markersize=ms * 0.85, color=col, markeredgecolor='#0F172A', markeredgewidth=0.45, linestyle='None', zorder=5)

  ax_in.set_xscale('log')
  ax_in.set_xlim(12.5, 175)
  ax_in.set_ylim(75.5, 99.0)
  ax_in.set_xticks([15, 30, 60, 130])
  ax_in.set_xticklabels(['15', '30', '60', '130'], fontsize=6.0)
  ax_in.xaxis.set_minor_formatter(plt.NullFormatter())
  ax_in.tick_params(axis='x', which='minor', length=2.0)
  ax_in.set_yticks([80, 90, 98])
  ax_in.tick_params(axis='y', labelsize=6.0)
  ax_in.set_title('Fine-scale zoom (13–175 km²)', fontsize=6.1, fontweight='bold', loc='left', pad=2)
  ax_in.grid(True, linestyle='--', alpha=0.35, color='#CBD5E1', linewidth=0.4)
  for sp in ['top', 'right', 'left', 'bottom']:
    ax_in.spines[sp].set_color('#64748B')
    ax_in.spines[sp].set_linewidth(0.7)

  fab_handles = [
      Line2D([0], [0], color='none', marker=mk, markerfacecolor='#475569', markeredgecolor='#0F172A', markeredgewidth=0.5, markersize=ms * 0.82, label=name)
      for _, name, mk, ms in fine_fabrics
  ]
  ax.legend(
      handles=fab_handles, loc='upper right', bbox_to_anchor=(0.99, 0.99),
      frameon=True, facecolor='white', edgecolor='#CBD5E1', framealpha=0.95,
      fontsize=6.0, borderpad=0.30, handletextpad=0.25, labelspacing=0.20,
  )

  # --- Panel (b): Visual Table 1 Multi-Hydrofabric Range Summary ---
  ax_b = fig.add_subplot(gs[0, 1])
  six_hf = ['hydrosheds_l12', 'hydrosheds_rivers', 'merit_basins', 'tdx_hydro', 'grit_v06_mainstem_tree', 'grit_v06_bifurcating_dag']
  flow_4 = ['hydrosheds_l12', 'hydrosheds_rivers', 'merit_basins', 'tdx_hydro']
  flow_3 = ['hydrosheds_rivers', 'merit_basins', 'tdx_hydro']
  canon_2d = filter_canonical_2d(m2[(m2['domain_name'] == 'common_no_grl_ata') & (m2['hydrofabric'].isin(six_hf))])
  has_exp = 'exposure_eae' in set(canon_2d['weight'].unique())

  row_specs = [
      ('sec_never', 'Never passes a gauge', None, None, None),
      ('row', 'Land area', 'U_never', 'area', six_hf),
      ('row', 'River runoff', 'U_never', 'runoff', flow_4),
      ('row', 'Population', 'U_never', 'population', six_hf),
  ]
  if has_exp:
    row_specs.append(('row', 'Flood exposure', 'U_never', 'exposure_eae', six_hf))
  row_specs.extend([
      ('sec_frac', 'Fractional ungaugedness', None, None, None),
      ('row', 'Land area', 'U_net_Eq4', 'area', six_hf),
      ('row', 'River runoff', 'U_net_Eq4', 'runoff', flow_4),
      ('row', 'Population', 'U_net_Eq4', 'population', six_hf),
  ])
  if has_exp:
    row_specs.append(('row', 'Flood exposure', 'U_net_Eq4', 'exposure_eae', six_hf))
  row_specs.append(('row_1d', 'River flow (1D)', 'U_net_1d', 'Q_ds', flow_3))

  y_coords, y_labels = [], []
  y_cur = 0.0
  plotted_rows = []
  for item in reversed(row_specs):
    kind, lbl, met, wt, fabs = item
    if kind.startswith('sec_'):
      y_cur += 0.15
      plotted_rows.append((kind, y_cur, lbl, None))
      y_cur += 1.00
    else:
      plotted_rows.append((kind, y_cur, lbl, (met, wt, fabs)))
      y_coords.append(y_cur)
      y_labels.append(lbl)
      y_cur += 1.08

  tier_offsets = {'tier1_open_compilations': +0.23, 'tier2_public_discharge': 0.0, 'tier3_all_known': -0.23}
  tier_cols = {'tier1_open_compilations': COL_T1, 'tier2_public_discharge': COL_T2, 'tier3_all_known': COL_T3}

  def _fmt_rng(lo: float, hi: float) -> str:
    ilo = int(np.floor(lo + 0.5 + 1e-9))
    ihi = int(np.floor(hi + 0.5 + 1e-9))
    return f'{ilo}%' if ilo == ihi else f'{ilo}–{ihi}%'

  for kind, y, lbl, spec in plotted_rows:
    if kind.startswith('sec_'):
      ax_b.text(1.5, y, lbl, fontsize=6.6, fontweight='bold', color='#0F172A', va='center', ha='left')
      if kind == 'sec_frac':
        ax_b.axhline(y + 0.50, color='#94A3B8', linewidth=0.75, linestyle='-', zorder=1)
      continue

    met, wt, fabs = spec
    ax_b.barh(y, 100.0, left=0.0, height=0.76, color='#F8FAFC', edgecolor='none', zorder=0)
    ranges = {}
    for tk, _, _ in tiers:
      if kind == 'row':
        vals = canon_2d[(canon_2d['metric'] == met) & (canon_2d['weight'] == wt) & (canon_2d['hydrofabric'].isin(fabs)) & (canon_2d['tier'] == tk)]['value'].astype(float).values * 100.0
      else:
        vals = m1[(m1['domain_name'] == 'common_no_grl_ata') & (m1['hydrofabric'].isin(fabs)) & (m1['metric'] == met) & (m1['weight'] == wt) & (m1['ratio_basis'] == 'flow') & (m1['stratum'] == 'all_cutoff_free') & (m1['tier'] == tk)]['value'].astype(float).values * 100.0
      if len(vals) == 0:
        continue
      vmin, vmax = float(np.min(vals)), float(np.max(vals))
      ranges[tk] = (vmin, vmax)
      yy = y + tier_offsets[tk]
      col = tier_cols[tk]
      ax_b.plot([vmin, vmax], [yy, yy], color=col, linewidth=2.6, solid_capstyle='round', zorder=4)
      ax_b.scatter([vmin, vmax], [yy, yy], s=10, color=col, edgecolor='white', linewidth=0.35, zorder=5)

    # Crisp inline labels: Tier 3 (left of min) and Tier 1 (right of max, or left if >78)
    if 'tier3_all_known' in ranges:
      lo3, hi3 = ranges['tier3_all_known']
      txt3 = _fmt_rng(lo3, hi3)
      if lo3 >= 22:
        ax_b.text(lo3 - 2.2, y - 0.23, txt3, ha='right', va='center', fontsize=6.0, fontweight='bold', color='#9A3412', zorder=6)
      else:
        ax_b.text(hi3 + 2.2, y - 0.23, txt3, ha='left', va='center', fontsize=6.0, fontweight='bold', color='#9A3412', zorder=6)
    if 'tier1_open_compilations' in ranges:
      lo1, hi1 = ranges['tier1_open_compilations']
      txt1 = _fmt_rng(lo1, hi1)
      if hi1 <= 78:
        ax_b.text(hi1 + 2.2, y + 0.23, txt1, ha='left', va='center', fontsize=6.0, fontweight='bold', color='#0369A1', zorder=6)
      else:
        ax_b.text(lo1 - 2.2, y + 0.23, txt1, ha='right', va='center', fontsize=6.0, fontweight='bold', color='#0369A1', zorder=6)

  ax_b.set_yticks(y_coords)
  ax_b.set_yticklabels(y_labels, fontsize=6.3, fontweight='bold')
  ax_b.set_xlim(0, 102)
  ax_b.set_ylim(-0.65, y_cur - 0.10)
  ax_b.set_xticks([0, 20, 40, 60, 80, 100])
  ax_b.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=6.2)
  ax_b.set_xlabel('Ungauged share across 6 hydrofabrics (%)', fontsize=6.8, fontweight='bold')
  ax_b.set_title('(b) Global ungaugedness ranges across hydrofabrics', fontsize=7.3, fontweight='bold', loc='left', pad=6)
  ax_b.grid(True, axis='x', linestyle='--', alpha=0.35, color='#CBD5E1', linewidth=0.45)
  ax_b.spines['top'].set_visible(False)
  ax_b.spines['right'].set_visible(False)

  # Shared 3-tier top legend
  tier_handles = [
      Line2D([0], [0], color=col, linewidth=2.6, marker='o', markerfacecolor=col, markeredgecolor='white', markersize=4.8, label=lbl)
      for _, col, lbl in tiers
  ]
  fig.legend(
      handles=tier_handles, loc='upper center', bbox_to_anchor=(0.53, 0.985), ncol=3,
      frameon=True, facecolor='white', edgecolor='#CBD5E1', framealpha=0.98,
      fontsize=6.4, columnspacing=1.1, handletextpad=0.38, borderpad=0.35,
  )

  out_path = out_dir / 'Fig_2_scale_and_global_summary.png'
  fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.05)
  plt.close(fig)
  print(f'  Saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)', flush=True)
  return out_path


# ==============================================================================
# FIGURE 3: 2x3 Regional, Aridity, Dissimilarity, Alerting & INFORM Correlations
# ==============================================================================
def build_fig3(world_all: gpd.GeoDataFrame, basins: pd.DataFrame, out_dir: Path) -> Path:
  print('[Fig 3] Generating 2x3 geography, aridity, OOD, CAP-gap & INFORM figure...', flush=True)
  _set_pub_rc(6.5)
  c2cont = dict(zip(world_all['map_name'], world_all['continent']))
  c2iso = dict(zip(world_all['map_name'], world_all['iso_clean']))
  w_no_ata = world_all[world_all['name'] != 'Antarctica'].copy()
  w_cent = w_no_ata.dissolve(by='map_name')
  rp = w_cent.geometry.representative_point()
  c2lat = dict(zip(w_cent.index, rp.y))
  c2lon = dict(zip(w_cent.index, rp.x))

  no_grl = basins[basins['country_name'].notna() & (basins['country_name'] != 'Greenland')].copy()
  no_grl['continent'] = no_grl['country_name'].map(c2cont)

  t1 = pd.read_parquet(TIER_DIR / 'tier1_open_compilations.parquet')
  t2 = pd.read_parquet(TIER_DIR / 'tier2_public_discharge.parquet')
  t4 = pd.read_parquet(TIER_DIR / 'tier4_all_discoverable.parquet')
  for df in (t1, t2, t4):
    df.index = norm_hybas_id(df.index)

  m2 = pd.read_parquet(HYDRO_RESULTS_DIR / 'master' / 'master_2d_metrics.parquet')
  six = ['hydrosheds_l12', 'hydrosheds_rivers', 'merit_basins', 'tdx_hydro', 'grit_v06_mainstem_tree', 'grit_v06_bifurcating_dag']
  sub_frac = filter_canonical_2d(m2[(m2['hydrofabric'].isin(six)) & (m2['metric'] == 'U_net_Eq4')])

  conts = ['Africa', 'Asia', 'Oceania', 'South America', 'Europe', 'North America']
  cont_short = {'Africa': 'Africa', 'Asia': 'Asia', 'Oceania': 'Oceania', 'South America': 'S. America', 'Europe': 'Europe', 'North America': 'N. America'}
  arids = ['Humid', 'Dry sub-humid', 'Semi-arid', 'Arid', 'Hyper-arid']
  arid_labels = ['Humid', 'Dry sub-\nhumid', 'Semi-\narid', 'Arid', 'Hyper-\narid']

  # Country-level ungaugedness + INFORM 2026
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
  ctry['clat'] = ctry['country_name'].map(c2lat)
  ctry['clon'] = ctry['country_name'].map(c2lon)

  # Optional Fathom country-level flood-exposure ungaugedness
  t3_csv = CF_DIR / 'Table_S3_175territories_r2.csv'
  has_fathom_exp = False
  if t3_csv.exists():
    df_t3 = pd.read_csv(t3_csv)
    if 'flood_exp_ung_pct_t4' in df_t3.columns:
      exp_map = df_t3.set_index('country_name')['flood_exp_ung_pct_t4'].to_dict()
      ctry['U4_exp_pct'] = ctry['country_name'].map(exp_map)
      has_fathom_exp = ctry['U4_exp_pct'].notna().sum() >= 50

  df_inf = pd.read_csv(file_paths.RISK_FILE, skiprows=[0, 2])
  df_inf['ISO3'] = df_inf['ISO3'].astype(str).str.strip()
  df_inf['COUNTRY_CLEAN'] = df_inf['COUNTRY'].astype(str).str.strip().replace(file_paths.country_name_map)
  for col in ['VULNERABILITY', 'LACK OF COPING CAPACITY']:
    df_inf[col] = pd.to_numeric(df_inf[col], errors='coerce')
    iso_m = df_inf.dropna(subset=['ISO3', col]).drop_duplicates('ISO3').set_index('ISO3')[col]
    nam_m = df_inf.dropna(subset=['COUNTRY_CLEAN', col]).drop_duplicates('COUNTRY_CLEAN').set_index('COUNTRY_CLEAN')[col]
    ctry[col] = [iso_m.get(c2iso.get(c), nam_m.get(c, np.nan)) for c in ctry['country_name']]

  fig = plt.figure(figsize=(SCIENCE_W, 5.45), dpi=SAVE_DPI)
  gs = gridspec.GridSpec(
      2, 3, wspace=0.33, hspace=0.44,
      left=0.095, right=0.985, top=0.895, bottom=0.090,
  )
  y_pos = np.arange(len(conts))[::-1]

  # ---- (a) Continental land & population ungaugedness dumbbells ----
  ax_a = fig.add_subplot(gs[0, 0])
  for idx, c in enumerate(conts):
    y = y_pos[idx]
    ax_a.barh(y, 100.0, left=0.0, height=0.68, color='#F8FAFC', edgecolor='none', zorder=0)
    for dy, w_key, mk_scale, ls_conn in [(+0.17, 'area', 1.0, '-'), (-0.17, 'population', 0.85, '--')]:
      meds, mins, maxs = {}, {}, {}
      for tkey in ['tier1_open_compilations', 'tier2_public_discharge', 'tier3_all_known']:
        v = sub_frac[(sub_frac['domain_type'] == 'continent') & (sub_frac['domain_name'] == c) & (sub_frac['tier'] == tkey) & (sub_frac['weight'] == w_key)]['value'].values * 100.0
        meds[tkey], mins[tkey], maxs[tkey] = float(np.median(v)), float(np.min(v)), float(np.max(v))
      yy = y + dy
      ax_a.plot([mins['tier3_all_known'], maxs['tier3_all_known']], [yy, yy], color='#FED7AA', lw=2.6, solid_capstyle='round', zorder=2)
      ax_a.plot([meds['tier3_all_known'], meds['tier1_open_compilations']], [yy, yy], color='#475569' if w_key == 'area' else '#64748B', lw=1.35, ls=ls_conn, zorder=3)
      if w_key == 'area':
        ax_a.scatter(meds['tier1_open_compilations'], yy, s=22, color=COL_T1, marker='D', edgecolor='white', lw=0.45, zorder=5)
        ax_a.scatter(meds['tier2_public_discharge'], yy, s=19, color=COL_T2, marker='s', edgecolor='white', lw=0.45, zorder=6)
        ax_a.scatter(meds['tier3_all_known'], yy, s=26, color=COL_T3, marker='o', edgecolor='white', lw=0.50, zorder=7)
      else:
        ax_a.scatter(meds['tier1_open_compilations'], yy, s=20, facecolor='white', edgecolor=COL_T1, marker='D', lw=0.9, zorder=5)
        ax_a.scatter(meds['tier2_public_discharge'], yy, s=17, facecolor='white', edgecolor=COL_T2, marker='s', lw=0.9, zorder=6)
        ax_a.scatter(meds['tier3_all_known'], yy, s=24, facecolor='white', edgecolor=COL_T3, marker='o', lw=0.95, zorder=7)
  ax_a.set_yticks(y_pos)
  ax_a.set_yticklabels([cont_short[c] for c in conts], fontsize=6.2, fontweight='bold')
  ax_a.set_xlim(24, 104)
  ax_a.set_ylim(-0.55, len(conts) - 0.40)
  ax_a.set_xticks([30, 50, 70, 90])
  ax_a.set_xticklabels(['30%', '50%', '70%', '90%'], fontsize=6.1)
  ax_a.set_xlabel('Ungauged land / pop. (%)', fontsize=6.5, fontweight='bold')
  ax_a.set_title('(a) Continental land & pop.', fontsize=7.0, fontweight='bold', loc='left', pad=5)
  ax_a.legend(
      handles=[
          Line2D([0], [0], color='#475569', lw=1.35, marker='o', markersize=4.0, markerfacecolor=COL_T3, markeredgecolor='white', label='Land (filled)'),
          Line2D([0], [0], color='#64748B', lw=1.2, ls='--', marker='o', markersize=4.0, markerfacecolor='white', markeredgecolor=COL_T3, label='Pop. (open)'),
      ],
      loc='upper left', frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=6.0, borderpad=0.22, handlelength=1.4, labelspacing=0.18,
  )
  ax_a.grid(True, axis='x', linestyle='--', alpha=0.35, color='#CBD5E1', lw=0.45)
  ax_a.spines['top'].set_visible(False)
  ax_a.spines['right'].set_visible(False)

  # ---- (b) Aridity-zone slope ribbons ----
  ax_b = fig.add_subplot(gs[0, 1])
  xa = np.arange(len(arids))
  med_by_tier = {}
  for tkey, col, mk in [('tier1_open_compilations', COL_T1, 'D'), ('tier2_public_discharge', COL_T2, 's'), ('tier3_all_known', COL_T3, 'o')]:
    meds, mins, maxs = [], [], []
    for a in arids:
      v = sub_frac[(sub_frac['domain_type'] == 'aridity_zone') & (sub_frac['domain_name'] == a) & (sub_frac['tier'] == tkey) & (sub_frac['weight'] == 'area')]['value'].values * 100.0
      meds.append(np.median(v))
      mins.append(np.min(v))
      maxs.append(np.max(v))
    meds, mins, maxs = np.array(meds), np.array(mins), np.array(maxs)
    med_by_tier[tkey] = meds
    if tkey != 'tier2_public_discharge':
      ax_b.fill_between(xa, mins, maxs, color=col, alpha=0.14, zorder=1)
    ax_b.plot(xa, meds, color=col, lw=1.65, marker=mk, markersize=3.8, zorder=4)

  ax_b.text(0.05, med_by_tier['tier1_open_compilations'][0] + 1.5, f"{med_by_tier['tier1_open_compilations'][0]:.0f}%", fontsize=6.0, fontweight='bold', color='#0369A1', ha='left', va='bottom')
  ax_b.text(0.05, med_by_tier['tier3_all_known'][0] - 2.0, f"{med_by_tier['tier3_all_known'][0]:.0f}%", fontsize=6.0, fontweight='bold', color='#9A3412', ha='left', va='top')
  ax_b.text(3.96, med_by_tier['tier3_all_known'][4] - 2.0, f"{med_by_tier['tier3_all_known'][4]:.1f}%", fontsize=6.0, fontweight='bold', color='#9A3412', ha='right', va='top')
  ax_b.set_xticks(xa)
  ax_b.set_xticklabels(arid_labels, fontsize=6.0)
  ax_b.set_ylim(68, 103.5)
  ax_b.set_yticks([70, 80, 90, 100])
  ax_b.set_yticklabels(['70%', '80%', '90%', '100%'], fontsize=6.1)
  ax_b.set_ylabel('Ungauged land (%)', fontsize=6.5, fontweight='bold')
  ax_b.set_title('(b) Ungaugedness by aridity', fontsize=7.0, fontweight='bold', loc='left', pad=5)
  ax_b.grid(True, axis='y', linestyle='--', alpha=0.35, color='#CBD5E1', lw=0.45)
  ax_b.spines['top'].set_visible(False)
  ax_b.spines['right'].set_visible(False)

  # ---- (c) Continental OOD dissimilarity dumbbells ----
  ax_c = fig.add_subplot(gs[0, 2])
  pct_t1 = gauged_ref_pct(t1, no_grl.index)
  pct_t2 = gauged_ref_pct(t2, no_grl.index)
  pct_t4 = gauged_ref_pct(t4, no_grl.index)
  ax_c.axvline(50.0, color='#475569', linestyle='--', lw=0.9, zorder=1)
  ax_c.text(51.2, len(conts) - 0.25, 'Gauged med.', fontsize=6.0, fontweight='bold', color='#475569', ha='left', va='bottom')
  for idx, c in enumerate(conts):
    y = y_pos[idx]
    m = (no_grl['continent'] == c).values
    wa = no_grl['sub_area'].values[m]
    d1, d2, d3 = w_quantile(pct_t1.values[m], wa, 0.50), w_quantile(pct_t2.values[m], wa, 0.50), w_quantile(pct_t4.values[m], wa, 0.50)
    q25, q75 = w_quantile(pct_t4.values[m], wa, 0.25), w_quantile(pct_t4.values[m], wa, 0.75)
    ax_c.barh(y, 100.0, left=0.0, height=0.50, color='#F8FAFC', edgecolor='none', zorder=0)
    ax_c.plot([q25, q75], [y, y], color='#FED7AA', lw=3.2, solid_capstyle='round', zorder=2)
    ax_c.plot([d3, d1], [y, y], color='#475569', lw=1.45, zorder=3)
    ax_c.scatter(d1, y, s=24, color=COL_T1, marker='D', edgecolor='white', lw=0.45, zorder=5)
    ax_c.scatter(d2, y, s=21, color=COL_T2, marker='s', edgecolor='white', lw=0.45, zorder=6)
    ax_c.scatter(d3, y, s=28, color=COL_T3, marker='o', edgecolor='white', lw=0.50, zorder=7)
  ax_c.set_yticks(y_pos)
  ax_c.set_yticklabels([cont_short[c] for c in conts], fontsize=6.1, fontweight='bold')
  ax_c.set_xlim(32, 102)
  ax_c.set_ylim(-0.55, len(conts) + 0.15)
  ax_c.set_xticks([40, 60, 80, 100])
  ax_c.set_xticklabels(['40th', '60th', '80th', '100th'], fontsize=6.1)
  ax_c.set_xlabel('Dissimilarity percentile', fontsize=6.5, fontweight='bold')
  ax_c.set_title('(c) Climate/landscape OOD', fontsize=7.0, fontweight='bold', loc='left', pad=5)
  ax_c.grid(True, axis='x', linestyle='--', alpha=0.35, color='#CBD5E1', lw=0.45)
  ax_c.spines['top'].set_visible(False)
  ax_c.spines['right'].set_visible(False)

  # ---- (d) Country-level ungaugedness in Alerting (60) vs. CAP-gap (114) ----
  ax_d = fig.add_subplot(gs[1, 0])
  w_df = ctry[ctry['has_system'] == True]
  g_df = ctry[ctry['has_system'] == False]
  groups_d = [('Land', 'U4_area_pct'), ('Pop.', 'U4_pop_pct')]
  if has_fathom_exp:
    groups_d.append(('Flood\nexp.', 'U4_exp_pct'))

  box_data, box_cols, positions, xticks_d, xlabels_d = [], [], [], [], []
  rng = np.random.default_rng(42)
  for gi, (glbl, col_m) in enumerate(groups_d):
    base_x = gi * 1.85
    p1, p2 = base_x + 0.35, base_x + 0.95
    v_sys = w_df[col_m].dropna().values
    v_gap = g_df[col_m].dropna().values
    box_data.extend([v_sys, v_gap])
    box_cols.extend([COL_T1, COL_T3])
    positions.extend([p1, p2])
    xticks_d.append(0.5 * (p1 + p2))
    xlabels_d.append(glbl)
    mw_p = mannwhitneyu(v_gap, v_sys).pvalue
    ax_d.plot([p1, p2], [102.5, 102.5], color='#334155', lw=0.65)
    ax_d.text(0.5 * (p1 + p2), 104.5, f'p={_fmt_p_tex(mw_p)}', ha='center', va='bottom', fontsize=5.6, fontweight='bold', color='#1E293B')

  bp = ax_d.boxplot(
      box_data, positions=positions, widths=0.44, patch_artist=True, showfliers=False,
      medianprops=dict(color='#0F172A', linewidth=1.2),
      whiskerprops=dict(color='#334155', linewidth=0.7),
      capprops=dict(color='#334155', linewidth=0.7),
      boxprops=dict(linewidth=0.65, edgecolor='#0F172A'),
  )
  for patch, col, pos, vals in zip(bp['boxes'], box_cols, positions, box_data):
    patch.set_facecolor(col)
    patch.set_alpha(0.75)
    ax_d.scatter(np.full_like(vals, pos) + rng.uniform(-0.11, 0.11, size=len(vals)), vals, s=4.0, color=col, alpha=0.38, edgecolors='none', zorder=2)

  ax_d.set_xticks(xticks_d)
  ax_d.set_xticklabels(xlabels_d, fontsize=6.2, fontweight='bold')
  ax_d.set_ylim(-4, 117)
  ax_d.set_yticks([0, 25, 50, 75, 100])
  ax_d.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=6.1)
  ax_d.set_ylabel('National ungaugedness (%)', fontsize=6.5, fontweight='bold')
  ax_d.set_title('(d) Alerting vs. CAP-gap', fontsize=7.0, fontweight='bold', loc='left', pad=5)
  ax_d.grid(True, axis='y', linestyle='--', alpha=0.35, color='#CBD5E1', lw=0.45)
  ax_d.spines['top'].set_visible(False)
  ax_d.spines['right'].set_visible(False)

  # ---- (e) & (f) INFORM 2026 Lack of Coping Capacity & Vulnerability with p_spatial ----
  sub_inf = ctry.dropna(subset=['VULNERABILITY', 'LACK OF COPING CAPACITY', 'U4_pop_pct', 'clat', 'clon']).copy()
  n_sys_inf = int((sub_inf['has_system'] == True).sum())
  n_gap_inf = int((sub_inf['has_system'] == False).sum())

  for slot, col_x, xlabel, title in [
      (gs[1, 1], 'LACK OF COPING CAPACITY', 'INFORM Lack of Coping (0–10)', '(e) Ungaugedness vs. coping gap'),
      (gs[1, 2], 'VULNERABILITY', 'INFORM Vulnerability (0–10)', '(f) Ungaugedness vs. vulnerability'),
  ]:
    ax = fig.add_subplot(slot)
    for has_sys, col, mk in [(True, COL_T1, 'o'), (False, COL_T3, '^')]:
      s = sub_inf[sub_inf['has_system'] == has_sys]
      ax.scatter(s[col_x], s['U4_pop_pct'], s=13, c=col, marker=mk, alpha=0.72, edgecolors='#0F172A', linewidths=0.30, zorder=3)

    # Trend lines & spatially adjusted p_spatial for Pop, Land, and optional Flood exposure
    stat_lines = []
    trend_specs = [('Pop', 'U4_pop_pct', '#0F172A', '-'), ('Land', 'U4_area_pct', '#475569', '--')]
    if has_fathom_exp:
      trend_specs.append(('Flood', 'U4_exp_pct', '#991B1B', ':'))

    for short_lbl, ycol, lcol, lstyle in trend_specs:
      ss = sub_inf.dropna(subset=[col_x, ycol, 'clat', 'clon'])
      coef = np.polyfit(ss[col_x].values, ss[ycol].values, 1)
      xg = np.linspace(ss[col_x].min(), ss[col_x].max(), 120)
      yg = np.polyval(coef, xg)
      keep = (yg >= 0) & (yg <= 100)
      ax.plot(xg[keep], yg[keep], color=lcol, linestyle=lstyle, linewidth=1.15, zorder=4, label=short_lbl)
      sp = spatial_stats.spatial_spearman_neff(ss[col_x].values, ss[ycol].values, ss['clat'].values, ss['clon'].values, k=5)
      stat_lines.append(f"{short_lbl}: $r_s$={sp['spearman_r']:+.2f} ($p_{{sp}}$={_fmt_p_tex(sp['p_spatial'])})")

    ax.text(
        0.97, 0.04, '\n'.join(stat_lines),
        transform=ax.transAxes, ha='right', va='bottom', fontsize=6.0,
        bbox=dict(boxstyle='round,pad=0.20', facecolor='#F8FAFC', edgecolor='#CBD5E1', alpha=0.94),
        zorder=6,
    )
    ax.set_xlim(0.2, 9.8)
    ax.set_ylim(-4, 106)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=6.1)
    ax.tick_params(axis='x', labelsize=6.1)
    ax.set_xlabel(xlabel, fontsize=6.5, fontweight='bold')
    ax.set_title(title, fontsize=7.0, fontweight='bold', loc='left', pad=5)
    ax.grid(True, linestyle='--', alpha=0.35, color='#CBD5E1', lw=0.45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

  # Top legend (gauge tiers) + bottom legend (alerting status & trend lines)
  leg_top = [
      Line2D([0], [0], marker='D', color='none', markerfacecolor=COL_T1, markeredgecolor='white', markersize=5.2, label='Open datasets (32,369)'),
      Line2D([0], [0], marker='s', color='none', markerfacecolor=COL_T2, markeredgecolor='white', markersize=5.0, label='Open discharge (74,078)'),
      Line2D([0], [0], marker='o', color='none', markerfacecolor=COL_T3, markeredgecolor='white', markersize=5.4, label='All discoverable (119,169)'),
  ]
  fig.legend(
      handles=leg_top, loc='upper center', bbox_to_anchor=(0.53, 0.990), ncol=3,
      frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=6.2,
      columnspacing=1.0, handletextpad=0.30, borderpad=0.30,
  )
  leg_bot = [
      Line2D([0], [0], marker='o', color='none', markerfacecolor=COL_T1, markeredgecolor='#0F172A', markersize=4.8, label=f'Alerting system ({n_sys_inf})'),
      Line2D([0], [0], marker='^', color='none', markerfacecolor=COL_T3, markeredgecolor='#0F172A', markersize=4.8, label=f'CAP-gap ({n_gap_inf} in INFORM)'),
      Line2D([0], [0], color='#0F172A', lw=1.2, ls='-', label='Pop. fit'),
      Line2D([0], [0], color='#475569', lw=1.2, ls='--', label='Land fit'),
  ]
  if has_fathom_exp:
    leg_bot.append(Line2D([0], [0], color='#991B1B', lw=1.2, ls=':', label='Flood fit'))
  fig.legend(
      handles=leg_bot, loc='lower center', bbox_to_anchor=(0.53, -0.005), ncol=len(leg_bot),
      frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=6.0,
      columnspacing=0.9, handletextpad=0.30, borderpad=0.25,
  )

  out_path = out_dir / 'Fig_3_geography_and_alerting.png'
  fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.05)
  plt.close(fig)
  print(f'  Saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)', flush=True)
  return out_path


# ==============================================================================
# FIGURE 4 (OPTIONAL — Fathom v3): 2x2 Global Flood Exposure & Gain Maps
# ==============================================================================
def build_fig4_optional(world_all: gpd.GeoDataFrame, out_dir: Path) -> Path | None:
  basin_pq = CF_DIR / 'basin_level_exposure_counterfactuals_r2.parquet'
  sum_csv = CF_DIR / 'multi_mechanism_tiered_exposure_r2.csv'
  if not (basin_pq.exists() and sum_csv.exists()):
    print(
        f'[Fig 4] SKIPPED: Licensed Fathom v3 flood-exposure outputs not found under {CF_DIR}.',
        flush=True,
    )
    return None

  print('[Fig 4] Generating 2x2 global flood exposure & gauge counterfactual maps...', flush=True)
  _set_pub_rc(6.8)
  world = world_all[world_all['name'] != 'Antarctica'].copy()
  df_b = pd.read_parquet(basin_pq)
  df_sum = pd.read_csv(sum_csv).set_index(['tier_key', 'mechanism'])

  pts = df_b.dropna(subset=['lat', 'lon']).copy()
  pts = pts[pts['country_name'] != 'Greenland'].copy()

  col_121k = 'dnet__tier4_all_discoverable__111_all_three_combined_snap'
  col_univ = 'dnet__tier5_universal_idealized_ceiling__111_all_three_combined_snap'

  tot_m = float(df_sum.loc[('tier1a_active_post2016_ccby_8811', '000_global_train_joint'), 'E_total_global']) / 1e6
  w_today_m = float(df_sum.loc[('tier1a_active_post2016_ccby_8811', '000_global_train_joint'), 'warned_global']) / 1e6
  g_121k_m = float(df_sum.loc[('tier4_all_discoverable', '111_all_three_combined_snap'), 'delta_net_global']) / 1e6
  g_121k_cap = float(df_sum.loc[('tier4_all_discoverable', '111_all_three_combined_snap'), 'delta_net_nosys115']) / 1e6
  g_u_ceil_m = float(df_sum.loc[('tier5_universal_idealized_ceiling', '111_all_three_combined_snap'), 'delta_net_global']) / 1e6
  g_u_ceil_cap = float(df_sum.loc[('tier5_universal_idealized_ceiling', '111_all_three_combined_snap'), 'delta_net_nosys115']) / 1e6
  if ('tier5_universal_no_ood_clamp', '111_all_three_combined_snap') in df_sum.index:
    g_u_nood_m = float(df_sum.loc[('tier5_universal_no_ood_clamp', '111_all_three_combined_snap'), 'delta_net_global']) / 1e6
    u_lbl = f'+{g_u_nood_m:.2f}–{g_u_ceil_m:.2f}M persons/yr'
  else:
    u_lbl = f'+{g_u_ceil_m:.2f}M persons/yr'

  xmin, xmax, ymin, ymax = -180.0, 180.0, -58.0, 84.0
  cmap_exp = mcolors.LinearSegmentedColormap.from_list('YlGnBu_trunc', plt.get_cmap('YlGnBu')(np.linspace(0.20, 1.0, 256)))
  cmap_gain = plt.get_cmap('YlOrRd')
  norm_exp = mcolors.LogNorm(vmin=1.0, vmax=10000.0)
  norm_gain = mcolors.LogNorm(vmin=0.5, vmax=5000.0)

  fig = plt.figure(figsize=(SCIENCE_W, 5.15), dpi=SAVE_DPI)
  gs = gridspec.GridSpec(
      2, 2, wspace=0.035, hspace=0.40,
      left=0.010, right=0.990, top=0.955, bottom=0.105,
  )

  def _plot_panel(ax, col, cmap, norm, title):
    world.plot(ax=ax, color='#F1F5F9', edgecolor='none', zorder=1, aspect=None)
    p = pts[pts[col] > norm.vmin * 0.25].sort_values(col, ascending=True)
    z = pts[pts[col] <= norm.vmin * 0.25]
    ax.scatter(z['lon'], z['lat'], s=0.04, marker='.', color='#E2E8F0', linewidths=0, rasterized=True, zorder=2)
    sc = ax.scatter(
        p['lon'], p['lat'], c=np.clip(p[col].values, norm.vmin, norm.vmax),
        s=0.14, marker='.', cmap=cmap, norm=norm, linewidths=0, rasterized=True, zorder=2.5,
    )
    _overlay_greenland_gray(ax, world, zorder=3.0)
    world.boundary.plot(ax=ax, color='#475569', linewidth=0.18, zorder=3.5, aspect=None)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect(1.0)
    ax.set_axis_off()
    ax.set_title(title, fontsize=6.9, fontweight='bold', loc='left', pad=3.0)
    return sc

  sc_a = _plot_panel(fig.add_subplot(gs[0, 0]), 'E_total', cmap_exp, norm_exp, f'(a) Annual flood exposure ({tot_m:.2f}M persons/yr)')
  _plot_panel(fig.add_subplot(gs[0, 1]), 'warned_t1a', cmap_exp, norm_exp, f'(b) Forecasted today by open ML ({w_today_m:.2f}M persons/yr)')
  sc_c = _plot_panel(fig.add_subplot(gs[1, 0]), col_121k, cmap_gain, norm_gain, f'(c) Gain: 119,169 existing gauges (+{g_121k_m:.2f}M persons/yr)')
  _plot_panel(fig.add_subplot(gs[1, 1]), col_univ, cmap_gain, norm_gain, f'(d) Gain: universal gauging ({u_lbl})')

  # Shared horizontal colorbar for row 1 (exposure & forecasted today)
  cax1 = fig.add_axes([0.23, 0.545, 0.54, 0.017])
  cb1 = fig.colorbar(sc_a, cax=cax1, orientation='horizontal', extend='both')
  cb1.set_ticks([1, 10, 100, 1000, 10000])
  cb1.set_ticklabels(['1', '10', '100', '1k', '10k+'])
  cb1.ax.tick_params(labelsize=6.1, pad=1.2)
  cb1.set_label('Expected annual flood exposure or forecasted exposure (persons/yr, log scale)', fontsize=6.2, fontweight='bold', labelpad=2.0)

  # Shared horizontal colorbar for row 2 (exact same log scale for c and d!)
  cax2 = fig.add_axes([0.23, 0.048, 0.54, 0.017])
  cb2 = fig.colorbar(sc_c, cax=cax2, orientation='horizontal', extend='both')
  cb2.set_ticks([1, 10, 100, 1000, 5000])
  cb2.set_ticklabels(['1', '10', '100', '1k', '5k+'])
  cb2.ax.tick_params(labelsize=6.1, pad=1.2)
  cb2.set_label('Cumulative forecasted exposure gain above today (persons/yr; identical log scale in c–d)', fontsize=6.2, fontweight='bold', labelpad=2.0)

  out_path = out_dir / 'Fig_4_global_flood_maps.png'
  fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.04)
  plt.close(fig)
  print(f'  Saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)', flush=True)
  return out_path


# ==============================================================================
# FIGURE 5 (OPTIONAL — Fathom v3): Full Quantitative Exposure Synthesis
# ==============================================================================
def build_fig5_optional(world_all: gpd.GeoDataFrame, out_dir: Path) -> Path | None:
  sum_csv = CF_DIR / 'multi_mechanism_tiered_exposure_r2.csv'
  t3_csv = CF_DIR / 'Table_S3_175territories_r2.csv'
  if not (sum_csv.exists() and t3_csv.exists()):
    print(
        f'[Fig 5] SKIPPED: Licensed Fathom v3 exposure outputs not found under {CF_DIR}.',
        flush=True,
    )
    return None

  print('[Fig 5] Generating full quantitative flood-exposure synthesis (A1 + A2 + A3 + A5)...', flush=True)
  _set_pub_rc(6.5)
  df_sum = pd.read_csv(sum_csv).set_index(['tier_key', 'mechanism'])
  df_c = pd.read_csv(t3_csv)
  boot_csv = CF_DIR / 'bootstrap_table1_fig3_ci95_r2.csv'
  df_ci = pd.read_csv(boot_csv) if boot_csv.exists() else None
  sp_json = CF_DIR / 'spatial_neff_stats_r3.json'
  sp_stats = json.loads(sp_json.read_text()) if sp_json.exists() else {}

  has_nood = ('tier5_universal_no_ood_clamp', '111_all_three_combined_snap') in df_sum.index

  def get_v(tk: str, mk: str, col: str) -> float:
    return float(df_sum.loc[(tk, mk), col]) / 1e6

  def get_ci_err(scen: str, metric: str, pt_m: float) -> tuple[float, float]:
    if df_ci is None:
      return 0.0, 0.0
    sub = df_ci[(df_ci['scenario'] == scen) & (df_ci['metric'] == metric)]
    if sub.empty:
      return 0.0, 0.0
    lo = float(sub.iloc[0]['boot_p025']) / 1e6
    hi = float(sub.iloc[0]['boot_p975']) / 1e6
    return max(0.0, pt_m - lo), max(0.0, hi - pt_m)

  fig = plt.figure(figsize=(SCIENCE_W, 5.85), dpi=SAVE_DPI)
  gs = gridspec.GridSpec(
      2, 2, width_ratios=[1.14, 1.00], height_ratios=[1.05, 1.00],
      wspace=0.33, hspace=0.40, left=0.115, right=0.980, top=0.905, bottom=0.085,
  )

  # ---- Panel (a): Exposure Decomposition across Global, CAP-gap (114), Alerting (60)
  #                 comparing alpha=1 (Recall) vs alpha=CSI/R (FA-discounted) ----
  ax_a = fig.add_subplot(gs[0, 0])
  regions = [
      ('Global\n(29.5M)', 'global', 'E_total_global'),
      ('CAP-gap\n(9.2M)', 'nosys115', 'E_total_nosys115'),
      ('Alerting\n(20.0M)', 'sys60', 'E_total_sys60'),
  ]
  bar_y = []
  bar_lbls = []
  y_cur = 0.0
  # Plot top-to-bottom: Global, CAP-gap, Alerting
  for r_idx, (r_lbl, r_suf, e_col) in enumerate(reversed(regions)):
    et = get_v('tier1a_active_post2016_ccby_8811', '000_global_train_joint', e_col)
    for m_idx, (mode_lbl, w_pref, d_pref, ci_pref) in enumerate([
        ('CSI/R', 'warned_csi_', 'delta_net_csi_', 'csi_'),
        ('Recall', 'warned_', 'delta_net_', ''),
    ]):
      yy = y_cur + m_idx * 0.44
      bar_y.append(yy)
      bar_lbls.append(f'{r_lbl.split(chr(10))[0]} ({mode_lbl})')
      w0 = get_v('tier1a_active_post2016_ccby_8811', '000_global_train_joint', f'{w_pref}{r_suf}')
      d121 = max(0.0, get_v('tier4_all_discoverable', '111_all_three_combined_snap', f'{d_pref}{r_suf}'))
      if has_nood:
        d_nood = max(d121, get_v('tier5_universal_no_ood_clamp', '111_all_three_combined_snap', f'{d_pref}{r_suf}'))
      else:
        d_nood = d121
      d_ceil = max(d_nood, get_v('tier5_universal_idealized_ceiling', '111_all_three_combined_snap', f'{d_pref}{r_suf}'))

      s_today = 100.0 * w0 / et
      s_share = 100.0 * d121 / et
      s_new_dir = 100.0 * max(0.0, d_nood - d121) / et
      s_new_ood = 100.0 * max(0.0, d_ceil - d_nood) / et
      s_other = max(0.0, 100.0 - (s_today + s_share + s_new_dir + s_new_ood))

      h = 0.35
      ax_a.barh(yy, s_today, height=h, color='#2563EB', edgecolor='white', lw=0.45, zorder=3)
      ax_a.barh(yy, s_share, left=s_today, height=h, color='#0D9488', edgecolor='white', lw=0.45, zorder=3)
      ax_a.barh(yy, s_new_dir, left=s_today + s_share, height=h, color='#D97706', edgecolor='white', lw=0.45, zorder=3)
      ax_a.barh(yy, s_new_ood, left=s_today + s_share + s_new_dir, height=h, color='#FDE68A', edgecolor='#B45309', hatch='///', lw=0.45, zorder=3)
      ax_a.barh(yy, s_other, left=s_today + s_share + s_new_dir + s_new_ood, height=h, color='#E2E8F0', edgecolor='white', lw=0.45, zorder=2)

      # Inline % text for today and remaining
      if s_today >= 12.0:
        ax_a.text(s_today * 0.5, yy, f'{s_today:.0f}%', ha='center', va='center', fontsize=6.0, fontweight='bold', color='white', zorder=5)
      tot_cov = s_today + s_share + s_new_dir + s_new_ood
      ax_a.text(tot_cov + 1.5, yy, f'{tot_cov:.0f}%', ha='left', va='center', fontsize=6.0, fontweight='bold', color='#1E293B', zorder=5)
    y_cur += 1.25

  ax_a.axvline(50.0, color='#64748B', ls=':', lw=0.8, zorder=1)
  ax_a.set_yticks(bar_y)
  ax_a.set_yticklabels(bar_lbls, fontsize=6.1, fontweight='bold')
  ax_a.set_xlim(0, 101)
  ax_a.set_xticks([0, 20, 40, 60, 80, 100])
  ax_a.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=6.1)
  ax_a.set_xlabel('Share of expected annual flood exposure (%)', fontsize=6.6, fontweight='bold')
  ax_a.set_title('(a) Exposure decomposition: Recall vs. CSI/R', fontsize=7.1, fontweight='bold', loc='left', pad=5)
  ax_a.spines['top'].set_visible(False)
  ax_a.spines['right'].set_visible(False)

  # Top shared legend for panel (a) segments
  seg_handles = [
      Patch(facecolor='#2563EB', edgecolor='white', label='Today'),
      Patch(facecolor='#0D9488', edgecolor='white', label='Existing 119.2k'),
      Patch(facecolor='#D97706', edgecolor='white', label='+New gauges (local)'),
      Patch(facecolor='#FDE68A', edgecolor='#B45309', hatch='///', label='+Under-sampled climates'),
      Patch(facecolor='#E2E8F0', edgecolor='#CBD5E1', label='Other limits'),
  ]
  fig.legend(
      handles=seg_handles, loc='upper center', bbox_to_anchor=(0.54, 0.990), ncol=5,
      frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=6.0,
      columnspacing=0.65, handletextpad=0.28, borderpad=0.28,
  )

  # ---- Panel (b): Tiered CAP-gap Exposure Gain (M/yr) with 95% CIs (Recall vs CSI/R) ----
  ax_b = fig.add_subplot(gs[0, 1])
  stage_specs = [
      ('tier1_open_compilations', '32.4k\nopen'),
      ('tier2_public_discharge', '74.1k\nopen Q'),
      ('tier4_all_discoverable', '119.2k\nexist.'),
  ]
  if has_nood:
    stage_specs.append(('tier5_universal_no_ood_clamp', 'Univ.\n(local)'))
  stage_specs.append(('tier5_universal_idealized_ceiling', 'Univ.\n(upper)'))

  xb = np.arange(len(stage_specs))
  bw = 0.36
  rec_v, rec_lo, rec_hi = [], [], []
  csi_v, csi_lo, csi_hi = [], [], []
  for tk, _ in stage_specs:
    vr = get_v(tk, '111_all_three_combined_snap', 'delta_net_nosys115')
    vc = get_v(tk, '111_all_three_combined_snap', 'delta_net_csi_nosys115')
    lr, hr = get_ci_err(f'{tk}__111_combined', 'delta_nosys115', vr)
    lc, hc = get_ci_err(f'{tk}__111_combined', 'delta_csi_nosys115', vc)
    rec_v.append(vr); rec_lo.append(lr); rec_hi.append(hr)
    csi_v.append(vc); csi_lo.append(lc); csi_hi.append(hc)

  ax_b.bar(xb - bw / 2, rec_v, width=bw, color='#D97706', edgecolor='#1E293B', lw=0.45, label='α = 1 (Recall)', zorder=3)
  ax_b.bar(xb + bw / 2, csi_v, width=bw, color='#0D9488', edgecolor='#1E293B', lw=0.45, label='α = CSI/R (FA-disc.)', zorder=3)
  if df_ci is not None:
    ax_b.errorbar(xb - bw / 2, rec_v, yerr=[rec_lo, rec_hi], fmt='none', ecolor='#0F172A', elinewidth=0.75, capsize=1.8, zorder=5)
    ax_b.errorbar(xb + bw / 2, csi_v, yerr=[csi_lo, csi_hi], fmt='none', ecolor='#0F172A', elinewidth=0.75, capsize=1.8, zorder=5)

  ymax_b = max(v + h for v, h in zip(rec_v, rec_hi))
  for i in range(len(stage_specs)):
    ax_b.text(xb[i] - bw / 2, rec_v[i] + rec_hi[i] + 0.03 * ymax_b, f'{rec_v[i]:.2f}', ha='center', va='bottom', fontsize=6.0, fontweight='bold', color='#9A3412')
    ax_b.text(xb[i] + bw / 2, csi_v[i] + csi_hi[i] + 0.03 * ymax_b, f'{csi_v[i]:.2f}', ha='center', va='bottom', fontsize=6.0, fontweight='bold', color='#0F766E')

  ax_b.axvline(2.5, color='#94A3B8', ls='--', lw=0.75)
  ax_b.set_xticks(xb)
  ax_b.set_xticklabels([lbl for _, lbl in stage_specs], fontsize=6.0, fontweight='bold')
  ax_b.set_ylim(0, ymax_b * 1.24)
  ax_b.tick_params(axis='y', labelsize=6.1)
  ax_b.set_ylabel('CAP-gap gain above today (M/yr)', fontsize=6.6, fontweight='bold')
  ax_b.set_title('(b) CAP-gap gains by tier (with 95% CIs)', fontsize=7.1, fontweight='bold', loc='left', pad=5)
  ax_b.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=6.0, borderpad=0.25)
  ax_b.grid(True, axis='y', linestyle='--', alpha=0.35, color='#CBD5E1', lw=0.45)
  ax_b.spines['top'].set_visible(False)
  ax_b.spines['right'].set_visible(False)

  # ---- Panel (c): How gauges improve forecasts (CAP-gap mechanism bars + 95% CIs) ----
  ax_c = fig.add_subplot(gs[1, 0])
  mech_rows = [
      ('000_global_train_joint', '000_train', 'Pool history\nfor training'),
      ('010_ft_only_snap', '010_ft_only', 'Tune model\nto river (FT)'),
      ('001_da_only_snap', '001_da_only', 'Assimilate live Q\n(DA)'),
      ('111_all_three_combined_snap', '111_combined', 'All three\ncombined'),
  ]
  ym = np.arange(len(mech_rows))[::-1]
  hm = 0.34
  v_121, v_nood, v_ceil = [], [], []
  e_121, e_ceil = [], []
  for mk, bmk, _ in mech_rows:
    v1 = get_v('tier4_all_discoverable', mk, 'delta_net_nosys115')
    v3 = get_v('tier5_universal_idealized_ceiling', mk, 'delta_net_nosys115')
    v2 = get_v('tier5_universal_no_ood_clamp', mk, 'delta_net_nosys115') if (has_nood and ('tier5_universal_no_ood_clamp', mk) in df_sum.index) else v3
    v_121.append(v1); v_nood.append(v2); v_ceil.append(v3)
    e_121.append(get_ci_err(f'tier4_all_discoverable__{bmk}', 'delta_nosys115', v1))
    e_ceil.append(get_ci_err(f'tier5_universal_idealized_ceiling__{bmk}', 'delta_nosys115', v3))

  ax_c.barh(ym + hm / 2, v_121, height=hm, color='#0D9488', edgecolor='white', lw=0.45, label='Existing 119,169 gauges', zorder=3)
  ax_c.barh(ym - hm / 2, v_nood, height=hm, color='#D97706', edgecolor='white', lw=0.45, label='Universal (local gain)', zorder=3)
  ext_ood = [max(0.0, c - n) for n, c in zip(v_nood, v_ceil)]
  if any(x > 0.01 for x in ext_ood):
    ax_c.barh(ym - hm / 2, ext_ood, left=v_nood, height=hm, color='#FDE68A', edgecolor='#B45309', hatch='///', lw=0.45, label='Universal (+climate bound)', zorder=3)

  if df_ci is not None:
    ax_c.errorbar(v_121, ym + hm / 2, xerr=[[e[0] for e in e_121], [e[1] for e in e_121]], fmt='none', ecolor='#0F172A', elinewidth=0.70, capsize=1.6, zorder=5)
    ax_c.errorbar(v_ceil, ym - hm / 2, xerr=[[e[0] for e in e_ceil], [e[1] for e in e_ceil]], fmt='none', ecolor='#0F172A', elinewidth=0.70, capsize=1.6, zorder=5)

  xmax_c = max(v + e[1] for v, e in zip(v_ceil, e_ceil))
  for idx in range(len(mech_rows)):
    ax_c.text(v_121[idx] + e_121[idx][1] + 0.03 * xmax_c, ym[idx] + hm / 2, f'+{v_121[idx]:.2f}M', va='center', ha='left', fontsize=6.0, fontweight='bold', color='#0F766E')
    if abs(v_ceil[idx] - v_nood[idx]) > 0.04:
      lbl_u = f'+{v_nood[idx]:.2f}–{v_ceil[idx]:.2f}M'
    else:
      lbl_u = f'+{v_ceil[idx]:.2f}M'
    ax_c.text(v_ceil[idx] + e_ceil[idx][1] + 0.03 * xmax_c, ym[idx] - hm / 2, lbl_u, va='center', ha='left', fontsize=6.0, fontweight='bold', color='#9A3412')

  ax_c.set_yticks(ym)
  ax_c.set_yticklabels([r[2] for r in mech_rows], fontsize=6.1, fontweight='bold')
  ax_c.set_xlim(0, xmax_c * 1.52)
  ax_c.tick_params(axis='x', labelsize=6.1)
  ax_c.set_xlabel('CAP-gap exposure gained at α = 1 (M/yr)', fontsize=6.6, fontweight='bold')
  ax_c.set_title('(c) How gauges improve CAP-gap forecasts', fontsize=7.1, fontweight='bold', loc='left', pad=5)
  ax_c.legend(loc='center right', bbox_to_anchor=(0.99, 0.56), frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=5.9, borderpad=0.22, labelspacing=0.18)
  ax_c.grid(True, axis='x', linestyle='--', alpha=0.35, color='#CBD5E1', lw=0.45)
  ax_c.spines['top'].set_visible(False)
  ax_c.spines['right'].set_visible(False)

  # ---- Panel (d): A5 Per-Capita Unforecasted Exposure vs. INFORM 2026 Vulnerability ----
  ax_d = fig.add_subplot(gs[1, 1])
  c2iso = dict(zip(world_all['map_name'], world_all['iso_clean']))
  df_inf = pd.read_csv(file_paths.RISK_FILE, skiprows=[0, 2])
  df_inf['ISO3'] = df_inf['ISO3'].astype(str).str.strip()
  df_inf['COUNTRY_CLEAN'] = df_inf['COUNTRY'].astype(str).str.strip().replace(file_paths.country_name_map)
  df_inf['VULNERABILITY'] = pd.to_numeric(df_inf['VULNERABILITY'], errors='coerce')
  iso_m = df_inf.dropna(subset=['ISO3', 'VULNERABILITY']).drop_duplicates('ISO3').set_index('ISO3')['VULNERABILITY']
  nam_m = df_inf.dropna(subset=['COUNTRY_CLEAN', 'VULNERABILITY']).drop_duplicates('COUNTRY_CLEAN').set_index('COUNTRY_CLEAN')['VULNERABILITY']

  df_c['country_clean'] = df_c['country_name'].astype(str).str.strip().replace(file_paths.country_name_map)
  df_c['vuln'] = [iso_m.get(c2iso.get(c), nam_m.get(c, np.nan)) for c in df_c['country_clean']]
  col_u = 'dnet__tier5_universal_idealized_ceiling__111_all_three_combined_snap'
  sc = df_c[(df_c['E_total'] > 0) & (df_c['total_pop'] > 0) & (df_c['country_name'] != 'Greenland') & (df_c['vuln'].notna())].copy()
  sc['percap_1m'] = sc[col_u] / sc['total_pop'] * 1e6
  sc['cap_gap'] = ~(sc['has_system'].astype(bool))
  sc_pos = sc[sc['percap_1m'] > 0.05].copy()

  sc_warn = sc_pos[~sc_pos['cap_gap']]
  sc_cap = sc_pos[sc_pos['cap_gap']]
  n_warn_inf = int((~sc['cap_gap']).sum())
  n_cap_inf = int(sc['cap_gap'].sum())
  ax_d.scatter(sc_warn['vuln'], sc_warn['percap_1m'], s=14, color='#94A3B8', alpha=0.58, edgecolor='white', lw=0.35, zorder=3, label=f'Alerting ({n_warn_inf})')
  ax_d.scatter(sc_cap['vuln'], sc_cap['percap_1m'], s=21, color='#DC2626', alpha=0.82, edgecolor='white', lw=0.40, zorder=4, label=f'CAP-gap ({n_cap_inf})')

  coef_d = np.polyfit(sc_pos['vuln'].values, np.log10(sc_pos['percap_1m'].values), 1)
  xs_d = np.linspace(0.5, 9.6, 100)
  ax_d.plot(xs_d, 10.0 ** (coef_d[0] * xs_d + coef_d[1]), color='#1E293B', lw=1.15, ls='--', zorder=5)

  for cname, dx, mult, ha, va, col in [
      ('South Sudan', +0.18, 1.45, 'left',  'bottom', '#991B1B'),
      ('Bangladesh',  -0.18, 1.45, 'right', 'bottom', '#475569'),
      ('Myanmar',     +0.22, 1.05, 'left',  'center', '#991B1B'),
      ('Nigeria',     +0.28, 0.92, 'left',  'center', '#991B1B'),
      ('Congo DR',    -0.25, 0.42, 'right', 'top',    '#991B1B'),
      ('India',       -0.20, 0.72, 'right', 'top',    '#475569'),
      ('Viet Nam',    -0.18, 0.75, 'right', 'top',    '#475569'),
  ]:
    row = sc_pos[sc_pos['country_name'] == cname]
    if not row.empty:
      r0 = row.iloc[0]
      short = 'DR Congo' if cname == 'Congo DR' else ('Vietnam' if cname == 'Viet Nam' else cname)
      ax_d.text(r0['vuln'] + dx, r0['percap_1m'] * mult, short, fontsize=5.9, fontweight='bold', color=col, ha=ha, va=va, zorder=7)

  sp_pc = sp_stats.get('inform_vuln_vs_unforecasted_per_capita_idealized', {})
  sp_fr = sp_stats.get('inform_vuln_vs_unforecasted_frac_idealized', {})
  rs_pc = sp_pc.get('spearman_r', spearmanr(sc['vuln'], sc['percap_1m'])[0])
  psp_pc = sp_pc.get('p_spatial', spearmanr(sc['vuln'], sc['percap_1m'])[1])
  rs_fr = sp_fr.get('spearman_r', spearmanr(sc['vuln'], sc[col_u] / sc['E_total'])[0])
  p_fr = sp_fr.get('p_spatial', spearmanr(sc['vuln'], sc[col_u] / sc['E_total'])[1])

  ax_d.text(
      0.03, 0.96,
      f'Per-capita: $r_s$={rs_pc:+.2f} ($p_{{sp}}$={_fmt_p_tex(psp_pc)})\n'
      f'Share of exp.: $r_s$={rs_fr:+.2f} ($p_{{sp}}$={p_fr:.2f})',
      transform=ax_d.transAxes, ha='left', va='top', fontsize=5.9, fontweight='bold', color='#0F172A',
      bbox=dict(boxstyle='round,pad=0.20', facecolor='white', edgecolor='#CBD5E1', alpha=0.94),
      zorder=6,
  )
  ax_d.set_yscale('log')
  ax_d.set_xlim(0.2, 10.1)
  ax_d.set_ylim(0.5, max(8e4, float(sc_pos['percap_1m'].max()) * 18.0))
  ax_d.tick_params(labelsize=6.1)
  ax_d.set_xlabel('INFORM 2026 Vulnerability Index (0–10)', fontsize=6.6, fontweight='bold')
  ax_d.set_ylabel('Unforecasted exposure\n(persons/yr per 1M residents)', fontsize=6.5, fontweight='bold')
  ax_d.set_title('(d) Unforecasted exposure concentrates in vulnerable nations', fontsize=6.8, fontweight='bold', loc='left', pad=5)
  ax_d.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=6.0, borderpad=0.22)
  ax_d.grid(True, linestyle='--', alpha=0.35, color='#CBD5E1', lw=0.45)
  ax_d.spines['top'].set_visible(False)
  ax_d.spines['right'].set_visible(False)

  out_path = out_dir / 'Fig_5_exposure_synthesis.png'
  fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.05)
  plt.close(fig)
  print(f'  Saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)', flush=True)
  return out_path


# ==============================================================================
# FIGURE 6: Exact 2^7=128 Group Shapley Factor Attribution (All Fonts >= 6 pt)
# ==============================================================================
def build_fig6(out_dir: Path) -> Path:
  print('[Fig 6] Generating Exact Group Shapley factor attribution figure...', flush=True)
  _set_pub_rc(6.5)

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

  fig = plt.figure(figsize=(SCIENCE_W, 8.20), dpi=SAVE_DPI, facecolor='#ffffff')
  gs = gridspec.GridSpec(
      4, 4, figure=fig, height_ratios=[1.32, 1.0, 1.0, 1.0],
      hspace=0.56, wspace=0.96, left=0.148, right=0.978, top=0.960, bottom=0.055,
  )

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
      meds.append(float(q50)); q25s.append(float(q25)); q75s.append(float(q75))
    meds, q25s, q75s = np.array(meds), np.array(q25s), np.array(q75s)
    xpos = x_cats + (mi - (n_m - 1) / 2.0) * w_bar
    ax_a.bar(xpos, meds, width=w_bar, color=metric_palette[m], edgecolor='#222222', linewidth=0.25, label=metric_labels[m])
    ax_a.errorbar(xpos, meds, yerr=[np.maximum(0.0, meds - q25s), np.maximum(0.0, q75s - meds)], fmt='none', ecolor='#222222', elinewidth=0.45, zorder=4)

  ax_a.set_xticks(x_cats)
  ax_a.set_xticklabels([short_cat.get(c, c) for c in cat_order], fontsize=6.4, fontweight='bold', rotation=10, ha='right')
  ax_a.set_ylim(0, 82)
  ax_a.tick_params(axis='y', labelsize=6.2)
  ax_a.set_ylabel('Normalized |SHAP| share (%)', fontsize=6.7, fontweight='bold')
  ax_a.set_title('(a) Normalized Shapley feature attribution across groups and metrics', fontsize=7.0, fontweight='bold', loc='left', pad=4)
  ax_a.legend(loc='upper right', ncol=4, fontsize=6.0, frameon=True, framealpha=0.95, edgecolor='#CFD8DC', handlelength=0.9, columnspacing=0.55, borderpad=0.25)
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
    ax.barh(y_pos, meds, height=0.68, color=cols_p, edgecolor='#222222', linewidth=0.40)
    ax.errorbar(meds, y_pos, xerr=[np.maximum(0.0, meds - q25s), np.maximum(0.0, q75s - meds)], fmt='none', ecolor='#222222', elinewidth=0.60, capsize=1.4, zorder=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([short_cat.get(c, c) for c in sub['category']], fontsize=6.0, fontweight='bold')
    ax.set_xlim(0.0, max(1e-4, float(q75s.max()) * 1.16))
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.tick_params(axis='x', labelsize=6.0)
    ax.set_title(ptitle, fontsize=6.8, fontweight='bold', loc='left', pad=2.5)
    if row == 3:
      ax.set_xlabel('Med. |Shapley|', fontsize=6.1, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.4, linewidth=0.4)
    sns.despine(ax=ax)

  out_path = out_dir / 'Fig_6_exact_group_shapley.png'
  fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.05)
  plt.close(fig)
  print(f'  Saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)', flush=True)
  return out_path


def main() -> None:
  parser = argparse.ArgumentParser(description='Generate main-text figures (Figs. 1–3, 6; optional Figs. 4–5).')
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
  if 'fig3' in targets:
    t4_idx = pd.read_parquet(TIER_DIR / 'tier4_all_discoverable.parquet', columns=['U_net_tree_phys']).index
    basins = load_non_fathom_basin_meta(pd.Index(norm_hybas_id(t4_idx)))

  if 'fig1' in targets:
    build_fig1(world_all, out_dir)
  if 'fig2' in targets:
    build_fig2(out_dir)
  if 'fig3' in targets:
    build_fig3(world_all, basins, out_dir)
  if 'fig4' in targets:
    build_fig4_optional(world_all, out_dir)
  if 'fig5' in targets:
    build_fig5_optional(world_all, out_dir)
  if 'fig6' in targets:
    build_fig6(out_dir)

  print('Main figure generation complete!', flush=True)


if __name__ == '__main__':
  main()
