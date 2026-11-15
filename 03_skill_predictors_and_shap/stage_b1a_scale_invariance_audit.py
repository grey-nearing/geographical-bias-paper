#!/usr/bin/env python3
"""Stage B1a: Scale-Invariance Audit, Continuum Resolution Curve U(A_target),
Runoff-Weighted Ungaugedness, and Physical-G vs ge95 Audit across ALL 1,034,083
HydroBASINS v1c watersheds.
"""

from collections import deque
import json
import os
import time

import numpy as np
import pandas as pd

CANON = os.environ.get('GEO_BIAS_DATA_DIR', './data')
OUT_DIR = os.path.join(CANON, 'round2', 'scale_invariance')
os.makedirs(OUT_DIR, exist_ok=True)


def build_dag_from_edges(node_ids: np.ndarray, next_down_ids: np.ndarray):
  """Returns topological sort order (headwaters -> outlets) and next_idx array."""
  n = len(node_ids)
  id_to_idx = {int(h): i for i, h in enumerate(node_ids)}
  next_idx = np.array([id_to_idx.get(int(nd), -1) for nd in next_down_ids], dtype=np.int32)
  in_deg = np.zeros(n, dtype=np.int32)
  for i in range(n):
    nd = next_idx[i]
    if nd >= 0 and nd != i:
      in_deg[nd] += 1
    elif nd == i:
      next_idx[i] = -1
  q = deque(np.where(in_deg == 0)[0].tolist())
  topo = []
  while q:
    u = q.popleft()
    topo.append(u)
    v = next_idx[u]
    if v >= 0:
      in_deg[v] -= 1
      if in_deg[v] == 0:
        q.append(v)
  return next_idx, np.array(topo, dtype=np.int32), id_to_idx


def compute_level_metrics(
    node_ids: np.ndarray,
    next_idx: np.ndarray,
    topo: np.ndarray,
    sub_area: np.ndarray,
    sub_runoff: np.ndarray,
    is_gauged: np.ndarray,
    hw_area: np.ndarray | None = None,
):
  """Computes exact tree-based ungaugedness metrics on any HydroBASINS level L."""
  n = len(node_ids)
  if hw_area is None:
    hw_area = np.zeros(n, dtype=np.float64)

  # Total upstream drainage area A(u) and gauged upstream area A_g_up(u)
  tot_area = sub_area.copy()
  up_gauged_area = np.where(is_gauged, sub_area, hw_area).astype(np.float64)
  for u in topo:
    if is_gauged[u]:
      # Outlet gauge at u measures its entire upstream drainage area A(u) BEFORE propagating downstream!
      up_gauged_area[u] = tot_area[u]
    else:
      up_gauged_area[u] = min(up_gauged_area[u], tot_area[u])
    v = next_idx[u]
    if v >= 0:
      tot_area[v] += tot_area[u]
      if not is_gauged[v]:
        up_gauged_area[v] += up_gauged_area[u]
  c_up = np.where(tot_area > 0, up_gauged_area / tot_area, 0.0)
  u_up = 1.0 - c_up

  # Inter-gauge incremental drainage area delta_A_in(u) and downstream reach area
  unmon_inc = np.where(is_gauged, sub_area, np.maximum(sub_area - hw_area, 0.0))
  delta_A_in = unmon_inc.copy()
  for u in topo:
    v = next_idx[u]
    if v >= 0 and not is_gauged[u]:
      delta_A_in[v] += delta_A_in[u]

  # Nearest downstream gauge total drainage A(g*) and inter-gauge reach area Delta_A(g*)
  down_reach_area = np.full(n, np.inf, dtype=np.float64)
  down_total_area = np.full(n, np.inf, dtype=np.float64)
  has_down_gauge = np.zeros(n, dtype=bool)
  for u in reversed(topo):
    if is_gauged[u]:
      down_reach_area[u] = delta_A_in[u]
      down_total_area[u] = tot_area[u]
      has_down_gauge[u] = True
    else:
      v = next_idx[u]
      if v >= 0:
        down_reach_area[u] = down_reach_area[v]
        down_total_area[u] = down_total_area[v]
        has_down_gauge[u] = has_down_gauge[v]

  ratio_down_eq4 = np.where(
      np.isinf(down_reach_area) | (down_reach_area <= 0),
      0.0,
      np.minimum(delta_A_in / down_reach_area, 1.0),
  )
  u_net_eq4 = u_up * np.where(is_gauged, 0.0, 1.0 - ratio_down_eq4)

  c_down_tot = np.where(
      np.isinf(down_total_area) | (down_total_area <= 0),
      0.0,
      np.minimum(tot_area / down_total_area, 1.0),
  )
  u_net_prod = (1.0 - c_up) * (1.0 - c_down_tot)

  u_out = (~is_gauged).astype(np.float64)
  zero_up = (c_up <= 0.0).astype(np.float64)
  no_down = (~has_down_gauge).astype(np.float64)
  zero_net = ((c_up <= 0.0) & (~has_down_gauge)).astype(np.float64)

  res = {
      'n_units': int(n),
      'n_gauged_units': int(is_gauged.sum()),
      'mean_unit_area_km2': float(sub_area.mean()),
      'median_unit_area_km2': float(np.median(sub_area)),
      # Area-weighted (%)
      'U_out_area_pct': float(np.average(u_out * 100.0, weights=sub_area)),
      'U_up_area_pct': float(np.average(u_up * 100.0, weights=sub_area)),
      'U_net_eq4_area_pct': float(np.average(u_net_eq4 * 100.0, weights=sub_area)),
      'U_net_prod_area_pct': float(np.average(u_net_prod * 100.0, weights=sub_area)),
      'U_zero_up_area_pct': float(np.average(zero_up * 100.0, weights=sub_area)),
      'U_no_down_area_pct': float(np.average(no_down * 100.0, weights=sub_area)),
      'U_zero_net_area_pct': float(np.average(zero_net * 100.0, weights=sub_area)),
      # Runoff-weighted (%)
      'U_out_runoff_pct': float(np.average(u_out * 100.0, weights=sub_runoff)),
      'U_up_runoff_pct': float(np.average(u_up * 100.0, weights=sub_runoff)),
      'U_net_eq4_runoff_pct': float(np.average(u_net_eq4 * 100.0, weights=sub_runoff)),
      'U_no_down_runoff_pct': float(np.average(no_down * 100.0, weights=sub_runoff)),
      'U_zero_net_runoff_pct': float(np.average(zero_net * 100.0, weights=sub_runoff)),
      # Unweighted (%)
      'U_out_unwt_pct': float(u_out.mean() * 100.0),
      'U_up_unwt_pct': float(u_up.mean() * 100.0),
      'U_net_eq4_unwt_pct': float(u_net_eq4.mean() * 100.0),
      'U_zero_net_unwt_pct': float(zero_net.mean() * 100.0),
  }
  return res, down_reach_area


def build_pfaf_level_graph(
    hybas_ids: np.ndarray,
    next_down_ids: np.ndarray,
    pfaf_ids: np.ndarray,
    sub_area: np.ndarray,
    sub_runoff: np.ndarray,
    level: int,
):
  """Aggregates L12 HydroBASINS graph to exact Pfafstetter level L."""
  if level == 12:
    next_idx, topo, id_to_idx = build_dag_from_edges(hybas_ids, next_down_ids)
    return {
        'node_ids': hybas_ids,
        'next_idx': next_idx,
        'topo': topo,
        'id_to_idx': id_to_idx,
        'sub_area': sub_area,
        'sub_runoff': sub_runoff,
        'l12_to_parent': hybas_ids,
    }

  div = np.int64(10 ** (12 - level))
  p_l = pfaf_ids // div
  id_to_l12 = {int(h): i for i, h in enumerate(hybas_ids)}

  # Aggregate areas & runoff by p_l
  df_agg = pd.DataFrame({
      'p_l': p_l,
      'sub_area': sub_area,
      'sub_runoff': sub_runoff,
      'hybas_id': hybas_ids,
      'next_down': next_down_ids,
  })
  grp = df_agg.groupby('p_l', sort=True)
  p_nodes = grp['sub_area'].sum().index.values.astype(np.int64)
  p_area = grp['sub_area'].sum().values.astype(np.float64)
  p_runoff = grp['sub_runoff'].sum().values.astype(np.float64)

  # Find the downstream parent p_l for each p_l (cross-group edge from the outlet L12 unit)
  next_p_map = {}
  # If a level-L polygon splits across multiple coastal outlets (Pfafstetter interbasins/coastal),
  # prefer the edge leaving the largest upstream area node inside p_l.
  # First compute L12 total upstream area to break any coastal ties cleanly.
  l12_next_idx, l12_topo, _ = build_dag_from_edges(hybas_ids, next_down_ids)
  l12_tot_area = sub_area.copy()
  for u in l12_topo:
    v = l12_next_idx[u]
    if v >= 0:
      l12_tot_area[v] += l12_tot_area[u]

  best_out_area = {}
  for i in range(len(hybas_ids)):
    nd = int(next_down_ids[i])
    pu = int(p_l[i])
    if nd <= 0 or nd not in id_to_l12:
      pv = 0
    else:
      pv = int(p_l[id_to_l12[nd]])
    if pv != pu:
      a = float(l12_tot_area[i])
      if pu not in best_out_area or a > best_out_area[pu]:
        best_out_area[pu] = a
        next_p_map[pu] = pv

  p_next = np.array([next_p_map.get(int(p), 0) for p in p_nodes], dtype=np.int64)
  next_idx, topo, id_to_idx = build_dag_from_edges(p_nodes, p_next)
  return {
      'node_ids': p_nodes,
      'next_idx': next_idx,
      'topo': topo,
      'id_to_idx': id_to_idx,
      'sub_area': p_area,
      'sub_runoff': p_runoff,
      'l12_to_parent': p_l,
  }


def main():
  t0 = time.time()
  print('1. Loading 1,034,083 HydroBASINS topology, Pfafstetter IDs, runoff, and gauge catalog...', flush=True)
  topo_pq = f'{CANON}/attributes/hybas_l12_topology_and_runoff.parquet'
  if os.path.exists(topo_pq):
    topo_df = pd.read_parquet(topo_pq)
    hybas_ids = topo_df['HYBAS_ID'].astype(np.int64).values
    next_down_ids = topo_df['NEXT_DOWN'].astype(np.int64).values
    sub_area = topo_df['area_m2'].values.astype(np.float64) / 1e6  # km^2
    pfaf_ids = topo_df['PFAF_ID'].astype(np.int64).values
    run_mm = topo_df['run_mm_syr'].fillna(0.0).clip(lower=0.0).values.astype(np.float64)
  else:
    topo_df = pd.read_csv(f'{CANON}/attributes/hybas_routing_topology.csv')
    hatt = pd.read_csv(
        f'{CANON}/attributes/hybas_attributes.csv',
        usecols=['Unnamed: 0', 'HYDRO_ATLAS:PFAF_ID', 'HYDRO_ATLAS:SUB_AREA', 'HYDRO_ATLAS:run_mm_syr'],
        dtype={'Unnamed: 0': str},
    )
    hatt['HYBAS_ID'] = hatt['Unnamed: 0'].str.replace('hybas_', '', regex=False).astype(np.int64)
    hatt = hatt.drop_duplicates('HYBAS_ID').set_index('HYBAS_ID')

    hybas_ids = topo_df['HYBAS_ID'].astype(np.int64).values
    next_down_ids = topo_df['NEXT_DOWN'].astype(np.int64).values
    sub_area = topo_df['area_m2'].values.astype(np.float64) / 1e6  # km^2
    pfaf_ids = hatt['HYDRO_ATLAS:PFAF_ID'].reindex(hybas_ids).values.astype(np.int64)
    run_mm = hatt['HYDRO_ATLAS:run_mm_syr'].reindex(hybas_ids).fillna(0.0).clip(lower=0.0).values
  sub_runoff = run_mm * sub_area * 1e-6  # km^3/yr

  l12_id_to_pos = {int(h): i for i, h in enumerate(hybas_ids)}

  dedup = pd.read_parquet(f'{CANON}/attributes/global_gauges_deduplicated.parquet')
  arch = pd.read_parquet(f'{CANON}/attributes/global_gauges_by_source_archive.parquet')
  grdc_8811 = pd.read_csv(f'{CANON}/attributes/grdc_attributes.csv', index_col=0)

  native_to_cid = dict(zip(arch['native_station_id'].astype(str), arch['canonical_gauge_id'].astype(str)))
  grdc_arch = arch[arch['source_dataset'] == 'GRDC']
  for nid, cid in zip(grdc_arch['native_station_id'].astype(str), grdc_arch['canonical_gauge_id'].astype(str)):
    native_to_cid[f'GRDC_{nid}'] = cid
  t1a_ids = ['CARAVAN_' + str(x).upper() for x in grdc_8811.index]
  t1a_cids = set(native_to_cid[x] for x in t1a_ids if x in native_to_cid)

  tiers = {
      'Baseline_8811': dedup[dedup['canonical_gauge_id'].isin(t1a_cids)],
      'Tier1_34577': dedup[
          dedup['all_source_datasets'].str.contains('Caravan', na=False)
          | dedup['all_source_datasets'].str.contains('EStreams', na=False)
      ],
      'Tier2_93827': dedup[
          dedup['best_data_access_policy'].isin(['open_public_api_or_bulk', 'open_academic_compilation'])
          & dedup['best_observation_type'].isin(['daily_discharge', 'monthly_discharge'])
      ],
      'Tier3_121819': dedup,
  }

  # 2. Multi-level Pfafstetter scale sensitivity (Levels 5, 6, 7, 8, 9, 10, 12)
  levels = [5, 6, 7, 8, 9, 10, 12]
  level_graphs = {}
  for L in levels:
    print(f'2. Building exact Pfafstetter Level {L} graph...', flush=True)
    level_graphs[L] = build_pfaf_level_graph(
        hybas_ids, next_down_ids, pfaf_ids, sub_area, sub_runoff, L
    )

  rows_scale = []
  rows_curve = []
  a_targets = [50, 100, 130, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 500000, 1e12]

  for tier_name, d_sub in tiers.items():
    valid_snapped = d_sub[d_sub['snapped_hybas_id'] > 0]
    snapped_main = valid_snapped[~valid_snapped['is_subbasin_headwater_gauge']]
    snapped_all_l12 = set(int(h) for h in valid_snapped['snapped_hybas_id'].values if int(h) in l12_id_to_pos)
    snapped_main_l12 = set(int(h) for h in snapped_main['snapped_hybas_id'].values if int(h) in l12_id_to_pos)

    for g_def, l12_set in [('physical_any', snapped_all_l12), ('physical_mainstem', snapped_main_l12)]:
      for L in levels:
        g = level_graphs[L]
        parent_map = g['l12_to_parent']
        id_to_idx = g['id_to_idx']
        is_gauged = np.zeros(len(g['node_ids']), dtype=bool)
        for h12 in l12_set:
          pos12 = l12_id_to_pos[h12]
          p_id = int(parent_map[pos12])
          if p_id in id_to_idx:
            is_gauged[id_to_idx[p_id]] = True

        # Optional headwater effective area at L12 for physical_mainstem
        hw_area = None
        if L == 12 and g_def == 'physical_mainstem':
          hw_df = valid_snapped[valid_snapped['is_subbasin_headwater_gauge']]
          hw_by_id = hw_df.groupby(hw_df['snapped_hybas_id'].astype(np.int64))['effective_gauged_area_km2'].sum()
          hw_area = np.array([
              min(float(hw_by_id.get(int(h), 0.0)), 0.95 * g['sub_area'][i])
              for i, h in enumerate(g['node_ids'])
          ])

        met, down_reach_area = compute_level_metrics(
            g['node_ids'], g['next_idx'], g['topo'], g['sub_area'], g['sub_runoff'], is_gauged, hw_area
        )
        met['tier'] = tier_name
        met['gauge_def'] = g_def
        met['level'] = L
        met['n_phys_gauges'] = int(len(d_sub))
        rows_scale.append(met)

        if g_def == 'physical_any' and L in (6, 8, 10, 12):
          for at in a_targets:
            frac_area = float(np.average((down_reach_area > at).astype(float) * 100.0, weights=g['sub_area']))
            frac_run = float(np.average((down_reach_area > at).astype(float) * 100.0, weights=g['sub_runoff']))
            rows_curve.append({
                'tier': tier_name,
                'level': L,
                'A_target_km2': at if at < 1e11 else np.inf,
                'U_area_pct': frac_area,
                'U_runoff_pct': frac_run,
            })

  df_scale = pd.DataFrame(rows_scale)
  df_curve = pd.DataFrame(rows_curve)
  df_scale.to_csv(os.path.join(OUT_DIR, 'hydrobasins_scale_invariance_L5_to_L12.csv'), index=False)
  df_curve.to_csv(os.path.join(OUT_DIR, 'continuum_resolution_curve_U_A_target.csv'), index=False)

  print('\n=== SCALE SENSITIVITY ACROSS HYDROBASINS LEVELS (physical_any, 1,034,083 L12 universe) ===')
  sub = df_scale[df_scale['gauge_def'] == 'physical_any']
  cols_show = ['tier', 'level', 'n_units', 'n_gauged_units', 'mean_unit_area_km2', 'U_out_area_pct', 'U_up_area_pct', 'U_net_eq4_area_pct', 'U_no_down_area_pct', 'U_zero_net_area_pct', 'U_net_eq4_runoff_pct', 'U_zero_net_runoff_pct']
  print(sub[cols_show].to_string(index=False))

  print('\n=== CONTINUUM MONITORING-RESOLUTION CURVE U(A_target) ACROSS L6, L8, L10, L12 ===')
  piv = df_curve[df_curve['tier'].isin(['Baseline_8811', 'Tier3_121819'])].pivot(
      index=['tier', 'A_target_km2'], columns='level', values='U_area_pct'
  )
  print(piv.round(2).to_string())

  print(f'\nSaved scale-invariance diagnostics to {OUT_DIR} in {time.time() - t0:.1f}s!', flush=True)


if __name__ == '__main__':
  main()
