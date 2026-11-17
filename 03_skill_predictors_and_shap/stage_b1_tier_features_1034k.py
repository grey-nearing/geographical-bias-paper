#!/usr/bin/env python3
"""Stage B1: Rebuilds all institutional gauge-tier coverage features (Factor 2),
88-feature HydroATLAS anomaly scores (Factor 3), physical G-set masks (active & all),
and Section 2 ungaugedness summaries across ALL 1,034,083 HydroBASINS v1c watersheds.

Round-2 fixes:
1. Full 1,034,083 global HydroBASINS L12 universe (all 9 continents incl. Canada `ar`,
   Greenland `gr`, and all 6,932 repaired polygons).
2. Pure feature-space top-quartile OOD reference (`X_train.quantile(0.75)`, ZERO target Y)
   for Universal Gauging, applied uniformly via `np.maximum(ood_t4, ref_q75)` across ALL
   3 OOD scores including `anomaly_score_lof`.
3. Strict physical gauge polygon masks (`snap_all`, `snap_active`, `snap_main_all`,
   `snap_main_active`) as primary G-set alongside `ge95` & `union` sensitivity columns.
4. Supports optional `--threads` so OOD scoring doesn't contend with other jobs.
"""

import argparse
from collections import defaultdict, deque
import concurrent.futures
import json
import multiprocessing as mp
import os
import sys
import time
import warnings

REPO_DIR = os.environ.get(
    'GEO_BIAS_REPO_DIR',
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
if REPO_DIR not in sys.path:
  sys.path.insert(0, REPO_DIR)

CANON_DIR = os.environ.get('GEO_BIAS_DATA_DIR', './data')
os.environ.setdefault('GEO_BIAS_DATA_DIR', CANON_DIR)
os.environ.setdefault('GEO_BIAS_THIRD_PARTY_DIR', CANON_DIR)
os.environ.setdefault('GEO_BIAS_OUTPUT_DIR', CANON_DIR)

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import torch

from utils import coverage_utils as overlap_mod

warnings.filterwarnings('ignore')

OVERLAPS_38K_DIR = os.path.join(CANON_DIR, 'coverage_fractions', 'polygon_overlaps_38k')
BACKUP_38K_DIR = os.path.join(CANON_DIR, 'backup_v2_38069_gauges')
ROUND_SUBDIR = os.environ.get('GEO_BIAS_ROUND_SUBDIR', 'round2')
ROUND2_DIR = os.path.join(CANON_DIR, ROUND_SUBDIR)
OUT_DIR = os.path.join(ROUND2_DIR, 'tier_features')
os.makedirs(OUT_DIR, exist_ok=True)


def normalize_ocsvm_series(vals: np.ndarray) -> np.ndarray:
  """Scale-normalizes OneClassSVM decision_function by |rho| = |min| so scores lie in [-1, ~+0.45]
  regardless of training-set size N (fixing the N-proportional scaling of sklearn OneClassSVM)."""
  arr = np.asarray(vals, dtype=np.float64)
  mn = float(np.nanmin(arr))
  if mn < -2.0:
    return (arr / abs(mn)).astype(np.float32)
  return arr.astype(np.float32)


def _read_table(base_path_no_ext: str, **csv_kwargs) -> pd.DataFrame:
  """Read a .parquet table if present, otherwise fall back to .csv."""
  pq = f'{base_path_no_ext}.parquet'
  if os.path.exists(pq):
    return pd.read_parquet(pq)
  return pd.read_csv(f'{base_path_no_ext}.csv', **csv_kwargs)


def _resolve_38k_table(rel_new_no_ext: str, rel_legacy_no_ext: str, **csv_kwargs) -> pd.DataFrame:
  for root, rel in ((OVERLAPS_38K_DIR, rel_new_no_ext), (BACKUP_38K_DIR, rel_legacy_no_ext)):
    for ext in ('.parquet', '.csv'):
      p = os.path.join(root, rel + ext)
      if os.path.exists(p):
        return pd.read_parquet(p) if ext == '.parquet' else pd.read_csv(p, **csv_kwargs)
  raise FileNotFoundError(f'Missing 38k overlap table: {rel_new_no_ext} (checked {OVERLAPS_38K_DIR} and {BACKUP_38K_DIR})')

CONTINENTS = ['AF', 'AR', 'AS', 'AU', 'EU', 'GR', 'NA', 'SA', 'SI']
COV_COLS = [
    'total_upstream_coverage',
    'max_upstream_coverage',
    'max_downstream_coverage',
    'upstream_gauge_count',
]
OOD_COLS = [
    'anomaly_score_if',
    'anomaly_score_ocsvm',
    'anomaly_score_lof',
]


def load_zarr_df(zarr_path: str) -> pd.DataFrame:
  import zarr

  root = zarr.open_group(store=zarr_path, mode='r', zarr_format=2)
  feats = root['features'][:]
  hybas_ids = [
      x.decode('utf-8') if isinstance(x, bytes) else str(x)
      for x in root['hybas_id'][:]
  ]
  hybas_ids = [x if x.startswith('hybas_') else f'hybas_{x}' for x in hybas_ids]
  cols = [
      x.decode('utf-8') if isinstance(x, bytes) else str(x)
      for x in root['columns'][:]
  ]
  return pd.DataFrame(feats, index=hybas_ids, columns=cols)


def fast_calculate_anomaly_scores(
    X_train: pd.DataFrame, X_test: pd.DataFrame, n_threads: int = 32
) -> pd.DataFrame:
  """PyTorch-accelerated exact anomaly scorer (IF, OCSVM, LOF) over all 1,034,083 basins."""
  torch.set_num_threads(n_threads)
  feature_order = list(X_train.columns)
  X_test_aligned = X_test[feature_order].copy()
  X_train_aligned = X_train.copy()
  if len(X_train_aligned) > 20000:
    non_area_cols = [
        c for c in feature_order
        if 'drain_area' not in c.lower() and 'drainage area' not in c.lower()
    ]
    X_train_aligned = X_train_aligned.drop_duplicates(
        subset=non_area_cols if non_area_cols else None
    )

  scaler = StandardScaler()
  X_train_scaled = np.ascontiguousarray(scaler.fit_transform(X_train_aligned), dtype=np.float64)
  X_test_scaled = np.ascontiguousarray(scaler.transform(X_test_aligned), dtype=np.float64)

  # 1. Isolation Forest
  iso = IsolationForest(random_state=42, n_jobs=n_threads)
  iso.fit(X_train_scaled)
  scores_if = iso.decision_function(X_test_scaled)

  # 2. One-Class SVM (normalized by |intercept| = rho so decision scores are N-invariant in [-1, ~+0.45])
  ocsvm = OneClassSVM(cache_size=4000)
  ocsvm.fit(X_train_scaled)
  sv = np.ascontiguousarray(ocsvm.support_vectors_, dtype=np.float64)
  coef = np.ascontiguousarray(ocsvm.dual_coef_.ravel(), dtype=np.float64)
  gamma = float(ocsvm._gamma)
  intercept = float(ocsvm.intercept_[0])
  rho_scale = max(abs(intercept), 1e-12)
  sv_sq = np.einsum('ij,ij->i', sv, sv)
  ocsvm_vals = np.empty(len(X_test_scaled), dtype=np.float64)
  bs = 16384
  for i in range(0, len(X_test_scaled), bs):
    xb = X_test_scaled[i : i + bs]
    xb_sq = np.einsum('ij,ij->i', xb, xb)[:, None]
    d2 = xb_sq + sv_sq[None, :] - 2.0 * (xb @ sv.T)
    np.maximum(d2, 0.0, out=d2)
    ocsvm_vals[i : i + bs] = (np.exp(-gamma * d2) @ coef + intercept) / rho_scale

  # 3. Local Outlier Factor (exact k=20 Euclidean, matching sklearn LOF novelty=True)
  n_tr = len(X_train_scaled)
  n_te = len(X_test_scaled)
  k_n = 20
  t_tr = torch.from_numpy(np.ascontiguousarray(X_train_scaled, dtype=np.float32))
  dists_tr = np.empty((n_tr, k_n), dtype=np.float64)
  nbr_tr = np.empty((n_tr, k_n), dtype=np.int64)
  with torch.no_grad():
    for s in range(0, n_tr, 8192):
      e = min(n_tr, s + 8192)
      d_mat = torch.cdist(t_tr[s:e], t_tr)
      ar = torch.arange(e - s)
      d_mat[ar, ar + s] = -1.0
      vals, idx = torch.topk(d_mat, k=k_n + 1, dim=1, largest=False, sorted=True)
      dists_tr[s:e] = np.maximum(vals[:, 1:].numpy().astype(np.float64), 0.0)
      nbr_tr[s:e] = idx[:, 1:].numpy()

  k_dist_tr = dists_tr[:, -1]
  reach_tr = np.maximum(dists_tr, k_dist_tr[nbr_tr])
  lrd_tr = 1.0 / (np.mean(reach_tr, axis=1) + 1e-10)

  lof_vals = np.empty(n_te, dtype=np.float64)
  with torch.no_grad():
    for s in range(0, n_te, 16384):
      e = min(n_te, s + 16384)
      t_q = torch.from_numpy(np.ascontiguousarray(X_test_scaled[s:e], dtype=np.float32))
      d_mat = torch.cdist(t_q, t_tr)
      vals, idx = torch.topk(d_mat, k=k_n, dim=1, largest=False, sorted=True)
      d_te = np.maximum(vals.numpy().astype(np.float64), 0.0)
      n_idx = idx.numpy()
      reach_te = np.maximum(d_te, k_dist_tr[n_idx])
      lrd_te = 1.0 / (np.mean(reach_te, axis=1) + 1e-10)
      score_te = -np.mean(lrd_tr[n_idx] / lrd_te[:, None], axis=1)
      lof_vals[s:e] = score_te + 1.5

  return pd.DataFrame(
      {
          'anomaly_score_if': scores_if.astype(np.float32),
          'anomaly_score_ocsvm': ocsvm_vals.astype(np.float32),
          'anomaly_score_lof': lof_vals.astype(np.float32),
      },
      index=X_test_aligned.index,
  )


def build_continent_topology(df_topo_cont: pd.DataFrame):
  df_t = df_topo_cont.copy()
  df_t['HYBAS_ID'] = (
      df_t['HYBAS_ID'].astype(str).str.replace(r'\.0$', '', regex=True).str.replace('hybas_', '', regex=False)
  )
  df_t['NEXT_DOWN'] = (
      df_t['NEXT_DOWN'].astype(str).str.replace(r'\.0$', '', regex=True).str.replace('hybas_', '', regex=False)
  )
  df_t = df_t.drop_duplicates(subset=['HYBAS_ID']).set_index('HYBAS_ID')
  hybas_ids = ['hybas_' + idx for idx in df_t.index]
  hybas_sub_areas = pd.Series(df_t['area_m2'].astype(float).values, index=hybas_ids)

  adj = defaultdict(list)
  parent_map_raw = {}
  for child, next_down in zip(df_t.index, df_t['NEXT_DOWN']):
    if pd.notna(next_down) and next_down not in ('0', 0, 'nan', ''):
      parent = str(int(next_down)) if isinstance(next_down, float) else str(next_down)
      adj[parent].append(child)
      parent_map_raw[child] = parent

  sub_area_raw = dict(zip(df_t.index, df_t['area_m2'].astype(float).values))
  total_area_raw = {}
  sys.setrecursionlimit(max(sys.getrecursionlimit(), 50000))

  def get_total_area(node):
    if node in total_area_raw:
      return total_area_raw[node]
    tot_a = sub_area_raw.get(node, 0.0)
    for child in adj.get(node, ()):
      tot_a += get_total_area(child)
    total_area_raw[node] = tot_a
    return tot_a

  for node in df_t.index:
    get_total_area(node)

  downstream_map = {}
  total_areas_dict = {}
  for node, tot_a in total_area_raw.items():
    p_node = 'hybas_' + node
    total_areas_dict[p_node] = tot_a
    d_path = []
    curr = node
    visited = {curr}
    while curr in parent_map_raw:
      nxt = parent_map_raw[curr]
      if nxt in visited or nxt not in total_area_raw:
        break
      visited.add(nxt)
      d_path.append('hybas_' + nxt)
      curr = nxt
    downstream_map[p_node] = d_path

  return hybas_ids, downstream_map, hybas_sub_areas, pd.Series(total_areas_dict)


def build_tree_arrays(topo_df: pd.DataFrame, run_mm: np.ndarray):
  hybas_ids = topo_df['HYBAS_ID'].astype(np.int64).values
  hybas_str_ids = [f'hybas_{h}' for h in hybas_ids]
  n = len(hybas_ids)
  id_to_idx = {int(h): i for i, h in enumerate(hybas_ids)}
  str_to_idx = {f'hybas_{int(h)}': i for i, h in enumerate(hybas_ids)}
  next_down_ids = topo_df['NEXT_DOWN'].astype(np.int64).values
  next_idx = np.array([id_to_idx.get(int(nd), -1) for nd in next_down_ids], dtype=np.int32)
  inc_area = topo_df['area_m2'].values.astype(np.float64) / 1e6  # km^2
  inc_runoff = np.maximum(run_mm, 0.0) * inc_area * 1e-6  # km^3/yr

  in_deg = np.zeros(n, dtype=np.int32)
  for i in range(n):
    nd = next_idx[i]
    if nd >= 0:
      in_deg[nd] += 1
  q = deque(np.where(in_deg == 0)[0].tolist())
  topo_order = []
  while q:
    u = q.popleft()
    topo_order.append(u)
    v = next_idx[u]
    if v >= 0:
      in_deg[v] -= 1
      if in_deg[v] == 0:
        q.append(v)
  return {
      'hybas_ids': hybas_ids,
      'hybas_str_ids': hybas_str_ids,
      'id_to_idx': id_to_idx,
      'str_to_idx': str_to_idx,
      'next_idx': next_idx,
      'inc_area': inc_area,
      'inc_runoff': inc_runoff,
      'topo_order': np.array(topo_order, dtype=np.int32),
  }


def compute_combined_coverage_continent(args):
  """Computes exact coverage features for any subset of 38k + net-new dedup gauges on 1 continent."""
  cont, valid_38k_ids, net_new_df, old_caravan_tt, old_caravan_areas, df_topo_c = args
  hybas_ids, downstream_map, hybas_sub_areas, hybas_total_areas = build_continent_topology(df_topo_c)
  hybas_set = set(hybas_ids)

  old_ti_df = _resolve_38k_table(
      f'infer_pairwise_overlaps_{cont}',
      f'coverage_fractions/hybas/infer_pairwise_overlaps_{cont}',
  )
  old_ti_df = old_ti_df[old_ti_df['basin2'].isin(valid_38k_ids)].copy()

  old_feat_df = _resolve_38k_table(
      f'infer_ml_features_{cont}',
      f'coverage_fractions/hybas/infer_ml_features_{cont}',
      index_col=0,
  )
  infer_ids = list(old_feat_df['hybas_id'].values) if 'hybas_id' in old_feat_df.columns else list(old_feat_df.index)

  tt_sub_38k = old_caravan_tt[
      old_caravan_tt['basin1'].isin(valid_38k_ids) & old_caravan_tt['basin2'].isin(valid_38k_ids)
  ].copy()

  old_overlap_by_hybas = defaultdict(list)
  for b1, b2, ov_a in zip(old_ti_df['basin1'], old_ti_df['basin2'], old_ti_df['overlap_area']):
    if ov_a > 0:
      old_overlap_by_hybas[b1].append((b2, float(ov_a)))

  sub_new = net_new_df[net_new_df['hybas_str'].isin(hybas_set)] if len(net_new_df) > 0 else net_new_df
  main_gauges = sub_new[~sub_new['is_subbasin_headwater_gauge']] if len(sub_new) > 0 else sub_new
  hw_gauges = sub_new[sub_new['is_subbasin_headwater_gauge']] if len(sub_new) > 0 else sub_new

  w_main_list = sorted(set(main_gauges['hybas_str'].values)) if len(main_gauges) > 0 else []
  w_main_set = set(w_main_list)

  hw_only = hw_gauges[~hw_gauges['hybas_str'].isin(w_main_set)] if len(hw_gauges) > 0 else hw_gauges
  hw_area_by_w = (
      hw_only.groupby('hybas_str')['effective_gauged_area_km2'].sum() * 1e6
      if len(hw_only) > 0
      else pd.Series(dtype=float)
  )

  new_rows_ti = []
  new_rows_tt = []
  new_gauge_areas = {}
  tot_area_dict = hybas_total_areas.to_dict()
  sub_area_dict = hybas_sub_areas.to_dict()

  for w in w_main_list:
    if w not in tot_area_dict:
      continue
    g_w = f'NEW_MAIN_{w}'
    w_area = float(tot_area_dict[w])
    new_gauge_areas[g_w] = w_area
    new_rows_tt.append({'basin1': g_w, 'basin2': g_w, 'overlap_area': w_area})
    new_rows_ti.append({'basin1': w, 'basin2': g_w, 'overlap_area': w_area})
    for d in downstream_map.get(w, ()):
      new_rows_ti.append({'basin1': d, 'basin2': g_w, 'overlap_area': w_area})
      if d in w_main_set:
        g_d = f'NEW_MAIN_{d}'
        new_rows_tt.append({'basin1': g_w, 'basin2': g_d, 'overlap_area': w_area})
        new_rows_tt.append({'basin1': g_d, 'basin2': g_w, 'overlap_area': w_area})
    for g_caravan, ov_area in old_overlap_by_hybas.get(w, ()):
      new_rows_tt.append({'basin1': g_w, 'basin2': g_caravan, 'overlap_area': ov_area})
      new_rows_tt.append({'basin1': g_caravan, 'basin2': g_w, 'overlap_area': ov_area})

  if w_main_set:
    for u in hybas_ids:
      u_area = float(tot_area_dict.get(u, 0.0))
      if u_area <= 0:
        continue
      for d in downstream_map.get(u, ()):
        if d in w_main_set:
          new_rows_ti.append({'basin1': u, 'basin2': f'NEW_MAIN_{d}', 'overlap_area': u_area})
          break

  for w, raw_hw_area_m2 in hw_area_by_w.items():
    if w not in sub_area_dict or w not in tot_area_dict:
      continue
    sub_a_m2 = float(sub_area_dict[w])
    tot_a_m2 = float(tot_area_dict[w])
    hw_area_m2 = min(float(raw_hw_area_m2), 0.99 * sub_a_m2, 0.99 * tot_a_m2)
    if hw_area_m2 <= 0:
      continue
    g_hw = f'NEW_HW_{w}'
    new_gauge_areas[g_hw] = hw_area_m2
    new_rows_tt.append({'basin1': g_hw, 'basin2': g_hw, 'overlap_area': hw_area_m2})
    new_rows_ti.append({'basin1': w, 'basin2': g_hw, 'overlap_area': hw_area_m2})
    for d in downstream_map.get(w, ()):
      new_rows_ti.append({'basin1': d, 'basin2': g_hw, 'overlap_area': hw_area_m2})
      if d in w_main_set:
        g_d = f'NEW_MAIN_{d}'
        new_rows_tt.append({'basin1': g_hw, 'basin2': g_d, 'overlap_area': hw_area_m2})
        new_rows_tt.append({'basin1': g_d, 'basin2': g_hw, 'overlap_area': hw_area_m2})
    for g_caravan, ov_area in old_overlap_by_hybas.get(w, ()):
      if ov_area >= 0.95 * tot_a_m2:
        new_rows_tt.append({'basin1': g_hw, 'basin2': g_caravan, 'overlap_area': hw_area_m2})
        new_rows_tt.append({'basin1': g_caravan, 'basin2': g_hw, 'overlap_area': hw_area_m2})

  df_new_ti = pd.DataFrame(new_rows_ti) if new_rows_ti else pd.DataFrame(columns=['basin1', 'basin2', 'overlap_area'])
  df_new_tt = pd.DataFrame(new_rows_tt) if new_rows_tt else pd.DataFrame(columns=['basin1', 'basin2', 'overlap_area'])

  ti_df = pd.concat([old_ti_df, df_new_ti], ignore_index=True) if not df_new_ti.empty else old_ti_df
  tt_df = pd.concat([tt_sub_38k, df_new_tt], ignore_index=True) if not df_new_tt.empty else tt_sub_38k

  combined_areas = pd.concat([old_caravan_areas, pd.Series(new_gauge_areas, dtype=float), hybas_total_areas])
  combined_areas = combined_areas[~combined_areas.index.duplicated(keep='first')]

  infer_up_df = overlap_mod.calculate_total_upstream_coverage(
      infer_ids=infer_ids, tt_df=tt_df, ti_df=ti_df, basin_areas=combined_areas
  )
  infer_down = overlap_mod.calculate_max_downstream_coverage(
      infer_ids=infer_ids, ti_df=ti_df, basin_areas=combined_areas
  )
  infer_df = infer_up_df.copy()
  infer_df['max_downstream_coverage'] = infer_down
  w_main_valid = [w for w in w_main_list if w in infer_df.index]
  if w_main_valid:
    infer_df.loc[w_main_valid, 'total_upstream_coverage'] = 1.0
    infer_df.loc[w_main_valid, 'max_upstream_coverage'] = 1.0
    infer_df.loc[w_main_valid, 'max_downstream_coverage'] = 1.0
    infer_df.loc[w_main_valid, 'upstream_gauge_count'] = np.maximum(
        infer_df.loc[w_main_valid, 'upstream_gauge_count'].values, 1.0
    )
  infer_df.index = [x if str(x).startswith('hybas_') else f'hybas_{x}' for x in infer_df.index]
  return infer_df


def compute_sec2_metrics(
    tier_key: str,
    tier_label: str,
    cov_df: pd.DataFrame,
    tier_dedup_df: pd.DataFrame | None,
    n_archive_records: int,
    tree_info: dict,
    all_ids: list[str],
):
  """Computes Section 2 metrics over all 1,034,083 watersheds for both physical G & ge95."""
  hybas_ids = tree_info['hybas_ids']
  hybas_str_ids = tree_info['hybas_str_ids']
  next_idx = tree_info['next_idx']
  inc_area = tree_info['inc_area']
  inc_runoff = tree_info['inc_runoff']
  topo_order = tree_info['topo_order']
  n = len(hybas_ids)

  up_cov = cov_df['total_upstream_coverage'].reindex(hybas_str_ids).fillna(0.0).clip(0.0, 1.0).values
  dn_cov = cov_df['max_downstream_coverage'].reindex(hybas_str_ids).fillna(0.0).clip(0.0, 1.0).values

  # Physical gauged polygons vs >=0.95 upstream coverage
  if tier_dedup_df is not None and len(tier_dedup_df) > 0:
    valid_sn = tier_dedup_df[tier_dedup_df['snapped_hybas_id'] > 0]
    snapped_any_set = set(valid_sn['snapped_hybas_id'].astype(np.int64).values)
    snapped_main_set = set(
        valid_sn.loc[~valid_sn['is_subbasin_headwater_gauge'], 'snapped_hybas_id'].astype(np.int64).values
    )
    is_phys_any = np.fromiter((int(h) in snapped_any_set for h in hybas_ids), dtype=bool, count=n)
    is_phys_main = np.fromiter((int(h) in snapped_main_set for h in hybas_ids), dtype=bool, count=n)
    hw_gauges = valid_sn[valid_sn['is_subbasin_headwater_gauge']]
    hw_by_id = hw_gauges.groupby(hw_gauges['snapped_hybas_id'].astype(np.int64))['effective_gauged_area_km2'].sum()
    hw_area = np.array([min(float(hw_by_id.get(int(h), 0.0)), 0.95 * inc_area[i]) for i, h in enumerate(hybas_ids)])
    n_phys_gauges = int(len(tier_dedup_df))
  elif tier_dedup_df is None:
    # Idealized universal ceiling
    is_phys_any = np.ones(n, dtype=bool)
    is_phys_main = np.ones(n, dtype=bool)
    hw_area = np.zeros(n, dtype=np.float64)
    n_phys_gauges = n
  else:
    is_phys_any = np.zeros(n, dtype=bool)
    is_phys_main = np.zeros(n, dtype=bool)
    hw_area = np.zeros(n, dtype=np.float64)
    n_phys_gauges = 0

  is_ge95 = up_cov >= 0.95

  def _run_tree_eq4(is_outlet: np.ndarray, use_hw: bool) -> tuple[np.ndarray, np.ndarray]:
    unmon_inc = np.where(is_outlet, inc_area, np.maximum(inc_area - (hw_area if use_hw else 0.0), 0.0))
    delta_A_in = unmon_inc.copy()
    for u in topo_order:
      v = next_idx[u]
      if v >= 0 and not is_outlet[u]:
        delta_A_in[v] += delta_A_in[u]
    down_reach = np.full(n, np.inf, dtype=np.float64)
    for u in reversed(topo_order):
      if is_outlet[u]:
        down_reach[u] = delta_A_in[u]
      else:
        v = next_idx[u]
        if v >= 0:
          down_reach[u] = down_reach[v]
    ratio_down = np.where(
        np.isinf(down_reach) | (down_reach <= 0), 0.0, np.minimum(delta_A_in / down_reach, 1.0)
    )
    u_tree = (1.0 - up_cov) * np.where(is_outlet, 0.0, 1.0 - ratio_down)
    return u_tree, down_reach

  # Primary Round-2: Physical G-set (snapped physical gauge polygon)
  u_tree_phys, down_reach_phys = _run_tree_eq4(is_phys_any, use_hw=False)
  u_tree_ge95, _ = _run_tree_eq4(is_ge95, use_hw=True)
  u_prod = (1.0 - up_cov) * (1.0 - dn_cov)

  u_up = 1.0 - up_cov
  zero_up = (up_cov <= 0.0).astype(np.float64)
  zero_net = ((up_cov <= 0.0) & (dn_cov <= 0.0)).astype(np.float64)
  out_phys = (~is_phys_any).astype(np.float64)
  out_ge95 = (~is_ge95).astype(np.float64)

  row = {
      'tier_key': tier_key,
      'tier_label': tier_label,
      'n_archive_records': int(n_archive_records),
      'n_physical_gauges': int(n_phys_gauges),
      'n_phys_gauged_polygons': int(is_phys_any.sum()),
      'n_phys_mainstem_polygons': int(is_phys_main.sum()),
      'n_ge95_polygons': int(is_ge95.sum()),
      'total_land_area_mkm2': float(inc_area.sum() / 1e6),
      'total_runoff_km3_yr': float(inc_runoff.sum()),
      # Area-weighted (%)
      'U_out_phys_area_pct': float(np.average(out_phys * 100.0, weights=inc_area)),
      'U_out_ge95_area_pct': float(np.average(out_ge95 * 100.0, weights=inc_area)),
      'U_up_area_pct': float(np.average(u_up * 100.0, weights=inc_area)),
      'U_net_phys_area_pct': float(np.average(u_tree_phys * 100.0, weights=inc_area)),
      'U_net_ge95_area_pct': float(np.average(u_tree_ge95 * 100.0, weights=inc_area)),
      'U_net_prod_area_pct': float(np.average(u_prod * 100.0, weights=inc_area)),
      'U_zero_up_area_pct': float(np.average(zero_up * 100.0, weights=inc_area)),
      'U_zero_net_area_pct': float(np.average(zero_net * 100.0, weights=inc_area)),
      # Runoff-weighted (%)
      'U_out_phys_runoff_pct': float(np.average(out_phys * 100.0, weights=inc_runoff)),
      'U_up_runoff_pct': float(np.average(u_up * 100.0, weights=inc_runoff)),
      'U_net_phys_runoff_pct': float(np.average(u_tree_phys * 100.0, weights=inc_runoff)),
      'U_zero_net_runoff_pct': float(np.average(zero_net * 100.0, weights=inc_runoff)),
      # Unweighted (%)
      'U_out_phys_unwt_pct': float(out_phys.mean() * 100.0),
      'U_up_unwt_pct': float(u_up.mean() * 100.0),
      'U_net_phys_unwt_pct': float(u_tree_phys.mean() * 100.0),
      'U_zero_net_unwt_pct': float(zero_net.mean() * 100.0),
  }

  s_utree = pd.Series(u_tree_phys.astype(np.float32), index=hybas_str_ids, name='U_net_tree_phys').reindex(all_ids)
  s_utree_ge95 = pd.Series(u_tree_ge95.astype(np.float32), index=hybas_str_ids, name='U_net_tree_ge95').reindex(all_ids)
  s_uprod = pd.Series(u_prod.astype(np.float32), index=hybas_str_ids, name='U_net_prod').reindex(all_ids)
  s_dreach = pd.Series(
      np.where(np.isfinite(down_reach_phys), down_reach_phys, -1.0).astype(np.float32),
      index=hybas_str_ids,
      name='intergauge_reach_area_km2',
  ).reindex(all_ids)
  return row, s_utree, s_utree_ge95, s_uprod, s_dreach


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--threads', type=int, default=min(32, os.cpu_count() or 8))
  parser.add_argument(
      '--core-tiers-only',
      action='store_true',
      help='Build only the 7 core manuscript tiers (zero, baseline 8.1k, T1 34.6k, T2 93.8k, T3 121.8k, universal ceilings).',
  )
  parser.add_argument(
      '--reuse-tier-dir',
      type=str,
      default='',
      help='Optional read-only tier_features dir (e.g., round2/tier_features) to reuse unchanged tier0..tier4 COV/OOD with OCSVM normalization.',
  )
  parser.add_argument('--overwrite', action='store_true')
  args = parser.parse_args()

  t0 = time.time()
  print('1. Loading canonical 1,034,083 index, topology, runoff, and gauge databases...', flush=True)
  inf_pq_candidates = [
      os.path.join(CANON_DIR, 'generic_training_data', 'hybas_generic_inference_data.parquet'),
      os.path.join(ROUND2_DIR, 'meta_models', 'hybas_inference_95_float32.parquet'),
  ]
  inf_pq = next((p for p in inf_pq_candidates if os.path.exists(p)), None)
  if inf_pq is not None:
    df_inf_121k = pd.read_parquet(inf_pq)
  else:
    df_inf_121k = load_zarr_df(os.path.join(CANON_DIR, 'generic_training_data', 'hybas_generic_inference_data.zarr'))
  all_ids = df_inf_121k.index.astype(str).tolist()
  assert len(all_ids) == 1034083, f'Expected 1,034,083 basins, got {len(all_ids)}'

  topo_pq = os.path.join(CANON_DIR, 'attributes', 'hybas_l12_topology_and_runoff.parquet')
  if os.path.exists(topo_pq):
    topo_df = pd.read_parquet(topo_pq)
    run_mm = topo_df['run_mm_syr'].fillna(0.0).values
  else:
    topo_df = pd.read_csv(
        os.path.join(CANON_DIR, 'attributes', 'hybas_routing_topology.csv'),
        dtype={'HYBAS_ID': str, 'NEXT_DOWN': str, 'continent': str},
    )
    hatt = pd.read_csv(
        os.path.join(CANON_DIR, 'attributes', 'hybas_attributes.csv'),
        usecols=['Unnamed: 0', 'HYDRO_ATLAS:run_mm_syr'],
        dtype={'Unnamed: 0': str},
    )
    hatt['HYBAS_ID'] = hatt['Unnamed: 0'].str.replace('hybas_', '', regex=False).astype(np.int64)
    hatt = hatt.drop_duplicates('HYBAS_ID').set_index('HYBAS_ID')
    run_mm = hatt['HYDRO_ATLAS:run_mm_syr'].reindex(topo_df['HYBAS_ID'].astype(np.int64).values).fillna(0.0).values
  tree_info = build_tree_arrays(topo_df, run_mm)

  dedup = pd.read_parquet(os.path.join(CANON_DIR, 'attributes', 'global_gauges_deduplicated.parquet'))
  arch = pd.read_parquet(os.path.join(CANON_DIR, 'attributes', 'global_gauges_by_source_archive.parquet'))
  grdc_8811 = pd.read_csv(os.path.join(CANON_DIR, 'attributes', 'grdc_attributes.csv'), index_col=0)
  ca_38k = _resolve_38k_table(
      'caravan_attributes_38k',
      'attributes/caravan_attributes',
      index_col=0,
  )
  ca_121k = _read_table(os.path.join(CANON_DIR, 'attributes', 'caravan_attributes'), index_col=0)
  ca_38k_clean = ca_38k.copy()
  ca_38k_clean.columns = [c.replace('HYDRO_ATLAS:', '') for c in ca_38k_clean.columns]
  ca_121k_clean = ca_121k.copy()
  ca_121k_clean.columns = [c.replace('HYDRO_ATLAS:', '') for c in ca_121k_clean.columns]

  shared_88_cols = sorted(list(set(ca_38k_clean.columns).intersection(set(df_inf_121k.columns))))
  assert len(shared_88_cols) == 88, f'Expected 88 shared cols, got {len(shared_88_cols)}'
  X_test_hybas_88 = df_inf_121k.reindex(all_ids)[shared_88_cols].astype(np.float64)
  assert X_test_hybas_88.isna().sum().sum() == 0

  # Pure feature-space top-quartile reference (ZERO target-Y conditioning!)
  train_candidates = [
      os.path.join(ROUND2_DIR, 'meta_models', 'train_95_float32.parquet'),
      os.path.join(CANON_DIR, 'round2', 'meta_models', 'train_95_float32.parquet'),
      os.path.join(CANON_DIR, 'generic_training_data', 'ungauged_generic_training_data.parquet'),
      os.path.join(CANON_DIR, 'generic_training_data', 'ungauged_generic_training_data.csv'),
  ]
  train_p = next((p for p in train_candidates if os.path.exists(p)), None)
  if train_p is None:
    raise FileNotFoundError(f'Missing training features file; checked {train_candidates}')
  df_train_95 = pd.read_parquet(train_p) if train_p.endswith('.parquet') else pd.read_csv(train_p, index_col=0)
  # Normalize training anomaly_score_ocsvm per split to canonical [-1, ~+0.45] scale!
  df_train_95 = df_train_95.copy()
  n_g_tr = len(df_train_95) // 3
  if len(df_train_95) == 3 * n_g_tr and n_g_tr > 1000:
    oc_col = df_train_95['anomaly_score_ocsvm'].values.astype(np.float64).copy()
    for sp_i in range(3):
      sl = slice(sp_i * n_g_tr, (sp_i + 1) * n_g_tr)
      oc_col[sl] = normalize_ocsvm_series(oc_col[sl])
    df_train_95['anomaly_score_ocsvm'] = oc_col.astype(np.float32)
  else:
    df_train_95['anomaly_score_ocsvm'] = normalize_ocsvm_series(df_train_95['anomaly_score_ocsvm'].values)
  ref_q75_ood = {c: float(df_train_95[c].quantile(0.75)) for c in OOD_COLS}
  dense_q75_cov = {c: float(df_train_95.loc[df_train_95[c] > 0, c].quantile(0.75)) for c in COV_COLS}
  print('   Pure feature-space Q75 OOD reference (OCSVM normalized):', {k: round(v, 5) for k, v in ref_q75_ood.items()}, flush=True)
  print('   Pure feature-space Q75 positive coverage reference:', {k: round(v, 5) for k, v in dense_q75_cov.items()}, flush=True)
  with open(os.path.join(OUT_DIR, 'universal_feature_space_references.json'), 'w') as f:
    json.dump({'ref_q75_ood': ref_q75_ood, 'dense_q75_cov': dense_q75_cov, 'ocsvm_normalized': True}, f, indent=2)

  native_to_cid = dict(zip(arch['native_station_id'].astype(str), arch['canonical_gauge_id'].astype(str)))
  grdc_arch = arch[arch['source_dataset'] == 'GRDC']
  for nid, cid in zip(grdc_arch['native_station_id'].astype(str), grdc_arch['canonical_gauge_id'].astype(str)):
    native_to_cid[f'GRDC_{nid}'] = cid

  old_caravan_tt = _resolve_38k_table(
      'caravan_pairwise_overlap_areas',
      'coverage_fractions/caravan_pairwise_overlap_areas',
  )
  old_caravan_self = old_caravan_tt[old_caravan_tt['basin1'] == old_caravan_tt['basin2']]
  old_caravan_areas = old_caravan_self.set_index('basin1')['overlap_area']
  cont_topo_dfs = {c: topo_df[topo_df['continent'].str.upper() == c] for c in CONTINENTS}

  def compute_cov_for_subset(sub_dedup: pd.DataFrame, valid_38k_override: set[str] | None = None) -> pd.DataFrame:
    cids = set(sub_dedup['canonical_gauge_id'].values)
    valid_38k = valid_38k_override if valid_38k_override is not None else set(
        x for x in ca_38k.index if native_to_cid.get(str(x)) in cids
    )
    net_new = sub_dedup[(~sub_dedup['in_canonical_38069']) & (sub_dedup['snapped_hybas_id'] > 0)][
        ['snapped_hybas_id', 'is_subbasin_headwater_gauge', 'effective_gauged_area_km2']
    ].copy()
    net_new['hybas_str'] = 'hybas_' + net_new['snapped_hybas_id'].astype(np.int64).astype(str)
    tasks = [(cont, valid_38k, net_new, old_caravan_tt, old_caravan_areas, cont_topo_dfs[cont]) for cont in CONTINENTS]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(9, os.cpu_count() or 4),
        mp_context=mp.get_context('spawn'),
    ) as ex:
      dfs = list(ex.map(compute_combined_coverage_continent, tasks))
    df_all = pd.concat(dfs, axis=0)
    return df_all[~df_all.index.duplicated(keep='first')].reindex(all_ids).fillna(0.0)

  # Define exact 11 gauge-based tiers
  t1a_ids = ['CARAVAN_' + str(x).upper() for x in grdc_8811.index]
  t1a_cids = set(native_to_cid[x] for x in t1a_ids if x in native_to_cid)
  t1b_ccby_ids = [x for x in ca_38k.index if str(x).startswith('CARAVAN_') and not str(x).startswith('CARAVAN_GRDC_')]
  t1b_ccby_cids = set(native_to_cid[x] for x in t1b_ccby_ids if x in native_to_cid)
  t1b_full_ids = [x for x in ca_38k.index if str(x).startswith('CARAVAN_')]
  t1b_full_cids = set(native_to_cid[x] for x in t1b_full_ids if x in native_to_cid)
  t2a_ids = [x for x in ca_38k.index if not str(x).startswith('GRDC_')]
  t2a_cids = set(native_to_cid[x] for x in t2a_ids if x in native_to_cid)

  has_caravan = dedup['all_source_datasets'].str.contains('Caravan', na=False)
  has_estreams = dedup['all_source_datasets'].str.contains('EStreams', na=False)
  dedup_34577 = dedup[has_caravan | has_estreams].copy()
  dedup_93827 = dedup[
      dedup['best_data_access_policy'].isin(['open_public_api_or_bulk', 'open_academic_compilation'])
      & dedup['best_observation_type'].isin(['daily_discharge', 'monthly_discharge'])
  ].copy()

  ood_38k_path = os.path.join(BACKUP_38K_DIR, 'anomaly_scores', 'infer_hybas_caravan_anomaly_scores.csv')
  ood_38k_cached = None
  cov_38k_cached = None
  if os.path.exists(ood_38k_path):
    ood_38k_cached = pd.read_csv(ood_38k_path, index_col=0)
    ood_38k_cached.index = [x if str(x).startswith('hybas_') else f'hybas_{x}' for x in ood_38k_cached.index]
    dfs_cov_38k = []
    for cont in CONTINENTS:
      d = _resolve_38k_table(
          f'infer_ml_features_{cont}',
          f'coverage_fractions/hybas/infer_ml_features_{cont}',
          index_col=0,
      )
      idx_col = d['hybas_id'] if 'hybas_id' in d.columns else d.index
      d.index = [x if str(x).startswith('hybas_') else f'hybas_{x}' for x in idx_col]
      dfs_cov_38k.append(d[COV_COLS])
    cov_38k_cached = pd.concat(dfs_cov_38k, axis=0)
    cov_38k_cached = cov_38k_cached[~cov_38k_cached.index.duplicated(keep='first')].reindex(all_ids).fillna(0.0)

  tier_specs = [
      ('tier0_zero_gauges', 'Zero-Gauge Counterfactual (0 Gauges)', dedup.iloc[0:0], 0, 'zero', None),
      ('tier1a_active_post2016_ccby_8811', 'Baseline: Active Post-2016 Caravan (8,811 Caravan / 8,075 Physical)', dedup[dedup['canonical_gauge_id'].isin(t1a_cids)], len(t1a_ids), 'ca38k', set(t1a_ids)),
      ('tier1b_strict_ccby_19491', 'Strict CC-BY-4.0 Regional Caravan (19,491 Caravan / 18,037 Physical)', dedup[dedup['canonical_gauge_id'].isin(t1b_ccby_cids)], len(t1b_ccby_ids), 'ca38k', set(t1b_ccby_ids)),
      ('tier1b_full_caravan_24847', 'Full Caravan Compendium (24,847 Caravan / 22,638 Physical)', dedup[dedup['canonical_gauge_id'].isin(t1b_full_cids)], len(t1b_full_ids), 'ca38k', set(t1b_full_ids)),
      ('tier2a_caravan_estreams_32713', 'Caravan + EStreams (32,713 Archive / 27,876 Physical)', dedup[dedup['canonical_gauge_id'].isin(t2a_cids)], len(t2a_ids), 'ca38k', set(t2a_ids)),
      ('tier2_academic_compilations_38069', 'Caravan + EStreams + GRDC (38,069 Archive / 32,174 Physical)', dedup[dedup['in_canonical_38069']], 38069, 'cached_38k' if ood_38k_cached is not None else 'ca38k', None if ood_38k_cached is not None else set(ca_38k.index)),
      ('tier_caravan_estreams_34577', 'Tier 1: Standardized Open Compilations — Caravan + EStreams (34,577 Physical)', dedup_34577, 40911, 'ca121k', None),
      ('tier2_all_open_academic_34630', 'All Open Academic Compilations (34,630 Physical)', dedup[dedup['in_canonical_38069'] | (dedup['best_data_access_policy'] == 'open_academic_compilation')], 34630, 'ca121k', None),
      ('tier_public_discharge_93827', 'Tier 2: Confirmed Publicly Accessible Discharge Records (93,827 Physical)', dedup_93827, 93827, 'ca121k', None),
      ('tier3_open_public_apis_and_bulk_106410', 'Open Public APIs & Bulk Portals incl. Stage-Only (106,410 Physical)', dedup[dedup['best_data_access_policy'].isin(['open_public_api_or_bulk', 'open_academic_compilation'])], 106410, 'ca121k', None),
      ('tier4_all_publicly_knowable_121819', 'Tier 3: All Publicly Knowable In Situ Gauges (121,819 Physical)', dedup, len(dedup), 'cached_121k', None),
  ]
  if args.core_tiers_only:
    core_keys = {
        'tier0_zero_gauges',
        'tier1a_active_post2016_ccby_8811',
        'tier_caravan_estreams_34577',
        'tier_public_discharge_93827',
        'tier4_all_publicly_knowable_121819',
    }
    tier_specs = [t for t in tier_specs if t[0] in core_keys]
    print(f'   --core-tiers-only active: building {len(tier_specs)} core tiers + 2 universal ceilings.', flush=True)

  sec2_rows = []
  mask_cols = {}
  ood_t1a_cached = None
  ood_t4_cached = None
  reuse_masks_df = None
  if args.reuse_tier_dir:
    rm_p = os.path.join(args.reuse_tier_dir, 'tier_active_and_all_masks_r2.parquet')
    if os.path.exists(rm_p):
      reuse_masks_df = pd.read_parquet(rm_p).reindex(all_ids)
      print(f'   Loaded read-only reuse masks from {rm_p} ({reuse_masks_df.shape})', flush=True)

  for t_key, t_lbl, d_sub, n_arch, mode, ids_override in tier_specs:
    ts = time.time()
    out_pq = os.path.join(OUT_DIR, f'{t_key}.parquet')
    print(f'\nBuilding {t_key} ({len(d_sub):,} physical gauges, mode={mode})...', flush=True)

    df_existing = None
    if (not args.overwrite) and os.path.exists(out_pq):
      df_existing = pd.read_parquet(out_pq)
      if len(df_existing) == len(all_ids):
        cov_df = df_existing[COV_COLS].copy()
        ood_df = df_existing[OOD_COLS].copy()
        ood_df['anomaly_score_ocsvm'] = normalize_ocsvm_series(ood_df['anomaly_score_ocsvm'].values)
        print(f'   Loaded existing {out_pq} ({len(df_existing):,} rows).', flush=True)
      else:
        df_existing = None
    elif args.reuse_tier_dir and mode != 'zero':
      reuse_pq = os.path.join(args.reuse_tier_dir, f'{t_key}.parquet')
      if os.path.exists(reuse_pq):
        df_re = pd.read_parquet(reuse_pq).reindex(all_ids)
        if len(df_re) == len(all_ids):
          cov_df = df_re[COV_COLS].copy()
          ood_df = df_re[OOD_COLS].copy()
          ood_df['anomaly_score_ocsvm'] = normalize_ocsvm_series(ood_df['anomaly_score_ocsvm'].values)
          df_existing = df_re
          print(f'   Reused {reuse_pq} with normalized OCSVM ({len(df_re):,} rows).', flush=True)

    if df_existing is None:
      if mode == 'zero':
        cov_df = pd.DataFrame(0.0, index=all_ids, columns=COV_COLS, dtype=np.float32)
        ood_df = None  # filled from tier1a below!
      elif mode == 'cached_38k':
        cov_df = cov_38k_cached
        ood_df = ood_38k_cached.reindex(all_ids)[OOD_COLS].copy()
        ood_df['anomaly_score_ocsvm'] = normalize_ocsvm_series(ood_df['anomaly_score_ocsvm'].values)
      elif mode == 'cached_121k':
        cov_df = df_inf_121k.reindex(all_ids)[COV_COLS].fillna(0.0)
        ood_df = df_inf_121k.reindex(all_ids)[OOD_COLS].copy()
        ood_df['anomaly_score_ocsvm'] = normalize_ocsvm_series(ood_df['anomaly_score_ocsvm'].values)
      elif mode == 'ca38k':
        cov_df = compute_cov_for_subset(d_sub, valid_38k_override=ids_override)
        ood_df = fast_calculate_anomaly_scores(ca_38k_clean.loc[sorted(ids_override), shared_88_cols], X_test_hybas_88, n_threads=args.threads)
      elif mode == 'ca121k':
        cov_df = compute_cov_for_subset(d_sub)
        ood_df = fast_calculate_anomaly_scores(ca_121k_clean.loc[d_sub['canonical_gauge_id'].values, shared_88_cols], X_test_hybas_88, n_threads=args.threads)

    if t_key == 'tier1a_active_post2016_ccby_8811':
      ood_t1a_cached = ood_df.copy()
    if t_key == 'tier4_all_publicly_knowable_121819':
      ood_t4_cached = ood_df.copy()

    # Wait: if zero-gauge tier was processed first before t1a, compute t1a OOD after loop
    if ood_df is not None:
      for c in ['total_upstream_coverage', 'max_upstream_coverage', 'max_downstream_coverage']:
        cov_df[c] = cov_df[c].clip(0.0, 1.0)
      cov_df['upstream_gauge_count'] = cov_df['upstream_gauge_count'].clip(lower=0.0)

      sec2_row, s_utree, s_utree_ge95, s_uprod, s_dreach = compute_sec2_metrics(
          t_key, t_lbl, cov_df, d_sub, n_arch, tree_info, all_ids
      )
      sec2_rows.append(sec2_row)
      out_df = pd.concat([cov_df[COV_COLS].astype(np.float32), ood_df[OOD_COLS].astype(np.float32), s_utree, s_utree_ge95, s_uprod, s_dreach], axis=1)
      out_df.to_parquet(out_pq)
      print(
          f"   Saved {t_key} ({time.time()-ts:.1f}s) | phys_poly={sec2_row['n_phys_gauged_polygons']:,} "
          f"| U_out_phys={sec2_row['U_out_phys_area_pct']:.2f}% | U_net_phys={sec2_row['U_net_phys_area_pct']:.2f}% "
          f"| U_zero_net={sec2_row['U_zero_net_area_pct']:.2f}%",
          flush=True,
      )

    # Physical & active masks for this tier across ALL 1,034,083 watersheds
    m_ge95_all = (cov_df['total_upstream_coverage'].reindex(all_ids).values >= 0.95)
    valid_sn = d_sub[d_sub['snapped_hybas_id'] > 0]
    sn_all = set('hybas_' + valid_sn['snapped_hybas_id'].astype(np.int64).astype(str))
    sn_main_all = set('hybas_' + valid_sn.loc[~valid_sn['is_subbasin_headwater_gauge'], 'snapped_hybas_id'].astype(np.int64).astype(str))
    m_snap_all = np.fromiter((h in sn_all for h in all_ids), dtype=bool, count=len(all_ids))
    m_snap_main_all = np.fromiter((h in sn_main_all for h in all_ids), dtype=bool, count=len(all_ids))

    # Strict >=2022 daily_discharge active mask for A6 DA sensitivity
    if len(d_sub) == 0:
      m_snap_strict2022 = np.zeros(len(all_ids), dtype=bool)
      m_snap_act = np.zeros(len(all_ids), dtype=bool)
      m_snap_main_act = np.zeros(len(all_ids), dtype=bool)
      m_ge95_act = np.zeros(len(all_ids), dtype=bool)
    else:
      d_strict = d_sub[
          (d_sub['best_observation_type'] == 'daily_discharge')
          & (pd.to_numeric(d_sub['verified_record_end_year'], errors='coerce') >= 2022)
          & (d_sub['snapped_hybas_id'] > 0)
      ]
      sn_strict = set('hybas_' + d_strict['snapped_hybas_id'].astype(np.int64).astype(str))
      m_snap_strict2022 = np.fromiter((h in sn_strict for h in all_ids), dtype=bool, count=len(all_ids))

      if d_sub['is_active_opt1'].all():
        m_snap_act = m_snap_all.copy()
        m_snap_main_act = m_snap_main_all.copy()
        m_ge95_act = m_ge95_all.copy()
      else:
        d_act = d_sub[d_sub['is_active_opt1']]
        valid_sn_a = d_act[d_act['snapped_hybas_id'] > 0]
        sn_act = set('hybas_' + valid_sn_a['snapped_hybas_id'].astype(np.int64).astype(str))
        sn_main_act = set('hybas_' + valid_sn_a.loc[~valid_sn_a['is_subbasin_headwater_gauge'], 'snapped_hybas_id'].astype(np.int64).astype(str))
        m_snap_act = np.fromiter((h in sn_act for h in all_ids), dtype=bool, count=len(all_ids))
        m_snap_main_act = np.fromiter((h in sn_main_act for h in all_ids), dtype=bool, count=len(all_ids))
        if reuse_masks_df is not None and f'{t_key}__ge95_active' in reuse_masks_df.columns:
          m_ge95_act = reuse_masks_df[f'{t_key}__ge95_active'].values.astype(bool)
        else:
          cov_act = compute_cov_for_subset(d_act)
          m_ge95_act = (cov_act['total_upstream_coverage'].reindex(all_ids).values >= 0.95)

    mask_cols[f'{t_key}__snap_all'] = m_snap_all
    mask_cols[f'{t_key}__snap_main_all'] = m_snap_main_all
    mask_cols[f'{t_key}__ge95_all'] = m_ge95_all
    mask_cols[f'{t_key}__union_all'] = m_snap_all | m_ge95_all
    mask_cols[f'{t_key}__snap_active'] = m_snap_act
    mask_cols[f'{t_key}__snap_main_active'] = m_snap_main_act
    mask_cols[f'{t_key}__ge95_active'] = m_ge95_act
    mask_cols[f'{t_key}__union_active'] = m_snap_act | m_ge95_act
    mask_cols[f'{t_key}__snap_active_strict2022'] = m_snap_strict2022

  # Finalize tier0_zero_gauges now that ood_t1a_cached is available
  t0_pq = os.path.join(OUT_DIR, 'tier0_zero_gauges.parquet')
  cov_t0 = pd.DataFrame(0.0, index=all_ids, columns=COV_COLS, dtype=np.float32)
  sec2_t0, s_ut0, s_ut0_g, s_up0, s_dr0 = compute_sec2_metrics(
      'tier0_zero_gauges', 'Zero-Gauge Counterfactual (0 Gauges)', cov_t0, dedup.iloc[0:0], 0, tree_info, all_ids
  )
  sec2_rows.insert(0, sec2_t0)
  pd.concat([cov_t0, ood_t1a_cached[OOD_COLS].astype(np.float32), s_ut0, s_ut0_g, s_up0, s_dr0], axis=1).to_parquet(t0_pq)

  # Universal Gauging Counterfactuals (A2):
  # (1) Conservative bound: tier5_universal_no_ood_clamp (Cup=Cdown=1 + FT+DA only; actual physical tier4 OOD, ZERO OOD clamping!)
  # (2) Idealized ceiling:  tier5_universal_idealized_ceiling (+ scale-consistent Q75 OOD floor on normalized OCSVM/IF/LOF!)
  if ood_t4_cached is not None:
    ood_t4 = ood_t4_cached[OOD_COLS].copy()
  else:
    ood_t4 = df_inf_121k.reindex(all_ids)[OOD_COLS].copy()
    ood_t4['anomaly_score_ocsvm'] = normalize_ocsvm_series(ood_t4['anomaly_score_ocsvm'].values)
  ood_t5 = pd.DataFrame(index=all_ids)
  for c in OOD_COLS:
    ood_t5[c] = np.maximum(ood_t4[c].values, ref_q75_ood[c]).astype(np.float32)

  cov_t5 = pd.DataFrame(index=all_ids, dtype=np.float32)
  cov_t5['total_upstream_coverage'] = np.float32(1.0)
  cov_t5['max_upstream_coverage'] = np.float32(1.0)
  cov_t5['max_downstream_coverage'] = np.float32(1.0)
  cov_t5['upstream_gauge_count'] = np.maximum(
      df_inf_121k['upstream_gauge_count'].reindex(all_ids).fillna(0.0).values, 1.0
  ).astype(np.float32)

  cov_t5_dense = cov_t5.copy()
  for c in COV_COLS:
    cov_t5_dense[c] = np.maximum(cov_t5_dense[c].values, dense_q75_cov[c]).astype(np.float32)

  for t_key, t_lbl, c_df, o_df in [
      ('tier5_universal_no_ood_clamp', 'Universal Gauging Conservative (100% Coverage + FT+DA Only, Actual OOD)', cov_t5, ood_t4),
      ('tier5_universal_idealized_ceiling', 'Universal Gauging Ceiling (100% Coverage + FT+DA + Feature-Space Q75 OOD)', cov_t5, ood_t5),
      ('tier5_dense_corridor_ceiling', 'Dense Corridor Ceiling (Q75 Multi-Gauge Coverage + Q75 OOD)', cov_t5_dense, ood_t5),
  ]:
    r_c, s_u, s_ug, s_up, s_dr = compute_sec2_metrics(t_key, t_lbl, c_df, None, len(all_ids), tree_info, all_ids)
    sec2_rows.append(r_c)
    pd.concat([c_df[COV_COLS], o_df[OOD_COLS].astype(np.float32), s_u, s_ug, s_up, s_dr], axis=1).to_parquet(os.path.join(OUT_DIR, f'{t_key}.parquet'))
    ones = np.ones(len(all_ids), dtype=bool)
    for sfx in ['snap_all', 'snap_main_all', 'ge95_all', 'union_all', 'snap_active', 'snap_main_active', 'ge95_active', 'union_active', 'snap_active_strict2022']:
      mask_cols[f'{t_key}__{sfx}'] = ones

  df_masks = pd.DataFrame(mask_cols, index=all_ids)
  df_masks.to_parquet(os.path.join(OUT_DIR, 'tier_active_and_all_masks_r2.parquet'))
  df_sec2 = pd.DataFrame(sec2_rows)
  df_sec2.to_csv(os.path.join(OUT_DIR, 'sec2_tiered_ungaugedness_summary_r2.csv'), index=False)
  print(f'\nSTAGE B1 COMPLETE in {(time.time() - t0)/60:.1f} min! Saved to {OUT_DIR}', flush=True)


if __name__ == '__main__':
  main()
