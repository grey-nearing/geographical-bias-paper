#!/usr/bin/env python3
"""Builds and caches river-network coverage features (Factor 2), continuous river-network
ungaugedness U(I) (Eq. 4), and 88-feature HydroATLAS anomaly scores (Factor 3) across all
964,084 global watersheds for every institutional data-sharing & licensing tier.
"""

from collections import defaultdict, deque
import concurrent.futures
import glob
import importlib
import json
import os
import sys
import time
import types
import warnings

os.environ.setdefault("GEO_BIAS_DATA_DIR", "./data")
os.environ.setdefault("GEO_BIAS_THIRD_PARTY_DIR", "./data")
os.environ.setdefault("GEO_BIAS_OUTPUT_DIR", "./data")

# Remove any local_pkg path that shadows standard-library calendar
sys.path = [p for p in sys.path if not p.endswith("/local_pkg")]

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import torch
import zarr

torch.set_num_threads(56)

# Inject lightweight gfile shim before importing local_pkg modules
gfile_mod = types.ModuleType("local_pkg.pyglib.gfile")
gfile_mod.Open = open
gfile_mod.Exists = os.path.exists
gfile_mod.MakeDirs = lambda p: os.makedirs(p, exist_ok=True)
gfile_mod.Glob = glob.glob
gfile_mod.ListDir = os.listdir
pyglib_mod = types.ModuleType("local_pkg.pyglib")
pyglib_mod.gfile = gfile_mod
sys.modules["local_pkg.pyglib"] = pyglib_mod
sys.modules["local_pkg.pyglib.gfile"] = gfile_mod

if "./data" not in sys.path:
  sys.path.insert(0, "./data")

from local_pkg.intelligence.gsnearing.geographical_bias_paper_v2.utils import file_paths
from local_pkg.intelligence.gsnearing.geographical_bias_paper_v2.utils import io_utils

overlap_mod = importlib.import_module(
    "local_pkg.intelligence.gsnearing.geographical_bias_paper_v2.0_calculate_overlap_fractions"
)

warnings.filterwarnings("ignore")

CANON_V2_DIR = "./data"
BACKUP_38K_DIR = f"{CANON_V2_DIR}/backup_v2_38069_gauges"
BACKUP_32K_DIR = f"{CANON_V2_DIR}/backup_v2_32713_gauges"
OUT_DIR = "./data/science_results"
FEAT_DIR = f"{OUT_DIR}/features"
os.makedirs(FEAT_DIR, exist_ok=True)

CONTINENTS = ["AF", "AR", "AS", "AU", "EU", "GR", "NA", "SA", "SI"]
COV_COLS = [
    "total_upstream_coverage",
    "max_upstream_coverage",
    "max_downstream_coverage",
    "upstream_gauge_count",
]
OOD_COLS = [
    "anomaly_score_if",
    "anomaly_score_ocsvm",
    "anomaly_score_lof",
]


def fast_calculate_anomaly_scores(X_train: pd.DataFrame, X_test: pd.DataFrame) -> dict:
  """Bit-exact PyTorch-accelerated version of 1_anomaly_scores.calculate_anomaly_scores."""
  feature_order = list(X_train.columns)
  X_test_aligned = X_test[feature_order].copy()
  X_train_aligned = X_train.copy()
  if len(X_train_aligned) > 20000:
    non_area_cols = [
        c
        for c in feature_order
        if "drain_area" not in c.lower() and "drainage area" not in c.lower()
    ]
    X_train_aligned = X_train_aligned.drop_duplicates(
        subset=non_area_cols if non_area_cols else None
    )

  scaler = StandardScaler()
  X_train_scaled = np.ascontiguousarray(scaler.fit_transform(X_train_aligned), dtype=np.float64)
  X_test_scaled = np.ascontiguousarray(scaler.transform(X_test_aligned), dtype=np.float64)

  # 1. Isolation Forest
  isolation_forest = IsolationForest(random_state=42, n_jobs=-1)
  isolation_forest.fit(X_train_scaled)
  scores_if = pd.Series(
      isolation_forest.decision_function(X_test_scaled),
      index=X_test_aligned.index,
  )

  # 2. One-Class SVM
  one_class_svm = OneClassSVM(cache_size=4000)
  one_class_svm.fit(X_train_scaled)
  sv = np.ascontiguousarray(one_class_svm.support_vectors_, dtype=np.float64)
  coef = np.ascontiguousarray(one_class_svm.dual_coef_.ravel(), dtype=np.float64)
  gamma = float(one_class_svm._gamma)
  intercept = float(one_class_svm.intercept_[0])
  sv_sq = np.einsum("ij,ij->i", sv, sv)
  ocsvm_vals = np.empty(len(X_test_scaled), dtype=np.float64)
  bs = 16384
  for i in range(0, len(X_test_scaled), bs):
    xb = X_test_scaled[i : i + bs]
    xb_sq = np.einsum("ij,ij->i", xb, xb)[:, None]
    d2 = xb_sq + sv_sq[None, :] - 2.0 * (xb @ sv.T)
    np.maximum(d2, 0.0, out=d2)
    ocsvm_vals[i : i + bs] = np.exp(-gamma * d2) @ coef + intercept
  scores_ocsvm = pd.Series(ocsvm_vals, index=X_test_aligned.index)

  # 3. Local Outlier Factor (exact float32/float64 cdist on 56 threads, matching sklearn LOF)
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
      # Ensure self distance is strictly -1 so self is index 0
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
      lof_vals[s:e] = score_te + 1.5  # offset_ = -1.5 for contamination='auto'

  scores_lof = pd.Series(lof_vals, index=X_test_aligned.index)
  return {
      "anomaly_score_if": scores_if,
      "anomaly_score_ocsvm": scores_ocsvm,
      "anomaly_score_lof": scores_lof,
  }


def load_zarr_df(zarr_path: str) -> pd.DataFrame:
  root = zarr.open_group(store=zarr_path, mode="r", zarr_format=2)
  feats = root["features"][:]
  hybas_ids = [
      x.decode("utf-8") if isinstance(x, bytes) else str(x)
      for x in root["hybas_id"][:]
  ]
  hybas_ids = [x if x.startswith("hybas_") else f"hybas_{x}" for x in hybas_ids]
  cols = [
      x.decode("utf-8") if isinstance(x, bytes) else str(x)
      for x in root["columns"][:]
  ]
  return pd.DataFrame(feats, index=hybas_ids, columns=cols)


def build_continent_topology(df_topo_cont: pd.DataFrame):
  """Builds downstream_map, hybas_sub_areas, and hybas_total_areas in m^2 in O(N) time."""
  df_t = df_topo_cont.copy()
  df_t["HYBAS_ID"] = (
      df_t["HYBAS_ID"]
      .astype(str)
      .str.replace(r"\.0$", "", regex=True)
      .str.replace("hybas_", "", regex=False)
  )
  df_t["NEXT_DOWN"] = (
      df_t["NEXT_DOWN"]
      .astype(str)
      .str.replace(r"\.0$", "", regex=True)
      .str.replace("hybas_", "", regex=False)
  )
  df_t = df_t.drop_duplicates(subset=["HYBAS_ID"]).set_index("HYBAS_ID")
  hybas_ids = ["hybas_" + idx for idx in df_t.index]
  hybas_sub_areas = pd.Series(df_t["area_m2"].astype(float).values, index=hybas_ids)

  adj = defaultdict(list)
  parent_map_raw = {}
  for child, next_down in zip(df_t.index, df_t["NEXT_DOWN"]):
    if pd.notna(next_down) and next_down not in ("0", 0, "nan", ""):
      parent = str(int(next_down)) if isinstance(next_down, float) else str(next_down)
      adj[parent].append(child)
      parent_map_raw[child] = parent

  sub_area_raw = dict(zip(df_t.index, df_t["area_m2"].astype(float).values))
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
    p_node = "hybas_" + node
    total_areas_dict[p_node] = tot_a

    d_path = []
    curr = node
    visited = {curr}
    while curr in parent_map_raw:
      nxt = parent_map_raw[curr]
      if nxt in visited or nxt not in total_area_raw:
        break
      visited.add(nxt)
      d_path.append("hybas_" + nxt)
      curr = nxt
    downstream_map[p_node] = d_path

  hybas_total_areas = pd.Series(total_areas_dict)
  return hybas_ids, downstream_map, hybas_sub_areas, hybas_total_areas


def compute_subset_38k_coverage_continent(args):
  """Computes exact coverage features for a subset of the 38,069 canonical gauges on one continent."""
  cont, valid_gauge_set, old_caravan_tt, old_caravan_areas, hybas_total_areas = args
  old_ti_path = f"{BACKUP_38K_DIR}/coverage_fractions/hybas/infer_pairwise_overlaps_{cont}.csv"
  old_ti_df = pd.read_csv(old_ti_path)
  old_feat_path = f"{BACKUP_38K_DIR}/coverage_fractions/hybas/infer_ml_features_{cont}.csv"
  old_feat_df = pd.read_csv(old_feat_path, index_col=0)
  infer_ids = (
      list(old_feat_df["hybas_id"].values)
      if "hybas_id" in old_feat_df.columns
      else list(old_feat_df.index)
  )

  ti_sub = old_ti_df[old_ti_df["basin2"].isin(valid_gauge_set)].copy()
  tt_sub = old_caravan_tt[
      old_caravan_tt["basin1"].isin(valid_gauge_set)
      & old_caravan_tt["basin2"].isin(valid_gauge_set)
  ].copy()

  combined_areas = pd.concat([old_caravan_areas, hybas_total_areas])
  combined_areas = combined_areas[~combined_areas.index.duplicated(keep="first")]

  infer_up_df = overlap_mod.calculate_total_upstream_coverage(
      infer_ids=infer_ids,
      tt_df=tt_sub,
      ti_df=ti_sub,
      basin_areas=combined_areas,
  )
  infer_down_series = overlap_mod.calculate_max_downstream_coverage(
      infer_ids=infer_ids,
      ti_df=ti_sub,
      basin_areas=combined_areas,
  )
  infer_df = infer_up_df.copy()
  infer_df["max_downstream_coverage"] = infer_down_series
  return infer_df


def compute_expanded_coverage_continent(args):
  """Computes exact coverage features when adding a subset of net_new_gauges on top of 38,069."""
  cont, net_new_gauges_cont, old_caravan_tt, old_caravan_areas, df_topo_c = args
  hybas_ids, downstream_map, hybas_sub_areas, hybas_total_areas = build_continent_topology(df_topo_c)
  hybas_set = set(hybas_ids)

  old_ti_path = f"{BACKUP_38K_DIR}/coverage_fractions/hybas/infer_pairwise_overlaps_{cont}.csv"
  old_ti_df = pd.read_csv(old_ti_path)
  old_feat_path = f"{BACKUP_38K_DIR}/coverage_fractions/hybas/infer_ml_features_{cont}.csv"
  old_feat_df = pd.read_csv(old_feat_path, index_col=0)
  infer_ids = (
      list(old_feat_df["hybas_id"].values)
      if "hybas_id" in old_feat_df.columns
      else list(old_feat_df.index)
  )

  old_overlap_by_hybas = defaultdict(list)
  for b1, b2, ov_a in zip(old_ti_df["basin1"], old_ti_df["basin2"], old_ti_df["overlap_area"]):
    if ov_a > 0:
      old_overlap_by_hybas[b1].append((b2, float(ov_a)))

  sub_new = net_new_gauges_cont[net_new_gauges_cont["hybas_str"].isin(hybas_set)]
  main_gauges = sub_new[~sub_new["is_subbasin_headwater_gauge"]]
  hw_gauges = sub_new[sub_new["is_subbasin_headwater_gauge"]]

  w_main_list = sorted(set(main_gauges["hybas_str"].values))
  w_main_set = set(w_main_list)

  hw_only = hw_gauges[~hw_gauges["hybas_str"].isin(w_main_set)]
  hw_area_by_w = (
      hw_only.groupby("hybas_str")["effective_gauged_area_km2"].sum() * 1e6
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
    g_w = f"NEW_MAIN_{w}"
    w_area = float(tot_area_dict[w])
    new_gauge_areas[g_w] = w_area
    new_rows_tt.append({"basin1": g_w, "basin2": g_w, "overlap_area": w_area})
    new_rows_ti.append({"basin1": w, "basin2": g_w, "overlap_area": w_area})

    for d in downstream_map.get(w, ()):
      new_rows_ti.append({"basin1": d, "basin2": g_w, "overlap_area": w_area})
      if d in w_main_set:
        g_d = f"NEW_MAIN_{d}"
        new_rows_tt.append({"basin1": g_w, "basin2": g_d, "overlap_area": w_area})
        new_rows_tt.append({"basin1": g_d, "basin2": g_w, "overlap_area": w_area})

    for g_caravan, ov_area in old_overlap_by_hybas.get(w, ()):
      new_rows_tt.append({"basin1": g_w, "basin2": g_caravan, "overlap_area": ov_area})
      new_rows_tt.append({"basin1": g_caravan, "basin2": g_w, "overlap_area": ov_area})

  for u in hybas_ids:
    u_area = float(tot_area_dict.get(u, 0.0))
    if u_area <= 0:
      continue
    for d in downstream_map.get(u, ()):
      if d in w_main_set:
        new_rows_ti.append({
            "basin1": u,
            "basin2": f"NEW_MAIN_{d}",
            "overlap_area": u_area,
        })
        break

  for w, raw_hw_area_m2 in hw_area_by_w.items():
    if w not in sub_area_dict or w not in tot_area_dict:
      continue
    sub_a_m2 = float(sub_area_dict[w])
    tot_a_m2 = float(tot_area_dict[w])
    hw_area_m2 = min(float(raw_hw_area_m2), 0.99 * sub_a_m2, 0.99 * tot_a_m2)
    if hw_area_m2 <= 0:
      continue
    g_hw = f"NEW_HW_{w}"
    new_gauge_areas[g_hw] = hw_area_m2
    new_rows_tt.append({"basin1": g_hw, "basin2": g_hw, "overlap_area": hw_area_m2})
    new_rows_ti.append({"basin1": w, "basin2": g_hw, "overlap_area": hw_area_m2})

    for d in downstream_map.get(w, ()):
      new_rows_ti.append({"basin1": d, "basin2": g_hw, "overlap_area": hw_area_m2})
      if d in w_main_set:
        g_d = f"NEW_MAIN_{d}"
        new_rows_tt.append({"basin1": g_hw, "basin2": g_d, "overlap_area": hw_area_m2})
        new_rows_tt.append({"basin1": g_d, "basin2": g_hw, "overlap_area": hw_area_m2})

    for g_caravan, ov_area in old_overlap_by_hybas.get(w, ()):
      if ov_area >= 0.95 * tot_a_m2:
        new_rows_tt.append({"basin1": g_hw, "basin2": g_caravan, "overlap_area": hw_area_m2})
        new_rows_tt.append({"basin1": g_caravan, "basin2": g_hw, "overlap_area": hw_area_m2})

  df_new_ti = pd.DataFrame(new_rows_ti) if new_rows_ti else pd.DataFrame(columns=["basin1", "basin2", "overlap_area"])
  df_new_tt = pd.DataFrame(new_rows_tt) if new_rows_tt else pd.DataFrame(columns=["basin1", "basin2", "overlap_area"])

  ti_df = pd.concat([old_ti_df, df_new_ti], ignore_index=True) if not df_new_ti.empty else old_ti_df.copy()
  tt_df_cont = pd.concat([old_caravan_tt, df_new_tt], ignore_index=True) if not df_new_tt.empty else old_caravan_tt

  combined_areas = pd.concat([
      old_caravan_areas,
      pd.Series(new_gauge_areas, dtype=float),
      hybas_total_areas,
  ])
  combined_areas = combined_areas[~combined_areas.index.duplicated(keep="first")]

  infer_up_df = overlap_mod.calculate_total_upstream_coverage(
      infer_ids=infer_ids,
      tt_df=tt_df_cont,
      ti_df=ti_df,
      basin_areas=combined_areas,
  )
  infer_down_series = overlap_mod.calculate_max_downstream_coverage(
      infer_ids=infer_ids,
      ti_df=ti_df,
      basin_areas=combined_areas,
  )

  infer_df = infer_up_df.copy()
  infer_df["max_downstream_coverage"] = infer_down_series
  w_main_valid = [w for w in w_main_list if w in infer_df.index]
  if w_main_valid:
    infer_df.loc[w_main_valid, "total_upstream_coverage"] = 1.0
    infer_df.loc[w_main_valid, "max_upstream_coverage"] = 1.0
    infer_df.loc[w_main_valid, "max_downstream_coverage"] = 1.0
    infer_df.loc[w_main_valid, "upstream_gauge_count"] = np.maximum(
        infer_df.loc[w_main_valid, "upstream_gauge_count"].values, 1
    )
  return infer_df


def build_tree_ungaugedness_arrays(topo_df: pd.DataFrame):
  """Precomputes topological sort order and incremental areas across all 1,034,083 HydroBASINS units."""
  hybas_ids = topo_df["HYBAS_ID"].astype(int).values
  hybas_str_ids = [f"hybas_{h}" for h in hybas_ids]
  n = len(hybas_ids)
  id_to_idx = {int(h): i for i, h in enumerate(hybas_ids)}
  str_to_idx = {f"hybas_{int(h)}": i for i, h in enumerate(hybas_ids)}
  next_down_ids = topo_df["NEXT_DOWN"].astype(int).values
  next_idx = np.array(
      [id_to_idx.get(int(nd), -1) for nd in next_down_ids], dtype=np.int32
  )
  inc_area = topo_df["area_m2"].values / 1e6  # km^2

  in_degree = np.zeros(n, dtype=np.int32)
  for i in range(n):
    nd = next_idx[i]
    if nd >= 0:
      in_degree[nd] += 1

  q = deque(np.where(in_degree == 0)[0])
  topo_order = []
  while q:
    u = q.popleft()
    topo_order.append(u)
    v = next_idx[u]
    if v >= 0:
      in_degree[v] -= 1
      if in_degree[v] == 0:
        q.append(v)
  topo_order = np.array(topo_order, dtype=np.int32)
  return {
      "hybas_ids": hybas_ids,
      "hybas_str_ids": hybas_str_ids,
      "id_to_idx": id_to_idx,
      "str_to_idx": str_to_idx,
      "next_idx": next_idx,
      "inc_area": inc_area,
      "topo_order": topo_order,
  }


def compute_sec2_metrics_for_tier(
    tier_key: str,
    tier_label: str,
    cov_df_all: pd.DataFrame,
    tier_dedup_df: pd.DataFrame,
    n_archive_records: int,
    tree_info: dict,
    study_ids: list[str],
    study_idx: np.ndarray,
    e_total_study: np.ndarray,
) -> tuple[dict, pd.Series, pd.Series]:
  """Computes all Section 2 Topological & River-Network Ungaugedness metrics for a tier."""
  hybas_ids = tree_info["hybas_ids"]
  hybas_str_ids = tree_info["hybas_str_ids"]
  next_idx = tree_info["next_idx"]
  inc_area = tree_info["inc_area"]
  topo_order = tree_info["topo_order"]
  n = len(hybas_ids)

  up_cov_all = (
      cov_df_all["total_upstream_coverage"]
      .reindex(hybas_str_ids)
      .fillna(0.0)
      .clip(0.0, 1.0)
      .values
  )
  dn_cov_all = (
      cov_df_all["max_downstream_coverage"]
      .reindex(hybas_str_ids)
      .fillna(0.0)
      .clip(0.0, 1.0)
      .values
  )
  is_outlet_gauged = up_cov_all >= 0.95

  if tier_dedup_df is not None and len(tier_dedup_df) > 0:
    hw_gauges = tier_dedup_df[
        (tier_dedup_df["snapped_hybas_id"] > 0)
        & (tier_dedup_df["is_subbasin_headwater_gauge"])
    ]
    hw_by_id = hw_gauges.groupby(hw_gauges["snapped_hybas_id"].astype(int))[
        "effective_gauged_area_km2"
    ].sum()
    hw_area = np.array([
        min(float(hw_by_id.get(int(h), 0.0)), 0.95 * inc_area[i])
        for i, h in enumerate(hybas_ids)
    ])
    n_phys_gauges = int(len(tier_dedup_df))
    n_gauged_hybas_snapped = int(
        tier_dedup_df.loc[tier_dedup_df["snapped_hybas_id"] > 0, "snapped_hybas_id"].nunique()
    )
  else:
    hw_area = np.zeros(n, dtype=np.float64)
    n_phys_gauges = int(n_archive_records)
    n_gauged_hybas_snapped = int((up_cov_all[study_idx] >= 0.95).sum())

  unmon_inc = np.where(is_outlet_gauged, inc_area, inc_area - hw_area)

  delta_A_in = unmon_inc.copy()
  for u in topo_order:
    v = next_idx[u]
    if v >= 0:
      if not is_outlet_gauged[u]:
        delta_A_in[v] += delta_A_in[u]

  down_reach_area = np.full(n, np.inf, dtype=np.float64)
  for u in reversed(topo_order):
    if is_outlet_gauged[u]:
      down_reach_area[u] = delta_A_in[u]
    else:
      v = next_idx[u]
      if v >= 0:
        down_reach_area[u] = down_reach_area[v]

  U_up_exact = 1.0 - up_cov_all
  ratio_down = np.where(
      np.isinf(down_reach_area) | (down_reach_area <= 0),
      0.0,
      np.minimum(delta_A_in / down_reach_area, 1.0),
  )
  U_net_tree = U_up_exact * np.where(is_outlet_gauged, 0.0, 1.0 - ratio_down)
  U_net_prod = (1.0 - up_cov_all) * (1.0 - dn_cov_all)

  # Restrict to the 964,084 study watersheds
  w_inc = inc_area[study_idx]
  tot_area_mkm2 = float(w_inc.sum()) / 1e6

  cu_study = up_cov_all[study_idx]
  cd_study = dn_cov_all[study_idx]
  u_up_study = 1.0 - cu_study
  u_tree_study = U_net_tree[study_idx]
  u_prod_study = U_net_prod[study_idx]

  outlet_ungauged = (cu_study < 0.95).astype(np.float64)
  zero_up = (cu_study <= 0.0).astype(np.float64)
  zero_net = ((cu_study <= 0.0) & (cd_study <= 0.0)).astype(np.float64)

  n_outlet_gauged_95 = int((cu_study >= 0.95).sum())
  exp_zero_net = float(e_total_study[zero_net > 0.5].sum())
  exp_zero_up = float(e_total_study[zero_up > 0.5].sum())
  exp_outlet_ungauged = float(e_total_study[outlet_ungauged > 0.5].sum())
  tot_exp = float(e_total_study.sum())

  q25_tree, q50_tree, q75_tree = np.percentile(u_tree_study * 100.0, [25, 50, 75])

  metrics = {
      "tier_key": tier_key,
      "tier_label": tier_label,
      "n_archive_records": int(n_archive_records),
      "n_physical_gauges": int(n_phys_gauges),
      "n_gauged_hybas_snapped": int(n_gauged_hybas_snapped),
      "n_outlet_gauged_ge95": int(n_outlet_gauged_95),
      "total_land_area_mkm2": tot_area_mkm2,
      # Area-weighted percentages (canonical Figure 1 / Section 2 headline basis)
      "pct_outlet_ungauged_area_wt": float(np.average(outlet_ungauged * 100.0, weights=w_inc)),
      "pct_upstream_area_ungauged_area_wt": float(np.average(u_up_study * 100.0, weights=w_inc)),
      "pct_zero_upstream_area_wt": float(np.average(zero_up * 100.0, weights=w_inc)),
      "pct_U_net_tree_eq4_area_wt": float(np.average(u_tree_study * 100.0, weights=w_inc)),
      "pct_U_net_prod_area_wt": float(np.average(u_prod_study * 100.0, weights=w_inc)),
      "pct_entire_network_ungauged_area_wt": float(np.average(zero_net * 100.0, weights=w_inc)),
      # Unweighted watershed means across 964,084 sub-basins
      "pct_outlet_ungauged_unweighted": float(outlet_ungauged.mean() * 100.0),
      "pct_upstream_area_ungauged_unweighted": float(u_up_study.mean() * 100.0),
      "pct_zero_upstream_unweighted": float(zero_up.mean() * 100.0),
      "pct_U_net_tree_eq4_unweighted": float(u_tree_study.mean() * 100.0),
      "pct_U_net_prod_unweighted": float(u_prod_study.mean() * 100.0),
      "pct_entire_network_ungauged_unweighted": float(zero_net.mean() * 100.0),
      "n_basins_entire_network_ungauged": int(zero_net.sum()),
      "n_basins_zero_upstream": int(zero_up.sum()),
      "n_basins_outlet_ungauged": int(outlet_ungauged.sum()),
      "U_net_tree_q25": float(q25_tree),
      "U_net_tree_median": float(q50_tree),
      "U_net_tree_q75": float(q75_tree),
      # Expected annual flood exposure in ungauged networks
      "exposure_entire_network_ungauged_people_yr": exp_zero_net,
      "pct_exposure_entire_network_ungauged": float(100.0 * exp_zero_net / tot_exp),
      "exposure_zero_upstream_people_yr": exp_zero_up,
      "pct_exposure_zero_upstream": float(100.0 * exp_zero_up / tot_exp),
      "exposure_outlet_ungauged_people_yr": exp_outlet_ungauged,
      "pct_exposure_outlet_ungauged": float(100.0 * exp_outlet_ungauged / tot_exp),
  }
  return (
      metrics,
      pd.Series(u_tree_study, index=study_ids, name="U_net_tree"),
      pd.Series(u_prod_study, index=study_ids, name="U_net_prod"),
  )


def main():
  t0 = time.time()
  file_paths.init("5.0-year-2.0-window-recall")

  print("1. Loading study watersheds (964,084), routing topology (1,034,083), and gauge databases...", flush=True)
  df_basin_v2 = pd.read_parquet(
      f"{CANON_V2_DIR}/v2_exposure_blockers/statistics/basin_exposure_and_blockers_v2.parquet"
  )
  study_ids = [
      x if str(x).startswith("hybas_") else f"hybas_{x}"
      for x in df_basin_v2.index
  ]
  df_basin_v2.index = study_ids
  e_total_study = df_basin_v2["E_total"].values.astype(np.float64)

  topo_df = pd.read_csv(
      f"{CANON_V2_DIR}/attributes/hybas_routing_topology.csv",
      dtype={"HYBAS_ID": str, "NEXT_DOWN": str, "continent": str},
  )
  tree_info = build_tree_ungaugedness_arrays(topo_df)
  study_idx = np.array([tree_info["str_to_idx"][h] for h in study_ids], dtype=np.int32)

  dedup = pd.read_parquet(f"{CANON_V2_DIR}/attributes/global_gauges_deduplicated.parquet")
  arch = pd.read_parquet(f"{CANON_V2_DIR}/attributes/global_gauges_by_source_archive.parquet")
  grdc_8811 = pd.read_csv(f"{CANON_V2_DIR}/attributes/grdc_attributes.csv", index_col=0)
  ca_38k = pd.read_csv(f"{BACKUP_38K_DIR}/attributes/caravan_attributes.csv", index_col=0)
  ca_121k = pd.read_csv(f"{CANON_V2_DIR}/attributes/caravan_attributes.csv", index_col=0)

  ca_38k_clean = ca_38k.copy()
  ca_38k_clean.columns = [c.replace("HYDRO_ATLAS:", "") for c in ca_38k_clean.columns]
  ca_121k_clean = ca_121k.copy()
  ca_121k_clean.columns = [c.replace("HYDRO_ATLAS:", "") for c in ca_121k_clean.columns]

  print("2. Loading cached Zarr inference matrices for Tier 4 (121,819) and Tier 2a (32,713)...", flush=True)
  df_zarr_121k = load_zarr_df(f"{CANON_V2_DIR}/generic_training_data/hybas_generic_inference_data.zarr")
  df_zarr_32k = load_zarr_df(f"{BACKUP_32K_DIR}/generic_training_data/hybas_generic_inference_data.zarr")

  shared_88_cols = sorted(list(set(ca_38k_clean.columns).intersection(set(df_zarr_121k.columns))))
  print(f"   Verified {len(shared_88_cols)} shared numeric HydroATLAS columns for anomaly scoring.", flush=True)
  X_test_hybas_88 = df_zarr_121k.loc[study_ids, shared_88_cols].copy()

  native_to_cid = dict(zip(arch["native_station_id"].astype(str), arch["canonical_gauge_id"].astype(str)))
  grdc_arch = arch[arch["source_dataset"] == "GRDC"]
  for nid, cid in zip(grdc_arch["native_station_id"].astype(str), grdc_arch["canonical_gauge_id"].astype(str)):
    native_to_cid[f"GRDC_{nid}"] = cid

  x_train_u, y_train_u_5 = io_utils.load_and_preprocess_data(
      "ungauged", "5.0-year-2.0-window-recall", None
  )
  top_q_thresh_u = y_train_u_5.quantile(0.75)
  top_q_basins_u = x_train_u.loc[y_train_u_5 >= top_q_thresh_u]
  ref_medians_u = top_q_basins_u.median()
  dense_q75_cov = {
      c: float(x_train_u[x_train_u[c] > 0][c].quantile(0.75))
      for c in COV_COLS
      if c in x_train_u.columns
  }

  old_caravan_tt = pd.read_csv(f"{BACKUP_38K_DIR}/coverage_fractions/caravan_pairwise_overlap_areas.csv")
  old_caravan_self = old_caravan_tt[old_caravan_tt["basin1"] == old_caravan_tt["basin2"]]
  old_caravan_areas = old_caravan_self.set_index("basin1")["overlap_area"]

  cont_total_areas = {}
  cont_topo_dfs = {}
  for cont in CONTINENTS:
    df_tc = topo_df[topo_df["continent"].str.upper() == cont]
    cont_topo_dfs[cont] = df_tc
    _, _, _, h_tot = build_continent_topology(df_tc)
    cont_total_areas[cont] = h_tot

  def run_38k_subset_coverage(valid_ids_list: list[str]) -> pd.DataFrame:
    valid_set = set(valid_ids_list)
    tasks = [
        (cont, valid_set, old_caravan_tt, old_caravan_areas, cont_total_areas[cont])
        for cont in CONTINENTS
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=9) as ex:
      dfs = list(ex.map(compute_subset_38k_coverage_continent, tasks))
    df_all = pd.concat(dfs, axis=0)
    df_all.index = [x if str(x).startswith("hybas_") else f"hybas_{x}" for x in df_all.index]
    return df_all[~df_all.index.duplicated(keep="first")]

  def run_expanded_coverage(net_new_df: pd.DataFrame) -> pd.DataFrame:
    tasks = [
        (cont, net_new_df, old_caravan_tt, old_caravan_areas, cont_topo_dfs[cont])
        for cont in CONTINENTS
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=9) as ex:
      dfs = list(ex.map(compute_expanded_coverage_continent, tasks))
    df_all = pd.concat(dfs, axis=0)
    df_all.index = [x if str(x).startswith("hybas_") else f"hybas_{x}" for x in df_all.index]
    return df_all[~df_all.index.duplicated(keep="first")]

  sec2_summaries = []

  def save_and_record_tier(
      tier_key: str,
      tier_label: str,
      cov_df_all: pd.DataFrame,
      ood_df_study: pd.DataFrame,
      tier_dedup_df: pd.DataFrame,
      n_archive_records: int,
  ):
    sec2_dict, s_utree, s_uprod = compute_sec2_metrics_for_tier(
        tier_key=tier_key,
        tier_label=tier_label,
        cov_df_all=cov_df_all,
        tier_dedup_df=tier_dedup_df,
        n_archive_records=n_archive_records,
        tree_info=tree_info,
        study_ids=study_ids,
        study_idx=study_idx,
        e_total_study=e_total_study,
    )
    sec2_summaries.append(sec2_dict)

    cov_study = cov_df_all.reindex(study_ids)[COV_COLS].fillna(0.0).copy()
    cov_study["total_upstream_coverage"] = cov_study["total_upstream_coverage"].clip(0.0, 1.0)
    cov_study["max_upstream_coverage"] = cov_study["max_upstream_coverage"].clip(0.0, 1.0)
    cov_study["max_downstream_coverage"] = cov_study["max_downstream_coverage"].clip(0.0, 1.0)
    cov_study["upstream_gauge_count"] = cov_study["upstream_gauge_count"].clip(lower=0.0)

    out_df = pd.concat(
        [cov_study, ood_df_study.reindex(study_ids)[OOD_COLS], s_utree, s_uprod],
        axis=1,
    )
    out_pq = f"{FEAT_DIR}/{tier_key}.parquet"
    out_df.to_parquet(out_pq)
    print(
        f"   [{tier_key}] phys_gauges={sec2_dict['n_physical_gauges']:,} | "
        f"outlet_ungauged={sec2_dict['pct_outlet_ungauged_area_wt']:.2f}% (area-wt) / {sec2_dict['pct_outlet_ungauged_unweighted']:.2f}% (unwt) | "
        f"U(I)_tree={sec2_dict['pct_U_net_tree_eq4_area_wt']:.2f}% (area-wt) | "
        f"0%-net_area={sec2_dict['pct_entire_network_ungauged_area_wt']:.2f}% | "
        f"0%-net_exp={sec2_dict['exposure_entire_network_ungauged_people_yr']/1e6:.2f}M ({sec2_dict['pct_exposure_entire_network_ungauged']:.2f}%)",
        flush=True,
    )
    return out_df

  # -------------------------------------------------------------------------
  # Tier 1a: Active Post-2016 CC-BY ML Training Network (8,811 Caravan records / 8,075-8,104 physical)
  # -------------------------------------------------------------------------
  t_step = time.time()
  print("\n3. Building Tier 1a (8,811 Active Post-2016 CC-BY Training Network)...", flush=True)
  t1a_ids = ["CARAVAN_" + str(x).upper() for x in grdc_8811.index]
  t1a_cids = set(native_to_cid[x] for x in t1a_ids if x in native_to_cid)
  t1a_dedup = dedup[dedup["canonical_gauge_id"].isin(t1a_cids)]
  cov_t1a = run_38k_subset_coverage(t1a_ids)
  t1a_pq = f"{FEAT_DIR}/tier1a_active_post2016_ccby_8811.parquet"
  if os.path.exists(t1a_pq):
    ood_t1a = pd.read_parquet(t1a_pq)[OOD_COLS]
  else:
    ood_t1a = pd.DataFrame(
        fast_calculate_anomaly_scores(ca_38k_clean.loc[t1a_ids, shared_88_cols], X_test_hybas_88),
        index=study_ids,
    )
  save_and_record_tier(
      "tier1a_active_post2016_ccby_8811",
      "Tier 1a: Active Post-2016 CC-BY ML Training Network (8,811 Caravan / 8,104 Physical)",
      cov_t1a,
      ood_t1a,
      t1a_dedup,
      len(t1a_ids),
  )
  print(f"   Done Tier 1a in {time.time() - t_step:.1f}s.", flush=True)

  # -------------------------------------------------------------------------
  # Tier 0: Counterfactual Zero-Gauge Baseline (0 gauges)
  # -------------------------------------------------------------------------
  print("\n4. Building Tier 0 (Counterfactual Zero-Gauge Baseline)...", flush=True)
  cov_t0 = pd.DataFrame(0.0, index=tree_info["hybas_str_ids"], columns=COV_COLS)
  save_and_record_tier(
      "tier0_zero_gauges",
      "Tier 0: Counterfactual Zero-Gauge Baseline (0 Gauges)",
      cov_t0,
      ood_t1a,
      dedup.iloc[0:0],
      0,
  )

  # -------------------------------------------------------------------------
  # Tier 1b (Strict CC-BY): All Strictly CC-BY-4.0 Regional Caravan (19,491 records / 18,037 physical)
  # -------------------------------------------------------------------------
  t_step = time.time()
  print("\n5. Building Tier 1b-CCBY (19,491 Strict CC-BY-4.0 Regional Caravan)...", flush=True)
  t1b_ccby_ids = [
      x for x in ca_38k.index
      if str(x).startswith("CARAVAN_") and not str(x).startswith("CARAVAN_GRDC_")
  ]
  t1b_ccby_cids = set(native_to_cid[x] for x in t1b_ccby_ids if x in native_to_cid)
  t1b_ccby_dedup = dedup[dedup["canonical_gauge_id"].isin(t1b_ccby_cids)]
  cov_t1b_ccby = run_38k_subset_coverage(t1b_ccby_ids)
  t1b_ccby_pq = f"{FEAT_DIR}/tier1b_strict_ccby_19491.parquet"
  if os.path.exists(t1b_ccby_pq):
    ood_t1b_ccby = pd.read_parquet(t1b_ccby_pq)[OOD_COLS]
  else:
    ood_t1b_ccby = pd.DataFrame(
        fast_calculate_anomaly_scores(ca_38k_clean.loc[t1b_ccby_ids, shared_88_cols], X_test_hybas_88),
        index=study_ids,
    )
  save_and_record_tier(
      "tier1b_strict_ccby_19491",
      "Tier 1b (Strict CC-BY): Regional Caravan CC-BY-4.0 Archive (19,491 Caravan / 18,037 Physical)",
      cov_t1b_ccby,
      ood_t1b_ccby,
      t1b_ccby_dedup,
      len(t1b_ccby_ids),
  )
  print(f"   Done Tier 1b-CCBY in {time.time() - t_step:.1f}s.", flush=True)

  # -------------------------------------------------------------------------
  # Tier 1b (Full Caravan): Full Caravan Compendium (24,847 records / 22,638 physical)
  # -------------------------------------------------------------------------
  t_step = time.time()
  print("\n6. Building Tier 1b-Full (24,847 Full Caravan Archive)...", flush=True)
  t1b_full_ids = [x for x in ca_38k.index if str(x).startswith("CARAVAN_")]
  t1b_full_cids = set(native_to_cid[x] for x in t1b_full_ids if x in native_to_cid)
  t1b_full_dedup = dedup[dedup["canonical_gauge_id"].isin(t1b_full_cids)]
  cov_t1b_full = run_38k_subset_coverage(t1b_full_ids)
  ood_t1b_full = pd.DataFrame(
      fast_calculate_anomaly_scores(
          ca_38k_clean.loc[t1b_full_ids, shared_88_cols], X_test_hybas_88
      ),
      index=study_ids,
  )
  save_and_record_tier(
      "tier1b_full_caravan_24847",
      "Tier 1b (Full Caravan): All Caravan Including Pre-2016 & GRDC Sub-Dataset (24,847 Caravan / 22,638 Physical)",
      cov_t1b_full,
      ood_t1b_full,
      t1b_full_dedup,
      len(t1b_full_ids),
  )
  print(f"   Done Tier 1b-Full in {time.time() - t_step:.1f}s.", flush=True)

  # -------------------------------------------------------------------------
  # Tier 2a: Caravan + EStreams (32,713 records / 27,876 physical, cached in backup_v2_32713_gauges)
  # -------------------------------------------------------------------------
  print("\n7. Building Tier 2a (32,713 Caravan + EStreams, from cached Zarr)...", flush=True)
  t2a_ids = [x for x in ca_38k.index if not str(x).startswith("GRDC_")]
  t2a_cids = set(native_to_cid[x] for x in t2a_ids if x in native_to_cid)
  t2a_dedup = dedup[dedup["canonical_gauge_id"].isin(t2a_cids)]
  save_and_record_tier(
      "tier2a_caravan_estreams_32713",
      "Tier 2a: Caravan + EStreams Compilation (32,713 Archive / 27,876 Physical)",
      df_zarr_32k[COV_COLS],
      df_zarr_32k.loc[study_ids, OOD_COLS],
      t2a_dedup,
      len(t2a_ids),
  )

  # -------------------------------------------------------------------------
  # Tier 2: Open Academic Compilations (38,069 Caravan + EStreams + GRDC / 32,174 physical)
  # -------------------------------------------------------------------------
  print("\n8. Building Tier 2 (38,069 Caravan + EStreams + GRDC, from cached backup_v2_38069_gauges)...", flush=True)
  dfs_cov_38k = []
  for cont in CONTINENTS:
    d = pd.read_csv(f"{BACKUP_38K_DIR}/coverage_fractions/hybas/infer_ml_features_{cont}.csv", index_col=0)
    idx_col = d["hybas_id"] if "hybas_id" in d.columns else d.index
    d.index = [x if str(x).startswith("hybas_") else f"hybas_{x}" for x in idx_col]
    dfs_cov_38k.append(d[COV_COLS])
  cov_t2_38k = pd.concat(dfs_cov_38k, axis=0)
  cov_t2_38k = cov_t2_38k[~cov_t2_38k.index.duplicated(keep="first")]

  ood_t2_38k = pd.read_csv(
      f"{BACKUP_38K_DIR}/anomaly_scores/infer_hybas_caravan_anomaly_scores.csv",
      index_col=0,
  )
  ood_t2_38k.index = [
      x if str(x).startswith("hybas_") else f"hybas_{x}" for x in ood_t2_38k.index
  ]
  t2_38k_dedup = dedup[dedup["in_canonical_38069"]]
  save_and_record_tier(
      "tier2_academic_compilations_38069",
      "Tier 2: Open Academic Compilations — Caravan + EStreams + GRDC (38,069 Archive / 32,174 Physical)",
      cov_t2_38k,
      ood_t2_38k.reindex(study_ids)[OOD_COLS],
      t2_38k_dedup,
      38069,
  )

  # -------------------------------------------------------------------------
  # Tier 2+ (All Open Academic Compilations in Unified Catalog: 34,630 physical stations)
  # -------------------------------------------------------------------------
  t_step = time.time()
  print("\n9. Building Tier 2+ (34,630 Physical Stations Across All Open Academic Compilations)...", flush=True)
  t2_all_acad_dedup = dedup[
      dedup["in_canonical_38069"] | (dedup["best_data_access_policy"] == "open_academic_compilation")
  ].copy()
  net_new_t2_acad = t2_all_acad_dedup[
      (~t2_all_acad_dedup["in_canonical_38069"]) & (t2_all_acad_dedup["snapped_hybas_id"] > 0)
  ].copy()
  net_new_t2_acad["hybas_str"] = "hybas_" + net_new_t2_acad["snapped_hybas_id"].astype(str)
  cov_t2_all_acad = run_expanded_coverage(net_new_t2_acad)
  ood_t2_all_acad = pd.DataFrame(
      fast_calculate_anomaly_scores(
          ca_121k_clean.loc[t2_all_acad_dedup["canonical_gauge_id"].values, shared_88_cols],
          X_test_hybas_88,
      ),
      index=study_ids,
  )
  save_and_record_tier(
      "tier2_all_open_academic_34630",
      "Tier 2+ (All Academic): All Open Academic Compilations Including Full GRDC/GSHA/CAMELS (34,630 Physical)",
      cov_t2_all_acad,
      ood_t2_all_acad,
      t2_all_acad_dedup,
      len(t2_all_acad_dedup),
  )
  print(f"   Done Tier 2+ in {time.time() - t_step:.1f}s.", flush=True)

  # -------------------------------------------------------------------------
  # Tier 3: All Open Public APIs & Bulk Downloads (106,410 physical stations)
  # -------------------------------------------------------------------------
  t_step = time.time()
  print("\n10. Building Tier 3 (106,410 Physical Stations — Open Public APIs, Bulk Portals & Academic)...", flush=True)
  t3_dedup = dedup[
      dedup["best_data_access_policy"].isin(["open_public_api_or_bulk", "open_academic_compilation"])
  ].copy()
  net_new_t3 = t3_dedup[
      (~t3_dedup["in_canonical_38069"]) & (t3_dedup["snapped_hybas_id"] > 0)
  ].copy()
  net_new_t3["hybas_str"] = "hybas_" + net_new_t3["snapped_hybas_id"].astype(str)
  cov_t3 = run_expanded_coverage(net_new_t3)
  ood_t3 = pd.DataFrame(
      fast_calculate_anomaly_scores(
          ca_121k_clean.loc[t3_dedup["canonical_gauge_id"].values, shared_88_cols],
          X_test_hybas_88,
      ),
      index=study_ids,
  )
  save_and_record_tier(
      "tier3_open_public_apis_and_bulk_106410",
      "Tier 3: All Open Public APIs, National Portals & Bulk Downloads (106,410 Physical)",
      cov_t3,
      ood_t3,
      t3_dedup,
      len(t3_dedup),
  )
  print(f"   Done Tier 3 in {time.time() - t_step:.1f}s.", flush=True)

  # -------------------------------------------------------------------------
  # Tier 4: All Publicly Knowable Gauges (121,819 physical stations, cached in canonical_v2)
  # -------------------------------------------------------------------------
  print("\n11. Building Tier 4 (121,819 All Publicly Knowable Physical Stations, from cached Zarr)...", flush=True)
  save_and_record_tier(
      "tier4_all_publicly_knowable_121819",
      "Tier 4: All Publicly Knowable In Situ Gauges Including Restricted & Index-Only (121,819 Physical)",
      df_zarr_121k[COV_COLS],
      df_zarr_121k.loc[study_ids, OOD_COLS],
      dedup,
      len(dedup),
  )

  # -------------------------------------------------------------------------
  # Tier 5: Idealized Universal Physical Coverage (100% River-Network Coverage + Top-Quartile Climate)
  # -------------------------------------------------------------------------
  print("\n12. Building Tier 5 (Idealized Universal 100% Coverage + Top-Quartile Climate Similarity)...", flush=True)
  cov_t5 = pd.DataFrame(index=tree_info["hybas_str_ids"], columns=COV_COLS, dtype=float)
  cov_t5["total_upstream_coverage"] = 1.0
  cov_t5["max_upstream_coverage"] = 1.0
  cov_t5["max_downstream_coverage"] = 1.0
  cov_t5["upstream_gauge_count"] = np.maximum(
      df_zarr_121k["upstream_gauge_count"].reindex(tree_info["hybas_str_ids"]).fillna(0.0).values,
      1.0,
  )
  ood_t5 = df_zarr_121k.loc[study_ids, OOD_COLS].copy()
  for c in OOD_COLS:
    if c in ref_medians_u.index:
      ood_t5[c] = (
          np.maximum(ood_t5[c].values, float(ref_medians_u[c]))
          if c != "anomaly_score_lof"
          else float(ref_medians_u[c])
      )
  save_and_record_tier(
      "tier5_universal_idealized_ceiling",
      "Tier 5: Idealized Universal Physical Coverage (100% Coverage + Top-Quartile Climate Similarity)",
      cov_t5,
      ood_t5,
      None,
      len(study_ids),
  )

  cov_t5_dense = cov_t5.copy()
  for c in COV_COLS:
    if c in dense_q75_cov:
      cov_t5_dense[c] = np.maximum(cov_t5_dense[c].values, dense_q75_cov[c])
  save_and_record_tier(
      "tier5_dense_corridor_ceiling",
      "Tier 5 (Dense Corridor): Multi-Gauge Nested Corridor Coverage (75th Pct) + Top-Quartile Climate",
      cov_t5_dense,
      ood_t5,
      None,
      len(study_ids),
  )

  df_sec2 = pd.DataFrame(sec2_summaries)
  df_sec2.to_csv(f"{OUT_DIR}/sec2_tiered_ungaugedness_summary.csv", index=False)
  with open(f"{OUT_DIR}/sec2_tiered_ungaugedness_summary.json", "w") as f:
    json.dump(sec2_summaries, f, indent=2)

  print(f"\nALL TIER FEATURES AND SECTION 2 METRICS COMPLETED IN {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
  main()
