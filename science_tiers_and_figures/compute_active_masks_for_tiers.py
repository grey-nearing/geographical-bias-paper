#!/usr/bin/env python3
"""Computes and caches exact Active-Only (Option 1: is_active_opt1 == True) and All-Gauges
watershed masks (snapped, ge95 upstream coverage, and union) across all 964,084 study watersheds
for all tiers in TIER_ORDER and ANCHORED_TIERS.
"""

from concurrent.futures import ProcessPoolExecutor
import os
import sys
import time

os.environ.setdefault("GEO_BIAS_DATA_DIR", "./data/canonical_v2")
os.environ.setdefault("GEO_BIAS_THIRD_PARTY_DIR", "./data/canonical_v2")
os.environ.setdefault("GEO_BIAS_OUTPUT_DIR", "./data/canonical_v2")

import calendar  # noqa: F401
sys.path = [p for p in sys.path if not p.endswith("/google3")]
sys.path.insert(0, "./.gemini/jetski/brain/9b5bd334-007b-439b-8c84-eb401fd08e38/scratch/tiered_counterfactual")
sys.path.insert(0, "./data/scratch")

import numpy as np
import pandas as pd
import build_tier_features as btf

CANON_V2_DIR = btf.CANON_V2_DIR
BACKUP_38K_DIR = btf.BACKUP_38K_DIR
CONTINENTS = btf.CONTINENTS
FEAT_DIR = "./.gemini/jetski/brain/9b5bd334-007b-439b-8c84-eb401fd08e38/scratch/tiered_counterfactual/features"
MASKS_OUT_PATH = "./data/scratch/tier_active_and_all_masks.parquet"

_G_TOPO = {}
_G_TI = {}
_G_INFER_IDS = {}
_G_OLD_OV = {}
_G_OLD_TT = None
_G_OLD_AREAS = None


def _init_Globals():
  global _G_TOPO, _G_TI, _G_INFER_IDS, _G_OLD_OV, _G_OLD_TT, _G_OLD_AREAS
  topo_df = pd.read_csv(
      f"{CANON_V2_DIR}/attributes/hybas_routing_topology.csv",
      dtype={"HYBAS_ID": str, "NEXT_DOWN": str, "continent": str},
  )
  _G_OLD_TT = pd.read_csv(f"{BACKUP_38K_DIR}/coverage_fractions/caravan_pairwise_overlap_areas.csv")
  old_self = _G_OLD_TT[_G_OLD_TT["basin1"] == _G_OLD_TT["basin2"]]
  _G_OLD_AREAS = old_self.set_index("basin1")["overlap_area"]

  from concurrent.futures import ThreadPoolExecutor

  def _load_cont(cont):
    df_tc = topo_df[topo_df["continent"].str.upper() == cont]
    topo_res = btf.build_continent_topology(df_tc)
    old_ti_df = pd.read_csv(f"{BACKUP_38K_DIR}/coverage_fractions/hybas/infer_pairwise_overlaps_{cont}.csv")
    old_feat_df = pd.read_csv(f"{BACKUP_38K_DIR}/coverage_fractions/hybas/infer_ml_features_{cont}.csv", index_col=0)
    inf_ids = list(old_feat_df["hybas_id"].values) if "hybas_id" in old_feat_df.columns else list(old_feat_df.index)
    ov_by_h = btf.defaultdict(list)
    for b1, b2, ov_a in zip(old_ti_df["basin1"], old_ti_df["basin2"], old_ti_df["overlap_area"]):
      if ov_a > 0:
        ov_by_h[b1].append((b2, float(ov_a)))
    return cont, topo_res, old_ti_df, inf_ids, ov_by_h

  with ThreadPoolExecutor(max_workers=9) as tpool:
    for cont, topo_res, old_ti_df, inf_ids, ov_by_h in tpool.map(_load_cont, CONTINENTS):
      _G_TOPO[cont] = topo_res
      _G_TI[cont] = old_ti_df
      _G_INFER_IDS[cont] = inf_ids
      _G_OLD_OV[cont] = ov_by_h


def _worker_cont_up_cov(args):
  t_key, cont, valid_38k_ids, sub_new, ge95_all_set = args
  hybas_ids, downstream_map, hybas_sub_areas, hybas_total_areas = _G_TOPO[cont]
  hybas_set = set(hybas_ids)
  infer_ids = [
      x for x in _G_INFER_IDS[cont]
      if (x if str(x).startswith("hybas_") else f"hybas_{x}") in ge95_all_set
      or str(x) in ge95_all_set
  ]
  if not infer_ids:
    return t_key, cont, pd.Series(dtype=float)

  old_ti_df = _G_TI[cont]
  old_ti_sub = old_ti_df[old_ti_df["basin2"].isin(valid_38k_ids)].copy()
  tt_sub_38k = _G_OLD_TT[
      _G_OLD_TT["basin1"].isin(valid_38k_ids) & _G_OLD_TT["basin2"].isin(valid_38k_ids)
  ].copy()
  old_overlap_by_hybas = _G_OLD_OV[cont]

  sub_new_c = sub_new[sub_new["hybas_str"].isin(hybas_set)]
  main_gauges = sub_new_c[~sub_new_c["is_subbasin_headwater_gauge"]]
  hw_gauges = sub_new_c[sub_new_c["is_subbasin_headwater_gauge"]]

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
      if g_caravan in valid_38k_ids:
        new_rows_tt.append({"basin1": g_w, "basin2": g_caravan, "overlap_area": ov_area})
        new_rows_tt.append({"basin1": g_caravan, "basin2": g_w, "overlap_area": ov_area})

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
      if g_caravan in valid_38k_ids and ov_area >= 0.95 * tot_a_m2:
        new_rows_tt.append({"basin1": g_hw, "basin2": g_caravan, "overlap_area": hw_area_m2})
        new_rows_tt.append({"basin1": g_caravan, "basin2": g_hw, "overlap_area": hw_area_m2})

  df_new_ti = pd.DataFrame(new_rows_ti) if new_rows_ti else pd.DataFrame(columns=["basin1", "basin2", "overlap_area"])
  df_new_tt = pd.DataFrame(new_rows_tt) if new_rows_tt else pd.DataFrame(columns=["basin1", "basin2", "overlap_area"])

  ti_df = pd.concat([old_ti_sub, df_new_ti], ignore_index=True) if not df_new_ti.empty else old_ti_sub
  tt_df_cont = pd.concat([tt_sub_38k, df_new_tt], ignore_index=True) if not df_new_tt.empty else tt_sub_38k

  combined_areas = pd.concat([_G_OLD_AREAS, pd.Series(new_gauge_areas, dtype=float), hybas_total_areas])
  combined_areas = combined_areas[~combined_areas.index.duplicated(keep="first")]

  infer_up_df = btf.overlap_mod.calculate_total_upstream_coverage(
      infer_ids=infer_ids, tt_df=tt_df_cont, ti_df=ti_df, basin_areas=combined_areas
  )
  w_main_valid = [w for w in w_main_list if w in infer_up_df.index]
  if w_main_valid:
    infer_up_df.loc[w_main_valid, "total_upstream_coverage"] = 1.0
  infer_up_df.index = [x if str(x).startswith("hybas_") else f"hybas_{x}" for x in infer_up_df.index]
  return t_key, cont, infer_up_df["total_upstream_coverage"]


def main():
  t0 = time.time()
  print("Initializing continent topologies & pairwise overlaps in parent process...", flush=True)
  _init_Globals()
  print(f"Initialized in {time.time() - t0:.1f}s.", flush=True)

  df_basin_v2 = pd.read_parquet(f"{CANON_V2_DIR}/v2_exposure_blockers/statistics/basin_exposure_and_blockers_v2.parquet")
  study_ids = [x if str(x).startswith("hybas_") else f"hybas_{x}" for x in df_basin_v2.index]
  df_basin_v2.index = study_ids

  dedup = pd.read_parquet(f"{CANON_V2_DIR}/attributes/global_gauges_deduplicated.parquet")
  arch = pd.read_parquet(f"{CANON_V2_DIR}/attributes/global_gauges_by_source_archive.parquet")
  grdc_8811 = pd.read_csv(f"{CANON_V2_DIR}/attributes/grdc_attributes.csv", index_col=0)
  ca_38k = pd.read_csv(f"{BACKUP_38K_DIR}/attributes/caravan_attributes.csv", index_col=0)

  native_to_cid = dict(zip(arch["native_station_id"].astype(str), arch["canonical_gauge_id"].astype(str)))
  grdc_arch = arch[arch["source_dataset"] == "GRDC"]
  for nid, cid in zip(grdc_arch["native_station_id"].astype(str), grdc_arch["canonical_gauge_id"].astype(str)):
    native_to_cid[f"GRDC_{nid}"] = cid

  t1a_ids = ["CARAVAN_" + str(x).upper() for x in grdc_8811.index]
  t1a_cids = set(native_to_cid[x] for x in t1a_ids if x in native_to_cid)
  t1b_ccby_ids = [x for x in ca_38k.index if str(x).startswith("CARAVAN_") and not str(x).startswith("CARAVAN_GRDC_")]
  t1b_ccby_cids = set(native_to_cid[x] for x in t1b_ccby_ids if x in native_to_cid)
  t1b_full_ids = [x for x in ca_38k.index if str(x).startswith("CARAVAN_")]
  t1b_full_cids = set(native_to_cid[x] for x in t1b_full_ids if x in native_to_cid)
  t2a_ids = [x for x in ca_38k.index if not str(x).startswith("GRDC_")]
  t2a_cids = set(native_to_cid[x] for x in t2a_ids if x in native_to_cid)

  has_caravan = dedup["all_source_datasets"].str.contains("Caravan", na=False)
  has_estreams = dedup["all_source_datasets"].str.contains("EStreams", na=False)
  dedup_34577 = dedup[has_caravan | has_estreams]
  dedup_93827 = dedup[
      dedup["best_data_access_policy"].isin(["open_public_api_or_bulk", "open_academic_compilation"])
      & dedup["best_observation_type"].isin(["daily_discharge", "monthly_discharge"])
  ]

  tier_dedup_map = {
      "tier0_zero_gauges": dedup.iloc[0:0],
      "tier1a_active_post2016_ccby_8811": dedup[dedup["canonical_gauge_id"].isin(t1a_cids)],
      "tier1b_strict_ccby_19491": dedup[dedup["canonical_gauge_id"].isin(t1b_ccby_cids)],
      "tier1b_full_caravan_24847": dedup[dedup["canonical_gauge_id"].isin(t1b_full_cids)],
      "tier2a_caravan_estreams_32713": dedup[dedup["canonical_gauge_id"].isin(t2a_cids)],
      "tier2_academic_compilations_38069": dedup[dedup["in_canonical_38069"]],
      "tier2_all_open_academic_34630": dedup[dedup["in_canonical_38069"] | (dedup["best_data_access_policy"] == "open_academic_compilation")],
      "tier3_open_public_apis_and_bulk_106410": dedup[dedup["best_data_access_policy"].isin(["open_public_api_or_bulk", "open_academic_compilation"])],
      "tier4_all_publicly_knowable_121819": dedup,
      "tier_caravan_estreams_34577": dedup_34577,
      "tier_public_discharge_93827": dedup_93827,
  }

  work_items = []
  for t_key, d_sub in tier_dedup_map.items():
    if len(d_sub) > 0 and not d_sub["is_active_opt1"].all():
      tf_all = pd.read_parquet(f"{FEAT_DIR}/{t_key}.parquet").loc[study_ids]
      ge95_all_ids = set(tf_all.index[tf_all["total_upstream_coverage"].values >= 0.95])
      ge95_all_ids |= {x.replace("hybas_", "") for x in ge95_all_ids}
      d_act = d_sub[d_sub["is_active_opt1"]].copy()
      cids_in_sub = set(d_act["canonical_gauge_id"].values)
      valid_38k_ids = set(x for x in ca_38k.index if native_to_cid.get(str(x)) in cids_in_sub)
      net_new_df = d_act[(~d_act["in_canonical_38069"]) & (d_act["snapped_hybas_id"] > 0)][
          ["snapped_hybas_id", "is_subbasin_headwater_gauge", "effective_gauged_area_km2"]
      ].copy()
      net_new_df["hybas_str"] = "hybas_" + net_new_df["snapped_hybas_id"].astype(int).astype(str)
      for cont in CONTINENTS:
        work_items.append((t_key, cont, valid_38k_ids, net_new_df, ge95_all_ids))

  print(f"Running {len(work_items)} (tier, continent) active-coverage tasks across 36 forked workers...", flush=True)
  import multiprocessing as mp
  ctx = mp.get_context("fork")
  t_by_tier = btf.defaultdict(list)
  with ProcessPoolExecutor(max_workers=36, mp_context=ctx) as pool:
    for tk, cont, s_up in pool.map(_worker_cont_up_cov, work_items):
      t_by_tier[tk].append(s_up)

  mask_cols = {}
  print(f"\n{'Tier Key':42s} | {'All Gauges':>10s} | {'Act Gauges':>10s} | {'Union All':>10s} | {'Union Act':>10s} | {'NonAct Only':>11s}")
  print("-" * 106)

  for t_key, d_sub in tier_dedup_map.items():
    tf_all = pd.read_parquet(f"{FEAT_DIR}/{t_key}.parquet").loc[study_ids]
    m_ge95_all = tf_all["total_upstream_coverage"].values >= 0.95
    snapped_all = set("hybas_" + d_sub.loc[d_sub["snapped_hybas_id"] > 0, "snapped_hybas_id"].astype(int).astype(str))
    m_snap_all = np.array([h in snapped_all for h in study_ids], dtype=bool)
    m_union_all = m_snap_all | m_ge95_all

    if len(d_sub) == 0:
      m_snap_act = np.zeros(len(study_ids), dtype=bool)
      m_ge95_act = np.zeros(len(study_ids), dtype=bool)
      m_union_act = np.zeros(len(study_ids), dtype=bool)
      n_act = 0
    elif d_sub["is_active_opt1"].all():
      m_snap_act = m_snap_all.copy()
      m_ge95_act = m_ge95_all.copy()
      m_union_act = m_union_all.copy()
      n_act = len(d_sub)
    else:
      d_act = d_sub[d_sub["is_active_opt1"]]
      n_act = len(d_act)
      snapped_act = set("hybas_" + d_act.loc[d_act["snapped_hybas_id"] > 0, "snapped_hybas_id"].astype(int).astype(str))
      m_snap_act = np.array([h in snapped_act for h in study_ids], dtype=bool)
      s_all_cont = pd.concat(t_by_tier[t_key], axis=0)
      s_all_cont = s_all_cont[~s_all_cont.index.duplicated(keep="first")]
      up_cov_act = s_all_cont.reindex(study_ids).fillna(0.0).values
      m_ge95_act = up_cov_act >= 0.95
      m_union_act = m_snap_act | m_ge95_act

    mask_cols[f"{t_key}__snap_all"] = m_snap_all
    mask_cols[f"{t_key}__ge95_all"] = m_ge95_all
    mask_cols[f"{t_key}__union_all"] = m_union_all
    mask_cols[f"{t_key}__snap_active"] = m_snap_act
    mask_cols[f"{t_key}__ge95_active"] = m_ge95_act
    mask_cols[f"{t_key}__union_active"] = m_union_act

    print(
        f"{t_key:42s} | {len(d_sub):10,d} | {n_act:10,d} | "
        f"{int(m_union_all.sum()):10,d} | {int(m_union_act.sum()):10,d} | {int((m_union_all & ~m_union_act).sum()):11,d}",
        flush=True,
    )

  for t_ceil in ["tier5_universal_idealized_ceiling", "tier5_dense_corridor_ceiling"]:
    ones = np.ones(len(study_ids), dtype=bool)
    for sfx in ["snap_all", "ge95_all", "union_all", "snap_active", "ge95_active", "union_active"]:
      mask_cols[f"{t_ceil}__{sfx}"] = ones

  df_masks = pd.DataFrame(mask_cols, index=study_ids)
  df_masks.to_parquet(MASKS_OUT_PATH)
  print(f"\nSaved all active and all-gauge masks to {MASKS_OUT_PATH} in {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
  main()
