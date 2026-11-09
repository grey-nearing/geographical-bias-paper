#!/usr/bin/env python3
"""Computes exact Section 2 ungaugedness metrics and Section 3 Lead-Time-1 flood exposure
metrics for the 3 anchored existing-gauge tiers defined in Section 2 Paragraph 7:
  - Tier 1 (Standardized open compilations: Caravan + EStreams, N = 34,577 physical gauges)
  - Tier 2 (Confirmed publicly accessible daily/monthly discharge records, N = 93,827 physical gauges)
  - Tier 3 (All publicly knowable gauges, N = 121,819 physical gauges)
plus the baseline training set (8,811 post-2016 Caravan gauges) and universal gauging (964,084 watersheds).
"""

import os
import sys
import time
import json

os.environ.setdefault("GEO_BIAS_DATA_DIR", "./data")
os.environ.setdefault("GEO_BIAS_THIRD_PARTY_DIR", "./data")
os.environ.setdefault("GEO_BIAS_OUTPUT_DIR", "./data")

sys.path = [p for p in sys.path if not p.endswith("/local_pkg")]
sys.path.insert(0, "./data/science_results")
sys.path.insert(0, "./data/science_results")

import numpy as np
import pandas as pd
import build_tier_features as btf
import evaluate_multi_mechanism_tiered_exposure as emm

CANON_V2_DIR = btf.CANON_V2_DIR
BACKUP_38K_DIR = btf.BACKUP_38K_DIR
CONTINENTS = btf.CONTINENTS
COV_COLS = btf.COV_COLS
OOD_COLS = btf.OOD_COLS
FEAT_DIR = "./data/coverage_fractions"

def compute_exact_coverage_for_dedup_subset(sub_dedup: pd.DataFrame, arch: pd.DataFrame, ca_38k: pd.DataFrame, old_caravan_tt: pd.DataFrame, old_caravan_areas: pd.Series, cont_total_areas: dict, cont_topo_dfs: dict) -> pd.DataFrame:
    """Computes exact coverage features for any subset of deduplicated stations `sub_dedup`."""
    # 1. Identify which canonical_gauge_ids in sub_dedup are in the 38,069 set, and map them to their 38,069 native IDs
    cids_in_sub = set(sub_dedup["canonical_gauge_id"].values)
    native_to_cid = dict(zip(arch["native_station_id"].astype(str), arch["canonical_gauge_id"].astype(str)))
    grdc_arch = arch[arch["source_dataset"] == "GRDC"]
    for nid, cid in zip(grdc_arch["native_station_id"].astype(str), grdc_arch["canonical_gauge_id"].astype(str)):
        native_to_cid[f"GRDC_{nid}"] = cid

    valid_38k_ids = set(x for x in ca_38k.index if native_to_cid.get(str(x)) in cids_in_sub)
    net_new_df = sub_dedup[(~sub_dedup["in_canonical_38069"]) & (sub_dedup["snapped_hybas_id"] > 0)].copy()
    net_new_df["hybas_str"] = "hybas_" + net_new_df["snapped_hybas_id"].astype(str)

    # Run per continent combining valid_38k_ids and net_new_df
    dfs = []
    for cont in CONTINENTS:
        df_topo_c = cont_topo_dfs[cont]
        hybas_ids, downstream_map, hybas_sub_areas, hybas_total_areas = btf.build_continent_topology(df_topo_c)
        hybas_set = set(hybas_ids)

        old_ti_path = f"{BACKUP_38K_DIR}/coverage_fractions/hybas/infer_pairwise_overlaps_{cont}.csv"
        old_ti_df = pd.read_csv(old_ti_path)
        old_ti_df = old_ti_df[old_ti_df["basin2"].isin(valid_38k_ids)].copy()

        old_feat_path = f"{BACKUP_38K_DIR}/coverage_fractions/hybas/infer_ml_features_{cont}.csv"
        old_feat_df = pd.read_csv(old_feat_path, index_col=0)
        infer_ids = list(old_feat_df["hybas_id"].values) if "hybas_id" in old_feat_df.columns else list(old_feat_df.index)

        tt_sub_38k = old_caravan_tt[
            old_caravan_tt["basin1"].isin(valid_38k_ids) & old_caravan_tt["basin2"].isin(valid_38k_ids)
        ].copy()

        old_overlap_by_hybas = btf.defaultdict(list)
        for b1, b2, ov_a in zip(old_ti_df["basin1"], old_ti_df["basin2"], old_ti_df["overlap_area"]):
            if ov_a > 0:
                old_overlap_by_hybas[b1].append((b2, float(ov_a)))

        sub_new = net_new_df[net_new_df["hybas_str"].isin(hybas_set)]
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
                    new_rows_ti.append({"basin1": u, "basin2": f"NEW_MAIN_{d}", "overlap_area": u_area})
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
        tt_df_cont = pd.concat([tt_sub_38k, df_new_tt], ignore_index=True) if not df_new_tt.empty else tt_sub_38k

        combined_areas = pd.concat([old_caravan_areas, pd.Series(new_gauge_areas, dtype=float), hybas_total_areas])
        combined_areas = combined_areas[~combined_areas.index.duplicated(keep="first")]

        infer_up_df = btf.overlap_mod.calculate_total_upstream_coverage(
            infer_ids=infer_ids, tt_df=tt_df_cont, ti_df=ti_df, basin_areas=combined_areas
        )
        infer_down_series = btf.overlap_mod.calculate_max_downstream_coverage(
            infer_ids=infer_ids, ti_df=ti_df, basin_areas=combined_areas
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
        infer_df.index = [x if str(x).startswith("hybas_") else f"hybas_{x}" for x in infer_df.index]
        dfs.append(infer_df)

    df_all = pd.concat(dfs, axis=0)
    return df_all[~df_all.index.duplicated(keep="first")]


def main():
    t0 = time.time()
    df_basin_v2 = pd.read_parquet(f"{CANON_V2_DIR}/v2_exposure_blockers/statistics/basin_exposure_and_blockers_v2.parquet")
    study_ids = [x if str(x).startswith("hybas_") else f"hybas_{x}" for x in df_basin_v2.index]
    df_basin_v2.index = study_ids
    e_total_study = df_basin_v2["E_total"].values.astype(np.float64)

    topo_df = pd.read_csv(f"{CANON_V2_DIR}/attributes/hybas_routing_topology.csv", dtype={"HYBAS_ID": str, "NEXT_DOWN": str, "continent": str})
    tree_info = btf.build_tree_ungaugedness_arrays(topo_df)
    study_idx = np.array([tree_info["str_to_idx"][h] for h in study_ids], dtype=np.int32)

    dedup = pd.read_parquet(f"{CANON_V2_DIR}/attributes/global_gauges_deduplicated.parquet")
    arch = pd.read_parquet(f"{CANON_V2_DIR}/attributes/global_gauges_by_source_archive.parquet")
    ca_38k = pd.read_csv(f"{BACKUP_38K_DIR}/attributes/caravan_attributes.csv", index_col=0)
    ca_121k = pd.read_csv(f"{CANON_V2_DIR}/attributes/caravan_attributes.csv", index_col=0)
    ca_38k_clean = ca_38k.copy()
    ca_38k_clean.columns = [c.replace("HYDRO_ATLAS:", "") for c in ca_38k_clean.columns]
    ca_121k_clean = ca_121k.copy()
    ca_121k_clean.columns = [c.replace("HYDRO_ATLAS:", "") for c in ca_121k_clean.columns]

    df_zarr_121k = btf.load_zarr_df(f"{CANON_V2_DIR}/generic_training_data/hybas_generic_inference_data.zarr")
    shared_88_cols = sorted(list(set(ca_38k_clean.columns).intersection(set(df_zarr_121k.columns))))
    X_test_hybas_88 = df_zarr_121k.loc[study_ids, shared_88_cols].copy()

    old_caravan_tt = pd.read_csv(f"{BACKUP_38K_DIR}/coverage_fractions/caravan_pairwise_overlap_areas.csv")
    old_caravan_self = old_caravan_tt[old_caravan_tt["basin1"] == old_caravan_tt["basin2"]]
    old_caravan_areas = old_caravan_self.set_index("basin1")["overlap_area"]

    cont_total_areas = {}
    cont_topo_dfs = {}
    for cont in CONTINENTS:
        df_tc = topo_df[topo_df["continent"].str.upper() == cont]
        cont_topo_dfs[cont] = df_tc
        _, _, _, h_tot = btf.build_continent_topology(df_tc)
        cont_total_areas[cont] = h_tot

    # Define the exact 2 new subsets to check alongside cached Tier 4 (121,819), Tier 3 (106,410), Tier 2+ (40,053 / 34,630)
    has_caravan = dedup["all_source_datasets"].str.contains("Caravan", na=False)
    has_estreams = dedup["all_source_datasets"].str.contains("EStreams", na=False)
    dedup_34577 = dedup[has_caravan | has_estreams].copy()

    is_93827 = (
        dedup["best_data_access_policy"].isin(["open_public_api_or_bulk", "open_academic_compilation"]) &
        dedup["best_observation_type"].isin(["daily_discharge", "monthly_discharge"])
    )
    dedup_93827 = dedup[is_93827].copy()

    for key, label, sub_d, n_rec in [
        ("tier_caravan_estreams_34577", "Tier 1: Standardized Open Compilations — Caravan + EStreams (34,577 Physical)", dedup_34577, 40911),
        ("tier_public_discharge_93827", "Tier 2: Confirmed Publicly Accessible Discharge Records (93,827 Physical)", dedup_93827, 93827),
    ]:
        t_s = time.time()
        print(f"\nComputing {key} ({len(sub_d):,} physical stations)...", flush=True)
        cov_df = compute_exact_coverage_for_dedup_subset(
            sub_d, arch, ca_38k, old_caravan_tt, old_caravan_areas, cont_total_areas, cont_topo_dfs
        )
        ood_df = pd.DataFrame(
            btf.fast_calculate_anomaly_scores(
                ca_121k_clean.loc[sub_d["canonical_gauge_id"].values, shared_88_cols],
                X_test_hybas_88,
            ),
            index=study_ids,
        )
        sec2_dict, s_utree, s_uprod = btf.compute_sec2_metrics_for_tier(
            tier_key=key,
            tier_label=label,
            cov_df_all=cov_df,
            tier_dedup_df=sub_d,
            n_archive_records=n_rec,
            tree_info=tree_info,
            study_ids=study_ids,
            study_idx=study_idx,
            e_total_study=e_total_study,
        )
        cov_study = cov_df.reindex(study_ids)[COV_COLS].fillna(0.0).copy()
        for c in ["total_upstream_coverage", "max_upstream_coverage", "max_downstream_coverage"]:
            cov_study[c] = cov_study[c].clip(0.0, 1.0)
        cov_study["upstream_gauge_count"] = cov_study["upstream_gauge_count"].clip(lower=0.0)
        out_df = pd.concat([cov_study, ood_df.reindex(study_ids)[OOD_COLS], s_utree, s_uprod], axis=1)
        out_df.to_parquet(f"{FEAT_DIR}/{key}.parquet")
        print(json.dumps(sec2_dict, indent=2), flush=True)
        print(f"Finished {key} in {time.time() - t_s:.1f}s.", flush=True)

if __name__ == "__main__":
    main()
