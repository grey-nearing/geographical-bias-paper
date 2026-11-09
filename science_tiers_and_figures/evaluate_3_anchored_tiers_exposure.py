#!/usr/bin/env python3
"""Evaluates Lead-Time-1 flood exposure and skill across the 3 anchored existing-gauge tiers:
  - Baseline: tier1a_active_post2016_ccby_8811 (8,811 post-2016 Caravan gauges)
  - Tier 1: tier_caravan_estreams_34577 (34,577 deduplicated physical Caravan + EStreams gauges)
  - Tier 2: tier_public_discharge_93827 (93,827 deduplicated physical gauges with confirmed public discharge records)
  - Tier 3: tier4_all_publicly_knowable_121819 (121,819 deduplicated physical publicly knowable gauges)
  - Universal Gauging: tier5_universal_idealized_ceiling (964,084 global watersheds)
"""

import json
import os
import sys
import time

os.environ.setdefault("GEO_BIAS_DATA_DIR", "./data")
os.environ.setdefault("GEO_BIAS_THIRD_PARTY_DIR", "./data")
os.environ.setdefault("GEO_BIAS_OUTPUT_DIR", "./data")

sys.path = [p for p in sys.path if not p.endswith("/local_pkg")]
sys.path.insert(0, "./data/science_results")

import numpy as np
import pandas as pd
import evaluate_multi_mechanism_tiered_exposure as emm

CANON_V2_DIR = emm.CANON_V2_DIR
FEAT_DIR = emm.FEAT_DIR
OUT_DIR = emm.OUT_DIR
PRED_DIR = emm.PRED_DIR
COV_COLS = emm.COV_COLS
OOD_COLS = emm.OOD_COLS

ANCHORED_TIERS = [
    ("tier1a_active_post2016_ccby_8811", "Post-2016 Caravan training (8,811)"),
    ("tier_caravan_estreams_34577", "Tier 1: Caravan + EStreams (34,577)"),
    ("tier_public_discharge_93827", "Tier 2: Public discharge (93,827)"),
    ("tier4_all_publicly_knowable_121819", "Tier 3: All knowable gauges (121,819)"),
    ("tier5_universal_idealized_ceiling", "Universal gauging (964,084 watersheds)"),
]


def main():
    t0 = time.time()
    emm.file_paths.init("5.0-year-2.0-window-recall")

    df_basin_v2 = pd.read_parquet(
        f"{CANON_V2_DIR}/v2_exposure_blockers/statistics/basin_exposure_and_blockers_v2.parquet"
    )
    study_ids = [x if str(x).startswith("hybas_") else f"hybas_{x}" for x in df_basin_v2.index]
    df_basin_v2.index = study_ids
    inc_area_km2 = df_basin_v2["sub_area_km2"].values if "sub_area_km2" in df_basin_v2.columns else None

    # Compute exact SD of U_net_tree_eq4 for the 3 anchored tiers for Table S4
    print("0. Computing exact Table S4 Part C summary statistics across the 3 anchored tiers...", flush=True)
    for tk, tlbl in [
        ("tier4_all_publicly_knowable_121819", "Tier 3: All publicly knowable gauges (121,819)"),
        ("tier_public_discharge_93827", "Tier 2: Confirmed publicly accessible discharge records (93,827)"),
        ("tier_caravan_estreams_34577", "Tier 1: Standardized open compilations — Caravan + EStreams (34,577)"),
        ("tier1a_active_post2016_ccby_8811", "Post-2016 Caravan global training subset (8,811)"),
    ]:
        pq = pd.read_parquet(f"{FEAT_DIR}/{tk}.parquet").loc[study_ids]
        u_vals = pq["U_net_tree"].values * 100.0
        q25, q50, q75 = np.percentile(u_vals, [25, 50, 75])
        print(
            f"   {tlbl}: mean={u_vals.mean():.1f}% ± {u_vals.std():.1f}% (median {q50:.1f}%; IQR [{q25:.1f}%, {q75:.1f}%])",
            flush=True,
        )

    df_exp_raw = pd.read_csv(emm.file_paths.GLOBAL_BASIN_EXPOSURE_DEFENDED_BY_RP_PATH)
    id_col = "hybas_id" if "hybas_id" in df_exp_raw.columns else df_exp_raw.columns[0]
    df_exp_raw[id_col] = (
        df_exp_raw[id_col]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .apply(lambda x: x if x.startswith("hybas_") else f"hybas_{x}")
    )
    df_exp_raw = df_exp_raw.drop_duplicates(subset=[id_col]).set_index(id_col)

    def _get_exp_col(df, t):
        for c in [f"exposed_pop_{t}", f"exposure_{t}yr", str(t)]:
            if c in df.columns:
                return df[c].fillna(0.0)
        raise KeyError(f"Could not find exposure column for T={t}")

    e5 = (emm.RP_WEIGHTS[5] * _get_exp_col(df_exp_raw, 5).reindex(study_ids).fillna(0.0)).values
    e10 = (emm.RP_WEIGHTS[10] * _get_exp_col(df_exp_raw, 10).reindex(study_ids).fillna(0.0)).values
    e20 = (
        emm.RP_WEIGHTS[20] * _get_exp_col(df_exp_raw, 20).reindex(study_ids).fillna(0.0)
        + emm.RP_WEIGHTS[50] * _get_exp_col(df_exp_raw, 50).reindex(study_ids).fillna(0.0)
        + emm.RP_WEIGHTS[100] * _get_exp_col(df_exp_raw, 100).reindex(study_ids).fillna(0.0)
    ).values
    etot = e5 + e10 + e20

    has_sys = df_basin_v2["has_system"].values.astype(bool)

    dedup = pd.read_parquet(f"{CANON_V2_DIR}/attributes/global_gauges_deduplicated.parquet")
    arch = pd.read_parquet(f"{CANON_V2_DIR}/attributes/global_gauges_by_source_archive.parquet")
    grdc_8811 = pd.read_csv(f"{CANON_V2_DIR}/attributes/grdc_attributes.csv", index_col=0)
    native_to_cid = dict(zip(arch["native_station_id"].astype(str), arch["canonical_gauge_id"].astype(str)))
    t1a_ids = ["CARAVAN_" + str(x).upper() for x in grdc_8811.index]
    t1a_cids = set(native_to_cid[x] for x in t1a_ids if x in native_to_cid)

    has_caravan = dedup["all_source_datasets"].str.contains("Caravan", na=False)
    has_estreams = dedup["all_source_datasets"].str.contains("EStreams", na=False)
    dedup_34577 = dedup[has_caravan | has_estreams]
    dedup_93827 = dedup[
        dedup["best_data_access_policy"].isin(["open_public_api_or_bulk", "open_academic_compilation"])
        & dedup["best_observation_type"].isin(["daily_discharge", "monthly_discharge"])
    ]

    tier_dedup_map = {
        "tier1a_active_post2016_ccby_8811": dedup[dedup["canonical_gauge_id"].isin(t1a_cids)],
        "tier_caravan_estreams_34577": dedup_34577,
        "tier_public_discharge_93827": dedup_93827,
        "tier4_all_publicly_knowable_121819": dedup,
    }

    df_base = emm.load_zarr_df(f"{CANON_V2_DIR}/generic_training_data/hybas_generic_inference_data.zarr").loc[study_ids].copy()
    x_train_u, y_train_u_5 = emm.io_utils.load_and_preprocess_data(
        "ungauged", "5.0-year-2.0-window-recall", None, exp_suffix="", lead_time="1"
    )
    top_q_thresh_u = y_train_u_5.quantile(0.75)
    ref_medians_u = x_train_u.loc[y_train_u_5 >= top_q_thresh_u, OOD_COLS].median()

    tier_feats = {}
    tier_masks_union = {}
    tier_masks_ge95 = {}

    for t_key, _ in ANCHORED_TIERS:
        tf = pd.read_parquet(f"{FEAT_DIR}/{t_key}.parquet").loc[study_ids].copy()
        if t_key == "tier5_universal_idealized_ceiling":
            ood_t4_base = pd.read_parquet(f"{FEAT_DIR}/tier4_all_publicly_knowable_121819.parquet").loc[study_ids, OOD_COLS]
            for c in OOD_COLS:
                tf[c] = (
                    np.maximum(ood_t4_base[c].values, float(ref_medians_u[c]))
                    if c != "anomaly_score_lof"
                    else float(ref_medians_u[c])
                )
            tier_masks_ge95[t_key] = np.ones(len(study_ids), dtype=bool)
            tier_masks_union[t_key] = np.ones(len(study_ids), dtype=bool)
        else:
            d_sub = tier_dedup_map[t_key]
            snapped_set = set("hybas_" + d_sub.loc[d_sub["snapped_hybas_id"] > 0, "snapped_hybas_id"].astype(int).astype(str))
            m_snap = np.array([h in snapped_set for h in study_ids], dtype=bool)
            m_ge95 = tf["total_upstream_coverage"].values >= 0.95
            tier_masks_ge95[t_key] = m_ge95
            tier_masks_union[t_key] = m_snap | m_ge95
        tier_feats[t_key] = tf

    families = {}
    for fam_key, sfx in [("base", ""), ("da", "_da"), ("ft", "_ft"), ("ft_da", "_ft_da")]:
        loaded = emm.load_model_family(sfx)
        if loaded is not None:
            families[fam_key] = loaded
            print(f"Loaded family '{fam_key}'", flush=True)

    models_u_base, _ = families["base"]
    rp_cols_u = list(models_u_base[emm.RP_METRICS[0]].impl.feature_columns)
    X_rp_base = models_u_base[emm.RP_METRICS[0]].impl._preprocess_features(df_base).reindex(columns=rp_cols_u, fill_value=0)

    def eval_or_load(fam_key: str, t_key: str):
        cache_pq = f"{PRED_DIR}/{fam_key}__{t_key}__joint.parquet"
        if os.path.exists(cache_pq):
            df_c = pd.read_parquet(cache_pq).loc[study_ids]
            return {c: df_c[c].values for c in df_c.columns}
        t_s = time.time()
        models_u, _ = families[fam_key]
        cov_df = tier_feats[t_key][COV_COLS]
        ood_df = tier_feats[t_key][OOD_COLS]
        X_rp = X_rp_base.copy()
        for c in COV_COLS:
            X_rp[c] = cov_df[c].values
        for c in OOD_COLS:
            X_rp[c] = ood_df[c].values
        full_rp = emm.fast_predict_rp_triplet(models_u, X_rp)
        res = {
            "r5_u": full_rp["5.0-year-2.0-window-recall"],
            "r10_u": full_rp["10.0-year-2.0-window-recall"],
            "r20_u": full_rp["20.0-year-2.0-window-recall"],
        }
        pd.DataFrame(res, index=study_ids).to_parquet(cache_pq)
        print(f"   Evaluated & cached {fam_key}__{t_key}__joint in {time.time() - t_s:.1f}s", flush=True)
        return res

    preds = {}
    for fam_key in families:
        for t_key, _ in ANCHORED_TIERS:
            preds[(fam_key, t_key)] = eval_or_load(fam_key, t_key)

    def calc_w(p):
        return e5 * p["r5_u"] + e10 * p["r10_u"] + e20 * p["r20_u"]

    def calc_clip_delta(p_new, p_ref):
        return (
            e5 * np.maximum(p_new["r5_u"] - p_ref["r5_u"], 0.0)
            + e10 * np.maximum(p_new["r10_u"] - p_ref["r10_u"], 0.0)
            + e20 * np.maximum(p_new["r20_u"] - p_ref["r20_u"], 0.0)
        )

    def blend(p_glob, p_loc, mask):
        return {k: np.where(mask, p_loc[k], p_glob[k]) for k in ["r5_u", "r10_u", "r20_u"]}

    p_b1a = preds[("base", "tier1a_active_post2016_ccby_8811")]
    p_da1a = preds[("da", "tier1a_active_post2016_ccby_8811")]
    w_b1a = calc_w(p_b1a)
    u_b1a = etot - w_b1a

    rows = []
    for t_key, t_lbl in ANCHORED_TIERS:
        p_bk = preds[("base", t_key)]
        p_dak = preds[("da", t_key)]
        for mask_name, mask_arr in [("ge95_outlet", tier_masks_ge95[t_key]), ("union_snapped_or_ge95", tier_masks_union[t_key])]:
            p_000 = p_bk
            p_001 = blend(p_b1a, p_da1a, mask_arr)
            p_101 = blend(p_bk, p_dak, mask_arr)
            for mech_key, p_m in [("000_global_train", p_000), ("001_da_only", p_001), ("101_combined", p_101)]:
                w_m = calc_w(p_m)
                d_clip = calc_clip_delta(p_m, p_b1a)
                d_net = w_m - w_b1a
                rows.append({
                    "tier_key": t_key,
                    "tier_label": t_lbl,
                    "mask_type": mask_name,
                    "mechanism": mech_key,
                    "warned_global_M": float(w_m.sum()) / 1e6,
                    "warned_sys60_M": float(w_m[has_sys].sum()) / 1e6,
                    "warned_nosys113_M": float(w_m[~has_sys].sum()) / 1e6,
                    "unwarned_global_M": float((etot - w_m).sum()) / 1e6,
                    "delta_clip_global_M": float(d_clip.sum()) / 1e6,
                    "delta_clip_pct_unw": float(100.0 * d_clip.sum() / u_b1a.sum()),
                    "delta_clip_sys60_M": float(d_clip[has_sys].sum()) / 1e6,
                    "delta_clip_nosys113_M": float(d_clip[~has_sys].sum()) / 1e6,
                    "delta_net_global_M": float(d_net.sum()) / 1e6,
                    "delta_net_pct_unw": float(100.0 * d_net.sum() / u_b1a.sum()),
                    "delta_net_sys60_M": float(d_net[has_sys].sum()) / 1e6,
                    "delta_net_nosys113_M": float(d_net[~has_sys].sum()) / 1e6,
                    "r5_median_pct": float(np.median(p_m["r5_u"]) * 100.0),
                })

    df_out = pd.DataFrame(rows)
    out_csv = f"{OUT_DIR}/anchored_3tiers_exposure_summary_lead1.csv"
    df_out.to_csv(out_csv, index=False)
    print("\n=== ANCHORED 3-TIER EXPOSURE SUMMARY (union_snapped_or_ge95) ===")
    print(
        df_out[df_out["mask_type"] == "union_snapped_or_ge95"][
            [
                "tier_label",
                "mechanism",
                "warned_global_M",
                "delta_clip_global_M",
                "delta_clip_pct_unw",
                "delta_clip_sys60_M",
                "delta_clip_nosys113_M",
            ]
        ].to_string(index=False)
    )
    print("\n=== ANCHORED 3-TIER EXPOSURE SUMMARY (ge95_outlet) ===")
    print(
        df_out[df_out["mask_type"] == "ge95_outlet"][
            [
                "tier_label",
                "mechanism",
                "warned_global_M",
                "delta_clip_global_M",
                "delta_clip_pct_unw",
                "delta_clip_sys60_M",
                "delta_clip_nosys113_M",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
