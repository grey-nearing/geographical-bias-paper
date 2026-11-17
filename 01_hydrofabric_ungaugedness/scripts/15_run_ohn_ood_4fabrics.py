#!/usr/bin/env python3
"""Compute exact OHN OOD detectors (LOF, Isolation Forest, One-Class SVM) across all 4 hydrofabrics.

Matches the exact 88 shared HydroATLAS feature set and scikit-learn detectors from
`geographical_bias_paper_v2/1_anomaly_scores.py`, evaluated across:
  - 4 hydrofabrics: hydrosheds_l12, merit_basins, tdx_hydro, grit_v06_mainstem_tree
  - 4 gauge tiers: baseline (8,075), tier1_open_compilations (32,369),
                   tier2_public_discharge (74,078), tier3_all_known (119,169)
  - 4 accounting variables: area, runoff (where available), population, exposure_eae
  - Macro domains: common_no_grl_ata, full_native, 6 continents, 5 UNEP aridity zones

Reports both:
  1. Native OHN / sklearn threshold (decision_function < 0)
  2. Uniform 95% gauged-basin envelope threshold (decision_function < q05_train)
  3. Continuous summary stats (weighted mean, median)
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
from configs.benchmark_config import (
    GAUGES_DIR,
    GEO_BIAS_DATA_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
)
from global_ungaugedness.spatial_enrichment import (
    ARIDITY_LABELS,
    transfer_exposure_and_aridity_from_l12,
)

ATTR_DIR = GEO_BIAS_DATA_DIR / "attributes"
OUT_DIR = RESULTS_DIR / "ohn_ood_4fabrics"
FEAT_CACHE = PROCESSED_DIR / "hydrosheds" / "hybas_88_features.parquet"

_NON_HYDROATLAS_COLS = {
    "anomaly_score_if",
    "anomaly_score_ocsvm",
    "anomaly_score_lof",
    "total_upstream_coverage",
    "max_upstream_coverage",
    "max_downstream_coverage",
    "upstream_gauge_count",
    "record_length",
}

TIERS = [
    ("baseline", "in_tier_baseline"),
    ("tier1_open_compilations", "in_tier_1"),
    ("tier2_public_discharge", "in_tier_2"),
    ("tier3_all_known", "in_tier_3"),
]

CONTINENTS = ["Africa", "Asia", "Europe", "North America", "South America", "Oceania"]


def load_or_build_hybas_88() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load exact 88 shared OHN HydroATLAS features for all 1,034,083 HydroSHEDS L12 basins."""
    t0 = time.time()
    if FEAT_CACHE.exists():
        df = pd.read_parquet(FEAT_CACHE)
        hybas_ids = df["HYBAS_ID"].to_numpy(dtype=np.int64)
        feat_cols = [c for c in df.columns if c != "HYBAS_ID"]
        X = df[feat_cols].to_numpy(dtype=np.float32)
        print(f"Loaded cached 88-feature matrix {X.shape} in {time.time() - t0:.1f}s", flush=True)
        return hybas_ids, X, feat_cols

    inf_pq = GEO_BIAS_DATA_DIR / "generic_training_data" / "hybas_generic_inference_data.parquet"
    if inf_pq.exists():
        df_inf = pd.read_parquet(inf_pq)
        feat_cols = sorted([c for c in df_inf.columns if c not in _NON_HYDROATLAS_COLS])
        hybas_ids = df_inf.index.astype(str).str.replace("hybas_", "", regex=False).astype(np.int64).to_numpy()
        out_df = df_inf[feat_cols].astype(np.float32).reset_index(drop=True)
        out_df.insert(0, "HYBAS_ID", hybas_ids)
        FEAT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(FEAT_CACHE, index=False)
        print(f"Built & cached 88-feature matrix from {inf_pq.name} {out_df[feat_cols].shape} in {time.time() - t0:.1f}s", flush=True)
        return hybas_ids, out_df[feat_cols].to_numpy(dtype=np.float32), feat_cols

    g_cols = {c.replace("HYDRO_ATLAS:", "") for c in pd.read_csv(ATTR_DIR / "grdc_attributes.csv", nrows=1).columns}
    c_cols = {c.replace("HYDRO_ATLAS:", "") for c in pd.read_csv(ATTR_DIR / "caravan_attributes.csv", nrows=1).columns}
    h_raw_cols = list(pd.read_csv(ATTR_DIR / "hybas_attributes.csv", nrows=1).columns)
    shared = sorted((g_cols & c_cols & {c.replace("HYDRO_ATLAS:", "") for c in h_raw_cols}) - {"Unnamed: 0"})
    use_raw = [h_raw_cols[0]] + [c for c in h_raw_cols if c.replace("HYDRO_ATLAS:", "") in set(shared)]

    hybas = pd.read_csv(ATTR_DIR / "hybas_attributes.csv", usecols=use_raw, index_col=0)
    hybas.columns = [c.replace("HYDRO_ATLAS:", "") for c in hybas.columns]
    feat_cols = sorted(hybas.columns.tolist())
    hybas = hybas[feat_cols]
    hybas_ids = hybas.index.str.replace("hybas_", "", regex=False).astype(np.int64).to_numpy()

    out_df = hybas.astype(np.float32).reset_index(drop=True)
    out_df.insert(0, "HYBAS_ID", hybas_ids)
    FEAT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(FEAT_CACHE, index=False)
    print(f"Built & cached 88-feature matrix {hybas.shape} in {time.time() - t0:.1f}s", flush=True)
    return hybas_ids, out_df[feat_cols].to_numpy(dtype=np.float32), feat_cols


def _read_parquet_present(path: Path, wanted: list[str]) -> pd.DataFrame:
    avail = set(pq.read_schema(path).names)
    return pd.read_parquet(path, columns=[c for c in wanted if c in avail])


def load_hydrofabric_meta(fabric_id: str, gauges_df: pd.DataFrame) -> dict:
    """Load unit weights, domain masks, hybas_l12 mapping, and snapped tier hybas sets."""
    proc = PROCESSED_DIR
    if fabric_id == "hydrosheds_l12":
        u = _read_parquet_present(
            proc / "hydrosheds" / "hydrosheds_l12_units.parquet",
            [
                "unit_id", "local_area_km2", "local_runoff_m3s", "population",
                "exposure_eae", "in_common_domain", "continent", "aridity_zone",
            ],
        )
        s = pd.read_parquet(proc / "hydrosheds" / "hydrosheds_l12_gauge_snaps.parquet", columns=["snapped_unit_id"])
        hyb_col = u["unit_id"].to_numpy(dtype=np.int64)
        local_q = u["local_runoff_m3s"].to_numpy(dtype=np.float64)
    elif fabric_id == "merit_basins":
        u = _read_parquet_present(
            proc / "merit_basins" / "merit_basins_units_grades.parquet",
            [
                "unit_id", "local_area_km2", "local_runoff_m3s", "population",
                "exposure_eae", "in_common_domain", "continent", "aridity_zone", "hybas_l12_id",
            ],
        )
        s = pd.read_parquet(proc / "merit_basins" / "merit_basins_gauge_snaps.parquet", columns=["snapped_unit_id"])
        hyb_col = u["hybas_l12_id"].to_numpy(dtype=np.int64)
        local_q = u["local_runoff_m3s"].to_numpy(dtype=np.float64)
    elif fabric_id == "tdx_hydro":
        u = _read_parquet_present(
            proc / "tdx_hydro" / "tdx_hydro_units.parquet",
            [
                "unit_id", "local_area_km2", "local_runoff_m3s", "population",
                "exposure_eae", "in_common_domain", "continent", "aridity_zone", "hybas_l12_id",
            ],
        )
        s = pd.read_parquet(proc / "tdx_hydro" / "tdx_hydro_gauge_snaps.parquet", columns=["snapped_unit_id"])
        hyb_col = u["hybas_l12_id"].to_numpy(dtype=np.int64)
        local_q = u["local_runoff_m3s"].to_numpy(dtype=np.float64)
    elif fabric_id == "grit_v06_mainstem_tree":
        all_df = pd.read_parquet(
            proc / "grit_v06" / "grit_v06_all_units.parquet",
            columns=[
                "unit_id", "is_mainstem", "local_area_dag_km2", "lon", "lat",
                "in_common_domain", "continent", "hybas_l12_id",
            ],
        )
        u = all_df.loc[all_df["is_mainstem"].to_numpy()].reset_index(drop=True)
        local_a = u["local_area_dag_km2"].to_numpy(dtype=np.float64)
        l12_ref = proc / "hydrosheds" / "hydrosheds_l12_enrichment_ref.parquet"
        pop_m, eae_m, ari_m = transfer_exposure_and_aridity_from_l12(
            u["hybas_l12_id"].to_numpy(dtype=np.int64),
            local_a,
            l12_aridity_parquet=l12_ref,
            unit_lons=u["lon"].to_numpy(dtype=np.float64),
            unit_lats=u["lat"].to_numpy(dtype=np.float64),
        )
        u["local_area_km2"] = local_a
        u["population"] = pop_m
        if eae_m is not None:
            u["exposure_eae"] = eae_m
        u["aridity_zone"] = ari_m
        s = pd.read_parquet(proc / "grit_v06" / "grit_v06_mainstem_gauge_snaps.parquet", columns=["snapped_unit_id"])
        hyb_col = u["hybas_l12_id"].to_numpy(dtype=np.int64)
        local_q = None
    else:
        raise ValueError(fabric_id)

    uid_arr = u["unit_id"].to_numpy(dtype=np.int64)
    uid_to_hyb = dict(zip(uid_arr.tolist(), hyb_col.tolist()))

    tier_hybas = {}
    snapped_uids = s["snapped_unit_id"].to_numpy(dtype=np.int64)
    for tkey, tcol in TIERS:
        m = gauges_df[tcol].to_numpy(dtype=bool)
        su = snapped_uids[m]
        su = su[su >= 0]
        hybs = sorted({uid_to_hyb[int(x)] for x in su if int(x) in uid_to_hyb})
        tier_hybas[tkey] = np.array(hybs, dtype=np.int64)

    return {
        "fabric_id": fabric_id,
        "hybas_l12_id": hyb_col,
        "area": u["local_area_km2"].to_numpy(dtype=np.float64),
        "runoff": local_q,
        "population": u["population"].fillna(0.0).to_numpy(dtype=np.float64),
        "exposure_eae": u["exposure_eae"].fillna(0.0).to_numpy(dtype=np.float64) if "exposure_eae" in u.columns else None,
        "in_common_domain": u["in_common_domain"].to_numpy(dtype=bool),
        "continent": u["continent"].astype(str).to_numpy(),
        "aridity_zone": u["aridity_zone"].to_numpy(dtype=np.int8),
        "tier_hybas": tier_hybas,
    }


def aggregate_scores_to_records(
    fabric_meta: dict,
    hybas_id_to_row: dict[int, int],
    scores_by_det: dict[str, tuple[np.ndarray, float]],
    tier_key: str,
    n_train_unique: int,
) -> list[dict]:
    """Map per-HYBAS_ID detector scores onto hydrofabric units and aggregate across weights & domains."""
    unit_hyb = fabric_meta["hybas_l12_id"]
    # Vectorized map via pandas index lookup (fast even for 6.8M units)
    row_idx = pd.Index(list(hybas_id_to_row.keys())).get_indexer(unit_hyb)
    valid_feat = row_idx >= 0

    in_common = fabric_meta["in_common_domain"] & valid_feat
    full_valid = valid_feat
    continents = fabric_meta["continent"]
    aridity = fabric_meta["aridity_zone"]

    domains = [
        ("global", "common_no_grl_ata", in_common),
        ("global", "full_native", full_valid),
    ]
    for cont in CONTINENTS:
        cm = in_common & (continents == cont)
        if np.any(cm):
            domains.append(("continent", cont, cm))
    for code, label in ARIDITY_LABELS.items():
        if code == 0:
            continue
        am = in_common & (aridity == code)
        if np.any(am):
            domains.append(("aridity_zone", label, am))

    weights = {
        "area": fabric_meta["area"],
        "runoff": fabric_meta["runoff"],
        "population": fabric_meta["population"],
        "exposure_eae": fabric_meta["exposure_eae"],
    }

    records = []
    safe_idx = np.where(valid_feat, row_idx, 0)

    # Also compute calibrated majority-vote (>=2 of 3) if all 3 detectors present
    det_items = list(scores_by_det.items())
    for det_name, (scores_hyb, q05_train) in det_items:
        u_scores = scores_hyb[safe_idx]
        ood_lt0 = (u_scores < 0.0).astype(np.float64)
        ood_q05 = (u_scores < q05_train).astype(np.float64)

        for d_type, d_name, d_mask in domains:
            idx_d = np.flatnonzero(d_mask)
            if idx_d.size == 0:
                continue
            sc_d = u_scores[idx_d]
            f0_d = ood_lt0[idx_d]
            fq_d = ood_q05[idx_d]
            for w_name, w_full in weights.items():
                if w_full is None:
                    continue
                w = w_full[idx_d]
                wt = float(np.sum(w))
                if wt <= 0.0:
                    continue
                records.append({
                    "hydrofabric": fabric_meta["fabric_id"],
                    "tier": tier_key,
                    "n_train_unique_hybas": n_train_unique,
                    "detector": det_name,
                    "domain_type": d_type,
                    "domain_name": d_name,
                    "weight": w_name,
                    "frac_lt_0": float(np.dot(w, f0_d) / wt),
                    "frac_lt_train_q05": float(np.dot(w, fq_d) / wt),
                    "mean_score": float(np.dot(w, sc_d) / wt),
                    "q05_train_thresh": float(q05_train),
                    "total_weight": wt,
                })

    if all(k in scores_by_det for k in ("lof", "if", "ocsvm")):
        votes_q05 = (
            (scores_by_det["lof"][0][safe_idx] < scores_by_det["lof"][1]).astype(np.int8)
            + (scores_by_det["if"][0][safe_idx] < scores_by_det["if"][1]).astype(np.int8)
            + (scores_by_det["ocsvm"][0][safe_idx] < scores_by_det["ocsvm"][1]).astype(np.int8)
        )
        maj_q05 = (votes_q05 >= 2).astype(np.float64)
        any_q05 = (votes_q05 >= 1).astype(np.float64)
        for d_type, d_name, d_mask in domains:
            idx_d = np.flatnonzero(d_mask)
            if idx_d.size == 0:
                continue
            m_d = maj_q05[idx_d]
            a_d = any_q05[idx_d]
            for w_name, w_full in weights.items():
                if w_full is None:
                    continue
                w = w_full[idx_d]
                wt = float(np.sum(w))
                if wt <= 0.0:
                    continue
                records.append({
                    "hydrofabric": fabric_meta["fabric_id"],
                    "tier": tier_key,
                    "n_train_unique_hybas": n_train_unique,
                    "detector": "ensemble_2of3_q05",
                    "domain_type": d_type,
                    "domain_name": d_name,
                    "weight": w_name,
                    "frac_lt_0": np.nan,
                    "frac_lt_train_q05": float(np.dot(w, m_d) / wt),
                    "mean_score": float(np.dot(w, a_d) / wt),  # stores >=1 of 3 for reference
                    "q05_train_thresh": np.nan,
                    "total_weight": wt,
                })

    return records


def fit_score_ocsvm_worker(args: tuple) -> tuple[str, str, np.ndarray, float, float]:
    """Worker to fit OneClassSVM and score all 1.03M HydroBASINS units using vectorized RBF."""
    fabric_id, tier_key, train_rows = args
    t0 = time.time()
    df = pd.read_parquet(FEAT_CACHE)
    feat_cols = [c for c in df.columns if c != "HYBAS_ID"]
    X_all = df[feat_cols].to_numpy(dtype=np.float64)
    X_tr = X_all[train_rows]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_all_s = scaler.transform(X_all)

    ocsvm = OneClassSVM(cache_size=4000)
    ocsvm.fit(X_tr_s)

    sv = np.ascontiguousarray(ocsvm.support_vectors_, dtype=np.float64)
    coef = np.ascontiguousarray(ocsvm.dual_coef_.ravel(), dtype=np.float64)
    gamma = float(ocsvm._gamma)
    intercept = float(ocsvm.intercept_[0])
    sv_sq = np.einsum("ij,ij->i", sv, sv)

    def eval_rbf(X_mat: np.ndarray) -> np.ndarray:
        out = np.empty(len(X_mat), dtype=np.float64)
        bs = 8192
        for i in range(0, len(X_mat), bs):
            xb = np.ascontiguousarray(X_mat[i : i + bs], dtype=np.float64)
            xb_sq = np.einsum("ij,ij->i", xb, xb)[:, None]
            d2 = xb_sq + sv_sq[None, :] - 2.0 * (xb @ sv.T)
            np.maximum(d2, 0.0, out=d2)
            out[i : i + bs] = np.exp(-gamma * d2) @ coef + intercept
        return out

    scores_all = eval_rbf(X_all_s).astype(np.float32)
    scores_tr = scores_all[train_rows]
    q05_tr = float(np.percentile(scores_tr, 5))
    elapsed = time.time() - t0
    print(
        f"  [OCSVM] {fabric_id} / {tier_key} (N_tr={len(train_rows):,}, SVs={len(sv):,}): "
        f"done in {elapsed:.1f}s | <0={(scores_all < 0).mean()*100:.1f}% | <q05={(scores_all < q05_tr).mean()*100:.1f}%",
        flush=True,
    )
    return fabric_id, tier_key, scores_all, q05_tr, elapsed


def main() -> None:
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hybas_ids, X_all_f32, feat_cols = load_or_build_hybas_88()
    hybas_id_to_row = {int(h): i for i, h in enumerate(hybas_ids)}
    gauges_df = pd.read_parquet(GAUGES_DIR / "global_gauges.parquet")

    fabrics = ["hydrosheds_l12", "merit_basins", "tdx_hydro", "grit_v06_mainstem_tree"]
    metas = {}
    for f in fabrics:
        tf = time.time()
        metas[f] = load_hydrofabric_meta(f, gauges_df)
        print(f"Loaded {f} ({len(metas[f]['area']):,} units) in {time.time() - tf:.1f}s", flush=True)

    # PHASE A: LOF + Isolation Forest across all 4 fabrics x 4 tiers (super fast with 64 cores!)
    print("\n=== PHASE A: LOF + Isolation Forest across 4 hydrofabrics x 4 tiers ===", flush=True)
    X_all = X_all_f32.astype(np.float64)
    phase_a_scores: dict[tuple[str, str], dict[str, tuple[np.ndarray, float]]] = {}
    train_row_map: dict[tuple[str, str], np.ndarray] = {}
    rows_a = []

    for f in fabrics:
        for tkey, _ in TIERS:
            t0 = time.time()
            th = metas[f]["tier_hybas"][tkey]
            tr_rows = np.array([hybas_id_to_row[int(h)] for h in th if int(h) in hybas_id_to_row], dtype=np.int64)
            train_row_map[(f, tkey)] = tr_rows

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_all[tr_rows])
            X_all_s = scaler.transform(X_all)

            # 1. Local Outlier Factor (exact 1_anomaly_scores.py settings)
            lof = LocalOutlierFactor(novelty=True, n_jobs=-1)
            lof.fit(X_tr_s)
            s_lof = lof.decision_function(X_all_s).astype(np.float32)
            # Exact leave-one-out training decision scores: negative_outlier_factor_ - offset_
            tr_lof_loo = lof.negative_outlier_factor_ - lof.offset_
            q05_lof = float(np.percentile(tr_lof_loo, 5))

            # 2. Isolation Forest (exact 1_anomaly_scores.py settings)
            iso = IsolationForest(random_state=42, n_jobs=-1)
            iso.fit(X_tr_s)
            s_if = iso.decision_function(X_all_s).astype(np.float32)
            q05_if = float(np.percentile(s_if[tr_rows], 5))

            phase_a_scores[(f, tkey)] = {
                "lof": (s_lof, q05_lof),
                "if": (s_if, q05_if),
            }
            print(
                f"  [LOF+IF] {f:22s} / {tkey:24s} (N_tr={len(tr_rows):6,d}) in {time.time()-t0:4.1f}s | "
                f"LOF<0={((s_lof<0).mean()*100):5.1f}%, LOF<q05={((s_lof<q05_lof).mean()*100):5.1f}%, "
                f"IF<q05={((s_if<q05_if).mean()*100):5.1f}%",
                flush=True,
            )
            rows_a.extend(
                aggregate_scores_to_records(
                    metas[f], hybas_id_to_row, phase_a_scores[(f, tkey)], tkey, len(tr_rows)
                )
            )

    df_a = pd.DataFrame(rows_a)
    df_a.to_parquet(OUT_DIR / "ohn_ood_lof_if_4fabrics.parquet", index=False)
    print(f"\nSaved Phase A ({len(df_a):,} rows) in {time.time() - t_start:.1f}s!", flush=True)

    # Print quick summary of Phase A
    sub_a = df_a[(df_a["domain_name"] == "common_no_grl_ata") & (df_a["weight"].isin(["area", "runoff", "population", "exposure_eae"]))]
    print("\n--- PHASE A GLOBAL SUMMARY (common_no_grl_ata, frac_lt_0) ---")
    print(sub_a.pivot_table(index=["detector", "weight", "tier"], columns="hydrofabric", values="frac_lt_0").round(4).to_string())
    print("\n--- PHASE A GLOBAL SUMMARY (common_no_grl_ata, frac_lt_train_q05) ---")
    print(sub_a.pivot_table(index=["detector", "weight", "tier"], columns="hydrofabric", values="frac_lt_train_q05").round(4).to_string())

    # PHASE B: One-Class SVM in parallel across workers (exact sklearn settings)
    # Use "spawn" context to prevent Linux fork()-after-OpenBLAS/OpenMP deadlocks
    # inherited from Phase A's parallel LOF/IsolationForest calls.
    n_cpus = os.cpu_count() or 8
    n_workers = min(8, max(1, n_cpus // 4))
    blas_threads = max(1, n_cpus // n_workers)
    print(f"\n=== PHASE B: One-Class SVM across 4 hydrofabrics x 4 tiers (workers={n_workers}, threads={blas_threads}, spawn) ===", flush=True)
    os.environ["OMP_NUM_THREADS"] = str(blas_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)
    os.environ["MKL_NUM_THREADS"] = str(blas_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(blas_threads)

    # Schedule largest tiers first to minimize tail stragglers
    tasks = sorted(
        [(f, tkey, train_row_map[(f, tkey)]) for f in fabrics for tkey, _ in TIERS],
        key=lambda item: -len(item[2]),
    )
    spawn_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=spawn_ctx) as ex:
        futs = [ex.submit(fit_score_ocsvm_worker, t) for t in tasks]
        for fut in as_completed(futs):
            f, tkey, s_ocsvm, q05_ocsvm, _ = fut.result()
            phase_a_scores[(f, tkey)]["ocsvm"] = (s_ocsvm, q05_ocsvm)

    # Aggregate full 3-detector + ensemble table
    rows_all = []
    for f in fabrics:
        for tkey, _ in TIERS:
            rows_all.extend(
                aggregate_scores_to_records(
                    metas[f], hybas_id_to_row, phase_a_scores[(f, tkey)], tkey, len(train_row_map[(f, tkey)])
                )
            )
    df_all = pd.DataFrame(rows_all)
    df_all.to_parquet(OUT_DIR / "ohn_ood_all3_4fabrics.parquet", index=False)
    df_all.to_csv(OUT_DIR / "ohn_ood_all3_4fabrics.csv", index=False)
    print(f"\nSaved COMPLETE 3-detector table ({len(df_all):,} rows) in {time.time() - t_start:.1f}s!", flush=True)

    sub_all = df_all[(df_all["domain_name"] == "common_no_grl_ata") & (df_all["weight"].isin(["area", "runoff", "population", "exposure_eae"]))]
    print("\n--- COMPLETE 3-DETECTOR GLOBAL SUMMARY (common_no_grl_ata, frac_lt_0) ---")
    print(sub_all[sub_all["detector"] != "ensemble_2of3_q05"].pivot_table(index=["detector", "weight", "tier"], columns="hydrofabric", values="frac_lt_0").round(4).to_string())
    print("\n--- COMPLETE 3-DETECTOR GLOBAL SUMMARY (common_no_grl_ata, frac_lt_train_q05) ---")
    print(sub_all.pivot_table(index=["detector", "weight", "tier"], columns="hydrofabric", values="frac_lt_train_q05").round(4).to_string())

    # Continental breakdown for Fig. 3c (area-weighted, common_no_grl_ata)
    sub_cont = df_all[(df_all["domain_type"] == "continent") & (df_all["weight"] == "area") & (df_all["tier"].isin(["baseline", "tier3_all_known"]))]
    print("\n--- CONTINENTAL BREAKDOWN (area-weighted, LOF < 0) ---")
    print(sub_cont[sub_cont["detector"] == "lof"].pivot_table(index=["domain_name", "tier"], columns="hydrofabric", values="frac_lt_0").round(4).to_string())
    print("\n--- CONTINENTAL BREAKDOWN (area-weighted, LOF < q05_train) ---")
    print(sub_cont[sub_cont["detector"] == "lof"].pivot_table(index=["domain_name", "tier"], columns="hydrofabric", values="frac_lt_train_q05").round(4).to_string())
    print("\n--- CONTINENTAL BREAKDOWN (area-weighted, ensemble_2of3_q05) ---")
    print(sub_cont[sub_cont["detector"] == "ensemble_2of3_q05"].pivot_table(index=["domain_name", "tier"], columns="hydrofabric", values="frac_lt_train_q05").round(4).to_string())


if __name__ == "__main__":
    main()
