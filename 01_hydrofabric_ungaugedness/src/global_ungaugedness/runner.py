"""Unified evaluation runner across all hydrofabrics, gauge tiers, domains, and metric families."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from configs.benchmark_config import (
    AREA_SCALE_GRID_KM2,
    DISCHARGE_SCALE_GRID_M3S,
    GAUGE_TIERS,
    RESULTS_DIR,
    RIVER_AREA_BINS_KM2,
    RIVER_DISCHARGE_BINS_M3S,
)
from global_ungaugedness.metrics_1d import compute_1d_metrics_tree_multi_domain
from global_ungaugedness.metrics_2d import compute_2d_metrics_tree_multi_domain
from global_ungaugedness.metrics_ood import compute_ood_metrics
from global_ungaugedness.spatial_enrichment import ARIDITY_LABELS


def aggregate_gauge_snaps_to_units(
    n_units: int,
    unit_id_to_idx: dict[int, int],
    gauges_df: pd.DataFrame,
    snap_df: pd.DataFrame,
    tier_col: str,
    local_area: np.ndarray,
    total_area_out: np.ndarray,
    local_runoff: np.ndarray | None = None,
    total_runoff_out: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Map gauges in a given tier onto unit arrays for 2D and 1D metric kernels.

    Returns:
        has_gauge (uint8[N]),
        effective_gauge_area (float64[N], -1 where no sub-unit cap applies),
        effective_gauge_runoff (float64[N] or None),
        gauge_unit_indices (int32[M_snapped]),
        gauge_frac_pos (float64[M_snapped]).
    """
    m_tier = gauges_df[tier_col].to_numpy(dtype=bool)
    sub_snap = snap_df.loc[m_tier]
    valid = sub_snap["snapped_unit_id"].to_numpy(dtype=np.int64) >= 0
    sub_snap = sub_snap.loc[valid]

    uids = sub_snap["snapped_unit_id"].to_numpy(dtype=np.int64)
    u_idx = np.array([unit_id_to_idx.get(int(u), -1) for u in uids], dtype=np.int32)
    keep = u_idx >= 0
    u_idx = u_idx[keep]

    if "snap_frac_geom" in sub_snap.columns:
        fracs = sub_snap["snap_frac_geom"].to_numpy(dtype=np.float64)[keep]
        fracs = np.where(np.isfinite(fracs), np.clip(fracs, 0.0, 1.0), 1.0)
    else:
        fracs = np.ones(len(u_idx), dtype=np.float64)

    has_g = np.zeros(n_units, dtype=np.uint8)
    eff_a = np.full(n_units, -1.0, dtype=np.float64)
    if u_idx.size > 0:
        has_g[u_idx] = 1
        # Effective gauged area per unit (if multiple gauges land in one unit, take max monitored area,
        # capped to total_area_out[u]; if any gauge is non-headwater, monitors full total_area_out[u])
        if "effective_gauged_area_km2" in sub_snap.columns:
            g_eff = sub_snap["effective_gauged_area_km2"].to_numpy(dtype=np.float64)[keep]
            g_is_sub = (
                sub_snap["is_sub_headwater"].to_numpy(dtype=bool)[keep]
                if "is_sub_headwater" in sub_snap.columns
                else np.zeros(len(u_idx), dtype=bool)
            )
            # Default: full unit outlet monitored; track sum of sub-headwater areas if ALL gauges in u are sub-headwater
            has_full = np.zeros(n_units, dtype=np.uint8)
            sub_sum = np.zeros(n_units, dtype=np.float64)
            np.add.at(has_full, u_idx[~g_is_sub], 1)
            np.add.at(sub_sum, u_idx[g_is_sub], np.where(np.isfinite(g_eff[g_is_sub]), g_eff[g_is_sub], 0.0))

            only_sub = (has_g == 1) & (has_full == 0) & (sub_sum > 0.0)
            eff_a[only_sub] = np.minimum(sub_sum[only_sub], total_area_out[only_sub])

    eff_q = None
    if local_runoff is not None and total_runoff_out is not None:
        eff_q = np.full(n_units, -1.0, dtype=np.float64)
        sub_mask = eff_a >= 0.0
        if np.any(sub_mask):
            ratio = np.where(
                total_area_out[sub_mask] > 0.0,
                np.clip(eff_a[sub_mask] / total_area_out[sub_mask], 0.0, 1.0),
                1.0,
            )
            eff_q[sub_mask] = total_runoff_out[sub_mask] * ratio

    return has_g, eff_a, eff_q, u_idx, fracs


def run_tree_hydrofabric_suite(
    fabric_id: str,
    units_df: pd.DataFrame,
    order: np.ndarray,
    next_down: np.ndarray,
    gauges_df: pd.DataFrame,
    snap_df: pd.DataFrame,
    has_flow: bool,
    run_1d: bool,
    ood_features: np.ndarray | None = None,
    tiers: Sequence[tuple[str, str, str]] = GAUGE_TIERS,
    run_subbasin_eff_mode: bool = True,
    out_dir: Path | None = None,
) -> Path:
    """Execute full metric suite across all tiers and spatial domains for a tree hydrofabric."""
    t0 = time.time()
    out_path = out_dir or (RESULTS_DIR / fabric_id)
    out_path.mkdir(parents=True, exist_ok=True)

    n = len(units_df)
    unit_ids = units_df["unit_id"].to_numpy(dtype=np.int64)
    unit_id_to_idx = {int(uid): i for i, uid in enumerate(unit_ids)}

    local_a = units_df["local_area_km2"].to_numpy(dtype=np.float64)
    a_in = units_df["up_area_in_km2"].to_numpy(dtype=np.float64)
    a_out = units_df["up_area_out_km2"].to_numpy(dtype=np.float64)
    length_km = units_df["length_km"].to_numpy(dtype=np.float64) if "length_km" in units_df.columns else np.zeros(n)

    pop = units_df["population"].to_numpy(dtype=np.float64) if "population" in units_df.columns else None
    eae = units_df["exposure_eae"].to_numpy(dtype=np.float64) if "exposure_eae" in units_df.columns else None
    vol = units_df["channel_volume_m3"].to_numpy(dtype=np.float64) if "channel_volume_m3" in units_df.columns else None

    if has_flow:
        local_q = units_df["local_runoff_m3s"].to_numpy(dtype=np.float64)
        q_cons_in = units_df["q_cons_in_m3s"].to_numpy(dtype=np.float64)
        q_cons_out = units_df["q_cons_out_m3s"].to_numpy(dtype=np.float64)
        q_phys_in = units_df["q_phys_in_m3s"].to_numpy(dtype=np.float64) if "q_phys_in_m3s" in units_df.columns else q_cons_in
        q_phys_out = units_df["q_phys_out_m3s"].to_numpy(dtype=np.float64) if "q_phys_out_m3s" in units_df.columns else q_cons_out
    else:
        local_q = q_cons_in = q_cons_out = q_phys_in = q_phys_out = None

    weights_2d: dict[str, np.ndarray | None] = {
        "count": np.ones(n, dtype=np.float64),
        "area": local_a,
        "runoff": local_q,
        "population": pop,
        "exposure_eae": eae,
    }

    # Spatial reporting domains
    in_common = units_df["in_common_domain"].to_numpy(dtype=bool)
    continents = units_df["continent"].astype(str).to_numpy()
    countries = units_df["country_iso3"].astype(str).to_numpy()
    aridity = units_df["aridity_zone"].to_numpy(dtype=np.int8) if "aridity_zone" in units_df.columns else np.zeros(n, dtype=np.int8)

    macro_domains: list[tuple[str, str, np.ndarray]] = [
        ("global", "full_native", np.ones(n, dtype=bool)),
        ("global", "common_no_grl_ata", in_common),
    ]
    for cont in ["Africa", "Asia", "Europe", "North America", "South America", "Oceania"]:
        cm = in_common & (continents == cont)
        if np.any(cm):
            macro_domains.append(("continent", cont, cm))
    for code, label in ARIDITY_LABELS.items():
        if code == 0:
            continue
        am = in_common & (aridity == code)
        if np.any(am):
            macro_domains.append(("aridity_zone", label, am))

    rows_2d: list[dict] = []
    rows_1d: list[dict] = []
    rows_ood: list[dict] = []
    rows_country: list[dict] = []

    # Precompute country index groups once outside the tier loop
    uniq_countries = sorted(c for c in set(countries[in_common]) if c not in ("UNK", "-99", "nan", "ATA", "GRL"))
    c_series = pd.Series(countries)
    c_groups = c_series.groupby(c_series).indices
    w_country = {k: weights_2d[k] for k in ["area", "runoff", "exposure_eae"] if weights_2d.get(k) is not None}

    for tier_key, tier_label, tier_col in tiers:
        t_tier = time.time()
        has_g, eff_a, eff_q, g_u_idx, g_fracs = aggregate_gauge_snaps_to_units(
            n, unit_id_to_idx, gauges_df, snap_df, tier_col,
            local_a, a_out, local_q, q_cons_out,
        )
        n_snapped = int(len(g_u_idx))
        n_units_gauged = int(has_g.sum())
        print(
            f"[{fabric_id}] Tier={tier_key}: {n_snapped:,} snapped gauges across {n_units_gauged:,} units",
            flush=True,
        )

        # Modes for 2D metrics (ONE graph sweep + ONE sort per mode across all macro domains!)
        outlet_area_fields = None
        modes = [False, True] if run_subbasin_eff_mode else [False]
        for sub_eff in modes:
            recs, area_fields = compute_2d_metrics_tree_multi_domain(
                order=order,
                next_down=next_down,
                local_area=local_a,
                total_area_out=a_out,
                has_gauge=has_g,
                weights_dict=weights_2d,
                domains=macro_domains,
                effective_gauge_area=eff_a if sub_eff else None,
                use_subbasin_effective=sub_eff,
                local_runoff=local_q,
                total_runoff_out=q_cons_out,
                effective_gauge_runoff=eff_q if sub_eff else None,
                area_scale_grid=AREA_SCALE_GRID_KM2,
                discharge_scale_grid=DISCHARGE_SCALE_GRID_M3S,
                return_area_fields=True,
            )
            if not sub_eff:
                outlet_area_fields = area_fields
            for r in recs:
                r["hydrofabric"] = fabric_id
                r["tier"] = tier_key
                r["n_gauges_snapped"] = n_snapped
            rows_2d.extend(recs)

        # Country breakdown (reusing outlet_area_fields computed above!)
        c_up_a, c_down_a, u_net_a, f_nev_a, net_g, _fd, _dia, res_ia, _iu = outlet_area_fields
        for iso in uniq_countries:
            c_idx = c_groups.get(iso)
            if c_idx is None or len(c_idx) == 0:
                continue
            for w_name, w_arr in w_country.items():
                w = w_arr[c_idx]
                wt = float(np.sum(w))
                if wt <= 0.0:
                    continue
                ia_c = res_ia[c_idx]
                fin_pos = np.isfinite(ia_c) & (ia_c > 0.0)
                safe_ia = np.where(fin_pos, ia_c, 1.0)
                u_nev_c = float(np.dot(w, f_nev_a[c_idx]) / wt)
                u_znet_c = float(np.dot(w, (net_g[c_idx] == 0).astype(np.float64)) / wt)
                cred_1k = np.where(fin_pos, np.minimum(1.0, 1_000.0 / safe_ia), np.where(ia_c == 0.0, 1.0, 0.0))
                cred_10k = np.where(fin_pos, np.minimum(1.0, 10_000.0 / safe_ia), np.where(ia_c == 0.0, 1.0, 0.0))
                rows_country.append({
                    "hydrofabric": fabric_id,
                    "tier": tier_key,
                    "country_iso3": iso,
                    "weight": w_name,
                    "total_weight": wt,
                    "U_out": float(np.dot(w, (has_g[c_idx] == 0).astype(np.float64)) / wt),
                    "U_up": float(np.dot(w, 1.0 - c_up_a[c_idx]) / wt),
                    "U_down": float(np.dot(w, 1.0 - c_down_a[c_idx]) / wt),
                    "U_net_Eq4": float(np.dot(w, u_net_a[c_idx]) / wt),
                    "U_zero_net": u_znet_c,
                    "U_never": u_nev_c,
                    "U_tail": u_nev_c - u_znet_c,
                    "U_res_1k": float(np.sum(w[ia_c > 1_000.0]) / wt),
                    "U_res_10k": float(np.sum(w[ia_c > 10_000.0]) / wt),
                    "U_cont_1k": float(1.0 - np.dot(w, cred_1k) / wt),
                    "U_cont_10k": float(1.0 - np.dot(w, cred_10k) / wt),
                })

        # Family 4: 1D continuous river-network metrics (ONE graph sweep across all macro domains!)
        if run_1d and np.any(length_km > 0.0):
            recs_1d = compute_1d_metrics_tree_multi_domain(
                order=order,
                next_down=next_down,
                length_km=length_km,
                area_in_km2=a_in,
                area_out_km2=a_out,
                gauge_reach_idx=g_u_idx,
                domains=macro_domains,
                gauge_frac_pos=g_fracs,
                q_cons_in_m3s=q_cons_in,
                q_cons_out_m3s=q_cons_out,
                q_phys_in_m3s=q_phys_in,
                q_phys_out_m3s=q_phys_out,
                exposure_eae=eae,
                channel_volume_m3=vol,
                area_bins=RIVER_AREA_BINS_KM2,
                q_bins=RIVER_DISCHARGE_BINS_M3S,
            )
            for r in recs_1d:
                r["hydrofabric"] = fabric_id
                r["tier"] = tier_key
                r["n_gauges_snapped"] = n_snapped
            rows_1d.extend(recs_1d)

        # Family 5: Hydroclimatic feature-space OOD metrics
        if ood_features is not None and g_u_idx.size >= 10:
            X_train = ood_features[np.unique(g_u_idx)]
            grp_ids = units_df["hybas_l12_id"].to_numpy() if "hybas_l12_id" in units_df.columns else None
            # Compute k-NN distances once on full target, then summarize across domains
            w_ood = {k: weights_2d[k] for k in ["area", "runoff", "population", "exposure_eae"] if weights_2d.get(k) is not None}
            recs_ood_full, dists_by_k = compute_ood_metrics(
                ood_features, X_train, w_ood, domain_mask=None, k_neighbors=(1, 5), workers=8,
                target_group_ids=grp_ids,
            )
            for r in recs_ood_full:
                r["hydrofabric"] = fabric_id
                r["tier"] = tier_key
                r["n_gauges_snapped"] = n_snapped
                r["domain_type"] = "global"
                r["domain_name"] = "full_native"
            rows_ood.extend(recs_ood_full)

            # Reuse precomputed distances for common domain, continents, aridity!
            from global_ungaugedness.metrics_2d import weighted_quantiles
            for d_type, d_name, d_mask in macro_domains[1:]:
                idx_d = np.flatnonzero(d_mask)
                if idx_d.size == 0:
                    continue
                for k_val, dk_arr in dists_by_k.items():
                    dk_d = dk_arr[idx_d]
                    for w_name, w_full in w_ood.items():
                        w = w_full[idx_d]
                        wt = float(np.sum(w))
                        if wt <= 0.0:
                            continue
                        q25, q50, q75 = weighted_quantiles(dk_d, w, (0.25, 0.50, 0.75))
                        mean_d = float(np.dot(w, dk_d) / wt)
                        for m_name, val in [
                            (f"U_OOD_knn{k_val}_mean", mean_d),
                            (f"U_OOD_knn{k_val}_q25", q25),
                            (f"U_OOD_knn{k_val}_q50", q50),
                            (f"U_OOD_knn{k_val}_q75", q75),
                        ]:
                            rows_ood.append({
                                "hydrofabric": fabric_id,
                                "tier": tier_key,
                                "n_gauges_snapped": n_snapped,
                                "domain_type": d_type,
                                "domain_name": d_name,
                                "family": "F5_hydroclimatic_ood",
                                "metric": m_name,
                                "ratio_basis": "feature_space",
                                "snapping_mode": "standardized_knn",
                                "weight": w_name,
                                "scale_param": float(k_val),
                                "value": float(val),
                                "total_weight": wt,
                            })

        print(f"[{fabric_id}] Finished {tier_key} in {time.time() - t_tier:.1f}s", flush=True)

    pd.DataFrame(rows_2d).to_parquet(out_path / "metrics_2d_global.parquet", index=False)
    if rows_country:
        pd.DataFrame(rows_country).to_parquet(out_path / "metrics_2d_country.parquet", index=False)
    if rows_1d:
        pd.DataFrame(rows_1d).to_parquet(out_path / "metrics_1d_global.parquet", index=False)
    if rows_ood:
        pd.DataFrame(rows_ood).to_parquet(out_path / "metrics_ood.parquet", index=False)

    print(f"[{fabric_id}] Saved complete results to {out_path} in {time.time() - t0:.1f}s", flush=True)
    return out_path
