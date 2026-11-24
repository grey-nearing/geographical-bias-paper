"""Families 1, 2, and 3: 2D Polygon, Mass-Balance, and Inter-Gauge Resolution Metrics.

Computes all 11 2D/polygon metrics across any tree hydrofabric (and DAG
generalization for GRIT v1.0):

Family 1 — Hydrofabric-Dependent Polygon Metrics:
  1. U_out          : directly ungauged unit fraction (has_gauge == 0)
  2. U_zero_up      : zero upstream gauge coverage (c_up == 0)
  3. U_up           : weighted mean of (1 - c_up)
  4. U_down         : weighted mean of (1 - c_down)
  5. U_net_Eq4      : weighted mean of (1 - c_up) * (1 - c_down)
                      (both outlet-snapped and sub-polygon headwater effective area)
  6. U_zero_net     : entire connected river network ungauged

Family 2 — Strictly Scale-Invariant 2D Mass-Balance Metrics:
  7. U_never        : fraction of land/runoff/exposure never passing through any gauge
  8. U_tail         : U_never - U_zero_net (below-lowest-gauge coastal/sink tail)

Family 3 — Scale-Parameterized 2D Inter-Gauge Resolution Metrics:
  9. U_res(A_0)     : strict pass/fail fraction in inter-gauge catchments > A_0 or never gauged
 10. A_tilde_quant  : weighted quantiles (q25, q50, q75) of inter-gauge area A_tilde(p)
                      both unconditional (all land, never-gauged = inf) and conditional on monitored land
 11. U_cont(A_0)    : smooth partial-credit metric capped at A_0 per gauge:
                      - area-weighted: 1 - (1/|Omega|) * sum_g min(A_tilde_g, A_0)
                      - general w-weighted: sum_i w_i * (1 - min(1, A_0 / A_tilde_i)) / sum_i w_i
                      (identical to the per-gauge cap when w_i = area_i!)
                      - discharge-scale Q_0 equivalent: 1 - (1/Q_tot) * sum_g min(Q_tilde_g, Q_0)
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
from numba import njit

from global_ungaugedness.graph import propagate_tree_gauges


def weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    probs: Sequence[float] = (0.25, 0.50, 0.75),
) -> list[float]:
    """Compute exact weighted quantiles, preserving np.inf values cleanly."""
    mask = (weights > 0.0) & (~np.isnan(values))
    v = values[mask]
    w = weights[mask]
    total_w = float(np.sum(w))
    if v.size == 0 or total_w <= 0.0:
        return [float("nan") for _ in probs]

    order = np.argsort(v, kind="mergesort")
    v_s = v[order]
    w_s = w[order]
    cdf = np.cumsum(w_s) / total_w

    out = []
    for p in probs:
        idx = int(np.searchsorted(cdf, p, side="left"))
        if idx >= v_s.size:
            idx = v_s.size - 1
        out.append(float(v_s[idx]))
    return out


@njit(cache=True)
def _compute_unit_fields_tree(
    order: np.ndarray,
    next_down: np.ndarray,
    local_w: np.ndarray,
    total_w_out: np.ndarray,
    has_gauge: np.ndarray,
    effective_gauge_w: np.ndarray,
    use_subbasin_effective: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Compute per-unit coverage arrays (c_up, c_down, u_net, is_never, inter_w, net_gauged).

    Also returns per-unit sub-polygon decomposition arrays so that when
    use_subbasin_effective=True, the directly gauged fraction of unit u
    (frac_direct) has inter-gauge scale `direct_inter_w` and is monitored,
    while the residual fraction (1 - frac_direct) drains to the strictly
    downstream gauge (or is never gauged if none).
    """
    n = next_down.shape[0]
    g_up_in, g_up_out, down_gauge_node, inter_w, net_has_gauge = propagate_tree_gauges(
        order,
        next_down,
        local_w,
        total_w_out,
        has_gauge,
        effective_gauge_w,
        use_subbasin_effective,
    )

    # Compute strictly downstream gauge (ignoring any gauge inside u itself)
    strict_down_node = np.full(n, -1, dtype=np.int32)
    strict_down_inter_w = np.full(n, np.inf, dtype=np.float64)
    for k in range(order.shape[0] - 1, -1, -1):
        u = order[k]
        d = next_down[u]
        if d >= 0:
            strict_down_node[u] = down_gauge_node[d]
            strict_down_inter_w[u] = inter_w[d]

    c_up = np.zeros(n, dtype=np.float64)
    c_down = np.zeros(n, dtype=np.float64)
    u_net = np.ones(n, dtype=np.float64)
    frac_never = np.ones(n, dtype=np.float64)
    frac_direct_gauged = np.zeros(n, dtype=np.float64)
    direct_inter_w = np.full(n, np.inf, dtype=np.float64)
    resid_inter_w = np.full(n, np.inf, dtype=np.float64)

    for u in range(n):
        tot = total_w_out[u]
        loc = local_w[u]
        # Upstream coverage ratio c_up = g_up_out / total_w_out
        if tot > 0.0:
            cu = g_up_out[u] / tot
            if cu > 1.0:
                cu = 1.0
            elif cu < 0.0:
                cu = 0.0
        else:
            cu = 1.0 if has_gauge[u] else 0.0
        c_up[u] = cu

        # Downstream coverage ratio c_down
        if has_gauge[u]:
            if use_subbasin_effective and effective_gauge_w[u] >= 0.0 and effective_gauge_w[u] < tot:
                # Sub-polygon headwater gauge: gauge captures eff - g_up_in of local_w
                # Residual drains to strictly downstream gauge
                direct_w = g_up_out[u] - g_up_in[u]
                if direct_w < 0.0:
                    direct_w = 0.0
                if loc > 0.0:
                    fd = direct_w / loc
                    if fd > 1.0:
                        fd = 1.0
                else:
                    fd = 1.0
                frac_direct_gauged[u] = fd
                direct_inter_w[u] = inter_w[u]

                sdg = strict_down_node[u]
                if sdg >= 0 and total_w_out[sdg] > 0.0:
                    cd_resid = tot / total_w_out[sdg]
                    if cd_resid > 1.0:
                        cd_resid = 1.0
                    resid_inter_w[u] = strict_down_inter_w[u]
                    frac_never[u] = 0.0
                else:
                    cd_resid = 0.0
                    resid_inter_w[u] = np.inf
                    frac_never[u] = 1.0 - fd

                # Polygon-level c_down matches original paper Eq. 4 convention:
                # uses strictly downstream gauge on residual (or max with direct_w/tot)
                c_down[u] = cd_resid
            else:
                # Full-outlet gauge in unit u
                c_down[u] = 1.0
                frac_direct_gauged[u] = 1.0
                direct_inter_w[u] = inter_w[u]
                resid_inter_w[u] = inter_w[u]
                frac_never[u] = 0.0
        else:
            frac_direct_gauged[u] = 0.0
            dg = down_gauge_node[u]
            if dg >= 0:
                # Downstream gauge total area (or effective area if dg is sub-unit,
                # though strictly upstream u only reaches dg if dg monitors u's inflow)
                denom = g_up_out[dg] if (use_subbasin_effective and g_up_out[dg] > 0.0) else total_w_out[dg]
                if denom > 0.0:
                    cd = tot / denom
                    if cd > 1.0:
                        cd = 1.0
                else:
                    cd = 0.0
                c_down[u] = cd
                resid_inter_w[u] = inter_w[u]
                frac_never[u] = 0.0
            else:
                c_down[u] = 0.0
                resid_inter_w[u] = np.inf
                frac_never[u] = 1.0

        u_net[u] = (1.0 - c_up[u]) * (1.0 - c_down[u])

    return (
        c_up,
        c_down,
        u_net,
        frac_never,
        net_has_gauge,
        frac_direct_gauged,
        direct_inter_w,
        resid_inter_w,
        inter_w,
    )


@njit(cache=True)
def _f3_multi_weight_sorted_kernel(
    s_val: np.ndarray,
    s_unit: np.ndarray,
    s_mult: np.ndarray,
    w_mat: np.ndarray,
    domain_u8: np.ndarray,
    finite_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact Family 3 quantiles, U_res(scale), and U_cont(scale) on pre-sorted values.

    Computes all weights and all scale thresholds in just two O(M) passes with
    zero temporary array allocations — mathematically identical to calling
    `weighted_quantiles` + per-threshold vector operations, but 30-100x faster.

    Args:
        s_val: Sorted ascending float64[M] inter-gauge scale values (np.inf at end).
        s_unit: int32[M] unit index for each item.
        s_mult: float64[M] sub-unit weight multiplier in (0, 1].
        w_mat: float64[n_w, N] matrix of unit weight arrays.
        domain_u8: uint8[N] domain inclusion mask.
        finite_scales: Sorted ascending strictly-positive finite float64[K] thresholds.

    Returns:
        w_tot: float64[n_w] total domain weight per weight type.
        q_all: float64[n_w, 3] (q25, q50, q75) over all weight (never-gauged = inf).
        q_mon: float64[n_w, 3] (q25, q50, q75) conditional on finite scale.
        u_res: float64[n_w, K] strict pass/fail U_res(scale).
        u_cont: float64[n_w, K] smooth partial-credit U_cont(scale).
    """
    n_w = w_mat.shape[0]
    m_len = s_val.shape[0]
    k_len = finite_scales.shape[0]

    w_tot = np.zeros(n_w, dtype=np.float64)
    w_fin = np.zeros(n_w, dtype=np.float64)
    bin_w = np.zeros((n_w, k_len + 2), dtype=np.float64)
    bin_winv = np.zeros((n_w, k_len + 1), dtype=np.float64)

    # Pass 1: accumulate exact totals and scale-bin sums in a single linear scan
    b = 0
    for m in range(m_len):
        u = s_unit[m]
        if domain_u8[u] == 0:
            continue
        v = s_val[m]
        if np.isnan(v):
            continue
        mult = s_mult[m]
        is_fin = np.isfinite(v)
        if is_fin:
            while b < k_len and v > finite_scales[b]:
                b += 1
            cur_b = b
            inv_v = 1.0 / v if (cur_b > 0 and v > 0.0) else 0.0
        else:
            cur_b = k_len + 1
            inv_v = 0.0

        for wi in range(n_w):
            w = w_mat[wi, u] * mult
            if w <= 0.0:
                continue
            w_tot[wi] += w
            bin_w[wi, cur_b] += w
            if is_fin:
                w_fin[wi] += w
                if cur_b > 0 and inv_v > 0.0:
                    bin_winv[wi, cur_b] += w * inv_v

    u_res = np.full((n_w, k_len), np.nan, dtype=np.float64)
    u_cont = np.full((n_w, k_len), np.nan, dtype=np.float64)
    for wi in range(n_w):
        wt = w_tot[wi]
        if wt <= 0.0:
            continue
        pref = 0.0
        for k in range(k_len):
            pref += bin_w[wi, k]
            suff_w = 0.0
            for j in range(k + 1, k_len + 2):
                suff_w += bin_w[wi, j]
            suff_winv = 0.0
            for j in range(k + 1, k_len + 1):
                suff_winv += bin_winv[wi, j]
            u_res[wi, k] = suff_w / wt
            credit_sum = pref + finite_scales[k] * suff_winv
            u_cont[wi, k] = 1.0 - credit_sum / wt

    # Pass 2: exact weighted quantiles (0.25, 0.50, 0.75) for all & finite
    probs = np.array([0.25, 0.50, 0.75], dtype=np.float64)
    q_all = np.full((n_w, 3), np.nan, dtype=np.float64)
    q_mon = np.full((n_w, 3), np.nan, dtype=np.float64)
    cum = np.zeros(n_w, dtype=np.float64)
    p_all = np.zeros(n_w, dtype=np.int32)
    p_mon = np.zeros(n_w, dtype=np.int32)
    last_v = np.full(n_w, np.nan, dtype=np.float64)
    last_fv = np.full(n_w, np.nan, dtype=np.float64)

    n_done = 0
    for wi in range(n_w):
        if w_tot[wi] <= 0.0:
            p_all[wi] = 3
            p_mon[wi] = 3
            n_done += 1
        elif w_fin[wi] <= 0.0:
            p_mon[wi] = 3

    for m in range(m_len):
        if n_done == n_w:
            break
        u = s_unit[m]
        if domain_u8[u] == 0:
            continue
        v = s_val[m]
        if np.isnan(v):
            continue
        mult = s_mult[m]
        is_fin = np.isfinite(v)

        for wi in range(n_w):
            if p_all[wi] == 3 and p_mon[wi] == 3:
                continue
            w = w_mat[wi, u] * mult
            if w <= 0.0:
                continue
            cum[wi] += w
            last_v[wi] = v
            if is_fin:
                last_fv[wi] = v
            c_all = cum[wi] / w_tot[wi]
            while p_all[wi] < 3 and c_all >= probs[p_all[wi]]:
                q_all[wi, p_all[wi]] = v
                p_all[wi] += 1
            if is_fin and w_fin[wi] > 0.0:
                c_mon = cum[wi] / w_fin[wi]
                while p_mon[wi] < 3 and c_mon >= probs[p_mon[wi]]:
                    q_mon[wi, p_mon[wi]] = v
                    p_mon[wi] += 1
            if p_all[wi] == 3 and p_mon[wi] == 3:
                n_done += 1

    # Guard float rounding at tail
    for wi in range(n_w):
        while p_all[wi] < 3 and w_tot[wi] > 0.0:
            q_all[wi, p_all[wi]] = last_v[wi]
            p_all[wi] += 1
        while p_mon[wi] < 3 and w_fin[wi] > 0.0:
            q_mon[wi, p_mon[wi]] = last_fv[wi]
            p_mon[wi] += 1

    return w_tot, q_all, q_mon, u_res, u_cont


def _build_sorted_scale_index(
    fd: np.ndarray,
    dir_scale: np.ndarray,
    res_scale: np.ndarray,
    use_subbasin_effective: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build globally sorted (s_val, s_unit, s_mult) index ONCE across all domains & weights."""
    n = fd.shape[0]
    if not use_subbasin_effective:
        # Every unit has a single entry (dir_scale where fd==1, res_scale where fd==0 -> identical to res_scale)
        vals = np.where(fd > 0.5, dir_scale, res_scale)
        units = np.arange(n, dtype=np.int32)
        mults = np.ones(n, dtype=np.float64)
    else:
        # Compact representation: 1 entry where fd==0 or fd==1, 2 entries only where 0 < fd < 1
        pure_d = fd >= 1.0
        pure_r = fd <= 0.0
        split = (~pure_d) & (~pure_r)
        split_idx = np.flatnonzero(split).astype(np.int32)
        base_vals = np.where(pure_d, dir_scale, res_scale)
        base_mults = np.where(split, 1.0 - fd, 1.0)
        if split_idx.size > 0:
            vals = np.concatenate([base_vals, dir_scale[split_idx]])
            units = np.concatenate([np.arange(n, dtype=np.int32), split_idx])
            mults = np.concatenate([base_mults, fd[split_idx]])
        else:
            vals = base_vals
            units = np.arange(n, dtype=np.int32)
            mults = base_mults

    order = np.argsort(vals, kind="mergesort")
    return vals[order], units[order], mults[order]


def compute_2d_metrics_tree_multi_domain(
    order: np.ndarray,
    next_down: np.ndarray,
    local_area: np.ndarray,
    total_area_out: np.ndarray,
    has_gauge: np.ndarray,
    weights_dict: dict[str, np.ndarray | None],
    domains: Sequence[tuple[str, str, np.ndarray]],
    effective_gauge_area: np.ndarray | None = None,
    use_subbasin_effective: bool = False,
    local_runoff: np.ndarray | None = None,
    total_runoff_out: np.ndarray | None = None,
    effective_gauge_runoff: np.ndarray | None = None,
    area_scale_grid: Sequence[float] = (
        50.0, 100.0, 250.0, 500.0, 1_000.0, 2_500.0, 5_000.0,
        10_000.0, 25_000.0, 50_000.0, 100_000.0, 500_000.0, float("inf"),
    ),
    discharge_scale_grid: Sequence[float] = (
        0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1_000.0, 5_000.0, float("inf"),
    ),
    return_area_fields: bool = False,
) -> list[dict] | tuple[list[dict], tuple]:
    """Compute Family 1, 2, and 3 metrics across multiple reporting domains in ONE graph sweep.

    Topological propagation and inter-gauge scale sorting run ONCE globally, then
    every domain and every weight is aggregated in O(N) without re-running graph
    kernels or re-sorting.
    """
    n = next_down.shape[0]
    has_g_u8 = np.asarray(has_gauge, dtype=np.uint8)
    eff_a = (
        np.asarray(effective_gauge_area, dtype=np.float64)
        if effective_gauge_area is not None
        else np.full(n, -1.0, dtype=np.float64)
    )

    area_fields = _compute_unit_fields_tree(
        order,
        next_down,
        local_area,
        total_area_out,
        has_g_u8,
        eff_a,
        use_subbasin_effective,
    )
    (
        c_up_a,
        c_down_a,
        u_net_a,
        frac_never_a,
        net_has_gauge,
        fd_a,
        dir_ia,
        res_ia,
        _ia_unit,
    ) = area_fields

    has_flow = local_runoff is not None and total_runoff_out is not None
    if has_flow:
        eff_q = (
            np.asarray(effective_gauge_runoff, dtype=np.float64)
            if effective_gauge_runoff is not None
            else np.full(n, -1.0, dtype=np.float64)
        )
        (
            c_up_q,
            c_down_q,
            u_net_q,
            frac_never_q,
            _net_q,
            fd_q,
            dir_iq,
            res_iq,
            _iq_unit,
        ) = _compute_unit_fields_tree(
            order,
            next_down,
            local_runoff,
            total_runoff_out,
            has_g_u8,
            eff_q,
            use_subbasin_effective,
        )

    # Active weights matrix (n_w, N)
    w_names = [k for k, v in weights_dict.items() if v is not None]
    n_w = len(w_names)
    if n_w == 0:
        return ([], area_fields) if return_area_fields else []
    w_mat = np.empty((n_w, n), dtype=np.float64)
    for wi, wn in enumerate(w_names):
        w_mat[wi] = np.asarray(weights_dict[wn], dtype=np.float64)

    # Pre-build indicator arrays once
    u_out_full = (has_g_u8 == 0).astype(np.float64)
    u_zero_up_a_full = (c_up_a <= 0.0).astype(np.float64)
    u_zero_net_full = (net_has_gauge == 0).astype(np.float64)
    one_m_cup_a = 1.0 - c_up_a
    one_m_cdown_a = 1.0 - c_down_a

    # Sort inter-gauge area scale ONCE across all domains & weights
    s_val_a, s_unit_a, s_mult_a = _build_sorted_scale_index(fd_a, dir_ia, res_ia, use_subbasin_effective)
    fin_a_scales = np.array(sorted(float(x) for x in area_scale_grid if np.isfinite(x)), dtype=np.float64)
    a_scale_to_pos = {float(x): i for i, x in enumerate(fin_a_scales)}

    if has_flow:
        u_zero_up_q_full = (c_up_q <= 0.0).astype(np.float64)
        one_m_cup_q = 1.0 - c_up_q
        one_m_cdown_q = 1.0 - c_down_q
        s_val_q, s_unit_q, s_mult_q = _build_sorted_scale_index(fd_q, dir_iq, res_iq, use_subbasin_effective)
        fin_q_scales = np.array(sorted(float(x) for x in discharge_scale_grid if np.isfinite(x)), dtype=np.float64)
        q_scale_to_pos = {float(x): i for i, x in enumerate(fin_q_scales)}

    mode_tag = "subbasin_eff" if use_subbasin_effective else "outlet"
    records: list[dict] = []

    for d_type, d_name, d_mask in domains:
        dm_u8 = np.ones(n, dtype=np.uint8) if d_mask is None else np.asarray(d_mask, dtype=np.uint8)
        idx = np.flatnonzero(dm_u8)
        if idx.size == 0:
            continue

        # Fast F3 pass for ALL weights & ALL area scales at once
        wt_a_arr, qa_all, qa_mon, ures_a, ucont_a = _f3_multi_weight_sorted_kernel(
            s_val_a, s_unit_a, s_mult_a, w_mat, dm_u8, fin_a_scales
        )
        if has_flow:
            _wt_q_arr, qq_all, qq_mon, ures_q, ucont_q = _f3_multi_weight_sorted_kernel(
                s_val_q, s_unit_q, s_mult_q, w_mat, dm_u8, fin_q_scales
            )

        # Slice unit fields once per domain for F1/F2 dot products
        u_out_m = u_out_full[idx]
        u_zup_a_m = u_zero_up_a_full[idx]
        u_znet_m = u_zero_net_full[idx]
        om_cup_a_m = one_m_cup_a[idx]
        om_cdn_a_m = one_m_cdown_a[idx]
        u_net_a_m = u_net_a[idx]
        f_nev_a_m = frac_never_a[idx]

        if has_flow:
            u_zup_q_m = u_zero_up_q_full[idx]
            om_cup_q_m = one_m_cup_q[idx]
            om_cdn_q_m = one_m_cdown_q[idx]
            u_net_q_m = u_net_q[idx]
            f_nev_q_m = frac_never_q[idx]

        for wi, w_name in enumerate(w_names):
            w = w_mat[wi, idx]
            w_tot = float(np.sum(w))
            if w_tot <= 0.0:
                continue

            def add_rec(family: str, metric: str, ratio_basis: str, scale_val: float, val: float):
                records.append({
                    "domain_type": d_type,
                    "domain_name": d_name,
                    "family": family,
                    "metric": metric,
                    "ratio_basis": ratio_basis,
                    "snapping_mode": mode_tag,
                    "weight": w_name,
                    "scale_param": scale_val,
                    "value": float(val),
                    "total_weight": w_tot,
                })

            # Family 1
            add_rec("F1_polygon", "U_out", "topology", np.nan, np.dot(w, u_out_m) / w_tot)
            add_rec("F1_polygon", "U_zero_up", "area", np.nan, np.dot(w, u_zup_a_m) / w_tot)
            add_rec("F1_polygon", "U_up", "area", np.nan, np.dot(w, om_cup_a_m) / w_tot)
            add_rec("F1_polygon", "U_down", "area", np.nan, np.dot(w, om_cdn_a_m) / w_tot)
            add_rec("F1_polygon", "U_net_Eq4", "area", np.nan, np.dot(w, u_net_a_m) / w_tot)
            u_znet = float(np.dot(w, u_znet_m) / w_tot)
            add_rec("F1_polygon", "U_zero_net", "topology", np.nan, u_znet)

            if has_flow:
                add_rec("F1_polygon", "U_zero_up", "flow", np.nan, np.dot(w, u_zup_q_m) / w_tot)
                add_rec("F1_polygon", "U_up", "flow", np.nan, np.dot(w, om_cup_q_m) / w_tot)
                add_rec("F1_polygon", "U_down", "flow", np.nan, np.dot(w, om_cdn_q_m) / w_tot)
                add_rec("F1_polygon", "U_net_Eq4", "flow", np.nan, np.dot(w, u_net_q_m) / w_tot)

            # Family 2
            u_nev = float(np.dot(w, f_nev_a_m) / w_tot)
            add_rec("F2_mass_balance", "U_never", "area", np.nan, u_nev)
            add_rec("F2_mass_balance", "U_tail", "area", np.nan, u_nev - u_znet)

            # Family 3 (area scale)
            add_rec("F3_resolution", "A_tilde_q25_all", "area", np.nan, qa_all[wi, 0])
            add_rec("F3_resolution", "A_tilde_q50_all", "area", np.nan, qa_all[wi, 1])
            add_rec("F3_resolution", "A_tilde_q75_all", "area", np.nan, qa_all[wi, 2])
            add_rec("F3_resolution", "A_tilde_q25_monitored", "area", np.nan, qa_mon[wi, 0])
            add_rec("F3_resolution", "A_tilde_q50_monitored", "area", np.nan, qa_mon[wi, 1])
            add_rec("F3_resolution", "A_tilde_q75_monitored", "area", np.nan, qa_mon[wi, 2])

            for a0 in area_scale_grid:
                if np.isinf(a0):
                    add_rec("F3_resolution", "U_res", "area", float(a0), u_nev)
                    add_rec("F3_resolution", "U_cont", "area", float(a0), u_nev)
                else:
                    kp = a_scale_to_pos[float(a0)]
                    add_rec("F3_resolution", "U_res", "area", float(a0), ures_a[wi, kp])
                    add_rec("F3_resolution", "U_cont", "area", float(a0), ucont_a[wi, kp])

            # Family 3 (discharge scale)
            if has_flow:
                add_rec("F3_resolution", "Q_tilde_q25_all", "flow", np.nan, qq_all[wi, 0])
                add_rec("F3_resolution", "Q_tilde_q50_all", "flow", np.nan, qq_all[wi, 1])
                add_rec("F3_resolution", "Q_tilde_q75_all", "flow", np.nan, qq_all[wi, 2])
                add_rec("F3_resolution", "Q_tilde_q25_monitored", "flow", np.nan, qq_mon[wi, 0])
                add_rec("F3_resolution", "Q_tilde_q50_monitored", "flow", np.nan, qq_mon[wi, 1])
                add_rec("F3_resolution", "Q_tilde_q75_monitored", "flow", np.nan, qq_mon[wi, 2])

                u_nev_q = float(np.dot(w, f_nev_q_m) / w_tot)
                for q0 in discharge_scale_grid:
                    if np.isinf(q0):
                        add_rec("F3_resolution", "U_res", "flow", float(q0), u_nev_q)
                        add_rec("F3_resolution", "U_cont", "flow", float(q0), u_nev_q)
                    else:
                        kp = q_scale_to_pos[float(q0)]
                        add_rec("F3_resolution", "U_res", "flow", float(q0), ures_q[wi, kp])
                        add_rec("F3_resolution", "U_cont", "flow", float(q0), ucont_q[wi, kp])

    return (records, area_fields) if return_area_fields else records


def compute_2d_metrics_tree(
    order: np.ndarray,
    next_down: np.ndarray,
    local_area: np.ndarray,
    total_area_out: np.ndarray,
    has_gauge: np.ndarray,
    weights_dict: dict[str, np.ndarray],
    domain_mask: np.ndarray | None = None,
    effective_gauge_area: np.ndarray | None = None,
    use_subbasin_effective: bool = False,
    local_runoff: np.ndarray | None = None,
    total_runoff_out: np.ndarray | None = None,
    effective_gauge_runoff: np.ndarray | None = None,
    area_scale_grid: Sequence[float] = (
        50.0, 100.0, 250.0, 500.0, 1_000.0, 2_500.0, 5_000.0,
        10_000.0, 25_000.0, 50_000.0, 100_000.0, 500_000.0, float("inf"),
    ),
    discharge_scale_grid: Sequence[float] = (
        0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1_000.0, 5_000.0, float("inf"),
    ),
) -> list[dict]:
    """Compute all Family 1, 2, and 3 metrics for a single reporting domain on a tree hydrofabric."""
    recs = compute_2d_metrics_tree_multi_domain(
        order=order,
        next_down=next_down,
        local_area=local_area,
        total_area_out=total_area_out,
        has_gauge=has_gauge,
        weights_dict=weights_dict,
        domains=[("", "", domain_mask)],
        effective_gauge_area=effective_gauge_area,
        use_subbasin_effective=use_subbasin_effective,
        local_runoff=local_runoff,
        total_runoff_out=total_runoff_out,
        effective_gauge_runoff=effective_gauge_runoff,
        area_scale_grid=area_scale_grid,
        discharge_scale_grid=discharge_scale_grid,
    )
    for r in recs:
        r.pop("domain_type", None)
        r.pop("domain_name", None)
    return recs


def compute_2d_metrics_dag_multi_domain(
    edge_local_area: np.ndarray,
    edge_area_out: np.ndarray,
    edge_has_gauge: np.ndarray,
    g_up_in: np.ndarray,
    g_up_out: np.ndarray,
    edge_p_never_out: np.ndarray,
    edge_inv_a_down_out: np.ndarray,
    edge_inv_inter_a_out: np.ndarray,
    weights_dict: dict[str, np.ndarray | None],
    domains: Sequence[tuple[str, str, np.ndarray]],
    area_scale_grid: Sequence[float] = (
        50.0, 100.0, 250.0, 500.0, 1_000.0, 2_500.0, 5_000.0,
        10_000.0, 25_000.0, 50_000.0, 100_000.0, 500_000.0, float("inf"),
    ),
) -> list[dict]:
    """Compute Family 1, 2, and 3 metrics on a bifurcating DAG across multiple domains in ONE pass."""
    n = edge_local_area.shape[0]
    has_g = np.asarray(edge_has_gauge, dtype=np.uint8)
    a_out = edge_area_out
    safe_a_out = np.where(a_out > 0.0, a_out, 1.0)
    c_up = np.where(a_out > 0.0, np.clip(g_up_out / safe_a_out, 0.0, 1.0), has_g.astype(np.float64))
    c_down = np.where(has_g == 1, 1.0, np.clip(a_out * edge_inv_a_down_out, 0.0, 1.0))
    u_net = (1.0 - c_up) * (1.0 - c_down)
    p_never = np.where(has_g == 1, 0.0, np.clip(edge_p_never_out, 0.0, 1.0))

    edge_inter = np.maximum(a_out - g_up_in, 0.0)
    safe_ei = np.where(edge_inter > 0.0, edge_inter, 1.0)
    inv_ia = np.where(
        has_g == 1,
        np.where(edge_inter > 0.0, 1.0 / safe_ei, np.inf),
        edge_inv_inter_a_out,
    )
    safe_inv = np.where((inv_ia > 0.0) & np.isfinite(inv_ia), inv_ia, 1.0)
    ia_eff = np.where(np.isinf(inv_ia), 0.0, np.where(inv_ia > 0.0, 1.0 / safe_inv, np.inf))

    w_names = [k for k, v in weights_dict.items() if v is not None]
    n_w = len(w_names)
    if n_w == 0:
        return []
    w_mat = np.empty((n_w, n), dtype=np.float64)
    for wi, wn in enumerate(w_names):
        w_mat[wi] = np.asarray(weights_dict[wn], dtype=np.float64)

    order_ia = np.argsort(ia_eff, kind="mergesort")
    s_val = ia_eff[order_ia]
    s_unit = order_ia.astype(np.int32)
    s_mult = np.ones(n, dtype=np.float64)
    fin_scales = np.array(sorted(float(x) for x in area_scale_grid if np.isfinite(x)), dtype=np.float64)
    scale_to_pos = {float(x): i for i, x in enumerate(fin_scales)}

    u_out_full = (has_g == 0).astype(np.float64)
    u_zup_full = (c_up <= 0.0).astype(np.float64)
    om_cup = 1.0 - c_up
    om_cdn = 1.0 - c_down

    records: list[dict] = []
    for d_type, d_name, d_mask in domains:
        dm_u8 = np.ones(n, dtype=np.uint8) if d_mask is None else np.asarray(d_mask, dtype=np.uint8)
        idx = np.flatnonzero(dm_u8)
        if idx.size == 0:
            continue

        _wt, qa_all, qa_mon, ures_a, ucont_a = _f3_multi_weight_sorted_kernel(
            s_val, s_unit, s_mult, w_mat, dm_u8, fin_scales
        )
        u_out_m = u_out_full[idx]
        u_zup_m = u_zup_full[idx]
        om_cup_m = om_cup[idx]
        om_cdn_m = om_cdn[idx]
        u_net_m = u_net[idx]
        p_nev_m = p_never[idx]

        for wi, w_name in enumerate(w_names):
            w = w_mat[wi, idx]
            w_tot = float(np.sum(w))
            if w_tot <= 0.0:
                continue

            def add_rec(family: str, metric: str, scale_val: float, val: float):
                records.append({
                    "domain_type": d_type,
                    "domain_name": d_name,
                    "family": family,
                    "metric": metric,
                    "ratio_basis": "area_dag",
                    "snapping_mode": "dag_width_weighted",
                    "weight": w_name,
                    "scale_param": scale_val,
                    "value": float(val),
                    "total_weight": w_tot,
                })

            add_rec("F1_polygon", "U_out", np.nan, np.dot(w, u_out_m) / w_tot)
            add_rec("F1_polygon", "U_zero_up", np.nan, np.dot(w, u_zup_m) / w_tot)
            add_rec("F1_polygon", "U_up", np.nan, np.dot(w, om_cup_m) / w_tot)
            add_rec("F1_polygon", "U_down", np.nan, np.dot(w, om_cdn_m) / w_tot)
            add_rec("F1_polygon", "U_net_Eq4", np.nan, np.dot(w, u_net_m) / w_tot)
            u_nev = float(np.dot(w, p_nev_m) / w_tot)
            add_rec("F2_mass_balance", "U_never", np.nan, u_nev)

            add_rec("F3_resolution", "A_tilde_q25_all", np.nan, qa_all[wi, 0])
            add_rec("F3_resolution", "A_tilde_q50_all", np.nan, qa_all[wi, 1])
            add_rec("F3_resolution", "A_tilde_q75_all", np.nan, qa_all[wi, 2])
            add_rec("F3_resolution", "A_tilde_q25_monitored", np.nan, qa_mon[wi, 0])
            add_rec("F3_resolution", "A_tilde_q50_monitored", np.nan, qa_mon[wi, 1])
            add_rec("F3_resolution", "A_tilde_q75_monitored", np.nan, qa_mon[wi, 2])

            for a0 in area_scale_grid:
                if np.isinf(a0):
                    add_rec("F3_resolution", "U_res", float(a0), u_nev)
                    add_rec("F3_resolution", "U_cont", float(a0), u_nev)
                else:
                    kp = scale_to_pos[float(a0)]
                    add_rec("F3_resolution", "U_res", float(a0), ures_a[wi, kp])
                    add_rec("F3_resolution", "U_cont", float(a0), ucont_a[wi, kp])

    return records


def compute_2d_metrics_dag(
    edge_local_area: np.ndarray,
    edge_area_out: np.ndarray,
    edge_has_gauge: np.ndarray,
    g_up_in: np.ndarray,
    g_up_out: np.ndarray,
    edge_p_never_out: np.ndarray,
    edge_inv_a_down_out: np.ndarray,
    edge_inv_inter_a_out: np.ndarray,
    weights_dict: dict[str, np.ndarray],
    domain_mask: np.ndarray | None = None,
    area_scale_grid: Sequence[float] = (
        50.0, 100.0, 250.0, 500.0, 1_000.0, 2_500.0, 5_000.0,
        10_000.0, 25_000.0, 50_000.0, 100_000.0, 500_000.0, float("inf"),
    ),
) -> list[dict]:
    """Compute Family 1, 2, and 3 metrics on a bifurcating DAG (single domain)."""
    recs = compute_2d_metrics_dag_multi_domain(
        edge_local_area=edge_local_area,
        edge_area_out=edge_area_out,
        edge_has_gauge=edge_has_gauge,
        g_up_in=g_up_in,
        g_up_out=g_up_out,
        edge_p_never_out=edge_p_never_out,
        edge_inv_a_down_out=edge_inv_a_down_out,
        edge_inv_inter_a_out=edge_inv_inter_a_out,
        weights_dict=weights_dict,
        domains=[("", "", domain_mask)],
        area_scale_grid=area_scale_grid,
    )
    for r in recs:
        r.pop("domain_type", None)
        r.pop("domain_name", None)
    return recs
