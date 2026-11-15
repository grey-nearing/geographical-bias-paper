"""Family 4: Strictly Scale-Invariant 1D River-Network Ungaugedness Metrics.

Uses EXACT analytical closed-form integration along every continuous river
sub-segment [a_0, a_1] (and [q_0, q_1]) with exact boundary splitting at every
drainage-area or discharge bin edge and every gauge fractional position t in [0, 1].

Mathematical guarantee:
  Subdividing any river reach into M >= 1 sub-reaches changes EVERY metric in
  this module by 0.0 (up to IEEE-754 floating-point machine epsilon ~1e-14).

Metrics computed for both causal u_up(x) = 1 - c_up(x), downstream u_down(x) = 1 - c_down(x),
and zero-range Top-Kriging / BLUP unexplained variance
u_net(x) = [1 - c_up(x)][1 - c_down(x)] = 1 - R^2_net(x):
  12. River-length weighted stratified by physical drainage area A(x)
      (binned [0-10, 10-100, ..., >1M] AND cumulative A(x) >= A_min)
  13. River-length weighted stratified by physical mean discharge Q(x)
      (binned AND cumulative Q(x) >= Q_min; skipped on GRIT)
  14. Cutoff-free drainage-area-kilometer weighted: int u(x) A(x) ds / int A(x) ds
  15. Cutoff-free discharge-kilometer weighted: int u(x) Q(x) ds / int Q(x) ds
      and channel-volume weighted (skipped on GRIT)
  16. Flood-exposure-weighted river ungaugedness: int u(x) E(x) ds / int E(x) ds
  17. Exact along-channel hydrologic distance to nearest gauge (d_up, d_down, d_any)
      via exact piecewise-linear continuous CDF accumulation (medians, IQRs, means,
      threshold fractions <= 10..1000 km) stratified by A >= A_min, A*ds, Q*ds, exposure.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
from numba import njit

from global_ungaugedness.graph import (
    propagate_tree_channel_distances,
    propagate_tree_gauges,
)


# ---------------------------------------------------------------------------
# Exact analytical 1D sub-segment integration primitives
# ---------------------------------------------------------------------------

@njit(cache=True)
def _exact_linear_integrals(
    ell: float,
    x0: float,
    x1: float,
    g_up: float,
    inv_x_down: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Exact closed-form integrals of u_up, u_down, u_net on a gauge-free sub-segment.

    Assumes x(s) = x0 + (x1 - x0) * (s / ell) for s in [0, ell], with
    0 <= g_up <= x0 <= x1 and 0 <= inv_x_down <= 1/x1 (or 0 if never gauged).

    Note: inv_x_down = 1 / X_down on a tree, or E[1 / X_down] on a bifurcating DAG!

    Returns:
      (int_u_up_ds, int_u_down_ds, int_u_net_ds,
       int_x_ds, int_u_up_x_ds, int_u_down_x_ds, int_u_net_x_ds,
       int_never_ds)
    """
    if ell <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Enforce physical bounds for numerical safety
    if x0 < 0.0:
        x0 = 0.0
    if x1 < x0:
        x1 = x0
    if g_up < 0.0:
        g_up = 0.0
    if g_up > x0:
        g_up = x0
    if inv_x_down < 0.0:
        inv_x_down = 0.0
    if x1 > 0.0 and inv_x_down * x1 > 1.0:
        inv_x_down = 1.0 / x1

    mean_x = 0.5 * (x0 + x1)
    int_x_ds = ell * mean_x
    dx = x1 - x0

    # Integral of (g_up / x) ds
    if g_up <= 0.0:
        int_cup_ds = 0.0
    elif dx <= 1e-12 * (x1 + 1.0):
        # Constant x along sub-segment
        int_cup_ds = ell * (g_up / x0) if x0 > 0.0 else 0.0
    else:
        # Since g_up <= x0 and g_up > 0, x0 > 0 is guaranteed
        int_cup_ds = ell * g_up * np.log(x1 / x0) / dx

    # Integral of (x * inv_x_down) ds
    int_cdown_ds = ell * mean_x * inv_x_down

    # Integral of (g_up * inv_x_down) ds = ell * g_up * inv_x_down
    int_cup_cdown_ds = ell * g_up * inv_x_down

    int_u_up_ds = ell - int_cup_ds
    int_u_down_ds = ell - int_cdown_ds
    int_u_net_ds = ell - int_cup_ds - int_cdown_ds + int_cup_cdown_ds

    # Clamp tiny floating-point roundoff
    if int_u_up_ds < 0.0:
        int_u_up_ds = 0.0
    if int_u_down_ds < 0.0:
        int_u_down_ds = 0.0
    if int_u_net_ds < 0.0:
        int_u_net_ds = 0.0

    # Exact x*ds weighted integrals (pure polynomials -- zero singularity even at x0 == 0!)
    # int (1 - g_up/x) * x ds = int (x - g_up) ds = ell * (mean_x - g_up)
    int_u_up_x_ds = ell * (mean_x - g_up)
    # int (1 - x * inv_x_down) * x ds = ell * (mean_x - inv_x_down * (x0^2 + x0*x1 + x1^2)/3)
    mean_x2 = (x0 * x0 + x0 * x1 + x1 * x1) / 3.0
    int_u_down_x_ds = ell * (mean_x - inv_x_down * mean_x2)
    # int (1 - g_up/x)(1 - x*inv_x_down) * x ds
    #   = int [(1 + g_up*inv_x_down)*x - g_up - inv_x_down*x^2] ds
    int_u_net_x_ds = ell * (
        (1.0 + g_up * inv_x_down) * mean_x - g_up - inv_x_down * mean_x2
    )
    if int_u_up_x_ds < 0.0:
        int_u_up_x_ds = 0.0
    if int_u_down_x_ds < 0.0:
        int_u_down_x_ds = 0.0
    if int_u_net_x_ds < 0.0:
        int_u_net_x_ds = 0.0

    int_never_ds = ell if inv_x_down <= 0.0 else 0.0

    return (
        int_u_up_ds,
        int_u_down_ds,
        int_u_net_ds,
        int_x_ds,
        int_u_up_x_ds,
        int_u_down_x_ds,
        int_u_net_x_ds,
        int_never_ds,
    )


@njit(cache=True)
def _deposit_linear_dist_hist(
    hist: np.ndarray,
    bin_edges: np.ndarray,
    d0: float,
    d1: float,
    weight: float,
) -> float:
    """Deposit a uniform distribution on [min(d0,d1), max(d0,d1)] with total mass `weight`
    into histogram `hist` (len = len(bin_edges) - 1, plus last bin for >= bin_edges[-1],
    and final overflow bin for inf).

    Returns total finite mass deposited (0 if infinite).
    """
    n_bins = bin_edges.shape[0] - 1  # finite interior bins: 0..n_bins-1
    # hist has shape n_bins + 2:
    #   [0 .. n_bins-1] : finite intervals [edge[b], edge[b+1])
    #   [n_bins]        : finite >= edge[-1]
    #   [n_bins + 1]    : np.inf (never gauged in that direction)
    if weight <= 0.0:
        return 0.0
    if np.isinf(d0) or np.isinf(d1):
        hist[n_bins + 1] += weight
        return 0.0

    lo = d0 if d0 <= d1 else d1
    hi = d1 if d0 <= d1 else d0
    if lo < 0.0:
        lo = 0.0
    if hi < lo:
        hi = lo

    span = hi - lo
    if span <= 1e-12:
        # Point mass at lo
        if lo >= bin_edges[n_bins]:
            hist[n_bins] += weight
        else:
            # Binary search
            l_i = 0
            r_i = n_bins - 1
            while l_i < r_i:
                mid = (l_i + r_i) >> 1
                if bin_edges[mid + 1] <= lo:
                    l_i = mid + 1
                else:
                    r_i = mid
            hist[l_i] += weight
        return weight

    # Find starting bin for lo
    if lo >= bin_edges[n_bins]:
        hist[n_bins] += weight
        return weight

    b = 0
    r_i = n_bins - 1
    while b < r_i:
        mid = (b + r_i) >> 1
        if bin_edges[mid + 1] <= lo:
            b = mid + 1
        else:
            r_i = mid

    cur = lo
    while b < n_bins and cur < hi:
        right = bin_edges[b + 1]
        seg_hi = hi if hi < right else right
        if seg_hi > cur:
            hist[b] += weight * ((seg_hi - cur) / span)
            cur = seg_hi
        b += 1

    if cur < hi:
        hist[n_bins] += weight * ((hi - cur) / span)

    return weight


@njit(cache=True)
def _deposit_piecewise_d_any(
    hist: np.ndarray,
    bin_edges: np.ndarray,
    d_up_start: float,
    d_down_end: float,
    ell: float,
    weight: float,
) -> tuple[float, float]:
    """Deposit exact distribution of d_any(s) = min(d_up_start + s, d_down_end + ell - s)
    for s in [0, ell] with uniform weight along length ell.

    Returns (weighted_sum_finite_d_any, total_finite_weight).
    """
    if weight <= 0.0 or ell <= 0.0:
        return 0.0, 0.0

    up_inf = np.isinf(d_up_start)
    dn_inf = np.isinf(d_down_end)
    if up_inf and dn_inf:
        _deposit_linear_dist_hist(hist, bin_edges, np.inf, np.inf, weight)
        return 0.0, 0.0
    if up_inf:
        _deposit_linear_dist_hist(hist, bin_edges, d_down_end, d_down_end + ell, weight)
        return weight * (d_down_end + 0.5 * ell), weight
    if dn_inf:
        _deposit_linear_dist_hist(hist, bin_edges, d_up_start, d_up_start + ell, weight)
        return weight * (d_up_start + 0.5 * ell), weight

    # Both finite: breakpoint s* where d_up_start + s* = d_down_end + ell - s*
    s_star = 0.5 * (d_down_end + ell - d_up_start)
    if s_star <= 0.0:
        # Downstream is always closer
        _deposit_linear_dist_hist(hist, bin_edges, d_down_end, d_down_end + ell, weight)
        return weight * (d_down_end + 0.5 * ell), weight
    if s_star >= ell:
        # Upstream is always closer
        _deposit_linear_dist_hist(hist, bin_edges, d_up_start, d_up_start + ell, weight)
        return weight * (d_up_start + 0.5 * ell), weight

    # Splits into [0, s_star] (d_up increases from d_up_start to peak)
    # and [s_star, ell] (d_down decreases from peak to d_down_end)
    d_peak = d_up_start + s_star
    w1 = weight * (s_star / ell)
    w2 = weight - w1
    _deposit_linear_dist_hist(hist, bin_edges, d_up_start, d_peak, w1)
    _deposit_linear_dist_hist(hist, bin_edges, d_down_end, d_peak, w2)
    mean_d_sum = w1 * 0.5 * (d_up_start + d_peak) + w2 * 0.5 * (d_down_end + d_peak)
    return mean_d_sum, weight


@njit(cache=True)
def _integrate_1d_network_kernel(
    length_km: np.ndarray,
    x_in: np.ndarray,
    x_out: np.ndarray,
    g_up_in: np.ndarray,
    inv_x_down_out: np.ndarray,
    phys_in: np.ndarray,
    phys_out: np.ndarray,
    d_up_in: np.ndarray,
    d_down_out: np.ndarray,
    gauge_ptr: np.ndarray,
    gauge_fracs: np.ndarray,
    extra_weight: np.ndarray,
    vol_weight: np.ndarray,
    domain_mask: np.ndarray,
    bin_bounds: np.ndarray,
    dist_bin_edges: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Exact Numba integration kernel over all N reaches.

    Args:
        length_km: Reach lengths (km).
        x_in, x_out: Monotonic ratio-basis coordinate along each reach (area km^2 or
            mass-conserving discharge m^3/s) for computing c_up(x), c_down(x).
        g_up_in: Gauged upstream x entering reach inlet.
        inv_x_down_out: 1 / X_down_gauge leaving reach outlet (or E[1/X_down] for DAG;
            0 if never gauged downstream).
        phys_in, phys_out: Physical binning & weighting quantity along reach (area km^2
            for area basis, or actual modeled discharge Q_phys for discharge basis).
        d_up_in, d_down_out: Along-channel distances (km) at reach inlet / outlet.
        gauge_ptr, gauge_fracs: CSR list of sorted gauge fractional positions t in [0, 1]
            snapped onto reach e.
        extra_weight: Flood exposure (or other per-reach weight E_e) distributed uniformly
            along reach length.
        vol_weight: Channel volume per reach (m^3), distributed uniformly along reach length.
        domain_mask: uint8 mask (1 if reach is inside reporting domain).
        bin_bounds: Sorted boundary array [b_0, b_1, ..., b_K] defining K bins
            [b_k, b_{k+1}) on phys(s) (with b_0=0, b_K=inf).
        dist_bin_edges: Sorted edges for continuous channel-distance histograms.

    Returns:
        bin_acc: shape (K, 9) -> [len, u_up_len, u_down_len, u_net_len, never_len,
                                  phys_ds, u_up_pds, u_down_pds, u_net_pds]
        glob_acc: shape (13,) -> [
            total_len, u_up_len, u_down_len, u_net_len,
            total_xds, u_up_xds, u_down_xds, u_net_xds,
            total_exp, u_up_exp, u_net_exp,
            total_vol, u_net_vol,
        ]
        dist_hists_len: shape (3, n_dbins+2) for [d_up, d_down, d_any] length-weighted (all reaches)
        dist_hists_gt100: shape (3, n_dbins+2) length-weighted for phys >= bin_bounds[2] (e.g. A>=100 or Q>=1)
        dist_hists_gt1000: shape (3, n_dbins+2) length-weighted for phys >= bin_bounds[3] (e.g. A>=1k or Q>=10)
        dist_hists_xds: shape (3, n_dbins+2) weighted by x*ds (cutoff-free!)
        dist_hists_exp: shape (3, n_dbins+2) weighted by exposure
        dist_means: shape (5, 2) -> [sum_d_any_finite, sum_w_finite] for the 5 weightings above
    """
    n = length_km.shape[0]
    k_bins = bin_bounds.shape[0] - 1
    n_db = dist_bin_edges.shape[0] + 1  # (n_edges - 1) + 2

    bin_acc = np.zeros((k_bins, 9), dtype=np.float64)
    glob_acc = np.zeros(13, dtype=np.float64)

    dist_h_len = np.zeros((3, n_db), dtype=np.float64)
    dist_h_b2 = np.zeros((3, n_db), dtype=np.float64)
    dist_h_b3 = np.zeros((3, n_db), dtype=np.float64)
    dist_h_xds = np.zeros((3, n_db), dtype=np.float64)
    dist_h_exp = np.zeros((3, n_db), dtype=np.float64)
    dist_means = np.zeros((5, 2), dtype=np.float64)

    thresh_b2 = bin_bounds[2] if k_bins >= 3 else 0.0
    thresh_b3 = bin_bounds[3] if k_bins >= 4 else 0.0

    for e in range(n):
        if domain_mask[e] == 0:
            continue
        L = length_km[e]
        if L <= 0.0:
            continue

        xin = x_in[e]
        xout = x_out[e]
        if xout < xin:
            xout = xin
        dx_tot = xout - xin

        pin = phys_in[e]
        pout = phys_out[e]
        dp_tot = pout - pin

        exp_e = extra_weight[e]
        vol_e = vol_weight[e]

        gp0 = gauge_ptr[e]
        gp1 = gauge_ptr[e + 1]
        n_g = gp1 - gp0

        # Iterate over gauge-free sub-segments [ta, tb]
        for seg_idx in range(n_g + 1):
            ta = 0.0 if seg_idx == 0 else gauge_fracs[gp0 + seg_idx - 1]
            tb = 1.0 if seg_idx == n_g else gauge_fracs[gp0 + seg_idx]
            if tb <= ta:
                continue

            sub_frac = tb - ta
            ell = sub_frac * L
            xa = xin + ta * dx_tot
            xb = xin + tb * dx_tot
            pa = pin + ta * dp_tot
            pb = pin + tb * dp_tot

            # Upstream gauged area entering this gauge-free sub-segment
            g_up_sub = g_up_in[e] if seg_idx == 0 else xa
            # Downstream gauge inverse area exiting this sub-segment
            if seg_idx < n_g:
                inv_xd_sub = 1.0 / xb if xb > 0.0 else 0.0
            else:
                inv_xd_sub = inv_x_down_out[e]

            # Exact channel distances at boundaries of this gauge-free sub-segment
            d_up_sub = d_up_in[e] if seg_idx == 0 else 0.0
            d_dn_sub = 0.0 if seg_idx < n_g else d_down_out[e]

            # Global exact integrals on this sub-segment
            (
                i_uup, i_udn, i_unet, i_xds,
                i_uup_x, i_udn_x, i_unet_x, i_nev
            ) = _exact_linear_integrals(ell, xa, xb, g_up_sub, inv_xd_sub)

            glob_acc[0] += ell
            glob_acc[1] += i_uup
            glob_acc[2] += i_udn
            glob_acc[3] += i_unet

            # Physical weighting x*ds (for area basis phys == x; for Q basis use phys_mean if phys != x)
            if abs(pa - xa) <= 1e-9 * (abs(xa) + 1.0) and abs(pb - xb) <= 1e-9 * (abs(xb) + 1.0):
                p_xds = i_xds
                p_uup_x = i_uup_x
                p_udn_x = i_udn_x
                p_unet_x = i_unet_x
            else:
                # Modeled physical discharge differs from monotonic runoff-routed Q (e.g. transmission loss)
                # Use exact average physical discharge weight on sub-segment
                p_mean = 0.5 * (pa + pb)
                if p_mean < 0.0:
                    p_mean = 0.0
                p_xds = ell * p_mean
                p_uup_x = i_uup * p_mean
                p_udn_x = i_udn * p_mean
                p_unet_x = i_unet * p_mean

            glob_acc[4] += p_xds
            glob_acc[5] += p_uup_x
            glob_acc[6] += p_udn_x
            glob_acc[7] += p_unet_x

            sub_exp = exp_e * sub_frac
            sub_vol = vol_e * sub_frac
            if sub_exp > 0.0:
                glob_acc[8] += sub_exp
                glob_acc[9] += sub_exp * (i_uup / ell)
                glob_acc[10] += sub_exp * (i_unet / ell)
            if sub_vol > 0.0:
                glob_acc[11] += sub_vol
                glob_acc[12] += sub_vol * (i_unet / ell)

            # Channel distances (all reaches, x*ds weighted, exposure weighted)
            _deposit_linear_dist_hist(dist_h_len[0], dist_bin_edges, d_up_sub, d_up_sub + ell, ell)
            _deposit_linear_dist_hist(dist_h_len[1], dist_bin_edges, d_dn_sub, d_dn_sub + ell, ell)
            ds_m, dw_m = _deposit_piecewise_d_any(dist_h_len[2], dist_bin_edges, d_up_sub, d_dn_sub, ell, ell)
            dist_means[0, 0] += ds_m
            dist_means[0, 1] += dw_m

            if p_xds > 0.0:
                _deposit_linear_dist_hist(dist_h_xds[0], dist_bin_edges, d_up_sub, d_up_sub + ell, p_xds)
                _deposit_linear_dist_hist(dist_h_xds[1], dist_bin_edges, d_dn_sub, d_dn_sub + ell, p_xds)
                ds_m, dw_m = _deposit_piecewise_d_any(dist_h_xds[2], dist_bin_edges, d_up_sub, d_dn_sub, ell, p_xds)
                dist_means[3, 0] += ds_m
                dist_means[3, 1] += dw_m

            if sub_exp > 0.0:
                _deposit_linear_dist_hist(dist_h_exp[0], dist_bin_edges, d_up_sub, d_up_sub + ell, sub_exp)
                _deposit_linear_dist_hist(dist_h_exp[1], dist_bin_edges, d_dn_sub, d_dn_sub + ell, sub_exp)
                ds_m, dw_m = _deposit_piecewise_d_any(dist_h_exp[2], dist_bin_edges, d_up_sub, d_dn_sub, ell, sub_exp)
                dist_means[4, 0] += ds_m
                dist_means[4, 1] += dw_m

            # Exact boundary splitting across phys bins [b_k, b_{k+1})
            p_lo = pa if pa <= pb else pb
            p_hi = pb if pa <= pb else pa
            dp_sub = pb - pa

            for kb in range(k_bins):
                blo = bin_bounds[kb]
                bhi = bin_bounds[kb + 1]
                if p_hi < blo or (p_lo >= bhi and not (p_lo == bhi and kb == k_bins - 1)):
                    continue
                if p_lo == p_hi:
                    # Constant phys on sub-segment: check strict half-open membership
                    if not (p_lo >= blo and p_lo < bhi):
                        continue
                    f0 = 0.0
                    f1 = 1.0
                else:
                    if dp_sub > 0.0:
                        f0 = (blo - pa) / dp_sub
                        f1 = (bhi - pa) / dp_sub
                    else:
                        f0 = (bhi - pa) / dp_sub
                        f1 = (blo - pa) / dp_sub
                    if f0 < 0.0:
                        f0 = 0.0
                    if f1 > 1.0:
                        f1 = 1.0
                    if f1 <= f0:
                        continue

                b_ell = (f1 - f0) * ell
                b_xa = xa + f0 * (xb - xa)
                b_xb = xa + f1 * (xb - xa)
                b_pa = pa + f0 * dp_sub
                b_pb = pa + f1 * dp_sub

                (
                    bi_uup, bi_udn, bi_unet, bi_xds,
                    bi_uup_x, bi_udn_x, bi_unet_x, bi_nev
                ) = _exact_linear_integrals(b_ell, b_xa, b_xb, g_up_sub, inv_xd_sub)

                bin_acc[kb, 0] += b_ell
                bin_acc[kb, 1] += bi_uup
                bin_acc[kb, 2] += bi_udn
                bin_acc[kb, 3] += bi_unet
                bin_acc[kb, 4] += bi_nev

                if abs(b_pa - b_xa) <= 1e-9 * (abs(b_xa) + 1.0) and abs(b_pb - b_xb) <= 1e-9 * (abs(b_xb) + 1.0):
                    bin_acc[kb, 5] += bi_xds
                    bin_acc[kb, 6] += bi_uup_x
                    bin_acc[kb, 7] += bi_udn_x
                    bin_acc[kb, 8] += bi_unet_x
                else:
                    bp_m = 0.5 * (b_pa + b_pb)
                    if bp_m < 0.0:
                        bp_m = 0.0
                    bin_acc[kb, 5] += b_ell * bp_m
                    bin_acc[kb, 6] += bi_uup * bp_m
                    bin_acc[kb, 7] += bi_udn * bp_m
                    bin_acc[kb, 8] += bi_unet * bp_m

                # Stratified channel distances for phys >= thresh_b2 and phys >= thresh_b3
                if blo >= thresh_b2 and b_ell > 0.0:
                    sub_dup = d_up_sub + f0 * ell
                    sub_ddn = d_dn_sub + (1.0 - f1) * ell
                    _deposit_linear_dist_hist(dist_h_b2[0], dist_bin_edges, sub_dup, sub_dup + b_ell, b_ell)
                    _deposit_linear_dist_hist(dist_h_b2[1], dist_bin_edges, sub_ddn, sub_ddn + b_ell, b_ell)
                    ds_m, dw_m = _deposit_piecewise_d_any(dist_h_b2[2], dist_bin_edges, sub_dup, sub_ddn, b_ell, b_ell)
                    dist_means[1, 0] += ds_m
                    dist_means[1, 1] += dw_m

                if blo >= thresh_b3 and b_ell > 0.0:
                    sub_dup = d_up_sub + f0 * ell
                    sub_ddn = d_dn_sub + (1.0 - f1) * ell
                    _deposit_linear_dist_hist(dist_h_b3[0], dist_bin_edges, sub_dup, sub_dup + b_ell, b_ell)
                    _deposit_linear_dist_hist(dist_h_b3[1], dist_bin_edges, sub_ddn, sub_ddn + b_ell, b_ell)
                    ds_m, dw_m = _deposit_piecewise_d_any(dist_h_b3[2], dist_bin_edges, sub_dup, sub_ddn, b_ell, b_ell)
                    dist_means[2, 0] += ds_m
                    dist_means[2, 1] += dw_m

    return (
        bin_acc,
        glob_acc,
        dist_h_len,
        dist_h_b2,
        dist_h_b3,
        dist_h_xds,
        dist_h_exp,
        dist_means,
    )


def _hist_quantiles_and_cdf(
    hist: np.ndarray,
    bin_edges: np.ndarray,
    thresholds: Sequence[float],
    probs: Sequence[float] = (0.25, 0.50, 0.75),
) -> dict[str, float]:
    """Extract continuous interpolated quantiles (all + conditional finite) and CDF thresholds."""
    n_bins = bin_edges.shape[0] - 1
    total = float(np.sum(hist))
    if total <= 0.0:
        out = {f"q{int(p*100)}_all": float("nan") for p in probs}
        out.update({f"q{int(p*100)}_finite": float("nan") for p in probs})
        out["frac_infinite"] = float("nan")
        for t in thresholds:
            out[f"frac_le_{t:g}km"] = float("nan")
        return out

    inf_mass = float(hist[n_bins + 1])
    fin_mass = total - inf_mass
    out: dict[str, float] = {"frac_infinite": inf_mass / total}

    # Cumulative mass up to each interior bin edge
    cum = np.zeros(n_bins + 1, dtype=np.float64)
    cum[1:] = np.cumsum(hist[:n_bins])

    def _interp_q(target_mass: float, denom: float) -> float:
        if denom <= 0.0:
            return float("nan")
        if target_mass > fin_mass:
            return float("inf")
        if target_mass >= cum[n_bins]:
            return float(bin_edges[n_bins])
        idx = int(np.searchsorted(cum, target_mass, side="right")) - 1
        if idx < 0:
            idx = 0
        if idx >= n_bins:
            return float(bin_edges[n_bins])
        b_mass = hist[idx]
        if b_mass <= 0.0:
            return float(bin_edges[idx])
        frac = (target_mass - cum[idx]) / b_mass
        return float(bin_edges[idx] + frac * (bin_edges[idx + 1] - bin_edges[idx]))

    for p in probs:
        out[f"q{int(p*100)}_all"] = _interp_q(p * total, total)
        out[f"q{int(p*100)}_finite"] = _interp_q(p * fin_mass, fin_mass) if fin_mass > 0.0 else float("nan")

    for t in thresholds:
        if t <= bin_edges[0]:
            m_le = 0.0
        elif t >= bin_edges[n_bins]:
            m_le = float(cum[n_bins])
        else:
            idx = int(np.searchsorted(bin_edges, t, side="right")) - 1
            frac = (t - bin_edges[idx]) / (bin_edges[idx + 1] - bin_edges[idx])
            m_le = float(cum[idx] + frac * hist[idx])
        out[f"frac_le_{t:g}km"] = m_le / total

    return out


def build_gauge_csr(
    n_reaches: int,
    gauge_reach_idx: np.ndarray,
    gauge_frac_pos: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack snapped gauges into sorted CSR arrays per reach.

    Args:
        n_reaches: Total number of river reaches.
        gauge_reach_idx: int32 array of reach indices for gauges in the active tier.
        gauge_frac_pos: Optional float64 array of fractional positions t in [0, 1]
            along each gauge's snapped reach (defaults to 1.0 = reach outlet).

    Returns:
        has_gauge (uint8[N]), gauge_ptr (int32[N+1]), gauge_fracs (float64[M]),
        frac_min (float64[N]), frac_max (float64[N]).
    """
    valid = (gauge_reach_idx >= 0) & (gauge_reach_idx < n_reaches)
    r_idx = np.asarray(gauge_reach_idx[valid], dtype=np.int32)
    if gauge_frac_pos is None:
        fracs = np.ones(r_idx.shape[0], dtype=np.float64)
    else:
        fracs = np.clip(np.asarray(gauge_frac_pos[valid], dtype=np.float64), 0.0, 1.0)

    has_gauge = np.zeros(n_reaches, dtype=np.uint8)
    frac_min = np.ones(n_reaches, dtype=np.float64)
    frac_max = np.ones(n_reaches, dtype=np.float64)

    if r_idx.size == 0:
        return (
            has_gauge,
            np.zeros(n_reaches + 1, dtype=np.int32),
            np.empty(0, dtype=np.float64),
            frac_min,
            frac_max,
        )

    # Sort by (reach_idx, frac_pos)
    lex = np.lexsort((fracs, r_idx))
    r_sorted = r_idx[lex]
    f_sorted = fracs[lex]

    counts = np.bincount(r_sorted, minlength=n_reaches).astype(np.int32)
    ptr = np.zeros(n_reaches + 1, dtype=np.int32)
    np.cumsum(counts, out=ptr[1:])

    nz = np.flatnonzero(counts > 0)
    has_gauge[nz] = 1
    frac_min[nz] = f_sorted[ptr[nz]]
    frac_max[nz] = f_sorted[ptr[nz + 1] - 1]

    return has_gauge, ptr, f_sorted, frac_min, frac_max


def compute_1d_metrics_tree_multi_domain(
    order: np.ndarray,
    next_down: np.ndarray,
    length_km: np.ndarray,
    area_in_km2: np.ndarray,
    area_out_km2: np.ndarray,
    gauge_reach_idx: np.ndarray,
    domains: Sequence[tuple[str, str, np.ndarray | None]],
    gauge_frac_pos: np.ndarray | None = None,
    q_cons_in_m3s: np.ndarray | None = None,
    q_cons_out_m3s: np.ndarray | None = None,
    q_phys_in_m3s: np.ndarray | None = None,
    q_phys_out_m3s: np.ndarray | None = None,
    exposure_eae: np.ndarray | None = None,
    channel_volume_m3: np.ndarray | None = None,
    area_bins: Sequence[tuple[float, float, str]] = (
        (0.0, 10.0, "0-10"),
        (10.0, 100.0, "10-100"),
        (100.0, 1_000.0, "100-1k"),
        (1_000.0, 10_000.0, "1k-10k"),
        (10_000.0, 100_000.0, "10k-100k"),
        (100_000.0, 1_000_000.0, "100k-1M"),
        (1_000_000.0, float("inf"), ">1M"),
    ),
    q_bins: Sequence[tuple[float, float, str]] = (
        (0.0, 0.1, "<0.1"),
        (0.1, 1.0, "0.1-1"),
        (1.0, 10.0, "1-10"),
        (10.0, 100.0, "10-100"),
        (100.0, 1_000.0, "100-1k"),
        (1_000.0, 10_000.0, "1k-10k"),
        (10_000.0, float("inf"), ">10k"),
    ),
    dist_thresholds_km: Sequence[float] = (10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1_000.0),
) -> list[dict]:
    """Compute all Family 4 1D river-network metrics across multiple domains in ONE graph sweep."""
    n = length_km.shape[0]
    has_g, g_ptr, g_fracs, f_min, f_max = build_gauge_csr(n, gauge_reach_idx, gauge_frac_pos)

    exp_arr = (
        np.zeros(n, dtype=np.float64)
        if exposure_eae is None
        else np.asarray(exposure_eae, dtype=np.float64)
    )
    vol_arr = (
        np.zeros(n, dtype=np.float64)
        if channel_volume_m3 is None
        else np.asarray(channel_volume_m3, dtype=np.float64)
    )

    # Fine 400-bin log-spaced distance grid from 0.05 km to 15,000 km (plus 0 and inf)
    dist_edges = np.concatenate([[0.0], np.geomspace(0.05, 15_000.0, 400)])

    # Along-channel distances — computed ONCE globally!
    d_up_in, d_down_out = propagate_tree_channel_distances(
        order, next_down, length_km, has_g, f_min, f_max
    )

    # Propagate gauged area along the tree ONCE globally!
    local_a = np.maximum(area_out_km2 - area_in_km2, 0.0)
    eff_a_at_tmax = np.where(has_g == 1, area_in_km2 + f_max * local_a, -1.0)
    g_up_in_a, _g_out_a, dg_node_a, _inter_a, _net_g = propagate_tree_gauges(
        order, next_down, local_a, area_out_km2, has_g, eff_a_at_tmax, True
    )

    inv_a_down_out = np.zeros(n, dtype=np.float64)
    has_d = next_down >= 0
    d_nodes = next_down[has_d]
    strict_dg = dg_node_a[d_nodes]
    has_sdg = strict_dg >= 0
    if np.any(has_sdg):
        valid_e = np.flatnonzero(has_d)[has_sdg]
        sdg_idx = strict_dg[has_sdg]
        a_at_sdg = area_in_km2[sdg_idx] + f_min[sdg_idx] * local_a[sdg_idx]
        safe_a = np.where(a_at_sdg > 0.0, a_at_sdg, 1.0)
        inv_a_down_out[valid_e] = np.where(a_at_sdg > 0.0, 1.0 / safe_a, 0.0)

    a_bounds = np.array([b[0] for b in area_bins] + [area_bins[-1][1]], dtype=np.float64)
    k_a = len(area_bins)

    # Optional flow propagation — computed ONCE globally!
    has_flow = q_cons_in_m3s is not None and q_cons_out_m3s is not None
    if has_flow:
        q_p_in = q_cons_in_m3s if q_phys_in_m3s is None else q_phys_in_m3s
        q_p_out = q_cons_out_m3s if q_phys_out_m3s is None else q_phys_out_m3s

        local_q = np.maximum(q_cons_out_m3s - q_cons_in_m3s, 0.0)
        eff_q_at_tmax = np.where(has_g == 1, q_cons_in_m3s + f_max * local_q, -1.0)
        g_up_in_q, _gq_out, dg_node_q, _iq, _nq = propagate_tree_gauges(
            order, next_down, local_q, q_cons_out_m3s, has_g, eff_q_at_tmax, True
        )

        inv_q_down_out = np.zeros(n, dtype=np.float64)
        strict_dg_q = dg_node_q[d_nodes]
        has_sdgq = strict_dg_q >= 0
        if np.any(has_sdgq):
            valid_eq = np.flatnonzero(has_d)[has_sdgq]
            sdgq_idx = strict_dg_q[has_sdgq]
            q_at_sdg = q_cons_in_m3s[sdgq_idx] + f_min[sdgq_idx] * local_q[sdgq_idx]
            safe_q = np.where(q_at_sdg > 0.0, q_at_sdg, 1.0)
            inv_q_down_out[valid_eq] = np.where(q_at_sdg > 0.0, 1.0 / safe_q, 0.0)

        q_bounds = np.array([b[0] for b in q_bins] + [q_bins[-1][1]], dtype=np.float64)

    records: list[dict] = []

    for d_type, d_name, d_mask in domains:
        dm_u8 = np.ones(n, dtype=np.uint8) if d_mask is None else np.asarray(d_mask, dtype=np.uint8)

        def add_1d(metric: str, ratio_basis: str, weight: str, stratum: str, value: float, total_w: float):
            records.append({
                "domain_type": d_type,
                "domain_name": d_name,
                "family": "F4_river_1d",
                "metric": metric,
                "ratio_basis": ratio_basis,
                "snapping_mode": "continuous_reach",
                "weight": weight,
                "stratum": stratum,
                "value": float(value),
                "total_weight": float(total_w),
            })

        (
            a_bin_acc,
            a_glob_acc,
            dh_len,
            dh_gt100,
            dh_gt1k,
            dh_ads,
            dh_exp,
            d_means,
        ) = _integrate_1d_network_kernel(
            length_km,
            area_in_km2,
            area_out_km2,
            g_up_in_a,
            inv_a_down_out,
            area_in_km2,
            area_out_km2,
            d_up_in,
            d_down_out,
            g_ptr,
            g_fracs,
            exp_arr,
            vol_arr,
            dm_u8,
            a_bounds,
            dist_edges,
        )

        tot_len = a_glob_acc[0]
        if tot_len > 0.0:
            add_1d("U_up_1d", "area", "river_length", "all", a_glob_acc[1] / tot_len, tot_len)
            add_1d("U_down_1d", "area", "river_length", "all", a_glob_acc[2] / tot_len, tot_len)
            add_1d("U_net_1d", "area", "river_length", "all", a_glob_acc[3] / tot_len, tot_len)

        tot_ads = a_glob_acc[4]
        if tot_ads > 0.0:
            add_1d("U_up_1d", "area", "A_ds", "all_cutoff_free", a_glob_acc[5] / tot_ads, tot_ads)
            add_1d("U_down_1d", "area", "A_ds", "all_cutoff_free", a_glob_acc[6] / tot_ads, tot_ads)
            add_1d("U_net_1d", "area", "A_ds", "all_cutoff_free", a_glob_acc[7] / tot_ads, tot_ads)

        tot_exp = a_glob_acc[8]
        if tot_exp > 0.0:
            add_1d("U_up_1d", "area", "exposure_eae", "all", a_glob_acc[9] / tot_exp, tot_exp)
            add_1d("U_net_1d", "area", "exposure_eae", "all", a_glob_acc[10] / tot_exp, tot_exp)

        tot_vol = a_glob_acc[11]
        if tot_vol > 0.0:
            add_1d("U_net_1d", "area", "channel_volume", "all", a_glob_acc[12] / tot_vol, tot_vol)

        for k, (lo, _hi, label) in enumerate(area_bins):
            blen = a_bin_acc[k, 0]
            if blen > 0.0:
                add_1d("U_up_1d", "area", "river_length", f"A_bin_{label}", a_bin_acc[k, 1] / blen, blen)
                add_1d("U_down_1d", "area", "river_length", f"A_bin_{label}", a_bin_acc[k, 2] / blen, blen)
                add_1d("U_net_1d", "area", "river_length", f"A_bin_{label}", a_bin_acc[k, 3] / blen, blen)
                add_1d("U_never_1d", "area", "river_length", f"A_bin_{label}", a_bin_acc[k, 4] / blen, blen)
            clen = float(np.sum(a_bin_acc[k:, 0]))
            if clen > 0.0 and lo > 0.0:
                add_1d("U_up_1d", "area", "river_length", f"A_ge_{lo:g}", np.sum(a_bin_acc[k:, 1]) / clen, clen)
                add_1d("U_down_1d", "area", "river_length", f"A_ge_{lo:g}", np.sum(a_bin_acc[k:, 2]) / clen, clen)
                add_1d("U_net_1d", "area", "river_length", f"A_ge_{lo:g}", np.sum(a_bin_acc[k:, 3]) / clen, clen)
                add_1d("U_never_1d", "area", "river_length", f"A_ge_{lo:g}", np.sum(a_bin_acc[k:, 4]) / clen, clen)

        dist_groups = [
            ("river_length", "all", dh_len, d_means[0]),
            ("river_length", f"A_ge_{a_bounds[2]:g}" if k_a >= 3 else "all", dh_gt100, d_means[1]),
            ("river_length", f"A_ge_{a_bounds[3]:g}" if k_a >= 4 else "all", dh_gt1k, d_means[2]),
            ("A_ds", "all_cutoff_free", dh_ads, d_means[3]),
            ("exposure_eae", "all", dh_exp, d_means[4]),
        ]
        for w_label, strat_label, h3, dm_pair in dist_groups:
            tot_w_d = float(np.sum(h3[2]))
            if tot_w_d <= 0.0:
                continue
            for dir_i, dir_name in enumerate(["d_up", "d_down", "d_any"]):
                stats = _hist_quantiles_and_cdf(h3[dir_i], dist_edges, dist_thresholds_km)
                for k_stat, v_stat in stats.items():
                    add_1d(f"{dir_name}_{k_stat}", "distance_km", w_label, strat_label, v_stat, tot_w_d)
            if dm_pair[1] > 0.0:
                add_1d("d_any_mean_finite", "distance_km", w_label, strat_label, dm_pair[0] / dm_pair[1], dm_pair[1])

        if has_flow:
            (
                q_bin_acc,
                q_glob_acc,
                _dq_len,
                _dq_gt1,
                _dq_gt10,
                dq_qds,
                _dq_exp,
                dq_means,
            ) = _integrate_1d_network_kernel(
                length_km,
                q_cons_in_m3s,
                q_cons_out_m3s,
                g_up_in_q,
                inv_q_down_out,
                q_p_in,
                q_p_out,
                d_up_in,
                d_down_out,
                g_ptr,
                g_fracs,
                exp_arr,
                vol_arr,
                dm_u8,
                q_bounds,
                dist_edges,
            )

            tot_qlen = q_glob_acc[0]
            if tot_qlen > 0.0:
                add_1d("U_up_1d", "flow", "river_length", "all", q_glob_acc[1] / tot_qlen, tot_qlen)
                add_1d("U_down_1d", "flow", "river_length", "all", q_glob_acc[2] / tot_qlen, tot_qlen)
                add_1d("U_net_1d", "flow", "river_length", "all", q_glob_acc[3] / tot_qlen, tot_qlen)

            tot_qds = q_glob_acc[4]
            if tot_qds > 0.0:
                add_1d("U_up_1d", "flow", "Q_ds", "all_cutoff_free", q_glob_acc[5] / tot_qds, tot_qds)
                add_1d("U_down_1d", "flow", "Q_ds", "all_cutoff_free", q_glob_acc[6] / tot_qds, tot_qds)
                add_1d("U_net_1d", "flow", "Q_ds", "all_cutoff_free", q_glob_acc[7] / tot_qds, tot_qds)

            if q_glob_acc[8] > 0.0:
                add_1d("U_up_1d", "flow", "exposure_eae", "all", q_glob_acc[9] / q_glob_acc[8], q_glob_acc[8])
                add_1d("U_net_1d", "flow", "exposure_eae", "all", q_glob_acc[10] / q_glob_acc[8], q_glob_acc[8])
            if q_glob_acc[11] > 0.0:
                add_1d("U_net_1d", "flow", "channel_volume", "all", q_glob_acc[12] / q_glob_acc[11], q_glob_acc[11])

            for k, (lo, _hi, label) in enumerate(q_bins):
                blen = q_bin_acc[k, 0]
                if blen > 0.0:
                    add_1d("U_up_1d", "flow", "river_length", f"Q_bin_{label}", q_bin_acc[k, 1] / blen, blen)
                    add_1d("U_down_1d", "flow", "river_length", f"Q_bin_{label}", q_bin_acc[k, 2] / blen, blen)
                    add_1d("U_net_1d", "flow", "river_length", f"Q_bin_{label}", q_bin_acc[k, 3] / blen, blen)
                    add_1d("U_never_1d", "flow", "river_length", f"Q_bin_{label}", q_bin_acc[k, 4] / blen, blen)
                clen = float(np.sum(q_bin_acc[k:, 0]))
                if clen > 0.0 and lo > 0.0:
                    add_1d("U_up_1d", "flow", "river_length", f"Q_ge_{lo:g}", np.sum(q_bin_acc[k:, 1]) / clen, clen)
                    add_1d("U_down_1d", "flow", "river_length", f"Q_ge_{lo:g}", np.sum(q_bin_acc[k:, 2]) / clen, clen)
                    add_1d("U_net_1d", "flow", "river_length", f"Q_ge_{lo:g}", np.sum(q_bin_acc[k:, 3]) / clen, clen)
                    add_1d("U_never_1d", "flow", "river_length", f"Q_ge_{lo:g}", np.sum(q_bin_acc[k:, 4]) / clen, clen)

            tot_w_qds = float(np.sum(dq_qds[2]))
            if tot_w_qds > 0.0:
                for dir_i, dir_name in enumerate(["d_up", "d_down", "d_any"]):
                    stats = _hist_quantiles_and_cdf(dq_qds[dir_i], dist_edges, dist_thresholds_km)
                    for k_stat, v_stat in stats.items():
                        add_1d(f"{dir_name}_{k_stat}", "distance_km", "Q_ds", "all_cutoff_free", v_stat, tot_w_qds)
                if dq_means[3, 1] > 0.0:
                    add_1d("d_any_mean_finite", "distance_km", "Q_ds", "all_cutoff_free", dq_means[3, 0] / dq_means[3, 1], dq_means[3, 1])

    return records


def compute_1d_metrics_tree(
    order: np.ndarray,
    next_down: np.ndarray,
    length_km: np.ndarray,
    area_in_km2: np.ndarray,
    area_out_km2: np.ndarray,
    gauge_reach_idx: np.ndarray,
    gauge_frac_pos: np.ndarray | None = None,
    domain_mask: np.ndarray | None = None,
    q_cons_in_m3s: np.ndarray | None = None,
    q_cons_out_m3s: np.ndarray | None = None,
    q_phys_in_m3s: np.ndarray | None = None,
    q_phys_out_m3s: np.ndarray | None = None,
    exposure_eae: np.ndarray | None = None,
    channel_volume_m3: np.ndarray | None = None,
    area_bins: Sequence[tuple[float, float, str]] = (
        (0.0, 10.0, "0-10"),
        (10.0, 100.0, "10-100"),
        (100.0, 1_000.0, "100-1k"),
        (1_000.0, 10_000.0, "1k-10k"),
        (10_000.0, 100_000.0, "10k-100k"),
        (100_000.0, 1_000_000.0, "100k-1M"),
        (1_000_000.0, float("inf"), ">1M"),
    ),
    q_bins: Sequence[tuple[float, float, str]] = (
        (0.0, 0.1, "<0.1"),
        (0.1, 1.0, "0.1-1"),
        (1.0, 10.0, "1-10"),
        (10.0, 100.0, "10-100"),
        (100.0, 1_000.0, "100-1k"),
        (1_000.0, 10_000.0, "1k-10k"),
        (10_000.0, float("inf"), ">10k"),
    ),
    dist_thresholds_km: Sequence[float] = (10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1_000.0),
) -> list[dict]:
    """Compute all Family 4 1D river-network metrics for a single domain on a tree hydrofabric."""
    recs = compute_1d_metrics_tree_multi_domain(
        order=order,
        next_down=next_down,
        length_km=length_km,
        area_in_km2=area_in_km2,
        area_out_km2=area_out_km2,
        gauge_reach_idx=gauge_reach_idx,
        domains=[("", "", domain_mask)],
        gauge_frac_pos=gauge_frac_pos,
        q_cons_in_m3s=q_cons_in_m3s,
        q_cons_out_m3s=q_cons_out_m3s,
        q_phys_in_m3s=q_phys_in_m3s,
        q_phys_out_m3s=q_phys_out_m3s,
        exposure_eae=exposure_eae,
        channel_volume_m3=channel_volume_m3,
        area_bins=area_bins,
        q_bins=q_bins,
        dist_thresholds_km=dist_thresholds_km,
    )
    for r in recs:
        r.pop("domain_type", None)
        r.pop("domain_name", None)
    return recs


def compute_1d_metrics_dag_multi_domain(
    length_km: np.ndarray,
    area_in_km2: np.ndarray,
    area_out_km2: np.ndarray,
    g_up_in: np.ndarray,
    inv_a_down_out: np.ndarray,
    gauge_reach_idx: np.ndarray,
    domains: Sequence[tuple[str, str, np.ndarray | None]],
    gauge_frac_pos: np.ndarray | None = None,
    exposure_eae: np.ndarray | None = None,
    area_bins: Sequence[tuple[float, float, str]] = (
        (0.0, 10.0, "0-10"),
        (10.0, 100.0, "10-100"),
        (100.0, 1_000.0, "100-1k"),
        (1_000.0, 10_000.0, "1k-10k"),
        (10_000.0, 100_000.0, "10k-100k"),
        (100_000.0, 1_000_000.0, "100k-1M"),
        (1_000_000.0, float("inf"), ">1M"),
    ),
) -> list[dict]:
    """Compute Family 4 1D continuous river metrics on a bifurcating DAG across multiple domains."""
    n = length_km.shape[0]
    _has_g, g_ptr, g_fracs, _fmin, _fmax = build_gauge_csr(n, gauge_reach_idx, gauge_frac_pos)
    exp_arr = np.zeros(n, dtype=np.float64) if exposure_eae is None else np.asarray(exposure_eae, dtype=np.float64)
    vol_arr = np.zeros(n, dtype=np.float64)
    d_inf = np.full(n, np.inf, dtype=np.float64)
    dist_edges = np.concatenate([[0.0], np.geomspace(0.05, 15_000.0, 400)])
    a_bounds = np.array([b[0] for b in area_bins] + [area_bins[-1][1]], dtype=np.float64)

    records: list[dict] = []
    for d_type, d_name, d_mask in domains:
        dm_u8 = np.ones(n, dtype=np.uint8) if d_mask is None else np.asarray(d_mask, dtype=np.uint8)
        a_bin_acc, a_glob_acc, *_ = _integrate_1d_network_kernel(
            length_km,
            area_in_km2,
            area_out_km2,
            g_up_in,
            inv_a_down_out,
            area_in_km2,
            area_out_km2,
            d_inf,
            d_inf,
            g_ptr,
            g_fracs,
            exp_arr,
            vol_arr,
            dm_u8,
            a_bounds,
            dist_edges,
        )

        def add_1d(metric: str, weight: str, stratum: str, value: float, total_w: float):
            records.append({
                "domain_type": d_type,
                "domain_name": d_name,
                "family": "F4_river_1d",
                "metric": metric,
                "ratio_basis": "area_dag",
                "snapping_mode": "dag_width_weighted",
                "weight": weight,
                "stratum": stratum,
                "value": float(value),
                "total_weight": float(total_w),
            })

        tot_len = a_glob_acc[0]
        if tot_len > 0.0:
            add_1d("U_up_1d", "river_length", "all", a_glob_acc[1] / tot_len, tot_len)
            add_1d("U_down_1d", "river_length", "all", a_glob_acc[2] / tot_len, tot_len)
            add_1d("U_net_1d", "river_length", "all", a_glob_acc[3] / tot_len, tot_len)
        tot_ads = a_glob_acc[4]
        if tot_ads > 0.0:
            add_1d("U_up_1d", "A_ds", "all_cutoff_free", a_glob_acc[5] / tot_ads, tot_ads)
            add_1d("U_down_1d", "A_ds", "all_cutoff_free", a_glob_acc[6] / tot_ads, tot_ads)
            add_1d("U_net_1d", "A_ds", "all_cutoff_free", a_glob_acc[7] / tot_ads, tot_ads)
        tot_exp = a_glob_acc[8]
        if tot_exp > 0.0:
            add_1d("U_up_1d", "exposure_eae", "all", a_glob_acc[9] / tot_exp, tot_exp)
            add_1d("U_net_1d", "exposure_eae", "all", a_glob_acc[10] / tot_exp, tot_exp)

        for k, (lo, _hi, label) in enumerate(area_bins):
            blen = a_bin_acc[k, 0]
            if blen > 0.0:
                add_1d("U_up_1d", "river_length", f"A_bin_{label}", a_bin_acc[k, 1] / blen, blen)
                add_1d("U_down_1d", "river_length", f"A_bin_{label}", a_bin_acc[k, 2] / blen, blen)
                add_1d("U_net_1d", "river_length", f"A_bin_{label}", a_bin_acc[k, 3] / blen, blen)
                add_1d("U_never_1d", "river_length", f"A_bin_{label}", a_bin_acc[k, 4] / blen, blen)
            clen = float(np.sum(a_bin_acc[k:, 0]))
            if clen > 0.0 and lo > 0.0:
                add_1d("U_up_1d", "river_length", f"A_ge_{lo:g}", np.sum(a_bin_acc[k:, 1]) / clen, clen)
                add_1d("U_down_1d", "river_length", f"A_ge_{lo:g}", np.sum(a_bin_acc[k:, 2]) / clen, clen)
                add_1d("U_net_1d", "river_length", f"A_ge_{lo:g}", np.sum(a_bin_acc[k:, 3]) / clen, clen)
                add_1d("U_never_1d", "river_length", f"A_ge_{lo:g}", np.sum(a_bin_acc[k:, 4]) / clen, clen)

    return records


def compute_1d_metrics_dag(
    length_km: np.ndarray,
    area_in_km2: np.ndarray,
    area_out_km2: np.ndarray,
    g_up_in: np.ndarray,
    inv_a_down_out: np.ndarray,
    gauge_reach_idx: np.ndarray,
    gauge_frac_pos: np.ndarray | None = None,
    domain_mask: np.ndarray | None = None,
    exposure_eae: np.ndarray | None = None,
    area_bins: Sequence[tuple[float, float, str]] = (
        (0.0, 10.0, "0-10"),
        (10.0, 100.0, "10-100"),
        (100.0, 1_000.0, "100-1k"),
        (1_000.0, 10_000.0, "1k-10k"),
        (10_000.0, 100_000.0, "10k-100k"),
        (100_000.0, 1_000_000.0, "100k-1M"),
        (1_000_000.0, float("inf"), ">1M"),
    ),
) -> list[dict]:
    """Compute Family 4 1D continuous river metrics on a bifurcating DAG (single domain)."""
    recs = compute_1d_metrics_dag_multi_domain(
        length_km=length_km,
        area_in_km2=area_in_km2,
        area_out_km2=area_out_km2,
        g_up_in=g_up_in,
        inv_a_down_out=inv_a_down_out,
        gauge_reach_idx=gauge_reach_idx,
        domains=[("", "", domain_mask)],
        gauge_frac_pos=gauge_frac_pos,
        exposure_eae=exposure_eae,
        area_bins=area_bins,
    )
    for r in recs:
        r.pop("domain_type", None)
        r.pop("domain_name", None)
    return recs
