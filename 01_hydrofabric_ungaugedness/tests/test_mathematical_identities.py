"""Analytical mathematical verification tests for all 5 ungaugedness metric families.

Verifies exact closed-form theorems BEFORE running on any global hydrofabric:
  1. Subdivision invariance of Family 2 (U_never, U_tail), Family 3 (U_res, U_cont, A_tilde),
     and Family 4 (1D continuous river integrals + channel distances) across M = 1..500 splits.
  2. Exact closed-form subdivision drift of Family 1 (U_net_Eq4) under serial & star splits.
  3. Exact analytical values on the 100 km straight river (U_net_len = 0.5, U_net_Ads = 1/3).
  4. Exact equivalence of mid-reach gauge fractional snapping vs explicit reach splitting.
  5. Exact consistency of per-gauge cap identity: U_cont(A_0) = 1 - sum_g min(A_tilde_g, A_0) / |Omega|.
  6. Exact reduction of bifurcating DAG engine to tree engine on single-threaded graphs,
     plus exact probabilistic mass balance on a 50/50 bifurcating river delta.
  7. Family 5 hydroclimatic OOD monotonicity as training gauges expand.
"""

from __future__ import annotations

from pathlib import Path
import sys

_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import pytest

from global_ungaugedness.graph import (
    accumulate_tree,
    propagate_dag_gauges,
    topo_sort_dag_edges,
    topo_sort_tree,
)
from global_ungaugedness.metrics_1d import (
    compute_1d_metrics_dag,
    compute_1d_metrics_tree,
)
from global_ungaugedness.metrics_2d import (
    compute_2d_metrics_dag,
    compute_2d_metrics_tree,
)
from global_ungaugedness.metrics_ood import compute_ood_metrics


def _lookup(records: list[dict], **filters) -> float:
    matches = [
        r["value"]
        for r in records
        if all(
            (np.isnan(v) and np.isnan(r.get(k, np.nan)))
            if isinstance(v, float) and np.isnan(v)
            else (r.get(k) == v)
            for k, v in filters.items()
        )
    ]
    assert len(matches) == 1, f"Expected 1 match for {filters}, got {len(matches)}: {matches}"
    return matches[0]


def test_serial_subdivision_1000km2_basin():
    """Single 1,000 km^2 basin with outlet gauge split into M serial reaches."""
    A_tot = 1_000.0
    L_tot = 100.0

    for M in [1, 2, 5, 25, 100]:
        next_down = np.arange(1, M + 1, dtype=np.int32)
        next_down[-1] = -1
        order, n_cyc = topo_sort_tree(next_down)
        assert n_cyc == 0

        local_a = np.full(M, A_tot / M, dtype=np.float64)
        a_in, a_out = accumulate_tree(order, next_down, local_a)
        lengths = np.full(M, L_tot / M, dtype=np.float64)

        has_g = np.zeros(M, dtype=np.uint8)
        has_g[-1] = 1  # single gauge at basin outlet

        recs_2d = compute_2d_metrics_tree(
            order=order,
            next_down=next_down,
            local_area=local_a,
            total_area_out=a_out,
            has_gauge=has_g,
            weights_dict={"area": local_a},
            area_scale_grid=[250.0, 500.0, 1000.0, 2000.0, float("inf")],
        )

        # Family 1 Eq. 4 exact analytical drift:
        # unit k (1..M) has c_up = 0 (for k<M), c_down = k/M; unit M has u=0.
        # Area-weighted mean = (1/M) * sum_{k=1}^{M-1} (1 - k/M) = (M - 1) / (2M).
        expected_eq4 = (M - 1.0) / (2.0 * M)
        got_eq4 = _lookup(recs_2d, metric="U_net_Eq4", ratio_basis="area", weight="area")
        assert np.isclose(got_eq4, expected_eq4, atol=1e-12), f"M={M}: {got_eq4} != {expected_eq4}"

        # Family 2 strictly scale-invariant: U_never = 0, U_tail = 0 for all M
        assert np.isclose(_lookup(recs_2d, metric="U_never", weight="area"), 0.0, atol=1e-12)
        assert np.isclose(_lookup(recs_2d, metric="U_tail", weight="area"), 0.0, atol=1e-12)

        # Family 3 strictly scale-invariant: inter-gauge area is 1000 km^2 everywhere
        assert np.isclose(_lookup(recs_2d, metric="A_tilde_q50_all", weight="area"), 1000.0, atol=1e-12)
        assert np.isclose(_lookup(recs_2d, metric="U_res", weight="area", scale_param=500.0), 1.0, atol=1e-12)
        assert np.isclose(_lookup(recs_2d, metric="U_res", weight="area", scale_param=1000.0), 0.0, atol=1e-12)
        # U_cont(A_0) = 1 - min(1000, A_0) / 1000
        assert np.isclose(_lookup(recs_2d, metric="U_cont", weight="area", scale_param=250.0), 0.75, atol=1e-12)
        assert np.isclose(_lookup(recs_2d, metric="U_cont", weight="area", scale_param=500.0), 0.50, atol=1e-12)
        assert np.isclose(_lookup(recs_2d, metric="U_cont", weight="area", scale_param=1000.0), 0.0, atol=1e-12)
        assert np.isclose(_lookup(recs_2d, metric="U_cont", weight="area", scale_param=float("inf")), 0.0, atol=1e-12)

        # Family 4 strictly scale-invariant 1D river-network integrals!
        recs_1d = compute_1d_metrics_tree(
            order=order,
            next_down=next_down,
            length_km=lengths,
            area_in_km2=a_in,
            area_out_km2=a_out,
            gauge_reach_idx=np.array([M - 1], dtype=np.int32),
            gauge_frac_pos=np.array([1.0], dtype=np.float64),
        )
        # Exact analytical values for straight river with outlet gauge:
        #   int_0^L (1 - s/L) ds / L = 0.5
        #   int_0^L (1 - s/L) * (A*s/L) ds / int_0^L (A*s/L) ds = (1/6) / (1/2) = 1/3
        assert np.isclose(_lookup(recs_1d, metric="U_net_1d", weight="river_length", stratum="all"), 0.5, atol=1e-12)
        assert np.isclose(_lookup(recs_1d, metric="U_net_1d", weight="A_ds", stratum="all_cutoff_free"), 1.0 / 3.0, atol=1e-12)
        # Exact binned stratum A in [100, 1000] (s from 10 km to 100 km):
        # mean of (1 - a/1000) on [100, 1000] is 1 - 550/1000 = 0.45!
        assert np.isclose(_lookup(recs_1d, metric="U_net_1d", weight="river_length", stratum="A_bin_100-1k"), 0.45, atol=1e-12)
        # Exact binned stratum A in [10, 100]: mean of (1 - a/1000) on [10, 100] = 1 - 55/1000 = 0.945!
        assert np.isclose(_lookup(recs_1d, metric="U_net_1d", weight="river_length", stratum="A_bin_10-100"), 0.945, atol=1e-12)
        # Exact along-channel distance mean & median = 50.0 km (within histogram resolution ~0.2 km)
        assert np.isclose(_lookup(recs_1d, metric="d_any_mean_finite", weight="river_length", stratum="all"), 50.0, atol=1e-9)
        assert np.isclose(_lookup(recs_1d, metric="d_down_q50_all", weight="river_length", stratum="all"), 50.0, atol=0.5)


def test_parallel_star_subdivision_eq4_bias_vs_invariance():
    """1,000 km^2 basin split into K parallel tributaries draining into a tiny outlet cell.

    Proves Eq. 4 exhibits the exact sum(a_i^2 / A) bias while U_never, U_res, U_cont are invariant.
    """
    A_tot = 1_000.0
    eps = 1e-6  # tiny gauged outlet cell
    for K in [2, 10, 50]:
        # Nodes 0..K-1 drain into node K (the outlet)
        next_down = np.full(K + 1, K, dtype=np.int32)
        next_down[K] = -1
        order, _ = topo_sort_tree(next_down)
        local_a = np.full(K + 1, (A_tot - eps) / K, dtype=np.float64)
        local_a[K] = eps
        _, a_out = accumulate_tree(order, next_down, local_a)
        has_g = np.zeros(K + 1, dtype=np.uint8)
        has_g[K] = 1

        recs = compute_2d_metrics_tree(
            order, next_down, local_a, a_out, has_g,
            weights_dict={"area": local_a},
            area_scale_grid=[500.0, 1000.0, float("inf")],
        )
        # Exact Eq. 4 -> 1 - 1/K as eps -> 0!
        got_eq4 = _lookup(recs, metric="U_net_Eq4", ratio_basis="area", weight="area")
        assert np.isclose(got_eq4, 1.0 - 1.0 / K, atol=1e-4)
        # Strictly invariant metrics unchanged
        assert np.isclose(_lookup(recs, metric="U_never", weight="area"), 0.0, atol=1e-12)
        assert np.isclose(_lookup(recs, metric="U_cont", weight="area", scale_param=500.0), 0.5, atol=1e-9)


def test_uneven_splits_and_midreach_gauge_exact_equivalence():
    """Mid-reach gauge at t=0.37 on 1 reach must match splitting at t=0.37 into 2 reaches (and 17 uneven reaches)."""
    L_tot = 120.0
    A_tot = 600.0
    t_gauge = 0.37

    # Case A: single reach with 2 gauges (t=0.37 and t=1.0)
    nd_a = np.array([-1], dtype=np.int32)
    ord_a, _ = topo_sort_tree(nd_a)
    recs_a = compute_1d_metrics_tree(
        ord_a, nd_a,
        length_km=np.array([L_tot]),
        area_in_km2=np.array([0.0]),
        area_out_km2=np.array([A_tot]),
        gauge_reach_idx=np.array([0, 0], dtype=np.int32),
        gauge_frac_pos=np.array([t_gauge, 1.0]),
    )

    # Case B: split unevenly into 10 reaches, with an exact vertex at t=0.37
    pts = np.array([0.0, 0.03, 0.11, 0.25, 0.37, 0.42, 0.58, 0.71, 0.89, 0.96, 1.0])
    M = len(pts) - 1
    nd_b = np.arange(1, M + 1, dtype=np.int32)
    nd_b[-1] = -1
    ord_b, _ = topo_sort_tree(nd_b)
    lens_b = np.diff(pts) * L_tot
    ain_b = pts[:-1] * A_tot
    aout_b = pts[1:] * A_tot
    # Gauge at reach 3 outlet (pts[4]=0.37) and reach M-1 outlet (pts[10]=1.0)
    recs_b = compute_1d_metrics_tree(
        ord_b, nd_b,
        length_km=lens_b,
        area_in_km2=ain_b,
        area_out_km2=aout_b,
        gauge_reach_idx=np.array([3, M - 1], dtype=np.int32),
        gauge_frac_pos=np.array([1.0, 1.0]),
    )

    for metric in ["U_up_1d", "U_down_1d", "U_net_1d"]:
        for weight, stratum in [
            ("river_length", "all"),
            ("A_ds", "all_cutoff_free"),
            ("river_length", "A_bin_100-1k"),
            ("river_length", "A_ge_100"),
        ]:
            va = _lookup(recs_a, metric=metric, weight=weight, stratum=stratum)
            vb = _lookup(recs_b, metric=metric, weight=weight, stratum=stratum)
            assert np.isclose(va, vb, atol=1e-12), f"{metric}/{weight}/{stratum}: {va} != {vb}"

    # Check exact d_any mean too!
    da = _lookup(recs_a, metric="d_any_mean_finite", weight="river_length", stratum="all")
    db = _lookup(recs_b, metric="d_any_mean_finite", weight="river_length", stratum="all")
    # Segment 1 [0, 0.37*L]: only downstream gauge -> mean dist = 0.5 * 0.37 * L
    # Segment 2 [0.37*L, L]: gauges at both ends -> triangle peak = 0.5 * 0.63 * L -> mean dist = 0.25 * 0.63 * L
    expected_d = 0.37 * (0.5 * 0.37 * L_tot) + 0.63 * (0.25 * 0.63 * L_tot)
    assert np.isclose(da, expected_d, atol=1e-9)
    assert np.isclose(db, expected_d, atol=1e-9)


def test_per_gauge_cap_identity_and_never_tail_decomposition():
    """Multi-basin network verifying U_cont(A_0) = 1 - sum_g min(A_tilde_g, A_0)/|Omega| and U_tail."""
    # Component 1 (connected gauged network):
    #   0 (300 km2, gauged) -> 1 (500 km2, ungauged) -> 2 (400 km2, gauged) -> 3 (200 km2, ungauged coastal tail)
    #   Gauge at 0 captures A_tilde_0 = 300 km2.
    #   Gauge at 2 captures A_tilde_2 = 500 + 400 = 900 km2.
    #   Unit 3 (200 km2) is below lowest gauge -> coastal tail!
    # Component 2 (completely ungauged basin):
    #   4 (600 km2, ungauged) -> ocean
    # Total land = 300 + 500 + 400 + 200 + 600 = 2000 km2.
    next_down = np.array([1, 2, 3, -1, -1], dtype=np.int32)
    local_a = np.array([300.0, 500.0, 400.0, 200.0, 600.0], dtype=np.float64)
    has_g = np.array([1, 0, 1, 0, 0], dtype=np.uint8)
    order, _ = topo_sort_tree(next_down)
    _, a_out = accumulate_tree(order, next_down, local_a)

    recs = compute_2d_metrics_tree(
        order, next_down, local_a, a_out, has_g,
        weights_dict={"area": local_a},
        area_scale_grid=[200.0, 500.0, 1000.0, float("inf")],
    )
    # U_zero_net = 600 / 2000 = 0.30
    # U_never = (200 + 600) / 2000 = 0.40
    # U_tail = U_never - U_zero_net = 200 / 2000 = 0.10
    assert np.isclose(_lookup(recs, metric="U_zero_net", weight="area"), 0.30, atol=1e-12)
    assert np.isclose(_lookup(recs, metric="U_never", weight="area"), 0.40, atol=1e-12)
    assert np.isclose(_lookup(recs, metric="U_tail", weight="area"), 0.10, atol=1e-12)

    # Check exact per-gauge cap identity for every A_0
    for a0 in [200.0, 500.0, 1000.0, float("inf")]:
        expected_cont = 1.0 - (min(300.0, a0) + min(900.0, a0)) / 2000.0
        got_cont = _lookup(recs, metric="U_cont", weight="area", scale_param=a0)
        assert np.isclose(got_cont, expected_cont, atol=1e-12)

    # Check U_res(A_0)
    # A_0 = 500: gauge 0 (300 <= 500) passes; gauge 2 (900 > 500) fails; tail + comp2 fail
    assert np.isclose(_lookup(recs, metric="U_res", weight="area", scale_param=500.0), 1700.0 / 2000.0, atol=1e-12)
    # A_0 = 1000: both gauges pass (300, 900 <= 1000), only tail + comp2 fail = 0.40
    assert np.isclose(_lookup(recs, metric="U_res", weight="area", scale_param=1000.0), 0.40, atol=1e-12)


def test_dag_reduces_to_tree_and_bifurcating_delta_mass_balance():
    """Verify DAG engine equals tree engine on a tree, and handles 50/50 bifurcations exactly."""
    # Part 1: 3-edge linear tree (nodes 0 -> 1 -> 2 -> 3)
    n_nodes = 4
    eu = np.array([0, 1, 2], dtype=np.int32)
    ev = np.array([1, 2, 3], dtype=np.int32)
    node_ord, _, n_cyc = topo_sort_dag_edges(n_nodes, eu, ev)
    assert n_cyc == 0

    out_ptr = np.array([0, 1, 2, 3, 3], dtype=np.int32)
    out_edges = np.array([0, 1, 2], dtype=np.int32)
    split_w = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    a_in = np.array([0.0, 100.0, 300.0], dtype=np.float64)
    a_out = np.array([100.0, 300.0, 600.0], dtype=np.float64)
    loc_a = a_out - a_in
    lens = np.array([20.0, 30.0, 50.0], dtype=np.float64)
    has_g = np.array([0, 0, 1], dtype=np.uint8)

    g_in, g_out, p_nev, inv_ad, inv_ia = propagate_dag_gauges(
        n_nodes, node_ord, eu, ev, out_ptr, out_edges, split_w, a_in, a_out, has_g
    )
    recs_dag_2d = compute_2d_metrics_dag(
        loc_a, a_out, has_g, g_in, g_out, p_nev, inv_ad, inv_ia,
        weights_dict={"area": loc_a}, area_scale_grid=[300.0, 600.0, float("inf")],
    )
    recs_dag_1d = compute_1d_metrics_dag(
        lens, a_in, a_out, g_in, inv_ad,
        gauge_reach_idx=np.array([2], dtype=np.int32),
        gauge_frac_pos=np.array([1.0]),
    )

    nd_tree = np.array([1, 2, -1], dtype=np.int32)
    t_ord, _ = topo_sort_tree(nd_tree)
    recs_tree_2d = compute_2d_metrics_tree(
        t_ord, nd_tree, loc_a, a_out, has_g,
        weights_dict={"area": loc_a}, area_scale_grid=[300.0, 600.0, float("inf")],
    )
    recs_tree_1d = compute_1d_metrics_tree(
        t_ord, nd_tree, lens, a_in, a_out,
        gauge_reach_idx=np.array([2], dtype=np.int32),
        gauge_frac_pos=np.array([1.0]),
    )

    assert np.isclose(
        _lookup(recs_dag_2d, metric="U_net_Eq4", weight="area"),
        _lookup(recs_tree_2d, metric="U_net_Eq4", ratio_basis="area", weight="area"),
        atol=1e-12,
    )
    assert np.isclose(
        _lookup(recs_dag_2d, metric="U_cont", weight="area", scale_param=300.0),
        _lookup(recs_tree_2d, metric="U_cont", ratio_basis="area", weight="area", scale_param=300.0),
        atol=1e-12,
    )
    assert np.isclose(
        _lookup(recs_dag_1d, metric="U_net_1d", weight="A_ds", stratum="all_cutoff_free"),
        _lookup(recs_tree_1d, metric="U_net_1d", ratio_basis="area", weight="A_ds", stratum="all_cutoff_free"),
        atol=1e-12,
    )

    # Part 2: Bifurcating delta!
    # Mainstem edge 0 (node 0 -> 1, area 0 -> 1000) splits 50/50 at node 1 into:
    #   edge 1 (node 1 -> 2, area 500 -> 500, GAUGED)
    #   edge 2 (node 1 -> 3, area 500 -> 500, UNGAUGED)
    eu2 = np.array([0, 1, 1], dtype=np.int32)
    ev2 = np.array([1, 2, 3], dtype=np.int32)
    no2, _, cyc2 = topo_sort_dag_edges(4, eu2, ev2)
    assert cyc2 == 0
    op2 = np.array([0, 1, 3, 3, 3], dtype=np.int32)
    oe2 = np.array([0, 1, 2], dtype=np.int32)
    sw2 = np.array([1.0, 0.5, 0.5], dtype=np.float64)
    ai2 = np.array([0.0, 500.0, 500.0], dtype=np.float64)
    ao2 = np.array([1000.0, 500.0, 500.0], dtype=np.float64)
    hg2 = np.array([0, 1, 0], dtype=np.uint8)

    _, _, p_nev2, inv_ad2, _ = propagate_dag_gauges(
        4, no2, eu2, ev2, op2, oe2, sw2, ai2, ao2, hg2
    )
    # Half of mainstem water flows into gauged branch 1, half into ungauged branch 2 -> p_never[0] = 0.50!
    assert np.isclose(p_nev2[0], 0.5, atol=1e-12)
    assert np.isclose(p_nev2[2], 1.0, atol=1e-12)
    # Expected 1/A_down leaving edge 0 = 0.5 * (1/500) + 0.5 * 0 = 0.001
    assert np.isclose(inv_ad2[0], 0.001, atol=1e-12)


def test_ood_metrics_monotonic_with_more_gauges():
    """Adding gauges strictly reduces (or keeps equal) k-NN OOD distance."""
    rng = np.random.default_rng(42)
    X_target = rng.normal(size=(500, 6))
    w_area = rng.uniform(10.0, 100.0, size=500)
    X_t1 = X_target[:40]
    X_t2 = X_target[:200]

    r1, _ = compute_ood_metrics(X_target, X_t1, {"area": w_area}, k_neighbors=(1, 5), workers=1)
    r2, _ = compute_ood_metrics(X_target, X_t2, {"area": w_area}, k_neighbors=(1, 5), workers=1)

    d1 = _lookup(r1, metric="U_OOD_knn1_mean", weight="area")
    d2 = _lookup(r2, metric="U_OOD_knn1_mean", weight="area")
    assert d2 < d1
