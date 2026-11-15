"""Numba-accelerated hydrological graph kernels for trees and bifurcating DAGs.

Supports:
1. Standard single-downstream river trees (HydroBASINS L1-L12, HydroRIVERS,
   MERIT-Basins, TDX-Hydro, GRIT mainstem-tree projection).
2. Multi-threaded / bifurcating river DAGs (GRIT v0.6 full segment network)
   with width-weighted discharge/area partitioning and automatic cycle breaking.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Tree topology kernels (O(N), handles 20M+ nodes in ~0.5s)
# ---------------------------------------------------------------------------

@njit(cache=True)
def topo_sort_tree(next_down: np.ndarray) -> tuple[np.ndarray, int]:
    """Kahn topological sort (upstream -> downstream) for a directed forest.

    Args:
        next_down: int32 array of length N; next_down[i] in [0, N) or -1 for outlet.

    Returns:
        order: int32 array of length <= N in upstream-to-downstream order.
        n_cycle_nodes: number of nodes trapped in cycles (0 for valid forest).
    """
    n = next_down.shape[0]
    indeg = np.zeros(n, dtype=np.int32)
    for i in range(n):
        d = next_down[i]
        if 0 <= d < n and d != i:
            indeg[d] += 1

    order = np.empty(n, dtype=np.int32)
    head = 0
    tail = 0
    for i in range(n):
        if indeg[i] == 0:
            order[tail] = i
            tail += 1

    while head < tail:
        u = order[head]
        head += 1
        d = next_down[u]
        if 0 <= d < n and d != u:
            indeg[d] -= 1
            if indeg[d] == 0:
                order[tail] = d
                tail += 1

    return order[:tail], n - tail


@njit(cache=True)
def break_tree_cycles_inplace(
    next_down: np.ndarray, up_area: np.ndarray
) -> int:
    """Break any rare DEM routing cycles by severing the highest-area edge in each cycle."""
    n = next_down.shape[0]
    # Remove self-loops first
    n_broken = 0
    for i in range(n):
        if next_down[i] == i or next_down[i] < -1 or next_down[i] >= n:
            if next_down[i] != -1:
                next_down[i] = -1
                n_broken += 1

    order, n_cyc = topo_sort_tree(next_down)
    if n_cyc == 0:
        return n_broken

    in_order = np.zeros(n, dtype=np.uint8)
    for k in range(order.shape[0]):
        in_order[order[k]] = 1

    # Walk remaining nodes to find cycle members and break at largest up_area
    visited = np.zeros(n, dtype=np.uint8)
    for start in range(n):
        if in_order[start] == 1 or visited[start] == 1:
            continue
        # Follow pointers until we hit a visited node or -1
        path = []
        cur = start
        while cur != -1 and in_order[cur] == 0 and visited[cur] == 0:
            visited[cur] = 2  # on current stack
            path.append(cur)
            cur = next_down[cur]
        if cur != -1 and visited[cur] == 2:
            # Extract cycle from cur onwards
            best_node = cur
            best_area = up_area[cur]
            found = False
            for node in path:
                if node == cur:
                    found = True
                if found and up_area[node] >= best_area:
                    best_area = up_area[node]
                    best_node = node
            next_down[best_node] = -1
            n_broken += 1
        for node in path:
            visited[node] = 1

    return n_broken


@njit(cache=True)
def accumulate_tree(
    order: np.ndarray, next_down: np.ndarray, local_val: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (inflow_from_upstream, total_outflow) on a tree."""
    n = next_down.shape[0]
    val_in = np.zeros(n, dtype=np.float64)
    val_out = np.zeros(n, dtype=np.float64)
    for k in range(order.shape[0]):
        u = order[k]
        tot = val_in[u] + local_val[u]
        val_out[u] = tot
        d = next_down[u]
        if d >= 0:
            val_in[d] += tot
    return val_in, val_out


@njit(cache=True)
def find_root_ids(order: np.ndarray, next_down: np.ndarray) -> np.ndarray:
    """Assign each node the index of its terminal outlet (connected component root)."""
    n = next_down.shape[0]
    root = np.empty(n, dtype=np.int32)
    for k in range(order.shape[0] - 1, -1, -1):
        u = order[k]
        d = next_down[u]
        if d < 0:
            root[u] = u
        else:
            root[u] = root[d]
    return root


@njit(cache=True)
def propagate_tree_gauges(
    order: np.ndarray,
    next_down: np.ndarray,
    local_w: np.ndarray,
    total_w_out: np.ndarray,
    has_gauge: np.ndarray,
    effective_gauge_w: np.ndarray,
    use_subbasin_effective: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Propagate upstream and downstream gauge state across a tree for any additive weight w.

    Args:
        order: Upstream-to-downstream topological order (length N).
        next_down: Downstream node index (-1 for terminal outlet).
        local_w: Nonnegative local incremental weight (area km2 or runoff m3/s) per unit.
        total_w_out: Total cumulative weight leaving unit outlet (= sum of upstream + local_w).
        has_gauge: uint8 array (1 if unit contains >=1 gauge, else 0).
        effective_gauge_w: float64 array; if use_subbasin_effective=True and has_gauge[u]==1,
            the effective weight monitored by the gauge(s) inside unit u (e.g., headwater
            reported area capped to local_w[u] if no upstream inflows, else total_w_out[u]).
            If use_subbasin_effective=False, any gauge in unit u monitors the entire unit
            outlet total_w_out[u].
        use_subbasin_effective: Whether to use sub-unit effective gauge weight for headwater
            gauges (exact match to original paper's sub-polygon headwater sensitivity).

    Returns:
        g_up_in: Gauged upstream weight entering unit u from strictly upstream tributaries.
        g_up_out: Total gauged weight leaving unit u (includes gauge in u itself).
        down_gauge_node: Index of nearest downstream gauge node (including u if u has full-outlet
            gauge, or strictly downstream if u has no gauge; -1 if never gauged downstream).
            Note: for a sub-unit headwater gauge (effective_gauge_w < total_w_out), the
            *ungauged residual* of u drains to the strictly downstream gauge! We handle that
            separately via strictly_down_gauge_node.
        intergauge_w: For each unit u, the inter-gauge catchment weight W_tilde_g of the nearest
            downstream gauge g that captures u's runoff (np.inf if never gauged).
        network_has_gauge: uint8 array (1 if unit u belongs to a connected basin with >=1 gauge).
    """
    n = next_down.shape[0]
    g_up_in = np.zeros(n, dtype=np.float64)
    g_up_out = np.zeros(n, dtype=np.float64)
    # Track exact inter-gauge incremental weight accumulating into each gauge
    # (resets to 0 immediately downstream of a gauge!)
    ungauged_inflow_to_u = np.zeros(n, dtype=np.float64)
    gauge_inter_w = np.zeros(n, dtype=np.float64)

    for k in range(order.shape[0]):
        u = order[k]
        gin = g_up_in[u]
        if has_gauge[u]:
            if use_subbasin_effective and effective_gauge_w[u] >= 0.0:
                eff = effective_gauge_w[u]
                if eff > total_w_out[u]:
                    eff = total_w_out[u]
                if eff < gin:
                    eff = gin
            else:
                eff = total_w_out[u]
            gout = eff
            # Inter-gauge incremental weight captured by gauge at u:
            # all upstream ungauged inflow + portion of u monitored by gauge
            # = eff - gin (exact by conservation: gin was already captured by upstream gauges!)
            inter_u = eff - gin
            if inter_u < 0.0:
                inter_u = 0.0
            gauge_inter_w[u] = inter_u
            # Any residual ungauged portion of u (only > 0 for sub-unit headwater gauge)
            # continues downstream to the next downstream gauge
            residual_u = total_w_out[u] - eff
            if residual_u < 0.0:
                residual_u = 0.0
            out_ungauged = residual_u
        else:
            gout = gin
            out_ungauged = ungauged_inflow_to_u[u] + local_w[u]

        g_up_out[u] = gout
        d = next_down[u]
        if d >= 0:
            g_up_in[d] += gout
            ungauged_inflow_to_u[d] += out_ungauged

    # Reverse pass: nearest downstream gauge (including u itself if full-unit gauge)
    # and strictly downstream gauge (for residual / polygon Eq. 4 downstream lookup)
    down_gauge_node = np.full(n, -1, dtype=np.int32)
    intergauge_w = np.full(n, np.inf, dtype=np.float64)
    root_has_gauge = np.zeros(n, dtype=np.uint8)

    # First mark roots that have any gauge in their connected component
    for k in range(order.shape[0]):
        u = order[k]
        if has_gauge[u]:
            root_has_gauge[u] = 1
        d = next_down[u]
        if d >= 0 and root_has_gauge[u] == 1:
            root_has_gauge[d] = 1

    network_has_gauge = np.zeros(n, dtype=np.uint8)
    for k in range(order.shape[0] - 1, -1, -1):
        u = order[k]
        d = next_down[u]
        if d < 0:
            network_has_gauge[u] = root_has_gauge[u]
            strict_dg = -1
        else:
            network_has_gauge[u] = network_has_gauge[d]
            strict_dg = down_gauge_node[d]

        if has_gauge[u]:
            down_gauge_node[u] = u
            intergauge_w[u] = gauge_inter_w[u]
        else:
            down_gauge_node[u] = strict_dg
            if strict_dg >= 0:
                intergauge_w[u] = gauge_inter_w[strict_dg]
            else:
                intergauge_w[u] = np.inf

    return g_up_in, g_up_out, down_gauge_node, intergauge_w, network_has_gauge


@njit(cache=True)
def propagate_tree_channel_distances(
    order: np.ndarray,
    next_down: np.ndarray,
    length_km: np.ndarray,
    has_gauge: np.ndarray,
    gauge_frac_min: np.ndarray,
    gauge_frac_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact along-channel distances to nearest upstream and downstream gauges.

    For reach u of length L_u with upstream end (s=0) and downstream end (s=L_u):
    - d_up_in[u]: minimum along-channel distance (km) from ANY upstream gauge to the
      upstream inlet of reach u (np.inf if no gauge upstream of u).
    - d_down_out[u]: distance (km) from the downstream outlet of reach u to the nearest
      downstream gauge (np.inf if no gauge downstream of u).

    If reach u itself contains one or more gauges, gauge_frac_min[u] is the most-upstream
    gauge fractional position in [0, 1], and gauge_frac_max[u] is the most-downstream
    gauge fractional position in [0, 1].
    """
    n = next_down.shape[0]
    d_up_in = np.full(n, np.inf, dtype=np.float64)
    d_down_out = np.full(n, np.inf, dtype=np.float64)

    # Forward pass (upstream -> downstream): minimum distance from nearest upstream gauge
    for k in range(order.shape[0]):
        u = order[k]
        L = length_km[u]
        if has_gauge[u]:
            # Most-downstream gauge in u sits at position gauge_frac_max[u] * L,
            # so distance from that gauge to outlet of u is (1 - gauge_frac_max[u]) * L
            dist_to_out = (1.0 - gauge_frac_max[u]) * L
            if dist_to_out < 0.0:
                dist_to_out = 0.0
        else:
            dist_to_out = d_up_in[u] + L

        d = next_down[u]
        if d >= 0 and dist_to_out < d_up_in[d]:
            d_up_in[d] = dist_to_out

    # Reverse pass (downstream -> upstream): distance to nearest downstream gauge
    for k in range(order.shape[0] - 1, -1, -1):
        u = order[k]
        d = next_down[u]
        if d >= 0:
            if has_gauge[d]:
                # Distance from outlet of u (= inlet of d) to most-upstream gauge in d
                d_down_out[u] = gauge_frac_min[d] * length_km[d]
            else:
                d_down_out[u] = d_down_out[d] + length_km[d]

    return d_up_in, d_down_out


# ---------------------------------------------------------------------------
# Bifurcating DAG kernels (for GRIT v0.6 multi-threaded river networks)
# ---------------------------------------------------------------------------

@njit(cache=True)
def topo_sort_dag_edges(
    n_nodes: int,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Kahn topological sort of a node-edge DAG (e.g., GRIT segments connecting nodes).

    Returns:
        node_order: topological order of acyclic nodes.
        edge_order: topological order of edges (sorted by upstream node order).
        n_cycle_nodes: number of nodes in directed cycles.
    """
    n_edges = edge_u.shape[0]
    indeg = np.zeros(n_nodes, dtype=np.int32)
    out_cnt = np.zeros(n_nodes, dtype=np.int32)
    for e in range(n_edges):
        u = edge_u[e]
        v = edge_v[e]
        if u >= 0 and v >= 0 and u != v:
            indeg[v] += 1
            out_cnt[u] += 1

    # CSR adjacency of outgoing edges per node
    out_ptr = np.zeros(n_nodes + 1, dtype=np.int32)
    for i in range(n_nodes):
        out_ptr[i + 1] = out_ptr[i] + out_cnt[i]
    out_edges = np.empty(out_ptr[n_nodes], dtype=np.int32)
    cursor = out_ptr[:-1].copy()
    for e in range(n_edges):
        u = edge_u[e]
        v = edge_v[e]
        if u >= 0 and v >= 0 and u != v:
            out_edges[cursor[u]] = e
            cursor[u] += 1

    node_order = np.empty(n_nodes, dtype=np.int32)
    head = 0
    tail = 0
    for i in range(n_nodes):
        if indeg[i] == 0:
            node_order[tail] = i
            tail += 1

    edge_order = np.empty(n_edges, dtype=np.int32)
    e_tail = 0
    while head < tail:
        u = node_order[head]
        head += 1
        for p in range(out_ptr[u], out_ptr[u + 1]):
            e = out_edges[p]
            edge_order[e_tail] = e
            e_tail += 1
            v = edge_v[e]
            indeg[v] -= 1
            if indeg[v] == 0:
                node_order[tail] = v
                tail += 1

    return node_order[:tail], edge_order[:e_tail], n_nodes - tail


@njit(cache=True)
def propagate_dag_gauges(
    n_nodes: int,
    node_order: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    out_ptr: np.ndarray,
    out_edges: np.ndarray,
    edge_split_w: np.ndarray,
    edge_area_in: np.ndarray,
    edge_area_out: np.ndarray,
    edge_has_gauge: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mass-conserving width-weighted gauge propagation on a bifurcating DAG.

    At every bifurcation node u with outgoing edges e_1..e_k and normalized split
    weights sum_j edge_split_w[e_j] = 1:
    - Forward (upstream -> downstream): gauged upstream area entering node u is
      distributed across outgoing edges proportionally to edge_split_w[e_j]
      (clipped to edge_area_in[e_j]). If edge e has a gauge, its outgoing gauged
      area becomes edge_area_out[e].
    - Reverse (downstream -> upstream):
      * p_never_out[e]: probability that water exiting edge e never encounters a
        downstream gauge before an ocean/sink outlet (0 if a downstream edge is
        gauged, 1 if terminal outlet).
      * inv_a_down_out[e]: expected reciprocal downstream gauge area E[1 / A_g]
        at the downstream end of edge e, so c_down(x) = clip(A(x) * inv_a_down(x), 0, 1).
        On a single-thread tree (all split weights = 1), this reduces identically
        to c_down(x) = A(x) / A_g_down!
      * inv_inter_a_out[e]: expected reciprocal inter-gauge area E[1 / A_tilde_g]
        for continuous resolution U_cont / effective scale on bifurcating networks.
    """
    n_edges = edge_u.shape[0]
    g_up_in = np.zeros(n_edges, dtype=np.float64)
    g_up_out = np.zeros(n_edges, dtype=np.float64)
    node_g_in = np.zeros(n_nodes, dtype=np.float64)
    node_ungauged_in = np.zeros(n_nodes, dtype=np.float64)
    edge_inter_w = np.zeros(n_edges, dtype=np.float64)

    # Forward topological pass
    for k in range(node_order.shape[0]):
        u = node_order[k]
        gin_u = node_g_in[u]
        ung_u = node_ungauged_in[u]
        for p in range(out_ptr[u], out_ptr[u + 1]):
            e = out_edges[p]
            sw = edge_split_w[e]
            ge_in = gin_u * sw
            if ge_in > edge_area_in[e]:
                ge_in = edge_area_in[e]
            g_up_in[e] = ge_in

            local_a = edge_area_out[e] - edge_area_in[e]
            if local_a < 0.0:
                local_a = 0.0

            if edge_has_gauge[e]:
                ge_out = edge_area_out[e]
                inter_e = ge_out - ge_in
                if inter_e < 0.0:
                    inter_e = 0.0
                edge_inter_w[e] = inter_e
                ung_out = 0.0
            else:
                ge_out = ge_in
                ung_out = ung_u * sw + local_a

            g_up_out[e] = ge_out
            v = edge_v[e]
            if v >= 0:
                node_g_in[v] += ge_out
                node_ungauged_in[v] += ung_out

    # Reverse topological pass
    node_p_never = np.ones(n_nodes, dtype=np.float64)
    node_inv_a_down = np.zeros(n_nodes, dtype=np.float64)
    node_inv_inter_a = np.zeros(n_nodes, dtype=np.float64)

    edge_p_never_out = np.ones(n_edges, dtype=np.float64)
    edge_inv_a_down_out = np.zeros(n_edges, dtype=np.float64)
    edge_inv_inter_a_out = np.zeros(n_edges, dtype=np.float64)

    for k in range(node_order.shape[0] - 1, -1, -1):
        u = node_order[k]
        start = out_ptr[u]
        end = out_ptr[u + 1]
        if start == end:
            # Terminal ocean/sink node
            node_p_never[u] = 1.0
            node_inv_a_down[u] = 0.0
            node_inv_inter_a[u] = 0.0
            continue

        p_nev_sum = 0.0
        inv_ad_sum = 0.0
        inv_ia_sum = 0.0
        for p in range(start, end):
            e = out_edges[p]
            sw = edge_split_w[e]
            v = edge_v[e]
            if v >= 0:
                edge_p_never_out[e] = node_p_never[v]
                edge_inv_a_down_out[e] = node_inv_a_down[v]
                edge_inv_inter_a_out[e] = node_inv_inter_a[v]
            else:
                edge_p_never_out[e] = 1.0
                edge_inv_a_down_out[e] = 0.0
                edge_inv_inter_a_out[e] = 0.0

            if edge_has_gauge[e]:
                e_p_in = 0.0
                e_inv_ad_in = 1.0 / edge_area_out[e] if edge_area_out[e] > 0.0 else 0.0
                e_inv_ia_in = 1.0 / edge_inter_w[e] if edge_inter_w[e] > 0.0 else 0.0
            else:
                e_p_in = edge_p_never_out[e]
                e_inv_ad_in = edge_inv_a_down_out[e]
                e_inv_ia_in = edge_inv_inter_a_out[e]

            p_nev_sum += sw * e_p_in
            inv_ad_sum += sw * e_inv_ad_in
            inv_ia_sum += sw * e_inv_ia_in

        node_p_never[u] = p_nev_sum
        node_inv_a_down[u] = inv_ad_sum
        node_inv_inter_a[u] = inv_ia_sum

    return (
        g_up_in,
        g_up_out,
        edge_p_never_out,
        edge_inv_a_down_out,
        edge_inv_inter_a_out,
    )
