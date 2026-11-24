"""End-to-end ETL, gauge snapping, and full metric computation for GRIT v1.0 (30m FABDEM).

Runs TWO complete evaluations on GRIT v1.0 (flow-weighted metrics skipped by design):
  1. `grit_v10_mainstem_tree`: Single-thread mainstem tree projection (`is_mainstem == 1`,
     1,682,135 river segments + direct coastal catchments `is_coastal == 1`).
  2. `grit_v10_bifurcating_dag`: Full multi-threaded / bifurcating river DAG (all
     1,764,069 river segments with width-weighted split fractions + coastal catchments).

Directly quantifies the impact of braided/deltaic bifurcations on global ungaugedness!
"""

from __future__ import annotations

import os
os.environ.setdefault("OGR_ORGANIZE_POLYGONS", "SKIP")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import shapely

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from configs.benchmark_config import (
    AREA_SCALE_GRID_KM2,
    GAUGE_TIERS,
    GAUGES_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    RESULTS_DIR,
    RIVER_AREA_BINS_KM2,
)
from global_ungaugedness.graph import (
    accumulate_tree,
    break_tree_cycles_inplace,
    propagate_dag_gauges,
    topo_sort_dag_edges,
    topo_sort_tree,
)
from global_ungaugedness.metrics_1d import compute_1d_metrics_dag_multi_domain
from global_ungaugedness.metrics_2d import compute_2d_metrics_dag_multi_domain
from global_ungaugedness.runner import aggregate_gauge_snaps_to_units, run_tree_hydrofabric_suite
from global_ungaugedness.snapping import StreamingPolygonSnapper, StreamingReachSnapper
from global_ungaugedness.spatial_enrichment import (
    ARIDITY_LABELS,
    assign_country_continent_and_domain,
    map_points_to_hydrosheds_l12,
    transfer_exposure_and_aridity_from_l12,
)

GRIT_RAW = RAW_DIR / "grit_v10"
OUT_PROC = PROCESSED_DIR / "grit_v10"
REGIONS = ["AF", "AS", "EU", "NA", "SA", "SI", "SP"]


def build_grit_tables() -> pd.DataFrame:
    """Load all 7 regional GRIT v1.0 segment networks + coastal component catchments."""
    OUT_PROC.mkdir(parents=True, exist_ok=True)
    seg_pq = OUT_PROC / "grit_v10_all_units.parquet"
    if seg_pq.exists():
        print(f"[cache] Loading {seg_pq}", flush=True)
        all_df = pd.read_parquet(seg_pq)
        # Upgrade cached table if orphan L12 exposure conservation or coastal L12 fallback was not yet applied
        if all_df["population"].sum() < 7.5e9 or (all_df["hybas_l12_id"] < 0).sum() > 5000:
            print("  Upgrading cached GRIT exposure & coastal L12 IDs for 100% mass conservation...", flush=True)
            lons = all_df["lon"].to_numpy(dtype=np.float64)
            lats = all_df["lat"].to_numpy(dtype=np.float64)
            hybas_l12 = all_df["hybas_l12_id"].to_numpy(dtype=np.int64).copy()
            miss = np.flatnonzero(hybas_l12 < 0)
            l12_units_pq = PROCESSED_DIR / "hydrosheds" / "hydrosheds_l12_units.parquet"
            if miss.size > 0 and l12_units_pq.exists():
                from scipy.spatial import cKDTree
                from global_ungaugedness.spatial_enrichment import _lonlat_to_xyz
                l12_u = pd.read_parquet(l12_units_pq, columns=["unit_id", "lon", "lat"])
                tree = cKDTree(_lonlat_to_xyz(l12_u["lon"].to_numpy(), l12_u["lat"].to_numpy()))
                max_chord = 2.0 * np.sin(50.0 / (2.0 * 6371.0088))
                dists, nn = tree.query(_lonlat_to_xyz(lons[miss], lats[miss]), k=1, distance_upper_bound=max_chord, workers=4)
                ok = np.isfinite(dists) & (nn < len(l12_u))
                if np.any(ok):
                    hybas_l12[miss[ok]] = l12_u["unit_id"].to_numpy(dtype=np.int64)[nn[ok]]
            l12_ref = PROCESSED_DIR / "hydrosheds" / "hydrosheds_l12_enrichment_ref.parquet"
            pop, eae, aridity = transfer_exposure_and_aridity_from_l12(
                hybas_l12,
                all_df["local_area_dag_km2"].to_numpy(dtype=np.float64),
                l12_aridity_parquet=l12_ref if l12_ref.exists() else None,
                unit_lons=lons,
                unit_lats=lats,
            )
            all_df["hybas_l12_id"] = hybas_l12
            all_df["population"] = pop
            if eae is not None:
                all_df["exposure_eae"] = eae
            all_df["aridity_zone"] = aridity
            all_df.to_parquet(seg_pq, index=False)
            eae_msg = f", EAE={eae.sum()/1e6:.2f}M/yr" if eae is not None else ""
            print(
                f"  Upgraded GRIT cache: pop={pop.sum()/1e9:.3f}B{eae_msg}, "
                f"unmapped_l12={(hybas_l12 < 0).sum():,}",
                flush=True,
            )
        return all_df

    t0 = time.time()
    seg_dfs = []
    coastal_dfs = []

    print("Loading GRIT v1.0 segments and coastal component catchments across 7 regions...", flush=True)
    for reg in REGIONS:
        gpkg = GRIT_RAW / f"GRITv1.0_segments_{reg}_EPSG4326.gpkg"
        gdf = pyogrio.read_dataframe(
            str(gpkg),
            layer="lines",
            columns=[
                "global_id",
                "catchment_id",
                "upstream_node_id",
                "downstream_node_id",
                "is_mainstem",
                "width_adjusted",
                "length",
                "drainage_area_in",
                "drainage_area_out",
                "drainage_area_mainstem_in",
                "drainage_area_mainstem_out",
            ],
        )
        b = shapely.bounds(gdf.geometry.to_numpy())
        df = pd.DataFrame({
            "unit_id": gdf["global_id"].to_numpy(dtype=np.int64),
            "catchment_id": gdf["catchment_id"].to_numpy(dtype=np.int64),
            "upstream_node_id": gdf["upstream_node_id"].to_numpy(dtype=np.int64),
            "downstream_node_id": gdf["downstream_node_id"].to_numpy(dtype=np.int64),
            "is_mainstem": (gdf["is_mainstem"].to_numpy() == 1),
            "is_coastal_catchment": np.zeros(len(gdf), dtype=bool),
            "width_m": np.maximum(gdf["width_adjusted"].fillna(1.0).to_numpy(dtype=np.float64), 1.0),
            "length_km": np.maximum(gdf["length"].fillna(0.0).to_numpy(dtype=np.float64) / 1e3, 0.0),
            "da_in_km2": np.maximum(gdf["drainage_area_in"].fillna(0.0).to_numpy(dtype=np.float64), 0.0),
            "da_out_km2": np.maximum(gdf["drainage_area_out"].fillna(0.0).to_numpy(dtype=np.float64), 0.0),
            "da_main_in_km2": np.maximum(gdf["drainage_area_mainstem_in"].fillna(0.0).to_numpy(dtype=np.float64), 0.0),
            "da_main_out_km2": np.maximum(gdf["drainage_area_mainstem_out"].fillna(0.0).to_numpy(dtype=np.float64), 0.0),
            "lon": 0.5 * (b[:, 0] + b[:, 2]),
            "lat": 0.5 * (b[:, 1] + b[:, 3]),
            "region": reg,
        })
        seg_dfs.append(df)

        print(f"  {reg}: {len(df):,} segments", flush=True)

    # Unchannelized coastal and endorheic sink component catchments (no >=50 km^2 segment)
    cgpkg = GRIT_RAW / "GRITv1.0_component_catchments_GLOBAL_EPSG4326.gpkg"
    clyr = str(pyogrio.list_layers(str(cgpkg))[0][0])
    cgdf = pyogrio.read_dataframe(str(cgpkg), layer=clyr, columns=["global_id", "area", "is_coastal", "is_sink", "domain"])
    seg_cids = set().union(*(set(d["catchment_id"].to_numpy()) for d in seg_dfs))
    cmask = ~cgdf["global_id"].isin(seg_cids).to_numpy()
    cgdf = cgdf.loc[cmask]
    if len(cgdf) > 0:
        cb = shapely.bounds(cgdf.geometry.to_numpy())
        cids = 800_000_000_000 + cgdf["global_id"].to_numpy(dtype=np.int64)
        areas = np.maximum(cgdf["area"].fillna(0.0).to_numpy(dtype=np.float64), 0.0)
        coastal_dfs.append(pd.DataFrame({
            "unit_id": cids,
            "catchment_id": cgdf["global_id"].to_numpy(dtype=np.int64),
            "upstream_node_id": np.full(len(cgdf), -1, dtype=np.int64),
            "downstream_node_id": np.full(len(cgdf), -1, dtype=np.int64),
            "is_mainstem": np.ones(len(cgdf), dtype=bool),
            "is_coastal_catchment": np.ones(len(cgdf), dtype=bool),
            "width_m": np.ones(len(cgdf), dtype=np.float64),
            "length_km": np.zeros(len(cgdf), dtype=np.float64),
            "da_in_km2": np.zeros(len(cgdf), dtype=np.float64),
            "da_out_km2": areas,
            "da_main_in_km2": np.zeros(len(cgdf), dtype=np.float64),
            "da_main_out_km2": areas,
            "lon": 0.5 * (cb[:, 0] + cb[:, 2]),
            "lat": 0.5 * (cb[:, 1] + cb[:, 3]),
            "region": cgdf["domain"].astype(str).to_numpy(),
        }))
    print(f"  GLOBAL unchannelized coastal/sink catchments: {len(cgdf):,}", flush=True)

    all_df = pd.concat(seg_dfs + coastal_dfs, ignore_index=True)

    # Local incremental catchment area for DAG (inlet segments own da_in + lateral; non-inlets own lateral)
    is_seg = ~all_df["is_coastal_catchment"].to_numpy()
    dn_nodes = set(all_df.loc[is_seg, "downstream_node_id"].to_numpy())
    is_inlet = is_seg & (~all_df["upstream_node_id"].isin(dn_nodes).to_numpy())
    lateral = np.maximum(all_df["da_out_km2"].to_numpy() - all_df["da_in_km2"].to_numpy(), 0.0)
    local_a_dag = np.where(
        all_df["is_coastal_catchment"].to_numpy(),
        all_df["da_out_km2"].to_numpy(),
        np.where(is_inlet, all_df["da_out_km2"].to_numpy(), lateral),
    )
    all_df["local_area_dag_km2"] = local_a_dag

    lons = all_df["lon"].to_numpy(dtype=np.float64)
    lats = all_df["lat"].to_numpy(dtype=np.float64)
    iso3, cont, in_common = assign_country_continent_and_domain(lons, lats)
    hybas_l12 = map_points_to_hydrosheds_l12(lons, lats)
    l12_ref = PROCESSED_DIR / "hydrosheds" / "hydrosheds_l12_enrichment_ref.parquet"
    pop, eae, aridity = transfer_exposure_and_aridity_from_l12(
        hybas_l12,
        local_a_dag,
        l12_aridity_parquet=l12_ref if l12_ref.exists() else None,
        unit_lons=lons,
        unit_lats=lats,
    )

    all_df["population"] = pop
    if eae is not None:
        all_df["exposure_eae"] = eae
    all_df["country_iso3"] = iso3
    all_df["continent"] = cont
    all_df["in_common_domain"] = in_common
    all_df["aridity_zone"] = aridity
    all_df["hybas_l12_id"] = hybas_l12
    all_df["lon"] = lons.astype(np.float32)
    all_df["lat"] = lats.astype(np.float32)

    eae_msg = f", EAE={eae.sum()/1e6:.2f}M/yr" if eae is not None else ""
    print(
        f"GRIT built in {time.time() - t0:.1f}s: {len(all_df):,} total units "
        f"({int(is_seg.sum()):,} segments, {int((~is_seg).sum()):,} coastal), "
        f"total land={local_a_dag.sum()/1e6:.2f}M km2, pop={pop.sum()/1e9:.3f}B{eae_msg}",
        flush=True,
    )
    all_df.to_parquet(seg_pq, index=False)
    return all_df


def snap_gauges_grit(all_df: pd.DataFrame, gauges_df: pd.DataFrame, mainstem_only: bool) -> pd.DataFrame:
    tag = "mainstem" if mainstem_only else "dag"
    snap_pq = OUT_PROC / f"grit_v10_{tag}_gauge_snaps.parquet"
    if snap_pq.exists():
        print(f"[cache] Loading {snap_pq}", flush=True)
        return pd.read_parquet(snap_pq)

    t0 = time.time()
    snapper = StreamingReachSnapper(
        gauge_ids=gauges_df["canonical_gauge_id"].to_numpy(),
        lons=gauges_df["longitude"].to_numpy(dtype=np.float64),
        lats=gauges_df["latitude"].to_numpy(dtype=np.float64),
        reported_area_km2=gauges_df["drainage_area_km2"].to_numpy(dtype=np.float64),
        max_search_km=10.0,
    )

    print(f"Snapping 119,169 gauges to GRIT v1.0 ({tag}, flip_orientation=False verified)...", flush=True)
    for reg in REGIONS:
        gpkg = GRIT_RAW / f"GRITv1.0_segments_{reg}_EPSG4326.gpkg"
        gdf = pyogrio.read_dataframe(
            str(gpkg),
            layer="lines",
            columns=["global_id", "is_mainstem", "drainage_area_in", "drainage_area_out",
                     "drainage_area_mainstem_in", "drainage_area_mainstem_out"],
        )
        if mainstem_only:
            gdf = gdf.loc[gdf["is_mainstem"].to_numpy() == 1]
            ain = np.maximum(gdf["drainage_area_mainstem_in"].fillna(0.0).to_numpy(dtype=np.float64), 0.0)
            aout = np.maximum(gdf["drainage_area_mainstem_out"].fillna(0.0).to_numpy(dtype=np.float64), 0.0)
        else:
            ain = np.maximum(gdf["drainage_area_in"].fillna(0.0).to_numpy(dtype=np.float64), 0.0)
            aout = np.maximum(gdf["drainage_area_out"].fillna(0.0).to_numpy(dtype=np.float64), 0.0)

        r_ids = gdf["global_id"].to_numpy(dtype=np.int64)
        geoms = gdf.geometry.to_numpy()
        ok = ~shapely.is_missing(geoms)
        pairs = snapper.update_chunk(r_ids[ok], geoms[ok], ain[ok], aout[ok], flip_orientation=None)
        print(f"  {reg}: {int(ok.sum()):,} segments -> {pairs:,} pairs", flush=True)

    snap_df = snapper.to_dataframe()

    # Fallback: match coastal gauges outside any >=50 km^2 segment to is_coastal==1 catchments
    unmatched = snap_df["snapped_unit_id"].to_numpy() < 0
    if np.any(unmatched):
        u_sub_g = gauges_df.loc[unmatched].reset_index(drop=True)
        psnap = StreamingPolygonSnapper(
            gauge_ids=u_sub_g["canonical_gauge_id"].to_numpy(),
            lons=u_sub_g["longitude"].to_numpy(dtype=np.float64),
            lats=u_sub_g["latitude"].to_numpy(dtype=np.float64),
            reported_area_km2=u_sub_g["drainage_area_km2"].to_numpy(dtype=np.float64),
            max_fallback_km=5.0,
        )
        cgpkg = GRIT_RAW / "GRITv1.0_component_catchments_GLOBAL_EPSG4326.gpkg"
        clyr = str(pyogrio.list_layers(str(cgpkg))[0][0])
        cgdf = pyogrio.read_dataframe(str(cgpkg), layer=clyr, columns=["global_id", "area", "is_coastal", "is_sink"])
        coast_ids = set(all_df.loc[all_df["is_coastal_catchment"].to_numpy(), "catchment_id"].to_numpy())
        cgdf = cgdf.loc[cgdf["global_id"].isin(coast_ids).to_numpy()]
        if len(cgdf) > 0:
            cids = 800_000_000_000 + cgdf["global_id"].to_numpy(dtype=np.int64)
            areas = np.maximum(cgdf["area"].fillna(0.0).to_numpy(dtype=np.float64), 0.0)
            psnap.update_chunk(cids, cgdf.geometry.to_numpy(), areas, areas)
        csnap = psnap.to_dataframe(use_area_tiebreak=False)
        hit = csnap["snapped_unit_id"].to_numpy() >= 0
        if np.any(hit):
            unm_idx = np.flatnonzero(unmatched)[hit]
            snap_df.loc[unm_idx, "snapped_unit_id"] = csnap.loc[hit, "snapped_unit_id"].to_numpy()
            snap_df.loc[unm_idx, "snap_dist_km"] = csnap.loc[hit, "snap_dist_km"].to_numpy()
            snap_df.loc[unm_idx, "snapped_area_in_km2"] = 0.0
            snap_df.loc[unm_idx, "snapped_area_out_km2"] = csnap.loc[hit, "snapped_area_out_km2"].to_numpy()
            snap_df.loc[unm_idx, "effective_gauged_area_km2"] = csnap.loc[hit, "effective_gauged_area_km2"].to_numpy()
            print(f"  Matched {int(hit.sum()):,} additional coastal gauges to GRIT coastal catchments", flush=True)

    n_snapped = int((snap_df["snapped_unit_id"] >= 0).sum())
    med_d = float(np.nanmedian(snap_df.loc[snap_df["snapped_unit_id"] >= 0, "snap_dist_km"]))
    med_err = float(np.nanmedian(snap_df["log10_area_err"]))
    print(
        f"  GRIT ({tag}) snapping done in {time.time() - t0:.1f}s: {n_snapped:,}/{len(snap_df):,} snapped "
        f"(median dist={med_d:.3f} km, median |log10 area err|={med_err:.3f})",
        flush=True,
    )
    snap_df.to_parquet(snap_pq, index=False)
    return snap_df


def run_grit_mainstem_tree(all_df: pd.DataFrame, gauges_df: pd.DataFrame, ood_l12: pd.DataFrame | None) -> None:
    """Run GRIT v1.0 single-threaded mainstem tree projection (no flow metrics)."""
    df = all_df.loc[all_df["is_mainstem"].to_numpy()].reset_index(drop=True)
    n = len(df)
    is_seg = ~df["is_coastal_catchment"].to_numpy()

    # For each upstream node, map to the single outgoing mainstem segment (if >1 due to tie, pick largest da_main_out)
    seg_idx = np.flatnonzero(is_seg)
    up_nodes = df["upstream_node_id"].to_numpy(dtype=np.int64)[seg_idx]
    da_main_out = df["da_main_out_km2"].to_numpy(dtype=np.float64)
    # Sort descending by da_main_out so first wins per upstream_node_id
    sort_o = np.argsort(-da_main_out[seg_idx])
    node_to_out_seg: dict[int, int] = {}
    for s in sort_o:
        idx_u = int(seg_idx[s])
        un = int(up_nodes[s])
        if un not in node_to_out_seg:
            node_to_out_seg[un] = idx_u

    next_down = np.full(n, -1, dtype=np.int32)
    dn_nodes = df["downstream_node_id"].to_numpy(dtype=np.int64)
    for idx_u in seg_idx:
        nxt = node_to_out_seg.get(int(dn_nodes[idx_u]), -1)
        if nxt != idx_u:
            next_down[idx_u] = nxt

    n_broken = break_tree_cycles_inplace(next_down, da_main_out)
    order, n_cyc = topo_sort_tree(next_down)
    assert n_cyc == 0

    # Exact tree local area conservation using GRIT's verified local incremental area
    # (avoids double-counting giant upstream basins when a multi-outgoing mainstem node disconnects a branch)
    local_a = df["local_area_dag_km2"].to_numpy(dtype=np.float64)
    _, a_out_acc = accumulate_tree(order, next_down, local_a)
    a_in_1d = np.minimum(df["da_main_in_km2"].to_numpy(dtype=np.float64), a_out_acc)

    print(
        f"[grit_v10_mainstem_tree] n={n:,}, broken_cyc={n_broken}, "
        f"total land={local_a.sum()/1e6:.2f}M km2, length={df['length_km'].sum()/1e6:.2f}M km",
        flush=True,
    )

    # Re-transfer exposure onto mainstem-only units so population & EAE are 100% mass-conserved
    l12_ref = PROCESSED_DIR / "hydrosheds" / "hydrosheds_l12_enrichment_ref.parquet"
    pop_m, eae_m, ari_m = transfer_exposure_and_aridity_from_l12(
        df["hybas_l12_id"].to_numpy(dtype=np.int64),
        local_a,
        l12_aridity_parquet=l12_ref if l12_ref.exists() else None,
        unit_lons=df["lon"].to_numpy(dtype=np.float64),
        unit_lats=df["lat"].to_numpy(dtype=np.float64),
    )
    eae_msg = f", EAE={eae_m.sum()/1e6:.2f}M/yr" if eae_m is not None else ""
    print(
        f"[grit_v10_mainstem_tree] enrichment: pop={pop_m.sum()/1e9:.3f}B{eae_msg}",
        flush=True,
    )

    units_cols = {
        "unit_id": df["unit_id"].to_numpy(dtype=np.int64),
        "next_down_idx": next_down,
        "local_area_km2": local_a,
        "up_area_in_km2": a_in_1d,
        "up_area_out_km2": a_out_acc,
        "length_km": df["length_km"].to_numpy(dtype=np.float64),
        "population": pop_m,
        "lon": df["lon"].to_numpy(dtype=np.float32),
        "lat": df["lat"].to_numpy(dtype=np.float32),
        "country_iso3": df["country_iso3"].to_numpy(),
        "continent": df["continent"].to_numpy(),
        "in_common_domain": df["in_common_domain"].to_numpy(dtype=bool),
        "aridity_zone": ari_m,
        "hybas_l12_id": df["hybas_l12_id"].to_numpy(dtype=np.int64),
    }
    if eae_m is not None:
        units_cols["exposure_eae"] = eae_m
    units_df = pd.DataFrame(units_cols)
    snap_df = snap_gauges_grit(all_df, gauges_df, mainstem_only=True)
    ood_feats = ood_l12.reindex(df["hybas_l12_id"].to_numpy()).to_numpy(dtype=np.float64) if ood_l12 is not None else None

    run_tree_hydrofabric_suite(
        fabric_id="grit_v10_mainstem_tree",
        units_df=units_df,
        order=order,
        next_down=next_down,
        gauges_df=gauges_df,
        snap_df=snap_df,
        has_flow=False,  # GRIT has no global operational flow model
        run_1d=True,
        ood_features=ood_feats,
    )


def run_grit_bifurcating_dag(all_df: pd.DataFrame, gauges_df: pd.DataFrame) -> None:
    """Run GRIT v1.0 full multi-threaded bifurcating DAG with width-weighted partitioning."""
    t0 = time.time()
    out_dir = RESULTS_DIR / "grit_v10_bifurcating_dag"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_units = len(all_df)
    is_seg = ~all_df["is_coastal_catchment"].to_numpy()
    seg_mask_idx = np.flatnonzero(is_seg)
    coast_mask_idx = np.flatnonzero(~is_seg)

    # Remap GRIT node IDs to compact 0..n_nodes-1 (plus dummy virtual nodes for coastal polygons)
    seg_u_raw = all_df["upstream_node_id"].to_numpy(dtype=np.int64)[seg_mask_idx]
    seg_v_raw = all_df["downstream_node_id"].to_numpy(dtype=np.int64)[seg_mask_idx]
    uniq_nodes = np.unique(np.concatenate([seg_u_raw, seg_v_raw]))
    node_map = pd.Series(np.arange(len(uniq_nodes), dtype=np.int32), index=uniq_nodes)

    n_real_nodes = len(uniq_nodes)
    n_coast = len(coast_mask_idx)
    # Give each coastal polygon a virtual single-edge (u_c -> v_c) so it participates seamlessly in DAG arrays
    n_nodes = n_real_nodes + 2 * n_coast

    edge_u = np.empty(n_units, dtype=np.int32)
    edge_v = np.empty(n_units, dtype=np.int32)
    edge_u[seg_mask_idx] = node_map.reindex(seg_u_raw).to_numpy(dtype=np.int32)
    edge_v[seg_mask_idx] = node_map.reindex(seg_v_raw).to_numpy(dtype=np.int32)
    edge_u[coast_mask_idx] = np.arange(n_real_nodes, n_real_nodes + n_coast, dtype=np.int32)
    edge_v[coast_mask_idx] = np.arange(n_real_nodes + n_coast, n_nodes, dtype=np.int32)

    node_ord, _, n_cyc = topo_sort_dag_edges(n_nodes, edge_u, edge_v)
    print(f"[grit_v10_bifurcating_dag] nodes={n_nodes:,}, edges={n_units:,}, cycle_nodes={n_cyc}", flush=True)

    # Build CSR outgoing edges per node & width-weighted bifurcation split fractions
    out_cnt = np.bincount(edge_u, minlength=n_nodes).astype(np.int32)
    out_ptr = np.zeros(n_nodes + 1, dtype=np.int32)
    np.cumsum(out_cnt, out=out_ptr[1:])
    order_by_u = np.argsort(edge_u, kind="mergesort").astype(np.int32)
    out_edges = order_by_u

    widths = all_df["width_m"].to_numpy(dtype=np.float64)
    node_w_sum = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(node_w_sum, edge_u, widths)
    edge_split_w = widths / np.maximum(node_w_sum[edge_u], 1e-12)

    a_in = all_df["da_in_km2"].to_numpy(dtype=np.float64)
    a_out = all_df["da_out_km2"].to_numpy(dtype=np.float64)
    local_a = all_df["local_area_dag_km2"].to_numpy(dtype=np.float64)
    length_km = all_df["length_km"].to_numpy(dtype=np.float64)
    pop = all_df["population"].to_numpy(dtype=np.float64)
    eae = all_df["exposure_eae"].to_numpy(dtype=np.float64) if "exposure_eae" in all_df.columns else None

    unit_ids = all_df["unit_id"].to_numpy(dtype=np.int64)
    uid_to_idx = {int(u): i for i, u in enumerate(unit_ids)}
    snap_df = snap_gauges_grit(all_df, gauges_df, mainstem_only=False)

    weights_2d = {
        "count": np.ones(n_units, dtype=np.float64),
        "area": local_a,
        "population": pop,
        "exposure_eae": eae,
    }
    in_common = all_df["in_common_domain"].to_numpy(dtype=bool)
    continents = all_df["continent"].astype(str).to_numpy()
    aridity = all_df["aridity_zone"].to_numpy(dtype=np.int8)

    macro_domains = [
        ("global", "full_native", np.ones(n_units, dtype=bool)),
        ("global", "common_no_grl_ata", in_common),
    ]
    for cont in ["Africa", "Asia", "Europe", "North America", "South America", "Oceania"]:
        cm = in_common & (continents == cont)
        if np.any(cm):
            macro_domains.append(("continent", cont, cm))
    for code, label in ARIDITY_LABELS.items():
        if code != 0:
            am = in_common & (aridity == code)
            if np.any(am):
                macro_domains.append(("aridity_zone", label, am))

    rows_2d: list[dict] = []
    rows_1d: list[dict] = []
    rows_country: list[dict] = []

    countries = all_df["country_iso3"].astype(str).to_numpy()
    uniq_countries = sorted(c for c in set(countries[in_common]) if c not in ("UNK", "-99", "nan", "ATA", "GRL"))
    c_series = pd.Series(countries)
    c_groups = c_series.groupby(c_series).indices

    for tier_key, _tlabel, tier_col in GAUGE_TIERS:
        has_g, _eff_a, _eff_q, g_u_idx, g_fracs = aggregate_gauge_snaps_to_units(
            n_units, uid_to_idx, gauges_df, snap_df, tier_col, local_a, a_out
        )
        n_snap = int(len(g_u_idx))
        g_in, g_out, p_nev, inv_ad, inv_ia = propagate_dag_gauges(
            n_nodes, node_ord, edge_u, edge_v, out_ptr, out_edges,
            edge_split_w, a_in, a_out, has_g,
        )

        # Country breakdown for GRIT bifurcating DAG
        c_up = np.where(a_out > 0.0, np.clip(g_out / np.where(a_out > 0.0, a_out, 1.0), 0.0, 1.0), 0.0)
        c_down = np.where(has_g == 1, 1.0, np.clip(a_out * inv_ad, 0.0, 1.0))
        u_net = (1.0 - c_up) * (1.0 - c_down)
        f_nev = np.where(has_g == 1, 0.0, p_nev)
        eff_inv_ia = np.where(
            has_g == 1,
            np.where(a_out - g_in > 0.0, 1.0 / np.where(a_out - g_in > 0.0, a_out - g_in, 1.0), np.inf),
            inv_ia,
        )
        w_country = {"area": local_a}
        if eae is not None:
            w_country["exposure_eae"] = eae
        for iso in uniq_countries:
            c_idx = c_groups.get(iso)
            if c_idx is None or len(c_idx) == 0:
                continue
            for w_name, w_arr in w_country.items():
                w = w_arr[c_idx]
                wt = float(np.sum(w))
                if wt <= 0.0:
                    continue
                inv_c = eff_inv_ia[c_idx]
                rows_country.append({
                    "hydrofabric": "grit_v10_bifurcating_dag",
                    "tier": tier_key,
                    "country_iso3": iso,
                    "weight": w_name,
                    "total_weight": wt,
                    "U_out": float(np.dot(w, (has_g[c_idx] == 0).astype(np.float64)) / wt),
                    "U_up": float(np.dot(w, 1.0 - c_up[c_idx]) / wt),
                    "U_down": float(np.dot(w, 1.0 - c_down[c_idx]) / wt),
                    "U_net_Eq4": float(np.dot(w, u_net[c_idx]) / wt),
                    "U_never": float(np.dot(w, f_nev[c_idx]) / wt),
                    "U_cont_1k": float(1.0 - np.dot(w, np.minimum(1.0, 1_000.0 * inv_c)) / wt),
                    "U_cont_10k": float(1.0 - np.dot(w, np.minimum(1.0, 10_000.0 * inv_c)) / wt),
                })

        r2 = compute_2d_metrics_dag_multi_domain(
            local_a, a_out, has_g, g_in, g_out, p_nev, inv_ad, inv_ia,
            weights_dict=weights_2d, domains=macro_domains, area_scale_grid=AREA_SCALE_GRID_KM2,
        )
        for r in r2:
            r["hydrofabric"] = "grit_v10_bifurcating_dag"
            r["tier"] = tier_key
            r["n_gauges_snapped"] = n_snap
        rows_2d.extend(r2)

        r1 = compute_1d_metrics_dag_multi_domain(
            length_km, a_in, a_out, g_in, inv_ad,
            gauge_reach_idx=g_u_idx, domains=macro_domains, gauge_frac_pos=g_fracs,
            exposure_eae=eae, area_bins=RIVER_AREA_BINS_KM2,
        )
        for r in r1:
            r["hydrofabric"] = "grit_v10_bifurcating_dag"
            r["tier"] = tier_key
            r["n_gauges_snapped"] = n_snap
        rows_1d.extend(r1)

        print(f"[grit_v10_bifurcating_dag] Finished tier={tier_key} ({n_snap:,} snapped)", flush=True)

    pd.DataFrame(rows_2d).to_parquet(out_dir / "metrics_2d_global.parquet", index=False)
    pd.DataFrame(rows_1d).to_parquet(out_dir / "metrics_1d_global.parquet", index=False)
    if rows_country:
        pd.DataFrame(rows_country).to_parquet(out_dir / "metrics_2d_country.parquet", index=False)
    print(f"[grit_v10_bifurcating_dag] Saved to {out_dir} in {time.time() - t0:.1f}s", flush=True)


def main() -> None:
    gauges_df = pd.read_parquet(GAUGES_DIR / "global_gauges.parquet")
    all_df = build_grit_tables()
    l12_ood_pq = PROCESSED_DIR / "hydrosheds" / "hydrosheds_l12_ood_features.parquet"
    ood_l12 = pd.read_parquet(l12_ood_pq).set_index("HYBAS_ID") if l12_ood_pq.exists() else None
    run_grit_mainstem_tree(all_df, gauges_df, ood_l12)
    run_grit_bifurcating_dag(all_df, gauges_df)


if __name__ == "__main__":
    main()
