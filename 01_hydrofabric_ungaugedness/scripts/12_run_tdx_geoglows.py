"""End-to-end ETL, gauge snapping, and full metric computation for TDX-Hydro / GEOGLOWS v2.

Inputs (all self-contained in data/):
  - data/raw/tdx_geoglows/v2-model-table.parquet         (6,838,900 reaches)
  - data/raw/tdx_geoglows/v2-countries-table.parquet     (lon, lat, RiverCountry)
  - data/raw/tdx_geoglows/global_streams_simplified.gpkg (6.84M stream polylines)
  - data/raw/tdx_geoglows/geoglows_v2_mean_annual_q.parquet (1940-2024 mean Q)
  - data/gauges/global_gauges.parquet
  - data/processed/hydrosheds_l12_enrichment_ref.parquet (optional; created by script 11)
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

from configs.benchmark_config import GAUGES_DIR, PROCESSED_DIR, RAW_DIR
from global_ungaugedness.graph import (
    accumulate_tree,
    break_tree_cycles_inplace,
    topo_sort_tree,
)
from global_ungaugedness.runner import run_tree_hydrofabric_suite
from global_ungaugedness.snapping import StreamingReachSnapper
from global_ungaugedness.spatial_enrichment import (
    assign_country_continent_and_domain,
    map_points_to_hydrosheds_l12,
    transfer_exposure_and_aridity_from_l12,
)

TDX_RAW = RAW_DIR / "tdx_geoglows"
OUT_PROC = PROCESSED_DIR / "tdx_hydro"


def detect_stream_flip(
    gpkg_path: Path, layer: str, id_col: str, u_ids: np.ndarray, u_dn: np.ndarray, idx_series: pd.Series
) -> bool:
    """Empirically verify whether polylines are stored upstream->downstream (False) or reversed (True)."""
    # Pick 500 reaches that actually have a downstream neighbor, skipping headwaters (take middle of table)
    has_d = np.flatnonzero(u_dn >= 0)
    # Sample evenly across the network so we hit LineString mainstems (not just order-1 MultiLineStrings)
    step = max(1, len(has_d) // 500)
    sel = has_d[::step][:500]
    u_sel = u_ids[sel]
    d_sel = u_ids[u_dn[sel]]
    all_ids = sorted(set(u_sel.tolist() + d_sel.tolist()))
    id_csv = ",".join(str(x) for x in all_ids)
    sub = pyogrio.read_dataframe(str(gpkg_path), layer=layer, where=f"{id_col} IN ({id_csv})")
    geom_map = {
        int(uid): geom
        for uid, geom in zip(sub[id_col].to_numpy(), sub.geometry.to_numpy())
        if geom is not None and geom.geom_type == "LineString"
    }

    d_fwd, d_rev = [], []
    for uid, dn in zip(u_sel.tolist(), d_sel.tolist()):
        if uid in geom_map and dn in geom_map:
            cu = shapely.get_coordinates(geom_map[uid])
            cd = shapely.get_coordinates(geom_map[dn])
            if len(cu) >= 2 and len(cd) >= 2:
                d_fwd.append(float(np.hypot(*(cu[-1] - cd[0]))))
                d_rev.append(float(np.hypot(*(cu[0] - cd[-1]))))
    if not d_fwd:
        raise RuntimeError("Could not detect TDX polyline orientation from sampled parent-child pairs!")
    med_fwd = float(np.median(d_fwd))
    med_rev = float(np.median(d_rev))
    flip = med_rev < med_fwd
    print(
        f"  TDX polyline orientation check (n={len(d_fwd)} LineString pairs): "
        f"median(end->dn_start)={med_fwd:.8f}, median(start->dn_end)={med_rev:.8f} -> flip={flip}",
        flush=True,
    )
    return flip


def build_tdx_graph() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    OUT_PROC.mkdir(parents=True, exist_ok=True)
    graph_pq = OUT_PROC / "tdx_hydro_units.parquet"
    if graph_pq.exists():
        print(f"[cache] Loading {graph_pq}", flush=True)
        df = pd.read_parquet(graph_pq)
        nd = df["next_down_idx"].to_numpy(dtype=np.int32)
        order, n_cyc = topo_sort_tree(nd)
        assert n_cyc == 0
        return df, order, nd

    t0 = time.time()
    print("Building unified TDX-Hydro + GEOGLOWS v2 graph...", flush=True)
    model = pd.read_parquet(
        TDX_RAW / "v2-model-table.parquet",
        columns=["LINKNO", "DSLINKNO", "strmOrder", "USContArea", "DSContArea", "LengthGeodesicMeters", "VPUCode"],
    )
    countries = pd.read_parquet(
        TDX_RAW / "v2-countries-table.parquet",
        columns=["LINKNO", "lon", "lat"],
    )
    q_df = pd.read_parquet(
        TDX_RAW / "geoglows_v2_mean_annual_q.parquet",
        columns=["LINKNO", "q_mean_m3s"],
    )

    df = model.merge(countries, on="LINKNO", how="left").merge(q_df, on="LINKNO", how="left")
    n = len(df)
    link = df["LINKNO"].to_numpy(dtype=np.int64)
    ds = df["DSLINKNO"].to_numpy(dtype=np.int64)

    id_map = pd.Series(np.arange(n, dtype=np.int32), index=link)
    next_down = id_map.reindex(ds).fillna(-1).to_numpy(dtype=np.int32)

    ds_area_km2 = np.maximum(df["DSContArea"].to_numpy(dtype=np.float64) / 1e6, 0.0)
    us_area_km2 = np.maximum(df["USContArea"].to_numpy(dtype=np.float64) / 1e6, 0.0)
    length_km = np.maximum(df["LengthGeodesicMeters"].to_numpy(dtype=np.float64) / 1e3, 0.0)

    n_broken = break_tree_cycles_inplace(next_down, ds_area_km2)
    order, n_cyc = topo_sort_tree(next_down)
    assert n_cyc == 0, f"TDX still has {n_cyc} cycle nodes!"
    print(f"  TDX topology sorted: n={n:,}, broken_cycles={n_broken}", flush=True)

    # Exact 2D local catchment area: DSContArea - sum(upstream children DSContArea)
    # Properly retains headwater drainage area (USContArea) on order-1 streams!
    sum_child_a = np.zeros(n, dtype=np.float64)
    has_dn = next_down >= 0
    np.add.at(sum_child_a, next_down[has_dn], ds_area_km2[has_dn])
    local_area = np.maximum(ds_area_km2 - sum_child_a, 0.0)
    _, a_out_acc = accumulate_tree(order, next_down, local_area)
    # 1D physical channel area along reach: [us_area_km2, max(ds_area_km2, us_area_km2)]
    a_in_1d = np.minimum(us_area_km2, a_out_acc)
    print(
        f"  TDX area closure: sum(local)={local_area.sum()/1e6:.2f}M km2, "
        f"sum(terminal DSContArea)={ds_area_km2[~has_dn].sum()/1e6:.2f}M km2, "
        f"total length={length_km.sum()/1e6:.2f}M km",
        flush=True,
    )

    # GEOGLOWS v2 1940-2024 mean annual discharge -> physical Q and nonnegative routed Q
    q_phys_out = np.where(
        np.isfinite(df["q_mean_m3s"].to_numpy(dtype=np.float64)) & (df["q_mean_m3s"].to_numpy(dtype=np.float64) >= 0.0),
        df["q_mean_m3s"].to_numpy(dtype=np.float64),
        0.0,
    )
    q_phys_in = np.zeros(n, dtype=np.float64)
    np.add.at(q_phys_in, next_down[has_dn], q_phys_out[has_dn])
    local_runoff = np.maximum(q_phys_out - q_phys_in, 0.0)
    q_cons_in, q_cons_out = accumulate_tree(order, next_down, local_runoff)
    print(
        f"  GEOGLOWS Q closure: sum(terminal Q_phys)={q_phys_out[~has_dn].sum()/1e6:.3f}M m3/s, "
        f"sum(local_runoff)={local_runoff.sum()/1e6:.3f}M m3/s, "
        f"transmission-loss reaches={(q_phys_out < q_phys_in - 1e-6).sum():,}",
        flush=True,
    )

    lons = df["lon"].to_numpy(dtype=np.float64)
    lats = df["lat"].to_numpy(dtype=np.float64)
    iso3, cont, in_common = assign_country_continent_and_domain(lons, lats)

    # Mass-conserving transfer of WorldPop population, Fathom EAE, and HydroATLAS aridity
    hybas_l12 = map_points_to_hydrosheds_l12(lons, lats)
    l12_ref = PROCESSED_DIR / "hydrosheds" / "hydrosheds_l12_enrichment_ref.parquet"
    pop, eae, aridity = transfer_exposure_and_aridity_from_l12(
        hybas_l12,
        local_area,
        l12_aridity_parquet=l12_ref if l12_ref.exists() else None,
        unit_lons=lons,
        unit_lats=lats,
    )
    eae_msg = f", EAE={eae.sum()/1e6:.2f}M/yr" if eae is not None else ""
    print(
        f"  TDX enrichment: pop={pop.sum()/1e9:.3f}B{eae_msg}, "
        f"common_domain_area={local_area[in_common].sum()/1e6:.2f}M km2",
        flush=True,
    )

    out_cols = {
        "unit_id": link,
        "next_down_idx": next_down,
        "local_area_km2": local_area,
        "up_area_in_km2": a_in_1d,
        "up_area_out_km2": a_out_acc,
        "length_km": length_km,
        "local_runoff_m3s": local_runoff,
        "q_cons_in_m3s": q_cons_in,
        "q_cons_out_m3s": q_cons_out,
        "q_phys_in_m3s": q_phys_in,
        "q_phys_out_m3s": q_phys_out,
        "population": pop,
        "lon": lons.astype(np.float32),
        "lat": lats.astype(np.float32),
        "country_iso3": iso3,
        "continent": cont,
        "in_common_domain": in_common,
        "aridity_zone": aridity,
        "hybas_l12_id": hybas_l12,
    }
    if eae is not None:
        out_cols["exposure_eae"] = eae
    out_df = pd.DataFrame(out_cols)
    out_df.to_parquet(graph_pq, index=False)
    print(f"Saved {graph_pq} in {time.time() - t0:.1f}s", flush=True)
    return out_df, order, next_down


def snap_gauges_tdx(units_df: pd.DataFrame, gauges_df: pd.DataFrame) -> pd.DataFrame:
    snap_pq = OUT_PROC / "tdx_hydro_gauge_snaps.parquet"
    if snap_pq.exists():
        print(f"[cache] Loading {snap_pq}", flush=True)
        return pd.read_parquet(snap_pq)

    t0 = time.time()
    gpkg = TDX_RAW / "global_streams_simplified.gpkg"
    layers = pyogrio.list_layers(str(gpkg))
    layer_name = str(layers[0][0])
    info = pyogrio.read_info(str(gpkg), layer=layer_name)
    n_feat = int(info["features"])
    fields = list(info["fields"])
    id_col = next((c for c in ["LINKNO", "linkno", "river_id"] if c in fields), fields[0])
    print(f"Snapping 119,169 gauges to TDX-Hydro ({n_feat:,} features, layer={layer_name}, id={id_col})...", flush=True)

    # Build lookup arrays for area_in and area_out by LINKNO
    u_ids = units_df["unit_id"].to_numpy(dtype=np.int64)
    u_ain = units_df["up_area_in_km2"].to_numpy(dtype=np.float64)
    u_aout = units_df["up_area_out_km2"].to_numpy(dtype=np.float64)
    u_dn = units_df["next_down_idx"].to_numpy(dtype=np.int32)
    idx_series = pd.Series(np.arange(len(u_ids), dtype=np.int32), index=u_ids)

    # Detect polyline orientation from first 5,000 features
    flip = detect_stream_flip(gpkg, layer_name, id_col, u_ids, u_dn, idx_series)

    snapper = StreamingReachSnapper(
        gauge_ids=gauges_df["canonical_gauge_id"].to_numpy(),
        lons=gauges_df["longitude"].to_numpy(dtype=np.float64),
        lats=gauges_df["latitude"].to_numpy(dtype=np.float64),
        reported_area_km2=gauges_df["drainage_area_km2"].to_numpy(dtype=np.float64),
        max_search_km=10.0,
    )

    chunk = 250_000
    total_pairs = 0
    for start in range(0, n_feat, chunk):
        gdf = pyogrio.read_dataframe(
            str(gpkg),
            layer=layer_name,
            columns=[id_col],
            skip_features=start,
            max_features=chunk,
        )
        if len(gdf) == 0:
            break
        r_ids = gdf[id_col].to_numpy(dtype=np.int64)
        geoms = gdf.geometry.to_numpy()
        valid_g = ~shapely.is_missing(geoms)
        r_ids = r_ids[valid_g]
        geoms = geoms[valid_g]

        pos = idx_series.reindex(r_ids).fillna(-1).to_numpy(dtype=np.int32)
        ok = pos >= 0
        r_ids = r_ids[ok]
        geoms = geoms[ok]
        pos = pos[ok]

        fl_arr = np.full(len(r_ids), flip, dtype=bool)
        pairs = snapper.update_chunk(r_ids, geoms, u_ain[pos], u_aout[pos], flip_orientation=fl_arr)
        total_pairs += pairs
        if (start // chunk) % 6 == 0 or start + chunk >= n_feat:
            print(f"  processed {min(start + chunk, n_feat):,}/{n_feat:,} reaches ({total_pairs:,} pairs)", flush=True)

    snap_df = snapper.to_dataframe()
    n_snapped = int((snap_df["snapped_unit_id"] >= 0).sum())
    med_d = float(np.nanmedian(snap_df.loc[snap_df["snapped_unit_id"] >= 0, "snap_dist_km"]))
    med_err = float(np.nanmedian(snap_df["log10_area_err"]))
    print(
        f"  TDX snapping done in {time.time() - t0:.1f}s: {n_snapped:,}/{len(snap_df):,} snapped "
        f"(median dist={med_d:.3f} km, median |log10 area err|={med_err:.3f})",
        flush=True,
    )
    snap_df.to_parquet(snap_pq, index=False)
    return snap_df


def main() -> None:
    gauges_df = pd.read_parquet(GAUGES_DIR / "global_gauges.parquet")
    units_df, order, next_down = build_tdx_graph()
    snap_df = snap_gauges_tdx(units_df, gauges_df)

    # Load OOD hydroclimatic features transferred via HYBAS_L12 if available
    ood_feats = None
    l12_ood_pq = PROCESSED_DIR / "hydrosheds" / "hydrosheds_l12_ood_features.parquet"
    if l12_ood_pq.exists():
        l12_ood = pd.read_parquet(l12_ood_pq).set_index("HYBAS_ID")
        ood_feats = l12_ood.reindex(units_df["hybas_l12_id"].to_numpy()).to_numpy(dtype=np.float64)

    run_tree_hydrofabric_suite(
        fabric_id="tdx_hydro",
        units_df=units_df,
        order=order,
        next_down=next_down,
        gauges_df=gauges_df,
        snap_df=snap_df,
        has_flow=True,
        run_1d=True,
        ood_features=ood_feats,
    )


if __name__ == "__main__":
    main()
