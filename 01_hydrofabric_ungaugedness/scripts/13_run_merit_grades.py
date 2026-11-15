"""End-to-end ETL, gauge snapping, and full metric computation for MERIT-Hydro / MERIT-Basins.

Native global flow model:
  - Default: GRADES v0.1 (Lin et al., 2019, WRR) 1979-2013 35-yr mean annual runoff
    routed cleanly through all ~2.94M MERIT-Basins reaches (+ coastal hillslopes).
  - Pluggable adapter: pass `--flow-model todays_earth --flow-file <path.nc|parquet>`
    to swap in JAXA Today's Earth (CaMa-Flood) mean annual runoff or reach discharge
    without modifying any metric code.
"""

from __future__ import annotations

import os
os.environ.setdefault("OGR_ORGANIZE_POLYGONS", "SKIP")

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
import xarray as xr
from scipy.spatial import cKDTree

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
from global_ungaugedness.snapping import StreamingPolygonSnapper, StreamingReachSnapper
from global_ungaugedness.spatial_enrichment import (
    assign_country_continent_and_domain,
    map_points_to_hydrosheds_l12,
    transfer_exposure_and_aridity_from_l12,
)

MERIT_RAW = RAW_DIR / "merit"
FLOW_RAW = RAW_DIR / "flow_models"
OUT_PROC = PROCESSED_DIR / "merit_basins"

# 1 mm/day over 1 km^2 = 1e-3 m * 1e6 m^2 / 86400 s = 0.011574074 m^3/s
MM_DAY_KM2_TO_M3S = 1000.0 / 86400.0


def sample_gridded_runoff_mm_day(
    lons: np.ndarray, lats: np.ndarray, nc_path: Path, var_name: str = "runoff_mean_mm_day"
) -> np.ndarray:
    """Sample 0.25-deg global runoff grid at (lon, lat), with nearest-valid-land fallback for coasts."""
    with xr.open_dataset(nc_path) as ds:
        grid_lat = ds["lat"].values.astype(np.float64)
        grid_lon = ds["lon"].values.astype(np.float64)
        vals = ds[var_name].values.astype(np.float64)  # (nlat, nlon)

    # Handle lon convention [0, 360) vs [-180, 180]
    if grid_lon.max() > 180.0:
        q_lon = np.mod(lons, 360.0)
    else:
        q_lon = np.where(lons > 180.0, lons - 360.0, lons)

    dlat = float(grid_lat[1] - grid_lat[0])
    dlon = float(grid_lon[1] - grid_lon[0])
    iy = np.clip(np.rint((lats - grid_lat[0]) / dlat).astype(np.int32), 0, grid_lat.size - 1)
    ix = np.clip(np.rint((q_lon - grid_lon[0]) / dlon).astype(np.int32), 0, grid_lon.size - 1)

    sampled = vals[iy, ix]

    # Nearest valid land cell fallback (within ~1.5 deg) for coastal pixels / Arctic edges
    miss = ~np.isfinite(sampled)
    if np.any(miss):
        v_iy, v_ix = np.nonzero(np.isfinite(vals))
        v_lat = grid_lat[v_iy]
        v_lon = grid_lon[v_ix]
        # Convert to 3D unit sphere for wrap-around-safe nearest neighbor
        def to_xyz(lo_d: np.ndarray, la_d: np.ndarray) -> np.ndarray:
            lr = np.deg2rad(lo_d)
            ar = np.deg2rad(la_d)
            ca = np.cos(ar)
            return np.column_stack([ca * np.cos(lr), ca * np.sin(lr), np.sin(ar)])

        tree = cKDTree(to_xyz(v_lon, v_lat))
        _, nn = tree.query(to_xyz(q_lon[miss], lats[miss]), k=1, workers=4)
        sampled[miss] = vals[v_iy[nn], v_ix[nn]]

    return np.maximum(np.where(np.isfinite(sampled), sampled, 0.0), 0.0)


def build_merit_graph(flow_model: str = "grades", flow_file: Path | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    OUT_PROC.mkdir(parents=True, exist_ok=True)
    graph_pq = OUT_PROC / f"merit_basins_units_{flow_model}.parquet"
    if graph_pq.exists():
        print(f"[cache] Loading {graph_pq}", flush=True)
        df = pd.read_parquet(graph_pq)
        nd = df["next_down_idx"].to_numpy(dtype=np.int32)
        order, n_cyc = topo_sort_tree(nd)
        assert n_cyc == 0
        return df, order, nd

    t0 = time.time()
    p1_dir = MERIT_RAW / "pfaf_level_01"
    riv_dfs = []
    cat_dfs = []

    print("Loading MERIT-Basins pfaf 1..9 rivers & catchments...", flush=True)
    for pf in range(1, 10):
        riv_shp = sorted(p1_dir.glob(f"riv_pfaf_{pf}_*.shp"))[0]
        cat_shp = sorted(p1_dir.glob(f"cat_pfaf_{pf}_*.shp"))[0]
        rdf = pyogrio.read_dataframe(
            str(riv_shp),
            columns=["COMID", "lengthkm", "uparea", "NextDownID"],
        )
        # Fast midpoint coordinate from river polyline bounds
        b = shapely.bounds(rdf.geometry.to_numpy())
        rdf_out = pd.DataFrame({
            "COMID": rdf["COMID"].to_numpy(dtype=np.int64),
            "NextDownID": rdf["NextDownID"].to_numpy(dtype=np.int64),
            "length_km": np.maximum(rdf["lengthkm"].to_numpy(dtype=np.float64), 0.0),
            "uparea_raw": np.maximum(rdf["uparea"].to_numpy(dtype=np.float64), 0.0),
            "lon": (0.5 * (b[:, 0] + b[:, 2])).astype(np.float64),
            "lat": (0.5 * (b[:, 1] + b[:, 3])).astype(np.float64),
        })
        riv_dfs.append(rdf_out)

        cdf = pyogrio.read_dataframe(str(cat_shp), columns=["COMID", "unitarea"], read_geometry=False)
        cat_dfs.append(cdf)
        print(f"  pfaf_{pf}: {len(rdf_out):,} reaches, {len(cdf):,} unit catchments", flush=True)

    riv_all = pd.concat(riv_dfs, ignore_index=True)
    cat_all = pd.concat(cat_dfs, ignore_index=True).drop_duplicates("COMID")
    merged = riv_all.merge(cat_all, on="COMID", how="left")
    merged["unitarea"] = merged["unitarea"].fillna(0.0).clip(lower=0.0)
    merged["is_coastal_hillslope"] = False

    # Load MERIT coastal hillslopes (direct-to-ocean land without channelized reach)
    hs_dir = MERIT_RAW / "coastal_hillslopes"
    hs_shps = sorted(hs_dir.glob("*.shp"))
    hs_rows = []
    next_hs_id = 900_000_000_000  # non-overlapping synthetic ID block for coastal hillslopes
    if hs_shps:
        print(f"Loading {len(hs_shps)} MERIT coastal hillslope shapefiles...", flush=True)
        for shp in hs_shps:
            gdf = pyogrio.read_dataframe(str(shp))
            if len(gdf) == 0:
                continue
            # Check for existing area field, otherwise compute equal-area km2 via EPSG:6933
            area_col = next((c for c in ["unitarea", "area", "AREA", "Area_km2"] if c in gdf.columns), None)
            if area_col is not None:
                areas = np.maximum(gdf[area_col].to_numpy(dtype=np.float64), 0.0)
            else:
                if gdf.crs is None:
                    gdf = gdf.set_crs("EPSG:4326")
                areas = np.maximum(gdf.to_crs("EPSG:6933").geometry.area.to_numpy(dtype=np.float64) / 1e6, 0.0)
            b = shapely.bounds(gdf.geometry.to_numpy())
            m_n = len(gdf)
            ids = np.arange(next_hs_id, next_hs_id + m_n, dtype=np.int64)
            next_hs_id += m_n
            hs_rows.append(pd.DataFrame({
                "COMID": ids,
                "NextDownID": np.zeros(m_n, dtype=np.int64),
                "length_km": np.zeros(m_n, dtype=np.float64),
                "uparea_raw": areas,
                "lon": 0.5 * (b[:, 0] + b[:, 2]),
                "lat": 0.5 * (b[:, 1] + b[:, 3]),
                "unitarea": areas,
                "is_coastal_hillslope": np.ones(m_n, dtype=bool),
            }))

    if hs_rows:
        hs_df = pd.concat(hs_rows, ignore_index=True)
        print(f"  Coastal hillslopes: {len(hs_df):,} polygons, {hs_df['unitarea'].sum()/1e6:.2f}M km2", flush=True)
        all_df = pd.concat([merged, hs_df], ignore_index=True)
    else:
        all_df = merged

    n = len(all_df)
    comid = all_df["COMID"].to_numpy(dtype=np.int64)
    dn_raw = all_df["NextDownID"].to_numpy(dtype=np.int64)
    id_map = pd.Series(np.arange(n, dtype=np.int32), index=comid)
    next_down = np.where(dn_raw > 0, id_map.reindex(dn_raw).fillna(-1).to_numpy(dtype=np.int32), -1).astype(np.int32)

    local_area = all_df["unitarea"].to_numpy(dtype=np.float64)
    n_broken = break_tree_cycles_inplace(next_down, all_df["uparea_raw"].to_numpy(dtype=np.float64))
    order, n_cyc = topo_sort_tree(next_down)
    assert n_cyc == 0
    a_in, a_out = accumulate_tree(order, next_down, local_area)
    print(
        f"  MERIT topology sorted: n={n:,} ({(~all_df['is_coastal_hillslope']).sum():,} reaches), "
        f"broken_cycles={n_broken}, total land={local_area.sum()/1e6:.2f}M km2, "
        f"total length={all_df['length_km'].sum()/1e6:.2f}M km",
        flush=True,
    )

    lons = all_df["lon"].to_numpy(dtype=np.float64)
    lats = all_df["lat"].to_numpy(dtype=np.float64)

    # Native flow model (GRADES v0.1 default; pluggable Today's Earth adapter)
    if flow_model == "grades":
        nc_path = flow_file or (FLOW_RAW / "grades_mean_annual_runoff_1979_2013.nc")
        r_mm_day = sample_gridded_runoff_mm_day(lons, lats, nc_path, var_name="runoff_mean_mm_day")
        local_runoff = r_mm_day * local_area * MM_DAY_KM2_TO_M3S
        q_in, q_out = accumulate_tree(order, next_down, local_runoff)
        q_phys_in, q_phys_out = q_in, q_out
    elif flow_model == "todays_earth":
        if flow_file is None or not flow_file.exists():
            raise FileNotFoundError(f"Today's Earth flow file required: {flow_file}")
        if flow_file.suffix == ".parquet":
            te = pd.read_parquet(flow_file).set_index("COMID")["q_mean_m3s"]
            q_phys_out = te.reindex(comid).fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64)
            q_phys_in = np.zeros(n, dtype=np.float64)
            has_dn = next_down >= 0
            np.add.at(q_phys_in, next_down[has_dn], q_phys_out[has_dn])
            local_runoff = np.maximum(q_phys_out - q_phys_in, 0.0)
            q_in, q_out = accumulate_tree(order, next_down, local_runoff)
        else:
            r_mm_day = sample_gridded_runoff_mm_day(lons, lats, flow_file)
            local_runoff = r_mm_day * local_area * MM_DAY_KM2_TO_M3S
            q_in, q_out = accumulate_tree(order, next_down, local_runoff)
            q_phys_in, q_phys_out = q_in, q_out
    else:
        raise ValueError(f"Unsupported flow_model={flow_model}")

    print(
        f"  MERIT ({flow_model}) global runoff: {local_runoff.sum()/1e6:.3f}M m3/s "
        f"({local_runoff.sum() * 86400 * 365.25 / 1e12:.1f}k km3/yr)",
        flush=True,
    )

    iso3, cont, in_common = assign_country_continent_and_domain(lons, lats)
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
        f"  MERIT enrichment: pop={pop.sum()/1e9:.3f}B{eae_msg}, "
        f"common_domain_area={local_area[in_common].sum()/1e6:.2f}M km2",
        flush=True,
    )

    out_cols = {
        "unit_id": comid,
        "next_down_idx": next_down,
        "local_area_km2": local_area,
        "up_area_in_km2": a_in,
        "up_area_out_km2": a_out,
        "length_km": all_df["length_km"].to_numpy(dtype=np.float64),
        "is_coastal_hillslope": all_df["is_coastal_hillslope"].to_numpy(dtype=bool),
        "local_runoff_m3s": local_runoff,
        "q_cons_in_m3s": q_in,
        "q_cons_out_m3s": q_out,
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


def snap_gauges_merit(units_df: pd.DataFrame, gauges_df: pd.DataFrame) -> pd.DataFrame:
    snap_pq = OUT_PROC / "merit_basins_gauge_snaps.parquet"
    if snap_pq.exists():
        print(f"[cache] Loading {snap_pq}", flush=True)
        return pd.read_parquet(snap_pq)

    t0 = time.time()
    p1_dir = MERIT_RAW / "pfaf_level_01"
    u_ids = units_df["unit_id"].to_numpy(dtype=np.int64)
    u_ain = units_df["up_area_in_km2"].to_numpy(dtype=np.float64)
    u_aout = units_df["up_area_out_km2"].to_numpy(dtype=np.float64)
    idx_series = pd.Series(np.arange(len(u_ids), dtype=np.int32), index=u_ids)

    snapper = StreamingReachSnapper(
        gauge_ids=gauges_df["canonical_gauge_id"].to_numpy(),
        lons=gauges_df["longitude"].to_numpy(dtype=np.float64),
        lats=gauges_df["latitude"].to_numpy(dtype=np.float64),
        reported_area_km2=gauges_df["drainage_area_km2"].to_numpy(dtype=np.float64),
        max_search_km=10.0,
    )

    print("Snapping 121,819 gauges to MERIT-Basins reaches (flip_orientation=True verified)...", flush=True)
    for pf in range(1, 10):
        riv_shp = sorted(p1_dir.glob(f"riv_pfaf_{pf}_*.shp"))[0]
        gdf = pyogrio.read_dataframe(str(riv_shp), columns=["COMID"])
        r_ids = gdf["COMID"].to_numpy(dtype=np.int64)
        geoms = gdf.geometry.to_numpy()
        pos = idx_series.reindex(r_ids).fillna(-1).to_numpy(dtype=np.int32)
        ok = (pos >= 0) & (~shapely.is_missing(geoms))
        # MERIT-Basins polylines are verified downstream->upstream (vertex 0 = outlet), so flip=True!
        fl = np.ones(int(ok.sum()), dtype=bool)
        pairs = snapper.update_chunk(r_ids[ok], geoms[ok], u_ain[pos[ok]], u_aout[pos[ok]], flip_orientation=fl)
        print(f"  pfaf_{pf}: {int(ok.sum()):,} reaches -> {pairs:,} candidate pairs", flush=True)

    snap_df = snapper.to_dataframe()

    # For any coastal gauges unmatched to a channelized reach, check coastal hillslope polygons
    unmatched = snap_df["snapped_unit_id"].to_numpy() < 0
    hs_dir = MERIT_RAW / "coastal_hillslopes"
    hs_shps = sorted(hs_dir.glob("*.shp"))
    if np.any(unmatched) and hs_shps:
        print(f"Checking {int(unmatched.sum()):,} unmatched gauges against MERIT coastal hillslopes...", flush=True)
        u_sub_g = gauges_df.loc[unmatched].reset_index(drop=True)
        psnap = StreamingPolygonSnapper(
            gauge_ids=u_sub_g["canonical_gauge_id"].to_numpy(),
            lons=u_sub_g["longitude"].to_numpy(dtype=np.float64),
            lats=u_sub_g["latitude"].to_numpy(dtype=np.float64),
            reported_area_km2=u_sub_g["drainage_area_km2"].to_numpy(dtype=np.float64),
            max_fallback_km=5.0,
        )
        next_hs_id = 900_000_000_000
        for shp in hs_shps:
            gdf = pyogrio.read_dataframe(str(shp))
            if len(gdf) == 0:
                continue
            m_n = len(gdf)
            ids = np.arange(next_hs_id, next_hs_id + m_n, dtype=np.int64)
            next_hs_id += m_n
            pos = idx_series.reindex(ids).fillna(-1).to_numpy(dtype=np.int32)
            ok = pos >= 0
            psnap.update_chunk(ids[ok], gdf.geometry.to_numpy()[ok], u_aout[pos[ok]], u_aout[pos[ok]])
        hs_snaps = psnap.to_dataframe(use_area_tiebreak=False)
        hit = hs_snaps["snapped_unit_id"].to_numpy() >= 0
        if np.any(hit):
            unm_idx = np.flatnonzero(unmatched)[hit]
            snap_df.loc[unm_idx, "snapped_unit_id"] = hs_snaps.loc[hit, "snapped_unit_id"].to_numpy()
            snap_df.loc[unm_idx, "snap_dist_km"] = hs_snaps.loc[hit, "snap_dist_km"].to_numpy()
            snap_df.loc[unm_idx, "snapped_area_in_km2"] = 0.0
            snap_df.loc[unm_idx, "snapped_area_out_km2"] = hs_snaps.loc[hit, "snapped_area_out_km2"].to_numpy()
            snap_df.loc[unm_idx, "effective_gauged_area_km2"] = hs_snaps.loc[hit, "effective_gauged_area_km2"].to_numpy()
            print(f"  Matched {int(hit.sum()):,} additional coastal gauges to MERIT hillslopes", flush=True)

    n_snapped = int((snap_df["snapped_unit_id"] >= 0).sum())
    med_d = float(np.nanmedian(snap_df.loc[snap_df["snapped_unit_id"] >= 0, "snap_dist_km"]))
    med_err = float(np.nanmedian(snap_df["log10_area_err"]))
    print(
        f"  MERIT snapping done in {time.time() - t0:.1f}s: {n_snapped:,}/{len(snap_df):,} snapped "
        f"(median dist={med_d:.3f} km, median |log10 area err|={med_err:.3f})",
        flush=True,
    )
    snap_df.to_parquet(snap_pq, index=False)
    return snap_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-model", default="grades", choices=["grades", "todays_earth"])
    parser.add_argument("--flow-file", type=Path, default=None)
    args = parser.parse_args()

    gauges_df = pd.read_parquet(GAUGES_DIR / "global_gauges.parquet")
    units_df, order, next_down = build_merit_graph(args.flow_model, args.flow_file)
    snap_df = snap_gauges_merit(units_df, gauges_df)

    ood_feats = None
    l12_ood_pq = PROCESSED_DIR / "hydrosheds" / "hydrosheds_l12_ood_features.parquet"
    if l12_ood_pq.exists():
        l12_ood = pd.read_parquet(l12_ood_pq).set_index("HYBAS_ID")
        ood_feats = l12_ood.reindex(units_df["hybas_l12_id"].to_numpy()).to_numpy(dtype=np.float64)

    fid = "merit_basins" if args.flow_model == "grades" else f"merit_basins_{args.flow_model}"
    run_tree_hydrofabric_suite(
        fabric_id=fid,
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
