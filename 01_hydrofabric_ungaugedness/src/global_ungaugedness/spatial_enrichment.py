"""Spatial enrichment: Natural Earth countries/continents, common domain mask,
and mass-conserving HydroATLAS exposure + aridity transfer across all hydrofabrics.
"""

from __future__ import annotations

import os
os.environ.setdefault("OGR_ORGANIZE_POLYGONS", "SKIP")

from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from scipy.spatial import cKDTree
from shapely import STRtree

from configs.benchmark_config import (
    COMMON_DOMAIN_LAT_MAX,
    COMMON_DOMAIN_LAT_MIN,
    DATA_DIR,
    EXCLUDED_ISO3_COMMON_DOMAIN,
    EXPOSURE_DIR,
    FATHOM_EXPOSURE_PARQUET,
    PROCESSED_DIR,
    RAW_DIR,
    WORLDPOP_POP_PARQUET,
)

# UNEP aridity index (HydroATLAS ari_ix_sav is scaled by 100: AI = ari_ix_sav / 100, verified against pre_mm_syr / pet_mm_syr)
# Standard UNEP classes:
#   < 0.03  : Hyper-arid
#   0.03-0.2: Arid
#   0.2-0.5 : Semi-arid
#   0.5-0.65: Dry sub-humid
#   >= 0.65 : Humid
ARIDITY_LABELS = {
    0: "Unknown",
    1: "Hyper-arid",
    2: "Arid",
    3: "Semi-arid",
    4: "Dry sub-humid",
    5: "Humid",
}


def classify_aridity_from_hydroatlas(ari_ix_sav: np.ndarray) -> np.ndarray:
    """Map HydroATLAS ari_ix_sav (AI * 100) to int8 aridity zone codes 1..5 (0=missing)."""
    ai = np.asarray(ari_ix_sav, dtype=np.float64) / 100.0
    out = np.zeros(ai.shape[0], dtype=np.int8)
    valid = np.isfinite(ai) & (ai >= 0.0)
    out[valid & (ai < 0.03)] = 1
    out[valid & (ai >= 0.03) & (ai < 0.20)] = 2
    out[valid & (ai >= 0.20) & (ai < 0.50)] = 3
    out[valid & (ai >= 0.50) & (ai < 0.65)] = 4
    out[valid & (ai >= 0.65)] = 5
    return out


def assign_country_continent_and_domain(
    lons: np.ndarray,
    lats: np.ndarray,
    ne_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign ISO-3 country code, continent name, and harmonized common-domain mask."""
    ne_file = ne_path or (EXPOSURE_DIR / "naturalearth_lowres.geojson")
    ne = gpd.read_file(ne_file)

    # Identify ISO3 and continent columns robustly
    iso_col = next(
        (c for c in ["iso_a3", "ISO_A3", "ADM0_A3", "adm0_a3", "SOV_A3"] if c in ne.columns),
        None,
    )
    cont_col = next(
        (c for c in ["continent", "CONTINENT", "REGION_UN"] if c in ne.columns),
        None,
    )
    name_col = next((c for c in ["name", "NAME", "ADMIN"] if c in ne.columns), None)

    iso_vals = ne[iso_col].astype(str).to_numpy() if iso_col else np.full(len(ne), "UNK")
    # Natural Earth quirk: France/Norway sometimes have iso_a3 == '-99'; fix from name or ADM0_A3
    if name_col and "ADM0_A3" in ne.columns:
        bad = (iso_vals == "-99") | (iso_vals == "nan")
        iso_vals[bad] = ne["ADM0_A3"].astype(str).to_numpy()[bad]
    elif name_col:
        name_arr = ne[name_col].astype(str).to_numpy()
        iso_vals = np.where((iso_vals == "-99") & (name_arr == "France"), "FRA", iso_vals)
        iso_vals = np.where((iso_vals == "-99") & (name_arr == "Norway"), "NOR", iso_vals)
        iso_vals = np.where((iso_vals == "-99") & (name_arr == "Somaliland"), "SOM", iso_vals)
        iso_vals = np.where((iso_vals == "-99") & (name_arr == "Kosovo"), "XKX", iso_vals)

    cont_vals = ne[cont_col].astype(str).to_numpy() if cont_col else np.full(len(ne), "Unknown")
    geoms = ne.geometry.to_numpy()
    poly_tree = STRtree(geoms)

    n = len(lons)
    out_iso = np.full(n, "UNK", dtype=object)
    out_cont = np.full(n, "Unknown", dtype=object)

    # Batch point queries for low peak RAM
    batch = 500_000
    for s in range(0, n, batch):
        e = min(s + batch, n)
        pts = shapely.points(lons[s:e], lats[s:e])
        # Exact point-in-polygon
        pairs = poly_tree.query(pts, predicate="intersects")
        matched_local = np.zeros(e - s, dtype=bool)
        if pairs.shape[1] > 0:
            # First hit wins for shared boundary points
            p_i = pairs[0]
            poly_i = pairs[1]
            out_iso[s + p_i] = iso_vals[poly_i]
            out_cont[s + p_i] = cont_vals[poly_i]
            matched_local[p_i] = True

        # Nearest polygon fallback for coastal units outside low-res Natural Earth coastline
        miss = np.flatnonzero(~matched_local)
        if miss.size > 0:
            near_idx = poly_tree.nearest(pts[miss])
            out_iso[s + miss] = iso_vals[near_idx]
            out_cont[s + miss] = cont_vals[near_idx]

    in_common = (
        np.isfinite(lats)
        & (lats >= COMMON_DOMAIN_LAT_MIN)
        & (lats <= COMMON_DOMAIN_LAT_MAX)
        & (~pd.Series(out_iso).isin(EXCLUDED_ISO3_COMMON_DOMAIN).to_numpy())
    )
    # Also exclude Antarctica by continent name
    in_common &= out_cont != "Antarctica"

    return out_iso, out_cont, in_common


def _lonlat_to_xyz(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Convert geographic (lon, lat) degrees to 3D unit-sphere Cartesian coordinates."""
    lr = np.deg2rad(np.asarray(lons, dtype=np.float64))
    ar = np.deg2rad(np.asarray(lats, dtype=np.float64))
    ca = np.cos(ar)
    return np.column_stack([ca * np.cos(lr), ca * np.sin(lr), np.sin(ar)])


def map_points_to_hydrosheds_l12(
    lons: np.ndarray,
    lats: np.ndarray,
    basinatlas_gdb: Path | None = None,
    chunk_size: int = 100_000,
    fill_nearest_km: float = 50.0,
) -> np.ndarray:
    """Assign each (lon, lat) point to its containing HydroBASINS L12 HYBAS_ID.

    Streams BasinATLAS_v10_lev12 polygons against an STRtree of target points
    so memory stays bounded even for 20M target points. Coastal points falling
    slightly outside HydroBASINS L12 polygons are optionally matched to the
    nearest L12 centroid within `fill_nearest_km`.
    """
    gdb = basinatlas_gdb or (RAW_DIR / "hydrosheds" / "BasinATLAS_v10.gdb")
    n = len(lons)
    hybas_out = np.full(n, -1, dtype=np.int64)

    info = pyogrio.read_info(str(gdb), layer="BasinATLAS_v10_lev12")
    n_polys = int(info["features"])

    pts = shapely.points(lons, lats)
    pt_tree = STRtree(pts)

    for start in range(0, n_polys, chunk_size):
        gdf = pyogrio.read_dataframe(
            str(gdb),
            layer="BasinATLAS_v10_lev12",
            columns=["HYBAS_ID"],
            skip_features=start,
            max_features=chunk_size,
        )
        if len(gdf) == 0:
            break
        geoms = gdf.geometry.to_numpy()
        hids = gdf["HYBAS_ID"].to_numpy(dtype=np.int64)
        pairs = pt_tree.query(geoms, predicate="intersects")
        if pairs.shape[1] > 0:
            hybas_out[pairs[1]] = hids[pairs[0]]

    # Nearest L12 centroid fallback for coastal points slightly outside HydroBASINS coastline
    miss = np.flatnonzero(hybas_out < 0)
    l12_units_pq = PROCESSED_DIR / "hydrosheds" / "hydrosheds_l12_units.parquet"
    if miss.size > 0 and fill_nearest_km > 0.0 and l12_units_pq.exists():
        l12_u = pd.read_parquet(l12_units_pq, columns=["unit_id", "lon", "lat"])
        tree = cKDTree(_lonlat_to_xyz(l12_u["lon"].to_numpy(), l12_u["lat"].to_numpy()))
        max_chord = 2.0 * np.sin(fill_nearest_km / (2.0 * 6371.0088))
        dists, nn = tree.query(
            _lonlat_to_xyz(lons[miss], lats[miss]),
            k=1,
            distance_upper_bound=max_chord,
            workers=4,
        )
        ok = np.isfinite(dists) & (nn < len(l12_u))
        if np.any(ok):
            hybas_out[miss[ok]] = l12_u["unit_id"].to_numpy(dtype=np.int64)[nn[ok]]

    return hybas_out


def transfer_exposure_and_aridity_from_l12(
    hybas_l12_per_unit: np.ndarray,
    unit_weight_for_split: np.ndarray,
    exposure_parquet: Path | None = None,
    l12_aridity_parquet: Path | None = None,
    unit_lons: np.ndarray | None = None,
    unit_lats: np.ndarray | None = None,
    orphan_max_km: float = 50.0,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Mass-conserving transfer of population, optional EAE (>10cm), and aridity from HydroBASINS L12.

    Loads public WorldPop 2020 basin population from `hydrosheds_l12_worldpop_population.parquet`.
    If optional licensed Fathom v3 flood exposure (`hydrosheds_l12_fathom_exposure.parquet` or an
    explicit `exposure_parquet` containing `eae_gt10cm`) is present, transfers `eae_gt10cm`
    symmetrically; otherwise returns `eae_out = None`.
    """
    n = len(hybas_l12_per_unit)

    if exposure_parquet is not None and Path(exposure_parquet).exists():
        raw_df = pd.read_parquet(exposure_parquet)
    elif WORLDPOP_POP_PARQUET.exists():
        raw_df = pd.read_parquet(WORLDPOP_POP_PARQUET, columns=["HYBAS_ID", "total_population"])
        if FATHOM_EXPOSURE_PARQUET.exists():
            f_df = pd.read_parquet(FATHOM_EXPOSURE_PARQUET, columns=["HYBAS_ID", "eae_gt10cm"])
            raw_df = raw_df.merge(f_df, on="HYBAS_ID", how="left")
            raw_df["eae_gt10cm"] = raw_df["eae_gt10cm"].fillna(0.0)
    else:
        raise FileNotFoundError(
            f"WorldPop L12 population file not found at {WORLDPOP_POP_PARQUET}. "
            "Run scripts/00_stage_gauges_and_population.py first."
        )

    has_eae = "eae_gt10cm" in raw_df.columns
    keep_cols = ["HYBAS_ID", "total_population"] + (["eae_gt10cm"] if has_eae else [])
    exp_df = raw_df[keep_cols].copy()

    # Compute sum of unit_weight_for_split per HYBAS_ID
    w_pos = np.maximum(np.asarray(unit_weight_for_split, dtype=np.float64), 0.0)
    tmp = pd.DataFrame({
        "HYBAS_ID": hybas_l12_per_unit,
        "w": w_pos,
        "cnt": np.ones(n, dtype=np.float64),
    })
    valid_h = tmp["HYBAS_ID"] > 0
    grp = tmp.loc[valid_h].groupby("HYBAS_ID", sort=False).agg(w_sum=("w", "sum"), c_sum=("cnt", "sum"))

    exp_idxed = exp_df.set_index("HYBAS_ID")
    merged = exp_idxed.join(grp, how="inner")
    u_idx = pd.Index(hybas_l12_per_unit)
    pop_l12 = merged["total_population"].reindex(u_idx).fillna(0.0).to_numpy(dtype=np.float64)
    w_sum_u = merged["w_sum"].reindex(u_idx).fillna(0.0).to_numpy(dtype=np.float64)
    c_sum_u = merged["c_sum"].reindex(u_idx).fillna(0.0).to_numpy(dtype=np.float64)

    share = np.where(
        w_sum_u > 0.0,
        w_pos / np.where(w_sum_u > 0.0, w_sum_u, 1.0),
        np.where(c_sum_u > 0.0, 1.0 / np.where(c_sum_u > 0.0, c_sum_u, 1.0), 0.0),
    )
    pop_out = pop_l12 * share
    eae_out = (
        merged["eae_gt10cm"].reindex(u_idx).fillna(0.0).to_numpy(dtype=np.float64) * share
        if has_eae
        else None
    )

    # Orphan L12 mass conservation via nearest target unit within orphan_max_km
    l12_units_pq = PROCESSED_DIR / "hydrosheds" / "hydrosheds_l12_units.parquet"
    if unit_lons is not None and unit_lats is not None and l12_units_pq.exists():
        hit_ids = grp.index
        cond = exp_df["total_population"] > 0.0
        if has_eae:
            cond = cond | (exp_df["eae_gt10cm"] > 0.0)
        orphan = exp_df.loc[(~exp_df["HYBAS_ID"].isin(hit_ids)) & cond]
        if len(orphan) > 0:
            l12_coords = pd.read_parquet(l12_units_pq, columns=["unit_id", "lon", "lat"]).set_index("unit_id")
            o_coords = l12_coords.reindex(orphan["HYBAS_ID"].to_numpy())
            valid_o = np.isfinite(o_coords["lon"].to_numpy()) & np.isfinite(o_coords["lat"].to_numpy())
            if np.any(valid_o):
                o_lon = o_coords["lon"].to_numpy()[valid_o]
                o_lat = o_coords["lat"].to_numpy()[valid_o]
                o_pop = orphan["total_population"].to_numpy(dtype=np.float64)[valid_o]
                o_eae = orphan["eae_gt10cm"].to_numpy(dtype=np.float64)[valid_o] if has_eae else None

                valid_u = np.isfinite(unit_lons) & np.isfinite(unit_lats)
                u_pos_idx = np.flatnonzero(valid_u)
                u_tree = cKDTree(_lonlat_to_xyz(unit_lons[u_pos_idx], unit_lats[u_pos_idx]))
                max_chord = 2.0 * np.sin(orphan_max_km / (2.0 * 6371.0088))
                dists, nn = u_tree.query(
                    _lonlat_to_xyz(o_lon, o_lat),
                    k=1,
                    distance_upper_bound=max_chord,
                    workers=4,
                )
                ok = np.isfinite(dists) & (nn < len(u_pos_idx))
                if np.any(ok):
                    tgt = u_pos_idx[nn[ok]]
                    np.add.at(pop_out, tgt, o_pop[ok])
                    if eae_out is not None and o_eae is not None:
                        np.add.at(eae_out, tgt, o_eae[ok])

    aridity_out = np.zeros(n, dtype=np.int8)
    if l12_aridity_parquet is not None and l12_aridity_parquet.exists():
        ari_df = pd.read_parquet(l12_aridity_parquet, columns=["HYBAS_ID", "aridity_zone"])
        aridity_out = (
            ari_df.set_index("HYBAS_ID")["aridity_zone"]
            .reindex(u_idx)
            .fillna(0)
            .to_numpy(dtype=np.int8)
        )

    return pop_out, eae_out, aridity_out
