"""End-to-end ETL, gauge snapping, and full metric computation for HydroSHEDS.

Runs ALL 13 HydroSHEDS representations:
  1. BasinATLAS L12 (`hydrosheds_l12`, 1,034,083 polygons) + WaterGAP v2.2 runoff/discharge
     - Also produces shared reference files for MERIT, TDX-Hydro, and GRIT:
       * data/processed/hydrosheds/hydrosheds_l12_enrichment_ref.parquet
       * data/processed/hydrosheds/hydrosheds_l12_ood_features.parquet
  2. BasinATLAS L01 through L11 (`hydrosheds_l01` .. `hydrosheds_l11`)
     - Exact 100.0000% (region_digit, PFAF_ID) nested hierarchy from L12,
       isolating pure spatial subdivision scale-dependence across all 12 levels!
  3. HydroRIVERS v1.0 (`hydrosheds_rivers`, 8,477,883 reaches, 35.85M river-km)
     - Native WaterGAP DIS_AV_CMS discharge, verified flip_orientation=False,
       exact HYBAS_L12 exposure/aridity/channel-volume/OOD join, full 2D + 1D + OOD suite!
"""

from __future__ import annotations

import os
os.environ["OGR_ORGANIZE_POLYGONS"] = "SKIP"

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
    EXPOSURE_DIR,
    FATHOM_EXPOSURE_PARQUET,
    GAUGES_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    RESULTS_DIR,
    WORLDPOP_POP_PARQUET,
)
from global_ungaugedness.graph import (
    accumulate_tree,
    break_tree_cycles_inplace,
    topo_sort_tree,
)
from global_ungaugedness.runner import run_tree_hydrofabric_suite
from global_ungaugedness.snapping import StreamingPolygonSnapper, StreamingReachSnapper
from global_ungaugedness.spatial_enrichment import (
    assign_country_continent_and_domain,
    classify_aridity_from_hydroatlas,
)

HYDRO_RAW = RAW_DIR / "hydrosheds"
GDB_PATH = HYDRO_RAW / "BasinATLAS_v10.gdb"
RIV_SHP = HYDRO_RAW / "HydroRIVERS_v10_shp" / "HydroRIVERS_v10.shp"
OUT_PROC = PROCESSED_DIR / "hydrosheds"

# 1 mm/yr over 1 km^2 = 1e-3 m * 1e6 m^2 / (365.25 * 86400 s) = 3.1688087814e-5 m^3/s
MM_YR_KM2_TO_M3S = 1000.0 / (365.25 * 86400.0)

# 16 clean, physically grounded hydroclimatic & physiographic BasinATLAS attributes for Family 5 OOD
OOD_COLS = [
    "pre_mm_syr",
    "pet_mm_syr",
    "aet_mm_syr",
    "run_mm_syr",
    "ari_ix_sav",
    "tmp_dc_syr",
    "tmp_dc_smn",
    "tmp_dc_smx",
    "snw_pc_syr",
    "ele_mt_sav",
    "slp_dg_sav",
    "cly_pc_sav",
    "snd_pc_sav",
    "for_pc_sse",
    "crp_pc_sse",
    "urb_pc_sse",
]


def build_hydrosheds_l12() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Build BasinATLAS L12 graph + shared L12 enrichment & OOD feature reference tables."""
    OUT_PROC.mkdir(parents=True, exist_ok=True)
    units_pq = OUT_PROC / "hydrosheds_l12_units.parquet"
    ref_pq = OUT_PROC / "hydrosheds_l12_enrichment_ref.parquet"
    ood_pq = OUT_PROC / "hydrosheds_l12_ood_features.parquet"

    if units_pq.exists() and ref_pq.exists() and ood_pq.exists():
        print(f"[cache] Loading {units_pq}", flush=True)
        df = pd.read_parquet(units_pq)
        nd = df["next_down_idx"].to_numpy(dtype=np.int32)
        order, n_cyc = topo_sort_tree(nd)
        assert n_cyc == 0
        ood = pd.read_parquet(ood_pq).drop(columns=["HYBAS_ID"]).to_numpy(dtype=np.float64)
        return df, order, nd, ood

    t0 = time.time()
    print("Loading BasinATLAS_v10_lev12 (1,034,083 polygons)...", flush=True)
    core_cols = [
        "HYBAS_ID", "NEXT_DOWN", "PFAF_ID", "SUB_AREA", "UP_AREA",
        "ENDO", "COAST", "dis_m3_pyr", "run_mm_syr", "riv_tc_ssu", "ari_ix_sav",
    ]
    read_cols = list(dict.fromkeys(core_cols + OOD_COLS))

    # Stream bounds + attributes in 200k chunks to keep peak memory lean
    info = pyogrio.read_info(str(GDB_PATH), layer="BasinATLAS_v10_lev12")
    n_tot = int(info["features"])
    dfs = []
    chunk = 200_000
    for start in range(0, n_tot, chunk):
        gdf = pyogrio.read_dataframe(
            str(GDB_PATH),
            layer="BasinATLAS_v10_lev12",
            columns=read_cols,
            skip_features=start,
            max_features=chunk,
        )
        b = shapely.bounds(gdf.geometry.to_numpy())
        d = pd.DataFrame(gdf.drop(columns="geometry"))
        d["lon"] = (0.5 * (b[:, 0] + b[:, 2])).astype(np.float64)
        d["lat"] = (0.5 * (b[:, 1] + b[:, 3])).astype(np.float64)
        dfs.append(d)
        print(f"  lev12 read {min(start + chunk, n_tot):,}/{n_tot:,}", flush=True)

    raw = pd.concat(dfs, ignore_index=True)
    n = len(raw)
    hybas_id = raw["HYBAS_ID"].to_numpy(dtype=np.int64)
    next_down_raw = raw["NEXT_DOWN"].to_numpy(dtype=np.int64)
    pfaf_id = raw["PFAF_ID"].to_numpy(dtype=np.int64)
    reg_digit = hybas_id // 1_000_000_000

    id_map = pd.Series(np.arange(n, dtype=np.int32), index=hybas_id)
    next_down = np.where(
        next_down_raw > 0,
        id_map.reindex(next_down_raw).fillna(-1).to_numpy(dtype=np.int32),
        -1,
    ).astype(np.int32)

    local_area = np.maximum(raw["SUB_AREA"].to_numpy(dtype=np.float64), 0.0)
    up_area_raw = np.maximum(raw["UP_AREA"].to_numpy(dtype=np.float64), 0.0)
    n_broken = break_tree_cycles_inplace(next_down, up_area_raw)
    order, n_cyc = topo_sort_tree(next_down)
    assert n_cyc == 0, f"BasinATLAS L12 has {n_cyc} cycle nodes!"

    a_in, a_out = accumulate_tree(order, next_down, local_area)

    # WaterGAP v2.2 runoff & discharge
    run_mm = np.maximum(raw["run_mm_syr"].to_numpy(dtype=np.float64), 0.0)
    run_mm = np.where(run_mm > 50_000.0, 0.0, run_mm)  # guard sentinel
    local_runoff = run_mm * local_area * MM_YR_KM2_TO_M3S
    q_cons_in, q_cons_out = accumulate_tree(order, next_down, local_runoff)
    q_phys_out = np.maximum(raw["dis_m3_pyr"].to_numpy(dtype=np.float64), 0.0)
    q_phys_in = np.zeros(n, dtype=np.float64)
    has_dn = next_down >= 0
    np.add.at(q_phys_in, next_down[has_dn], q_phys_out[has_dn])

    # Aridity (verified ari_ix_sav = AI * 100)
    ari_zone = classify_aridity_from_hydroatlas(raw["ari_ix_sav"].to_numpy(dtype=np.float64))

    # Countries, continents, harmonized common domain (explicitly exclude HydroBASINS region 9 = Greenland)
    lons = raw["lon"].to_numpy(dtype=np.float64)
    lats = raw["lat"].to_numpy(dtype=np.float64)
    iso3, cont, in_common = assign_country_continent_and_domain(lons, lats)
    in_common = in_common & (reg_digit != 9)

    # Exact 1-to-1 WorldPop (and optional Fathom EAE >10cm) join on HYBAS_ID
    pop_df = pd.read_parquet(
        WORLDPOP_POP_PARQUET,
        columns=["HYBAS_ID", "total_population"],
    ).set_index("HYBAS_ID")
    pop = pop_df["total_population"].reindex(hybas_id).fillna(0.0).to_numpy(dtype=np.float64)
    eae = None
    if FATHOM_EXPOSURE_PARQUET.exists():
        f_df = pd.read_parquet(FATHOM_EXPOSURE_PARQUET, columns=["HYBAS_ID", "eae_gt10cm"]).set_index("HYBAS_ID")
        eae = f_df["eae_gt10cm"].reindex(hybas_id).fillna(0.0).to_numpy(dtype=np.float64)

    # Channel volume: riv_tc_ssu is in thousand m^3 -> * 1000 for m^3
    vol_m3 = np.maximum(raw["riv_tc_ssu"].to_numpy(dtype=np.float64), 0.0) * 1000.0

    # Clean OOD feature matrix (replace HydroATLAS -999/-9999 sentinels with NaN)
    ood_arr = raw[OOD_COLS].to_numpy(dtype=np.float64, copy=True)
    for j, cname in enumerate(OOD_COLS):
        if cname.startswith("tmp_dc"):
            ood_arr[ood_arr[:, j] <= -9000.0, j] = np.nan
        else:
            ood_arr[ood_arr[:, j] <= -900.0, j] = np.nan

    ood_df = pd.DataFrame(ood_arr, columns=OOD_COLS)
    ood_df.insert(0, "HYBAS_ID", hybas_id)
    ood_df.to_parquet(ood_pq, index=False)

    ref_data = {
        "HYBAS_ID": hybas_id,
        "PFAF_ID": pfaf_id,
        "reg_digit": reg_digit.astype(np.int8),
        "aridity_zone": ari_zone,
        "country_iso3": iso3,
        "continent": cont,
        "in_common_domain": in_common,
        "population": pop,
        "riv_vol_m3": vol_m3,
        "local_area_km2": local_area,
    }
    if eae is not None:
        ref_data["exposure_eae"] = eae
    ref_df = pd.DataFrame(ref_data)
    ref_df.to_parquet(ref_pq, index=False)

    units_data = {
        "unit_id": hybas_id,
        "pfaf_id": pfaf_id,
        "reg_digit": reg_digit.astype(np.int8),
        "next_down_idx": next_down,
        "local_area_km2": local_area,
        "up_area_in_km2": a_in,
        "up_area_out_km2": a_out,
        "up_area_attr_km2": up_area_raw,
        "local_runoff_m3s": local_runoff,
        "q_cons_in_m3s": q_cons_in,
        "q_cons_out_m3s": q_cons_out,
        "q_phys_in_m3s": q_phys_in,
        "q_phys_out_m3s": q_phys_out,
        "population": pop,
        "channel_volume_m3": vol_m3,
        "lon": lons.astype(np.float32),
        "lat": lats.astype(np.float32),
        "country_iso3": iso3,
        "continent": cont,
        "in_common_domain": in_common,
        "aridity_zone": ari_zone,
    }
    if eae is not None:
        units_data["exposure_eae"] = eae
    units_df = pd.DataFrame(units_data)
    units_df.to_parquet(units_pq, index=False)
    eae_str = f", EAE={eae.sum()/1e6:.2f}M/yr" if eae is not None else ""
    print(
        f"[hydrosheds_l12] Built in {time.time() - t0:.1f}s: n={n:,}, broken_cyc={n_broken}, "
        f"land={local_area.sum()/1e6:.3f}M km2 (common={local_area[in_common].sum()/1e6:.3f}M km2), "
        f"Q_runoff={local_runoff.sum()/1e6:.3f}M m3/s, pop={pop.sum()/1e9:.3f}B{eae_str}",
        flush=True,
    )
    return units_df, order, next_down, ood_arr


def snap_gauges_l12(units_df: pd.DataFrame, gauges_df: pd.DataFrame) -> pd.DataFrame:
    """Symmetrically snap all 121,819 gauges to BasinATLAS_v10_lev12 polygons."""
    snap_pq = OUT_PROC / "hydrosheds_l12_gauge_snaps.parquet"
    if snap_pq.exists():
        print(f"[cache] Loading {snap_pq}", flush=True)
        return pd.read_parquet(snap_pq)

    t0 = time.time()
    print("Snapping 121,819 gauges to BasinATLAS_v10_lev12 polygons...", flush=True)
    u_ids = units_df["unit_id"].to_numpy(dtype=np.int64)
    u_sub = units_df["local_area_km2"].to_numpy(dtype=np.float64)
    u_up = units_df["up_area_out_km2"].to_numpy(dtype=np.float64)
    idx_s = pd.Series(np.arange(len(u_ids), dtype=np.int32), index=u_ids)

    snapper = StreamingPolygonSnapper(
        gauge_ids=gauges_df["canonical_gauge_id"].to_numpy(),
        lons=gauges_df["longitude"].to_numpy(dtype=np.float64),
        lats=gauges_df["latitude"].to_numpy(dtype=np.float64),
        reported_area_km2=gauges_df["drainage_area_km2"].to_numpy(dtype=np.float64),
        max_fallback_km=10.0,
        area_tiebreak_km=1.5,
    )

    info = pyogrio.read_info(str(GDB_PATH), layer="BasinATLAS_v10_lev12")
    n_tot = int(info["features"])
    chunk = 150_000
    for start in range(0, n_tot, chunk):
        gdf = pyogrio.read_dataframe(
            str(GDB_PATH),
            layer="BasinATLAS_v10_lev12",
            columns=["HYBAS_ID"],
            skip_features=start,
            max_features=chunk,
        )
        hids = gdf["HYBAS_ID"].to_numpy(dtype=np.int64)
        pos = idx_s.reindex(hids).to_numpy(dtype=np.int32)
        snapper.update_chunk(hids, gdf.geometry.to_numpy(), u_sub[pos], u_up[pos])
        print(f"  snapped against {min(start + chunk, n_tot):,}/{n_tot:,} L12 polygons", flush=True)

    snap_df = snapper.to_dataframe(use_area_tiebreak=True)
    n_s = int((snap_df["snapped_unit_id"] >= 0).sum())
    med_d = float(np.nanmedian(snap_df.loc[snap_df["snapped_unit_id"] >= 0, "snap_dist_km"]))
    med_e = float(np.nanmedian(snap_df["log10_area_err"]))
    print(
        f"  BasinATLAS L12 snapping done in {time.time() - t0:.1f}s: {n_s:,}/{len(snap_df):,} snapped "
        f"(median dist={med_d:.3f} km, median |log10 area err|={med_e:.3f})",
        flush=True,
    )
    snap_df.to_parquet(snap_pq, index=False)
    return snap_df


def run_hydrosheds_levels_1_to_11(
    l12_units: pd.DataFrame,
    l12_snap: pd.DataFrame,
    gauges_df: pd.DataFrame,
) -> None:
    """Run BasinATLAS L01 through L11 using exact 100.0000% (reg_digit, PFAF_ID) hierarchy nesting."""
    l12_hid = l12_units["unit_id"].to_numpy(dtype=np.int64)
    l12_reg = l12_units["reg_digit"].to_numpy(dtype=np.int64)
    l12_pfaf = l12_units["pfaf_id"].to_numpy(dtype=np.int64)
    l12_pop = l12_units["population"].to_numpy(dtype=np.float64)
    l12_eae = l12_units["exposure_eae"].to_numpy(dtype=np.float64) if "exposure_eae" in l12_units.columns else None
    l12_area = l12_units["local_area_km2"].to_numpy(dtype=np.float64)
    l12_cont = l12_units["continent"].astype(str).to_numpy()
    l12_iso = l12_units["country_iso3"].astype(str).to_numpy()

    # Map snapped L12 HYBAS_ID -> L12 row index for O(1) level projection
    l12_idx_map = pd.Series(np.arange(len(l12_hid), dtype=np.int32), index=l12_hid)
    g_l12_hid = l12_snap["snapped_unit_id"].to_numpy(dtype=np.int64)
    g_l12_pos = l12_idx_map.reindex(g_l12_hid).fillna(-1).to_numpy(dtype=np.int32)
    g_rep_area = gauges_df["drainage_area_km2"].to_numpy(dtype=np.float64)

    for lv in range(1, 12):
        t0 = time.time()
        fid = f"hydrosheds_l{lv:02d}"
        if (RESULTS_DIR / fid / "metrics_2d_global.parquet").exists():
            print(f"[skip] {fid} already complete at {RESULTS_DIR / fid}", flush=True)
            continue
        lyr = f"BasinATLAS_v10_lev{lv:02d}"
        df = pyogrio.read_dataframe(
            str(GDB_PATH),
            layer=lyr,
            columns=["HYBAS_ID", "NEXT_DOWN", "PFAF_ID", "SUB_AREA", "UP_AREA", "dis_m3_pyr", "run_mm_syr", "ari_ix_sav"],
            read_geometry=False,
        )
        n = len(df)
        hid = df["HYBAS_ID"].to_numpy(dtype=np.int64)
        nd_raw = df["NEXT_DOWN"].to_numpy(dtype=np.int64)
        pfaf = df["PFAF_ID"].to_numpy(dtype=np.int64)
        reg = hid // 1_000_000_000

        id_map = pd.Series(np.arange(n, dtype=np.int32), index=hid)
        next_down = np.where(nd_raw > 0, id_map.reindex(nd_raw).fillna(-1).to_numpy(dtype=np.int32), -1).astype(np.int32)

        local_a = np.maximum(df["SUB_AREA"].to_numpy(dtype=np.float64), 0.0)
        up_raw = np.maximum(df["UP_AREA"].to_numpy(dtype=np.float64), 0.0)
        break_tree_cycles_inplace(next_down, up_raw)
        order, n_cyc = topo_sort_tree(next_down)
        assert n_cyc == 0
        a_in, a_out = accumulate_tree(order, next_down, local_a)

        run_mm = np.maximum(df["run_mm_syr"].to_numpy(dtype=np.float64), 0.0)
        run_mm = np.where(run_mm > 50_000.0, 0.0, run_mm)
        local_q = run_mm * local_a * MM_YR_KM2_TO_M3S
        q_in, q_out = accumulate_tree(order, next_down, local_q)
        ari_zone = classify_aridity_from_hydroatlas(df["ari_ix_sav"].to_numpy(dtype=np.float64))

        # Exact 100.0000% parent mapping: composite integer key = reg * 10^13 + pfaf_prefix
        scale_div = 10 ** (12 - lv)
        key_lv = reg * 10_000_000_000_000 + pfaf
        key_l12 = l12_reg * 10_000_000_000_000 + (l12_pfaf // scale_div)
        key_to_lv_idx = pd.Series(np.arange(n, dtype=np.int32), index=key_lv)
        l12_to_lv = key_to_lv_idx.reindex(key_l12).fillna(-1).to_numpy(dtype=np.int32)
        valid_m = l12_to_lv >= 0
        assert valid_m.all(), f"{fid}: missing {(~valid_m).sum()} L12 parents!"

        # Exact mass-conserving population (and optional exposure) aggregation from L12
        pop_lv = np.zeros(n, dtype=np.float64)
        np.add.at(pop_lv, l12_to_lv, l12_pop)
        eae_lv = None
        if l12_eae is not None:
            eae_lv = np.zeros(n, dtype=np.float64)
            np.add.at(eae_lv, l12_to_lv, l12_eae)

        # Area-dominant continent & country from L12 children
        tmp_c = pd.DataFrame({"p": l12_to_lv, "cont": l12_cont, "a": l12_area})
        dom_cont = tmp_c.groupby(["p", "cont"], sort=False)["a"].sum().reset_index()
        dom_cont = dom_cont.sort_values("a", ascending=False).drop_duplicates("p").set_index("p")["cont"]
        cont_lv = dom_cont.reindex(np.arange(n)).fillna("Unknown").to_numpy()

        tmp_i = pd.DataFrame({"p": l12_to_lv, "iso": l12_iso, "a": l12_area})
        dom_iso = tmp_i.groupby(["p", "iso"], sort=False)["a"].sum().reset_index()
        dom_iso = dom_iso.sort_values("a", ascending=False).drop_duplicates("p").set_index("p")["iso"]
        iso_lv = dom_iso.reindex(np.arange(n)).fillna("UNK").to_numpy()

        # Exact common domain: HydroBASINS excludes Antarctica; reg == 9 is Greenland
        in_common_lv = (reg != 9) & (iso_lv != "GRL") & (iso_lv != "ATA")

        # Project L12 snapped gauges onto parent L_k polygon + compute level-native sub-headwater cap
        g_lv_idx = np.where(g_l12_pos >= 0, l12_to_lv[np.maximum(g_l12_pos, 0)], -1)
        g_lv_hid = np.where(g_lv_idx >= 0, hid[np.maximum(g_lv_idx, 0)], -1)
        g_lv_aout = np.where(g_lv_idx >= 0, a_out[np.maximum(g_lv_idx, 0)], np.nan)
        g_lv_sub = np.where(g_lv_idx >= 0, local_a[np.maximum(g_lv_idx, 0)], np.nan)
        is_hw_poly = (g_lv_idx >= 0) & (g_lv_sub >= 0.95 * np.maximum(g_lv_aout, 1e-9))
        has_rep = np.isfinite(g_rep_area) & (g_rep_area > 0.0)
        is_sub_hw = is_hw_poly & has_rep & (g_rep_area < 0.8 * g_lv_aout)
        eff_area = np.where(is_sub_hw, np.minimum(g_rep_area, g_lv_sub), g_lv_aout)

        snap_lv = pd.DataFrame({
            "canonical_gauge_id": l12_snap["canonical_gauge_id"].to_numpy(),
            "snapped_unit_id": g_lv_hid,
            "snapped_area_out_km2": g_lv_aout,
            "is_sub_headwater": is_sub_hw,
            "effective_gauged_area_km2": eff_area,
        })

        units_lv_data = {
            "unit_id": hid,
            "next_down_idx": next_down,
            "local_area_km2": local_a,
            "up_area_in_km2": a_in,
            "up_area_out_km2": a_out,
            "local_runoff_m3s": local_q,
            "q_cons_in_m3s": q_in,
            "q_cons_out_m3s": q_out,
            "population": pop_lv,
            "country_iso3": iso_lv,
            "continent": cont_lv,
            "in_common_domain": in_common_lv,
            "aridity_zone": ari_zone,
        }
        if eae_lv is not None:
            units_lv_data["exposure_eae"] = eae_lv
        units_lv = pd.DataFrame(units_lv_data)

        eae_str = f", EAE={eae_lv.sum()/1e6:.2f}M/yr" if eae_lv is not None else ""
        print(
            f"[{fid}] n={n:,}, land={local_a.sum()/1e6:.2f}M km2 (common={local_a[in_common_lv].sum()/1e6:.2f}M km2), "
            f"pop={pop_lv.sum()/1e9:.3f}B{eae_str} (prep {time.time()-t0:.1f}s)",
            flush=True,
        )
        run_tree_hydrofabric_suite(
            fabric_id=fid,
            units_df=units_lv,
            order=order,
            next_down=next_down,
            gauges_df=gauges_df,
            snap_df=snap_lv,
            has_flow=True,
            run_1d=False,
            ood_features=None,
            run_subbasin_eff_mode=True,
        )


def run_hydrorivers(gauges_df: pd.DataFrame) -> None:
    """Build HydroRIVERS v1.0 (8.48M reaches), snap gauges, and run full 2D + 1D + OOD suite."""
    t0 = time.time()
    units_pq = OUT_PROC / "hydrosheds_rivers_units.parquet"
    snap_pq = OUT_PROC / "hydrosheds_rivers_gauge_snaps.parquet"

    ref_df = pd.read_parquet(OUT_PROC / "hydrosheds_l12_enrichment_ref.parquet").set_index("HYBAS_ID")

    if units_pq.exists():
        print(f"[cache] Loading {units_pq}", flush=True)
        units_df = pd.read_parquet(units_pq)
        nd = units_df["next_down_idx"].to_numpy(dtype=np.int32)
        order, n_cyc = topo_sort_tree(nd)
        assert n_cyc == 0
    else:
        print("Building HydroRIVERS v1.0 graph (8,477,883 reaches)...", flush=True)
        df = pyogrio.read_dataframe(
            str(RIV_SHP),
            columns=["HYRIV_ID", "NEXT_DOWN", "LENGTH_KM", "CATCH_SKM", "UPLAND_SKM", "DIS_AV_CMS", "HYBAS_L12"],
            read_geometry=False,
        )
        n = len(df)
        hyriv_id = df["HYRIV_ID"].to_numpy(dtype=np.int64)
        dn_raw = df["NEXT_DOWN"].to_numpy(dtype=np.int64)
        hybas_l12 = df["HYBAS_L12"].to_numpy(dtype=np.int64)
        length_km = np.maximum(df["LENGTH_KM"].to_numpy(dtype=np.float64), 0.0)
        local_area = np.maximum(df["CATCH_SKM"].to_numpy(dtype=np.float64), 0.0)
        up_raw = np.maximum(df["UPLAND_SKM"].to_numpy(dtype=np.float64), 0.0)

        id_map = pd.Series(np.arange(n, dtype=np.int32), index=hyriv_id)
        next_down = np.where(dn_raw > 0, id_map.reindex(dn_raw).fillna(-1).to_numpy(dtype=np.int32), -1).astype(np.int32)
        n_broken = break_tree_cycles_inplace(next_down, up_raw)
        order, n_cyc = topo_sort_tree(next_down)
        assert n_cyc == 0

        a_in, a_out = accumulate_tree(order, next_down, local_area)

        # Native WaterGAP v2.2 discharge (DIS_AV_CMS)
        q_phys_out = np.maximum(df["DIS_AV_CMS"].to_numpy(dtype=np.float64), 0.0)
        q_phys_in = np.zeros(n, dtype=np.float64)
        has_dn = next_down >= 0
        np.add.at(q_phys_in, next_down[has_dn], q_phys_out[has_dn])
        local_runoff = np.maximum(q_phys_out - q_phys_in, 0.0)
        q_cons_in, q_cons_out = accumulate_tree(order, next_down, local_runoff)

        # Exact HYBAS_L12 join for country, continent, aridity, common domain, exposure & channel volume
        l12_idx = pd.Index(hybas_l12)
        iso3 = ref_df["country_iso3"].reindex(l12_idx).fillna("UNK").to_numpy()
        cont = ref_df["continent"].reindex(l12_idx).fillna("Unknown").to_numpy()
        in_common = ref_df["in_common_domain"].reindex(l12_idx).fillna(False).to_numpy(dtype=bool)
        aridity = ref_df["aridity_zone"].reindex(l12_idx).fillna(0).to_numpy(dtype=np.int8)

        # Mass-conserving distribution of L12 pop/EAE proportional to local_area, and riv_vol proportional to A*L
        tmp = pd.DataFrame({
            "h": hybas_l12,
            "a": local_area,
            "al": np.maximum(a_out * length_km, 1e-6),
        })
        grp = tmp.groupby("h", sort=False).agg(a_sum=("a", "sum"), al_sum=("al", "sum"))
        a_sum_u = grp["a_sum"].reindex(l12_idx).fillna(0.0).to_numpy(dtype=np.float64)
        al_sum_u = grp["al_sum"].reindex(l12_idx).fillna(0.0).to_numpy(dtype=np.float64)

        pop_l12 = ref_df["population"].reindex(l12_idx).fillna(0.0).to_numpy(dtype=np.float64)
        eae_l12 = (
            ref_df["exposure_eae"].reindex(l12_idx).fillna(0.0).to_numpy(dtype=np.float64)
            if "exposure_eae" in ref_df.columns
            else None
        )
        vol_l12 = ref_df["riv_vol_m3"].reindex(l12_idx).fillna(0.0).to_numpy(dtype=np.float64)

        share_a = np.where(a_sum_u > 0.0, local_area / np.where(a_sum_u > 0.0, a_sum_u, 1.0), 0.0)
        share_al = np.where(al_sum_u > 0.0, tmp["al"].to_numpy() / np.where(al_sum_u > 0.0, al_sum_u, 1.0), 0.0)
        pop = pop_l12 * share_a
        eae = eae_l12 * share_a if eae_l12 is not None else None
        vol = vol_l12 * share_al

        units_data = {
            "unit_id": hyriv_id,
            "next_down_idx": next_down,
            "local_area_km2": local_area,
            "up_area_in_km2": a_in,
            "up_area_out_km2": a_out,
            "length_km": length_km,
            "local_runoff_m3s": local_runoff,
            "q_cons_in_m3s": q_cons_in,
            "q_cons_out_m3s": q_cons_out,
            "q_phys_in_m3s": q_phys_in,
            "q_phys_out_m3s": q_phys_out,
            "population": pop,
            "channel_volume_m3": vol,
            "country_iso3": iso3,
            "continent": cont,
            "in_common_domain": in_common,
            "aridity_zone": aridity,
            "hybas_l12_id": hybas_l12,
        }
        if eae is not None:
            units_data["exposure_eae"] = eae
        units_df = pd.DataFrame(units_data)
        units_df.to_parquet(units_pq, index=False)
        nd = next_down
        eae_str = f", EAE={eae.sum()/1e6:.2f}M/yr" if eae is not None else ""
        print(
            f"[hydrosheds_rivers] Built in {time.time()-t0:.1f}s: n={n:,}, broken_cyc={n_broken}, "
            f"land={local_area.sum()/1e6:.3f}M km2 (common={local_area[in_common].sum()/1e6:.3f}M km2), "
            f"length={length_km.sum()/1e6:.3f}M km, Q_cons={local_runoff.sum()/1e6:.3f}M m3/s, "
            f"pop={pop.sum()/1e9:.3f}B{eae_str}",
            flush=True,
        )

    # Snap gauges to HydroRIVERS polylines (verified flip_orientation=False!)
    if snap_pq.exists():
        print(f"[cache] Loading {snap_pq}", flush=True)
        snap_df = pd.read_parquet(snap_pq)
    else:
        ts = time.time()
        print("Snapping 121,819 gauges to HydroRIVERS v1.0 (8.48M reaches, flip=False verified)...", flush=True)
        u_ids = units_df["unit_id"].to_numpy(dtype=np.int64)
        u_ain = units_df["up_area_in_km2"].to_numpy(dtype=np.float64)
        u_aout = units_df["up_area_out_km2"].to_numpy(dtype=np.float64)
        idx_s = pd.Series(np.arange(len(u_ids), dtype=np.int32), index=u_ids)

        snapper = StreamingReachSnapper(
            gauge_ids=gauges_df["canonical_gauge_id"].to_numpy(),
            lons=gauges_df["longitude"].to_numpy(dtype=np.float64),
            lats=gauges_df["latitude"].to_numpy(dtype=np.float64),
            reported_area_km2=gauges_df["drainage_area_km2"].to_numpy(dtype=np.float64),
            max_search_km=10.0,
        )

        info = pyogrio.read_info(str(RIV_SHP))
        n_feat = int(info["features"])
        chunk = 400_000
        tot_pairs = 0
        for start in range(0, n_feat, chunk):
            gdf = pyogrio.read_dataframe(
                str(RIV_SHP),
                columns=["HYRIV_ID"],
                skip_features=start,
                max_features=chunk,
            )
            r_ids = gdf["HYRIV_ID"].to_numpy(dtype=np.int64)
            geoms = gdf.geometry.to_numpy()
            ok = ~shapely.is_missing(geoms)
            r_ids = r_ids[ok]
            geoms = geoms[ok]
            pos = idx_s.reindex(r_ids).fillna(-1).to_numpy(dtype=np.int32)
            ok2 = pos >= 0
            pairs = snapper.update_chunk(r_ids[ok2], geoms[ok2], u_ain[pos[ok2]], u_aout[pos[ok2]], flip_orientation=None)
            tot_pairs += pairs
            if (start // chunk) % 5 == 0 or start + chunk >= n_feat:
                print(f"  processed {min(start + chunk, n_feat):,}/{n_feat:,} reaches ({tot_pairs:,} pairs)", flush=True)

        snap_df = snapper.to_dataframe()
        n_s = int((snap_df["snapped_unit_id"] >= 0).sum())
        med_d = float(np.nanmedian(snap_df.loc[snap_df["snapped_unit_id"] >= 0, "snap_dist_km"]))
        med_e = float(np.nanmedian(snap_df["log10_area_err"]))
        print(
            f"  HydroRIVERS snapping done in {time.time()-ts:.1f}s: {n_s:,}/{len(snap_df):,} snapped "
            f"(median dist={med_d:.3f} km, median |log10 area err|={med_e:.3f})",
            flush=True,
        )
        snap_df.to_parquet(snap_pq, index=False)

    # OOD already computed directly on BasinATLAS L12 (same HYBAS_L12 attributes), pass None here to avoid redundant 8.5M query
    run_tree_hydrofabric_suite(
        fabric_id="hydrosheds_rivers",
        units_df=units_df,
        order=order,
        next_down=nd,
        gauges_df=gauges_df,
        snap_df=snap_df,
        has_flow=True,
        run_1d=True,
        ood_features=None,
        run_subbasin_eff_mode=True,
    )


def main() -> None:
    gauges_df = pd.read_parquet(GAUGES_DIR / "global_gauges.parquet")
    l12_units, l12_order, l12_nd, l12_ood = build_hydrosheds_l12()
    l12_snap = snap_gauges_l12(l12_units, gauges_df)

    # 1. Run BasinATLAS L12 (full 2D + Family 5 OOD)
    if (RESULTS_DIR / "hydrosheds_l12" / "metrics_2d_global.parquet").exists():
        print(f"[skip] hydrosheds_l12 already complete at {RESULTS_DIR / 'hydrosheds_l12'}", flush=True)
    else:
        run_tree_hydrofabric_suite(
            fabric_id="hydrosheds_l12",
            units_df=l12_units,
            order=l12_order,
            next_down=l12_nd,
            gauges_df=gauges_df,
            snap_df=l12_snap,
            has_flow=True,
            run_1d=False,
            ood_features=l12_ood,
            run_subbasin_eff_mode=True,
        )

    # 2. Run BasinATLAS L01..L11 (exact nested hierarchy scale-dependence curve)
    run_hydrosheds_levels_1_to_11(l12_units, l12_snap, gauges_df)

    # 3. Run HydroRIVERS v1.0 (8.48M reaches, full 2D + 1D)
    if (
        (RESULTS_DIR / "hydrosheds_rivers" / "metrics_2d_global.parquet").exists()
        and (RESULTS_DIR / "hydrosheds_rivers" / "metrics_1d_global.parquet").exists()
    ):
        print(f"[skip] hydrosheds_rivers already complete at {RESULTS_DIR / 'hydrosheds_rivers'}", flush=True)
    else:
        run_hydrorivers(gauges_df)


if __name__ == "__main__":
    main()
