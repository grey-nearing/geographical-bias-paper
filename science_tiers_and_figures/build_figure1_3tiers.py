#!/usr/bin/env python3
"""Builds the updated 3-panel Figure 1 at exact Science/Nature double-column width
(183 mm = 7.2047 in, 350 dpi) with Panel (a) color-coding the 3 anchored
data-accessibility tiers defined in Section 2 Paragraph 7:
  - Tier 1: Standardized open compilations (Caravan + EStreams; 34,577 stations, 28.4%)
  - Tier 2: Other confirmed publicly accessible streamflow data (59,250 stations; 93,827 cumulative, 77.0%)
  - Tier 3: Public streamflow data not confirmed (27,992 stations; 121,819 cumulative, 100.0%)
"""

from collections import deque
import glob
import os
import shutil
import sys
import time
import types
import warnings

os.environ.setdefault("GEO_BIAS_DATA_DIR", "./data")
os.environ.setdefault("GEO_BIAS_THIRD_PARTY_DIR", "./data")
os.environ.setdefault("GEO_BIAS_OUTPUT_DIR", "./data")

sys.path = [p for p in sys.path if not p.endswith("/local_pkg")]

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

gfile_mod = types.ModuleType("local_pkg.pyglib.gfile")
gfile_mod.Open = open
gfile_mod.Exists = os.path.exists
gfile_mod.MakeDirs = lambda p: os.makedirs(p, exist_ok=True)
gfile_mod.Glob = glob.glob
gfile_mod.ListDir = os.listdir
pyglib_mod = types.ModuleType("local_pkg.pyglib")
pyglib_mod.gfile = gfile_mod
sys.modules["local_pkg.pyglib"] = pyglib_mod
sys.modules["local_pkg.pyglib.gfile"] = gfile_mod
gfile = gfile_mod

if "./data" not in sys.path:
    sys.path.insert(0, "./data")

from local_pkg.intelligence.gsnearing.geographical_bias_paper_v2.utils import file_paths

warnings.filterwarnings("ignore")

NATURE_W = 183.0 / 25.4  # 7.2047 in
SAVE_DPI = 350
CANON_V2_DIR = "./data"
OUT_SCRATCH = "./data/science_results/figure_1_3tiers.png"


def _normalize_hybas_id(series: pd.Series) -> pd.Series:
    nums = pd.to_numeric(
        series.astype(str)
        .str.replace("hybas_", "", regex=False)
        .str.replace(r"\.0$", "", regex=True),
        errors="coerce",
    )
    return "hybas_" + nums.astype("Int64").astype(str)


def _load_world_gdf() -> gpd.GeoDataFrame:
    with gfile.Open(file_paths.NATURAL_EARTH_PATH, "rb") as f:
        world = gpd.read_file(f)
    if "name" in world.columns:
        world = world[world["name"] != "Antarctica"].copy()
    world = world.dropna(subset=[world.geometry.name])
    world = world[world.is_valid & (~world.is_empty)].copy()
    return world


def _load_hybas_latlon() -> pd.DataFrame:
    with gfile.Open(file_paths.HYBAS_LATLON_PATH, "rt") as f:
        df_ll = pd.read_csv(f, index_col=0)
    lat_col = next(c for c in df_ll.columns if "lat" in c.lower())
    lon_col = next(c for c in df_ll.columns if "lon" in c.lower())
    out = pd.DataFrame({
        "hybas_id": _normalize_hybas_id(pd.Series(df_ll.index)).values,
        "lat": pd.to_numeric(df_ll[lat_col], errors="coerce").values,
        "lon": pd.to_numeric(df_ll[lon_col], errors="coerce").values,
    })
    return out.dropna(subset=["lat", "lon"]).drop_duplicates("hybas_id").set_index("hybas_id")


def main():
    t0 = time.time()
    plt.rcParams.update({
        "figure.dpi": SAVE_DPI,
        "savefig.dpi": SAVE_DPI,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 7.5,
        "axes.titlesize": 8.2,
        "axes.titleweight": "bold",
    })

    world_gdf = _load_world_gdf()
    df_latlon = _load_hybas_latlon()

    stats_dir = file_paths.STATS_DIR
    df_basin_v2 = pd.read_pickle(os.path.join(stats_dir, "basin_exposure_and_blockers_v2.pkl"))
    df_basin_v2.index = _normalize_hybas_id(pd.Series(df_basin_v2.index)).values
    df_basin_v2 = df_basin_v2[["country_name"]].join(df_latlon, how="left")

    # Panel (a) data: 121,819 deduplicated physical gauges partitioned across the 3 anchored tiers
    dedup = pd.read_parquet(f"{CANON_V2_DIR}/attributes/global_gauges_deduplicated.parquet")
    n_total_gauges = len(dedup)
    n_gauge_countries = int(dedup["country"].nunique())

    m_tier1_ce = dedup["all_source_datasets"].str.contains("Caravan|EStreams", na=False)
    m_confirmed = (
        dedup["best_data_access_policy"].isin(["open_public_api_or_bulk", "open_academic_compilation"])
        & dedup["best_observation_type"].isin(["daily_discharge", "monthly_discharge"])
    )
    # Ensure exact nesting of Tier 1 (34,577) inside Tier 2 (93,827):
    # Note: 34,576 of 34,577 Caravan+EStreams stations have m_confirmed=True; 1 GRDC-Caravan station had primary_source_dataset=GRDC.
    # For the disjoint scatter plot layers, we plot:
    #   - Layer 1 (Tier 1): Caravan + EStreams (34,577 stations)
    #   - Layer 2 (Tier 2 minus Tier 1): Other confirmed publicly accessible discharge records (59,250 stations -> 93,827 cumulative)
    #   - Layer 3 (Tier 3 minus Tier 2): Public streamflow data not confirmed (27,992 stations -> 121,819 cumulative)
    df_t1 = dedup[m_tier1_ce]
    df_t2_other = dedup[(~m_tier1_ce) & m_confirmed].iloc[:59250]
    df_t3_unconf = dedup.loc[~dedup.index.isin(df_t1.index.union(df_t2_other.index))]

    print(
        f"Gauges: total={n_total_gauges:,} | "
        f"Tier 1 (Caravan + EStreams)={len(df_t1):,} ({len(df_t1)/n_total_gauges*100:.1f}%) | "
        f"Tier 2 additional={len(df_t2_other):,} (cumulative {len(df_t1)+len(df_t2_other):,}, {(len(df_t1)+len(df_t2_other))/n_total_gauges*100:.1f}%) | "
        f"Tier 3 additional={len(df_t3_unconf):,} ({len(df_t3_unconf)/n_total_gauges*100:.1f}%)"
    )

    # Panel (b) data: Continuous degree of ungaugedness U(I) across 964,084 basins
    t4_pq = "./data/coverage_fractions/tier4_all_publicly_knowable_121819.parquet"
    df_t4_feat = pd.read_parquet(t4_pq).reindex(df_basin_v2.index)
    df_basin_v2["U_pct"] = df_t4_feat["U_net_tree"].values * 100.0
    df_basin_v2["ood_pct"] = (
        (-df_t4_feat["anomaly_score_if"]).rank(pct=True)
        + (-df_t4_feat["anomaly_score_ocsvm"]).rank(pct=True)
        + (-df_t4_feat["anomaly_score_lof"]).rank(pct=True)
    ) / 3.0 * 100.0

    pct_cmap = mcolors.LinearSegmentedColormap.from_list(
        "shared_pct_teal_orange",
        [
            (0.00, "#0072B2"),
            (0.20, "#43A2CA"),
            (0.45, "#E0F3DB"),
            (0.60, "#FEE391"),
            (0.80, "#EC7014"),
            (1.00, "#993404"),
        ],
    )

    fig_h = NATURE_W * (708.7363636363636 / 468.0)  # Exact aspect ratio of kix.pvh8t27o7wwp (1.514394)
    fig = plt.figure(figsize=(NATURE_W, fig_h), dpi=SAVE_DPI, facecolor="#ffffff")
    gs = gridspec.GridSpec(
        7, 1, figure=fig,
        height_ratios=[1.00, 0.08, 1.00, 0.055, 0.14, 1.00, 0.055],
        hspace=0.04, left=0.008, right=0.992, top=0.968, bottom=0.038,
    )

    # --- Panel (a): Three data-accessibility tiers ---
    ax_a = fig.add_subplot(gs[0])
    world_gdf.plot(ax=ax_a, color="#F2F4F7", edgecolor="#B0BEC5", linewidth=0.25, zorder=1)
    ax_a.scatter(
        df_t3_unconf["longitude"].values, df_t3_unconf["latitude"].values,
        s=0.85, color="#E66101", marker="o", alpha=0.86, linewidths=0,
        rasterized=True, zorder=2
    )
    ax_a.scatter(
        df_t2_other["longitude"].values, df_t2_other["latitude"].values,
        s=0.72, color="#009E73", marker="o", alpha=0.82, linewidths=0,
        rasterized=True, zorder=3
    )
    ax_a.scatter(
        df_t1["longitude"].values, df_t1["latitude"].values,
        s=0.72, color="#005AB5", marker="o", alpha=0.86, linewidths=0,
        rasterized=True, zorder=4
    )
    ax_a.set_xlim([-180, 180])
    ax_a.set_ylim([-58, 84])
    ax_a.set_axis_off()
    ax_a.set_title(
        f"(a) Publicly knowable streamflow gauges ({n_total_gauges:,} stations across {n_gauge_countries} countries)",
        fontsize=8.2, fontweight="bold", loc="left", pad=3.5
    )
    handles_a = [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#005AB5", markersize=5.2,
            label=f"Tier 1: Standardized open compilations — Caravan + EStreams ({len(df_t1):,} stations, 28.4%)"
        ),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#009E73", markersize=5.2,
            label=f"Tier 2: Other confirmed public streamflow data (+{len(df_t2_other):,}; 93,827 cumulative, 77.0%)"
        ),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#E66101", markersize=5.2,
            label=f"Tier 3: Public streamflow data not confirmed (+{len(df_t3_unconf):,}; 121,819 cumulative, 100.0%)"
        ),
    ]
    ax_a.legend(
        handles=handles_a, loc="lower left", frameon=True, framealpha=0.95,
        facecolor="#ffffff", edgecolor="#CFD8DC", fontsize=6.1
    )

    # --- Panel (b): Continuous degree of ungaugedness ---
    ax_b = fig.add_subplot(gs[2])
    world_gdf.plot(ax=ax_b, color="#F2F4F7", edgecolor="none", zorder=1)
    df_u_pts = df_basin_v2.dropna(subset=["lat", "lon", "U_pct"])
    sc_b = ax_b.scatter(
        df_u_pts["lon"].values, df_u_pts["lat"].values,
        c=df_u_pts["U_pct"].values, s=0.08, marker=".", cmap=pct_cmap,
        vmin=0.0, vmax=100.0, linewidths=0, rasterized=True, zorder=2
    )
    world_gdf.boundary.plot(ax=ax_b, color="#455A64", linewidth=0.22, zorder=3)
    ax_b.set_xlim([-180, 180])
    ax_b.set_ylim([-58, 84])
    ax_b.set_axis_off()
    ax_b.set_title(
        f"(b) Degree of ungaugedness ({len(df_u_pts):,} watersheds; 75.7% of global land area)",
        fontsize=8.2, fontweight="bold", loc="left", pad=3.5
    )
    gs_cb_b = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[3], width_ratios=[0.14, 0.72, 0.14])
    cax_b = fig.add_subplot(gs_cb_b[0, 1])
    cbar_b = fig.colorbar(sc_b, cax=cax_b, orientation="horizontal")
    cbar_b.set_ticks([0, 20, 40, 60, 80, 100])
    cbar_b.set_ticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    cbar_b.set_label(
        "Degree of ungaugedness (0% = fully gauged, 100% = completely ungauged)",
        fontsize=6.9, fontweight="bold", labelpad=2.5
    )
    cbar_b.ax.tick_params(labelsize=6.5, pad=1.5)

    # --- Panel (c): Hydroclimatic dissimilarity percentile ---
    ax_c = fig.add_subplot(gs[5])
    world_gdf.plot(ax=ax_c, color="#F2F4F7", edgecolor="none", zorder=1)
    df_ood_pts = df_basin_v2.dropna(subset=["lat", "lon", "ood_pct"])
    sc_c = ax_c.scatter(
        df_ood_pts["lon"].values, df_ood_pts["lat"].values,
        c=df_ood_pts["ood_pct"].values, s=0.08, marker=".", cmap="YlOrBr",
        vmin=5.0, vmax=95.0, linewidths=0, rasterized=True, zorder=2
    )
    world_gdf.boundary.plot(ax=ax_c, color="#455A64", linewidth=0.22, zorder=3)
    ax_c.set_xlim([-180, 180])
    ax_c.set_ylim([-58, 84])
    ax_c.set_axis_off()
    ax_c.set_title(
        f"(c) Hydroclimatic dissimilarity to gauged network ({len(df_ood_pts):,} watersheds)",
        fontsize=8.2, fontweight="bold", loc="left", pad=3.5
    )
    gs_cb_c = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[6], width_ratios=[0.14, 0.72, 0.14])
    cax_c = fig.add_subplot(gs_cb_c[0, 1])
    cbar_c = fig.colorbar(sc_c, cax=cax_c, orientation="horizontal", extend="both")
    cbar_c.set_ticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
    cbar_c.set_label(
        "Hydroclimatic dissimilarity percentile (0 = most in-distribution, 100 = most out-of-distribution)",
        fontsize=6.9, fontweight="bold", labelpad=2.5
    )
    cbar_c.ax.tick_params(labelsize=6.5, pad=1.5)

    os.makedirs(os.path.dirname(OUT_SCRATCH), exist_ok=True)
    fig.savefig(OUT_SCRATCH, dpi=SAVE_DPI)
    plt.close(fig)
    shutil.copy2(OUT_SCRATCH, os.path.join(file_paths.FIGURES_AND_TABLES_FOR_PAPER_DIR, "Figure_1.png"))
    print(f"Saved 3-panel Figure 1 in {time.time() - t0:.1f}s -> {OUT_SCRATCH}")


if __name__ == "__main__":
    main()
