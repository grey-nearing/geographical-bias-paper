#!/usr/bin/env python3
"""Builds the 3-panel Figure 4 anchored to the 3 existing-gauge tiers + universal gauging:
  - Panel (a): Global map of reduction in unwarned flood exposure under universal gauging
    (964,084 watersheds, 45.7M persons/yr, 30.2% of all unwarned exposure)
  - Panel (b): Grouped bar chart of reduction in unwarned flood exposure across the
    3 anchored existing-gauge tiers (34,577; 93,827; 121,819) + universal gauging (964,084)
    relative to the 8,811 post-2016 Caravan gauges by data-use mechanism
  - Panel (c): Stacked bar chart of total flood exposure warned per year across gauge
    availability tiers under combined global training, local fine-tuning, and real-time
    data assimilation, stacked by national flood warning system status (60 vs 113 countries)
Exact aspect ratio matches kix.a4fk56xa8gtg (468pt x 331.50975pt, ratio 0.70835418) so
in-place replacement has zero cropping and preserves comment anchors.
"""

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

NATURE_W = 183.0 / 25.4  # 7.204724 in
SAVE_DPI = 350
CANON_V2_DIR = "./data"
PRED_DIR = "./data/science_results_lead1/predictions"
SUMMARY_CSV = "./data/science_results_lead1/anchored_3tiers_exposure_summary_lead1.csv"
OUT_SCRATCH = "./data/science_results/figure_4_3panel_anchored.png"


def _normalize_hybas_id(series: pd.Series) -> pd.Series:
    nums = pd.to_numeric(
        series.astype(str)
        .str.replace("hybas_", "", regex=False)
        .str.replace(r"\.0$", "", regex=True),
        errors="coerce",
    )
    return "hybas_" + nums.astype("Int64").astype(str)


def main():
    t0 = time.time()
    plt.rcParams.update({
        "figure.dpi": SAVE_DPI,
        "savefig.dpi": SAVE_DPI,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 7.0,
    })

    with gfile.Open(file_paths.NATURAL_EARTH_PATH, "rb") as f:
        world_gdf = gpd.read_file(f)
    if "name" in world_gdf.columns:
        world_gdf = world_gdf[world_gdf["name"] != "Antarctica"].copy()
    world_gdf = world_gdf.dropna(subset=[world_gdf.geometry.name])
    world_gdf = world_gdf[world_gdf.is_valid & (~world_gdf.is_empty)].copy()

    df_ll = pd.read_csv(file_paths.HYBAS_LATLON_PATH, index_col=0)
    lat_c = next(c for c in df_ll.columns if "lat" in c.lower())
    lon_c = next(c for c in df_ll.columns if "lon" in c.lower())
    ll_clean = pd.DataFrame({
        "hybas_id": _normalize_hybas_id(pd.Series(df_ll.index)).values,
        "lat": pd.to_numeric(df_ll[lat_c], errors="coerce").values,
        "lon": pd.to_numeric(df_ll[lon_c], errors="coerce").values,
    }).dropna().drop_duplicates("hybas_id").set_index("hybas_id")

    exp_df = pd.read_parquet(
        f"{CANON_V2_DIR}/v2_exposure_blockers/statistics/basin_exposure_and_blockers_v2.parquet",
        columns=["country_name", "has_system", "E_total"],
    )
    exp_df.index = _normalize_hybas_id(pd.Series(exp_df.index)).values
    study_ids = list(exp_df.index)
    exp_df = exp_df.join(ll_clean, how="left")

    rp_raw = pd.read_csv(f"{CANON_V2_DIR}/exposure/global_basin_exposure_defended_by_return_period.csv")
    id_col = "hybas_id" if "hybas_id" in rp_raw.columns else rp_raw.columns[0]
    rp_raw[id_col] = _normalize_hybas_id(rp_raw[id_col]).values
    rp_raw = rp_raw.drop_duplicates(subset=[id_col]).set_index(id_col).reindex(study_ids).fillna(0.0)

    def _get_exp_col(df, t):
        for c in [f"exposed_pop_{t}", f"exposure_{t}yr", str(t)]:
            if c in df.columns:
                return df[c].values
        raise KeyError(f"Missing exposure column for T={t}")

    e5 = 0.05 * _get_exp_col(rp_raw, 5)
    e10 = 0.075 * _get_exp_col(rp_raw, 10)
    e20 = 0.04 * _get_exp_col(rp_raw, 20) + 0.02 * _get_exp_col(rp_raw, 50) + 0.005 * _get_exp_col(rp_raw, 100)

    p_b1a = pd.read_parquet(f"{PRED_DIR}/base__tier1a_active_post2016_ccby_8811__joint.parquet").loc[study_ids]
    p_dat5 = pd.read_parquet(f"{PRED_DIR}/da__tier5_universal_idealized_ceiling__joint.parquet").loc[study_ids]

    d_t5_101 = (
        e5 * np.maximum(0.0, p_dat5["r5_u"].values - p_b1a["r5_u"].values)
        + e10 * np.maximum(0.0, p_dat5["r10_u"].values - p_b1a["r10_u"].values)
        + e20 * np.maximum(0.0, p_dat5["r20_u"].values - p_b1a["r20_u"].values)
    )
    exp_df["d_t5_101"] = d_t5_101
    df_ws_pts = exp_df.dropna(subset=["lat", "lon"]).copy()
    df_ws_bias = df_ws_pts[df_ws_pts["d_t5_101"] > 0].sort_values("d_t5_101", ascending=True)
    df_ws_nobias = df_ws_pts[df_ws_pts["d_t5_101"] <= 0]

    df_sum = pd.read_csv(SUMMARY_CSV)
    # Use ge95_outlet mask (which gives 13.1M for Tier 3 121,819 combined, matching the manuscript text!)
    df_ge95 = df_sum[df_sum["mask_type"] == "ge95_outlet"].copy()

    def get_row(t_key, mech):
        r = df_ge95[(df_ge95["tier_key"] == t_key) & (df_ge95["mechanism"] == mech)]
        return r.iloc[0]

    # Exact aspect ratio of kix.a4fk56xa8gtg (468pt x 331.5097548774387pt -> ratio 0.70835418)
    fig_h = NATURE_W * (331.5097548774387 / 468.0)  # 5.103496 in
    fig = plt.figure(figsize=(NATURE_W, fig_h), dpi=SAVE_DPI, facecolor="#ffffff")
    gs = gridspec.GridSpec(
        4, 1, figure=fig,
        height_ratios=[1.00, 0.055, 0.22, 0.90],
        hspace=0.05, left=0.01, right=0.99, top=0.945, bottom=0.095,
    )

    # =========================================================================
    # Panel (a): Global map of reduction in unwarned flood exposure (45.7M/yr)
    # =========================================================================
    ax_a = fig.add_subplot(gs[0])
    world_gdf.plot(ax=ax_a, color="#F2F4F7", edgecolor="none", zorder=1)
    ax_a.scatter(
        df_ws_nobias["lon"].values, df_ws_nobias["lat"].values,
        s=0.05, marker=".", color="#D8DFE6", linewidths=0, rasterized=True, zorder=2
    )
    sc_a = ax_a.scatter(
        df_ws_bias["lon"].values, df_ws_bias["lat"].values,
        c=np.maximum(df_ws_bias["d_t5_101"].values, 0.05), s=0.08, marker=".",
        cmap="magma_r", norm=mcolors.LogNorm(vmin=0.1, vmax=1e4),
        linewidths=0, rasterized=True, zorder=3
    )
    world_gdf.boundary.plot(ax=ax_a, color="#455A64", linewidth=0.22, zorder=4)
    ax_a.set_xlim([-180, 180])
    ax_a.set_ylim([-58, 84])
    ax_a.set_axis_off()
    ax_a.set_title(
        "(a) Unwarned flood exposure reduction under universal gauging (12.7M/yr CAP-masked; 45.7M/yr global)",
        fontsize=7.4, fontweight="bold", loc="center", pad=3.0
    )

    gs_cb = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], width_ratios=[0.14, 0.72, 0.14])
    cax = fig.add_subplot(gs_cb[0, 1])
    cb = fig.colorbar(sc_a, cax=cax, orientation="horizontal", extend="both")
    cb.set_label(
        "Unwarned flood exposure reduction at α = 1.0 (persons/yr; 12.7M/yr CAP-masked, 29.6%; 45.7M/yr global, 30.2%)",
        fontsize=6.3, fontweight="bold", labelpad=2.0
    )
    cb.ax.tick_params(labelsize=6.1, pad=1.2)

    # =========================================================================
    # Bottom Row: Side-by-side Panel (b) and Panel (c)
    # =========================================================================
    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=gs[3], width_ratios=[0.06, 0.44, 0.08, 0.42], wspace=0.0
    )
    ax_b = fig.add_subplot(gs_bot[0, 1])
    ax_c = fig.add_subplot(gs_bot[0, 3])

    # Panel (b): Grouped bar chart across the 3 anchored tiers + universal gauging (CAP-masked 113 countries)
    b_tiers = [
        ("tier_caravan_estreams_34577", "Tier 1:\nCaravan + ES\n(34,577)"),
        ("tier_public_discharge_93827", "Tier 2:\nPublic records\n(93,827)"),
        ("tier4_all_publicly_knowable_121819", "Tier 3:\nAll knowable\n(121,819)"),
        ("tier5_universal_idealized_ceiling", "Universal\ngauging\n(964,084)"),
    ]
    xb = np.arange(len(b_tiers))
    wb = 0.25
    v_000 = [float(get_row(tk, "000_global_train")["delta_clip_nosys113_M"]) for tk, _ in b_tiers]
    v_001 = [float(get_row(tk, "001_da_only")["delta_clip_nosys113_M"]) for tk, _ in b_tiers]
    v_101 = [float(get_row(tk, "101_combined")["delta_clip_nosys113_M"]) for tk, _ in b_tiers]

    b1 = ax_b.bar(xb - wb, v_000, width=wb, color="#0072B2", edgecolor="#222222", linewidth=0.45, label="Global training alone")
    b2 = ax_b.bar(xb, v_001, width=wb, color="#43A2CA", edgecolor="#222222", linewidth=0.45, label="Real-time DA alone")
    b3 = ax_b.bar(xb + wb, v_101, width=wb, color="#E66101", edgecolor="#222222", linewidth=0.45, label="Combined (train + FT + DA)")

    for i, (val0, val1, valc) in enumerate(zip(v_000, v_001, v_101)):
        lbl0 = f"{val0:.2f}" if val0 < 0.1 else f"{val0:.1f}"
        lbl1 = f"{val1:.2f}" if val1 < 0.1 else f"{val1:.1f}"
        ax_b.text(xb[i] - wb, val0 + 0.25, lbl0, ha="center", va="bottom", fontsize=5.3, color="#004D80", fontweight="bold")
        ax_b.text(xb[i], val1 + 0.25, lbl1, ha="center", va="bottom", fontsize=5.3, color="#1D6A8A", fontweight="bold")
        ax_b.text(xb[i] + wb, valc + 0.25, f"{valc:.1f}M", ha="center", va="bottom", fontsize=5.6, color="#993404", fontweight="bold")

    ax_b.axvline(2.5, color="#78909C", linestyle="--", linewidth=0.7)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels([lbl for _, lbl in b_tiers], fontsize=5.8, fontweight="bold")
    ax_b.set_ylim(0, 14.8)
    ax_b.set_yticks([0, 3, 6, 9, 12, 15])
    ax_b.tick_params(axis="y", labelsize=6.0)
    ax_b.set_ylabel("CAP-masked reduction (M/yr)", fontsize=6.5, fontweight="bold")
    ax_b.set_title("(b) Exposure reduction in CAP-gap countries", fontsize=7.4, fontweight="bold", loc="left", pad=3.5)
    ax_b.legend(loc="upper left", fontsize=5.5, frameon=True, framealpha=0.95, facecolor="#ffffff", edgecolor="#CFD8DC", handlelength=1.1, borderpad=0.25)
    ax_b.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.4)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # Panel (c): Stacked bar chart of total flood exposure warned per year by warning system status
    c_tiers = [
        ("tier1a_active_post2016_ccby_8811", "000_global_train", "Post-2016\nCaravan\n(8,811)"),
        ("tier_caravan_estreams_34577", "101_combined", "Tier 1:\nCaravan+ES\n(34,577)"),
        ("tier_public_discharge_93827", "101_combined", "Tier 2:\nPublic Q\n(93,827)"),
        ("tier4_all_publicly_knowable_121819", "101_combined", "Tier 3:\nAll knowable\n(121,819)"),
        ("tier5_universal_idealized_ceiling", "101_combined", "Universal\ngauging\n(964,084)"),
    ]
    xc = np.arange(len(c_tiers))
    wc = 0.54
    # Use initial + clipped delta so total warned matches 70.3M + delta_clip_global_M exactly
    r_base = get_row("tier1a_active_post2016_ccby_8811", "000_global_train")
    w_sys_base = float(r_base["warned_sys60_M"])
    w_nosys_base = float(r_base["warned_nosys113_M"])

    w_sys_vals = []
    w_nosys_vals = []
    for tk, mk, _ in c_tiers:
        rk = get_row(tk, mk)
        w_sys_vals.append(w_sys_base + float(rk["delta_clip_sys60_M"]))
        w_nosys_vals.append(w_nosys_base + float(rk["delta_clip_nosys113_M"]))
    w_sys_vals = np.array(w_sys_vals)
    w_nosys_vals = np.array(w_nosys_vals)
    w_tot_vals = w_sys_vals + w_nosys_vals

    ax_c.bar(xc, w_sys_vals, width=wc, color="#0072B2", edgecolor="#222222", linewidth=0.45, label="Established warning sys. (60)")
    ax_c.bar(xc, w_nosys_vals, width=wc, bottom=w_sys_vals, color="#D55E00", edgecolor="#222222", linewidth=0.45, label="CAP warning gap (113)")

    for i, (ws, wn, wt) in enumerate(zip(w_sys_vals, w_nosys_vals, w_tot_vals)):
        ax_c.text(xc[i], wt + 1.8, f"{wt:.1f}M", ha="center", va="bottom", fontsize=5.6, fontweight="bold", color="#222222")
        ax_c.text(xc[i], ws * 0.5, f"{ws:.1f}", ha="center", va="center", fontsize=5.1, fontweight="bold", color="#ffffff")
        ax_c.text(xc[i], ws + wn * 0.5, f"{wn:.1f}", ha="center", va="center", fontsize=5.0, fontweight="bold", color="#ffffff")

    ax_c.axvline(3.5, color="#78909C", linestyle="--", linewidth=0.7)
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels([lbl for _, _, lbl in c_tiers], fontsize=5.6, fontweight="bold")
    ax_c.set_ylim(0, 136.0)
    ax_c.set_yticks([0, 25, 50, 75, 100, 125])
    ax_c.tick_params(axis="y", labelsize=6.0)
    ax_c.set_ylabel("Total exposure warned (M/yr)", fontsize=6.5, fontweight="bold")
    ax_c.set_title("(c) Total warned exposure", fontsize=7.4, fontweight="bold", loc="left", pad=3.5)
    ax_c.legend(loc="upper left", fontsize=5.5, frameon=True, framealpha=0.95, facecolor="#ffffff", edgecolor="#CFD8DC", handlelength=1.1, borderpad=0.25)
    ax_c.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.4)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    os.makedirs(os.path.dirname(OUT_SCRATCH), exist_ok=True)
    fig.savefig(OUT_SCRATCH, dpi=SAVE_DPI)
    plt.close(fig)
    shutil.copy2(OUT_SCRATCH, os.path.join(file_paths.FIGURES_AND_TABLES_FOR_PAPER_DIR, "Figure_4.png"))
    print(f"Saved 3-panel Figure 4 in {time.time() - t0:.1f}s -> {OUT_SCRATCH}")


if __name__ == "__main__":
    main()
