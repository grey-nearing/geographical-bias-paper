"""Synthesize cross-hydrofabric master tables and generate publication-grade figures.

Consolidates all hydrofabric outputs in results/*/ into:
  - results/master/master_2d_metrics.{parquet,csv}
  - results/master/master_1d_metrics.{parquet,csv}
  - results/master/master_ood_metrics.{parquet,csv}
  - results/master/master_2d_country.parquet
  - results/master/master_cross_hydrofabric_summary.csv
  - results/master/snapping_audit_summary.csv

And generates 6 diagnostic/publication figures in figures/:
  1. fig01_hydrosheds_l1_to_l12_scale_dependence.png
  2. fig02_cross_hydrofabric_comparison_all_families.png
  3. fig03_resolution_curves_Ures_Ucont.png
  4. fig04_river_1d_stratified_and_channel_distance.png
  5. fig05_continental_and_aridity_breakdown.png
  6. fig06_grit_bifurcation_impact.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from configs.benchmark_config import FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR

MASTER_DIR = RESULTS_DIR / "master"

# Clean colorblind-safe Okabe-Ito / Tol palette
PALETTE = {
    "hydrosheds_l12": "#0072B2",
    "hydrosheds_rivers": "#56B4E9",
    "merit_basins": "#009E73",
    "tdx_hydro": "#E69F00",
    "grit_v06_mainstem_tree": "#CC79A7",
    "grit_v06_bifurcating_dag": "#D55E00",
}

FABRIC_LABELS = {
    "hydrosheds_l12": "HydroSHEDS L12",
    "hydrosheds_rivers": "HydroRIVERS v1",
    "merit_basins": "MERIT-Basins (GRADES)",
    "tdx_hydro": "TDX-Hydro (GEOGLOWS)",
    "grit_v06_mainstem_tree": "GRIT v0.6 (Mainstem)",
    "grit_v06_bifurcating_dag": "GRIT v0.6 (Bifurc. DAG)",
}

TIER_ORDER = [
    "baseline",
    "tier1_open_compilations",
    "tier2_public_discharge",
    "tier3_all_known",
]
TIER_SHORT = {
    "baseline": "Baseline\n(8.1k)",
    "tier1_open_compilations": "Tier 1\n(32.4k)",
    "tier2_public_discharge": "Tier 2\n(74.1k)",
    "tier3_all_known": "Tier 3\n(119.2k)",
}
TIER_COLORS = {
    "baseline": "#D55E00",
    "tier1_open_compilations": "#E69F00",
    "tier2_public_discharge": "#009E73",
    "tier3_all_known": "#0072B2",
}


def setup_mpl() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fontsize": 8.2,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def consolidate_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    MASTER_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    f_2d = sorted(p for p in RESULTS_DIR.glob("*/metrics_2d_global.parquet") if p.parent.name != "master")
    f_1d = sorted(p for p in RESULTS_DIR.glob("*/metrics_1d_global.parquet") if p.parent.name != "master")
    f_ood = sorted(p for p in RESULTS_DIR.glob("*/metrics_ood.parquet") if p.parent.name != "master")
    f_cty = sorted(p for p in RESULTS_DIR.glob("*/metrics_2d_country.parquet") if p.parent.name != "master")

    df_2d = pd.concat([pd.read_parquet(p) for p in f_2d], ignore_index=True) if f_2d else pd.DataFrame()
    df_1d = pd.concat([pd.read_parquet(p) for p in f_1d], ignore_index=True) if f_1d else pd.DataFrame()
    df_ood = pd.concat([pd.read_parquet(p) for p in f_ood], ignore_index=True) if f_ood else pd.DataFrame()
    df_cty = pd.concat([pd.read_parquet(p) for p in f_cty], ignore_index=True) if f_cty else pd.DataFrame()

    if len(df_2d):
        df_2d.to_parquet(MASTER_DIR / "master_2d_metrics.parquet", index=False)
        df_2d.to_csv(MASTER_DIR / "master_2d_metrics.csv", index=False)
    if len(df_1d):
        df_1d.to_parquet(MASTER_DIR / "master_1d_metrics.parquet", index=False)
        df_1d.to_csv(MASTER_DIR / "master_1d_metrics.csv", index=False)
    if len(df_ood):
        df_ood.to_parquet(MASTER_DIR / "master_ood_metrics.parquet", index=False)
        df_ood.to_csv(MASTER_DIR / "master_ood_metrics.csv", index=False)
    if len(df_cty):
        df_cty.to_parquet(MASTER_DIR / "master_2d_country.parquet", index=False)

    print(
        f"Consolidated: 2D={len(df_2d):,} rows ({len(f_2d)} fabrics), "
        f"1D={len(df_1d):,} rows ({len(f_1d)} fabrics), "
        f"OOD={len(df_ood):,} rows ({len(f_ood)} fabrics), "
        f"Country={len(df_cty):,} rows ({len(f_cty)} fabrics)",
        flush=True,
    )
    return df_2d, df_1d, df_ood, df_cty


def audit_snapping() -> pd.DataFrame:
    snaps = sorted(PROCESSED_DIR.rglob("*gauge_snaps*.parquet"))
    rows = []
    for p in snaps:
        df = pd.read_parquet(p)
        snapped = df["snapped_unit_id"].to_numpy() >= 0
        n_tot = len(df)
        n_s = int(snapped.sum())
        d = df.loc[snapped, "snap_dist_km"].to_numpy(dtype=np.float64) if "snap_dist_km" in df.columns else np.array([])
        err = df.loc[snapped, "log10_area_err"].to_numpy(dtype=np.float64) if "log10_area_err" in df.columns else np.array([])
        err = err[np.isfinite(err)]
        rows.append({
            "snap_file": p.stem,
            "n_total_gauges": n_tot,
            "n_snapped": n_s,
            "snap_rate_pct": 100.0 * n_s / n_tot if n_tot else np.nan,
            "n_unique_units": int(df.loc[snapped, "snapped_unit_id"].nunique()),
            "median_snap_dist_km": float(np.nanmedian(d)) if len(d) else np.nan,
            "p90_snap_dist_km": float(np.nanpercentile(d[np.isfinite(d)], 90)) if np.any(np.isfinite(d)) else np.nan,
            "median_abs_log10_area_err": float(np.nanmedian(err)) if len(err) else np.nan,
            "frac_area_err_le_0p3": float(np.mean(err <= 0.3)) if len(err) else np.nan,
        })
    out = pd.DataFrame(rows)
    out.to_csv(MASTER_DIR / "snapping_audit_summary.csv", index=False)
    print(f"Saved snapping audit ({len(out)} files) -> {MASTER_DIR / 'snapping_audit_summary.csv'}", flush=True)
    return out


def build_headline_summary(df_2d: pd.DataFrame, df_1d: pd.DataFrame, df_ood: pd.DataFrame) -> pd.DataFrame:
    """Create wide comparison CSV across all hydrofabrics & tiers on common_no_grl_ata and full_native."""
    rows = []
    target_doms = ("common_no_grl_ata", "full_native")
    all_fabrics = sorted(set(df_2d["hydrofabric"].unique()) | (set(df_1d["hydrofabric"].unique()) if len(df_1d) else set()))

    # Pre-group by (hydrofabric, tier, domain_name) for fast O(1) lookup
    g2d = {k: v for k, v in df_2d[df_2d["domain_name"].isin(target_doms)].groupby(["hydrofabric", "tier", "domain_name"], sort=False)}
    g1d = {k: v for k, v in df_1d[df_1d["domain_name"].isin(target_doms)].groupby(["hydrofabric", "tier", "domain_name"], sort=False)} if len(df_1d) else {}
    good = {k: v for k, v in df_ood[df_ood["domain_name"].isin(target_doms)].groupby(["hydrofabric", "tier", "domain_name"], sort=False)} if len(df_ood) else {}

    def get_2d(sub_ftd: pd.DataFrame | None, metric: str, weight: str,
               ratio: tuple[str, ...] = ("area", "area_dag", "topology"),
               mode: tuple[str, ...] = ("outlet", "dag_width_weighted"),
               scale: float | None = None) -> float:
        if sub_ftd is None or sub_ftd.empty:
            return np.nan
        sub = sub_ftd[
            (sub_ftd["metric"] == metric)
            & (sub_ftd["weight"] == weight)
            & (sub_ftd["ratio_basis"].isin(ratio))
            & (sub_ftd["snapping_mode"].isin(mode))
        ]
        if scale is not None:
            sub = sub[np.isclose(sub["scale_param"].fillna(-999.0), scale)]
        return float(sub["value"].iloc[0]) if len(sub) else np.nan

    def get_1d(sub_ftd: pd.DataFrame | None, metric: str, weight: str, stratum: str,
               ratio: tuple[str, ...] = ("area", "area_dag", "distance_km")) -> float:
        if sub_ftd is None or sub_ftd.empty:
            return np.nan
        sub = sub_ftd[
            (sub_ftd["metric"] == metric)
            & (sub_ftd["weight"] == weight)
            & (sub_ftd["stratum"] == stratum)
            & (sub_ftd["ratio_basis"].isin(ratio))
        ]
        return float(sub["value"].iloc[0]) if len(sub) else np.nan

    def get_ood(sub_ftd: pd.DataFrame | None, metric: str, weight: str = "area") -> float:
        if sub_ftd is None or sub_ftd.empty:
            return np.nan
        sub = sub_ftd[
            (sub_ftd["metric"] == metric)
            & (sub_ftd["weight"] == weight)
        ]
        return float(sub["value"].iloc[0]) if len(sub) else np.nan

    has_eae = "exposure_eae" in set(df_2d["weight"].unique())

    for dom in target_doms:
        for fab in all_fabrics:
            for tier in TIER_ORDER:
                s2 = g2d.get((fab, tier, dom))
                if s2 is None or s2.empty:
                    continue
                s1 = g1d.get((fab, tier, dom))
                so = good.get((fab, tier, dom))
                # Total land area from U_never row
                sub_a = s2[
                    (s2["metric"] == "U_never")
                    & (s2["weight"] == "area")
                    & (s2["snapping_mode"].isin(["outlet", "dag_width_weighted"]))
                ]
                if not len(sub_a):
                    continue
                tot_area_mkm2 = float(sub_a["total_weight"].iloc[0]) / 1e6
                n_snapped = int(sub_a["n_gauges_snapped"].iloc[0])

                rec = {
                    "domain": dom,
                    "hydrofabric": fab,
                    "tier": tier,
                    "n_gauges_snapped": n_snapped,
                    "total_area_Mkm2": tot_area_mkm2,
                    # Family 1 (2D area-weighted)
                    "U_out_area": get_2d(s2, "U_out", "area"),
                    "U_zero_up_area": get_2d(s2, "U_zero_up", "area"),
                    "U_up_area": get_2d(s2, "U_up", "area"),
                    "U_down_area": get_2d(s2, "U_down", "area"),
                    "U_net_Eq4_area": get_2d(s2, "U_net_Eq4", "area"),
                    "U_net_Eq4_area_subeff": get_2d(s2, "U_net_Eq4", "area", mode=("subbasin_eff",)),
                    "U_zero_net_area": get_2d(s2, "U_zero_net", "area"),
                    # Family 2 (2D strictly scale-invariant)
                    "U_never_area": get_2d(s2, "U_never", "area"),
                    "U_tail_area": get_2d(s2, "U_tail", "area"),
                    "U_never_runoff": get_2d(s2, "U_never", "runoff"),
                    "U_never_pop": get_2d(s2, "U_never", "population"),
                    # Flow-ratio Eq4 (runoff-weighted)
                    "U_net_Eq4_runoff_Q": get_2d(s2, "U_net_Eq4", "runoff", ratio=("flow",)),
                    # Family 3 (2D resolution)
                    "U_res_1k_area": get_2d(s2, "U_res", "area", scale=1_000.0),
                    "U_res_10k_area": get_2d(s2, "U_res", "area", scale=10_000.0),
                    "U_cont_1k_area": get_2d(s2, "U_cont", "area", scale=1_000.0),
                    "U_cont_10k_area": get_2d(s2, "U_cont", "area", scale=10_000.0),
                    "A_tilde_q50_mon_km2": get_2d(s2, "A_tilde_q50_monitored", "area"),
                    "U_cont_Q10_runoff": get_2d(s2, "U_cont", "runoff", ratio=("flow",), scale=10.0),
                    # Family 4 (1D river network)
                    "U_net_1d_Ads": get_1d(s1, "U_net_1d", "A_ds", "all_cutoff_free"),
                    "U_up_1d_Ads": get_1d(s1, "U_up_1d", "A_ds", "all_cutoff_free"),
                    "U_net_1d_Qds": get_1d(s1, "U_net_1d", "Q_ds", "all_cutoff_free", ratio=("flow",)),
                    "U_net_1d_len_A_ge_100": get_1d(s1, "U_net_1d", "river_length", "A_ge_100"),
                    "U_net_1d_len_A_ge_1k": get_1d(s1, "U_net_1d", "river_length", "A_ge_1000"),
                    "U_net_1d_len_A_ge_10k": get_1d(s1, "U_net_1d", "river_length", "A_ge_10000"),
                    "d_any_q50_Ads_km": get_1d(s1, "d_any_q50_all", "A_ds", "all_cutoff_free"),
                    "d_any_le100km_Ads": get_1d(s1, "d_any_frac_le_100km", "A_ds", "all_cutoff_free"),
                    # Family 5 (OOD)
                    "U_OOD_knn1_mean_area": get_ood(so, "U_OOD_knn1_mean", "area"),
                    "U_OOD_knn5_mean_area": get_ood(so, "U_OOD_knn5_mean", "area"),
                }
                if has_eae:
                    rec["U_never_eae"] = get_2d(s2, "U_never", "exposure_eae")
                    rec["U_net_Eq4_eae"] = get_2d(s2, "U_net_Eq4", "exposure_eae")
                    rec["U_cont_1k_eae"] = get_2d(s2, "U_cont", "exposure_eae", scale=1_000.0)
                    rec["U_net_1d_eae"] = get_1d(s1, "U_net_1d", "exposure_eae", "all")
                rows.append(rec)

    summary = pd.DataFrame(rows)
    summary.to_csv(MASTER_DIR / "master_cross_hydrofabric_summary.csv", index=False)
    print(f"Saved headline summary ({len(summary)} rows) -> {MASTER_DIR / 'master_cross_hydrofabric_summary.csv'}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# Figures 1-6
# ---------------------------------------------------------------------------

def plot_fig01_hydrosheds_scale(df_2d: pd.DataFrame, df_1d: pd.DataFrame) -> None:
    levels = list(range(1, 13))
    fnames = [f"hydrosheds_l{lv:02d}" for lv in levels]
    sub = df_2d[(df_2d["hydrofabric"].isin(fnames)) & (df_2d["domain_name"] == "common_no_grl_ata")].copy()
    if sub.empty:
        return
    sub["level"] = sub["hydrofabric"].str.extract(r"l(\d+)").astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), sharex=True)

    def series_lv(tier: str, metric: str, weight: str = "area", ratio: str = "area",
                  mode: str = "outlet", scale: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        q = sub[(sub["tier"] == tier) & (sub["metric"] == metric) & (sub["weight"] == weight)
                & (sub["ratio_basis"].isin([ratio, "topology"])) & (sub["snapping_mode"] == mode)]
        if scale is not None:
            q = q[np.isclose(q["scale_param"].fillna(-999.0), scale)]
        q = q.sort_values("level")
        return q["level"].to_numpy(), q["value"].to_numpy() * 100.0

    # (a) Family 1 hydrofabric-dependent polygon metrics (Tier 3)
    ax = axes[0, 0]
    for m, lbl, col, ls in [
        ("U_out", r"$U_{\mathrm{out}}$ (directly ungauged)", "#000000", ":"),
        ("U_up", r"$U_{\mathrm{up}}$ (upstream)", "#0072B2", "-"),
        ("U_down", r"$U_{\mathrm{down}}$ (downstream)", "#009E73", "-"),
        ("U_net_Eq4", r"$U_{\mathrm{net}}$ Eq. 4 (outlet)", "#D55E00", "-"),
    ]:
        x, y = series_lv("tier3_all_known", m)
        ax.plot(x, y, marker="o", ms=4.5, lw=2.0, color=col, ls=ls, label=lbl)
    x, y = series_lv("tier3_all_known", "U_net_Eq4", mode="subbasin_eff")
    if len(x):
        ax.plot(x, y, marker="s", ms=4.0, lw=1.8, color="#D55E00", ls="--", label=r"$U_{\mathrm{net}}$ Eq. 4 (sub-basin eff.)")
    # Overlay HydroRIVERS 1D A*ds reference
    r1 = df_1d[(df_1d["hydrofabric"] == "hydrosheds_rivers") & (df_1d["tier"] == "tier3_all_known")
               & (df_1d["domain_name"] == "common_no_grl_ata") & (df_1d["metric"] == "U_net_1d")
               & (df_1d["weight"] == "A_ds")]
    if len(r1):
        ax.axhline(r1["value"].iloc[0] * 100.0, color="#56B4E9", ls="-.", lw=1.6,
                   label=f"HydroRIVERS 1D $U_{{\\mathrm{{net}}}}$ ($A\\,ds$): {r1['value'].iloc[0]*100:.1f}%")
    ax.set_title("(a) Family 1 Polygon Metrics Drift Across L1–L12 (Tier 3, Area)")
    ax.set_ylabel("Ungauged fraction (%)")
    ax.set_ylim(0, 102)
    ax.legend(loc="lower right")

    # (b) Family 2 & 3 scale-invariant 2D metrics (Tier 3)
    ax = axes[0, 1]
    for m, sc, lbl, col, ls in [
        ("U_never", None, r"$U_{\mathrm{never}}$ (never gauged)", "#000000", "-"),
        ("U_zero_net", None, r"$U_{\mathrm{zero,net}}$ (unmonitored basins)", "#7F7F7F", ":"),
        ("U_res", 1_000.0, r"$U_{\mathrm{res}}(1{,}000\,\mathrm{km}^2)$", "#D55E00", "-"),
        ("U_cont", 1_000.0, r"$U_{\mathrm{cont}}(1{,}000\,\mathrm{km}^2)$", "#E69F00", "--"),
        ("U_res", 10_000.0, r"$U_{\mathrm{res}}(10{,}000\,\mathrm{km}^2)$", "#0072B2", "-"),
        ("U_cont", 10_000.0, r"$U_{\mathrm{cont}}(10{,}000\,\mathrm{km}^2)$", "#56B4E9", "--"),
    ]:
        x, y = series_lv("tier3_all_known", m, scale=sc)
        ax.plot(x, y, marker="o", ms=4.2, lw=1.9, color=col, ls=ls, label=lbl)
    ax.set_title("(b) Families 2 & 3 Stabilize Once Polygon < Gauge Scale (Tier 3)")
    ax.set_ylabel("Ungauged fraction (%)")
    ax.set_ylim(0, 102)
    ax.legend(loc="lower right", ncol=2, fontsize=7.1, columnspacing=0.9, handlelength=1.8)

    # (c) Weight comparison across L1-L12 (Baseline vs Tier 3)
    ax = axes[1, 0]
    for w_name, lbl, col in [
        ("area", "Land area", "#0072B2"),
        ("runoff", "Runoff (WaterGAP)", "#009E73"),
        ("population", "Population (WorldPop)", "#CC79A7"),
        ("exposure_eae", "Flood exposure (Fathom EAE)", "#D55E00"),
    ]:
        x, y = series_lv("tier3_all_known", "U_net_Eq4", weight=w_name)
        if len(x):
            ax.plot(x, y, marker="o", ms=4.0, lw=2.0, color=col, ls="-", label=f"{lbl} (T3)")
        xb, yb = series_lv("baseline", "U_net_Eq4", weight=w_name)
        if len(xb):
            ax.plot(xb, yb, marker="^", ms=3.5, lw=1.4, color=col, ls=":", alpha=0.85, label=f"{lbl} (Base)")
    ax.set_title("(c) $U_{\\mathrm{net}}$ (Eq. 4) Scale Drift by Weighting Variable")
    ax.set_xlabel("HydroBASINS Pfafstetter Level")
    ax.set_ylabel("Ungauged fraction (%)")
    ax.set_ylim(0, 102)
    ax.legend(loc="lower right", ncol=2, fontsize=7.0, columnspacing=0.8, handlelength=1.7)

    # (d) All 4 gauge tiers: Eq. 4 (solid) vs U_never (dashed)
    ax = axes[1, 1]
    for tk in TIER_ORDER:
        x, y = series_lv(tk, "U_net_Eq4")
        ax.plot(x, y, marker="o", ms=4.0, lw=2.0, color=TIER_COLORS[tk], ls="-",
                label=f"{TIER_SHORT[tk].replace(chr(10), ' ')} — Eq. 4")
        xn, yn = series_lv(tk, "U_never")
        ax.plot(xn, yn, marker="x", ms=4.5, lw=1.5, color=TIER_COLORS[tk], ls="--")
    ax.set_title("(d) All 4 Tiers: Eq. 4 (solid) vs. $U_{\\mathrm{never}}$ (dashed)")
    ax.set_xlabel("HydroBASINS Pfafstetter Level")
    ax.set_ylabel("Ungauged fraction (%)")
    ax.set_ylim(0, 102)
    ax.set_xticks(levels)
    ax.legend(loc="lower right", fontsize=7.8)

    fig.suptitle("HydroSHEDS BasinATLAS L1–L12 Resolution Sensitivity (Harmonized Global Domain)", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig01_hydrosheds_l1_to_l12_scale_dependence.png")
    plt.close(fig)


def plot_fig02_cross_hydrofabric(summary: pd.DataFrame) -> None:
    fine_fabs = [
        "hydrosheds_l12",
        "hydrosheds_rivers",
        "merit_basins",
        "tdx_hydro",
        "grit_v06_mainstem_tree",
        "grit_v06_bifurcating_dag",
    ]
    sub = summary[(summary["domain"] == "common_no_grl_ata") & (summary["hydrofabric"].isin(fine_fabs))].copy()
    present_fabs = [f for f in fine_fabs if f in set(sub["hydrofabric"])]
    if not present_fabs:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    panels = [
        (axes[0, 0], "U_net_Eq4_area", "(a) 2D $U_{\\mathrm{net}}$ (Eq. 4, Area-Weighted)"),
        (axes[0, 1], "U_never_area", "(b) 2D $U_{\\mathrm{never}}$ (Never Gauged, Area-Weighted)"),
        (axes[1, 0], "U_cont_1k_area", "(c) 2D $U_{\\mathrm{cont}}(1{,}000\\,\\mathrm{km}^2)$ (Area-Weighted)"),
        (axes[1, 1], "U_net_1d_Ads", "(d) 1D River $U_{\\mathrm{net,1D}}$ ($A\\,ds$-Weighted, Cutoff-Free)"),
    ]

    x = np.arange(len(TIER_ORDER))
    n_f = len(present_fabs)
    width = 0.80 / max(n_f, 1)

    for ax, col, title in panels:
        for i, fab in enumerate(present_fabs):
            vals = []
            for tk in TIER_ORDER:
                r = sub[(sub["hydrofabric"] == fab) & (sub["tier"] == tk)]
                v = float(r[col].iloc[0]) * 100.0 if len(r) and np.isfinite(r[col].iloc[0]) else np.nan
                vals.append(v)
            offs = x + (i - (n_f - 1) / 2.0) * width
            bars = ax.bar(offs, vals, width=width * 0.92, color=PALETTE[fab], label=FABRIC_LABELS[fab], edgecolor="white", lw=0.4)
            for b, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2.0, v + 0.8, f"{v:.0f}", ha="center", va="bottom", fontsize=6.6, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels([TIER_SHORT[t] for t in TIER_ORDER])
        ax.set_ylabel("Ungauged fraction (%)")
        ax.set_ylim(0, 108)
        ax.set_title(title)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=8.8)
    fig.suptitle("Cross-Hydrofabric Comparison Across All 4 Gauge Tiers (Harmonized Global Domain, Excl. GRL/ATA)",
                 fontsize=12.2, fontweight="bold", y=1.06)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig02_cross_hydrofabric_comparison_all_families.png")
    plt.close(fig)


def plot_fig03_resolution_curves(df_2d: pd.DataFrame) -> None:
    fine_fabs = [
        "hydrosheds_l12",
        "hydrosheds_rivers",
        "merit_basins",
        "tdx_hydro",
        "grit_v06_mainstem_tree",
        "grit_v06_bifurcating_dag",
    ]
    sub = df_2d[
        (df_2d["hydrofabric"].isin(fine_fabs))
        & (df_2d["domain_name"] == "common_no_grl_ata")
        & (df_2d["snapping_mode"].isin(["outlet", "dag_width_weighted"]))
        & (df_2d["metric"].isin(["U_res", "U_cont"]))
        & (np.isfinite(df_2d["scale_param"]))
    ].copy()
    if sub.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    specs = [
        (axes[0, 0], "U_res", "area", ("area", "area_dag"), r"$A_0$ (km²)", "(a) $U_{\\mathrm{res}}(A_0)$ — Area-Weighted (Baseline & Tier 3)"),
        (axes[0, 1], "U_cont", "area", ("area", "area_dag"), r"$A_0$ (km²)", "(b) $U_{\\mathrm{cont}}(A_0)$ — Area-Weighted (Baseline & Tier 3)"),
        (axes[1, 0], "U_res", "runoff", ("flow",), r"$Q_0$ (m³/s)", "(c) $U_{\\mathrm{res}}(Q_0)$ — Runoff-Weighted (Flow-Enabled Fabrics)"),
        (axes[1, 1], "U_cont", "runoff", ("flow",), r"$Q_0$ (m³/s)", "(d) $U_{\\mathrm{cont}}(Q_0)$ — Runoff-Weighted (Flow-Enabled Fabrics)"),
    ]

    for ax, met, wt, ratios, xlabel, title in specs:
        for fab in fine_fabs:
            for tk, ls, alpha in [("baseline", ":", 0.85), ("tier3_all_known", "-", 1.0)]:
                q = sub[
                    (sub["hydrofabric"] == fab)
                    & (sub["tier"] == tk)
                    & (sub["metric"] == met)
                    & (sub["weight"] == wt)
                    & (sub["ratio_basis"].isin(ratios))
                ].sort_values("scale_param")
                if q.empty:
                    continue
                lbl = FABRIC_LABELS[fab] if tk == "tier3_all_known" else None
                ax.plot(q["scale_param"], q["value"] * 100.0, color=PALETTE[fab], ls=ls, lw=1.9 if tk == "tier3_all_known" else 1.4,
                        alpha=alpha, marker="o" if tk == "tier3_all_known" else None, ms=3.2, label=lbl)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Ungauged fraction (%)")
        ax.set_ylim(0, 102)
        ax.set_title(title)

    axes[0, 0].legend(loc="lower left", fontsize=8.0, title="Solid = Tier 3 | Dotted = Baseline", title_fontsize=7.6)
    axes[1, 0].legend(loc="lower left", fontsize=8.0, title="GRIT skipped (no flow model)", title_fontsize=7.6)
    fig.suptitle("Family 3 Resolution Curves Across All Hydrofabrics (Harmonized Global Domain)", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig03_resolution_curves_Ures_Ucont.png")
    plt.close(fig)


def plot_fig04_river_1d(df_1d: pd.DataFrame) -> None:
    river_fabs = ["hydrosheds_rivers", "merit_basins", "tdx_hydro", "grit_v06_mainstem_tree", "grit_v06_bifurcating_dag"]
    sub = df_1d[(df_1d["hydrofabric"].isin(river_fabs)) & (df_1d["domain_name"] == "common_no_grl_ata")].copy()
    if sub.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))

    # (a) Stratified by drainage area A(x)
    ax = axes[0, 0]
    a_bins = ["0-10", "10-100", "100-1k", "1k-10k", "10k-100k", "100k-1M", ">1M"]
    x_a = np.arange(len(a_bins))
    for fab in river_fabs:
        for tk, ls in [("baseline", ":"), ("tier3_all_known", "-")]:
            q = sub[(sub["hydrofabric"] == fab) & (sub["tier"] == tk) & (sub["metric"] == "U_net_1d")
                    & (sub["weight"] == "river_length") & (sub["ratio_basis"].isin(["area", "area_dag"]))]
            ys = [float(q[q["stratum"] == f"A_bin_{b}"]["value"].iloc[0]) * 100.0 if len(q[q["stratum"] == f"A_bin_{b}"]) else np.nan for b in a_bins]
            if any(np.isfinite(ys)):
                ax.plot(x_a, ys, color=PALETTE[fab], ls=ls, lw=2.0 if tk == "tier3_all_known" else 1.4,
                        marker="o" if tk == "tier3_all_known" else "^", ms=4.0,
                        label=FABRIC_LABELS[fab] if tk == "tier3_all_known" else None)
    ax.set_xticks(x_a)
    ax.set_xticklabels(a_bins, rotation=20)
    ax.set_xlabel("Upstream drainage area bin $A(x)$ (km²)")
    ax.set_ylabel("1D $U_{\\mathrm{net,1D}}$ (%)")
    ax.set_ylim(0, 102)
    ax.set_title("(a) 1D River Ungaugedness by River Size $A(x)$ (Solid=T3, Dotted=Base)")
    ax.legend(loc="lower left", fontsize=7.8)

    # (b) Stratified by physical discharge Q(x)
    ax = axes[0, 1]
    q_bins = ["<0.1", "0.1-1", "1-10", "10-100", "100-1k", "1k-10k", ">10k"]
    x_q = np.arange(len(q_bins))
    for fab in ["hydrosheds_rivers", "merit_basins", "tdx_hydro"]:
        for tk, ls in [("baseline", ":"), ("tier3_all_known", "-")]:
            q = sub[(sub["hydrofabric"] == fab) & (sub["tier"] == tk) & (sub["metric"] == "U_net_1d")
                    & (sub["weight"] == "river_length") & (sub["ratio_basis"] == "flow")]
            ys = [float(q[q["stratum"] == f"Q_bin_{b}"]["value"].iloc[0]) * 100.0 if len(q[q["stratum"] == f"Q_bin_{b}"]) else np.nan for b in q_bins]
            if any(np.isfinite(ys)):
                ax.plot(x_q, ys, color=PALETTE[fab], ls=ls, lw=2.0 if tk == "tier3_all_known" else 1.4,
                        marker="o" if tk == "tier3_all_known" else "^", ms=4.0,
                        label=FABRIC_LABELS[fab] if tk == "tier3_all_known" else None)
    ax.set_xticks(x_q)
    ax.set_xticklabels(q_bins, rotation=20)
    ax.set_xlabel("Mean annual discharge bin $Q(x)$ (m³/s)")
    ax.set_ylabel("1D $U_{\\mathrm{net,1D}}$ (Flow-Ratio, %)")
    ax.set_ylim(0, 102)
    ax.set_title("(b) 1D River Ungaugedness by Discharge $Q(x)$ (Flow-Enabled Fabrics)")
    ax.legend(loc="lower left", fontsize=8.0)

    # (c) Along-channel distance CDF (A*ds weighted)
    ax = axes[1, 0]
    dists = [10, 25, 50, 100, 250, 500, 1000]
    for fab in ["hydrosheds_rivers", "merit_basins", "tdx_hydro", "grit_v06_mainstem_tree"]:
        for tk, ls in [("baseline", ":"), ("tier3_all_known", "-")]:
            q = sub[(sub["hydrofabric"] == fab) & (sub["tier"] == tk) & (sub["weight"] == "A_ds") & (sub["ratio_basis"] == "distance_km")]
            ys = [float(q[q["metric"] == f"d_any_frac_le_{d}km"]["value"].iloc[0]) * 100.0 if len(q[q["metric"] == f"d_any_frac_le_{d}km"]) else np.nan for d in dists]
            if any(np.isfinite(ys)):
                ax.plot(dists, ys, color=PALETTE[fab], ls=ls, lw=2.0 if tk == "tier3_all_known" else 1.4,
                        marker="o" if tk == "tier3_all_known" else None, ms=3.8,
                        label=FABRIC_LABELS[fab] if tk == "tier3_all_known" else None)
    ax.set_xscale("log")
    ax.set_xlabel("Distance along river channel to nearest gauge $d_{\\mathrm{any}}$ (km)")
    ax.set_ylabel("Cumulative $A\\,ds$ within distance (%)")
    ax.set_ylim(0, 102)
    ax.set_title("(c) Along-Channel Proximity CDF ($A\\,ds$-Weighted)")
    ax.legend(loc="upper left", fontsize=7.8)

    # (d) Fraction of A*ds with NO gauge within 100 km (or infinite) across all 4 tiers
    ax = axes[1, 1]
    tree_riv = [f for f in ["hydrosheds_rivers", "merit_basins", "tdx_hydro", "grit_v06_mainstem_tree"] if f in set(sub["hydrofabric"])]
    xt = np.arange(len(TIER_ORDER))
    w = 0.78 / max(len(tree_riv), 1)
    for i, fab in enumerate(tree_riv):
        vals = []
        for tk in TIER_ORDER:
            q = sub[(sub["hydrofabric"] == fab) & (sub["tier"] == tk) & (sub["weight"] == "A_ds") & (sub["metric"] == "d_any_frac_le_100km")]
            vals.append((1.0 - float(q["value"].iloc[0])) * 100.0 if len(q) else np.nan)
        offs = xt + (i - (len(tree_riv) - 1) / 2.0) * w
        ax.bar(offs, vals, width=w * 0.9, color=PALETTE[fab], label=FABRIC_LABELS[fab])
    ax.set_xticks(xt)
    ax.set_xticklabels([TIER_SHORT[t] for t in TIER_ORDER])
    ax.set_ylabel("Share of global $A\\,ds$ > 100 river-km from gauge (%)")
    ax.set_ylim(0, 102)
    ax.set_title("(d) Share of River $A\\,ds$ > 100 km From Nearest Gauge")
    ax.legend(loc="upper right", fontsize=7.8)

    fig.suptitle("Family 4 Continuous 1D River-Network Metrics Across All 4 Global Hydrofabrics", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig04_river_1d_stratified_and_channel_distance.png")
    plt.close(fig)


def plot_fig05_regional_and_ood(df_2d: pd.DataFrame, df_1d: pd.DataFrame, df_ood: pd.DataFrame) -> None:
    fine_fabs = ["hydrosheds_rivers", "merit_basins", "tdx_hydro", "grit_v06_mainstem_tree"]
    present_fabs = [f for f in fine_fabs if f in set(df_2d["hydrofabric"])]
    if not present_fabs:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.4))

    # (a) Continents: U_never (area-weighted, Tier 3 vs Baseline markers)
    ax = axes[0, 0]
    conts = ["Africa", "Asia", "Europe", "North America", "South America", "Oceania"]
    xc = np.arange(len(conts))
    w = 0.78 / len(present_fabs)
    for i, fab in enumerate(present_fabs):
        q3 = df_2d[(df_2d["hydrofabric"] == fab) & (df_2d["tier"] == "tier3_all_known") & (df_2d["domain_type"] == "continent")
                   & (df_2d["metric"] == "U_never") & (df_2d["weight"] == "area") & (df_2d["snapping_mode"] == "outlet")]
        qb = df_2d[(df_2d["hydrofabric"] == fab) & (df_2d["tier"] == "baseline") & (df_2d["domain_type"] == "continent")
                   & (df_2d["metric"] == "U_never") & (df_2d["weight"] == "area") & (df_2d["snapping_mode"] == "outlet")]
        y3 = [float(q3[q3["domain_name"] == c]["value"].iloc[0]) * 100.0 if len(q3[q3["domain_name"] == c]) else np.nan for c in conts]
        yb = [float(qb[qb["domain_name"] == c]["value"].iloc[0]) * 100.0 if len(qb[qb["domain_name"] == c]) else np.nan for c in conts]
        offs = xc + (i - (len(present_fabs) - 1) / 2.0) * w
        ax.bar(offs, y3, width=w * 0.9, color=PALETTE[fab], label=FABRIC_LABELS[fab])
        ax.scatter(offs, yb, color="black", marker="_", s=38, lw=1.6, zorder=4)
    ax.set_xticks(xc)
    ax.set_xticklabels(conts, rotation=18)
    ax.set_ylabel("$U_{\\mathrm{never}}$ (%)")
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("(a) Continental $U_{\\mathrm{never}}$ (Bars = Tier 3, Black ticks = Baseline)")
    ax.legend(loc="upper center", ncol=2, fontsize=7.5)

    # (b) Continents: 1D U_net_1d (A*ds)
    ax = axes[0, 1]
    riv_fabs = [f for f in ["hydrosheds_rivers", "merit_basins", "tdx_hydro", "grit_v06_mainstem_tree"] if f in set(df_1d["hydrofabric"])]
    w2 = 0.78 / max(len(riv_fabs), 1)
    for i, fab in enumerate(riv_fabs):
        q3 = df_1d[(df_1d["hydrofabric"] == fab) & (df_1d["tier"] == "tier3_all_known") & (df_1d["domain_type"] == "continent")
                   & (df_1d["metric"] == "U_net_1d") & (df_1d["weight"] == "A_ds")]
        qb = df_1d[(df_1d["hydrofabric"] == fab) & (df_1d["tier"] == "baseline") & (df_1d["domain_type"] == "continent")
                   & (df_1d["metric"] == "U_net_1d") & (df_1d["weight"] == "A_ds")]
        y3 = [float(q3[q3["domain_name"] == c]["value"].iloc[0]) * 100.0 if len(q3[q3["domain_name"] == c]) else np.nan for c in conts]
        yb = [float(qb[qb["domain_name"] == c]["value"].iloc[0]) * 100.0 if len(qb[qb["domain_name"] == c]) else np.nan for c in conts]
        offs = xc + (i - (len(riv_fabs) - 1) / 2.0) * w2
        ax.bar(offs, y3, width=w2 * 0.9, color=PALETTE[fab], label=FABRIC_LABELS[fab])
        ax.scatter(offs, yb, color="black", marker="_", s=38, lw=1.6, zorder=4)
    ax.set_xticks(xc)
    ax.set_xticklabels(conts, rotation=18)
    ax.set_ylabel("1D $U_{\\mathrm{net,1D}}$ ($A\\,ds$, %)")
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("(b) Continental 1D River $U_{\\mathrm{net,1D}}$ (Bars = Tier 3, Ticks = Baseline)")
    ax.legend(loc="upper center", ncol=2, fontsize=7.5)

    # (c) UNEP Aridity zones: U_cont(1k) across fabrics
    ax = axes[1, 0]
    arids = ["Hyper-arid", "Arid", "Semi-arid", "Dry sub-humid", "Humid"]
    xa = np.arange(len(arids))
    for fab in present_fabs:
        q3 = df_2d[(df_2d["hydrofabric"] == fab) & (df_2d["tier"] == "tier3_all_known") & (df_2d["domain_type"] == "aridity_zone")
                   & (df_2d["metric"] == "U_cont") & (df_2d["weight"] == "area") & (df_2d["snapping_mode"] == "outlet")
                   & np.isclose(df_2d["scale_param"].fillna(-999.0), 1_000.0)]
        y3 = [float(q3[q3["domain_name"] == a]["value"].iloc[0]) * 100.0 if len(q3[q3["domain_name"] == a]) else np.nan for a in arids]
        ax.plot(xa, y3, marker="o", ms=5.0, lw=2.0, color=PALETTE[fab], label=FABRIC_LABELS[fab])
    ax.set_xticks(xa)
    ax.set_xticklabels(arids, rotation=15)
    ax.set_ylabel("$U_{\\mathrm{cont}}(1{,}000\\,\\mathrm{km}^2)$ (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(c) Aridity Gradient of $U_{\\mathrm{cont}}(1{,}000\\,\\mathrm{km}^2)$ (Tier 3)")
    ax.legend(loc="lower left", fontsize=7.8)

    # (d) Family 5 Hydroclimatic OOD distance across continents & tiers (HydroSHEDS L12 native features)
    ax = axes[1, 1]
    if len(df_ood):
        ood_fab = "hydrosheds_l12" if "hydrosheds_l12" in set(df_ood["hydrofabric"]) else df_ood["hydrofabric"].iloc[0]
        qo = df_ood[(df_ood["hydrofabric"] == ood_fab) & (df_ood["domain_type"] == "continent")
                    & (df_ood["metric"] == "U_OOD_knn5_mean") & (df_ood["weight"] == "area")]
        w3 = 0.78 / len(TIER_ORDER)
        for i, tk in enumerate(TIER_ORDER):
            qt = qo[qo["tier"] == tk]
            ys = [float(qt[qt["domain_name"] == c]["value"].iloc[0]) if len(qt[qt["domain_name"] == c]) else np.nan for c in conts]
            offs = xc + (i - (len(TIER_ORDER) - 1) / 2.0) * w3
            ax.bar(offs, ys, width=w3 * 0.9, color=TIER_COLORS[tk], label=TIER_SHORT[tk].replace("\n", " "))
        ax.set_xticks(xc)
        ax.set_xticklabels(conts, rotation=18)
        ax.set_ylabel("Mean 5-NN standardized RMS distance")
        ax.set_ylim(0, 0.49)
        ax.set_title(f"(d) Family 5 Hydroclimatic OOD Distance by Continent ({FABRIC_LABELS.get(ood_fab, ood_fab)})")
        ax.legend(loc="upper center", ncol=2, fontsize=7.5)

    fig.suptitle("Regional, Climatic, and Feature-Space Ungaugedness Across Global Hydrofabrics", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig05_continental_and_aridity_breakdown.png")
    plt.close(fig)


def plot_fig06_grit_bifurcation(df_2d: pd.DataFrame, df_1d: pd.DataFrame) -> None:
    if not {"grit_v06_mainstem_tree", "grit_v06_bifurcating_dag"}.issubset(set(df_2d["hydrofabric"])):
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))

    # (a) Global delta (DAG - Mainstem Tree) across tiers for key metrics
    ax = axes[0]
    metrics_spec = [
        ("2D", "U_never", "area", None, r"$U_{\mathrm{never}}$"),
        ("2D", "U_net_Eq4", "area", None, r"$U_{\mathrm{net}}$ (Eq. 4)"),
        ("2D", "U_cont", "area", 1_000.0, r"$U_{\mathrm{cont}}(1\mathrm{k})$"),
        ("1D", "U_net_1d", "A_ds", None, r"1D $U_{\mathrm{net}}$ ($A\,ds$)"),
    ]
    xm = np.arange(len(metrics_spec))
    w = 0.78 / len(TIER_ORDER)
    for ti, tk in enumerate(TIER_ORDER):
        deltas = []
        for kind, met, wt, sc, _lbl in metrics_spec:
            if kind == "2D":
                q_m = df_2d[(df_2d["hydrofabric"] == "grit_v06_mainstem_tree") & (df_2d["tier"] == tk)
                            & (df_2d["domain_name"] == "common_no_grl_ata") & (df_2d["metric"] == met)
                            & (df_2d["weight"] == wt) & (df_2d["snapping_mode"] == "outlet")]
                q_d = df_2d[(df_2d["hydrofabric"] == "grit_v06_bifurcating_dag") & (df_2d["tier"] == tk)
                            & (df_2d["domain_name"] == "common_no_grl_ata") & (df_2d["metric"] == met)
                            & (df_2d["weight"] == wt)]
                if sc is not None:
                    q_m = q_m[np.isclose(q_m["scale_param"].fillna(-999.0), sc)]
                    q_d = q_d[np.isclose(q_d["scale_param"].fillna(-999.0), sc)]
            else:
                q_m = df_1d[(df_1d["hydrofabric"] == "grit_v06_mainstem_tree") & (df_1d["tier"] == tk)
                            & (df_1d["domain_name"] == "common_no_grl_ata") & (df_1d["metric"] == met)
                            & (df_1d["weight"] == wt)]
                q_d = df_1d[(df_1d["hydrofabric"] == "grit_v06_bifurcating_dag") & (df_1d["tier"] == tk)
                            & (df_1d["domain_name"] == "common_no_grl_ata") & (df_1d["metric"] == met)
                            & (df_1d["weight"] == wt)]
            vm = float(q_m["value"].iloc[0]) if len(q_m) else np.nan
            vd = float(q_d["value"].iloc[0]) if len(q_d) else np.nan
            deltas.append((vd - vm) * 100.0)
        offs = xm + (ti - (len(TIER_ORDER) - 1) / 2.0) * w
        ax.bar(offs, deltas, width=w * 0.9, color=TIER_COLORS[tk], label=TIER_SHORT[tk].replace("\n", " "))
    ax.axhline(0.0, color="black", lw=0.9)
    ax.set_xticks(xm)
    ax.set_xticklabels([s[4] for s in metrics_spec])
    ax.set_ylabel("Bifurcating DAG minus Mainstem Tree (percentage points)")
    ax.set_title("(a) Global Impact of Multi-Threaded River Bifurcations (GRIT v0.6)")
    ax.legend(loc="best", fontsize=7.8)

    # (b) Continental delta in U_never (Tier 3)
    ax = axes[1]
    conts = ["Africa", "Asia", "Europe", "North America", "South America", "Oceania"]
    xc = np.arange(len(conts))
    deltas_c = []
    for c in conts:
        qm = df_2d[(df_2d["hydrofabric"] == "grit_v06_mainstem_tree") & (df_2d["tier"] == "tier3_all_known")
                   & (df_2d["domain_name"] == c) & (df_2d["metric"] == "U_never")
                   & (df_2d["weight"] == "area") & (df_2d["snapping_mode"] == "outlet")]
        qd = df_2d[(df_2d["hydrofabric"] == "grit_v06_bifurcating_dag") & (df_2d["tier"] == "tier3_all_known")
                   & (df_2d["domain_name"] == c) & (df_2d["metric"] == "U_never") & (df_2d["weight"] == "area")]
        vm = float(qm["value"].iloc[0]) if len(qm) else np.nan
        vd = float(qd["value"].iloc[0]) if len(qd) else np.nan
        deltas_c.append((vd - vm) * 100.0)
    ax.bar(xc, deltas_c, color="#D55E00", width=0.6)
    ax.axhline(0.0, color="black", lw=0.9)
    ax.set_xticks(xc)
    ax.set_xticklabels(conts, rotation=18)
    ax.set_ylabel("$\\Delta U_{\\mathrm{never}}$ (DAG − Mainstem, percentage points)")
    ax.set_title("(b) Regional Bifurcation Impact on $U_{\\mathrm{never}}$ (Tier 3)")

    fig.suptitle("GRIT v0.6 Multi-Threaded / Deltaic Bifurcation Sensitivity", fontsize=12.0, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig06_grit_bifurcation_impact.png")
    plt.close(fig)


def main() -> None:
    setup_mpl()
    df_2d, df_1d, df_ood, _df_cty = consolidate_results()
    audit_snapping()
    summary = build_headline_summary(df_2d, df_1d, df_ood)

    plot_fig01_hydrosheds_scale(df_2d, df_1d)
    plot_fig02_cross_hydrofabric(summary)
    plot_fig03_resolution_curves(df_2d)
    plot_fig04_river_1d(df_1d)
    plot_fig05_regional_and_ood(df_2d, df_1d, df_ood)
    plot_fig06_grit_bifurcation(df_2d, df_1d)

    print(f"All master tables saved in {MASTER_DIR} and figures saved in {FIGURES_DIR}", flush=True)


if __name__ == "__main__":
    main()
