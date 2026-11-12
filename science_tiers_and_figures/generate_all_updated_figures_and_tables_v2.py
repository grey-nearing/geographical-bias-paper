#!/usr/bin/env python3
"""Computes 4-regime Lead-Time-1 Table S1 & Table S3 (with Option B flood events, Sudan repaired, and Option 1 Active-only DA + FT fallback) and renders all 6 updated figures at exact kix aspect ratios."""

from concurrent.futures import ProcessPoolExecutor
import glob
import io
import json
import os
import shutil
import sys
import time
import types
import warnings

os.environ.setdefault("GEO_BIAS_DATA_DIR", "./data/canonical_v2")
os.environ.setdefault("GEO_BIAS_THIRD_PARTY_DIR", "./data/canonical_v2")
os.environ.setdefault("GEO_BIAS_OUTPUT_DIR", "./data/canonical_v2")

sys.path = [p for p in sys.path if not p.endswith("/google3")]

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
import seaborn as sns
from sklearn.metrics import r2_score

gfile_mod = types.ModuleType("google3.pyglib.gfile")
gfile_mod.Open = open
gfile_mod.Exists = os.path.exists
gfile_mod.MakeDirs = lambda p: os.makedirs(p, exist_ok=True)
gfile_mod.Glob = glob.glob
gfile_mod.ListDir = os.listdir
pyglib_mod = types.ModuleType("google3.pyglib")
pyglib_mod.gfile = gfile_mod
sys.modules["google3.pyglib"] = pyglib_mod
sys.modules["google3.pyglib.gfile"] = gfile_mod
gfile = gfile_mod

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils import file_paths
from utils.io_utils import load_experiment_metrics

warnings.filterwarnings("ignore")

NATURE_W = 183.0 / 25.4  # 7.204724 in
SAVE_DPI = 350
CANON_V2_DIR = "./data/canonical_v2"
SCRATCH_DIR = "./data/scratch"
CUR_SCRATCH_DIR = "./data/scratch"
MASKS_PATH = f"{CUR_SCRATCH_DIR}/tier_active_and_all_masks.parquet"
PRED_DIR = f"{SCRATCH_DIR}/tiered_counterfactual_lead1/predictions"
FEAT_DIR = "./data/canonical_v2/coverage_fractions_tiered_v2"
if not os.path.exists(f"{FEAT_DIR}/tier1a_active_post2016_ccby_8811.parquet"):
    FEAT_DIR = f"{CANON_V2_DIR}/coverage_fractions"
SUMMARY_CSV = f"{SCRATCH_DIR}/tiered_counterfactual_lead1/anchored_3tiers_exposure_summary_lead1.csv"
OUT_DIR = f"{SCRATCH_DIR}/updated_manuscript_assets_lead1"
os.makedirs(OUT_DIR, exist_ok=True)

_G_STUDY_INDEX = None
_G_HAS_SYS = None


def _init_worker(study_index, has_sys):
    global _G_STUDY_INDEX, _G_HAS_SYS
    _G_STUDY_INDEX = study_index
    _G_HAS_SYS = has_sys


def _normalize_hybas_id(series: pd.Series) -> pd.Series:
    nums = pd.to_numeric(
        series.astype(str)
        .str.replace("hybas_", "", regex=False)
        .str.replace(r"\.0$", "", regex=True),
        errors="coerce",
    )
    return "hybas_" + nums.astype("Int64").astype(str)


def pad_image_to_exact_aspect_ratio(img_path: str, target_h_over_w: float, out_path: str = None):
    """Pads an image with white margin so its pixel aspect ratio h/w matches target_h_over_w exactly."""
    if out_path is None:
        out_path = img_path
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    cur_ratio = h / float(w)
    if abs(cur_ratio - target_h_over_w) < 1e-4:
        if out_path != img_path:
            im.save(out_path)
        return
    if cur_ratio < target_h_over_w:
        new_w = w
        new_h = int(round(w * target_h_over_w))
    else:
        new_h = h
        new_w = int(round(h / target_h_over_w))
    canvas = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    dx = (new_w - w) // 2
    dy = (new_h - h) // 2
    canvas.paste(im, (dx, dy))
    canvas.save(out_path)


def _process_one_metric(m: str) -> list[dict]:
    sname = file_paths.METRIC_SHORT_NAME_MAP.get(m, m)
    out_min = file_paths.OUTLIER_MIN_MAP.get(m, 0.0)
    out_max = file_paths.OUTLIER_MAX_MAP.get(m, 1.0)
    out_rows = []

    regimes = [
        ("ungauged", "", "M_base (000)"),
        ("ungauged_ft", "_ft", "M_FT (100)"),
        ("ungauged_da", "_da", "M_DA (001)"),
        ("ungauged_ft_da", "_ft_da", "M_FT_DA (111)"),
    ]
    for model_tag, exp_sfx, label_tag in regimes:
        r_sk = load_experiment_metrics(file_paths.METRICS_DIR, f"random{exp_sfx}", m, lead_time="1").dropna()
        s_sk = load_experiment_metrics(file_paths.METRICS_DIR, f"separated{exp_sfx}", m, lead_time="1").dropna()
        g_sk = load_experiment_metrics(file_paths.METRICS_DIR, f"gauged{exp_sfx}", m, lead_time="1").dropna()

        u_obs_raw = pd.concat([r_sk, s_sk], axis=0)
        all3_obs_raw = pd.concat([g_sk, r_sk, s_sk], axis=0)

        u_obs = u_obs_raw.clip(lower=out_min, upper=out_max)
        all3_obs = all3_obs_raw.clip(lower=out_min, upper=out_max)

        u_q25, u_med, u_q75 = np.percentile(u_obs.values, [25, 50, 75])
        u_iqr = u_q75 - u_q25

        r_q25, r_med, r_q75 = np.percentile(r_sk.clip(lower=out_min, upper=out_max).values, [25, 50, 75])
        s_q25, s_med, s_q75 = np.percentile(s_sk.clip(lower=out_min, upper=out_max).values, [25, 50, 75])
        g_q25, g_med_cv, g_q75 = np.percentile(g_sk.clip(lower=out_min, upper=out_max).values, [25, 50, 75])

        r_raw_q25, r_raw_med, r_raw_q75 = np.percentile(r_sk.values, [25, 50, 75])
        s_raw_q25, s_raw_med, s_raw_q75 = np.percentile(s_sk.values, [25, 50, 75])
        g_raw_q25, g_raw_med, g_raw_q75 = np.percentile(g_sk.values, [25, 50, 75])

        cv_path = os.path.join(file_paths.CV_PREDS_DIR, f"{model_tag}_regression_cv_preds_{m}.csv")
        df_cv = pd.read_csv(cv_path).dropna(subset=["true", "pred"])
        cv_r2_all = float(r2_score(df_cv["true"], df_cv["pred"]))
        n_cv = len(df_cv)
        df_cv_ung = df_cv.iloc[n_cv // 3 :]
        cv_r2_ung_only = float(r2_score(df_cv_ung["true"], df_cv_ung["pred"]))

        hybas_path = os.path.join(file_paths.HYBAS_PREDS_DIR, f"global_hybas_predictions_{model_tag}_{m}.csv")
        df_hybas_pred = pd.read_csv(hybas_path, index_col=0)
        s_hybas = pd.to_numeric(df_hybas_pred.iloc[:, 0], errors="coerce")
        if len(s_hybas) == len(_G_STUDY_INDEX) and str(s_hybas.index[0]) == str(_G_STUDY_INDEX[0]):
            vals = s_hybas.values
        else:
            idx_str = s_hybas.index.astype(str)
            if not str(idx_str[0]).startswith("hybas_"):
                idx_str = "hybas_" + idx_str
            s_hybas.index = idx_str
            s_hybas = s_hybas.reindex(_G_STUDY_INDEX)
            vals = s_hybas.fillna(s_hybas.median()).values

        vals_clipped = np.clip(vals, out_min, out_max)
        g_q10, g_q25_p, g_med_p, g_q75_p, g_q90 = np.nanpercentile(vals_clipped, [10, 25, 50, 75, 90])
        g_iqr_p = g_q75_p - g_q25_p

        sys_vals = vals_clipped[_G_HAS_SYS]
        nosys_vals = vals_clipped[~_G_HAS_SYS]
        sys_q25, sys_med, sys_q75 = np.nanpercentile(sys_vals, [25, 50, 75])
        nosys_q25, nosys_med, nosys_q75 = np.nanpercentile(nosys_vals, [25, 50, 75])

        out_rows.append({
            "model_regime": label_tag,
            "model_tag": model_tag,
            "metric_id": m,
            "metric_short": sname,
            "ungauged_cv_n": int(len(u_obs)),
            "obs_ungauged_median": float(u_med),
            "obs_ungauged_iqr": float(u_iqr),
            "obs_ungauged_q25": float(u_q25),
            "obs_ungauged_q75": float(u_q75),
            "cv_gauged_raw_median": float(g_raw_med),
            "cv_gauged_raw_q25": float(g_raw_q25),
            "cv_gauged_raw_q75": float(g_raw_q75),
            "cv_random_raw_median": float(r_raw_med),
            "cv_random_raw_q25": float(r_raw_q25),
            "cv_random_raw_q75": float(r_raw_q75),
            "cv_separated_raw_median": float(s_raw_med),
            "cv_separated_raw_q25": float(s_raw_q25),
            "cv_separated_raw_q75": float(s_raw_q75),
            "skill_predictor_cv_r2": cv_r2_all,
            "skill_predictor_cv_r2_ungauged_subset": cv_r2_ung_only,
            "global_pred_median": float(g_med_p),
            "global_pred_iqr": float(g_iqr_p),
            "global_pred_q10": float(g_q10),
            "global_pred_q25": float(g_q25_p),
            "global_pred_q75": float(g_q75_p),
            "global_pred_q90": float(g_q90),
            "pred_median_with_warning_sys_60": float(sys_med),
            "pred_iqr_with_warning_sys_60": float(sys_q75 - sys_q25),
            "pred_median_no_warning_sys_113": float(nosys_med),
            "pred_iqr_no_warning_sys_113": float(nosys_q75 - nosys_q25),
            "pred_median_no_warning_sys_114": float(nosys_med),
            "pred_iqr_no_warning_sys_114": float(nosys_q75 - nosys_q25),
        })
    return out_rows


def _load_world_gdf() -> gpd.GeoDataFrame:
    with gfile.Open(file_paths.NATURAL_EARTH_PATH, "rb") as f:
        world = gpd.read_file(f)
    if "name" in world.columns:
        world = world[world["name"] != "Antarctica"].copy()
    world = world.dropna(subset=[world.geometry.name])
    world = world[world.is_valid & (~world.is_empty)].copy()
    world["map_name"] = world["name"].replace(file_paths.country_name_map)
    iso_fix = {"France": "FRA", "Norway": "NOR"}
    world["iso_clean"] = [
        iso_fix.get(mname, str(iso).strip())
        for mname, iso in zip(world["map_name"], world.get("iso_a3", ""))
    ]
    return world


def _load_inform_vulnerability(world_gdf: gpd.GeoDataFrame) -> pd.Series:
    country_to_iso = dict(zip(world_gdf["map_name"], world_gdf["iso_clean"]))
    with gfile.Open(file_paths.RISK_FILE, "rt") as f:
        df_inform = pd.read_csv(io.StringIO(f.read()), skiprows=[0, 2])
    df_inform["ISO3"] = df_inform["ISO3"].astype(str).str.strip()
    df_inform["COUNTRY_CLEAN"] = (
        df_inform["COUNTRY"].astype(str).str.strip().replace(file_paths.country_name_map)
    )
    df_inform["VULN_NUM"] = pd.to_numeric(df_inform["VULNERABILITY"], errors="coerce")
    iso_vuln = (
        df_inform.dropna(subset=["ISO3", "VULN_NUM"])
        .drop_duplicates("ISO3")
        .set_index("ISO3")["VULN_NUM"]
    )
    name_vuln = (
        df_inform.dropna(subset=["COUNTRY_CLEAN", "VULN_NUM"])
        .drop_duplicates("COUNTRY_CLEAN")
        .set_index("COUNTRY_CLEAN")["VULN_NUM"]
    )
    all_countries = sorted(set(country_to_iso.keys()).union(set(name_vuln.index)))
    out = {}
    for c in all_countries:
        iso = country_to_iso.get(c)
        val = iso_vuln.get(iso, np.nan) if iso else np.nan
        if pd.isna(val):
            val = name_vuln.get(c, np.nan)
        out[c] = val
    return pd.Series(out, name="Vulnerability")


def _fmt_sci_tex(p: float) -> str:
    if p <= 0 or np.isnan(p):
        return "0"
    if p >= 0.001:
        return f"{p:.3f}"
    exp = int(np.floor(np.log10(p)))
    mant = p / (10 ** exp)
    return f"{mant:.1f} \\times 10^{{{exp}}}"


def main():
    t0 = time.time()
    print("Step 1: Loading study watersheds and computing 4-regime Table S1...", flush=True)
    exp_df = pd.read_parquet(
        f"{CANON_V2_DIR}/v2_exposure_blockers/statistics/basin_exposure_and_blockers_v2.parquet",
        columns=["country_name", "has_system", "E_total"],
    )
    exp_df.index = _normalize_hybas_id(pd.Series(exp_df.index)).values
    study_index = exp_df.index.astype(str).values
    has_sys = exp_df["has_system"].values.astype(bool)

    table_s1_metrics = [
        "NSE", "KGE", "log-NSE", "log-KGE", "Beta-NSE", "Beta-KGE", "Alpha-NSE", "pearson-r",
        "1.01-year-2.0-window-recall", "2.0-year-2.0-window-recall", "5.0-year-2.0-window-recall",
        "10.0-year-2.0-window-recall", "20.0-year-2.0-window-recall",
        "1.01-year-2.0-window-precision", "2.0-year-2.0-window-precision", "5.0-year-2.0-window-precision",
        "10.0-year-2.0-window-precision", "20.0-year-2.0-window-precision",
        "1.01-year-1.0-window-recall", "2.0-year-1.0-window-recall", "5.0-year-1.0-window-recall",
        "10.0-year-1.0-window-recall", "20.0-year-1.0-window-recall",
        "1.01-year-1.0-window-precision", "2.0-year-1.0-window-precision", "5.0-year-1.0-window-precision",
        "10.0-year-1.0-window-precision", "20.0-year-1.0-window-precision",
        "1.01-year-0.0-window-recall", "2.0-year-0.0-window-recall", "5.0-year-0.0-window-recall",
        "10.0-year-0.0-window-recall", "20.0-year-0.0-window-recall",
        "1.01-year-0.0-window-precision", "2.0-year-0.0-window-precision", "5.0-year-0.0-window-precision",
        "10.0-year-0.0-window-precision", "20.0-year-0.0-window-precision",
    ]

    if os.path.exists(f"{OUT_DIR}/Table_S1_4regimes_lead1.csv") and os.path.getmtime(f"{OUT_DIR}/Table_S1_4regimes_lead1.csv") > 1794348000:
        df_ts1 = pd.read_csv(f"{OUT_DIR}/Table_S1_4regimes_lead1.csv")
        print(f"   Loaded existing {OUT_DIR}/Table_S1_4regimes_lead1.csv ({len(df_ts1)} rows)", flush=True)
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=12, initializer=_init_worker, initargs=(study_index, has_sys)) as ex:
            for m_rows in ex.map(_process_one_metric, table_s1_metrics):
                rows.extend(m_rows)
        df_ts1 = pd.DataFrame(rows)
        df_ts1.to_csv(f"{OUT_DIR}/Table_S1_4regimes_lead1.csv", index=False)
        print(f"   Saved {OUT_DIR}/Table_S1_4regimes_lead1.csv in {time.time() - t0:.1f}s", flush=True)

    # Print key NSE, KGE, 5y/10y/20y recall numbers across all 4 regimes
    for m_check in ["NSE", "KGE", "5.0-year-2.0-window-recall", "10.0-year-2.0-window-recall", "20.0-year-2.0-window-recall"]:
        sub = df_ts1[df_ts1["metric_id"] == m_check]
        print(f"   === {m_check} ===")
        for _, r in sub.iterrows():
            print(
                f"      {r['model_tag']:15s}: CV Sep={r['cv_separated_raw_median']:.3f} ({r['cv_separated_raw_q25']:.3f}..{r['cv_separated_raw_q75']:.3f}), "
                f"Rand={r['cv_random_raw_median']:.3f}, Gauged={r['cv_gauged_raw_median']:.3f} | "
                f"ObsUng={r['obs_ungauged_median']:.2f} ({r['obs_ungauged_iqr']:.2f}) | "
                f"R2={r['skill_predictor_cv_r2']:.2f} | "
                f"GlobalPred={r['global_pred_median']:.2f} (IQR {r['global_pred_iqr']:.2f}; {r['global_pred_q25']:.2f}..{r['global_pred_q75']:.2f}) | "
                f"Sys60={r['pred_median_with_warning_sys_60']:.2f} | NoSys114={r['pred_median_no_warning_sys_114']:.2f}"
            )

    # =========================================================================
    # Step 2: Compute 4-regime Table S3 (174 countries, Sudan included) and Country Correlations
    # =========================================================================
    print("Step 2: Computing 4-regime Table S3 and country correlations (with Option 1 Active-only DA + FT fallback)...", flush=True)
    world_gdf = _load_world_gdf()
    vuln_series = _load_inform_vulnerability(world_gdf)

    df_ll = pd.read_csv(file_paths.HYBAS_LATLON_PATH, index_col=0)
    lat_c = next(c for c in df_ll.columns if "lat" in c.lower())
    lon_c = next(c for c in df_ll.columns if "lon" in c.lower())
    ll_clean = pd.DataFrame({
        "hybas_id": _normalize_hybas_id(pd.Series(df_ll.index)).values,
        "lat": pd.to_numeric(df_ll[lat_c], errors="coerce").values,
        "lon": pd.to_numeric(df_ll[lon_c], errors="coerce").values,
    }).dropna().drop_duplicates("hybas_id").set_index("hybas_id")

    exp_df["country_name"] = exp_df["country_name"].replace(file_paths.country_name_map)
    exp_df = exp_df.join(ll_clean, how="left")

    rp_raw = pd.read_csv(f"{CANON_V2_DIR}/exposure/global_basin_exposure_defended_by_return_period.csv")
    id_col = "hybas_id" if "hybas_id" in rp_raw.columns else rp_raw.columns[0]
    rp_raw[id_col] = _normalize_hybas_id(rp_raw[id_col]).values
    rp_raw = rp_raw.drop_duplicates(subset=[id_col]).set_index(id_col).reindex(study_index).fillna(0.0)

    def _get_exp_col(df, t):
        for c in [f"exposed_pop_{t}", f"exposure_{t}yr", str(t)]:
            if c in df.columns:
                return df[c].values
        raise KeyError(f"Missing exposure column for T={t}")

    e5 = 0.05 * _get_exp_col(rp_raw, 5)
    e10 = 0.075 * _get_exp_col(rp_raw, 10)
    e20 = 0.04 * _get_exp_col(rp_raw, 20) + 0.02 * _get_exp_col(rp_raw, 50) + 0.005 * _get_exp_col(rp_raw, 100)
    e_tot = e5 + e10 + e20
    exp_df["E_total"] = e_tot

    f_t1a = pd.read_parquet(f"{FEAT_DIR}/tier1a_active_post2016_ccby_8811.parquet", columns=["total_upstream_coverage"]).reindex(study_index)
    f_t4 = pd.read_parquet(f"{FEAT_DIR}/tier4_all_publicly_knowable_121819.parquet", columns=["total_upstream_coverage"]).reindex(study_index)
    masks_df = pd.read_parquet(MASKS_PATH).reindex(study_index)
    out_t4_all = masks_df["tier4_all_publicly_knowable_121819__ge95_all"].fillna(False).values.astype(bool)
    out_t4_active = masks_df["tier4_all_publicly_knowable_121819__ge95_active"].fillna(False).values.astype(bool)

    exp_df["zero_up_t4"] = (f_t4["total_upstream_coverage"].fillna(0.0).values <= 0.0).astype(float) * 100.0
    exp_df["zero_up_t1a"] = (f_t1a["total_upstream_coverage"].fillna(0.0).values <= 0.0).astype(float) * 100.0

    def load_p(m_tag, tier, mode="joint"):
        return pd.read_parquet(f"{PRED_DIR}/{m_tag}__{tier}__{mode}.parquet").reindex(study_index)

    p_b1a = load_p("base", "tier1a_active_post2016_ccby_8811")
    p_b4 = load_p("base", "tier4_all_publicly_knowable_121819")
    p_ft4 = load_p("ft", "tier4_all_publicly_knowable_121819")
    p_da4 = load_p("da", "tier4_all_publicly_knowable_121819")
    p_ftda4 = load_p("ft_da", "tier4_all_publicly_knowable_121819")

    p_b5 = load_p("base", "tier5_universal_idealized_ceiling")
    p_ft1a = load_p("ft", "tier1a_active_post2016_ccby_8811")
    p_da1a = load_p("da", "tier1a_active_post2016_ccby_8811")
    p_da5 = load_p("da", "tier5_universal_idealized_ceiling")
    p_ftda5 = load_p("ft_da", "tier5_universal_idealized_ceiling")

    r5_b1, r10_b1, r20_b1 = p_b1a["r5_u"].values, p_b1a["r10_u"].values, p_b1a["r20_u"].values
    w_b1 = e5 * r5_b1 + e10 * r10_b1 + e20 * r20_b1
    u_b1 = e_tot - w_b1
    exp_df["warned_t1a"] = w_b1
    exp_df["unwarned_t1a"] = u_b1

    r5_b4, r10_b4, r20_b4 = p_b4["r5_u"].values, p_b4["r10_u"].values, p_b4["r20_u"].values
    w_b4 = e5 * r5_b4 + e10 * r10_b4 + e20 * r20_b4
    u_b4 = e_tot - w_b4
    exp_df["warned_t4"] = w_b4
    exp_df["unwarned_t4"] = u_b4

    def clip_gain(r5, r10, r20, rb5=r5_b1, rb10=r10_b1, rb20=r20_b1):
        return (
            e5 * np.maximum(0.0, r5 - rb5)
            + e10 * np.maximum(0.0, r10 - rb10)
            + e20 * np.maximum(0.0, r20 - rb20)
        )

    def net_gain(r5, r10, r20, rb5=r5_b1, rb10=r10_b1, rb20=r20_b1):
        return (
            e5 * (r5 - rb5)
            + e10 * (r10 - rb10)
            + e20 * (r20 - rb20)
        )

    # Tier 3 (121,819 Knowable Gauges) from Tier 1a (8,811) with Option 1 Active-only DA + FT fallback
    exp_df["d_t4_000"] = clip_gain(p_b4["r5_u"].values, p_b4["r10_u"].values, p_b4["r20_u"].values)
    exp_df["d_t4_000_net"] = net_gain(p_b4["r5_u"].values, p_b4["r10_u"].values, p_b4["r20_u"].values)

    r5_t4_100 = np.where(out_t4_all, p_ft1a["r5_u"].values, r5_b1)
    r10_t4_100 = np.where(out_t4_all, p_ft1a["r10_u"].values, r10_b1)
    r20_t4_100 = np.where(out_t4_all, p_ft1a["r20_u"].values, r20_b1)
    exp_df["d_t4_100"] = clip_gain(r5_t4_100, r10_t4_100, r20_t4_100)
    exp_df["d_t4_100_net"] = net_gain(r5_t4_100, r10_t4_100, r20_t4_100)

    r5_t4_001 = np.where(out_t4_active, p_da1a["r5_u"].values, r5_b1)
    r10_t4_001 = np.where(out_t4_active, p_da1a["r10_u"].values, r10_b1)
    r20_t4_001 = np.where(out_t4_active, p_da1a["r20_u"].values, r20_b1)
    exp_df["d_t4_001"] = clip_gain(r5_t4_001, r10_t4_001, r20_t4_001)
    exp_df["d_t4_001_net"] = net_gain(r5_t4_001, r10_t4_001, r20_t4_001)

    r5_t4_101 = np.where(out_t4_active, p_da4["r5_u"].values, p_b4["r5_u"].values)
    r10_t4_101 = np.where(out_t4_active, p_da4["r10_u"].values, p_b4["r10_u"].values)
    r20_t4_101 = np.where(out_t4_active, p_da4["r20_u"].values, p_b4["r20_u"].values)
    exp_df["d_t4_101"] = clip_gain(r5_t4_101, r10_t4_101, r20_t4_101)
    exp_df["d_t4_101_net"] = net_gain(r5_t4_101, r10_t4_101, r20_t4_101)

    # Combined 111: Active gauges get M_FT_DA, Non-Active gauges get M_FT, Ungauged get M_base
    r5_t4_111 = np.where(out_t4_active, p_ftda4["r5_u"].values, np.where(out_t4_all, p_ft4["r5_u"].values, p_b4["r5_u"].values))
    r10_t4_111 = np.where(out_t4_active, p_ftda4["r10_u"].values, np.where(out_t4_all, p_ft4["r10_u"].values, p_b4["r10_u"].values))
    r20_t4_111 = np.where(out_t4_active, p_ftda4["r20_u"].values, np.where(out_t4_all, p_ft4["r20_u"].values, p_b4["r20_u"].values))
    exp_df["d_t4_111"] = clip_gain(r5_t4_111, r10_t4_111, r20_t4_111)
    exp_df["d_t4_111_net"] = net_gain(r5_t4_111, r10_t4_111, r20_t4_111)

    # Tier 5 (Universal Gauging, 964,084) from Tier 1a (8,811)
    exp_df["d_t5_000"] = clip_gain(p_b5["r5_u"].values, p_b5["r10_u"].values, p_b5["r20_u"].values)
    exp_df["d_t5_000_net"] = net_gain(p_b5["r5_u"].values, p_b5["r10_u"].values, p_b5["r20_u"].values)
    exp_df["d_t5_100"] = clip_gain(p_ft1a["r5_u"].values, p_ft1a["r10_u"].values, p_ft1a["r20_u"].values)
    exp_df["d_t5_001"] = clip_gain(p_da1a["r5_u"].values, p_da1a["r10_u"].values, p_da1a["r20_u"].values)
    exp_df["d_t5_101"] = clip_gain(p_da5["r5_u"].values, p_da5["r10_u"].values, p_da5["r20_u"].values)
    exp_df["d_t5_111"] = clip_gain(p_ftda5["r5_u"].values, p_ftda5["r10_u"].values, p_ftda5["r20_u"].values)
    exp_df["d_t5_111_net"] = net_gain(p_ftda5["r5_u"].values, p_ftda5["r10_u"].values, p_ftda5["r20_u"].values)

    # Also from Tier 4 baseline (for reference)
    exp_df["d_t5_000_from_t4"] = clip_gain(p_b5["r5_u"].values, p_b5["r10_u"].values, p_b5["r20_u"].values, r5_b4, r10_b4, r20_b4)

    sum_cols = [
        "E_total", "warned_t1a", "unwarned_t1a", "warned_t4", "unwarned_t4",
        "d_t4_000", "d_t4_100", "d_t4_001", "d_t4_101", "d_t4_111",
        "d_t4_000_net", "d_t4_100_net", "d_t4_001_net", "d_t4_101_net", "d_t4_111_net",
        "d_t5_000", "d_t5_100", "d_t5_001", "d_t5_101", "d_t5_111",
        "d_t5_000_net", "d_t5_111_net",
        "d_t5_000_from_t4",
    ]
    grp = exp_df.groupby("country_name", as_index=False)
    df_c = grp[sum_cols].sum()
    df_c["n_basins"] = grp.size()["size"].values
    df_c["has_system"] = grp["has_system"].first()["has_system"].values.astype(bool)
    df_c["zero_up_pct_t4"] = grp["zero_up_t4"].mean()["zero_up_t4"].values
    df_c["zero_up_pct_t1a"] = grp["zero_up_t1a"].mean()["zero_up_t1a"].values
    df_c["inform_vulnerability"] = df_c["country_name"].map(vuln_series)

    df_c["unwarned_pct_t1a"] = np.where(df_c["E_total"] > 0, 100.0 * df_c["unwarned_t1a"] / df_c["E_total"], 0.0)
    df_c["unwarned_pct_t4"] = np.where(df_c["E_total"] > 0, 100.0 * df_c["unwarned_t4"] / df_c["E_total"], 0.0)
    for col in ["d_t4_000", "d_t4_111", "d_t5_000", "d_t5_101", "d_t5_111", "d_t4_000_net", "d_t4_111_net", "d_t5_000_net", "d_t5_111_net"]:
        df_c[f"pct_unw_t1a_{col}"] = np.where(df_c["unwarned_t1a"] > 0, 100.0 * df_c[col] / df_c["unwarned_t1a"], 0.0)
    df_c["pct_unw_t4_d_t5_000_from_t4"] = np.where(df_c["unwarned_t4"] > 0, 100.0 * df_c["d_t5_000_from_t4"] / df_c["unwarned_t4"], 0.0)

    df_c = df_c.sort_values("E_total", ascending=False).reset_index(drop=True)
    df_c.to_csv(f"{OUT_DIR}/Table_S3_4regimes_lead1.csv", index=False)
    print(f"   Saved {OUT_DIR}/Table_S3_4regimes_lead1.csv ({len(df_c)} countries: {int(df_c['has_system'].sum())} with CAP, {int((~df_c['has_system']).sum())} without CAP)", flush=True)

    df_c_exp = df_c[df_c["E_total"] > 0].copy()
    df_c_vuln = df_c_exp.dropna(subset=["inform_vulnerability"]).copy()

    corrs = {}
    for name, zcol, pct_col, abs_col in [
        ("t1a_to_t5_000_vs_zup_t1a", "zero_up_pct_t1a", "pct_unw_t1a_d_t5_000", "d_t5_000"),
        ("t1a_to_t5_000_vs_zup_t4", "zero_up_pct_t4", "pct_unw_t1a_d_t5_000", "d_t5_000"),
        ("t4_to_t5_000_vs_zup_t4", "zero_up_pct_t4", "pct_unw_t4_d_t5_000_from_t4", "d_t5_000_from_t4"),
        ("t1a_to_t5_111_vs_zup_t1a", "zero_up_pct_t1a", "pct_unw_t1a_d_t5_111", "d_t5_111"),
        ("t1a_to_t5_111_vs_zup_t4", "zero_up_pct_t4", "pct_unw_t1a_d_t5_111", "d_t5_111"),
        ("t1a_to_t4_111_vs_zup_t1a", "zero_up_pct_t1a", "pct_unw_t1a_d_t4_111", "d_t4_111"),
        ("t1a_to_t4_000_vs_zup_t4", "zero_up_pct_t4", "pct_unw_t1a_d_t4_000", "d_t4_000"),
    ]:
        rz_p, pz_p = spearmanr(df_c_exp[zcol], df_c_exp[pct_col])
        rz_a, pz_a = spearmanr(df_c_exp[zcol], df_c_exp[abs_col])
        rv_p, pv_p = spearmanr(df_c_vuln["inform_vulnerability"], df_c_vuln[pct_col])
        rv_a, pv_a = spearmanr(df_c_vuln["inform_vulnerability"], df_c_vuln[abs_col])
        corrs[name] = {
            "r_zup_pct": float(rz_p), "p_zup_pct": float(pz_p),
            "r_zup_abs": float(rz_a), "p_zup_abs": float(pz_a),
            "r_vuln_pct": float(rv_p), "p_vuln_pct": float(pv_p),
            "r_vuln_abs": float(rv_a), "p_vuln_abs": float(pv_a),
        }
        print(
            f"   Corr {name:28s}: zup_pct r={rz_p:+.3f} (p={pz_p:.2e}) | "
            f"vuln_pct r={rv_p:+.3f} (p={pv_p:.2e}) | vuln_abs r={rv_a:+.3f} (p={pv_a:.2e})"
        )

    with open(f"{OUT_DIR}/country_corrs_4regimes_lead1.json", "w") as f:
        json.dump(corrs, f, indent=2)

    # =========================================================================
    # Step 3: Render Figure 3 (kix.a4fk56xa8gtg, ratio 331.5097548774387 / 468.0)
    # =========================================================================
    print("Step 3: Rendering updated Figure 3 (3-panel counterfactual exposure reduction with FT + DA)...", flush=True)
    plt.rcParams.update({
        "figure.dpi": SAVE_DPI,
        "savefig.dpi": SAVE_DPI,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 7.0,
    })

    df_ws_pts = exp_df.dropna(subset=["lat", "lon"]).copy()
    df_ws_bias = df_ws_pts[df_ws_pts["d_t5_111"] > 0].sort_values("d_t5_111", ascending=True)
    df_ws_nobias = df_ws_pts[df_ws_pts["d_t5_111"] <= 0]

    df_sum = pd.read_csv(SUMMARY_CSV)
    df_ge95 = df_sum[df_sum["mask_type"] == "ge95_outlet"].copy()

    def get_row(t_key, mech):
        r = df_ge95[(df_ge95["tier_key"] == t_key) & (df_ge95["mechanism"] == mech)]
        return r.iloc[0]

    r_univ_111 = get_row("tier5_universal_idealized_ceiling", "111_all_three_combined")
    cap_m_u111 = float(r_univ_111["delta_clip_nosys113_M"])
    unw_nosys_base_m = float(df_c[~df_c["has_system"]]["unwarned_t1a"].sum()) / 1e6
    cap_pct_u111 = 100.0 * cap_m_u111 / unw_nosys_base_m
    glob_m_u111 = float(r_univ_111["delta_clip_global_M"])
    glob_pct_u111 = float(r_univ_111["delta_clip_pct_unw"])
    n_sys_cnt = int(df_c["has_system"].sum())
    n_nosys_cnt = int((~df_c["has_system"]).sum())

    target_ratio_fig3 = 331.5097548774387 / 468.0
    fig_h = NATURE_W * target_ratio_fig3
    fig = plt.figure(figsize=(NATURE_W, fig_h), dpi=SAVE_DPI, facecolor="#ffffff")
    gs = gridspec.GridSpec(
        4, 1, figure=fig,
        height_ratios=[1.00, 0.055, 0.22, 0.90],
        hspace=0.05, left=0.01, right=0.99, top=0.945, bottom=0.095,
    )

    ax_a = fig.add_subplot(gs[0])
    world_gdf.plot(ax=ax_a, color="#F2F4F7", edgecolor="none", zorder=1)
    ax_a.scatter(
        df_ws_nobias["lon"].values, df_ws_nobias["lat"].values,
        s=0.05, marker=".", color="#D8DFE6", linewidths=0, rasterized=True, zorder=2
    )
    sc_a = ax_a.scatter(
        df_ws_bias["lon"].values, df_ws_bias["lat"].values,
        c=np.maximum(df_ws_bias["d_t5_111"].values, 0.05), s=0.08, marker=".",
        cmap="magma_r", norm=mcolors.LogNorm(vmin=0.1, vmax=1e4),
        linewidths=0, rasterized=True, zorder=3
    )
    world_gdf.boundary.plot(ax=ax_a, color="#455A64", linewidth=0.22, zorder=4)
    ax_a.set_xlim([-180, 180])
    ax_a.set_ylim([-58, 84])
    ax_a.set_axis_off()
    ax_a.set_title(
        f"(a) Unforecasted flood exposure reduction under universal gauging ({cap_m_u111:.1f}M/yr CAP-masked; {glob_m_u111:.1f}M/yr global)",
        fontsize=7.3, fontweight="bold", loc="center", pad=3.0
    )

    gs_cb = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], width_ratios=[0.12, 0.76, 0.12])
    cax = fig.add_subplot(gs_cb[0, 1])
    cb = fig.colorbar(sc_a, cax=cax, orientation="horizontal", extend="both")
    cb.set_label(
        f"Unforecasted flood exposure reduction at α = 1.0 (persons/yr; {cap_m_u111:.1f}M/yr CAP-masked, {cap_pct_u111:.1f}%; {glob_m_u111:.1f}M/yr global, {glob_pct_u111:.1f}%)",
        fontsize=6.2, fontweight="bold", labelpad=2.0
    )
    cb.ax.tick_params(labelsize=6.0, pad=1.2)

    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=gs[3], width_ratios=[0.06, 0.45, 0.07, 0.42], wspace=0.0
    )
    ax_b = fig.add_subplot(gs_bot[0, 1])
    ax_c = fig.add_subplot(gs_bot[0, 3])

    b_tiers = [
        ("tier_caravan_estreams_34577", "Tier 1:\nCaravan + ES\n(34,577)"),
        ("tier_public_discharge_93827", "Tier 2:\nPublic records\n(93,827)"),
        ("tier4_all_publicly_knowable_121819", "Tier 3:\nAll knowable\n(121,819)"),
        ("tier5_universal_idealized_ceiling", "Universal\ngauging\n(964,084)"),
    ]
    xb = np.arange(len(b_tiers))
    wb = 0.195
    v_000 = [float(get_row(tk, "000_global_train")["delta_clip_nosys113_M"]) for tk, _ in b_tiers]
    v_100 = [float(get_row(tk, "010_ft_only")["delta_clip_nosys113_M"]) for tk, _ in b_tiers]
    v_001 = [float(get_row(tk, "001_da_only")["delta_clip_nosys113_M"]) for tk, _ in b_tiers]
    v_111 = [float(get_row(tk, "111_all_three_combined")["delta_clip_nosys113_M"]) for tk, _ in b_tiers]

    ax_b.bar(xb - 1.5 * wb, v_000, width=wb, color="#0072B2", edgecolor="#222222", linewidth=0.4, label="Global training alone")
    ax_b.bar(xb - 0.5 * wb, v_100, width=wb, color="#009E73", edgecolor="#222222", linewidth=0.4, label="Local FT alone")
    ax_b.bar(xb + 0.5 * wb, v_001, width=wb, color="#43A2CA", edgecolor="#222222", linewidth=0.4, label="Real-time DA alone (Active)")
    ax_b.bar(xb + 1.5 * wb, v_111, width=wb, color="#E66101", edgecolor="#222222", linewidth=0.4, label="Combined (train + FT + Active DA)")

    for i, (val0, val_ft, val_da, val_c) in enumerate(zip(v_000, v_100, v_001, v_111)):
        lbl0 = f"{val0:.2f}" if val0 < 0.1 else f"{val0:.1f}"
        lbl_ft = f"{val_ft:.2f}" if val_ft < 0.1 else f"{val_ft:.1f}"
        lbl_da = f"{val_da:.2f}" if val_da < 0.1 else f"{val_da:.1f}"
        dy_ft = 0.24 if (i == 0 and abs(val_ft - val_da) < 0.05) else 0.10
        dy_da = 0.08 if (i == 0 and abs(val_ft - val_da) < 0.05) else 0.10
        ax_b.text(xb[i] - 1.5 * wb, val0 + 0.10, lbl0, ha="center", va="bottom", fontsize=4.6, color="#004D80", fontweight="bold")
        ax_b.text(xb[i] - 0.5 * wb, val_ft + dy_ft, lbl_ft, ha="center", va="bottom", fontsize=4.5, color="#00664B", fontweight="bold")
        ax_b.text(xb[i] + 0.5 * wb, val_da + dy_da, lbl_da, ha="center", va="bottom", fontsize=4.5, color="#1D6A8A", fontweight="bold")
        ax_b.text(xb[i] + 1.5 * wb, val_c + 0.12, f"{val_c:.1f}M", ha="center", va="bottom", fontsize=5.2, color="#993404", fontweight="bold")

    ax_b.axvline(2.5, color="#78909C", linestyle="--", linewidth=0.7)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels([lbl for _, lbl in b_tiers], fontsize=5.7, fontweight="bold")
    y_max_b = max(7.2, max(v_111) * 1.22)
    ax_b.set_ylim(0, y_max_b)
    ax_b.set_yticks([0, 1.5, 3.0, 4.5, 6.0, 7.5])
    ax_b.tick_params(axis="y", labelsize=6.0)
    ax_b.set_ylabel("CAP-masked reduction (M/yr)", fontsize=6.5, fontweight="bold")
    ax_b.set_title("(b) Exposure reduction in CAP-gap countries", fontsize=7.3, fontweight="bold", loc="left", pad=3.5)
    ax_b.legend(loc="upper left", fontsize=5.0, frameon=True, framealpha=0.95, facecolor="#ffffff", edgecolor="#CFD8DC", handlelength=1.0, borderpad=0.22, labelspacing=0.22)
    ax_b.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.4)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    c_tiers = [
        ("tier1a_active_post2016_ccby_8811", "000_global_train", "Post-2016\nCaravan\n(8,811)"),
        ("tier_caravan_estreams_34577", "111_all_three_combined", "Tier 1:\nCaravan+ES\n(34,577)"),
        ("tier_public_discharge_93827", "111_all_three_combined", "Tier 2:\nPublic Q\n(93,827)"),
        ("tier4_all_publicly_knowable_121819", "111_all_three_combined", "Tier 3:\nAll knowable\n(121,819)"),
        ("tier5_universal_idealized_ceiling", "111_all_three_combined", "Universal\ngauging\n(964,084)"),
    ]
    xc = np.arange(len(c_tiers))
    wc = 0.54
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

    ax_c.bar(xc, w_sys_vals, width=wc, color="#0072B2", edgecolor="#222222", linewidth=0.45, label=f"Established warning sys. ({n_sys_cnt})")
    ax_c.bar(xc, w_nosys_vals, width=wc, bottom=w_sys_vals, color="#D55E00", edgecolor="#222222", linewidth=0.45, label=f"CAP warning gap ({n_nosys_cnt})")

    for i, (ws, wn, wt) in enumerate(zip(w_sys_vals, w_nosys_vals, w_tot_vals)):
        ax_c.text(xc[i], wt + 1.5, f"{wt:.1f}M", ha="center", va="bottom", fontsize=5.6, fontweight="bold", color="#222222")
        ax_c.text(xc[i], ws * 0.5, f"{ws:.1f}", ha="center", va="center", fontsize=5.1, fontweight="bold", color="#ffffff")
        ax_c.text(xc[i], ws + wn * 0.5, f"{wn:.1f}", ha="center", va="center", fontsize=5.0, fontweight="bold", color="#ffffff")

    ax_c.axvline(3.5, color="#78909C", linestyle="--", linewidth=0.7)
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels([lbl for _, _, lbl in c_tiers], fontsize=5.6, fontweight="bold")
    ax_c.set_ylim(0, max(112.0, float(w_tot_vals.max()) * 1.18))
    ax_c.set_yticks([0, 25, 50, 75, 100])
    ax_c.tick_params(axis="y", labelsize=6.0)
    ax_c.set_ylabel("Total exposure forecasted (M/yr)", fontsize=6.5, fontweight="bold")
    ax_c.set_title("(c) Total forecasted exposure", fontsize=7.3, fontweight="bold", loc="left", pad=3.5)
    ax_c.legend(loc="upper left", fontsize=5.4, frameon=True, framealpha=0.95, facecolor="#ffffff", edgecolor="#CFD8DC", handlelength=1.1, borderpad=0.25)
    ax_c.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.4)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    fig3_path = f"{OUT_DIR}/Fig3_counterfactual_exposure_lead1.png"
    fig.savefig(fig3_path, dpi=SAVE_DPI)
    plt.close(fig)
    pad_image_to_exact_aspect_ratio(fig3_path, target_ratio_fig3, fig3_path)
    shutil.copy2(fig3_path, f"{SCRATCH_DIR}/figure_4_3panel_anchored.png")
    print(f"   Saved {fig3_path}", flush=True)

    # =========================================================================
    # Step 4: Render & Pad Fig. S2, Fig. S3, Fig. S4 to exact kix aspect ratios
    # =========================================================================
    print("Step 4: Rendering Fig. S2, Fig. S3 (Option B 5y recall vs OOD), and Fig. S4 (Option B global predicted skill) at exact kix aspect ratios...", flush=True)
    repo_fig_dir = "./science_results/figures"
    figs2_path = f"{OUT_DIR}/FigS2_delta_nse_lead1.png"
    figs3_path = f"{OUT_DIR}/FigS3_recall_vs_ood_lead1.png"
    figs4_path = f"{OUT_DIR}/FigS4_global_predicted_skill_lead1.png"

    if os.path.exists(f"{repo_fig_dir}/Fig_S2_delta_nse_kge_lead1.png"):
        pad_image_to_exact_aspect_ratio(f"{repo_fig_dir}/Fig_S2_delta_nse_kge_lead1.png", 295.6898449224612 / 468.0, figs2_path)

    # Render Fig. S3 (3x3 OOD sextile plot of 5-yr 2-day flood recall under Option B)
    sns.set_theme(style="whitegrid", palette="colorblind")
    plt.rcParams.update({
        "figure.dpi": SAVE_DPI,
        "savefig.dpi": SAVE_DPI,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 7.5,
    })
    file_paths.init("5.0-year-2.0-window-recall")
    rand_anom_df = pd.read_csv(file_paths.RANDOM_ANOMALIES_PATH).set_index("gauge_id")
    sep_anom_df = pd.read_csv(file_paths.SEPARATED_ANOMALIES_PATH).set_index("gauge_id")
    g_sk_ed4 = load_experiment_metrics(file_paths.METRICS_DIR, "gauged", "5.0-year-2.0-window-recall", rename_to="gauged_rec5", lead_time="1")
    r_sk_ed4 = load_experiment_metrics(file_paths.METRICS_DIR, "random", "5.0-year-2.0-window-recall", rename_to="random_rec5", lead_time="1")
    s_sk_ed4 = load_experiment_metrics(file_paths.METRICS_DIR, "separated", "5.0-year-2.0-window-recall", rename_to="separated_rec5", lead_time="1")

    df_top_ed4 = rand_anom_df.join(g_sk_ed4, how="inner").join(r_sk_ed4, how="inner").dropna(subset=["gauged_rec5", "random_rec5"])
    df_top_ed4 = df_top_ed4[(df_top_ed4["gauged_rec5"] >= 0.0) & (df_top_ed4["random_rec5"] >= 0.0)]
    df_bot_ed4 = sep_anom_df.join(g_sk_ed4, how="inner").join(s_sk_ed4, how="inner").dropna(subset=["gauged_rec5", "separated_rec5"])
    df_bot_ed4 = df_bot_ed4[(df_bot_ed4["gauged_rec5"] >= 0.0) & (df_bot_ed4["separated_rec5"] >= 0.0)]

    anom_cols = ["anomaly_score_if", "anomaly_score_ocsvm", "anomaly_score_lof"]
    anom_titles = ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"]
    ed4_ltrs = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"], ["(g)", "(h)", "(i)"]]
    sextile_labels = ["Q1\n(OOD)", "Q2", "Q3", "Q4", "Q5", "Q6\n(In-dist)"]

    fig_s3_ood, axes_s3 = plt.subplots(3, 3, figsize=(NATURE_W, 5.80), sharey=True)
    for i, (acol, atitle) in enumerate(zip(anom_cols, anom_titles)):
        df_top_ed4[f"{acol}_bin"] = pd.qcut(df_top_ed4[acol], q=6, labels=sextile_labels, duplicates="drop")
        df_bot_ed4[f"{acol}_bin"] = pd.qcut(df_bot_ed4[acol], q=6, labels=sextile_labels, duplicates="drop")

        ax0 = axes_s3[0, i]
        sns.boxplot(
            data=df_top_ed4, x=f"{acol}_bin", y="random_rec5", hue=f"{acol}_bin",
            palette="Blues_r", legend=False, showfliers=False, whis=[10, 90], linewidth=0.55, ax=ax0
        )
        ax0.axhline(df_top_ed4["random_rec5"].median(), color="#0d47a1", ls="--", lw=0.9)
        ax0.set_xlabel("")
        ax0.set_title(f"{ed4_ltrs[0][i]} Random: {atitle}", fontsize=7.4, fontweight="bold", loc="left", pad=2)
        ax0.set_ylabel("5-yr recall (Random)" if i == 0 else "", fontsize=7.0, fontweight="bold")
        ax0.tick_params(axis="x", labelsize=6.0)
        sns.despine(ax=ax0)

        ax1 = axes_s3[1, i]
        sns.boxplot(
            data=df_bot_ed4, x=f"{acol}_bin", y="separated_rec5", hue=f"{acol}_bin",
            palette="Oranges_r", legend=False, showfliers=False, whis=[10, 90], linewidth=0.55, ax=ax1
        )
        ax1.axhline(df_bot_ed4["separated_rec5"].median(), color="#e65100", ls="--", lw=0.9)
        ax1.set_xlabel("")
        ax1.set_title(f"{ed4_ltrs[1][i]} Separated: {atitle}", fontsize=7.4, fontweight="bold", loc="left", pad=2)
        ax1.set_ylabel("5-yr recall (Separated)" if i == 0 else "", fontsize=7.0, fontweight="bold")
        ax1.tick_params(axis="x", labelsize=6.0)
        sns.despine(ax=ax1)

        ax2 = axes_s3[2, i]
        sns.boxplot(
            data=df_top_ed4, x=f"{acol}_bin", y="gauged_rec5", hue=f"{acol}_bin",
            palette="Greens_r", legend=False, showfliers=False, whis=[10, 90], linewidth=0.55, ax=ax2
        )
        ax2.axhline(df_top_ed4["gauged_rec5"].median(), color="#1b5e20", ls="--", lw=0.9)
        ax2.set_xlabel(f"{atitle} sextile", fontsize=6.8, fontweight="bold")
        ax2.set_title(f"{ed4_ltrs[2][i]} Gauged: {atitle}", fontsize=7.4, fontweight="bold", loc="left", pad=2)
        ax2.set_ylabel("5-yr recall (Gauged)" if i == 0 else "", fontsize=7.0, fontweight="bold")
        ax2.tick_params(axis="x", labelsize=6.0)
        sns.despine(ax=ax2)

    fig_s3_ood.tight_layout()
    fig_s3_ood.savefig(figs3_path, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.03)
    if os.path.exists(repo_fig_dir):
        fig_s3_ood.savefig(f"{repo_fig_dir}/Fig_S3_recall_vs_ood_lead1.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig_s3_ood)
    pad_image_to_exact_aspect_ratio(figs3_path, 375.99199599799897 / 468.0, figs3_path)
    if os.path.exists(repo_fig_dir):
        shutil.copy2(figs3_path, f"{repo_fig_dir}/Fig_S3_recall_vs_ood_lead1.png")
    print(f"   Saved {figs3_path}", flush=True)

    # Render Fig. S4 (horizontal boxplots of global predicted skill across NSE, KGE, Recall 1y..20y, Precision 1y..20y under Option B)
    metrics_f2 = [
        ("NSE", "NSE"),
        ("KGE", "KGE"),
        ("1.01-year-2.0-window-recall", "Recall (1y)"),
        ("2.0-year-2.0-window-recall", "Recall (2y)"),
        ("5.0-year-2.0-window-recall", "Recall (5y)"),
        ("10.0-year-2.0-window-recall", "Recall (10y)"),
        ("20.0-year-2.0-window-recall", "Recall (20y)"),
        ("1.01-year-2.0-window-precision", "Precision (1y)"),
        ("2.0-year-2.0-window-precision", "Precision (2y)"),
        ("5.0-year-2.0-window-precision", "Precision (5y)"),
        ("10.0-year-2.0-window-precision", "Precision (10y)"),
        ("20.0-year-2.0-window-precision", "Precision (20y)"),
    ]
    df_base_idx = df_ts1[df_ts1["model_tag"] == "ungauged"].set_index("metric_id")
    box_stats_f2 = []
    for m_id, m_lbl in metrics_f2:
        r = df_base_idx.loc[m_id]
        box_stats_f2.append({
            "label": m_lbl,
            "med": float(r["global_pred_median"]),
            "q1": float(r["global_pred_q25"]),
            "q3": float(r["global_pred_q75"]),
            "whislo": float(r["global_pred_q10"]),
            "whishi": float(r["global_pred_q90"]),
            "fliers": [],
        })

    fig_s4_bxp, ax_s4 = plt.subplots(figsize=(NATURE_W, 3.85))
    bp = ax_s4.bxp(
        list(reversed(box_stats_f2)),
        vert=False, patch_artist=True, showfliers=False, widths=0.62,
    )
    colors_bxp = ["#CC79A7"] * 5 + ["#009E73"] * 5 + ["#0072B2"] * 2
    for patch, col in zip(bp["boxes"], colors_bxp):
        patch.set_facecolor(col)
        patch.set_alpha(0.82)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.55)
    for med_line in bp["medians"]:
        med_line.set_color("black")
        med_line.set_linewidth(1.2)
    for w_line in bp["whiskers"] + bp["caps"]:
        w_line.set_linewidth(0.6)
    ax_s4.axvline(0.0, color="#616161", linestyle="--", linewidth=0.7)
    ax_s4.set_xlim([-0.46, 0.72])
    ax_s4.set_xlabel("Predicted hydrological skill across 964,084 global watersheds", fontsize=7.5, fontweight="bold")
    ax_s4.set_title("Global watershed forecast skill distributions", fontsize=8.2, fontweight="bold", loc="left", pad=4)
    handles_s4 = [
        Patch(facecolor="#0072B2", edgecolor="black", alpha=0.82, label="Continuous hydrograph (NSE, KGE)"),
        Patch(facecolor="#009E73", edgecolor="black", alpha=0.82, label="Flood recall (2-day window)"),
        Patch(facecolor="#CC79A7", edgecolor="black", alpha=0.82, label="Flood precision (2-day window)"),
    ]
    ax_s4.legend(handles=handles_s4, loc="lower left", frameon=True, framealpha=0.94, fontsize=6.4)
    ax_s4.grid(axis="x", linestyle="--", alpha=0.45)
    sns.despine(ax=ax_s4)
    fig_s4_bxp.tight_layout()
    fig_s4_bxp.savefig(figs4_path, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.03)
    if os.path.exists(repo_fig_dir):
        fig_s4_bxp.savefig(f"{repo_fig_dir}/Fig_S4_global_predicted_skill_lead1.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig_s4_bxp)
    pad_image_to_exact_aspect_ratio(figs4_path, 245.12056028014007 / 468.0, figs4_path)
    if os.path.exists(repo_fig_dir):
        shutil.copy2(figs4_path, f"{repo_fig_dir}/Fig_S4_global_predicted_skill_lead1.png")
    print(f"   Saved {figs4_path}", flush=True)

    # =========================================================================
    # Step 5: Render Fig. S5 (kix.apc8scdtoi0m, ratio 305.0 / 467.55751533742335)
    # =========================================================================
    print("Step 5: Rendering Fig. S5 (4-panel exposure & warning coverage)...", flush=True)
    sns.set_theme(style="whitegrid", palette="colorblind")
    plt.rcParams.update({
        "figure.dpi": SAVE_DPI,
        "savefig.dpi": SAVE_DPI,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 7.5,
    })

    global_tot_exp = float(e_tot.sum())
    global_warned_t1a = float(w_b1.sum())
    rho_zup_exp_t4, p_zup_exp_t4 = spearmanr(df_c_exp["zero_up_pct_t4"], df_c_exp["E_total"])

    target_ratio_figs5 = 305.0 / 467.55751533742335
    fig_s5 = plt.figure(figsize=(NATURE_W, NATURE_W * target_ratio_figs5), dpi=SAVE_DPI, facecolor="#ffffff")
    gs_s5 = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.05], hspace=0.28, wspace=0.16, left=0.06, right=0.98, top=0.94, bottom=0.09)

    ax_s5a = fig_s5.add_subplot(gs_s5[0, 0])
    w_s5a = world_gdf.copy()
    cap_map = df_c.set_index("country_name")["has_system"].to_dict()
    w_s5a["has_cap"] = w_s5a["map_name"].map(cap_map)
    world_gdf.plot(ax=ax_s5a, color="#ECEFF1", edgecolor="#B0BEC5", linewidth=0.20)
    w_s5a[w_s5a["has_cap"] == True].plot(ax=ax_s5a, color="#0072B2", edgecolor="#263238", linewidth=0.18, alpha=0.86)
    w_s5a[w_s5a["has_cap"] == False].plot(ax=ax_s5a, color="#D55E00", edgecolor="#263238", linewidth=0.18, alpha=0.84)
    ax_s5a.set_xlim([-180, 180])
    ax_s5a.set_ylim([-58, 84])
    ax_s5a.set_axis_off()
    ax_s5a.set_title("(a) National flood warning systems", fontsize=8.0, fontweight="bold", loc="left", pad=3)
    handles_s5a = [
        Patch(facecolor="#0072B2", edgecolor="#263238", alpha=0.86, label=f"Established system ({n_sys_cnt} countries)"),
        Patch(facecolor="#D55E00", edgecolor="#263238", alpha=0.84, label=f"Lacking system ({n_nosys_cnt} countries)"),
    ]
    ax_s5a.legend(handles=handles_s5a, loc="lower left", frameon=True, framealpha=0.94, fontsize=6.0)

    df_ws_zero = df_ws_pts[df_ws_pts["E_total"] <= 0]
    df_ws_pos = df_ws_pts[df_ws_pts["E_total"] > 0].sort_values("E_total", ascending=True)
    df_ws_pos["pct_warned_ws"] = (100.0 * df_ws_pos["warned_t1a"] / df_ws_pos["E_total"]).clip(0.0, 100.0)

    base_ylgnbu = plt.get_cmap("YlGnBu")
    cmap_exp = mcolors.LinearSegmentedColormap.from_list("YlGnBu_trunc", base_ylgnbu(np.linspace(0.24, 1.0, 256)))

    ax_s5b = fig_s5.add_subplot(gs_s5[0, 1])
    world_gdf.plot(ax=ax_s5b, color="#F4F6F8", edgecolor="none", zorder=1)
    ax_s5b.scatter(df_ws_zero["lon"].values, df_ws_zero["lat"].values, s=0.04, marker=".", color="#D8DFE6", linewidths=0, rasterized=True, zorder=2)
    sc_s5b = ax_s5b.scatter(
        df_ws_pos["lon"].values, df_ws_pos["lat"].values,
        c=np.maximum(df_ws_pos["E_total"].values, 0.1), s=0.08, marker=".",
        cmap=cmap_exp, norm=mcolors.LogNorm(vmin=1.0, vmax=1e5),
        linewidths=0, rasterized=True, zorder=3
    )
    world_gdf.boundary.plot(ax=ax_s5b, color="#455A64", linewidth=0.18, zorder=4)
    cbar_s5b = fig_s5.colorbar(sc_s5b, ax=ax_s5b, orientation="horizontal", fraction=0.052, pad=0.02, shrink=0.84, extend="both")
    cbar_s5b.set_label(f"Flood exposure (persons/yr; total {global_tot_exp/1e6:.1f}M/yr)", fontsize=6.4, fontweight="bold")
    cbar_s5b.ax.tick_params(labelsize=6.0)
    ax_s5b.set_xlim([-180, 180])
    ax_s5b.set_ylim([-58, 84])
    ax_s5b.set_axis_off()
    ax_s5b.set_title("(b) Baseline annual flood exposure", fontsize=8.0, fontweight="bold", loc="left", pad=3)

    ax_s5c = fig_s5.add_subplot(gs_s5[1, 0])
    for has_sys_flag, lbl, col, mk in [
        (True, "Established system", "#0072B2", "o"),
        (False, "Lacking system", "#D55E00", "^"),
    ]:
        sub = df_c_exp[df_c_exp["has_system"] == has_sys_flag]
        ax_s5c.scatter(
            sub["zero_up_pct_t4"], sub["E_total"],
            s=16, color=col, marker=mk, alpha=0.78, edgecolor="black", linewidth=0.35, label=lbl
        )

    x_grid = np.linspace(0, 100, 200)
    log_y = np.log10(np.maximum(df_c_exp["E_total"].values, 1.0))
    slope, intercept = np.polyfit(df_c_exp["zero_up_pct_t4"].values, log_y, 1)
    ax_s5c.plot(x_grid, 10 ** (intercept + slope * x_grid), color="#263238", linestyle="--", linewidth=1.0)

    for cname, label_txt, dx_pt, dy_pt in [
        ("India", "India", 7, 2),
        ("Bangladesh", "Bangladesh", -38, -14),
        ("China", "China", -32, 8),
        ("Brazil", "Brazil", -42, -11),
        ("Nigeria", "Nigeria", 6, 6),
        ("United States of America", "USA", -26, -14),
    ]:
        rc = df_c_exp[df_c_exp["country_name"] == cname]
        if not rc.empty:
            xv = float(rc["zero_up_pct_t4"].iloc[0])
            yv = float(rc["E_total"].iloc[0])
            ax_s5c.annotate(
                label_txt, xy=(xv, yv), xytext=(dx_pt, dy_pt),
                textcoords="offset points",
                fontsize=6.0, fontweight="bold", color="#263238",
                arrowprops=dict(arrowstyle="-", color="#546E7A", lw=0.4, shrinkA=1, shrinkB=1)
            )

    ax_s5c.set_yscale("log")
    ax_s5c.set_xlim(-4, 108)
    ax_s5c.set_ylim(5.0, 4.5e8)
    ax_s5c.text(
        0.03, 0.06,
        f"Spearman $r_s = {rho_zup_exp_t4:+.2f}$ ($p = {p_zup_exp_t4:.3f}$, $N = {len(df_c_exp)}$)",
        transform=ax_s5c.transAxes, fontsize=6.2,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="#FFF8E1", edgecolor="#B0BEC5", alpha=0.94)
    )
    ax_s5c.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=6.0)
    ax_s5c.set_xlabel("Watersheds with zero upstream gauges (%)", fontsize=6.8, fontweight="bold")
    ax_s5c.set_ylabel("Flood exposure (persons/yr)", fontsize=7.0, fontweight="bold")
    ax_s5c.set_title("(c) Zero upstream gauges vs. flood exposure", fontsize=8.0, fontweight="bold", loc="left", pad=3)
    ax_s5c.grid(True, which="both", linestyle="--", alpha=0.4)
    sns.despine(ax=ax_s5c)

    ax_s5d = fig_s5.add_subplot(gs_s5[1, 1])
    world_gdf.plot(ax=ax_s5d, color="#F4F6F8", edgecolor="none", zorder=1)
    ax_s5d.scatter(df_ws_zero["lon"].values, df_ws_zero["lat"].values, s=0.04, marker=".", color="#D8DFE6", linewidths=0, rasterized=True, zorder=2)
    sc_s5d = ax_s5d.scatter(
        df_ws_pos["lon"].values, df_ws_pos["lat"].values,
        c=df_ws_pos["pct_warned_ws"].values, s=0.08, marker=".",
        cmap="viridis", vmin=10.0, vmax=50.0,
        linewidths=0, rasterized=True, zorder=3
    )
    world_gdf.boundary.plot(ax=ax_s5d, color="#455A64", linewidth=0.18, zorder=4)
    pct_global_warned_t1a = (global_warned_t1a / global_tot_exp) * 100.0
    cbar_s5d = fig_s5.colorbar(sc_s5d, ax=ax_s5d, orientation="horizontal", fraction=0.052, pad=0.02, shrink=0.84, extend="both")
    cbar_s5d.set_label(
        f"Exposure warned at α = 1.0 (%; global {pct_global_warned_t1a:.1f}%, {global_warned_t1a/1e6:.1f}M/yr)",
        fontsize=6.4, fontweight="bold"
    )
    cbar_s5d.ax.tick_params(labelsize=6.0)
    ax_s5d.set_xlim([-180, 180])
    ax_s5d.set_ylim([-58, 84])
    ax_s5d.set_axis_off()
    ax_s5d.set_title("(d) Flood exposure warned by global model", fontsize=8.0, fontweight="bold", loc="left", pad=3)

    figs5_path = f"{OUT_DIR}/FigS5_exposure_and_warnings_lead1.png"
    fig_s5.savefig(figs5_path, dpi=SAVE_DPI)
    plt.close(fig_s5)
    pad_image_to_exact_aspect_ratio(figs5_path, target_ratio_figs5, figs5_path)
    print(f"   Saved {figs5_path}", flush=True)

    # =========================================================================
    # Step 6: Render Fig. S6 (kix.ud4n2ku0blj8, ratio 219.83591795897948 / 468.0)
    # =========================================================================
    print("Step 6: Rendering Fig. S6 (2-panel country data-bias gap vs. zero upstream gauges & vulnerability)...", flush=True)
    target_ratio_figs6 = 219.83591795897948 / 468.0
    fig_s6 = plt.figure(figsize=(NATURE_W, NATURE_W * target_ratio_figs6), dpi=SAVE_DPI, facecolor="#ffffff")
    gs_s6 = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.0], wspace=0.22, left=0.07, right=0.98, top=0.90, bottom=0.15)

    df_c_bias = df_c[df_c["unwarned_t1a"] > 0].copy()
    df_c_bias["bias_pct_unw"] = df_c_bias["pct_unw_t1a_d_t5_000"]
    df_c_bias["bias_abs"] = df_c_bias["d_t5_000"]

    ax_s6a = fig_s6.add_subplot(gs_s6[0, 0])
    rho_bcov, p_bcov = spearmanr(df_c_bias["zero_up_pct_t4"], df_c_bias["bias_pct_unw"])
    for has_sys_flag, lbl, col, mk in [
        (True, "Established system", "#0072B2", "o"),
        (False, "Lacking system", "#D55E00", "^"),
    ]:
        sub = df_c_bias[df_c_bias["has_system"] == has_sys_flag]
        sizes = np.clip(np.sqrt(sub["E_total"]) / 45.0, 10.0, 95.0)
        ax_s6a.scatter(sub["zero_up_pct_t4"], sub["bias_pct_unw"], s=sizes, color=col, marker=mk, alpha=0.75, edgecolor="black", linewidth=0.35, label=lbl)

    sns.regplot(data=df_c_bias, x="zero_up_pct_t4", y="bias_pct_unw", scatter=False, ax=ax_s6a, color="#263238", line_kws={"linestyle": "--", "linewidth": 1.0}, ci=95)
    for cname, label_txt, dx_pt, dy_pt in [
        ("India", "India", -34, 14),
        ("Bangladesh", "Bangladesh", -54, 4),
        ("Pakistan", "Pakistan", -32, 20),
        ("Somalia", "Somalia", -34, 22),
        ("China", "China", -36, 6),
        ("Brazil", "Brazil", -40, -16),
        ("Nigeria", "Nigeria", -54, -10),
        ("United States of America", "USA", -26, 10),
    ]:
        rc = df_c_bias[df_c_bias["country_name"] == cname]
        if not rc.empty:
            xv = float(rc["zero_up_pct_t4"].iloc[0])
            yv = float(rc["bias_pct_unw"].iloc[0])
            ax_s6a.annotate(
                label_txt, xy=(xv, yv), xytext=(dx_pt, dy_pt),
                textcoords="offset points",
                fontsize=6.0, fontweight="bold", color="#263238",
                arrowprops=dict(arrowstyle="-", color="#546E7A", lw=0.4, shrinkA=1, shrinkB=1)
            )

    ax_s6a.text(
        0.03, 0.89,
        f"$r_s = {rho_bcov:+.2f}$ ($p = {_fmt_sci_tex(p_bcov)}$)",
        transform=ax_s6a.transAxes, fontsize=6.2,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="#FFF8E1", edgecolor="#B0BEC5", alpha=0.94)
    )
    ax_s6a.set_xlabel("Watersheds with zero upstream gauges (%)", fontsize=6.9, fontweight="bold")
    ax_s6a.set_ylabel("Unwarned exposure from data bias (%)", fontsize=7.0, fontweight="bold")
    ax_s6a.set_title("(a) Data-bias gap vs. zero upstream gauges", fontsize=7.8, fontweight="bold", loc="left", pad=3)
    ax_s6a.legend(loc="lower right", fontsize=6.0, frameon=True, framealpha=0.94)
    ax_s6a.grid(True, linestyle="--", alpha=0.45)
    sns.despine(ax=ax_s6a)

    ax_s6b = fig_s6.add_subplot(gs_s6[0, 1])
    df_c_bv = df_c_bias.dropna(subset=["inform_vulnerability"]).copy()
    df_c_bv["t4_pct_unw"] = df_c_bv["pct_unw_t1a_d_t4_000"]
    df_c_bv["t4_abs"] = df_c_bv["d_t4_000"]
    rho_bv_pct, p_bv_pct = spearmanr(df_c_bv["inform_vulnerability"], df_c_bv["t4_pct_unw"])
    rho_bv_abs, p_bv_abs = spearmanr(df_c_bv["inform_vulnerability"], df_c_bv["t4_abs"])
    for has_sys_flag, lbl, col, mk in [
        (True, "Established system", "#0072B2", "o"),
        (False, "Lacking system", "#D55E00", "^"),
    ]:
        sub = df_c_bv[df_c_bv["has_system"] == has_sys_flag]
        sizes = np.clip(np.sqrt(sub["E_total"]) / 45.0, 10.0, 95.0)
        ax_s6b.scatter(sub["inform_vulnerability"], sub["t4_pct_unw"], s=sizes, color=col, marker=mk, alpha=0.75, edgecolor="black", linewidth=0.35, label=lbl)

    sns.regplot(data=df_c_bv, x="inform_vulnerability", y="t4_pct_unw", scatter=False, ax=ax_s6b, color="#263238", line_kws={"linestyle": "--", "linewidth": 1.0}, ci=95)
    for cname, label_txt, dx_pt, dy_pt in [
        ("United States of America", "USA", -24, 14),
        ("China", "China", -8, 12),
        ("India", "India", -24, 12),
        ("Bangladesh", "Bangladesh", -32, 12),
        ("Pakistan", "Pakistan", -40, -14),
        ("Nigeria", "Nigeria", 8, -12),
        ("Somalia", "Somalia", -36, 14),
    ]:
        rc = df_c_bv[df_c_bv["country_name"] == cname]
        if not rc.empty:
            xv = float(rc["inform_vulnerability"].iloc[0])
            yv = float(rc["t4_pct_unw"].iloc[0])
            ax_s6b.annotate(
                label_txt, xy=(xv, yv), xytext=(dx_pt, dy_pt),
                textcoords="offset points",
                fontsize=6.0, fontweight="bold", color="#263238",
                arrowprops=dict(arrowstyle="-", color="#546E7A", lw=0.4, shrinkA=1, shrinkB=1)
            )

    ax_s6b.text(
        0.03, 0.80,
        f"% gap: $r_s = {rho_bv_pct:+.2f}$ ($p = {_fmt_sci_tex(p_bv_pct)}$)\n"
        f"Persons/yr: $r_s = {rho_bv_abs:+.2f}$ ($p = {_fmt_sci_tex(p_bv_abs)}$)",
        transform=ax_s6b.transAxes, fontsize=6.1,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="#FFF8E1", edgecolor="#B0BEC5", alpha=0.94)
    )
    ax_s6b.set_xlabel("INFORM 2026 Vulnerability Index", fontsize=6.9, fontweight="bold")
    ax_s6b.set_ylabel("Unwarned exposure from data bias (%)", fontsize=7.0, fontweight="bold")
    ax_s6b.set_title("(b) Data-bias gap vs. vulnerability", fontsize=7.8, fontweight="bold", loc="left", pad=3)
    ax_s6b.legend(loc="lower right", fontsize=6.0, frameon=True, framealpha=0.94)
    ax_s6b.grid(True, linestyle="--", alpha=0.45)
    sns.despine(ax=ax_s6b)

    figs6_path = f"{OUT_DIR}/FigS6_vulnerability_and_bias_lead1.png"
    fig_s6.savefig(figs6_path, dpi=SAVE_DPI)
    plt.close(fig_s6)
    pad_image_to_exact_aspect_ratio(figs6_path, target_ratio_figs6, figs6_path)

    # Also pad Fig 4 (KernelSHAP categories) and render Fig S7 (KernelSHAP top 35) at exact kix aspect ratios
    fig4_src = f"{CANON_V2_DIR}/science_results/Fig5_TreeSHAP_lead1.png"
    fig4_dst = f"{OUT_DIR}/Fig4_shap_categories_lead1.png"
    if os.path.exists(fig4_src):
        pad_image_to_exact_aspect_ratio(fig4_src, 493.16394306800214 / 468.0, fig4_dst)
        print(f"   Saved {fig4_dst}", flush=True)

    top35_csv = f"{CANON_V2_DIR}/science_results/treeshap_top35_features_lead1.csv"
    figs7_dst = f"{OUT_DIR}/FigS7_shap_top35_lead1.png"
    if os.path.exists(top35_csv):
        df_t35 = pd.read_csv(top35_csv, index_col=0)
        all_12_m = [
            "NSE", "KGE",
            "1.01-year-2.0-window-precision", "2.0-year-2.0-window-precision", "5.0-year-2.0-window-precision", "10.0-year-2.0-window-precision", "20.0-year-2.0-window-precision",
            "1.01-year-2.0-window-recall", "2.0-year-2.0-window-recall", "5.0-year-2.0-window-recall", "10.0-year-2.0-window-recall", "20.0-year-2.0-window-recall",
        ]
        short_col_labels = [
            "NSE", "KGE",
            "Prec (1y)", "Prec (2y)", "Prec (5y)", "Prec (10y)", "Prec (20y)",
            "Rec (1y)", "Rec (2y)", "Rec (5y)", "Rec (10y)", "Rec (20y)",
        ]
        df_s7_mat = df_t35[all_12_m].copy()
        df_s7_mat.columns = short_col_labels
        y_labels_s7 = [f"{r['description']} ({r['category']})" for _, r in df_t35.iterrows()]
        target_ratio_figs7 = 417.0 / 468.0
        fig_s7, ax_s7 = plt.subplots(figsize=(NATURE_W, NATURE_W * target_ratio_figs7), dpi=SAVE_DPI)
        sns.heatmap(
            df_s7_mat,
            ax=ax_s7,
            cmap="viridis",
            annot=True,
            fmt=".4f",
            annot_kws={"size": 4.7},
            linewidths=0.30,
            cbar_kws={"label": "Mean Absolute Shapley (KernelSHAP) Value", "shrink": 0.72, "pad": 0.02},
        )
        ax_s7.set_yticks(np.arange(len(df_t35)) + 0.5)
        ax_s7.set_yticklabels(y_labels_s7, rotation=0, fontsize=5.4)
        ax_s7.set_xticklabels(short_col_labels, rotation=35, ha="right", fontsize=6.0, fontweight="bold")
        ax_s7.set_title(
            "Global Feature Importance Heatmap (Top 35 Features Across 12 Metrics, Lead Time 1)",
            fontsize=7.6, fontweight="bold", pad=8,
        )
        ax_s7.set_xlabel("Hydrological Evaluation Metric", fontsize=6.8, fontweight="bold")
        ax_s7.set_ylabel("Watershed Predictor Feature (Category)", fontsize=6.8, fontweight="bold")
        fig_s7.tight_layout()
        fig_s7.savefig(figs7_dst, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig_s7)
        pad_image_to_exact_aspect_ratio(figs7_dst, target_ratio_figs7, figs7_dst)
        print(f"   Saved {figs7_dst}", flush=True)

    print(f"   Saved {figs6_path} in {time.time() - t0:.1f}s total!", flush=True)


if __name__ == "__main__":
    main()
