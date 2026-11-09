#!/usr/bin/env python3
"""Evaluates M_ungauged (and confirmation M_gauged) across all 964,084 global watersheds
for every institutional data-sharing & licensing tier under three clean decompositions:
  1. Joint Data-Sharing & Integration (Factor 2 Coverage at Tier k + Factor 3 OOD at Tier k)
  2. River-Network Coverage Only (Factor 2 at Tier k, holding Factor 3 OOD fixed at Tier 1a [8,811] and at Tier 4 [121,819])
  3. Climate Representation Only (Factor 3 OOD at Tier k, holding Factor 2 Coverage fixed at Tier 1a [8,811] and at Tier 4 [121,819])

Computes global median & IQR predicted NSE, KGE, 5-yr, 10-yr, and 20-yr 2-day flood recall,
Total Warned Exposure, Total Unwarned Exposure, and Incremental People Warned (Net and RP-Clipped)
relative to Tier 1a (8,811), Tier 0 (0), Tier 4 (121,819), and adjacent waterfall tiers, stratified by:
  - 60 countries with operational warning capacity vs. 113 countries without (canonical cap_utils.py)
  - Harmonized 62 countries with operational WMO CAP systems vs. 111 without (Justin Ginnetti CAP paper #8)
  - Full 4-tier CAP classification (Tier 1 [41], Tier 2 [28], Tier 3 [19], Tier 4 [85])
  - World Bank income groups (High, Upper middle, Lower middle, Low income)
  - Continents (Asia, Africa, South America, North America, Europe, Oceania)
"""

import glob
import json
import os
import pickle
import sys
import time
import types
import warnings

os.environ.setdefault("GEO_BIAS_DATA_DIR", "./data")
os.environ.setdefault("GEO_BIAS_THIRD_PARTY_DIR", "./data")
os.environ.setdefault("GEO_BIAS_OUTPUT_DIR", "./data")

sys.path = [p for p in sys.path if not p.endswith("/local_pkg")]

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import torch
import zarr

torch.set_num_threads(56)

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

if "./data" not in sys.path:
  sys.path.insert(0, "./data")

from local_pkg.intelligence.gsnearing.geographical_bias_paper_v2.utils import cap_utils
from local_pkg.intelligence.gsnearing.geographical_bias_paper_v2.utils import file_paths
from local_pkg.intelligence.gsnearing.geographical_bias_paper_v2.utils import skill_predictor as sp

sys.modules["local_pkg.intelligence.gsnearing.geographical_bias_paper.skill_predictor"] = sp
sys.modules["local_pkg.intelligence.gsnearing.geographical_bias_paper.utils.skill_predictor"] = sp
sys.modules["local_pkg.intelligence.gsnearing.geographical_bias_paper_v2.skill_predictor"] = sp

warnings.filterwarnings("ignore")

CANON_V2_DIR = "./data"
OUT_DIR = "./data/science_results"
FEAT_DIR = f"{OUT_DIR}/features"
PRED_DIR = f"{OUT_DIR}/predictions"
os.makedirs(PRED_DIR, exist_ok=True)

RP_WEIGHTS = {5: 0.05, 10: 0.075, 20: 0.04, 50: 0.02, 100: 0.005}
RP_METRICS = [
    "5.0-year-2.0-window-recall",
    "10.0-year-2.0-window-recall",
    "20.0-year-2.0-window-recall",
]
CLASSIC_METRICS = ["NSE", "KGE"]
COV_COLS = [
    "total_upstream_coverage",
    "max_upstream_coverage",
    "max_downstream_coverage",
    "upstream_gauge_count",
]
OOD_COLS = [
    "anomaly_score_if",
    "anomaly_score_ocsvm",
    "anomaly_score_lof",
]

TIER_ORDER = [
    "tier0_zero_gauges",
    "tier1a_active_post2016_ccby_8811",
    "tier1b_strict_ccby_19491",
    "tier1b_full_caravan_24847",
    "tier2a_caravan_estreams_32713",
    "tier2_academic_compilations_38069",
    "tier2_all_open_academic_34630",
    "tier3_open_public_apis_and_bulk_106410",
    "tier4_all_publicly_knowable_121819",
    "tier5_universal_idealized_ceiling",
    "tier5_dense_corridor_ceiling",
]

TIER_LABELS = {
    "tier0_zero_gauges": "Tier 0: Zero-Gauge Baseline (0)",
    "tier1a_active_post2016_ccby_8811": "Tier 1a: Active Post-2016 CC-BY Training (8,811 / 8,104 phys)",
    "tier1b_strict_ccby_19491": "Tier 1b (Strict CC-BY): Regional Caravan CC-BY-4.0 (19,491 / 18,037 phys)",
    "tier1b_full_caravan_24847": "Tier 1b (Full Caravan): All Caravan Pre/Post-2016 (24,847 / 22,638 phys)",
    "tier2a_caravan_estreams_32713": "Tier 2a: Caravan + EStreams (32,713 / 27,876 phys)",
    "tier2_academic_compilations_38069": "Tier 2: Caravan + EStreams + GRDC (38,069 / 32,174 phys)",
    "tier2_all_open_academic_34630": "Tier 2+: All Open Academic Compilations (34,630 phys)",
    "tier3_open_public_apis_and_bulk_106410": "Tier 3: All Open Public APIs & Bulk Portals (106,410 phys)",
    "tier4_all_publicly_knowable_121819": "Tier 4: All Publicly Knowable Gauges (121,819 phys)",
    "tier5_universal_idealized_ceiling": "Tier 5: Universal 100% Coverage + Top-Quartile Climate",
    "tier5_dense_corridor_ceiling": "Tier 5 (Dense Corridor): 75th-Pct Multi-Gauge Corridor + Top-Quartile Climate",
}

# World Bank FY2024 Income Group Classification by ISO-A3
WB_INCOME_BY_ISO3 = {
    # High income
    "ARE": "High income", "AUS": "High income", "AUT": "High income", "BEL": "High income",
    "BGR": "High income", "BHS": "High income", "BRN": "High income", "CAN": "High income",
    "CHE": "High income", "CHL": "High income", "CYP": "High income", "CZE": "High income",
    "DEU": "High income", "DNK": "High income", "ESP": "High income", "EST": "High income",
    "FIN": "High income", "FRA": "High income", "GBR": "High income", "GRC": "High income",
    "GUY": "High income", "HRV": "High income", "HUN": "High income", "IRL": "High income",
    "ISL": "High income", "ISR": "High income", "ITA": "High income", "JPN": "High income",
    "KOR": "High income", "KWT": "High income", "LTU": "High income", "LUX": "High income",
    "LVA": "High income", "NCL": "High income", "NLD": "High income", "NOR": "High income",
    "NZL": "High income", "OMN": "High income", "PAN": "High income", "POL": "High income",
    "PRI": "High income", "PRT": "High income", "QAT": "High income", "ROU": "High income",
    "RUS": "High income", "SAU": "High income", "SVK": "High income", "SVN": "High income",
    "SWE": "High income", "TTO": "High income", "TWN": "High income", "URY": "High income",
    "USA": "High income",
    # Upper middle income
    "ALB": "Upper middle income", "ARG": "Upper middle income", "ARM": "Upper middle income",
    "AZE": "Upper middle income", "BIH": "Upper middle income", "BLR": "Upper middle income",
    "BLZ": "Upper middle income", "BRA": "Upper middle income", "BWA": "Upper middle income",
    "CHN": "Upper middle income", "COL": "Upper middle income", "CRI": "Upper middle income",
    "CUB": "Upper middle income", "DOM": "Upper middle income", "DZA": "Upper middle income",
    "ECU": "Upper middle income", "FJI": "Upper middle income", "GAB": "Upper middle income",
    "GEO": "Upper middle income", "GNQ": "Upper middle income", "GTM": "Upper middle income",
    "IDN": "Upper middle income", "IRN": "Upper middle income", "IRQ": "Upper middle income",
    "JAM": "Upper middle income", "KAZ": "Upper middle income", "LBY": "Upper middle income",
    "MDA": "Upper middle income", "MEX": "Upper middle income", "MKD": "Upper middle income",
    "MNE": "Upper middle income", "MNG": "Upper middle income", "MYS": "Upper middle income",
    "NAM": "Upper middle income", "PER": "Upper middle income", "PRY": "Upper middle income",
    "SLV": "Upper middle income", "SRB": "Upper middle income", "SUR": "Upper middle income",
    "THA": "Upper middle income", "TKM": "Upper middle income", "TUR": "Upper middle income",
    "UKR": "Upper middle income", "VEN": "Upper middle income", "XKX": "Upper middle income",
    "ZAF": "Upper middle income",
    # Lower middle income
    "AGO": "Lower middle income", "BEN": "Lower middle income", "BGD": "Lower middle income",
    "BOL": "Lower middle income", "BTN": "Lower middle income", "CIV": "Lower middle income",
    "CMR": "Lower middle income", "COG": "Lower middle income", "DJI": "Lower middle income",
    "EGY": "Lower middle income", "GHA": "Lower middle income", "GIN": "Lower middle income",
    "HND": "Lower middle income", "HTI": "Lower middle income", "IND": "Lower middle income",
    "JOR": "Lower middle income", "KEN": "Lower middle income", "KGZ": "Lower middle income",
    "KHM": "Lower middle income", "LAO": "Lower middle income", "LBN": "Lower middle income",
    "LKA": "Lower middle income", "LSO": "Lower middle income", "MAR": "Lower middle income",
    "MMR": "Lower middle income", "MRT": "Lower middle income", "NIC": "Lower middle income",
    "NGA": "Lower middle income", "NPL": "Lower middle income", "PAK": "Lower middle income",
    "PHL": "Lower middle income", "PNG": "Lower middle income", "PSE": "Lower middle income",
    "SEN": "Lower middle income", "SLB": "Lower middle income", "SWZ": "Lower middle income",
    "TJK": "Lower middle income", "TLS": "Lower middle income", "TUN": "Lower middle income",
    "TZA": "Lower middle income", "UZB": "Lower middle income", "VNM": "Lower middle income",
    "VUT": "Lower middle income", "ZMB": "Lower middle income", "ZWE": "Lower middle income",
    # Low income
    "AFG": "Low income", "BDI": "Low income", "BFA": "Low income", "CAF": "Low income",
    "COD": "Low income", "ERI": "Low income", "ETH": "Low income", "GMB": "Low income",
    "GNB": "Low income", "LBR": "Low income", "MDG": "Low income", "MLI": "Low income",
    "MOZ": "Low income", "MWI": "Low income", "NER": "Low income", "PRK": "Low income",
    "RWA": "Low income", "SDN": "Low income", "SLE": "Low income", "SOM": "Low income",
    "SSD": "Low income", "SYR": "Low income", "TCD": "Low income", "TGO": "Low income",
    "UGA": "Low income", "YEM": "Low income", "ESH": "Low income", "SOL": "Low income",
}

COUNTRY_TO_ISO3_ALIASES = {
    "United States": "USA", "Democratic Republic of the Congo": "COD",
    "Central African Republic": "CAF", "South Sudan": "SSD", "Czech Republic": "CZE",
    "Bosnia and Herzegovina": "BIH", "Dominican Republic": "DOM", "Ivory Coast": "CIV",
    "Equatorial Guinea": "GNQ", "Swaziland": "SWZ", "Solomon Islands": "SLB",
    "Western Sahara": "ESH", "North Macedonia": "MKD", "France": "FRA",
    "Norway": "NOR", "Kosovo": "XKX", "Somaliland": "SOL", "Northern Cyprus": "CYP",
    "East Timor": "TLS",
}


def build_country_metadata_map() -> dict:
  ne = gpd.read_file(f"{CANON_V2_DIR}/naturalearth_lowres.geojson")
  name_to_iso = dict(zip(ne["name"], ne["iso_a3"]))
  name_to_cont = dict(zip(ne["name"], ne["continent"]))
  name_to_iso.update(COUNTRY_TO_ISO3_ALIASES)
  name_to_cont.update({
      "United States": "North America",
      "Democratic Republic of the Congo": "Africa",
      "Central African Republic": "Africa",
      "South Sudan": "Africa",
      "Czech Republic": "Europe",
      "Bosnia and Herzegovina": "Europe",
      "Dominican Republic": "North America",
      "Ivory Coast": "Africa",
      "Equatorial Guinea": "Africa",
      "Swaziland": "Africa",
      "Solomon Islands": "Oceania",
      "Western Sahara": "Africa",
      "North Macedonia": "Europe",
      "France": "Europe",
      "Norway": "Europe",
      "Kosovo": "Europe",
      "Somaliland": "Africa",
      "Northern Cyprus": "Asia",
      "East Timor": "Asia",
  })
  return {"iso3": name_to_iso, "continent": name_to_cont}


def load_zarr_df(zarr_path: str) -> pd.DataFrame:
  root = zarr.open_group(store=zarr_path, mode="r", zarr_format=2)
  feats = root["features"][:]
  hybas_ids = [
      x.decode("utf-8") if isinstance(x, bytes) else str(x)
      for x in root["hybas_id"][:]
  ]
  hybas_ids = [x if x.startswith("hybas_") else f"hybas_{x}" for x in hybas_ids]
  cols = [
      x.decode("utf-8") if isinstance(x, bytes) else str(x)
      for x in root["columns"][:]
  ]
  return pd.DataFrame(feats, index=hybas_ids, columns=cols)


def fast_predict_rp_triplet(models_dict: dict, X_aligned: pd.DataFrame) -> dict[str, np.ndarray]:
  """Evaluates 5-yr, 10-yr, and 20-yr recall using 56-thread torch.cdist KNN + HGBR + ET."""
  out = {}
  for m_name in RP_METRICS:
    impl = models_dict[m_name].impl
    X_in = X_aligned.reindex(columns=impl.feature_columns, fill_value=0)
    est_hgbr, est_et, est_knn = impl.ensemble.estimators_

    X_knn = np.ascontiguousarray(est_knn[:-1].transform(X_in), dtype=np.float64)
    knn_reg = est_knn.named_steps["kneighborsregressor"]
    fit_X = np.ascontiguousarray(knn_reg._fit_X, dtype=np.float64)
    fit_y = np.asarray(knn_reg._y, dtype=np.float64).ravel()
    k_n = int(knn_reg.n_neighbors)
    n_q = len(X_knn)

    dists = np.empty((n_q, k_n), dtype=np.float64)
    nbr_idx = np.empty((n_q, k_n), dtype=np.int64)
    t_ref = torch.from_numpy(np.ascontiguousarray(fit_X, dtype=np.float32))
    with torch.no_grad():
      for s in range(0, n_q, 16384):
        e = min(n_q, s + 16384)
        t_q = torch.from_numpy(np.ascontiguousarray(X_knn[s:e], dtype=np.float32))
        d_mat = torch.cdist(t_q, t_ref)
        vals, idx = torch.topk(d_mat, k=k_n, dim=1, largest=False, sorted=True)
        dists[s:e] = vals.numpy().astype(np.float64)
        nbr_idx[s:e] = idx.numpy()

    zero_mask = dists <= 1e-15
    any_zero = zero_mask.any(axis=1)
    with np.errstate(divide="ignore"):
      nbr_w = np.where(any_zero[:, None], zero_mask.astype(np.float64), 1.0 / dists)
    nbr_w /= nbr_w.sum(axis=1, keepdims=True)
    p_knn = np.einsum("ij,ij->i", fit_y[nbr_idx], nbr_w)

    p_hgbr = est_hgbr._loss.link.inverse(
        est_hgbr._raw_predict(X_in, n_threads=56).ravel()
    )
    if hasattr(est_et, "named_steps") and "extratreesregressor" in est_et.named_steps:
      est_et.named_steps["extratreesregressor"].n_jobs = 56
    with joblib.parallel_backend("threading"):
      p_et = est_et.predict(X_in)

    preds = impl.ensemble.final_estimator_.predict(
        np.column_stack([p_hgbr, p_et, p_knn])
    )
    out[m_name] = np.clip(preds, 0.0, 1.0)
  return out


def fast_predict_classic_pair(models_dict: dict, X_aligned_classic: pd.DataFrame) -> dict[str, np.ndarray]:
  """Evaluates NSE and KGE on pre-engineered X_aligned_classic using 56 threads and inplace_predict."""
  X_np = np.ascontiguousarray(X_aligned_classic.values, dtype=np.float32)
  out = {}
  for m_name in CLASSIC_METRICS:
    impl = models_dict[m_name].impl
    base_preds = []
    for est in impl.model.estimators_:
      cls_name = est.__class__.__name__
      if cls_name == "HistGradientBoostingRegressor":
        base_preds.append(
            est._loss.link.inverse(est._raw_predict(X_np, n_threads=56).ravel())
        )
      elif cls_name == "XGBRegressor":
        booster = est.get_booster()
        booster.set_param({"nthread": 56})
        base_preds.append(booster.inplace_predict(X_np))
      elif cls_name == "LGBMRegressor":
        base_preds.append(est.booster_.predict(X_np, num_threads=56))
      else:
        base_preds.append(est.predict(X_aligned_classic))
    preds = impl.model.final_estimator_.predict(np.column_stack(base_preds))
    out[m_name] = np.clip(preds, -5.0, 1.0)
  return out


def main():
  t0 = time.time()
  file_paths.init("5.0-year-2.0-window-recall")

  print("1. Loading global basin exposure (964,084 study watersheds) and country/CAP/income metadata...", flush=True)
  df_basin_v2 = pd.read_parquet(
      f"{CANON_V2_DIR}/v2_exposure_blockers/statistics/basin_exposure_and_blockers_v2.parquet"
  )
  study_ids = [
      x if str(x).startswith("hybas_") else f"hybas_{x}"
      for x in df_basin_v2.index
  ]
  df_basin_v2.index = study_ids

  df_exp_raw = pd.read_csv(file_paths.GLOBAL_BASIN_EXPOSURE_DEFENDED_BY_RP_PATH)
  id_col = (
      "hybas_id"
      if "hybas_id" in df_exp_raw.columns
      else ("HYBAS_ID" if "HYBAS_ID" in df_exp_raw.columns else df_exp_raw.columns[0])
  )
  df_exp_raw[id_col] = (
      df_exp_raw[id_col]
      .astype(str)
      .str.replace(r"\.0$", "", regex=True)
      .apply(lambda x: x if x.startswith("hybas_") else f"hybas_{x}")
  )
  df_exp_raw = df_exp_raw.drop_duplicates(subset=[id_col]).set_index(id_col)

  def _get_exp_col(df, t):
    for c in [f"exposed_pop_{t}", f"exposure_{t}yr", str(t)]:
      if c in df.columns:
        return df[c].fillna(0.0)
    raise KeyError(f"Could not find exposure column for T={t}")

  df_exp = pd.DataFrame(index=study_ids)
  df_exp["E_5"] = RP_WEIGHTS[5] * _get_exp_col(df_exp_raw, 5).reindex(study_ids).fillna(0.0)
  df_exp["E_10"] = RP_WEIGHTS[10] * _get_exp_col(df_exp_raw, 10).reindex(study_ids).fillna(0.0)
  df_exp["E_20_50_100"] = (
      RP_WEIGHTS[20] * _get_exp_col(df_exp_raw, 20).reindex(study_ids).fillna(0.0)
      + RP_WEIGHTS[50] * _get_exp_col(df_exp_raw, 50).reindex(study_ids).fillna(0.0)
      + RP_WEIGHTS[100] * _get_exp_col(df_exp_raw, 100).reindex(study_ids).fillna(0.0)
  )
  df_exp["E_total"] = df_exp["E_5"] + df_exp["E_10"] + df_exp["E_20_50_100"]

  df_map = pd.read_csv(file_paths.HYBAS_MAPPING_PATH)
  if "Unnamed: 0" in df_map.columns and "hybas_id" not in df_map.columns:
    df_map = df_map.rename(columns={"Unnamed: 0": "hybas_id"})
  df_map["hybas_id"] = (
      df_map["hybas_id"]
      .astype(str)
      .str.replace(r"\.0$", "", regex=True)
      .apply(lambda x: x if x.startswith("hybas_") else f"hybas_{x}")
  )
  df_map["country_name_clean"] = df_map["country_name"].replace(file_paths.country_name_map)
  df_map_dedup = df_map.drop_duplicates("hybas_id").set_index("hybas_id")

  df_exp["country_name"] = df_map_dedup["country_name_clean"].reindex(study_ids)
  # Canonical 60 vs 113 warning capacity (Tier 1 [41] + Tier 3 [19] = 60)
  df_exp["has_system_60"] = (
      df_exp["country_name"]
      .map(cap_utils.has_effective_flood_warning_system)
      .fillna(False)
      .astype(bool)
  )
  df_exp["cap_tier"] = (
      df_exp["country_name"]
      .map(cap_utils.assign_cap_tier)
      .fillna("Tier 4: Complete Systemic Gap (No CAP)")
  )
  # Harmonized 62 countries with WMO CAP systems (matching Justin Ginnetti's CAP paper #8):
  # (A) WMO CAP Agency Lookup (Tier 1 [41] + 21 WMO CAP Tier 2 countries = 62 countries in 173-country dataset)
  t2_non_wmo_7 = {
      "Bosnia and Herzegovina", "Bosnia and Herz.", "Eswatini", "eSwatini", "Swaziland",
      "Guinea-Bissau", "Lesotho", "Montenegro", "North Macedonia", "Macedonia", "Trinidad and Tobago",
  }
  wmo_cap_lookup_countries = (
      set(cap_utils.TIER_1_COMPREHENSIVE_COUNTRIES)
      | (set(cap_utils.TIER_2_PROPORTIONALITY_COUNTRIES) - t2_non_wmo_7)
  )
  df_exp["has_wmo_cap_62"] = (
      df_exp["country_name"].isin(wmo_cap_lookup_countries).fillna(False).astype(bool)
  )
  # (B) Canonical 60 + Indonesia + Nigeria = 62
  df_exp["has_cap_60_idn_nga_62"] = (
      df_exp["has_system_60"] | df_exp["country_name"].isin({"Indonesia", "Nigeria"})
  ).astype(bool)

  c_meta = build_country_metadata_map()
  df_exp["iso3"] = df_exp["country_name"].map(c_meta["iso3"]).fillna("UNK")
  df_exp["income_group"] = df_exp["iso3"].map(WB_INCOME_BY_ISO3).fillna("Unclassified")

  # Continent mapping (from country + fallback to HydroBASINS continent for unmapped coastal basins)
  topo_df = pd.read_csv(
      f"{CANON_V2_DIR}/attributes/hybas_routing_topology.csv",
      usecols=["HYBAS_ID", "continent"],
      dtype=str,
  )
  topo_df["hybas_str"] = "hybas_" + topo_df["HYBAS_ID"].str.replace(r"\.0$", "", regex=True)
  reg_to_cont = {
      "NA": "North America", "AR": "North America", "GR": "North America",
      "SA": "South America", "EU": "Europe", "AU": "Oceania",
      "AF": "Africa", "SI": "Asia", "AS": "Asia",
  }
  hybas_cont_fallback = (
      topo_df.drop_duplicates("hybas_str")
      .set_index("hybas_str")["continent"]
      .str.upper()
      .map(reg_to_cont)
      .reindex(study_ids)
  )
  df_exp["continent"] = df_exp["country_name"].map(c_meta["continent"]).fillna(hybas_cont_fallback).fillna("Other")

  print(
      f"   Verified exposure total: {df_exp['E_total'].sum():,.2f} people/yr across {len(df_exp):,} watersheds | "
      f"60-sys countries: {df_exp.loc[df_exp['has_system_60'], 'country_name'].nunique()}, "
      f"62-WMO-CAP countries: {df_exp.loc[df_exp['has_wmo_cap_62'], 'country_name'].nunique()}",
      flush=True,
  )

  print("\n2. Loading base Zarr features and pre-computing invariant engineered features...", flush=True)
  df_base = load_zarr_df(f"{CANON_V2_DIR}/generic_training_data/hybas_generic_inference_data.zarr").loc[study_ids].copy()

  models_u, models_g = {}, {}
  for m in RP_METRICS + CLASSIC_METRICS:
    with open(os.path.join(file_paths.MODELS_DIR, f"final_ungauged_model_{m}.pkl"), "rb") as f:
      models_u[m] = pickle.load(f)
    with open(os.path.join(file_paths.MODELS_DIR, f"final_gauged_model_{m}.pkl"), "rb") as f:
      models_g[m] = pickle.load(f)

  rp_cols_u = list(models_u[RP_METRICS[0]].impl.feature_columns)
  classic_cols_u = list(models_u[CLASSIC_METRICS[0]].impl.feature_names)

  X_rp_base = models_u[RP_METRICS[0]].impl._preprocess_features(df_base).reindex(columns=rp_cols_u, fill_value=0)
  X_classic_base = models_u[CLASSIC_METRICS[0]].impl._engineer_features(df_base).reindex(columns=classic_cols_u, fill_value=0)

  # Load all per-tier feature parquets
  tier_feats = {}
  for t_key in TIER_ORDER:
    pq_path = f"{FEAT_DIR}/{t_key}.parquet"
    if not os.path.exists(pq_path):
      raise FileNotFoundError(f"Missing tier feature file: {pq_path}")
    tier_feats[t_key] = pd.read_parquet(pq_path).loc[study_ids]

  # Reference feature sets for holding Factor 2 or Factor 3 fixed
  cov_ref_t1a = tier_feats["tier1a_active_post2016_ccby_8811"][COV_COLS]
  ood_ref_t1a = tier_feats["tier1a_active_post2016_ccby_8811"][OOD_COLS]
  cov_ref_t4 = tier_feats["tier4_all_publicly_knowable_121819"][COV_COLS]
  ood_ref_t4 = tier_feats["tier4_all_publicly_knowable_121819"][OOD_COLS]

  # Pre-load cached Tier 4 Joint predictions directly from canonical_v2/saved_predictions/hybas_preds
  t4_joint_pq = f"{PRED_DIR}/tier4_all_publicly_knowable_121819__joint.parquet"
  if not os.path.exists(t4_joint_pq):
    t4_preload = {}
    for m_short, m_full in [
        ("r5", "5.0-year-2.0-window-recall"),
        ("r10", "10.0-year-2.0-window-recall"),
        ("r20", "20.0-year-2.0-window-recall"),
        ("NSE", "NSE"),
        ("KGE", "KGE"),
    ]:
      su = pd.read_csv(f"{CANON_V2_DIR}/saved_predictions/hybas_preds/global_hybas_predictions_ungauged_{m_full}.csv", index_col=0).iloc[:, 0]
      su.index = [x if str(x).startswith("hybas_") else f"hybas_{x}" for x in su.index.astype(str).str.replace(r"\.0$", "", regex=True)]
      t4_preload[f"{m_short}_u"] = su.reindex(study_ids).fillna(su.median()).values
      if m_short in ("r5", "r10", "r20"):
        t4_preload[f"{m_short}_u"] = np.clip(t4_preload[f"{m_short}_u"], 0.0, 1.0)
        sg = pd.read_csv(f"{CANON_V2_DIR}/saved_predictions/hybas_preds/global_hybas_predictions_gauged_{m_full}.csv", index_col=0).iloc[:, 0]
        sg.index = [x if str(x).startswith("hybas_") else f"hybas_{x}" for x in sg.index.astype(str).str.replace(r"\.0$", "", regex=True)]
        t4_preload[f"{m_short}_g"] = np.clip(sg.reindex(study_ids).fillna(sg.median()).values, 0.0, 1.0)
    pd.DataFrame(t4_preload, index=study_ids).to_parquet(t4_joint_pq)
    print("   Pre-loaded Tier 4 (121,819) Joint predictions from canonical_v2/saved_predictions/hybas_preds!", flush=True)

  # Pre-load cached Tier 2a (32,713) Joint predictions from backup_v2_32713_gauges/saved_predictions/hybas_preds
  t2a_joint_pq = f"{PRED_DIR}/tier2a_caravan_estreams_32713__joint.parquet"
  if not os.path.exists(t2a_joint_pq) and os.path.exists(f"{CANON_V2_DIR}/backup_v2_32713_gauges/saved_predictions/hybas_preds/global_hybas_predictions_ungauged_5.0-year-2.0-window-recall.csv"):
    t2a_preload = {}
    for m_short, m_full in [
        ("r5", "5.0-year-2.0-window-recall"),
        ("r10", "10.0-year-2.0-window-recall"),
        ("r20", "20.0-year-2.0-window-recall"),
        ("NSE", "NSE"),
        ("KGE", "KGE"),
    ]:
      su = pd.read_csv(f"{CANON_V2_DIR}/backup_v2_32713_gauges/saved_predictions/hybas_preds/global_hybas_predictions_ungauged_{m_full}.csv", index_col=0).iloc[:, 0]
      su.index = [x if str(x).startswith("hybas_") else f"hybas_{x}" for x in su.index.astype(str).str.replace(r"\.0$", "", regex=True)]
      t2a_preload[f"{m_short}_u"] = su.reindex(study_ids).fillna(su.median()).values
      if m_short in ("r5", "r10", "r20"):
        t2a_preload[f"{m_short}_u"] = np.clip(t2a_preload[f"{m_short}_u"], 0.0, 1.0)
    pd.DataFrame(t2a_preload, index=study_ids).to_parquet(t2a_joint_pq)
    print("   Pre-loaded Tier 2a (32,713) Joint predictions from backup_v2_32713_gauges/saved_predictions/hybas_preds!", flush=True)

  combo_mem_cache = {}

  def evaluate_feature_combo(
      combo_key: str,
      cov_df: pd.DataFrame,
      ood_df: pd.DataFrame,
      cov_tag: str = "",
      ood_tag: str = "",
      eval_classic: bool = True,
      eval_mgauged: bool = False,
      ref_preds: dict = None,
      ref_cov: pd.DataFrame = None,
      ref_ood: pd.DataFrame = None,
  ) -> dict[str, np.ndarray]:
    """Evaluates M_ungauged (and optionally M_gauged) on (cov_df, ood_df), using delta-row acceleration when possible."""
    pair_sig = (cov_tag, ood_tag) if (cov_tag and ood_tag) else None
    if pair_sig is not None and pair_sig in combo_mem_cache:
      cached_res = combo_mem_cache[pair_sig]
      if (not eval_classic or "NSE_u" in cached_res) and (not eval_mgauged or "r5_g" in cached_res):
        return cached_res

    cache_pq = f"{PRED_DIR}/{combo_key}.parquet"
    if os.path.exists(cache_pq):
      df_cached = pd.read_parquet(cache_pq)
      if (not eval_classic or "NSE_u" in df_cached.columns) and (not eval_mgauged or "r5_g" in df_cached.columns):
        res_c = {c: df_cached[c].values for c in df_cached.columns}
        if pair_sig is not None:
          combo_mem_cache[pair_sig] = res_c
        return res_c

    t_eval = time.time()
    X_rp = X_rp_base.copy()
    for c in COV_COLS:
      X_rp[c] = cov_df[c].values
    for c in OOD_COLS:
      X_rp[c] = ood_df[c].values

    # Check if only a subset of rows changed relative to ref
    changed_mask = None
    if ref_preds is not None and ref_cov is not None and ref_ood is not None:
      diff_cov = np.abs(cov_df[COV_COLS].values - ref_cov[COV_COLS].values).max(axis=1) > 1e-9
      diff_ood = np.abs(ood_df[OOD_COLS].values - ref_ood[OOD_COLS].values).max(axis=1) > 1e-9
      changed_mask = diff_cov | diff_ood

    res = {}
    if changed_mask is not None and changed_mask.sum() < 0.75 * len(X_rp):
      n_chg = int(changed_mask.sum())
      res["r5_u"] = ref_preds["r5_u"].copy()
      res["r10_u"] = ref_preds["r10_u"].copy()
      res["r20_u"] = ref_preds["r20_u"].copy()
      if n_chg > 0:
        sub_rp = fast_predict_rp_triplet(models_u, X_rp.iloc[changed_mask])
        res["r5_u"][changed_mask] = sub_rp["5.0-year-2.0-window-recall"]
        res["r10_u"][changed_mask] = sub_rp["10.0-year-2.0-window-recall"]
        res["r20_u"][changed_mask] = sub_rp["20.0-year-2.0-window-recall"]
    else:
      n_chg = len(X_rp)
      full_rp = fast_predict_rp_triplet(models_u, X_rp)
      res["r5_u"] = full_rp["5.0-year-2.0-window-recall"]
      res["r10_u"] = full_rp["10.0-year-2.0-window-recall"]
      res["r20_u"] = full_rp["20.0-year-2.0-window-recall"]

    if eval_classic:
      X_cl = X_classic_base.copy()
      for c in COV_COLS:
        X_cl[c] = cov_df[c].values
      for c in OOD_COLS:
        X_cl[c] = ood_df[c].values
      if changed_mask is not None and changed_mask.sum() < 0.75 * len(X_cl) and "NSE_u" in ref_preds:
        res["NSE_u"] = ref_preds["NSE_u"].copy()
        res["KGE_u"] = ref_preds["KGE_u"].copy()
        if changed_mask.sum() > 0:
          sub_cl = fast_predict_classic_pair(models_u, X_cl.iloc[changed_mask])
          res["NSE_u"][changed_mask] = sub_cl["NSE"]
          res["KGE_u"][changed_mask] = sub_cl["KGE"]
      else:
        full_cl = fast_predict_classic_pair(models_u, X_cl)
        res["NSE_u"] = full_cl["NSE"]
        res["KGE_u"] = full_cl["KGE"]

    if eval_mgauged:
      if changed_mask is not None and changed_mask.sum() < 0.75 * len(X_rp) and "r5_g" in ref_preds:
        res["r5_g"] = ref_preds["r5_g"].copy()
        res["r10_g"] = ref_preds["r10_g"].copy()
        res["r20_g"] = ref_preds["r20_g"].copy()
        if changed_mask.sum() > 0:
          sub_g = fast_predict_rp_triplet(models_g, X_rp.iloc[changed_mask])
          res["r5_g"][changed_mask] = sub_g["5.0-year-2.0-window-recall"]
          res["r10_g"][changed_mask] = sub_g["10.0-year-2.0-window-recall"]
          res["r20_g"][changed_mask] = sub_g["20.0-year-2.0-window-recall"]
      else:
        full_g = fast_predict_rp_triplet(models_g, X_rp)
        res["r5_g"] = full_g["5.0-year-2.0-window-recall"]
        res["r10_g"] = full_g["10.0-year-2.0-window-recall"]
        res["r20_g"] = full_g["20.0-year-2.0-window-recall"]

    pd.DataFrame(res, index=study_ids).to_parquet(cache_pq)
    if pair_sig is not None:
      combo_mem_cache[pair_sig] = res
    print(
        f"   [{combo_key}] evaluated ({n_chg:,} active rows) in {time.time() - t_eval:.1f}s | "
        f"med_r5_u={np.median(res['r5_u'])*100:.2f}%",
        flush=True,
    )
    return res

  # =========================================================================
  # Evaluate Baseline Reference States First (Tier 1a Joint and Tier 4 Joint)
  # =========================================================================
  print("\n3. Evaluating Anchor Baselines (Tier 1a [8,811] Joint and Tier 4 [121,819] Joint)...", flush=True)
  preds_t4_joint = evaluate_feature_combo(
      "tier4_all_publicly_knowable_121819__joint",
      cov_ref_t4,
      ood_ref_t4,
      cov_tag="tier4_all_publicly_knowable_121819",
      ood_tag="tier4_all_publicly_knowable_121819",
      eval_classic=True,
      eval_mgauged=True,
  )
  preds_t1a_joint = evaluate_feature_combo(
      "tier1a_active_post2016_ccby_8811__joint",
      cov_ref_t1a,
      ood_ref_t1a,
      cov_tag="tier1a_active_post2016_ccby_8811",
      ood_tag="tier1a_active_post2016_ccby_8811",
      eval_classic=True,
      eval_mgauged=True,
      ref_preds=preds_t4_joint,
      ref_cov=cov_ref_t4,
      ref_ood=ood_ref_t4,
  )

  # =========================================================================
  # Evaluate All Tiers Across All Decompositions
  # =========================================================================
  print("\n4. Evaluating All Tiers Across Decompositions (Joint, Factor 2 Only, Factor 3 Only)...", flush=True)
  all_evals = {}

  for t_key in TIER_ORDER:
    cov_k = tier_feats[t_key][COV_COLS]
    ood_k = tier_feats[t_key][OOD_COLS]
    c_tag = t_key
    o_tag = "tier1a_active_post2016_ccby_8811" if t_key == "tier0_zero_gauges" else (
        "tier5_universal_idealized_ceiling" if t_key == "tier5_dense_corridor_ceiling" else t_key
    )

    # 1. Joint (Factor 2 at Tier k + Factor 3 at Tier k)
    all_evals[(t_key, "joint")] = evaluate_feature_combo(
        f"{t_key}__joint",
        cov_k,
        ood_k,
        cov_tag=c_tag,
        ood_tag=o_tag,
        eval_classic=True,
        eval_mgauged=(t_key in ("tier1a_active_post2016_ccby_8811", "tier4_all_publicly_knowable_121819")),
        ref_preds=preds_t1a_joint,
        ref_cov=cov_ref_t1a,
        ref_ood=ood_ref_t1a,
    )

    # 2. Factor 2 Only (Coverage at Tier k, OOD fixed at Tier 1a [8,811])
    all_evals[(t_key, "f2_cov_only_base1a")] = evaluate_feature_combo(
        f"{t_key}__f2_cov_only_base1a",
        cov_k,
        ood_ref_t1a,
        cov_tag=c_tag,
        ood_tag="tier1a_active_post2016_ccby_8811",
        eval_classic=True,
        eval_mgauged=False,
        ref_preds=preds_t1a_joint,
        ref_cov=cov_ref_t1a,
        ref_ood=ood_ref_t1a,
    )

    # 3. Factor 3 Only (OOD at Tier k, Coverage fixed at Tier 1a [8,811])
    all_evals[(t_key, "f3_ood_only_base1a")] = evaluate_feature_combo(
        f"{t_key}__f3_ood_only_base1a",
        cov_ref_t1a,
        ood_k,
        cov_tag="tier1a_active_post2016_ccby_8811",
        ood_tag=o_tag,
        eval_classic=True,
        eval_mgauged=False,
        ref_preds=all_evals[(t_key, "joint")],
        ref_cov=cov_k,
        ref_ood=ood_k,
    )

    # 4. Factor 2 Only relative to Tier 4 (Coverage at Tier k, OOD fixed at Tier 4 [121,819])
    all_evals[(t_key, "f2_cov_only_base4")] = evaluate_feature_combo(
        f"{t_key}__f2_cov_only_base4",
        cov_k,
        ood_ref_t4,
        cov_tag=c_tag,
        ood_tag="tier4_all_publicly_knowable_121819",
        eval_classic=True,
        eval_mgauged=False,
        ref_preds=preds_t4_joint,
        ref_cov=cov_ref_t4,
        ref_ood=ood_ref_t4,
    )

    # 5. Factor 3 Only relative to Tier 4 (OOD at Tier k, Coverage fixed at Tier 4 [121,819])
    all_evals[(t_key, "f3_ood_only_base4")] = evaluate_feature_combo(
        f"{t_key}__f3_ood_only_base4",
        cov_ref_t4,
        ood_k,
        cov_tag="tier4_all_publicly_knowable_121819",
        ood_tag=o_tag,
        eval_classic=True,
        eval_mgauged=False,
        ref_preds=all_evals[(t_key, "joint")],
        ref_cov=cov_k,
        ref_ood=ood_k,
    )

  # =========================================================================
  # Compute Comprehensive Summary Tables & Stratified Breakdowns
  # =========================================================================
  print("\n5. Computing global skill summaries, warned/unwarned exposure, and stratified breakdowns...", flush=True)
  e5 = df_exp["E_5"].values
  e10 = df_exp["E_10"].values
  e20 = df_exp["E_20_50_100"].values
  etot = df_exp["E_total"].values

  def calc_warned_vec(p_dict):
    return e5 * p_dict["r5_u"] + e10 * p_dict["r10_u"] + e20 * p_dict["r20_u"]

  def calc_rp_clipped_delta_vec(p_new, p_base):
    return (
        e5 * np.maximum(p_new["r5_u"] - p_base["r5_u"], 0.0)
        + e10 * np.maximum(p_new["r10_u"] - p_base["r10_u"], 0.0)
        + e20 * np.maximum(p_new["r20_u"] - p_base["r20_u"], 0.0)
    )

  p_t0_joint = all_evals[("tier0_zero_gauges", "joint")]
  p_t1a_joint = all_evals[("tier1a_active_post2016_ccby_8811", "joint")]
  p_t4_joint = all_evals[("tier4_all_publicly_knowable_121819", "joint")]

  w_t0 = calc_warned_vec(p_t0_joint)
  w_t1a = calc_warned_vec(p_t1a_joint)
  w_t4 = calc_warned_vec(p_t4_joint)

  mask_sys60 = df_exp["has_system_60"].values
  mask_nosys113 = (~df_exp["has_system_60"].values) & df_exp["country_name"].notna().values
  mask_cap62 = df_exp["has_wmo_cap_62"].values
  mask_nocap111 = (~df_exp["has_wmo_cap_62"].values) & df_exp["country_name"].notna().values
  mask_cap62_alt = df_exp["has_cap_60_idn_nga_62"].values
  mask_nocap111_alt = (~df_exp["has_cap_60_idn_nga_62"].values) & df_exp["country_name"].notna().values

  summary_rows = []
  stratified_rows = []

  for t_key in TIER_ORDER:
    for decomp in [
        "joint",
        "f2_cov_only_base1a",
        "f3_ood_only_base1a",
        "f2_cov_only_base4",
        "f3_ood_only_base4",
    ]:
      p = all_evals[(t_key, decomp)]
      w_vec = calc_warned_vec(p)
      unw_vec = etot - w_vec

      # Net deltas relative to Tier 1a, Tier 0, Tier 4
      d_net_vs_t1a = w_vec - w_t1a
      d_net_vs_t0 = w_vec - w_t0
      d_net_vs_t4 = w_vec - w_t4

      # RP-clipped deltas relative to Tier 1a, Tier 0, Tier 4
      d_rp_vs_t1a = calc_rp_clipped_delta_vec(p, p_t1a_joint)
      d_rp_vs_t0 = calc_rp_clipped_delta_vec(p, p_t0_joint)
      d_rp_vs_t4 = calc_rp_clipped_delta_vec(p, p_t4_joint)

      def _qstats(arr):
        q25, q50, q75 = np.percentile(arr, [25, 50, 75])
        return float(q50), float(q25), float(q75), float(q75 - q25), float(np.mean(arr))

      r5_med, r5_q25, r5_q75, r5_iqr, r5_mean = _qstats(p["r5_u"] * 100.0)
      r10_med, r10_q25, r10_q75, r10_iqr, r10_mean = _qstats(p["r10_u"] * 100.0)
      r20_med, r20_q25, r20_q75, r20_iqr, r20_mean = _qstats(p["r20_u"] * 100.0)
      nse_med, nse_q25, nse_q75, nse_iqr, nse_mean = _qstats(p["NSE_u"])
      kge_med, kge_q25, kge_q75, kge_iqr, kge_mean = _qstats(p["KGE_u"])

      row = {
          "tier_key": t_key,
          "tier_label": TIER_LABELS[t_key],
          "decomposition": decomp,
          # Skill metrics across all 964,084 global watersheds
          "r5_median_pct": r5_med, "r5_q25_pct": r5_q25, "r5_q75_pct": r5_q75, "r5_iqr_pct": r5_iqr, "r5_mean_pct": r5_mean,
          "r10_median_pct": r10_med, "r10_q25_pct": r10_q25, "r10_q75_pct": r10_q75, "r10_iqr_pct": r10_iqr, "r10_mean_pct": r10_mean,
          "r20_median_pct": r20_med, "r20_q25_pct": r20_q25, "r20_q75_pct": r20_q75, "r20_iqr_pct": r20_iqr, "r20_mean_pct": r20_mean,
          "NSE_median": nse_med, "NSE_q25": nse_q25, "NSE_q75": nse_q75, "NSE_iqr": nse_iqr, "NSE_mean": nse_mean,
          "KGE_median": kge_med, "KGE_q25": kge_q25, "KGE_q75": kge_q75, "KGE_iqr": kge_iqr, "KGE_mean": kge_mean,
          # Global Exposure (people/yr)
          "E_total_global": float(etot.sum()),
          "warned_global": float(w_vec.sum()),
          "unwarned_global": float(unw_vec.sum()),
          "pct_warned_global": float(100.0 * w_vec.sum() / etot.sum()),
          "pct_unwarned_global": float(100.0 * unw_vec.sum() / etot.sum()),
          # Incremental People Warned vs Tier 1a (8,811 baseline)
          "delta_net_vs_t1a_global": float(d_net_vs_t1a.sum()),
          "delta_rp_clipped_vs_t1a_global": float(d_rp_vs_t1a.sum()),
          "pct_unwarned_closed_net_vs_t1a": float(100.0 * d_net_vs_t1a.sum() / (etot.sum() - w_t1a.sum())),
          "pct_unwarned_closed_rp_vs_t1a": float(100.0 * d_rp_vs_t1a.sum() / (etot.sum() - w_t1a.sum())),
          # Incremental People Warned vs Tier 0 (0-gauge baseline)
          "delta_net_vs_t0_global": float(d_net_vs_t0.sum()),
          "delta_rp_clipped_vs_t0_global": float(d_rp_vs_t0.sum()),
          # Incremental People Warned vs Tier 4 (121,819 canonical v2 baseline)
          "delta_net_vs_t4_global": float(d_net_vs_t4.sum()),
          "delta_rp_clipped_vs_t4_global": float(d_rp_vs_t4.sum()),
          # Stratified: 60 With Warning System vs 113 Without (Canonical cap_utils.py)
          "warned_sys60": float(w_vec[mask_sys60].sum()),
          "unwarned_sys60": float(unw_vec[mask_sys60].sum()),
          "delta_net_vs_t1a_sys60": float(d_net_vs_t1a[mask_sys60].sum()),
          "delta_rp_vs_t1a_sys60": float(d_rp_vs_t1a[mask_sys60].sum()),
          "delta_net_vs_t4_sys60": float(d_net_vs_t4[mask_sys60].sum()),
          "delta_rp_vs_t4_sys60": float(d_rp_vs_t4[mask_sys60].sum()),
          "warned_nosys113": float(w_vec[mask_nosys113].sum()),
          "unwarned_nosys113": float(unw_vec[mask_nosys113].sum()),
          "delta_net_vs_t1a_nosys113": float(d_net_vs_t1a[mask_nosys113].sum()),
          "delta_rp_vs_t1a_nosys113": float(d_rp_vs_t1a[mask_nosys113].sum()),
          "delta_net_vs_t4_nosys113": float(d_net_vs_t4[mask_nosys113].sum()),
          "delta_rp_vs_t4_nosys113": float(d_rp_vs_t4[mask_nosys113].sum()),
          # Stratified: 62 With WMO CAP System vs 111 Without (Justin Ginnetti CAP Paper #8)
          "warned_cap62": float(w_vec[mask_cap62].sum()),
          "unwarned_cap62": float(unw_vec[mask_cap62].sum()),
          "delta_net_vs_t1a_cap62": float(d_net_vs_t1a[mask_cap62].sum()),
          "delta_rp_vs_t1a_cap62": float(d_rp_vs_t1a[mask_cap62].sum()),
          "delta_net_vs_t4_cap62": float(d_net_vs_t4[mask_cap62].sum()),
          "delta_rp_vs_t4_cap62": float(d_rp_vs_t4[mask_cap62].sum()),
          "warned_nocap111": float(w_vec[mask_nocap111].sum()),
          "unwarned_nocap111": float(unw_vec[mask_nocap111].sum()),
          "delta_net_vs_t1a_nocap111": float(d_net_vs_t1a[mask_nocap111].sum()),
          "delta_rp_vs_t1a_nocap111": float(d_rp_vs_t1a[mask_nocap111].sum()),
          "delta_net_vs_t4_nocap111": float(d_net_vs_t4[mask_nocap111].sum()),
          "delta_rp_vs_t4_nocap111": float(d_rp_vs_t4[mask_nocap111].sum()),
          # Alternative 62 (60 + Indonesia + Nigeria)
          "warned_cap62_alt": float(w_vec[mask_cap62_alt].sum()),
          "unwarned_cap62_alt": float(unw_vec[mask_cap62_alt].sum()),
          "delta_net_vs_t1a_cap62_alt": float(d_net_vs_t1a[mask_cap62_alt].sum()),
          "delta_rp_vs_t1a_cap62_alt": float(d_rp_vs_t1a[mask_cap62_alt].sum()),
          "warned_nocap111_alt": float(w_vec[mask_nocap111_alt].sum()),
          "unwarned_nocap111_alt": float(unw_vec[mask_nocap111_alt].sum()),
          "delta_net_vs_t1a_nocap111_alt": float(d_net_vs_t1a[mask_nocap111_alt].sum()),
          "delta_rp_vs_t1a_nocap111_alt": float(d_rp_vs_t1a[mask_nocap111_alt].sum()),
      }

      if "r5_g" in p:
        w_g_vec = e5 * p["r5_g"] + e10 * p["r10_g"] + e20 * p["r20_g"]
        d_mg_rp = (
            e5 * np.maximum(p["r5_g"] - p["r5_u"], 0.0)
            + e10 * np.maximum(p["r10_g"] - p["r10_u"], 0.0)
            + e20 * np.maximum(p["r20_g"] - p["r20_u"], 0.0)
        )
        row["mgauged_r5_median_pct"] = float(np.median(p["r5_g"]) * 100.0)
        row["mgauged_warned_global"] = float(w_g_vec.sum())
        row["mgauged_confirm_delta_rp_global"] = float(d_mg_rp.sum())
        row["mgauged_confirm_delta_rp_sys60"] = float(d_mg_rp[mask_sys60].sum())
        row["mgauged_confirm_delta_rp_nosys113"] = float(d_mg_rp[mask_nosys113].sum())

      summary_rows.append(row)

      # Stratified breakdowns for Joint, F2_base1a, F3_base1a, F2_base4, F3_base4
      for strat_col in ["cap_tier", "income_group", "continent"]:
        for grp_val, grp_idx in df_exp.groupby(strat_col).groups.items():
          pos = df_exp.index.get_indexer(grp_idx)
          if len(pos) == 0:
            continue
          stratified_rows.append({
              "tier_key": t_key,
              "tier_label": TIER_LABELS[t_key],
              "decomposition": decomp,
              "stratum_type": strat_col,
              "stratum_value": grp_val,
              "n_basins": int(len(pos)),
              "E_total": float(etot[pos].sum()),
              "warned": float(w_vec[pos].sum()),
              "unwarned": float(unw_vec[pos].sum()),
              "pct_warned": float(100.0 * w_vec[pos].sum() / etot[pos].sum()) if etot[pos].sum() > 0 else 0.0,
              "r5_median_pct": float(np.median(p["r5_u"][pos]) * 100.0),
              "NSE_median": float(np.median(p["NSE_u"][pos])),
              "KGE_median": float(np.median(p["KGE_u"][pos])),
              "delta_net_vs_t1a": float(d_net_vs_t1a[pos].sum()),
              "delta_rp_vs_t1a": float(d_rp_vs_t1a[pos].sum()),
              "delta_net_vs_t0": float(d_net_vs_t0[pos].sum()),
              "delta_rp_vs_t0": float(d_rp_vs_t0[pos].sum()),
              "delta_net_vs_t4": float(d_net_vs_t4[pos].sum()),
              "delta_rp_vs_t4": float(d_rp_vs_t4[pos].sum()),
          })

  df_sum = pd.DataFrame(summary_rows)
  df_sum.to_csv(f"{OUT_DIR}/tiered_skill_and_exposure_summary.csv", index=False)

  df_strat = pd.DataFrame(stratified_rows)
  df_strat.to_csv(f"{OUT_DIR}/tiered_stratified_breakdowns.csv", index=False)

  # =========================================================================
  # Step-by-Step Waterfall Attribution Table (Sequential Step Deltas)
  # =========================================================================
  waterfall_steps = [
      ("tier0_zero_gauges", "tier1a_active_post2016_ccby_8811", "Step 0->1a: Existing Post-2016 CC-BY ML Training Network (0 -> 8,104 phys / 8,811 records)"),
      ("tier1a_active_post2016_ccby_8811", "tier1b_strict_ccby_19491", "Step 1a->1b(CC-BY): Modern Weather-Archive Overlap Bottleneck — Strict CC-BY Caravan (8,104 -> 18,037 phys)"),
      ("tier1b_strict_ccby_19491", "tier1b_full_caravan_24847", "Step 1b(CC-BY)->1b(Full): Pre-2016 Caravan GRDC Sub-Dataset (18,037 -> 22,638 phys)"),
      ("tier1a_active_post2016_ccby_8811", "tier1b_full_caravan_24847", "Step 1a->1b(Full Combined): Total Modern Weather-Archive Overlap Bottleneck (8,104 -> 22,638 phys / 24,847 Caravan)"),
      ("tier1b_full_caravan_24847", "tier2_academic_compilations_38069", "Step 1b->2(38k): Academic License & Compilation Fragmentation — EStreams + GRDC (22,638 -> 32,174 phys / 38,069 records)"),
      ("tier1b_full_caravan_24847", "tier2_all_open_academic_34630", "Step 1b->2+(34.6k): Full Open Academic Compilations (22,638 -> 34,630 phys)"),
      ("tier2_academic_compilations_38069", "tier3_open_public_apis_and_bulk_106410", "Step 2(38k)->3: National API & Portal Fragmentation — RivRetrieve/GSHA/ROBIN/CDR2/CIS/JPDH (32,174 -> 106,410 phys)"),
      ("tier2_all_open_academic_34630", "tier3_open_public_apis_and_bulk_106410", "Step 2+(34.6k)->3: National API & Portal Fragmentation above All Academic (34,630 -> 106,410 phys)"),
      ("tier3_open_public_apis_and_bulk_106410", "tier4_all_publicly_knowable_121819", "Step 3->4: Institutional Withholding & Summary-Only Catalogs — GSIM/SWOT/ADHI + Bilateral (106,410 -> 121,819 phys)"),
      ("tier1a_active_post2016_ccby_8811", "tier4_all_publicly_knowable_121819", "TOTAL EXISTING GAUGE BOTTLENECK (Step 1a->4): All Licensing, Archive Overlap, Portal Fragmentation & Withholding (8,104 -> 121,819 phys)"),
      ("tier4_all_publicly_knowable_121819", "tier5_universal_idealized_ceiling", "Step 4->5: True Physical Gauge Scarcity on the Ground (121,819 phys -> Universal 100% Coverage & Top-Quartile Climate)"),
      ("tier4_all_publicly_knowable_121819", "tier5_dense_corridor_ceiling", "Step 4->5(Dense): True Physical Gauge Scarcity to Dense Multi-Gauge Corridor Ceiling"),
      ("tier1a_active_post2016_ccby_8811", "tier5_universal_idealized_ceiling", "TOTAL OBSERVATIONAL DATA CEILING (Step 1a->5): Existing Gauge Sharing + New Physical Gauges (8,104 -> Universal 100%)"),
  ]

  wf_rows = []
  for t_from, t_to, step_label in waterfall_steps:
    for decomp in ["joint", "f2_cov_only_base1a", "f3_ood_only_base1a", "f2_cov_only_base4", "f3_ood_only_base4"]:
      p_from = all_evals[(t_from, decomp)]
      p_to = all_evals[(t_to, decomp)]
      w_from = calc_warned_vec(p_from)
      w_to = calc_warned_vec(p_to)
      d_net = w_to - w_from
      d_rp = calc_rp_clipped_delta_vec(p_to, p_from)
      wf_rows.append({
          "step_label": step_label,
          "tier_from": t_from,
          "tier_to": t_to,
          "decomposition": decomp,
          "delta_r5_median_pp": float((np.median(p_to["r5_u"]) - np.median(p_from["r5_u"])) * 100.0),
          "delta_NSE_median": float(np.median(p_to["NSE_u"]) - np.median(p_from["NSE_u"])),
          "delta_KGE_median": float(np.median(p_to["KGE_u"]) - np.median(p_from["KGE_u"])),
          "delta_net_global": float(d_net.sum()),
          "delta_rp_clipped_global": float(d_rp.sum()),
          "delta_net_sys60": float(d_net[mask_sys60].sum()),
          "delta_rp_sys60": float(d_rp[mask_sys60].sum()),
          "delta_net_nosys113": float(d_net[mask_nosys113].sum()),
          "delta_rp_nosys113": float(d_rp[mask_nosys113].sum()),
          "delta_net_cap62": float(d_net[mask_cap62].sum()),
          "delta_rp_cap62": float(d_rp[mask_cap62].sum()),
          "delta_net_nocap111": float(d_net[mask_nocap111].sum()),
          "delta_rp_nocap111": float(d_rp[mask_nocap111].sum()),
      })

  df_wf = pd.DataFrame(wf_rows)
  df_wf.to_csv(f"{OUT_DIR}/policy_attribution_waterfall.csv", index=False)

  print(f"\nALL SKILL & EXPOSURE COUNTERFACTUALS COMPLETED IN {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
  main()
