#!/usr/bin/env python3
"""Evaluates Lead-Time-1 SkillPredictor meta-models across all 964,084 global watersheds
and all institutional data-sharing & licensing tiers (Tier 0 through Tier 5) across:
  1. Global Training Only (M_base, '000'):
     - Joint (Factor 2 Coverage at Tier k + Factor 3 OOD at Tier k)
     - Factor 2 Coverage Only (holding Factor 3 OOD fixed at Tier 1a [8,811] and Tier 4 [121,819])
     - Factor 3 Climate OOD Only (holding Factor 2 Coverage fixed at Tier 1a [8,811] and Tier 4 [121,819])
  2. 30-Day Variational Data Assimilation Only (M_DA, '001'):
     - Holding global training fixed at Tier 1a (8,811), applying M_DA at gauged sub-basins in Tier k
  3. Global Training + 30-Day Variational Data Assimilation ('101'):
     - Updating global training to Tier k across all 964,084 sub-basins + applying M_DA at gauged sub-basins in Tier k
  4. Target Fine-Tuning Only (M_FT, '010', when trained):
     - Holding global training fixed at Tier 1a (8,811), applying M_FT at gauged sub-basins in Tier k
  5. All Three Combined: Global Training + Target Fine-Tuning + 30-Day DA (M_FT_DA, '011', when trained):
     - Updating global training to Tier k across all 964,084 sub-basins + applying M_FT_DA at gauged sub-basins in Tier k
"""

import glob
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

if "./data" not in sys.path:
  sys.path.insert(0, "./data")

import local_pkg
gfile_mod = types.ModuleType("local_pkg.pyglib.gfile")
gfile_mod.Open = open
gfile_mod.Exists = os.path.exists
gfile_mod.MakeDirs = lambda p: os.makedirs(p, exist_ok=True)
gfile_mod.Glob = glob.glob
gfile_mod.ListDir = os.listdir
pyglib_mod = types.ModuleType("local_pkg.pyglib")
pyglib_mod.gfile = gfile_mod
local_pkg.pyglib = pyglib_mod
sys.modules["local_pkg.pyglib"] = pyglib_mod
sys.modules["local_pkg.pyglib.gfile"] = gfile_mod

from local_pkg.intelligence.gsnearing.geographical_bias_paper.utils import cap_utils
from local_pkg.intelligence.gsnearing.geographical_bias_paper.utils import file_paths
from local_pkg.intelligence.gsnearing.geographical_bias_paper.utils import io_utils
from local_pkg.intelligence.gsnearing.geographical_bias_paper.utils import skill_predictor as sp

sys.modules["local_pkg.intelligence.gsnearing.geographical_bias_paper.skill_predictor"] = sp
sys.modules["local_pkg.intelligence.gsnearing.geographical_bias_paper.utils.skill_predictor"] = sp
sys.modules["local_pkg.intelligence.gsnearing.geographical_bias_paper_v2.skill_predictor"] = sp
sys.modules["local_pkg.intelligence.gsnearing.geographical_bias_paper_v2.utils.skill_predictor"] = sp

warnings.filterwarnings("ignore")

CANON_V2_DIR = "./data"
FEAT_DIR = "./data/coverage_fractions"
OUT_DIR = "./data/science_results_lead1"
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
    "tier1a_active_post2016_ccby_8811": "Tier 1a: Active Post-2016 CC-BY Training (8,811 / 8,075 phys)",
    "tier1b_strict_ccby_19491": "Tier 1b (Strict CC-BY): Regional Caravan CC-BY-4.0 (19,491 / 18,037 phys)",
    "tier1b_full_caravan_24847": "Tier 1b (Full Caravan): All Caravan Pre/Post-2016 (24,847 / 21,231 phys)",
    "tier2a_caravan_estreams_32713": "Tier 2a: Caravan + EStreams (32,713 / 27,876 phys)",
    "tier2_academic_compilations_38069": "Tier 2: Caravan + EStreams + GRDC (38,069 / 32,174 phys)",
    "tier2_all_open_academic_34630": "Tier 2+: All Open Academic Compilations (40,053 phys)",
    "tier3_open_public_apis_and_bulk_106410": "Tier 3: All Open Public APIs & Bulk Portals (106,410 phys)",
    "tier4_all_publicly_knowable_121819": "Tier 4: All Publicly Knowable Gauges (121,819 phys)",
    "tier5_universal_idealized_ceiling": "Tier 5: Universal 100% Coverage + Top-Quartile Climate",
    "tier5_dense_corridor_ceiling": "Tier 5 (Dense Corridor): 75th-Pct Multi-Gauge Corridor + Top-Quartile Climate",
}

WB_INCOME_BY_ISO3 = {
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


def load_model_family(exp_suffix: str) -> tuple[dict, dict] | None:
  """Loads ungauged and gauged SkillPredictor dicts for a given exp_suffix ('', '_da', '_ft', '_ft_da')."""
  u_tag = f"ungauged{exp_suffix}"
  g_tag = f"gauged{exp_suffix}"
  models_u, models_g = {}, {}
  for m in RP_METRICS + CLASSIC_METRICS:
    u_path = os.path.join(file_paths.MODELS_DIR, f"final_{u_tag}_model_{m}.pkl")
    g_path = os.path.join(file_paths.MODELS_DIR, f"final_{g_tag}_model_{m}.pkl")
    if not (os.path.exists(u_path) and os.path.exists(g_path)):
      return None
    with open(u_path, "rb") as f:
      models_u[m] = pickle.load(f)
    with open(g_path, "rb") as f:
      models_g[m] = pickle.load(f)
  return models_u, models_g


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
  df_exp["has_cap_60_idn_nga_62"] = (
      df_exp["has_system_60"] | df_exp["country_name"].isin({"Indonesia", "Nigeria"})
  ).astype(bool)

  c_meta = build_country_metadata_map()
  df_exp["iso3"] = df_exp["country_name"].map(c_meta["iso3"]).fillna("UNK")
  df_exp["income_group"] = df_exp["iso3"].map(WB_INCOME_BY_ISO3).fillna("Unclassified")

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
      f"   Verified exposure total: {df_exp['E_total'].sum():,.2f} people/yr across {len(df_exp):,} watersheds.",
      flush=True,
  )

  print("\n2. Loading gauge deduplication catalog to build per-tier gauged sub-basin masks...", flush=True)
  dedup = pd.read_parquet(f"{CANON_V2_DIR}/attributes/global_gauges_deduplicated.parquet")
  arch = pd.read_parquet(f"{CANON_V2_DIR}/attributes/global_gauges_by_source_archive.parquet")
  grdc_8811 = pd.read_csv(f"{CANON_V2_DIR}/attributes/grdc_attributes.csv", index_col=0)
  ca_38k = pd.read_csv(f"{CANON_V2_DIR}/backup_v2_38069_gauges/attributes/caravan_attributes.csv", index_col=0)

  native_to_cid = dict(zip(arch["native_station_id"].astype(str), arch["canonical_gauge_id"].astype(str)))
  grdc_arch = arch[arch["source_dataset"] == "GRDC"]
  for nid, cid in zip(grdc_arch["native_station_id"].astype(str), grdc_arch["canonical_gauge_id"].astype(str)):
    native_to_cid[f"GRDC_{nid}"] = cid

  t1a_ids = ["CARAVAN_" + str(x).upper() for x in grdc_8811.index]
  t1a_cids = set(native_to_cid[x] for x in t1a_ids if x in native_to_cid)
  t1b_ccby_ids = [x for x in ca_38k.index if str(x).startswith("CARAVAN_") and not str(x).startswith("CARAVAN_GRDC_")]
  t1b_ccby_cids = set(native_to_cid[x] for x in t1b_ccby_ids if x in native_to_cid)
  t1b_full_ids = [x for x in ca_38k.index if str(x).startswith("CARAVAN_")]
  t1b_full_cids = set(native_to_cid[x] for x in t1b_full_ids if x in native_to_cid)
  t2a_ids = [x for x in ca_38k.index if not str(x).startswith("GRDC_")]
  t2a_cids = set(native_to_cid[x] for x in t2a_ids if x in native_to_cid)

  tier_dedup_map = {
      "tier0_zero_gauges": dedup.iloc[0:0],
      "tier1a_active_post2016_ccby_8811": dedup[dedup["canonical_gauge_id"].isin(t1a_cids)],
      "tier1b_strict_ccby_19491": dedup[dedup["canonical_gauge_id"].isin(t1b_ccby_cids)],
      "tier1b_full_caravan_24847": dedup[dedup["canonical_gauge_id"].isin(t1b_full_cids)],
      "tier2a_caravan_estreams_32713": dedup[dedup["canonical_gauge_id"].isin(t2a_cids)],
      "tier2_academic_compilations_38069": dedup[dedup["in_canonical_38069"]],
      "tier2_all_open_academic_34630": dedup[dedup["in_canonical_38069"] | (dedup["best_data_access_policy"] == "open_academic_compilation")],
      "tier3_open_public_apis_and_bulk_106410": dedup[dedup["best_data_access_policy"].isin(["open_public_api_or_bulk", "open_academic_compilation"])],
      "tier4_all_publicly_knowable_121819": dedup,
  }

  print("\n3. Loading base Zarr features and Lead-Time-1 SkillPredictor model families...", flush=True)
  df_base = load_zarr_df(f"{CANON_V2_DIR}/generic_training_data/hybas_generic_inference_data.zarr").loc[study_ids].copy()

  # Recompute exact Lead-Time-1 top-quartile OOD medians for Tier 5
  x_train_u, y_train_u_5 = io_utils.load_and_preprocess_data(
      "ungauged", "5.0-year-2.0-window-recall", None, exp_suffix="", lead_time="1"
  )
  top_q_thresh_u = y_train_u_5.quantile(0.75)
  ref_medians_u = x_train_u.loc[y_train_u_5 >= top_q_thresh_u, OOD_COLS].median()

  tier_feats = {}
  tier_masks_union = {}
  tier_masks_snapped = {}
  tier_masks_ge95 = {}

  for t_key in TIER_ORDER:
    pq_path = f"{FEAT_DIR}/{t_key}.parquet"
    tf = pd.read_parquet(pq_path).loc[study_ids].copy()
    if t_key in ("tier5_universal_idealized_ceiling", "tier5_dense_corridor_ceiling"):
      ood_t4_base = pd.read_parquet(f"{FEAT_DIR}/tier4_all_publicly_knowable_121819.parquet").loc[study_ids, OOD_COLS]
      for c in OOD_COLS:
        tf[c] = (
            np.maximum(ood_t4_base[c].values, float(ref_medians_u[c]))
            if c != "anomaly_score_lof"
            else float(ref_medians_u[c])
        )
      tier_masks_snapped[t_key] = np.ones(len(study_ids), dtype=bool)
      tier_masks_ge95[t_key] = np.ones(len(study_ids), dtype=bool)
      tier_masks_union[t_key] = np.ones(len(study_ids), dtype=bool)
    else:
      d_sub = tier_dedup_map[t_key]
      snapped_set = set("hybas_" + d_sub.loc[d_sub["snapped_hybas_id"] > 0, "snapped_hybas_id"].astype(int).astype(str))
      m_snap = np.array([h in snapped_set for h in study_ids], dtype=bool)
      m_ge95 = tf["total_upstream_coverage"].values >= 0.95
      tier_masks_snapped[t_key] = m_snap
      tier_masks_ge95[t_key] = m_ge95
      tier_masks_union[t_key] = m_snap | m_ge95
    tier_feats[t_key] = tf

  families = {}
  for fam_key, sfx in [("base", ""), ("da", "_da"), ("ft", "_ft"), ("ft_da", "_ft_da")]:
    loaded = load_model_family(sfx)
    if loaded is not None:
      families[fam_key] = loaded
      print(f"   Loaded Lead-Time-1 model family '{fam_key}' (exp_suffix='{sfx}')", flush=True)
    else:
      print(f"   Model family '{fam_key}' (exp_suffix='{sfx}') not yet available — skipping for now.", flush=True)

  models_u_base, models_g_base = families["base"]
  rp_cols_u = list(models_u_base[RP_METRICS[0]].impl.feature_columns)
  classic_cols_u = list(models_u_base[CLASSIC_METRICS[0]].impl.feature_names)

  X_rp_base = models_u_base[RP_METRICS[0]].impl._preprocess_features(df_base).reindex(columns=rp_cols_u, fill_value=0)
  X_classic_base = models_u_base[CLASSIC_METRICS[0]].impl._engineer_features(df_base).reindex(columns=classic_cols_u, fill_value=0)

  cov_ref_t1a = tier_feats["tier1a_active_post2016_ccby_8811"][COV_COLS]
  ood_ref_t1a = tier_feats["tier1a_active_post2016_ccby_8811"][OOD_COLS]
  cov_ref_t4 = tier_feats["tier4_all_publicly_knowable_121819"][COV_COLS]
  ood_ref_t4 = tier_feats["tier4_all_publicly_knowable_121819"][OOD_COLS]

  combo_mem_cache = {}

  def evaluate_family_on_features(
      fam_key: str,
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
    """Evaluates a SkillPredictor family (base, da, ft, ft_da) on (cov_df, ood_df)."""
    models_u, models_g = families[fam_key]
    full_cache_key = f"{fam_key}__{combo_key}"
    pair_sig = (fam_key, cov_tag, ood_tag) if (cov_tag and ood_tag) else None
    if pair_sig is not None and pair_sig in combo_mem_cache:
      cached_res = combo_mem_cache[pair_sig]
      if (not eval_classic or "NSE_u" in cached_res) and (not eval_mgauged or "r5_g" in cached_res):
        return cached_res

    cache_pq = f"{PRED_DIR}/{full_cache_key}.parquet"
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
      if eval_classic:
        X_cl = X_classic_base.copy()
        for c in COV_COLS:
          X_cl[c] = cov_df[c].values
        for c in OOD_COLS:
          X_cl[c] = ood_df[c].values
        full_cl_g = fast_predict_classic_pair(models_g, X_cl)
        res["NSE_g"] = full_cl_g["NSE"]
        res["KGE_g"] = full_cl_g["KGE"]

    pd.DataFrame(res, index=study_ids).to_parquet(cache_pq)
    if pair_sig is not None:
      combo_mem_cache[pair_sig] = res
    print(
        f"   [{full_cache_key}] evaluated ({n_chg:,} active rows) in {time.time() - t_eval:.1f}s | "
        f"med_r5_u={np.median(res['r5_u'])*100:.2f}% | med_NSE_u={np.median(res['NSE_u']):.3f}",
        flush=True,
    )
    return res

  # =========================================================================
  # Step 4. Evaluate Anchor Baselines & All Tiers for Each Available Family
  # =========================================================================
  print("\n4. Evaluating Anchor Baselines and All Tiers across model families...", flush=True)
  family_tier_evals = {}

  for fam_key in families:
    print(f"\n--- Evaluating Model Family: {fam_key} ---", flush=True)
    p_t4 = evaluate_family_on_features(
        fam_key,
        "tier4_all_publicly_knowable_121819__joint",
        cov_ref_t4,
        ood_ref_t4,
        cov_tag="tier4_all_publicly_knowable_121819",
        ood_tag="tier4_all_publicly_knowable_121819",
        eval_classic=True,
        eval_mgauged=True,
    )
    p_t1a = evaluate_family_on_features(
        fam_key,
        "tier1a_active_post2016_ccby_8811__joint",
        cov_ref_t1a,
        ood_ref_t1a,
        cov_tag="tier1a_active_post2016_ccby_8811",
        ood_tag="tier1a_active_post2016_ccby_8811",
        eval_classic=True,
        eval_mgauged=True,
        ref_preds=p_t4,
        ref_cov=cov_ref_t4,
        ref_ood=ood_ref_t4,
    )

    for t_key in TIER_ORDER:
      cov_k = tier_feats[t_key][COV_COLS]
      ood_k = tier_feats[t_key][OOD_COLS]
      c_tag = t_key
      o_tag = "tier1a_active_post2016_ccby_8811" if t_key == "tier0_zero_gauges" else (
          "tier5_universal_idealized_ceiling" if t_key == "tier5_dense_corridor_ceiling" else t_key
      )

      family_tier_evals[(fam_key, t_key, "joint")] = evaluate_family_on_features(
          fam_key,
          f"{t_key}__joint",
          cov_k,
          ood_k,
          cov_tag=c_tag,
          ood_tag=o_tag,
          eval_classic=True,
          eval_mgauged=(t_key in ("tier1a_active_post2016_ccby_8811", "tier4_all_publicly_knowable_121819", "tier5_universal_idealized_ceiling", "tier5_dense_corridor_ceiling")),
          ref_preds=p_t1a,
          ref_cov=cov_ref_t1a,
          ref_ood=ood_ref_t1a,
      )

      if fam_key == "base":
        family_tier_evals[(fam_key, t_key, "f2_cov_only_base1a")] = evaluate_family_on_features(
            fam_key,
            f"{t_key}__f2_cov_only_base1a",
            cov_k,
            ood_ref_t1a,
            cov_tag=c_tag,
            ood_tag="tier1a_active_post2016_ccby_8811",
            eval_classic=True,
            eval_mgauged=False,
            ref_preds=p_t1a,
            ref_cov=cov_ref_t1a,
            ref_ood=ood_ref_t1a,
        )
        family_tier_evals[(fam_key, t_key, "f3_ood_only_base1a")] = evaluate_family_on_features(
            fam_key,
            f"{t_key}__f3_ood_only_base1a",
            cov_ref_t1a,
            ood_k,
            cov_tag="tier1a_active_post2016_ccby_8811",
            ood_tag=o_tag,
            eval_classic=True,
            eval_mgauged=False,
            ref_preds=family_tier_evals[(fam_key, t_key, "joint")],
            ref_cov=cov_k,
            ref_ood=ood_k,
        )

  # =========================================================================
  # Step 5. Construct Mechanism-Specific Predictions Across All Tiers
  # =========================================================================
  print("\n5. Constructing multi-mechanism global skill & exposure tables...", flush=True)
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

  def blend_local_and_global(p_global: dict, p_local: dict, mask_local: np.ndarray, use_gauged_head: bool = False) -> dict:
    """Replaces predictions at mask_local with p_local (either _u or _g head)."""
    out = {}
    for k in ["r5_u", "r10_u", "r20_u", "NSE_u", "KGE_u"]:
      arr = p_global[k].copy()
      src_key = k.replace("_u", "_g") if (use_gauged_head and k.replace("_u", "_g") in p_local) else k
      arr[mask_local] = p_local[src_key][mask_local]
      out[k] = arr
    return out

  p_base_t0 = family_tier_evals[("base", "tier0_zero_gauges", "joint")]
  p_base_t1a = family_tier_evals[("base", "tier1a_active_post2016_ccby_8811", "joint")]
  p_base_t4 = family_tier_evals[("base", "tier4_all_publicly_knowable_121819", "joint")]

  w_base_t0 = calc_warned_vec(p_base_t0)
  w_base_t1a = calc_warned_vec(p_base_t1a)
  w_base_t4 = calc_warned_vec(p_base_t4)

  mask_sys60 = df_exp["has_system_60"].values
  mask_nosys113 = (~df_exp["has_system_60"].values) & df_exp["country_name"].notna().values
  mask_cap62 = df_exp["has_wmo_cap_62"].values
  mask_nocap111 = (~df_exp["has_wmo_cap_62"].values) & df_exp["country_name"].notna().values

  summary_rows = []

  for t_key in TIER_ORDER:
    mask_u = tier_masks_union[t_key]
    mask_s = tier_masks_snapped[t_key]

    mechanisms_for_tier = [
        ("000_global_train_joint", family_tier_evals[("base", t_key, "joint")]),
        ("000_global_train_f2_cov_only_base1a", family_tier_evals[("base", t_key, "f2_cov_only_base1a")]),
        ("000_global_train_f3_ood_only_base1a", family_tier_evals[("base", t_key, "f3_ood_only_base1a")]),
    ]

    if "da" in families:
      p_da_t1a = family_tier_evals[("da", "tier1a_active_post2016_ccby_8811", "joint")]
      p_da_tk = family_tier_evals[("da", t_key, "joint")]
      # 001: 30-Day DA Only (Global Training fixed at Tier 1a; local DA at gauged basins in Tier k)
      mechanisms_for_tier.append((
          "001_da_only_union_mask",
          blend_local_and_global(p_base_t1a, p_da_t1a, mask_u, use_gauged_head=False),
      ))
      mechanisms_for_tier.append((
          "001_da_only_snapped_mask",
          blend_local_and_global(p_base_t1a, p_da_t1a, mask_s, use_gauged_head=False),
      ))
      # 101: Global Training (Tier k) + 30-Day DA at gauged basins in Tier k
      mechanisms_for_tier.append((
          "101_global_train_plus_da_union_mask",
          blend_local_and_global(family_tier_evals[("base", t_key, "joint")], p_da_tk, mask_u, use_gauged_head=False),
      ))
      mechanisms_for_tier.append((
          "101_global_train_plus_da_snapped_mask",
          blend_local_and_global(family_tier_evals[("base", t_key, "joint")], p_da_tk, mask_s, use_gauged_head=False),
      ))
      # Universal DA evaluated on Tier k features across all 964,084 basins (pure M_DA curve)
      mechanisms_for_tier.append((
          "001_pure_M_DA_all_basins",
          p_da_tk,
      ))

    if "ft" in families:
      p_ft_t1a = family_tier_evals[("ft", "tier1a_active_post2016_ccby_8811", "joint")]
      p_ft_tk = family_tier_evals[("ft", t_key, "joint")]
      mechanisms_for_tier.append((
          "010_ft_only_union_mask",
          blend_local_and_global(p_base_t1a, p_ft_t1a, mask_u, use_gauged_head=False),
      ))
      mechanisms_for_tier.append((
          "010_ft_only_snapped_mask",
          blend_local_and_global(p_base_t1a, p_ft_t1a, mask_s, use_gauged_head=False),
      ))
      mechanisms_for_tier.append((
          "010_pure_M_FT_all_basins",
          p_ft_tk,
      ))

    if "ft_da" in families:
      p_ft_da_tk = family_tier_evals[("ft_da", t_key, "joint")]
      mechanisms_for_tier.append((
          "011_all_three_combined_union_mask",
          blend_local_and_global(family_tier_evals[("base", t_key, "joint")], p_ft_da_tk, mask_u, use_gauged_head=False),
      ))
      mechanisms_for_tier.append((
          "011_all_three_combined_snapped_mask",
          blend_local_and_global(family_tier_evals[("base", t_key, "joint")], p_ft_da_tk, mask_s, use_gauged_head=False),
      ))
      mechanisms_for_tier.append((
          "011_pure_M_FT_DA_all_basins",
          p_ft_da_tk,
      ))

    for mech_name, p in mechanisms_for_tier:
      w_vec = calc_warned_vec(p)
      unw_vec = etot - w_vec

      d_net_vs_t1a = w_vec - w_base_t1a
      d_net_vs_t0 = w_vec - w_base_t0
      d_net_vs_t4 = w_vec - w_base_t4

      d_rp_vs_t1a = calc_rp_clipped_delta_vec(p, p_base_t1a)
      d_rp_vs_t0 = calc_rp_clipped_delta_vec(p, p_base_t0)
      d_rp_vs_t4 = calc_rp_clipped_delta_vec(p, p_base_t4)

      def _qstats(arr):
        q25, q50, q75 = np.percentile(arr, [25, 50, 75])
        return float(q50), float(q25), float(q75), float(q75 - q25), float(np.mean(arr))

      r5_med, r5_q25, r5_q75, r5_iqr, r5_mean = _qstats(p["r5_u"] * 100.0)
      r10_med, r10_q25, r10_q75, r10_iqr, r10_mean = _qstats(p["r10_u"] * 100.0)
      r20_med, r20_q25, r20_q75, r20_iqr, r20_mean = _qstats(p["r20_u"] * 100.0)
      nse_med, nse_q25, nse_q75, nse_iqr, nse_mean = _qstats(p["NSE_u"])
      kge_med, kge_q25, kge_q75, kge_iqr, kge_mean = _qstats(p["KGE_u"])

      summary_rows.append({
          "tier_key": t_key,
          "tier_label": TIER_LABELS[t_key],
          "mechanism": mech_name,
          "n_gauged_union": int(mask_u.sum()),
          "n_gauged_snapped": int(mask_s.sum()),
          "r5_median_pct": r5_med, "r5_q25_pct": r5_q25, "r5_q75_pct": r5_q75, "r5_iqr_pct": r5_iqr, "r5_mean_pct": r5_mean,
          "r10_median_pct": r10_med, "r10_q25_pct": r10_q25, "r10_q75_pct": r10_q75, "r10_iqr_pct": r10_iqr, "r10_mean_pct": r10_mean,
          "r20_median_pct": r20_med, "r20_q25_pct": r20_q25, "r20_q75_pct": r20_q75, "r20_iqr_pct": r20_iqr, "r20_mean_pct": r20_mean,
          "NSE_median": nse_med, "NSE_q25": nse_q25, "NSE_q75": nse_q75, "NSE_iqr": nse_iqr, "NSE_mean": nse_mean,
          "KGE_median": kge_med, "KGE_q25": kge_q25, "KGE_q75": kge_q75, "KGE_iqr": kge_iqr, "KGE_mean": kge_mean,
          "E_total_global": float(etot.sum()),
          "warned_global": float(w_vec.sum()),
          "unwarned_global": float(unw_vec.sum()),
          "pct_warned_global": float(100.0 * w_vec.sum() / etot.sum()),
          "pct_unwarned_global": float(100.0 * unw_vec.sum() / etot.sum()),
          "delta_net_vs_base_t1a_global": float(d_net_vs_t1a.sum()),
          "delta_rp_clipped_vs_base_t1a_global": float(d_rp_vs_t1a.sum()),
          "pct_unwarned_closed_net_vs_base_t1a": float(100.0 * d_net_vs_t1a.sum() / (etot.sum() - w_base_t1a.sum())),
          "pct_unwarned_closed_rp_vs_base_t1a": float(100.0 * d_rp_vs_t1a.sum() / (etot.sum() - w_base_t1a.sum())),
          "delta_net_vs_base_t0_global": float(d_net_vs_t0.sum()),
          "delta_rp_clipped_vs_base_t0_global": float(d_rp_vs_t0.sum()),
          "delta_net_vs_base_t4_global": float(d_net_vs_t4.sum()),
          "delta_rp_clipped_vs_base_t4_global": float(d_rp_vs_t4.sum()),
          "warned_sys60": float(w_vec[mask_sys60].sum()),
          "unwarned_sys60": float(unw_vec[mask_sys60].sum()),
          "delta_net_vs_base_t1a_sys60": float(d_net_vs_t1a[mask_sys60].sum()),
          "delta_rp_vs_base_t1a_sys60": float(d_rp_vs_t1a[mask_sys60].sum()),
          "warned_nosys113": float(w_vec[mask_nosys113].sum()),
          "unwarned_nosys113": float(unw_vec[mask_nosys113].sum()),
          "delta_net_vs_base_t1a_nosys113": float(d_net_vs_t1a[mask_nosys113].sum()),
          "delta_rp_vs_base_t1a_nosys113": float(d_rp_vs_t1a[mask_nosys113].sum()),
          "warned_cap62": float(w_vec[mask_cap62].sum()),
          "unwarned_cap62": float(unw_vec[mask_cap62].sum()),
          "delta_net_vs_base_t1a_cap62": float(d_net_vs_t1a[mask_cap62].sum()),
          "delta_rp_vs_base_t1a_cap62": float(d_rp_vs_t1a[mask_cap62].sum()),
          "warned_nocap111": float(w_vec[mask_nocap111].sum()),
          "unwarned_nocap111": float(unw_vec[mask_nocap111].sum()),
          "delta_net_vs_base_t1a_nocap111": float(d_net_vs_t1a[mask_nocap111].sum()),
          "delta_rp_vs_base_t1a_nocap111": float(d_rp_vs_t1a[mask_nocap111].sum()),
      })

  df_sum = pd.DataFrame(summary_rows)
  out_csv = f"{OUT_DIR}/multi_mechanism_tiered_skill_and_exposure_lead1.csv"
  df_sum.to_csv(out_csv, index=False)
  print(f"\nSaved full summary to {out_csv} in {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
  main()
