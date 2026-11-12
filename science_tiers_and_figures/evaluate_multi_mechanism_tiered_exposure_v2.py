#!/usr/bin/env python3
"""Evaluates Lead-Time-1 SkillPredictor meta-models across all 964,084 global watersheds
and all institutional data-sharing & licensing tiers (Tier 0 through Tier 5) with Option B
trajectory-crossing return-period metrics and Option 1 Active-Only DA filtering:
  1. Global Training Only (M_base, '000'):
     - Joint (Factor 2 Coverage at Tier k + Factor 3 OOD at Tier k, using all gauges in Tier k)
     - Factor 2 Coverage Only (holding Factor 3 OOD fixed at Tier 1a [8,811])
     - Factor 3 Climate OOD Only (holding Factor 2 Coverage fixed at Tier 1a [8,811])
  2. 30-Day Variational Data Assimilation Only (M_DA, '001'):
     - Holding global training fixed at Tier 1a (8,811), applying M_DA ONLY at Active gauged sub-basins (`mask_active`) in Tier k
  3. Global Training + 30-Day Variational Data Assimilation ('101'):
     - Updating global training to Tier k across all 964,084 sub-basins + applying M_DA ONLY at Active gauged sub-basins (`mask_active`) in Tier k
  4. Target Fine-Tuning Only (M_FT, '010'):
     - Holding global training fixed at Tier 1a (8,811), applying M_FT at All gauged sub-basins (`mask_all`, Active + Non-Active) in Tier k
  5. Global Training + Target Fine-Tuning ('110'):
     - Updating global training to Tier k + applying M_FT at All gauged sub-basins (`mask_all`) in Tier k
  6. Local FT + Local DA Only ('011_ft_plus_da_only'):
     - Holding global training fixed at Tier 1a (8,811), applying M_FT_DA at Active gauged sub-basins (`mask_active`) and falling back to M_FT at Non-Active gauged sub-basins (`mask_all & ~mask_active`)
  7. All Three Combined: Global Training + Target Fine-Tuning + 30-Day DA ('011_all_three_combined' / '111'):
     - Updating global training to Tier k across all 964,084 sub-basins + applying M_FT_DA at Active gauged sub-basins (`mask_active`) and falling back to M_FT at Non-Active gauged sub-basins (`mask_all & ~mask_active`)
"""

import os
import sys
import time

os.environ.setdefault("GEO_BIAS_DATA_DIR", "./data/canonical_v2")
os.environ.setdefault("GEO_BIAS_THIRD_PARTY_DIR", "./data/canonical_v2")
os.environ.setdefault("GEO_BIAS_OUTPUT_DIR", "./data/canonical_v2")

import calendar  # noqa: F401
sys.path = [p for p in sys.path if not p.endswith("/google3")]
sys.path.insert(0, "./data/scratch")

import numpy as np
import pandas as pd
import evaluate_multi_mechanism_tiered_exposure as emm

CANON_V2_DIR = emm.CANON_V2_DIR
FEAT_DIR = emm.FEAT_DIR
OUT_DIR = emm.OUT_DIR
PRED_DIR = emm.PRED_DIR
MASKS_PATH = "./data/scratch/tier_active_and_all_masks.parquet"
RP_WEIGHTS = emm.RP_WEIGHTS
RP_METRICS = emm.RP_METRICS
CLASSIC_METRICS = emm.CLASSIC_METRICS
COV_COLS = emm.COV_COLS
OOD_COLS = emm.OOD_COLS
TIER_ORDER = emm.TIER_ORDER
TIER_LABELS = emm.TIER_LABELS
WB_INCOME_BY_ISO3 = emm.WB_INCOME_BY_ISO3


def main():
  t0 = time.time()
  emm.file_paths.init("5.0-year-2.0-window-recall")

  print("1. Loading global basin exposure (964,084 study watersheds) and country/CAP/income metadata...", flush=True)
  df_basin_v2 = pd.read_parquet(
      f"{CANON_V2_DIR}/v2_exposure_blockers/statistics/basin_exposure_and_blockers_v2.parquet"
  )
  study_ids = [
      x if str(x).startswith("hybas_") else f"hybas_{x}"
      for x in df_basin_v2.index
  ]
  df_basin_v2.index = study_ids

  df_exp_raw = pd.read_csv(emm.file_paths.GLOBAL_BASIN_EXPOSURE_DEFENDED_BY_RP_PATH)
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

  df_map = pd.read_csv(emm.file_paths.HYBAS_MAPPING_PATH)
  if "Unnamed: 0" in df_map.columns and "hybas_id" not in df_map.columns:
    df_map = df_map.rename(columns={"Unnamed: 0": "hybas_id"})
  df_map["hybas_id"] = (
      df_map["hybas_id"]
      .astype(str)
      .str.replace(r"\.0$", "", regex=True)
      .apply(lambda x: x if x.startswith("hybas_") else f"hybas_{x}")
  )
  df_map["country_name_clean"] = df_map["country_name"].replace(emm.file_paths.country_name_map)
  df_map_dedup = df_map.drop_duplicates("hybas_id").set_index("hybas_id")

  df_exp["country_name"] = df_map_dedup["country_name_clean"].reindex(study_ids)
  df_exp["has_system_60"] = (
      df_exp["country_name"]
      .map(emm.cap_utils.has_effective_flood_warning_system)
      .fillna(False)
      .astype(bool)
  )
  df_exp["cap_tier"] = (
      df_exp["country_name"]
      .map(emm.cap_utils.assign_cap_tier)
      .fillna("Tier 4: Complete Systemic Gap (No CAP)")
  )
  t2_non_wmo_7 = {
      "Bosnia and Herzegovina", "Bosnia and Herz.", "Eswatini", "eSwatini", "Swaziland",
      "Guinea-Bissau", "Lesotho", "Montenegro", "North Macedonia", "Macedonia", "Trinidad and Tobago",
  }
  wmo_cap_lookup_countries = (
      set(emm.cap_utils.TIER_1_COMPREHENSIVE_COUNTRIES)
      | (set(emm.cap_utils.TIER_2_PROPORTIONALITY_COUNTRIES) - t2_non_wmo_7)
  )
  df_exp["has_wmo_cap_62"] = (
      df_exp["country_name"].isin(wmo_cap_lookup_countries).fillna(False).astype(bool)
  )

  print("\n2. Loading Active-Only and All-Gauges watershed masks...", flush=True)
  df_masks = pd.read_parquet(MASKS_PATH).loc[study_ids]

  print("\n3. Loading base Zarr features and Lead-Time-1 SkillPredictor model families...", flush=True)
  df_base = emm.load_zarr_df(f"{CANON_V2_DIR}/generic_training_data/hybas_generic_inference_data.zarr").loc[study_ids].copy()

  x_train_u, y_train_u_5 = emm.io_utils.load_and_preprocess_data(
      "ungauged", "5.0-year-2.0-window-recall", None, exp_suffix="", lead_time="1"
  )
  top_q_thresh_u = y_train_u_5.quantile(0.75)
  ref_medians_u = x_train_u.loc[y_train_u_5 >= top_q_thresh_u, OOD_COLS].median()

  tier_feats = {}
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
    tier_feats[t_key] = tf

  families = {}
  for fam_key, sfx in [("base", ""), ("da", "_da"), ("ft", "_ft"), ("ft_da", "_ft_da")]:
    loaded = emm.load_model_family(sfx)
    if loaded is not None:
      families[fam_key] = loaded
      print(f"   Loaded Lead-Time-1 model family '{fam_key}' (exp_suffix='{sfx}')", flush=True)

  models_u_base, _ = families["base"]
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
    models_u, models_g = families[fam_key]
    full_cache_key = f"{fam_key}__{combo_key}"
    pair_sig = (fam_key, cov_tag, ood_tag) if (cov_tag and ood_tag) else None
    if pair_sig is not None and pair_sig in combo_mem_cache:
      cached_res = combo_mem_cache[pair_sig]
      if (not eval_classic or "NSE_u" in cached_res) and (not eval_mgauged or "r5_g" in cached_res):
        return cached_res

    cache_pq = f"{PRED_DIR}/{full_cache_key}.parquet"
    if os.path.exists(cache_pq):
      df_cached = pd.read_parquet(cache_pq).loc[study_ids]
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
        sub_rp = emm.fast_predict_rp_triplet(models_u, X_rp.iloc[changed_mask])
        res["r5_u"][changed_mask] = sub_rp["5.0-year-2.0-window-recall"]
        res["r10_u"][changed_mask] = sub_rp["10.0-year-2.0-window-recall"]
        res["r20_u"][changed_mask] = sub_rp["20.0-year-2.0-window-recall"]
    else:
      n_chg = len(X_rp)
      full_rp = emm.fast_predict_rp_triplet(models_u, X_rp)
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
          sub_cl = emm.fast_predict_classic_pair(models_u, X_cl.iloc[changed_mask])
          res["NSE_u"][changed_mask] = sub_cl["NSE"]
          res["KGE_u"][changed_mask] = sub_cl["KGE"]
      else:
        full_cl = emm.fast_predict_classic_pair(models_u, X_cl)
        res["NSE_u"] = full_cl["NSE"]
        res["KGE_u"] = full_cl["KGE"]

    if eval_mgauged:
      if changed_mask is not None and changed_mask.sum() < 0.75 * len(X_rp) and "r5_g" in ref_preds:
        res["r5_g"] = ref_preds["r5_g"].copy()
        res["r10_g"] = ref_preds["r10_g"].copy()
        res["r20_g"] = ref_preds["r20_g"].copy()
        if changed_mask.sum() > 0:
          sub_g = emm.fast_predict_rp_triplet(models_g, X_rp.iloc[changed_mask])
          res["r5_g"][changed_mask] = sub_g["5.0-year-2.0-window-recall"]
          res["r10_g"][changed_mask] = sub_g["10.0-year-2.0-window-recall"]
          res["r20_g"][changed_mask] = sub_g["20.0-year-2.0-window-recall"]
      else:
        full_g = emm.fast_predict_rp_triplet(models_g, X_rp)
        res["r5_g"] = full_g["5.0-year-2.0-window-recall"]
        res["r10_g"] = full_g["10.0-year-2.0-window-recall"]
        res["r20_g"] = full_g["20.0-year-2.0-window-recall"]
      if eval_classic:
        X_cl = X_classic_base.copy()
        for c in COV_COLS:
          X_cl[c] = cov_df[c].values
        for c in OOD_COLS:
          X_cl[c] = ood_df[c].values
        full_cl_g = emm.fast_predict_classic_pair(models_g, X_cl)
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

  print("\n5. Constructing multi-mechanism global skill & exposure tables (with Active-Only DA and FT-Fallback)...", flush=True)
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

  def blend_local_and_global(p_global: dict, p_local: dict, mask_local: np.ndarray) -> dict:
    out = {}
    for k in ["r5_u", "r10_u", "r20_u", "NSE_u", "KGE_u"]:
      arr = p_global[k].copy()
      arr[mask_local] = p_local[k][mask_local]
      out[k] = arr
    return out

  def blend_three_way(p_global: dict, p_ft_local: dict, p_ft_da_local: dict, mask_all: np.ndarray, mask_act: np.ndarray) -> dict:
    """Applies M_FT_DA on Active gauges (`mask_act`), M_FT on Non-Active gauges (`mask_all & ~mask_act`), and M_base elsewhere."""
    out = {}
    for k in ["r5_u", "r10_u", "r20_u", "NSE_u", "KGE_u"]:
      arr = p_global[k].copy()
      arr[mask_all] = p_ft_local[k][mask_all]
      arr[mask_act] = p_ft_da_local[k][mask_act]
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
    mask_u_all = df_masks[f"{t_key}__union_all"].values
    mask_s_all = df_masks[f"{t_key}__snap_all"].values
    mask_u_act = df_masks[f"{t_key}__union_active"].values
    mask_s_act = df_masks[f"{t_key}__snap_active"].values

    mechanisms_for_tier = [
        ("000_global_train_joint", family_tier_evals[("base", t_key, "joint")]),
        ("000_global_train_f2_cov_only_base1a", family_tier_evals[("base", t_key, "f2_cov_only_base1a")]),
        ("000_global_train_f3_ood_only_base1a", family_tier_evals[("base", t_key, "f3_ood_only_base1a")]),
    ]

    if "da" in families:
      p_da_t1a = family_tier_evals[("da", "tier1a_active_post2016_ccby_8811", "joint")]
      p_da_tk = family_tier_evals[("da", t_key, "joint")]
      # 001: 30-Day DA Only (Global Training fixed at Tier 1a; local DA at ACTIVE gauged basins in Tier k)
      mechanisms_for_tier.append((
          "001_da_only_union_mask",
          blend_local_and_global(p_base_t1a, p_da_t1a, mask_u_act),
      ))
      mechanisms_for_tier.append((
          "001_da_only_snapped_mask",
          blend_local_and_global(p_base_t1a, p_da_t1a, mask_s_act),
      ))
      # 101: Global Training (Tier k) + 30-Day DA at ACTIVE gauged basins in Tier k
      mechanisms_for_tier.append((
          "101_global_train_plus_da_union_mask",
          blend_local_and_global(family_tier_evals[("base", t_key, "joint")], p_da_tk, mask_u_act),
      ))
      mechanisms_for_tier.append((
          "101_global_train_plus_da_snapped_mask",
          blend_local_and_global(family_tier_evals[("base", t_key, "joint")], p_da_tk, mask_s_act),
      ))
      mechanisms_for_tier.append((
          "001_pure_M_DA_all_basins",
          p_da_tk,
      ))

    if "ft" in families:
      p_ft_t1a = family_tier_evals[("ft", "tier1a_active_post2016_ccby_8811", "joint")]
      p_ft_tk = family_tier_evals[("ft", t_key, "joint")]
      # 010: FT Only (applied at ALL gauged basins in Tier k: Active + Non-Active)
      mechanisms_for_tier.append((
          "010_ft_only_union_mask",
          blend_local_and_global(p_base_t1a, p_ft_t1a, mask_u_all),
      ))
      mechanisms_for_tier.append((
          "010_ft_only_snapped_mask",
          blend_local_and_global(p_base_t1a, p_ft_t1a, mask_s_all),
      ))
      mechanisms_for_tier.append((
          "110_global_train_plus_ft_union_mask",
          blend_local_and_global(family_tier_evals[("base", t_key, "joint")], p_ft_tk, mask_u_all),
      ))
      mechanisms_for_tier.append((
          "110_global_train_plus_ft_snapped_mask",
          blend_local_and_global(family_tier_evals[("base", t_key, "joint")], p_ft_tk, mask_s_all),
      ))
      mechanisms_for_tier.append((
          "010_pure_M_FT_all_basins",
          p_ft_tk,
      ))

    if "ft_da" in families and "ft" in families:
      p_ft_t1a = family_tier_evals[("ft", "tier1a_active_post2016_ccby_8811", "joint")]
      p_ft_tk = family_tier_evals[("ft", t_key, "joint")]
      p_ft_da_t1a = family_tier_evals[("ft_da", "tier1a_active_post2016_ccby_8811", "joint")]
      p_ft_da_tk = family_tier_evals[("ft_da", t_key, "joint")]
      # Local FT + DA Only (holding global training at Tier 1a; M_FT_DA on Active gauges, M_FT on Non-Active gauges)
      mechanisms_for_tier.append((
          "011_ft_plus_da_only_union_mask",
          blend_three_way(p_base_t1a, p_ft_t1a, p_ft_da_t1a, mask_u_all, mask_u_act),
      ))
      mechanisms_for_tier.append((
          "011_ft_plus_da_only_snapped_mask",
          blend_three_way(p_base_t1a, p_ft_t1a, p_ft_da_t1a, mask_s_all, mask_s_act),
      ))
      # All Three Combined (Global Training on Tier k + M_FT_DA on Active gauges + M_FT on Non-Active gauges)
      mechanisms_for_tier.append((
          "011_all_three_combined_union_mask",
          blend_three_way(family_tier_evals[("base", t_key, "joint")], p_ft_tk, p_ft_da_tk, mask_u_all, mask_u_act),
      ))
      mechanisms_for_tier.append((
          "011_all_three_combined_snapped_mask",
          blend_three_way(family_tier_evals[("base", t_key, "joint")], p_ft_tk, p_ft_da_tk, mask_s_all, mask_s_act),
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
          "n_gauged_union_all": int(mask_u_all.sum()),
          "n_gauged_union_active": int(mask_u_act.sum()),
          "n_gauged_snapped_all": int(mask_s_all.sum()),
          "n_gauged_snapped_active": int(mask_s_act.sum()),
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
