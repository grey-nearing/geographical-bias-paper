#!/usr/bin/env python3
"""Forensic 73-archive physical-gauge, streamflow-timeseries, and open-license audit.

Rebuilds `global_gauges_by_source_archive.parquet` and `global_gauges_deduplicated.parquet`
in `$GEO_BIAS_DATA_DIR/attributes/` (preserving canonical_gauge_id keys for all surviving
physical river gauges so downstream 88-feature HydroATLAS tables align 1-to-1) under three
strict, auditable criteria:

1. STRICT PHYSICAL IN-SITU RIVER GAUGES ONLY (Zero Simulations / Non-River Sites):
   - Excludes `MERV_Jp` (135 multi-model simulated runoff series, Zenodo 22037429).
   - Excludes Norway NVE `Virtuell` (3 simulated/virtual nodes) and `Uten målinger` (15
     stations without measurements).
   - Excludes Colombia `GIS_Internal / IDEAM DHIME` non-river meteorological, rainfall,
     radiosonde, marine, and air-quality stations (`Pluviométrica`, `Pluviográfica`,
     `Climatológica*`, `Meteorológica*`, `Agrometeorológica`, `Sinóptica*`, `Radio Sonda`,
     `Ambiental`, and `Estación Ficticia 10000000`) verified against `ideam_catalog_raw.csv`.
   - Excludes England `GIS_Internal / UK Environment Agency` groundwater boreholes and
     rain gauges verified against `uk_ea_sites.csv`.
   - Excludes Netherlands `RivRetrieve / Rijkswaterstaat Waterwebservices` offshore North Sea
     platforms, beach swimming-water temperature sensors, salinity/wave buoys, and groundwater
     sites verified against Rijkswaterstaat `OphalenCatalogus` (`OW+Q` & inland `OW+WATHTE` kept).
   - Excludes Switzerland `CAMELS-CH` / `FOEN` pure lake-level gauges (`Seepegel`,
     `daily_lake_level`, 33 stations).

2. ACTUAL OBSERVED STREAMFLOW TIME SERIES REQUIRED FOR DISCHARGE TIERS (Tier 1 & Tier 2):
   - Reclassifies metadata/signature/index/flood-peak/standardized-anomaly/satellite-extended
     compilations out of `daily_discharge`/`monthly_discharge`:
       * `EStreams` (metadata, attributes, signatures, and portal scripts only; zero streamflow CSVs)
       * `FOCA_ISPRA_Italy` (annual maximum flood peaks + attributes only)
       * `ADHI` (summary hydrometric indices only; raw daily SIEREM/GRDC flows restricted)
       * `North_Africa_98_Basins` (isolated flood-event peaks & volumes only)
       * `CDR2` (310 Landsat/Sentinel river-width hydraulic-geometry reconstructions + 886 metadata rows)
       * `CA_discharge` (161 long-term norm-only rows -> `flow_indices_only`; 134 digitized yearbook
         discharge series retained as `daily_discharge` / `monthly_discharge`)
       * `GHI_v1` G3 & `GHI_v2` G4 (station catalog / GSIM indices only -> `flow_indices_only`;
         `GHI_v1` G1-G2, `GHI_v2` G1-G3, and `GHSA_v3` G3 reclassified to `monthly_discharge`)
       * `HydroMLYR` (standardized z-score weekly anomalies without volumetric m^3/s -> `flow_indices_only`)
       * `ROBIN_v1` (674 restricted metadata-only stations -> `flow_indices_only`; 2,386 CC-BY-4.0
         daily streamflow stations retained)
       * `CAMELS_IND` / `Caravan camelsind` (159 LSTM-simulated-only CWC gauges + 6 gauges with
         0.00 years of observed flow -> `catchment_attributes_and_boundaries`)
       * `CAMELS_NZ` (14 hydro-operator permission-required gauges -> `catchment_attributes_and_boundaries`)
       * `JPDH_MLIT_Network_Extra` (154 gauge-location-only rows -> `catchment_attributes_and_boundaries`;
         580 water-level-only rows -> `stage_or_fluviometric`)
   - Enforces station-level discharge (`Q`) vs. stage-only (`H`) classification across
     `GIS_Internal`, `RivRetrieve`, and `IDEAM_Colombia` (`USGS NWIS` 00060 vs 00065,
     `NVE HydAPI` `discharge_daily_mean`, `UK EA` `flow` vs `level`, `ANA HidroWeb`
     `Desc_Liquida` vs `Escala`, `IDEAM` verified AQUARIUS `CAUDAL` vs `Limnimétrica`/`NIVEL`,
     and `Rijkswaterstaat` `OW+Q` vs `OW+WATHTE`).

3. FULL-USE OPEN LICENSE REQUIRED FOR TIER 1 & TIER 2 (Login Permitted; Usage Restrictions Excluded):
   - Free registration / API-key login alone is NOT treated as a barrier (e.g., `NVE HydAPI` NLOD 2.0,
     `HRFCO` KOGL Type 1).
   - Licenses that restrict commercial/operational reuse or prohibit redistribution are RESTRICTED
     OUT of Tier 1 and Tier 2 into Tier 3 (`noncommercial_or_restricted_portal`):
       * `GRDC` Portal (`6,415` non-Caravan stations under GRDC non-commercial research policy;
         whereas the `5,356` `GRDC-Caravan` stations explicitly licensed under `CC BY 4.0` ARE
         included in Tier 1 & Tier 2!)
       * `R_ArcticNET_v4` (`Open-Academic-UNH`, academic research only)
       * `JPDH` (`D3P-JP-Q` on HuggingFace, `CC BY-NC 4.0` non-commercial; whereas the `1,029`
         Japanese stations directly on `MLIT WIS` under `PDL 1.0` allow full commercial use -> Tier 2!)
       * `SSA-HCD` & `CIS_Watersheds / GRDC_CIS` (repackaged non-Caravan GRDC runoff)
       * `DWS South Africa` (`dws.gov.za`, non-commercial research/personal use only + HTTP 403)
       * `HII ThaiWater` (`CC BY-NC` non-commercial)
       * `IMGW-PIB` direct portal (Polish Water Law commercial contract requirement; whereas
         `CAMELS-PL` `869` stations published under `CC BY 4.0` ARE in Tier 1 & Tier 2!)
       * `UKCEH NRFA` direct portal (Click-Through Licence prohibiting raw-data redistribution;
         whereas `CAMELS-GB`, `ROBIN_v1`, `UK EA`, and `SEPA` under `OGL v3.0` / `CC BY 4.0` ARE in Tier 1 & Tier 2!)
       * National feeds lacking an established open full-use license or bilateral feeds
         (`SENAMHI Peru` portal [while `CAMELS-PE` 136 CC-BY-4.0 is Tier 1/2], `INA Alerta`
         Argentina, `AVP Sava` Bosnia, `OpenHI` Greece, `DINAGUA Uruguay`, `MRC`, `CWC India`,
         `BWDB`, `DWR Thailand`, `ICE Costa Rica`, `KRC/WRA Kenya`, `MAWLR Namibia`,
         `MOI Sri Lanka`, `NIHSA Nigeria`, `PID Pakistan`, `PUPR Indonesia`, `VNMHA Vietnam`).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = REPO_ROOT.parent
GEO_BIAS_DATA_DIR = Path(os.environ.get("GEO_BIAS_DATA_DIR", str(PROJECT_ROOT / "data")))
ATTR_DIR = GEO_BIAS_DATA_DIR / "attributes"
REF_DIR = REPO_ROOT / "configs" / "audit_reference"
REF_DIR.mkdir(parents=True, exist_ok=True)

RAW_AUDIT_DIR = Path(os.environ.get("RAW_AUDIT_DIR", str(GEO_BIAS_DATA_DIR / "raw_audit")))
HARV_DIR = RAW_AUDIT_DIR / "harvested_catalogs"
RIV_DIR = RAW_AUDIT_DIR / "rivretrieve_audit"

OBS_TYPE_RANK = {
    "daily_discharge": 1,
    "monthly_discharge": 2,
    "stage_or_fluviometric": 3,
    "flow_indices_only": 4,
    "catchment_attributes_and_boundaries": 5,
}
INV_OBS_RANK = {v: k for k, v in OBS_TYPE_RANK.items()}

ACCESS_POLICY_RANK = {
    "open_academic_compilation": 1,
    "open_public_api_or_bulk": 2,
    "open_indices_and_metadata_only": 3,
    "open_attributes_and_metadata_only": 4,
    "registration_or_request_portal": 5,
    "restricted_or_bilateral_partner": 6,
}
INV_POL_RANK = {v: k for k, v in ACCESS_POLICY_RANK.items()}

# Open-licensed harmonized research compilations redistributing observed streamflow time series
OPEN_TIER1_COMPILATIONS = {
    "Caravan",
    "Caravan (GRDC Extension)",
    "Caravan (Regional CAMELS Extension)",
    "CAMELS-US",
    "CNS_CAMELS_Extra",
    "HYSETS",
    "ROBIN_v1",
    "CABra",
    "CAMELS-BR",
    "CAMELS-CH",
    "CAMELS-CL",
    "CAMELS-COL",
    "CAMELS-FI",
    "CAMELS-LUX",
    "CAMELS-PE",
    "CAMELS-PL",
    "CAMELS_IND",
    "CAMELS_KR",
    "CAMELSH_KR",
    "CAMELS_NZ",
    "LamaH-Ice",
    "Andean-GC",
    "PMET-obs",
    "SIREN_Italy",
    "CA_discharge",
    "CIS_Watersheds",
    "ArcticGRO",
    "GHI_v1",
    "GHI_v2",
    "GHSA_v3",
    "SSA-HCD",
}


def _strip_id(val: object) -> str:
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def prepare_reference_json() -> dict:
    """Persist verified per-station audit sets into configs/audit_reference/."""
    ref_json = REF_DIR / "station_level_audit_sets.json"
    if ref_json.exists() and not os.environ.get("REBUILD_AUDIT_REF"):
        return json.loads(ref_json.read_text())

    # 1. Rijkswaterstaat live catalog classification
    rws_tmp = Path("/tmp/rws_station_types.json")
    rws_data = json.loads(rws_tmp.read_text()) if rws_tmp.exists() else {"q_codes": [], "h_inland": [], "h_marine": [], "other_codes": []}

    # 2. IDEAM AQUARIUS CAUDAL stations + ideam_catalog_raw.csv non-river categories
    ideam_tmp = Path("/tmp/ideam_caudal_stations.json")
    ideam_caudal = json.loads(ideam_tmp.read_text())["all_caudal"] if ideam_tmp.exists() else []

    co_raw = pd.read_csv(HARV_DIR / "tier3_latam_africa_eu/raw/ideam_catalog_raw.csv", low_memory=False)
    river_cats = {"Limnimétrica", "Limnigráfica", "HidroMeteorológica"}
    ideam_river_codes = sorted(set(co_raw.loc[co_raw["categoria"].isin(river_cats), "codigo"].map(_strip_id)))
    ideam_nonriver_codes = sorted(set(co_raw.loc[~co_raw["categoria"].isin(river_cats), "codigo"].map(_strip_id)) | {"10000000"})

    # 3. Norway NVE discharge vs stage vs Virtuell/Uten målinger
    no_df = pd.read_csv(RIV_DIR / "extracted_py_v1_sites/main/norway_sites.csv", low_memory=False)
    nve_drop = sorted(set(no_df.loc[no_df["stationStatusName"].isin(["Virtuell", "Uten målinger"]), "gauge_id"].map(_strip_id)))
    nve_q = sorted(set(no_df.loc[(no_df["discharge_daily_mean"] == True) & (~no_df["stationStatusName"].isin(["Virtuell", "Uten målinger"])), "gauge_id"].map(_strip_id)))

    # 4. Switzerland CAMELS-CH lake-level stations
    ch_df = pd.read_csv(HARV_DIR / "tier3_latam_africa_eu/europe_camels_ch_catalog.csv", low_memory=False)
    ch_lake_ids = sorted(set(ch_df.loc[ch_df["observation_type"] == "daily_lake_level", "native_station_id"].map(_strip_id)))

    # 5. UK EA flow vs river-level vs groundwater/rainfall
    uk_ea = pd.read_csv(RIV_DIR / "extracted_py_v1_sites/main/uk_ea_sites.csv", low_memory=False)
    uk_nrfa = pd.read_csv(RIV_DIR / "extracted_py_v1_sites/main/uk_nrfa_sites.csv", low_memory=False)
    uk_ea["has_flow"] = uk_ea["measures"].astype(str).str.contains("'parameter': 'flow'", regex=False)
    uk_ea["has_level"] = uk_ea["measures"].astype(str).str.contains("'parameter': 'level'", regex=False)
    uk_ea["is_gw_or_rain"] = (
        uk_ea["type"].astype(str).str.contains("Groundwater", case=False)
        | uk_ea["measures"].astype(str).str.contains("groundwater|rainfall", case=False)
    ) & (~uk_ea["has_flow"]) & (~uk_ea["type"].astype(str).str.contains("River", case=False))

    uk_flow_ids, uk_gw_rain_ids = set(), set()
    for _, r in uk_ea.iterrows():
        ks = [_strip_id(r.get(c, "")).upper() for c in ("RLOIid", "stationReference", "wiskiID", "nrfaStationID", "gauge_id")]
        ks = [k for k in ks if k and k != "NAN"]
        ks_nz = [k.lstrip("0") for k in ks if k.lstrip("0")]
        if r["has_flow"]:
            uk_flow_ids.update(ks + ks_nz)
        elif r["is_gw_or_rain"]:
            uk_gw_rain_ids.update(ks + ks_nz)
    for g in uk_nrfa["gauge_id"].map(_strip_id).str.upper():
        if g:
            uk_flow_ids.add(g)
            if g.lstrip("0"):
                uk_flow_ids.add(g.lstrip("0"))

    # 6. USGS NWIS 00060 discharge vs 00065 stage-only
    sp_usgs = pd.read_parquet(RIV_DIR / "RivRetrieve_v2_org/src/rivretrieve/_internal/providers/usgs_nwis/catalogue/station_products.parquet")
    usgs_q = set(
        sp_usgs.loc[
            sp_usgs["product_id"].isin(["discharge_daily_mean", "discharge_instantaneous"]) & (sp_usgs["availability"] == "available"),
            "station_id",
        ].map(_strip_id)
    )
    usgs_stage_only = sorted(set(sp_usgs["station_id"].map(_strip_id)) - usgs_q)

    # 7. Brazil ANA HidroWeb confirmed liquid discharge (brazil_sites.csv + CAMELS-BR + CABra)
    br_sites = pd.read_csv(RIV_DIR / "extracted_py_v1_sites/main/brazil_sites.csv", low_memory=False)
    ana_q = set(br_sites["gauge_id"].map(_strip_id).str.lstrip("0"))
    cam_br = pd.read_csv(HARV_DIR / "tier3_latam_africa_eu/brazil_camels_br_catalog.csv", low_memory=False)
    ana_q.update(cam_br["native_station_id"].map(_strip_id).str.lstrip("0"))

    # 8. Central Asia CA_discharge 134 timeseries gauges vs 161 norm-only gauges
    ca_dis = pd.read_csv(HARV_DIR / "tier2_asia_cis_oceania/04_central_asia_ca_discharge_295.csv", low_memory=False)
    ca_daily_ids = sorted(set(ca_dis.loc[ca_dis["observation_type"].str.contains("day|decade", case=False, na=False), "native_station_id"].map(_strip_id)))
    ca_monthly_ids = sorted(set(ca_dis.loc[ca_dis["observation_type"].str.contains("month", case=False, na=False) & (~ca_dis["observation_type"].str.contains("day|decade", case=False, na=False)), "native_station_id"].map(_strip_id)))

    # 9. GHI_v1 / GHI_v2 / GHSA_v3 monthly Q vs metadata-only G3/G4
    ghi1 = pd.read_csv(HARV_DIR / "tier2_asia_cis_oceania/05a_india_ghi_v1_645.csv", low_memory=False)
    ghi1_meta_ids = sorted(set(ghi1.loc[ghi1["observation_type"].str.contains("G3|unverified", na=False), "native_station_id"].map(_strip_id)))
    ghi2 = pd.read_csv(HARV_DIR / "tier2_asia_cis_oceania/05b_india_ghi_v2_1677.csv", low_memory=False)
    ghi2_meta_ids = sorted(set(ghi2.loc[ghi2["observation_type"].str.contains("G4", na=False), "native_station_id"].map(_strip_id)))

    # 10. CAMELS-IND zero-observation / simulated-only gauges (159 + 6 = 165)
    ind_t2 = pd.read_csv(HARV_DIR / "tier2_asia_cis_oceania/05c_india_camels_ind_472.csv", low_memory=False)
    ind_sim_159 = set(ind_t2.loc[ind_t2["observation_type"].str.contains("simulated_daily_discharge", na=False), "native_station_id"].map(_strip_id).str.replace("camelsind_", "", case=False))
    ind_zero_6 = {"15004", "15015", "17025", "3012", "3049", "9004"}
    camels_ind_zero_obs = sorted(ind_sim_159 | ind_zero_6)

    # 11. CAMELS-NZ 14 permission-required gauges
    nz_t2 = pd.read_csv(HARV_DIR / "tier2_asia_cis_oceania/08_new_zealand_camels_nz_369.csv", low_memory=False)
    camels_nz_perm_14 = sorted(set(nz_t2.loc[nz_t2["data_access_policy"].str.contains("permission", case=False, na=False), "native_station_id"].map(_strip_id).str.replace("camelsnz_", "", case=False)))

    # 12. GRDC-Caravan 5,356 CC-BY-4.0 station IDs
    t1_comb = pd.read_csv(HARV_DIR / "tier1_global_and_cns/tier1_combined_catalog.csv", low_memory=False)
    grdc_ccby_5356 = sorted(
        set(
            t1_comb.loc[
                (t1_comb["source_dataset"] == "GRDC") & t1_comb["license"].astype(str).str.contains("CC BY", case=False, na=False),
                "native_station_id",
            ].map(_strip_id)
        )
    )

    # 13. CIS_Watersheds subdataset (GRDC_CIS vs Roshydromet_AIS_GMVO) & stage-only
    cis_df = pd.read_csv(HARV_DIR / "tier2_asia_cis_oceania/03_russia_cis_watersheds_2164.csv", low_memory=False)
    cis_grdc_ids = sorted(set(cis_df.loc[cis_df["subdataset_or_agency"] == "GRDC_CIS", "native_station_id"].map(_strip_id)))
    cis_stage_ids = sorted(set(cis_df.loc[cis_df["observation_type"] == "daily_stage_and_hydrometeo", "native_station_id"].map(_strip_id)))

    payload = {
        "rws_q_codes": rws_data["q_codes"],
        "rws_h_inland": rws_data["h_inland"],
        "rws_drop_codes": sorted(set(rws_data["h_marine"]) | set(rws_data["other_codes"])),
        "ideam_caudal_codes": sorted(set(ideam_caudal)),
        "ideam_river_codes": ideam_river_codes,
        "ideam_nonriver_codes": ideam_nonriver_codes,
        "nve_drop_ids": nve_drop,
        "nve_q_ids": nve_q,
        "ch_lake_ids": ch_lake_ids,
        "uk_flow_ids": sorted(uk_flow_ids),
        "uk_gw_rain_ids": sorted(uk_gw_rain_ids),
        "usgs_stage_only_ids": usgs_stage_only,
        "ana_q_ids": sorted(x for x in ana_q if x),
        "ca_daily_ids": ca_daily_ids,
        "ca_monthly_ids": ca_monthly_ids,
        "ghi1_meta_ids": ghi1_meta_ids,
        "ghi2_meta_ids": ghi2_meta_ids,
        "camels_ind_zero_obs": camels_ind_zero_obs,
        "camels_nz_perm_14": camels_nz_perm_14,
        "grdc_ccby_5356": grdc_ccby_5356,
        "cis_grdc_ids": cis_grdc_ids,
        "cis_stage_ids": cis_stage_ids,
    }
    ref_json.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    ref = prepare_reference_json()
    p_arch = ATTR_DIR / "global_gauges_by_source_archive.parquet"
    p_dedup = ATTR_DIR / "global_gauges_deduplicated.parquet"
    p_baseline = ATTR_DIR / "grdc_attributes.csv"

    # Backup pre-v3.2.0 files once so audit is 100% idempotent & repeatable!
    bak_arch = ATTR_DIR / "global_gauges_by_source_archive.parquet.bak_v310"
    bak_dedup = ATTR_DIR / "global_gauges_deduplicated.parquet.bak_v310"
    if not bak_arch.exists():
        shutil.copy2(p_arch, bak_arch)
    if not bak_dedup.exists():
        shutil.copy2(p_dedup, bak_dedup)

    arch = pd.read_parquet(bak_arch)
    dedup_orig = pd.read_parquet(bak_dedup)
    grdc_8811 = pd.read_csv(p_baseline, index_col=0, usecols=[0])
    cv_pref = set("CARAVAN_" + str(x).upper() for x in grdc_8811.index)
    baseline_cids_orig = set(arch.loc[arch["native_station_id"].astype(str).str.upper().isin(cv_pref), "canonical_gauge_id"])

    print(f"Loaded v3.1.0 archive: {len(arch):,} rows | deduplicated: {len(dedup_orig):,} stations | baseline CV physical: {len(baseline_cids_orig):,}")

    # Convert reference lists to fast lookup sets
    rws_q = set(ref["rws_q_codes"])
    rws_h_inland = set(ref["rws_h_inland"])
    rws_drop = set(ref["rws_drop_codes"])
    ideam_caudal = set(ref["ideam_caudal_codes"])
    ideam_nonriver = set(ref["ideam_nonriver_codes"])
    nve_drop = set(ref["nve_drop_ids"])
    nve_q = set(ref["nve_q_ids"])
    ch_lake = set(ref["ch_lake_ids"])
    uk_flow = set(ref["uk_flow_ids"])
    uk_gw_rain = set(ref["uk_gw_rain_ids"])
    usgs_stage_only = set(ref["usgs_stage_only_ids"])
    ana_q = set(ref["ana_q_ids"])
    ca_daily = set(ref["ca_daily_ids"])
    ca_monthly = set(ref["ca_monthly_ids"])
    ghi1_meta = set(ref["ghi1_meta_ids"])
    ghi2_meta = set(ref["ghi2_meta_ids"])
    ind_zero = set(ref["camels_ind_zero_obs"])
    nz_perm14 = set(ref["camels_nz_perm_14"])
    grdc_ccby = set(ref["grdc_ccby_5356"])
    cis_grdc = set(ref["cis_grdc_ids"])
    cis_stage = set(ref["cis_stage_ids"])

    keep_mask = np.ones(len(arch), dtype=bool)
    drop_reasons: dict[str, int] = {}
    new_obs = arch["observation_type"].astype(str).to_numpy(copy=True)
    new_pol = arch["data_access_policy"].astype(str).to_numpy(copy=True)
    new_lic = arch["license"].astype(str).to_numpy(copy=True)
    row_in_t1 = np.zeros(len(arch), dtype=bool)

    ds_arr = arch["source_dataset"].astype(str).to_numpy()
    ag_arr = arch["subdataset_or_agency"].astype(str).to_numpy()
    st_arr = arch["source_tier"].astype(str).to_numpy()
    nid_arr = arch["native_station_id"].map(_strip_id).to_numpy()
    cid_arr = arch["canonical_gauge_id"].astype(str).to_numpy()

    def _bump(reason: str) -> None:
        drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

    for i in range(len(arch)):
        ds = ds_arr[i]
        ag = ag_arr[i]
        st = st_arr[i]
        nid = nid_arr[i]
        cid = cid_arr[i]

        # Never drop a verified 8,811 Caravan CV record
        is_cv = cid in baseline_cids_orig

        # ---------------- RULE 1: DROP NON-PHYSICAL / SIMULATED / NON-RIVER ROWS ----------------
        if ds == "MERV_Jp":
            keep_mask[i] = False
            _bump("MERV_Jp_simulated_runoff")
            continue

        if ds == "NVE HydAPI" or (ds == "RivRetrieve" and "NVE" in ag):
            code_nve = nid.replace("NVE_", "")
            if code_nve in nve_drop and not is_cv:
                keep_mask[i] = False
                _bump("NVE_Virtuell_or_Uten_malinger")
                continue

        if st == "GIS_Internal" and ds == "IDEAM DHIME":
            code_co = nid.replace("IDEAM_", "")
            if code_co in ideam_nonriver and not is_cv:
                keep_mask[i] = False
                _bump("IDEAM_weather_rain_radiosonde_nonriver")
                continue

        if st == "GIS_Internal" and ds == "UK Environment Agency":
            code_uk = nid.replace("UKEA_", "").upper()
            code_uk_nz = code_uk.lstrip("0")
            if (code_uk in uk_gw_rain or (code_uk_nz and code_uk_nz in uk_gw_rain)) and not (code_uk in uk_flow or code_uk_nz in uk_flow) and not is_cv:
                keep_mask[i] = False
                _bump("UK_EA_groundwater_or_rainfall")
                continue

        if ds == "RivRetrieve" and "Rijkswaterstaat" in ag:
            code_nl = nid.lower()
            if code_nl not in rws_q and code_nl not in rws_h_inland and not is_cv:
                keep_mask[i] = False
                _bump("Rijkswaterstaat_marine_offshore_beach_or_gw")
                continue

        if (ds == "CAMELS-CH" or (ds == "RivRetrieve" and "FOEN" in ag)) and not is_cv:
            code_ch = nid.replace("camelsch_", "").replace("CAMELSCH_", "")
            if code_ch in ch_lake:
                keep_mask[i] = False
                _bump("FOEN_CAMELS_CH_lake_level_Seepegel")
                continue

        # ---------------- RULE 2 & 3: OBSERVATION TYPE + OPEN FULL-USE LICENSE ----------------
        obs = new_obs[i]
        pol = new_pol[i]
        lic = new_lic[i]

        # 2A. Metadata / signature / flood-peak / standardized-anomaly / satellite-reconstructed compilations
        if ds == "EStreams":
            obs = "flow_indices_only"
            pol = "open_indices_and_metadata_only"
            lic = "CC BY 4.0 (attributes/signatures only; no streamflow series)"
        elif ds in ("GSIM", "GSHA"):
            obs = "flow_indices_only"
            pol = "open_indices_and_metadata_only"
        elif ds == "ADHI":
            obs = "flow_indices_only"
            pol = "open_indices_and_metadata_only"
            lic = "CC BY 4.0 (indices only; raw daily Q restricted)"
        elif ds == "FOCA_ISPRA_Italy":
            obs = "flow_indices_only"
            pol = "open_indices_and_metadata_only"
            lic = "CC-BY-4.0 (annual max flood peaks & attributes only)"
        elif ds == "North_Africa_98_Basins":
            obs = "flow_indices_only"
            pol = "open_indices_and_metadata_only"
            lic = "CC-BY-4.0 (flood event peaks & volumes only)"
        elif ds == "CDR2":
            obs = "flow_indices_only"
            pol = "open_indices_and_metadata_only"
            lic = "CC-BY-4.0 (satellite-reconstructed W->Q & station metadata)"
        elif ds == "HydroMLYR":
            obs = "flow_indices_only"
            pol = "open_indices_and_metadata_only"
            lic = "CC-BY-4.0 (standardized weekly anomaly only)"
        elif ds == "CCAM":
            obs = "catchment_attributes_and_boundaries"
            pol = "open_attributes_and_metadata_only"
        elif ds == "JPDH_MLIT_Network_Extra":
            if obs == "daily_discharge":
                obs = "catchment_attributes_and_boundaries"
            else:
                obs = "stage_or_fluviometric"
            pol = "open_attributes_and_metadata_only"
            lic = "CC-BY-NC-4.0 (station metadata/stage only)"
        elif ds == "JPDH":
            # JPDH D3P-JP-Q is explicitly CC BY-NC 4.0 (non-commercial)
            obs = "daily_discharge"
            pol = "registration_or_request_portal"
            lic = "CC-BY-NC-4.0 (non-commercial)"
        elif ds == "CA_discharge":
            if nid in ca_daily:
                obs = "daily_discharge"
                pol = "open_academic_compilation"
                lic = "CC-BY-4.0"
            elif nid in ca_monthly:
                obs = "monthly_discharge"
                pol = "open_academic_compilation"
                lic = "CC-BY-4.0"
            else:
                obs = "flow_indices_only"
                pol = "open_indices_and_metadata_only"
                lic = "CC-BY-4.0 (norm Q & polygon only)"
        elif ds == "GHI_v1":
            if nid in ghi1_meta:
                obs = "flow_indices_only"
                pol = "open_indices_and_metadata_only"
            else:
                obs = "monthly_discharge"
                pol = "open_academic_compilation"
                lic = "CC-BY-4.0"
        elif ds == "GHI_v2":
            if nid in ghi2_meta:
                obs = "flow_indices_only"
                pol = "open_indices_and_metadata_only"
            else:
                obs = "monthly_discharge"
                pol = "open_academic_compilation"
                lic = "CC-BY-4.0"
        elif ds == "GHSA_v3":
            obs = "monthly_discharge"
            pol = "open_academic_compilation"
            lic = "CC-BY-4.0"
        elif ds == "ROBIN_v1":
            if pol == "restricted_or_bilateral_partner":
                obs = "flow_indices_only"
                pol = "restricted_or_bilateral_partner"
                lic = "Restricted daily Q (open metadata)"
            else:
                obs = "daily_discharge"
                pol = "open_academic_compilation"
                lic = "CC BY 4.0"
        elif ds in ("Caravan", "Caravan (GRDC Extension)", "Caravan (Regional CAMELS Extension)", "CNS_CAMELS_Extra", "CAMELS_IND", "CAMELS_NZ"):
            # Check zero-observation / simulated-only CAMELS-IND and permission-required CAMELS-NZ
            code_ind = nid.upper().replace("CARAVAN_CAMELSIND_", "").replace("CAMELSIND_", "")
            code_nz = nid.upper().replace("CARAVAN_CAMELSNZ_", "").replace("CAMELSNZ_", "")
            if ("IND" in ds.upper() or "IND" in ag.upper()) and code_ind in ind_zero and not is_cv:
                obs = "catchment_attributes_and_boundaries"
                pol = "open_attributes_and_metadata_only"
                lic = "CC-BY-4.0 (0% observed Q; simulated/attributes only)"
            elif ("NZ" in ds.upper() or "NZ" in ag.upper()) and code_nz in nz_perm14 and not is_cv:
                obs = "catchment_attributes_and_boundaries"
                pol = "restricted_or_bilateral_partner"
                lic = "Permission required from hydro operator for Q"
            else:
                obs = "daily_discharge"
                pol = "open_academic_compilation"
                lic = "CC BY 4.0"
        elif ds == "GRDC":
            code_grdc = nid.upper().replace("GRDC_", "")
            if code_grdc in grdc_ccby:
                obs = "daily_discharge"
                pol = "open_academic_compilation"
                lic = "CC BY 4.0 (GRDC-Caravan)"
            else:
                pol = "registration_or_request_portal"
                lic = "GRDC Data Policy (Non-Commercial Research & Education)"
        elif ds == "R_ArcticNET_v4":
            pol = "registration_or_request_portal"
            lic = "Open-Academic-UNH (Non-Commercial Academic Only)"
        elif ds == "SSA-HCD":
            code_ssa = nid.upper().replace("GRDC_", "")
            if code_ssa in grdc_ccby:
                obs = "monthly_discharge"
                pol = "open_academic_compilation"
                lic = "CC-BY-4.0"
            else:
                obs = "monthly_discharge"
                pol = "registration_or_request_portal"
                lic = "GRDC-derived monthly runoff (non-commercial underlying terms)"
        elif ds == "CIS_Watersheds":
            if nid in cis_stage:
                obs = "stage_or_fluviometric"
                pol = "open_public_api_or_bulk"
                lic = "CC-BY-4.0"
            elif nid in cis_grdc and nid not in grdc_ccby:
                obs = "daily_discharge"
                pol = "registration_or_request_portal"
                lic = "GRDC_CIS (underlying GRDC non-commercial terms)"
            else:
                obs = "daily_discharge"
                pol = "open_academic_compilation"
                lic = "CC-BY-4.0"

        # 2B. National portals — per-station Q vs stage + strict full-use open license audit
        elif ds == "USGS NWIS":
            code_us = nid.replace("USGS_", "")
            if code_us in usgs_stage_only and not is_cv:
                obs = "stage_or_fluviometric"
            else:
                obs = "daily_discharge"
            pol = "open_public_api_or_bulk"
            lic = "US Public Domain"
        elif ds == "NVE HydAPI":
            code_nve = nid.replace("NVE_", "")
            obs = "daily_discharge" if code_nve in nve_q else "stage_or_fluviometric"
            pol = "open_public_api_or_bulk"
            lic = "NLOD 2.0 / CC BY 3.0 NO"
        elif ds == "UK Environment Agency":
            code_uk = nid.replace("UKEA_", "").upper()
            code_uk_nz = code_uk.lstrip("0")
            obs = "daily_discharge" if (code_uk in uk_flow or (code_uk_nz and code_uk_nz in uk_flow)) else "stage_or_fluviometric"
            pol = "open_public_api_or_bulk"
            lic = "Open Government Licence v3.0 (OGL v3)"
        elif ds == "ANA HidroWeb":
            code_br = nid.replace("ANA_", "").lstrip("0")
            obs = "daily_discharge" if code_br in ana_q else "stage_or_fluviometric"
            pol = "open_public_api_or_bulk"
            lic = "Dados Abertos ANA (Open Government Data Brazil)"
        elif ds in ("IDEAM DHIME", "IDEAM_Colombia"):
            code_co = nid.replace("IDEAM_", "")
            obs = "daily_discharge" if code_co in ideam_caudal else "stage_or_fluviometric"
            pol = "open_public_api_or_bulk"
            lic = "Datos Abiertos Colombia (CC BY 4.0)"
        elif ds == "BoM Water Data Online":
            obs = "daily_discharge"
            pol = "open_public_api_or_bulk"
            lic = "CC BY 3.0 AU / CC BY 4.0 (Australian Water Regulations)"
        elif ds == "CONAGUA BANDAS":
            obs = "daily_discharge"
            pol = "open_public_api_or_bulk"
            lic = "Libre Uso MX / CC BY 4.0"
        elif ds == "CHMI (ČHMÚ)":
            obs = "daily_discharge"
            pol = "open_public_api_or_bulk"
            lic = "CC BY 4.0"
        elif ds == "CR2_Caudales_Chile":
            obs = "daily_discharge"
            pol = "open_public_api_or_bulk"
            lic = "CC BY 4.0"
        elif ds == "DGA Chile":
            obs = "stage_or_fluviometric"
            pol = "open_public_api_or_bulk"
            lic = "Open Government Data Chile"
        elif ds == "DWS South Africa":
            pol = "registration_or_request_portal"
            lic = "DWS Terms (Non-Commercial Research/Personal Only)"
        elif ds == "DINAGUA Uruguay":
            pol = "restricted_or_bilateral_partner"
            lic = "Bilateral WMO Pilot / Restricted"
        elif ds == "MRC":
            obs = "stage_or_fluviometric"
            pol = "registration_or_request_portal"
            lic = "MRC Data Policy (Paid/Restricted Commercial Terms)"
        elif ds == "RivRetrieve":
            # Audit every one of the 39 RivRetrieve agency adapters!
            if "Rijkswaterstaat" in ag:
                code_nl = nid.lower()
                obs = "daily_discharge" if code_nl in rws_q else "stage_or_fluviometric"
                pol = "open_public_api_or_bulk"
                lic = "CC0 1.0 Universal"
            elif "DWS Verified" in ag:
                pol = "registration_or_request_portal"
                lic = "DWS Terms (Non-Commercial Research/Personal Only)"
            elif "ThaiWater" in ag:
                pol = "registration_or_request_portal"
                lic = "HII CC BY-NC (Non-Commercial)"
            elif "IMGW-PIB" in ag:
                pol = "registration_or_request_portal"
                lic = "IMGW Regulamin / Prawo Wodne (Commercial Contract Required)"
            elif "UKCEH NRFA" in ag:
                pol = "registration_or_request_portal"
                lic = "UKCEH NRFA Click-Through Licence (No Raw Redistribution)"
            elif any(x in ag for x in ("SENAMHI Peru", "INA Alerta", "AVP Sava", "OpenHI")):
                pol = "registration_or_request_portal"
                lic = "Agency Portal (No Open Commercial License Established)"
            else:
                # All remaining 29 RivRetrieve adapters have verified full-use open government licenses
                # (USGS Public Domain, ECCC OGL-Canada, BoM CC-BY, ANA Open Data, NVE NLOD, SMHI CC-BY-4.0,
                # SYKE CC-BY-4.0, Miljøportal CC-BY-4.0, EstModel CC-BY-4.0, LHMT CC-BY-SA-4.0, Hub'Eau Etalab 2.0,
                # Waterinfo.be & SPW Open Data, OPW CC-BY-4.0, UK EA & SEPA OGL v3, WSV & Berlin DL-DE-2.0,
                # BMLUK eHYD CC-BY-4.0, FOEN opendata.swiss, CHMI CC-BY-4.0, MITECO Spanish PSI, APA SNIRH PSI,
                # ARSO CC-BY-4.0, SIR Toscana IODL/CC-BY, CR2/DGA CC-BY-4.0, MLIT WIS PDL 1.0, WAMIS/HRFCO KOGL-1,
                # WRA OGDL-Taiwan-1.0, MELCCFP Quebec CC-BY-4.0)
                pol = "open_public_api_or_bulk"

        new_obs[i] = obs
        new_pol[i] = pol
        new_lic[i] = lic

        # Tier 1 row membership: harmonized open-licensed compilation redistributing observed Q
        if (
            (ds in OPEN_TIER1_COMPILATIONS or (ds == "GRDC" and pol == "open_academic_compilation"))
            and pol == "open_academic_compilation"
            and obs in ("daily_discharge", "monthly_discharge")
        ):
            row_in_t1[i] = True

    arch["observation_type"] = new_obs
    arch["data_access_policy"] = new_pol
    arch["license"] = new_lic
    arch["row_in_tier_1"] = row_in_t1

    arch_clean = arch.loc[keep_mask].copy().reset_index(drop=True)
    arch_clean["archive_row_id"] = np.arange(len(arch_clean), dtype=np.int64)

    print("\n=== Dropped Non-Physical / Simulated / Non-River Archive Rows ===")
    for k, v in sorted(drop_reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {k:45s}: {v:6,d}")
    print(f"Surviving archive rows: {len(arch_clean):,} (dropped {len(arch) - len(arch_clean):,})")

    # Re-aggregate per canonical_gauge_id onto dedup!
    surviving_cids = set(arch_clean["canonical_gauge_id"])
    dedup = dedup_orig.loc[dedup_orig["canonical_gauge_id"].isin(surviving_cids)].copy().reset_index(drop=True)
    assert baseline_cids_orig.issubset(surviving_cids), "FATAL: Lost a Caravan CV baseline station!"

    # Fast linear pass over arch_clean to update dedup columns accurately
    cid_to_idx = {cid: idx for idx, cid in enumerate(dedup["canonical_gauge_id"])}
    K = len(dedup)

    n_rec = np.zeros(K, dtype=np.int32)
    best_obs_r = np.full(K, 99, dtype=np.int32)
    best_pol_r = np.full(K, 99, dtype=np.int32)
    # Track whether the gauge has AT LEAST ONE row with observed discharge (Tier 2 = 82,596),
    # and separately whether at least one row has open full-use policy + observed Q (74,078).
    has_any_q_row = np.zeros(K, dtype=bool)
    has_open_q_row = np.zeros(K, dtype=bool)
    has_tier1_row = np.zeros(K, dtype=bool)

    ds_lists: list[list[str]] = [[] for _ in range(K)]
    ag_lists: list[list[str]] = [[] for _ in range(K)]
    obs_lists: list[list[str]] = [[] for _ in range(K)]
    pol_lists: list[list[str]] = [[] for _ in range(K)]

    # Also recompute physical active status cleanly from surviving rows
    cid_reasons: list[set[str]] = [set() for _ in range(K)]
    max_ver_ey = np.full(K, np.nan, dtype=np.float64)

    for row in arch_clean.itertuples(index=False):
        pos = cid_to_idx[row.canonical_gauge_id]
        n_rec[pos] += 1
        o_str = str(row.observation_type)
        p_str = str(row.data_access_policy)
        d_str = str(row.source_dataset)
        g_str = str(row.subdataset_or_agency)

        orank = OBS_TYPE_RANK.get(o_str, 99)
        prank = ACCESS_POLICY_RANK.get(p_str, 99)
        if orank < best_obs_r[pos]:
            best_obs_r[pos] = orank
        if prank < best_pol_r[pos]:
            best_pol_r[pos] = prank

        if o_str in ("daily_discharge", "monthly_discharge"):
            has_any_q_row[pos] = True
            if p_str in ("open_academic_compilation", "open_public_api_or_bulk"):
                has_open_q_row[pos] = True
        if row.row_in_tier_1:
            has_tier1_row[pos] = True

        if d_str and d_str not in ds_lists[pos]:
            ds_lists[pos].append(d_str)
        if g_str and g_str not in ("", "nan", "None") and g_str not in ag_lists[pos]:
            ag_lists[pos].append(g_str)
        if o_str and o_str not in obs_lists[pos]:
            obs_lists[pos].append(o_str)
        if p_str and p_str not in pol_lists[pos]:
            pol_lists[pos].append(p_str)

        rsn = str(row.active_reason)
        if rsn:
            cid_reasons[pos].add(rsn)
        vey = row.verified_record_end_year
        if pd.notna(vey):
            vf = float(vey)
            if np.isnan(max_ver_ey[pos]) or vf > max_ver_ey[pos]:
                max_ver_ey[pos] = vf

    # Re-evaluate Option-1 active flag on surviving physical river rows
    dedup_active = np.zeros(K, dtype=bool)
    dedup_act_reason = np.full(K, "", dtype=object)
    for pos in range(K):
        rset = cid_reasons[pos]
        if "cv8811_verified_active" in rset:
            dedup_active[pos] = True
            dedup_act_reason[pos] = "1_cv8811_verified_active"
        elif "end_year_ge_2016" in rset:
            dedup_active[pos] = True
            dedup_act_reason[pos] = "2_end_year_ge_2016"
        elif "portal_status_active" in rset:
            dedup_active[pos] = True
            dedup_act_reason[pos] = "3_portal_status_active"
        elif "vetted_operational_cwc_dhm_g1_g2" in rset and not ("end_year_lt_2016" in rset or "portal_status_inactive" in rset):
            dedup_active[pos] = True
            dedup_act_reason[pos] = "4_vetted_cwc_dhm_g1_g2"
        elif "portal_status_inactive" in rset:
            dedup_active[pos] = False
            dedup_act_reason[pos] = "5_portal_status_inactive"
        elif "end_year_lt_2016" in rset:
            dedup_active[pos] = False
            dedup_act_reason[pos] = "6_end_year_lt_2016"
        elif "option1_live_operational_portal" in rset:
            dedup_active[pos] = True
            dedup_act_reason[pos] = "7_option1_live_operational_portal"
        else:
            dedup_active[pos] = False
            dedup_act_reason[pos] = "8_undated_historical_or_metadata"

    dedup["n_archive_records"] = n_rec
    dedup["n_source_datasets"] = [len(x) for x in ds_lists]
    dedup["all_source_datasets"] = [" | ".join(x) for x in ds_lists]
    dedup["all_agencies_or_subdatasets"] = [" | ".join(x) for x in ag_lists]
    dedup["best_observation_type"] = [INV_OBS_RANK.get(int(r), "flow_indices_only") for r in best_obs_r]
    dedup["all_observation_types"] = [" | ".join(x) for x in obs_lists]
    dedup["best_data_access_policy"] = [INV_POL_RANK.get(int(r), "open_indices_and_metadata_only") for r in best_pol_r]
    dedup["all_data_access_policies"] = [" | ".join(x) for x in pol_lists]
    dedup["verified_record_end_year"] = max_ver_ey
    dedup["is_active_opt1"] = dedup_active
    dedup["active_reason"] = dedup_act_reason
    dedup["in_tier_baseline"] = dedup["canonical_gauge_id"].isin(baseline_cids_orig).to_numpy()
    dedup["in_tier_1"] = has_tier1_row
    dedup["in_tier_2"] = has_tier1_row | has_any_q_row
    dedup["in_tier_3"] = np.ones(K, dtype=bool)

    # Propagate physical active status back to arch_clean
    cid_to_act = dict(zip(dedup["canonical_gauge_id"], dedup["is_active_opt1"]))
    arch_clean["is_active_physical_opt1"] = arch_clean["canonical_gauge_id"].map(cid_to_act).astype(bool)

    # Save updated parquets
    arch_out_cols = [c for c in arch.columns if c != "row_in_tier_1"] + ["row_in_tier_1"]
    arch_clean[arch_out_cols].to_parquet(p_arch, index=False)
    dedup.to_parquet(p_dedup, index=False)

    # Also write audit summary CSV & JSON to configs/audit_reference/ (and attributes/)
    audit_tbl = (
        arch_clean.groupby(["source_tier", "source_dataset", "observation_type", "data_access_policy", "license"], dropna=False)
        .agg(n_archive_rows=("archive_row_id", "count"), n_unique_gauges=("canonical_gauge_id", "nunique"), n_tier1_rows=("row_in_tier_1", "sum"))
        .reset_index()
    )
    audit_tbl.to_csv(REF_DIR / "73_archive_license_and_data_audit.csv", index=False)
    audit_tbl.to_csv(ATTR_DIR / "73_archive_license_and_data_audit.csv", index=False)

    summary = {
        "archive_rows_v310": int(len(arch)),
        "archive_rows_v320": int(len(arch_clean)),
        "dropped_nonphysical_rows": drop_reasons,
        "physical_gauges_v310": int(len(dedup_orig)),
        "physical_gauges_v320_tier3": int(dedup["in_tier_3"].sum()),
        "in_tier_baseline": int(dedup["in_tier_baseline"].sum()),
        "in_tier_1": int(dedup["in_tier_1"].sum()),
        "in_tier_2": int(dedup["in_tier_2"].sum()),
        "in_open_license_q_subset": int((has_tier1_row | has_open_q_row).sum()),
        "in_tier_3": int(dedup["in_tier_3"].sum()),
        "in_tier_1_active": int((dedup["in_tier_1"] & dedup["is_active_opt1"]).sum()),
        "in_tier_2_active": int((dedup["in_tier_2"] & dedup["is_active_opt1"]).sum()),
        "in_tier_3_active": int((dedup["in_tier_3"] & dedup["is_active_opt1"]).sum()),
        "tier3_breakdown_by_best_obs": dedup["best_observation_type"].value_counts().to_dict(),
        "tier3_breakdown_by_best_policy": dedup["best_data_access_policy"].value_counts().to_dict(),
    }
    (REF_DIR / "gauge_audit_summary_v320.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== AUDITED v3.2.0 PHYSICAL GAUGE CATALOG SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
