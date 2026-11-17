"""Stage clean, self-contained gauge catalog, WorldPop population, and Natural Earth boundaries.

Extracts pure, hydrofabric-agnostic gauge metadata (coordinates, reported drainage areas,
country/continent, data-access tiers) with all HydroBASINS-specific snapped columns stripped
out so that every global hydrofabric snaps gauges symmetrically from scratch.

Stages public WorldPop 2020 population (`hydrosheds_l12_worldpop_population.parquet`) and
Natural Earth country boundaries (`naturalearth_lowres.geojson`) from `$GEO_BIAS_DATA_DIR`.
If optional licensed Fathom v3 basin exposure (`round2/exposure/global_basin_exposure_defended_r2.parquet`)
is present locally, also stages `hydrosheds_l12_fathom_exposure.parquet`; otherwise skips it cleanly.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.benchmark_config import (
    DATA_DIR,
    EXPOSURE_DIR,
    FATHOM_EXPOSURE_PARQUET,
    GAUGES_DIR,
    GEO_BIAS_DATA_DIR,
    WORLDPOP_POP_PARQUET,
)

GAUGES_OUT = GAUGES_DIR / "global_gauges.parquet"
NE_OUT = EXPOSURE_DIR / "naturalearth_lowres.geojson"

_SRC_GAUGES_PREBUILT = GEO_BIAS_DATA_DIR / "attributes" / "global_gauges.parquet"
_SRC_GAUGES = GEO_BIAS_DATA_DIR / "attributes" / "global_gauges_deduplicated.parquet"
_SRC_ARCHIVE = GEO_BIAS_DATA_DIR / "attributes" / "global_gauges_by_source_archive.parquet"
_SRC_BASELINE = GEO_BIAS_DATA_DIR / "attributes" / "grdc_attributes.csv"
_SRC_WORLDPOP = GEO_BIAS_DATA_DIR / "attributes" / "hydrosheds_l12_worldpop_population.parquet"
_SRC_FATHOM_OPTIONAL = GEO_BIAS_DATA_DIR / "round2" / "exposure" / "global_basin_exposure_defended_r2.parquet"
_SRC_NE = GEO_BIAS_DATA_DIR / "naturalearth_lowres.geojson"


def stage_gauges() -> dict:
    GAUGES_OUT.parent.mkdir(parents=True, exist_ok=True)
    if _SRC_GAUGES.exists() and _SRC_ARCHIVE.exists() and _SRC_BASELINE.exists():
        g = pd.read_parquet(_SRC_GAUGES)
        a = pd.read_parquet(_SRC_ARCHIVE, columns=["canonical_gauge_id", "source_dataset", "native_station_id"])
        grdc_8811 = pd.read_csv(_SRC_BASELINE, index_col=0, usecols=[0])

        idx_pref = set("CARAVAN_" + str(x).upper() for x in grdc_8811.index)
        baseline_cids = set(a[a["native_station_id"].astype(str).str.upper().isin(idx_pref)]["canonical_gauge_id"])

        if "in_tier_1" in g.columns and "in_tier_2" in g.columns:
            in_tier_1 = g["in_tier_1"].astype(bool).to_numpy()
            in_tier_2 = g["in_tier_2"].astype(bool).to_numpy()
        else:
            open_policies = {"open_public_api_or_bulk", "open_academic_compilation"}
            q_obs = {"daily_discharge", "monthly_discharge"}
            in_tier_1 = (
                (g["best_data_access_policy"] == "open_academic_compilation")
                & g["best_observation_type"].isin(q_obs)
            ).to_numpy()
            in_tier_2 = (
                g["best_data_access_policy"].isin(open_policies)
                & g["best_observation_type"].isin(q_obs)
            ).to_numpy()
        in_tier_3 = np.ones(len(g), dtype=bool)

        in_tier_baseline = g["canonical_gauge_id"].isin(baseline_cids).to_numpy()
        is_active = g["is_active_opt1"].astype(bool).to_numpy()

        keep_cols = [
            "canonical_gauge_id",
            "latitude",
            "longitude",
            "country",
            "continent",
            "station_name",
            "river_name",
            "drainage_area_km2",
            "primary_source_dataset",
            "all_source_datasets",
            "n_source_datasets",
            "best_observation_type",
            "best_data_access_policy",
            "record_start_year",
            "record_end_year",
            "record_length_years",
            "verified_record_end_year",
        ]
        clean = g[keep_cols].copy()
        clean["is_active"] = is_active
        clean["in_tier_baseline"] = in_tier_baseline
        clean["in_tier_1"] = in_tier_1
        clean["in_tier_2"] = in_tier_2
        clean["in_tier_3"] = in_tier_3
        clean["in_tier_1_active"] = clean["in_tier_1"] & clean["is_active"]
        clean["in_tier_2_active"] = clean["in_tier_2"] & clean["is_active"]
        clean["in_tier_3_active"] = clean["in_tier_3"] & clean["is_active"]
        clean.to_parquet(GAUGES_OUT, index=False)
    elif _SRC_GAUGES_PREBUILT.exists():
        shutil.copy2(_SRC_GAUGES_PREBUILT, GAUGES_OUT)
        clean = pd.read_parquet(GAUGES_OUT)
    else:
        raise FileNotFoundError(
            f"Could not locate gauge catalog under {GEO_BIAS_DATA_DIR / 'attributes'}. "
            "Download the Zenodo data archive and set GEO_BIAS_DATA_DIR."
        )

    assert int(clean["in_tier_3"].sum()) == 119_169
    assert int(clean["in_tier_2"].sum()) == 74_078
    assert int(clean["in_tier_1"].sum()) == 32_369
    assert 8_050 <= int(clean["in_tier_baseline"].sum()) <= 8_120

    summary = {
        "total_physical_gauges": int(len(clean)),
        "in_tier_baseline": int(clean["in_tier_baseline"].sum()),
        "in_tier_1": int(clean["in_tier_1"].sum()),
        "in_tier_2": int(clean["in_tier_2"].sum()),
        "in_tier_3": int(clean["in_tier_3"].sum()),
        "with_reported_drainage_area": int((clean["drainage_area_km2"] > 0).sum()),
        "active_total": int(clean["is_active"].sum()),
    }
    print("Staged clean gauges:", json.dumps(summary, indent=2), flush=True)
    return summary


def stage_population_and_boundaries() -> dict:
    EXPOSURE_DIR.mkdir(parents=True, exist_ok=True)
    if _SRC_WORLDPOP.exists():
        shutil.copy2(_SRC_WORLDPOP, WORLDPOP_POP_PARQUET)
        pop_df = pd.read_parquet(WORLDPOP_POP_PARQUET)
    elif _SRC_FATHOM_OPTIONAL.exists():
        exp = pd.read_parquet(_SRC_FATHOM_OPTIONAL).reset_index()
        idx_col = exp.columns[0]
        exp = exp.rename(columns={idx_col: "HYBAS_ID"})
        exp["HYBAS_ID"] = (
            exp["HYBAS_ID"].astype(str).str.replace("^hybas_", "", regex=True).astype(np.int64)
        )
        pop_df = exp[["HYBAS_ID", "total_population"]].copy()
        pop_df.to_parquet(WORLDPOP_POP_PARQUET, index=False)
    else:
        raise FileNotFoundError(
            f"Public WorldPop L12 population table not found at {_SRC_WORLDPOP}."
        )

    fathom_staged = False
    total_eae = None
    if _SRC_FATHOM_OPTIONAL.exists():
        exp = pd.read_parquet(_SRC_FATHOM_OPTIONAL).reset_index()
        idx_col = exp.columns[0]
        exp = exp.rename(columns={idx_col: "HYBAS_ID"})
        exp["HYBAS_ID"] = (
            exp["HYBAS_ID"].astype(str).str.replace("^hybas_", "", regex=True).astype(np.int64)
        )
        keep_f = [c for c in ["HYBAS_ID", "eae_gt10cm", "eae_gt0cm", "exposed_pop_gt10cm_10", "exposed_pop_gt10cm_100"] if c in exp.columns]
        if "eae_gt10cm" in keep_f:
            exp[keep_f].to_parquet(FATHOM_EXPOSURE_PARQUET, index=False)
            fathom_staged = True
            total_eae = float(exp["eae_gt10cm"].sum())
    else:
        print(
            "[info] Optional licensed Fathom v3 exposure file not present — running in public "
            "(non-Fathom) mode (land area, runoff, population, and 1D river discharge).",
            flush=True,
        )

    if _SRC_NE.exists():
        shutil.copy2(_SRC_NE, NE_OUT)

    summary = {
        "n_basins": int(len(pop_df)),
        "total_population": float(pop_df["total_population"].sum()),
        "optional_fathom_staged": fathom_staged,
        "naturalearth_copied": NE_OUT.exists(),
    }
    if total_eae is not None:
        summary["total_eae_gt10cm"] = total_eae
    print("Staged population & boundaries:", json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    s_g = stage_gauges()
    s_p = stage_population_and_boundaries()
    manifest_path = DATA_DIR / "MANIFEST.json"
    manifest = {"gauges": s_g, "population_and_boundaries": s_p}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path}", flush=True)
