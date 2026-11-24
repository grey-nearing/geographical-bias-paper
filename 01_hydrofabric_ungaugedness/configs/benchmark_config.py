"""Central configuration for the Global Hydrological Ungaugedness Benchmark."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = REPO_ROOT.parent
GEO_BIAS_DATA_DIR = Path(os.environ.get("GEO_BIAS_DATA_DIR", str(PROJECT_ROOT / "data")))
DATA_DIR = Path(os.environ.get("GLOBAL_UNGAUGEDNESS_DATA_DIR", str(REPO_ROOT / "data")))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GAUGES_DIR = DATA_DIR / "gauges"
EXPOSURE_DIR = DATA_DIR / "exposure"
WORLDPOP_POP_PARQUET = EXPOSURE_DIR / "hydrosheds_l12_worldpop_population.parquet"
FATHOM_EXPOSURE_PARQUET = EXPOSURE_DIR / "hydrosheds_l12_fathom_exposure.parquet"
RESULTS_DIR = Path(os.environ.get("GLOBAL_UNGAUGEDNESS_RESULTS_DIR", str(REPO_ROOT / "results")))
FIGURES_DIR = RESULTS_DIR / "figures"

# Four nested gauge data-accessibility tiers (physical deduplicated gauges)
GAUGE_TIERS = [
    ("baseline", "Baseline: Active Post-2016 CC-BY ML Training (8,075)", "in_tier_baseline"),
    ("tier1_open_compilations", "Tier 1: Standardized, Open-License Discharge Datasets (32,369)", "in_tier_1"),
    ("tier2_public_discharge", "Tier 2: Gauges with Discharge Records (82,596)", "in_tier_2"),
    ("tier3_all_known", "Tier 3: All Publicly Discoverable Stream Gauges (119,169)", "in_tier_3"),
]

# Active-only sensitivity subset (optional secondary evaluation)
ACTIVE_GAUGE_TIERS = [
    ("tier1_active", "Tier 1 Active (Standardized Open-License Discharge, active)", "in_tier_1_active"),
    ("tier2_active", "Tier 2 Active (Discharge Records, active)", "in_tier_2_active"),
    ("tier3_active", "Tier 3 Active (All Discoverable, active)", "in_tier_3_active"),
]

# Family 3: Scale-parameterized 2D inter-gauge area resolution grid (km^2)
# Includes np.inf (mathematically equal to U_never) as the upper asymptote.
AREA_SCALE_GRID_KM2 = [
    50.0,
    100.0,
    250.0,
    500.0,
    1_000.0,
    2_500.0,
    5_000.0,
    10_000.0,
    25_000.0,
    50_000.0,
    100_000.0,
    500_000.0,
    float("inf"),
]

# Family 3: Discharge resolution grid (m^3/s) for flow-enabled hydrofabrics
DISCHARGE_SCALE_GRID_M3S = [
    0.5,
    1.0,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1_000.0,
    5_000.0,
    float("inf"),
]

# Family 4: 1D river-network physical drainage area strata (km^2)
RIVER_AREA_BINS_KM2 = [
    (0.0, 10.0, "0-10"),
    (10.0, 100.0, "10-100"),
    (100.0, 1_000.0, "100-1k"),
    (1_000.0, 10_000.0, "1k-10k"),
    (10_000.0, 100_000.0, "10k-100k"),
    (100_000.0, 1_000_000.0, "100k-1M"),
    (1_000_000.0, float("inf"), ">1M"),
]

RIVER_AREA_CUTOFFS_KM2 = [0.0, 10.0, 100.0, 500.0, 1_000.0, 10_000.0, 100_000.0]

# Family 4: 1D river-network physical mean annual discharge strata (m^3/s)
RIVER_DISCHARGE_BINS_M3S = [
    (0.0, 0.1, "<0.1"),
    (0.1, 1.0, "0.1-1"),
    (1.0, 10.0, "1-10"),
    (10.0, 100.0, "10-100"),
    (100.0, 1_000.0, "100-1k"),
    (1_000.0, 10_000.0, "1k-10k"),
    (10_000.0, float("inf"), ">10k"),
]

RIVER_DISCHARGE_CUTOFFS_M3S = [0.1, 1.0, 10.0, 100.0, 1_000.0]

# Along-channel hydrologic distance CDF thresholds (river-km)
CHANNEL_DIST_THRESHOLDS_KM = [10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1_000.0]

# Harmonized global domain bounds (excludes Antarctica and Greenland for
# apples-to-apples comparison across all 4 hydrofabrics including GRIT v1.0)
COMMON_DOMAIN_LAT_MIN = -60.0
COMMON_DOMAIN_LAT_MAX = 84.0
EXCLUDED_ISO3_COMMON_DOMAIN = frozenset({"ATA", "GRL"})

# Hydrofabric registry
HYDROFABRICS = {
    "hydrosheds_basins": {
        "name": "HydroSHEDS / HydroBASINS (L1-L12) + BasinATLAS",
        "levels": list(range(1, 13)),
        "has_flow": True,
        "flow_model": "HydroATLAS (WaterGAP v2.2)",
        "topology": "tree",
    },
    "hydrosheds_rivers": {
        "name": "HydroSHEDS / HydroRIVERS v1.0 + RiverATLAS",
        "has_flow": True,
        "flow_model": "HydroATLAS (WaterGAP v2.2)",
        "topology": "tree",
    },
    "merit_basins": {
        "name": "MERIT-Hydro / MERIT-Basins v0.1",
        "has_flow": True,
        "flow_model": "GRADES v0.1 (Lin et al., 2019; pluggable Today's Earth adapter)",
        "topology": "tree",
    },
    "tdx_hydro": {
        "name": "TDX-Hydro / GEOGLOWS v2",
        "has_flow": True,
        "flow_model": "GEOGLOWS v2 Retrospective (1940-2024)",
        "topology": "tree",
    },
    "grit_v10": {
        "name": "GRIT v1.0 (30m FABDEM)",
        "has_flow": False,
        "flow_model": None,
        "topology": "dag_and_mainstem_tree",
    },
}
