# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Grey Nearing
# Created: 2026-07-14

"""Feature description catalog and classification utilities for the Geographical Bias project.

This module maps raw attribute names to human-readable descriptions, classifies
features into thematic groups (e.g. Topography, Climate, Anthropogenic), and
sets standard plotting color palettes.
"""

import numpy as np

FEATURE_DESCRIPTIONS = {
    "calculated_drain_area": "Drainage area.",
    "aet_mm_uyr": "Actual evapotranspiration; multi-year annual average (mm/yr).",
    "ari_ix_uav": "Aridity index; multi-year average.",
    "cly_pc_uav": "Clay content of soil; percentage multi-year average.",
    "cmi_ix_uyr": "Climate Moisture Index; multi-year annual average.",
    "crp_pc_use": "Cropland extent; percentage of sub-basin area.",
    "ele_mt_uav": "Elevation; multi-year average in meters above sea level.",
    "ero_kh_uav": "Soil erosion; multi-year average in kg/ha/yr.",
    "for_pc_use": "Forest cover; percentage of sub-basin area.",
    "gdp_ud_usu": "Gross Domestic Product (GDP); sum for sub-basin (USD).",
    "gla_pc_use": "Glacier cover; percentage of sub-basin area.",
    "glc_pc_u01": "Land cover: Evergreen Needleleaf Forest (percentage).",
    "glc_pc_u02": "Land cover: Evergreen Broadleaf Forest (percentage).",
    "glc_pc_u03": "Land cover: Deciduous Needleleaf Forest (percentage).",
    "glc_pc_u04": "Land cover: Deciduous Broadleaf Forest (percentage).",
    "glc_pc_u05": "Land cover: Mixed Forest (percentage).",
    "glc_pc_u06": "Land cover: Closed Shrublands (percentage).",
    "glc_pc_u07": "Land cover: Open Shrublands (percentage).",
    "glc_pc_u08": "Land cover: Woody Savannas (percentage).",
    "glc_pc_u09": "Land cover: Savannas (percentage).",
    "glc_pc_u10": "Land cover: Grasslands (percentage).",
    "glc_pc_u11": "Land cover: Permanent Wetlands (percentage).",
    "glc_pc_u12": "Land cover: Croplands (percentage).",
    "glc_pc_u13": "Land cover: Urban and Built-up (percentage).",
    "glc_pc_u14": "Land cover: Cropland/Natural Vegetation Mosaic (percentage).",
    "glc_pc_u15": "Land cover: Snow and Ice (percentage).",
    "glc_pc_u16": "Land cover: Barren or Sparsely Vegetated (percentage).",
    "glc_pc_u17": "Land cover: Water Bodies (percentage).",
    "glc_pc_u18": "Land cover: Tundra (percentage).",
    "glc_pc_u19": "Land cover: Permanent Snow/Ice (percentage).",
    "glc_pc_u20": "Land cover: Sparse Vegetation (percentage).",
    "glc_pc_u21": "Land cover: Salt Pans (percentage).",
    "glc_pc_u22": "Land cover: Moss and Lichen (percentage).",
    "hft_ix_u09": "Human Footprint index for year 2009 (multi-year average).",
    "hft_ix_u93": "Human Footprint index for year 1993 (multi-year average).",
    "inu_pc_ult": "Inundation (long-term); percentage of sub-basin area.",
    "inu_pc_umn": "Inundation (minimum); percentage of sub-basin area.",
    "inu_pc_umx": "Inundation (maximum); percentage of sub-basin area.",
    "ire_pc_use": "Irrigated area; percentage of sub-basin area.",
    "kar_pc_use": "Karst area; percentage of sub-basin area.",
    "lka_pc_use": "Lake area; percentage of sub-basin area.",
    "lkv_mc_usu": "Lake volume; total sum in sub-basin (million m3).",
    "nli_ix_uav": "Nighttime lights index; multi-year average.",
    "pac_pc_use": "Protected area cover; percentage of sub-basin area.",
    "pet_mm_uyr": "Potential evapotranspiration; multi-year annual average (mm/yr).",
    "pnv_pc_u01": "Potential Natural Veg: Tropical/Subtropical Moist Broadleaf Forests (%).",
    "pnv_pc_u02": "Potential Natural Veg: Tropical/Subtropical Dry Broadleaf Forests (%).",
    "pnv_pc_u03": "Potential Natural Veg: Tropical/Subtropical Coniferous Forests (%).",
    "pnv_pc_u04": "Potential Natural Veg: Temperate Broadleaf and Mixed Forests (%).",
    "pnv_pc_u05": "Potential Natural Veg: Temperate Coniferous Forests (%).",
    "pnv_pc_u06": "Potential Natural Veg: Boreal Forests/Taiga (%).",
    "pnv_pc_u07": "Potential Natural Veg: Tropical/Subtropical Grasslands/Savannas (%).",
    "pnv_pc_u08": "Potential Natural Veg: Temperate Grasslands/Savannas (%).",
    "pnv_pc_u09": "Potential Natural Veg: Flooded Grasslands/Savannas (%).",
    "pnv_pc_u10": "Potential Natural Veg: Montane Grasslands/Shrublands (%).",
    "pnv_pc_u11": "Potential Natural Veg: Tundra (%).",
    "pnv_pc_u12": "Potential Natural Veg: Mediterranean Forests/Woodlands (%).",
    "pnv_pc_u13": "Potential Natural Veg: Deserts and Xeric Shrublands (%).",
    "pnv_pc_u14": "Potential Natural Veg: Mangroves (%).",
    "pnv_pc_u15": "Potential Natural Veg: Inland Water (%).",
    "pop_ct_usu": "Population count; total sum within sub-basin.",
    "ppd_pk_uav": "Population density; multi-year average (people/km2).",
    "pre_mm_uyr": "Precipitation; multi-year annual average (mm/yr).",
    "prm_pc_use": "Permafrost extent; percentage of sub-basin area.",
    "pst_pc_use": "Pasture extent; percentage of sub-basin area.",
    "rdd_mk_uav": "Road density; multi-year average (m/km2).",
    "rev_mc_usu": "Reservoir volume; total sum in sub-basin (million m3).",
    "ria_ha_usu": "Reservoir area; total sum in sub-basin (hectares).",
    "riv_tc_usu": "River volume; total sum in sub-basin (thousand m3).",
    "slp_dg_uav": "Terrain slope; multi-year average in degrees.",
    "slt_pc_uav": "Silt content of soil; percentage multi-year average.",
    "snd_pc_uav": "Sand content of soil; percentage multi-year average.",
    "snw_pc_uyr": "Snow cover; percentage multi-year annual average.",
    "soc_th_uav": "Soil organic carbon; multi-year average (tonnes/ha).",
    "swc_pc_uyr": "Soil water content; percentage multi-year annual average.",
    "tmp_dc_uyr": "Air temperature; multi-year annual average (degrees Celsius).",
    "urb_pc_use": "Urban area; percentage of sub-basin area.",
    "wet_pc_u01": "Wetlands: Lakes (percentage).",
    "wet_pc_u02": "Wetlands: Reservoirs (percentage).",
    "wet_pc_u03": "Wetlands: Rivers (percentage).",
    "wet_pc_u04": "Wetlands: Fresh Marsh/Floodplain (percentage).",
    "wet_pc_u05": "Wetlands: Swamp/Flooded Forest (percentage).",
    "wet_pc_u06": "Wetlands: Coastal (percentage).",
    "wet_pc_u07": "Wetlands: Pan/Brackish/Saline (percentage).",
    "wet_pc_u08": "Wetlands: Bog/Fen/Mire (percentage).",
    "wet_pc_u09": "Wetlands: Intermittent (percentage).",
    "wet_pc_ug1": "Wetlands Global Class: Lake/Reservoir (percentage).",
    "wet_pc_ug2": "Wetlands Global Class: Floodplain/Marsh/Swamp (percentage).",

    # Overlap Fraction Features
    "total_upstream_coverage": "Total Upstream Coverage Fraction",
    "max_upstream_coverage": "Max Individual Upstream Coverage Fraction",
    "max_downstream_coverage": "Max Individual Downstream Coverage Fraction",
    "upstream_gauge_count": "Independent Upstream Gauge Count",

    # Anomaly Features
    "anomaly_score_if": "Anomaly Score (Isolation Forest)",
    "anomaly_score_ocsvm": "Anomaly Score (One-Class SVM)",
    "anomaly_score_lof": "Anomaly Score (Local Outlier Factor)",

    # Record Length Features
    "record_length": "Observation Record Length (years)"
}

def get_feature_type(feature):
    """Categorizes a feature name into its broader physical or metadata category.

    Args:
        feature (str): The name of the feature to categorize.

    Returns:
        str: The name of the category (e.g. 'Topography', 'Climate & Hydrology').

    Raises:
        ValueError: If the feature name is not recognized or cannot be categorized.
    """
    if feature.startswith('anomaly_score_'):
        return 'Anomaly Score'
    elif 'coverage' in feature or 'gauge_count' in feature:
        return 'Coverage/Overlap'
    elif feature == 'record_length':
        return 'Record Length'
    elif feature in ['calculated_drain_area', 'ele_mt_uav', 'slp_dg_uav', 'riv_tc_usu']:
        return 'Topography'
    elif 'climate' in FEATURE_DESCRIPTIONS[feature].lower() or \
         'evapotranspiration' in FEATURE_DESCRIPTIONS[feature].lower() or \
         feature in ['snw_pc_uyr', 'pre_mm_uyr', 'tmp_dc_uyr', 'ari_ix_uav', 'swc_pc_uyr']:
        return 'Climate & Hydrology'
    elif 'karst' in FEATURE_DESCRIPTIONS[feature].lower() or \
         feature in ['snd_pc_uav', 'slt_pc_uav', 'cly_pc_uav', 'ero_kh_uav', 'soc_th_uav']:
        return 'Soil and Geology'
    elif 'human' in FEATURE_DESCRIPTIONS[feature].lower() or \
         'irrigated' in FEATURE_DESCRIPTIONS[feature].lower() or \
         'population' in FEATURE_DESCRIPTIONS[feature].lower() or \
         'reservoir' in FEATURE_DESCRIPTIONS[feature].lower() or \
         feature in ['urb_pc_use', 'rdd_mk_uav', 'ppd_pk_uav', 'gdp_ud_usu', 'hft_ix_u09', 'nli_ix_uav']:
        return 'Anthropogenic Factors'
    elif 'Natural Veg' in FEATURE_DESCRIPTIONS[feature] or \
         'Wetlands' in FEATURE_DESCRIPTIONS[feature] or \
         'Lake' in FEATURE_DESCRIPTIONS[feature] or \
         'Inundation' in FEATURE_DESCRIPTIONS[feature] or \
         'Permafrost' in FEATURE_DESCRIPTIONS[feature] or \
         'Pasture' in FEATURE_DESCRIPTIONS[feature] or \
         'Forest' in FEATURE_DESCRIPTIONS[feature] or \
         'Land cover' in FEATURE_DESCRIPTIONS[feature] or \
         'Glacier' in FEATURE_DESCRIPTIONS[feature] or \
         'Cropland' in FEATURE_DESCRIPTIONS[feature] or \
         feature in ['pac_pc_use']:
        return 'Land Cover'
    else:
        raise ValueError('Unclassified Attribute: ', feature, FEATURE_DESCRIPTIONS[feature])

FEATURE_CLASSIFICATION = {feature: get_feature_type(feature) for feature in FEATURE_DESCRIPTIONS}

FEATURE_PALETTE = {
    'Topography': 'tab:pink',
    'Climate & Hydrology': 'tab:blue',
    'Land Cover': 'tab:green',
    'Soil and Geology': 'saddlebrown',
    'Anthropogenic Factors': 'tab:red',
    'Anomaly Score': 'tab:orange',
    'Coverage/Overlap': 'tab:purple',
    'Record Length': 'dimgray'
}
