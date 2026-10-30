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

"""Centralized filesystem paths and metrics mapping catalog.

This module registers paths to geographic vector files, hydrological model
metrics, and programmatically manages plotting names for extreme events
return-period metrics.
"""

import os

from utils import gfile_compat as gfile

# Base Directories
_DEFAULT_DATA_DIR = os.path.join(os.getcwd(), 'data')
PROJECT_DIR = os.getenv('GEO_BIAS_DATA_DIR', _DEFAULT_DATA_DIR)
_DEFAULT_THIRD_PARTY_DIR = (
    PROJECT_DIR
    if os.path.exists(os.path.join(PROJECT_DIR, 'attributes'))
    else os.path.join(PROJECT_DIR, 'third_party_data')
)
THIRD_PARTY_DIR = os.getenv('GEO_BIAS_THIRD_PARTY_DIR', _DEFAULT_THIRD_PARTY_DIR)
OUTPUT_DIR = os.getenv('GEO_BIAS_OUTPUT_DIR', PROJECT_DIR)
EXPOSURE_FULL_WORLD_DIR = os.path.join(THIRD_PARTY_DIR, 'exposure_full_world')


def _get_workspace_root():
  """Retrieves the absolute path to the repository root directory.

  Returns:
      str: The repository root path.
  """
  try:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  except NameError:
    return os.getcwd()


BASE_DIR = _get_workspace_root()
PAPER_ARTIFACTS_DIR = os.path.join(
    OUTPUT_DIR, 'v2_exposure_blockers', 'figures_and_tables_for_paper'
)


# ==============================================================================
# Pipeline Metrics Definition & Formatting Configurations
# ==============================================================================

# List of all supported target metrics in the hydrological evaluation.
ALL_METRICS = [
    # --- 2.0-Day Return Window Metrics ---
    '1.01-year-2.0-window-precision',
    '2.0-year-2.0-window-precision',
    '5.0-year-2.0-window-precision',
    '10.0-year-2.0-window-precision',
    '20.0-year-2.0-window-precision',
    '1.01-year-2.0-window-recall',
    '2.0-year-2.0-window-recall',
    '5.0-year-2.0-window-recall',
    '10.0-year-2.0-window-recall',
    '20.0-year-2.0-window-recall',
    # --- 1.0-Day Return Window Metrics ---
    '1.01-year-1.0-window-precision',
    '2.0-year-1.0-window-precision',
    '5.0-year-1.0-window-precision',
    '10.0-year-1.0-window-precision',
    '20.0-year-1.0-window-precision',
    '1.01-year-1.0-window-recall',
    '2.0-year-1.0-window-recall',
    '5.0-year-1.0-window-recall',
    '10.0-year-1.0-window-recall',
    '20.0-year-1.0-window-recall',
    # --- 0.0-Day Return Window Metrics ---
    '1.01-year-0.0-window-precision',
    '2.0-year-0.0-window-precision',
    '5.0-year-0.0-window-precision',
    '10.0-year-0.0-window-precision',
    '20.0-year-0.0-window-precision',
    '1.01-year-0.0-window-recall',
    '2.0-year-0.0-window-recall',
    '5.0-year-0.0-window-recall',
    '10.0-year-0.0-window-recall',
    '20.0-year-0.0-window-recall',
    # --- Classic Continuous Metrics ---
    'NSE',
    'KGE',
    'log-NSE',
    'log-KGE',
    'Beta-NSE',
    'Beta-KGE',
    'Alpha-NSE',
    'pearson-r',
]

# Outlier filtering minimum thresholds map.
OUTLIER_MIN_MAP = {
    'NSE': -0.5,
    'KGE': -0.5,
    'log-NSE': -0.5,
    'log-KGE': -0.5,
    'Beta-NSE': -3.0,
    'Beta-KGE': 0.0,
    'Alpha-NSE': 0.0,
    'pearson-r': -0.5,
}

# Outlier filtering maximum thresholds map.
OUTLIER_MAX_MAP = {
    'Beta-NSE': 3.0,
    'Beta-KGE': 3.0,
    'Alpha-NSE': 3.0,
}


def _clean_metric_val(val_str):
  """Truncates '.0' decimals for clean plot displays (e.g. '2.0' -> '2')."""
  if val_str.endswith('.0'):
    return val_str[:-2]
  return val_str


METRIC_SHORT_NAME_MAP = {
    'NSE': 'NSE',
    'KGE': 'KGE',
    'log-NSE': 'log-NSE',
    'log-KGE': 'log-KGE',
    'Beta-NSE': 'Beta-NSE',
    'Beta-KGE': 'Beta-KGE',
    'Alpha-NSE': 'Alpha-NSE',
    'pearson-r': 'r',
}

_years = ['1.01', '2.0', '5.0', '10.0', '20.0']
_windows = ['0.0', '1.0', '2.0']
_score_types = ['precision', 'recall']

_score_short_labels = {
    'precision': 'Precision',
    'recall': 'Recall',
}

# Dynamically populate plotting names for all return-period metric permutations
for win in _windows:
  for score in _score_types:
    for yr in _years:
      m_id = f'{yr}-year-{win}-window-{score}'
      yr_lbl = _clean_metric_val(yr)
      win_lbl = _clean_metric_val(win)
      s_label = _score_short_labels[score]
      METRIC_SHORT_NAME_MAP[m_id] = f'{s_label} ({yr_lbl}y, {win_lbl}d)'

METRIC_NAME_MAP = METRIC_SHORT_NAME_MAP


BANNED_SPATIAL_COLS = [
    'latitude',
    'longitude',
    'lat',
    'long',
    'gauge_lat',
    'gauge_lon',
    'is_external_polygon',
]

# Top Level Files
NATURAL_EARTH_PATH = os.path.join(THIRD_PARTY_DIR, 'naturalearth_lowres.geojson')

RERUN_ALL = os.getenv('GEO_BIAS_RERUN_ALL', '0').lower() in ('1', 'true', 'yes')
_DERIVED_BASE_DIR = OUTPUT_DIR if RERUN_ALL else THIRD_PARTY_DIR

# Directories
POLYGONS_DIR = os.path.join(THIRD_PARTY_DIR, 'polygons')
ATTRIBUTES_DIR = os.path.join(THIRD_PARTY_DIR, 'attributes')
METRICS_DIR = os.path.join(PROJECT_DIR, 'hydro_model_metrics')
COVERAGE_DIR = os.getenv(
    'GEO_BIAS_COVERAGE_DIR', os.path.join(_DERIVED_BASE_DIR, 'coverage_fractions')
)
ANOMALIES_DIR = os.path.join(OUTPUT_DIR, 'anomaly_scores')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'saved_models')
PREDS_DIR = os.path.join(OUTPUT_DIR, 'saved_predictions')
CV_PREDS_DIR = os.path.join(PREDS_DIR, 'cv_preds')
HYBAS_PREDS_DIR = os.path.join(PREDS_DIR, 'hybas_preds')
COUNTRY_MAPPINGS_DIR = os.path.join(PREDS_DIR, 'country_mappings')
V2_RESULTS_DIR = os.getenv(
    'GEO_BIAS_V2_RESULTS_DIR',
    os.path.join(OUTPUT_DIR, 'v2_exposure_blockers'),
)
STATS_DIR = os.path.join(V2_RESULTS_DIR, 'statistics')
INFORM_DIR = os.path.join(THIRD_PARTY_DIR, 'risk_data')
FIGURES_AND_TABLES_FOR_PAPER_DIR = os.path.join(
    V2_RESULTS_DIR, 'figures_and_tables_for_paper'
)
FIGURES_DIR = FIGURES_AND_TABLES_FOR_PAPER_DIR
GENERIC_DATA_DIR = os.getenv(
    'GEO_BIAS_GENERIC_DATA_DIR',
    os.path.join(_DERIVED_BASE_DIR, 'generic_training_data'),
)
EXPOSURE_DIR = os.getenv(
    'GEO_BIAS_EXPOSURE_DIR', os.path.join(THIRD_PARTY_DIR, 'exposure')
)
GLOBAL_BASIN_EXPOSURE_PATH = os.path.join(EXPOSURE_DIR, 'global_basin_exposure.csv')
GLOBAL_BASIN_EXPOSURE_UNDEFENDED_BY_RP_PATH = os.path.join(
    EXPOSURE_DIR, 'global_basin_exposure_by_return_period.csv'
)
GLOBAL_BASIN_EXPOSURE_DEFENDED_BY_RP_PATH = os.path.join(
    EXPOSURE_DIR, 'global_basin_exposure_defended_by_return_period.csv'
)
GLOBAL_BASIN_EXPOSURE_BY_RP_PATH = GLOBAL_BASIN_EXPOSURE_DEFENDED_BY_RP_PATH

# Evaluation Results Directories (Per Return Period & Integrated)
EVALUATION_RESULTS_DIR = os.path.join(OUTPUT_DIR, 'evaluation_results_run2')
EVALUATION_RESULTS_DIR_INTEGRATED = os.path.join(EVALUATION_RESULTS_DIR, 'integrated')
EVALUATION_RESULTS_DIR_5Y = os.path.join(EVALUATION_RESULTS_DIR, 'rp_5y')
EVALUATION_RESULTS_DIR_10Y = os.path.join(EVALUATION_RESULTS_DIR, 'rp_10y')
EVALUATION_RESULTS_DIR_20Y = os.path.join(EVALUATION_RESULTS_DIR, 'rp_20y')
EVALUATION_RESULTS_DIR_50Y = os.path.join(EVALUATION_RESULTS_DIR, 'rp_50y')
EVALUATION_RESULTS_DIR_100Y = os.path.join(EVALUATION_RESULTS_DIR, 'rp_100y')

EVALUATION_RESULTS_DIR_RP = {
    5: EVALUATION_RESULTS_DIR_5Y,
    10: EVALUATION_RESULTS_DIR_10Y,
    20: EVALUATION_RESULTS_DIR_20Y,
    50: EVALUATION_RESULTS_DIR_50Y,
    100: EVALUATION_RESULTS_DIR_100Y,
}


GAUGED_GENERIC_DATA_PATH = os.path.join(
    GENERIC_DATA_DIR, 'gauged_generic_training_data.csv'
)
UNGAUGED_GENERIC_DATA_PATH = os.path.join(
    GENERIC_DATA_DIR, 'ungauged_generic_training_data.csv'
)
HYBAS_GENERIC_DATA_PATH = os.path.join(
    GENERIC_DATA_DIR, 'hybas_generic_inference_data.csv'
)
HYBAS_GENERIC_DATA_ZARR_PATH = os.path.join(
    GENERIC_DATA_DIR, 'hybas_generic_inference_data.zarr'
)

# Specific Input Files
GRDC_POLYGONS_FILE = os.path.join(POLYGONS_DIR, 'grdc_polygons.geojson')
CARAVAN_POLYGONS_FILE = os.path.join(POLYGONS_DIR, 'caravan_polygons.geojson')
HYBAS_POLYGONS_DIR = os.path.join(POLYGONS_DIR, 'hybas')
HYBAS_SUBBASINS_DIR = os.path.join(POLYGONS_DIR, 'hybas_subbasins')
HYBAS_FULL_DRAINAGE_DIR = os.path.join(POLYGONS_DIR, 'hybas_full_drainage')

GRDC_ATTRIBUTES_PATH = os.path.join(ATTRIBUTES_DIR, 'grdc_attributes.csv')
GRDC_RECORD_LENGTHS_PATH = os.path.join(
    ATTRIBUTES_DIR, 'grdc_record_lengths.csv'
)
HYBAS_ATTRIBUTES_PATH = os.path.join(ATTRIBUTES_DIR, 'hybas_attributes.csv')
HYBAS_ROUTING_TOPOLOGY_PATH = os.path.join(
    ATTRIBUTES_DIR, 'hybas_routing_topology.csv'
)
CARAVAN_ATTRIBUTES_PATH = os.path.join(ATTRIBUTES_DIR, 'caravan_attributes.csv')
HYBAS_LATLON_PATH = os.path.join(ATTRIBUTES_DIR, 'hybas_latlon.csv')
CARAVAN_LATLON_PATH = os.path.join(ATTRIBUTES_DIR, 'caravan_coordinates.csv')
TABLE_1_PRECOMPUTED_STATS_PATH = os.path.join(
    STATS_DIR, 'main_text_table_1_summary_statistics.csv'
)
HYBAS_MAPPING_PATH = os.path.join(
    COUNTRY_MAPPINGS_DIR, 'hybas_to_country_mapping.csv'
)

CROSS_VALIDATION_DIR = os.path.join(PROJECT_DIR, 'cross_validation_splits')
RANDOM_CV_DIR = os.path.join(CROSS_VALIDATION_DIR, 'random')

VULN_FILE = os.path.join(
    INFORM_DIR, 'INFORM_Risk_2026_v072 - Vulnerability.csv'
)
INDICATOR_FILE = os.path.join(
    INFORM_DIR, 'INFORM_Risk_2026_v072 - Indicator Data.csv'
)
RISK_FILE = os.path.join(
    INFORM_DIR, 'INFORM_Risk_2026_v072 - INFORM Risk 2026 (a-z).csv'
)

# Output Files - Coverage Fractions
TRAIN_PAIRWISE_OVERLAPS_PATH = os.path.join(
    COVERAGE_DIR, 'train_pairwise_overlap_areas.csv'
)
TRAIN_ML_FEATURES_PATH = os.path.join(
    COVERAGE_DIR, 'train_fold_ml_features.csv'
)
CARAVAN_PAIRWISE_OVERLAPS_PATH = os.path.join(
    COVERAGE_DIR, 'caravan_pairwise_overlap_areas.csv'
)
HYBAS_COVERAGE_DIR = os.path.join(COVERAGE_DIR, 'hybas')

# Output Files - Anomaly Scores
RANDOM_ANOMALIES_PATH = os.path.join(ANOMALIES_DIR, 'random_anomaly_scores.csv')
SEPARATED_ANOMALIES_PATH = os.path.join(
    ANOMALIES_DIR, 'separated_anomaly_scores.csv'
)
GAUGED_ANOMALIES_PATH = os.path.join(ANOMALIES_DIR, 'gauged_anomaly_scores.csv')
INFER_HYBAS_CARAVAN_OUT_PATH = os.path.join(
    ANOMALIES_DIR, 'infer_hybas_caravan_anomaly_scores.csv'
)

SHAP_SAMPLING_METHODS = [
    'sorted_by_area',
    'random',
    'sorted_by_population',
    'sorted_by_gdp',
    'all_basins',
]

country_name_map = {
    'Bosnia and Herz.': 'Bosnia and Herzegovina',
    'Brunei': 'Brunei Darussalam',
    'Central African Rep.': 'Central African Republic',
    'Czechia': 'Czech Republic',
    'Dem. Rep. Congo': 'Congo DR',
    'Dominican Rep.': 'Dominican Republic',
    'Eq. Guinea': 'Equatorial Guinea',
    'Laos': 'Lao PDR',
    'Macedonia': 'North Macedonia',
    'Moldova': 'Moldova Republic of',
    'North Korea': 'Korea DPR',
    'Russia': 'Russian Federation',
    'S. Sudan': 'South Sudan',
    'Solomon Is.': 'Solomon Islands',
    'South Korea': 'Korea Republic of',
    'Syria': 'Syrian Arab Republic',
    'Turkey': 'Türkiye',
    'Vietnam': 'Viet Nam',
    'eSwatini': 'Eswatini',
}


# ==============================================================================
# Dynamic (Metric-Dependent) Variables and Paths
# ==============================================================================
METRIC = None
OUTLIER_MIN = None
OUTLIER_MAX = None
METRIC_SHORT_NAME = None

GAUGED_REGRESSION_CV_PREDS = None
UNGAUGED_REGRESSION_CV_PREDS = None
GLOBAL_HYBAS_PREDICTIONS_GAUGED = None
GLOBAL_HYBAS_PREDICTIONS_UNGAUGED = None
HYBAS_MAPPING_FILE = None
SHAP_BASE_DIR = None

FINAL_GAUGED_MODEL = None
FINAL_UNGAUGED_MODEL = None
SKILL_DIFFERENCE_BY_COUNTRY = None


def init(metric: str):
  """Initializes all metric-dependent paths and variables."""
  global METRIC, OUTLIER_MIN, OUTLIER_MAX, METRIC_SHORT_NAME
  global GAUGED_REGRESSION_CV_PREDS, UNGAUGED_REGRESSION_CV_PREDS
  global GLOBAL_HYBAS_PREDICTIONS_GAUGED, GLOBAL_HYBAS_PREDICTIONS_UNGAUGED
  global HYBAS_MAPPING_FILE, SHAP_BASE_DIR
  global FINAL_GAUGED_MODEL, FINAL_UNGAUGED_MODEL
  global SKILL_DIFFERENCE_BY_COUNTRY

  METRIC = metric
  OUTLIER_MIN = OUTLIER_MIN_MAP.get(METRIC, 0.0)
  OUTLIER_MAX = OUTLIER_MAX_MAP.get(METRIC, None)
  METRIC_SHORT_NAME = METRIC_SHORT_NAME_MAP.get(METRIC, METRIC)

  GAUGED_REGRESSION_CV_PREDS = os.path.join(
      CV_PREDS_DIR, f'gauged_regression_cv_preds_{METRIC}.csv'
  )
  UNGAUGED_REGRESSION_CV_PREDS = os.path.join(
      CV_PREDS_DIR, f'ungauged_regression_cv_preds_{METRIC}.csv'
  )
  GLOBAL_HYBAS_PREDICTIONS_GAUGED = os.path.join(
      HYBAS_PREDS_DIR, f'global_hybas_predictions_gauged_{METRIC}.csv'
  )
  GLOBAL_HYBAS_PREDICTIONS_UNGAUGED = os.path.join(
      HYBAS_PREDS_DIR, f'global_hybas_predictions_ungauged_{METRIC}.csv'
  )
  HYBAS_MAPPING_FILE = os.path.join(
      COUNTRY_MAPPINGS_DIR, f'hybas_to_country_mapping_{METRIC}.csv'
  )
  SHAP_BASE_DIR = os.path.join(PREDS_DIR, 'shap', METRIC, 'ungauged')

  FINAL_GAUGED_MODEL = os.path.join(
      MODELS_DIR, f'final_gauged_model_{METRIC}.pkl'
  )
  FINAL_UNGAUGED_MODEL = os.path.join(
      MODELS_DIR, f'final_ungauged_model_{METRIC}.pkl'
  )
  SKILL_DIFFERENCE_BY_COUNTRY = os.path.join(
      STATS_DIR, f'skill_difference_by_country_{METRIC}.csv'
  )


def ensure_directories_exist():
  """Ensures all required output directories exist. Call after InitGoogle."""
  for directory in [
      COVERAGE_DIR,
      HYBAS_COVERAGE_DIR,
      HYBAS_FULL_DRAINAGE_DIR,
      MODELS_DIR,
      PREDS_DIR,
      CV_PREDS_DIR,
      HYBAS_PREDS_DIR,
      COUNTRY_MAPPINGS_DIR,
      STATS_DIR,
      ANOMALIES_DIR,
      SHAP_BASE_DIR,
      GENERIC_DATA_DIR,
      EXPOSURE_DIR,
      PAPER_ARTIFACTS_DIR,
  ]:
    if not gfile.Exists(directory):
      gfile.MakeDirs(directory)
      print(f'Created directory: {directory}')

  for method in SHAP_SAMPLING_METHODS:
    method_dir = os.path.join(SHAP_BASE_DIR, method)
    if not gfile.Exists(method_dir):
      gfile.MakeDirs(method_dir)
      print(f'Created directory: {method_dir}')


# Initialize with default metric from env var or fallback
default_metric = os.getenv(
    'GEOGRAPHIC_BIAS_METRIC', '5.0-year-2.0-window-recall'
)
init(default_metric)
