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


"""Calculate Basin Attribute Anomaly Scores.

This script calculates how "anomalous" or dissimilar a specific basin is
compared
to its training set, based on its physical/climatic attributes, using Isolation
Forest, One-Class SVM, and Local Outlier Factor.

Main Steps:
1. Load and clean GRDC, Caravan, and HyBAS attribute datasets.
2. Standardize features to strictly shared numeric capabilities across datasets.
3. Load cross-validation splits (Random and Separated).
4. Compute train-to-train anomalies respecting CV splits.
5. Compute global gauged anomalies.
6. Compute HyBAS anomalies using Caravan as the training distribution.
7. Perform visual exploratory validation analysis.

Required Input Files:
- Attribute datasets: file_paths.GRDC_ATTRIBUTES_PATH,
file_paths.CARAVAN_ATTRIBUTES_PATH, file_paths.HYBAS_ATTRIBUTES_PATH
- Cross-Validation Splits: file_paths.CROSS_VALIDATION_DIR
- Performance Metrics (for EDA): file_paths.METRICS_DIR

Saved Output Files:
- Random Anomalies: file_paths.RANDOM_ANOMALIES_PATH
- Separated Anomalies: file_paths.SEPARATED_ANOMALIES_PATH
- Gauged Anomalies: file_paths.GAUGED_ANOMALIES_PATH
- Global Inference Anomalies: file_paths.INFER_HYBAS_CARAVAN_OUT_PATH
"""

import concurrent.futures
import os
import time

from absl import app
from absl import flags
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Machine Learning / Anomaly Detection Imports
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from tqdm.auto import tqdm

from utils import file_paths
from utils import stats_utils
from utils.feature_utils import FEATURE_DESCRIPTIONS
from utils.io_utils import load_experiment_metrics
from utils import gfile_compat as gfile

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'force_recalculate_random',
    False,
    'Set to True to force recalculation, overwriting existing files',
)
flags.DEFINE_bool(
    'force_recalculate_separated',
    False,
    'Set to True to force recalculation, overwriting existing files',
)
flags.DEFINE_bool(
    'force_recalculate_gauged',
    False,
    'Set to True to force recalculation, overwriting existing files',
)
flags.DEFINE_bool(
    'force_recalculate_hybas_caravan',
    False,
    'Set to True to force recalculation, overwriting existing files',
)


# Banned spatial columns (case-insensitive) to prevent direct geographic bias
BANNED_SPATIAL_COLS = [
    'latitude',
    'longitude',
    'lat',
    'long',
    'gauge_lat',
    'gauge_lon',
]


def clean_dataset(
    df: pd.DataFrame, dataset_name: str, index_hints: list
) -> pd.DataFrame:
  """Standardizes index and aggressively drops banned spatial coordinates.

  Args:
    df: The input pandas DataFrame containing the dataset to be cleaned.
    dataset_name: A string Name of the dataset being processed, used for logging
      purposes.
    index_hints: A list of string column names to test when searching for a
      unique basin identifier column to use as the index.

  Returns:
    pd.DataFrame: A cleaned DataFrame with a standard index name and without
    spatial coordinate columns.
  """
  # Try to set the dataframe index based on the provided list of potential ID column names.
  # This normalizes the identifiers across datasets so they align properly later.
  for hint in index_hints:
    if hint in df.columns:
      df.set_index(hint, inplace=True)
      break
  else:
    # If none of the preferred names are found, fallback to common raw index names
    # or just forcefully use the first column as a final resort.
    if 'Unnamed: 0' in df.columns:
      df.set_index('Unnamed: 0', inplace=True)
    else:
      df.set_index(df.columns[0], inplace=True)

  # Standardize the index name universally to 'basin_id' across all datasets.
  df.index.name = 'basin_id'

  # The HydroATLAS dataset prepends 'HYDRO_ATLAS:' to its columns. We strip this prefix
  # so that feature names match perfectly across all our distinct datasets.
  df.columns = df.columns.str.replace('HYDRO_ATLAS:', '', regex=False)

  # Identify and drop columns that contain latitude/longitude data.
  # If we don't drop these, the machine learning models might learn to simply memorize
  # geographic locations rather than learning relationships based on physical attributes.
  cols_to_drop = [
      c for c in df.columns if c.lower() in file_paths.BANNED_SPATIAL_COLS
  ]
  if cols_to_drop:
    print(
        f"[{time.strftime('%H:%M:%S')}] {dataset_name}: Dropped spatial columns"
        f' -> {cols_to_drop}'
    )
    df.drop(columns=cols_to_drop, inplace=True)

  return df


def load_and_clean_data():
  """Loads attribute datasets, harmonizes their features, and completely removes rows with NaNs.

  Reads GRDC, HyBAS, and Caravan datasets from file paths. Cleans each using
  `clean_dataset`,
  determines the strict subset of numeric features that exist in ALL THREE
  datasets, and cuts
  down the dataframes to ONLY contain those shared features. Finally, removes
  any basin missing
  a value and renames columns to be human-readable.

  Args: None

  Returns:
    tuple: Returns three pandas DataFrames:
      - grdc_df: The cleaned GRDC basin attributes dataset (training reference).
      - hybas_df: The cleaned HyBAS attributes dataset (global inference
      reference).
      - caravan_df: The cleaned Caravan attributes dataset (auxiliary inference
      reference).
  """
  # We begin the data preparation pipeline by loading GRDC, HyBAS, and Caravan datasets.
  # The goal is to clean up spatial bias and guarantee that every dataset has the exact same
  # set of features so a model trained on one can run inference on another smoothly.

  # Load the GRDC dataset which acts as our primary training dataset for gauged basins.
  print(f"\n[{time.strftime('%H:%M:%S')}] Loading GRDC Attributes...")
  start = time.time()
  with gfile.Open(file_paths.GRDC_ATTRIBUTES_PATH, 'rt') as f:
    grdc_df = pd.read_csv(f)
  grdc_df = clean_dataset(grdc_df, 'GRDC', ['gauge_id', 'unique_gauge_id'])
  print(
      f"[{time.strftime('%H:%M:%S')}] GRDC loaded: {len(grdc_df)} basins,"
      f' {len(grdc_df.columns)} potential features. Shape: {grdc_df.shape}.'
      f' Elapsed time: {time.time() - start:.2f} seconds.'
  )

  # Load the HyBAS dataset which acts as our global inference target (basins we want to predict).
  print(f"\n[{time.strftime('%H:%M:%S')}] Loading HyBAS Attributes...")
  start = time.time()
  with gfile.Open(file_paths.HYBAS_ATTRIBUTES_PATH, 'rt') as f:
    hybas_df = pd.read_csv(f)

  # Remove 'HYDRO_ATLAS:' prefix directly prior to cleaning to ensure standard parsing.
  hybas_df = hybas_df.rename(
      columns={col: col.replace('HYDRO_ATLAS:', '') for col in hybas_df.columns}
  )
  hybas_df = clean_dataset(
      hybas_df, 'HyBAS', ['HYBAS_ID', 'hybas_id', 'Unnamed: 0']
  )
  print(
      f"[{time.strftime('%H:%M:%S')}] HyBAS loaded: {len(hybas_df)} basins,"
      f' {len(hybas_df.columns)} potential features. Shape: {hybas_df.shape}.'
      f' Elapsed time: {time.time() - start:.2f} seconds.'
  )

  # Load the Caravan dataset which is used as an alternative inference baseline.
  print(f"\n[{time.strftime('%H:%M:%S')}] Loading Caravan Attributes...")
  start = time.time()
  with gfile.Open(file_paths.CARAVAN_ATTRIBUTES_PATH, 'rt') as f:
    caravan_df = pd.read_csv(f)
  caravan_df = clean_dataset(
      caravan_df, 'Caravan', ['gauge_id', 'unique_gauge_id']
  )
  print(
      f"[{time.strftime('%H:%M:%S')}] Caravan loaded: {len(caravan_df)} basins,"
      f' {len(caravan_df.columns)} potential features. Shape:'
      f' {caravan_df.shape}. Elapsed time: {time.time() - start:.2f} seconds.'
  )

  # Cross-reference the columns across datasets to discover exactly which features
  # are reliably present in all three sources.
  print(f"\n[{time.strftime('%H:%M:%S')}] {'-'*40}")
  print(f"[{time.strftime('%H:%M:%S')}] COLUMN INTERSECTION REPORT")
  print(f"[{time.strftime('%H:%M:%S')}] {'-'*40}")

  grdc_cols = set(grdc_df.columns)
  caravan_cols = set(caravan_df.columns)
  hybas_cols = set(hybas_df.columns)

  shared_cols = grdc_cols.intersection(caravan_cols).intersection(hybas_cols)
  all_unique_cols = grdc_cols.union(caravan_cols).union(hybas_cols)
  grdc_only = grdc_cols - shared_cols
  caravan_only = caravan_cols - shared_cols
  hybas_only = hybas_cols - shared_cols

  print(
      '\nTotal numbers of unique columns in either dataset:'
      f' {len(all_unique_cols)}'
  )
  print(f'Total number of columns in ALL THREE datasets: {len(shared_cols)}')

  print(f'\nColumns in ALL THREE ({len(shared_cols)}):')
  for col in sorted(shared_cols):
    print(f'  - {col}')

  # Extract ONLY the features that are shared AND are numeric data types (integer/float),
  # sorting them to guarantee exact order. Models will break if text/string columns leak in.
  shared_numeric_cols = sorted(
      grdc_df[list(shared_cols)]
      .select_dtypes(include=[np.number])
      .columns.tolist()
  )
  non_numeric_shared = shared_cols - set(shared_numeric_cols)

  if non_numeric_shared:
    print(
        '\nShared columns dropped for being non-numeric'
        f' ({len(non_numeric_shared)}):'
    )
    for col in sorted(non_numeric_shared):
      print(f'  - {col}')

  print(
      '\nFINAL SHARED NUMERIC COLUMNS USED FOR ML'
      f' ({len(shared_numeric_cols)}):'
  )
  for col in shared_numeric_cols:
    print(f'  - {col}')

  # Debugging analysis: calculate and log which datasets are restricting the features.
  # This provides insight if a dataset is accidentally dropping powerful predictors.
  grdc_numeric_cols = set(grdc_df.select_dtypes(include=[np.number]).columns)
  caravan_numeric_cols = set(
      caravan_df.select_dtypes(include=[np.number]).columns
  )
  hybas_numeric_cols = set(hybas_df.select_dtypes(include=[np.number]).columns)

  lost_to_caravan = grdc_numeric_cols - caravan_numeric_cols
  print(
      'Columns in GRDC (numeric) but missing in Caravan:'
      f' {len(lost_to_caravan)}'
  )
  for col in sorted(lost_to_caravan):
    print(f'  - {col}')

  lost_to_hybas = grdc_numeric_cols - hybas_numeric_cols
  print(
      f'\nColumns in GRDC (numeric) but missing in HyBAS: {len(lost_to_hybas)}'
  )
  for col in sorted(lost_to_hybas):
    print(f'  - {col}')

  print(f'\nTotal columns used for ML: {len(shared_numeric_cols)}')

  # Sub-select only the valid numeric columns from each dataframe. Use .copy() to firmly
  # separate memory references so pandas won't warn us about modifying slices later.
  start = time.time()
  grdc_df = grdc_df[shared_numeric_cols].copy()
  hybas_df = hybas_df[shared_numeric_cols].copy()
  caravan_df = caravan_df[shared_numeric_cols].copy()

  # Remove basins carrying 'NaN' (missing values) in any feature column.
  # Distance algorithms fail critically if they encounter missing values.
  grdc_len_pre = len(grdc_df)
  grdc_df.dropna(inplace=True)
  print(
      f"\n[{time.strftime('%H:%M:%S')}] GRDC: Dropped"
      f' {grdc_len_pre - len(grdc_df)} basins strictly due to missing/NaN'
      ' attribute values.'
  )

  hybas_len_pre = len(hybas_df)
  hybas_df.dropna(inplace=True)
  print(
      f"[{time.strftime('%H:%M:%S')}] HyBAS: Dropped"
      f' {hybas_len_pre - len(hybas_df)} basins strictly due to missing/NaN'
      ' attribute values.'
  )

  caravan_len_pre = len(caravan_df)
  caravan_df.dropna(inplace=True)
  print(
      f"[{time.strftime('%H:%M:%S')}] Caravan: Dropped"
      f' {caravan_len_pre - len(caravan_df)} basins strictly due to missing/NaN'
      ' attribute values.'
  )

  # Translate the heavily abbreviated raw feature codes to legible labels making plots clearer.
  grdc_df.rename(columns=FEATURE_DESCRIPTIONS, inplace=True)
  hybas_df.rename(columns=FEATURE_DESCRIPTIONS, inplace=True)
  caravan_df.rename(columns=FEATURE_DESCRIPTIONS, inplace=True)
  print(
      f"[{time.strftime('%H:%M:%S')}] Renamed feature columns using"
      ' FEATURE_DESCRIPTIONS mapping. Clean NaNs and Copy took'
      f' {time.time() - start:.2f} seconds. Shapes: GRDC: {grdc_df.shape},'
      f' HyBAS: {hybas_df.shape}, Caravan: {caravan_df.shape}'
  )
  return grdc_df, hybas_df, caravan_df


def load_cv_splits(split_dir, valid_gauges):
  """Helper to load train/test folds from a given sub-directory.

  Reads text files outlining basins allocated to testing sets (one fold per
  file),
  then automatically resolves what the exact corresponding training gauge list
  must be.

  Args:
    split_dir: String path holding the plain text files defining test fold
      splits.
    valid_gauges: A list of gauge IDs known to be valid and loaded properly in
      the earlier data ingest step.

  Returns:
    tuple: A tuple containing:
      - fold_gauges (dict): Maps integer fold IDs to dicts containing 'train'
      and 'test' basin ID lists.
      - all_active_gauges (list): The master universe of all valid/used test
      gauges identified.
  """
  # Parse the directory looking for testing segment definition files
  split_files = gfile.Glob(os.path.join(split_dir, '*.txt'))
  if not split_files:
    print(
        f"[{time.strftime('%H:%M:%S')}] WARNING: No split files found in"
        f' {split_dir}'
    )
    return {}, []

  # Open each text file and record which basin IDs belong inside that fold's test set.
  fold_gauges = {}
  for fold, file_path in enumerate(split_files):
    fold_gauges[fold] = {}
    with gfile.Open(file_path, 'rt') as f:
      lines = f.readlines()
      fold_gauges[fold]['test'] = [
          line.strip() for line in lines if line.strip()
      ]

  # Make a comprehensive list uniting every gauge tested across any fold.
  all_test_gauges = [g for fold in fold_gauges.values() for g in fold['test']]

  # Cross-reference this list with our known safe data (valid_gauges) to ensure
  # we ignore gauges whose attribute data could be missing/deleted during clean-up.
  all_active_gauges = list(set(all_test_gauges).intersection(set(valid_gauges)))

  # Programmatically define the training data sets. For a fold's given test set,
  # its matching training set simply consists of all active gauges EXCEPT those in the test set.
  for fold in fold_gauges:
    fold_test_gauges = fold_gauges[fold]['test']
    fold_gauges[fold]['train'] = list(
        set(all_active_gauges) - set(fold_test_gauges)
    )

  return fold_gauges, all_active_gauges


def setup_cv_splits(grdc_index):
  """Loads lists of testing and training basins representing different geographical cross validation tests.

  Reads splitting manifests for Random (testing subsets separated randomly) and
  Separated
  (testing subsets intentionally clustered apart from the training network)
  scenarios.

  Args:
    grdc_index: A pandas Index or list declaring all basin IDs successfully
      surviving data cleanings.

  Returns:
    tuple: containing:
      - random_fold_gauges (dict): Maps random folds training and testing
      arrays.
      - separated_fold_gauges (dict): Maps physically disjoint folds training
      and testing arrays.
      - all_active_grdc_gauges (list): Amalgamated list of gauge IDs actively
      evaluated.
  """
  # Parse out the subsets designating the completely "randomly selected" validation blocks.
  print(
      f"\n[{time.strftime('%H:%M:%S')}] Loading Random Cross Validation"
      ' Splits...'
  )
  start = time.time()
  random_cv_dir = os.path.join(file_paths.CROSS_VALIDATION_DIR, 'random')
  random_fold_gauges, active_random = load_cv_splits(random_cv_dir, grdc_index)
  print(
      f"[{time.strftime('%H:%M:%S')}] Found {len(active_random)} active random"
      ' CV gauges matching our GRDC dataset. Elapsed time:'
      f' {time.time() - start:.2f} seconds.'
  )

  # Parse the subsets designating "hydrologically separated" logic blocks, essentially
  # guaranteeing that training basins DO NOT share direct river networks with testing basins.
  print(
      f"\n[{time.strftime('%H:%M:%S')}] Loading Separated Cross Validation"
      ' Splits...'
  )
  start = time.time()
  separated_cv_dir = os.path.join(file_paths.CROSS_VALIDATION_DIR, 'separated')
  separated_fold_gauges, active_separated = load_cv_splits(
      separated_cv_dir, grdc_index
  )
  print(
      f"[{time.strftime('%H:%M:%S')}] Found {len(active_separated)} active"
      ' separated CV gauges matching our GRDC dataset. Elapsed time:'
      f' {time.time() - start:.2f} seconds.'
  )

  # Merge ID records across scenarios producing one master sequence designating exactly
  # what basins form our active baseline modeling universe.
  all_active_grdc_gauges = list(set(active_random).union(set(active_separated)))
  return random_fold_gauges, separated_fold_gauges, all_active_grdc_gauges


# ### Function Definition: calculate_anomaly_scores
def calculate_anomaly_scores(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> dict:
  """Calculates anomaly scores for testing data based on training data distributions.

  Fits three outlier detection algorithms (Isolation Forest, One-Class SVM,
  Local
  Outlier Factor) on `X_train` and predicts on `X_test`. Both dataframes are
  automatically
  scaled to have a mean of 0 and variance of 1 prior to prediction.

  Args:
    X_train: A pandas DataFrame containing the training features (the baseline
      "normal" distribution).
    X_test: A pandas DataFrame containing the testing features (the instances to
      score for outliers).

  Returns:
    dict: A dictionary containing three pandas Series keyed by the model name
    ('anomaly_score_if',
    'anomaly_score_ocsvm', 'anomaly_score_lof'). Note that lower decision
    function scores
    always indicate HIGHER anomaly level.
  """
  # Catch logic errors: We must enforce that the model testing phase uses the exact
  # same features evaluated during training, otherwise matrix mathematics will fail.
  if set(X_train.columns) != set(X_test.columns):
    raise ValueError(
        'CRITICAL: X_train and X_test feature sets do not match exactly!'
    )

  # Lock in the feature column order exactly. Scikit-learn expects feature matrices
  # to present columns in identical sequence across splits; failure to comply yields ValueErrors.
  feature_order = list(X_train.columns)
  X_test_aligned = X_test[feature_order].copy()
  X_train_aligned = X_train.copy()

  # Scale data so all attributes roughly span -3 to +3 standard deviations.
  # Anomaly detectors are distance-based; without scaling, features with large absolute values
  # like 'GDP' would completely overshadow features with tiny bounds like 'Aridity Index'.
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train_aligned)
  X_test_scaled = scaler.transform(X_test_aligned)

  # Deploy Isolation Forest: A tree-based algorithm that determines anomalies
  # by how few random splits it takes to isolate a point from the rest of the dataset.
  isolation_forest = IsolationForest(random_state=42, n_jobs=-1)
  isolation_forest.fit(X_train_scaled)
  scores_if = pd.Series(
      isolation_forest.decision_function(X_test_scaled),
      index=X_test_aligned.index,
  )

  # Deploy One-Class Support Vector Machine: Fits a hypersphere boundary around the
  # 'normal' data in high-dimensional space and measures how far out points lie.
  one_class_svm = OneClassSVM()
  one_class_svm.fit(X_train_scaled)
  if len(X_test_scaled) >= 2000:
    n_threads = min(56, os.cpu_count() or 32)
    chunks = np.array_split(X_test_scaled, n_threads)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as ex:
      ocsvm_vals = np.concatenate(
          list(ex.map(one_class_svm.decision_function, chunks))
      )
  else:
    ocsvm_vals = one_class_svm.decision_function(X_test_scaled)
  scores_ocsvm = pd.Series(ocsvm_vals, index=X_test_aligned.index)

  # Deploy Local Outlier Factor: Compares the local anomaly density around a test point
  # to the densities of its training phase neighbors, flagging unusually sparse regions.
  local_outlier_factor = LocalOutlierFactor(novelty=True, n_jobs=-1)
  local_outlier_factor.fit(X_train_scaled)
  scores_lof = pd.Series(
      local_outlier_factor.decision_function(X_test_scaled),
      index=X_test_aligned.index,
  )

  return {
      'anomaly_score_if': scores_if,
      'anomaly_score_ocsvm': scores_ocsvm,
      'anomaly_score_lof': scores_lof,
  }


# ### Function Definition: `compute_cv_anomalies`
def compute_cv_anomalies(fold_gauges, out_path, calc_flag, name_label, grdc_df):
  """Helper to repeatedly process train/test splits, extract anomalies, and save results.

  Iterates over folds in the cross-validation setup, constructs training and
  testing
  subsets, runs anomaly detection models, and aggregates the scores into a
  single DataFrame.

  Args:
    fold_gauges: A dictionary containing test and train gauge IDs for each fold.
    out_path: A string path marking where the resulting CSV will be saved (or
      loaded from).
    calc_flag: A boolean dictating whether to re-calculate (True) or load from
      cache (False).
    name_label: A string used in logging to identify this cross-validation split
      (e.g. 'Random CV').
    grdc_df: The primary pandas DataFrame containing the GRDC basin attributes.

  Returns:
    pd.DataFrame: A pandas DataFrame pairing gauge identifiers and their fold
    index to their computed
    anomaly scores.
  """
  if calc_flag:
    # Build anomaly scores dynamically if calculation is requested/required.
    print(
        f"\n[{time.strftime('%H:%M:%S')}] Calculating {name_label} Anomaly"
        ' Scores...'
    )
    start = time.time()
    scores_list = []

    # Loop over every discrete cross-validation split in the dataset.
    for fold, split_info in tqdm(
        fold_gauges.items(), desc=f'Processing {name_label} Folds'
    ):
      # Extract valid gauge instances specifically matching our clean GRDC dataframe.
      test_gauges = [g for g in split_info['test'] if g in grdc_df.index]
      train_gauges = [g for g in split_info['train'] if g in grdc_df.index]

      # Skip processing if a fold had all of its test basins stripped during cleaning.
      if not test_gauges:
        continue

      # Pull matching basin rows from the dataframe to create isolated Train and Test sets.
      X_train = grdc_df.loc[train_gauges]
      X_test = grdc_df.loc[test_gauges]

      # Generate anomaly scores for testing data based purely on the training distribution.
      fold_scores_dict = calculate_anomaly_scores(X_train, X_test)

      # Organize the outputs with identifying metadata (fold id) for downstream merging.
      fold_df = pd.DataFrame(fold_scores_dict)
      fold_df.index.name = 'gauge_id'
      fold_df['fold'] = fold
      scores_list.append(fold_df)

    # Concatenate all fold-level predictions into a single massive dataframe.
    anomaly_df = pd.concat(scores_list)
    print(
        f"[{time.strftime('%H:%M:%S')}] Extracted anomaly scores for"
        f' {len(anomaly_df)} {name_label} basins. Shape: {anomaly_df.shape}.'
        f' Elapsed time: {time.time() - start:.2f} seconds.'
    )

    # Save to disk as a CSV to allow for faster reloading next time the script executes.
    print(f"[{time.strftime('%H:%M:%S')}] Saving to {out_path}")
    with gfile.Open(out_path, 'w') as f:
      anomaly_df.to_csv(f)
    return anomaly_df
  else:
    # Quickly read pre-calculated scores straight from disk if caching allows.
    print(
        f"\n[{time.strftime('%H:%M:%S')}] Loading cached {name_label} Anomaly"
        f' Scores from {out_path}'
    )
    start = time.time()
    with gfile.Open(out_path, 'r') as f:
      anomaly_df = pd.read_csv(f)

    # Ensure correct index format holds after pandas loading finishes.
    if 'gauge_id' in anomaly_df.columns:
      anomaly_df.set_index('gauge_id', inplace=True)
    print(
        f"[{time.strftime('%H:%M:%S')}] Loaded cached shape:"
        f' {anomaly_df.shape}, elapsed: {time.time() - start:.2f} seconds.'
    )
    return anomaly_df


def calculate_gauged_anomalies(
    grdc_df, all_active_grdc_gauges, CALC_NEW_GAUGED
):
  """Computes anomalies globally assessing "gauged" basins against themselves.

  Unlike cross-validation splits which hide the target basin during training,
  this
  scenario models what happens when a basin exists squarely in the global
  training data.

  Args:
    grdc_df: The primary pandas DataFrame containing the GRDC basin attributes.
    all_active_grdc_gauges: List of valid gauge IDs active in our analysis
      pipeline.
    CALC_NEW_GAUGED: A boolean flag, True to run calculation; False to load
      purely from cache.

  Returns:
    pd.DataFrame: A pandas DataFrame containing global anomaly scores for gauged
    basins.
  """
  # For fully surveyed 'gauged' basins, the model technically 'saw' all of them during training.
  # Thus we measure normal performance by training our anomaly detector dynamically on the
  # entire global dataset, then scoring that same entire dataset.
  if CALC_NEW_GAUGED:
    print(
        f"\n[{time.strftime('%H:%M:%S')}] Calculating GAUGED Anomaly Scores..."
    )
    start = time.time()

    # Isolate all allowed global instances forming our baseline matrix.
    X_all_grdc = grdc_df.loc[all_active_grdc_gauges]

    # Calculate anomaly deviations for members of the training set relative strictly to that training set.
    gauged_scores_dict = calculate_anomaly_scores(
        X_train=X_all_grdc, X_test=X_all_grdc
    )

    # Frame dictionary contents and lock fold index flag to 'all' reflecting its global nature.
    gauged_anomaly_df = pd.DataFrame(gauged_scores_dict)
    gauged_anomaly_df.index.name = 'gauge_id'
    gauged_anomaly_df['fold'] = 'all'

    print(
        f"[{time.strftime('%H:%M:%S')}] Extracted gauged anomaly scores for"
        f' {len(gauged_anomaly_df)} GRDC basins. Shape:'
        f' {gauged_anomaly_df.shape}. Elapsed time: {time.time() - start:.2f}'
        ' seconds.'
    )

    # Persist records to file to expedite future pipeline launches.
    print(
        f"[{time.strftime('%H:%M:%S')}] Saving to"
        f' {file_paths.GAUGED_ANOMALIES_PATH}'
    )
    with gfile.Open(file_paths.GAUGED_ANOMALIES_PATH, 'w') as f:
      gauged_anomaly_df.to_csv(f)
  else:
    # Use pre-saved anomaly records if re-calculation wasn't manually forced.
    print(
        f"\n[{time.strftime('%H:%M:%S')}] Loading cached GAUGED Anomaly Scores"
        f' from {file_paths.GAUGED_ANOMALIES_PATH}'
    )
    start = time.time()
    with gfile.Open(file_paths.GAUGED_ANOMALIES_PATH, 'r') as f:
      gauged_anomaly_df = pd.read_csv(f)
    if 'gauge_id' in gauged_anomaly_df.columns:
      gauged_anomaly_df.set_index('gauge_id', inplace=True)
    print(
        f"[{time.strftime('%H:%M:%S')}] Loaded cached shape:"
        f' {gauged_anomaly_df.shape}, elapsed: {time.time() - start:.2f}'
        ' seconds.'
    )
  return gauged_anomaly_df


def calculate_hybas_inference_anomalies(
    caravan_df, hybas_df, CALC_NEW_HYBAS_CARAVAN
):
  """Measures the 'strangeness' of target HyBAS basins compared strictly against Caravan data.

  Determines if a desired prediction region (HyBAS) diverges largely from
  regions well
  covered inside the available Caravan datasets, reflecting real-world inference
  bounds.

  Args:
    caravan_df: Pandas DataFrame representing physical attributes across the
      Caravan regime.
    hybas_df: Pandas DataFrame representing target attributes across global
      inference points (HyBAS).
    CALC_NEW_HYBAS_CARAVAN: A boolean denoting if computations should occur
      (True) or load from cache (False).

  Returns:
    pd.DataFrame: A pandas DataFrame populated securely with predictive anomaly
    metrics denoting
    HyBAS basin abnormality relative to Caravan constraints.
  """
  # As part of global extrapolation, we want to know how far outside normal parameter bounds
  # a newly supplied HyBAS inference catchment resides when specifically compared to a known baseline (Caravan).
  if CALC_NEW_HYBAS_CARAVAN:
    print(
        f"\n[{time.strftime('%H:%M:%S')}] Calculating HYBAS GLOBAL INFERENCE"
        ' Anomaly Scores (based on Caravan)...'
    )
    start = time.time()

    # Declare Caravan gauges exclusively as the foundation representing 'normal' distributions.
    X_train_caravan = caravan_df

    # Declare our true target HyBAS basins entirely as the testing regime to query metrics for.
    X_test_hybas = hybas_df

    hybas_scores_dict = calculate_anomaly_scores(
        X_train=X_train_caravan, X_test=X_test_hybas
    )

    # Convert results firmly back into our working dataframe structure ready to merge globally.
    hybas_anomaly_df = pd.DataFrame(hybas_scores_dict)
    hybas_anomaly_df.index.name = 'hybas_id'
    hybas_anomaly_df['fold'] = 'infer'

    print(
        f"[{time.strftime('%H:%M:%S')}] Extracted inference anomaly scores for"
        f' {len(hybas_anomaly_df)} HyBAS basins. Shape:'
        f' {hybas_anomaly_df.shape}. Elapsed time: {time.time() - start:.2f}'
        ' seconds.'
    )

    # Store these values permanently reflecting output anomaly results ready for future validation checks.
    print(
        f"[{time.strftime('%H:%M:%S')}] Saving to"
        f' {file_paths.INFER_HYBAS_CARAVAN_OUT_PATH}'
    )
    with gfile.Open(file_paths.INFER_HYBAS_CARAVAN_OUT_PATH, 'w') as f:
      hybas_anomaly_df.to_csv(f)
  else:
    # Recover these results directly from persistent storage instead.
    print(
        f"\n[{time.strftime('%H:%M:%S')}] Loading cached HYBAS GLOBAL INFERENCE"
        ' Anomaly Scores (based on Caravan) from'
        f' {file_paths.INFER_HYBAS_CARAVAN_OUT_PATH}'
    )
    start = time.time()
    with gfile.Open(file_paths.INFER_HYBAS_CARAVAN_OUT_PATH, 'r') as f:
      hybas_anomaly_df = pd.read_csv(f)
    if 'hybas_id' in hybas_anomaly_df.columns:
      hybas_anomaly_df.set_index('hybas_id', inplace=True)
    print(
        f"[{time.strftime('%H:%M:%S')}] Loaded cached shape:"
        f' {hybas_anomaly_df.shape}, elapsed: {time.time() - start:.2f}'
        ' seconds.'
    )
  return hybas_anomaly_df


def load_metrics_for_eda(
    gauged_anomaly_df, random_anomaly_df, separated_anomaly_df
):
  """Merges generated anomaly evaluations firmly with physical model performance records dynamically.

  Allows alignment between statistical detection algorithms (what the paper
  tests) and
  neural-driven hydrological model outcomes (e.g. NSE score drops, what end
  users care about).
  Strips extreme performance outliers based strictly on user configuration
  constraints.

  Args:
    gauged_anomaly_df: DataFrame denoting global score assessments.
    random_anomaly_df: DataFrame matching completely shuffled validation
      metrics.
    separated_anomaly_df: DataFrame anchoring geographically separated basin
      evaluations.

  Returns:
    tuple: A tuple enclosing 6 comprehensive outputs supporting subsequent deep
    exploratory analysis tracking:
      (random_skill, separated_skill, gauged_skill, gauged_eval_df,
      random_eval_df, separated_eval_df)
  """
  # Up to now, we computed physical anomalies. Next we must load the *actual modelling skills*
  # (how well the Dual-LSTM predicted runoff) to test if higher anomalies systematically caused
  # worse modelling performance.
  print(f"\n[{time.strftime('%H:%M:%S')}] Loading Metrics for EDA...")

  # Load raw modelling skills strictly referencing specific experiment configurations (e.g., 'random').
  random_skill = load_experiment_metrics(
      file_paths.METRICS_DIR,
      'random',
      file_paths.METRIC,
      rename_to=f'random_{file_paths.METRIC}',
  )
  separated_skill = load_experiment_metrics(
      file_paths.METRICS_DIR,
      'separated',
      file_paths.METRIC,
      rename_to=f'separated_{file_paths.METRIC}',
  )
  gauged_skill = load_experiment_metrics(
      file_paths.METRICS_DIR,
      'gauged',
      file_paths.METRIC,
      rename_to=f'gauged_{file_paths.METRIC}',
  )

  # Fuse anomaly scores with respective modelling skills for the Global Gauged scenario, matching inherently on the gauge index.
  gauged_eval_df = gauged_anomaly_df.join(gauged_skill, how='inner')

  # Drop predictions where standard skill boundaries (like extreme negatives) broke configuration tolerances securely limiting noise.
  if file_paths.OUTLIER_MIN is not None:
    gauged_eval_df = gauged_eval_df[
        gauged_eval_df[f'gauged_{file_paths.METRIC}'] >= file_paths.OUTLIER_MIN
    ]
  if file_paths.OUTLIER_MAX is not None:
    gauged_eval_df = gauged_eval_df[
        gauged_eval_df[f'gauged_{file_paths.METRIC}'] <= file_paths.OUTLIER_MAX
    ]

  # Fuse anomaly scores specifically intersecting random subsets ensuring matching sizes cleanly.
  random_eval_df = random_anomaly_df.join(random_skill, how='inner')
  if file_paths.OUTLIER_MIN is not None:
    random_eval_df = random_eval_df[
        random_eval_df[f'random_{file_paths.METRIC}'] >= file_paths.OUTLIER_MIN
    ]
  if file_paths.OUTLIER_MAX is not None:
    random_eval_df = random_eval_df[
        random_eval_df[f'random_{file_paths.METRIC}'] <= file_paths.OUTLIER_MAX
    ]

  # Fuse anomaly scores aligning directly onto physically separated evaluation metrics targeting extrapolative validation limits securely.
  separated_eval_df = separated_anomaly_df.join(separated_skill, how='inner')
  if file_paths.OUTLIER_MIN is not None:
    separated_eval_df = separated_eval_df[
        separated_eval_df[f'separated_{file_paths.METRIC}']
        >= file_paths.OUTLIER_MIN
    ]
  if file_paths.OUTLIER_MAX is not None:
    separated_eval_df = separated_eval_df[
        separated_eval_df[f'separated_{file_paths.METRIC}']
        <= file_paths.OUTLIER_MAX
    ]

  print(
      f"[{time.strftime('%H:%M:%S')}] Gauged evaluation dataset size:"
      f' {len(gauged_eval_df)}'
  )
  print(
      f"[{time.strftime('%H:%M:%S')}] Random evaluation dataset size:"
      f' {len(random_eval_df)}'
  )
  print(
      f"[{time.strftime('%H:%M:%S')}] Separated evaluation dataset size:"
      f' {len(separated_eval_df)}'
  )

  print(
      f"\n[{time.strftime('%H:%M:%S')}] Generating Skill vs. Anomaly Plots..."
  )
  return (
      random_skill,
      separated_skill,
      gauged_skill,
      gauged_eval_df,
      random_eval_df,
      separated_eval_df,
  )


def prepare_delta_skill_data(
    random_anomaly_df,
    separated_anomaly_df,
    gauged_skill,
    random_skill,
    separated_skill,
):
  """Computes performance penalties contrasting Gauged baselines versus Regionalized drops.

  This evaluates whether predicting upon completely unseen basins causes
  disproportionate accuracy collapse when the test basin is physically
  "anomalous".

  Args:
    random_anomaly_df: DataFrame providing anomaly scores tied to our randomly
      separated models.
    separated_anomaly_df: DataFrame outlining geographic separation validation
      scores.
    gauged_skill: Ideal baseline metrics captured when network actively
      memorized regions globally.
    random_skill: Target validation outcomes tracking entirely randomized
      network inferences.
    separated_skill: Target validation outcomes capturing spatially disconnected
      predictions.

  Returns:
    tuple: Containing detailed Dataframes framing combined skills/metrics
    alongside standard calculated medians.
  """
  # We want to measure explicitly how much WORSE a hydrological model does when it's forced
  # to predict blindly (Random CV) compared to when it had data practically identical to the target (Gauged).
  # We construct subsets for evaluating this "Delta" (difference in skill).

  # Prepare Top plotting row explicitly assessing Gauged Native vs Random Extrapolation differences.
  df_top = random_anomaly_df.copy()
  df_top = df_top.join(gauged_skill, how='inner').join(
      random_skill, how='inner'
  )
  if file_paths.OUTLIER_MIN is not None:
    df_top = df_top[
        (df_top[f'gauged_{file_paths.METRIC}'] >= file_paths.OUTLIER_MIN)
        & (df_top[f'random_{file_paths.METRIC}'] >= file_paths.OUTLIER_MIN)
    ]
  if file_paths.OUTLIER_MAX is not None:
    df_top = df_top[
        (df_top[f'gauged_{file_paths.METRIC}'] <= file_paths.OUTLIER_MAX)
        & (df_top[f'random_{file_paths.METRIC}'] <= file_paths.OUTLIER_MAX)
    ]
  df_top = df_top.copy()

  # Calculate the exact difference between the "best case scenario" (gauged) and our validation limit.
  df_top['\u0394 (Gauged - Random)'] = (
      df_top[f'gauged_{file_paths.METRIC}']
      - df_top[f'random_{file_paths.METRIC}']
  )
  median_g_r = df_top['\u0394 (Gauged - Random)'].median()

  # Prepare Bottom plotting row explicitly assessing Gauged Native vs Fully Separated limits.
  df_bot = separated_anomaly_df.copy()
  df_bot = df_bot.join(gauged_skill, how='inner').join(
      separated_skill, how='inner'
  )
  if file_paths.OUTLIER_MIN is not None:
    df_bot = df_bot[
        (df_bot[f'gauged_{file_paths.METRIC}'] >= file_paths.OUTLIER_MIN)
        & (df_bot[f'separated_{file_paths.METRIC}'] >= file_paths.OUTLIER_MIN)
    ]
  if file_paths.OUTLIER_MAX is not None:
    df_bot = df_bot[
        (df_bot[f'gauged_{file_paths.METRIC}'] <= file_paths.OUTLIER_MAX)
        & (df_bot[f'separated_{file_paths.METRIC}'] <= file_paths.OUTLIER_MAX)
    ]
  df_bot = df_bot.copy()
  df_bot['\u0394 (Gauged - Separated)'] = (
      df_bot[f'gauged_{file_paths.METRIC}']
      - df_bot[f'separated_{file_paths.METRIC}']
  )
  median_g_s = df_bot['\u0394 (Gauged - Separated)'].median()

  return df_top, df_bot, median_g_r, median_g_s


# ## Spearman Correlations Analysis
#
# We compute and visualize the non-parametric Spearman rank correlation coefficients between physical anomaly scores and prediction skills/deltas. This quantifies the strength of the monotonic relationship between attribute anomaly and skill loss.
#
# ### Function Definition: calculate_summary_stats
def calculate_summary_stats(df, skill_col, delta_col=None):
  """Computes non-parametric statistical correlation strengths testing physical mapping dependencies.

  Determines formally using standard Spearman R values whether as anomaly
  severities grow,
  skills strictly decline monotonically avoiding linear normality assumptions
  logically.

  Args:
    df: Target DataFrame capturing grouped modelling outcomes tied sequentially
      matching anomalies.
    skill_col: Name identifier evaluating strict native performances logically.
    delta_col: Optional column label calculating exact skill drop penalties
      accurately.

  Returns:
    pd.DataFrame: Tabular array summarising raw Rho variables effectively mapped
    across testing domains.
  """
  # We compute and visualize the non-parametric Spearman rank correlation coefficients
  # between physical anomaly scores and prediction skills/deltas. This quantifies the strength
  # of the monotonic relationship between attribute anomaly and skill loss firmly.

  metrics = ['anomaly_score_if', 'anomaly_score_ocsvm', 'anomaly_score_lof']
  results = []

  # Validate impacts globally testing every anomaly technique available natively.
  for m in metrics:
    stat_row = {'Method': m.replace('anomaly_score_', '').upper()}
    # Determine the explicit Correlation evaluating absolute skill baseline capabilities seamlessly.
    stat_row['Corr with Skill'] = df[m].corr(df[skill_col], method='spearman')

    # Identify dynamic Correlation shifts mapping directly against validation performance gaps reliably.
    if delta_col:
      stat_row['Corr with Delta'] = df[m].corr(df[delta_col], method='spearman')

    results.append(stat_row)

  return pd.DataFrame(results)


def analyze_spearman_correlations(df_top, df_bot, gauged_eval_df):
  """Visually parses and plots comprehensive Spearman Rho tracking metrics effectively formatting printouts.

  Renders text based tables directly mapping correlation bounds cleanly while
  instantiating
  summary graphs depicting anomaly measurement power logically.

  Args:
    df_top: Array identifying baseline randomness structures mapped globally.
    df_bot: Physical distance constraint limits isolating validation metrics
      securely.
    gauged_eval_df: Complete array capturing unrestricted baseline model limits
      directly.

  Returns:
    None. Triggers text dumps logically validating pipeline assertions before
    printing a barplot chart comprehensively.
  """
  # Formally expose statistical properties highlighting direct correlations natively.
  print(f'--- Spearman Correlations: Random CV ---')
  random_stats = calculate_summary_stats(
      df_top, f'random_{file_paths.METRIC}', 'Δ (Gauged - Random)'
  )
  print(random_stats.to_string(index=False))

  # Test physical geographic exclusions effectively mapping penalties explicitly.
  print(f'\n--- Spearman Correlations: Separated CV ---')
  separated_stats = calculate_summary_stats(
      df_bot, f'separated_{file_paths.METRIC}', 'Δ (Gauged - Separated)'
  )
  print(separated_stats.to_string(index=False))

  # Track general baseline boundaries defining network limitations logically.
  print(f'\n--- Spearman Correlations: Gauged (Global) ---')
  gauged_stats = calculate_summary_stats(
      gauged_eval_df, f'gauged_{file_paths.METRIC}'
  )
  print(gauged_stats.to_string(index=False))
  # (Plotting of correlations removed as we are removing figures outside notebook 9)


def run_hypothesis_testing(df_top, df_bot, gauged_eval_df):
  """Conducts core statistical tests formally evaluating penalty significance boundaries.

  Delegates highly specific non-parametric validations against custom background
  libraries,
  synthesizing detailed statistical summaries output directly to paper drafts
  natively.

  Args:
    df_top: Dataset configuring validation tracking strictly bounding physical
      boundaries natively.
    df_bot: Array tracing separated distance validation limit thresholds
      exactly.
    gauged_eval_df: Global evaluation properties explicitly recording ideal
      configuration baselines.

  Returns:
    None. Outputs comprehensive standardized test tables summarizing evaluations
    rigorously.
  """
  # To formalize the claim that catchments with highly anomalous physical properties
  # exhibit significantly lower predictive skill compared to well-represented ones,
  # we run rigorous non-parametric statistical tests across all 9 modeling scenarios
  # using our shared stats_utils library.

  # Configure testing matrices tracking explicitly matching evaluation targets seamlessly.
  scenarios = [
      (
          'Gauged (Global Binning)',
          gauged_eval_df,
          f'gauged_{file_paths.METRIC}',
      ),
      ('Random CV (Spatial Binning)', df_top, f'random_{file_paths.METRIC}'),
      (
          'Separated CV (Extrapolation)',
          df_bot,
          f'separated_{file_paths.METRIC}',
      ),
  ]

  # Map dictionary pointers structuring accurate labeling outputs fundamentally.
  anomaly_methods = {
      'anomaly_score_if': 'Isolation Forest',
      'anomaly_score_ocsvm': 'One-Class SVM',
      'anomaly_score_lof': 'Local Outlier Factor',
  }

  # Process analytical comparisons leveraging standardized shared library functions completely.
  results_df, formatted_df, manuscript_text = (
      stats_utils.run_anomaly_significance_sweeps(
          scenarios=scenarios, anomaly_methods=anomaly_methods
      )
  )

  # Dump perfectly grouped statistical findings structured flawlessly mapping publication expectations directly.
  print('\n=== RIGOROUS HYPOTHESIS TESTING SUMMARY TABLE (NATURE STYLE) ===')
  print(formatted_df.to_string(index=False))

  # Provide raw textual outputs allowing direct copy-paste manuscript alignments natively.
  print('\n=== PAPER TEXT GENERATION HELPER ===')
  print(manuscript_text)


def main(argv):
  """Orchestrates comprehensive processing routing from data loading directly towards artifact creation.

  Controls execution flow validating input flags, organizing analytical
  checkpoints globally, and
  triggering explicit analytical plotting components linearly building towards
  rigorous evaluation targets.

  Args:
    argv: List detailing target command-line instructions capturing flag
      property bindings securely.

  Returns:
    None. Triggers complete process execution finalizing output records
    automatically onto disk seamlessly.
  """
  # Enforce rigorous argument checks matching python script execution limits tightly.
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Retrieve forced recalculation flags bypassing caching logic deliberately entirely.
  FORCE_RECALCULATE_RANDOM = FLAGS.force_recalculate_random
  FORCE_RECALCULATE_SEPARATED = FLAGS.force_recalculate_separated
  FORCE_RECALCULATE_GAUGED = FLAGS.force_recalculate_gauged
  FORCE_RECALCULATE_HYBAS_CARAVAN = FLAGS.force_recalculate_hybas_caravan

  # Determine explicitly whether recalculations must compute fresh rather than pull from pre-computed paths.
  CALC_NEW_RANDOM = FORCE_RECALCULATE_RANDOM or not gfile.Exists(
      file_paths.RANDOM_ANOMALIES_PATH
  )
  CALC_NEW_SEPARATED = FORCE_RECALCULATE_SEPARATED or not gfile.Exists(
      file_paths.SEPARATED_ANOMALIES_PATH
  )
  CALC_NEW_GAUGED = FORCE_RECALCULATE_GAUGED or not gfile.Exists(
      file_paths.GAUGED_ANOMALIES_PATH
  )
  CALC_NEW_HYBAS_CARAVAN = FORCE_RECALCULATE_HYBAS_CARAVAN or not gfile.Exists(
      file_paths.INFER_HYBAS_CARAVAN_OUT_PATH
  )

  file_paths.ensure_directories_exist()
  print(f"[{time.strftime('%H:%M:%S')}] Configuration Loaded.")

  # Ensure target visual structure exists fully capable of catching output representations natively.

  # Validate primary network input definitions perfectly preparing global processing targets.
  grdc_df, hybas_df, caravan_df = load_and_clean_data()

  # Identify targeted CV groupings securely partitioning limits explicitly properly.
  random_fold_gauges, separated_fold_gauges, all_active_grdc_gauges = (
      setup_cv_splits(grdc_df.index)
  )

  # Process spatial validation structures identifying physical boundaries intrinsically natively.
  random_anomaly_df = compute_cv_anomalies(
      random_fold_gauges,
      file_paths.RANDOM_ANOMALIES_PATH,
      CALC_NEW_RANDOM,
      'Random Ungauged',
      grdc_df,
  )

  # Re-evaluate geographical isolation dependencies measuring separation explicitly safely.
  separated_anomaly_df = compute_cv_anomalies(
      separated_fold_gauges,
      file_paths.SEPARATED_ANOMALIES_PATH,
      CALC_NEW_SEPARATED,
      'Hydrologically Separated',
      grdc_df,
  )

  # Check standard native boundaries limiting anomaly scores fully securely.
  gauged_anomaly_df = calculate_gauged_anomalies(
      grdc_df, all_active_grdc_gauges, CALC_NEW_GAUGED
  )

  # Compare inference inputs rigorously modeling HyBAS limits naturally.
  hybas_anomaly_df = calculate_hybas_inference_anomalies(
      caravan_df, hybas_df, CALC_NEW_HYBAS_CARAVAN
  )

  # Evaluate categorical distribution similarities explicitly mapping variance reliably.

  # Match true performance outputs validating anomaly limits formally comprehensively.
  try:
    (
        random_skill,
        separated_skill,
        gauged_skill,
        gauged_eval_df,
        random_eval_df,
        separated_eval_df,
    ) = load_metrics_for_eda(
        gauged_anomaly_df, random_anomaly_df, separated_anomaly_df
    )
    if (
        len(gauged_eval_df) > 0
        and len(random_eval_df) > 0
        and len(separated_eval_df) > 0
    ):
      df_top, df_bot, median_g_r, median_g_s = prepare_delta_skill_data(
          random_anomaly_df,
          separated_anomaly_df,
          gauged_skill,
          random_skill,
          separated_skill,
      )
      analyze_spearman_correlations(df_top, df_bot, gauged_eval_df)
      run_hypothesis_testing(df_top, df_bot, gauged_eval_df)
    else:
      print(
          f"[{time.strftime('%H:%M:%S')}] Skipping EDA correlation printout:"
          ' matching hydrological model metrics are not yet populated.'
      )
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"[{time.strftime('%H:%M:%S')}] Skipping optional EDA printout: {e}")


if __name__ == '__main__':
  app.run(main)
