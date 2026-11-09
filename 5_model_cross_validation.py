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


"""Cross-Validation Analysis for Skill Predictor Models.

This script executes K-Fold cross-validation for skill prediction models (both
Gauged and Ungauged model types). It loads generic pre-computed physical/social
catchment features, aligns them with target hydrological model performance
metrics (skills), applies outlier clamping, trains a stacked regression model
(integrating LightGBM, XGBoost, and HistGradientBoosting), and evaluates
performance.

Main Steps:
1. Initialize configuration flags and centralized paths (via file_paths.init).
2. Load training features and target skill metrics (calls
load_and_preprocess_data).
3. Preprocess data (inner join, drop NaNs, optional subsampling, apply outlier
clamping).
4. Perform K-Fold cross-validation (train stacked model on K-1 folds, predict on
out-of-fold split).
5. Save out-of-fold predictions.
6. Evaluate and print final regression metrics (R², MAE, RMSE) to the screen.

Required Input Files:
- Gauged Features (if model_type='gauged'): file_paths.GAUGED_GENERIC_DATA_PATH
- Ungauged Features (if model_type='ungauged'):
file_paths.UNGAUGED_GENERIC_DATA_PATH
- Target Skills: file_paths.METRICS_DIR/{model_type}/{metric}.csv
  (Note: For ungauged model_type, both 'random' and 'separated' targets are
  loaded and combined)

Saved Output Files:
- Gauged CV Predictions (if model_type='gauged'):
file_paths.GAUGED_REGRESSION_CV_PREDS
- Ungauged CV Predictions (if model_type='ungauged'):
file_paths.UNGAUGED_REGRESSION_CV_PREDS
"""

import time
import warnings

from absl import app
from absl import flags
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from utils import file_paths
from utils.io_utils import load_and_preprocess_data
from utils.skill_predictor import SkillPredictor
from utils import gfile_compat as gfile

# Suppress third-party warnings to provide a clean execution log. Deprecation 
# and future warnings are common from external dependencies but unactionable here.
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Define customizable command-line flags managed by Abseil so that users can 
# easily modify parameters (like metric type or fold count) without changing code.
flags.DEFINE_string(
    'metric',
    'NSE',
    'The target performance metric to predict (e.g. NSE, KGE).',
)
flags.DEFINE_enum(
    'model_type',
    'gauged',
    ['gauged', 'ungauged'],
    'Whether to run the gauged or ungauged model.',
)
flags.DEFINE_integer(
    'cv_folds',
    10,
    'Number of splits (folds) to use for K-Fold cross-validation.',
)
flags.DEFINE_integer(
    'sample_size',
    None,
    'Integer limit on dataset size for rapid debugging.',
)
# Expose the defined command-line configurations directly under `FLAGS`
FLAGS = flags.FLAGS


def main(argv):
  """Executes the cross-validation pipeline for predicting hydrological skill.

  This main function loads the requisite datasets based on user configurations,
  sets up a K-Fold cross-validation loop to fairly evaluate the regressor, and
  outputs performance metrics and prediction files to disk.

  Args:
    argv: A list of command-line arguments. The first element is the script name.
      Any extra elements signify unrecognized parameters.

  Returns:
    None. Outputs are written to standard out and the filesystem.
  """
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  script_start_time = time.time()
  script_wall_start = time.strftime('%H:%M:%S')

  metric = FLAGS.metric
  model_type = FLAGS.model_type
  cv_folds = FLAGS.cv_folds
  sample_size = FLAGS.sample_size
  if sample_size == 0:
    sample_size = None

  # Configure the global outputs directory using the currently active metric 
  # to segregate generated performance logs and files correctly.
  file_paths.init(metric)

  print(
      f'[{script_wall_start}] Starting CV Script for Metric: {metric} | Model'
      f' Type: {model_type}'
  )

  # ----------------------------------------------------------------------------
  # Load & Preprocess Data: Establish foundational datasets. These operations 
  # ingest local CSVs representing physical characteristics and target labels.
  # ----------------------------------------------------------------------------
  x, y = load_and_preprocess_data(model_type, metric, sample_size)
  print(f'   -> Features shape: {x.shape} | Targets shape: {y.shape}')

  # Dynamically assign export paths and report headers depending on if we are 
  # evaluating completely ungauged areas, or well-monitored gauged sections.
  if model_type == 'gauged':
    cv_save_path = file_paths.GAUGED_REGRESSION_CV_PREDS
    title_prefix = 'Gauged Regression'
  else:
    cv_save_path = file_paths.UNGAUGED_REGRESSION_CV_PREDS
    title_prefix = 'Ungauged Regression'

  # ----------------------------------------------------------------------------
  # Run Cross-Validation: Partition the dataset into randomly split folds to 
  # minimize potential model overfitting biases through rotating hold-out sets.
  # ----------------------------------------------------------------------------
  print(f"\n[{time.strftime('%H:%M:%S')}] Running CV Estimation...")
  cv_start_time = time.time()
  cv_wall_start = time.strftime('%H:%M:%S')
  
  # Configure the KFold cross-validator with random shuffling enabled to 
  # ensure evenly distributed, relatively independent data selections per fold.
  kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
  oof_preds, oof_true = [], []

  for fold, (train_index, test_index) in enumerate(kf.split(x)):
    fold_start_time = time.time()
    fold_wall_start = time.strftime('%H:%M:%S')

    # Subselect our training and testing data splits dynamically for this fold.
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Instantiate, configure, and fully train our predictor pipeline architecture.
    fold_model = SkillPredictor(metric)
    fit_start = time.time()
    fold_model.fit(x_train, y_train)
    fit_time = time.time() - fit_start

    # Run inference across our holdout test set so we can record performance.
    pred_start = time.time()
    preds = fold_model.predict(x_test)
    pred_time = time.time() - pred_start

    # Cumulate inference outcomes and reference truth markers to aggregate later.
    oof_preds.append(preds)
    oof_true.append(y_test)
    print(
        f'   - Fold {fold + 1:02d}/{cv_folds} | Fit: {fit_time:.2f}s | Predict:'
        f' {pred_time:.2f}s | Total: {time.time() - fold_start_time:.2f}s'
        ' (Wall:'
        f" {fold_wall_start} to {time.strftime('%H:%M:%S')})"
    )

  all_true = pd.concat(oof_true)
  all_preds = pd.concat(oof_preds)
  print(
      f"[{time.strftime('%H:%M:%S')}] CV Loop complete in"
      f' {time.time() - cv_start_time:.2f}s (Wall time: {cv_wall_start} to'
      f" {time.strftime('%H:%M:%S')})"
  )

  # Consolidate all hold-out (out-of-fold) predictions alongside their actual 
  # true target values into a local summary file for subsequent plotting/analysis.
  save_start_time = time.time()
  print(
      f"\n[{time.strftime('%H:%M:%S')}] Saving CV predictions to"
      f' {cv_save_path}...'
  )
  file_paths.ensure_directories_exist()
  with gfile.Open(cv_save_path, 'wt') as f:
    pd.DataFrame({'true': all_true, 'pred': all_preds}).to_csv(f)
  print(f'   -> Saved in {time.time() - save_start_time:.2f}s')

  # ----------------------------------------------------------------------------
  # Evaluate Metrics: Compute critical overall statistical goodness-of-fit 
  # indicators (like Coefficient of Determination) referencing every data point.
  # ----------------------------------------------------------------------------
  eval_start_time = time.time()
  print(f"\n[{time.strftime('%H:%M:%S')}] Evaluating metrics...")
  r2 = r2_score(all_true, all_preds)
  mae = mean_absolute_error(all_true, all_preds)
  rmse = np.sqrt(mean_squared_error(all_true, all_preds))
  print(f'   -> Evaluated in {time.time() - eval_start_time:.4f}s')

  print(f'\n--- {title_prefix} Overall Estimation Metrics ---')
  print(f'R2: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}')

  print(
      f"\n[{time.strftime('%H:%M:%S')}] Script finished in"
      f' {time.time() - script_start_time:.2f}s (Wall time: {script_wall_start}'
      f" to {time.strftime('%H:%M:%S')})"
  )


if __name__ == '__main__':
  app.run(main)
