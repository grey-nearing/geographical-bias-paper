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


"""Global Training and Inference for Skill Predictor Models.

This script trains the final skill prediction model on the full available
dataset
(Gauged or Ungauged configuration) and then runs global inference over all HyBAS
catchments.

Main Steps:
1. Initialize configuration flags and centralized paths (via file_paths.init).
2. Load and preprocess training features and target skills (calls
load_and_preprocess_data).
3. Train the final stacked model (SkillPredictor) on the full dataset.
4. Serialize and save the trained model.
5. Load global HyBAS catchment inference features.
6. Run inference to predict performance metrics globally.
7. Save global predictions.

Required Input Files:
- Gauged Features (if model_type='gauged'): file_paths.GAUGED_GENERIC_DATA_PATH
- Ungauged Features (if model_type='ungauged'):
file_paths.UNGAUGED_GENERIC_DATA_PATH
- Target Skills: file_paths.METRICS_DIR/{model_type}/{metric}.csv
  (Note: For ungauged model_type, both 'random' and 'separated' targets are
  loaded and combined)
- Global HyBAS Features (for global inference):
file_paths.HYBAS_GENERIC_DATA_PATH

Saved Output Files:
- Final Serialized Model: file_paths.FINAL_GAUGED_MODEL (gauged) or
file_paths.FINAL_UNGAUGED_MODEL (ungauged)
- Global Predictions: file_paths.GLOBAL_HYBAS_PREDICTIONS_GAUGED (gauged) or
file_paths.GLOBAL_HYBAS_PREDICTIONS_UNGAUGED (ungauged)
"""

import pickle
import time
import warnings

from absl import app
from absl import flags
import pandas as pd

from utils import file_paths
from utils import io_utils
from utils.io_utils import load_and_preprocess_data
from utils.skill_predictor import SkillPredictor
from utils import gfile_compat as gfile

# We suppress specific warning messages (like those for deprecated features) 
# to ensure the console output remains readable and focused on our application's progress.
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# We use the absl flags library to allow users to configure the script's behavior
# from the command line (e.g., passing `--metric=KGE` when running the script).
flags.DEFINE_string(
    'metric',
    'NSE',
    'The target performance metric to predict (e.g. NSE, KGE).',
)
flags.DEFINE_enum(
    'model_type',
    'gauged',
    ['gauged', 'ungauged'],
    'Whether to run gauged (local) or ungauged (regionalized) models.',
)
flags.DEFINE_integer(
    'sample_size',
    None,
    'Set to an integer to limit datasets for debugging, or None for full run.',
)
FLAGS = flags.FLAGS


def main(argv):
  """Executes the global training and inference pipeline.

  This main function acts as the entry point for the script. It sets up 
  the file paths, loads the training data, trains the skill prediction model, 
  and finally applies this model to predict over a global dataset.

  Args:
    argv: A list of command-line arguments passed to the script.

  Returns:
    None
  """
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  script_start_time = time.time()
  script_wall_start = time.strftime('%H:%M:%S')

  metric = FLAGS.metric
  model_type = FLAGS.model_type
  sample_size = FLAGS.sample_size
  if sample_size == 0:
    sample_size = None

  # Here we dynamically configure our file paths. By passing the chosen metric,
  # the file_paths module automatically sets up the correct locations to read
  # the target data and where to save the final models and predictions.
  file_paths.init(metric)

  print(
      f'[{script_wall_start}] Starting Global Train & Inference Script for'
      f' Metric: {metric} | Model Type: {model_type}'
  )

  # ----------------------------------------------------------------------------
  # Step 0: Load and Preprocess the Training Data
  # In this section, we retrieve the historical features and the known target 
  # performance metrics for our catchments so the model can learn from them.
  # ----------------------------------------------------------------------------
  x_train, y_train = load_and_preprocess_data(model_type, metric, sample_size)
  print(
      f'   -> Training features shape: {x_train.shape} | Targets shape:'
      f' {y_train.shape}'
  )

  if model_type == 'gauged':
    model_save_path = file_paths.FINAL_GAUGED_MODEL
    hybas_preds_save_path = file_paths.GLOBAL_HYBAS_PREDICTIONS_GAUGED
    pred_col_name = 'gauged_skill'
  else:
    model_save_path = file_paths.FINAL_UNGAUGED_MODEL
    hybas_preds_save_path = file_paths.GLOBAL_HYBAS_PREDICTIONS_UNGAUGED
    pred_col_name = 'ungauged_skill'

  # ----------------------------------------------------------------------------
  # Step 1: Train the Core Model
  # Now that the data is ready, we initialize the SkillPredictor (which encapsulates
  # our machine learning logic, like an XGBoost pipeline) and fit it against our 
  # training examples. This is where the actual "learning" takes place.
  # ----------------------------------------------------------------------------
  print(f"\n[{time.strftime('%H:%M:%S')}] Training final model...")
  model = SkillPredictor(metric)
  train_start = time.time()
  train_wall_start = time.strftime('%H:%M:%S')
  model.fit(x_train, y_train)
  print(
      f"[{time.strftime('%H:%M:%S')}] Final model trained in"
      f' {time.time() - train_start:.2f}s (Wall time: {train_wall_start} to'
      f" {time.strftime('%H:%M:%S')})."
  )

  # ----------------------------------------------------------------------------
  # Step 2: Serialize internal state
  # We serialize (or "pickle") the fully trained model to disk. This allows
  # other scripts or future runs to reuse the model without retraining from scratch.
  # ----------------------------------------------------------------------------
  save_start = time.time()
  print(
      f"[{time.strftime('%H:%M:%S')}] Saving serialized model to"
      f' {model_save_path}...'
  )
  file_paths.ensure_directories_exist()
  with gfile.Open(model_save_path, 'wb') as f:
    pickle.dump(model, f)
  print(f'   -> Saved in {time.time() - save_start:.2f}s')

  # ----------------------------------------------------------------------------
  # Step 3: Loading Global Features
  # To make predictions across the entire world, we load the "HyBAS" dataset, 
  # which contains the geographic and climate features for every global catchment.
  # ----------------------------------------------------------------------------
  hybas_zarr_path = file_paths.HYBAS_GENERIC_DATA_ZARR_PATH
  print(
      f"\n[{time.strftime('%H:%M:%S')}] Loading HyBAS Generic Inference"
      f' Dataset from Zarr: {hybas_zarr_path}...'
  )
  load_start = time.time()
  load_wall_start = time.strftime('%H:%M:%S')
  df_hybas = io_utils.load_hybas_zarr(hybas_zarr_path)
  print(
      f'   -> Loaded in {time.time() - load_start:.2f}s | Shape:'
      f' {df_hybas.shape} (Wall: {load_wall_start} to'
      f" {time.strftime('%H:%M:%S')})"
  )

  if sample_size and len(df_hybas) > sample_size:
    df_hybas = df_hybas.sample(n=sample_size, random_state=42)
    print(f'   -> Subsampled HyBAS to shape: {df_hybas.shape}')

  # Machine learning models expect the input columns to be in exactly the same 
  # order, and contain the same names, as they did during training. We align 
  # the upcoming HyBAS dataset with the training columns to prevent shape mismatch errors.
  x_infer = df_hybas.reindex(columns=x_train.columns, fill_value=0)
  print(f'   -> Inference features shape: {x_infer.shape}')

  # ----------------------------------------------------------------------------
  # Step 4: Run Global Inference & Save Results
  # Finally, we ask our trained model to predict the targeted performance metric 
  # (e.g., NSE scores) for all the global catchments, then save the results.
  # ----------------------------------------------------------------------------
  print(
      f"\n[{time.strftime('%H:%M:%S')}] Running inference on global HyBAS"
      ' catchments...'
  )
  pred_start = time.time()
  pred_wall_start = time.strftime('%H:%M:%S')
  preds = model.predict(x_infer)
  print(
      f"[{time.strftime('%H:%M:%S')}] Inference complete in"
      f' {time.time() - pred_start:.2f}s (Wall: {pred_wall_start} to'
      f" {time.strftime('%H:%M:%S')})."
  )

  # Write out the predicted global scores to a CSV file for analytical usage 
  # down the line (e.g., generating global result maps/figures in notebooks).
  save_start = time.time()
  print(
      f"[{time.strftime('%H:%M:%S')}] Saving predictions to"
      f' {hybas_preds_save_path}...'
  )
  preds_df = preds.rename(pred_col_name).to_frame()
  with gfile.Open(hybas_preds_save_path, 'wt') as f:
    preds_df.to_csv(f, index=True)
  print(f'   -> Saved in {time.time() - save_start:.2f}s')

  print(
      f"\n[{time.strftime('%H:%M:%S')}] Script finished in"
      f' {time.time() - script_start_time:.2f}s (Wall time: {script_wall_start}'
      f" to {time.strftime('%H:%M:%S')})"
  )


if __name__ == '__main__':
  app.run(main)
