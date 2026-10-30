# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Master runner to recalculate hydro model performance metrics and return period metrics.

Runs identical metric calculations to the Nature paper (Nearing et al., 2024 /
global_streamflow_model_paper repository).
"""

import argparse
import pathlib
import sys
from typing import Optional

import pandas as pd
import tqdm
import xarray as xr

from . import data_paths
from . import loading_utils
from . import metrics_utils
from . import return_period_metrics


def run_recalculate_metrics(
    ds: xr.Dataset,
    output_dir: pathlib.Path,
    experiment_name: str = 'full_run',
    sim_variable: str = 'google_prediction',
    obs_variable: str = 'observation',
    gauge_list: Optional[list[str]] = None,
    lead_times: Optional[list[int]] = None,
    calc_hydrograph: bool = True,
    calc_return_periods: bool = True,
) -> None:
  """Recalculates hydro performance and return period metrics for a dataset.

  Args:
    ds: xarray Dataset containing predictions and observations.
    output_dir: Root output directory to store metrics CSV files.
    experiment_name: Identifier for the experiment run.
    sim_variable: Name of prediction variable in ds.
    obs_variable: Name of observation variable in ds.
    gauge_list: Optional list of gauge IDs to process.
    lead_times: Optional list of lead times.
    calc_hydrograph: Whether to calculate standard hydrograph metrics (NSE, KGE, etc.).
    calc_return_periods: Whether to calculate return period precision/recall metrics.
  """
  loading_utils.create_remote_folder_if_necessary(output_dir)

  if gauge_list is None:
    if 'gauge_id' in ds.dims or 'gauge_id' in ds.coords:
      gauge_list = [str(g) for g in ds.gauge_id.values]
    else:
      raise ValueError('No gauge_id dimension/coordinate found in dataset.')

  if lead_times is None:
    lead_times = data_paths.LEAD_TIMES

  print(f'Starting metric recalculations for experiment "{experiment_name}"')
  print(f'Gauges count: {len(gauge_list)}, Lead times: {lead_times}')
  print(f'Output directory: {output_dir}')

  # 1. Calculate Standard Hydrograph Performance Metrics
  if calc_hydrograph:
    print('\n--- Recalculating Standard Hydrograph Metrics (NSE, KGE, RMSE, etc.) ---')
    hydro_output_path = output_dir / 'hydrograph_metrics'
    for gauge in tqdm.tqdm(gauge_list, desc='Hydrograph metrics'):
      metrics_utils.calculate_and_save_metrics_for_one_gauge(
          ds=ds,
          gauge=gauge,
          sim_variable=sim_variable,
          obs_variable=obs_variable,
          base_path=hydro_output_path,
          lead_times=lead_times,
          path_modifier=experiment_name,
      )

  # 2. Calculate Return Period Performance Metrics (Bulletin 17c methodology)
  if calc_return_periods:
    print('\n--- Recalculating Return Period Performance Metrics (Precision/Recall) ---')
    rp_output_path = output_dir / 'return_period_metrics'
    for gauge in tqdm.tqdm(gauge_list, desc='Return period metrics'):
      return_period_metrics.calculate_and_save_metrics_for_one_gauge(
          ds=ds,
          gauge=gauge,
          sim_variable=sim_variable,
          obs_variable=obs_variable,
          base_path=rp_output_path,
          lead_times=lead_times,
          path_modifier=experiment_name,
      )

  print('\nMetric recalculation completed successfully.')


def main():
  parser = argparse.ArgumentParser(
      description='Recalculate Hydro Performance & Return Period Metrics'
  )
  parser.add_argument(
      '--dataset_path',
      type=str,
      required=True,
      help='Path to NetCDF/Zarr dataset containing predictions and observations.',
  )
  parser.add_argument(
      '--output_dir',
      type=str,
      default=str(data_paths.METRICS_DIR),
      help='Directory to save output metric CSV files.',
  )
  parser.add_argument(
      '--experiment_name',
      type=str,
      default='full_run',
      help='Experiment identifier.',
  )
  parser.add_argument(
      '--sim_variable',
      type=str,
      default='google_prediction',
      help='Variable name for predictions.',
  )
  parser.add_argument(
      '--obs_variable',
      type=str,
      default='observation',
      help='Variable name for observations.',
  )
  args = parser.parse_args()

  dataset_path = pathlib.Path(args.dataset_path)
  output_dir = pathlib.Path(args.output_dir)

  print(f'Loading dataset from {dataset_path}...')
  if dataset_path.suffix == '.zarr' or dataset_path.is_dir():
    ds = xr.open_zarr(dataset_path)
  else:
    ds = xr.open_dataset(dataset_path)

  run_recalculate_metrics(
      ds=ds,
      output_dir=output_dir,
      experiment_name=args.experiment_name,
      sim_variable=args.sim_variable,
      obs_variable=args.obs_variable,
  )


if __name__ == '__main__':
  main()
