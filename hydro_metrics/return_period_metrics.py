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

"""Utilities to calculate return period precision and recall metrics."""

import os
import pathlib
import shutil
from typing import Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import tqdm
import xarray

from . import data_paths
from . import evaluation_utils
from . import loading_utils
from . import metrics_utils
from .return_period_calculator import exceptions
from .return_period_calculator import return_period_calculator


RETURN_PERIOD_TIME_WINDOWS = [
    pd.Timedelta(0, unit='d'),
    pd.Timedelta(1, unit='d'),
    pd.Timedelta(2, unit='d'),
]


def _true_positives_fraction_in_window(
    a_crossovers: np.ndarray,
    b_crossovers: np.ndarray,
    discard_nans_in_window: bool,
    window_in_timesteps: int,
):
  """Calculates fraction of crossovers that were hit within a window."""
  a_crossover_idxs = np.where((a_crossovers != 0) & ~np.isnan(a_crossovers))[0]
  b_crossover_idxs = np.where((b_crossovers != 0) & ~np.isnan(b_crossovers))[0]
  b_nan_crossover_idxs = np.where(np.isnan(b_crossovers))[0]

  total_count = 0
  true_positives = 0
  for a_idx in a_crossover_idxs:
    if discard_nans_in_window and np.any(
        np.abs(b_nan_crossover_idxs - a_idx) <= window_in_timesteps + 1e-6
    ):
      continue
    total_count += 1
    if np.any(np.abs(b_crossover_idxs - a_idx) <= window_in_timesteps + 1e-6):
      true_positives += 1

  if total_count > 0:
    return true_positives / total_count
  else:
    return np.nan


def _single_return_period_performance_metric(
    observations: np.ndarray,
    simulations: np.ndarray,
    obs_return_period_calculator: Optional[
        return_period_calculator.ReturnPeriodCalculator
    ],
    sim_return_period_calculator: Optional[
        return_period_calculator.ReturnPeriodCalculator
    ],
    return_period: float,
    window_in_timesteps: int,
) -> Tuple[Mapping[str, float], float]:
  """Calculates hit/miss rates for a single return period and time window."""

  if sim_return_period_calculator:
    sim_flow_value = sim_return_period_calculator.flow_value_from_return_period(
        return_period)
  else:
    sim_flow_value = np.nan

  if obs_return_period_calculator:
    obs_flow_value = obs_return_period_calculator.flow_value_from_return_period(
        return_period)
  else:
    return {'precision': np.nan, 'recall': np.nan}, sim_flow_value

  above_threshold = np.where(
      np.isnan(observations),
      np.nan,
      (observations >= obs_flow_value).astype(float),
  )
  obs_crossovers = np.maximum(0, np.diff(above_threshold))

  above_threshold = np.where(
      np.isnan(simulations),
      np.nan,
      (simulations >= sim_flow_value).astype(float),
  )
  sim_crossovers = np.maximum(0, np.diff(above_threshold))

  precision = _true_positives_fraction_in_window(
      sim_crossovers, obs_crossovers, True, window_in_timesteps
  )
  
  recall = _true_positives_fraction_in_window(
      obs_crossovers, sim_crossovers, False, window_in_timesteps
  )

  return {'precision': precision, 'recall': recall}, sim_flow_value


def calculate_return_period_performance_metrics(
    observations: pd.Series,
    predictions: pd.Series,
    temporal_resolution: str = '1D',
    evaluation_time_period: Optional[list[str]] = None,
) -> Tuple[Mapping[str, float], Mapping[str, float]]:
  """Calculates metrics for cross-product of return periods and time windows."""

  temporal_resolution_dt = pd.Timedelta(temporal_resolution)

  if not isinstance(predictions.index, pd.DatetimeIndex):
    predictions.index = [
        pd.to_datetime(t, unit='s', origin='unix') if isinstance(t, (int, float)) else pd.to_datetime(t)
        for t in predictions.index
    ]
  if not isinstance(observations.index, pd.DatetimeIndex):
    observations.index = [
        pd.to_datetime(t, unit='s', origin='unix') if isinstance(t, (int, float)) else pd.to_datetime(t)
        for t in observations.index
    ]

  try:
    obs_return_period_calculator = return_period_calculator.ReturnPeriodCalculator(
        hydrograph_series=observations,
        hydrograph_series_frequency=temporal_resolution_dt,
        use_simple_fitting=True,
        verbose=False,
    )
  except exceptions.NotEnoughDataError:
    obs_return_period_calculator = None

  try:
    sim_return_period_calculator = return_period_calculator.ReturnPeriodCalculator(
        hydrograph_series=predictions,
        hydrograph_series_frequency=temporal_resolution_dt,
        use_simple_fitting=True,
        verbose=False,
    )
  except exceptions.NotEnoughDataError:
    sim_return_period_calculator = None

  start_date = min([min(observations.index), min(predictions.index)])
  end_date = max([max(observations.index), max(predictions.index)])
  new_date_range = pd.date_range(
      start=start_date,
      end=end_date,
      freq=temporal_resolution_dt,
  )

  observations = observations[~observations.index.duplicated(keep='first')]
  predictions = predictions[~predictions.index.duplicated(keep='first')]
  observations = observations.reindex(new_date_range, fill_value=np.nan)
  predictions = predictions.reindex(new_date_range, fill_value=np.nan)

  if evaluation_time_period is not None:
    obs = observations.loc[slice(*evaluation_time_period)]
    sim = predictions.loc[slice(*evaluation_time_period)]
  else:
    obs = observations.copy()
    sim = predictions.copy()

  return_period_metrics = {}
  sim_return_period_dict = {}
  for return_period in evaluation_utils.RETURN_PERIODS:
    for window in RETURN_PERIOD_TIME_WINDOWS:
      window_in_timesteps = int(window / temporal_resolution_dt)

      metrics_dict, sim_return_period_dict[
          return_period] = _single_return_period_performance_metric(
              observations=obs.values,
              simulations=sim.values,
              obs_return_period_calculator=obs_return_period_calculator,
              sim_return_period_calculator=sim_return_period_calculator,
              return_period=return_period,
              window_in_timesteps=window_in_timesteps,
          )
    
      metric_name_base = f'return_period_{return_period}_window_{window_in_timesteps}_'
      return_period_metrics.update({
          metric_name_base + metric_name: metric_value
          for metric_name, metric_value in metrics_dict.items()
      })

  flow_values_from_return_periods = {
      f'return_period_{return_period}': flow_value
      for return_period, flow_value in sim_return_period_dict.items()
  }

  return return_period_metrics, flow_values_from_return_periods


def calculate_and_save_metrics_for_one_gauge(
    ds: xarray.Dataset,
    gauge: str,
    sim_variable: str,
    obs_variable: str,
    base_path: pathlib.Path,
    temporal_resolution: str = '1D',
    lead_times: list[int] = data_paths.LEAD_TIMES,
    distribution_time_period: Optional[list[str]] = None,
    evaluation_time_period: Optional[list[str]] = None,
    path_modifier: Optional[str] = None,
) -> None:
  """Calculates return period precision and recall for one gauge."""
  if gauge not in ds.gauge_id:
    return

  resolution_dt = pd.Timedelta(temporal_resolution)
  return_period_time_windows = [int(w / resolution_dt) for w in RETURN_PERIOD_TIME_WINDOWS]

  precision_metrics = pd.DataFrame(
      index=pd.MultiIndex.from_product(
          [evaluation_utils.RETURN_PERIODS, return_period_time_windows]),
      columns=lead_times
  )
  recall_metrics = pd.DataFrame(
      index=pd.MultiIndex.from_product(
          [evaluation_utils.RETURN_PERIODS, return_period_time_windows]),
      columns=lead_times
  )

  for lead_time in lead_times:
    if 'lead_time' in ds.dims:
      sim = ds.sel(gauge_id=gauge, lead_time=lead_time)[sim_variable].to_series()
      obs = ds.sel(gauge_id=gauge, lead_time=lead_time)[obs_variable].to_series()
    else:
      sim = ds.sel(gauge_id=gauge)[sim_variable].to_series()
      obs = ds.sel(gauge_id=gauge)[obs_variable].to_series()

    if distribution_time_period is not None:
      obs = obs.loc[slice(*distribution_time_period)]
      sim = sim.loc[slice(*distribution_time_period)]

    metrics_dict, _ = calculate_return_period_performance_metrics(
        predictions=sim,
        observations=obs,
        evaluation_time_period=evaluation_time_period,
    )

    for rp in evaluation_utils.RETURN_PERIODS:
      for w in return_period_time_windows:
        metric_name = f'return_period_{rp}_window_{w}_precision'
        precision_metrics.loc[(rp, w), lead_time] = metrics_dict.get(metric_name, np.nan)
        metric_name = f'return_period_{rp}_window_{w}_recall'
        recall_metrics.loc[(rp, w), lead_time] = metrics_dict.get(metric_name, np.nan)

  metrics_utils.save_metrics_df(
      df=recall_metrics,
      metric=gauge,
      base_path=base_path,
      path_modifier=f'{path_modifier}/recall' if path_modifier else 'recall',
  )
  metrics_utils.save_metrics_df(
      df=precision_metrics,
      metric=gauge,
      base_path=base_path,
      path_modifier=f'{path_modifier}/precision' if path_modifier else 'precision',
  )


def compute_metrics(
    restart: bool,
    working_path: pathlib.Path,
    experiments: list[str],
    gauge_list: list[str],
    ds_dict: dict[str, xarray.Dataset],
    sim_variable: str,
    obs_variable: str,
    evaluation_time_periods: Optional[dict] = None,
    lead_times: Optional[list[int]] = None
) -> dict[str, list[str]]:
  """Computes return period metrics across experiments and gauges."""
  if lead_times is None:
    lead_times = data_paths.LEAD_TIMES

  if os.path.exists(working_path) and restart:
    shutil.rmtree(working_path)
  loading_utils.create_remote_folder_if_necessary(working_path)

  if evaluation_time_periods is None:
    evaluation_time_periods = {gauge: None for gauge in gauge_list}

  missing_gauges = {experiment: [] for experiment in experiments}
  for experiment in experiments:
    print(f'Working on experiment: {experiment} ...')
    for gauge in tqdm.tqdm(gauge_list):
      try:
        calculate_and_save_metrics_for_one_gauge(
            ds=ds_dict[experiment],
            gauge=gauge,
            sim_variable=sim_variable,
            obs_variable=obs_variable,
            base_path=working_path,
            path_modifier=experiment,
            evaluation_time_period=evaluation_time_periods.get(gauge),
            lead_times=lead_times,
        )
      except Exception as e:
        missing_gauges[experiment].append(gauge)

  return missing_gauges
