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

"""Utilities for calculating and saving performance metrics across model runs."""

import os
import pathlib
from typing import Optional

import pandas as pd
import tqdm
import xarray

from . import data_paths
from . import loading_utils
from . import metrics

OBS_VARIABLE = 'observation'
GOOGLE_VARIABLE = 'google_prediction'
GLOFAS_VARIABLE = 'glofas_prediction'

METRICS = [
    'MSE',
    'RMSE',
    'NSE',
    'log-NSE',
    'Beta-NSE',
    'Alpha-NSE',
    'KGE',
    'log-KGE',
    'Pearson-r',
    'Beta-KGE',
    'Peak-MAPE',
    'FLV',
    'FHV',
    'FMS',
]


def save_metrics_df(
    df: Optional[pd.DataFrame],
    metric: str,
    base_path: pathlib.Path,
    path_modifier: Optional[str] = None,
) -> pathlib.Path:
  """Saves a metrics dataframe and returns the path."""
  path = base_path
  if path_modifier is not None:
    path = path / path_modifier

  loading_utils.create_remote_folder_if_necessary(path)
  filepath = path / f'{metric}.csv'

  if df is None:
    return filepath

  loading_utils.create_remote_folder_if_necessary(path)
  with open(filepath, 'w') as f:
    df.to_csv(f)

  return filepath


def load_metrics_df(
    filepath: pathlib.Path,
) -> Optional[pd.DataFrame]:
  """Loads a metrics dataframe."""
  if not os.path.exists(filepath):
    raise ValueError('Metrics file does not exist at path: ', filepath)

  with open(filepath, 'r') as f:
    return pd.read_csv(f, index_col=0)


def load_metrics_for_experiment(
    metric: str,
    experiment: str,
    base_path: pathlib.Path,
):
  filepath = save_metrics_df(
      df=None,
      metric=metric,
      base_path=base_path,
      path_modifier=experiment,
  )
  return load_metrics_df(filepath=filepath)


def _metrics_for_gauge_time_period_and_lead_time(
    ds: xarray.Dataset,
    gauge: str,
    lead_time: int,
    metrics_to_calculate: list[str],
    sim_variable: str,
    obs_variable: str,
    time_period: Optional[list[str]] = None,
) -> dict[str, float]:
  """Pulls sim and obs & calculate metrics from Dataset."""
  obs = ds[obs_variable]
  sim = ds[sim_variable]

  if 'lead_time' in obs.dims:
    obs = obs.sel(lead_time=lead_time)
    if 'lead_time' in obs.coords:
      obs = obs.drop_vars(['lead_time'])
  if 'lead_time' in sim.dims:
    sim = sim.sel(lead_time=lead_time)
    if 'lead_time' in sim.coords:
      sim = sim.drop_vars(['lead_time'])

  if 'gauge_id' in sim.dims:
    sim = sim.sel(gauge_id=gauge)
  if 'gauge_id' in obs.dims:
    obs = obs.sel(gauge_id=gauge)

  if time_period is not None:
    obs = obs.sel(time=slice(*time_period))
    sim = sim.sel(time=slice(*time_period))

  if 'gauge_id' in obs.coords:
    obs = obs.drop_vars(['gauge_id'])
  if 'gauge_id' in sim.coords:
    sim = sim.drop_vars(['gauge_id'])

  return metrics.calculate_metrics(
      obs=obs,
      sim=sim,
      metrics=metrics_to_calculate,
      resolution='1D',
  )


def _metrics_for_one_gauge(
    ds: xarray.Dataset,
    gauge: str,
    sim_variable: str,
    obs_variable: str,
    metrics_to_calculate: list[str],
    lead_times: list[int] = data_paths.LEAD_TIMES,
    time_period: Optional[list[str]] = None,
) -> pd.DataFrame:
  """Calculates metrics for one gauge across lead times."""
  series_at_lead_time = []
  for lead_time in lead_times:
    metrics_dict = _metrics_for_gauge_time_period_and_lead_time(
        ds=ds,
        gauge=gauge,
        lead_time=lead_time,
        metrics_to_calculate=metrics_to_calculate,
        sim_variable=sim_variable,
        obs_variable=obs_variable,
        time_period=time_period,
    )
    series_at_lead_time.append(
        pd.Series(
            metrics_dict.values(),
            index=metrics_dict.keys(),
            name=lead_time,
        )
    )

  return pd.concat(series_at_lead_time, axis=1)


def calculate_and_save_metrics_for_one_gauge(
    ds: xarray.Dataset,
    gauge: str,
    sim_variable: str,
    obs_variable: str,
    base_path: pathlib.Path,
    metrics_to_calculate: Optional[list[str]] = None,
    lead_times: list[int] = data_paths.LEAD_TIMES,
    time_period: Optional[list[str]] = None,
    path_modifier: Optional[str] = None,
) -> pd.DataFrame:
  """Calculates standard hydro performance metrics for a gauge and saves CSVs."""
  if metrics_to_calculate is None:
    metrics_to_calculate = METRICS

  if gauge not in ds.gauge_id:
    return pd.DataFrame()

  df = _metrics_for_one_gauge(
      ds=ds,
      gauge=gauge,
      sim_variable=sim_variable,
      obs_variable=obs_variable,
      metrics_to_calculate=metrics_to_calculate,
      lead_times=lead_times,
      time_period=time_period,
  )

  save_metrics_df(
      df=df,
      metric=gauge,
      base_path=base_path,
      path_modifier=path_modifier,
  )
  return df
