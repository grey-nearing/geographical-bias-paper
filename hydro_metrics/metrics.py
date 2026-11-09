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

"""Definitions of performance metrics for hydrological models."""
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from scipy import signal
import xarray
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset


_MIN_DATA_FOR_CALCULATING_METRICS = 365*2

_METRICS = [
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
    'Peak-Timing',
    'Missed-Peaks',
    'Peak-MAPE',
    'FLV',
    'FHV',
    'FMS',
]

_UNIFORM_WEIGHTS = [1., 1., 1.]


def get_frequency_factor(freq_one: str, freq_two: str) -> float:
  if freq_one == freq_two:
    return 1

  offset_one = to_offset(freq_one)
  offset_two = to_offset(freq_two)
  if offset_one.n < 0 or offset_two.n < 0:
    raise NotImplementedError('Cannot compare negative frequencies.')
  if offset_one.n == offset_two.n == 0:
    return 1
  if offset_two.n == 0:
    return np.inf
  if offset_one.name == offset_two.name:
    return offset_one.n / offset_two.n

  factor = None
  regex_month_or_day = '-(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC|MON|TUE|WED|THU|FRI|SAT|SUN)$'
  for i, (one, two) in enumerate(
      [(offset_one, offset_two), (offset_two, offset_one)]):
    name_one = re.sub(regex_month_or_day, '', one.name)
    name_two = re.sub(regex_month_or_day, '', two.name)
    if ((name_one in ['A', 'Y'] and name_two == 'M') or
        (name_one in ['AS', 'YS'] and name_two == 'MS')):
      factor = 12 * one.n / two.n
    if ((name_one in ['A', 'Y'] and name_two == 'Q') or
        (name_one in ['AS', 'YS'] and name_two == 'QS')):
      factor = 4 * one.n / two.n
    if ((name_one == 'Q' and name_two == 'M') or
        (name_one == 'QS' and name_two == 'MS')):
      factor = 3 * one.n / two.n
    if name_one == 'W' and name_two == 'D':
      factor = 7 * one.n / two.n

    if factor is not None:
      if i == 1:
        return 1 / factor
      return factor

  if any(
      re.sub(regex_month_or_day, '', offset.name) in ['M', 'Y', 'A', 'y']
      for offset in [offset_one, offset_two]
  ):
    raise ValueError(
        f'Frequencies {freq_one} and/or {freq_two} are not comparable.'
    )
  try:
    factor = pd.to_timedelta(freq_one) / pd.to_timedelta(freq_two)
  except ValueError as err:
    raise ValueError(
        f'Frequencies {freq_one} and/or {freq_two} are not comparable.'
    ) from err
  return factor


def infer_datetime_coord(xr: Union[DataArray, Dataset]) -> str:
  candidates = [c for c in list(xr.coords) if 'date' in c]
  if len(candidates) > 1:
    raise RuntimeError('Found multiple coordinates with "date" in its name.')
  if not candidates:
    raise RuntimeError('Did not find any coordinate with "date" in its name')

  return candidates[0]


def get_available_metrics() -> List[str]:
  return _METRICS


def _validate_inputs(
    obs: DataArray,
    sim: DataArray
):
  if obs.shape != sim.shape:
    raise RuntimeError('Shapes of observations and simulations must match')

  if (len(obs.shape) > 1) and (obs.shape[1] > 1):
    raise RuntimeError(
        'Metrics only defined for time series (1d or 2d with second'
        ' dimension 1)'
    )


def _mask_valid(
    obs: DataArray,
    sim: DataArray
) -> Tuple[DataArray, DataArray]:
  idx = (~sim.isnull()) & (~obs.isnull())
  obs = obs[idx]
  sim = sim[idx]
  return obs, sim


def _get_fdc(
    da: DataArray
) -> np.ndarray:
  return da.sortby(da, ascending=False).values


def nse(
    obs: DataArray,
    sim: DataArray
) -> float:
  _validate_inputs(obs, sim)
  obs, sim = _mask_valid(obs, sim)
  denominator = ((obs - obs.mean()) ** 2).sum()
  numerator = ((sim - obs) ** 2).sum()
  value = 1 - numerator / denominator
  return float(value)


def log_nse(
    obs: DataArray,
    sim: DataArray
) -> float:
  obs = np.log10(np.maximum(1e-4, obs))
  sim = np.log10(np.maximum(1e-4, sim))

  _validate_inputs(obs, sim)
  obs, sim = _mask_valid(obs, sim)
  denominator = ((obs - obs.mean()) ** 2).sum()
  numerator = ((sim - obs) ** 2).sum()
  value = 1 - numerator / denominator
  return float(value)


def mse(
    obs: DataArray,
    sim: DataArray
) -> float:
  _validate_inputs(obs, sim)
  obs, sim = _mask_valid(obs, sim)
  return float(((sim - obs)**2).mean())


def rmse(obs: DataArray, sim: DataArray) -> float:
  return np.sqrt(mse(obs, sim))


def alpha_nse(
    obs: DataArray,
    sim: DataArray
) -> float:
  _validate_inputs(obs, sim)
  obs, sim = _mask_valid(obs, sim)
  return float(sim.std() / obs.std())


def beta_nse(
    obs: DataArray,
    sim: DataArray
) -> float:
  _validate_inputs(obs, sim)
  obs, sim = _mask_valid(obs, sim)
  return float((sim.mean() - obs.mean()) / obs.std())


def beta_kge(obs: DataArray, sim: DataArray) -> float:
  _validate_inputs(obs, sim)
  obs, sim = _mask_valid(obs, sim)
  return float(sim.mean() / obs.mean())


def kge(
    obs: DataArray,
    sim: DataArray,
    weights: List[float] = _UNIFORM_WEIGHTS
) -> float:
  if len(weights) != 3:
    raise ValueError('Weights of the KGE must be a list of three values')
  _validate_inputs(obs, sim)
  obs, sim = _mask_valid(obs, sim)

  if len(obs) < 2:
    return np.nan

  r = np.corrcoef(obs.values, sim.values)[0, 1]
  alpha = sim.std() / obs.std()
  beta = sim.mean() / obs.mean()

  value = (weights[0] * (r - 1)**2 + weights[1] * 
           (alpha - 1)**2 + weights[2] * (beta - 1)**2)

  return 1 - np.sqrt(float(value))


def log_kge(
    obs: DataArray,
    sim: DataArray,
    weights: List[float] = _UNIFORM_WEIGHTS
) -> float:
  obs = np.log10(np.maximum(1e-4, obs))
  sim = np.log10(np.maximum(1e-4, sim))

  if len(weights) != 3:
    raise ValueError('Weights of the KGE must be a list of three values')

  _validate_inputs(obs, sim)
  obs, sim = _mask_valid(obs, sim)

  if len(obs) < 2:
    return np.nan

  r = np.corrcoef(obs.values, sim.values)[0, 1]
  alpha = sim.std() / obs.std()
  beta = sim.mean() / obs.mean()

  value = (weights[0] * (r - 1)**2 + weights[1] * (alpha - 1)**2 +
           weights[2] * (beta - 1)**2)

  return 1 - np.sqrt(float(value))


def pearsonr(
    obs: DataArray,
    sim: DataArray
) -> float:
  _validate_inputs(obs, sim)
  obs, sim = _mask_valid(obs, sim)

  if len(obs) < 2:
    return np.nan

  r = np.corrcoef(obs.values, sim.values)[0, 1]
  return float(r)


def mean_peak_timing(
    obs: DataArray,
    sim: DataArray,
    window: int = None,
    resolution: str = '1D',
    datetime_coord: str = None
) -> float:
  _validate_inputs(obs, sim)
  obs, sim = _mask_valid(obs, sim)

  peaks, _ = signal.find_peaks(
      obs.values, distance=100, prominence=np.std(obs.values)
  )

  if datetime_coord is None:
    datetime_coord = infer_datetime_coord(obs)

  if window is None:
    window = max(int(get_frequency_factor('12h', resolution)), 3)

  timing_errors = []
  for idx in peaks:
    if ((idx - window < 0) or(idx + window >= len(obs)) or
        pd.date_range(
            obs[idx - window][datetime_coord].values,
            obs[idx + window][datetime_coord].values,
            freq=resolution).size != 2 * window + 1):
      continue

    if (sim[idx] > sim[idx - 1]) and (sim[idx] > sim[idx + 1]):
      peak_sim = sim[idx]
    else:
      values = sim[idx - window:idx + window + 1]
      peak_sim = values[values.argmax()]

    peak_obs = obs[idx]
    delta = peak_obs.coords[datetime_coord] - peak_sim.coords[datetime_coord]
    timing_error = np.abs(delta.values / pd.to_timedelta(resolution))
    timing_errors.append(timing_error)

  return np.mean(timing_errors) if len(timing_errors) > 0 else np.nan


def missed_peaks(
    obs: DataArray,
    sim: DataArray,
    window: int = None,
    resolution: str = '1D',
    percentile: float = 80,
    datetime_coord: str = None
) -> float:
  _validate_inputs(obs, sim)
  obs, sim = _mask_valid(obs, sim)

  min_obs_height = np.percentile(obs.values, percentile)
  min_sim_height = np.percentile(sim.values, percentile)

  peaks_obs_times, _ = signal.find_peaks(
      obs, distance=30, height=min_obs_height)
  peaks_sim_times, _ = signal.find_peaks(
      sim, distance=30, height=min_sim_height)

  if len(peaks_obs_times) == 0:
    return 0.

  if datetime_coord is None:
    datetime_coord = infer_datetime_coord(obs)

  if window is None:
    window = max(int(get_frequency_factor('12h', resolution)), 1)

  missed_events = 0
  for idx in peaks_obs_times:
    if ((idx - window < 0) or(idx + window >= len(obs)) or
        pd.date_range(
            obs[idx - window][datetime_coord].values,
            obs[idx + window][datetime_coord].values,
            freq=resolution).size != 2 * window + 1):
      continue

    nearby_peak_sim_index = np.where(np.abs(peaks_sim_times - idx) <= window)[0]
    if len(nearby_peak_sim_index) == 0:
      missed_events += 1

  return missed_events / len(peaks_obs_times)


def fdc_fms(
    obs: DataArray, 
    sim: DataArray, 
    lower: float = 0.2, 
    upper: float = 0.7
) -> float:
  _validate_inputs(obs, sim)
  obs, sim = _mask_valid(obs, sim)

  if len(obs) < 1:
    return np.nan

  if any([(x <= 0) or (x >= 1) for x in [upper, lower]]):
    raise ValueError('upper and lower have to be in range ]0,1[')

  if lower >= upper:
    raise ValueError('The lower threshold has to be smaller than the upper.')

  obs = _get_fdc(obs)
  sim = _get_fdc(sim)

  sim[sim <= 0] = 1e-6
  obs[obs == 0] = 1e-6

  qsm_lower = np.log(sim[np.round(lower * len(sim)).astype(int)])
  qsm_upper = np.log(sim[np.round(upper * len(sim)).astype(int)])
  qom_lower = np.log(obs[np.round(lower * len(obs)).astype(int)])
  qom_upper = np.log(obs[np.round(upper * len(obs)).astype(int)])

  fms = ((qsm_lower - qsm_upper) - (qom_lower - qom_upper)) / (qom_lower - qom_upper + 1e-6)

  return fms * 100


def fdc_fhv(obs: DataArray, sim: DataArray, h: float = 0.02) -> float:
  _validate_inputs(obs, sim)
  obs, sim = _mask_valid(obs, sim)

  if len(obs) < 1:
    return np.nan

  if (h <= 0) or (h >= 1):
    raise ValueError('h must be in range [0, 1].')

  obs = _get_fdc(obs)
  sim = _get_fdc(sim)

  obs = obs[:np.round(h * len(obs)).astype(int)]
  sim = sim[:np.round(h * len(sim)).astype(int)]

  fhv = np.sum(sim - obs) / np.sum(obs)

  return fhv * 100


def fdc_flv(
    obs: DataArray,
    sim: DataArray,
    l: float = 0.3
) -> float:
  _validate_inputs(obs, sim)
  obs, sim = _mask_valid(obs, sim)

  if len(obs) < 1:
    return np.nan

  if (l <= 0) or (l >= 1):
    raise ValueError('l must be in range [0, 1].')

  obs = _get_fdc(obs)
  sim = _get_fdc(sim)

  sim[sim <= 0] = 1e-6
  obs[obs == 0] = 1e-6

  obs = obs[-np.round(l * len(obs)).astype(int) :]
  sim = sim[-np.round(l * len(sim)).astype(int) :]

  obs = np.log(obs)
  sim = np.log(sim)

  qsl = np.sum(sim - sim.min())
  qol = np.sum(obs - obs.min())

  flv = -1 * (qsl - qol) / (qol + 1e-6)

  return flv * 100


def mean_absolute_percentage_peak_error(obs: DataArray, sim: DataArray) -> float:
  _validate_inputs(obs, sim)
  obs, sim = _mask_valid(obs, sim)

  if obs.size == 0 or sim.size == 0:
    return np.nan

  peaks, _ = signal.find_peaks(obs.values, distance=10, prominence=np.std(obs.values))

  if peaks.size == 0:
    return np.nan

  obs = obs[peaks].values
  sim = sim[peaks].values

  peak_mape = np.sum(np.abs((sim - obs) / obs)) / peaks.size * 100

  return peak_mape


def _check_enough_data(
    obs: DataArray,
    sim: DataArray,
    minimum_data_points: Optional[int] = None,
):
  if minimum_data_points is None:
    minimum_data_points = _MIN_DATA_FOR_CALCULATING_METRICS

  obs, sim = _mask_valid(obs, sim)

  if obs.shape[0] <= minimum_data_points:
    return False
  return True


def calculate_all_metrics(
    obs: DataArray,
    sim: DataArray,
    resolution: str = '1D',
    datetime_coord: str = None,
    minimum_data_points: Optional[int] = None,
) -> Dict[str, float]:
  if not _check_enough_data(obs, sim, minimum_data_points):
    return {metric: np.nan for metric in get_available_metrics()}

  results = {
      'NSE': nse(obs, sim),
      'log-NSE': log_nse(obs, sim),
      'MSE': mse(obs, sim),
      'RMSE': rmse(obs, sim),
      'KGE': kge(obs, sim),
      'log-KGE': log_kge(obs, sim),
      'Alpha-NSE': alpha_nse(obs, sim),
      'Beta-KGE': beta_kge(obs, sim),
      'Beta-NSE': beta_nse(obs, sim),
      'Pearson-r': pearsonr(obs, sim),
      'Peak-Timing': mean_peak_timing(
          obs, sim, resolution=resolution, datetime_coord=datetime_coord
      ),
      'FLV': fdc_flv(obs, sim),
      'FHV': fdc_fhv(obs, sim),
      'FMS': fdc_fms(obs, sim),
      'Missed-Peaks': missed_peaks(
          obs, sim, resolution=resolution, datetime_coord=datetime_coord
      ),
      'Peak-MAPE': mean_absolute_percentage_peak_error(obs, sim)
  }
  return results


def calculate_metrics(
    obs: DataArray,
    sim: DataArray,
    metrics: List[str],
    resolution: str = '1D',
    datetime_coord: str = None,
    minimum_data_points: Optional[int] = None,
) -> Dict[str, float]:
  if 'all' in metrics:
    return calculate_all_metrics(obs, sim, resolution=resolution)

  if not _check_enough_data(obs, sim, minimum_data_points):
    return {metric: np.nan for metric in metrics}

  values = {}
  for metric in metrics:
    m = metric.lower()
    if m == 'nse':
      values['NSE'] = nse(obs, sim)
    elif m == 'log-nse':
      values['log-NSE'] = log_nse(obs, sim)
    elif m == 'mse':
      values['MSE'] = mse(obs, sim)
    elif m == 'rmse':
      values['RMSE'] = rmse(obs, sim)
    elif m == 'kge':
      values['KGE'] = kge(obs, sim)
    elif m == 'log-kge':
      values['log-KGE'] = log_kge(obs, sim)
    elif m == 'alpha-nse':
      values['Alpha-NSE'] = alpha_nse(obs, sim)
    elif m == 'beta-kge':
      values['Beta-KGE'] = beta_kge(obs, sim)
    elif m == 'beta-nse':
      values['Beta-NSE'] = beta_nse(obs, sim)
    elif m == 'pearson-r':
      values['Pearson-r'] = pearsonr(obs, sim)
    elif m == 'peak-timing':
      values['Peak-Timing'] = mean_peak_timing(
          obs, sim, resolution=resolution, datetime_coord=datetime_coord
      )
    elif m == 'missed-peaks':
      values['Missed-Peaks'] = missed_peaks(
          obs, sim, resolution=resolution, datetime_coord=datetime_coord
      )
    elif m == 'flv':
      values['FLV'] = fdc_flv(obs, sim)
    elif m == 'fhv':
      values['FHV'] = fdc_fhv(obs, sim)
    elif m == 'fms':
      values['FMS'] = fdc_fms(obs, sim)
    elif m == 'peak-mape':
      values['Peak-MAPE'] = mean_absolute_percentage_peak_error(obs, sim)
    else:
      raise RuntimeError(f'Unknown metric {metric}')

  return values
