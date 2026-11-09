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

"""Unit tests for hydro_metrics infrastructure."""

import os
import sys
import unittest

# Ensure repository root is in sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
import xarray as xr

from hydro_metrics.metrics import calculate_all_metrics
from hydro_metrics.return_period_calculator.return_period_calculator import ReturnPeriodCalculator
from hydro_metrics.return_period_metrics import calculate_return_period_performance_metrics


class HydroMetricsTest(unittest.TestCase):

  def setUp(self):
    dates = pd.date_range('2010-01-01', '2020-12-31', freq='1D')
    np.random.seed(42)
    obs_vals = 10 + np.abs(np.random.normal(0, 5, len(dates)))
    # Add occasional flood peaks
    obs_vals[100] = 150
    obs_vals[460] = 180
    obs_vals[820] = 140
    obs_vals[1180] = 200
    obs_vals[1540] = 160
    obs_vals[1900] = 190

    sim_vals = obs_vals + np.random.normal(0, 1, len(dates))

    self.obs_series = pd.Series(obs_vals, index=dates)
    self.sim_series = pd.Series(sim_vals, index=dates)

    self.obs_da = xr.DataArray(self.obs_series, dims=['time'], coords={'time': dates})
    self.sim_da = xr.DataArray(self.sim_series, dims=['time'], coords={'time': dates})

  def test_standard_metrics(self):
    m = calculate_all_metrics(self.obs_da, self.sim_da, datetime_coord='time')
    self.assertIn('NSE', m)
    self.assertIn('KGE', m)
    self.assertIn('RMSE', m)
    self.assertGreater(m['NSE'], 0.9)
    self.assertGreater(m['KGE'], 0.9)

  def test_return_period_calculator(self):
    rpc = ReturnPeriodCalculator(
        hydrograph_series=self.obs_series,
        hydrograph_series_frequency=pd.Timedelta('1D'),
        use_simple_fitting=True,
    )
    self.assertGreater(len(rpc), 0)
    rp_flow = rpc.flow_value_from_return_period(5.0)
    self.assertGreater(rp_flow, 0)

  def test_return_period_performance_metrics(self):
    rp_metrics, flow_vals = calculate_return_period_performance_metrics(
        observations=self.obs_series,
        predictions=self.sim_series,
    )
    self.assertIn('return_period_2_window_0_precision', rp_metrics)
    self.assertIn('return_period_2_window_0_recall', rp_metrics)
    self.assertIn('return_period_2', flow_vals)


if __name__ == '__main__':
  unittest.main()
