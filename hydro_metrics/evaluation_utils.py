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

"""Evaluation utilities and constants for hydro model performance metrics."""

import os
import pathlib
from typing import Optional

import numpy as np
import pandas as pd
import tqdm

from . import data_paths
from . import metrics_utils

# Standard return period thresholds evaluated (in years)
RETURN_PERIODS = [1.01, 2, 5, 10, 20, 50]

METRIC_NAMES = {
    'NSE': 'Nash Sutcliffe Efficiency (NSE)',
    'log-NSE': 'NSE of Log-Transformed Values',
    'Alpha-NSE': 'Standard Deviation Ratio',
    'Beta-NSE': 'Bias',
    'KGE': 'Kling-Gupta Efficiency (KGE)',
    'log-KGE': 'KGE of Log-Transformed Values',
    'Pearson-r': 'Correlation Coefficient',
    'Beta-KGE': 'Bias Ratio',
}


def f1_from_precision_and_recall_dfs(
    precision_df: pd.DataFrame,
    recall_df: pd.DataFrame
) -> pd.DataFrame:
  """Calculates F1 score DataFrame from precision and recall DataFrames."""
  f1 = 2 * (precision_df * recall_df / (precision_df + recall_df))
  f1[(precision_df + recall_df) == 0] = 0
  return f1


def load_return_period_metrics(
    base_path: pathlib.Path,
    experiments: list[str],
    gauges: list[str],
    metric: str,
) -> dict[str, dict[str, pd.DataFrame]]:
  """Load and concatenate return period metrics for experiments across gauges."""
  if metric.lower() not in ['precision', 'recall']:
    raise ValueError('Can only load precision or recall.')

  scores_by_lead_time = {
      experiment: {
          lead_time: pd.DataFrame(
              index=gauges,
              columns=RETURN_PERIODS,
              dtype=float
          ) for lead_time in data_paths.LEAD_TIMES
      } for experiment in experiments
  }

  for experiment in experiments:
    print(f'Working on experiment {experiment} ...')
    for gauge in tqdm.tqdm(gauges):
      filepath = metrics_utils.save_metrics_df(
          df=None,
          metric=gauge,
          base_path=base_path,
          path_modifier=f'{experiment}/{metric.lower()}',
      )
      if os.path.exists(filepath):
        with open(filepath, 'r') as f:
          scores = pd.read_csv(f, index_col=[0, 1])
        idx = pd.IndexSlice
        for lead_time in data_paths.LEAD_TIMES:
          if str(lead_time) in scores.columns:
            scores_by_lead_time[experiment][lead_time].loc[gauge] = (
                scores.loc[idx[:, 2], str(lead_time)].values
            )
      else:
        for lead_time in data_paths.LEAD_TIMES:
          scores_by_lead_time[experiment][lead_time].loc[gauge] = np.nan

  return scores_by_lead_time
