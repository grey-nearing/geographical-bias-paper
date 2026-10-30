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

"""Paths and configuration for hydro model performance metrics calculations."""

import pathlib

_BASE_DIR = pathlib.Path(__file__).parent.parent

_WORKING_DIR = _BASE_DIR

# Lead times (days)
LEAD_TIMES = list(range(8))

UNGAUGED_EXPERIMENTS = [
    'kfold_splits',
    'continent_splits',
    'climate_splits',
    'hydrologically_separated',
]
EXPERIMENTS = UNGAUGED_EXPERIMENTS + ['full_run']

GOOGLE_MODEL = 'dual_lstm'

# Output directory for calculated metrics
METRICS_DIR = _WORKING_DIR / 'paper_artifacts' / 'metrics'
RETURN_PERIOD_METRICS_DIR = METRICS_DIR / 'return_period_metrics'
HYDROGRAPH_METRICS_DIR = METRICS_DIR / 'hydrograph_metrics'
