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

"""Code to plot visual representations of return period distributions.

This code implements flood frequency interval calculations from guidelines
in the USGS Bulletin 17c (2019):

https://pubs.usgs.gov/tm/04/b05/tm4b5.pdf
"""
from typing import Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import base_fitter
from . import empirical_distribution_utilities as edu

_DEFAULT_FIG_SIZE = (10, 6)


def plot_fitted_distribution(
    data: np.ndarray,
    fitter: base_fitter.BaseFitter,
    min_allowed_bins: int = edu.MIN_ALLOWED_HISTOGRAM_BINS,
):
  """Plots a fitted distribution with a histogram of flow values."""
  _ = plt.figure(figsize=_DEFAULT_FIG_SIZE)

  bin_heights, bin_centers, bin_width, bin_edges = edu.empirical_pdf(
      data=data,
      min_allowed_bins=min_allowed_bins,
  )
  plt.bar(
      x=bin_centers,
      height=bin_heights,
      width=bin_width*0.9,
      label='empirical distribution',
  )

  probability_mass = fitter.probability_mass_function(
      edges=bin_edges[bin_edges > 0]) / bin_width
  plt.plot(
      bin_centers[-len(probability_mass):],
      probability_mass,
      '-*',
      color='g',
      label='fitted_distribution',
  )

  plt.grid()
  plt.legend()
  plt.xlabel('Annual Peak Streamflow (original data units)')
  plt.ylabel('Probability Mass Function')


def plot_exceedence_probability_distribution(
    fitter: base_fitter.BaseFitter,
):
  """Plots an exceedence probability distribution."""
  _, ax = plt.subplots(1, 1, figsize=_DEFAULT_FIG_SIZE)

  systematic_flows = fitter.in_population_sample
  pilf_flows = fitter.pilf_sample

  empirical_plotting_positions = edu.simple_empirical_plotting_position(
      data=np.concatenate([pilf_flows, systematic_flows])) * 100

  pilf_plotting_positions = empirical_plotting_positions[:len(pilf_flows)]
  systematic_plotting_positions = empirical_plotting_positions[len(pilf_flows):]

  ax.scatter(x=pilf_plotting_positions, y=pilf_flows,
             color='r', marker='o', label='PILF flows')
  ax.scatter(x=systematic_plotting_positions, y=systematic_flows,
             color='k', marker='o', label='systematic flows')

  theoretical_cdf = np.linspace(0, 1, 200)[1:-1]
  theoretical_exceedance_probabilities = (1 - theoretical_cdf)
  theoretical_flows = fitter.flow_values_from_exceedance_probabilities(
      exceedance_probabilities=theoretical_exceedance_probabilities)
  ax.plot(
      theoretical_exceedance_probabilities * 100,
      theoretical_flows,
      label='theoretical distribution',
    )

  ax.grid()
  ax.legend()
  ax.invert_xaxis()
  ax.set_yscale('log')
  ax.set_xscale('log')
  ax.set_xlabel('Annual Exceedance Probability (%)')
  ax.set_ylabel('Annual Peak Streamflow (original data units)')


def plot_hydrograph_with_peaks(
    hydrograph_series: pd.Series,
    peaks_series: pd.Series,
):
  """Plots a hyetograph with peak flows identified."""
  plt.figure(figsize=_DEFAULT_FIG_SIZE)

  plt.plot(hydrograph_series, label='hydrograph')
  plt.plot(peaks_series, 'o', label='extracted peaks')

  plt.grid()
  plt.legend()
  plt.ylabel('Streamflow (original data units)s')


def plot_hydrograph_with_return_periods(
    hydrograph_series: pd.Series,
    return_period_values: Mapping[float, float],
):
  """Plots a hydrograph with return overlaid return periods."""
  plt.figure(figsize=_DEFAULT_FIG_SIZE)

  plt.plot(hydrograph_series, label='hydrograph')

  for rp, flow_value in return_period_values.items():
    series = pd.Series(flow_value, index=hydrograph_series.index, name=rp)
    plt.plot(series, '--', label=f'{rp}-year return period')

  plt.grid()
  plt.legend()
  plt.ylabel('Streamflow (original data units)')


def plot_full_record(
    systematic_record: pd.Series,
    historical_records: Sequence[pd.Series],
    historic_events: Mapping[pd.Timestamp, Tuple[float, float]],
):
  raise NotImplementedError()
