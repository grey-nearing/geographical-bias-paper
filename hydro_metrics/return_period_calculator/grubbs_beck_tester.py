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

"""Implements utilities for handling Potentially Impactful Low Floods (PILFs).

This procedure for handling PILFs is described in Appendix 4 of Bulletin 17b.
It was supersceded by the Multiple Grubbs-Beck Test in Appendix 6 of Bulletin
17c.

This code was developed as part of an open source Python package for
calcultaing streamflow return periods according to guidelines in the USGS
Bulletin 17c (2019):

https://pubs.usgs.gov/tm/04/b05/tm4b5.pdf
"""
import abc
import logging
import pathlib
from typing import Optional, Mapping

import numpy as np
import pandas as pd

from . import exceptions

_KN_TABLE_FILENAME = str(pathlib.Path(__file__).parent / 'bulletin17b_kn_table.csv')


def _load_kn_table(
    file: str = _KN_TABLE_FILENAME
) -> Mapping[int, float]:
  kn_table_series = pd.read_csv(file, index_col='Sample Size')
  kn_table_series.index = kn_table_series.index.astype(int)
  return kn_table_series.to_dict()['KN Value']


def _grubbs_beck_test(
    sorted_data: np.ndarray,
    kn_table: Mapping[int, float],
) -> int:
  """Performs one sweep of a GBT.

  This routine implements Equation 8a in the USGS Bulletin 17b (not 17c):
  https://water.usgs.gov/osw/bulletin17b/dl_flow.pdf

  Cohn et al (2013) argue that a multiple Grubbs Beck Test should be used
  instead.

  Cohn, T. A., et al. "A generalized Grubbs-Beck test statistic for detecting
  multiple potentially influential low outliers in flood series."
  Water Resources Research 49.8 (2013): 5047-5058.

  Args:
    sorted_data: Systematic flood record sorted in ascending order. Must be
      in log-transformed units.
    kn_table: Kn table for Grubbs Beck test (see Bulletin 17b Table 4).

  Returns:
    Number of low outliers in the data series.
  """
  sample_size = len(sorted_data)
  if sample_size not in kn_table:
    logging.warning('Sample size %d not in Kn Table', sample_size)
    return 0

  kn_value = kn_table[sample_size]

  mean = np.mean(sorted_data)
  std = np.std(sorted_data)

  x_star = mean - kn_value * std

  outliers = sorted_data < x_star
  return int(np.sum(outliers))


def _standard_grubbs_beck_test(
    sorted_data: np.ndarray,
    kn_table: Optional[Mapping[int, float]] = None,
) -> int:
  """Implements the standard Grubbs Beck Test from Bulletin 17b.

  Args:
    sorted_data: Systematic flood record sorted in ascending order. Must be
      in log-transformed units.
    kn_table: Kn table for Grubbs Beck test (see Bulletin 17b Table 4).

  Returns:
    Number of low outliers in the data series.
  """
  if kn_table is None:
    kn_table = _load_kn_table()

  num_pilfs = 0

  while True:
    pilfs = _grubbs_beck_test(
        sorted_data=sorted_data[num_pilfs:],
        kn_table=kn_table,
    )
    num_pilfs += pilfs
    if pilfs == 0:
      break
    if len(sorted_data) - num_pilfs <= 0:
      break

  return num_pilfs


class BaseGrubbsBeckTester(abc.ABC):
  """Base class for Grubbs Beck test routines."""

  def __init__(
      self,
      data: np.ndarray,
      kn_table: Optional[Mapping[int, float]] = None,
  ):
    """Constructor for Grubbs Beck test objects.

    Args:
      data: Transformed systematic flood record.
      kn_table: Kn table for Grubbs Beck test (see Bulletin 17b Table 4).
    """
    self._data = np.sort(data)
    self._kn_table = kn_table
    self._run_test()

  def _run_test(self):
    raise NotImplementedError()

  @property
  def pilf_threshold(self) -> float:
    """PILF threshold in transformed units."""
    if self._num_pilfs == 0:
      return min(self._data) / 2
    else:
      return self._data[self._num_pilfs - 1]

  @property
  def in_population_sample(self) -> np.ndarray:
    """Systematic sample with PILFs removed (in transformed units)."""
    return self._data[self._num_pilfs:]

  @property
  def pilf_sample(self) -> np.ndarray:
    """PILFs (in transformed units)."""
    return self._data[:self._num_pilfs]


class GrubbsBeckTester(BaseGrubbsBeckTester):
  """Implements the standard Grubbs Beck Test from Bulletin 17b."""

  def _run_test(self):
    self._num_pilfs = _standard_grubbs_beck_test(
        sorted_data=self._data,
        kn_table=self._kn_table,
    )


class MultipleGrubbsBeckTester(BaseGrubbsBeckTester):
  """Implements the Multiple Grubbs Beck Test from Bulletin 17c."""

  def _run_test(self):
    # TODO(gsnearing): Implement MGBT.
    raise NotImplementedError()
