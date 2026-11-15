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

"""Polygon-overlap upstream and downstream gauge-coverage helpers."""

import pandas as pd
from tqdm import tqdm


def calculate_total_upstream_coverage(
    infer_ids: list,
    tt_df: pd.DataFrame,
    ti_df: pd.DataFrame,
    basin_areas: pd.Series,
    containment_threshold: float = 0.98,
) -> pd.DataFrame:
  """Calculates predictive upstream geospatial features for inference basins.

  Isolates maximal (non-nested) upstream training basins inside each target
  basin to prevent double-counting overlapping areas.
  """
  containment_dict = {}
  if not tt_df.empty:
    child_areas = tt_df['basin1'].map(basin_areas).values
    parent_areas = tt_df['basin2'].map(basin_areas).values
    is_contained = (
        tt_df['overlap_area'].values / child_areas
    ) >= containment_threshold
    is_strictly_larger = parent_areas > child_areas
    is_tie_breaker = (parent_areas == child_areas) & (
        tt_df['basin2'].values > tt_df['basin1'].values
    )
    valid = is_contained & (is_strictly_larger | is_tie_breaker)
    contained_df = tt_df[valid]
    containment_dict = (
        contained_df.groupby('basin1')['basin2'].apply(set).to_dict()
    )

  up_ti_df = ti_df.copy()
  up_ti_df['area1'] = up_ti_df['basin1'].map(basin_areas)
  up_ti_df['area2'] = up_ti_df['basin2'].map(basin_areas)
  up_ti_df = up_ti_df[up_ti_df['area2'] < up_ti_df['area1']]

  results_dict = {
      infer_id: {
          'total_upstream_coverage': 0.0,
          'max_upstream_coverage': 0.0,
          'upstream_gauge_count': 0,
      }
      for infer_id in infer_ids
  }

  for infer_id, group in tqdm(
      up_ti_df.groupby('basin1'), desc='Upstream coverage', leave=False
  ):
    active_train = set(group['basin2'])
    maximal_train_ids = set()
    for t_id in active_train:
      parents = containment_dict.get(t_id, set())
      if not (parents & active_train):
        maximal_train_ids.add(t_id)

    maximal_overlap_sum = group[group['basin2'].isin(maximal_train_ids)][
        'overlap_area'
    ].sum()
    fraction = maximal_overlap_sum / basin_areas.loc[infer_id]
    max_fraction = group['overlap_area'].max() / basin_areas.loc[infer_id]

    results_dict[infer_id]['total_upstream_coverage'] = min(fraction, 1.0)
    results_dict[infer_id]['max_upstream_coverage'] = min(max_fraction, 1.0)
    results_dict[infer_id]['upstream_gauge_count'] = len(maximal_train_ids)

  return pd.DataFrame.from_dict(results_dict, orient='index')


def calculate_max_downstream_coverage(
    infer_ids: list, ti_df: pd.DataFrame, basin_areas: pd.Series
) -> pd.Series:
  """Calculates maximum downstream gauge-coverage fraction for target basins."""
  if ti_df.empty:
    return pd.Series(0.0, index=infer_ids, name='max_downstream_coverage')

  down_ti_df = ti_df.copy()
  down_ti_df['area1'] = down_ti_df['basin1'].map(basin_areas)
  down_ti_df['area2'] = down_ti_df['basin2'].map(basin_areas)
  is_downstream = down_ti_df['area2'] >= down_ti_df['area1']
  down_df = down_ti_df[is_downstream].copy()

  if down_df.empty:
    return pd.Series(0.0, index=infer_ids, name='max_downstream_coverage')

  down_df['downstream_frac'] = down_df['overlap_area'] / down_df['area2']
  max_down = down_df.groupby('basin1')['downstream_frac'].max()
  return max_down.reindex(infer_ids, fill_value=0.0).rename(
      'max_downstream_coverage'
  )
