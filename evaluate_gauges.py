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

# Author: Grey Nearing
# Created: 2026-07-14

"""Gauge Value Evaluator (v2: M_ungauged-Centered with Exact River Topology).

Evaluates the expected annual human flood exposure reduction achieved by adding
streamflow gauges at candidate HydroBASINS Level 12 sub-basins on the real-world
(`M_ungauged`) skill predictor, with `M_gauged` retained solely as an optional
independent confirmation cross-check at newly gauged outlets.
"""

from collections import defaultdict
import importlib
import os
import pickle
import sys
import time
import types

from absl import app
from absl import flags
import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from utils import skill_predictor as sp
sys.modules['utils.skill_predictor'] = sp

from utils import file_paths
from utils import io_utils
from utils import stats_utils
from utils.io_utils import load_csv
from utils import gfile_compat as gfile

FLAGS = flags.FLAGS

def _define_flags():
  if 'metric' not in FLAGS:
    flags.DEFINE_string('metric', '5.0-year-2.0-window-recall', 'Metric to evaluate.')
  if 'continent' not in FLAGS:
    flags.DEFINE_string('continent', 'au', 'Continent code (e.g. au, af, as).')
  if 'candidates_file' not in FLAGS:
    flags.DEFINE_string('candidates_file', '', 'CSV file containing candidate hybas_ids.')
  if 'start_idx' not in FLAGS:
    flags.DEFINE_integer('start_idx', 0, 'Start index in candidates list.')
  if 'end_idx' not in FLAGS:
    flags.DEFINE_integer('end_idx', -1, 'End index in candidates list (exclusive, -1 for end of list).')
  if 'output_dir' not in FLAGS:
    flags.DEFINE_string('output_dir', '', 'Directory to write evaluation results.')


if __name__ == '__main__':
  _define_flags()

# Module-level shared cache across continental evaluators so multi-continent
# scripts load global Zarr features, pickled models, Caravan tables, and base
# HyBAS skill predictions once instead of 9 times.
_SHARED_GLOBAL_CACHE = {}

NON_PHYSICAL_COLS = {
    'anomaly_score_if',
    'anomaly_score_ocsvm',
    'anomaly_score_lof',
    'total_upstream_coverage',
    'max_upstream_coverage',
    'max_downstream_coverage',
    'upstream_gauge_count',
    'record_length',
}


def _enable_parallel_n_jobs(obj):
  """Recursively enables multi-core CPU inference (`n_jobs=-1`) on sklearn estimators."""
  if obj is None:
    return
  if hasattr(obj, 'n_jobs'):
    try:
      obj.n_jobs = -1
    except Exception:
      pass
  for attr in ('impl', 'ensemble', 'model', 'final_estimator'):
    if hasattr(obj, attr):
      _enable_parallel_n_jobs(getattr(obj, attr))
  if hasattr(obj, 'estimators_'):
    for est in obj.estimators_:
      _enable_parallel_n_jobs(est)
  if hasattr(obj, 'steps'):
    for _, step in obj.steps:
      _enable_parallel_n_jobs(step)


def get_shared_global_resources(default_metric: str = '5.0-year-2.0-window-recall') -> dict:
  """Loads and caches global models, base Zarr features, Caravan overlaps, and base skills."""
  if _SHARED_GLOBAL_CACHE:
    return _SHARED_GLOBAL_CACHE

  t0_all = time.time()
  print(f"[{time.strftime('%H:%M:%S')}] Loading shared global resources into cache...")

  rp_metric_map = {
      5: '5.0-year-2.0-window-recall',
      10: '10.0-year-2.0-window-recall',
      20: '20.0-year-2.0-window-recall',
  }
  gauged_models = {}
  ungauged_models = {}
  for T, m_str in rp_metric_map.items():
    file_paths.init(m_str)
    with gfile.Open(file_paths.FINAL_GAUGED_MODEL, 'rb') as f:
      gm = pickle.loads(f.read())
      _enable_parallel_n_jobs(gm)
      gauged_models[T] = gm
    with gfile.Open(file_paths.FINAL_UNGAUGED_MODEL, 'rb') as f:
      um = pickle.loads(f.read())
      _enable_parallel_n_jobs(um)
      ungauged_models[T] = um
  file_paths.init(default_metric)

  # Global HyBAS Zarr feature matrix
  df_global = io_utils.load_hybas_zarr(file_paths.HYBAS_GENERIC_DATA_ZARR_PATH)
  df_global.index = [
      idx if str(idx).startswith('hybas_') else 'hybas_' + str(idx)
      for idx in df_global.index
  ]
  physical_cols = [c for c in df_global.columns if c not in NON_PHYSICAL_COLS]

  # Caravan attributes aligned to physical_cols
  df_caravan_raw = load_csv(file_paths.CARAVAN_ATTRIBUTES_PATH)
  df_caravan_raw = df_caravan_raw.drop(
      columns=[c for c in df_caravan_raw.columns if c.lower() in file_paths.BANNED_SPATIAL_COLS]
  )
  df_caravan_num = df_caravan_raw.select_dtypes(include=[np.number]).copy()
  df_caravan_num.columns = [
      c.replace('HYDRO_ATLAS:', '') for c in df_caravan_num.columns
  ]
  caravan_features = df_caravan_num.reindex(columns=physical_cols).dropna()

  # Caravan train-to-train pairwise overlaps & Caravan areas
  with gfile.Open(file_paths.CARAVAN_PAIRWISE_OVERLAPS_PATH, 'r') as f:
    caravan_tt_df = pd.read_csv(f)
  caravan_self = caravan_tt_df[caravan_tt_df['basin1'] == caravan_tt_df['basin2']]
  caravan_areas = caravan_self.set_index('basin1')['overlap_area']

  # Global topology CSV (if present)
  df_topo_all = None
  if gfile.Exists(file_paths.HYBAS_ROUTING_TOPOLOGY_PATH):
    with gfile.Open(file_paths.HYBAS_ROUTING_TOPOLOGY_PATH, 'rt') as f:
      df_topo_all = pd.read_csv(
          f, dtype={'HYBAS_ID': str, 'NEXT_DOWN': str, 'continent': str}
      )

  # Country mapping, INFORM Vulnerability, MPI, and multi-RP Exposure
  try:
    with gfile.Open(file_paths.HYBAS_MAPPING_PATH, 'rt') as f:
      mapping_df = pd.read_csv(f, index_col='hybas_id')
    mapping_df.index = [
        idx if str(idx).startswith('hybas_') else 'hybas_' + str(idx)
        for idx in mapping_df.index
    ]
    mapping_df = mapping_df[~mapping_df.index.duplicated(keep='first')]
    mapping_df['country_name'] = mapping_df['country_name'].replace(
        file_paths.country_name_map
    )
  except Exception as e:
    print(f'   -> Warning: Failed to load HyBAS country mapping: {e}')
    mapping_df = pd.DataFrame(columns=['country_name'])

  try:
    with gfile.Open(file_paths.RISK_FILE, 'rt') as f:
      df_risk = pd.read_csv(f, header=1, index_col=0)
    df_risk = df_risk.drop('(a-z)', errors='ignore')
    df_risk.index = df_risk.index.to_series().replace(file_paths.country_name_map)
    vuln_index = (pd.to_numeric(df_risk['VULNERABILITY'], errors='coerce') / 10.0).groupby(level=0).first()
  except Exception:
    vuln_index = pd.Series(dtype=float)

  try:
    with gfile.Open(file_paths.INDICATOR_FILE, 'rt') as f:
      df_ind = pd.read_csv(f, header=1, index_col=0)
    df_ind = df_ind.drop('(a-z)', errors='ignore')
    df_ind.index = df_ind.index.to_series().replace(file_paths.country_name_map)
    mpi_index = pd.to_numeric(df_ind['Multidimensional Poverty Index'], errors='coerce').groupby(level=0).first()
  except Exception:
    mpi_index = pd.Series(dtype=float)

  country_series = mapping_df['country_name'].reindex(df_global.index)
  vuln_global = country_series.map(vuln_index).fillna(0.5).astype(float)
  mpi_global = country_series.map(mpi_index).fillna(0.5).astype(float)

  return_periods = [5, 10, 20, 50, 100]
  exposures_global_df = pd.DataFrame(0.0, index=df_global.index, columns=return_periods)
  if gfile.Exists(file_paths.GLOBAL_BASIN_EXPOSURE_BY_RP_PATH):
    with gfile.Open(file_paths.GLOBAL_BASIN_EXPOSURE_BY_RP_PATH, 'rt') as f:
      exp_df = pd.read_csv(f, index_col='hybas_id')
    idx_s = exp_df.index.astype(str).str.replace(r'\.0$', '', regex=True)
    exp_df.index = np.where(idx_s.str.startswith('hybas_'), idx_s, 'hybas_' + idx_s)
    exp_df = exp_df[~exp_df.index.duplicated(keep='first')]
    for T in return_periods:
      col_name = f'exposed_pop_{T}'
      if col_name in exp_df.columns:
        exposures_global_df[T] = exp_df[col_name].reindex(df_global.index, fill_value=0.0).astype(float)

  # Precomputed baseline M_ungauged skill predictions across all 964,084 basins
  skill_base_global_by_rp = {}
  for T, m_str in rp_metric_map.items():
    pred_csv = os.path.join(
        file_paths.HYBAS_PREDS_DIR, f'global_hybas_predictions_ungauged_{m_str}.csv'
    )
    if gfile.Exists(pred_csv):
      with gfile.Open(pred_csv, 'rt') as f:
        s_pred = pd.read_csv(f, index_col=0).iloc[:, 0]
      s_pred.index = [
          idx if str(idx).startswith('hybas_') else 'hybas_' + str(idx)
          for idx in s_pred.index
      ]
      s_pred = s_pred[~s_pred.index.duplicated(keep='first')]
      s_aligned = s_pred.reindex(df_global.index)
      missing_ids = s_aligned[s_aligned.isna()].index.tolist()
      if missing_ids:
        s_aligned.loc[missing_ids] = ungauged_models[T].predict(df_global.loc[missing_ids])
      skill_base_global_by_rp[T] = s_aligned.astype(float)
    else:
      skill_base_global_by_rp[T] = pd.Series(
          ungauged_models[T].predict(df_global), index=df_global.index, dtype=float
      )
  skill_base_global_by_rp[50] = skill_base_global_by_rp[20]
  skill_base_global_by_rp[100] = skill_base_global_by_rp[20]

  _SHARED_GLOBAL_CACHE.update({
      'rp_metric_map': rp_metric_map,
      'gauged_models': gauged_models,
      'ungauged_models': ungauged_models,
      'df_global': df_global,
      'physical_cols': physical_cols,
      'caravan_features': caravan_features,
      'caravan_tt_df': caravan_tt_df,
      'caravan_areas': caravan_areas,
      'df_topo_all': df_topo_all,
      'country_series': country_series,
      'vuln_global': vuln_global,
      'mpi_global': mpi_global,
      'exposures_global_df': exposures_global_df,
      'skill_base_global_by_rp': skill_base_global_by_rp,
  })
  print(f"[{time.strftime('%H:%M:%S')}] Cached shared global resources in {time.time() - t0_all:.2f}s.")
  return _SHARED_GLOBAL_CACHE


class GaugeValueEvaluator:
  """Evaluates streamflow gauge placement value using canonical pipeline functions."""

  def __init__(self, metric: str, continent_code: str):
    self.metric = metric
    self.continent = continent_code.lower()
    file_paths.init(self.metric)

    shared = get_shared_global_resources(self.metric)
    self.rp_metric_map = shared['rp_metric_map']
    self.gauged_models = shared['gauged_models']
    self.ungauged_models = shared['ungauged_models']
    self.physical_cols = shared['physical_cols']
    self.caravan_features = shared['caravan_features']
    self.caravan_tt_df = shared['caravan_tt_df']
    self.caravan_areas = shared['caravan_areas']

    # 1. Load HydroBASINS topology & exact full upstream drainage areas
    if self.continent == 'all':
      self.hybas_ids_set = set()
      self.df_base = shared['df_global'].copy()
      self.upstream_map = {}
      self.downstream_map = {}
      self.hybas_sub_areas = pd.Series(dtype=float)
      self.hybas_total_areas = pd.Series(dtype=float)
    else:
      t0 = time.time()
      if shared['df_topo_all'] is not None:
        df_topo = shared['df_topo_all'][
            shared['df_topo_all']['continent'].str.lower() == self.continent
        ].copy()
        df_topo['HYBAS_ID'] = (
            df_topo['HYBAS_ID']
            .astype(str)
            .str.replace(r'\.0$', '', regex=True)
            .str.replace('hybas_', '', regex=False)
        )
        df_topo['NEXT_DOWN'] = (
            df_topo['NEXT_DOWN']
            .astype(str)
            .str.replace(r'\.0$', '', regex=True)
            .str.replace('hybas_', '', regex=False)
        )
        df_topo.set_index('HYBAS_ID', inplace=True)
        self.hybas = types.SimpleNamespace(gdf=df_topo)
        self.hybas_ids = ['hybas_' + idx for idx in df_topo.index]
        self.hybas_ids_set = set(self.hybas_ids)
        self.hybas_sub_areas = pd.Series(
            df_topo['area_m2'].astype(float).values, index=self.hybas_ids
        )
      else:
        subbasin_file = os.path.join(
            file_paths.HYBAS_SUBBASINS_DIR, f'hybas_{self.continent}_lev12_v1c.geojson'
        )
        with gfile.Open(subbasin_file, 'r') as f:
          gdf = gpd.read_file(f)
        for col_name in ['HYBAS_ID', 'NEXT_DOWN']:
          if col_name in gdf.columns:
            gdf[col_name] = (
                gdf[col_name]
                .astype(str)
                .str.replace(r'\.0$', '', regex=True)
                .str.replace('hybas_', '', regex=False)
            )
        gdf.set_index('HYBAS_ID', inplace=True)
        if gdf.crs and gdf.crs.to_epsg() == 4326:
          gdf = gdf.to_crs('EPSG:6933')
        self.hybas = types.SimpleNamespace(gdf=gdf)
        self.hybas_ids = [
            idx if idx.startswith('hybas_') else 'hybas_' + idx for idx in gdf.index
        ]
        self.hybas_ids_set = set(self.hybas_ids)
        self.hybas_sub_areas = pd.Series(gdf.geometry.area.values, index=self.hybas_ids)

      # Build upstream/downstream routing trees and exact full upstream drainage areas
      self._build_upstream_map()
      self.hybas_areas = self.hybas_total_areas
      self.df_base = shared['df_global'].loc[
          shared['df_global'].index.isin(self.hybas_ids_set)
      ].copy()
      print(
          f"[{time.strftime('%H:%M:%S')}] Initialized {self.continent.upper()} "
          f"({len(self.df_base)} basins) in {time.time() - t0:.2f}s"
      )

    # 2. Load continent-specific HyBAS-to-Caravan overlaps & O(1) lookup index
    if self.continent.upper() == 'ALL':
      overlap_files = gfile.Glob(
          os.path.join(file_paths.HYBAS_COVERAGE_DIR, 'infer_pairwise_overlaps_*.csv')
      )
      dfs = []
      for f_name in overlap_files:
        with gfile.Open(f_name, 'r') as f:
          dfs.append(pd.read_csv(f))
      self.infer_overlap_df = pd.concat(dfs, ignore_index=True)
    else:
      pairwise_file = os.path.join(
          file_paths.HYBAS_COVERAGE_DIR,
          f'infer_pairwise_overlaps_{self.continent.upper()}.csv',
      )
      with gfile.Open(pairwise_file, 'r') as f:
        self.infer_overlap_df = pd.read_csv(f)

    self.infer_overlap_by_hybas = defaultdict(list)
    for b1, b2, area in zip(
        self.infer_overlap_df['basin1'],
        self.infer_overlap_df['basin2'],
        self.infer_overlap_df['overlap_area'],
    ):
      if area > 0:
        self.infer_overlap_by_hybas[b1].append((b2, float(area)))

    # 3. Slice Risk Data & Precomputed Base Skills to continent
    idx = self.df_base.index
    self.country_series = shared['country_series'].reindex(idx)
    self.vulnerability = shared['vuln_global'].reindex(idx).fillna(0.5)
    self.mpi = shared['mpi_global'].reindex(idx).fillna(0.5)
    self.exposures_df = shared['exposures_global_df'].reindex(idx).fillna(0.0)
    self.exposure = self.exposures_df[20]

    self.skill_base_by_rp = {
        T: shared['skill_base_global_by_rp'][T].reindex(idx)
        for T in [5, 10, 20, 50, 100]
    }
    self.skill_base = self.skill_base_by_rp[5]

    # Load canonical modules via importlib
    self.cov_lib = importlib.import_module(
        '0_calculate_overlap_fractions'
    )
    self.anom_lib = importlib.import_module(
        '1_anomaly_scores'
    )

  def _build_upstream_map(self):
    """Builds upstream/downstream maps and exact full upstream drainage areas (`m^2`)."""
    gdf = self.hybas.gdf
    adj = defaultdict(list)
    parent_map_raw = {}
    for child, next_down in zip(gdf.index, gdf['NEXT_DOWN']):
      if pd.notna(next_down) and next_down != '0' and next_down != 0 and next_down != 'nan':
        parent = str(int(next_down)) if isinstance(next_down, float) else str(next_down)
        adj[parent].append(child)
        parent_map_raw[child] = parent

    sub_area_raw = dict(
        zip(
            [i.replace('hybas_', '') for i in self.hybas_sub_areas.index],
            self.hybas_sub_areas.values,
        )
    )
    upstream_map_raw = {}
    total_area_raw = {}

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 50000))

    def get_upstream(node):
      if node in upstream_map_raw:
        return upstream_map_raw[node], total_area_raw[node]
      upstreams = [node]
      tot_a = sub_area_raw.get(node, 0.0)
      for child in adj.get(node, ()):
        c_ups, c_area = get_upstream(child)
        upstreams.extend(c_ups)
        tot_a += c_area
      upstream_map_raw[node] = upstreams
      total_area_raw[node] = tot_a
      return upstreams, tot_a

    for node in gdf.index:
      get_upstream(node)

    self.upstream_map = {}
    self.downstream_map = {}
    total_areas_dict = {}

    for node, upstreams in upstream_map_raw.items():
      p_node = node if node.startswith('hybas_') else 'hybas_' + node
      self.upstream_map[p_node] = [
          u if u.startswith('hybas_') else 'hybas_' + u for u in upstreams
      ]
      total_areas_dict[p_node] = total_area_raw[node]

      # Trace downstream path to coastal/endorheic outlet
      d_path = []
      curr = node
      visited = {curr}
      while curr in parent_map_raw:
        nxt = parent_map_raw[curr]
        if nxt in visited or nxt not in upstream_map_raw:
          break
        visited.add(nxt)
        d_path.append(nxt if nxt.startswith('hybas_') else 'hybas_' + nxt)
        curr = nxt
      self.downstream_map[p_node] = d_path

    self.hybas_total_areas = pd.Series(total_areas_dict)

  def _get_downstream_ids(self, w: str) -> list[str]:
    """Returns the downstream path from w to the outlet."""
    return self.downstream_map.get(w, [])

  def _get_affected_ids(self, new_gauges: list[str]) -> list[str]:
    """Returns all basins on this continent whose river-network features change."""
    valid_index = self.df_base.index
    affected = set()
    for w in new_gauges:
      if w not in self.upstream_map and w not in valid_index:
        continue
      affected.add(w)
      affected.update(self.upstream_map.get(w, [w]))
      affected.update(self.downstream_map.get(w, []))
    return [b for b in affected if b in valid_index]

  def _recalculate_coverage(self, new_gauges: list[str]) -> pd.DataFrame:
    """Computes exact upstream/downstream/self coverage on affected basins."""
    affected_ids = self._get_affected_ids(new_gauges)
    if not affected_ids:
      return pd.DataFrame(
          columns=[
              'total_upstream_coverage',
              'max_upstream_coverage',
              'upstream_gauge_count',
              'max_downstream_coverage',
          ]
      )
    affected_set = set(affected_ids)

    # Filter base HyBAS-to-Caravan overlaps strictly to affected_ids BEFORE concat
    ti_base = self.infer_overlap_df[
        self.infer_overlap_df['basin1'].isin(affected_set)
    ]

    new_rows_ti = []
    new_rows_tt = []
    new_gauge_areas = {}
    upstream_sets = {w: set(self.upstream_map.get(w, [w])) for w in new_gauges}

    for w in new_gauges:
      if w not in self.hybas_total_areas.index:
        continue
      g_w = f'NEW_GAUGE_{w}'
      w_area = float(self.hybas_total_areas.loc[w])
      new_gauge_areas[g_w] = w_area
      new_rows_tt.append({'basin1': g_w, 'basin2': g_w, 'overlap_area': w_area})

      # 1. Downstream basins d of w: U(d) \cap U(w) = U(w), area(g_w) < area(d)
      for d in self.downstream_map.get(w, ()):
        if d in affected_set:
          new_rows_ti.append({
              'basin1': d,
              'basin2': g_w,
              'overlap_area': w_area,
          })

      # 2. Upstream basins u of w (u != w): U(u) \cap U(w) = U(u), area(g_w) > area(u)
      for u in self.upstream_map.get(w, ()):
        if u != w and u in affected_set and u in self.hybas_total_areas.index:
          new_rows_ti.append({
              'basin1': u,
              'basin2': g_w,
              'overlap_area': float(self.hybas_total_areas.loc[u]),
          })

      # 3. Exact full-drainage overlaps between U(w) and existing Caravan gauges
      for g_caravan, ov_area in self.infer_overlap_by_hybas.get(w, ()):
        new_rows_tt.append({'basin1': g_w, 'basin2': g_caravan, 'overlap_area': ov_area})
        new_rows_tt.append({'basin1': g_caravan, 'basin2': g_w, 'overlap_area': ov_area})

      # 4. Containment overlaps between nested new gauges (prevents double-counting!)
      for w2 in new_gauges:
        if w == w2 or w2 not in self.hybas_total_areas.index:
          continue
        if w in upstream_sets[w2]:
          g_w2 = f'NEW_GAUGE_{w2}'
          new_rows_tt.append({'basin1': g_w, 'basin2': g_w2, 'overlap_area': w_area})
          new_rows_tt.append({'basin1': g_w2, 'basin2': g_w, 'overlap_area': w_area})

    ti_df = (
        pd.concat([ti_base, pd.DataFrame(new_rows_ti)], ignore_index=True)
        if new_rows_ti
        else ti_base
    )
    tt_df = (
        pd.concat([self.caravan_tt_df, pd.DataFrame(new_rows_tt)], ignore_index=True)
        if new_rows_tt
        else self.caravan_tt_df
    )

    combined_areas = pd.concat([
        self.caravan_areas,
        pd.Series(new_gauge_areas, dtype=float),
        self.hybas_total_areas,
    ])
    combined_areas = combined_areas[~combined_areas.index.duplicated(keep='first')]

    updated_up = self.cov_lib.calculate_total_upstream_coverage(
        infer_ids=affected_ids,
        tt_df=tt_df,
        ti_df=ti_df,
        basin_areas=combined_areas,
    )
    updated_down = self.cov_lib.calculate_max_downstream_coverage(
        infer_ids=affected_ids,
        ti_df=ti_df,
        basin_areas=combined_areas,
    )
    updated_cov = updated_up.copy()
    updated_cov['max_downstream_coverage'] = updated_down

    # At each newly gauged outlet w itself, 100% of U(w) is gauged (consistent with Factor 2)
    for w in new_gauges:
      if w in updated_cov.index:
        updated_cov.loc[w, 'total_upstream_coverage'] = 1.0
        updated_cov.loc[w, 'max_upstream_coverage'] = 1.0
        updated_cov.loc[w, 'max_downstream_coverage'] = 1.0
        updated_cov.loc[w, 'upstream_gauge_count'] = max(
            int(updated_cov.loc[w, 'upstream_gauge_count']), 1
        )
    return updated_cov

  def _update_anomaly_scores(
      self, new_gauges: list[str], eval_ids: list[str]
  ) -> pd.DataFrame:
    """Refits anomaly models on Caravan + new_gauges and predicts on eval_ids."""
    valid_new = [g for g in new_gauges if g in self.df_base.index]
    new_gauges_phys = self.df_base.loc[valid_new, self.physical_cols]
    X_train = pd.concat([self.caravan_features, new_gauges_phys], axis=0)
    X_test = self.df_base.loc[eval_ids, self.physical_cols].copy()
    scores_dict = self.anom_lib.calculate_anomaly_scores(X_train, X_test)
    return pd.DataFrame(scores_dict, index=eval_ids)

  def evaluate(
      self,
      new_gauges: list[str],
      use_gauged_for_self: bool = False,
      precomputed_anom_df: pd.DataFrame | None = None,
      extra_eval_ids: list[str] | None = None,
      update_anomalies: bool = True,
  ) -> dict:
    """Calculates Expected Annual Exposure Reduction of adding `new_gauges`.

    By default (`use_gauged_for_self=False`), all predictions—including at newly
    gauged outlets via 100% topological coverage and updated multivariate OOD
    scores—are evaluated on `M_ungauged`, while also recording the independent
    `M_gauged` self-site confirmation metric (`global_risk_2_exp_mgauged_confirm`).
    """
    if not new_gauges and not extra_eval_ids:
      return {}

    new_gauges_prefixed = [
        g if g.startswith('hybas_') else 'hybas_' + g for g in new_gauges
    ]
    local_new_gauges = [g for g in new_gauges_prefixed if g in self.df_base.index]
    affected_ids = self._get_affected_ids(local_new_gauges)

    eval_ids_set = set(affected_ids)
    if extra_eval_ids:
      eval_ids_set.update(b for b in extra_eval_ids if b in self.df_base.index)
    eval_ids = list(eval_ids_set)
    if not eval_ids:
      return {}

    X_eval = self.df_base.loc[eval_ids].copy()

    # 1. Update topological coverage on river-network affected_ids
    if affected_ids:
      updated_cov = self._recalculate_coverage(local_new_gauges)
      for col in updated_cov.columns:
        X_eval.loc[affected_ids, col] = updated_cov[col].astype(X_eval[col].dtype)

    # 2. Update multivariate OOD anomaly scores on eval_ids
    if precomputed_anom_df is not None:
      common_anom = [b for b in eval_ids if b in precomputed_anom_df.index]
      if common_anom:
        for col in ('anomaly_score_if', 'anomaly_score_ocsvm', 'anomaly_score_lof'):
          if col in precomputed_anom_df.columns:
            X_eval.loc[common_anom, col] = precomputed_anom_df.loc[
                common_anom, col
            ].astype(X_eval[col].dtype)
    elif update_anomalies and local_new_gauges:
      df_anom = self._update_anomaly_scores(local_new_gauges, eval_ids)
      for col in df_anom.columns:
        X_eval.loc[eval_ids, col] = df_anom[col].astype(X_eval[col].dtype)

    # 3. Predict multi-return-period skill ONLY on eval_ids
    delta_eval_by_rp = {}
    delta_confirm_eval_by_rp = {}
    skill_new_eval_5 = None

    for T in [5, 10, 20]:
      preds_u = pd.Series(
          self.ungauged_models[T].predict(X_eval), index=eval_ids, dtype=float
      )
      base_u = self.skill_base_by_rp[T].loc[eval_ids]

      preds_confirm = preds_u.copy()
      if local_new_gauges:
        preds_g_self = pd.Series(
            self.gauged_models[T].predict(X_eval.loc[local_new_gauges]),
            index=local_new_gauges,
            dtype=float,
        )
        preds_confirm.loc[local_new_gauges] = preds_g_self

      preds_primary = preds_confirm if use_gauged_for_self else preds_u
      if T == 5:
        skill_new_eval_5 = preds_primary

      delta_eval_by_rp[T] = (preds_primary - base_u).clip(lower=0.0)
      delta_confirm_eval_by_rp[T] = (preds_confirm - base_u).clip(lower=0.0)

    delta_eval_by_rp[50] = delta_eval_by_rp[20]
    delta_eval_by_rp[100] = delta_eval_by_rp[20]
    delta_confirm_eval_by_rp[50] = delta_confirm_eval_by_rp[20]
    delta_confirm_eval_by_rp[100] = delta_confirm_eval_by_rp[20]

    # 4. Trapezoidal integration on eval_ids (all non-eval_ids have delta == 0)
    return_periods = [5, 10, 20, 50, 100]
    vuln_sub = self.vulnerability.loc[eval_ids]
    mpi_sub = self.mpi.loc[eval_ids]
    exp_sub = self.exposures_df.loc[eval_ids]

    r1 = pd.Series(
        stats_utils.calculate_trapezoidal_expected_annual_risk(
            {T: (delta_eval_by_rp[T] * vuln_sub).values for T in return_periods},
            return_periods=return_periods,
        ),
        index=eval_ids,
    )
    r2 = pd.Series(
        stats_utils.calculate_trapezoidal_expected_annual_risk(
            {T: (delta_eval_by_rp[T] * exp_sub[T]).values for T in return_periods},
            return_periods=return_periods,
        ),
        index=eval_ids,
    )
    r2_confirm = pd.Series(
        stats_utils.calculate_trapezoidal_expected_annual_risk(
            {
                T: (delta_confirm_eval_by_rp[T] * exp_sub[T]).values
                for T in return_periods
            },
            return_periods=return_periods,
        ),
        index=eval_ids,
    )
    r3 = pd.Series(
        stats_utils.calculate_trapezoidal_expected_annual_risk(
            {
                T: (delta_eval_by_rp[T] * vuln_sub * exp_sub[T]).values
                for T in return_periods
            },
            return_periods=return_periods,
        ),
        index=eval_ids,
    )
    r4 = pd.Series(
        stats_utils.calculate_trapezoidal_expected_annual_risk(
            {
                T: (delta_eval_by_rp[T] * exp_sub[T] * mpi_sub).values
                for T in return_periods
            },
            return_periods=return_periods,
        ),
        index=eval_ids,
    )

    affected_idx = [b for b in affected_ids if b in r2.index]
    r2_river_network = float(r2.loc[affected_idx].sum()) if affected_idx else 0.0

    candidate_w = new_gauges_prefixed[0] if new_gauges_prefixed else eval_ids[0]
    delta_5 = delta_eval_by_rp[5]
    top_10_basins = r2.nlargest(min(10, len(r2))).index.tolist()

    cand_country = (
        self.country_series.get(candidate_w, None)
        if candidate_w in self.country_series.index
        else None
    )
    within_country_exp = 0.0
    if pd.notna(cand_country):
      same_c = [
          b for b in eval_ids if self.country_series.get(b, None) == cand_country
      ]
      if same_c:
        within_country_exp = float(r2.loc[same_c].sum())

    res = {
        'candidate_hybas_id': candidate_w,
        'value': float(r2.sum()),
        'candidate_base_skill': float(self.skill_base.get(candidate_w, 0.0)),
        'candidate_new_skill': float(skill_new_eval_5.get(candidate_w, 0.0)),
        'candidate_delta_skill': float(delta_5.get(candidate_w, 0.0)),
        'candidate_risk_1_vuln': float(r1.get(candidate_w, 0.0)),
        'candidate_risk_2_exp': float(r2.get(candidate_w, 0.0)),
        'candidate_risk_2_exp_mgauged_confirm': float(
            r2_confirm.get(candidate_w, 0.0)
        ),
        'candidate_risk_3_vuln_exp': float(r3.get(candidate_w, 0.0)),
        'candidate_risk_4_exp_mpi': float(r4.get(candidate_w, 0.0)),
        'within_country_risk_2_exp': within_country_exp,
        'river_network_risk_2_exp': r2_river_network,
        'global_risk_1_vuln': float(r1.sum()),
        'global_risk_2_exp': float(r2.sum()),
        'global_risk_2_exp_mgauged_confirm': float(r2_confirm.sum()),
        'global_risk_3_vuln_exp': float(r3.sum()),
        'global_risk_4_exp_mpi': float(r4.sum()),
    }

    for idx_k, b_id in enumerate(top_10_basins):
      k = idx_k + 1
      res[f'top_{k}_basin_id'] = b_id
      res[f'top_{k}_base_skill'] = float(self.skill_base.get(b_id, 0.0))
      res[f'top_{k}_new_skill'] = float(skill_new_eval_5.get(b_id, 0.0))
      res[f'top_{k}_delta_skill'] = float(delta_5.get(b_id, 0.0))
      res[f'top_{k}_risk_1_vuln'] = float(r1.get(b_id, 0.0))
      res[f'top_{k}_risk_2_exp'] = float(r2.get(b_id, 0.0))
      res[f'top_{k}_risk_3_vuln_exp'] = float(r3.get(b_id, 0.0))
      res[f'top_{k}_risk_4_exp_mpi'] = float(r4.get(b_id, 0.0))

    return res

  def evaluate_sets_batched(
      self,
      named_sets: list[tuple[str, list[str], str | None]],
  ) -> list[dict]:
    """Evaluates many gauge sets in one vectorized continent batch on `M_ungauged`.

    Args:
      named_sets: List of `(set_id, gauge_list, target_country_or_None)` tuples.
        For single-gauge or country-level `N <= 20` evaluations where adding
        `<= 20` unweighted stations to `26,572` Caravan stations changes global
        OOD scores by `< 0.08%`, batching all modified river-network rows into a
        single `M_ungauged.predict()` call per return period evaluates thousands
        of candidate configurations in seconds with zero river-network
        double-counting.

    Returns:
      List of evaluation dictionaries corresponding to `named_sets`.
    """
    batch_frames = []
    batch_meta = []

    for set_id, gauges, target_country in named_sets:
      local_gauges = [g for g in gauges if g in self.df_base.index]
      if not local_gauges:
        continue
      aff_ids = self._get_affected_ids(local_gauges)
      if not aff_ids:
        continue
      updated_cov = self._recalculate_coverage(local_gauges)
      X_sub = self.df_base.loc[aff_ids].copy()
      for col in updated_cov.columns:
        X_sub.loc[aff_ids, col] = updated_cov[col].astype(X_sub[col].dtype)
      start_row = sum(len(f) for f in batch_frames)
      batch_frames.append(X_sub)
      batch_meta.append(
          (set_id, local_gauges, target_country, aff_ids, start_row, len(X_sub))
      )

    if not batch_frames:
      return []

    X_big = pd.concat(batch_frames, axis=0)
    self_mask = np.zeros(len(X_big), dtype=bool)
    for _, local_gauges, _, aff_ids, start_row, n_rows in batch_meta:
      lg_set = set(local_gauges)
      for offset, b_id in enumerate(aff_ids):
        if b_id in lg_set:
          self_mask[start_row + offset] = True

    preds_u_big = {}
    preds_g_self = {}
    for T in [5, 10, 20]:
      preds_u_big[T] = np.asarray(self.ungauged_models[T].predict(X_big), dtype=float)
      if self_mask.any():
        preds_g_self[T] = np.asarray(
            self.gauged_models[T].predict(X_big.iloc[np.where(self_mask)[0]]),
            dtype=float,
        )

    self_positions = np.where(self_mask)[0]
    self_pos_to_g_idx = {pos: i for i, pos in enumerate(self_positions)}

    return_periods = [5, 10, 20, 50, 100]
    outputs = []

    for set_id, local_gauges, target_country, aff_ids, start_row, n_rows in batch_meta:
      sl = slice(start_row, start_row + n_rows)
      delta_rp = {}
      delta_confirm_rp = {}
      skill_new_5 = None

      for T in [5, 10, 20]:
        pu = pd.Series(preds_u_big[T][sl], index=aff_ids, dtype=float)
        bu = self.skill_base_by_rp[T].loc[aff_ids]
        pc = pu.copy()
        for offset, b_id in enumerate(aff_ids):
          abs_pos = start_row + offset
          if abs_pos in self_pos_to_g_idx:
            pc.loc[b_id] = preds_g_self[T][self_pos_to_g_idx[abs_pos]]
        if T == 5:
          skill_new_5 = pu
        delta_rp[T] = (pu - bu).clip(lower=0.0)
        delta_confirm_rp[T] = (pc - bu).clip(lower=0.0)

      delta_rp[50] = delta_rp[20]
      delta_rp[100] = delta_rp[20]
      delta_confirm_rp[50] = delta_confirm_rp[20]
      delta_confirm_rp[100] = delta_confirm_rp[20]

      exp_sub = self.exposures_df.loc[aff_ids]
      vuln_sub = self.vulnerability.loc[aff_ids]
      mpi_sub = self.mpi.loc[aff_ids]

      r2 = pd.Series(
          stats_utils.calculate_trapezoidal_expected_annual_risk(
              {T: (delta_rp[T] * exp_sub[T]).values for T in return_periods},
              return_periods=return_periods,
          ),
          index=aff_ids,
      )
      r2_confirm = pd.Series(
          stats_utils.calculate_trapezoidal_expected_annual_risk(
              {
                  T: (delta_confirm_rp[T] * exp_sub[T]).values
                  for T in return_periods
              },
              return_periods=return_periods,
          ),
          index=aff_ids,
      )
      r3 = pd.Series(
          stats_utils.calculate_trapezoidal_expected_annual_risk(
              {
                  T: (delta_rp[T] * vuln_sub * exp_sub[T]).values
                  for T in return_periods
              },
              return_periods=return_periods,
          ),
          index=aff_ids,
      )
      r4 = pd.Series(
          stats_utils.calculate_trapezoidal_expected_annual_risk(
              {
                  T: (delta_rp[T] * exp_sub[T] * mpi_sub).values
                  for T in return_periods
              },
              return_periods=return_periods,
          ),
          index=aff_ids,
      )

      cand_w = local_gauges[0]
      c_name = (
          target_country
          if target_country is not None
          else self.country_series.get(cand_w, None)
      )
      within_c_exp = 0.0
      within_c_confirm = 0.0
      if pd.notna(c_name):
        c_basins = [
            b for b in aff_ids if self.country_series.get(b, None) == c_name
        ]
        if c_basins:
          within_c_exp = float(r2.loc[c_basins].sum())
          within_c_confirm = float(r2_confirm.loc[c_basins].sum())

      row_out = {
          'set_id': set_id,
          'candidate_hybas_id': cand_w,
          'country_name': c_name,
          'n_local_gauges': len(local_gauges),
          'n_affected_basins': len(aff_ids),
          'value': float(r2.sum()),
          'candidate_base_skill': float(self.skill_base.get(cand_w, 0.0)),
          'candidate_new_skill': float(skill_new_5.get(cand_w, 0.0)),
          'candidate_delta_skill': float(delta_rp[5].get(cand_w, 0.0)),
          'candidate_risk_2_exp': float(r2.get(cand_w, 0.0)),
          'candidate_risk_2_exp_mgauged_confirm': float(
              r2_confirm.get(cand_w, 0.0)
          ),
          'within_country_risk_2_exp': within_c_exp,
          'within_country_risk_2_exp_mgauged_confirm': within_c_confirm,
          'global_risk_2_exp': float(r2.sum()),
          'global_risk_2_exp_mgauged_confirm': float(r2_confirm.sum()),
          'global_risk_3_vuln_exp': float(r3.sum()),
          'global_risk_4_exp_mpi': float(r4.sum()),
      }

      top_10 = r2.nlargest(min(10, len(r2))).index.tolist()
      for idx_k, b_id in enumerate(top_10):
        k = idx_k + 1
        row_out[f'top_{k}_basin_id'] = b_id
        row_out[f'top_{k}_risk_2_exp'] = float(r2.get(b_id, 0.0))
        row_out[f'top_{k}_delta_skill'] = float(delta_rp[5].get(b_id, 0.0))

      outputs.append(row_out)

    return outputs

  def test_load(self):
    print('Base features shape:', self.df_base.shape)
    print('Vulnerability shape:', self.vulnerability.shape)
    print('Exposure shape:', self.exposure.shape)
    print('Upstream map size:', len(self.upstream_map))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  evaluator = GaugeValueEvaluator(FLAGS.metric, FLAGS.continent)
  evaluator.test_load()

  if FLAGS.candidates_file:
    with gfile.Open(FLAGS.candidates_file, 'r') as f:
      candidates_df = pd.read_csv(f)
    candidates_list = [
        c if str(c).startswith('hybas_') else 'hybas_' + str(c)
        for c in candidates_df['hybas_id']
    ]
    candidates_list = [c for c in candidates_list if c in evaluator.hybas_ids_set]
    start = FLAGS.start_idx
    end = FLAGS.end_idx if FLAGS.end_idx != -1 else len(candidates_list)
    sliced = candidates_list[start:end]

    out_dir = (
        FLAGS.output_dir
        if FLAGS.output_dir
        else os.path.join(file_paths.V2_RESULTS_DIR, 'evaluation_results_single_gauge')
    )
    if not gfile.Exists(out_dir):
      gfile.MakeDirs(out_dir)

    named_sets = [(c, [c], None) for c in sliced]
    t0 = time.time()
    batch_res = evaluator.evaluate_sets_batched(named_sets)
    out_file = os.path.join(out_dir, f'results_{FLAGS.continent}_{start}_{end}.csv')
    with gfile.Open(out_file, 'w') as f:
      pd.DataFrame(batch_res).to_csv(f, index=False)
    print(
        f"[{time.strftime('%H:%M:%S')}] Evaluated {len(batch_res)} candidates "
        f"in {time.time() - t0:.2f}s -> {out_file}"
    )
  else:
    import random
    candidate = random.choice(evaluator.hybas_ids)
    t0 = time.time()
    value_dict = evaluator.evaluate([candidate])
    print(
        f"Evaluated {candidate} in {time.time() - t0:.2f}s | "
        f"Global Exposure Reduced (M_ungauged): {value_dict.get('global_risk_2_exp', 0.0):.2f}"
    )


if __name__ == '__main__':
  app.run(main)
