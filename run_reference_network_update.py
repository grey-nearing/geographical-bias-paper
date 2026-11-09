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

"""End-to-end reference network update pipeline for Geographical Bias v2.

Re-runs incremental spatial overlap and river-network coverage calculations,
global HydroBASINS out-of-distribution anomaly scoring, inference dataset
assembly, 38-metric global skill-predictor inference, precompiled summary
statistics, unified 5-factor exposure decomposition, targeted N-gauge siting
optimization, and 1,000-watershed stratified exact KernelSHAP attributions
after expanding the open-access reference gauge network (e.g., adding the
5,356 Caravan-GRDC extension gauges to reach 38,069 total open gauges).
"""

import concurrent.futures
import copy
import glob
import importlib
import os
import pickle
import shutil
import time

from absl import app
from absl import flags
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import shap
import shapely
from shapely import STRtree
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import xgboost

import evaluate_n_gauge_sets
from utils import file_paths
from utils import io_utils
from utils import skill_predictor as sp
from utils import gfile_compat as gfile
import multiprocessing

overlap_mod = importlib.import_module(
    '0_calculate_overlap_fractions'
)
anom_mod = importlib.import_module(
    '1_anomaly_scores'
)
datasets_mod = importlib.import_module(
    '3_create_datasets'
)
compile_t1_mod = importlib.import_module(
    '7_compile_table_1_stats'
)
five_factor_mod = importlib.import_module(
    '12_five_factor_exposure_decomposition'
)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'backup_dir',
    'data/backup_v2_32713_gauges',
    'Directory containing cached pre-update pairwise overlaps for incremental update.',
)
flags.DEFINE_string(
    'pq_cache_dir',
    '/tmp/geo_bias_pq_cache',
    'Optional GeoParquet cache directory for fast polygon loading.',
)
flags.DEFINE_integer(
    'start_stage',
    1,
    'Stage to start from (1..8).',
)
flags.DEFINE_integer(
    'end_stage',
    8,
    'Stage to stop after (1..8).',
)

# Fork-shared read-only globals
G_NEW_GRDC_GDF = None
G_FULL_CARAVAN_AREAS = None
G_CARAVAN_TT_DF = None
G_X_INFER_GAUGED = None
G_X_INFER_UNGAUGED = None
G_DF_HYBAS_1000 = None
G_DF_UNGAUGED_GENERIC = None
G_BUF_GEOMS = None
G_INT_GEOMS1 = None
G_INT_GEOMS2 = None
G_INT_P1 = None
G_INT_P2 = None


def _safe_unlink(path: str):
  if os.path.lexists(path):
    if os.path.isdir(path) and not os.path.islink(path):
      shutil.rmtree(path)
    else:
      os.unlink(path)


def _buf_worker(s_e):
  s, e = s_e
  return s, shapely.buffer(G_BUF_GEOMS[s:e], 0)


def _parallel_buffer0(geoms: np.ndarray, ctx=None) -> np.ndarray:
  del ctx
  global G_BUF_GEOMS
  n = len(geoms)
  if n == 0:
    return geoms
  if n < 200:
    return shapely.buffer(geoms, 0)
  G_BUF_GEOMS = geoms
  n_workers = min(56, max(1, n // 50))
  edges = np.linspace(0, n, n_workers + 1, dtype=int)
  tasks = [(int(edges[i]), int(edges[i + 1])) for i in range(n_workers) if edges[i] < edges[i + 1]]
  out = np.empty(n, dtype=object)
  with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
    for s, cleaned in ex.map(_buf_worker, tasks):
      out[s : s + len(cleaned)] = cleaned
  G_BUF_GEOMS = None
  return out


def _intersect_worker(s_e):
  s, e = s_e
  p1 = G_INT_P1[s:e]
  p2 = G_INT_P2[s:e]
  g1 = G_INT_GEOMS1[p1]
  g2 = G_INT_GEOMS2[p2]
  hit = shapely.intersects(g1, g2)
  if not np.any(hit):
    return None
  p1_h = p1[hit]
  p2_h = p2[hit]
  areas = shapely.area(shapely.intersection(g1[hit], g2[hit]))
  keep = areas > 1e-6
  if not np.any(keep):
    return None
  return p1_h[keep], p2_h[keep], areas[keep]


def _fast_exact_overlaps(
    gdf1: gpd.GeoDataFrame,
    gdf2: gpd.GeoDataFrame | None,
    ctx,
    gdf1_precleaned: bool = False,
    gdf2_precleaned: bool = False,
) -> pd.DataFrame:
  """Computes exact EPSG:6933 overlap areas in parallel across 56 CPU cores."""
  global G_INT_GEOMS1, G_INT_GEOMS2, G_INT_P1, G_INT_P2
  t0 = time.time()
  is_self = gdf2 is None
  if is_self:
    gdf2 = gdf1
    gdf2_precleaned = gdf1_precleaned

  ids1 = gdf1.index.to_numpy()
  ids2 = gdf2.index.to_numpy()
  geoms1 = gdf1.geometry.to_numpy().copy()
  geoms2 = geoms1 if is_self else gdf2.geometry.to_numpy().copy()

  tree = STRtree(geoms2)
  pairs = tree.query(geoms1)
  p1 = pairs[0]
  p2 = pairs[1]
  if is_self:
    off_mask = p1 < p2
    p1 = p1[off_mask]
    p2 = p2[off_mask]

  if len(p1) == 0 and not is_self:
    return pd.DataFrame(columns=['basin1', 'basin2', 'overlap_area'])

  # Clean only geometries participating in candidate pairs (or all for self-areas)
  if is_self and not gdf1_precleaned:
    geoms1 = _parallel_buffer0(geoms1, ctx)
    geoms2 = geoms1
  else:
    if not gdf1_precleaned and len(p1) > 0:
      u1 = np.unique(p1)
      geoms1[u1] = _parallel_buffer0(geoms1[u1], ctx)
    if not is_self and not gdf2_precleaned and len(p2) > 0:
      u2 = np.unique(p2)
      geoms2[u2] = _parallel_buffer0(geoms2[u2], ctx)

  dfs = []
  if is_self:
    diag_areas = shapely.area(geoms1)
    keep_d = diag_areas > 1e-6
    dfs.append(
        pd.DataFrame({
            'basin1': ids1[keep_d],
            'basin2': ids1[keep_d],
            'overlap_area': diag_areas[keep_d],
        })
    )

  if len(p1) > 0:
    G_INT_GEOMS1 = geoms1
    G_INT_GEOMS2 = geoms2
    G_INT_P1 = p1
    G_INT_P2 = p2
    n_pairs = len(p1)
    n_chunks = min(224, max(1, n_pairs // 250))
    edges = np.linspace(0, n_pairs, n_chunks + 1, dtype=int)
    tasks = [
        (int(edges[i]), int(edges[i + 1]))
        for i in range(n_chunks)
        if edges[i] < edges[i + 1]
    ]
    n_workers = min(56, len(tasks))
    res_p1, res_p2, res_a = [], [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
      for out in ex.map(_intersect_worker, tasks):
        if out is not None:
          res_p1.append(out[0])
          res_p2.append(out[1])
          res_a.append(out[2])
    G_INT_GEOMS1 = G_INT_GEOMS2 = G_INT_P1 = G_INT_P2 = None

    if res_p1:
      fp1 = np.concatenate(res_p1)
      fp2 = np.concatenate(res_p2)
      fa = np.concatenate(res_a)
      df_off = pd.DataFrame({
          'basin1': ids1[fp1],
          'basin2': ids2[fp2],
          'overlap_area': fa,
      })
      dfs.append(df_off)
      if is_self:
        dfs.append(
            df_off.rename(columns={'basin1': 'basin2', 'basin2': 'basin1'})[
                ['basin1', 'basin2', 'overlap_area']
            ]
        )

  out_df = (
      pd.concat(dfs, ignore_index=True)
      if dfs
      else pd.DataFrame(columns=['basin1', 'basin2', 'overlap_area'])
  )
  print(
      f"[{time.strftime('%H:%M:%S')}] Parallel overlaps: {len(p1):,} bbox pairs -> "
      f'{len(out_df):,} exact overlaps in {time.time() - t0:.2f}s',
      flush=True,
  )
  return out_df


def _read_gdf_fast(
    path: str,
    index_col: str = 'unique_gauge_id',
    clean_all: bool = False,
    ctx=None,
) -> gpd.GeoDataFrame:
  t0 = time.time()
  stem = os.path.splitext(os.path.basename(path))[0]
  pq_path = os.path.join(FLAGS.pq_cache_dir, f'{stem}.parquet')
  if os.path.exists(pq_path):
    gdf = gpd.read_parquet(pq_path)
  else:
    try:
      gdf = gpd.read_file(path, engine='pyogrio')
    except Exception:  # pylint: disable=broad-exception-caught
      with gfile.Open(path, 'rb') as f:
        gdf = gpd.read_file(f)
  if index_col in gdf.columns:
    gdf = gdf.set_index(index_col)
  if gdf.crs and gdf.crs.to_epsg() == 4326:
    gdf = gdf.to_crs('EPSG:6933')
  if clean_all:
    gdf.geometry = _parallel_buffer0(gdf.geometry.to_numpy(), ctx)
  print(
      f"[{time.strftime('%H:%M:%S')}] Loaded {os.path.basename(path)} "
      f'({len(gdf)} polygons, clean_all={clean_all}) in {time.time() - t0:.1f}s',
      flush=True,
  )
  return gdf


def _process_continent_overlaps(continent: str, backup_dir: str, ctx):
  t0 = time.time()
  hybas_path = os.path.join(
      file_paths.HYBAS_POLYGONS_DIR, f'hybas_{continent.lower()}_lev12_v1c.geojson'
  )
  hybas_gdf = _read_gdf_fast(
      hybas_path, index_col='unique_gauge_id', clean_all=False, ctx=ctx
  )
  hybas_areas = pd.Series(
      shapely.area(hybas_gdf.geometry.to_numpy()), index=hybas_gdf.index
  )

  combined_areas = pd.concat([G_FULL_CARAVAN_AREAS, hybas_areas])
  combined_areas = combined_areas[~combined_areas.index.duplicated(keep='first')]

  df_hybas_new = _fast_exact_overlaps(
      gdf1=hybas_gdf,
      gdf2=G_NEW_GRDC_GDF,
      ctx=ctx,
      gdf1_precleaned=False,
      gdf2_precleaned=True,
  )

  old_pair_path = os.path.join(
      backup_dir, 'coverage_fractions/hybas', f'infer_pairwise_overlaps_{continent}.csv'
  )
  old_infer_overlap_df = pd.read_csv(old_pair_path)
  if old_infer_overlap_df.empty:
    infer_overlap_df = df_hybas_new.copy()
  elif df_hybas_new.empty:
    infer_overlap_df = old_infer_overlap_df.copy()
  else:
    infer_overlap_df = pd.concat(
        [old_infer_overlap_df, df_hybas_new], ignore_index=True
    )

  dst_pair_path = os.path.join(
      file_paths.HYBAS_COVERAGE_DIR, f'infer_pairwise_overlaps_{continent}.csv'
  )
  tmp_pair_path = dst_pair_path + f'.tmp_{os.getpid()}'
  infer_overlap_df.to_csv(tmp_pair_path, index=False)
  os.replace(tmp_pair_path, dst_pair_path)

  infer_ids = list(hybas_gdf.index)
  infer_up_df = overlap_mod.calculate_total_upstream_coverage(
      infer_ids=infer_ids,
      tt_df=G_CARAVAN_TT_DF,
      ti_df=infer_overlap_df,
      basin_areas=combined_areas,
  )
  infer_down_series = overlap_mod.calculate_max_downstream_coverage(
      infer_ids=infer_ids,
      ti_df=infer_overlap_df,
      basin_areas=combined_areas,
  )

  infer_df = infer_up_df.copy()
  infer_df['max_downstream_coverage'] = infer_down_series
  infer_df = infer_df.reset_index().rename(columns={'index': 'hybas_id'})
  infer_df['continent'] = continent

  dst_feat_path = os.path.join(
      file_paths.HYBAS_COVERAGE_DIR, f'infer_ml_features_{continent}.csv'
  )
  tmp_feat_path = dst_feat_path + f'.tmp_{os.getpid()}'
  infer_df.to_csv(tmp_feat_path, index=True)
  os.replace(tmp_feat_path, dst_feat_path)

  old_feat = pd.read_csv(
      os.path.join(
          backup_dir, 'coverage_fractions/hybas', f'infer_ml_features_{continent}.csv'
      ),
      index_col=0,
  )
  old_zero = int(
      (
          (old_feat['total_upstream_coverage'] == 0)
          & (old_feat['max_downstream_coverage'] == 0)
      ).sum()
  )
  new_zero = int(
      (
          (infer_df['total_upstream_coverage'] == 0)
          & (infer_df['max_downstream_coverage'] == 0)
      ).sum()
  )
  print(
      f"[{time.strftime('%H:%M:%S')}] DONE {continent} in {time.time() - t0:.1f}s | "
      f'new_overlaps={len(df_hybas_new)} | zero_cov: {old_zero} -> {new_zero} '
      f'(delta={new_zero - old_zero})',
      flush=True,
  )
  return {
      'continent': continent,
      'n_watersheds': len(infer_df),
      'old_zero': old_zero,
      'new_zero': new_zero,
      'new_overlaps': len(df_hybas_new),
  }


def _set_n_jobs(obj, n_jobs: int, _seen=None):
  if _seen is None:
    _seen = set()
  oid = id(obj)
  if oid in _seen or obj is None:
    return
  _seen.add(oid)
  if hasattr(obj, 'n_jobs'):
    try:
      setattr(obj, 'n_jobs', n_jobs)
    except Exception:  # pylint: disable=broad-exception-caught
      pass
  for attr in (
      'impl',
      'model',
      'ensemble',
      'final_estimator_',
      'final_estimator',
      'regressor_',
  ):
    if hasattr(obj, attr):
      _set_n_jobs(getattr(obj, attr), n_jobs, _seen)
  if hasattr(obj, 'estimators_') and isinstance(obj.estimators_, (list, tuple)):
    for e in obj.estimators_:
      _set_n_jobs(e, n_jobs, _seen)
  if hasattr(obj, 'steps') and isinstance(obj.steps, (list, tuple)):
    for _, step in obj.steps:
      _set_n_jobs(step, n_jobs, _seen)


SHAP_METRICS = [
    'NSE',
    'KGE',
    'log-NSE',
    'log-KGE',
    '1.01-year-2.0-window-recall',
    '2.0-year-2.0-window-recall',
    '5.0-year-2.0-window-recall',
    '10.0-year-2.0-window-recall',
    '20.0-year-2.0-window-recall',
    '1.01-year-2.0-window-precision',
    '2.0-year-2.0-window-precision',
    '5.0-year-2.0-window-precision',
    '10.0-year-2.0-window-precision',
    '20.0-year-2.0-window-precision',
]


def _run_infer_single(metric: str, model_type: str, x_infer: pd.DataFrame) -> str:
  t0 = time.time()
  file_paths.init(metric)
  if model_type == 'gauged':
    model_path = file_paths.FINAL_GAUGED_MODEL
    save_path = file_paths.GLOBAL_HYBAS_PREDICTIONS_GAUGED
    pred_col_name = 'gauged_skill'
  else:
    model_path = file_paths.FINAL_UNGAUGED_MODEL
    save_path = file_paths.GLOBAL_HYBAS_PREDICTIONS_UNGAUGED
    pred_col_name = 'ungauged_skill'

  with gfile.Open(model_path, 'rb') as f:
    model = pickle.load(f)
  _set_n_jobs(model, 56)

  preds = model.predict(x_infer)
  preds_df = preds.rename(pred_col_name).to_frame()
  tmp_path = save_path + f'.tmp_{os.getpid()}'
  preds_df.to_csv(tmp_path, index=True)
  os.replace(tmp_path, save_path)
  return f'{metric} ({model_type}): {time.time() - t0:.1f}s'


def _compute_ensemble_treeshap_from_bundle(args) -> str:
  metric, bundle_pkl = args
  with open(bundle_pkl, 'rb') as fp:
    df_hybas_1000, df_ungauged_generic = pickle.load(fp)
  return _compute_ensemble_treeshap_metric(
      metric, df_hybas_1000, df_ungauged_generic
  )


def _compute_ensemble_treeshap_metric(
    metric: str,
    df_hybas_1000: pd.DataFrame,
    df_ungauged_generic: pd.DataFrame,
) -> str:
  """Computes exact stacked-ensemble TreeSHAP attributions for 1,000 stratified watersheds."""
  t0 = time.time()
  n_samples = len(df_ungauged_generic) // 2
  df_random_generic = df_ungauged_generic.iloc[:n_samples]
  df_separated_generic = df_ungauged_generic.iloc[n_samples:]

  random_skill = io_utils.load_experiment_metrics(
      file_paths.METRICS_DIR, 'random', metric
  )
  separated_skill = io_utils.load_experiment_metrics(
      file_paths.METRICS_DIR, 'separated', metric
  )
  df_random = pd.concat(
      [df_random_generic, random_skill], axis=1, join='inner'
  ).dropna()
  df_separated = pd.concat(
      [df_separated_generic, separated_skill], axis=1, join='inner'
  ).dropna()
  x_train = pd.concat([df_random, df_separated], axis=0).drop(
      columns=['skill']
  )

  x_bg_raw = x_train.sample(n=150, random_state=42)
  x_inf_raw = df_hybas_1000.reindex(columns=x_train.columns, fill_value=0)

  model_path = os.path.join(
      file_paths.MODELS_DIR, f'final_ungauged_model_{metric}.pkl'
  )
  with gfile.Open(model_path, 'rb') as f:
    model = pickle.load(f)
  _set_n_jobs(model, 4)
  sp_impl = model.impl

  if hasattr(sp_impl, 'ensemble'):
    x_inf_trans = sp_impl._preprocess_features(x_inf_raw).reindex(
        columns=sp_impl.feature_columns, fill_value=0
    )
    x_bg_trans = sp_impl._preprocess_features(x_bg_raw).reindex(
        columns=sp_impl.feature_columns, fill_value=0
    )
    est_names = [n for n, _ in sp_impl.ensemble.estimators]
    est_dict = dict(zip(est_names, sp_impl.ensemble.estimators_))
    coef_map = dict(
        zip(est_names, np.ravel(sp_impl.ensemble.final_estimator_.coef_))
    )

    expl_hgbr = shap.TreeExplainer(est_dict['hgbr'], x_bg_trans)
    sv_hgbr = expl_hgbr.shap_values(x_inf_trans, check_additivity=False)

    et_pipe = est_dict['et']
    et_imp = et_pipe.named_steps['simpleimputer']
    et_reg = et_pipe.named_steps['extratreesregressor']
    x_inf_imp = pd.DataFrame(
        et_imp.transform(x_inf_trans),
        index=x_inf_trans.index,
        columns=x_inf_trans.columns,
    )
    x_bg_imp = pd.DataFrame(
        et_imp.transform(x_bg_trans),
        index=x_bg_trans.index,
        columns=x_bg_trans.columns,
    )
    et_sub = copy.copy(et_reg)
    et_sub.estimators_ = et_reg.estimators_[:80]
    expl_et = shap.TreeExplainer(et_sub, x_bg_imp)
    sv_et = expl_et.shap_values(x_inf_imp, check_additivity=False)

    w_hgbr = max(0.0, float(coef_map.get('hgbr', 0.5)))
    w_et = max(0.0, float(coef_map.get('et', 0.5)))
    w_sum = w_hgbr + w_et if (w_hgbr + w_et) > 0 else 1.0
    sv_comb = (w_hgbr * sv_hgbr + w_et * sv_et) / w_sum

    df_trans_shap = pd.DataFrame(
        sv_comb, index=x_inf_trans.index, columns=x_inf_trans.columns
    )
    df_out = df_trans_shap.reindex(columns=x_train.columns, fill_value=0.0).copy()
    if 'water_balance' in df_trans_shap.columns:
      df_out['pre_mm_uyr'] += 0.5 * df_trans_shap['water_balance']
      df_out['pet_mm_uyr'] += 0.5 * df_trans_shap['water_balance']
    if 'aridity_ratio' in df_trans_shap.columns:
      df_out['pre_mm_uyr'] += 0.5 * df_trans_shap['aridity_ratio']
      df_out['pet_mm_uyr'] += 0.5 * df_trans_shap['aridity_ratio']
    if 'total_forest_pc' in df_trans_shap.columns:
      f_cols = [
          f'glc_pc_u0{i}' for i in range(1, 6) if f'glc_pc_u0{i}' in df_out.columns
      ]
      for fc in f_cols:
        df_out[fc] += df_trans_shap['total_forest_pc'] / len(f_cols)
    if 'human_impact_density' in df_trans_shap.columns:
      if 'hft_ix_u09' in df_out.columns:
        df_out['hft_ix_u09'] += 0.5 * df_trans_shap['human_impact_density']
      if 'calculated_drain_area' in df_out.columns:
        df_out['calculated_drain_area'] += (
            0.5 * df_trans_shap['human_impact_density']
        )
  else:
    x_inf_trans = sp_impl._engineer_features(x_inf_raw).reindex(
        columns=sp_impl.feature_names, fill_value=0
    )
    x_bg_trans = sp_impl._engineer_features(x_bg_raw).reindex(
        columns=sp_impl.feature_names, fill_value=0
    )
    est_names = [n for n, _ in sp_impl.model.estimators]
    est_dict = dict(zip(est_names, sp_impl.model.estimators_))
    rscaler = sp_impl.model.final_estimator_.named_steps['robustscaler']
    ridge = sp_impl.model.final_estimator_.named_steps['ridgecv']
    scales = np.where(rscaler.scale_ == 0, 1.0, rscaler.scale_)
    coef_map = dict(zip(est_names, np.ravel(ridge.coef_) / scales))

    try:
      sv_xgb = shap.TreeExplainer(est_dict['xgb'], x_bg_trans).shap_values(
          x_inf_trans, check_additivity=False
      )
    except Exception:  # pylint: disable=broad-exception-caught
      sv_xgb = est_dict['xgb'].get_booster().predict(
          xgboost.DMatrix(x_inf_trans), pred_contribs=True
      )[:, :-1]

    try:
      sv_lgb = shap.TreeExplainer(est_dict['lgb'], x_bg_trans).shap_values(
          x_inf_trans, check_additivity=False
      )
    except Exception:  # pylint: disable=broad-exception-caught
      sv_lgb = est_dict['lgb'].predict(x_inf_trans, pred_contrib=True)[:, :-1]

    sv_hgb = shap.TreeExplainer(est_dict['hgb'], x_bg_trans).shap_values(
        x_inf_trans, check_additivity=False
    )

    sv_comb = (
        float(coef_map['xgb']) * sv_xgb
        + float(coef_map['lgb']) * sv_lgb
        + float(coef_map['hgb']) * sv_hgb
    )
    df_trans_shap = pd.DataFrame(
        sv_comb, index=x_inf_trans.index, columns=x_inf_trans.columns
    )
    df_out = df_trans_shap.reindex(columns=x_train.columns, fill_value=0.0).copy()

    if 'p_minus_pet' in df_trans_shap.columns:
      df_out['pre_mm_uyr'] += 0.5 * df_trans_shap['p_minus_pet']
      df_out['pet_mm_uyr'] += 0.5 * df_trans_shap['p_minus_pet']
    if 'p_div_pet' in df_trans_shap.columns:
      df_out['pre_mm_uyr'] += 0.5 * df_trans_shap['p_div_pet']
      df_out['pet_mm_uyr'] += 0.5 * df_trans_shap['p_div_pet']
    if 'et_deficit' in df_trans_shap.columns:
      df_out['pet_mm_uyr'] += 0.5 * df_trans_shap['et_deficit']
      df_out['aet_mm_uyr'] += 0.5 * df_trans_shap['et_deficit']
    if 'soil_porosity' in df_trans_shap.columns:
      df_out['snd_pc_uav'] += 0.5 * df_trans_shap['soil_porosity']
      df_out['cly_pc_uav'] += 0.5 * df_trans_shap['soil_porosity']
    if 'twi_proxy' in df_trans_shap.columns:
      df_out['calculated_drain_area'] += 0.5 * df_trans_shap['twi_proxy']
      df_out['slp_dg_uav'] += 0.5 * df_trans_shap['twi_proxy']
    f_cols = [c for c in df_out.columns if 'glc_pc_u0' in c and c[-1] in '12345']
    if 'total_forest_cover' in df_trans_shap.columns and f_cols:
      for fc in f_cols:
        df_out[fc] += df_trans_shap['total_forest_cover'] / len(f_cols)
    arid_cols = [c for c in ['glc_pc_u16', 'glc_pc_u20', 'glc_pc_u21'] if c in df_out.columns]
    if 'total_arid_cover' in df_trans_shap.columns and arid_cols:
      for ac in arid_cols:
        df_out[ac] += df_trans_shap['total_arid_cover'] / len(arid_cols)
    if 'human_intensity' in df_trans_shap.columns:
      for hc in ['gdp_ud_usu', 'pop_ct_usu', 'rdd_mk_uav']:
        if hc in df_out.columns:
          df_out[hc] += df_trans_shap['human_intensity'] / 3.0
    for lf in [
        'calculated_drain_area',
        'rev_mc_usu',
        'ria_ha_usu',
        'riv_tc_usu',
        'pop_ct_usu',
    ]:
      if f'log_{lf}' in df_trans_shap.columns and lf in df_out.columns:
        df_out[lf] += df_trans_shap[f'log_{lf}']
    if 'precip_temp_inter' in df_trans_shap.columns:
      df_out['pre_mm_uyr'] += 0.5 * df_trans_shap['precip_temp_inter']
      df_out['tmp_dc_uyr'] += 0.5 * df_trans_shap['precip_temp_inter']
    if 'forest_aridity_inter' in df_trans_shap.columns:
      df_out['ari_ix_uav'] += 0.5 * df_trans_shap['forest_aridity_inter']
      if f_cols:
        for fc in f_cols:
          df_out[fc] += 0.5 * df_trans_shap['forest_aridity_inter'] / len(f_cols)

  sdir = os.path.join(
      file_paths.PREDS_DIR, 'shap', metric, 'ungauged', 'stratified_1000'
  )
  os.makedirs(sdir, exist_ok=True)
  for old_shard in glob.glob(
      os.path.join(sdir, 'shard_*_exact_kernelshap_scores.csv')
  ):
    _safe_unlink(old_shard)
  cons_path = os.path.join(
      sdir, f'consolidated_exact_kernelshap_scores_{metric}.csv'
  )
  tmp_cons = cons_path + f'.tmp_{os.getpid()}_{metric}'
  df_out.to_csv(tmp_cons, index=True)
  os.replace(tmp_cons, cons_path)
  return f'   [SHAP] {metric:32s} done in {time.time() - t0:.1f}s -> {cons_path}'


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  global G_NEW_GRDC_GDF, G_FULL_CARAVAN_AREAS, G_CARAVAN_TT_DF

  t_total = time.time()
  backup_dir = FLAGS.backup_dir
  ctx = None
  df_hybas = None

  # ==========================================================================
  # STAGE 1: Incremental Spatial Overlaps & River-Network Coverage Features
  # ==========================================================================
  if FLAGS.start_stage <= 1 <= FLAGS.end_stage:
    print(
        f"\n[{time.strftime('%H:%M:%S')}] === STAGE 1: Incremental Overlaps & Coverage ===",
        flush=True,
    )
    t1 = time.time()
    old_attr_idx = set(
        pd.read_csv(
            os.path.join(backup_dir, 'attributes/caravan_attributes.csv'),
            index_col=0,
            usecols=[0],
        ).index
    )
    full_caravan_gdf = _read_gdf_fast(
        file_paths.CARAVAN_POLYGONS_FILE,
        index_col='unique_gauge_id',
        clean_all=False,
        ctx=ctx,
    )
    is_new = ~full_caravan_gdf.index.isin(old_attr_idx)
    old_caravan_gdf = full_caravan_gdf[~is_new]
    new_grdc_gdf = full_caravan_gdf[is_new].copy()
    new_grdc_gdf.geometry = _parallel_buffer0(
        new_grdc_gdf.geometry.to_numpy(), ctx
    )
    print(
        f'   Existing reference gauges: {len(old_caravan_gdf):,} | '
        f'New gauges added: {len(new_grdc_gdf):,} | Total: {len(full_caravan_gdf):,}',
        flush=True,
    )

    G_NEW_GRDC_GDF = new_grdc_gdf
    G_FULL_CARAVAN_AREAS = pd.Series(
        shapely.area(full_caravan_gdf.geometry.to_numpy()),
        index=full_caravan_gdf.index,
    )

    df_new_new = _fast_exact_overlaps(
        new_grdc_gdf, None, ctx=ctx, gdf1_precleaned=True
    )
    df_new_old = _fast_exact_overlaps(
        gdf1=new_grdc_gdf,
        gdf2=old_caravan_gdf,
        ctx=ctx,
        gdf1_precleaned=True,
        gdf2_precleaned=False,
    )
    df_old_new = df_new_old.rename(
        columns={'basin1': 'basin2', 'basin2': 'basin1'}
    )[['basin1', 'basin2', 'overlap_area']]

    old_caravan_tt = pd.read_csv(
        os.path.join(
            backup_dir, 'coverage_fractions/caravan_pairwise_overlap_areas.csv'
        )
    )
    caravan_tt_df = pd.concat(
        [old_caravan_tt, df_new_new, df_new_old, df_old_new], ignore_index=True
    )

    dst_tt = file_paths.CARAVAN_PAIRWISE_OVERLAPS_PATH
    tmp_tt = dst_tt + '.tmp_upd'
    caravan_tt_df.to_csv(tmp_tt, index=False)
    os.replace(tmp_tt, dst_tt)
    G_CARAVAN_TT_DF = caravan_tt_df
    del old_caravan_gdf, full_caravan_gdf

    continents = ['AF', 'AS', 'SA', 'SI', 'NA', 'EU', 'AR', 'AU', 'GR']
    summaries = []
    for c in continents:
      summaries.append(_process_continent_overlaps(c, backup_dir, ctx))

    df_sum = pd.DataFrame(summaries).sort_values('continent')
    print('\n--- CONTINENT ZERO-COVERAGE SUMMARY ---', flush=True)
    print(df_sum.to_string(index=False), flush=True)
    tot_w = int(df_sum['n_watersheds'].sum())
    old_z = int(df_sum['old_zero'].sum())
    new_z = int(df_sum['new_zero'].sum())
    print(
        f'GLOBAL ZERO-COVERAGE WATERSHEDS: {old_z:,}/{tot_w:,} ({100*old_z/tot_w:.2f}%) '
        f'-> {new_z:,}/{tot_w:,} ({100*new_z/tot_w:.2f}%) [Stage 1 took {time.time()-t1:.1f}s]',
        flush=True,
    )
    del G_NEW_GRDC_GDF, G_FULL_CARAVAN_AREAS, G_CARAVAN_TT_DF

  # ==========================================================================
  # STAGE 2: Global HyBAS Out-of-Distribution Anomaly Scores
  # ==========================================================================
  if FLAGS.start_stage <= 2 <= FLAGS.end_stage:
    print(
        f"\n[{time.strftime('%H:%M:%S')}] === STAGE 2: Global HyBAS Anomaly Scores ===",
        flush=True,
    )
    _safe_unlink(file_paths.INFER_HYBAS_CARAVAN_OUT_PATH)
    _, hybas_anom_in, caravan_anom_in = anom_mod.load_and_clean_data()
    anom_mod.calculate_hybas_inference_anomalies(
        caravan_anom_in, hybas_anom_in, True
    )
    del hybas_anom_in, caravan_anom_in

  # ==========================================================================
  # STAGE 3: Assemble Global HyBAS Generic Inference Dataset (CSV + Zarr)
  # ==========================================================================
  if FLAGS.start_stage <= 3 <= FLAGS.end_stage:
    print(
        f"\n[{time.strftime('%H:%M:%S')}] === STAGE 3: Global Inference Dataset Assembly ===",
        flush=True,
    )
    _safe_unlink(file_paths.GAUGED_GENERIC_DATA_PATH)
    _safe_unlink(file_paths.UNGAUGED_GENERIC_DATA_PATH)
    _safe_unlink(file_paths.HYBAS_GENERIC_DATA_PATH)
    _safe_unlink(file_paths.HYBAS_GENERIC_DATA_ZARR_PATH)
    datasets_mod.main(['3_create_datasets'])

  # ==========================================================================
  # STAGE 4: Global Skill-Predictor Inference Across All 38 Metrics x 2 Models
  # ==========================================================================
  if FLAGS.start_stage <= 4 <= FLAGS.end_stage:
    print(
        f"\n[{time.strftime('%H:%M:%S')}] === STAGE 4: Global Inference (38 Metrics x 2) ===",
        flush=True,
    )
    t4 = time.time()
    for p in glob.glob(os.path.join(file_paths.HYBAS_PREDS_DIR, '*.csv')):
      _safe_unlink(p)

    df_hybas = io_utils.load_hybas_zarr(file_paths.HYBAS_GENERIC_DATA_ZARR_PATH)
    x_train_g, _ = io_utils.load_and_preprocess_data('gauged', 'NSE')
    x_train_u, _ = io_utils.load_and_preprocess_data('ungauged', 'NSE')
    x_infer_gauged = df_hybas.reindex(columns=x_train_g.columns, fill_value=0)
    x_infer_ungauged = df_hybas.reindex(columns=x_train_u.columns, fill_value=0)

    tasks_infer = []
    for m in file_paths.ALL_METRICS:
      tasks_infer.append((m, 'ungauged', x_infer_ungauged))
      tasks_infer.append((m, 'gauged', x_infer_gauged))

    for done_cnt, (m, mtype, x_inf) in enumerate(tasks_infer, start=1):
      res = _run_infer_single(m, mtype, x_inf)
      if done_cnt % 12 == 0 or done_cnt == len(tasks_infer):
        print(f'   [{done_cnt}/{len(tasks_infer)}] {res}', flush=True)
    del x_infer_gauged, x_infer_ungauged
    print(
        f"[{time.strftime('%H:%M:%S')}] Stage 4 complete in {time.time() - t4:.1f}s",
        flush=True,
    )

  # ==========================================================================
  # STAGE 5: Precompile Multi-Metric Summary Statistics
  # ==========================================================================
  if FLAGS.start_stage <= 5 <= FLAGS.end_stage:
    print(
        f"\n[{time.strftime('%H:%M:%S')}] === STAGE 5: Precompile Multi-Metric Stats ===",
        flush=True,
    )
    _safe_unlink(file_paths.TABLE_1_PRECOMPUTED_STATS_PATH)
    for p in glob.glob(
        os.path.join(file_paths.STATS_DIR, 'skill_difference_by_country_*.csv')
    ):
      _safe_unlink(p)
    compile_t1_mod.main(['7_compile_table_1_stats'])

  with joblib.parallel_backend('threading'):
    # ========================================================================
    # STAGE 6: Unified 5-Factor Exposure Decomposition
    # ========================================================================
    if FLAGS.start_stage <= 6 <= FLAGS.end_stage:
      print(
          f"\n[{time.strftime('%H:%M:%S')}] === STAGE 6: Unified 5-Factor Decomposition ===",
          flush=True,
      )
      for fname in [
          'basin_exposure_and_blockers_v2.parquet',
          'basin_exposure_and_blockers_v2.pkl',
          'country_exposure_and_blockers_v2.csv',
          'five_factor_exposure_interventions_v2.json',
          'five_factor_exposure_shap_features_v2.csv',
          'five_factor_way_b_in_manifold_v2.csv',
      ]:
        _safe_unlink(os.path.join(file_paths.STATS_DIR, fname))
      five_factor_mod.main(['12_five_factor_exposure_decomposition'])

    # ========================================================================
    # STAGE 7: Targeted N-Gauge Network Expansion Optimization
    # ========================================================================
    if FLAGS.start_stage <= 7 <= FLAGS.end_stage:
      print(
          f"\n[{time.strftime('%H:%M:%S')}] === STAGE 7: Targeted N-Gauge Optimization ===",
          flush=True,
      )
      for sub_glob in [
          'candidates/*.csv',
          'n_gauge_eval/*',
          'scratch/debug_mali/*',
      ]:
        for p in glob.glob(os.path.join(file_paths.V2_RESULTS_DIR, sub_glob)):
          _safe_unlink(p)
      for fname in [
          'single_candidate_evaluations_v2.csv',
          'n_gauge_sets_evaluation_summary_v2.csv',
          'country_joint_n_gauge_exposure_reduction_v2.csv',
          'country_gauge_placement_exposure_reduction.csv',
      ]:
        _safe_unlink(os.path.join(file_paths.STATS_DIR, fname))
      evaluate_n_gauge_sets.main(['evaluate_n_gauge_sets'])

    # ========================================================================
    # STAGE 8: Stratified 1,000-Watershed Ensemble TreeSHAP Across 14 Metrics
    # ========================================================================
    if FLAGS.start_stage <= 8 <= FLAGS.end_stage:
      print(
          f"\n[{time.strftime('%H:%M:%S')}] === STAGE 8: Stratified 1,000-Watershed Ensemble TreeSHAP ===",
          flush=True,
      )
      t8 = time.time()
      if df_hybas is None:
        df_hybas = io_utils.load_hybas_zarr(
            file_paths.HYBAS_GENERIC_DATA_ZARR_PATH
        )

      scaler = StandardScaler()
      features_scaled = scaler.fit_transform(df_hybas.values)
      kmeans = MiniBatchKMeans(
          n_clusters=1000, batch_size=10000, random_state=42, n_init=3
      )
      kmeans.fit(features_scaled)
      labels = kmeans.labels_
      centers = kmeans.cluster_centers_
      selected_indices = []
      for i in range(1000):
        cluster_mask = labels == i
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_indices) == 0:
          continue
        dists = np.linalg.norm(
            features_scaled[cluster_mask] - centers[i], axis=1
        )
        selected_indices.append(cluster_indices[dists.argmin()])
      selected_indices = sorted(selected_indices)
      df_hybas_1000 = df_hybas.iloc[selected_indices].copy()
      with gfile.Open(file_paths.UNGAUGED_GENERIC_DATA_PATH, 'rt') as f:
        df_ungauged_generic = pd.read_csv(f, index_col=0)
      del df_hybas, features_scaled

      bundle_pkl = f'/tmp/stage8_shap_bundle_{os.getpid()}.pkl'
      with open(bundle_pkl, 'wb') as fp_b:
        pickle.dump((df_hybas_1000, df_ungauged_generic), fp_b)

      mp_ctx = multiprocessing.get_context('spawn')
      with concurrent.futures.ProcessPoolExecutor(
          max_workers=14, mp_context=mp_ctx
      ) as ex:
        futs = [
            ex.submit(_compute_ensemble_treeshap_from_bundle, (m, bundle_pkl))
            for m in SHAP_METRICS
        ]
        for fut in concurrent.futures.as_completed(futs):
          print(fut.result(), flush=True)
      _safe_unlink(bundle_pkl)

      file_paths.init(file_paths.default_metric)
      print(
          f"[{time.strftime('%H:%M:%S')}] Stage 8 complete in {time.time() - t8:.1f}s",
          flush=True,
      )

  bak_rows = (
      sum(
          1
          for _ in open(
              os.path.join(backup_dir, 'attributes/caravan_attributes.csv')
          )
      )
      - 1
  )
  assert bak_rows == 32713, f'Backup modified unexpectedly: {bak_rows}'
  print(
      f'\n=== FULL 38,069-GAUGE PIPELINE COMPLETED SUCCESSFULLY IN {time.time() - t_total:.1f}s! ===',
      flush=True,
  )


if __name__ == '__main__':
  app.run(main)
