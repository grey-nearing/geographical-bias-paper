"""Clean Round-2 SkillPredictor classes with:
1. Gauge-grouped inner CV (`GroupKFold` / precomputed grouped folds) inside StackingRegressor.
2. Metric-aware target & prediction clipping (fixing the 1.0 upper-clip bug on Beta-KGE, Alpha-NSE, Beta-NSE).
3. Paired within-gauge adaptation lift (`is_delta=True`) support (unconstrained signed targets).
4. Pure 95-feature interface (no `record_length`).
"""

import os
import joblib
import lightgbm
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

CLASSIC_METRICS = [
    'NSE',
    'KGE',
    'log-NSE',
    'log-KGE',
    'Beta-NSE',
    'Beta-KGE',
    'Alpha-NSE',
    'pearson-r',
]

# Natural/outlier-truncated bounds for each metric (matching Table S1 / file_paths.py).
METRIC_BOUNDS = {
    'NSE': (-0.5, 1.0),
    'KGE': (-0.5, 1.0),
    'log-NSE': (-0.5, 1.0),
    'log-KGE': (-0.5, 1.0),
    'pearson-r': (-0.5, 1.0),
    'Beta-NSE': (-3.0, 3.0),
    'Beta-KGE': (0.0, 3.0),
    'Alpha-NSE': (0.0, 3.0),
}


def get_metric_bounds(metric: str) -> tuple[float, float]:
  if metric in METRIC_BOUNDS:
    return METRIC_BOUNDS[metric]
  return (0.0, 1.0)  # Return-period precision/recall


def make_grouped_cv_splits(
    groups: pd.Index | np.ndarray, n_splits: int = 5, random_state: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
  """Creates deterministic shuffled gauge-grouped CV fold index pairs."""
  g_arr = np.asarray(groups)
  uniq = pd.Index(g_arr).unique()
  kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
  splits = []
  for _, va_u in kf.split(uniq):
    va_set = set(uniq[va_u])
    is_va = np.fromiter((g in va_set for g in g_arr), dtype=bool, count=len(g_arr))
    splits.append((np.where(~is_va)[0], np.where(is_va)[0]))
  return splits


def _make_classic_estimators(n_threads: int = 4):
  import xgboost  # pylint: disable=g-import-not-at-top

  xgb = xgboost.XGBRegressor(
      n_estimators=2000,
      learning_rate=0.015,
      max_depth=8,
      subsample=0.7,
      colsample_bytree=0.5,
      reg_alpha=1.0,
      reg_lambda=2.0,
      n_jobs=n_threads,
      random_state=42,
      tree_method='hist',
  )
  lgb = lightgbm.LGBMRegressor(
      n_estimators=2000,
      learning_rate=0.015,
      num_leaves=127,
      subsample=0.7,
      colsample_bytree=0.5,
      reg_alpha=1.0,
      reg_lambda=2.0,
      n_jobs=n_threads,
      random_state=42,
      verbose=-1,
  )
  hgb = HistGradientBoostingRegressor(
      max_iter=1000,
      max_depth=10,
      l2_regularization=1.5,
      random_state=42,
  )
  return [('xgb', xgb), ('lgb', lgb), ('hgb', hgb)]


def _make_rp_estimators(n_threads: int = 4):
  hgbr = HistGradientBoostingRegressor(
      loss='squared_error',
      learning_rate=0.04,
      max_iter=800,
      max_leaf_nodes=63,
      max_depth=15,
      min_samples_leaf=15,
      l2_regularization=1.0,
      random_state=42,
  )
  et = make_pipeline(
      SimpleImputer(strategy='median'),
      ExtraTreesRegressor(
          n_estimators=600,
          max_features=0.5,
          min_samples_leaf=3,
          n_jobs=n_threads,
          random_state=42,
      ),
  )
  knn = make_pipeline(
      SimpleImputer(strategy='median'),
      StandardScaler(),
      KNeighborsRegressor(n_neighbors=15, weights='distance', n_jobs=1),
  )
  return [('hgbr', hgbr), ('et', et), ('knn', knn)]


def _fast_predict_knn_torch(knn_pipe, X_aligned: pd.DataFrame) -> np.ndarray:
  """Exact torch-accelerated distance-weighted KNN prediction."""
  X_knn = np.ascontiguousarray(knn_pipe[:-1].transform(X_aligned), dtype=np.float64)
  knn_reg = knn_pipe.named_steps['kneighborsregressor']
  fit_X = np.ascontiguousarray(knn_reg._fit_X, dtype=np.float64)
  fit_y = np.asarray(knn_reg._y, dtype=np.float64).ravel()
  k_n = int(knn_reg.n_neighbors)
  n_q = len(X_knn)
  try:
    import torch

    dists = np.empty((n_q, k_n), dtype=np.float64)
    nbr_idx = np.empty((n_q, k_n), dtype=np.int64)
    if n_q >= 50000:
      torch.set_num_threads(max(1, int(os.environ.get('OMP_NUM_THREADS', '4'))))
    t_ref = torch.from_numpy(np.ascontiguousarray(fit_X, dtype=np.float32))
    step_sz = 8192
    with torch.no_grad():
      for s in range(0, n_q, step_sz):
        e = min(n_q, s + step_sz)
        t_q = torch.from_numpy(np.ascontiguousarray(X_knn[s:e], dtype=np.float32))
        d_mat = torch.cdist(t_q, t_ref)
        vals, idx = torch.topk(d_mat, k=k_n, dim=1, largest=False, sorted=True)
        dists[s:e] = vals.numpy().astype(np.float64)
        nbr_idx[s:e] = idx.numpy()
  except Exception:
    dists, nbr_idx = knn_reg.kneighbors(X_knn)
  zero_mask = dists <= 1e-15
  any_zero = zero_mask.any(axis=1)
  with np.errstate(divide='ignore'):
    nbr_w = np.where(any_zero[:, None], zero_mask.astype(np.float64), 1.0 / dists)
  nbr_w /= nbr_w.sum(axis=1, keepdims=True)
  return np.einsum('ij,ij->i', fit_y[nbr_idx], nbr_w)


class SkillPredictorClassicR2:
  """Stacked XGB+LGB+HGB+RidgeCV predictor for continuous metrics & paired deltas."""

  def __init__(
      self,
      metric: str,
      is_delta: bool = False,
      n_threads: int = 4,
  ):
    self.metric = metric
    self.is_delta = is_delta
    self.n_threads = n_threads
    lo, hi = get_metric_bounds(metric)
    span = hi - lo
    self.clip_bounds = (-span, span) if is_delta else (lo, hi)
    self.model = None
    self.feature_names = None

  def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
    X_eng = X.copy()
    if 'record_length' in X_eng.columns:
      X_eng = X_eng.drop(columns=['record_length'])

    X_eng['p_minus_pet'] = X_eng['pre_mm_uyr'] - X_eng['pet_mm_uyr']
    X_eng['p_div_pet'] = X_eng['pre_mm_uyr'] / (X_eng['pet_mm_uyr'] + 1e-3)
    X_eng['et_deficit'] = X_eng['pet_mm_uyr'] - X_eng['aet_mm_uyr']
    X_eng['soil_porosity'] = (X_eng['snd_pc_uav'] + 1e-3) / (
        X_eng['cly_pc_uav'] + 1e-3
    )
    X_eng['twi_proxy'] = np.log1p(np.maximum(X_eng['calculated_drain_area'], 0.0)) / (
        np.tan(np.radians(X_eng['slp_dg_uav'] + 0.1))
    )
    forest_cols = [
        c for c in X_eng.columns if 'glc_pc_u0' in c and c[-1] in '12345'
    ]
    X_eng['total_forest_cover'] = X_eng[forest_cols].sum(axis=1)
    arid_cols = ['glc_pc_u16', 'glc_pc_u20', 'glc_pc_u21']
    X_eng['total_arid_cover'] = X_eng[arid_cols].sum(axis=1)
    X_eng['human_intensity'] = (
        np.log1p(np.maximum(X_eng['gdp_ud_usu'], 0.0))
        * np.log1p(np.maximum(X_eng['pop_ct_usu'], 0.0))
        * np.log1p(np.maximum(X_eng['rdd_mk_uav'], 0.0))
    )
    for feat in [
        'calculated_drain_area',
        'rev_mc_usu',
        'ria_ha_usu',
        'riv_tc_usu',
        'pop_ct_usu',
    ]:
      if feat in X_eng.columns:
        X_eng[f'log_{feat}'] = np.log1p(np.maximum(X_eng[feat], 0.0))
    X_eng['precip_temp_inter'] = X_eng['pre_mm_uyr'] * X_eng['tmp_dc_uyr']
    X_eng['forest_aridity_inter'] = (
        X_eng['total_forest_cover'] * X_eng['ari_ix_uav']
    )
    return X_eng

  def fit(
      self,
      X: pd.DataFrame,
      y: pd.Series,
      groups: pd.Index | np.ndarray | None = None,
  ):
    X_trans = self._engineer_features(X)
    self.feature_names = X_trans.columns.tolist()
    y_fit = np.clip(np.asarray(y, dtype=np.float64), *self.clip_bounds)

    if groups is None:
      groups = X.index
    inner_cv = make_grouped_cv_splits(groups, n_splits=5, random_state=42)

    self.model = StackingRegressor(
        estimators=_make_classic_estimators(n_threads=self.n_threads),
        final_estimator=make_pipeline(
            RobustScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))
        ),
        cv=inner_cv,
        n_jobs=1,
    )
    self.model.fit(X_trans, y_fit)
    return self

  def predict_from_engineered(self, X_aligned: pd.DataFrame) -> np.ndarray:
    preds = self.model.predict(X_aligned)
    return np.clip(preds, *self.clip_bounds)

  def predict(self, X: pd.DataFrame) -> pd.Series:
    X_trans = self._engineer_features(X)
    X_aligned = X_trans.reindex(columns=self.feature_names, fill_value=0)
    preds = self.predict_from_engineered(X_aligned)
    return pd.Series(preds, index=X.index, name='skill')


class SkillPredictorReturnPeriodR2:
  """Stacked HGBR+ET+KNN+RidgeCV predictor for return-period metrics & paired deltas."""

  def __init__(
      self,
      metric: str,
      is_delta: bool = False,
      n_threads: int = 4,
  ):
    self.metric = metric
    self.is_delta = is_delta
    self.n_threads = n_threads
    self.clip_bounds = (-1.0, 1.0) if is_delta else (0.0, 1.0)
    self.ensemble = None
    self.feature_columns = None

  def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
    X_df = X.copy()
    if 'record_length' in X_df.columns:
      X_df = X_df.drop(columns=['record_length'])
    skewed_cols = [
        'calculated_drain_area',
        'pop_ct_usu',
        'gdp_ud_usu',
        'riv_tc_usu',
        'ria_ha_usu',
        'rev_mc_usu',
        'lkv_mc_usu',
    ]
    for col in skewed_cols:
      if col in X_df.columns:
        X_df[col] = np.log1p(X_df[col].clip(lower=0))
    if 'pre_mm_uyr' in X_df.columns and 'pet_mm_uyr' in X_df.columns:
      X_df['water_balance'] = X_df['pre_mm_uyr'] - X_df['pet_mm_uyr']
      X_df['aridity_ratio'] = X_df['pre_mm_uyr'] / (X_df['pet_mm_uyr'] + 1e-6)
    forest_cols = [f'glc_pc_u0{i}' for i in range(1, 6)]
    if all(c in X_df.columns for c in forest_cols):
      X_df['total_forest_pc'] = X_df[forest_cols].sum(axis=1)
    if 'hft_ix_u09' in X_df.columns and 'calculated_drain_area' in X_df.columns:
      X_df['human_impact_density'] = X_df['hft_ix_u09'] / (
          X_df['calculated_drain_area'] + 1.0
      )
    return X_df

  def fit(
      self,
      X: pd.DataFrame,
      y: pd.Series,
      groups: pd.Index | np.ndarray | None = None,
  ):
    X_trans = self._preprocess_features(X)
    self.feature_columns = X_trans.columns.tolist()
    y_fit = np.clip(np.asarray(y, dtype=np.float64), *self.clip_bounds)

    if groups is None:
      groups = X.index
    inner_cv = make_grouped_cv_splits(groups, n_splits=5, random_state=42)

    self.ensemble = StackingRegressor(
        estimators=_make_rp_estimators(n_threads=self.n_threads),
        final_estimator=RidgeCV(alphas=np.logspace(-2, 2, 10)),
        cv=inner_cv,
        n_jobs=1,
    )
    # Run ExtraTrees with threading backend so joblib doesn't spawn inside workers
    with joblib.parallel_backend('threading', n_jobs=self.n_threads):
      self.ensemble.fit(X_trans, y_fit)
    return self

  def predict_from_preprocessed(self, X_aligned: pd.DataFrame) -> np.ndarray:
    """Fast batch prediction bypassing slow sklearn multi-thread KNN."""
    hgbr = self.ensemble.estimators_[0]
    et = self.ensemble.estimators_[1]
    knn = self.ensemble.estimators_[2]
    p_hgbr = hgbr.predict(X_aligned)
    with joblib.parallel_backend('threading', n_jobs=self.n_threads):
      p_et = et.predict(X_aligned)
    p_knn = _fast_predict_knn_torch(knn, X_aligned)
    meta_X = np.column_stack([p_hgbr, p_et, p_knn])
    preds = self.ensemble.final_estimator_.predict(meta_X)
    return np.clip(preds, *self.clip_bounds)

  def predict(self, X: pd.DataFrame) -> pd.Series:
    X_trans = self._preprocess_features(X)
    X_aligned = X_trans.reindex(columns=self.feature_columns, fill_value=0)
    preds = self.predict_from_preprocessed(X_aligned)
    return pd.Series(preds, index=X.index, name='predicted_skill')


class SkillPredictorR2:
  """Unified Round-2 predictor wrapper for continuous/RP and base/delta targets."""

  def __init__(self, metric: str, is_delta: bool = False, n_threads: int = 4):
    self.metric = metric
    self.is_delta = is_delta
    if metric in CLASSIC_METRICS:
      self.impl = SkillPredictorClassicR2(metric, is_delta=is_delta, n_threads=n_threads)
    else:
      self.impl = SkillPredictorReturnPeriodR2(metric, is_delta=is_delta, n_threads=n_threads)

  def fit(self, X: pd.DataFrame, y: pd.Series, groups=None):
    self.impl.fit(X, y, groups=groups)
    return self

  def predict(self, X: pd.DataFrame) -> pd.Series:
    return self.impl.predict(X)
