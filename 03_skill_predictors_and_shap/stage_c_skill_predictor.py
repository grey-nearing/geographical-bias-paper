"""Clean Round-3 SkillPredictor classes (A3 cross-domain regularized architecture).

Upgrades from Round 2:
1. Removes non-extrapolating 15-NN and unregularized ExtraTrees(min_samples_leaf=3)
   that caused negative cross-continent (LOCO) R^2.
2. Uses a unified regularized 4-learner stack (HistGradientBoosting + LightGBM +
   XGBoost hist + WeightedRidgePipeline) with RidgeCV meta-learner and MAIN_BAS
   GroupKFold inner CV across BOTH continuous metrics and return-period metrics
   (preserving metric-family feature engineering).
3. Full `sample_weight` support across every base learner and meta-learner so
   Stage D Bayesian-bootstrap replicates use continuous Dirichlet block weights
   without duplicating rows.
4. `fit_with_fixed_stacker()` method for fast, unbiased B=200 block-bootstrap
   refits reusing Stage C's MAIN_BAS-validated RidgeCV stacking weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

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

# Natural/outlier-truncated bounds for each metric (matching Table S3 / file_paths.py).
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

# Active regularized preset (selected via A3 cross-domain CV benchmark).
DEFAULT_REG_PRESET = 'v2_strongreg'


def get_metric_bounds(metric: str) -> tuple[float, float]:
  if metric in METRIC_BOUNDS:
    return METRIC_BOUNDS[metric]
  return (0.0, 1.0)  # Return-period precision/recall/F1


def make_grouped_cv_splits(
    groups: pd.Index | np.ndarray, n_splits: int = 3, random_state: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
  """Creates deterministic GroupKFold splits on basin/station groups."""
  del random_state  # GroupKFold is deterministic and balances large basins cleanly.
  g_arr = np.asarray(groups)
  n_uniq = len(pd.Index(g_arr).unique())
  k = min(n_splits, max(2, n_uniq))
  gkf = GroupKFold(n_splits=k)
  dummy = np.zeros(len(g_arr), dtype=np.int8)
  return list(gkf.split(dummy, dummy, g_arr))


class WeightedRidgePipeline(RegressorMixin, BaseEstimator):
  """Median-imputing, RobustScaled RidgeCV regressor with full sample_weight support.

  Note: inherits (RegressorMixin, BaseEstimator) in that exact MRO order for
  scikit-learn >= 1.6 estimator-type tag detection inside StackingRegressor.
  """

  def __init__(self, alphas=None):
    self.alphas = alphas

  def fit(self, X, y, sample_weight=None):
    alphas = self.alphas if self.alphas is not None else np.logspace(0, 5, 15)
    self.imputer_ = SimpleImputer(strategy='median')
    self.scaler_ = RobustScaler()
    X_imp = self.imputer_.fit_transform(X)
    X_scl = self.scaler_.fit_transform(X_imp)
    self.ridge_ = RidgeCV(alphas=alphas)
    self.ridge_.fit(X_scl, y, sample_weight=sample_weight)
    return self

  def predict(self, X):
    X_imp = self.imputer_.transform(X)
    X_scl = self.scaler_.transform(X_imp)
    return self.ridge_.predict(X_scl)


def _make_regularized_estimators(
    n_threads: int = 2, preset: str = DEFAULT_REG_PRESET
):
  """Returns regularized (HGB, LGB, XGB, WeightedRidge) base learners for A3."""
  import xgboost  # pylint: disable=g-import-not-at-top

  if preset == 'v1':
    hgb = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.03,
        max_depth=8,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=5.0,
        early_stopping=False,
        random_state=42,
    )
    lgb = lightgbm.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=8,
        min_child_samples=30,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_alpha=2.0,
        reg_lambda=5.0,
        n_jobs=n_threads,
        random_state=42,
        verbose=-1,
    )
    xgb = xgboost.XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=25,
        subsample=0.75,
        colsample_bytree=0.5,
        reg_alpha=2.0,
        reg_lambda=5.0,
        tree_method='hist',
        n_jobs=n_threads,
        random_state=42,
    )
    rdg = WeightedRidgePipeline(alphas=np.logspace(-1, 4, 15))
  elif preset == 'v2_strongreg':
    hgb = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.025,
        max_depth=6,
        max_leaf_nodes=23,
        min_samples_leaf=40,
        l2_regularization=10.0,
        early_stopping=False,
        random_state=42,
    )
    lgb = lightgbm.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.025,
        num_leaves=23,
        max_depth=6,
        min_child_samples=50,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.45,
        reg_alpha=5.0,
        reg_lambda=10.0,
        n_jobs=n_threads,
        random_state=42,
        verbose=-1,
    )
    xgb = xgboost.XGBRegressor(
        n_estimators=500,
        learning_rate=0.025,
        max_depth=5,
        min_child_weight=40,
        subsample=0.75,
        colsample_bytree=0.45,
        reg_alpha=5.0,
        reg_lambda=10.0,
        tree_method='hist',
        n_jobs=n_threads,
        random_state=42,
    )
    rdg = WeightedRidgePipeline(alphas=np.logspace(0, 5, 15))
  else:
    raise ValueError(f'Unknown regularization preset: {preset}')

  return [('hgb', hgb), ('lgb', lgb), ('xgb', xgb), ('rdg', rdg)]


def _set_estimators_threads(estimators, n_threads: int) -> None:
  for est in estimators:
    if hasattr(est, 'set_params'):
      for pkey in ('n_jobs', 'nthread'):
        if pkey in est.get_params():
          try:
            est.set_params(**{pkey: n_threads})
          except Exception:
            pass
    if hasattr(est, 'get_booster'):
      try:
        est.get_booster().set_param({'nthread': n_threads})
      except Exception:
        pass


class _BaseRegularizedStackPredictor:
  """Shared fit / fixed-stacker / threaded-predict implementation."""

  def __init__(
      self,
      metric: str,
      is_delta: bool = False,
      n_threads: int = 2,
      preset: str = DEFAULT_REG_PRESET,
      inner_splits: int = 3,
  ):
    self.metric = metric
    self.is_delta = is_delta
    self.n_threads = n_threads
    self.preset = preset
    self.inner_splits = inner_splits
    lo, hi = get_metric_bounds(metric)
    span = hi - lo
    self.clip_bounds = (-span, span) if is_delta else (lo, hi)
    self._stacker = None
    self._fixed_estimators_ = None
    self._fixed_final_estimator_ = None

  def _fit_stacker(
      self,
      X_trans: pd.DataFrame,
      y: pd.Series | np.ndarray,
      groups: pd.Index | np.ndarray | None = None,
      sample_weight: np.ndarray | None = None,
  ):
    y_fit = np.clip(np.asarray(y, dtype=np.float64), *self.clip_bounds)
    if groups is None:
      groups = X_trans.index
    inner_cv = make_grouped_cv_splits(groups, n_splits=self.inner_splits, random_state=42)

    self._stacker = StackingRegressor(
        estimators=_make_regularized_estimators(
            n_threads=self.n_threads, preset=self.preset
        ),
        final_estimator=RidgeCV(alphas=np.logspace(-1, 3, 15)),
        cv=inner_cv,
        n_jobs=1,
    )
    if sample_weight is not None:
      self._stacker.fit(X_trans, y_fit, sample_weight=sample_weight)
    else:
      self._stacker.fit(X_trans, y_fit)
    self._fixed_estimators_ = None
    self._fixed_final_estimator_ = None
    return self._stacker

  def _fit_fixed_stacker(
      self,
      X_trans: pd.DataFrame,
      y: pd.Series | np.ndarray,
      sample_weight: np.ndarray | None,
      final_estimator,
  ):
    """Fits all 4 base learners once with sample_weight, reusing Stage C's RidgeCV stacker."""
    y_fit = np.clip(np.asarray(y, dtype=np.float64), *self.clip_bounds)
    fitted = []
    for _, est in _make_regularized_estimators(
        n_threads=self.n_threads, preset=self.preset
    ):
      if sample_weight is not None:
        est.fit(X_trans, y_fit, sample_weight=sample_weight)
      else:
        est.fit(X_trans, y_fit)
      fitted.append(est)
    self._stacker = None
    self._fixed_estimators_ = fitted
    self._fixed_final_estimator_ = final_estimator
    return self

  def _predict_aligned(self, X_aligned: pd.DataFrame) -> np.ndarray:
    if self._fixed_estimators_ is not None:
      # Fast zero-drift inference path for B=200 fixed-stacker bootstrap replicates:
      # float32 DataFrame + contiguous ndarray fits inside L3 cache (~2.4x faster HGB, 0.00e+00 maxdiff).
      if all(dt == np.float32 for dt in X_aligned.dtypes):
        X_f32_df = X_aligned
      else:
        X_f32_df = X_aligned.astype(np.float32)
      X_f32 = np.ascontiguousarray(X_f32_df.values, dtype=np.float32)
      cols = []
      for est in self._fixed_estimators_:
        cls_name = type(est).__name__
        if cls_name == 'HistGradientBoostingRegressor':
          cols.append(est.predict(X_f32_df))
        elif cls_name == 'LGBMRegressor':
          cols.append(
              est.booster_.predict(
                  X_f32, num_threads=self.n_threads, raw_score=True
              )
          )
        elif cls_name == 'XGBRegressor':
          cols.append(est.get_booster().inplace_predict(X_f32))
        else:
          cols.append(est.predict(X_f32))
      meta_X = np.column_stack(cols)
      preds = self._fixed_final_estimator_.predict(meta_X)
    else:
      preds = self._stacker.predict(X_aligned)
    return np.clip(preds, *self.clip_bounds)

  def set_n_threads(self, n_threads: int) -> None:
    self.n_threads = int(n_threads)
    if self._stacker is not None and hasattr(self._stacker, 'estimators_'):
      _set_estimators_threads(self._stacker.estimators_, self.n_threads)
    if self._fixed_estimators_ is not None:
      _set_estimators_threads(self._fixed_estimators_, self.n_threads)

  @property
  def final_estimator_(self):
    if self._fixed_final_estimator_ is not None:
      return self._fixed_final_estimator_
    if self._stacker is not None:
      return self._stacker.final_estimator_
    return None


class SkillPredictorClassicR2(_BaseRegularizedStackPredictor):
  """Stacked HGB+LGB+XGB+WeightedRidge predictor for continuous metrics & paired deltas."""

  def __init__(
      self,
      metric: str,
      is_delta: bool = False,
      n_threads: int = 2,
      preset: str = DEFAULT_REG_PRESET,
      inner_splits: int = 3,
  ):
    super().__init__(
        metric=metric,
        is_delta=is_delta,
        n_threads=n_threads,
        preset=preset,
        inner_splits=inner_splits,
    )
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
      sample_weight: np.ndarray | None = None,
  ):
    X_trans = self._engineer_features(X)
    self.feature_names = X_trans.columns.tolist()
    self.model = self._fit_stacker(
        X_trans, y, groups=groups, sample_weight=sample_weight
    )
    return self

  def fit_with_fixed_stacker(
      self,
      X: pd.DataFrame,
      y: pd.Series,
      sample_weight: np.ndarray | None,
      final_estimator,
  ):
    X_trans = self._engineer_features(X)
    self.feature_names = X_trans.columns.tolist()
    self._fit_fixed_stacker(X_trans, y, sample_weight=sample_weight, final_estimator=final_estimator)
    self.model = None
    return self

  def predict_from_engineered(self, X_aligned: pd.DataFrame) -> np.ndarray:
    return self._predict_aligned(X_aligned)

  def predict(self, X: pd.DataFrame) -> pd.Series:
    X_trans = self._engineer_features(X)
    X_aligned = X_trans.reindex(columns=self.feature_names, fill_value=0)
    preds = self.predict_from_engineered(X_aligned)
    return pd.Series(preds, index=X.index, name='skill')


class SkillPredictorReturnPeriodR2(_BaseRegularizedStackPredictor):
  """Stacked HGB+LGB+XGB+WeightedRidge predictor for return-period metrics & paired deltas."""

  def __init__(
      self,
      metric: str,
      is_delta: bool = False,
      n_threads: int = 2,
      preset: str = DEFAULT_REG_PRESET,
      inner_splits: int = 3,
  ):
    super().__init__(
        metric=metric,
        is_delta=is_delta,
        n_threads=n_threads,
        preset=preset,
        inner_splits=inner_splits,
    )
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
      sample_weight: np.ndarray | None = None,
  ):
    X_trans = self._preprocess_features(X)
    self.feature_columns = X_trans.columns.tolist()
    self.ensemble = self._fit_stacker(
        X_trans, y, groups=groups, sample_weight=sample_weight
    )
    return self

  def fit_with_fixed_stacker(
      self,
      X: pd.DataFrame,
      y: pd.Series,
      sample_weight: np.ndarray | None,
      final_estimator,
  ):
    X_trans = self._preprocess_features(X)
    self.feature_columns = X_trans.columns.tolist()
    self._fit_fixed_stacker(X_trans, y, sample_weight=sample_weight, final_estimator=final_estimator)
    self.ensemble = None
    return self

  def predict_from_preprocessed(self, X_aligned: pd.DataFrame) -> np.ndarray:
    return self._predict_aligned(X_aligned)

  def predict(self, X: pd.DataFrame) -> pd.Series:
    X_trans = self._preprocess_features(X)
    X_aligned = X_trans.reindex(columns=self.feature_columns, fill_value=0)
    preds = self.predict_from_preprocessed(X_aligned)
    return pd.Series(preds, index=X.index, name='predicted_skill')


class SkillPredictorR2:
  """Unified predictor wrapper for continuous/RP and base/delta targets."""

  def __init__(
      self,
      metric: str,
      is_delta: bool = False,
      n_threads: int = 2,
      preset: str = DEFAULT_REG_PRESET,
      inner_splits: int = 3,
  ):
    self.metric = metric
    self.is_delta = is_delta
    if metric in CLASSIC_METRICS:
      self.impl = SkillPredictorClassicR2(
          metric,
          is_delta=is_delta,
          n_threads=n_threads,
          preset=preset,
          inner_splits=inner_splits,
      )
    else:
      self.impl = SkillPredictorReturnPeriodR2(
          metric,
          is_delta=is_delta,
          n_threads=n_threads,
          preset=preset,
          inner_splits=inner_splits,
      )

  def fit(self, X: pd.DataFrame, y: pd.Series, groups=None, sample_weight=None):
    self.impl.fit(X, y, groups=groups, sample_weight=sample_weight)
    return self

  def fit_with_fixed_stacker(self, X: pd.DataFrame, y: pd.Series, sample_weight, final_estimator):
    self.impl.fit_with_fixed_stacker(X, y, sample_weight=sample_weight, final_estimator=final_estimator)
    return self

  def set_n_threads(self, n_threads: int) -> None:
    self.impl.set_n_threads(n_threads)

  def predict(self, X: pd.DataFrame) -> pd.Series:
    return self.impl.predict(X)
