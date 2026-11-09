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


"""Model definitions for predicting hydrological model skill.

This module defines the predictor architectures (classic vs return period)
and a unified wrapper to predict model performance across basins.
"""

import joblib
import lightgbm
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


class SkillPredictorClassic:
  """Model architecture optimized for classic continuous metrics (NSE, KGE, etc.)."""

  def __init__(self):
    """Initializes the classic metric ensemble model.

    Sets up a StackingRegressor combining XGBoost, LightGBM, and
    HistGradientBoosting regressors, using a RidgeCV meta-model.
    """
    import xgboost  # pylint: disable=g-import-not-at-top
    # 1. XGBoost: Deeper trees with high regularization
    xgb = xgboost.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.015,
        max_depth=8,
        subsample=0.7,
        colsample_bytree=0.5,
        reg_alpha=1.0,
        reg_lambda=2.0,
        n_jobs=-1,
        random_state=42,
        tree_method='hist',
    )

    # 2. LightGBM: Wide trees with leaf-wise growth
    lgb = lightgbm.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.015,
        num_leaves=127,
        subsample=0.7,
        colsample_bytree=0.5,
        reg_alpha=1.0,
        reg_lambda=2.0,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )

    # 3. HistGradientBoosting: Robust and handles potential non-linearities well
    hgb = HistGradientBoostingRegressor(
        max_iter=1000, max_depth=10, l2_regularization=1.5, random_state=42
    )

    # Stacking with a Ridge meta-model
    self.model = StackingRegressor(
        estimators=[('xgb', xgb), ('lgb', lgb), ('hgb', hgb)],
        final_estimator=make_pipeline(
            RobustScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))
        ),
        cv=5,
        n_jobs=-1,
    )
    self.feature_names = None

  def _engineer_features(self, X):
    """Engineers physical and socio-economic interaction features for classic metrics.

    Args:
        X (pd.DataFrame): Raw physical catchment attributes.

    Returns:
        pd.DataFrame: Transformed DataFrame with engineered features.
    """
    X_eng = X.copy()

    # --- Hydrological & Climatic Indicators ---
    X_eng['p_minus_pet'] = X_eng['pre_mm_uyr'] - X_eng['pet_mm_uyr']
    X_eng['p_div_pet'] = X_eng['pre_mm_uyr'] / (X_eng['pet_mm_uyr'] + 1e-3)
    X_eng['et_deficit'] = X_eng['pet_mm_uyr'] - X_eng['aet_mm_uyr']

    # --- Soil & Topographic interactions ---
    # Soil porosity proxy
    X_eng['soil_porosity'] = (X_eng['snd_pc_uav'] + 1e-3) / (
        X_eng['cly_pc_uav'] + 1e-3
    )
    # Topographic Wetness Index proxy
    X_eng['twi_proxy'] = np.log1p(X_eng['calculated_drain_area']) / (
        np.tan(np.radians(X_eng['slp_dg_uav'] + 0.1))
    )

    # --- Aggregated Land Cover / Biomes ---
    forest_cols = [
        c for c in X.columns if 'glc_pc_u0' in c and c[-1] in '12345'
    ]
    X_eng['total_forest_cover'] = X_eng[forest_cols].sum(axis=1)

    arid_cols = ['glc_pc_u16', 'glc_pc_u20', 'glc_pc_u21']
    X_eng['total_arid_cover'] = X_eng[arid_cols].sum(axis=1)

    # --- Human Footprint Metrics ---
    X_eng['human_intensity'] = (
        np.log1p(X_eng['gdp_ud_usu'])
        * np.log1p(X_eng['pop_ct_usu'])
        * np.log1p(X_eng['rdd_mk_uav'])
    )

    # --- Log Transformations ---
    log_feats = [
        'calculated_drain_area',
        'rev_mc_usu',
        'ria_ha_usu',
        'riv_tc_usu',
        'pop_ct_usu',
    ]
    for feat in log_feats:
      if feat in X_eng.columns:
        X_eng[f'log_{feat}'] = np.log1p(X_eng[feat])

    # --- Interaction Terms ---
    X_eng['precip_temp_inter'] = X_eng['pre_mm_uyr'] * X_eng['tmp_dc_uyr']
    X_eng['forest_aridity_inter'] = (
        X_eng['total_forest_cover'] * X_eng['ari_ix_uav']
    )

    return X_eng

  def fit(self, X, y):
    """Fits the classic skill predictor ensemble.

    Args:
        X (pd.DataFrame): Training features (catchment attributes).
        y (pd.Series): Target hydrological model skills.
    """
    X_transformed = self._engineer_features(X)
    self.feature_names = X_transformed.columns.tolist()
    
    # Clip targets to protect RidgeCV meta-learner from outlier skew
    y_clipped = np.clip(y, -2.0, 1.0)
    
    with joblib.parallel_backend('threading'):
      self.model.fit(X_transformed, y_clipped)

  def predict(self, X):
    """Predicts expected classic skill values for input catchments.

    Args:
        X (pd.DataFrame): Catchment attributes.

    Returns:
        pd.Series: Predicted skill scores, clipped to [-5.0, 1.0].
    """
    X_transformed = self._engineer_features(X)
    X_aligned = X_transformed.reindex(columns=self.feature_names, fill_value=0)
    
    with joblib.parallel_backend('threading'):
      preds = self.model.predict(X_aligned)
    return pd.Series(
        np.clip(preds, -5.0, 1.0), index=X.index, name='skill'
    )



class SkillPredictorReturnPeriod:
  """Model architecture optimized for return period scores (Precision, Recall)."""

  def __init__(self):
    """Initializes the return period metric ensemble model.

    Sets up a StackingRegressor combining HistGradientBoosting, ExtraTrees,
    and KNeighbors regressors, using a RidgeCV meta-model.
    """
    # Model 1: Gradient Boosting - natively handles NaNs and non-linear interactions
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

    # Model 2: Extra Trees - excellent for high-variance/noisy tabular data
    et = make_pipeline(
        SimpleImputer(strategy='median'),
        ExtraTreesRegressor(
            n_estimators=600,
            max_features=0.5,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42,
        ),
    )

    # Model 3: KNN - captures "physical similarity" (neighboring catchments in feature space)
    knn = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=15, weights='distance', n_jobs=-1),
    )

    # Stacking with RidgeCV as meta-learner
    self.ensemble = StackingRegressor(
        estimators=[
            ('hgbr', hgbr),
            ('et', et),
            ('knn', knn),
        ],
        final_estimator=RidgeCV(alphas=np.logspace(-2, 2, 10)),
        cv=5,
        n_jobs=-1,
    )

    self.feature_columns = None

  def _preprocess_features(self, X):
    """Engineers physical and socio-economic features for return period metrics.

    Log-transforms skewed physical attributes, aggregates land cover types,
    and computes aridity/water balance indicators.

    Args:
        X (pd.DataFrame): Raw physical catchment attributes.

    Returns:
        pd.DataFrame: Transformed DataFrame with engineered features.
    """
    X_df = X.copy()

    # 1. Log-transform highly skewed drainage and socio-economic variables
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

    # 2. Climate Indicators: Water Balance and Aridity
    if 'pre_mm_uyr' in X_df.columns and 'pet_mm_uyr' in X_df.columns:
      X_df['water_balance'] = X_df['pre_mm_uyr'] - X_df['pet_mm_uyr']
      X_df['aridity_ratio'] = X_df['pre_mm_uyr'] / (X_df['pet_mm_uyr'] + 1e-6)

    # 3. Aggregate Land Cover: Group specific forest/vegetation types
    # glc_pc_u01 to u05 are various forest types
    forest_cols = [f'glc_pc_u0{i}' for i in range(1, 6)]
    if all(c in X_df.columns for c in forest_cols):
      X_df['total_forest_pc'] = X_df[forest_cols].sum(axis=1)

    # 4. Human Impact Ratio
    if 'hft_ix_u09' in X_df.columns and 'calculated_drain_area' in X_df.columns:
      X_df['human_impact_density'] = X_df['hft_ix_u09'] / (
          X_df['calculated_drain_area'] + 1.0
      )

    return X_df

  def fit(self, X, y):
    """Fits the return period skill predictor ensemble.

    Args:
        X (pd.DataFrame): Training features (catchment attributes).
        y (pd.Series): Target return period skills (Precision/Recall).
    """
    X_transformed = self._preprocess_features(X)
    self.feature_columns = X_transformed.columns
    # Target y is recall [0, 1], tree-based models handle this range naturally
    with joblib.parallel_backend('threading'):
      self.ensemble.fit(X_transformed, y)

  def predict(self, X):
    """Predicts expected return period skill (Precision/Recall) for input catchments.

    Args:
        X (pd.DataFrame): Catchment attributes.

    Returns:
        pd.Series: Predicted skill scores, clipped to [0, 1].
    """
    X_transformed = self._preprocess_features(X)
    # Ensure column alignment
    X_aligned = X_transformed.reindex(
        columns=self.feature_columns, fill_value=0
    )

    with joblib.parallel_backend('threading'):
      preds = self.ensemble.predict(X_aligned)
    # Recall must be between 0 and 1
    preds = np.clip(preds, 0, 1)

    return pd.Series(preds, index=X.index, name='predicted_skill')


class SkillPredictor:
  """Unified wrapper that dynamically instantiates the correct architecture based on the active metric."""

  def __init__(self, metric: str = '5.0-year-2.0-window-recall'):
    """Initializes the SkillPredictor wrapper and chooses the correct backend.

    Args:
        metric (str): The name of the target hydrological performance metric.
    """
    self.metric = metric
    if self._is_classic_metric(metric):
      print(
          'Initializing SkillPredictor with Classic Architecture for metric:'
          f' {metric}'
      )
      self.impl = SkillPredictorClassic()
    else:
      print(
          'Initializing SkillPredictor with Return Period Architecture for'
          f' metric: {metric}'
      )
      self.impl = SkillPredictorReturnPeriod()

  def _is_classic_metric(self, metric: str) -> bool:
    """Checks if the metric belongs to the classic continuous category.

    Args:
        metric (str): The metric name to check.

    Returns:
        bool: True if the metric is classic (NSE, KGE, etc.), False if return period.
    """
    classic_list = [
        'NSE',
        'KGE',
        'log-NSE',
        'log-KGE',
        'Beta-NSE',
        'Beta-KGE',
        'Alpha-NSE',
        'pearson-r',
    ]
    return metric in classic_list

  def fit(self, X, y):
    """Fits the underlying model implementation on training data.

    Args:
        X (pd.DataFrame): Training features (attributes).
        y (pd.Series): Target performance skills.
    """
    return self.impl.fit(X, y)

  def predict(self, X):
    """Predicts performance skills using the underlying model implementation.

    Args:
        X (pd.DataFrame): Attributes for catchments to predict.

    Returns:
        pd.Series: Predicted skill scores.
    """
    return self.impl.predict(X)


import pickle


class _SkillPredictorUnpickler(pickle.Unpickler):
  """Unpickler that maps SkillPredictor class references to utils.skill_predictor."""

  def find_class(self, module, name):
    if name == 'SkillPredictor' or module.endswith('skill_predictor'):
      return SkillPredictor
    return super().find_class(module, name)


def load_skill_predictor(file_obj) -> SkillPredictor:
  """Loads a pickled SkillPredictor instance from a binary file object."""
  return _SkillPredictorUnpickler(file_obj).load()
