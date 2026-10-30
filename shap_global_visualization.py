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


# %% [markdown]
# # Global SHAP Feature & Category Importance Analysis Across Metrics
#
# This script performs the post-processing and visualization of the sharded SHAP
# attribution scores calculated for the 1000 representative (stratified) catchments
# across all 12 hydrological metrics (NSE, KGE, and Precision/Recall sweeps).
#
# The analysis is organized into the following sequential steps:
# 1. **Data Ingestion**: Loads the consolidated exact KernelSHAP scores from disk.
# 2. **Category Aggregation**: Maps individual physical features to standardized
#    scientific categories (Climate, Soil, Land Cover, Topography, Anthropogenic).
# 3. **Global Heatmaps**:
#    - *Category Level*: Broad comparison of what attribute classes drive models.
#    - *Feature Level*: Granular heatmap of the top 20 most important catchments attributes.
# 4. **Grouped Visualizations (Paper Figures)**:
#    - *Faceted Bar Chart*: Detailed top 10 features for each metric.
#    - *Unified Comparison*: A grouped bar chart displaying category rankings side-by-side.

import os
import sys
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utils import gfile_compat as gfile

from utils import file_paths
from utils.feature_utils import FEATURE_DESCRIPTIONS, FEATURE_CLASSIFICATION, FEATURE_PALETTE


def prepare_feature_map(feature_names: list[str]) -> dict[str, str]:
  """Maps individual feature names to their corresponding categories."""
  full_feature_map = {}
  for feat in feature_names:
    if feat in FEATURE_CLASSIFICATION:
      full_feature_map[feat] = FEATURE_CLASSIFICATION[feat]
    elif (
        'p_minus_pet' in feat
        or 'p_div_pet' in feat
        or 'et_deficit' in feat
        or 'precip' in feat
    ):
      full_feature_map[feat] = 'Climate & Hydrology'
    elif 'soil_porosity' in feat:
      full_feature_map[feat] = 'Soil and Geology'
    elif 'twi_proxy' in feat or 'log_calculated_drain_area' in feat:
      full_feature_map[feat] = 'Topography'
    elif (
        'total_forest_cover' in feat
        or 'total_arid_cover' in feat
        or 'forest_aridity' in feat
    ):
      full_feature_map[feat] = 'Land Cover'
    elif (
        'human_intensity' in feat
        or 'log_pop' in feat
        or 'log_rev' in feat
        or 'log_ria' in feat
        or 'log_riv' in feat
    ):
      full_feature_map[feat] = 'Anthropogenic Factors'
    else:
      matched = False
      for base_feat in FEATURE_CLASSIFICATION:
        if base_feat in feat:
          full_feature_map[feat] = FEATURE_CLASSIFICATION[base_feat]
          matched = True
          break
      if not matched:
        full_feature_map[feat] = 'Other'
  return full_feature_map

metrics = [
    'NSE', 'KGE',
    '1.01-year-2.0-window-precision', '2.0-year-2.0-window-precision', '5.0-year-2.0-window-precision', '10.0-year-2.0-window-precision', '20.0-year-2.0-window-precision',
    '1.01-year-2.0-window-recall', '2.0-year-2.0-window-recall', '5.0-year-2.0-window-recall', '10.0-year-2.0-window-recall', '20.0-year-2.0-window-recall'
]

# Dictionaries to store importance scores
feature_importances = {}
category_importances = {}

for metric in metrics:
  # Initialize file_paths for this metric
  file_paths.init(metric)

  csv_path = None
  # Check stratified directories (prefer 10k, fallback to 1k)
  for folder in ['stratified_10000', 'stratified_1000']:
    p = os.path.join(file_paths.SHAP_BASE_DIR, folder, f'consolidated_exact_kernelshap_scores_{metric}.csv')
    if gfile.Exists(p):
      csv_path = p
      break
  if csv_path is None:
    # Fallback to direct directory check
    p = os.path.join(file_paths.SHAP_BASE_DIR, f'consolidated_exact_kernelshap_scores_{metric}.csv')
    if gfile.Exists(p):
      csv_path = p

  if csv_path is None:
    print(f"Loading {metric}... -> File not found, skipping.")
    continue

  print(f"Loading {metric} from {csv_path}...")

  with gfile.Open(csv_path, 'rt') as f:
    df_shap = pd.read_csv(f, index_col=0)

  # Calculate mean absolute SHAP per feature
  mean_abs_shap = df_shap.abs().mean()
  feature_importances[metric] = mean_abs_shap

  # Aggregate by category
  feature_map = prepare_feature_map(df_shap.columns)
  category_shap = df_shap.T.groupby(feature_map).sum().T
  mean_abs_cat_shap = category_shap.abs().mean()
  category_importances[metric] = mean_abs_cat_shap

# Combine into DataFrames
df_features = pd.DataFrame(feature_importances).fillna(0)
df_categories = pd.DataFrame(category_importances).fillna(0)

print("Data loaded successfully!")


plt.figure(figsize=(12, 8))
# Sort categories by mean importance across all metrics
category_order = df_categories.mean(axis=1).sort_values(ascending=False).index
sns.heatmap(
    df_categories.loc[category_order],
    cmap='viridis',
    annot=True,
    fmt=".4f",
    linewidths=.5,
    cbar_kws={'label': 'Mean |SHAP Value|'}
)
plt.title('Global Feature Category Importance across Metrics (1000 Basins)', fontsize=16, fontweight='bold')
plt.xlabel('Metric', fontsize=12)
plt.ylabel('Category', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Select top 20 features based on max importance in any metric
top_features = df_features.max(axis=1).sort_values(ascending=False).head(20).index

plt.figure(figsize=(16, 12))
sns.heatmap(
    df_features.loc[top_features],
    cmap='viridis',
    annot=True,
    fmt=".4f",
    linewidths=.5,
    cbar_kws={'label': 'Mean |SHAP Value|'}
)
# Add category annotations to y-axis labels and use full feature descriptions
feature_map = prepare_feature_map(top_features)
y_labels = [f"{FEATURE_DESCRIPTIONS.get(feat, feat)} ({feature_map[feat]})" for feat in top_features]
plt.yticks(np.arange(len(top_features)) + 0.5, y_labels, rotation=0)

plt.title('Global Feature Importance Heatmap (Top 20 Features across Metrics)', fontsize=16, fontweight='bold')
plt.xlabel('Metric', fontsize=12)
plt.ylabel('Feature (Category)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



################################################################################
# # Grouped Bar Charts for Categorical Data (Paper Visualization)
# 
# Here we visualize the category importance using grouped bar charts. We offer two versions:
# 1. **Faceted Grouped Bar Chart (Recommended)**: Splits the metrics into subplots by type (Classic, Recall, Precision) to prevent visual clutter.
# 2. **Single Grouped Bar Chart**: Plots all 12 metrics side-by-side in one figure.
################################################################################


fig = plt.figure(figsize=(32, 22))
gs = gridspec.GridSpec(3, 5, hspace=0.5, wspace=1.0)

# 1. Top row: NSE and KGE (placed at columns 1 and 3 to center them)
classic_layout = {
    'NSE': (0, 1),
    'KGE': (0, 3)
}

# We define the metrics groups
recall_metrics = [
    '1.01-year-2.0-window-recall',
    '2.0-year-2.0-window-recall',
    '5.0-year-2.0-window-recall',
    '10.0-year-2.0-window-recall',
    '20.0-year-2.0-window-recall'
]

precision_metrics = [
    '1.01-year-2.0-window-precision',
    '2.0-year-2.0-window-precision',
    '5.0-year-2.0-window-precision',
    '10.0-year-2.0-window-precision',
    '20.0-year-2.0-window-precision'
]

# Helper to plot a single metric subplot
def plot_top_features_subplot(metric_name, ax_obj, title):
  """Plots a horizontal bar chart of the top 10 features for a single metric.

  Args:
      metric_name (str): The metric column to plot.
      ax_obj (plt.Axes): The matplotlib axes object.
      title (str): Subplot title.
  """
  if metric_name not in df_features.columns:
    ax_obj.text(0.5, 0.5, f"No data for {metric_name}", ha='center', va='center')
    return

  # Get top 10 features by absolute importance
  df_metric = df_features[metric_name].abs().sort_values(ascending=False).head(10)

  # Map categories for coloring
  feature_map = prepare_feature_map(df_metric.index)
  colors = [FEATURE_PALETTE.get(feature_map[feat], 'gray') for feat in df_metric.index]

  # Resolve full names for y-tick labels
  y_labels = [FEATURE_DESCRIPTIONS.get(feat, feat).rstrip('.') for feat in df_metric.index]

  sns.barplot(
      x=df_metric.values,
      y=y_labels,
      palette=colors,
      hue=y_labels,
      legend=False,
      ax=ax_obj,
      edgecolor='black',
      linewidth=0.5
  )
  ax_obj.set_title(title, fontsize=12, fontweight='bold')
  ax_obj.set_xlabel('Mean |SHAP Value|', fontsize=9)
  ax_obj.set_ylabel('')
  ax_obj.tick_params(labelsize=8)
  sns.despine(ax=ax_obj)
  ax_obj.grid(axis='x', linestyle='--', alpha=0.4)

# --- Plot Classic Metrics (Row 0) ---
for metric, (r, c) in classic_layout.items():
  ax = fig.add_subplot(gs[r, c])
  plot_top_features_subplot(metric, ax, metric)

# Hide axes of empty slots in row 0
for c in [0, 2, 4]:
  ax_empty = fig.add_subplot(gs[0, c])
  ax_empty.axis('off')

# --- Plot Recall Metrics (Row 1) ---
for idx, metric in enumerate(recall_metrics):
  ax = fig.add_subplot(gs[1, idx])
  short_title = f"Recall ({metric.split('-')[0]}y)"
  plot_top_features_subplot(metric, ax, short_title)

# --- Plot Precision Metrics (Row 2) ---
for idx, metric in enumerate(precision_metrics):
  ax = fig.add_subplot(gs[2, idx])
  short_title = f"Precision ({metric.split('-')[0]}y)"
  plot_top_features_subplot(metric, ax, short_title)

plt.suptitle('Top 10 Most Important Features by Metric', fontsize=18, fontweight='bold', y=0.95)
plt.show()



# Define custom metric color palette grouping Classic (Blues), Precision (Oranges/Reds), and Recall (Greens)
metric_palette = {
    'NSE': '#0d47a1',
    'KGE': '#42a5f5',
    '1.01-year-2.0-window-precision': '#ffcc80',
    '2.0-year-2.0-window-precision': '#ffa726',
    '5.0-year-2.0-window-precision': '#fb8c00',
    '10.0-year-2.0-window-precision': '#e65100',
    '20.0-year-2.0-window-precision': '#b71c1c',
    '1.01-year-2.0-window-recall': '#c8e6c9',
    '2.0-year-2.0-window-recall': '#81c784',
    '5.0-year-2.0-window-recall': '#4caf50',
    '10.0-year-2.0-window-recall': '#2e7d32',
    '20.0-year-2.0-window-recall': '#1b5e20',
}

metric_order = [
    'NSE', 'KGE',
    '1.01-year-2.0-window-precision', '2.0-year-2.0-window-precision', '5.0-year-2.0-window-precision', '10.0-year-2.0-window-precision', '20.0-year-2.0-window-precision',
    '1.01-year-2.0-window-recall', '2.0-year-2.0-window-recall', '5.0-year-2.0-window-recall', '10.0-year-2.0-window-recall', '20.0-year-2.0-window-recall'
]

# Set up the figure with GridSpec (3 rows, 5 columns)
# Row 0: Comparison plot (cols 0-2), NSE (col 3), KGE (col 4)
# Row 1: Recall metrics (cols 0-4)
# Row 2: Precision metrics (cols 0-4)
# We use a wider figure (28 inches) and larger wspace/hspace to prevent y-axis labels overlap
fig = plt.figure(figsize=(28, 20))
gs = gridspec.GridSpec(3, 5, height_ratios=[1.2, 1, 1], hspace=0.4, wspace=0.6)

# --- Row 0: Grouped Comparison Plot (spans cols 0-2) ---
ax_compare = fig.add_subplot(gs[0, 0:3])

df_cat_long = df_categories.reset_index().rename(columns={'index': 'category'}).melt(
    id_vars='category',
    var_name='metric',
    value_name='importance'
)
category_order = df_categories.mean(axis=1).sort_values(ascending=False).index

sns.barplot(
    x='category',
    y='importance',
    hue='metric',
    data=df_cat_long,
    order=category_order,
    hue_order=metric_order,
    palette=metric_palette,
    ax=ax_compare,
    edgecolor='black',
    linewidth=0.5
)
ax_compare.set_title('Category Importance Comparison Across All Metrics', fontsize=14, fontweight='bold')
ax_compare.set_xlabel('Feature Category', fontsize=11)
ax_compare.set_ylabel('Mean Absolute SHAP Value', fontsize=11)
ax_compare.tick_params(axis='x', rotation=20, labelsize=10)
# Place legend inside the plot to avoid overlapping with NSE subplot
ax_compare.legend(title='Metric', loc='upper right', fontsize=8, ncol=2)
sns.despine(ax=ax_compare)
ax_compare.grid(axis='y', linestyle='--', alpha=0.5)

# --- Row 0: NSE and KGE (cols 3 and 4) ---
classic_metrics = ['NSE', 'KGE']
for idx, metric in enumerate(classic_metrics):
  ax = fig.add_subplot(gs[0, 3 + idx])
  if metric in df_categories.columns:
    df_metric = df_categories[metric].sort_values(ascending=False)
    colors = [FEATURE_PALETTE.get(cat, 'gray') for cat in df_metric.index]
    sns.barplot(
        x=df_metric.values,
        y=df_metric.index,
        palette=colors,
        hue=df_metric.index,
        legend=False,
        ax=ax,
        edgecolor='black',
        linewidth=0.5
    )
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_xlabel('Mean |SHAP|', fontsize=10)
    ax.set_ylabel('')
    ax.tick_params(labelsize=9)
    sns.despine(ax=ax)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

# --- Row 1: Recall Metrics (cols 0-4) ---
recall_metrics = [
    '1.01-year-2.0-window-recall',
    '2.0-year-2.0-window-recall',
    '5.0-year-2.0-window-recall',
    '10.0-year-2.0-window-recall',
    '20.0-year-2.0-window-recall'
]
for idx, metric in enumerate(recall_metrics):
  ax = fig.add_subplot(gs[1, idx])
  if metric in df_categories.columns:
    df_metric = df_categories[metric].sort_values(ascending=False)
    colors = [FEATURE_PALETTE.get(cat, 'gray') for cat in df_metric.index]
    sns.barplot(
        x=df_metric.values,
        y=df_metric.index,
        palette=colors,
        hue=df_metric.index,
        legend=False,
        ax=ax,
        edgecolor='black',
        linewidth=0.5
    )
    short_title = f"Recall ({metric.split('-')[0]}y)"
    ax.set_title(short_title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Mean |SHAP|', fontsize=10)
    ax.set_ylabel('')
    ax.tick_params(labelsize=9)
    sns.despine(ax=ax)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

# --- Row 2: Precision Metrics (cols 0-4) ---
precision_metrics = [
    '1.01-year-2.0-window-precision',
    '2.0-year-2.0-window-precision',
    '5.0-year-2.0-window-precision',
    '10.0-year-2.0-window-precision',
    '20.0-year-2.0-window-precision'
]
for idx, metric in enumerate(precision_metrics):
  ax = fig.add_subplot(gs[2, idx])
  if metric in df_categories.columns:
    df_metric = df_categories[metric].sort_values(ascending=False)
    colors = [FEATURE_PALETTE.get(cat, 'gray') for cat in df_metric.index]
    sns.barplot(
        x=df_metric.values,
        y=df_metric.index,
        palette=colors,
        hue=df_metric.index,
        legend=False,
        ax=ax,
        edgecolor='black',
        linewidth=0.5
    )
    short_title = f"Precision ({metric.split('-')[0]}y)"
    ax.set_title(short_title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Mean |SHAP|', fontsize=10)
    ax.set_ylabel('')
    ax.tick_params(labelsize=9)
    sns.despine(ax=ax)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

plt.show()



