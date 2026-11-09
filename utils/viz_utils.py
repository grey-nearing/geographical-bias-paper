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

"""Visualization utility functions for the Geographical Bias project.

This module provides standardized helpers for saving figures, plotting global
maps of basin-level metrics, and generating cross-validation diagnostics.
"""

import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import gfile_compat as gfile

def save_figure(fig, figures_dir, filename=None):
  """Saves a matplotlib figure to a file using gfile.

  Args:
      fig (matplotlib.figure.Figure): The figure object to save.
      figures_dir (str): The directory where the figure should be saved (or
        filename if 2 args passed).
      filename (str, optional): The filename for the saved figure.
  """
  if filename is None:
    from utils import file_paths
    filename = figures_dir
    figures_dir = file_paths.FIGURES_DIR
  gfile.MakeDirs(figures_dir)
  filepath = os.path.join(figures_dir, filename)
  with gfile.Open(filepath, 'wb') as f:
    fig.savefig(f, dpi=300, bbox_inches='tight')
  print(f'   --> Saved plot to: {filepath}')

def plot_global_map_on_ax(
    geo_df: pd.DataFrame,
    data_series: pd.Series,
    title: str,
    basemap_gdf: gpd.GeoDataFrame,
    ax: plt.Axes,
    clip_val: float = None,
    cmap='coolwarm',
):
  """Plots basin-level values on a global map using latitude/longitude coordinates.

  Args:
      geo_df (pd.DataFrame): DataFrame containing coordinates ('latitude', 'longitude') for each basin.
      data_series (pd.Series): The data values to plot, indexed matching geo_df.
      title (str): The title of the plot.
      basemap_gdf (gpd.GeoDataFrame): Basemap geometry for continents/countries.
      ax (plt.Axes): The matplotlib axes object to draw on.
      clip_val (float, optional): If set, symmmetrically clips the colorbar between [-clip_val, clip_val].
      cmap (str): The color map to use.
  """
  basemap_gdf.plot(
      ax=ax, facecolor='whitesmoke', edgecolor='darkgray', linewidth=0.5
  )
  plot_df = geo_df.join(data_series.rename('val'), how='inner').dropna(
      subset=['val']
  )

  if clip_val:
    vmin, vmax = -clip_val, clip_val
  else:
    vmin, vmax = -plot_df['val'].abs().max(), plot_df['val'].abs().max()

  scatter = ax.scatter(
      plot_df['longitude'],
      plot_df['latitude'],
      c=plot_df['val'],
      cmap=cmap,
      vmin=vmin,
      vmax=vmax,
      s=15,
      alpha=0.8,
      edgecolor='k',
      linewidth=0.2,
  )

  cbar = plt.colorbar(
      scatter, ax=ax, orientation='horizontal', shrink=0.5, pad=0.05
  )
  cbar.set_label('Target Value', fontsize=12)
  ax.set_title(title, fontsize=16, fontweight='bold')
  ax.set_aspect('equal', adjustable='datalim')
  ax.axis('off')


def plot_cv_diagnostics(
    all_true: pd.Series,
    all_preds: pd.Series,
    title_prefix: str,
    geo_df: pd.DataFrame,
    continents_gdf: gpd.GeoDataFrame,
    metric_short_name: str,
    figures_dir: str,
    metric: str,
):
  """Generates and saves diagnostic plots for cross-validation."""
  r2 = r2_score(all_true, all_preds)
  mae = mean_absolute_error(all_true, all_preds)
  rmse = np.sqrt(mean_squared_error(all_true, all_preds))

  residuals = all_true - all_preds
  fig = plt.figure(figsize=(18, 14))
  gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

  # Plot A: Scatter True vs Predicted
  ax_scatter = fig.add_subplot(gs[0, 0])
  ax_scatter.scatter(
      all_true,
      all_preds,
      alpha=0.4,
      edgecolor='w',
      linewidth=0.3,
      s=30,
      c='teal',
      label='Predictions',
  )
  if len(all_true) > 0:
    min_val = min(all_true.min(), all_preds.min())
    max_val = max(all_true.max(), all_preds.max())
    ax_scatter.plot(
        [min_val, max_val],
        [min_val, max_val],
        'k--',
        lw=1.5,
        label='1:1 Line',
        alpha=0.7,
    )
    padding = (max_val - min_val) * 0.05
    ax_scatter.set_xlim([min_val - padding, max_val + padding])
    ax_scatter.set_ylim([min_val - padding, max_val + padding])
    ax_scatter.set_aspect('equal', adjustable='box')
    ax_scatter.text(
        0.05,
        0.95,
        f'R² = {r2:.3f}',
        transform=ax_scatter.transAxes,
        verticalalignment='top',
        fontsize=12,
        fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )
    ax_scatter.legend(loc='lower right', frameon=True)
  ax_scatter.set_xlabel(f'True Skill ({metric_short_name})')
  ax_scatter.set_ylabel(f'Predicted Skill ({metric_short_name})')
  ax_scatter.grid(True, linestyle=':', alpha=0.6)

  # Plot B: Residuals Distribution
  ax_hist = fig.add_subplot(gs[0, 1])
  sns.histplot(
      residuals.clip(-0.5, 0.5), bins=60, kde=True, ax=ax_hist, color='purple', alpha=0.6
  )
  ax_hist.axvline(0, color='k', linestyle='dashed', lw=2)
  ax_hist.set_title(
      'Error Distribution (Residuals)', fontweight='bold', fontsize=14
  )
  ax_hist.set_xlabel('Residual (True - Pred)')
  ax_hist.set_ylabel('Frequency')

  # Plot C: Global Map of Errors
  ax_map = fig.add_subplot(gs[1, :])
  plot_global_map_on_ax(
      geo_df,
      residuals,
      f'{title_prefix}: Spatial Distribution of Error',
      continents_gdf,
      ax=ax_map,
      clip_val=0.5,
  )

  plt.tight_layout(pad=3.0)
  safe_name = title_prefix.replace(' ', '_').lower()
  save_figure(
      fig,
      figures_dir,
      f'{safe_name}_cv_diagnostics_v2_{metric}.png',
  )
  plt.show()
