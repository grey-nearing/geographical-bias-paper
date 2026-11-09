# %%
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
# # Humanitarian Risk Reduction Analysis for the Geographical Bias project
#
# This script analyzes the potential reduction in humanitarian risk that can be
# achieved by improving flood forecast models (reducing the regionalization
# penalty).
#
# It compares two frameworks:
# 1. Population-Scaled Risk Reduction: Scaled by absolute population exposure.
# 2. Vulnerability-Scaled Risk Reduction: Focuses on socioeconomic vulnerability
#    (excluding population size to prevent larger nations from dominating).
#
# It integrates vulnerability-scaled risk reduction across return periods to
# compute the Expected Annual Vulnerability Reduction.

# %% [markdown]
# ## 1. Setup & Environment Configuration

# %%
import os
import time
import warnings
import geopandas as gpd
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from utils import gfile_compat as gfile

# Suppress warnings for clean output
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from utils import file_paths
from utils import io_utils

# Professional plotting standards
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'sans-serif'


def get_display_yr(yr):
  """Maps '1.01' to '1' and cleans other values for display."""
  return '1' if yr == '1.01' else file_paths._clean_metric_val(yr)


print(f"[{time.strftime('%H:%M:%S')}] Environment successfully configured.")

# %% [markdown]
# ## 2. Ingestion of Pipeline Datasets
#
# We load the INFORM dataset layers and Natural Earth boundary vectors.

# %%
load_start_time = time.time()

# 2.1 Ingest INFORM Humanitarian Index layers
df_vul = pd.read_csv(file_paths.VULN_FILE, header=1, index_col=0)
df_ind = pd.read_csv(file_paths.INDICATOR_FILE, header=1, index_col=0)
df_risk = pd.read_csv(file_paths.RISK_FILE, header=1, index_col=0)

# 2.2 Ingest Natural Earth boundary vectors
with gfile.Open(file_paths.NATURAL_EARTH_PATH, 'rb') as f:
  world_gdf = gpd.read_file(f)

print(
    f"[{time.strftime('%H:%M:%S')}] Core datasets ingested in"
    f' {time.time() - load_start_time:.2f}s.'
)

# %% [markdown]
# ## 3. Country-Level Boundary & Index Harmonization
#
# Reconcile country names across datasets to prevent systematic mismatches.

# %%
country_name_map = file_paths.country_name_map

# Reconcile geometry names
world_gdf['name'] = world_gdf['name'].replace(country_name_map)
world_gdf = world_gdf.dropna(subset=[world_gdf.geometry.name])
world_gdf = world_gdf[world_gdf.is_valid & ~world_gdf.is_empty]
world_bounds = world_gdf.total_bounds

# Reconcile INFORM index names
df_vul.index = df_vul.index.to_series().replace(country_name_map)
df_ind.index = df_ind.index.to_series().replace(country_name_map)
df_risk.index = df_risk.index.to_series().replace(country_name_map)

df_vul = df_vul.drop('(a-z)', errors='ignore')
df_ind = df_ind.drop('(a-z)', errors='ignore')
df_risk = df_risk.drop('(a-z)', errors='ignore')

# Extract and clean exposure (people), vulnerability (index), and coping capacity (lack of)
pop_exposure = df_ind['Annual Expected Exposed People to River Floods']
if pop_exposure.dtype == 'object':
  pop_exposure = pop_exposure.str.replace(',', '', regex=False)
pop_exposure = pd.to_numeric(pop_exposure, errors='coerce')

vuln_index = pd.to_numeric(df_risk['VULNERABILITY'], errors='coerce')

coping_capacity = df_risk['LACK OF COPING CAPACITY']
if coping_capacity.dtype == 'object':
  coping_capacity = coping_capacity.str.replace(',', '', regex=False)
coping_capacity = pd.to_numeric(coping_capacity, errors='coerce')

# Align Series
pop_exposure = pop_exposure.groupby(level=0).first()
vuln_index = vuln_index.groupby(level=0).first()
coping_capacity = coping_capacity.groupby(level=0).first()

inform_df = pd.DataFrame(index=df_risk.index.drop_duplicates(keep='first'))
inform_df.index.name = 'country_name'
inform_df['Exposed_Population'] = pop_exposure
inform_df['Vulnerability'] = vuln_index
inform_df['Coping_Capacity_Lack'] = coping_capacity

# Human Structural Vulnerability (Population-scaled vulnerability)
inform_df['Human_Structural_Vulnerability'] = inform_df[
    'Exposed_Population'
] * (inform_df['Vulnerability'] / 10.0)

print(f"[{time.strftime('%H:%M:%S')}] Country name harmonization completed.")
print(f'  - INFORM countries processed: {len(inform_df)}')

# %% [markdown]
# ## 4. Load Skill Gaps per Return Period (with Caching)
#
# We calculate the country-level median recall gaps. To optimize execution time, we load pre-computed country-level stats if they exist, otherwise we fallback to resolving the basin-level predictions.

# %%
years = ['2.0', '5.0', '10.0', '20.0']
window = '2.0'
raw_weights = {'2.0': 0.5, '5.0': 0.2, '10.0': 0.1, '20.0': 0.05}
sum_raw_weights = sum(raw_weights.values())
weights = {k: v / sum_raw_weights for k, v in raw_weights.items()}

# Load Hybas population attributes for weighting country-level aggregations
print('Loading Hybas population attributes for weighting country stats...')
with gfile.Open(file_paths.HYBAS_ATTRIBUTES_PATH, 'rt') as f:
  header = pd.read_csv(f, nrows=0)
  cols = [header.columns[0], 'HYDRO_ATLAS:pop_ct_ssu']
  f.seek(0)
  df_pop = pd.read_csv(f, usecols=cols)
  df_pop = df_pop.rename(
      columns={
          header.columns[0]: 'hybas_id',
          'HYDRO_ATLAS:pop_ct_ssu': 'population',
      }
  )
  # Convert from thousands of people to individuals
  df_pop['population'] = df_pop['population'] * 1000.0
  df_pop = df_pop.set_index('hybas_id')


def weighted_mean(values, weights):
  valid = ~np.isnan(values) & ~np.isnan(weights)
  w = weights[valid]
  v = values[valid]
  if w.sum() == 0:
    return np.nan
  return np.average(v, weights=w)


def weighted_quantile(values, weights, quantile):
  valid = ~np.isnan(values) & ~np.isnan(weights)
  w = weights[valid]
  v = values[valid]
  if w.sum() == 0:
    return np.nan
  sorter = np.argsort(v)
  v = v[sorter]
  w = w[sorter]
  cumulative_weight = np.cumsum(w) - 0.5 * w
  cumulative_weight = cumulative_weight / np.sum(w)
  return np.interp(quantile, cumulative_weight, v, left=v[0], right=v[-1])


dfs = {}

for yr in years:
  metric = f'{yr}-year-{window}-window-recall'
  print(f'Processing metric: {metric} ...')

  file_paths.init(metric)

  # Check for pre-computed country level skill stats
  recalculate = True
  if gfile.Exists(file_paths.SKILL_DIFFERENCE_BY_COUNTRY):
    try:
      country_median_gap = pd.read_csv(file_paths.SKILL_DIFFERENCE_BY_COUNTRY)
      if (
          'country_name' not in country_median_gap.columns
          and 'index' in country_median_gap.columns
      ):
        country_median_gap = country_median_gap.rename(
            columns={'index': 'country_name'}
        )

      # Ensure necessary columns are present in cached file
      if (
          'gauged_skill' in country_median_gap.columns
          and 'ungauged_skill' in country_median_gap.columns
      ):
        print(
            '  Loading pre-computed country median stats from'
            f' {file_paths.SKILL_DIFFERENCE_BY_COUNTRY}'
        )
        recalculate = False
      else:
        print(
            f'  Pre-computed file {file_paths.SKILL_DIFFERENCE_BY_COUNTRY} is'
            ' outdated (missing raw skill columns). Forcing recalculation...'
        )
    except Exception as e:
      print(f'  Error reading cached file, forcing recalculation: {e}')

  if recalculate:
    print(
        '  Resolving country stats from raw basin predictions (this will take'
        ' longer)...'
    )
    try:
      hybas_preds_gauged_df = pd.read_csv(
          file_paths.GLOBAL_HYBAS_PREDICTIONS_GAUGED, index_col=0
      )
      hybas_preds_ungauged_df = pd.read_csv(
          file_paths.GLOBAL_HYBAS_PREDICTIONS_UNGAUGED, index_col=0
      )
      mapping_df = pd.read_csv(file_paths.HYBAS_MAPPING_FILE)
      if 'Unnamed: 0' in mapping_df.columns:
        mapping_df = mapping_df.rename(columns={'Unnamed: 0': 'hybas_id'})
    except Exception as e:
      print(f'  Error loading files for {metric}: {e}')
      continue

    mapping_df['country_name'] = mapping_df['country_name'].replace(
        file_paths.country_name_map
    )
    preds_df = hybas_preds_gauged_df.join(hybas_preds_ungauged_df, how='inner')
    preds_df['skill_gap'] = (
        preds_df['gauged_skill'] - preds_df['ungauged_skill']
    )

    preds_with_country = preds_df.join(
        mapping_df.set_index('hybas_id'), how='inner'
    ).join(df_pop, how='inner')
    country_median_gap = (
        preds_with_country.groupby('country_name')
        .apply(
            lambda x: pd.Series({
                'gauged_skill': weighted_mean(
                    x['gauged_skill'], x['population']
                ),
                'ungauged_skill': weighted_mean(
                    x['ungauged_skill'], x['population']
                ),
                'skill_gap': weighted_mean(x['skill_gap'], x['population']),
                'skill_gap_q25': weighted_quantile(
                    x['skill_gap'], x['population'], 0.25
                ),
                'skill_gap_q75': weighted_quantile(
                    x['skill_gap'], x['population'], 0.75
                ),
            })
        )
        .reset_index()
    )

    file_paths.ensure_directories_exist()
    country_median_gap.to_csv(
        file_paths.SKILL_DIFFERENCE_BY_COUNTRY, index=False
    )
    print(
        '  Saved country median stats to'
        f' {file_paths.SKILL_DIFFERENCE_BY_COUNTRY}'
    )

  country_median_gap['country_name'] = country_median_gap[
      'country_name'
  ].replace(file_paths.country_name_map)

  df = inform_df.join(country_median_gap.set_index('country_name'), how='inner')

  df['Risk_Reduction'] = df['Human_Structural_Vulnerability'] * df['skill_gap']
  df['Risk_Reduction_No_Pop'] = (df['Vulnerability'] / 10.0) * df['skill_gap']

  dfs[yr] = df

# Restore default metric state
file_paths.init(file_paths.default_metric)
print(
    f"[{time.strftime('%H:%M:%S')}] Return period metrics successfully loaded."
)

# %% [markdown]
# ## 5. Compile Unified Metrics Dataset & Integrate Return Periods
#
# We align all countries across all return periods to construct a single, comprehensive dataset.
# We then integrate the metrics across return periods using exceedance probability weights (1/T):
# - **1-Year (1.01y)**: Weight 1.0
# - **2-Year**: Weight 0.5
# - **5-Year**: Weight 0.2
# - **10-Year**: Weight 0.1
# - **20-Year**: Weight 0.05

# %%
# Align countries across all return periods
common_countries = set(dfs[years[0]].index)
for yr in years[1:]:
  common_countries = common_countries.intersection(dfs[yr].index)
common_countries = list(common_countries)

metrics_df = inform_df.loc[common_countries].copy()

# Add individual return period gaps and reductions
for yr in years:
  metrics_df[f'skill_gap_{yr}'] = dfs[yr].loc[common_countries, 'skill_gap']
  metrics_df[f'gauged_skill_{yr}'] = dfs[yr].loc[
      common_countries, 'gauged_skill'
  ]
  metrics_df[f'ungauged_skill_{yr}'] = dfs[yr].loc[
      common_countries, 'ungauged_skill'
  ]
  metrics_df[f'Risk_Reduction_{yr}'] = dfs[yr].loc[
      common_countries, 'Risk_Reduction'
  ]
  metrics_df[f'Risk_Reduction_No_Pop_{yr}'] = dfs[yr].loc[
      common_countries, 'Risk_Reduction_No_Pop'
  ]

# Compute expected stats
metrics_df['skill_gap_expected'] = 0.0
metrics_df['gauged_skill_expected'] = 0.0
metrics_df['ungauged_skill_expected'] = 0.0
for yr in years:
  w = weights[yr]
  metrics_df['skill_gap_expected'] += w * metrics_df[f'skill_gap_{yr}'].fillna(0.0)
  metrics_df['gauged_skill_expected'] += w * metrics_df[f'gauged_skill_{yr}'].fillna(0.0)
  metrics_df['ungauged_skill_expected'] += w * metrics_df[f'ungauged_skill_{yr}'].fillna(0.0)

# Calculate Integrated Expected Risk Reduction (both pop-scaled and vulnerability-only)
metrics_df['Risk_Reduction_Integrated'] = (
    metrics_df['Human_Structural_Vulnerability']
    * metrics_df['skill_gap_expected']
)
metrics_df['Risk_Reduction_No_Pop_Integrated'] = (
    metrics_df['Vulnerability'] / 10.0
) * metrics_df['skill_gap_expected']

print(
    f"[{time.strftime('%H:%M:%S')}] Unified metrics compiled for"
    f' {len(metrics_df)} countries.'
)

# %% [markdown]
# ---
# # SECTION A: Population-Scaled Risk Reduction Plots
#
# This section produces figures accounting for absolute population exposure.
#
# **Scientific Significance**: Scaling risk reduction by the absolute population
# exposed to river floods highlights countries where model improvements will protect
# the largest absolute number of people. This framework favors densely populated
# regions (e.g., South and East Asia), where even modest relative model improvements
# can benefit millions of lives.
#
# **Key Formula**:
# $$\text{Risk Reduction} = \text{Exposed Population} \times \frac{\text{Vulnerability}}{10} \times \Delta\text{Skill}$$
# where $\Delta\text{Skill} = \text{Gauged Skill} - \text{Ungauged Skill}$.
# ---

# %% [markdown]
# ### Figure 5a: Global Maps of Population-Scaled Risk Reduction

# %% [markdown]
# **Data Preparation**

# %%
map_dfs_pop = {}
for yr in ['1.01', '2.0', '5.0', '10.0', '20.0', 'Integrated']:
  col = (
      f'Risk_Reduction_{yr}'
      if yr != 'Integrated'
      else 'Risk_Reduction_Integrated'
  )
  map_dfs_pop[yr] = world_gdf.merge(
      metrics_df[col].to_frame(), left_on='name', right_index=True, how='left'
  )

# %% [markdown]
# **Plotting**

# %%
fig_5a, axes = plt.subplots(3, 2, figsize=(20, 21))
axes = axes.flatten()

for i, yr in enumerate(['1.01', '2.0', '5.0', '10.0', '20.0', 'Integrated']):
  ax = axes[i]
  col = (
      f'Risk_Reduction_{yr}'
      if yr != 'Integrated'
      else 'Risk_Reduction_Integrated'
  )
  map_df = map_dfs_pop[yr]

  no_data_df = map_df[map_df[col].isna()]
  no_rr_df = map_df[map_df[col] <= 0]
  pos_rr_df = map_df[map_df[col] > 0]

  map_df.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.1)
  if not no_rr_df.empty:
    no_rr_df.plot(ax=ax, color='white', edgecolor='black', linewidth=0.1)

  if not pos_rr_df.empty:
    val_max = pos_rr_df[col].max()
    val_min = (
        max(1e-1, pos_rr_df[col].min())
        if not pd.isna(pos_rr_df[col].min())
        else 1e-1
    )
    if pd.isna(val_max) or val_max <= 0:
      val_max = 1.0
    if val_min >= val_max:
      val_min = val_max * 1e-4
    norm = mcolors.LogNorm(vmin=val_min, vmax=val_max)

    pos_rr_df.plot(
        ax=ax,
        column=col,
        cmap='YlOrRd',
        norm=norm,
        legend=True,
        legend_kwds={
            'shrink': 0.8,
            'label': 'Risk Reduction (People/year per alpha)',
            'orientation': 'horizontal',
            'pad': 0.05,
        },
        edgecolor='black',
        linewidth=0.1,
    )
  title = (
      'Integrated Expected Annual'
      if yr == 'Integrated'
      else f'{get_display_yr(yr)}-Year Return Period'
  )
  ax.set_title(title, fontsize=14, fontweight='bold')
  ax.set_xlim(world_bounds[0], world_bounds[2])
  ax.set_ylim(world_bounds[1], world_bounds[3])
  ax.axis('off')

fig_5a.suptitle(
    'Population-Scaled: Global Map of Risk Reduction Potential',
    fontsize=18,
    fontweight='bold',
    y=0.98,
)
plt.tight_layout()

# %% [markdown]
# **Save Figure**

# %%
plt.show()

# %% [markdown]
# **Figure 5a Caption**: Global maps showing population-scaled risk reduction potential across return periods and the integrated expected annual reduction. Gray represents missing data; white represents valid data but zero or negative potential; colors scale logarithmically by population size.

# %% [markdown]
# ### Figure 6a: Correlation Analysis (Population-Scaled)

# %% [markdown]
# **Data Preparation**

# %%
# No special preparation needed, using metrics_df directly

# %% [markdown]
# **Plotting**

# %%
fig_6a, axes = plt.subplots(3, 2, figsize=(16, 21), sharey=True)
axes = axes.flatten()

x_col = 'Human_Structural_Vulnerability'
x_label = (
    'Human Structural Vulnerability (Exposed Population x Vulnerability / 10)'
)

for i, yr in enumerate(['1.01', '2.0', '5.0', '10.0', '20.0', 'Integrated']):
  ax = axes[i]
  gap_col = f'skill_gap_{yr}' if yr != 'Integrated' else 'skill_gap_expected'

  corr_df = metrics_df.dropna(subset=[x_col, gap_col])
  corr_df = corr_df[corr_df[x_col] > 0]
  corr_df['log_x'] = np.log10(corr_df[x_col])

  slope, intercept, r_value, p_value, std_err = stats.linregress(
      corr_df['log_x'], corr_df[gap_col]
  )
  r2 = r_value**2

  sns.scatterplot(
      data=corr_df,
      x=x_col,
      y=gap_col,
      s=60,
      alpha=0.7,
      color='steelblue',
      ax=ax,
  )
  x_vals = np.logspace(
      np.log10(corr_df[x_col].min()), np.log10(corr_df[x_col].max()), 100
  )
  y_vals = slope * np.log10(x_vals) + intercept
  ax.plot(x_vals, y_vals, color='crimson', linewidth=1.8)
  ax.set_xscale('log')
  ax.set_box_aspect(1)

  stats_text = f'$R^2 = {r2:.4f}$ (log-X)\n$p = {p_value:.2e}$'
  title = (
      'Integrated Expected Annual'
      if yr == 'Integrated'
      else f'{get_display_yr(yr)}-Year Return Period'
  )
  ax.set_title(title, fontsize=12, fontweight='bold')
  if i >= 4:
    ax.set_xlabel(x_label, fontsize=10)
  else:
    ax.set_xlabel('')
  if i % 2 == 0:
    ax.set_ylabel('Recall Skill Gap (Gauged - Ungauged)', fontsize=10)
  else:
    ax.set_ylabel('')
  ax.grid(True, linestyle='--', alpha=0.5)
  ax.text(
      0.05,
      0.95,
      stats_text,
      transform=ax.transAxes,
      fontsize=10,
      verticalalignment='top',
      bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.55),
  )

fig_6a.suptitle(
    'Population-Scaled: Correlation vs. Recall Skill Gap',
    fontsize=16,
    fontweight='bold',
    y=0.98,
)
plt.tight_layout()

# %% [markdown]
# **Save Figure**

# %%
plt.show()

# %% [markdown]
# **Figure 6a Caption**: Scatter plots and log-linear regressions of country-level recall skill gap against Human Structural Vulnerability across return periods and for the integrated expected annual skill gap.

# %% [markdown]
# ### Figure 7a: Top Countries & Alpha Sensitivity (Population-Scaled)

# %% [markdown]
# **Data Preparation**

# %%
top_countries_pop = {}
for yr in ['1.01', '2.0', '5.0', '10.0', '20.0', 'Integrated']:
  col = (
      f'Risk_Reduction_{yr}'
      if yr != 'Integrated'
      else 'Risk_Reduction_Integrated'
  )
  top_10 = metrics_df.sort_values(by=col, ascending=False).head(10)
  top_countries_pop[yr] = top_10.sort_values(by=col, ascending=True)

# %% [markdown]
# **Plotting**

# %%
fig_7a, axes = plt.subplots(1, 6, figsize=(36, 8))
axes = axes.flatten()

for j, yr in enumerate(['1.01', '2.0', '5.0', '10.0', '20.0', 'Integrated']):
  ax_bar = axes[j]

  top_10 = top_countries_pop[yr]
  col = (
      f'Risk_Reduction_{yr}'
      if yr != 'Integrated'
      else 'Risk_Reduction_Integrated'
  )

  colors = plt.cm.viridis(np.linspace(0.3, 0.8, 10))
  ax_bar.barh(
      top_10.index, top_10[col], color=colors, edgecolor='black', height=0.6
  )

  title = (
      'Integrated Expected Annual'
      if yr == 'Integrated'
      else f'{get_display_yr(yr)}-Year Return Period'
  )
  ax_bar.set_title(title, fontsize=14, fontweight='bold')
  ax_bar.set_xlabel('Risk Reduction (People/year)', fontsize=11)
  ax_bar.grid(True, linestyle='--', alpha=0.5, axis='x')
  ax_bar.tick_params(axis='both', which='major', labelsize=10)

fig_7a.suptitle(
    'Population-Scaled: Top 10 Countries Risk Reduction Potential',
    fontsize=18,
    fontweight='bold',
    y=0.98,
)
plt.tight_layout()

# %% [markdown]
# **Save Figure**

# %%
plt.show()

# %% [markdown]
# **Figure 7a Caption**: Subplot grid showing the top 10 countries for population-scaled risk reduction potential across return periods and for the integrated expected annual reduction.

# %% [markdown]
# ---
# # SECTION B: Vulnerability-Scaled Risk Reduction Plots (No Pop)
#
# This section produces figures focused purely on socioeconomic vulnerability.
#
# **Scientific Significance**: To address equity, this framework scales risk
# reduction purely by the INFORM Vulnerability Index (from 0 to 10), omitting
# absolute population density. This prevents populous countries from dominating
# the metrics and instead highlights regions with the least capacity to adapt,
# prepare for, or recover from disasters (e.g., Sub-Saharan Africa).
#
# **Key Formula**:
# $$\text{Vulnerability-Scaled Risk Reduction} = \frac{\text{Vulnerability}}{10} \times \Delta\text{Skill}$$
# ---

# %% [markdown]
# ### Figure 5b: Global Maps of Vulnerability-Scaled Risk Reduction (No Pop)

# %% [markdown]
# **Data Preparation**

# %%
map_dfs_nopop = {}
for yr in ['1.01', '2.0', '5.0', '10.0', '20.0', 'Integrated']:
  col = (
      f'Risk_Reduction_No_Pop_{yr}'
      if yr != 'Integrated'
      else 'Risk_Reduction_No_Pop_Integrated'
  )
  map_dfs_nopop[yr] = world_gdf.merge(
      metrics_df[col].to_frame(), left_on='name', right_index=True, how='left'
  )

# %% [markdown]
# **Plotting**

# %%
fig_5b, axes = plt.subplots(3, 2, figsize=(20, 21))
axes = axes.flatten()

for i, yr in enumerate(['1.01', '2.0', '5.0', '10.0', '20.0', 'Integrated']):
  ax = axes[i]
  col = (
      f'Risk_Reduction_No_Pop_{yr}'
      if yr != 'Integrated'
      else 'Risk_Reduction_No_Pop_Integrated'
  )
  map_df = map_dfs_nopop[yr]

  no_data_df = map_df[map_df[col].isna()]
  no_rr_df = map_df[map_df[col] <= 0]
  pos_rr_df = map_df[map_df[col] > 0]

  map_df.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.1)
  if not no_rr_df.empty:
    no_rr_df.plot(ax=ax, color='white', edgecolor='black', linewidth=0.1)

  if not pos_rr_df.empty:
    val_max = pos_rr_df[col].max()
    if pd.isna(val_max) or val_max <= 0:
      val_max = 1.0
    norm = mcolors.Normalize(vmin=0, vmax=val_max)

    pos_rr_df.plot(
        ax=ax,
        column=col,
        cmap='YlOrRd',
        norm=norm,
        legend=True,
        legend_kwds={
            'shrink': 0.8,
            'label': 'Risk Reduction (Impact Score 0-1 per alpha)',
            'orientation': 'horizontal',
            'pad': 0.05,
        },
        edgecolor='black',
        linewidth=0.1,
    )
  title = (
      'Integrated Expected Annual'
      if yr == 'Integrated'
      else f'{get_display_yr(yr)}-Year Return Period'
  )
  ax.set_title(title, fontsize=14, fontweight='bold')
  ax.set_xlim(world_bounds[0], world_bounds[2])
  ax.set_ylim(world_bounds[1], world_bounds[3])
  ax.axis('off')

fig_5b.suptitle(
    'Vulnerability-Only (No Pop): Global Map of Risk Reduction Potential',
    fontsize=18,
    fontweight='bold',
    y=0.98,
)
plt.tight_layout()

# %% [markdown]
# **Save Figure**

# %%
plt.show()

# %% [markdown]
# **Figure 5b Caption**: Global maps showing vulnerability-scaled (no population term) risk reduction potential across return periods and the integrated expected annual vulnerability reduction. White represents valid data but <= 0 potential; gray represents missing data.

# %% [markdown]
# ### Figure 6b: Correlation Analysis (Vulnerability-Scaled, No Pop)

# %% [markdown]
# **Data Preparation**

# %%
# No special preparation needed, using metrics_df directly

# %% [markdown]
# **Plotting**

# %%
fig_6b, axes = plt.subplots(3, 2, figsize=(16, 21), sharey=True)
axes = axes.flatten()

x_col = 'Vulnerability'
x_label = 'Socioeconomic Vulnerability (INFORM)'

for i, yr in enumerate(['1.01', '2.0', '5.0', '10.0', '20.0', 'Integrated']):
  ax = axes[i]
  gap_col = f'skill_gap_{yr}' if yr != 'Integrated' else 'skill_gap_expected'

  corr_df = metrics_df.dropna(subset=[x_col, gap_col])

  slope, intercept, r_value, p_value, std_err = stats.linregress(
      corr_df[x_col], corr_df[gap_col]
  )
  r2 = r_value**2

  sns.scatterplot(
      data=corr_df,
      x=x_col,
      y=gap_col,
      s=60,
      alpha=0.7,
      color='steelblue',
      ax=ax,
  )
  x_vals = np.linspace(corr_df[x_col].min(), corr_df[x_col].max(), 100)
  y_vals = slope * x_vals + intercept
  ax.plot(x_vals, y_vals, color='crimson', linewidth=1.8)
  ax.set_box_aspect(1)

  stats_text = f'$R^2 = {r2:.4f}$\n$p = {p_value:.2e}$'
  title = (
      'Integrated Expected Annual'
      if yr == 'Integrated'
      else f'{get_display_yr(yr)}-Year Return Period'
  )
  ax.set_title(title, fontsize=12, fontweight='bold')
  if i >= 4:
    ax.set_xlabel(x_label, fontsize=10)
  else:
    ax.set_xlabel('')
  if i % 2 == 0:
    ax.set_ylabel('Recall Skill Gap (Gauged - Ungauged)', fontsize=10)
  else:
    ax.set_ylabel('')
  ax.grid(True, linestyle='--', alpha=0.5)
  ax.text(
      0.05,
      0.95,
      stats_text,
      transform=ax.transAxes,
      fontsize=10,
      verticalalignment='top',
      bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.55),
  )

fig_6b.suptitle(
    'Vulnerability-Only (No Pop): Correlation vs. Recall Skill Gap',
    fontsize=16,
    fontweight='bold',
    y=0.98,
)
plt.tight_layout()

# %% [markdown]
# **Save Figure**

# %%
plt.show()

# %% [markdown]
# **Figure 6b Caption**: Scatter plots and linear regressions of country-level recall skill gap against socioeconomic vulnerability index across return periods and for the integrated expected annual skill gap.

# %% [markdown]
# ### Figure 7b: Top Countries & Alpha Sensitivity (Vulnerability-Scaled, No Pop)

# %% [markdown]
# **Data Preparation**

# %%
top_countries_nopop = {}
for yr in ['1.01', '2.0', '5.0', '10.0', '20.0', 'Integrated']:
  col = (
      f'Risk_Reduction_No_Pop_{yr}'
      if yr != 'Integrated'
      else 'Risk_Reduction_No_Pop_Integrated'
  )
  top_10 = metrics_df.sort_values(by=col, ascending=False).head(10)
  top_countries_nopop[yr] = top_10.sort_values(by=col, ascending=True)

# %% [markdown]
# **Plotting**

# %%
fig_7b, axes = plt.subplots(1, 6, figsize=(36, 8))
axes = axes.flatten()

for j, yr in enumerate(['1.01', '2.0', '5.0', '10.0', '20.0', 'Integrated']):
  ax_bar = axes[j]

  top_10 = top_countries_nopop[yr]
  col = (
      f'Risk_Reduction_No_Pop_{yr}'
      if yr != 'Integrated'
      else 'Risk_Reduction_No_Pop_Integrated'
  )

  colors = plt.cm.viridis(np.linspace(0.3, 0.8, 10))
  ax_bar.barh(
      top_10.index, top_10[col], color=colors, edgecolor='black', height=0.6
  )

  title = (
      'Integrated Expected Annual'
      if yr == 'Integrated'
      else f'{get_display_yr(yr)}-Year Return Period'
  )
  ax_bar.set_title(title, fontsize=14, fontweight='bold')
  ax_bar.set_xlabel('Risk Reduction (Impact Score 0-1)', fontsize=11)
  ax_bar.grid(True, linestyle='--', alpha=0.5, axis='x')
  ax_bar.tick_params(axis='both', which='major', labelsize=10)

fig_7b.suptitle(
    'Vulnerability-Only (No Pop): Top 10 Countries Risk Reduction Potential',
    fontsize=18,
    fontweight='bold',
    y=0.98,
)
plt.tight_layout()

# %% [markdown]
# **Save Figure**

# %%
plt.show()

# %% [markdown]
# **Figure 7b Caption**: Subplot grid showing the top 10 countries for vulnerability-scaled risk reduction potential (no population term) across return periods and for the integrated expected annual reduction.

# %% [markdown]
# ---
# # SECTION C: Comprehensive Integrated Summary Dashboards
#
# Single-page dashboard layouts summarizing the integrated Expected Annual metrics.
#
# **Scientific Significance**: These dashboards compile return-period CDF curves,
# spatial distribution maps of expected skill gaps, geographic maps of expected
# annual risk reduction, and top 20 country rankings into a single, cohesive layout.
# - **Figure 12 (Population-Scaled)**: Integrates absolute human exposure data to show
#   global distributions of expected impact.
# - **Figure 13 (Vulnerability-Scaled)**: Integrates vulnerability index-adjusted skill
#   gap metrics for equity analyses.
# ---

# %% [markdown]
# ## 8. Figure 12: Integrated Summary Figure (Population-Scaled)
#
# This figure provides a comprehensive overview of the Integrated Expected Annual Risk Reduction
# (Population-Scaled) in a single 3-panel layout, showing:
# 1. Global distribution (Map, Top row)
# 2. Top 25 countries (Bar Chart, Bottom-Left)
# 3. Correlation with Human Structural Vulnerability (Scatterplot, Bottom-Right)
# The alpha parameter is omitted because warning effectiveness is not factored into this integrated expected annual presentation.

# %%
# Data Prep for Figure 12
# 1. Map Data
map_df_sum = world_gdf.merge(
    metrics_df['Risk_Reduction_Integrated'].to_frame(),
    left_on='name',
    right_index=True,
    how='left',
)

# 2. Bar Chart Data
top_25_sum = metrics_df.sort_values(
    by='Risk_Reduction_Integrated', ascending=False
).head(25)
top_25_sum_sorted = top_25_sum.sort_values(
    by='Risk_Reduction_Integrated', ascending=True
)

# 3. Correlation Data
x_col_sum = 'Human_Structural_Vulnerability'
corr_df_sum = metrics_df.dropna(subset=[x_col_sum, 'skill_gap_expected'])
corr_df_sum = corr_df_sum[corr_df_sum[x_col_sum] > 0]
corr_df_sum['log_x'] = np.log10(corr_df_sum[x_col_sum])

# %%
# Plot Figure 12
fig_12 = plt.figure(figsize=(20, 16))
gs = GridSpec(2, 2, figure=fig_12, height_ratios=[1.2, 1.0])

# Panel A: Map (Top Row, spans both columns)
ax_map = fig_12.add_subplot(gs[0, :])
no_data_df = map_df_sum[map_df_sum['Risk_Reduction_Integrated'].isna()]
no_rr_df = map_df_sum[map_df_sum['Risk_Reduction_Integrated'] <= 0]
pos_rr_df = map_df_sum[map_df_sum['Risk_Reduction_Integrated'] > 0]

map_df_sum.plot(ax=ax_map, color='lightgray', edgecolor='black', linewidth=0.1)
if not no_rr_df.empty:
  no_rr_df.plot(ax=ax_map, color='white', edgecolor='black', linewidth=0.1)
if not pos_rr_df.empty:
  val_max = pos_rr_df['Risk_Reduction_Integrated'].max()
  val_min = (
      max(1e-1, pos_rr_df['Risk_Reduction_Integrated'].min())
      if not pd.isna(pos_rr_df['Risk_Reduction_Integrated'].min())
      else 1e-1
  )
  norm = mcolors.LogNorm(vmin=val_min, vmax=val_max)
  pos_rr_df.plot(
      ax=ax_map,
      column='Risk_Reduction_Integrated',
      cmap='YlOrRd',
      norm=norm,
      legend=True,
      legend_kwds={
          'shrink': 0.8,
          'label': 'Risk Reduction (People/year)',
          'orientation': 'horizontal',
          'pad': 0.05,
      },
      edgecolor='black',
      linewidth=0.1,
  )
ax_map.set_title('A. Global Risk Reduction Map', fontsize=14, fontweight='bold')
ax_map.set_xlim(world_bounds[0], world_bounds[2])
ax_map.set_ylim(world_bounds[1], world_bounds[3])
ax_map.axis('off')

# Panel B: Bar Chart (Bottom-Left)
ax_bar = fig_12.add_subplot(gs[1, 0])
colors = plt.cm.viridis(np.linspace(0.3, 0.8, 25))
ax_bar.barh(
    top_25_sum_sorted.index,
    top_25_sum_sorted['Risk_Reduction_Integrated'],
    color=colors,
    edgecolor='black',
    height=0.6,
)
ax_bar.set_title('B. Top 25 Countries', fontsize=14, fontweight='bold')
ax_bar.set_xlabel('Risk Reduction (People/year)', fontsize=11)
ax_bar.set_ylabel('Country', fontsize=11)
ax_bar.grid(True, linestyle='--', alpha=0.5, axis='x')

# Panel C: Correlation (Bottom-Right)
ax_corr = fig_12.add_subplot(gs[1, 1])
slope, intercept, r_value, p_value, std_err = stats.linregress(
    corr_df_sum['log_x'], corr_df_sum['skill_gap_expected']
)
r2 = r_value**2
sns.scatterplot(
    data=corr_df_sum,
    x=x_col_sum,
    y='skill_gap_expected',
    s=60,
    alpha=0.7,
    color='steelblue',
    ax=ax_corr,
)
x_vals = np.logspace(
    np.log10(corr_df_sum[x_col_sum].min()),
    np.log10(corr_df_sum[x_col_sum].max()),
    100,
)
y_vals = slope * np.log10(x_vals) + intercept
ax_corr.plot(x_vals, y_vals, color='crimson', linewidth=1.8)
ax_corr.set_xscale('log')
ax_corr.set_box_aspect(1)
stats_text = f'$R^2 = {r2:.4f}$ (log-X)\n$p = {p_value:.2e}$'
ax_corr.text(
    0.05,
    0.95,
    stats_text,
    transform=ax_corr.transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.55),
)
ax_corr.set_title(
    'C. Vulnerability vs. Recall Skill Gap', fontsize=14, fontweight='bold'
)
ax_corr.set_xlabel(x_label, fontsize=11)
ax_corr.set_ylabel('Expected Recall Skill Gap', fontsize=11)
ax_corr.grid(True, linestyle='--', alpha=0.5)

fig_12.suptitle(
    'Population-Scaled: Integrated Expected Annual Risk Reduction Summary',
    fontsize=18,
    fontweight='bold',
    y=0.98,
)
plt.tight_layout()

# %%
# Save Figure 12
plt.show()

# %% [markdown]
# **Figure 12 Caption**: Summary dashboard for population-scaled integrated risk reduction, showing (A) global mapping of expected annual persons protected, (B) the top 25 countries, and (C) log-linear regression against Human Structural Vulnerability. The alpha parameter is omitted because warning effectiveness is not factored into this integrated expected annual presentation.

# %% [markdown]
# ## 9. Figure 13: Integrated Summary Figure (Vulnerability-Scaled, No Pop)
#
# This figure provides a comprehensive overview of the Integrated Expected Annual Risk Reduction
# (Vulnerability-Scaled, No Pop) in a single 3-panel layout, showing:
# 1. Global distribution (Map, Top row)
# 2. Top 25 countries (Bar Chart, Bottom-Left)
# 3. Correlation with socioeconomic vulnerability (Scatterplot, Bottom-Right)
# The alpha parameter is omitted because warning effectiveness is not factored into this vulnerability-only framework.

# %%
# Data Prep for Figure 13
# 1. Map Data
map_df_sum_no = world_gdf.merge(
    metrics_df['Risk_Reduction_No_Pop_Integrated'].to_frame(),
    left_on='name',
    right_index=True,
    how='left',
)

# 2. Bar Chart Data
top_25_sum_no = metrics_df.sort_values(
    by='Risk_Reduction_No_Pop_Integrated', ascending=False
).head(25)
top_25_sum_no_sorted = top_25_sum_no.sort_values(
    by='Risk_Reduction_No_Pop_Integrated', ascending=True
)

# 3. Correlation Data
x_col_sum_no = 'Vulnerability'
corr_df_sum_no = metrics_df.dropna(subset=[x_col_sum_no, 'skill_gap_expected'])

# %%
# Plot Figure 13
fig_13 = plt.figure(figsize=(20, 16))
gs = GridSpec(2, 2, figure=fig_13, height_ratios=[1.2, 1.0])

# Panel A: Map (Top Row, spans both columns)
ax_map = fig_13.add_subplot(gs[0, :])
no_data_df = map_df_sum_no[
    map_df_sum_no['Risk_Reduction_No_Pop_Integrated'].isna()
]
no_rr_df = map_df_sum_no[map_df_sum_no['Risk_Reduction_No_Pop_Integrated'] <= 0]
pos_rr_df = map_df_sum_no[map_df_sum_no['Risk_Reduction_No_Pop_Integrated'] > 0]

map_df_sum_no.plot(
    ax=ax_map, color='lightgray', edgecolor='black', linewidth=0.1
)
if not no_rr_df.empty:
  no_rr_df.plot(ax=ax_map, color='white', edgecolor='black', linewidth=0.1)
if not pos_rr_df.empty:
  val_max = pos_rr_df['Risk_Reduction_No_Pop_Integrated'].max()
  norm = mcolors.Normalize(vmin=0, vmax=val_max)
  pos_rr_df.plot(
      ax=ax_map,
      column='Risk_Reduction_No_Pop_Integrated',
      cmap='YlOrRd',
      norm=norm,
      legend=True,
      legend_kwds={
          'shrink': 0.8,
          'label': 'Risk Reduction (Impact Score 0-1)',
          'orientation': 'horizontal',
          'pad': 0.05,
      },
      edgecolor='black',
      linewidth=0.1,
  )
ax_map.set_title('A. Global Risk Reduction Map', fontsize=14, fontweight='bold')
ax_map.set_xlim(world_bounds[0], world_bounds[2])
ax_map.set_ylim(world_bounds[1], world_bounds[3])
ax_map.axis('off')

# Panel B: Bar Chart (Bottom-Left)
ax_bar = fig_13.add_subplot(gs[1, 0])
colors = plt.cm.viridis(np.linspace(0.3, 0.8, 25))
ax_bar.barh(
    top_25_sum_no_sorted.index,
    top_25_sum_no_sorted['Risk_Reduction_No_Pop_Integrated'],
    color=colors,
    edgecolor='black',
    height=0.6,
)
ax_bar.set_title('B. Top 25 Countries', fontsize=14, fontweight='bold')
ax_bar.set_xlabel('Risk Reduction (Impact Score)', fontsize=11)
ax_bar.set_ylabel('Country', fontsize=11)
ax_bar.grid(True, linestyle='--', alpha=0.5, axis='x')

# Panel C: Correlation (Bottom-Right)
ax_corr = fig_13.add_subplot(gs[1, 1])
slope, intercept, r_value, p_value, std_err = stats.linregress(
    corr_df_sum_no[x_col_sum_no], corr_df_sum_no['skill_gap_expected']
)
r2 = r_value**2
sns.scatterplot(
    data=corr_df_sum_no,
    x=x_col_sum_no,
    y='skill_gap_expected',
    s=60,
    alpha=0.7,
    color='steelblue',
    ax=ax_corr,
)
x_vals = np.linspace(
    corr_df_sum_no[x_col_sum_no].min(), corr_df_sum_no[x_col_sum_no].max(), 100
)
y_vals = slope * x_vals + intercept
ax_corr.plot(x_vals, y_vals, color='crimson', linewidth=1.8)
ax_corr.set_box_aspect(1)
stats_text = f'$R^2 = {r2:.4f}$\n$p = {p_value:.2e}$'
ax_corr.text(
    0.05,
    0.95,
    stats_text,
    transform=ax_corr.transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.55),
)
ax_corr.set_title(
    'C. Vulnerability vs. Recall Skill Gap', fontsize=14, fontweight='bold'
)
ax_corr.set_xlabel('Socioeconomic Vulnerability (INFORM)', fontsize=11)
ax_corr.set_ylabel('Expected Recall Skill Gap', fontsize=11)
ax_corr.grid(True, linestyle='--', alpha=0.5)

fig_13.suptitle(
    'Vulnerability-Only (No Pop): Integrated Expected Annual Risk Reduction'
    ' Summary',
    fontsize=18,
    fontweight='bold',
    y=0.98,
)
plt.tight_layout()

# %%
# Save Figure 13
plt.show()

# %% [markdown]
# **Figure 13 Caption**: Summary dashboard for vulnerability-scaled integrated risk reduction, showing (A) global mapping of expected annual vulnerability reduction, (B) the top 25 countries, and (C) linear regression against socioeconomic vulnerability. The alpha parameter is omitted because warning effectiveness is not factored into this vulnerability-only presentation.

# %% [markdown]
# ## 10. Figure 14: Correlation of Skill Gaps with Composite Flood Risk Index
#
# We correlate the gauged-to-ungauged skill gaps for all metrics (NSE, KGE, and recall
# at 1.01y, 2y, 5y, 10y, 20y return periods, plus the integrated expected recall gap)
# against a composite flood risk index defined as:
#   Composite Risk = Exposed Population * (Vulnerability / 10) * (Lack of Coping Capacity / 10)

# %%
t_start = time.time()
print(f"[{time.strftime('%H:%M:%S')}] Figure 14: Starting data compilation...")

# 1. Load Coping Capacity and define composite flood risk
coping_capacity = df_risk['LACK OF COPING CAPACITY']
if coping_capacity.dtype == 'object':
  coping_capacity = coping_capacity.str.replace(',', '', regex=False)
coping_capacity = pd.to_numeric(coping_capacity, errors='coerce')
coping_capacity = coping_capacity.groupby(level=0).first()

inform_df['Coping_Capacity_Lack'] = coping_capacity
inform_df['Composite_Flood_Risk'] = (
    inform_df['Exposed_Population']
    * (inform_df['Vulnerability'] / 10.0)
    * (inform_df['Coping_Capacity_Lack'] / 10.0)
)

# 2. Compile NSE and KGE country median gaps (with compute fallback)
extra_metrics = ['NSE', 'KGE']
extra_gaps = {}

for metric in extra_metrics:
  file_paths.init(metric)
  if gfile.Exists(file_paths.SKILL_DIFFERENCE_BY_COUNTRY):
    print(
        f'  - Loading pre-computed country gaps for {metric} from'
        f' {file_paths.SKILL_DIFFERENCE_BY_COUNTRY}'
    )
    df_gap = pd.read_csv(file_paths.SKILL_DIFFERENCE_BY_COUNTRY)
    if 'country_name' not in df_gap.columns and 'index' in df_gap.columns:
      df_gap = df_gap.rename(columns={'index': 'country_name'})
  else:
    print(
        f'  - Pre-computed file not found for {metric}. Calculating from raw'
        ' predictions...'
    )
    hybas_preds_gauged_df = pd.read_csv(
        file_paths.GLOBAL_HYBAS_PREDICTIONS_GAUGED, index_col=0
    )
    hybas_preds_ungauged_df = pd.read_csv(
        file_paths.GLOBAL_HYBAS_PREDICTIONS_UNGAUGED, index_col=0
    )
    mapping_df = pd.read_csv(file_paths.HYBAS_MAPPING_FILE)
    if 'Unnamed: 0' in mapping_df.columns:
      mapping_df = mapping_df.rename(columns={'Unnamed: 0': 'hybas_id'})
    mapping_df['country_name'] = mapping_df['country_name'].replace(
        file_paths.country_name_map
    )
    preds_df = hybas_preds_gauged_df.join(hybas_preds_ungauged_df, how='inner')
    preds_df['skill_gap'] = (
        preds_df['gauged_skill'] - preds_df['ungauged_skill']
    )
    preds_with_country = preds_df.join(
        mapping_df.set_index('hybas_id'), how='inner'
    )
    df_gap = (
        preds_with_country.groupby('country_name')['skill_gap']
        .median()
        .reset_index()
    )

    file_paths.ensure_directories_exist()
    df_gap.to_csv(file_paths.SKILL_DIFFERENCE_BY_COUNTRY, index=False)

  df_gap['country_name'] = df_gap['country_name'].replace(
      file_paths.country_name_map
  )
  extra_gaps[metric] = df_gap.set_index('country_name')

# Restore default metric state
file_paths.init(file_paths.default_metric)
print(f"[{time.strftime('%H:%M:%S')}] Figure 14: Extra metrics loaded.")

# %%
# Compile all metrics into a local dataframe for correlation
corr_df_f14 = inform_df[['Composite_Flood_Risk']].copy()

# Add NSE & KGE
for metric in ['NSE', 'KGE']:
  corr_df_f14[f'gap_{metric}'] = extra_gaps[metric]['skill_gap']

# Add Return Period Recalls & Integrated from metrics_df
common_countries = metrics_df.index
corr_df_f14 = corr_df_f14.loc[common_countries]

for yr in ['1.01', '2.0', '5.0', '10.0', '20.0']:
  corr_df_f14[f'gap_recall_{yr}'] = metrics_df[f'skill_gap_{yr}']
corr_df_f14['gap_recall_expected'] = metrics_df['skill_gap_expected']

# Clean up
x_col_f14 = 'Composite_Flood_Risk'
corr_df_f14 = corr_df_f14.dropna(subset=[x_col_f14])
corr_df_f14 = corr_df_f14[corr_df_f14[x_col_f14] > 0]
corr_df_f14['log_x'] = np.log10(corr_df_f14[x_col_f14])

print(f'Compiled correlation dataframe. Total countries: {len(corr_df_f14)}')

# %%
# Plot Figure 14
fig_14, axes = plt.subplots(4, 2, figsize=(16, 28), sharey=False)
axes = axes.flatten()

metrics_to_plot = [
    ('gap_NSE', 'NSE Skill Gap (Gauged - Ungauged)', 'NSE'),
    ('gap_KGE', 'KGE Skill Gap (Gauged - Ungauged)', 'KGE'),
    ('gap_recall_1.01', '1-Year Recall Skill Gap', '1y Recall'),
    ('gap_recall_2.0', '2-Year Recall Skill Gap', '2y Recall'),
    ('gap_recall_5.0', '5-Year Recall Skill Gap', '5y Recall'),
    ('gap_recall_10.0', '10-Year Recall Skill Gap', '10y Recall'),
    ('gap_recall_20.0', '20-Year Recall Skill Gap', '20y Recall'),
    (
        'gap_recall_expected',
        'Expected Annual Recall Skill Gap',
        'Expected Recall',
    ),
]

x_label_f14 = (
    'Composite Flood Risk Index (Exposed Pop x Vuln/10 x Coping Lack/10)'
)

for i, (col, title_name, short_name) in enumerate(metrics_to_plot):
  ax = axes[i]

  # Drop NaNs for this specific metric
  sub_df = corr_df_f14.dropna(subset=[col])

  slope, intercept, r_value, p_value, std_err = stats.linregress(
      sub_df['log_x'], sub_df[col]
  )
  r2 = r_value**2

  sns.scatterplot(
      data=sub_df, x=x_col_f14, y=col, s=60, alpha=0.7, color='teal', ax=ax
  )

  x_vals = np.logspace(
      np.log10(sub_df[x_col_f14].min()), np.log10(sub_df[x_col_f14].max()), 100
  )
  y_vals = slope * np.log10(x_vals) + intercept
  ax.plot(x_vals, y_vals, color='crimson', linewidth=1.8)

  ax.set_xscale('log')
  ax.set_box_aspect(1)

  stats_text = f'$R^2 = {r2:.4f}$ (log-X)\n$p = {p_value:.2e}$'
  ax.set_title(title_name, fontsize=12, fontweight='bold')

  if i >= 6:
    ax.set_xlabel(x_label_f14, fontsize=10)
  else:
    ax.set_xlabel('')

  ax.set_ylabel(f'{short_name} Gap', fontsize=10)
  ax.grid(True, linestyle='--', alpha=0.5)
  ax.text(
      0.05,
      0.95,
      stats_text,
      transform=ax.transAxes,
      fontsize=10,
      verticalalignment='top',
      bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.55),
  )

fig_14.suptitle(
    'Correlation of Skill Gaps with INFORM Composite Flood Risk Index',
    fontsize=16,
    fontweight='bold',
    y=0.98,
)
plt.tight_layout()

# %%
# Save Figure 14
plt.show()

# %% [markdown]
# **Figure 14 Caption**: Scatterplots and log-linear regressions of country-level gauged-to-ungauged skill penalties (for NSE, KGE, and flood recall metrics across five return periods and the integrated expected recall gap) against the composite INFORM Flood Risk Index.

# %% [markdown]
# # 8. Globally-Aggregated Warning Effectiveness & Dynamic Coping Capacity Analysis
#
# We perform a global summation of vulnerable exposure and potential risk reduction.
# We then conduct:
# 1. A sensitivity sweep of warning response rate (alpha) from 0.1 to 1.0.
# 2. A country-specific dynamic response mapping where alpha_c is defined by coping capacity:
#    alpha_c = 1.0 - Lack_of_Coping_Capacity / 10.0
# We output both numerical summaries and diagnostic plots.

# %%
# 8.1 Sensitivity Sweep Calculations
total_weight = sum(weights.values())  # sum of exceedance probability weights
total_vulnerable_exposure = metrics_df['Human_Structural_Vulnerability'].sum()
# Expected persons protected under perfect response (alpha = 1.0)
total_potential_risk_reduction = metrics_df['Risk_Reduction_Integrated'].sum()

alpha_vals = np.linspace(0.1, 1.0, 10)
persons_protected_sweep = []
unprotected_risk_sweep = []
pct_reduction_unprotected_sweep = []

for alpha in alpha_vals:
  unprotected_risk_ungauged = (
      metrics_df['Human_Structural_Vulnerability']
      * (total_weight - alpha * metrics_df['ungauged_skill_expected'])
  ).sum()
  persons_protected = alpha * total_potential_risk_reduction
  pct_reduction_unprotected = (persons_protected / unprotected_risk_ungauged) * 100

  persons_protected_sweep.append(persons_protected)
  unprotected_risk_sweep.append(unprotected_risk_ungauged)
  pct_reduction_unprotected_sweep.append(pct_reduction_unprotected)

# 8.2 Dynamic Alpha Calculations (Option C)
median_coping_deficit = metrics_df['Coping_Capacity_Lack'].median()
metrics_df['Coping_Capacity_Lack_Filled'] = metrics_df[
    'Coping_Capacity_Lack'
].fillna(median_coping_deficit)

metrics_df['alpha_c'] = 1.0 - (metrics_df['Coping_Capacity_Lack_Filled'] / 10.0)

# Calculate expected metrics under dynamic alpha
metrics_df['Persons_Protected_Dynamic'] = (
    metrics_df['alpha_c'] * metrics_df['Risk_Reduction_Integrated']
)
metrics_df['Unprotected_Risk_Dynamic'] = metrics_df[
    'Human_Structural_Vulnerability'
] * (
    total_weight - metrics_df['alpha_c'] * metrics_df['ungauged_skill_expected']
)

actual_persons_protected_dynamic = metrics_df['Persons_Protected_Dynamic'].sum()
total_unprotected_risk_dynamic = metrics_df['Unprotected_Risk_Dynamic'].sum()

pct_reduction_total_dynamic = (
    actual_persons_protected_dynamic / (total_vulnerable_exposure * total_weight)
) * 100
pct_reduction_unprotected_dynamic = (
    actual_persons_protected_dynamic / total_unprotected_risk_dynamic
) * 100
mean_alpha = metrics_df['alpha_c'].mean()
weighted_mean_alpha = (
    metrics_df['alpha_c'] * metrics_df['Human_Structural_Vulnerability']
).sum() / total_vulnerable_exposure

print('\n--- GLOBAL AGGREGATED METRICS ---')
print(
    'Total Vulnerable Exposure (People-years/year):'
    f' {total_vulnerable_exposure:,.2f}'
)
print(
    'Total Potential Risk Reduction (alpha=1.0):'
    f' {total_potential_risk_reduction:,.2f}'
)
print(
    'Mean Dynamic Response Rate (alpha_c, Exposure-weighted):'
    f' {weighted_mean_alpha:.2%}'
)
print(
    'Persons Protected under Dynamic Response Rates:'
    f' {actual_persons_protected_dynamic:,.2f}'
)
print(
    'Risk Reduction (% of early warning-unprotected risk):'
    f' {pct_reduction_unprotected_dynamic:.2f}%'
)

# %%
# 8.3 Plot Figure 15: Global Warning Effectiveness Sensitivity Dashboard
fig_15, (ax_sens, ax_bar) = plt.subplots(1, 2, figsize=(16, 7))

# Subplot A: Sensitivity Sweep
color = 'steelblue'
ax_sens.set_xlabel('Warning Response Rate ($\\alpha$)', fontsize=12)
ax_sens.set_ylabel(
    'Annual Persons Protected (People-Years / Year)', color=color, fontsize=12
)
line1 = ax_sens.plot(
    alpha_vals,
    persons_protected_sweep,
    color=color,
    marker='o',
    linewidth=2,
    label='Expected Persons Protected',
)
ax_sens.tick_params(axis='y', labelcolor=color)
ax_sens.grid(True, linestyle='--', alpha=0.5)

ax_sens_right = ax_sens.twinx()
color = 'forestgreen'
ax_sens_right.set_ylabel(
    'Reduction in Early Warning-Unprotected Risk (%)', color=color, fontsize=12
)
line2 = ax_sens_right.plot(
    alpha_vals,
    pct_reduction_unprotected_sweep,
    color=color,
    marker='s',
    linewidth=2,
    linestyle='--',
    label='% Unprotected Risk Reduction',
)
ax_sens_right.tick_params(axis='y', labelcolor=color)

# Add Option C point
ax_sens.plot(
    weighted_mean_alpha,
    actual_persons_protected_dynamic,
    color='crimson',
    marker='*',
    markersize=14,
    label='Option C (Dynamic $\\alpha_c$)',
    linestyle='None',
)
ax_sens.text(
    weighted_mean_alpha + 0.02,
    actual_persons_protected_dynamic - 80000,
    'Option C (Dynamic $\\alpha_c$)\n$\\alpha$ ='
    f' {weighted_mean_alpha:.1%}\n{actual_persons_protected_dynamic:,.0f} protected',
    color='crimson',
    fontsize=10,
    fontweight='bold',
    bbox=dict(
        boxstyle='round,pad=0.3',
        facecolor='white',
        edgecolor='crimson',
        alpha=0.8,
    ),
)

ax_sens.set_title(
    'Global Sensitivity of Flood Risk Reduction to Response Rate ($\\alpha$)',
    fontsize=13,
    fontweight='bold',
)
ax_sens.set_xlim(0.05, 1.05)

# Subplot B: Top Countries under Dynamic alpha_c
top_dynamic_countries = metrics_df.sort_values(
    by='Persons_Protected_Dynamic', ascending=False
).head(25)
y_pos = np.arange(len(top_dynamic_countries))
ax_bar.barh(
    y_pos,
    top_dynamic_countries['Persons_Protected_Dynamic'],
    align='center',
    color='teal',
    edgecolor='black',
    linewidth=0.5,
)
ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(top_dynamic_countries.index)
ax_bar.invert_yaxis()  # top-down
ax_bar.set_xlabel(
    'Expected Annual Persons Protected (People-Years / Year)', fontsize=12
)
ax_bar.set_title(
    'Top 25 Countries for Persons Protected under Dynamic Response Rates',
    fontsize=13,
    fontweight='bold',
)
ax_bar.grid(True, linestyle='--', alpha=0.5, axis='x')

plt.tight_layout()
plt.show()

# %% [markdown]
# **Figure 15 Caption**: Global early warning response rate analysis showing (A) expected annual persons protected (left axis, blue) and reduction in early warning-unprotected flood risk (right axis, green) across a sweep of uniform warning action rates ($\alpha \in [0.1, 1.0]$), with Option C (coping-capacity-scaled dynamic response rate $\alpha_c$) highlighted as a red star; and (B) the top 25 countries ranked by expected annual persons protected under the dynamic response rate model.

# %%
# 8.4 Plot Figure 16: World Map of Dynamic Early Warning Response Rate (alpha_c)
fig_16, ax_map = plt.subplots(1, 1, figsize=(15, 8))

world_gdf.plot(
    ax=ax_map, facecolor='whitesmoke', edgecolor='darkgray', linewidth=0.5
)

# Merge dynamic alpha with world geometry
world_alpha_gdf = world_gdf.merge(
    metrics_df[['alpha_c']], left_on='name', right_index=True, how='left'
)

world_alpha_gdf.dropna(subset=['alpha_c']).plot(
    column='alpha_c',
    ax=ax_map,
    cmap='RdYlGn',
    legend=True,
    legend_kwds={
        'shrink': 0.4,
        'label': 'Warning Response Rate ($\\alpha_c$)',
        'orientation': 'horizontal',
        'pad': 0.05,
    },
    edgecolor='black',
    linewidth=0.1,
)

ax_map.set_title(
    'Global Distribution of early warning response rate ($\\alpha_c$) based on'
    ' Coping Capacity',
    fontsize=15,
    fontweight='bold',
)
ax_map.axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# **Figure 16 Caption**: Choropleth world map visualizing early warning response rate ($\alpha_c$) derived from the INFORM Lack of Coping Capacity index ($\alpha_c = 1.0 - C_c/10.0$). Green indicates high response rates (strong capacity to disseminate and act on alerts); red indicates severe coping deficits that limit warning effectiveness.

# %% [markdown]
# ## 11. Figure 17: Relationship between Climate Aridity and Ungauged Model Recall
#
# We analyze how climate aridity (Potential Evapotranspiration and Aridity Index) drives
# the expected annual recall performance of the ungauged models at the country level.

# %%
t_start = time.time()
print(
    f"[{time.strftime('%H:%M:%S')}] Figure 17: Loading climate features and"
    ' mapping...'
)


# 1. Load global features from Zarr (with CSV fallback)
file_paths.init('NSE')
df_features_f17 = io_utils.load_hybas_zarr()

# 2. Extract specific aridity features (PET and UNEP Aridity Index)
# ari_ix_uav = Precipitation / PET (UNEP index)
aridity_features = ['pet_mm_uyr', 'ari_ix_uav']
df_aridity_f17 = df_features_f17[aridity_features].copy()

# 3. Load mapping file to aggregate to country level
mapping_df_f17 = pd.read_csv(file_paths.HYBAS_MAPPING_FILE)
if 'Unnamed: 0' in mapping_df_f17.columns:
  mapping_df_f17 = mapping_df_f17.rename(columns={'Unnamed: 0': 'hybas_id'})
mapping_df_f17['country_name'] = mapping_df_f17['country_name'].replace(
    file_paths.country_name_map
)

# 4. Join and calculate country-level median aridity metrics
df_aridity_with_country = df_aridity_f17.join(
    mapping_df_f17.set_index('hybas_id'), how='inner'
)
df_country_aridity = df_aridity_with_country.groupby('country_name')[
    aridity_features
].median()

# Restore default metric state
file_paths.init(file_paths.default_metric)

# 5. Merge with metrics_df (expected annual recall for ungauged)
df_plot_f17 = metrics_df[['ungauged_skill_expected']].join(
    df_country_aridity, how='inner'
)

print(f'Compiled aridity dataset for {len(df_plot_f17)} countries.')

# %%
# Plot Figure 17
fig_17, (ax_pet, ax_ari) = plt.subplots(1, 2, figsize=(16, 7))

# Panel A: Potential Evapotranspiration (PET) vs. Expected Recall
slope_pet, intercept_pet, r_pet, p_pet, _ = stats.linregress(
    df_plot_f17['pet_mm_uyr'], df_plot_f17['ungauged_skill_expected']
)
r_spearman_pet, p_spearman_pet = stats.spearmanr(
    df_plot_f17['pet_mm_uyr'], df_plot_f17['ungauged_skill_expected']
)

sns.scatterplot(
    data=df_plot_f17,
    x='pet_mm_uyr',
    y='ungauged_skill_expected',
    s=70,
    alpha=0.7,
    color='#e65100',
    edgecolor='black',
    linewidth=0.5,
    ax=ax_pet,
)
x_vals_pet = np.linspace(
    df_plot_f17['pet_mm_uyr'].min(), df_plot_f17['pet_mm_uyr'].max(), 100
)
y_vals_pet = slope_pet * x_vals_pet + intercept_pet
ax_pet.plot(
    x_vals_pet, y_vals_pet, color='black', linestyle='--', linewidth=1.5
)
ax_pet.set_title(
    'Ungauged Recall vs. Potential Evapotranspiration',
    fontsize=13,
    fontweight='bold',
)
ax_pet.set_xlabel('Country Median PET (mm/yr)', fontsize=11)
ax_pet.set_ylabel('Expected Annual Recall (Ungauged)', fontsize=11)
ax_pet.grid(True, linestyle='--', alpha=0.5)

stats_text_pet = (
    f'Pearson $r = {r_pet:.3f}$ ($p = {p_pet:.2e}$)\n'
    f'Spearman $\\rho = {r_spearman_pet:.3f}$ ($p = {p_spearman_pet:.2e}$)'
)
ax_pet.text(
    0.95,
    0.95,
    stats_text_pet,
    transform=ax_pet.transAxes,
    fontsize=10,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='whitesmoke', alpha=0.8),
)

# Panel B: Aridity Index vs. Expected Recall
slope_ari, intercept_ari, r_ari, p_ari, _ = stats.linregress(
    df_plot_f17['ari_ix_uav'], df_plot_f17['ungauged_skill_expected']
)
r_spearman_ari, p_spearman_ari = stats.spearmanr(
    df_plot_f17['ari_ix_uav'], df_plot_f17['ungauged_skill_expected']
)

sns.scatterplot(
    data=df_plot_f17,
    x='ari_ix_uav',
    y='ungauged_skill_expected',
    s=70,
    alpha=0.7,
    color='#1b5e20',
    edgecolor='black',
    linewidth=0.5,
    ax=ax_ari,
)
x_vals_ari = np.linspace(
    df_plot_f17['ari_ix_uav'].min(), df_plot_f17['ari_ix_uav'].max(), 100
)
y_vals_ari = slope_ari * x_vals_ari + intercept_ari
ax_ari.plot(
    x_vals_ari, y_vals_ari, color='black', linestyle='--', linewidth=1.5
)
ax_ari.set_title(
    'Ungauged Recall vs. UNEP Aridity Index', fontsize=13, fontweight='bold'
)
ax_ari.set_xlabel(
    'Country Median Aridity Index (Precipitation / PET)', fontsize=11
)
ax_ari.set_ylabel('Expected Annual Recall (Ungauged)', fontsize=11)
ax_ari.grid(True, linestyle='--', alpha=0.5)

stats_text_ari = (
    f'Pearson $r = {r_ari:.3f}$ ($p = {p_ari:.2e}$)\n'
    f'Spearman $\\rho = {r_spearman_ari:.3f}$ ($p = {p_spearman_ari:.2e}$)'
)
ax_ari.text(
    0.95,
    0.05,
    stats_text_ari,
    transform=ax_ari.transAxes,
    fontsize=10,
    verticalalignment='bottom',
    horizontalalignment='right',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='whitesmoke', alpha=0.8),
)

plt.suptitle(
    'Climate Aridity as the Primary Driver of Ungauged Model Recall Deficits',
    fontsize=16,
    fontweight='bold',
    y=0.98,
)
plt.tight_layout()

plt.show()

# %% [markdown]
# **Figure 17 Caption**: Scatterplots and linear regressions analyzing the relationship between country-level median climate aridity metrics and Expected Annual Recall of ungauged models across 174 countries. Panel (A) correlates performance against Potential Evapotranspiration (PET), demonstrating a severe drop in forecast skill in hotter, high-evaporative regimes. Panel (B) correlates performance against the UNEP Aridity Index (Precip/PET), showing that more arid regions (low Aridity Index) suffer from significantly lower forecast recall compared to humid regimes (high Aridity Index).
