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


"""Shared statistical analysis tools for geography bias testing in hydrology."""

import numpy as np
import pandas as pd
import scipy.stats as stats


def compute_mw_effect_size(x, y):
  """Computes Rank-Biserial Correlation and Common Language Effect Size (CLES).

  Args:
      x: Numeric array-like representing group 1.
      y: Numeric array-like representing group 2.

  Returns:
      r_rb: Standardized Rank-Biserial correlation (-1.0 to 1.0).
      cles: Probability that a randomly chosen element from group 2 is greater
        than one from group 1.
  """
  n1, n2 = len(x), len(y)
  if n1 == 0 or n2 == 0:
    return np.nan, np.nan
  combined = np.concatenate([x, y])
  ranks = stats.rankdata(combined)
  r1 = np.sum(ranks[:n1])
  # Mann-Whitney U statistic for group 1 (number of times x_i exceeds y_j)
  u1 = r1 - (n1 * (n1 + 1)) / 2.0
  # Common Language Effect Size: Probability that a Y element outperforms X
  cles = 1.0 - (u1 / (n1 * n2))
  # Rank-Biserial Correlation
  r_rb = (n2 * n1 - 2 * u1) / (n1 * n2)
  return r_rb, cles


def format_p_value(p):
  """Formats a significance p-value to a readable journal-grade string."""
  if pd.isna(p):
    return "N/A"
  if p < 0.001:
    return "< 0.001"
  return f"{p:.4f}"


def run_anomaly_significance_sweeps(scenarios, anomaly_methods):
  """Runs physical attribute anomaly severity vs. predictive skill sweeps (Notebook #3).

  Args:
      scenarios: List of tuples (split_name, eval_df, skill_col)
      anomaly_methods: Dict mapping anomaly score columns to method descriptions

  Returns:
      results_df: Pandas DataFrame containing raw statistical parameters.
      formatted_df: Pandas DataFrame containing clean formatted text representations.
      manuscript_sentence: String drafting a Nature-style results sentence.
  """
  results_summary = []

  for split_name, eval_df, skill_col in scenarios:
    for score_col, method_name in anomaly_methods.items():
      data = eval_df[[score_col, skill_col]].dropna()
      x_anomaly = data[score_col].values
      y_skill = data[skill_col].values
      
      if len(data) < 20:
        continue
        
      # Spearman Rank Correlation
      rho, p_spearman = stats.spearmanr(x_anomaly, y_skill)
      
      # Extreme Groups (Mann-Whitney U)
      bins = pd.qcut(x_anomaly, q=6, retbins=False, duplicates='drop')
      unique_bins = sorted(bins.unique())
      
      if len(unique_bins) < 2:
        continue
        
      bin_1_y = y_skill[bins == unique_bins[0]]
      bin_6_y = y_skill[bins == unique_bins[-1]]
      
      if len(bin_1_y) > 5 and len(bin_6_y) > 5:
        # Standard one-sided: physically most anomalous group has LESS skill than most normal group
        mw_stat, p_mw = stats.mannwhitneyu(bin_1_y, bin_6_y, alternative='less')
        r_rb, cles = compute_mw_effect_size(bin_1_y, bin_6_y)
      else:
        mw_stat, p_mw = np.nan, np.nan
        r_rb, cles = np.nan, np.nan
      
      results_summary.append({
          "Modeling Regime": split_name,
          "Anomaly Method": method_name,
          "Spearman Rho (\u03c1)": rho,
          "Spearman p-value": p_spearman,
          "Mann-Whitney U": mw_stat,
          "Mann-Whitney p-value": p_mw,
          "Rank-Biserial (r_rb)": r_rb,
          "CLES Effect Size": cles
      })

  results_df = pd.DataFrame(results_summary)

  formatted_df = results_df.copy()
  formatted_df["Spearman p-value"] = formatted_df["Spearman p-value"].apply(format_p_value)
  formatted_df["Mann-Whitney p-value"] = formatted_df["Mann-Whitney p-value"].apply(format_p_value)

  if len(results_df) > 0:
    max_rho = results_df["Spearman Rho (\u03c1)"].max()
    min_rho = results_df["Spearman Rho (\u03c1)"].min()
    max_p_spearman = results_df["Spearman p-value"].max()
    max_p_mw = results_df["Mann-Whitney p-value"].max()
    min_cles = results_df["CLES Effect Size"].min()
    
    manuscript_sentence = (
        f"\"Across all scenarios and outlier detection models, continuous physical anomaly scores show "
        f"a significant rank correlation with predictive skill (Spearman's \u03c1 range: {min_rho:.3f} to {max_rho:.3f}, "
        f"all p {format_p_value(max_p_spearman)}). Furthermore, when grouping catchments by their outlier severity into sextiles, "
        f"those physically most anomalous (leftmost bin) exhibit significantly lower predictive skill than catchments "
        f"statistically well-represented in the training set (rightmost bin; Mann-Whitney U test, all p {format_p_value(max_p_mw)}, "
        f"with probability of superiority CLES >= {min_cles:.2%}).\""
    )
  else:
    manuscript_sentence = "N/A"

  return results_df, formatted_df, manuscript_sentence


def run_coverage_significance_sweeps(
    df,
    melted_df,
    target_cols,
    coverage_cols,
    test_type="penalty"
):
  """Sweeps statistical significance for gauge coverage fractions vs. performance levels.

  Args:
      df: Master analysis dataframe.
      melted_df: Melted dataframe containing coverage feature mapping and ordered coverage bins.
      target_cols: Dictionary mapping dataframe column name -> printable scenario names.
      coverage_cols: List of coverage feature columns.
      test_type: String. "penalty" (for delta skill penalties) or "skill" (for absolute skill levels).

  Returns:
      results_df: Raw DataFrame of calculated test metrics.
      formatted_df: Formatted DataFrame (for readable printouts).
  """
  results = []
  
  for target_col, target_name in target_cols.items():
    clean_df = df.dropna(subset=[target_col] + coverage_cols)
    
    for cov_col in coverage_cols:
      x_cov = clean_df[cov_col].values
      y_perf = clean_df[target_col].values
      
      # 1. Spearman correlation
      rho, p_spearman = stats.spearmanr(x_cov, y_perf)
      
      # 2. Extreme group groupings (Zero Coverage vs High Coverage)
      bin_subset = melted_df[melted_df['Coverage_Type'] == cov_col].copy()
      bin_subset = bin_subset.dropna(subset=[target_col, 'Coverage_Bin'])
      unique_bins = sorted(bin_subset['Coverage_Bin'].unique())
      
      if len(unique_bins) < 2:
        continue
        
      bin_groups = [
          bin_subset[bin_subset['Coverage_Bin'] == b][target_col].values 
          for b in unique_bins
      ]
      
      zero_group = bin_groups[0]
      high_group = bin_groups[-1]
      
      if len(zero_group) > 5 and len(high_group) > 5:
        if test_type == "penalty":
          # Directional Hypothesis: penalty in zero-coverage group is GREATER than high-coverage group
          mw_stat, p_mw = stats.mannwhitneyu(zero_group, high_group, alternative='greater')
          # Effect size: probability that a zero-coverage basin has a GREATER penalty than a high-coverage one
          r_rb, cles = compute_mw_effect_size(high_group, zero_group)
        else:
          # Directional Hypothesis: absolute skill in zero-coverage group is LESS than high-coverage group
          mw_stat, p_mw = stats.mannwhitneyu(zero_group, high_group, alternative='less')
          # Effect size: probability that a high-coverage basin has a GREATER absolute skill than a zero-coverage one
          r_rb, cles = compute_mw_effect_size(zero_group, high_group)
      else:
        mw_stat, p_mw = np.nan, np.nan
        r_rb, cles = np.nan, np.nan
        
      results.append({
          "Scenario": target_name,
          "Coverage Metric": cov_col,
          "Spearman Rho": rho,
          "Spearman p-value": p_spearman,
          "Mann-Whitney p-value (Extremes)": p_mw,
          "Rank-Biserial (r_rb)": r_rb,
          "CLES Effect Size": cles
      })
      
  results_df = pd.DataFrame(results)
  
  formatted_df = results_df.copy()
  formatted_df["Spearman p-value"] = formatted_df["Spearman p-value"].apply(format_p_value)
  formatted_df["Mann-Whitney p-value (Extremes)"] = (
      formatted_df["Mann-Whitney p-value (Extremes)"].apply(format_p_value)
  )
  
  return results_df, formatted_df


def calculate_trapezoidal_expected_annual_risk(risk_data, return_periods=(5, 10, 20, 50, 100)):
  """Calculates Expected Annual Risk/Value across return periods using trapezoidal integration.

  Integrates risk values over exceedance probabilities p = 1 / T.

  Args:
      risk_data: DataFrame or Dict where columns/keys correspond to return periods.
      return_periods: Iterable of return period values T (in years).

  Returns:
      Integrated annual risk reduction (NumPy array, Series, or float).
  """
  rps = [float(T) for T in return_periods]
  probs = np.array([1.0 / T for T in rps])
  sort_idx = np.argsort(probs)
  x_probs = probs[sort_idx]

  if isinstance(risk_data, pd.DataFrame):
    # Extract matrix matching return periods sorted by ascending probability
    ordered_cols = [return_periods[i] for i in sort_idx]
    y_vals = risk_data[ordered_cols].values
    return np.trapezoid(y_vals, x=x_probs, axis=1)
  elif isinstance(risk_data, dict):
    y_vals = np.array([risk_data[T] for T in return_periods])[sort_idx]
    return np.trapezoid(y_vals, x=x_probs, axis=0)
  elif isinstance(risk_data, np.ndarray):
    y_vals = risk_data[:, sort_idx]
    return np.trapezoid(y_vals, x=x_probs, axis=1)
  else:
    raise ValueError("Unsupported input type for risk_data.")

