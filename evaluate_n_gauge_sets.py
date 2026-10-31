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

"""Global & National Multi-Gauge Siting Evaluator (v2: M_ungauged-Centered).

Evaluates:
1. Global N = 10, 50, 100, 1000 gauge network expansions across three siting
   strategies on `M_ungauged`:
   - Bucket 1 (Unweighted OOD Representativeness — fixed ascending OOD sort)
   - Bucket 2A (Global Flood Exposure)
   - Bucket 2B (Exposure-Weighted OOD)
   measuring both local/river-network gains and global physical-neighborhood OOD
   transfer, plus the independent `M_gauged` self-site confirmation check.
2. Exact river-network single-candidate evaluations across all country-top-20 &
   global-top-1000 candidate sub-basins (`single_candidate_evaluations_v2.csv`).
3. Exact joint multi-gauge (`N = 1, 5, 10, 20`) national river-network
   evaluations across all mapped countries with zero nested double-counting
   (`country_joint_n_gauge_exposure_reduction_v2.csv`).
"""

from collections import defaultdict
import importlib
import os
import time

from absl import app
from absl import flags
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import select_candidates
from evaluate_gauges import GaugeValueEvaluator
from evaluate_gauges import get_shared_global_resources
from utils import cap_utils
from utils import file_paths
from utils import stats_utils
from utils import gfile_compat as gfile

FLAGS = flags.FLAGS


def _load_candidates_csv(filepath: str) -> list[str]:
  with gfile.Open(filepath, 'r') as f:
    df = pd.read_csv(f)
  return [
      idx if str(idx).startswith('hybas_') else 'hybas_' + str(idx)
      for idx in df['hybas_id'].astype(str)
  ]


def _ensure_v2_candidates(candidates_dir: str):
  """Generates v2 candidate lists (with fixed ascending OOD sort) if absent."""
  req_files = [
      'representativeness_candidates.csv',
      'exposure_global_1000_candidates.csv',
      'exposure_ood_1000_candidates.csv',
      'exposure_country_top_20_candidates.csv',
  ]
  if all(gfile.Exists(os.path.join(candidates_dir, fn)) for fn in req_files):
    print(f"[{time.strftime('%H:%M:%S')}] Found existing v2 candidate lists in {candidates_dir}")
    return

  print(f"[{time.strftime('%H:%M:%S')}] Generating clean v2 candidate lists in {candidates_dir}...")
  gfile.MakeDirs(candidates_dir)
  df_base, physical_cols = select_candidates.load_global_data()
  rep_cands = select_candidates.generate_representativeness_bucket(
      df_base=df_base,
      physical_cols=physical_cols,
      n_candidates=1000,
      n_clusters=2000,
  )
  g1000, g_exp_ood_1000, c_top20 = select_candidates.generate_exposure_buckets(df_base)
  for fn, lst in [
      ('representativeness_candidates.csv', rep_cands),
      ('exposure_global_1000_candidates.csv', g1000),
      ('exposure_ood_1000_candidates.csv', g_exp_ood_1000),
      ('exposure_country_top_20_candidates.csv', c_top20),
      ('combined_candidates.csv', sorted(set(rep_cands + g1000 + g_exp_ood_1000 + c_top20))),
  ]:
    with gfile.Open(os.path.join(candidates_dir, fn), 'w') as f:
      pd.DataFrame({'hybas_id': lst}).to_csv(f, index=False)


def main(argv):
  with joblib.parallel_backend('threading'):
    return _main_impl(argv)


def _main_impl(argv):
  del argv
  t_script = time.time()
  file_paths.init('5.0-year-2.0-window-recall')
  file_paths.ensure_directories_exist()

  candidates_dir = os.path.join(file_paths.V2_RESULTS_DIR, 'candidates')
  _ensure_v2_candidates(candidates_dir)

  bucket_1 = _load_candidates_csv(
      os.path.join(candidates_dir, 'representativeness_candidates.csv')
  )
  bucket_2a = _load_candidates_csv(
      os.path.join(candidates_dir, 'exposure_global_1000_candidates.csv')
  )
  bucket_2b = _load_candidates_csv(
      os.path.join(candidates_dir, 'exposure_ood_1000_candidates.csv')
  )
  country_top_20 = _load_candidates_csv(
      os.path.join(candidates_dir, 'exposure_country_top_20_candidates.csv')
  )

  # Initialize all 9 continental evaluators (shares global cache!)
  continents = ['af', 'as', 'au', 'eu', 'na', 'sa', 'si', 'ar', 'gr']
  evaluators = {}
  for cont in continents:
    evaluators[cont] = GaugeValueEvaluator(
        continent_code=cont, metric='5.0-year-2.0-window-recall'
    )

  shared = get_shared_global_resources('5.0-year-2.0-window-recall')
  df_global = shared['df_global']
  physical_cols = shared['physical_cols']
  caravan_phys = shared['caravan_features']
  exposures_global_df = shared['exposures_global_df']
  country_series = shared['country_series']
  anom_lib = importlib.import_module(
      '1_anomaly_scores'
  )

  # Precompute 20th-nearest Caravan distance in standardized 78D physical space
  # for all exposed global sub-basins so we can identify exact physical-space
  # OOD neighbors whose local training density increases when N gauges are added.
  print(f"[{time.strftime('%H:%M:%S')}] Precomputing 20-NN Caravan radius across exposed global basins...")
  t0_nn = time.time()
  return_periods = [5, 10, 20, 50, 100]
  annual_exp_global = pd.Series(
      stats_utils.calculate_trapezoidal_expected_annual_risk(
          {T: exposures_global_df[T].values for T in return_periods},
          return_periods=return_periods,
      ),
      index=df_global.index,
  )
  exposed_ids = annual_exp_global[annual_exp_global > 0].index.tolist()

  scaler_phys = StandardScaler()
  X_caravan_sc = scaler_phys.fit_transform(caravan_phys)
  X_exposed_sc = scaler_phys.transform(
      df_global.loc[exposed_ids, physical_cols].fillna(0.0)
  )
  nn_caravan = NearestNeighbors(n_neighbors=20, n_jobs=-1)
  nn_caravan.fit(X_caravan_sc)
  dists_20_caravan, _ = nn_caravan.kneighbors(X_exposed_sc)
  radius_20_caravan = dists_20_caravan[:, -1]
  print(
      f"[{time.strftime('%H:%M:%S')}] Computed 20-NN Caravan radius for "
      f"{len(exposed_ids)} exposed basins in {time.time() - t0_nn:.2f}s."
  )

  # ============================================================================
  # PART 1: Global N = 10, 50, 100, 1000 Multi-Strategy Siting Evaluation
  # ============================================================================
  results_global_sets = []
  sample_sizes = [10, 50, 100, 1000]
  strategies = [
      ('Bucket 1 (Representativeness)', bucket_1),
      ('Bucket 2A (Global Exposure)', bucket_2a),
      ('Bucket 2B (Exposure-Weighted OOD)', bucket_2b),
  ]

  for bucket_name, full_list in strategies:
    for N in sample_sizes:
      if N > len(full_list):
        continue
      subset = [g for g in full_list[:N] if g in df_global.index]
      print(f"\n[{time.strftime('%H:%M:%S')}] Evaluating {bucket_name} for N={N}...")
      t0 = time.time()

      # 1. Collect all river-network affected_ids across continents
      all_river_affected = set()
      for cont, evaluator in evaluators.items():
        local_sub = [c for c in subset if c in evaluator.df_base.index]
        if local_sub:
          all_river_affected.update(evaluator._get_affected_ids(local_sub))

      # 2. Identify exposed physical-space neighbors where at least 1 new gauge
      # enters the 20-nearest training neighborhood
      X_sub_sc = scaler_phys.transform(
          df_global.loc[subset, physical_cols].fillna(0.0)
      )
      nn_new = NearestNeighbors(n_neighbors=1, n_jobs=-1)
      nn_new.fit(X_sub_sc)
      dist_to_new, _ = nn_new.kneighbors(X_exposed_sc)
      ood_neighbor_mask = dist_to_new[:, 0] < radius_20_caravan
      ood_neighbor_ids = [
          exposed_ids[i] for i in np.where(ood_neighbor_mask)[0]
      ]

      union_eval_ids = list(all_river_affected.union(ood_neighbor_ids))
      print(
          f"   -> River-network basins: {len(all_river_affected):,} | "
          f"20-NN OOD exposed neighbors: {len(ood_neighbor_ids):,} | "
          f"Union: {len(union_eval_ids):,}"
      )

      # 3. Fit canonical anomaly detectors ONCE on full global Caravan + subset
      X_train_anom = pd.concat(
          [caravan_phys, df_global.loc[subset, physical_cols]], axis=0
      )
      X_test_anom = df_global.loc[union_eval_ids, physical_cols].copy()
      t_anom = time.time()
      anom_dict = anom_lib.calculate_anomaly_scores(X_train_anom, X_test_anom)
      precomputed_anom_df = pd.DataFrame(anom_dict, index=union_eval_ids)
      # Preserve deterministic OCSVM & LOF updates on OOD-only neighbors while
      # avoiding IsolationForest 256-sample bootstrap seed jitter off-network:
      non_river_ids = [b for b in union_eval_ids if b not in all_river_affected]
      if non_river_ids:
        precomputed_anom_df.loc[non_river_ids, 'anomaly_score_if'] = (
            df_global.loc[non_river_ids, 'anomaly_score_if'].values
        )
      print(f"   -> Global anomaly update completed in {time.time() - t_anom:.2f}s")

      # 4. Evaluate across all 9 continents
      river_exp_reduced = 0.0
      total_with_ood_exp_reduced = 0.0
      mgauged_confirm_river_exp = 0.0

      for cont, evaluator in evaluators.items():
        local_sub = [c for c in subset if c in evaluator.df_base.index]
        local_extra = [
            b for b in ood_neighbor_ids if b in evaluator.df_base.index
        ]
        if not local_sub and not local_extra:
          continue

        # River-network scope (canonical Global_Risk_Reduced + M_gauged confirm)
        if local_sub:
          ev_river = evaluator.evaluate(
              local_sub,
              use_gauged_for_self=False,
              precomputed_anom_df=precomputed_anom_df,
          )
          river_exp_reduced += ev_river.get('global_risk_2_exp', 0.0)
          mgauged_confirm_river_exp += ev_river.get(
              'global_risk_2_exp_mgauged_confirm', 0.0
          )

        # Combined river-network + global 20-NN OOD transfer scope
        ev_full = evaluator.evaluate(
            local_sub,
            use_gauged_for_self=False,
            precomputed_anom_df=precomputed_anom_df,
            extra_eval_ids=local_extra,
        )
        total_with_ood_exp_reduced += ev_full.get('global_risk_2_exp', 0.0)

      elapsed = time.time() - t0
      print(
          f"   -> Done in {elapsed:.1f}s | River-Network M_ungauged: {river_exp_reduced:,.1f} "
          f"| +Global OOD Transfer: {total_with_ood_exp_reduced:,.1f} "
          f"| M_gauged Self-Confirm: {mgauged_confirm_river_exp:,.1f}"
      )
      results_global_sets.append({
          'Bucket': bucket_name,
          'N': N,
          'Global_Risk_Reduced': river_exp_reduced,
          'Global_Risk_Reduced_With_OOD_Transfer': total_with_ood_exp_reduced,
          'Global_Risk_Reduced_MGauged_Confirm': mgauged_confirm_river_exp,
          'N_River_Network_Basins': len(all_river_affected),
          'N_Global_OOD_Neighbor_Basins': len(ood_neighbor_ids),
          'Elapsed_Seconds': elapsed,
      })

  df_sets_summary = pd.DataFrame(results_global_sets)
  out_sets_csv = os.path.join(
      file_paths.STATS_DIR, 'n_gauge_sets_evaluation_summary_v2.csv'
  )
  with gfile.Open(out_sets_csv, 'w') as f:
    df_sets_summary.to_csv(f, index=False)
  print(f"\n[{time.strftime('%H:%M:%S')}] Saved {out_sets_csv}:")
  print(df_sets_summary.to_string(index=False))

  # ============================================================================
  # PART 2: Batched Single-Candidate Evaluations (Country Top-20 + Global Top-1000)
  # ============================================================================
  print(f"\n[{time.strftime('%H:%M:%S')}] Running batched single-gauge evaluations on M_ungauged...")
  t0_single = time.time()
  all_single_cands = sorted(set(country_top_20 + bucket_2a + bucket_2b))
  single_rows = []
  for cont, evaluator in evaluators.items():
    cont_cands = [c for c in all_single_cands if c in evaluator.df_base.index]
    if not cont_cands:
      continue
    t_c = time.time()
    named_sets = [(c, [c], None) for c in cont_cands]
    cont_res = evaluator.evaluate_sets_batched(named_sets)
    single_rows.extend(cont_res)
    print(
        f"   -> {cont.upper()}: evaluated {len(cont_res)} single-gauge candidates in {time.time() - t_c:.2f}s"
    )

  df_single = pd.DataFrame(single_rows)
  out_single_csv = os.path.join(
      file_paths.STATS_DIR, 'single_candidate_evaluations_v2.csv'
  )
  with gfile.Open(out_single_csv, 'w') as f:
    df_single.to_csv(f, index=False)
  print(
      f"[{time.strftime('%H:%M:%S')}] Saved {len(df_single)} single-candidate "
      f"evaluations in {time.time() - t0_single:.2f}s -> {out_single_csv}"
  )

  # ============================================================================
  # PART 3: Exact Joint Multi-Gauge (N = 1, 5, 10, 20) National Evaluations
  # ============================================================================
  print(f"\n[{time.strftime('%H:%M:%S')}] Running exact joint N=1,5,10,20 national gauge sets...")
  t0_nat = time.time()
  country_base_exp = (
      pd.DataFrame({'country': country_series, 'exp': annual_exp_global})
      .dropna(subset=['country'])
      .groupby('country')['exp']
      .sum()
  )

  # Rank each country's candidates by within-country exposure reduction on M_ungauged
  df_single_valid = df_single.dropna(subset=['country_name']).copy()
  country_ranked_cands = {}
  for c_name, grp in df_single_valid.groupby('country_name'):
    grp_sorted = grp.sort_values(
        by=['within_country_risk_2_exp', 'global_risk_2_exp'], ascending=False
    )
    country_ranked_cands[c_name] = grp_sorted['candidate_hybas_id'].tolist()[:20]

  N_list = [1, 5, 10, 20]
  country_joint_raw = {N: defaultdict(float) for N in N_list}
  country_joint_confirm = {N: defaultdict(float) for N in N_list}
  country_max_cands = {c: len(lst) for c, lst in country_ranked_cands.items()}

  for cont, evaluator in evaluators.items():
    named_country_sets = []
    for c_name, c_list in country_ranked_cands.items():
      for N in N_list:
        sub_n = c_list[: min(N, len(c_list))]
        local_sub_n = [g for g in sub_n if g in evaluator.df_base.index]
        if local_sub_n:
          named_country_sets.append((f'{c_name}__N{N}', local_sub_n, c_name))
    if not named_country_sets:
      continue
    cont_nat_res = evaluator.evaluate_sets_batched(named_country_sets)
    for r in cont_nat_res:
      c_name, n_tag = r['set_id'].rsplit('__N', 1)
      N_val = int(n_tag)
      country_joint_raw[N_val][c_name] += r['within_country_risk_2_exp']
      country_joint_confirm[N_val][c_name] += r[
          'within_country_risk_2_exp_mgauged_confirm'
      ]

  nat_rows = []
  for c_name in sorted(country_ranked_cands.keys()):
    base_e = float(country_base_exp.get(c_name, 0.0))
    if base_e <= 0:
      continue
    row = {
        'Country': c_name,
        'Baseline Exposure (people/yr)': base_e,
        'Max_Evaluated_Candidates': country_max_cands.get(c_name, 0),
    }
    for N in N_list:
      raw_n = float(country_joint_raw[N].get(c_name, 0.0))
      pct_n = min(100.0, (raw_n / base_e) * 100.0)
      conf_n = float(country_joint_confirm[N].get(c_name, 0.0))
      row[f'N={N} Exposure Reduced (people/yr)'] = raw_n
      row[f'N={N} Exposure Reduced (%)'] = pct_n
      row[f'N={N} Exposure Reduced MGauged Confirm (people/yr)'] = conf_n
    row['CAP Warning System Tier'] = cap_utils.assign_cap_tier(c_name)
    row['Operational System Status'] = cap_utils.assign_cap_binary_group(c_name)
    nat_rows.append(row)

  df_nat = pd.DataFrame(nat_rows)
  out_nat_csv = os.path.join(
      file_paths.STATS_DIR, 'country_joint_n_gauge_exposure_reduction_v2.csv'
  )
  with gfile.Open(out_nat_csv, 'w') as f:
    df_nat.to_csv(f, index=False)
  print(
      f"[{time.strftime('%H:%M:%S')}] Saved joint national N=1,5,10,20 table "
      f"({len(df_nat)} countries) in {time.time() - t0_nat:.2f}s -> {out_nat_csv}"
  )
  print(
      df_nat.groupby('Operational System Status')[
          [
              'N=1 Exposure Reduced (%)',
              'N=5 Exposure Reduced (%)',
              'N=10 Exposure Reduced (%)',
              'N=20 Exposure Reduced (%)',
          ]
      ]
      .median()
      .to_string()
  )
  print(f"\n[{time.strftime('%H:%M:%S')}] ALL N-GAUGE EVALUATIONS COMPLETE in {time.time() - t_script:.1f}s.")


if __name__ == '__main__':
  app.run(main)
