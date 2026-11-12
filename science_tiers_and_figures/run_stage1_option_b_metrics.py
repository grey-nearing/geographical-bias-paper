#!/usr/bin/env python3
"""Stage 1: Compute Option B (OptB-Traj) 7-day causal trajectory-crossing return-period metrics
for all 8,811 Caravan gauges across all 12 experiments (gauged, random, separated x base, _da, _ft, _ft_da)
over 1986-2023 and write them into ./data/canonical_v2/hydro_model_metrics/{dir_name}/{metric}.csv.
"""

import calendar  # noqa: F401
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from hydro_metrics.return_period_calculator import return_period_calculator as rp_calc_mod

CANON_DIR = Path('./data/canonical_v2')
STITCHED_DIR = CANON_DIR / 'model-runs' / 'stitched_preds'
METRICS_DIR = CANON_DIR / 'hydro_model_metrics'
BACKUP_METRICS_DIR = CANON_DIR / 'hydro_model_metrics_backup_pre_optB'

RETURN_PERIOD_STRS = ['1.01', '2.0', '5.0', '10.0', '20.0', '50.0']
RETURN_PERIOD_VALS = np.array([1.01, 2.0, 5.0, 10.0, 20.0, 50.0], dtype=np.float64)
WINDOW_STRS = ['0.0', '1.0', '2.0']
WINDOW_INTS = [0, 1, 2]

EXP_PAIRS = [
    ('gauged', 'gauged'),
    ('random', 'random'),
    ('basin', 'separated'),
    ('gauged_da', 'gauged_da'),
    ('random_da', 'random_da'),
    ('basin_da', 'separated_da'),
    ('gauged_ft', 'gauged_ft'),
    ('random_ft', 'random_ft'),
    ('basin_ft', 'separated_ft'),
    ('gauged_ft_da', 'gauged_ft_da'),
    ('random_ft_da', 'random_ft_da'),
    ('basin_ft_da', 'separated_ft_da'),
]

_G_OBS = None
_G_PRED = None
_G_OBS_TH = None
_G_SIM_TH = None


def _fit_sim_thresholds_chunk(args):
  exp_name, b_indices = args
  arr = np.load(STITCHED_DIR / f'{exp_name}_preds_1986_2023.npy', mmap_mode='r')
  dates = pd.date_range('1986-01-01', '2023-12-31', freq='1D')
  freq_dt = pd.Timedelta('1D')
  out = np.full((len(b_indices), 8, len(RETURN_PERIOD_VALS)), np.nan, dtype=np.float64)
  for i, b_idx in enumerate(b_indices):
    for lt in range(8):
      sim = arr[b_idx, :, lt].astype(np.float64)
      if not np.any(np.isfinite(sim)):
        continue
      try:
        calc = rp_calc_mod.ReturnPeriodCalculator(
            hydrograph_series=pd.Series(sim, index=dates),
            hydrograph_series_frequency=freq_dt,
            use_simple_fitting=True,
            verbose=False,
        )
        out[i, lt, :] = calc.flow_values_from_return_periods(RETURN_PERIOD_VALS)
      except Exception:
        pass
  return b_indices, out


def ensure_all_thresholds_cached(n_basins: int) -> dict[str, np.ndarray]:
  cache_path = Path('/tmp/rp_thresholds_8811_all12_cache_v2.npz')
  base_cache = Path('/tmp/rp_thresholds_8811_cache.npz')
  data_dict = {}
  if cache_path.exists():
    z = np.load(cache_path)
    for k in z.files:
      data_dict[k] = z[k]
  elif base_cache.exists():
    z = np.load(base_cache)
    for k in z.files:
      data_dict[k] = z[k]

  missing_exps = [
      st_name
      for st_name, _ in EXP_PAIRS
      if (f'th_{st_name}' not in data_dict) or (np.sum(~np.isnan(data_dict[f'th_{st_name}'])) == 0)
  ]
  if missing_exps:
    print(f'Fitting thresholds for {len(missing_exps)} remaining experiments: {missing_exps}...', flush=True)
    chunks = [np.arange(i, min(i + 160, n_basins)) for i in range(0, n_basins, 160)]
    with ProcessPoolExecutor(max_workers=56) as ex:
      for st_name in missing_exps:
        t0 = time.time()
        th_arr = np.full((n_basins, 8, len(RETURN_PERIOD_VALS)), np.nan, dtype=np.float64)
        for b_idxs, sub_th in ex.map(_fit_sim_thresholds_chunk, [(st_name, c) for c in chunks]):
          th_arr[b_idxs] = sub_th
        data_dict[f'th_{st_name}'] = th_arr
        n_valid = int(np.sum(~np.isnan(th_arr[:, 1, 2])))
        print(f'  Fitted {st_name} in {time.time() - t0:.1f}s (valid 5-yr LT1 basins: {n_valid}/{n_basins})', flush=True)
    np.savez(cache_path, **data_dict)
  return data_dict


def _init_eval_worker(obs_path, pred_path, obs_th, sim_th):
  global _G_OBS, _G_PRED, _G_OBS_TH, _G_SIM_TH
  _G_OBS = np.load(obs_path, mmap_mode='r')
  _G_PRED = np.load(pred_path, mmap_mode='r')
  _G_OBS_TH = obs_th
  _G_SIM_TH = sim_th


def _eval_opt_b_chunk(b_indices):
  """Evaluates Option B (OptB-Traj) across 6 return periods [1.01, 2.0, 5.0, 10.0, 20.0, 50.0]
  and 3 windows [0, 1, 2] over 1986-2023 (13,879 days).
  Columns: 6 RPs x 3 Windows x 3 (precision, recall, f1) = 54 columns.
  """
  n_days = _G_OBS.shape[1]
  out_rows = np.full((len(b_indices), 54), np.nan, dtype=np.float64)

  for i_local, b_idx in enumerate(b_indices):
    obs = _G_OBS[b_idx].astype(np.float64)  # (N_days,)
    sim = _G_PRED[b_idx].astype(np.float64)  # (N_days, 8) aligned by target verification date t

    for rp_i in range(6):  # 1.01, 2.0, 5.0, 10.0, 20.0, 50.0
      th_o = _G_OBS_TH[b_idx, rp_i]
      th_s = _G_SIM_TH[b_idx, :, rp_i]  # (8,)
      if np.isnan(th_o) or np.any(np.isnan(th_s)):
        continue

      # 1. Observed upward crossings at t^* (index u = t^* - 1 in 0..n_days-2):
      # obs[u] < th_o and obs[u+1] >= th_o
      obs_above = np.where(np.isnan(obs), np.nan, (obs >= th_o).astype(np.float64))
      obs_cross = np.maximum(0.0, np.diff(obs_above))  # length n_days - 1
      obs_u_idxs = np.where((obs_cross > 0) & ~np.isnan(obs_cross))[0]

      # 2. Forecast trajectory upward crossings along each issue day d:
      # Forecast issued on day d has trajectory (Q_{d|d}, Q_{d+1|d}, ..., Q_{d+7|d}).
      # Step tau (for tau = 1..7) crosses upward at target date t' = d + tau (crossing index u' = t' - 1 = d + tau - 1)
      # iff sim[d + tau - 1, tau - 1] < th_s[tau - 1] and sim[d + tau, tau] >= th_s[tau].
      sim_above = np.where(np.isnan(sim), np.nan, (sim >= th_s[None, :]).astype(np.float64))
      cross_mat = np.zeros((n_days - 1, 7), dtype=bool)
      nan_mat = np.zeros((n_days - 1, 7), dtype=bool)
      for tau in range(1, 8):
        prev_state = sim_above[tau - 1 : n_days - 1, tau - 1]
        curr_state = sim_above[tau : n_days, tau]
        is_nan = np.isnan(prev_state) | np.isnan(curr_state)
        is_cross = (~is_nan) & (prev_state == 0.0) & (curr_state == 1.0)
        cross_mat[tau - 1 :, tau - 1] = is_cross
        nan_mat[tau - 1 :, tau - 1] = is_nan

      pred_any_u = np.any(cross_mat, axis=1)
      pred_u_idxs = np.where(pred_any_u)[0]
      obs_nan_u = np.isnan(obs_cross)

      for w_i, w in enumerate(WINDOW_INTS):
        col_base = (rp_i * 3 + w_i) * 3
        # --- RECALL: fraction of observed crossings u^* detected by a causal forecast crossing at u' with |u' - u^*| <= w and d = u' + 1 - tau <= u^* ---
        # Note: matching stitch_and_compute_ft_da_metrics.py line 396 (`discard_nans_in_window=False` for recall, `True` for precision)
        if obs_u_idxs.size > 0:
          tp_rec = 0
          for u_star in obs_u_idxs:
            lo = max(0, u_star - w)
            hi = min(n_days - 2, u_star + w)
            hit = False
            for u_p in range(lo, hi + 1):
              min_tau = max(1, u_p - u_star + 1)
              if np.any(cross_mat[u_p, min_tau - 1 :]):
                hit = True
                break
            if hit:
              tp_rec += 1
          rec_val = tp_rec / obs_u_idxs.size
        else:
          rec_val = np.nan

        # --- PRECISION: fraction of predicted crossing target dates u' that match an observed crossing u^* within |u' - u^*| <= w with d <= u^* ---
        if pred_u_idxs.size > 0:
          tot_prec = 0
          tp_prec = 0
          for u_p in pred_u_idxs:
            lo = max(0, u_p - w)
            hi = min(n_days - 2, u_p + w)
            if np.any(obs_nan_u[lo : hi + 1]):
              continue
            tot_prec += 1
            taus_crossed = np.where(cross_mat[u_p])[0] + 1
            tau_max = int(taus_crossed.max())
            d_min = u_p + 1 - tau_max
            match_lo = max(lo, d_min)
            if match_lo <= hi and np.any((obs_cross[match_lo : hi + 1] > 0)):
              tp_prec += 1
          prec_val = (tp_prec / tot_prec) if tot_prec > 0 else np.nan
        else:
          prec_val = np.nan

        if np.isnan(prec_val) or np.isnan(rec_val):
          f1_val = np.nan
        elif (prec_val + rec_val) == 0.0:
          f1_val = 0.0
        else:
          f1_val = 2.0 * (prec_val * rec_val) / (prec_val + rec_val)

        out_rows[i_local, col_base] = prec_val
        out_rows[i_local, col_base + 1] = rec_val
        out_rows[i_local, col_base + 2] = f1_val

  return b_indices, out_rows


def main():
  t0 = time.time()
  grdc_8811 = pd.read_csv(CANON_DIR / 'attributes' / 'grdc_attributes.csv', index_col=0)
  basins = [str(b) for b in grdc_8811.index]
  n_basins = len(basins)

  # 1. Back up existing hydro_model_metrics if not already backed up
  if not BACKUP_METRICS_DIR.exists():
    print(f'Backing up {METRICS_DIR} -> {BACKUP_METRICS_DIR}...', flush=True)
    shutil.copytree(METRICS_DIR, BACKUP_METRICS_DIR)

  # 2. Ensure all 12 experiments have fitted return-period thresholds cached
  th_dict = ensure_all_thresholds_cached(n_basins)
  obs_th = th_dict['obs_th']
  obs_path = str(STITCHED_DIR / 'obs_1986_2023.npy')

  rp_metric_names = []
  for rp_s in RETURN_PERIOD_STRS:
    for w_s in WINDOW_STRS:
      rp_metric_names.append(f'{rp_s}-year-{w_s}-window-precision')
      rp_metric_names.append(f'{rp_s}-year-{w_s}-window-recall')
      rp_metric_names.append(f'{rp_s}-year-{w_s}-window-f1')

  chunks = [np.arange(i, min(i + 256, n_basins)) for i in range(0, n_basins, 256)]
  lt_cols = [str(lt) for lt in range(8)]

  # 3. Evaluate Option B for all 12 experiments and write `{m_name}.csv`
  for st_name, dir_name in EXP_PAIRS:
    t_exp = time.time()
    pred_path = str(STITCHED_DIR / f'{st_name}_preds_1986_2023.npy')
    sim_th = th_dict[f'th_{st_name}']

    res_mat = np.full((n_basins, 54), np.nan, dtype=np.float64)
    with ProcessPoolExecutor(
        max_workers=28,
        initializer=_init_eval_worker,
        initargs=(obs_path, pred_path, obs_th, sim_th),
    ) as ex:
      for b_idxs, sub_rows in ex.map(_eval_opt_b_chunk, chunks):
        res_mat[b_idxs] = sub_rows

    exp_dir = METRICS_DIR / dir_name
    for col_i, m_name in enumerate(rp_metric_names):
      csv_path = exp_dir / f'{m_name}.csv'
      vec = res_mat[:, col_i]
      # Tile across columns '0'..'7' so any lead_time lookup ('0' or '1') gets the exact 7-day Option B trajectory score
      mat_8 = np.tile(vec[:, None], (1, 8))
      pd.DataFrame(mat_8, index=basins, columns=lt_cols).to_csv(csv_path)

    idx_r5 = rp_metric_names.index('5.0-year-2.0-window-recall')
    idx_p5 = rp_metric_names.index('5.0-year-2.0-window-precision')
    idx_f5 = rp_metric_names.index('5.0-year-2.0-window-f1')
    r5_med = float(np.nanmedian(res_mat[:, idx_r5]) * 100.0)
    r5_mean = float(np.nanmean(res_mat[:, idx_r5]) * 100.0)
    p5_med = float(np.nanmedian(res_mat[:, idx_p5]) * 100.0)
    f5_med = float(np.nanmedian(res_mat[:, idx_f5]) * 100.0)
    print(
        f'  [{dir_name:16s}] Updated 54 CSVs in {time.time() - t_exp:.1f}s | '
        f'5-yr (w=2) Recall med={r5_med:.1f}% (mean={r5_mean:.1f}%), Prec med={p5_med:.1f}%, F1 med={f5_med:.1f}%',
        flush=True,
    )

  print(f'\nStage 1 complete in {time.time() - t0:.1f}s!', flush=True)


if __name__ == '__main__':
  main()
