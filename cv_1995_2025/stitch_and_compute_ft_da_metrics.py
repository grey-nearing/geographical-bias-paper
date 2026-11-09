"""Stitch 1986-2023 out-of-sample predictions and compute all 68 metrics for any exp_suffix ('_da', '_ft', '_ft_da').

Usage:
  cd /tmp && python3 stitch_and_compute_ft_da_metrics.py --exp-suffix _da --workers 32
"""

from __future__ import annotations

import argparse
import calendar  # Import stdlib calendar before adding local_pkg to sys.path
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any
import warnings

sys.path.insert(0, './data')

import numpy as np
import pandas as pd
from scipy import signal
from intelligence.gsnearing.geographical_bias_paper.hydro_metrics.return_period_calculator import return_period_calculator as rp_calc_mod

warnings.filterwarnings('ignore')

CANONICAL_V2_DIR = Path('./data')
RUNS_FT_DA_DIR = CANONICAL_V2_DIR / 'model-runs' / 'runs_ft_da'
STITCHED_DIR = CANONICAL_V2_DIR / 'model-runs' / 'stitched_preds'
METRICS_DIR = CANONICAL_V2_DIR / 'hydro_model_metrics'

RETURN_PERIOD_STRS = ['1.01', '2.0', '5.0', '10.0', '20.0', '50.0']
RETURN_PERIOD_VALS = np.array([1.01, 2.0, 5.0, 10.0, 20.0, 50.0], dtype=np.float64)
WINDOW_STRS = ['0.0', '1.0', '2.0']
WINDOW_INTS = [0, 1, 2]
LEAD_TIMES = list(range(8))

TEMPORAL_FOLDS: list[dict[str, Any]] = [
    {'tfold': 0, 'name': '1986_1990'},
    {'tfold': 1, 'name': '1991_1995'},
    {'tfold': 2, 'name': '1996_2000'},
    {'tfold': 3, 'name': '2001_2005'},
    {'tfold': 4, 'name': '2006_2010'},
    {'tfold': 5, 'name': '2011_2015'},
    {'tfold': 6, 'name': '2016_2017'},
    {'tfold': 7, 'name': '2018_2019'},
    {'tfold': 8, 'name': '2020_2021'},
    {'tfold': 9, 'name': '2022_2023'},
]

CONTINUOUS_METRIC_NAMES = [
    'NSE',
    'KGE',
    'log-NSE',
    'log-KGE',
    'Beta-NSE',
    'Beta-KGE',
    'Alpha-NSE',
    'pearson-r',
    'MSE',
    'RMSE',
    'FHV',
    'FLV',
    'FMS',
    'Peak-MAPE',
]

_G_OBS: np.ndarray | None = None
_G_GAUGED: np.ndarray | None = None
_G_RANDOM: np.ndarray | None = None
_G_BASIN: np.ndarray | None = None
_G_DATES: pd.DatetimeIndex | None = None


def stitch_predictions_for_experiment(
    run_exp_name: str,
    basins: list[str],
    dates_1986_2023: pd.DatetimeIndex,
    allow_partial: bool = False,
) -> tuple[Path, dict[str, Any]]:
  """Stitch all 100 (sfold_0..9, tfold_0..9) preds.npz files into (8811, 13879, 8) float32."""
  STITCHED_DIR.mkdir(parents=True, exist_ok=True)
  n_basins = len(basins)
  n_days = len(dates_1986_2023)
  basin_to_idx = {b: i for i, b in enumerate(basins)}
  date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates_1986_2023)}

  stitched_path = STITCHED_DIR / f'{run_exp_name}_preds_1986_2023.npy'
  stitched = np.lib.format.open_memmap(
      stitched_path,
      mode='w+',
      dtype=np.float32,
      shape=(n_basins, n_days, 8),
  )
  stitched[:] = np.nan

  n_sfolds = 10  # All experiments in runs_ft_da (gauged_*, random_*, basin_*) are sharded into sfold_0..9
  loaded_files = 0
  missing_files = []

  d0 = dates_1986_2023[0]
  d_max = dates_1986_2023[-1]

  for sfold in range(n_sfolds):
    for tf in TEMPORAL_FOLDS:
      tfold = tf['tfold']
      tfold_name = tf['name']
      npz_path = (
          RUNS_FT_DA_DIR
          / run_exp_name
          / f'sfold_{sfold}'
          / f'tfold_{tfold}_{tfold_name}'
          / 'preds.npz'
      )
      if not npz_path.exists():
        missing_files.append(str(npz_path))
        continue

      data = np.load(npz_path)
      preds = data['preds']  # (N_test_basins, N_test_days, 8), indexed by issue date
      test_basins = [str(b) for b in data['basins']]
      test_dates = pd.to_datetime(data['dates'])

      b_indices = np.array([basin_to_idx[b] for b in test_basins], dtype=np.int64)
      for lt in range(8):
        verif_dates = test_dates + pd.Timedelta(days=lt - 1)
        valid_mask = (verif_dates >= d0) & (verif_dates <= d_max)
        if not np.any(valid_mask):
          continue
        v_sub = verif_dates[valid_mask]
        v_indices = np.array([date_to_idx[pd.Timestamp(d)] for d in v_sub], dtype=np.int64)
        if len(v_indices) > 0 and np.all(np.diff(v_indices) == 1):
          v_slice = slice(int(v_indices[0]), int(v_indices[-1]) + 1)
          stitched[b_indices, v_slice, lt] = preds[:, valid_mask, lt]
        else:
          stitched[b_indices[:, None], v_indices[None, :], lt] = preds[:, valid_mask, lt]
      loaded_files += 1

  stitched.flush()

  if missing_files and not allow_partial:
    raise FileNotFoundError(
        f'Missing {len(missing_files)} prediction files for {run_exp_name}: {missing_files[:5]}'
    )

  valid_cells_lt0 = int(np.isfinite(stitched[:, :, 0]).sum())
  valid_cells_lt1 = int(np.isfinite(stitched[:, :, 1]).sum())
  total_cells = n_basins * n_days
  meta = {
      'experiment': run_exp_name,
      'loaded_files': loaded_files,
      'expected_files': n_sfolds * len(TEMPORAL_FOLDS),
      'n_basins': n_basins,
      'n_days': n_days,
      'valid_basin_day_cells_lt0': valid_cells_lt0,
      'valid_basin_day_cells_lt1': valid_cells_lt1,
      'coverage_fraction_lt1': round(valid_cells_lt1 / max(total_cells, 1), 6),
  }
  with open(STITCHED_DIR / f'{run_exp_name}_preds_1986_2023_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

  return stitched_path, meta


def _init_worker(obs_path: str, gauged_path: str, random_path: str, basin_path: str) -> None:
  global _G_OBS, _G_GAUGED, _G_RANDOM, _G_BASIN, _G_DATES
  _G_OBS = np.load(obs_path, mmap_mode='r')
  _G_GAUGED = np.load(gauged_path, mmap_mode='r')
  _G_RANDOM = np.load(random_path, mmap_mode='r')
  _G_BASIN = np.load(basin_path, mmap_mode='r')
  _G_DATES = pd.date_range('1986-01-01', '2023-12-31', freq='1D')


def _get_crossovers(arr: np.ndarray, thresh: float) -> np.ndarray:
  if np.isnan(thresh):
    return np.full(arr.shape[0] - 1, np.nan, dtype=np.float64)
  above = np.where(np.isnan(arr), np.nan, (arr >= thresh).astype(np.float64))
  return np.maximum(0.0, np.diff(above))


def _tp_fraction_in_window(
    a_cross: np.ndarray,
    b_cross: np.ndarray,
    discard_nans_in_window: bool,
    win: int,
) -> float:
  a_idxs = np.where((a_cross != 0) & ~np.isnan(a_cross))[0]
  if a_idxs.size == 0:
    return np.nan
  b_idxs = np.where((b_cross != 0) & ~np.isnan(b_cross))[0]
  b_nan_idxs = (
      np.where(np.isnan(b_cross))[0] if discard_nans_in_window else np.empty(0, dtype=np.int64)
  )

  total_count = 0
  true_positives = 0
  for a_idx in a_idxs:
    if (
        discard_nans_in_window
        and b_nan_idxs.size > 0
        and np.any(np.abs(b_nan_idxs - a_idx) <= win + 1e-6)
    ):
      continue
    total_count += 1
    if b_idxs.size > 0 and np.any(np.abs(b_idxs - a_idx) <= win + 1e-6):
      true_positives += 1

  return (true_positives / total_count) if total_count > 0 else np.nan


def _calc_continuous_14(obs: np.ndarray, sim: np.ndarray) -> dict[str, float]:
  valid = np.isfinite(obs) & np.isfinite(sim)
  if int(valid.sum()) <= 365 * 2:
    return {m: np.nan for m in CONTINUOUS_METRIC_NAMES}

  o = obs[valid]
  s = sim[valid]
  o_mean = float(np.mean(o))
  s_mean = float(np.mean(s))
  o_std = float(np.std(o))
  s_std = float(np.std(s))

  diff_sq = (s - o) ** 2
  mse_val = float(np.mean(diff_sq))
  rmse_val = float(np.sqrt(mse_val))
  denom = float(np.sum((o - o_mean) ** 2))
  nse_val = float(1.0 - np.sum(diff_sq) / denom) if denom > 0 else np.nan
  alpha_nse_val = float(s_std / o_std) if o_std > 0 else np.nan
  beta_nse_val = float((s_mean - o_mean) / o_std) if o_std > 0 else np.nan
  beta_kge_val = float(s_mean / o_mean) if o_mean != 0 else np.nan

  if o.size >= 2 and o_std > 0 and s_std > 0:
    r_val = float(np.corrcoef(o, s)[0, 1])
  else:
    r_val = np.nan

  if np.isfinite(r_val) and np.isfinite(alpha_nse_val) and np.isfinite(beta_kge_val):
    kge_val = float(
        1.0
        - np.sqrt((r_val - 1.0) ** 2 + (alpha_nse_val - 1.0) ** 2 + (beta_kge_val - 1.0) ** 2)
    )
  else:
    kge_val = np.nan

  o_log = np.log10(np.maximum(1e-4, o))
  s_log = np.log10(np.maximum(1e-4, s))
  o_log_mean = float(np.mean(o_log))
  s_log_mean = float(np.mean(s_log))
  o_log_std = float(np.std(o_log))
  s_log_std = float(np.std(s_log))
  denom_log = float(np.sum((o_log - o_log_mean) ** 2))
  log_nse_val = float(1.0 - np.sum((s_log - o_log) ** 2) / denom_log) if denom_log > 0 else np.nan
  if o_log.size >= 2 and o_log_std > 0 and s_log_std > 0 and o_log_mean != 0:
    r_log = float(np.corrcoef(o_log, s_log)[0, 1])
    a_log = s_log_std / o_log_std
    b_log = s_log_mean / o_log_mean
    log_kge_val = float(
        1.0 - np.sqrt((r_log - 1.0) ** 2 + (a_log - 1.0) ** 2 + (b_log - 1.0) ** 2)
    )
  else:
    log_kge_val = np.nan

  o_fdc = np.sort(o)[::-1].copy()
  s_fdc = np.sort(s)[::-1].copy()
  n_pts = len(o_fdc)

  n_h = int(np.round(0.02 * n_pts))
  if n_h > 0 and np.sum(o_fdc[:n_h]) != 0:
    fhv_val = float(np.sum(s_fdc[:n_h] - o_fdc[:n_h]) / np.sum(o_fdc[:n_h]) * 100.0)
  else:
    fhv_val = np.nan

  s_fdc_pos = np.where(s_fdc <= 0, 1e-6, s_fdc)
  o_fdc_pos = np.where(o_fdc == 0, 1e-6, o_fdc)

  idx_low = int(np.round(0.2 * n_pts))
  idx_up = int(np.round(0.7 * n_pts))
  if idx_up < n_pts:
    qsm_lower = np.log(s_fdc_pos[idx_low])
    qsm_upper = np.log(s_fdc_pos[idx_up])
    qom_lower = np.log(o_fdc_pos[idx_low])
    qom_upper = np.log(o_fdc_pos[idx_up])
    fms_val = float(
        ((qsm_lower - qsm_upper) - (qom_lower - qom_upper))
        / (qom_lower - qom_upper + 1e-6)
        * 100.0
    )
  else:
    fms_val = np.nan

  n_l = int(np.round(0.3 * n_pts))
  if n_l > 0:
    o_low = np.log(o_fdc_pos[-n_l:])
    s_low = np.log(s_fdc_pos[-n_l:])
    qsl = np.sum(s_low - np.min(s_low))
    qol = np.sum(o_low - np.min(o_low))
    flv_val = float(-1.0 * (qsl - qol) / (qol + 1e-6) * 100.0)
  else:
    flv_val = np.nan

  peaks, _ = signal.find_peaks(o, distance=10, prominence=o_std)
  if peaks.size > 0:
    peak_mape_val = float(np.sum(np.abs((s[peaks] - o[peaks]) / o[peaks])) / peaks.size * 100.0)
  else:
    peak_mape_val = np.nan

  return {
      'NSE': nse_val,
      'KGE': kge_val,
      'log-NSE': log_nse_val,
      'log-KGE': log_kge_val,
      'Beta-NSE': beta_nse_val,
      'Beta-KGE': beta_kge_val,
      'Alpha-NSE': alpha_nse_val,
      'pearson-r': r_val,
      'MSE': mse_val,
      'RMSE': rmse_val,
      'FHV': fhv_val,
      'FLV': flv_val,
      'FMS': fms_val,
      'Peak-MAPE': peak_mape_val,
  }


def _process_basin(b_idx: int) -> tuple[int, dict[str, dict[str, np.ndarray]]]:
  assert _G_OBS is not None
  assert _G_GAUGED is not None
  assert _G_RANDOM is not None
  assert _G_BASIN is not None
  assert _G_DATES is not None

  out: dict[str, dict[str, np.ndarray]] = {}
  for exp_name in ('gauged', 'random', 'separated'):
    exp_dict: dict[str, np.ndarray] = {}
    for yr_str in RETURN_PERIOD_STRS:
      for win_str in WINDOW_STRS:
        for score in ('precision', 'recall', 'f1'):
          exp_dict[f'{yr_str}-year-{win_str}-window-{score}'] = np.full(
              8, np.nan, dtype=np.float64
          )
    for m in CONTINUOUS_METRIC_NAMES:
      exp_dict[m] = np.full(8, np.nan, dtype=np.float64)
    out[exp_name] = exp_dict

  obs = _G_OBS[b_idx].astype(np.float64)
  if not np.any(np.isfinite(obs)):
    return b_idx, out

  freq_dt = pd.Timedelta('1D')
  try:
    obs_calc = rp_calc_mod.ReturnPeriodCalculator(
        hydrograph_series=pd.Series(obs, index=_G_DATES),
        hydrograph_series_frequency=freq_dt,
        use_simple_fitting=True,
        verbose=False,
    )
    obs_thresholds = obs_calc.flow_values_from_return_periods(RETURN_PERIOD_VALS)
    obs_cross_by_rp = [_get_crossovers(obs, float(th)) for th in obs_thresholds]
  except Exception:
    obs_calc = None
    obs_cross_by_rp = None

  exp_arrays = [
      ('gauged', _G_GAUGED[b_idx].astype(np.float64)),
      ('random', _G_RANDOM[b_idx].astype(np.float64)),
      ('separated', _G_BASIN[b_idx].astype(np.float64)),
  ]

  for exp_name, preds_mat in exp_arrays:
    if not np.any(np.isfinite(preds_mat)):
      continue
    exp_res = out[exp_name]
    for lt in LEAD_TIMES:
      sim = preds_mat[:, lt]
      if not np.any(np.isfinite(sim)):
        continue

      if obs_cross_by_rp is not None:
        try:
          sim_calc = rp_calc_mod.ReturnPeriodCalculator(
              hydrograph_series=pd.Series(sim, index=_G_DATES),
              hydrograph_series_frequency=freq_dt,
              use_simple_fitting=True,
              verbose=False,
          )
          sim_thresholds = sim_calc.flow_values_from_return_periods(RETURN_PERIOD_VALS)
        except Exception:
          sim_thresholds = np.full(len(RETURN_PERIOD_VALS), np.nan, dtype=np.float64)

        for rp_i, yr_str in enumerate(RETURN_PERIOD_STRS):
          obs_cross = obs_cross_by_rp[rp_i]
          sim_cross = _get_crossovers(sim, float(sim_thresholds[rp_i]))
          for win_int, win_str in zip(WINDOW_INTS, WINDOW_STRS):
            p_val = _tp_fraction_in_window(sim_cross, obs_cross, True, win_int)
            r_val = _tp_fraction_in_window(obs_cross, sim_cross, False, win_int)
            exp_res[f'{yr_str}-year-{win_str}-window-precision'][lt] = p_val
            exp_res[f'{yr_str}-year-{win_str}-window-recall'][lt] = r_val
            if np.isnan(p_val) or np.isnan(r_val):
              f1_val = np.nan
            elif (p_val + r_val) == 0.0:
              f1_val = 0.0
            else:
              f1_val = 2.0 * (p_val * r_val) / (p_val + r_val)
            exp_res[f'{yr_str}-year-{win_str}-window-f1'][lt] = f1_val

      c_vals = _calc_continuous_14(obs, sim)
      for m_name, m_val in c_vals.items():
        exp_res[m_name][lt] = m_val

  return b_idx, out


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--exp-suffix',
      type=str,
      required=True,
      choices=['_da', '_ft', '_ft_da'],
      help='Experiment suffix to stitch and evaluate (_da, _ft, _ft_da)',
  )
  parser.add_argument('--workers', type=int, default=32)
  parser.add_argument('--allow-partial', action='store_true')
  args = parser.parse_args()

  t0 = time.time()
  sfx = args.exp_suffix
  basins = pd.read_csv(METRICS_DIR / 'gauged' / 'NSE.csv', index_col=0).index.astype(str).tolist()
  n_basins = len(basins)
  dates_1986_2023 = pd.date_range('1986-01-01', '2023-12-31', freq='1D')

  stitched_paths: dict[str, str] = {}
  for base_exp in ('gauged', 'random', 'basin'):
    run_exp = f'{base_exp}{sfx}'
    print(f'[{run_exp}] Stitching 1986-2023 predictions from 100 folds...', flush=True)
    sp, meta = stitch_predictions_for_experiment(
        run_exp_name=run_exp,
        basins=basins,
        dates_1986_2023=dates_1986_2023,
        allow_partial=args.allow_partial,
    )
    stitched_paths[base_exp] = str(sp)
    print(f'[{run_exp}] Stitched in {time.time() - t0:.1f}s: {meta}', flush=True)

  obs_path = str(STITCHED_DIR / 'obs_1986_2023.npy')
  gauged_path = stitched_paths['gauged']
  random_path = stitched_paths['random']
  basin_path = stitched_paths['basin']

  all_tables: dict[str, dict[str, np.ndarray]] = {}
  for exp_name in ('gauged', 'random', 'separated'):
    all_tables[exp_name] = {}
    for yr_str in RETURN_PERIOD_STRS:
      for win_str in WINDOW_STRS:
        for score in ('precision', 'recall', 'f1'):
          all_tables[exp_name][f'{yr_str}-year-{win_str}-window-{score}'] = np.full(
              (n_basins, 8), np.nan, dtype=np.float64
          )
    for m in CONTINUOUS_METRIC_NAMES:
      all_tables[exp_name][m] = np.full((n_basins, 8), np.nan, dtype=np.float64)

  print(
      f'Computing all 68 metrics x 8 lead times x 3 experiments ({sfx}) across {n_basins} basins '
      f'(1986-2023) using {args.workers} workers...',
      flush=True,
  )
  ctx = mp.get_context('fork')
  with ctx.Pool(
      processes=args.workers,
      initializer=_init_worker,
      initargs=(obs_path, gauged_path, random_path, basin_path),
  ) as pool:
    for done_i, (b_idx, b_out) in enumerate(
        pool.imap_unordered(_process_basin, range(n_basins), chunksize=32), 1
    ):
      for exp_name, exp_dict in b_out.items():
        target_tables = all_tables[exp_name]
        for m_name, arr8 in exp_dict.items():
          target_tables[m_name][b_idx, :] = arr8
      if done_i % 1000 == 0 or done_i == n_basins:
        print(f'  Processed {done_i}/{n_basins} basins ({time.time() - t0:.1f}s)', flush=True)

  cols = [str(lt) for lt in LEAD_TIMES]
  for exp_name, exp_tables in all_tables.items():
    out_dir_name = f'{exp_name}{sfx}'
    exp_dir = METRICS_DIR / out_dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    for m_name, mat in sorted(exp_tables.items()):
      df = pd.DataFrame(mat, index=basins, columns=cols)
      df.to_csv(exp_dir / f'{m_name}.csv')

  print(f'\nAll 68 x 3 CSVs ({sfx}) written to {METRICS_DIR} in {time.time() - t0:.1f}s!', flush=True)
  for exp_name in ('gauged', 'random', 'separated'):
    for lt_idx, lt_label in [(0, 'LT0 (t0)'), (1, 'LT1 (t0+1)')]:
      nse_v = all_tables[exp_name]['NSE'][:, lt_idx]
      kge_v = all_tables[exp_name]['KGE'][:, lt_idx]
      r2_v = all_tables[exp_name]['2.0-year-2.0-window-recall'][:, lt_idx]
      r5_v = all_tables[exp_name]['5.0-year-2.0-window-recall'][:, lt_idx]
      r10_v = all_tables[exp_name]['10.0-year-2.0-window-recall'][:, lt_idx]
      r20_v = all_tables[exp_name]['20.0-year-2.0-window-recall'][:, lt_idx]
      print(
          f'  {exp_name + sfx:14s} [{lt_label}] | NSE med={np.nanmedian(nse_v):.4f} | KGE med={np.nanmedian(kge_v):.4f} | '
          f'2y-rec med={np.nanmedian(r2_v):.3f} (mean={np.nanmean(r2_v):.3f}) | '
          f'5y-rec med={np.nanmedian(r5_v):.3f} (mean={np.nanmean(r5_v):.3f}) | '
          f'10y-rec med={np.nanmedian(r10_v):.3f} (mean={np.nanmean(r10_v):.3f}) | '
          f'20y-rec med={np.nanmedian(r20_v):.3f} (mean={np.nanmean(r20_v):.3f})',
          flush=True,
      )


if __name__ == '__main__':
  main()
