#!/usr/bin/env python3
"""Stage A (Round 2): Compute all 14 continuous hydrograph metrics strictly over
2016-01-01..2023-12-31 (2,922 days of operational HRES/GraphCast forecasts, zero pre-2016
ERA5-Land hindsight) across all 12 experiments (gauged/random/separated x base/_da/_ft/_ft_da)
and all 8 lead times (0..7), using exact hydro_metrics/metrics.py definitions
(minimum_data_points = 730 valid days). Also copies the 54 Option B 7-day causal
return-period CSVs (1986-2023) into round2/hydro_model_metrics/.
"""

import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

CANON_DIR = Path(os.environ.get('GEO_BIAS_DATA_DIR', './data'))
STITCHED_DIR = CANON_DIR / 'model-runs' / 'stitched_preds'
ORIG_METRICS_DIR = CANON_DIR / 'hydro_model_metrics'
OUT_METRICS_DIR = CANON_DIR / 'round2' / 'hydro_model_metrics'

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

CONT_METRICS = [
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

MIN_DAYS = 730  # exact match to metrics.py: obs.shape[0] > 365 * 2
DATES = pd.date_range('1986-01-01', '2023-12-31', freq='1D')
IDX_2016 = int(np.where(DATES == pd.Timestamp('2016-01-01'))[0][0])

_G_OBS = None
_G_PRED = None


def _init_worker(obs_path: str, pred_path: str):
  global _G_OBS, _G_PRED
  _G_OBS = np.load(obs_path, mmap_mode='r')
  _G_PRED = np.load(pred_path, mmap_mode='r')


def _compute_metrics_1d(obs: np.ndarray, sim: np.ndarray) -> np.ndarray:
  """Exact vectorized/numpy translation of hydro_metrics/metrics.py for 1 basin, 1 lead."""
  out = np.full(len(CONT_METRICS), np.nan, dtype=np.float64)
  mask = np.isfinite(obs) & np.isfinite(sim)
  n = int(mask.sum())
  if n <= MIN_DAYS:
    return out
  o = obs[mask].astype(np.float64)
  s = sim[mask].astype(np.float64)

  o_mean = o.mean()
  s_mean = s.mean()
  o_std = o.std()
  s_std = s.std()

  diff = s - o
  mse_v = float(np.mean(diff * diff))
  rmse_v = float(np.sqrt(mse_v))

  denom_nse = float(np.sum((o - o_mean) ** 2))
  num_nse = float(np.sum(diff * diff))
  nse_v = 1.0 - num_nse / denom_nse if denom_nse > 0 else np.nan

  alpha_v = float(s_std / o_std) if o_std > 0 else np.nan
  beta_nse_v = float((s_mean - o_mean) / o_std) if o_std > 0 else np.nan
  beta_kge_v = float(s_mean / o_mean) if o_mean != 0 else np.nan

  if o_std > 0 and s_std > 0:
    r_v = float(np.mean((o - o_mean) * (s - s_mean)) / (o_std * s_std))
  else:
    r_v = np.nan

  if np.isfinite(r_v) and np.isfinite(alpha_v) and np.isfinite(beta_kge_v):
    kge_v = float(1.0 - np.sqrt((r_v - 1.0) ** 2 + (alpha_v - 1.0) ** 2 + (beta_kge_v - 1.0) ** 2))
  else:
    kge_v = np.nan

  # Log metrics: exact match to metrics.py lines 166-167, 242-243
  lo = np.log10(np.maximum(1e-4, o))
  ls = np.log10(np.maximum(1e-4, s))
  lo_mean = lo.mean()
  ls_mean = ls.mean()
  lo_std = lo.std()
  ls_std = ls.std()
  denom_lnse = float(np.sum((lo - lo_mean) ** 2))
  num_lnse = float(np.sum((ls - lo) ** 2))
  lnse_v = 1.0 - num_lnse / denom_lnse if denom_lnse > 0 else np.nan
  if lo_std > 0 and ls_std > 0 and lo_mean != 0:
    lr = float(np.mean((lo - lo_mean) * (ls - ls_mean)) / (lo_std * ls_std))
    la = float(ls_std / lo_std)
    lb = float(ls_mean / lo_mean)
    lkge_v = float(1.0 - np.sqrt((lr - 1.0) ** 2 + (la - 1.0) ** 2 + (lb - 1.0) ** 2))
  else:
    lkge_v = np.nan

  # FDC signatures (exact match to metrics.py lines 365-451)
  o_fdc = np.sort(o)[::-1]
  s_fdc = np.sort(s)[::-1]

  # FHV (top 2%)
  n_h = int(np.round(0.02 * n))
  if n_h > 0:
    oh = o_fdc[:n_h]
    sh = s_fdc[:n_h]
    sum_oh = float(np.sum(oh))
    fhv_v = float(np.sum(sh - oh) / sum_oh * 100.0) if sum_oh != 0 else np.nan
  else:
    fhv_v = np.nan

  # FMS (20%-70%)
  o_fms = o_fdc.copy()
  s_fms = s_fdc.copy()
  s_fms[s_fms <= 0] = 1e-6
  o_fms[o_fms == 0] = 1e-6
  i_lo = min(int(np.round(0.2 * n)), n - 1)
  i_hi = min(int(np.round(0.7 * n)), n - 1)
  qsm_lo = np.log(s_fms[i_lo])
  qsm_hi = np.log(s_fms[i_hi])
  qom_lo = np.log(o_fms[i_lo])
  qom_hi = np.log(o_fms[i_hi])
  fms_v = float(((qsm_lo - qsm_hi) - (qom_lo - qom_hi)) / (qom_lo - qom_hi + 1e-6) * 100.0)

  # FLV (bottom 30%)
  n_l = int(np.round(0.3 * n))
  if n_l > 0:
    ol = np.log(o_fms[-n_l:])
    sl = np.log(s_fms[-n_l:])
    qsl = float(np.sum(sl - sl.min()))
    qol = float(np.sum(ol - ol.min()))
    flv_v = float(-1.0 * (qsl - qol) / (qol + 1e-6) * 100.0)
  else:
    flv_v = np.nan

  # Peak-MAPE (exact match to metrics.py lines 454-471)
  if o_std > 0:
    peaks, _ = signal.find_peaks(o, distance=10, prominence=o_std)
    if peaks.size > 0:
      op = o[peaks]
      sp = s[peaks]
      if np.all(op != 0):
        pmape_v = float(np.sum(np.abs((sp - op) / op)) / peaks.size * 100.0)
      else:
        pmape_v = np.nan
    else:
      pmape_v = np.nan
  else:
    pmape_v = np.nan

  out[:] = [
      nse_v, kge_v, lnse_v, lkge_v, beta_nse_v, beta_kge_v, alpha_v,
      r_v, mse_v, rmse_v, fhv_v, flv_v, fms_v, pmape_v,
  ]
  return out


def _eval_chunk(b_indices):
  res = np.full((len(b_indices), 8, len(CONT_METRICS)), np.nan, dtype=np.float64)
  for i_local, b_idx in enumerate(b_indices):
    obs = _G_OBS[b_idx, IDX_2016:]
    sim_all = _G_PRED[b_idx, IDX_2016:, :]
    for lt in range(8):
      res[i_local, lt, :] = _compute_metrics_1d(obs, sim_all[:, lt])
  return b_indices, res


def main():
  t0 = time.time()
  OUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
  grdc_8811 = pd.read_csv(CANON_DIR / 'attributes' / 'grdc_attributes.csv', index_col=0)
  basins = [str(b) for b in grdc_8811.index]
  n_basins = len(basins)
  lt_cols = [str(lt) for lt in range(8)]
  obs_path = str(STITCHED_DIR / 'obs_1986_2023.npy')
  chunks = [np.arange(i, min(i + 160, n_basins)) for i in range(0, n_basins, 160)]

  print(f'Stage A: Computing 14 continuous metrics strictly over 2016-2023 (idx {IDX_2016}..{len(DATES)}, {len(DATES)-IDX_2016} days, min_valid>{MIN_DAYS})...', flush=True)

  summary_rows = []
  for st_name, dir_name in EXP_PAIRS:
    t_exp = time.time()
    out_dir = OUT_METRICS_DIR / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy the 54 Option B return-period CSVs from canonical hydro_model_metrics
    orig_dir = ORIG_METRICS_DIR / dir_name
    n_copied = 0
    for csv_p in sorted(orig_dir.glob('*-year-*-window-*.csv')):
      dst_p = out_dir / csv_p.name
      shutil.copy2(csv_p, dst_p)
      n_copied += 1

    # 2. Compute 14 continuous metrics strictly over 2016-2023
    pred_path = str(STITCHED_DIR / f'{st_name}_preds_1986_2023.npy')
    mat = np.full((n_basins, 8, len(CONT_METRICS)), np.nan, dtype=np.float64)
    with ProcessPoolExecutor(
        max_workers=56,
        initializer=_init_worker,
        initargs=(obs_path, pred_path),
    ) as ex:
      for b_idxs, sub in ex.map(_eval_chunk, chunks):
        mat[b_idxs] = sub

    for m_i, m_name in enumerate(CONT_METRICS):
      df_m = pd.DataFrame(mat[:, :, m_i], index=basins, columns=lt_cols)
      df_m.to_csv(out_dir / f'{m_name}.csv')

    # Print LT=1 medians for quick verification
    meds = {m: float(np.nanmedian(mat[:, 1, i])) for i, m in enumerate(CONT_METRICS)}
    n_valid = int(np.sum(np.isfinite(mat[:, 1, 0])))
    print(
        f"  [{dir_name:14s}] copied {n_copied} RP CSVs + 14 cont CSVs in {time.time()-t_exp:.1f}s | "
        f"N={n_valid} | NSE={meds['NSE']:.3f} KGE={meds['KGE']:.3f} r={meds['pearson-r']:.3f} "
        f"Alpha={meds['Alpha-NSE']:.3f} BetaKGE={meds['Beta-KGE']:.3f} BetaNSE={meds['Beta-NSE']:.3f}",
        flush=True,
    )
    row = {'split': dir_name, 'n_valid_lt1': n_valid}
    row.update(meds)
    summary_rows.append(row)

  df_sum = pd.DataFrame(summary_rows)
  df_sum.to_csv(CANON_DIR / 'round2' / 'stage_a_continuous_2016_2023_summary.csv', index=False)

  # Print exact quantiles of Beta-KGE, Alpha-NSE, Beta-NSE across all 3 splits (base) to check Winsorization bounds
  print('\nQuantiles of ratio/bias metrics on 2016-2023 (Lead 1, Base 3-split pool):', flush=True)
  for m in ['NSE', 'KGE', 'log-NSE', 'log-KGE', 'Beta-NSE', 'Beta-KGE', 'Alpha-NSE', 'pearson-r']:
    vals = []
    for sp in ['gauged', 'random', 'separated']:
      s = pd.read_csv(OUT_METRICS_DIR / sp / f'{m}.csv', index_col=0)['1'].replace([np.inf, -np.inf], np.nan).dropna()
      vals.append(s.values)
    v = np.concatenate(vals)
    qs = np.percentile(v, [0.5, 1, 5, 25, 50, 75, 95, 99, 99.5])
    pct_gt1 = float((v > 1.0).mean() * 100.0)
    print(
        f'  {m:10s} (N={len(v):5d}, >1.0: {pct_gt1:5.1f}%): '
        f'p0.5={qs[0]:7.3f} p1={qs[1]:7.3f} p5={qs[2]:6.3f} med={qs[4]:6.3f} '
        f'p95={qs[6]:6.3f} p99={qs[7]:7.3f} p99.5={qs[8]:7.3f}',
        flush=True,
    )

  print(f'\nStage A complete in {time.time() - t0:.1f}s!', flush=True)


if __name__ == '__main__':
  main()
