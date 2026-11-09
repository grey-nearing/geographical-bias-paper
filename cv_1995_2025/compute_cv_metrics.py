"""Stitch 1986-2023 out-of-sample CV predictions and compute 2016-2023 hydro_metrics CSVs.

For a given experiment (`gauged`, `random`, or `basin`):
1. Loads all (spatial_fold, temporal_fold) `preds.npz` files across the 10 temporal folds
   covering 1986-01-01 .. 2023-12-31 (13,879 days) and all 8,811 public Caravan basins.
2. Saves the stitched out-of-sample prediction tensor `(8811, 13879, 8)` float32 to
   `./data/results/stitched_preds/{experiment}_preds_1986_2023.npy`,
   where each lead time `lt in 0..7` is aligned to its target verification date.
3. Fits observed and simulated return-period distributions (`ReturnPeriodCalculator`,
   `use_simple_fitting=True`) on the full 38-year `1986-01-01 .. 2023-12-31` record.
4. Evaluates all 54 return-period metrics (precision, recall, f1 across return periods
   `[1.01, 2.0, 5.0, 10.0, 20.0, 50.0]` and windows `[0.0, 1.0, 2.0]`) and all 14 continuous
   hydrograph metrics (`NSE`, `KGE`, `log-NSE`, `log-KGE`, `Beta-NSE`, `Beta-KGE`, `Alpha-NSE`,
   `pearson-r`, `MSE`, `RMSE`, `FHV`, `FLV`, `FMS`, `Peak-MAPE`) strictly on `2016-01-01 .. 2023-12-31`
   across all 8 lead times (`0..7`).
5. Writes `{metric}.csv` files to
   `./data/results/hydro_model_metrics/{gauged,random,separated}/`.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from hydro_metrics import metrics as hm_metrics
from hydro_metrics import return_period_metrics as hm_rp_metrics
from hydro_metrics.return_period_calculator import exceptions as rp_exceptions
from hydro_metrics.return_period_calculator import return_period_calculator as rp_calc_mod

warnings.filterwarnings('ignore')

ROOT_DIR = Path('./data')
PRECOMPUTED_DIR = ROOT_DIR / 'data' / 'precomputed'
RUNS_DIR = ROOT_DIR / 'runs'
RESULTS_DIR = ROOT_DIR / 'results'
METRICS_OUT_DIR = RESULTS_DIR / 'hydro_model_metrics'
STITCHED_DIR = RESULTS_DIR / 'stitched_preds'

DIST_START = '1986-01-01'
DIST_END = '2023-12-31'
EVAL_START = '2016-01-01'
EVAL_END = '2023-12-31'
LEAD_TIMES = list(range(8))

RETURN_PERIOD_STRS = ['1.01', '2.0', '5.0', '10.0', '20.0', '50.0']
RETURN_PERIOD_VALS = [1.01, 2, 5, 10, 20, 50]
WINDOW_STRS = ['0.0', '1.0', '2.0']
WINDOW_INTS = [0, 1, 2]

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

CONTINUOUS_METRICS_TO_CALC = [
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

# Map key returned by hydro_metrics.metrics.calculate_metrics -> output CSV stem
CONTINUOUS_KEY_TO_FILENAME = {
    'NSE': 'NSE',
    'KGE': 'KGE',
    'log-NSE': 'log-NSE',
    'log-KGE': 'log-KGE',
    'Beta-NSE': 'Beta-NSE',
    'Beta-KGE': 'Beta-KGE',
    'Alpha-NSE': 'Alpha-NSE',
    'Pearson-r': 'pearson-r',
    'MSE': 'MSE',
    'RMSE': 'RMSE',
    'FHV': 'FHV',
    'FLV': 'FLV',
    'FMS': 'FMS',
    'Peak-MAPE': 'Peak-MAPE',
}

# Worker global arrays
_G_OBS_1986_2023: np.ndarray | None = None
_G_PREDS_1986_2023: np.ndarray | None = None
_G_DATES_1986_2023: pd.DatetimeIndex | None = None
_G_EVAL_SLICE: slice | None = None
_G_DATES_2016_2023: pd.DatetimeIndex | None = None


def _init_worker(
    obs_path: str,
    preds_path: str,
    dates_1986_2023_ns: np.ndarray,
    eval_start_idx: int,
    eval_end_idx: int,
) -> None:
    global _G_OBS_1986_2023, _G_PREDS_1986_2023, _G_DATES_1986_2023, _G_EVAL_SLICE, _G_DATES_2016_2023
    _G_OBS_1986_2023 = np.load(obs_path, mmap_mode='r')
    _G_PREDS_1986_2023 = np.load(preds_path, mmap_mode='r')
    _G_DATES_1986_2023 = pd.to_datetime(dates_1986_2023_ns)
    _G_EVAL_SLICE = slice(eval_start_idx, eval_end_idx)
    _G_DATES_2016_2023 = _G_DATES_1986_2023[_G_EVAL_SLICE]


def _compute_metrics_for_basin(b_idx: int) -> tuple[int, dict[str, np.ndarray]]:
    """Compute all 68 metrics across lead times 0..7 for a single basin index `b_idx`."""
    assert _G_OBS_1986_2023 is not None
    assert _G_PREDS_1986_2023 is not None
    assert _G_DATES_1986_2023 is not None
    assert _G_EVAL_SLICE is not None
    assert _G_DATES_2016_2023 is not None

    # Pre-initialize result dict mapping metric_name -> float64 array of length 8
    res: dict[str, np.ndarray] = {}
    for yr_str in RETURN_PERIOD_STRS:
        for win_str in WINDOW_STRS:
            for score in ('precision', 'recall', 'f1'):
                res[f'{yr_str}-year-{win_str}-window-{score}'] = np.full(8, np.nan, dtype=np.float64)
    for fname in CONTINUOUS_KEY_TO_FILENAME.values():
        res[fname] = np.full(8, np.nan, dtype=np.float64)

    preds_full = _G_PREDS_1986_2023[b_idx].astype(np.float64)  # (N_days, 8)
    if not np.any(np.isfinite(preds_full)):
        return b_idx, res

    obs_full = _G_OBS_1986_2023[b_idx].astype(np.float64)
    obs_eval_vals = obs_full[_G_EVAL_SLICE]
    if not np.any(np.isfinite(obs_eval_vals)):
        return b_idx, res

    obs_series_full = pd.Series(obs_full, index=_G_DATES_1986_2023)
    obs_da_eval = xr.DataArray(
        obs_eval_vals,
        coords={'date': _G_DATES_2016_2023},
        dims=['date'],
    )

    # Fit observed ReturnPeriodCalculator ONCE for this basin over 1986-2023
    freq_dt = pd.Timedelta('1D')
    try:
        obs_rp_calc = rp_calc_mod.ReturnPeriodCalculator(
            hydrograph_series=obs_series_full,
            hydrograph_series_frequency=freq_dt,
            use_simple_fitting=True,
            verbose=False,
        )
    except Exception:
        obs_rp_calc = None

    for lt in LEAD_TIMES:
        sim_full = preds_full[:, lt]
        sim_eval_vals = sim_full[_G_EVAL_SLICE]
        if not np.any(np.isfinite(sim_eval_vals)):
            continue

        # 1. Return Period Metrics (fit on 1986-2023, evaluated on 2016-2023)
        if obs_rp_calc is not None:
            sim_series_full = pd.Series(sim_full, index=_G_DATES_1986_2023)
            try:
                sim_rp_calc = rp_calc_mod.ReturnPeriodCalculator(
                    hydrograph_series=sim_series_full,
                    hydrograph_series_frequency=freq_dt,
                    use_simple_fitting=True,
                    verbose=False,
                )
            except rp_exceptions.NotEnoughDataError:
                sim_rp_calc = None
            except Exception:
                sim_rp_calc = None

            for rp_val, yr_str in zip(RETURN_PERIOD_VALS, RETURN_PERIOD_STRS):
                for win_int, win_str in zip(WINDOW_INTS, WINDOW_STRS):
                    try:
                        m_dict, _ = hm_rp_metrics._single_return_period_performance_metric(
                            observations=obs_eval_vals,
                            simulations=sim_eval_vals,
                            obs_return_period_calculator=obs_rp_calc,
                            sim_return_period_calculator=sim_rp_calc,
                            return_period=rp_val,
                            window_in_timesteps=win_int,
                        )
                        p_val = m_dict.get('precision', np.nan)
                        r_val = m_dict.get('recall', np.nan)
                    except Exception:
                        p_val = np.nan
                        r_val = np.nan
                    res[f'{yr_str}-year-{win_str}-window-precision'][lt] = p_val
                    res[f'{yr_str}-year-{win_str}-window-recall'][lt] = r_val
                    if np.isnan(p_val) or np.isnan(r_val):
                        f1_val = np.nan
                    elif (p_val + r_val) == 0.0:
                        f1_val = 0.0
                    else:
                        f1_val = 2.0 * (p_val * r_val) / (p_val + r_val)
                    res[f'{yr_str}-year-{win_str}-window-f1'][lt] = f1_val

        # 2. Continuous Hydrograph Metrics (evaluated on 2016-2023)
        sim_da_eval = xr.DataArray(
            sim_eval_vals,
            coords={'date': _G_DATES_2016_2023},
            dims=['date'],
        )
        try:
            c_dict = hm_metrics.calculate_metrics(
                obs=obs_da_eval,
                sim=sim_da_eval,
                metrics=CONTINUOUS_METRICS_TO_CALC,
                resolution='1D',
                datetime_coord='date',
            )
            for k, fname in CONTINUOUS_KEY_TO_FILENAME.items():
                res[fname][lt] = c_dict.get(k, np.nan)
        except Exception:
            pass

    return b_idx, res


def stitch_predictions_for_experiment(
    experiment: str,
    basins: list[str],
    dates_1986_2023: pd.DatetimeIndex,
    allow_partial: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Stitch all (sfold, tfold) predictions into `(N_basins, N_days_1986_2023, 8)` float32."""
    STITCHED_DIR.mkdir(parents=True, exist_ok=True)
    n_basins = len(basins)
    n_days = len(dates_1986_2023)
    basin_to_idx = {b: i for i, b in enumerate(basins)}
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates_1986_2023)}

    stitched_path = STITCHED_DIR / f'{experiment}_preds_1986_2023.npy'
    stitched = np.lib.format.open_memmap(
        stitched_path,
        mode='w+',
        dtype=np.float32,
        shape=(n_basins, n_days, 8),
    )
    stitched[:] = np.nan

    n_sfolds = 1 if experiment == 'gauged' else 10
    loaded_files = 0
    missing_files = []

    for sfold in range(n_sfolds):
        for tf in TEMPORAL_FOLDS:
            tfold = tf['tfold']
            tfold_name = tf['name']
            npz_path = (
                RUNS_DIR
                / experiment
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
            # Align each lead time lt (0..7) by its target verification date:
            # In Multimet (lead_time=7, min_lead_time=1, predict_last_n=8), column `lt`
            # issued on `test_dates` predicts streamflow on `test_dates + (lt - 1) days`.
            d0 = dates_1986_2023[0]
            d_max = dates_1986_2023[-1]
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
            f'Missing {len(missing_files)} prediction files for {experiment}: {missing_files[:5]}'
        )

    valid_cells = int(np.isfinite(stitched[:, :, 0]).sum())
    total_cells = n_basins * n_days
    meta = {
        'experiment': experiment,
        'loaded_files': loaded_files,
        'expected_files': n_sfolds * len(TEMPORAL_FOLDS),
        'n_basins': n_basins,
        'n_days': n_days,
        'valid_basin_day_cells_lt0': valid_cells,
        'coverage_fraction': round(valid_cells / max(total_cells, 1), 6),
    }
    with open(STITCHED_DIR / f'{experiment}_preds_1986_2023_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    return stitched_path, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        choices=['gauged', 'random', 'basin'],
    )
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument(
        '--allow-partial',
        action='store_true',
        help='Allow computing metrics when only a subset of folds exist (for smoke testing)',
    )
    parser.add_argument(
        '--output-subdir',
        type=str,
        default='',
        help='Optional override output directory name inside results/hydro_model_metrics/',
    )
    args = parser.parse_args()

    t0 = time.time()
    basins = (PRECOMPUTED_DIR / 'basins.txt').read_text().splitlines()
    basins = [b.strip() for b in basins if b.strip()]
    all_dates = pd.to_datetime(np.load(PRECOMPUTED_DIR / 'dates.npy'))

    dist_mask = (all_dates >= pd.Timestamp(DIST_START)) & (all_dates <= pd.Timestamp(DIST_END))
    dist_indices = np.where(dist_mask)[0]
    d_start, d_end = int(dist_indices[0]), int(dist_indices[-1]) + 1
    dates_1986_2023 = all_dates[d_start:d_end]

    eval_mask = (dates_1986_2023 >= pd.Timestamp(EVAL_START)) & (
        dates_1986_2023 <= pd.Timestamp(EVAL_END)
    )
    eval_indices = np.where(eval_mask)[0]
    eval_start_idx, eval_end_idx = int(eval_indices[0]), int(eval_indices[-1]) + 1

    # Save 1986-2023 observed streamflow slice once for worker mmap
    STITCHED_DIR.mkdir(parents=True, exist_ok=True)
    obs_1986_2023_path = STITCHED_DIR / 'obs_1986_2023.npy'
    if not obs_1986_2023_path.exists():
        streamflow_full = np.load(PRECOMPUTED_DIR / 'streamflow.npy', mmap_mode='r')
        obs_slice = np.ascontiguousarray(streamflow_full[:, d_start:d_end], dtype=np.float32)
        np.save(obs_1986_2023_path, obs_slice)

    print(f'[{args.experiment}] Stitching 1986-2023 out-of-sample predictions...', flush=True)
    stitched_path, stitch_meta = stitch_predictions_for_experiment(
        experiment=args.experiment,
        basins=basins,
        dates_1986_2023=dates_1986_2023,
        allow_partial=args.allow_partial,
    )
    print(f'[{args.experiment}] Stitched metadata: {stitch_meta}', flush=True)

    # Allocate master dictionary of (N_basins, 8) arrays for all 68 metrics
    n_basins = len(basins)
    all_metric_tables: dict[str, np.ndarray] = {}
    for yr_str in RETURN_PERIOD_STRS:
        for win_str in WINDOW_STRS:
            for score in ('precision', 'recall', 'f1'):
                all_metric_tables[f'{yr_str}-year-{win_str}-window-{score}'] = np.full(
                    (n_basins, 8), np.nan, dtype=np.float64
                )
    for fname in CONTINUOUS_KEY_TO_FILENAME.values():
        all_metric_tables[fname] = np.full((n_basins, 8), np.nan, dtype=np.float64)

    print(
        f'[{args.experiment}] Computing 68 metrics across {n_basins} basins x 8 lead times '
        f'(RP fit: {DIST_START}..{DIST_END}, skill eval: {EVAL_START}..{EVAL_END}) using {args.workers} workers...',
        flush=True,
    )
    t_calc0 = time.time()
    ctx = mp.get_context('fork')
    with ctx.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(
            str(obs_1986_2023_path),
            str(stitched_path),
            dates_1986_2023.values,
            eval_start_idx,
            eval_end_idx,
        ),
    ) as pool:
        for done_i, (b_idx, b_res) in enumerate(
            pool.imap_unordered(_compute_metrics_for_basin, range(n_basins), chunksize=32), 1
        ):
            for m_name, arr8 in b_res.items():
                all_metric_tables[m_name][b_idx, :] = arr8
            if done_i % 1000 == 0 or done_i == n_basins:
                print(
                    f'[{args.experiment}] Processed {done_i}/{n_basins} basins ({time.time() - t_calc0:.1f}s)',
                    flush=True,
                )

    if args.output_subdir:
        out_name = args.output_subdir
    else:
        out_name = 'separated' if args.experiment == 'basin' else args.experiment

    exp_out_dir = METRICS_OUT_DIR / out_name
    exp_out_dir.mkdir(parents=True, exist_ok=True)

    cols = [str(lt) for lt in LEAD_TIMES]
    for m_name, mat in sorted(all_metric_tables.items()):
        df = pd.DataFrame(mat, index=basins, columns=cols)
        df.to_csv(exp_out_dir / f'{m_name}.csv')

    # If experiment is 'basin' and we wrote to 'separated', also symlink 'basin' -> 'separated'
    if args.experiment == 'basin' and out_name == 'separated':
        symlink_path = METRICS_OUT_DIR / 'basin'
        if not symlink_path.exists():
            try:
                symlink_path.symlink_to('separated')
            except Exception:
                pass

    nse0 = all_metric_tables['NSE'][:, 0]
    kge0 = all_metric_tables['KGE'][:, 0]
    rp5_rec0 = all_metric_tables['5.0-year-2.0-window-recall'][:, 0]
    rp5_prec0 = all_metric_tables['5.0-year-2.0-window-precision'][:, 0]
    print(
        f'[{args.experiment}] Saved 68 metric CSVs to {exp_out_dir} in {time.time() - t0:.1f}s.\n'
        f'  Lead-time 0 summary (2016-2023 eval):\n'
        f'    NSE: median={float(np.nanmedian(nse0)):.4f} (n={int(np.isfinite(nse0).sum())})\n'
        f'    KGE: median={float(np.nanmedian(kge0)):.4f} (n={int(np.isfinite(kge0).sum())})\n'
        f'    5yr-2d-recall:    mean={float(np.nanmean(rp5_rec0)):.4f} (n={int(np.isfinite(rp5_rec0).sum())})\n'
        f'    5yr-2d-precision: mean={float(np.nanmean(rp5_prec0)):.4f} (n={int(np.isfinite(rp5_prec0).sum())})',
        flush=True,
    )


if __name__ == '__main__':
    main()
