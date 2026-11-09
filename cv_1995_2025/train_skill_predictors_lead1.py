#!/usr/bin/env python3
"""Trains Lead-Time-1 SkillPredictor meta-models for a given exp_suffix ('', '_da', '_ft', '_ft_da').

Each meta-model family (`M_base`, `M_DA`, `M_FT`, `M_FT_DA`) is trained on the unified
26,433-row `[gauged, random, separated]` bucket (`model_type='ungauged'`) as well as
the standalone `gauged` confirmation split (`model_type='gauged'`).
10-fold CV splits by unique basin ID (`unique_gauge_id`) so all 3 rows (`gauged`,
`random`, `separated`) of any basin are always in the same CV fold.
"""

import os

# Must be set BEFORE importing numpy/scipy/sklearn/lightgbm/xgboost/torch!
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '4')
os.environ['GEO_BIAS_DATA_DIR'] = './data'
os.environ['GEO_BIAS_THIRD_PARTY_DIR'] = './data'
os.environ['GEO_BIAS_OUTPUT_DIR'] = './data'

import argparse  # noqa: E402
import calendar  # noqa: F401, E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import glob  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
import warnings  # noqa: E402

sys.path.insert(0, './data')
import local_pkg  # noqa: E402

gfile_mod = types.ModuleType('local_pkg.pyglib.gfile')
gfile_mod.Open = open
gfile_mod.Exists = os.path.exists
gfile_mod.MakeDirs = lambda p: os.makedirs(p, exist_ok=True)
gfile_mod.Glob = glob.glob
gfile_mod.ListDir = os.listdir
pyglib_mod = types.ModuleType('local_pkg.pyglib')
pyglib_mod.gfile = gfile_mod
local_pkg.pyglib = pyglib_mod
sys.modules['local_pkg.pyglib'] = pyglib_mod
sys.modules['local_pkg.pyglib.gfile'] = gfile_mod

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from local_pkg.intelligence.gsnearing.geographical_bias_paper.utils import file_paths, io_utils  # noqa: E402
from local_pkg.intelligence.gsnearing.geographical_bias_paper.utils.skill_predictor import SkillPredictor  # noqa: E402

warnings.filterwarnings('ignore')

PRIORITY_METRICS = [
    '5.0-year-2.0-window-recall',
    '10.0-year-2.0-window-recall',
    '20.0-year-2.0-window-recall',
    '2.0-year-2.0-window-recall',
    '1.01-year-2.0-window-recall',
    '5.0-year-2.0-window-precision',
    '10.0-year-2.0-window-precision',
    '20.0-year-2.0-window-precision',
    '2.0-year-2.0-window-precision',
    '1.01-year-2.0-window-precision',
    'NSE',
    'KGE',
    'log-NSE',
    'log-KGE',
    'Beta-NSE',
    'Beta-KGE',
    'Alpha-NSE',
    'pearson-r',
]
REMAINING_METRICS = [m for m in file_paths.ALL_METRICS if m not in PRIORITY_METRICS]
ORDERED_METRICS = PRIORITY_METRICS + REMAINING_METRICS


def _fit_without_nested_threading(m: SkillPredictor, x: pd.DataFrame, y: pd.Series, n_threads: int = 4) -> None:
    impl = m.impl
    if hasattr(impl, 'ensemble'):
        impl.ensemble.n_jobs = 1
        for _, est in impl.ensemble.estimators:
            if hasattr(est, 'n_jobs'):
                est.n_jobs = n_threads
            elif hasattr(est, 'named_steps'):
                for step in est.named_steps.values():
                    if hasattr(step, 'n_jobs'):
                        step.n_jobs = n_threads
        X_trans = impl._preprocess_features(x)
        impl.feature_columns = X_trans.columns
        impl.ensemble.fit(X_trans, y)
    else:
        impl.model.n_jobs = 1
        for _, est in impl.model.estimators:
            if hasattr(est, 'n_jobs'):
                est.n_jobs = n_threads
        X_trans = impl._engineer_features(x)
        impl.feature_names = X_trans.columns.tolist()
        y_clipped = np.clip(y, -2.0, 1.0)
        impl.model.fit(X_trans, y_clipped)


def _spawn_worker(args_tuple):
    """Run CV fold `task_id` (0..9) or full model fit (`task_id == -1`) in a clean spawned process."""
    task_id, metric, x, y, unique_ids, id_splits = args_tuple
    m = SkillPredictor(metric)
    if task_id == -1:
        _fit_without_nested_threading(m, x, y, n_threads=4)
        return (-1, None, None, m)
    tr_id_idx, te_id_idx = id_splits[task_id]
    tr_ids = set(unique_ids[tr_id_idx])
    te_mask = ~x.index.isin(tr_ids)
    tr_mask = ~te_mask
    x_tr, x_te = x.loc[tr_mask], x.loc[te_mask]
    y_tr, y_te = y.loc[tr_mask], y.loc[te_mask]
    _fit_without_nested_threading(m, x_tr, y_tr, n_threads=4)
    p = m.predict(x_te)
    return (task_id, np.where(te_mask)[0], y_te.values, p.values)


def run_cv_and_global_for_metric_model(
    metric: str,
    model_type: str,
    df_hybas: pd.DataFrame,
    pool: ProcessPoolExecutor,
    exp_suffix: str = '',
    lead_time: str = '1',
    cv_folds: int = 10,
    skip_existing: bool = True,
) -> None:
    t0 = time.time()
    file_paths.init(metric)
    file_paths.ensure_directories_exist()

    tag = f'{model_type}{exp_suffix}'
    cv_save_path = os.path.join(
        file_paths.CV_PREDS_DIR, f'{tag}_regression_cv_preds_{metric}.csv'
    )
    model_save_path = os.path.join(
        file_paths.MODELS_DIR, f'final_{tag}_model_{metric}.pkl'
    )
    hybas_preds_save_path = os.path.join(
        file_paths.HYBAS_PREDS_DIR, f'global_hybas_predictions_{tag}_{metric}.csv'
    )
    pred_col_name = f'{model_type}_skill'

    if (
        skip_existing
        and os.path.exists(cv_save_path)
        and os.path.exists(model_save_path)
        and os.path.exists(hybas_preds_save_path)
    ):
        gen_mtime = os.path.getmtime(file_paths.UNGAUGED_GENERIC_DATA_PATH)
        if os.path.getmtime(hybas_preds_save_path) >= gen_mtime:
            print(f"[{time.strftime('%H:%M:%S')}] SKIP (already up-to-date) {metric:32s} | {tag:14s}", flush=True)
            return

    x, y = io_utils.load_and_preprocess_data(
        model_type, metric, None, exp_suffix=exp_suffix, lead_time=lead_time
    )

    unique_ids = pd.Index(x.index.unique())
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    id_splits = list(kf.split(unique_ids))

    tasks = [
        (tid, metric, x, y, unique_ids, id_splits)
        for tid in ([-1] + list(range(cv_folds)))
    ]
    results = list(pool.map(_spawn_worker, tasks))

    full_model = None
    all_true_arr = np.empty(len(y), dtype=np.float64)
    all_pred_arr = np.empty(len(y), dtype=np.float64)
    for tid, pos_idx, y_te_vals, p_te_vals_or_mod in results:
        if tid == -1:
            full_model = p_te_vals_or_mod
        else:
            all_true_arr[pos_idx] = y_te_vals
            all_pred_arr[pos_idx] = p_te_vals_or_mod

    cv_df = pd.DataFrame(
        {'true': all_true_arr, 'pred': all_pred_arr}, index=x.index
    )
    cv_df.to_csv(cv_save_path)
    r2 = r2_score(all_true_arr, all_pred_arr)
    mae = mean_absolute_error(all_true_arr, all_pred_arr)

    with open(model_save_path, 'wb') as f:
        pickle.dump(full_model, f)

    x_infer = df_hybas.reindex(columns=x.columns, fill_value=0)
    preds = full_model.predict(x_infer)
    preds.rename(pred_col_name).to_frame().to_csv(hybas_preds_save_path, index=True)

    elapsed = time.time() - t0
    print(
        f"[{time.strftime('%H:%M:%S')}] DONE {metric:32s} | {tag:14s} | "
        f"N={len(x):5d} | CV R2={r2:.3f} (MAE={mae:.3f}) | "
        f"Train med={y.median():.3f} | HyBAS med={preds.median():.3f} (mean={preds.mean():.3f}) | {elapsed:.1f}s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-suffix', type=str, default='', help="'' | '_da' | '_ft' | '_ft_da'")
    parser.add_argument('--lead-time', type=str, default='1')
    parser.add_argument('--no-skip', action='store_true')
    parser.add_argument('--metrics', nargs='*', default=None)
    args = parser.parse_args()

    t_start = time.time()
    file_paths.init('5.0-year-2.0-window-recall')
    print(f"[{time.strftime('%H:%M:%S')}] Loading HyBAS Zarr once into memory...", flush=True)
    df_hybas = io_utils.load_hybas_zarr(file_paths.HYBAS_GENERIC_DATA_ZARR_PATH)
    print(
        f"[{time.strftime('%H:%M:%S')}] Loaded HyBAS Zarr: {df_hybas.shape} in {time.time()-t_start:.1f}s",
        flush=True,
    )

    metrics_to_run = args.metrics if args.metrics else ORDERED_METRICS
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=11, mp_context=ctx) as pool:
        for idx, metric in enumerate(metrics_to_run, 1):
            print(
                f"\n=== [{idx}/{len(metrics_to_run)}] Processing {metric} (suffix='{args.exp_suffix}', lead={args.lead_time}) ===",
                flush=True,
            )
            for model_type in ['ungauged', 'gauged']:
                run_cv_and_global_for_metric_model(
                    metric,
                    model_type,
                    df_hybas,
                    pool=pool,
                    exp_suffix=args.exp_suffix,
                    lead_time=args.lead_time,
                    skip_existing=(not args.no_skip),
                )

    print(
        f"\n[{time.strftime('%H:%M:%S')}] Completed {len(metrics_to_run)} metrics (suffix='{args.exp_suffix}') in {time.time()-t_start:.1f}s!",
        flush=True,
    )


if __name__ == '__main__':
    main()
