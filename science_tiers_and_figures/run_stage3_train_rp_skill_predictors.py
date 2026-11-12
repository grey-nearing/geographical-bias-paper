#!/usr/bin/env python3
"""Stage 3: Retrain Lead-Time-1 SkillPredictor meta-models for all 30 Option B return-period metrics
for a given exp_suffix ('', '_da', '_ft', '_ft_da') or all 4 families.
Continuous metrics (NSE, KGE, log-NSE, log-KGE, Beta-NSE, Beta-KGE, Alpha-NSE, pearson-r) are
unchanged by Option B and remain untouched.
"""

import os

# Keep OpenMP single-threaded per worker to prevent libgomp barrier spinlock under CPU steal,
# while using joblib threading backend for ExtraTreesRegressor inside each worker.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
os.environ['GOMP_SPINCOUNT'] = '0'
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_default_data = os.environ.get('GEO_BIAS_DATA_DIR', './data/canonical_v2')
_default_code = _repo_root
os.environ.setdefault('GEO_BIAS_DATA_DIR', _default_data)
os.environ.setdefault('GEO_BIAS_THIRD_PARTY_DIR', _default_data)
os.environ.setdefault('GEO_BIAS_OUTPUT_DIR', _default_data)

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

import joblib  # noqa: E402

CODE_ROOT = os.environ.get('GEO_BIAS_CODE_ROOT', _default_code)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)
google3 = sys.modules.setdefault('google3', types.ModuleType('google3'))  # noqa: E402

gfile_mod = types.ModuleType('google3.pyglib.gfile')
gfile_mod.Open = open
gfile_mod.Exists = os.path.exists
gfile_mod.MakeDirs = lambda p: os.makedirs(p, exist_ok=True)
gfile_mod.Glob = glob.glob
gfile_mod.ListDir = os.listdir
pyglib_mod = types.ModuleType('google3.pyglib')
pyglib_mod.gfile = gfile_mod
google3.pyglib = pyglib_mod
sys.modules['google3.pyglib'] = pyglib_mod
sys.modules['google3.pyglib.gfile'] = gfile_mod
for _mod_name in ('lightgbm', 'geopandas', 'xgboost'):
    try:
        __import__(_mod_name)
    except ImportError:
        _m = types.ModuleType(_mod_name)
        _m.GeoDataFrame = object
        sys.modules[_mod_name] = _m

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import ExtraTreesRegressor as _ET  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

_orig_et_fit = _ET.fit


def _threaded_et_fit(self, X, y, sample_weight=None):
    nt = getattr(self, 'n_jobs', 1) or 1
    if nt > 1:
        with joblib.parallel_backend('threading', n_jobs=nt):
            return _orig_et_fit(self, X, y, sample_weight=sample_weight)
    return _orig_et_fit(self, X, y, sample_weight=sample_weight)


_ET.fit = _threaded_et_fit

from utils import file_paths, io_utils  # noqa: E402
from utils import skill_predictor as sp_mod  # noqa: E402
from utils.skill_predictor import SkillPredictor  # noqa: E402

warnings.filterwarnings('ignore')


def _fast_cached_preprocessed_features(predictor, X: pd.DataFrame) -> pd.DataFrame:
    if len(X) >= 50000:
        x_arr = X.to_numpy(copy=False)
        for slot in sp_mod._RP_FEAT_CACHE:
            if (
                slot['shape'] == x_arr.shape
                and list(X.columns) == slot['cols']
                and np.array_equal(x_arr, slot['x_arr'], equal_nan=True)
            ):
                return slot['X_transformed']
        X_transformed = predictor._preprocess_features(X)
        sp_mod._RP_FEAT_CACHE.append({
            'shape': x_arr.shape,
            'cols': list(X.columns),
            'x_arr': x_arr.copy(),
            'X_transformed': X_transformed,
        })
        if len(sp_mod._RP_FEAT_CACHE) > 2:
            sp_mod._RP_FEAT_CACHE.pop(0)
        return X_transformed
    return predictor._preprocess_features(X)


sp_mod._get_cached_preprocessed_features = _fast_cached_preprocessed_features


def _fast_predict_knn_cached(knn_pipe, X_aligned: pd.DataFrame) -> np.ndarray:
    X_knn = np.ascontiguousarray(knn_pipe[:-1].transform(X_aligned), dtype=np.float64)
    knn_reg = knn_pipe.named_steps['kneighborsregressor']
    fit_X = np.ascontiguousarray(knn_reg._fit_X, dtype=np.float64)
    fit_y = np.asarray(knn_reg._y, dtype=np.float64).ravel()

    fp = (
        X_knn.shape,
        fit_X.shape,
        float(X_knn[0, 0]),
        float(X_knn[-1, -1]),
        float(fit_X[0, 0]),
        float(fit_X[-1, -1]),
        float(fit_X.sum()),
    )
    nbr_idx = None
    nbr_w = None
    for slot in sp_mod._KNN_NBR_CACHE:
        if slot.get('fp') == fp:
            nbr_idx = slot['nbr_idx']
            nbr_w = slot['nbr_w']
            break

    if nbr_idx is None:
        try:
            import torch
            k_n = int(knn_reg.n_neighbors)
            n_q = len(X_knn)
            dists = np.empty((n_q, k_n), dtype=np.float64)
            nbr_idx = np.empty((n_q, k_n), dtype=np.int64)
            use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
            dev = f"cuda:{os.getpid() % torch.cuda.device_count()}" if use_cuda else "cpu"
            if not use_cuda:
                torch.set_num_threads(64 if n_q >= 100000 else 8)
            t_ref = torch.from_numpy(np.ascontiguousarray(fit_X, dtype=np.float32)).to(dev)
            step_sz = 16384 if use_cuda else 8192
            with torch.no_grad():
                for s in range(0, n_q, step_sz):
                    e = min(n_q, s + step_sz)
                    t_q = torch.from_numpy(np.ascontiguousarray(X_knn[s:e], dtype=np.float32)).to(dev)
                    d_mat = torch.cdist(t_q, t_ref)
                    vals, idx = torch.topk(d_mat, k=k_n, dim=1, largest=False, sorted=True)
                    dists[s:e] = vals.cpu().numpy().astype(np.float64)
                    nbr_idx[s:e] = idx.cpu().numpy()
        except Exception:
            knn_reg.n_jobs = 8
            dists, nbr_idx = knn_reg.kneighbors(X_knn)
        zero_mask = dists <= 1e-15
        any_zero = zero_mask.any(axis=1)
        with np.errstate(divide='ignore'):
            nbr_w = np.where(any_zero[:, None], zero_mask.astype(np.float64), 1.0 / dists)
        nbr_w /= nbr_w.sum(axis=1, keepdims=True)
        if len(X_knn) >= 5000:
            sp_mod._KNN_NBR_CACHE.append({
                'fp': fp,
                'nbr_idx': nbr_idx,
                'nbr_w': nbr_w,
            })
            if len(sp_mod._KNN_NBR_CACHE) > 16:
                sp_mod._KNN_NBR_CACHE.pop(0)

    return np.einsum('ij,ij->i', fit_y[nbr_idx], nbr_w)


sp_mod._predict_knn_cached = _fast_predict_knn_cached

PHASE_3A_RP_METRICS = [
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
]

PHASE_3B_RP_METRICS = [
    m for m in file_paths.ALL_METRICS
    if ('-year-' in m) and (m not in PHASE_3A_RP_METRICS)
]


def _fit_without_nested_threading(m: SkillPredictor, x: pd.DataFrame, y: pd.Series, n_threads: int = 5) -> None:
    impl = m.impl
    if hasattr(impl, 'ensemble'):
        impl.ensemble.n_jobs = 1
        for name, est in impl.ensemble.estimators:
            if hasattr(est, 'n_jobs'):
                est.n_jobs = n_threads if name == 'et' else 1
            elif hasattr(est, 'named_steps'):
                for sname, step in est.named_steps.items():
                    if hasattr(step, 'n_jobs'):
                        step.n_jobs = n_threads if 'extratrees' in sname else 1
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
    task_id, metric, x, y, unique_ids, id_splits, n_threads = args_tuple
    try:
        import torch
        torch.set_num_threads(n_threads)
    except Exception:
        pass
    m = SkillPredictor(metric)
    if task_id == -1:
        _fit_without_nested_threading(m, x, y, n_threads=n_threads)
        return (-1, None, None, m)
    tr_id_idx, te_id_idx = id_splits[task_id]
    tr_ids = set(unique_ids[tr_id_idx])
    te_mask = ~x.index.isin(tr_ids)
    tr_mask = ~te_mask
    x_tr, x_te = x.loc[tr_mask], x.loc[te_mask]
    y_tr, y_te = y.loc[tr_mask], y.loc[te_mask]
    _fit_without_nested_threading(m, x_tr, y_tr, n_threads=n_threads)
    # Predict in chunks < 2000 so worker does not spawn 16 extra threads
    chunks = [m.predict(x_te.iloc[i : i + 1500]) for i in range(0, len(x_te), 1500)]
    p = pd.concat(chunks)
    return (task_id, np.where(te_mask)[0], y_te.values, p.values)


def run_cv_and_global_for_metric_model(
    metric: str,
    model_type: str,
    df_hybas: pd.DataFrame,
    pool: ProcessPoolExecutor,
    exp_suffix: str = '',
    lead_time: str = '1',
    cv_folds: int = 10,
    n_threads: int = 5,
    skip_after_ts: float = 0.0,
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

    cv_fresh = os.path.exists(cv_save_path) and os.path.getmtime(cv_save_path) >= skip_after_ts
    mod_fresh = os.path.exists(model_save_path) and os.path.getmtime(model_save_path) >= skip_after_ts
    hyb_fresh = os.path.exists(hybas_preds_save_path) and os.path.getmtime(hybas_preds_save_path) >= skip_after_ts

    if skip_after_ts > 0 and cv_fresh and hyb_fresh:
        print(f"[{time.strftime('%H:%M:%S')}] SKIP (already computed) {metric:32s} | {tag:14s}", flush=True)
        return

    lock_path = hybas_preds_save_path + '.lock'
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
    except FileExistsError:
        try:
            if time.time() - os.path.getmtime(lock_path) < 1800:
                print(f"[{time.strftime('%H:%M:%S')}] SKIP (locked by another worker) {metric:32s} | {tag:14s}", flush=True)
                return
        except OSError:
            pass

    x, y = io_utils.load_and_preprocess_data(
        model_type, metric, None, exp_suffix=exp_suffix, lead_time=lead_time
    )

    unique_ids = pd.Index(x.index.unique())
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    id_splits = list(kf.split(unique_ids))

    full_model = None
    if mod_fresh:
        try:
            with open(model_save_path, 'rb') as f:
                full_model = pickle.load(f)
        except Exception:
            full_model = None

    tids_to_run = []
    if full_model is None:
        tids_to_run.append(-1)
    if not cv_fresh:
        tids_to_run.extend(range(cv_folds))

    tasks = [
        (tid, metric, x, y, unique_ids, id_splits, n_threads)
        for tid in tids_to_run
    ]
    results = list(pool.map(_spawn_worker, tasks)) if tasks else []

    all_true_arr = np.empty(len(y), dtype=np.float64)
    all_pred_arr = np.empty(len(y), dtype=np.float64)
    for tid, pos_idx, y_te_vals, p_te_vals_or_mod in results:
        if tid == -1:
            full_model = p_te_vals_or_mod
        else:
            all_true_arr[pos_idx] = y_te_vals
            all_pred_arr[pos_idx] = p_te_vals_or_mod

    if not cv_fresh:
        cv_df = pd.DataFrame(
            {'true': all_true_arr, 'pred': all_pred_arr}, index=x.index
        )
        if os.path.exists(cv_save_path):
            os.unlink(cv_save_path)
        cv_df.to_csv(cv_save_path)
    else:
        cv_df = pd.read_csv(cv_save_path).dropna(subset=['true', 'pred'])
        all_true_arr = cv_df['true'].values
        all_pred_arr = cv_df['pred'].values

    r2 = r2_score(all_true_arr, all_pred_arr)
    mae = mean_absolute_error(all_true_arr, all_pred_arr)

    if not mod_fresh and full_model is not None:
        if os.path.exists(model_save_path):
            os.unlink(model_save_path)
        with open(model_save_path, 'wb') as f:
            pickle.dump(full_model, f)

    if not hyb_fresh:
        if os.path.exists(hybas_preds_save_path):
            os.unlink(hybas_preds_save_path)
        x_infer = df_hybas.reindex(columns=x.columns, fill_value=0)
        preds = full_model.predict(x_infer)
        preds.rename(pred_col_name).to_frame().to_csv(hybas_preds_save_path, index=True)
    else:
        preds = pd.read_csv(hybas_preds_save_path, index_col=0).iloc[:, 0]
    try:
        if os.path.exists(lock_path):
            os.unlink(lock_path)
    except OSError:
        pass

    elapsed = time.time() - t0
    print(
        f"[{time.strftime('%H:%M:%S')}] DONE {metric:32s} | {tag:14s} | "
        f"N={len(x):5d} | CV R2={r2:.3f} (MAE={mae:.3f}) | "
        f"Train med={y.median():.3f} | HyBAS med={preds.median():.3f} (mean={preds.mean():.3f}) | {elapsed:.1f}s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-suffix', nargs='*', default=['', '_da', '_ft', '_ft_da'])
    parser.add_argument('--phase', choices=['3a', '3b', 'all', 'custom'], default='all')
    parser.add_argument('--metrics', nargs='*', default=None)
    parser.add_argument('--model-types', nargs='*', default=['ungauged', 'gauged'])
    parser.add_argument('--max-workers', type=int, default=11)
    parser.add_argument('--n-threads', type=int, default=5)
    parser.add_argument('--skip-after-ts', type=float, default=1794328000.0)
    args = parser.parse_args()

    t_start = time.time()
    file_paths.init('5.0-year-2.0-window-recall')
    print(f"[{time.strftime('%H:%M:%S')}] Loading HyBAS Zarr once into memory...", flush=True)
    df_hybas = io_utils.load_hybas_zarr(file_paths.HYBAS_GENERIC_DATA_ZARR_PATH)
    print(
        f"[{time.strftime('%H:%M:%S')}] Loaded HyBAS Zarr: {df_hybas.shape} in {time.time()-t_start:.1f}s",
        flush=True,
    )

    suffixes = args.exp_suffix
    model_types = args.model_types
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=args.max_workers, mp_context=ctx) as pool:
        if args.metrics:
            print(f"\n================ CUSTOM METRICS ({len(args.metrics)} metrics, suffixes={suffixes}, model_types={model_types}) ================", flush=True)
            for sfx in suffixes:
                for metric in args.metrics:
                    for model_type in model_types:
                        run_cv_and_global_for_metric_model(
                            metric,
                            model_type,
                            df_hybas,
                            pool=pool,
                            exp_suffix=sfx,
                            lead_time='1',
                            n_threads=args.n_threads,
                            skip_after_ts=args.skip_after_ts,
                        )
            print(f"\n[{time.strftime('%H:%M:%S')}] CUSTOM METRICS COMPLETE in {time.time()-t_start:.1f}s!", flush=True)
            return

        if args.phase in ('3a', 'all'):
            print(f"\n================ PHASE 3A: 2.0-WINDOW RETURN-PERIOD METRICS (suffixes={suffixes}) ================", flush=True)
            for sfx in suffixes:
                for idx, metric in enumerate(PHASE_3A_RP_METRICS, 1):
                    for model_type in model_types:
                        run_cv_and_global_for_metric_model(
                            metric,
                            model_type,
                            df_hybas,
                            pool=pool,
                            exp_suffix=sfx,
                            lead_time='1',
                            n_threads=args.n_threads,
                            skip_after_ts=args.skip_after_ts,
                        )
            print(f"\n[{time.strftime('%H:%M:%S')}] PHASE 3A COMPLETE for {suffixes} in {time.time()-t_start:.1f}s!", flush=True)

        if args.phase in ('3b', 'all'):
            print(f"\n================ PHASE 3B: 1.0 & 0.0-WINDOW RETURN-PERIOD METRICS (suffixes={suffixes}) ================", flush=True)
            for sfx in suffixes:
                for idx, metric in enumerate(PHASE_3B_RP_METRICS, 1):
                    for model_type in model_types:
                        run_cv_and_global_for_metric_model(
                            metric,
                            model_type,
                            df_hybas,
                            pool=pool,
                            exp_suffix=sfx,
                            lead_time='1',
                            n_threads=args.n_threads,
                            skip_after_ts=args.skip_after_ts,
                        )
            print(f"\n[{time.strftime('%H:%M:%S')}] PHASE 3B COMPLETE for {suffixes} in {time.time()-t_start:.1f}s!", flush=True)


if __name__ == '__main__':
    main()
