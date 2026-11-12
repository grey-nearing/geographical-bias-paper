#!/usr/bin/env python3
"""Stage 5: Compute full-stack empirical black-box KernelSHAP attributions
(K_bg = 100 k-means centroids of X_ungauged, nsamples = 2*M + 2048 = 2,240 coalitions)
across all 10 2.0-window return-period metrics on the N = 1,000 global representative medoids,
combine with exact KernelSHAP for NSE and KGE, and render Figure 3 (Fig5_TreeSHAP_lead1)
and Figure S7 (FigS6_TreeSHAP_top35_lead1).
"""

import os

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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

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

from sklearn.ensemble import ExtraTreesRegressor as _ET  # noqa: E402

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
from utils.feature_utils import (  # noqa: E402
    FEATURE_CLASSIFICATION,
    FEATURE_DESCRIPTIONS,
    FEATURE_PALETTE,
)
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
                torch.set_num_threads(4)
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
            knn_reg.n_jobs = 4
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
            if len(sp_mod._KNN_NBR_CACHE) > 10:
                sp_mod._KNN_NBR_CACHE.pop(0)

    return np.einsum('ij,ij->i', fit_y[nbr_idx], nbr_w)


sp_mod._predict_knn_cached = _fast_predict_knn_cached

RP_METRICS_10 = [
    '1.01-year-2.0-window-precision',
    '2.0-year-2.0-window-precision',
    '5.0-year-2.0-window-precision',
    '10.0-year-2.0-window-precision',
    '20.0-year-2.0-window-precision',
    '1.01-year-2.0-window-recall',
    '2.0-year-2.0-window-recall',
    '5.0-year-2.0-window-recall',
    '10.0-year-2.0-window-recall',
    '20.0-year-2.0-window-recall',
]

ALL_12_METRICS = ['NSE', 'KGE'] + RP_METRICS_10

RECALL_METRICS = [
    '1.01-year-2.0-window-recall',
    '2.0-year-2.0-window-recall',
    '5.0-year-2.0-window-recall',
    '10.0-year-2.0-window-recall',
    '20.0-year-2.0-window-recall',
]

PRECISION_METRICS = [
    '1.01-year-2.0-window-precision',
    '2.0-year-2.0-window-precision',
    '5.0-year-2.0-window-precision',
    '10.0-year-2.0-window-precision',
    '20.0-year-2.0-window-precision',
]

NATURE_DOUBLE_COL_IN = 183.0 / 25.4
SAVE_DPI = 350


def prepare_feature_map(feature_names: list[str]) -> dict[str, str]:
    full_feature_map = {}
    for feat in feature_names:
        if feat in FEATURE_CLASSIFICATION:
            full_feature_map[feat] = FEATURE_CLASSIFICATION[feat]
        else:
            full_feature_map[feat] = 'Other'
    return full_feature_map


def short_feature_label(feat: str, max_len: int = 44) -> str:
    custom = {
        'anomaly_score_if': 'Climate OOD (Isolation Forest)',
        'anomaly_score_ocsvm': 'Climate OOD (One-Class SVM)',
        'anomaly_score_lof': 'Climate OOD (Local Outlier Factor)',
        'total_upstream_coverage': 'Upstream gauge coverage (%)',
        'max_upstream_coverage': 'Max single-gauge upstream cov.',
        'max_downstream_coverage': 'Max downstream gauge cov.',
        'upstream_gauge_count': 'Upstream gauge count',
        'record_length': 'Gauge record length (yr)',
        'calculated_drain_area': 'Drainage area',
        'ele_mt_uav': 'Mean elevation (m)',
        'slp_dg_uav': 'Mean terrain slope (deg)',
        'pre_mm_uyr': 'Mean annual precipitation',
        'pet_mm_uyr': 'Potential evapotranspiration',
        'aet_mm_uyr': 'Actual evapotranspiration',
        'cmi_ix_uyr': 'Climate moisture index',
        'ari_ix_uav': 'Aridity index',
        'tmp_dc_uyr': 'Mean annual air temperature',
        'snw_pc_uyr': 'Snow cover extent (%)',
        'swc_pc_uyr': 'Soil water content (%)',
        'snd_pc_uav': 'Soil sand fraction (%)',
        'slt_pc_uav': 'Soil silt fraction (%)',
        'cly_pc_uav': 'Soil clay fraction (%)',
        'soc_th_uav': 'Soil organic carbon (t/ha)',
        'ero_kh_uav': 'Soil erosion rate (kg/ha/yr)',
        'kar_pc_use': 'Karst geology extent (%)',
        'for_pc_use': 'Forest cover extent (%)',
        'crp_pc_use': 'Cropland extent (%)',
        'pst_pc_use': 'Pasture extent (%)',
        'urb_pc_use': 'Urban extent (%)',
        'gla_pc_use': 'Glacier cover (%)',
        'prm_pc_use': 'Permafrost extent (%)',
        'pac_pc_use': 'Protected area extent (%)',
        'lka_pc_use': 'Lake area extent (%)',
        'lkv_mc_usu': 'Total lake volume',
        'rev_mc_usu': 'Total reservoir volume',
        'ria_ha_usu': 'Total reservoir area',
        'riv_tc_usu': 'Total river channel volume',
        'ire_pc_use': 'Irrigated area extent (%)',
        'pop_ct_usu': 'Watershed population count',
        'ppd_pk_uav': 'Population density (persons/km²)',
        'gdp_ud_usu': 'Watershed GDP (USD)',
        'rdd_mk_uav': 'Road network density',
        'nli_ix_uav': 'Nighttime lights index',
        'hft_ix_u09': 'Human footprint index (2009)',
        'hft_ix_u93': 'Human footprint index (1993)',
        'inu_pc_ult': 'Long-term inundation extent (%)',
        'inu_pc_umn': 'Minimum inundation extent (%)',
        'inu_pc_umx': 'Maximum inundation extent (%)',
        # Distinct PNV labels so truncation never produces duplicate rows:
        'pnv_pc_u01': 'PNV: Trop./Subtrop. Moist Broadleaf (%)',
        'pnv_pc_u02': 'PNV: Trop./Subtrop. Dry Broadleaf (%)',
        'pnv_pc_u03': 'PNV: Trop./Subtrop. Coniferous (%)',
        'pnv_pc_u04': 'PNV: Temp. Broadleaf & Mixed Forest (%)',
        'pnv_pc_u05': 'PNV: Temp. Coniferous Forest (%)',
        'pnv_pc_u06': 'PNV: Boreal Forest / Taiga (%)',
        'pnv_pc_u07': 'PNV: Trop./Subtrop. Grasslands (%)',
        'pnv_pc_u08': 'PNV: Temperate Grasslands (%)',
        'pnv_pc_u09': 'PNV: Flooded Grasslands/Savannas (%)',
        'pnv_pc_u10': 'PNV: Montane Grasslands/Shrublands (%)',
        'pnv_pc_u11': 'PNV: Tundra (%)',
        'pnv_pc_u12': 'PNV: Mediterranean Woodlands (%)',
        'pnv_pc_u13': 'PNV: Deserts & Xeric Shrublands (%)',
        'pnv_pc_u14': 'PNV: Mangroves (%)',
        'pnv_pc_u15': 'PNV: Inland Water (%)',
    }
    if feat in custom:
        return custom[feat]
    desc = FEATURE_DESCRIPTIONS.get(feat, feat)
    desc = desc.split(';')[0].rstrip('.')
    desc = (
        desc.replace('Potential Natural Veg:', 'PNV:')
        .replace('Land cover:', 'LC:')
        .replace('Wetlands Global Class:', 'Wetland Class:')
        .replace('Wetlands:', 'Wetland:')
        .replace('(percentage)', '(%)')
    )
    if len(desc) > max_len:
        desc = desc[: max_len - 1].rstrip() + '…'
    return desc


def _fit_one_full_model(args_tuple):
    metric, n_threads = args_tuple
    t0 = time.time()
    file_paths.init(metric)
    os.makedirs(file_paths.MODELS_DIR, exist_ok=True)
    pkl_path = os.path.join(file_paths.MODELS_DIR, f'final_ungauged_model_{metric}.pkl')
    if os.path.exists(pkl_path) and os.path.getmtime(pkl_path) >= 1794328000:
        try:
            with open(pkl_path, 'rb') as f:
                _ = pickle.load(f)
            print(f"[{time.strftime('%H:%M:%S')}] Loaded existing fresh model for {metric}", flush=True)
            return metric
        except Exception:
            pass
    x, y = io_utils.load_and_preprocess_data('ungauged', metric, None, exp_suffix='', lead_time='1')
    m = SkillPredictor(metric)
    impl = m.impl
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
    with open(pkl_path, 'wb') as f:
        pickle.dump(m, f)
    print(f"[{time.strftime('%H:%M:%S')}] Fitted & saved final_ungauged_model_{metric}.pkl in {time.time()-t0:.1f}s", flush=True)
    return metric


_WORKER_MODELS = None
_WORKER_BG = None
_WORKER_COLS = None
_WORKER_STRAT = None
_WORKER_SHARD_DIR = None


def _init_shap_worker(models_dir, bg_data, bg_weights, raw_cols, x_strat_df, shard_dir, n_threads):
    global _WORKER_MODELS, _WORKER_BG, _WORKER_COLS, _WORKER_STRAT, _WORKER_SHARD_DIR
    _WORKER_COLS = raw_cols
    _WORKER_STRAT = x_strat_df
    _WORKER_SHARD_DIR = shard_dir
    _WORKER_BG = shap.utils._legacy.DenseData(bg_data, list(raw_cols), None, bg_weights)
    _WORKER_MODELS = []
    for m_name in RP_METRICS_10:
        pkl_path = os.path.join(models_dir, f'final_ungauged_model_{m_name}.pkl')
        with open(pkl_path, 'rb') as f:
            mod = pickle.load(f)
        # Set threads per worker
        est_hgbr, est_et, est_knn = mod.impl.ensemble.estimators_
        if hasattr(est_et, 'named_steps') and 'extratreesregressor' in est_et.named_steps:
            est_et.named_steps['extratreesregressor'].n_jobs = n_threads
        _WORKER_MODELS.append((m_name, mod, n_threads))


def _predict_all_10(X):
    df = pd.DataFrame(X, columns=_WORKER_COLS) if isinstance(X, np.ndarray) else X
    X_transformed = sp_mod._get_cached_preprocessed_features(_WORKER_MODELS[0][1].impl, df)
    out_cols = []
    for _, mod, n_threads in _WORKER_MODELS:
        impl = mod.impl
        X_aligned = X_transformed.reindex(columns=impl.feature_columns, fill_value=0)
        est_hgbr, est_et, est_knn = impl.ensemble.estimators_
        p_hgbr = est_hgbr._loss.link.inverse(
            est_hgbr._raw_predict(X_aligned, n_threads=n_threads).ravel()
        )
        est_et.named_steps['extratreesregressor'].n_jobs = n_threads
        with joblib.parallel_backend('threading', n_jobs=n_threads):
            p_et = est_et.predict(X_aligned)
        p_knn = sp_mod._predict_knn_cached(est_knn, X_aligned)
        preds = impl.ensemble.final_estimator_.predict(
            np.column_stack([p_hgbr, p_et, p_knn])
        )
        out_cols.append(np.clip(preds, 0.0, 1.0))
    return np.column_stack(out_cols)


def _compute_one_basin_shap(idx: int):
    out_npz = os.path.join(_WORKER_SHARD_DIR, f'basin_{idx:04d}.npz')
    if os.path.exists(out_npz) and os.path.getsize(out_npz) > 1000:
        return idx, 0.0, True
    lock_path = out_npz + '.lock'
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
    except FileExistsError:
        try:
            if time.time() - os.path.getmtime(lock_path) < 600:
                return idx, 0.0, True
        except OSError:
            pass

    t0 = time.time()
    sp_mod._RP_FEAT_CACHE.clear()
    sp_mod._KNN_NBR_CACHE.clear()
    np.random.seed(42 + idx)
    explainer = shap.KernelExplainer(_predict_all_10, _WORKER_BG)
    x_row = _WORKER_STRAT.iloc[idx : idx + 1]
    sv = explainer.shap_values(x_row, silent=True)
    # sv is either a list of 10 arrays of shape (1, 96) or an array of shape (1, 96, 10)
    if isinstance(sv, list):
        sv_mat = np.vstack([arr.reshape(1, -1) for arr in sv])  # (10, 96)
    else:
        sv_mat = np.moveaxis(sv[0], -1, 0)  # (10, 96)

    np.savez_compressed(
        out_npz,
        idx=idx,
        basin_id=str(x_row.index[0]),
        sv_mat=sv_mat,
    )
    try:
        if os.path.exists(lock_path):
            os.unlink(lock_path)
    except OSError:
        pass
    return idx, time.time() - t0, False


def aggregate_and_plot(shard_dir: str, data_dir: str, repo_dir: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style='whitegrid', palette='colorblind')
    plt.rcParams.update({
        'figure.dpi': SAVE_DPI,
        'savefig.dpi': SAVE_DPI,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 7.5,
        'axes.titlesize': 8.2,
        'axes.titleweight': 'bold',
        'axes.labelsize': 7.5,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 6.8,
        'ytick.labelsize': 6.8,
        'legend.fontsize': 6.8,
        'legend.title_fontsize': 7.2,
        'axes.linewidth': 0.6,
        'grid.linewidth': 0.45,
        'lines.linewidth': 1.0,
        'patch.linewidth': 0.6,
    })

    nse_ref_path = os.path.join(
        data_dir,
        'saved_predictions/shap/NSE/ungauged/stratified_1000/consolidated_exact_kernelshap_scores_NSE.csv',
    )
    df_nse_ref = pd.read_csv(nse_ref_path, index_col=0)
    df_nse_ref.index = df_nse_ref.index.astype(str)
    strat_ids = df_nse_ref.index.tolist()
    raw_cols = df_nse_ref.columns.tolist()

    # Load all 1,000 .npz shards
    mats = {m: np.zeros((len(strat_ids), len(raw_cols)), dtype=np.float64) for m in RP_METRICS_10}
    missing = []
    for idx in range(len(strat_ids)):
        npz_path = os.path.join(shard_dir, f'basin_{idx:04d}.npz')
        if not os.path.exists(npz_path):
            missing.append(idx)
            continue
        d = np.load(npz_path, allow_pickle=True)
        sv_mat = d['sv_mat']
        for m_i, m_name in enumerate(RP_METRICS_10):
            mats[m_name][idx, :] = sv_mat[m_i, :]

    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} KernelSHAP basin shards: {missing[:20]}...")

    out_dirs = [
        os.path.join(data_dir, 'science_results'),
    ]
    if repo_dir and os.path.exists(repo_dir):
        out_dirs.append(os.path.join(repo_dir, 'data/science_results'))

    for od in out_dirs:
        os.makedirs(os.path.join(od, 'shap_scores_lead1'), exist_ok=True)

    # Save consolidated CSVs for all 12 metrics
    dfs_12 = {}
    for m_classic in ['NSE', 'KGE']:
        c_path = os.path.join(
            data_dir,
            f'saved_predictions/shap/{m_classic}/ungauged/stratified_1000/consolidated_exact_kernelshap_scores_{m_classic}.csv',
        )
        df_c = pd.read_csv(c_path, index_col=0)
        df_c.index = df_c.index.astype(str)
        df_c = df_c.reindex(index=strat_ids, columns=raw_cols)
        dfs_12[m_classic] = df_c
        for od in out_dirs:
            df_c.to_csv(os.path.join(od, 'shap_scores_lead1', f'consolidated_treeshap_scores_{m_classic}_lead1.csv'))

    for m_rp in RP_METRICS_10:
        df_rp = pd.DataFrame(mats[m_rp], index=strat_ids, columns=raw_cols)
        dfs_12[m_rp] = df_rp
        dest_dir = os.path.join(data_dir, f'saved_predictions/shap/{m_rp}/ungauged/stratified_1000')
        os.makedirs(dest_dir, exist_ok=True)
        df_rp.to_csv(os.path.join(dest_dir, f'consolidated_exact_kernelshap_scores_{m_rp}.csv'))
        for od in out_dirs:
            df_rp.to_csv(os.path.join(od, 'shap_scores_lead1', f'consolidated_treeshap_scores_{m_rp}_lead1.csv'))

    feature_importances = {}
    category_importances = {}
    category_std = {}
    shap_raw_list = []

    for metric in ALL_12_METRICS:
        df_shap = dfs_12[metric]
        feature_importances[metric] = df_shap.abs().mean()
        feature_map = prepare_feature_map(df_shap.columns.tolist())
        # Signed sum within category per watershed first, then absolute value
        cat_shap_abs = df_shap.T.groupby(feature_map).sum().T.abs()
        category_importances[metric] = cat_shap_abs.mean()
        category_std[metric] = cat_shap_abs.std()
        df_m_raw = (
            cat_shap_abs.reset_index()
            .rename(columns={'index': 'basin_id'})
            .melt(id_vars='basin_id', var_name='category', value_name='importance')
        )
        df_m_raw['metric'] = metric
        shap_raw_list.append(df_m_raw)

    df_raw_all_f5 = pd.concat(shap_raw_list, ignore_index=True)
    df_features = pd.DataFrame(feature_importances).fillna(0.0)
    df_categories = pd.DataFrame(category_importances).fillna(0.0)
    df_categories_pct = df_categories.div(df_categories.sum(axis=0), axis=1) * 100.0

    df_features_norm = df_features.div(df_features.max(axis=0), axis=1)
    top35_feats = df_features.max(axis=1).sort_values(ascending=False).head(35).index.tolist()
    fmap_all = prepare_feature_map(df_features.index.tolist())
    df_top35 = df_features.loc[top35_feats].copy()
    df_top35.insert(0, 'category', [fmap_all[f] for f in top35_feats])
    df_top35.insert(1, 'description', [short_feature_label(f, max_len=60) for f in top35_feats])
    df_top35['max_importance'] = df_features.loc[top35_feats].max(axis=1)
    df_top35['mean_normalized_importance'] = df_features_norm.loc[top35_feats].mean(axis=1)

    for od in out_dirs:
        df_categories.to_csv(os.path.join(od, 'treeshap_category_importances_lead1.csv'))
        df_categories_pct.to_csv(os.path.join(od, 'treeshap_category_importances_pct_lead1.csv'))
        df_top35.to_csv(os.path.join(od, 'treeshap_top35_features_lead1.csv'))

    # Render Fig5_TreeSHAP_lead1.png (Figure 3 in Science manuscript)
    metric_palette = {
        'NSE': '#0d47a1',
        'KGE': '#42a5f5',
        '1.01-year-2.0-window-precision': '#ffcc80',
        '2.0-year-2.0-window-precision': '#ffa726',
        '5.0-year-2.0-window-precision': '#fb8c00',
        '10.0-year-2.0-window-precision': '#e65100',
        '20.0-year-2.0-window-precision': '#b71c1c',
        '1.01-year-2.0-window-recall': '#c8e6c9',
        '2.0-year-2.0-window-recall': '#81c784',
        '5.0-year-2.0-window-recall': '#4caf50',
        '10.0-year-2.0-window-recall': '#2e7d32',
        '20.0-year-2.0-window-recall': '#1b5e20',
    }
    metric_labels_map = {
        'NSE': 'NSE',
        'KGE': 'KGE',
        '1.01-year-2.0-window-precision': 'Precision (1y)',
        '2.0-year-2.0-window-precision': 'Precision (2y)',
        '5.0-year-2.0-window-precision': 'Precision (5y)',
        '10.0-year-2.0-window-precision': 'Precision (10y)',
        '20.0-year-2.0-window-precision': 'Precision (20y)',
        '1.01-year-2.0-window-recall': 'Recall (1y)',
        '2.0-year-2.0-window-recall': 'Recall (2y)',
        '5.0-year-2.0-window-recall': 'Recall (5y)',
        '10.0-year-2.0-window-recall': 'Recall (10y)',
        '20.0-year-2.0-window-recall': 'Recall (20y)',
    }
    plot_palette = {**FEATURE_PALETTE, 'Other': 'gray'}
    short_cat_label = {
        'Climate & Hydrology': 'Climate/Hydrol.',
        'Land Cover': 'Land Cover',
        'Anomaly Score': 'Anomaly Score',
        'Topography': 'Topography',
        'Soil and Geology': 'Soil & Geology',
        'Anthropogenic Factors': 'Anthropogenic',
        'Coverage/Overlap': 'Gauge Coverage',
        'Record Length': 'Record Length',
    }

    fig5 = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 6.4))
    gs5 = gridspec.GridSpec(3, 5, height_ratios=[1.28, 1.0, 1.0], hspace=0.52, wspace=1.28)
    ax_comp = fig5.add_subplot(gs5[0, 0:3])
    cat_order = df_categories.mean(axis=1).sort_values(ascending=False).index.tolist()
    sns.barplot(
        x='category',
        y='importance',
        hue='metric',
        data=df_raw_all_f5,
        order=cat_order,
        hue_order=ALL_12_METRICS,
        palette=metric_palette,
        ax=ax_comp,
        edgecolor='black',
        linewidth=0.35,
        errorbar='sd',
        err_kws={'linewidth': 0.55},
    )
    ax_comp.set_title('(a) Feature category importance across all 12 metrics', fontsize=7.2, fontweight='bold', loc='left', pad=4)
    ax_comp.set_xlabel('')
    ax_comp.set_ylabel('Mean |SHAP|', fontsize=6.6, fontweight='bold')
    ax_comp.set_xticklabels([short_cat_label.get(c, c) for c in cat_order], rotation=22, ha='right', fontsize=5.8)
    ax_comp.tick_params(axis='y', labelsize=5.8)
    handles_f5, labels_f5 = ax_comp.get_legend_handles_labels()
    ax_comp.legend(
        handles_f5,
        [metric_labels_map.get(lbl, lbl) for lbl in labels_f5],
        title='Metric',
        loc='upper right',
        fontsize=4.9,
        title_fontsize=5.4,
        ncol=2,
        frameon=True,
        handlelength=1.0,
        columnspacing=0.7,
    )
    sns.despine(ax=ax_comp)
    ax_comp.grid(axis='y', linestyle='--', alpha=0.45)

    def plot_cat_panel(ax, metric, title_str, show_xlabel=True):
        df_m = df_categories[metric].sort_values(ascending=False)
        df_mr = df_raw_all_f5[df_raw_all_f5['metric'] == metric]
        sns.barplot(
            x='importance',
            y='category',
            data=df_mr,
            order=df_m.index,
            palette=plot_palette,
            hue='category',
            legend=False,
            ax=ax,
            edgecolor='black',
            linewidth=0.35,
            errorbar='sd',
            err_kws={'linewidth': 0.55},
        )
        ax.set_yticklabels([short_cat_label.get(c, c) for c in df_m.index], fontsize=4.8)
        ax.set_title(title_str, fontsize=6.2, fontweight='bold', loc='left', pad=3)
        ax.set_xlabel('Mean |SHAP|' if show_xlabel else '', fontsize=5.5, fontweight='bold')
        ax.set_ylabel('')
        ax.tick_params(axis='x', labelsize=4.8)
        sns.despine(ax=ax)
        ax.grid(axis='x', linestyle='--', alpha=0.45)

    for idx, (metric, pl_ltr) in enumerate([('NSE', '(b)'), ('KGE', '(c)')]):
        ax = fig5.add_subplot(gs5[0, 3 + idx])
        plot_cat_panel(ax, metric, f'{pl_ltr} {metric}', show_xlabel=True)

    sub_letters_rows = [
        ['(d)', '(e)', '(f)', '(g)', '(h)'],
        ['(i)', '(j)', '(k)', '(l)', '(m)'],
    ]
    for row_i, m_group, pfx, ltrs, show_x in [
        (1, RECALL_METRICS, 'Recall', sub_letters_rows[0], False),
        (2, PRECISION_METRICS, 'Precision', sub_letters_rows[1], True),
    ]:
        for idx, metric in enumerate(m_group):
            ax = fig5.add_subplot(gs5[row_i, idx])
            yr_str = '1y' if metric.startswith('1.01') else f"{metric.split('.')[0]}y"
            plot_cat_panel(ax, metric, f'{ltrs[idx]} {pfx} ({yr_str})', show_xlabel=show_x)

    for od in out_dirs:
        fig5.savefig(os.path.join(od, 'Fig5_TreeSHAP_lead1.png'), dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.03)
        fig5.savefig(os.path.join(od, 'Fig5_TreeSHAP_lead1.pdf'), bbox_inches='tight', pad_inches=0.03)
    plt.close(fig5)

    # Render FigS6_TreeSHAP_top35_lead1.png (Figure S7 in Science manuscript)
    short_col_labels = [
        'NSE', 'KGE',
        'Prec (1y)', 'Prec (2y)', 'Prec (5y)', 'Prec (10y)', 'Prec (20y)',
        'Rec (1y)', 'Rec (2y)', 'Rec (5y)', 'Rec (10y)', 'Rec (20y)',
    ]
    df_s6_raw = df_features.loc[top35_feats, ALL_12_METRICS].copy()
    df_s6_raw.columns = short_col_labels
    fmap_s6 = prepare_feature_map(top35_feats)
    y_labels_s6 = [
        f'{short_feature_label(f, max_len=42)} ({fmap_s6[f]})' for f in top35_feats
    ]
    assert len(set(y_labels_s6)) == len(y_labels_s6), f"Duplicate labels found in y_labels_s6: {y_labels_s6}"

    fig_s6, ax_s6 = plt.subplots(figsize=(NATURE_DOUBLE_COL_IN, 7.6))
    sns.heatmap(
        df_s6_raw,
        ax=ax_s6,
        cmap='viridis',
        annot=True,
        fmt='.4f',
        annot_kws={'size': 4.8},
        linewidths=0.30,
        cbar_kws={'label': 'Mean Absolute Shapley (KernelSHAP) Value', 'shrink': 0.82},
    )
    ax_s6.set_yticks(np.arange(len(top35_feats)) + 0.5)
    ax_s6.set_yticklabels(y_labels_s6, rotation=0, fontsize=5.6)
    ax_s6.set_xticklabels(short_col_labels, rotation=32, ha='right', fontsize=6.2, fontweight='bold')
    ax_s6.set_title(
        'Global Feature Importance Heatmap (Top 35 Features Across 12 Metrics, Lead Time 1)',
        fontsize=8.0, fontweight='bold', pad=8,
    )
    ax_s6.set_xlabel('Hydrological Evaluation Metric', fontsize=7.0, fontweight='bold')
    ax_s6.set_ylabel('Watershed Predictor Feature (Category)', fontsize=7.0, fontweight='bold')

    for od in out_dirs:
        fig_s6.savefig(os.path.join(od, 'FigS6_TreeSHAP_top35_lead1.png'), dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.03)
        fig_s6.savefig(os.path.join(od, 'FigS6_TreeSHAP_top35_lead1.pdf'), bbox_inches='tight', pad_inches=0.03)
    plt.close(fig_s6)
    print(f"[{time.strftime('%H:%M:%S')}] Saved Fig5_TreeSHAP_lead1 and FigS6_TreeSHAP_top35_lead1 to {out_dirs}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit-models', action='store_true')
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--end-idx', type=int, default=1000)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--n-threads', type=int, default=2)
    parser.add_argument('--shard-dir', type=str, default='/tmp/kernelshap_shards_lead1')
    parser.add_argument('--aggregate-and-plot', action='store_true')
    parser.add_argument('--repo-dir', type=str, default='.')
    args = parser.parse_args()

    os.makedirs(args.shard_dir, exist_ok=True)
    file_paths.init('5.0-year-2.0-window-recall')

    if args.fit_models:
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=min(10, args.num_workers), mp_context=ctx) as pool:
            list(pool.map(_fit_one_full_model, [(m, 6) for m in RP_METRICS_10]))

    if args.end_idx > args.start_idx:
        # Prepare background centroids & X_strat
        X_ungauged, _ = io_utils.load_and_preprocess_data('2.0-year-2.0-window-recall' if False else 'ungauged', '2.0-year-2.0-window-recall', None, exp_suffix='', lead_time='1')
        raw_cols = X_ungauged.columns.tolist()

        np.random.seed(42)
        bg_summary = shap.kmeans(X_ungauged, 100)
        bg_data = np.asarray(bg_summary.data, dtype=np.float64)
        bg_weights = np.asarray(bg_summary.weights, dtype=np.float64)

        nse_ref_path = os.path.join(
            os.environ['GEO_BIAS_DATA_DIR'],
            'saved_predictions/shap/NSE/ungauged/stratified_1000/consolidated_exact_kernelshap_scores_NSE.csv',
        )
        df_nse_ref = pd.read_csv(nse_ref_path, index_col=0)
        strat_ids = df_nse_ref.index.astype(str).tolist()

        df_hybas = io_utils.load_hybas_zarr(file_paths.HYBAS_GENERIC_DATA_ZARR_PATH)
        df_hybas.index = df_hybas.index.astype(str)
        X_strat = df_hybas.reindex(strat_ids).reindex(columns=raw_cols, fill_value=0)

        indices = list(range(args.start_idx, min(args.end_idx, len(strat_ids))))
        print(f"[{time.strftime('%H:%M:%S')}] Running exact multi-output KernelSHAP on indices [{args.start_idx}, {args.end_idx}) with {args.num_workers} workers...", flush=True)
        t_start = time.time()
        ctx = mp.get_context('spawn')
        done_cnt = 0
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            mp_context=ctx,
            initializer=_init_shap_worker,
            initargs=(file_paths.MODELS_DIR, bg_data, bg_weights, raw_cols, X_strat, args.shard_dir, args.n_threads),
        ) as pool:
            for idx, dt, skipped in pool.map(_compute_one_basin_shap, indices):
                done_cnt += 1
                if done_cnt % 10 == 0 or done_cnt == len(indices):
                    el = time.time() - t_start
                    rate = done_cnt / max(1e-6, el)
                    print(
                        f"[{time.strftime('%H:%M:%S')}] Progress: {done_cnt}/{len(indices)} basins "
                        f"(last idx={idx} in {dt:.1f}s, skipped={skipped}) | Elapsed={el:.1f}s ({rate*60:.1f} basins/min)",
                        flush=True,
                    )

    if args.aggregate_and_plot:
        aggregate_and_plot(args.shard_dir, os.environ['GEO_BIAS_DATA_DIR'], args.repo_dir)


if __name__ == '__main__':
    main()
