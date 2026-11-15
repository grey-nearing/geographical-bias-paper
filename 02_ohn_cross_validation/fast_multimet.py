"""High-performance memory-mapped Multimet dataset and dataloader for Caravan-MultiMet.

Provides `FastMultimet`, `FastMultimetDataLoader`, `install_fast_multimet()`, and
`fast_evaluate_period()`, backed by the pre-unioned float32 memory-mapped arrays in
`./data/data/precomputed/`.

Guarantees 100% numerical equivalence with `model.datasetzoo.multimet.Multimet`
(`Scaler`, valid sample index, and batch tensors) while reducing dataset/scaler
initialization from ~5.5 hours to <1 second and keeping per-worker private heap
RAM (`RssAnon`) under ~1 GB so all 4 GPU workers share the 30 GB OS page cache
comfortably inside 58 GB RAM.
"""

from __future__ import annotations

import functools
import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import Dataset
import xarray as xr

from model.datasetzoo.multimet import SampleIndexer, _needs_per_basin_target_stds
from model.datautils.scaler import Scaler
from model.datautils.utils import load_basin_file
from model.utils.config import Config
from model.utils.configutils import flatten_feature_list
from model.utils.errors import NoEvaluationDataError, NoTrainDataError

LOGGER = logging.getLogger(__name__)

PRECOMPUTED_DIR = Path('./data/data/precomputed')
POST2016_OFFSET = 12776  # Index of 2015-12-25 in dates.npy (start=1981-01-01)

HINDCAST_FEATURES_ORDER = [
    'hres_surface_net_solar_radiation',
    'hres_surface_net_thermal_radiation',
    'hres_surface_pressure',
    'hres_temperature_2m',
    'hres_total_precipitation',
    'graphcast_temperature_2m',
    'graphcast_total_precipitation',
    'imerg_precipitation',
    'cpc_precipitation',
]

FORECAST_FEATURES_ORDER = [
    'hres_surface_net_solar_radiation',
    'hres_surface_net_thermal_radiation',
    'hres_surface_pressure',
    'hres_temperature_2m',
    'hres_total_precipitation',
    'graphcast_temperature_2m',
    'graphcast_total_precipitation',
]

_PRECOMPUTED_CACHE: dict[str, Any] = {}


def get_precomputed_store(precomputed_dir: Path = PRECOMPUTED_DIR) -> dict[str, Any]:
    """Load and cache read-only memory-mapped arrays for the process."""
    key = str(precomputed_dir.resolve())
    if key in _PRECOMPUTED_CACHE:
        return _PRECOMPUTED_CACHE[key]

    basins = (precomputed_dir / 'basins.txt').read_text().strip().splitlines()
    basin_to_idx = {b: i for i, b in enumerate(basins)}
    dates_raw = np.load(precomputed_dir / 'dates.npy')
    dates = pd.DatetimeIndex(dates_raw, name='date')
    date_to_idx = pd.Series(np.arange(len(dates), dtype=np.int32), index=dates)
    static_attrs = (
        (precomputed_dir / 'static_attributes.txt').read_text().strip().splitlines()
    )
    static_attr_to_idx = {a: i for i, a in enumerate(static_attrs)}

    store = {
        'basins': basins,
        'basin_to_idx': basin_to_idx,
        'dates': dates,
        'dates_np': dates_raw,
        'date_to_idx': date_to_idx,
        'static_attrs': static_attrs,
        'static_attr_to_idx': static_attr_to_idx,
        'statics': np.load(precomputed_dir / 'statics.npy', mmap_mode='r'),
        'streamflow': np.load(precomputed_dir / 'streamflow.npy', mmap_mode='r'),
        'hindcast_2d': np.load(precomputed_dir / 'hindcast_2d.npy', mmap_mode='r'),
        'forecast_3d_post2016': np.load(
            precomputed_dir / 'forecast_3d_post2016.npy', mmap_mode='r'
        ),
        'forecast_3d_sum': np.load(
            precomputed_dir / 'forecast_3d_sum.npy', mmap_mode='r'
        ),
        'forecast_3d_sq_sum': np.load(
            precomputed_dir / 'forecast_3d_sq_sum.npy', mmap_mode='r'
        ),
        'forecast_3d_count': np.load(
            precomputed_dir / 'forecast_3d_count.npy', mmap_mode='r'
        ),
    }
    _PRECOMPUTED_CACHE[key] = store
    return store


def _indices_to_slices(idx: np.ndarray) -> list[slice]:
    """Convert a sorted 1D integer index array into a minimal list of contiguous `slice` objects."""
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) != 1) + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(idx)]])
    return [slice(int(idx[s]), int(idx[e - 1]) + 1) for s, e in zip(starts, ends)]


class FastMultimet(Dataset):
    """Memory-mapped drop-in replacement for `model.datasetzoo.multimet.Multimet`."""

    def __init__(
        self,
        cfg: Config,
        is_train: bool,
        period: str,
        basins: list[str] | None = None,
        compute_scaler: bool = True,
        precomputed_dir: Path = PRECOMPUTED_DIR,
    ):
        super().__init__()
        self.cfg = cfg
        self.is_train = is_train
        self._period = period
        if period not in ['train', 'validation', 'test']:
            raise ValueError("'period' must be one of 'train', 'validation' or 'test'")
        if period in ['validation', 'test'] or cfg.is_finetuning:
            if compute_scaler:
                raise ValueError(
                    'Scaler must be loaded (not computed) for validation, test, and finetuning.'
                )

        self._store = get_precomputed_store(precomputed_dir)

        self._seq_length = cfg.seq_length
        self.lead_time = cfg.lead_time
        self._forecast_overlap = cfg.forecast_overlap
        self._predict_last_n = cfg.predict_last_n
        self._min_lead_time = 1
        self._lead_times = list(range(self._min_lead_time, self.lead_time + 1))

        self._static_features = cfg.static_attributes
        self._hindcast_features = flatten_feature_list(cfg.hindcast_inputs)
        self._forecast_features = flatten_feature_list(cfg.forecast_inputs)
        self._target_features = cfg.target_variables

        # Verify requested features match our precomputed store
        self._static_col_idx = np.array(
            [self._store['static_attr_to_idx'][a] for a in self._static_features],
            dtype=np.int32,
        )
        self._hindcast_col_idx = np.array(
            [HINDCAST_FEATURES_ORDER.index(f) for f in self._hindcast_features],
            dtype=np.int32,
        )
        self._forecast_col_idx = np.array(
            [FORECAST_FEATURES_ORDER.index(f) for f in self._forecast_features],
            dtype=np.int32,
        )
        self._forecast_in_hindcast_col_idx = np.array(
            [HINDCAST_FEATURES_ORDER.index(f) for f in self._forecast_features],
            dtype=np.int32,
        )

        raw_basins = basins or load_basin_file(getattr(cfg, f'{period}_basin_file'))
        # Match xarray merge(join='outer') alphabetical basin coordinate ordering
        self._basins = sorted(raw_basins)
        self._global_basin_idx = np.array(
            [self._store['basin_to_idx'][b] for b in self._basins], dtype=np.int32
        )

        start_dates, end_dates = self._get_period_dates(cfg)
        self._sample_dates = self._union_ranges(start_dates, end_dates)
        extended_start_dates = [
            s - pd.Timedelta(days=self._seq_length) for s in start_dates
        ]
        extended_end_dates = [
            e + pd.Timedelta(days=self.lead_time) for e in end_dates
        ]
        self._extended_dates = self._union_ranges(
            extended_start_dates, extended_end_dates
        ).intersection(self._store['dates'])

        self._ext_global_date_idx = (
            self._store['date_to_idx'].loc[self._extended_dates].to_numpy(dtype=np.int32)
        )
        self._sample_in_ext_mask = self._extended_dates.isin(self._sample_dates)

        # Timestep counters
        self._hindcast_counter = None
        self._forecast_counter = None
        if cfg.timestep_counter:
            self._hindcast_counter = np.full(
                (self._seq_length, 1), 0.0, dtype=np.float32
            )
            fc = np.array(self._lead_times, dtype=np.float32)
            if self._forecast_overlap:
                overlap_c = np.full(
                    (self._forecast_overlap,), self._min_lead_time, dtype=np.float32
                )
                fc = np.concatenate([overlap_c, fc], axis=0)
            self._forecast_counter = np.expand_dims(fc, -1)

        # Scaler initialization
        if cfg.is_finetuning:
            if self.is_train:
                scaler_dir = cfg.base_run_dir
            else:
                scaler_dir = (
                    cfg.run_dir
                    if (cfg.run_dir is not None and (cfg.run_dir / 'scaler.zarr').exists())
                    else cfg.base_run_dir
                )
        else:
            scaler_dir = cfg.run_dir

        if compute_scaler:
            scaler_ds = self._compute_fast_scaler_dataset()
            self.scaler = Scaler(
                scaler_dir=scaler_dir,
                calculate_scaler=True,
                custom_normalization=cfg.custom_normalization,
                dataset=None,
            )
            self.scaler.scaler = scaler_ds
            self.scaler.check_zero_scale()
            if cfg.run_dir is not None:
                self.scaler.scaler_dir = cfg.run_dir
                self.scaler.save()
        else:
            self.scaler = Scaler(
                scaler_dir=scaler_dir,
                calculate_scaler=False,
                custom_normalization=cfg.custom_normalization,
                dataset=None,
            )
            self.scaler.check_zero_scale()

        # Extract contiguous float32 center/scale vectors for vectorized batch scaling
        sc = self.scaler.scaler
        self._s_center = np.array(
            [float(sc[a].sel(parameter='center').values) for a in self._static_features],
            dtype=np.float32,
        )
        self._s_scale = np.array(
            [float(sc[a].sel(parameter='scale').values) for a in self._static_features],
            dtype=np.float32,
        )
        self._h_center = np.array(
            [float(sc[f].sel(parameter='center').values) for f in self._hindcast_features],
            dtype=np.float32,
        )
        self._h_scale = np.array(
            [float(sc[f].sel(parameter='scale').values) for f in self._hindcast_features],
            dtype=np.float32,
        )
        self._f_center = np.array(
            [float(sc[f].sel(parameter='center').values) for f in self._forecast_features],
            dtype=np.float32,
        )
        self._f_scale = np.array(
            [float(sc[f].sel(parameter='scale').values) for f in self._forecast_features],
            dtype=np.float32,
        )
        self._y_center = float(sc['streamflow'].sel(parameter='center').values)
        self._y_scale = float(sc['streamflow'].sel(parameter='scale').values)

        # Build valid sample mask & compact SampleIndexer
        self._build_valid_sample_index()

        self._lazy_dataset_cache: xr.Dataset | None = None
        self._per_basin_target_stds = None
        if _needs_per_basin_target_stds(cfg):
            stds = np.empty(len(self._basins), dtype=np.float32)
            for i, gb in enumerate(self._global_basin_idx):
                row = self._store['streamflow'][gb, self._ext_global_date_idx]
                stds[i] = np.nanstd((row - self._y_center) / self._y_scale, dtype=np.float32)
            self._per_basin_target_stds = stds

        gc.collect()

    @property
    def _dataset(self) -> xr.Dataset:
        """Lazy _dataset property for BaseTester._calc_exclude_basins compatibility."""
        if self._lazy_dataset_cache is None:
            sf_sub = self._store['streamflow'][
                np.ix_(self._global_basin_idx, self._ext_global_date_idx)
            ]
            sf_scaled = (sf_sub - self._y_center) / self._y_scale
            self._lazy_dataset_cache = xr.Dataset(
                data_vars={
                    'streamflow': (('basin', 'date'), sf_scaled),
                },
                coords={
                    'basin': np.array(self._basins, dtype=object),
                    'date': self._extended_dates,
                },
            )
        return self._lazy_dataset_cache

    def _compute_fast_scaler_dataset(self) -> xr.Dataset:
        """Compute exact Scaler xarray.Dataset using zero-copy slice views over the mmap arrays."""
        b_idx = self._global_basin_idx
        d_idx = self._ext_global_date_idx
        n_b = len(b_idx)
        n_d = len(d_idx)
        n_total_basins = len(self._store['basins'])
        d_slices = _indices_to_slices(d_idx)

        b_mask = np.zeros(n_total_basins, dtype=bool)
        b_mask[b_idx] = True

        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}

        # 1. Static features: mean and std (ddof=0) over self._basins (in float32 to match xarray/dask)
        statics_sub = self._store['statics'][b_idx][:, self._static_col_idx]
        s_mean = statics_sub.mean(axis=0)
        s_std = statics_sub.std(axis=0, ddof=0)
        s_std = np.where(s_std == 0.0, 1.0, s_std)
        for i, attr in enumerate(self._static_features):
            val = np.array([s_mean[i], s_std[i], s_mean[i], s_std[i]], dtype=np.float32)
            for k in (attr, f'{attr}_obs', f'{attr}_sim'):
                data_vars[k] = (('parameter',), val)

        # 2. Streamflow, 3D Forecast features, and 2D-only Hindcast features using zero-copy slice views
        sf_sum = 0.0
        sf_sq_sum = 0.0
        sf_count = 0

        tot_f_sum = np.zeros(len(FORECAST_FEATURES_ORDER), dtype=np.float64)
        tot_f_sq_sum = np.zeros(len(FORECAST_FEATURES_ORDER), dtype=np.float64)
        tot_f_count = np.zeros(len(FORECAST_FEATURES_ORDER), dtype=np.float64)

        extra_h_feats = [
            f for f in self._hindcast_features if f not in FORECAST_FEATURES_ORDER
        ]
        extra_h_cols = [HINDCAST_FEATURES_ORDER.index(f) for f in extra_h_feats]
        tot_h_sum = np.zeros(len(extra_h_feats), dtype=np.float64)
        tot_h_sq_sum = np.zeros(len(extra_h_feats), dtype=np.float64)
        tot_h_count = float(n_b * n_d)

        sf_mmap = self._store['streamflow']
        f_sum_mmap = self._store['forecast_3d_sum']
        f_sq_mmap = self._store['forecast_3d_sq_sum']
        f_cnt_mmap = self._store['forecast_3d_count']
        h2d_mmap = self._store['hindcast_2d']

        # Reduce over date slices using zero-copy 2D/3D slice views in 512-basin blocks
        block_size = 512
        for b_start in range(0, n_total_basins, block_size):
            b_end = min(b_start + block_size, n_total_basins)
            b_sl = slice(b_start, b_end)
            sub_mask = b_mask[b_sl]
            if not np.any(sub_mask):
                continue

            for d_sl in d_slices:
                # Zero-copy slice views into mmap:
                f_sum_view = f_sum_mmap[b_sl, d_sl]
                f_sq_view = f_sq_mmap[b_sl, d_sl]
                f_cnt_view = f_cnt_mmap[b_sl, d_sl]

                tot_f_sum += f_sum_view.sum(axis=1)[sub_mask].sum(axis=0)
                tot_f_sq_sum += f_sq_view.sum(axis=1)[sub_mask].sum(axis=0)
                tot_f_count += f_cnt_view.sum(axis=1, dtype=np.int64)[sub_mask].sum(axis=0).astype(np.float64)

                sf_view = sf_mmap[b_sl, d_sl][sub_mask]
                valid_sf = np.isfinite(sf_view)
                sf_count += int(valid_sf.sum())
                sf_vals = sf_view[valid_sf].astype(np.float64)
                sf_sum += float(sf_vals.sum())
                sf_sq_sum += float(np.square(sf_vals).sum())

                for idx_h, h_col in enumerate(extra_h_cols):
                    h_view = h2d_mmap[b_sl, d_sl, h_col][sub_mask].astype(np.float64)
                    tot_h_sum[idx_h] += float(h_view.sum())
                    tot_h_sq_sum[idx_h] += float(np.square(h_view).sum())

        y_mean = sf_sum / max(sf_count, 1)
        y_var = max(0.0, (sf_sq_sum / max(sf_count, 1)) - (y_mean ** 2))
        y_std = float(np.sqrt(y_var))
        y_val = np.array([y_mean, y_std, y_mean, y_std], dtype=np.float32)
        for k in ('streamflow', 'streamflow_obs', 'streamflow_sim'):
            data_vars[k] = (('parameter',), y_val)

        f_mean_all = tot_f_sum / tot_f_count
        f_var_all = np.maximum(0.0, (tot_f_sq_sum / tot_f_count) - (f_mean_all ** 2))
        f_std_all = np.sqrt(f_var_all)
        for j, feat in enumerate(FORECAST_FEATURES_ORDER):
            f_val = np.array(
                [f_mean_all[j], f_std_all[j], f_mean_all[j], f_std_all[j]],
                dtype=np.float32,
            )
            for k in (feat, f'{feat}_obs', f'{feat}_sim'):
                data_vars[k] = (('parameter',), f_val)

        for idx_h, feat in enumerate(extra_h_feats):
            m = float(tot_h_sum[idx_h] / tot_h_count)
            v = max(0.0, float(tot_h_sq_sum[idx_h] / tot_h_count) - (m ** 2))
            s = float(np.sqrt(v))
            h_val = np.array([m, s, m, s], dtype=np.float32)
            for k in (feat, f'{feat}_obs', f'{feat}_sim'):
                data_vars[k] = (('parameter',), h_val)

        return xr.Dataset(
            data_vars=data_vars,
            coords={'parameter': pd.Index(['center', 'scale', 'mean', 'std'], name='parameter', dtype=object)},
        )

    def _build_valid_sample_index(self) -> None:
        """Construct valid sample mask and compact uint16 sample index arrays with minimal peak RAM."""
        n_b = len(self._basins)
        n_d = len(self._extended_dates)

        max_gd = len(self._store['dates']) - 1
        date_valid_1d = (
            self._sample_in_ext_mask
            & (self._ext_global_date_idx >= self._seq_length)
            & (self._ext_global_date_idx + (self.lead_time - self._min_lead_time) <= max_gd)
        )

        if not self.is_train:
            valid_mask = np.broadcast_to(date_valid_1d[None, :], (n_b, n_d))
        else:
            sf_mmap = self._store['streamflow']
            valid_mask = np.zeros((n_b, n_d), dtype=bool)
            start_offset = self.lead_time - self._min_lead_time - self._predict_last_n + 1  # -1
            end_offset = self.lead_time - self._min_lead_time  # +6
            max_gd = sf_mmap.shape[1] - 1
            offset_indices = [
                np.clip(self._ext_global_date_idx + k, 0, max_gd)
                for k in range(start_offset, end_offset + 1)
            ]
            # Process row-by-row from mmap view without allocating a 550 MB copy of sf_basins
            for i, gb in enumerate(self._global_basin_idx):
                row_valid = ~np.isnan(sf_mmap[gb])
                has_v = np.zeros(n_d, dtype=bool)
                for idx_k in offset_indices:
                    has_v |= row_valid[idx_k]
                valid_mask[i] = has_v & date_valid_1d

        # Count total valid samples first so we can pre-allocate uint16 arrays directly
        # instead of np.nonzero() allocating two 560 MB int64 arrays!
        row_counts = valid_mask.sum(axis=1, dtype=np.int64)
        num_samples = int(row_counts.sum())
        if num_samples == 0:
            if self._period == 'train':
                raise NoTrainDataError
            raise NoEvaluationDataError

        min_dt_b = np.min_scalar_type(n_b)
        min_dt_d = np.min_scalar_type(n_d)
        local_b_compact = np.empty(num_samples, dtype=min_dt_b)
        local_d_compact = np.empty(num_samples, dtype=min_dt_d)
        sample_global_b = np.empty(num_samples, dtype=np.uint16)
        sample_global_d = np.empty(num_samples, dtype=np.uint16)

        pos = 0
        for i in range(n_b):
            cnt = int(row_counts[i])
            if cnt == 0:
                continue
            d_pos = np.flatnonzero(valid_mask[i])
            end_pos = pos + cnt
            local_b_compact[pos:end_pos] = i
            local_d_compact[pos:end_pos] = d_pos
            sample_global_b[pos:end_pos] = self._global_basin_idx[i]
            sample_global_d[pos:end_pos] = self._ext_global_date_idx[d_pos]
            pos = end_pos

        del valid_mask, row_counts
        self._sample_local_b = local_b_compact
        self._sample_global_b = sample_global_b
        self._sample_global_d = sample_global_d

        self._sample_index = SampleIndexer(
            (('basin', local_b_compact), ('date', local_d_compact))
        )
        self._num_samples = num_samples

    def __len__(self) -> int:
        return self._num_samples

    @property
    def min_lead_time(self) -> int:
        return self._min_lead_time

    def get_batch(self, indices: list[int] | np.ndarray) -> dict[str, Any]:
        """Vectorized batch construction directly returning PyTorch tensors."""
        idx_arr = np.asarray(indices, dtype=np.int64)
        B = len(idx_arr)
        gb = self._sample_global_b[idx_arr].astype(np.int32)  # (B,)
        gd = self._sample_global_d[idx_arr].astype(np.int32)  # (B,)
        lb = self._sample_local_b[idx_arr].astype(np.int64)   # (B,)

        # 1. Static features: (B, 200)
        x_s_raw = self._store['statics'][gb][:, self._static_col_idx]
        x_s = torch.from_numpy((x_s_raw - self._s_center) / self._s_scale)

        # 2. Hindcast 2D features: [gd - 365 .. gd - 1], shape (B, 365, 9)
        h_offsets = np.arange(-self._seq_length, 0, dtype=np.int32)
        h_dates = gd[:, None] + h_offsets[None, :]  # (B, 365)
        h_raw = self._store['hindcast_2d'][gb[:, None], h_dates][
            :, :, self._hindcast_col_idx
        ]
        h_scaled = (h_raw - self._h_center) / self._h_scale  # (B, 365, 9)
        h_tensor = torch.from_numpy(h_scaled).unsqueeze(-1)  # (B, 365, 9, 1)
        x_d_hindcast: dict[str, torch.Tensor] = {
            feat: h_tensor[:, :, i, :]
            for i, feat in enumerate(self._hindcast_features)
        }
        if self._hindcast_counter is not None:
            hc = torch.from_numpy(self._hindcast_counter).unsqueeze(0).expand(B, -1, -1).contiguous()
            x_d_hindcast['hindcast_counter'] = hc

        # 3. Forecast features: overlap (365 steps) + lead_time (7 steps) = 372 steps
        f_overlap_scaled = h_scaled[:, :, self._forecast_in_hindcast_col_idx]  # (B, 365, 7)

        f_lead_raw = np.empty((B, self.lead_time, len(self._forecast_features)), dtype=np.float32)
        post_mask = gd >= POST2016_OFFSET
        if post_mask.all():
            f_lead_raw = self._store['forecast_3d_post2016'][
                gb, gd - POST2016_OFFSET
            ][:, : self.lead_time, self._forecast_col_idx]
        elif not post_mask.any():
            l_offsets = np.arange(0, self.lead_time, dtype=np.int32)
            l_dates = gd[:, None] + l_offsets[None, :]
            f_lead_raw = self._store['hindcast_2d'][gb[:, None], l_dates][
                :, :, self._forecast_in_hindcast_col_idx
            ]
        else:
            if post_mask.any():
                p_idx = np.flatnonzero(post_mask)
                f_lead_raw[p_idx] = self._store['forecast_3d_post2016'][
                    gb[p_idx], gd[p_idx] - POST2016_OFFSET
                ][:, : self.lead_time, self._forecast_col_idx]
            pre_idx = np.flatnonzero(~post_mask)
            if len(pre_idx) > 0:
                l_offsets = np.arange(0, self.lead_time, dtype=np.int32)
                l_dates = gd[pre_idx, None] + l_offsets[None, :]
                f_lead_raw[pre_idx] = self._store['hindcast_2d'][
                    gb[pre_idx, None], l_dates
                ][:, :, self._forecast_in_hindcast_col_idx]

        f_lead_scaled = (f_lead_raw - self._f_center) / self._f_scale  # (B, 7, 7)
        if self._forecast_overlap:
            f_full_scaled = np.concatenate([f_overlap_scaled, f_lead_scaled], axis=1)  # (B, 372, 7)
        else:
            f_full_scaled = f_lead_scaled
        f_tensor = torch.from_numpy(f_full_scaled).unsqueeze(-1)  # (B, 372, 7, 1)
        x_d_forecast: dict[str, torch.Tensor] = {
            feat: f_tensor[:, :, i, :]
            for i, feat in enumerate(self._forecast_features)
        }
        if self._forecast_counter is not None:
            fc = torch.from_numpy(self._forecast_counter).unsqueeze(0).expand(B, -1, -1).contiguous()
            x_d_forecast['forecast_counter'] = fc

        # 4. Target y and date: [gd + lead_time - min_lead_time - seq_length + 1 .. gd + lead_time - min_lead_time]
        end_shift = self.lead_time - self._min_lead_time  # +6
        y_offsets = np.arange(
            end_shift - (self._seq_length - 1), end_shift + 1, dtype=np.int32
        )  # [-358 .. +6]
        y_dates_idx = gd[:, None] + y_offsets[None, :]  # (B, 365)
        y_raw = self._store['streamflow'][gb[:, None], y_dates_idx]  # (B, 365)
        y_scaled = ((y_raw - self._y_center) / self._y_scale)[:, :, None]  # (B, 365, 1)
        y_tensor = torch.from_numpy(y_scaled)
        dates_batch = self._store['dates_np'][y_dates_idx]  # (B, 365) datetime64[ns]

        batch: dict[str, Any] = {
            'date': dates_batch,
            'x_s': x_s,
            'x_d_hindcast': x_d_hindcast,
            'x_d_forecast': x_d_forecast,
            'y': y_tensor,
            'basin_index': torch.from_numpy(lb),
        }
        if self._per_basin_target_stds is not None:
            stds = self._per_basin_target_stds[lb][:, None, None]
            batch['per_basin_target_stds'] = torch.from_numpy(stds)
        return batch

    def __getitem__(self, item: int) -> dict[str, Any]:
        """Single-item retrieval for compatibility with stock interfaces."""
        if item >= self._num_samples:
            raise IndexError(
                f'Requested index {item} > the total number of samples {self._num_samples}.'
            )
        if item < 0 or item % 1 != 0:
            raise ValueError(f'Invalid index {item}.')
        b = self.get_batch([item])
        return {
            'date': b['date'][0],
            'x_s': b['x_s'][0].numpy(),
            'x_d_hindcast': {k: v[0].numpy() for k, v in b['x_d_hindcast'].items()},
            'x_d_forecast': {k: v[0].numpy() for k, v in b['x_d_forecast'].items()},
            'y': b['y'][0].numpy(),
            'basin_index': np.array(b['basin_index'][0].item(), dtype=np.int64),
        }

    def _get_period_dates(
        self, cfg: Config
    ) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
        if self._period == 'train':
            start_dates, end_dates = cfg.train_start_date, cfg.train_end_date
        elif self._period == 'test':
            start_dates, end_dates = cfg.test_start_date, cfg.test_end_date
        elif self._period == 'validation':
            start_dates, end_dates = (
                cfg.validation_start_date,
                cfg.validation_end_date,
            )
        else:
            raise ValueError(f'Unknown period {self._period}')
        return start_dates, end_dates

    def _union_ranges(
        self, start_dates: list[pd.Timestamp], end_dates: list[pd.Timestamp]
    ) -> pd.DatetimeIndex:
        ranges = [pd.date_range(s, e) for s, e in zip(start_dates, end_dates)]
        return functools.reduce(pd.Index.union, ranges)

    @staticmethod
    def collate_fn(
        samples: list[dict[str, torch.Tensor | np.ndarray | dict]],
    ) -> dict[str, torch.Tensor | np.ndarray | dict]:
        batch: dict[str, Any] = {}
        if not samples:
            return batch
        for k in samples[0]:
            if k.startswith('x_d'):
                batch[k] = {
                     name: torch.stack([torch.as_tensor(s[k][name]) for s in samples], dim=0)
                    for name in samples[0][k]
                }
            elif k == 'date':
                batch[k] = np.stack([s[k] for s in samples], axis=0)
            else:
                batch[k] = torch.stack([torch.as_tensor(s[k]) for s in samples], dim=0)
        return batch


class FastMultimetDataLoader(torch.utils.data.DataLoader):
    """Vectorized DataLoader that calls `FastMultimet.get_batch(indices)` with background prefetching."""

    def __init__(self, *args, lazy_load: bool = True, logging_level: int = logging.INFO, prefetch_batches: int = 3, **kwargs):
        kwargs['num_workers'] = 0
        self._prefetch_batches = prefetch_batches
        super().__init__(*args, **kwargs)

    def __iter__(self):
        if isinstance(self.dataset, FastMultimet):
            if self._prefetch_batches <= 0:
                for indices in self.batch_sampler:
                    yield self.dataset.get_batch(indices)
                return

            import queue
            import threading

            q: queue.Queue = queue.Queue(maxsize=self._prefetch_batches)
            sentinel = object()
            stop_event = threading.Event()

            def _worker():
                try:
                    for indices in self.batch_sampler:
                        if stop_event.is_set():
                            break
                        batch = self.dataset.get_batch(indices)
                        while not stop_event.is_set():
                            try:
                                q.put(batch, timeout=0.5)
                                break
                            except queue.Full:
                                continue
                except Exception as e:
                    q.put(e)
                finally:
                    q.put(sentinel)

            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            try:
                while True:
                    item = q.get()
                    if item is sentinel:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                stop_event.set()
        else:
            yield from super().__iter__()


def install_fast_multimet() -> None:
    """Monkey-patch `googlehydrology` (`model`) to use `FastMultimet` and `FastMultimetDataLoader`."""
    import model.datasetzoo as datasetzoo
    import model.datasetzoo.multimet as multimet_mod
    import model.evaluation.tester as tester_mod

    multimet_mod.Multimet = FastMultimet
    multimet_mod.MultimetDataLoader = FastMultimetDataLoader
    tester_mod.MultimetDataLoader = FastMultimetDataLoader

    orig_get_dataset = datasetzoo.get_dataset

    def _fast_get_dataset(
        cfg: Config,
        is_train: bool,
        period: str,
        basins: list[str] | None = None,
        compute_scaler: bool = True,
    ) -> Dataset:
        if cfg.dataset.lower() == 'multimet':
            return FastMultimet(
                cfg=cfg,
                is_train=is_train,
                period=period,
                basins=basins,
                compute_scaler=compute_scaler,
            )
        return orig_get_dataset(
            cfg=cfg,
            is_train=is_train,
            period=period,
            basins=basins,
            compute_scaler=compute_scaler,
        )

    datasetzoo.get_dataset = _fast_get_dataset
    tester_mod.get_dataset = _fast_get_dataset
    import model.training.basetrainer as basetrainer_mod
    basetrainer_mod.get_dataset = _fast_get_dataset


@torch.inference_mode()
def fast_evaluate_period(
    model: torch.nn.Module,
    dataset: FastMultimet,
    device: torch.device,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DatetimeIndex]:
    """Evaluate `model` over all samples in `dataset` (`is_train=False`) using packed GPU batches.

    Unwraps `torch.compile` (`_orig_mod`) so evaluation never triggers TorchInductor
    recompilations or CUDA Graph private pool allocations on partial tail batches,
    and uses `FastMultimetDataLoader` background prefetching to overlap CPU batch
    assembly with GPU inference.
    """
    raw_model = getattr(model, '_orig_mod', model)
    raw_model.eval()
    n_basins = len(dataset._basins)
    n_dates = len(dataset._sample_dates)
    predict_last_n = dataset._predict_last_n
    total_samples = len(dataset)
    assert total_samples == n_basins * n_dates, (
        f'Expected {n_basins} * {n_dates} = {n_basins * n_dates} samples, got {total_samples}'
    )

    preds_flat = np.empty((total_samples, predict_last_n), dtype=np.float32)
    obs_flat = np.empty((total_samples, predict_last_n), dtype=np.float32)

    sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = torch.utils.data.BatchSampler(
        sampler, batch_size=batch_size, drop_last=False
    )
    loader = FastMultimetDataLoader(
        dataset, batch_sampler=batch_sampler, prefetch_batches=3
    )

    offset = 0
    for batch in loader:
        bsz = batch['x_s'].shape[0]
        for k in ('x_d_hindcast', 'x_d_forecast'):
            batch[k] = {
                name: t.to(device, non_blocking=True)
                for name, t in batch[k].items()
            }
        batch['x_s'] = batch['x_s'].to(device, non_blocking=True)
        batch['y'] = batch['y'].to(device, non_blocking=True)
        batch['basin_index'] = batch['basin_index'].to(device, non_blocking=True)

        with autocast(device.type, enabled=(device.type == 'cuda')):
            batch = raw_model.pre_model_hook(batch, is_train=False)
            out = raw_model(batch)

        y_hat = out['y_hat'][:, -predict_last_n:, 0].detach().cpu().numpy().astype(np.float32)
        y_obs = batch['y'][:, -predict_last_n:, 0].detach().cpu().numpy().astype(np.float32)
        preds_flat[offset : offset + bsz] = y_hat
        obs_flat[offset : offset + bsz] = y_obs
        offset += bsz

    # Unscale using streamflow center and scale
    y_center = dataset._y_center
    y_scale = dataset._y_scale
    preds_unscaled = preds_flat * y_scale + y_center
    obs_unscaled = obs_flat * y_scale + y_center

    if 'streamflow' in dataset.cfg.clip_targets_to_zero:
        preds_unscaled = np.where(preds_unscaled < 0.0, 0.0, preds_unscaled)

    return (
        preds_unscaled.reshape(n_basins, n_dates, predict_last_n),
        obs_unscaled.reshape(n_basins, n_dates, predict_last_n),
        dataset._basins,
        dataset._sample_dates,
    )
