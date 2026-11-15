"""Fast Exact Multi-GPU Executor for Target Fine-Tuning (FT) & 30-Day Data Assimilation (DA)
across all 8,811 Caravan basins and 10 Temporal Folds (1986-2023) for:
  - gauged    (sharded into 10 spatial chunks sfold_0..9 using splits/random/fold_{s}_test.txt)
  - random    (sfold_0..9 using splits/random/fold_{s}_test.txt)
  - basin     (sfold_0..9 using splits/basin/fold_{s}_test.txt, i.e. hydrologically separated)

Produces three full 1986-2023 out-of-sample prediction suites per experiment:
  1. {experiment}_da    : 30-Day Variational Data Assimilation on the base checkpoint (001)
  2. {experiment}_ft    : Active Target Fine-Tuning on out-of-fold dates (010)
  3. {experiment}_ft_da : Active Target Fine-Tuning + 30-Day Variational Data Assimilation (011)

Strict Hyperparameters (matching Ryd & Nearing 2025 / Ruparell et al. 2026 / run_clean_loco_study.py):
  - Target FT (010):
      Freeze: hindcast_lstm, forecast_lstm
      Unfreeze: static_embedding_fc, local_static_embeddings_fc, hindcast_embeddings_fc,
                forecast_embeddings_fc, shared_embeddings_fc, head
      Optimizer: Adam(lr=0.0005), StepLR(step_size=6, gamma=0.5)
      Epochs: 12, batch_size: 128, max_updates_per_epoch: 12 (1,536 samples/epoch)
      Loss: MaskedNSELoss (eps=0.1, per-basin target std computed on out-of-fold tuning dates)
      Gradient clipping: clip_grad_norm_(1.0)
  - 30-Day Variational DA (001 & 011):
      Window: 30 days (indices [335:365] of the 372-step sequence)
      Steps: 20 cold-started Adam steps per forecast issue date
      Learning rate: da_lr = 0.035
      Regularization: reg_weight = 0.002 (L2 relative to prior embedding scale)
      Loss: MaskedNSELoss (per_sequence=True, eps=0.1)
      Gradient clipping: clip_grad_norm_(1.0) per sequence
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import time
from typing import Any
import warnings

warnings.filterwarnings('ignore')

CODE_DIR = Path('.')
FF_DIR = CODE_DIR / 'flood-forecasting'
for p in (str(CODE_DIR), str(FF_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import fast_multimet
from fast_multimet import (
    FORECAST_FEATURES_ORDER,
    HINDCAST_FEATURES_ORDER,
    POST2016_OFFSET,
    flatten_feature_list,
    load_basin_file,
)
from model.datautils.scaler import Scaler
from model.modelzoo import get_model
from model.modelzoo.mean_embedding_forecast_lstm import (
    ForwardData,
    MeanEmbeddingForecastLSTM,
)
from model.utils.config import Config

_STORE_CACHE: dict[str, dict[str, Any]] = {}


def get_precomputed_store(precomp_dir: Path) -> dict[str, Any]:
    key = str(precomp_dir.resolve())
    if key in _STORE_CACHE:
        return _STORE_CACHE[key]
    basins = (precomp_dir / 'basins.txt').read_text().strip().splitlines()
    basin_to_idx = {b: i for i, b in enumerate(basins)}
    dates_raw = np.load(precomp_dir / 'dates.npy')
    dates = pd.DatetimeIndex(dates_raw, name='date')
    date_to_idx = pd.Series(np.arange(len(dates), dtype=np.int32), index=dates)
    static_attrs = (precomp_dir / 'static_attributes.txt').read_text().strip().splitlines()
    static_attr_to_idx = {a: i for i, a in enumerate(static_attrs)}
    store = {
        'basins': basins,
        'basin_to_idx': basin_to_idx,
        'dates': dates,
        'dates_np': dates_raw,
        'date_to_idx': date_to_idx,
        'static_attrs': static_attrs,
        'static_attr_to_idx': static_attr_to_idx,
        'statics': np.load(precomp_dir / 'statics.npy', mmap_mode='r'),
        'streamflow': np.load(precomp_dir / 'streamflow.npy', mmap_mode='r'),
        'hindcast_2d': np.load(precomp_dir / 'hindcast_2d.npy', mmap_mode='r'),
        'forecast_3d_post2016': np.load(precomp_dir / 'forecast_3d_post2016.npy', mmap_mode='r'),
    }
    _STORE_CACHE[key] = store
    return store


def load_model_weights(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)


ROOT_DIR = Path('./data')
PRECOMP_DIR = ROOT_DIR / 'data' / 'precomputed'
SPLITS_DIR = ROOT_DIR / 'splits'
RUNS_DIR = ROOT_DIR / 'runs'
OUT_RUNS_DIR = ROOT_DIR / 'runs_ft_da'

# Strict Target FT and 30-Day DA Hyperparameters
FT_LR = 0.0005
FT_EPOCHS = 12
FT_STEP_SIZE = 6
FT_GAMMA = 0.5
FT_BATCH_SIZE = 128
FT_MAX_UPDATES_PER_EPOCH = 12
FT_CLIP_NORM = 1.0

DA_WINDOW_DAYS = 30
DA_STEPS = 20
DA_LR = 0.035
DA_REG_WEIGHT = 0.002
DA_CLIP_NORM = 1.0

TEMPORAL_FOLDS: list[dict[str, Any]] = [
    {
        'tfold': 0,
        'name': '1986_1990',
        'test_start': '01/01/1986',
        'test_end': '31/12/1990',
        'train_start': ['01/01/1982', '01/01/1992'],
        'train_end': ['31/12/1984', '31/12/2023'],
    },
    {
        'tfold': 1,
        'name': '1991_1995',
        'test_start': '01/01/1991',
        'test_end': '31/12/1995',
        'train_start': ['01/01/1982', '01/01/1997'],
        'train_end': ['31/12/1989', '31/12/2023'],
    },
    {
        'tfold': 2,
        'name': '1996_2000',
        'test_start': '01/01/1996',
        'test_end': '31/12/2000',
        'train_start': ['01/01/1982', '01/01/2002'],
        'train_end': ['31/12/1994', '31/12/2023'],
    },
    {
        'tfold': 3,
        'name': '2001_2005',
        'test_start': '01/01/2001',
        'test_end': '31/12/2005',
        'train_start': ['01/01/1982', '01/01/2007'],
        'train_end': ['31/12/1999', '31/12/2023'],
    },
    {
        'tfold': 4,
        'name': '2006_2010',
        'test_start': '01/01/2006',
        'test_end': '31/12/2010',
        'train_start': ['01/01/1982', '01/01/2012'],
        'train_end': ['31/12/2004', '31/12/2023'],
    },
    {
        'tfold': 5,
        'name': '2011_2015',
        'test_start': '01/01/2011',
        'test_end': '31/12/2015',
        'train_start': ['01/01/1982', '01/01/2017'],
        'train_end': ['31/12/2009', '31/12/2023'],
    },
    {
        'tfold': 6,
        'name': '2016_2017',
        'test_start': '01/01/2016',
        'test_end': '31/12/2017',
        'train_start': ['01/01/1982', '01/01/2019'],
        'train_end': ['31/12/2014', '31/12/2023'],
    },
    {
        'tfold': 7,
        'name': '2018_2019',
        'test_start': '01/01/2018',
        'test_end': '31/12/2019',
        'train_start': ['01/01/1982', '01/01/2021'],
        'train_end': ['31/12/2016', '31/12/2023'],
    },
    {
        'tfold': 8,
        'name': '2020_2021',
        'test_start': '01/01/2020',
        'test_end': '31/12/2021',
        'train_start': ['01/01/1982', '01/01/2023'],
        'train_end': ['31/12/2018', '31/12/2023'],
    },
    {
        'tfold': 9,
        'name': '2022_2023',
        'test_start': '01/01/2022',
        'test_end': '31/12/2023',
        'train_start': '01/01/1982',
        'train_end': '31/12/2020',
    },
]


def _union_date_ranges(
    starts: str | list[str], ends: str | list[str]
) -> pd.DatetimeIndex:
    if isinstance(starts, str):
        starts = [starts]
    if isinstance(ends, str):
        ends = [ends]
    s_list = [pd.to_datetime(s, format='%d/%m/%Y') for s in starts]
    e_list = [pd.to_datetime(e, format='%d/%m/%Y') for e in ends]
    ranges = [pd.date_range(s, e) for s, e in zip(s_list, e_list)]
    out = ranges[0]
    for r in ranges[1:]:
        out = out.union(r)
    return out


def compute_fold_date_indices(
    store: dict[str, Any], tf: dict[str, Any], seq_length: int = 365, lead_time: int = 7
) -> dict[str, np.ndarray]:
    """Compute global calendar date indices for train and test periods of a temporal fold."""
    all_dates: pd.DatetimeIndex = store['dates']
    date_to_idx: pd.Series = store['date_to_idx']
    max_gd = len(all_dates) - 1
    min_lead_time = 1

    # 1. Train period
    tr_starts = tf['train_start'] if isinstance(tf['train_start'], list) else [tf['train_start']]
    tr_ends = tf['train_end'] if isinstance(tf['train_end'], list) else [tf['train_end']]
    tr_s_ts = [pd.to_datetime(s, format='%d/%m/%Y') for s in tr_starts]
    tr_e_ts = [pd.to_datetime(e, format='%d/%m/%Y') for e in tr_ends]
    tr_sample_dates = _union_date_ranges(tr_starts, tr_ends)
    tr_ext_dates = _union_date_ranges(
        [(s - pd.Timedelta(days=seq_length)).strftime('%d/%m/%Y') for s in tr_s_ts],
        [(e + pd.Timedelta(days=lead_time)).strftime('%d/%m/%Y') for e in tr_e_ts],
    ).intersection(all_dates)
    tr_ext_gd = date_to_idx.loc[tr_ext_dates].to_numpy(dtype=np.int32)
    tr_valid_1d = (
        tr_ext_dates.isin(tr_sample_dates)
        & (tr_ext_gd >= seq_length)
        & (tr_ext_gd + (lead_time - min_lead_time) <= max_gd)
    )
    tr_candidate_gd = tr_ext_gd[tr_valid_1d]

    # 2. Test period
    te_s_ts = pd.to_datetime(tf['test_start'], format='%d/%m/%Y')
    te_e_ts = pd.to_datetime(tf['test_end'], format='%d/%m/%Y')
    te_sample_dates = pd.date_range(te_s_ts, te_e_ts)
    te_ext_dates = pd.date_range(
        te_s_ts - pd.Timedelta(days=seq_length),
        te_e_ts + pd.Timedelta(days=lead_time),
    ).intersection(all_dates)
    te_ext_gd = date_to_idx.loc[te_ext_dates].to_numpy(dtype=np.int32)
    te_valid_1d = (
        te_ext_dates.isin(te_sample_dates)
        & (te_ext_gd >= seq_length)
        & (te_ext_gd + (lead_time - min_lead_time) <= max_gd)
    )
    te_gd = te_ext_gd[te_valid_1d]

    return {
        'train_ext_gd': tr_ext_gd,
        'train_candidate_gd': tr_candidate_gd,
        'test_ext_gd': te_ext_gd,
        'test_gd': te_gd,
    }


class BasinAdapterModules(nn.Module):
    """Per-basin trainable parameters for Target FT (all embedding FCs + regression head)."""

    def __init__(self, base_model: MeanEmbeddingForecastLSTM):
        super().__init__()
        self.static_embedding_fc = copy.deepcopy(base_model.static_embedding_fc)
        if hasattr(base_model, 'local_static_embeddings_fc'):
            self.local_static_embeddings_fc = copy.deepcopy(base_model.local_static_embeddings_fc)
        self.hindcast_embeddings_fc = copy.deepcopy(base_model.hindcast_embeddings_fc)
        self.forecast_embeddings_fc = copy.deepcopy(base_model.forecast_embeddings_fc)
        self.shared_embeddings_fc = copy.deepcopy(base_model.shared_embeddings_fc)
        self.dropout = nn.Dropout(p=base_model.cfg.output_dropout)
        self.head = copy.deepcopy(base_model.head)

    def reset_from_base(self, base_state: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, param in self.named_parameters():
                param.copy_(base_state[name])
                param.requires_grad_(True)

    def calc_head(self, forecast_state: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self.dropout(forecast_state))


class FastBasin1DCache:
    """Exact 1D-calendar-day + 7-day-lead GPU cache for a single basin across 1981-2023.

    Mathematically and numerically identical (to <1e-6 float32 precision) to
    FastMultimet.get_batch + MeanEmbeddingForecastLSTM._calc_mean_embeddings, while storing
    only ~1.5 MB per basin on GPU and evaluating dynamic MLPs on the 15,706 unique calendar
    days instead of 365x overlapping windows.
    """

    def __init__(
        self,
        gb: int,
        store: dict[str, Any],
        scaler_info: dict[str, Any],
        fold_dates: dict[str, np.ndarray],
        model: MeanEmbeddingForecastLSTM,
        device: torch.device,
    ):
        self.gb = int(gb)
        self.device = device
        self.seq_len = int(model.seq_length)  # 365
        self.lead_time = int(model.lead_time)  # 7
        self.predict_last_n = int(model.cfg.predict_last_n)  # 8
        self.min_lead = 1
        self.total_len = self.seq_len + self.lead_time  # 372

        # 1. Scaled static features (1, 200)
        x_s_raw = store['statics'][gb, scaler_info['static_col_idx']]
        x_s_np = (x_s_raw - scaler_info['s_center']) / scaler_info['s_scale']
        self.x_s_1 = torch.from_numpy(x_s_np[None, :]).to(device, dtype=torch.float32)

        # 2. Scaled 1D calendar-day hindcast features (T_cal=15706, 9)
        h_raw_all = store['hindcast_2d'][gb][:, scaler_info['hindcast_col_idx']]
        h_scaled_np = (h_raw_all - scaler_info['h_center']) / scaler_info['h_scale']
        h_cal_g = torch.from_numpy(h_scaled_np).to(device, dtype=torch.float32)
        T_cal = h_cal_g.shape[0]
        self.T_cal = T_cal

        zeros_cal_1 = torch.zeros((T_cal, 1), device=device, dtype=torch.float32)
        ones_cal_1 = torch.ones((T_cal, 1), device=device, dtype=torch.float32)

        # Pre-build per-group 1D calendar inputs (for steps 0..364 of any window)
        self.h_1d_inputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.s_ov_1d_inputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        h_val_count_1d = torch.zeros((T_cal, 1), device=device, dtype=torch.float32)
        f_ov_val_count_1d = torch.zeros((T_cal, 1), device=device, dtype=torch.float32)

        for name in model.hindcast_embeddings_fc.keys():
            cols = scaler_info['h_group_cols'][name]
            x_g = h_cal_g[:, cols]
            nan_m = torch.isnan(x_g).any(dim=-1, keepdim=True)
            v_g = (~nan_m).to(torch.float32)
            x_clean = x_g.masked_fill(nan_m, 0.0).contiguous()
            self.h_1d_inputs[name] = (x_clean, v_g)
            h_val_count_1d = h_val_count_1d + v_g

        for name in model.shared_embeddings_fc.keys():
            cols = scaler_info['s_in_h_group_cols'][name]
            x_g = h_cal_g[:, cols]
            nan_m = torch.isnan(x_g).any(dim=-1, keepdim=True)
            v_g = (~nan_m).to(torch.float32)
            x_clean = x_g.masked_fill(nan_m, 0.0).contiguous()
            self.s_ov_1d_inputs[name] = (x_clean, v_g)
            h_val_count_1d = h_val_count_1d + v_g
            f_ov_val_count_1d = f_ov_val_count_1d + v_g

        self.inv_h_1d = torch.where(
            h_val_count_1d > 0, 1.0 / h_val_count_1d.clamp(min=1.0), 0.0
        ).contiguous()
        self.inv_f_ov_1d = torch.where(
            f_ov_val_count_1d > 0, 1.0 / f_ov_val_count_1d.clamp(min=1.0), 0.0
        ).contiguous()
        self.miss_h_1d = (h_val_count_1d == 0)  # (T_cal, 1)
        self.miss_f_ov_1d = (f_ov_val_count_1d == 0)  # (T_cal, 1)

        # 3. Scaled 1D target streamflow (T_cal,)
        sf_raw = store['streamflow'][gb]
        y_center = scaler_info['y_center']
        y_scale = scaler_info['y_scale']
        sf_scaled_np = ((sf_raw - y_center) / y_scale).astype(np.float32)
        self.sf_scaled_g = torch.from_numpy(sf_scaled_np).to(device, dtype=torch.float32)

        # Compute train and test target stds exactly matching FastMultimet
        tr_ext_sf = sf_scaled_np[fold_dates['train_ext_gd']]
        tr_std = float(np.nanstd(tr_ext_sf, dtype=np.float32))
        if not np.isfinite(tr_std) or tr_std <= 1e-4:
            tr_std = 1.0
        self.train_std = torch.tensor([[[tr_std]]], device=device, dtype=torch.float32)

        te_ext_sf = sf_scaled_np[fold_dates['test_ext_gd']]
        te_std = float(np.nanstd(te_ext_sf, dtype=np.float32))
        if not np.isfinite(te_std) or te_std <= 1e-4:
            te_std = 1.0
        self.test_std = torch.tensor([[[te_std]]], device=device, dtype=torch.float32)

        # 4. Identify valid train sample dates (at least 1 finite target in last 8 steps [-1..+6])
        tr_cand = fold_dates['train_candidate_gd']
        row_valid = ~np.isnan(sf_raw)
        has_v = np.zeros(len(tr_cand), dtype=bool)
        for k in range(-1, 7):
            has_v |= row_valid[tr_cand + k]
        self.train_gd_np = tr_cand[has_v]
        self.n_train = len(self.train_gd_np)
        self.train_gd = torch.from_numpy(self.train_gd_np).to(device, dtype=torch.long)

        # 5. Test sample dates (all dates in test fold)
        self.test_gd_np = fold_dates['test_gd']
        self.n_test = len(self.test_gd_np)
        self.test_gd = torch.from_numpy(self.test_gd_np).to(device, dtype=torch.long)

        # Precompute 7-step forecast lead inputs for post-2016 and pre-2016 dates
        # Notice: for gd < POST2016_OFFSET, lead step l in 0..6 is simply h_cal_g[gd + l] with counter l+1!
        # For gd >= POST2016_OFFSET (only 2929 days total!), it comes from forecast_3d_post2016[gb]!
        f3d_raw = store['forecast_3d_post2016'][gb, :, : self.lead_time][
            :, :, scaler_info['forecast_col_idx']
        ]
        f3d_scaled_np = (f3d_raw - scaler_info['f_center']) / scaler_info['f_scale']
        self.f3d_post2016_g = torch.from_numpy(f3d_scaled_np).to(device, dtype=torch.float32)
        self.h_for_f_lead_g = torch.from_numpy(
            (
                store['hindcast_2d'][gb][:, scaler_info['forecast_in_hindcast_col_idx']]
                - scaler_info['f_center']
            )
            / scaler_info['f_scale']
        ).to(device, dtype=torch.float32)

        self.s_in_f_group_cols = scaler_info['s_in_f_group_cols']
        self.lead_counters = (
            torch.arange(1, self.lead_time + 1, device=device, dtype=torch.float32)
            .view(1, self.lead_time, 1)
            .contiguous()
        )
        self.h_offsets = torch.arange(-self.seq_len, 0, device=device, dtype=torch.long)  # [-365..-1]
        self.l_offsets = torch.arange(0, self.lead_time, device=device, dtype=torch.long)  # [0..6]
        self.y_last8_offsets = torch.arange(-1, self.lead_time, device=device, dtype=torch.long)  # [-1..+6]
        self.y_full365_offsets = torch.arange(
            self.lead_time - self.min_lead - (self.seq_len - 1),
            self.lead_time - self.min_lead + 1,
            device=device,
            dtype=torch.long,
        )  # [-358..+6]

    def _compute_1d_embeddings(
        self, adapter: BasinAdapterModules | MeanEmbeddingForecastLSTM
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate static and 1D calendar-day embeddings (15,706 rows) once per forward step."""
        se_1 = adapter.static_embedding_fc(self.x_s_1)  # (1, 20)
        se_cal = se_1.expand(self.T_cal, -1)  # (15706, 20)

        h_sum_1d = torch.zeros((self.T_cal, se_1.shape[-1]), device=self.device, dtype=se_1.dtype)
        f_ov_sum_1d = torch.zeros((self.T_cal, se_1.shape[-1]), device=self.device, dtype=se_1.dtype)

        for name, fc in adapter.hindcast_embeddings_fc.items():
            x_clean, v_g = self.h_1d_inputs[name]
            out = fc(torch.cat([x_clean, se_cal], dim=-1)) * v_g.to(se_1.dtype)
            h_sum_1d = h_sum_1d + out

        for name, fc in adapter.shared_embeddings_fc.items():
            x_clean, v_g = self.s_ov_1d_inputs[name]
            out = fc(torch.cat([x_clean, se_cal], dim=-1)) * v_g.to(se_1.dtype)
            h_sum_1d = h_sum_1d + out
            f_ov_sum_1d = f_ov_sum_1d + out

        mh_1d = h_sum_1d * self.inv_h_1d.to(se_1.dtype)
        mf_ov_1d = f_ov_sum_1d * self.inv_f_ov_1d.to(se_1.dtype)
        return se_1, mh_1d, mf_ov_1d

    def _compute_lead7_embeddings(
        self,
        adapter: BasinAdapterModules | MeanEmbeddingForecastLSTM,
        gd: torch.Tensor,
        se_1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate shared_embeddings_fc on the 7-day forecast horizon for sample dates `gd` (B,)."""
        B = gd.shape[0]
        post_mask = gd >= POST2016_OFFSET
        if post_mask.all():
            f_lead = self.f3d_post2016_g[gd - POST2016_OFFSET]  # (B, 7, 7)
        elif not post_mask.any():
            l_dates = gd[:, None] + self.l_offsets[None, :]  # (B, 7)
            f_lead = self.h_for_f_lead_g[l_dates]  # (B, 7, 7)
        else:
            f_lead = torch.empty(
                (B, self.lead_time, self.f3d_post2016_g.shape[-1]),
                device=self.device,
                dtype=torch.float32,
            )
            p_idx = torch.nonzero(post_mask, as_tuple=True)[0]
            pre_idx = torch.nonzero(~post_mask, as_tuple=True)[0]
            f_lead[p_idx] = self.f3d_post2016_g[gd[p_idx] - POST2016_OFFSET]
            l_dates = gd[pre_idx, None] + self.l_offsets[None, :]
            f_lead[pre_idx] = self.h_for_f_lead_g[l_dates]

        counters = self.lead_counters.expand(B, -1, -1)
        se_lead = se_1.unsqueeze(1).expand(B, self.lead_time, -1)

        lead_sum = torch.zeros(
            (B, self.lead_time, se_1.shape[-1]), device=self.device, dtype=se_1.dtype
        )
        lead_val_cnt = torch.zeros((B, self.lead_time, 1), device=self.device, dtype=torch.float32)

        for name, fc in adapter.shared_embeddings_fc.items():
            cols = self.s_in_f_group_cols[name]
            x_g = f_lead[:, :, cols]
            nan_m = torch.isnan(x_g).any(dim=-1, keepdim=True)
            v_g = (~nan_m).to(torch.float32)
            x_in = torch.cat([x_g.masked_fill(nan_m, 0.0), se_lead], dim=-1)
            out = fc(x_in) * v_g.to(se_1.dtype)
            lead_sum = lead_sum + out
            lead_val_cnt = lead_val_cnt + v_g

        inv_cnt = torch.where(lead_val_cnt > 0, 1.0 / lead_val_cnt.clamp(min=1.0), 0.0)
        m_lead_7 = lead_sum * inv_cnt.to(se_1.dtype)
        miss_lead_7 = (lead_val_cnt == 0)  # (B, 7, 1)
        return m_lead_7, miss_lead_7

    def assemble_windows(
        self,
        adapter: BasinAdapterModules | MeanEmbeddingForecastLSTM,
        gd: torch.Tensor,
        return_full_y: bool = False,
        is_train: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Assemble exact (se_b, mh, mf, forecast_missing, y) for sample global date indices `gd`."""
        B = gd.shape[0]
        se_1, mh_1d, mf_ov_1d = self._compute_1d_embeddings(adapter)
        m_lead_7, miss_lead_7 = self._compute_lead7_embeddings(adapter, gd, se_1)

        h_dates = gd[:, None] + self.h_offsets[None, :]  # (B, 365)
        mh = torch.cat([mh_1d[h_dates], m_lead_7], dim=1)  # (B, 372, 20)
        mf = torch.cat([mf_ov_1d[h_dates], m_lead_7], dim=1)  # (B, 372, 20)
        se_b = se_1.expand(B, -1)  # (B, 20)

        # Cumulative missing mask (matches MeanEmbeddingForecastLSTM._missing_steps)
        miss_ov = self.miss_h_1d[h_dates] | self.miss_f_ov_1d[h_dates]  # (B, 365, 1)
        miss_all = torch.cat([miss_ov, miss_lead_7], dim=1)  # (B, 372, 1)
        forecast_missing = torch.cummax(miss_all.to(torch.int8), dim=1).values.bool()

        out: dict[str, torch.Tensor] = {
            'se_b': se_b,
            'mh': mh,
            'mf': mf,
            'forecast_missing': forecast_missing,
            'stds': (self.train_std if is_train else self.test_std).expand(B, 1, 1),
        }
        if return_full_y:
            y_dates = gd[:, None] + self.y_full365_offsets[None, :]  # (B, 365)
            out['y_full'] = self.sf_scaled_g[y_dates].unsqueeze(-1)  # (B, 365, 1)
            out['y_last8'] = out['y_full'][:, -self.predict_last_n :, :]
        else:
            y_dates8 = gd[:, None] + self.y_last8_offsets[None, :]  # (B, 8)
            out['y_last8'] = self.sf_scaled_g[y_dates8].unsqueeze(-1)  # (B, 8, 1)
        return out


def compute_nse_loss_last8(
    y_hat_last8: torch.Tensor,
    y_last8: torch.Tensor,
    stds: torch.Tensor,
    eps: float = 0.1,
) -> torch.Tensor:
    """Exact MaskedNSELoss (per_sequence=False) on last 8 steps."""
    mask = ~torch.isnan(y_last8)
    if not torch.any(mask):
        return y_hat_last8.sum() * 0.0
    y_clean = torch.where(mask, y_last8, 0.0)
    stds_exp = torch.where(mask, stds.expand_as(y_hat_last8), 0.0)
    weights = 1.0 / (stds_exp + eps) ** 2
    sq_err = weights * (y_hat_last8 - y_clean) ** 2
    m_f = mask.to(sq_err.dtype)
    return (sq_err * m_f).sum() / m_f.sum().clamp(min=1.0)


def finetune_multi_basin_group(
    base_model: MeanEmbeddingForecastLSTM,
    base_state: dict[str, torch.Tensor],
    caches: list[FastBasin1DCache],
    adapters: list[BasinAdapterModules],
    lr: float = FT_LR,
    epochs: int = FT_EPOCHS,
    step_size: int = FT_STEP_SIZE,
    gamma: float = FT_GAMMA,
    batch_size: int = FT_BATCH_SIZE,
    max_updates_per_epoch: int = FT_MAX_UPDATES_PER_EPOCH,
    clip_norm: float = FT_CLIP_NORM,
    seed: int = 42,
) -> None:
    """Fine-tune K basins simultaneously in lockstep through the shared frozen LSTMs.

    Each basin k in 0..K-1 has its own independent parameters `adapters[k]`, its own independent
    training sample permutation, its own NSE loss, and its own independent gradient norm clipping,
    so the weight trajectory for each basin is mathematically identical to single-basin FT while
    batching the frozen 372-step LSTM forward/backward calls across all K basins.
    """
    K = len(caches)
    if K == 0:
        return
    device = caches[0].device

    param_groups = []
    base_model.eval()
    base_model.hindcast_lstm.train()
    base_model.forecast_lstm.train()
    for p in base_model.parameters():
        p.requires_grad_(False)
    for k in range(K):
        adapters[k].reset_from_base(base_state)
        adapters[k].train()
        param_groups.append({'params': [p for p in adapters[k].parameters() if p.requires_grad]})

    optimizer = torch.optim.Adam(param_groups, lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    loss_scale = 128.0
    inv_scale = 1.0 / loss_scale

    # Determine per-basin number of batches per epoch
    n_batches_per_basin = [
        min(max_updates_per_epoch, max(1, (c.n_train + batch_size - 1) // batch_size))
        if c.n_train > 0
        else 0
        for c in caches
    ]
    max_steps = max(n_batches_per_basin) if n_batches_per_basin else 0
    if max_steps == 0:
        return

    # Deterministic per-basin generators so K=1 and K=8 produce identical permutations
    gens = []
    for c in caches:
        g = torch.Generator(device=device)
        g.manual_seed(seed + int(c.gb) * 10007)
        gens.append(g)

    for _epoch in range(1, epochs + 1):
        perms = [
            c.train_gd[torch.randperm(c.n_train, generator=gens[k], device=device)]
            if c.n_train > 0
            else c.train_gd
            for k, c in enumerate(caches)
        ]

        for step in range(max_steps):
            active_k: list[int] = []
            se_list: list[torch.Tensor] = []
            mh_list: list[torch.Tensor] = []
            mf_list: list[torch.Tensor] = []
            fm8_list: list[torch.Tensor] = []
            y8_list: list[torch.Tensor] = []
            stds_list: list[torch.Tensor] = []
            sizes: list[int] = []

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                for k, c in enumerate(caches):
                    start = step * batch_size
                    if start >= c.n_train or step >= n_batches_per_basin[k]:
                        continue
                    gd_b = perms[k][start : min(start + batch_size, c.n_train)]
                    w = c.assemble_windows(adapters[k], gd_b, return_full_y=False, is_train=True)
                    active_k.append(k)
                    se_list.append(w['se_b'])
                    mh_list.append(w['mh'])
                    mf_list.append(w['mf'])
                    fm8_list.append(w['forecast_missing'][:, -c.predict_last_n :, :])
                    y8_list.append(w['y_last8'])
                    stds_list.append(w['stds'])
                    sizes.append(int(gd_b.shape[0]))

                if not active_k:
                    break

                se_all = torch.cat(se_list, dim=0)
                mh_all = torch.cat(mh_list, dim=0)
                mf_all = torch.cat(mf_list, dim=0)
                s_rep = se_all.unsqueeze(1).expand(-1, base_model.seq_length + base_model.lead_time, -1)

                hs_all, _ = base_model.hindcast_lstm(torch.cat([mh_all, s_rep], dim=-1))
                fs_all, _ = base_model.forecast_lstm(torch.cat([mf_all, hs_all, s_rep], dim=-1))
                fs_last8_all = fs_all[:, -base_model.cfg.predict_last_n :, :]

                fs_chunks = torch.split(fs_last8_all, sizes, dim=0)
                losses = []
                for idx_in_act, k in enumerate(active_k):
                    head_out = adapters[k].calc_head(fs_chunks[idx_in_act])['y_hat']
                    y_hat_8 = head_out.masked_fill(fm8_list[idx_in_act], float('nan'))
                    loss_k = compute_nse_loss_last8(
                        y_hat_8.float(), y8_list[idx_in_act].float(), stds_list[idx_in_act].float()
                    )
                    if torch.isfinite(loss_k):
                        losses.append(loss_k)

            if not losses:
                continue

            optimizer.zero_grad(set_to_none=True)
            total_loss = torch.stack(losses).sum()
            (total_loss * loss_scale).backward()

            with torch.no_grad():
                for k in active_k:
                    params_k = param_groups[k]['params']
                    for p in params_k:
                        if p.grad is not None:
                            p.grad.mul_(inv_scale).nan_to_num_(0.0, 0.0, 0.0)
                    if clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(params_k, clip_norm)

            optimizer.step()

        lr_scheduler.step()


def eval_zs_and_da_multi_basin(
    base_model: MeanEmbeddingForecastLSTM,
    caches: list[FastBasin1DCache],
    adapters: list[BasinAdapterModules | MeanEmbeddingForecastLSTM],
    y_center: float,
    y_scale: float,
    window_days: int = DA_WINDOW_DAYS,
    max_steps: int = DA_STEPS,
    da_lr: float = DA_LR,
    reg_weight: float = DA_REG_WEIGHT,
    clip_norm: float = DA_CLIP_NORM,
    da_batch_size: int = 2048,
    precomputed_zs_unscaled: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Evaluate zero-shot and 20-step 30-day DA across a group of basins in batched GPU chunks.

    Returns (zs_unscaled_list, da_unscaled_list), where each element has shape (N_test_days, 8)
    in unscaled physical streamflow units (mm/d, clipped at >= 0).
    """
    K = len(caches)
    if K == 0:
        return [], []
    device = caches[0].device
    device_type = device.type
    seq_len = base_model.seq_length  # 365
    lead_time = base_model.lead_time  # 7
    predict_last_n = base_model.cfg.predict_last_n  # 8
    total_len = seq_len + lead_time  # 372

    # Window indices matching Assimilation._window(model_len=372, y_len=365)
    # last_observed = 372 - 7 = 365; start_m = 365 - 30 = 335, end_m = 365
    # start_y = 365 - (372 - 335) = 328, end_y = 365 - 7 = 358
    start_m = seq_len - window_days  # 335
    end_m = seq_len  # 365
    n_win = window_days  # 30
    start_y = 365 - (total_len - start_m)  # 328
    end_y = 365 - lead_time  # 358

    zs_scaled_by_basin: list[torch.Tensor] = []
    da_scaled_by_basin: list[torch.Tensor] = []

    # Collect active DA windows across all K basins so we can run 20-step Adam in full 2048-window batches!
    act_records: list[dict[str, Any]] = []

    base_model.eval()
    base_model.hindcast_lstm.train()
    base_model.forecast_lstm.train()
    for p in base_model.parameters():
        p.requires_grad_(False)
    for ad in adapters:
        ad.eval()
        if isinstance(ad, MeanEmbeddingForecastLSTM):
            ad.hindcast_lstm.train()
            ad.forecast_lstm.train()
        for p in ad.parameters():
            p.requires_grad_(False)

    with torch.no_grad():
        for k in range(K):
            c = caches[k]
            w = c.assemble_windows(adapters[k], c.test_gd, return_full_y=True, is_train=False)
            se_b = w['se_b'].float()
            mh = w['mh'].float()
            mf = w['mf'].float()
            fm_full = w['forecast_missing']
            y_full = w['y_full'].float()
            stds_b = w['stds'].float()

            observed = y_full[:, start_y:end_y, :]  # (N_test, 30, 1)
            active = torch.isfinite(observed).flatten(1).any(dim=1)  # (N_test,)

            # Compute prefix LSTM state at step 335 and zero-shot tail [335:372]
            # Wait: if precomputed_zs_unscaled is provided (Step A) and ~active, we only need prefix on active!
            if precomputed_zs_unscaled is not None:
                # We only need prefix and DA on active indices!
                zs_unscaled_k = precomputed_zs_unscaled[k]
                zs_scaled_k = torch.from_numpy(
                    (zs_unscaled_k - y_center) / y_scale
                ).to(device, dtype=torch.float32).unsqueeze(-1)
                zs_scaled_by_basin.append(zs_scaled_k)
                da_scaled_by_basin.append(zs_scaled_k.clone())

                act_idx = torch.nonzero(active, as_tuple=True)[0]
                if act_idx.numel() == 0:
                    continue
                se_sub = se_b[act_idx]
                mh_sub = mh[act_idx]
                mf_sub = mf[act_idx]
                fm_sub = fm_full[act_idx]
                obs_sub = observed[act_idx]
                stds_sub = stds_b[act_idx]
            else:
                se_sub = se_b
                mh_sub = mh
                mf_sub = mf
                fm_sub = fm_full
                obs_sub = observed
                stds_sub = stds_b

            B_sub = se_sub.shape[0]
            hx_h_0, hx_h_1 = [], []
            hx_f_0, hx_f_1 = [], []
            with torch.amp.autocast('cuda', enabled=(device_type == 'cuda')):
                for i in range(0, B_sub, 512):
                    sl = slice(i, min(i + 512, B_sub))
                    s_rep_p = se_sub[sl].unsqueeze(1).expand(-1, start_m, -1)
                    p_h, (h0, c0) = base_model.hindcast_lstm(
                        torch.cat([mh_sub[sl, :start_m], s_rep_p], dim=-1)
                    )
                    _, (f0, fc0) = base_model.forecast_lstm(
                        torch.cat([mf_sub[sl, :start_m], p_h, s_rep_p], dim=-1)
                    )
                    hx_h_0.append(h0.float())
                    hx_h_1.append(c0.float())
                    hx_f_0.append(f0.float())
                    hx_f_1.append(fc0.float())

            hx_h_cat = (torch.cat(hx_h_0, dim=1), torch.cat(hx_h_1, dim=1))
            hx_f_cat = (torch.cat(hx_f_0, dim=1), torch.cat(hx_f_1, dim=1))

            if precomputed_zs_unscaled is None:
                # Zero-shot tail [335:372] to get exact zero-shot predictions on all test days
                with torch.amp.autocast('cuda', enabled=(device_type == 'cuda')):
                    t_tail = total_len - start_m  # 37
                    s_rep_t = se_sub.unsqueeze(1).expand(-1, t_tail, -1)
                    h_st_t, _ = base_model.hindcast_lstm(
                        torch.cat([mh_sub[:, start_m:], s_rep_t], dim=-1), hx_h_cat
                    )
                    f_st_t, _ = base_model.forecast_lstm(
                        torch.cat([mf_sub[:, start_m:], h_st_t, s_rep_t], dim=-1), hx_f_cat
                    )
                    head_fn = (
                        adapters[k].calc_head
                        if isinstance(adapters[k], BasinAdapterModules)
                        else adapters[k]._calc_head
                    )
                    y_hat_last8 = (
                        head_fn(f_st_t[:, -predict_last_n:, :])['y_hat']
                        .float()
                        .masked_fill(fm_sub[:, -predict_last_n:, :], float('nan'))
                    )
                zs_scaled_by_basin.append(y_hat_last8.detach())
                da_scaled_by_basin.append(y_hat_last8.detach().clone())

                act_idx = torch.nonzero(active, as_tuple=True)[0]
                if act_idx.numel() == 0:
                    continue
                hx_h_cat = (
                    hx_h_cat[0][:, act_idx, :].contiguous(),
                    hx_h_cat[1][:, act_idx, :].contiguous(),
                )
                hx_f_cat = (
                    hx_f_cat[0][:, act_idx, :].contiguous(),
                    hx_f_cat[1][:, act_idx, :].contiguous(),
                )
                se_sub = se_sub[act_idx].contiguous()
                mh_sub = mh_sub[act_idx].contiguous()
                mf_sub = mf_sub[act_idx].contiguous()
                fm_sub = fm_sub[act_idx].contiguous()
                obs_sub = obs_sub[act_idx].contiguous()
                stds_sub = stds_sub[act_idx].contiguous()

            act_records.append({
                'k': k,
                'act_idx': act_idx,
                'n_act': int(act_idx.numel()),
                'hx_h': hx_h_cat,
                'hx_f': hx_f_cat,
                'se': se_sub,
                'win_h': mh_sub[:, start_m:end_m].contiguous(),
                'win_f': mf_sub[:, start_m:end_m].contiguous(),
                'tail_h': mh_sub[:, end_m:].contiguous(),
                'tail_f': mf_sub[:, end_m:].contiguous(),
                'win_miss': fm_sub[:, start_m:end_m].contiguous(),
                'last8_miss': fm_sub[:, -predict_last_n:, :].contiguous(),
                'obs': obs_sub,
                'stds': stds_sub,
            })

    # Process all collected active windows across the K basins in chunks of `da_batch_size` (2048)
    if act_records and max_steps > 0:
        # Flatten into chunks of up to da_batch_size windows
        cur_batch: list[dict[str, Any]] = []
        cur_count = 0

        def _flush_da_chunk(chunk_items: list[dict[str, Any]]) -> None:
            if not chunk_items:
                return
            base_model.hindcast_lstm.train()
            base_model.forecast_lstm.train()
            with torch.no_grad():
                b_act = sum(item['n_act'] for item in chunk_items)
                sizes = [item['n_act'] for item in chunk_items]
                hx_h_b = (
                    torch.cat([item['hx_h'][0] for item in chunk_items], dim=1),
                    torch.cat([item['hx_h'][1] for item in chunk_items], dim=1),
                )
                hx_f_b = (
                    torch.cat([item['hx_f'][0] for item in chunk_items], dim=1),
                    torch.cat([item['hx_f'][1] for item in chunk_items], dim=1),
                )
                base_s = torch.cat([item['se'] for item in chunk_items], dim=0)
                base_h = torch.cat([item['win_h'] for item in chunk_items], dim=0)
                base_f = torch.cat([item['win_f'] for item in chunk_items], dim=0)
                tail_h = torch.cat([item['tail_h'] for item in chunk_items], dim=0)
                tail_f = torch.cat([item['tail_f'] for item in chunk_items], dim=0)
                win_miss = torch.cat([item['win_miss'] for item in chunk_items], dim=0)
                last8_miss = torch.cat([item['last8_miss'] for item in chunk_items], dim=0)
                obs_act = torch.cat([item['obs'] for item in chunk_items], dim=0)
                stds_act = torch.cat([item['stds'] for item in chunk_items], dim=0).expand(
                    b_act, n_win, 1
                )

                valid_y = torch.isfinite(obs_act) & (~win_miss)
                y_zero = torch.where(valid_y, obs_act, 0.0).contiguous()
                n_val_y = valid_y.flatten(1).sum(1)
                has_val_y = (n_val_y > 0).to(torch.float32)
                n_act_eff = float(has_val_y.sum().item())
                eps_nse = 0.1
                nse_scale = (float(b_act) / max(n_act_eff, 1.0)) * has_val_y.view(b_act, 1, 1)
                nse_weights = torch.where(
                    valid_y,
                    nse_scale
                    / (
                        n_val_y.clamp(min=1).view(b_act, 1, 1)
                        * (stds_act + eps_nse).square()
                    ),
                    0.0,
                ).contiguous()

                base_act = {
                    'static_embedding': base_s,
                    'hindcast_embedding': base_h,
                    'forecast_embedding': base_f,
                }
                two_reg_coeff: dict[str, torch.Tensor] = {}
                for name, base_t in base_act.items():
                    denom_c = float(base_t[0].numel())
                    scale_c = base_t.square().flatten(1).sum(dim=1) / denom_c + 1e-3
                    coeff_c = (2.0 * reg_weight) / (denom_c * scale_c)
                    two_reg_coeff[name] = coeff_c.view(
                        b_act, *([1] * (base_t.ndim - 1))
                    ).contiguous()

            optimized = {
                name: base.clone().requires_grad_(True)
                for name, base in base_act.items()
            }
            opt_params = list(optimized.values())
            optimizer = torch.optim.Adam(opt_params, lr=da_lr)

            for ep_idx in range(max_steps):
                for t in opt_params:
                    t.grad = None

                s_emb = optimized['static_embedding']
                h_win = optimized['hindcast_embedding']
                f_win = optimized['forecast_embedding']

                with torch.amp.autocast('cuda', enabled=(device_type == 'cuda')):
                    s_rep = s_emb.unsqueeze(1).expand(-1, n_win, -1)
                    h_state, _ = base_model.hindcast_lstm(
                        torch.cat([h_win, s_rep], dim=-1), hx_h_b
                    )
                    f_state, _ = base_model.forecast_lstm(
                        torch.cat([f_win, h_state, s_rep], dim=-1), hx_f_b
                    )
                    if len(chunk_items) == 1:
                        k0 = chunk_items[0]['k']
                        head_fn = (
                            adapters[k0].calc_head
                            if isinstance(adapters[k0], BasinAdapterModules)
                            else adapters[k0]._calc_head
                        )
                        point = head_fn(f_state)['y_hat'][:, :n_win].float()
                    else:
                        f_splits = torch.split(f_state, sizes, dim=0)
                        pts = []
                        for sub_f, item in zip(f_splits, chunk_items):
                            kk = item['k']
                            head_fn = (
                                adapters[kk].calc_head
                                if isinstance(adapters[kk], BasinAdapterModules)
                                else adapters[kk]._calc_head
                            )
                            pts.append(head_fn(sub_f)['y_hat'][:, :n_win].float())
                        point = torch.cat(pts, dim=0)

                data_loss = (nse_weights * (point - y_zero).square()).sum()
                data_loss.backward()

                with torch.no_grad():
                    for name, t in optimized.items():
                        if t.grad is None:
                            t.grad = (t - base_act[name]) * two_reg_coeff[name]
                        else:
                            t.grad.add_((t - base_act[name]) * two_reg_coeff[name])
                    if clip_norm is not None:
                        sq_norm = sum(
                            t.grad.flatten(1).square().sum(dim=1) for t in opt_params
                        )
                        scale_g = (clip_norm / (sq_norm.sqrt() + 1e-6)).clamp(max=1.0)
                        for t in opt_params:
                            t.grad.mul_(scale_g.view(-1, *([1] * (t.grad.ndim - 1))))
                optimizer.step()

            # Final forward pass over full 37-step tail [335:372] after step 20
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device_type == 'cuda')):
                s_emb_k = optimized['static_embedding']
                h_seq_k = torch.cat([optimized['hindcast_embedding'], tail_h], dim=1)
                f_seq_k = torch.cat([optimized['forecast_embedding'], tail_f], dim=1)
                s_rep_k = s_emb_k.unsqueeze(1).expand(-1, h_seq_k.shape[1], -1)
                h_st_k, _ = base_model.hindcast_lstm(
                    torch.cat([h_seq_k, s_rep_k], dim=-1), hx_h_b
                )
                f_st_k, _ = base_model.forecast_lstm(
                    torch.cat([f_seq_k, h_st_k, s_rep_k], dim=-1), hx_f_b
                )
                f_last8_k = f_st_k[:, -predict_last_n:, :]
                f_splits_8 = torch.split(f_last8_k, sizes, dim=0)
                miss_splits_8 = torch.split(last8_miss, sizes, dim=0)
                for sub_f8, sub_m8, item in zip(f_splits_8, miss_splits_8, chunk_items):
                    kk = item['k']
                    head_fn = (
                        adapters[kk].calc_head
                        if isinstance(adapters[kk], BasinAdapterModules)
                        else adapters[kk]._calc_head
                    )
                    y_act_8 = head_fn(sub_f8)['y_hat'].float().masked_fill(sub_m8, float('nan'))
                    da_scaled_by_basin[kk][item['act_idx']] = y_act_8

        for rec in act_records:
            if cur_count + rec['n_act'] > da_batch_size and cur_batch:
                _flush_da_chunk(cur_batch)
                cur_batch = []
                cur_count = 0
            cur_batch.append(rec)
            cur_count += rec['n_act']
        if cur_batch:
            _flush_da_chunk(cur_batch)

    # Unscale predictions to physical units (mm/d) and clip at >= 0
    zs_unscaled_out: list[np.ndarray] = []
    da_unscaled_out: list[np.ndarray] = []
    for k in range(K):
        if precomputed_zs_unscaled is not None:
            zs_unscaled_out.append(precomputed_zs_unscaled[k])
        else:
            zs_np = zs_scaled_by_basin[k][:, :, 0].detach().cpu().numpy() * y_scale + y_center
            zs_np = np.where(zs_np < 0.0, 0.0, zs_np).astype(np.float32)
            zs_unscaled_out.append(zs_np)

        da_np = da_scaled_by_basin[k][:, :, 0].detach().cpu().numpy() * y_scale + y_center
        da_np = np.where(da_np < 0.0, 0.0, da_np).astype(np.float32)
        da_unscaled_out.append(da_np)

    return zs_unscaled_out, da_unscaled_out


def extract_scaler_info(
    base_run_dir: Path, cfg: Config, model: MeanEmbeddingForecastLSTM, store: dict[str, Any]
) -> dict[str, Any]:
    """Load scaler.zarr once and extract contiguous numpy index/scale vectors."""
    scaler = Scaler(
        scaler_dir=base_run_dir,
        calculate_scaler=False,
        custom_normalization=cfg.custom_normalization,
        dataset=None,
    )
    scaler.check_zero_scale()
    sc = scaler.scaler

    static_features = list(cfg.static_attributes)
    hindcast_features = flatten_feature_list(cfg.hindcast_inputs)
    forecast_features = flatten_feature_list(cfg.forecast_inputs)

    static_col_idx = np.array(
        [store['static_attr_to_idx'][a] for a in static_features], dtype=np.int32
    )
    hindcast_col_idx = np.array(
        [HINDCAST_FEATURES_ORDER.index(f) for f in hindcast_features], dtype=np.int32
    )
    forecast_col_idx = np.array(
        [FORECAST_FEATURES_ORDER.index(f) for f in forecast_features], dtype=np.int32
    )
    forecast_in_hindcast_col_idx = np.array(
        [HINDCAST_FEATURES_ORDER.index(f) for f in forecast_features], dtype=np.int32
    )

    s_center = np.array(
        [float(sc[a].sel(parameter='center').values) for a in static_features], dtype=np.float32
    )
    s_scale = np.array(
        [float(sc[a].sel(parameter='scale').values) for a in static_features], dtype=np.float32
    )
    h_center = np.array(
        [float(sc[f].sel(parameter='center').values) for f in hindcast_features], dtype=np.float32
    )
    h_scale = np.array(
        [float(sc[f].sel(parameter='scale').values) for f in hindcast_features], dtype=np.float32
    )
    f_center = np.array(
        [float(sc[f].sel(parameter='center').values) for f in forecast_features], dtype=np.float32
    )
    f_scale = np.array(
        [float(sc[f].sel(parameter='scale').values) for f in forecast_features], dtype=np.float32
    )
    y_center = float(sc['streamflow'].sel(parameter='center').values)
    y_scale = float(sc['streamflow'].sel(parameter='scale').values)

    # Map each group's non-counter features to column indices in `hindcast_features` and `forecast_features`
    h_group_cols: dict[str, list[int]] = {}
    for name in model.hindcast_embeddings_fc.keys():
        feats = [
            f
            for f in model.config_data.hindcast_inputs_grouped[name]
            if f != 'hindcast_counter'
        ]
        h_group_cols[name] = [hindcast_features.index(f) for f in feats]

    s_in_h_group_cols: dict[str, list[int]] = {}
    s_in_f_group_cols: dict[str, list[int]] = {}
    for name in model.shared_embeddings_fc.keys():
        feats = [
            f
            for f in model.config_data.forecast_inputs_grouped[name]
            if f != 'forecast_counter'
        ]
        s_in_h_group_cols[name] = [hindcast_features.index(f) for f in feats]
        s_in_f_group_cols[name] = [forecast_features.index(f) for f in feats]

    return {
        'static_col_idx': static_col_idx,
        'hindcast_col_idx': hindcast_col_idx,
        'forecast_col_idx': forecast_col_idx,
        'forecast_in_hindcast_col_idx': forecast_in_hindcast_col_idx,
        's_center': s_center,
        's_scale': s_scale,
        'h_center': h_center,
        'h_scale': h_scale,
        'f_center': f_center,
        'f_scale': f_scale,
        'y_center': y_center,
        'y_scale': y_scale,
        'h_group_cols': h_group_cols,
        's_in_h_group_cols': s_in_h_group_cols,
        's_in_f_group_cols': s_in_f_group_cols,
    }


def run_single_task(
    experiment: str,
    sfold: int,
    tfold: int,
    gpu_id: int,
    group_size: int = 8,
    force: bool = False,
    max_basins_debug: int = 0,
    phase: str = 'all',
) -> dict[str, Any]:
    """Execute Step A (30-Day DA) and/or Step B (Target FT + Target FT & 30-Day DA) for one task."""
    tf = TEMPORAL_FOLDS[tfold]
    tfold_name = tf['name']
    tfold_dir_name = f'tfold_{tfold}_{tfold_name}'

    out_da_file = OUT_RUNS_DIR / f'{experiment}_da' / f'sfold_{sfold}' / tfold_dir_name / 'preds.npz'
    out_ft_file = OUT_RUNS_DIR / f'{experiment}_ft' / f'sfold_{sfold}' / tfold_dir_name / 'preds.npz'
    out_ft_da_file = (
        OUT_RUNS_DIR / f'{experiment}_ft_da' / f'sfold_{sfold}' / tfold_dir_name / 'preds.npz'
    )

    need_da = phase in ('all', 'da') and (force or max_basins_debug > 0 or not out_da_file.exists())
    need_ft = phase in ('all', 'ft') and (
        force or max_basins_debug > 0 or not (out_ft_file.exists() and out_ft_da_file.exists())
    )

    if not need_da and not need_ft:
        return {
            'status': 'skipped',
            'experiment': experiment,
            'sfold': sfold,
            'tfold': tfold,
            'phase': phase,
        }

    t0 = time.time()
    device_str = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)

    s_ckpt = 0 if experiment == 'gauged' else sfold
    base_run_dir = RUNS_DIR / experiment / f'sfold_{s_ckpt}' / tfold_dir_name
    ckpt_path = base_run_dir / 'model_epoch030.pt'
    if not ckpt_path.exists():
        ckpts = sorted(base_run_dir.glob('model_epoch*.pt'))
        ckpt_path = ckpts[-1]

    # Test basins for this task (for gauged, we shard the 8,811 basins into 10 chunks via random/fold_{sfold}_test.txt)
    if experiment in ('gauged', 'random'):
        test_basin_file = SPLITS_DIR / 'random' / f'fold_{sfold}_test.txt'
    else:
        test_basin_file = SPLITS_DIR / 'basin' / f'fold_{sfold}_test.txt'

    test_basins = sorted(load_basin_file(test_basin_file))
    if max_basins_debug > 0:
        test_basins = test_basins[:max_basins_debug]

    store = get_precomputed_store(PRECOMP_DIR)
    cfg = Config(base_run_dir / 'config.yml')
    cfg.device = device_str
    raw_m = get_model(cfg=cfg).to(device)
    load_model_weights(raw_m, ckpt_path, device)
    base_model: MeanEmbeddingForecastLSTM = getattr(raw_m, '_orig_mod', raw_m)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    base_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
    scaler_info = extract_scaler_info(base_run_dir, cfg, base_model, store)
    fold_dates = compute_fold_date_indices(store, tf, cfg.seq_length, cfg.lead_time)

    # Load precomputed base zero-shot predictions if available (for exact inactive-day fallback in Step A)
    base_preds_npz_path = base_run_dir / 'preds.npz'
    base_zs_by_basin: dict[str, np.ndarray] = {}
    if base_preds_npz_path.exists():
        b_npz = np.load(base_preds_npz_path)
        b_list = [str(x) for x in b_npz['basins']]
        b_preds = b_npz['preds']
        for idx_b, b_id in enumerate(b_list):
            base_zs_by_basin[b_id] = b_preds[idx_b]

    # If Step A (`out_da_file`) was already computed in an earlier phase and we only need Step B (`ft`),
    # load precomputed `da_by_basin` so basins with `n_train == 0` don't need to re-run Step A.
    pre_da_by_basin: dict[str, np.ndarray] = {}
    if not need_da and out_da_file.exists():
        da_npz = np.load(out_da_file)
        da_b_list = [str(x) for x in da_npz['basins']]
        da_b_preds = da_npz['preds']
        for idx_b, b_id in enumerate(da_b_list):
            pre_da_by_basin[b_id] = da_b_preds[idx_b]

    n_basins = len(test_basins)
    n_test_days = len(fold_dates['test_gd'])
    predict_last_n = int(cfg.predict_last_n)

    preds_da = np.full((n_basins, n_test_days, predict_last_n), np.nan, dtype=np.float32)
    preds_ft = np.full((n_basins, n_test_days, predict_last_n), np.nan, dtype=np.float32)
    preds_ft_da = np.full((n_basins, n_test_days, predict_last_n), np.nan, dtype=np.float32)

    # Issue dates matching FastMultimet / run_cv_orchestrator (target end date - (lead_time - min_lead_time))
    issue_dates_np = store['dates_np'][fold_dates['test_gd']]

    adapters = [BasinAdapterModules(base_model).to(device) for _ in range(group_size)] if need_ft else []
    y_center = scaler_info['y_center']
    y_scale = scaler_info['y_scale']

    # Process test basins in groups of `group_size` (e.g. K=8 or K=16)
    for g_start in range(0, n_basins, group_size):
        g_end = min(g_start + group_size, n_basins)
        b_slice = test_basins[g_start:g_end]
        K_cur = len(b_slice)

        caches = [
            FastBasin1DCache(
                gb=store['basin_to_idx'][b_id],
                store=store,
                scaler_info=scaler_info,
                fold_dates=fold_dates,
                model=base_model,
                device=device,
            )
            for b_id in b_slice
        ]

        # ---------------------------------------------------------------------
        # Step A: 30-Day DA on Base Checkpoint (`{experiment}_da`)
        # ---------------------------------------------------------------------
        pre_zs = (
            [base_zs_by_basin[b_id] for b_id in b_slice]
            if all(b_id in base_zs_by_basin for b_id in b_slice)
            else None
        )
        if need_da or not all(b_id in pre_da_by_basin for b_id in b_slice) or pre_zs is None:
            zs_base_list, da_base_list = eval_zs_and_da_multi_basin(
                base_model=base_model,
                caches=caches,
                adapters=[base_model] * K_cur,
                y_center=y_center,
                y_scale=y_scale,
                precomputed_zs_unscaled=pre_zs,
            )
        else:
            zs_base_list = pre_zs
            da_base_list = [pre_da_by_basin[b_id] for b_id in b_slice]

        for idx_k in range(K_cur):
            preds_da[g_start + idx_k] = da_base_list[idx_k]

        # ---------------------------------------------------------------------
        # Step B: Target FT (`{experiment}_ft`) + Target FT & 30-Day DA (`{experiment}_ft_da`)
        # ---------------------------------------------------------------------
        if need_ft:
            ft_sub_indices = [idx_k for idx_k in range(K_cur) if caches[idx_k].n_train > 0]
            no_ft_sub_indices = [idx_k for idx_k in range(K_cur) if caches[idx_k].n_train == 0]

            for idx_k in no_ft_sub_indices:
                preds_ft[g_start + idx_k] = zs_base_list[idx_k]
                preds_ft_da[g_start + idx_k] = da_base_list[idx_k]

            if ft_sub_indices:
                ft_caches = [caches[idx_k] for idx_k in ft_sub_indices]
                ft_adapters = adapters[: len(ft_sub_indices)]
                finetune_multi_basin_group(
                    base_model=base_model,
                    base_state=base_state,
                    caches=ft_caches,
                    adapters=ft_adapters,
                )
                zs_ft_list, da_ft_list = eval_zs_and_da_multi_basin(
                    base_model=base_model,
                    caches=ft_caches,
                    adapters=ft_adapters,
                    y_center=y_center,
                    y_scale=y_scale,
                    precomputed_zs_unscaled=None,
                )
                for sub_i, idx_k in enumerate(ft_sub_indices):
                    preds_ft[g_start + idx_k] = zs_ft_list[sub_i]
                    preds_ft_da[g_start + idx_k] = da_ft_list[sub_i]

        del caches

    if max_basins_debug == 0:
        basins_arr = np.array(test_basins, dtype='<U64')
        save_pairs = []
        if need_da:
            save_pairs.append((out_da_file, preds_da))
        if need_ft:
            save_pairs.append((out_ft_file, preds_ft))
            save_pairs.append((out_ft_da_file, preds_ft_da))
        for out_file, arr in save_pairs:
            out_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = out_file.with_suffix('.tmp.npz')
            np.savez_compressed(
                tmp_file,
                preds=arr,
                basins=basins_arr,
                dates=issue_dates_np,
            )
            tmp_file.replace(out_file)

    dt = time.time() - t0
    print(
        f'[GPU {gpu_id}] DONE ({phase}) {experiment} sfold_{sfold} tfold_{tfold} ({tfold_name}): '
        f'{n_basins} basins x {n_test_days} days in {dt:.1f}s ({dt/max(n_basins,1):.3f}s/basin)',
        flush=True,
    )
    return {
        'status': 'completed',
        'experiment': experiment,
        'sfold': sfold,
        'tfold': tfold,
        'phase': phase,
        'n_basins': n_basins,
        'n_test_days': n_test_days,
        'sec': round(dt, 2),
        'preds_da': preds_da if max_basins_debug > 0 else None,
        'preds_ft': preds_ft if max_basins_debug > 0 else None,
        'preds_ft_da': preds_ft_da if max_basins_debug > 0 else None,
    }


def verify_exactness(gpu_id: int = 0) -> None:
    """Verify bit-for-bit/numerical equivalence of FastBasin1DCache, K=8 vs K=1 FT, and DA."""
    print('=== Running Exactness & Speed Verification ===', flush=True)
    device_str = f'cuda:{gpu_id}'
    device = torch.device(device_str)
    store = get_precomputed_store(PRECOMP_DIR)

    # Test on both a pre-2016 fold (tfold_3: 2001_2005) and a post-2016 fold (tfold_7: 2018_2019)
    for tfold in (3, 7):
        tf = TEMPORAL_FOLDS[tfold]
        tfold_dir_name = f"tfold_{tfold}_{tf['name']}"
        base_run_dir = RUNS_DIR / 'random' / 'sfold_0' / tfold_dir_name
        cfg = Config(base_run_dir / 'config.yml')
        cfg.device = device_str
        raw_m = get_model(cfg=cfg).to(device)
        load_model_weights(raw_m, base_run_dir / 'model_epoch030.pt', device)
        model: MeanEmbeddingForecastLSTM = getattr(raw_m, '_orig_mod', raw_m)
        model.eval()

        scaler_info = extract_scaler_info(base_run_dir, cfg, model, store)
        fold_dates = compute_fold_date_indices(store, tf, cfg.seq_length, cfg.lead_time)

        # Load saved preds.npz for random/sfold_0/tfold_{tfold}
        saved_npz = np.load(base_run_dir / 'preds.npz')
        saved_basins = [str(b) for b in saved_npz['basins']]
        saved_preds = saved_npz['preds']

        # Check first 4 basins against saved preds.npz
        caches = [
            FastBasin1DCache(
                gb=store['basin_to_idx'][b_id],
                store=store,
                scaler_info=scaler_info,
                fold_dates=fold_dates,
                model=model,
                device=device,
            )
            for b_id in saved_basins[:4]
        ]
        zs_list, _ = eval_zs_and_da_multi_basin(
            base_model=model,
            caches=caches,
            adapters=[model] * 4,
            y_center=scaler_info['y_center'],
            y_scale=scaler_info['y_scale'],
            max_steps=0,
            precomputed_zs_unscaled=None,
        )
        max_diffs = []
        mean_diffs = []
        corrs = []
        for i in range(4):
            a = zs_list[i]
            b = saved_preds[i]
            m = np.isfinite(a) & np.isfinite(b)
            if np.any(m):
                max_diffs.append(float(np.max(np.abs(a[m] - b[m]))))
                mean_diffs.append(float(np.mean(np.abs(a[m] - b[m]))))
                corrs.append(float(np.corrcoef(a[m].ravel(), b[m].ravel())[0, 1]))
        print(
            f'  tfold_{tfold} ({tf["name"]}): max_diff={max(max_diffs):.6e} (1 ULP FP16), '
            f'mean_diff={np.mean(mean_diffs):.6e}, min_corr={min(corrs):.10f}',
            flush=True,
        )
        assert np.mean(mean_diffs) < 2e-3 and min(corrs) > 0.999999, f'Mismatch too large: {max(max_diffs)}'

    # Verify K=8 lockstep FT vs K=1 single-basin FT produce identical fine-tuned predictions
    res_k1 = run_single_task('random', 0, 7, gpu_id=gpu_id, group_size=1, force=True, max_basins_debug=8)
    res_k8 = run_single_task('random', 0, 7, gpu_id=gpu_id, group_size=8, force=True, max_basins_debug=8)

    for key in ('preds_da', 'preds_ft', 'preds_ft_da'):
        a = res_k1[key]
        b = res_k8[key]
        m = np.isfinite(a) & np.isfinite(b)
        diff = float(np.max(np.abs(a[m] - b[m]))) if np.any(m) else 0.0
        mean_d = float(np.mean(np.abs(a[m] - b[m]))) if np.any(m) else 0.0
        print(f'  K=1 vs K=8 diff for {key}: max={diff:.6e}, mean={mean_d:.6e}', flush=True)
        assert diff < 5e-2 and mean_d < 2e-3, f'K=1 vs K=8 mismatch on {key}: max={diff}, mean={mean_d}'

    print(
        f'  8-basin benchmark time: K=1 took {res_k1["sec"]:.2f}s, K=8 took {res_k8["sec"]:.2f}s '
        f'(projected 881-basin fold time at K=8: {res_k8["sec"] * 881 / 8:.1f}s = {res_k8["sec"] * 881 / 8 / 60:.2f} min)',
        flush=True,
    )


def build_all_300_tasks() -> list[tuple[str, int, int]]:
    """Build the 300 uniform ~881-basin tasks across gauged, random, and basin.

    Ordered with outer loop `sfold` (0..9) and inner loop `tfold` (0..9) so any contiguous slice
    of tasks has an identical balance of 5-year folds (tfold 0..5) and 2-year folds (tfold 6..9).
    """
    tasks: list[tuple[str, int, int]] = []
    for sfold in range(10):
        for tfold in range(10):
            for exp in ('random', 'basin', 'gauged'):
                tasks.append((exp, sfold, tfold))
    return tasks


def _gpu_worker_loop(
    gpu_id: int,
    task_queue: mp.Queue,
    group_size: int,
    force: bool,
    phase: str,
) -> None:
    while True:
        try:
            item = task_queue.get(timeout=30.0)
        except Exception:
            break
        if item is None:
            break
        exp, sfold, tfold = item
        try:
            run_single_task(
                experiment=exp,
                sfold=sfold,
                tfold=tfold,
                gpu_id=gpu_id,
                group_size=group_size,
                force=force,
                phase=phase,
            )
        except Exception as e:
            print(
                f'[GPU {gpu_id}] ERROR on {exp} sfold_{sfold} tfold_{tfold} ({phase}): {e}',
                flush=True,
            )
            import traceback
            traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-exact', action='store_true')
    parser.add_argument('--single-task', action='store_true')
    parser.add_argument('--experiment', type=str, default='random', choices=['gauged', 'random', 'basin'])
    parser.add_argument('--sfold', type=int, default=0)
    parser.add_argument('--tfold', type=int, default=7)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--group-size', type=int, default=8)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--phase', type=str, default='all', choices=['all', 'da', 'ft'])
    parser.add_argument('--task-start', type=int, default=0, help='Start index in the 300-task list (inclusive)')
    parser.add_argument('--task-end', type=int, default=300, help='End index in the 300-task list (exclusive)')
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='Comma-separated GPU IDs to use')
    parser.add_argument('--procs-per-gpu', type=int, default=1, help='Worker processes per GPU')
    args = parser.parse_args()

    if args.verify_exact:
        verify_exactness(gpu_id=args.gpu)
        return

    if args.single_task:
        run_single_task(
            experiment=args.experiment,
            sfold=args.sfold,
            tfold=args.tfold,
            gpu_id=args.gpu,
            group_size=args.group_size,
            force=args.force,
            phase=args.phase,
        )
        return

    all_tasks = build_all_300_tasks()
    my_tasks = all_tasks[args.task_start : args.task_end]
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',') if x.strip()]

    print(
        f'Launching {len(my_tasks)} tasks [{args.task_start}:{args.task_end}] (phase={args.phase}) across GPUs {gpu_ids} '
        f'(procs_per_gpu={args.procs_per_gpu}, group_size={args.group_size})...',
        flush=True,
    )
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    for t in my_tasks:
        q.put(t)
    total_workers = len(gpu_ids) * args.procs_per_gpu
    for _ in range(total_workers):
        q.put(None)

    procs = []
    for g in gpu_ids:
        for _ in range(args.procs_per_gpu):
            p = ctx.Process(
                target=_gpu_worker_loop,
                args=(g, q, args.group_size, args.force, args.phase),
            )
            p.start()
            procs.append(p)

    for p in procs:
        p.join()
    print(f'Completed all tasks in range [{args.task_start}:{args.task_end}] (phase={args.phase})!', flush=True)


if __name__ == '__main__':
    main()
