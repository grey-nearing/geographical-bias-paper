"""Verify exact numerical equivalence between FastMultimet and stock Multimet,
and benchmark full-scale training & evaluation speed on 1x Tesla T4 GPU.
"""

import copy
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch.amp import GradScaler, autocast
import yaml

from model.datasetzoo.multimet import Multimet, MultimetDataLoader
from model.modelzoo import get_model
from model.training import get_loss_obj, get_regularization_obj
from model.utils.config import Config

from fast_multimet import (
    FastMultimet,
    FastMultimetDataLoader,
    fast_evaluate_period,
)


def make_test_cfg(base_cfg_dict: dict, run_dir: Path, basin_file: Path, train_start, train_end, test_start, test_end) -> Config:
    d = copy.deepcopy(base_cfg_dict)
    root = Path('./data')
    d['run_dir'] = str(run_dir)
    d['train_basin_file'] = str(basin_file)
    d['validation_basin_file'] = str(basin_file)
    d['test_basin_file'] = str(basin_file)
    d['train_start_date'] = train_start
    d['train_end_date'] = train_end
    d['validation_start_date'] = test_start
    d['validation_end_date'] = test_end
    d['test_start_date'] = test_start
    d['test_end_date'] = test_end
    d['statics_data_dir'] = str(root / 'data' / 'attributes.zarr')
    d['targets_data_dir'] = str(root / 'data' / 'streamflow.zarr')
    d['dynamics_data_dir'] = str(root / 'data' / 'dynamics')
    d['device'] = 'cuda:0'
    d['lazy_load'] = False
    run_dir.mkdir(parents=True, exist_ok=True)
    return Config(d)


def verify_equivalence(base_cfg_dict: dict, tmp_dir: Path) -> None:
    print('=== STEP 1: Numerical Equivalence Check (FastMultimet vs Stock Multimet) ===', flush=True)
    basins_all = (
        Path('./data/data/precomputed/basins.txt')
        .read_text()
        .strip()
        .splitlines()
    )
    # Pick 10 basins across different subdatasets and keep static attributes with std > 0 on those 10
    sub_idx = np.linspace(0, len(basins_all) - 1, 10, dtype=int)
    sub_basins = sorted([basins_all[i] for i in sub_idx])
    statics_all = np.load('./data/data/precomputed/statics.npy', mmap_mode='r')
    static_attrs_all = (
        Path('./data/data/precomputed/static_attributes.txt')
        .read_text()
        .strip()
        .splitlines()
    )
    sub_std = statics_all[sub_idx].std(axis=0)
    nonzero_static_attrs = [a for a, s in zip(static_attrs_all, sub_std) if s > 1e-6]
    test_cfg_dict = copy.deepcopy(base_cfg_dict)
    test_cfg_dict['static_attributes'] = nonzero_static_attrs

    basin_file = tmp_dir / 'sub10_basins.txt'
    basin_file.write_text('\n'.join(sub_basins) + '\n')

    run_stock = tmp_dir / 'run_stock'
    run_fast = tmp_dir / 'run_fast'
    shutil.rmtree(run_stock, ignore_errors=True)
    shutil.rmtree(run_fast, ignore_errors=True)

    # Span 2015-06-01 .. 2016-06-30 so both pre-2016 (ERA5-Land fallback) and post-2016 (HRES/GraphCast) are tested!
    cfg_stock = make_test_cfg(
        test_cfg_dict,
        run_stock,
        basin_file,
        train_start='01/06/2015',
        train_end='30/06/2016',
        test_start='01/01/2016',
        test_end='31/03/2016',
    )
    cfg_fast = make_test_cfg(
        test_cfg_dict,
        run_fast,
        basin_file,
        train_start='01/06/2015',
        train_end='30/06/2016',
        test_start='01/01/2016',
        test_end='31/03/2016',
    )

    t0 = time.time()
    ds_stock = Multimet(cfg=cfg_stock, is_train=True, period='train', compute_scaler=True)
    t_stock = time.time() - t0

    t0 = time.time()
    ds_fast = FastMultimet(cfg=cfg_fast, is_train=True, period='train', compute_scaler=True)
    t_fast = time.time() - t0
    print(f'  10-basin init time: Stock Multimet = {t_stock:.2f}s vs FastMultimet = {t_fast:.4f}s', flush=True)

    # 1. Compare Scaler values
    max_rel_err = 0.0
    for var in ds_fast.scaler.scaler.data_vars:
        v_s = ds_stock.scaler.scaler[var].values
        v_f = ds_fast.scaler.scaler[var].values
        rel_err = np.max(np.abs(v_s - v_f) / np.maximum(np.abs(v_s), 1e-6))
        max_rel_err = max(max_rel_err, float(rel_err))
        assert np.allclose(v_s, v_f, rtol=1e-4, atol=1e-4), (
            f'Scaler mismatch for {var}: stock={v_s}, fast={v_f}'
        )
    print(f'  [OK] Scaler matches across all {len(ds_fast.scaler.scaler.data_vars)} variables (max rel diff = {max_rel_err:.2e})', flush=True)

    # 2. Compare sample counts & sample_index
    assert len(ds_stock) == len(ds_fast), f'Len mismatch: {len(ds_stock)} vs {len(ds_fast)}'
    assert np.array_equal(
        ds_stock._sample_index.get_column('basin'),
        ds_fast._sample_index.get_column('basin'),
    ), 'Basin sample_index column mismatch'
    assert np.array_equal(
        ds_stock._sample_index.get_column('date'),
        ds_fast._sample_index.get_column('date'),
    ), 'Date sample_index column mismatch'
    print(f'  [OK] Valid sample index matches identically ({len(ds_fast)} valid samples)', flush=True)

    # 3. Compare batch tensors across pre-2016 and post-2016 samples
    test_indices = np.linspace(0, len(ds_fast) - 1, 32, dtype=int).tolist()
    batch_fast = ds_fast.get_batch(test_indices)
    batch_stock = ds_fast.collate_fn([ds_stock[i] for i in test_indices])

    assert np.array_equal(batch_fast['date'], batch_stock['date']), 'Date array mismatch'
    assert torch.equal(batch_fast['basin_index'], batch_stock['basin_index'].to(torch.int64)), 'Basin index mismatch'
    assert torch.allclose(batch_fast['x_s'], batch_stock['x_s'], rtol=1e-3, atol=1e-3), 'x_s mismatch'
    for k in batch_stock['x_d_hindcast']:
        assert torch.allclose(
            batch_fast['x_d_hindcast'][k],
            batch_stock['x_d_hindcast'][k].to(torch.float32),
            rtol=1e-3,
            atol=1e-3,
        ), f'x_d_hindcast[{k}] mismatch'
    for k in batch_stock['x_d_forecast']:
        assert torch.allclose(
            batch_fast['x_d_forecast'][k],
            batch_stock['x_d_forecast'][k].to(torch.float32),
            rtol=1e-3,
            atol=1e-3,
        ), f'x_d_forecast[{k}] mismatch'

    y_f = batch_fast['y']
    y_s = batch_stock['y']
    assert torch.equal(torch.isnan(y_f), torch.isnan(y_s)), 'y NaN mask mismatch'
    valid_y = ~torch.isnan(y_s)
    assert torch.allclose(y_f[valid_y], y_s[valid_y], rtol=1e-3, atol=1e-3), 'y values mismatch'
    print('  [OK] Batch tensors (x_s, x_d_hindcast, x_d_forecast, y, date, basin_index) match 100%!', flush=True)


def benchmark_full_scale(base_cfg_dict: dict, tmp_dir: Path) -> None:
    print('\n=== STEP 2: Full-Scale 8,811-Basin Benchmark on 1x Tesla T4 GPU ===', flush=True)
    root = Path('./data')
    run_bench = tmp_dir / 'run_bench_full'
    shutil.rmtree(run_bench, ignore_errors=True)

    # Use random fold_0 train (7,929 basins) and test (882 basins) with test year 2016
    # and 1-year (365-day) buffer before and after 2016 withheld from training:
    # Train intervals: 01/01/1982 .. 31/12/2014 and 01/01/2018 .. 31/12/2023
    cfg = make_test_cfg(
        base_cfg_dict,
        run_bench,
        root / 'splits' / 'random' / 'fold_0_train.txt',
        train_start=['01/01/1982', '01/01/2018'],
        train_end=['31/12/2014', '31/12/2023'],
        test_start='01/01/2016',
        test_end='31/12/2016',
    )
    cfg._cfg['test_basin_file'] = root / 'splits' / 'random' / 'fold_0_test.txt'

    t0 = time.time()
    ds_train = FastMultimet(cfg=cfg, is_train=True, period='train', compute_scaler=True)
    t_init = time.time() - t0
    print(
        f'  Full 7,929-basin train dataset + Scaler init: {t_init:.2f}s '
        f'({len(ds_train):,} valid training samples)',
        flush=True,
    )

    device = torch.device('cuda:0')
    model = get_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_obj = get_loss_obj(cfg)
    loss_obj.set_regularization_terms(get_regularization_obj(cfg=cfg))
    grad_scaler = GradScaler('cuda', enabled=True)

    sampler = torch.utils.data.RandomSampler(
        ds_train, replacement=False, num_samples=cfg.max_updates_per_epoch * cfg.batch_size
    )
    batch_sampler = torch.utils.data.BatchSampler(
        sampler, batch_size=cfg.batch_size, drop_last=False
    )
    loader = FastMultimetDataLoader(ds_train, batch_sampler=batch_sampler)

    # Warmup 5 batches + benchmark 100 batches to project exact 2,000-batch epoch time
    model.train()
    t_start_100 = None
    losses = []
    for i, batch in enumerate(loader):
        if i == 5:
            torch.cuda.synchronize(device)
            t_start_100 = time.time()
        if i == 105:
            break

        optimizer.zero_grad(set_to_none=True)
        for k in ('x_d_hindcast', 'x_d_forecast'):
            batch[k] = {
                name: t.to(device, non_blocking=True) for name, t in batch[k].items()
            }
        batch['x_s'] = batch['x_s'].to(device, non_blocking=True)
        batch['y'] = batch['y'].to(device, non_blocking=True)
        batch['basin_index'] = batch['basin_index'].to(device, non_blocking=True)

        with autocast('cuda', enabled=True):
            batch = model.pre_model_hook(batch, is_train=True)
            preds = model(batch)
            loss, _ = loss_obj(preds, batch)

        grad_scaler.scale(loss).backward()
        if cfg.clip_gradient_norm is not None:
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_gradient_norm)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        losses.append(float(loss.detach().cpu()))

    torch.cuda.synchronize(device)
    t_100 = time.time() - t_start_100
    sec_per_batch = t_100 / 100.0
    sec_per_epoch_2000 = sec_per_batch * 2000.0
    print(
        f'  100 training batches (batch_size=512, seq_len=365, hidden=512, AMP): {t_100:.2f}s '
        f'({sec_per_batch*1000:.1f} ms/batch -> {sec_per_epoch_2000:.1f}s / 2,000-update epoch, '
        f'mean loss={np.nanmean(losses):.4f})',
        flush=True,
    )

    # Benchmark evaluation on 1-year test period (2016) for fold_0_test (882 basins)
    t0 = time.time()
    ds_test = FastMultimet(cfg=cfg, is_train=False, period='test', compute_scaler=False)
    preds_unscaled, obs_unscaled, test_basins, test_dates = fast_evaluate_period(
        model, ds_test, device=device, batch_size=512
    )
    torch.cuda.synchronize(device)
    t_eval = time.time() - t0
    print(
        f'  1-year test evaluation (882 basins x {len(test_dates)} days = {len(ds_test):,} samples): '
        f'{t_eval:.2f}s, output shape={preds_unscaled.shape}',
        flush=True,
    )


def main() -> None:
    root = Path('./data')
    with open(root / 'configs' / 'base_v2_config.yml', 'r') as f:
        base_cfg_dict = yaml.safe_load(f)
    with open(
        root
        / 'code'
        / 'flood-forecasting'
        / 'model'
        / 'example-configs'
        / 'floodhub-settings-config.yml',
        'r',
    ) as f:
        fh_cfg = yaml.safe_load(f)
    base_cfg_dict['union_mapping'] = fh_cfg['union_mapping']

    tmp_dir = root / 'scratch_test'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    verify_equivalence(base_cfg_dict, tmp_dir)
    benchmark_full_scale(base_cfg_dict, tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
