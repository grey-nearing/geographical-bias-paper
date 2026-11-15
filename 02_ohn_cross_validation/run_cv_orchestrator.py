"""Multi-GPU Cross-Validation Orchestrator for Geographical Bias Paper (Public Caravan).

Runs all 210 cross-validation models (21 spatial splits x 10 temporal folds covering 1986-2023
with +-1 year buffers) across the 4x NVIDIA Tesla T4 GPUs on watershed-theory-vm.

Features:
- Pre-warms precomputed memory-mapped arrays into OS page cache so all 4 GPU workers share
  the 26.5 GB dataset in RAM without duplication.
- Trains each model for 30 epochs x 2,000 updates/epoch = 60,000 updates (batch_size=512,
  hidden_size=512, seq_length=365, forecast_overlap=365, lead_time=7, predict_last_n=8, FP16 AMP).
- Saves resumable checkpoints every 5 epochs (`checkpoint_latest.pt`) and final weights
  (`model_epoch030.pt`).
- Evaluates out-of-sample predictions (`(N_test_basins, N_test_days, 8)` float32) on the
  held-out test basins and held-out test window (`preds.npz`) and writes a `DONE` marker.
- Automatically runs `compute_cv_metrics.py` as soon as an experiment (`gauged`, `random`,
  or `basin`) completes all its folds.
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
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
from torch.amp import GradScaler, autocast
import yaml

from model.modelzoo import get_model
from model.training import get_loss_obj, get_regularization_obj
from model.utils.config import Config

from fast_multimet import (
    FastMultimet,
    FastMultimetDataLoader,
    fast_evaluate_period,
    get_precomputed_store,
)

ROOT_DIR = Path('./data')
RUNS_DIR = ROOT_DIR / 'runs'
LOGS_DIR = ROOT_DIR / 'logs'
SPLITS_DIR = ROOT_DIR / 'splits'

# 10 Temporal Folds covering 1986-01-01 .. 2023-12-31 with +-1 year (365-day) buffers,
# matching ./data
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
        'train_start': ['01/01/1982'],
        'train_end': ['31/12/2020'],
    },
]


def get_lr_for_epoch(epoch: int, total_epochs: int = 30) -> float:
    """Step learning rate schedule for 30-epoch (60,000-update) training."""
    if total_epochs == 30:
        if epoch <= 15:
            return 5e-4
        elif epoch <= 25:
            return 1e-4
        else:
            return 2e-5
    # Fallback proportional schedule
    frac = epoch / max(total_epochs, 1)
    if frac <= 0.5:
        return 5e-4
    elif frac <= 0.833:
        return 1e-4
    return 2e-5


def build_all_tasks() -> list[dict[str, Any]]:
    """Build ordered list of 210 CV tasks: gauged (10), random (100), basin (100)."""
    tasks: list[dict[str, Any]] = []

    # 1. Gauged (1 spatial split x 10 temporal folds = 10 tasks)
    for tf in TEMPORAL_FOLDS:
        tasks.append({
            'experiment': 'gauged',
            'sfold': 0,
            'tfold': tf['tfold'],
            'tfold_name': tf['name'],
            'train_basin_file': str(SPLITS_DIR / 'gauged' / 'all_basins.txt'),
            'test_basin_file': str(SPLITS_DIR / 'gauged' / 'all_basins.txt'),
            'train_start': tf['train_start'],
            'train_end': tf['train_end'],
            'test_start': tf['test_start'],
            'test_end': tf['test_end'],
        })

    # 2. Random 10-fold CV (10 spatial folds x 10 temporal folds = 100 tasks)
    for sfold in range(10):
        for tf in TEMPORAL_FOLDS:
            tasks.append({
                'experiment': 'random',
                'sfold': sfold,
                'tfold': tf['tfold'],
                'tfold_name': tf['name'],
                'train_basin_file': str(SPLITS_DIR / 'random' / f'fold_{sfold}_train.txt'),
                'test_basin_file': str(SPLITS_DIR / 'random' / f'fold_{sfold}_test.txt'),
                'train_start': tf['train_start'],
                'train_end': tf['train_end'],
                'test_start': tf['test_start'],
                'test_end': tf['test_end'],
            })

    # 3. Hydrologically separated (MAIN_BAS) 10-fold CV (10 spatial folds x 10 temporal folds = 100 tasks)
    for sfold in range(10):
        for tf in TEMPORAL_FOLDS:
            tasks.append({
                'experiment': 'basin',
                'sfold': sfold,
                'tfold': tf['tfold'],
                'tfold_name': tf['name'],
                'train_basin_file': str(SPLITS_DIR / 'basin' / f'fold_{sfold}_train.txt'),
                'test_basin_file': str(SPLITS_DIR / 'basin' / f'fold_{sfold}_test.txt'),
                'train_start': tf['train_start'],
                'train_end': tf['train_end'],
                'test_start': tf['test_start'],
                'test_end': tf['test_end'],
            })

    return tasks


def get_task_run_dir(task: dict[str, Any]) -> Path:
    return (
        RUNS_DIR
        / task['experiment']
        / f"sfold_{task['sfold']}"
        / f"tfold_{task['tfold']}_{task['tfold_name']}"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_single_task(
    task: dict[str, Any],
    base_cfg_dict: dict[str, Any],
    gpu_id: int,
    epochs: int,
    max_updates_per_epoch: int,
    progress_log_path: Path,
) -> dict[str, Any]:
    """Train and evaluate a single (experiment, sfold, tfold) model on `cuda:{gpu_id}`."""
    run_dir = get_task_run_dir(task)
    done_file = run_dir / 'DONE'
    if done_file.exists() and (run_dir / 'preds.npz').exists():
        return {'status': 'skipped_already_done', 'run_dir': str(run_dir)}

    run_dir.mkdir(parents=True, exist_ok=True)
    device_str = f'cuda:{gpu_id}'
    device = torch.device(device_str)
    torch.cuda.set_device(device)

    seed = 42000 + task['sfold'] * 100 + task['tfold']
    set_seed(seed)

    cfg_dict = copy.deepcopy(base_cfg_dict)
    cfg_dict['experiment_name'] = (
        f"{task['experiment']}_sfold{task['sfold']}_tfold{task['tfold']}_{task['tfold_name']}"
    )
    cfg_dict['run_dir'] = str(run_dir)
    cfg_dict['train_dir'] = str(run_dir / 'train_data')
    cfg_dict['img_log_dir'] = str(run_dir / 'img_log')
    cfg_dict['device'] = device_str
    cfg_dict['seed'] = seed
    cfg_dict['epochs'] = epochs
    cfg_dict['max_updates_per_epoch'] = max_updates_per_epoch
    cfg_dict['train_basin_file'] = task['train_basin_file']
    cfg_dict['validation_basin_file'] = task['test_basin_file']
    cfg_dict['test_basin_file'] = task['test_basin_file']
    cfg_dict['train_start_date'] = task['train_start']
    cfg_dict['train_end_date'] = task['train_end']
    cfg_dict['validation_start_date'] = task['test_start']
    cfg_dict['validation_end_date'] = task['test_end']
    cfg_dict['test_start_date'] = task['test_start']
    cfg_dict['test_end_date'] = task['test_end']
    cfg_dict['statics_data_dir'] = str(ROOT_DIR / 'data' / 'attributes.zarr')
    cfg_dict['targets_data_dir'] = str(ROOT_DIR / 'data' / 'streamflow.zarr')
    cfg_dict['dynamics_data_dir'] = str(ROOT_DIR / 'data' / 'dynamics')
    cfg_dict['lazy_load'] = False

    cfg = Config(cfg_dict)
    cfg.dump_config(folder=run_dir)

    t_task_start = time.time()

    # Check if we are resuming from an intermediate checkpoint
    ckpt_latest = run_dir / 'checkpoint_latest.pt'
    scaler_exists = (run_dir / 'scaler.zarr').exists()
    ds_train = FastMultimet(
        cfg=cfg,
        is_train=True,
        period='train',
        compute_scaler=(not scaler_exists),
    )

    model = get_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=get_lr_for_epoch(1, epochs))
    loss_obj = get_loss_obj(cfg)
    loss_obj.set_regularization_terms(get_regularization_obj(cfg=cfg))
    grad_scaler = GradScaler('cuda', enabled=True)

    start_epoch = 1
    loss_history: list[dict[str, Any]] = []
    if ckpt_latest.exists():
        try:
            ckpt = torch.load(ckpt_latest, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'grad_scaler_state_dict' in ckpt:
                grad_scaler.load_state_dict(ckpt['grad_scaler_state_dict'])
            start_epoch = int(ckpt['epoch']) + 1
            loss_history = ckpt.get('loss_history', [])
            print(
                f"[GPU {gpu_id}] Resuming {cfg_dict['experiment_name']} from epoch {start_epoch}/{epochs}",
                flush=True,
            )
        except Exception as e:
            print(
                f"[GPU {gpu_id}] Warning: failed to load checkpoint {ckpt_latest} ({e}); starting from epoch 1.",
                flush=True,
            )
            start_epoch = 1

    noise_sampler_y = None
    if cfg.target_noise_std is not None and cfg.target_noise_std > 0:
        noise_sampler_y = torch.distributions.Normal(loc=0.0, scale=cfg.target_noise_std)

    num_samples_per_epoch = min(len(ds_train), max_updates_per_epoch * cfg.batch_size)

    for epoch in range(start_epoch, epochs + 1):
        t_ep0 = time.time()
        lr = get_lr_for_epoch(epoch, epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        model.train()
        sampler = torch.utils.data.RandomSampler(
            ds_train, replacement=False, num_samples=num_samples_per_epoch
        )
        batch_sampler = torch.utils.data.BatchSampler(
            sampler, batch_size=cfg.batch_size, drop_last=True
        )
        loader = FastMultimetDataLoader(
            ds_train, batch_sampler=batch_sampler, prefetch_batches=3
        )

        epoch_losses: list[float] = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            for k in ('x_d_hindcast', 'x_d_forecast'):
                batch[k] = {
                    name: t.to(device, non_blocking=True)
                    for name, t in batch[k].items()
                }
            batch['x_s'] = batch['x_s'].to(device, non_blocking=True)
            batch['y'] = batch['y'].to(device, non_blocking=True)
            batch['basin_index'] = batch['basin_index'].to(device, non_blocking=True)
            if 'per_basin_target_stds' in batch:
                batch['per_basin_target_stds'] = batch['per_basin_target_stds'].to(
                    device, non_blocking=True
                )

            with autocast('cuda', enabled=True):
                batch = model.pre_model_hook(batch, is_train=True)
                preds = model(batch)
                if noise_sampler_y is not None:
                    y_Target = batch['y'][:, -cfg.predict_last_n :, :]
                    batch['y'][:, -cfg.predict_last_n :, :] = (
                        y_Target + noise_sampler_y.sample(y_Target.shape).to(device=device, dtype=y_Target.dtype)
                    )
                loss, _ = loss_obj(preds, batch)

            grad_scaler.scale(loss).backward()
            if cfg.clip_gradient_norm is not None:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.clip_gradient_norm
                )
            grad_scaler.step(optimizer)
            grad_scaler.update()
            epoch_losses.append(float(loss.detach().cpu()))

        torch.cuda.synchronize(device)
        ep_time = time.time() - t_ep0
        mean_loss = float(np.nanmean(epoch_losses))
        loss_history.append({
            'epoch': epoch,
            'lr': lr,
            'mean_loss': mean_loss,
            'epoch_sec': round(ep_time, 2),
        })

        if epoch % 5 == 0 or epoch == epochs:
            ckpt_payload = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'grad_scaler_state_dict': grad_scaler.state_dict(),
                'loss_history': loss_history,
            }
            tmp_ckpt = run_dir / 'checkpoint_latest.pt.tmp'
            torch.save(ckpt_payload, tmp_ckpt)
            tmp_ckpt.replace(ckpt_latest)
            print(
                f"[GPU {gpu_id}] {cfg_dict['experiment_name']} | Epoch {epoch:02d}/{epochs:02d} "
                f"| lr={lr:.1e} | loss={mean_loss:.4f} | {ep_time:.1f}s",
                flush=True,
            )

    # Save final model weights
    final_weights_path = run_dir / f'model_epoch{epochs:03d}.pt'
    torch.save(model.state_dict(), final_weights_path)

    # Evaluate on held-out test period & held-out test basins
    t_eval0 = time.time()
    ds_test = FastMultimet(
        cfg=cfg, is_train=False, period='test', compute_scaler=False
    )
    preds_unscaled, _obs_unscaled, test_basins, test_dates = fast_evaluate_period(
        model=model,
        dataset=ds_test,
        device=device,
        batch_size=2048,
    )
    torch.cuda.synchronize(device)
    eval_sec = time.time() - t_eval0

    preds_tmp = run_dir / 'preds_tmp.npz'
    preds_final = run_dir / 'preds.npz'
    np.savez(
        preds_tmp,
        preds=preds_unscaled.astype(np.float32),
        basins=np.array(test_basins, dtype='<U32'),
        dates=test_dates.values,
    )
    preds_tmp.replace(preds_final)

    with open(run_dir / 'loss_history.json', 'w') as f:
        json.dump(loss_history, f, indent=2)

    total_sec = time.time() - t_task_start
    summary = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'gpu_id': gpu_id,
        'experiment': task['experiment'],
        'sfold': task['sfold'],
        'tfold': task['tfold'],
        'tfold_name': task['tfold_name'],
        'n_train_basins': len(ds_train._basins),
        'n_test_basins': len(test_basins),
        'n_test_days': len(test_dates),
        'preds_shape': list(preds_unscaled.shape),
        'final_loss': loss_history[-1]['mean_loss'] if loss_history else None,
        'eval_sec': round(eval_sec, 2),
        'total_sec': round(total_sec, 2),
        'run_dir': str(run_dir),
    }
    done_file.write_text(json.dumps(summary, indent=2) + '\n')

    with open(progress_log_path, 'a') as f:
        f.write(json.dumps(summary) + '\n')

    print(
        f"[GPU {gpu_id}] COMPLETED {cfg_dict['experiment_name']} in {total_sec/60:.1f} min "
        f"(eval {eval_sec:.1f}s, shape={preds_unscaled.shape})",
        flush=True,
    )
    del model, optimizer, loss_obj, grad_scaler, ds_train, ds_test, preds_unscaled
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def gpu_worker_loop(
    gpu_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    base_cfg_dict: dict[str, Any],
    epochs: int,
    max_updates_per_epoch: int,
    progress_log_path: Path,
) -> None:
    """Worker process pinned to `cuda:{gpu_id}` pulling tasks from `task_queue`."""
    print(f'[GPU {gpu_id}] Worker started (PID={os.getpid()})', flush=True)
    # Initialize mmap store once per worker
    get_precomputed_store()

    while True:
        task = task_queue.get()
        if task is None:
            break

        try:
            res = run_single_task(
                task=task,
                base_cfg_dict=base_cfg_dict,
                gpu_id=gpu_id,
                epochs=epochs,
                max_updates_per_epoch=max_updates_per_epoch,
                progress_log_path=progress_log_path,
            )
            result_queue.put({'ok': True, 'task': task, 'result': res})
        except Exception as e:
            tb = traceback.format_exc()
            print(
                f"[GPU {gpu_id}] ERROR on task {task['experiment']} sfold={task['sfold']} tfold={task['tfold']}: {e}\n{tb}",
                flush=True,
            )
            result_queue.put({'ok': False, 'task': task, 'error': str(e), 'traceback': tb})


def prewarm_page_cache() -> None:
    """Touch precomputed .npy files sequentially so OS caches them in the 125 GB RAM."""
    t0 = time.time()
    pre_dir = ROOT_DIR / 'data' / 'precomputed'
    total_bytes = 0
    for name in [
        'statics.npy',
        'streamflow.npy',
        'hindcast_2d.npy',
        'forecast_3d_post2016.npy',
        'forecast_3d_sum.npy',
        'forecast_3d_sq_sum.npy',
        'forecast_3d_count.npy',
    ]:
        p = pre_dir / name
        if p.exists():
            arr = np.load(p, mmap_mode='r')
            # Touch every 4096-byte page via fast strided sum
            _ = float(arr.reshape(-1)[::512].sum())
            total_bytes += p.stat().st_size
    print(
        f'Pre-warmed {total_bytes / 1e9:.2f} GB of precomputed arrays into OS page cache in {time.time() - t0:.2f}s',
        flush=True,
    )


def check_and_run_experiment_metrics(experiment: str, all_tasks: list[dict[str, Any]]) -> bool:
    """If all tasks for `experiment` are DONE, invoke `compute_cv_metrics.py`."""
    exp_tasks = [t for t in all_tasks if t['experiment'] == experiment]
    if not all((get_task_run_dir(t) / 'DONE').exists() for t in exp_tasks):
        return False

    metrics_done_marker = ROOT_DIR / 'results' / 'hydro_model_metrics' / f'{experiment}.DONE'
    if metrics_done_marker.exists():
        return True

    print(f'=== All folds for {experiment} are DONE! Computing stitched 1986-2023 -> 2016-2023 metrics... ===', flush=True)
    cmd = [
        sys.executable,
        str(ROOT_DIR / 'code' / 'compute_cv_metrics.py'),
        '--experiment',
        experiment,
    ]
    res = subprocess.run(cmd, check=False)
    if res.returncode == 0:
        metrics_done_marker.parent.mkdir(parents=True, exist_ok=True)
        metrics_done_marker.write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + '\n')
        return True
    print(f'Warning: compute_cv_metrics.py exited with code {res.returncode} for {experiment}', flush=True)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--max-updates-per-epoch', type=int, default=2000)
    parser.add_argument('--gpus', type=str, default='0,1,2,3')
    parser.add_argument(
        '--experiments',
        type=str,
        default='gauged,random,basin',
        help='Comma-separated list of experiments to run in order',
    )
    parser.add_argument(
        '--max-tasks',
        type=int,
        default=0,
        help='If > 0, limit total tasks run (useful for smoke tests)',
    )
    parser.add_argument(
        '--tfold',
        type=int,
        default=-1,
        help='If >= 0, filter tasks to only this temporal fold index',
    )
    args = parser.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    progress_log_path = LOGS_DIR / 'orchestrator_progress.jsonl'

    with open(ROOT_DIR / 'configs' / 'base_v2_config.yml', 'r') as f:
        base_cfg_dict = yaml.safe_load(f)
    with open(
        ROOT_DIR
        / 'code'
        / 'flood-forecasting'
        / 'model'
        / 'example-configs'
        / 'floodhub-settings-config.yml',
        'r',
    ) as f:
        fh_cfg = yaml.safe_load(f)
    base_cfg_dict['union_mapping'] = fh_cfg['union_mapping']

    wanted_exps = [e.strip() for e in args.experiments.split(',') if e.strip()]
    all_tasks = [t for t in build_all_tasks() if t['experiment'] in wanted_exps]
    if args.tfold >= 0:
        all_tasks = [t for t in all_tasks if t['tfold'] == args.tfold]
    pending_tasks = [
        t
        for t in all_tasks
        if not (
            (get_task_run_dir(t) / 'DONE').exists()
            and (get_task_run_dir(t) / 'preds.npz').exists()
        )
    ]
    already_completed = len(all_tasks) - len(pending_tasks)
    if args.max_tasks > 0:
        pending_tasks = pending_tasks[: args.max_tasks]

    gpu_ids = [int(x.strip()) for x in args.gpus.split(',') if x.strip()]
    print(
        f'=== CV Orchestrator Starting ===\n'
        f'  Total configured tasks: {len(all_tasks)}\n'
        f'  Already completed:      {already_completed}\n'
        f'  Pending tasks to run:   {len(pending_tasks)}\n'
        f'  GPUs:                   {gpu_ids}\n'
        f'  Training budget:        {args.epochs} epochs x {args.max_updates_per_epoch} updates/epoch '
        f'({args.epochs * args.max_updates_per_epoch:,} updates/model)',
        flush=True,
    )

    if not pending_tasks:
        for exp in wanted_exps:
            check_and_run_experiment_metrics(exp, all_tasks)
        print('All tasks already completed!', flush=True)
        return

    prewarm_page_cache()

    ctx = mp.get_context('spawn')
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    for t in pending_tasks:
        task_queue.put(t)
    for _ in gpu_ids:
        task_queue.put(None)

    workers = []
    for gid in gpu_ids:
        p = ctx.Process(
            target=gpu_worker_loop,
            args=(
                gid,
                task_queue,
                result_queue,
                base_cfg_dict,
                args.epochs,
                args.max_updates_per_epoch,
                progress_log_path,
            ),
        )
        p.start()
        workers.append(p)

    completed_count = 0
    failed_count = 0
    total_to_wait = len(pending_tasks)
    while completed_count + failed_count < total_to_wait:
        msg = result_queue.get()
        if msg['ok']:
            completed_count += 1
            exp = msg['task']['experiment']
            if args.max_tasks == 0 and args.tfold < 0:
                check_and_run_experiment_metrics(exp, all_tasks)
        else:
            failed_count += 1
        print(
            f'[Orchestrator] Progress: {completed_count}/{total_to_wait} succeeded, {failed_count} failed.',
            flush=True,
        )

    for p in workers:
        p.join()

    if args.max_tasks == 0 and args.tfold < 0:
        for exp in wanted_exps:
            check_and_run_experiment_metrics(exp, all_tasks)

    print(
        f'=== Orchestrator Finished: {completed_count} succeeded, {failed_count} failed ===',
        flush=True,
    )
    if failed_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
