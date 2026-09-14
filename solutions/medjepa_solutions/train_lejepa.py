from __future__ import annotations

import logging
import math
import time
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from medjepa.data import build_data_bundle, set_sampler_epoch
from medjepa.training.checkpoint import load_checkpoint, save_checkpoint
from medjepa.training.distributed import (
    DistributedEnvironment,
    barrier,
    cleanup_distributed,
    init_distributed,
    reduce_max,
    reduce_mean,
    unwrap_ddp,
    wrap_ddp,
)
from medjepa.training.metrics import RunningAverage, feature_diagnostics
from medjepa.training.optim import build_optimizer, build_scheduler, gradient_norm
from medjepa.training.precision import PrecisionManager
from medjepa.training.profiler import make_profiler
from medjepa.training.runtime import (
    configure_logging,
    make_run_dir,
    make_tracker,
    move_lejepa_batch,
    peak_memory_mb,
    resolve_config,
    save_resolved_config,
    synchronize_device,
    training_parser,
)
from medjepa.utils import environment_summary, parameter_count, seed_everything
from solutions.medjepa_solutions.lejepa import build_lejepa
from solutions.medjepa_solutions.sigreg import LeJEPAObjective, SIGReg


LOGGER = logging.getLogger(__name__)


def _sync_context(model: nn.Module, should_sync: bool):
    if isinstance(model, DistributedDataParallel) and not should_sync:
        return model.no_sync()
    return nullcontext()


def _reduced_float(value: float, env: DistributedEnvironment) -> float:
    tensor = torch.tensor(value, device=env.device, dtype=torch.float32)
    return float(reduce_mean(tensor, env).item())


def _reduced_max_float(value: float, env: DistributedEnvironment) -> float:
    tensor = torch.tensor(value, device=env.device, dtype=torch.float32)
    return float(reduce_max(tensor, env).item())


def train_one_epoch(
    *,
    model: nn.Module,
    objective: LeJEPAObjective,
    loader: object,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    precision: PrecisionManager,
    cfg: object,
    env: DistributedEnvironment,
    tracker: object,
    profiler: object,
    epoch: int,
    global_step: int,
    max_steps: int | None,
) -> tuple[int, bool, dict[str, float]]:
    model.train()
    objective.train()
    accumulation = int(cfg.optimization.gradient_accumulation_steps)
    clip = getattr(cfg.optimization, "gradient_clip_norm", None)
    clip = None if clip is None else float(clip)
    log_every = int(cfg.tracking.log_every_steps)
    total_average = RunningAverage()
    invariance_average = RunningAverage()
    sigreg_average = RunningAverage()
    optimizer.zero_grad(set_to_none=True)
    group_total = 0.0
    group_invariance = 0.0
    group_sigreg = 0.0
    group_examples = 0
    group_data_time = 0.0
    previous_iteration_end = time.perf_counter()
    group_start_time = previous_iteration_end
    group_lr = scheduler.value(global_step)
    last_projections: torch.Tensor | None = None
    epoch_completed = True

    for batch_index, batch in enumerate(loader):
        batch_ready_time = time.perf_counter()
        batch_data_time = batch_ready_time - previous_iteration_end
        group_start_index = (batch_index // accumulation) * accumulation
        group_size = min(accumulation, len(loader) - group_start_index)
        is_group_start = batch_index == group_start_index
        is_group_end = (batch_index + 1) == (group_start_index + group_size)
        if is_group_start:
            group_total = 0.0
            group_invariance = 0.0
            group_sigreg = 0.0
            group_examples = 0
            group_data_time = 0.0
            group_start_time = previous_iteration_end
            group_lr = scheduler.step(global_step)
        group_data_time += batch_data_time

        views, _labels = move_lejepa_batch(batch, env.device)
        with _sync_context(model, is_group_end):
            with precision.autocast():
                output = model(views)
                losses = objective(output["projections"])
                raw_loss = losses["loss"]
                loss = raw_loss / group_size
            if not torch.isfinite(raw_loss):
                raise FloatingPointError(f"Non-finite LeJEPA loss: {raw_loss.item()}")
            precision.backward(loss)

        group_total += float(losses["loss"].detach().item())
        group_invariance += float(losses["invariance"].detach().item())
        group_sigreg += float(losses["sigreg"].detach().item())
        group_examples += int(views.shape[0])
        last_projections = output["projections"].detach()
        profiler.step()

        if not is_group_end:
            previous_iteration_end = time.perf_counter()
            continue

        precision.prepare_gradients(optimizer, model, clip)
        grad_norm = gradient_norm(model)
        precision.step(optimizer)
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        synchronize_device(env.device)
        group_end_time = time.perf_counter()
        elapsed = max(group_end_time - group_start_time, 1e-9)
        previous_iteration_end = group_end_time

        step_total = _reduced_float(group_total / group_size, env)
        step_invariance = _reduced_float(group_invariance / group_size, env)
        step_sigreg = _reduced_float(group_sigreg / group_size, env)
        step_grad = _reduced_float(grad_norm, env)
        step_seconds = _reduced_max_float(elapsed, env)
        data_seconds = _reduced_max_float(group_data_time, env)
        step_throughput = group_examples * env.world_size / step_seconds
        total_average.update(step_total)
        invariance_average.update(step_invariance)
        sigreg_average.update(step_sigreg)

        if global_step == 1 or global_step % log_every == 0:
            diagnostics = feature_diagnostics(last_projections)
            diagnostics = {
                key: _reduced_float(value, env) for key, value in diagnostics.items()
            }
            peak_memory = _reduced_max_float(peak_memory_mb(env.device), env)
            metrics = {
                "train/loss": step_total,
                "train/invariance": step_invariance,
                "train/sigreg": step_sigreg,
                "train/lr": group_lr,
                "train/grad_norm": step_grad,
                "train/images_per_second": step_throughput,
                "train/encoded_views_per_second": step_throughput * int(views.shape[1]),
                "train/step_seconds": step_seconds,
                "train/data_seconds": data_seconds,
                "train/data_fraction": min(data_seconds / step_seconds, 1.0),
                "train/peak_memory_mb": peak_memory,
                "train/epoch": epoch + 1,
                **diagnostics,
            }
            tracker.log(metrics, global_step)
            if env.is_main:
                LOGGER.info(
                    "epoch=%d step=%d loss=%.5f inv=%.5f sigreg=%.5f lr=%.2e "
                    "img/s=%.1f data=%.3fs step=%.3fs mem=%.0fMB",
                    epoch + 1,
                    global_step,
                    step_total,
                    step_invariance,
                    step_sigreg,
                    group_lr,
                    step_throughput,
                    data_seconds,
                    step_seconds,
                    metrics["train/peak_memory_mb"],
                )

        if max_steps is not None and global_step >= max_steps:
            epoch_completed = batch_index == len(loader) - 1
            break

    return global_step, epoch_completed, {
        "train/loss": total_average.average,
        "train/invariance": invariance_average.average,
        "train/sigreg": sigreg_average.average,
    }


def main() -> None:
    args = training_parser("Train the reference LeJEPA implementation").parse_args()
    cfg = resolve_config(args.config, args.override)
    if cfg.algorithm != "lejepa":
        raise ValueError("The selected config is not a LeJEPA config")
    env = init_distributed()
    configure_logging(env)
    seed_everything(int(cfg.experiment.seed))
    run_dir = make_run_dir(cfg)
    save_resolved_config(cfg, run_dir, env)

    data = build_data_bundle(cfg, "lejepa", env)
    model = build_lejepa(cfg).to(env.device)
    if env.distributed and env.device.type == "cuda":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    sigreg = SIGReg(
        num_slices=int(cfg.sigreg.num_slices),
        t_max=float(cfg.sigreg.t_max),
        num_points=int(cfg.sigreg.num_points),
        seed=int(cfg.sigreg.seed),
    ).to(env.device)
    objective = LeJEPAObjective(sigreg, float(cfg.sigreg.weight)).to(env.device)
    optimizer = build_optimizer(model, cfg.optimization)
    accumulation = int(cfg.optimization.gradient_accumulation_steps)
    steps_per_epoch = math.ceil(len(data.train_loader) / accumulation)
    planned_steps = steps_per_epoch * int(cfg.optimization.epochs)
    total_steps = min(planned_steps, args.max_steps) if args.max_steps else planned_steps
    scheduler = build_scheduler(optimizer, cfg.optimization, steps_per_epoch, total_steps)
    precision = PrecisionManager(str(cfg.optimization.precision), env.device)

    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, env.device)
        model.load_state_dict(checkpoint["model"])
        objective.load_state_dict(checkpoint["objective"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler_state = checkpoint.get("scaler")
        if scaler_state:
            precision.scaler.load_state_dict(scaler_state)
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])

    model = wrap_ddp(model, env)
    tracker = make_tracker(cfg, run_dir, env)
    profiler = make_profiler(cfg.profiler, run_dir, env.rank)
    if env.is_main:
        LOGGER.info("Reference LeJEPA parameters: %s", f"{parameter_count(unwrap_ddp(model)):,}")
        LOGGER.info(
            "device=%s world=%d local_batch=%d global_batch=%d views=%d precision=%s",
            env.device,
            env.world_size,
            int(cfg.data.batch_size),
            int(cfg.data.batch_size) * env.world_size * accumulation,
            int(cfg.data.views.count),
            precision.name,
        )

    try:
        with profiler:
            for epoch in range(start_epoch, int(cfg.optimization.epochs)):
                set_sampler_epoch(data.train_sampler, epoch)
                global_step, completed, metrics = train_one_epoch(
                    model=model,
                    objective=objective,
                    loader=data.train_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    precision=precision,
                    cfg=cfg,
                    env=env,
                    tracker=tracker,
                    profiler=profiler,
                    epoch=epoch,
                    global_step=global_step,
                    max_steps=args.max_steps,
                )
                if env.is_main:
                    payload = {
                        "algorithm": "lejepa",
                        "implementation": "solution",
                        "config": cfg.to_dict(),
                        "environment": environment_summary(),
                        "epoch": epoch + int(completed),
                        "global_step": global_step,
                        "model": unwrap_ddp(model).state_dict(),
                        "objective": objective.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": precision.scaler.state_dict(),
                        "epoch_metrics": metrics,
                    }
                    save_checkpoint(run_dir / "latest.pt", payload)
                    every = int(cfg.checkpoint.every_epochs)
                    if completed and every > 0 and (epoch + 1) % every == 0:
                        save_checkpoint(run_dir / f"epoch-{epoch + 1:04d}.pt", payload)
                barrier()
                if args.max_steps is not None and global_step >= args.max_steps:
                    break
    finally:
        tracker.finish()
        cleanup_distributed()


if __name__ == "__main__":
    main()
