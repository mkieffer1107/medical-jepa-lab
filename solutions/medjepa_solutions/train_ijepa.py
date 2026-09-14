from __future__ import annotations

import logging
import math
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
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
from medjepa.training.optim import (
    build_optimizer,
    build_scheduler,
    ema_momentum,
    gradient_norm,
)
from medjepa.training.precision import PrecisionManager
from medjepa.training.profiler import make_profiler
from medjepa.training.runtime import (
    configure_logging,
    make_run_dir,
    make_tracker,
    move_ijepa_batch,
    peak_memory_mb,
    resolve_config,
    save_resolved_config,
    synchronize_device,
    training_parser,
)
from medjepa.utils import environment_summary, parameter_count, seed_everything
from solutions.medjepa_solutions.ijepa import (
    build_ijepa,
    extract_target_tokens,
    update_target_encoder,
)


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
    student: nn.Module,
    target_encoder: nn.Module,
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
    total_steps: int,
    max_steps: int | None,
) -> tuple[int, bool, dict[str, float]]:
    student.train()
    target_encoder.eval()
    accumulation = int(cfg.optimization.gradient_accumulation_steps)
    clip = getattr(cfg.optimization, "gradient_clip_norm", None)
    clip = None if clip is None else float(clip)
    log_every = int(cfg.tracking.log_every_steps)
    loss_average = RunningAverage()
    optimizer.zero_grad(set_to_none=True)
    group_loss = 0.0
    group_examples = 0
    group_data_time = 0.0
    previous_iteration_end = time.perf_counter()
    group_start_time = previous_iteration_end
    group_lr = scheduler.value(global_step)
    last_features: torch.Tensor | None = None
    epoch_completed = True

    for batch_index, batch in enumerate(loader):
        batch_ready_time = time.perf_counter()
        batch_data_time = batch_ready_time - previous_iteration_end
        group_start_index = (batch_index // accumulation) * accumulation
        group_size = min(accumulation, len(loader) - group_start_index)
        is_group_start = batch_index == group_start_index
        is_group_end = (batch_index + 1) == (group_start_index + group_size)
        if is_group_start:
            group_loss = 0.0
            group_examples = 0
            group_data_time = 0.0
            group_start_time = previous_iteration_end
            group_lr = scheduler.step(global_step)
        group_data_time += batch_data_time

        batch = move_ijepa_batch(batch, env.device)
        images = batch["images"]
        context_masks = batch["context_masks"]
        target_masks = batch["target_masks"]

        with _sync_context(student, is_group_end):
            with precision.autocast():
                with torch.no_grad():
                    full_target_tokens = target_encoder.forward_tokens(images)
                    full_target_tokens = F.layer_norm(
                        full_target_tokens, (full_target_tokens.shape[-1],)
                    )
                    targets = extract_target_tokens(
                        full_target_tokens, context_masks, target_masks
                    )
                predictions = student(images, context_masks, target_masks)
                if predictions.shape != targets.shape:
                    raise RuntimeError(
                        f"Prediction/target mismatch: {predictions.shape} vs {targets.shape}"
                    )
                raw_loss = F.smooth_l1_loss(predictions, targets)
                loss = raw_loss / group_size
            if not torch.isfinite(raw_loss):
                raise FloatingPointError(f"Non-finite I-JEPA loss: {raw_loss.item()}")
            precision.backward(loss)

        group_loss += float(raw_loss.detach().item())
        group_examples += int(images.shape[0])
        last_features = full_target_tokens.detach().mean(dim=1)
        profiler.step()

        if not is_group_end:
            previous_iteration_end = time.perf_counter()
            continue

        precision.prepare_gradients(optimizer, student, clip)
        grad_norm = gradient_norm(student)
        precision.step(optimizer)
        optimizer.zero_grad(set_to_none=True)

        unwrapped = unwrap_ddp(student)
        momentum = ema_momentum(
            global_step,
            total_steps,
            float(cfg.optimization.ema_start),
            float(cfg.optimization.ema_end),
        )
        update_target_encoder(unwrapped.context_encoder, target_encoder, momentum)
        global_step += 1
        synchronize_device(env.device)
        group_end_time = time.perf_counter()
        elapsed = max(group_end_time - group_start_time, 1e-9)
        previous_iteration_end = group_end_time

        step_loss = _reduced_float(group_loss / group_size, env)
        step_grad = _reduced_float(grad_norm, env)
        step_seconds = _reduced_max_float(elapsed, env)
        data_seconds = _reduced_max_float(group_data_time, env)
        step_throughput = group_examples * env.world_size / step_seconds
        loss_average.update(step_loss)

        if global_step == 1 or global_step % log_every == 0:
            diagnostics = feature_diagnostics(last_features)
            diagnostics = {
                key: _reduced_float(value, env) for key, value in diagnostics.items()
            }
            peak_memory = _reduced_max_float(peak_memory_mb(env.device), env)
            metrics = {
                "train/loss": step_loss,
                "train/lr": group_lr,
                "train/ema_momentum": momentum,
                "train/grad_norm": step_grad,
                "train/images_per_second": step_throughput,
                "train/step_seconds": step_seconds,
                "train/data_seconds": data_seconds,
                "train/data_fraction": min(data_seconds / step_seconds, 1.0),
                "train/peak_memory_mb": peak_memory,
                "train/context_patches": int(context_masks[0].shape[1]),
                "train/target_patches": int(target_masks[0].shape[1]),
                "train/epoch": epoch + 1,
                **diagnostics,
            }
            tracker.log(metrics, global_step)
            if env.is_main:
                LOGGER.info(
                    "epoch=%d step=%d loss=%.5f lr=%.2e ema=%.5f img/s=%.1f "
                    "data=%.3fs step=%.3fs mem=%.0fMB",
                    epoch + 1,
                    global_step,
                    step_loss,
                    group_lr,
                    momentum,
                    step_throughput,
                    data_seconds,
                    step_seconds,
                    metrics["train/peak_memory_mb"],
                )

        if max_steps is not None and global_step >= max_steps:
            epoch_completed = batch_index == len(loader) - 1
            break

    return global_step, epoch_completed, {"train/loss": loss_average.average}


def main() -> None:
    args = training_parser("Train the reference I-JEPA implementation").parse_args()
    cfg = resolve_config(args.config, args.override)
    if cfg.algorithm != "ijepa":
        raise ValueError("The selected config is not an I-JEPA config")
    env = init_distributed()
    configure_logging(env)
    seed_everything(int(cfg.experiment.seed))
    run_dir = make_run_dir(cfg)
    save_resolved_config(cfg, run_dir, env)

    data = build_data_bundle(cfg, "ijepa", env)
    student, target_encoder = build_ijepa(cfg)
    student.to(env.device)
    target_encoder.to(env.device)
    optimizer = build_optimizer(student, cfg.optimization)
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
        student.load_state_dict(checkpoint["student"])
        target_encoder.load_state_dict(checkpoint["target_encoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler_state = checkpoint.get("scaler")
        if scaler_state:
            precision.scaler.load_state_dict(scaler_state)
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])

    student = wrap_ddp(student, env)
    tracker = make_tracker(cfg, run_dir, env)
    profiler = make_profiler(cfg.profiler, run_dir, env.rank)
    if env.is_main:
        LOGGER.info("Reference I-JEPA parameters: %s", f"{parameter_count(unwrap_ddp(student)):,}")
        LOGGER.info(
            "device=%s world=%d local_batch=%d global_batch=%d precision=%s",
            env.device,
            env.world_size,
            int(cfg.data.batch_size),
            int(cfg.data.batch_size) * env.world_size * accumulation,
            precision.name,
        )

    try:
        with profiler:
            for epoch in range(start_epoch, int(cfg.optimization.epochs)):
                set_sampler_epoch(data.train_sampler, epoch)
                global_step, completed, metrics = train_one_epoch(
                    student=student,
                    target_encoder=target_encoder,
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
                    total_steps=total_steps,
                    max_steps=args.max_steps,
                )
                if env.is_main:
                    payload = {
                        "algorithm": "ijepa",
                        "implementation": "solution",
                        "config": cfg.to_dict(),
                        "environment": environment_summary(),
                        "epoch": epoch + int(completed),
                        "global_step": global_step,
                        "student": unwrap_ddp(student).state_dict(),
                        "target_encoder": target_encoder.state_dict(),
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
