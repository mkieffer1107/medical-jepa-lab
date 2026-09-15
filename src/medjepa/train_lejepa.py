from __future__ import annotations

import logging

import torch
from torch import nn

from medjepa.data import build_data_bundle, set_sampler_epoch
from medjepa.losses.sigreg import LeJEPAObjective, SIGReg
from medjepa.models.lejepa import build_lejepa
from medjepa.training.checkpoint import load_checkpoint, save_checkpoint
from medjepa.training.distributed import (
    DistributedEnvironment,
    barrier,
    cleanup_distributed,
    init_distributed,
    unwrap_ddp,
    wrap_ddp,
)
from medjepa.training.optim import build_optimizer, build_scheduler
from medjepa.training.precision import PrecisionManager
from medjepa.training.profiler import make_profiler
from medjepa.training.runtime import (
    configure_logging,
    make_run_dir,
    make_tracker,
    resolve_config,
    save_resolved_config,
    training_parser,
)
from medjepa.utils import environment_summary, parameter_count, seed_everything


LOGGER = logging.getLogger(__name__)


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
    """Train LeJEPA for one epoch or until the global optimizer-step limit.

    Args:
        model: LeJEPAModel, possibly DDP-wrapped, returning projection features.
        objective: LeJEPAObjective returning scalar loss, invariance, and sigreg.
        loader: Sized iterable of (views, labels) batches: views (B, V, 3, H, H)
            and labels (B,). Labels are unused during pretraining.
        optimizer: Optimizer for the trainable model parameters.
        scheduler: Learning-rate scheduler exposing value(step) and step(step).
        precision: PrecisionManager for autocast, backward, gradient preparation,
            and optimizer stepping.
        cfg: Resolved config; optimization controls accumulation, precision and
            clipping, and tracking.log_every_steps controls reporting frequency.
        env: DistributedEnvironment providing device, rank, world_size, and is_main.
        tracker: Logger with log(metrics, step); emit logs only on the main rank.
        profiler: Active profiler (or no-op profiler) with step().
        epoch: Zero-based epoch index for reporting.
        global_step: Number of optimizer steps before this call, not minibatches.
        max_steps: Absolute global optimizer-step limit, or None for no limit.

    Returns:
        (new_global_step, epoch_completed, epoch_metrics). The step count
        includes this call's optimizer steps. epoch_completed is True only if
        the whole loader was consumed, including when the limit hits its end.
        epoch_metrics contains Python floats under "train/loss",
        "train/invariance", and "train/sigreg", averaged over the processed
        portion of the epoch; component losses are unweighted.

    Side effects:
        Update model/optimizer/precision and objective state; step the LR scheduler
        and profiler and emit metrics. Train the model and objective; handle the
        final partial accumulation group. Report detached, distributed loss
        statistics and feature-collapse diagnostics without using labels.
    """
    # TODO 31:
    # - encode every augmented view with the one trainable model;
    # - compute invariance and SIGReg terms;
    # - optimize with accumulation/AMP as configured;
    # - report collapse diagnostics, timing/throughput, LR, and memory;
    # - step the profiler and respect max_steps.
    raise NotImplementedError("TODO 31: implement the LeJEPA training loop")


def main() -> None:
    args = training_parser("Train the exercise LeJEPA implementation").parse_args()
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
    steps_per_epoch = (len(data.train_loader) + accumulation - 1) // accumulation
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
        precision.scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])

    model = wrap_ddp(model, env)
    tracker = make_tracker(cfg, run_dir, env)
    profiler = make_profiler(cfg.profiler, run_dir, env.rank)
    if env.is_main:
        LOGGER.info("Trainable parameters: %s", f"{parameter_count(unwrap_ddp(model)):,}")
        LOGGER.info("Device: %s | world size: %d", env.device, env.world_size)

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
                        "implementation": "exercise",
                        "config": cfg.to_dict(),
                        "environment": environment_summary(),
                        "epoch": epoch + int(completed),
                        "global_step": global_step,
                        "training_complete": bool(
                            (completed and epoch + 1 >= int(cfg.optimization.epochs))
                            or (args.max_steps is not None and global_step >= args.max_steps)
                        ),
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
