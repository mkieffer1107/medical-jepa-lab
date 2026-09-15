from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import torch
import yaml

from medjepa.config import Config, load_config
from medjepa.training.distributed import DistributedEnvironment
from medjepa.training.tracking import Tracker
from medjepa.utils import ensure_dir


def training_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to a YAML config")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Repeatable dotted override, for example data.batch_size=16",
    )
    parser.add_argument("--resume", default=None, help="Checkpoint path")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after optimizer steps")
    return parser


def resolve_config(config_path: str, overrides: Iterable[str]) -> Config:
    return load_config(config_path, overrides)


def configure_logging(env: DistributedEnvironment) -> None:
    level = logging.INFO if env.is_main else logging.WARNING
    logging.basicConfig(
        level=level,
        format=f"%(asctime)s | rank={env.rank} | %(levelname)s | %(message)s",
        force=True,
    )


def make_run_dir(cfg: Config) -> Path:
    return ensure_dir(Path(cfg.experiment.output_dir) / str(cfg.experiment.run_name))


def save_resolved_config(cfg: Config, run_dir: Path, env: DistributedEnvironment) -> None:
    if not env.is_main:
        return
    log_path = (run_dir / "training.log").resolve()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger(__name__).info(
        "Starting %s run=%s device=%s world_size=%d | log=%s",
        cfg.algorithm, cfg.experiment.run_name, env.device, env.world_size, log_path,
    )
    with (run_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, sort_keys=False)


def make_tracker(cfg: Config, run_dir: Path, env: DistributedEnvironment) -> Tracker:
    return Tracker(
        backend=str(cfg.tracking.backend),
        project=str(cfg.tracking.project),
        run_name=str(cfg.experiment.run_name),
        config=cfg.to_dict(),
        run_dir=run_dir,
        enabled=env.is_main,
        space_id=getattr(cfg.tracking, "space_id", None),
    )


def move_ijepa_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        "images": batch["images"].to(device, non_blocking=True),
        "labels": batch["labels"].to(device, non_blocking=True),
        "context_masks": [mask.to(device, non_blocking=True) for mask in batch["context_masks"]],
        "target_masks": [mask.to(device, non_blocking=True) for mask in batch["target_masks"]],
    }


def move_lejepa_batch(batch: tuple[torch.Tensor, torch.Tensor], device: torch.device):
    views, labels = batch
    return views.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024**2
