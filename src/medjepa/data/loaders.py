from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from medjepa.data.datasets import DatasetInfo, build_dataset
from medjepa.data.masks import MultiBlockMaskCollator
from medjepa.data.transforms import MultiViewTransform, evaluation_transform, ijepa_transform
from medjepa.utils import seed_worker


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    info: DatasetInfo
    train_sampler: Any


def _sampler(dataset: object, distributed: bool, rank: int, world_size: int, train: bool):
    if distributed:
        return DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=train,
            drop_last=train,
        )
    return RandomSampler(dataset) if train else SequentialSampler(dataset)


def _loader(
    dataset: object,
    sampler: object,
    cfg: object,
    collate_fn: object | None,
    train: bool,
    seed: int,
    rank: int,
) -> DataLoader:
    workers = int(cfg.num_workers)
    generator = torch.Generator().manual_seed(seed + rank)
    return DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        sampler=sampler,
        num_workers=workers,
        pin_memory=bool(cfg.pin_memory),
        persistent_workers=bool(cfg.persistent_workers) and workers > 0,
        drop_last=train,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_data_bundle(cfg: object, algorithm: str, dist_env: object) -> DataBundle:
    image_size = int(cfg.data.image_size)
    if algorithm == "ijepa":
        train_transform = ijepa_transform(image_size, cfg.data.augmentation)
        mask_cfg = cfg.mask
        collator = MultiBlockMaskCollator(
            image_size=image_size,
            patch_size=int(cfg.model.patch_size),
            context_scale=tuple(float(v) for v in mask_cfg.context_scale),
            target_scale=tuple(float(v) for v in mask_cfg.target_scale),
            target_aspect_ratio=tuple(float(v) for v in mask_cfg.target_aspect_ratio),
            num_context_masks=int(mask_cfg.num_context_masks),
            num_target_masks=int(mask_cfg.num_target_masks),
            min_keep=int(mask_cfg.min_keep),
            allow_overlap=bool(mask_cfg.allow_overlap),
        )
    elif algorithm == "lejepa":
        train_transform = MultiViewTransform(image_size, cfg.data.views)
        collator = None
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    eval_transform = evaluation_transform(image_size)
    train_dataset, info = build_dataset(cfg.data, "train", train_transform)
    val_dataset, val_info = build_dataset(cfg.data, "val", eval_transform)
    test_dataset, test_info = build_dataset(cfg.data, "test", eval_transform)
    if info.class_names != val_info.class_names or info.class_names != test_info.class_names:
        raise RuntimeError("Dataset metadata differs across splits")

    train_sampler = _sampler(
        train_dataset, dist_env.distributed, dist_env.rank, dist_env.world_size, train=True
    )
    val_sampler = _sampler(
        val_dataset, dist_env.distributed, dist_env.rank, dist_env.world_size, train=False
    )
    test_sampler = _sampler(
        test_dataset, dist_env.distributed, dist_env.rank, dist_env.world_size, train=False
    )
    seed = int(cfg.experiment.seed)
    return DataBundle(
        train_loader=_loader(
            train_dataset, train_sampler, cfg.data, collator, True, seed, dist_env.rank
        ),
        val_loader=_loader(
            val_dataset, val_sampler, cfg.data, None, False, seed + 1, dist_env.rank
        ),
        test_loader=_loader(
            test_dataset, test_sampler, cfg.data, None, False, seed + 2, dist_env.rank
        ),
        info=info,
        train_sampler=train_sampler,
    )


def set_sampler_epoch(sampler: object, epoch: int) -> None:
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)


@dataclass
class EvaluationBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    info: DatasetInfo


def build_evaluation_bundle(cfg: object, batch_size: int | None = None) -> EvaluationBundle:
    """Build deterministic full-image loaders for frozen-representation evaluation."""
    transform = evaluation_transform(int(cfg.data.image_size))
    train_dataset, info = build_dataset(cfg.data, "train", transform)
    val_dataset, val_info = build_dataset(cfg.data, "val", transform)
    test_dataset, test_info = build_dataset(cfg.data, "test", transform)
    if info.class_names != val_info.class_names or info.class_names != test_info.class_names:
        raise RuntimeError("Dataset metadata differs across splits")

    workers = int(cfg.data.num_workers)
    loader_kwargs = {
        "batch_size": int(batch_size or cfg.data.batch_size),
        "num_workers": workers,
        "pin_memory": bool(cfg.data.pin_memory),
        "persistent_workers": bool(cfg.data.persistent_workers) and workers > 0,
        "drop_last": False,
        "worker_init_fn": seed_worker,
    }
    return EvaluationBundle(
        train_loader=DataLoader(train_dataset, shuffle=False, **loader_kwargs),
        val_loader=DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        test_loader=DataLoader(test_dataset, shuffle=False, **loader_kwargs),
        info=info,
    )
