from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, Subset


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    class_names: list[str]
    num_classes: int


class LabelSqueezingDataset(Dataset):
    """Normalize datasets to return `(image, int_label)` pairs."""

    def __init__(self, dataset: Dataset, info: DatasetInfo) -> None:
        self.dataset = dataset
        self.info = info

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.dataset[index]
        label_array = np.asarray(label).reshape(-1)
        if label_array.size != 1:
            raise ValueError("This lab currently supports single-label MedMNIST tasks only")
        return image, int(label_array[0])


class SyntheticMedicalDataset(Dataset):
    """Deterministic colored cell-like patterns for offline smoke tests."""

    def __init__(
        self,
        split: str,
        samples: int,
        classes: int,
        image_size: int,
        transform: Callable[[Image.Image], torch.Tensor],
        seed: int,
    ) -> None:
        self.split = split
        self.samples = samples
        self.classes = classes
        self.image_size = image_size
        self.transform = transform
        split_offset = {"train": 0, "val": 10_000, "test": 20_000}[split]
        self.seed = seed + split_offset

    def __len__(self) -> int:
        return self.samples

    def _render(self, index: int, label: int) -> Image.Image:
        rng = np.random.default_rng(self.seed + index)
        size = self.image_size
        background = np.array([238, 224, 224], dtype=np.float32)
        noise = rng.normal(0, 7, size=(size, size, 3))
        pixels = np.clip(background + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(pixels, mode="RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        margin = max(3, size // 8)
        jitter = max(1, size // 20)
        x0 = margin + int(rng.integers(-jitter, jitter + 1))
        y0 = margin + int(rng.integers(-jitter, jitter + 1))
        x1 = size - margin + int(rng.integers(-jitter, jitter + 1))
        y1 = size - margin + int(rng.integers(-jitter, jitter + 1))
        palette = [
            (111, 45, 150, 220),
            (35, 113, 179, 220),
            (206, 65, 75, 220),
            (235, 148, 55, 220),
            (50, 150, 95, 220),
            (120, 90, 55, 220),
        ]
        color = palette[label % len(palette)]
        pattern = label % 4
        if pattern == 0:
            draw.ellipse((x0, y0, x1, y1), fill=color, outline=(70, 30, 90, 255), width=2)
        elif pattern == 1:
            draw.rectangle((x0, y0, x1, y1), fill=color, outline=(20, 60, 120, 255), width=2)
        elif pattern == 2:
            points = [(size // 2, y0), (x1, y1), (x0, y1)]
            draw.polygon(points, fill=color, outline=(120, 30, 40, 255))
        else:
            width = max(3, size // 7)
            draw.line((x0, size // 2, x1, size // 2), fill=color, width=width)
            draw.line((size // 2, y0, size // 2, y1), fill=color, width=width)
        nucleus_radius = max(2, size // 10)
        cx = size // 2 + int(rng.integers(-jitter, jitter + 1))
        cy = size // 2 + int(rng.integers(-jitter, jitter + 1))
        draw.ellipse(
            (cx - nucleus_radius, cy - nucleus_radius, cx + nucleus_radius, cy + nucleus_radius),
            fill=(45, 25, 85, 230),
        )
        return image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        label = index % self.classes
        image = self._render(index, label)
        return self.transform(image), label


def _medmnist_dataset(
    name: str,
    split: str,
    root: str | Path,
    image_size: int,
    download: bool,
    transform: Callable,
) -> tuple[Dataset, DatasetInfo]:
    try:
        import medmnist
    except ImportError as exc:
        raise RuntimeError("Install project dependencies with `uv sync` to use MedMNIST") from exc

    if name not in medmnist.INFO:
        available = ", ".join(sorted(medmnist.INFO))
        raise ValueError(f"Unknown MedMNIST dataset {name!r}. Available: {available}")
    metadata: dict[str, Any] = medmnist.INFO[name]
    task = str(metadata.get("task", ""))
    if "multi-label" in task:
        raise ValueError(f"{name} is multi-label; this lab's evaluator expects one class per image")
    dataset_class = getattr(medmnist, metadata["python_class"])
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Loading %s split=%s size=%d root=%s (download enabled=%s)",
        name, split, image_size, root, download,
    )
    dataset = dataset_class(
        split=split,
        root=str(root),
        transform=transform,
        download=download,
        size=image_size,
        as_rgb=True,
    )
    label_map = metadata["label"]
    class_names = [
        str(label_map[str(i)] if str(i) in label_map else label_map[i])
        for i in range(len(label_map))
    ]
    info = DatasetInfo(name=name, class_names=class_names, num_classes=len(class_names))
    return LabelSqueezingDataset(dataset, info), info


def _synthetic_dataset(
    cfg: object,
    split: str,
    transform: Callable,
) -> tuple[Dataset, DatasetInfo]:
    count_key = f"synthetic_{split}_samples"
    samples = int(getattr(cfg, count_key, 64 if split == "train" else 24))
    classes = int(getattr(cfg, "synthetic_classes", 4))
    dataset = SyntheticMedicalDataset(
        split=split,
        samples=samples,
        classes=classes,
        image_size=int(cfg.image_size),
        transform=transform,
        seed=0,
    )
    info = DatasetInfo(
        name="synthetic",
        class_names=[f"synthetic-{i}" for i in range(classes)],
        num_classes=classes,
    )
    return dataset, info


def build_dataset(cfg: object, split: str, transform: Callable) -> tuple[Dataset, DatasetInfo]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")
    if str(cfg.dataset) == "synthetic":
        dataset, info = _synthetic_dataset(cfg, split, transform)
    else:
        dataset, info = _medmnist_dataset(
            name=str(cfg.dataset),
            split=split,
            root=cfg.root,
            image_size=int(cfg.image_size),
            download=bool(cfg.download),
            transform=transform,
        )

    fraction = float(getattr(cfg, "train_fraction", 1.0)) if split == "train" else 1.0
    if not 0 < fraction <= 1:
        raise ValueError("data.train_fraction must be in (0, 1]")
    if fraction < 1:
        keep = max(1, math.floor(len(dataset) * fraction))
        generator = torch.Generator().manual_seed(0)
        indices = torch.randperm(len(dataset), generator=generator)[:keep].tolist()
        dataset = Subset(dataset, indices)
    LOGGER.info("Dataset ready: %s split=%s samples=%d", info.name, split, len(dataset))
    return dataset, info
