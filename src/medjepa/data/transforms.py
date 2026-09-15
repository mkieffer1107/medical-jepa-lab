from __future__ import annotations

from collections.abc import Callable

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def evaluation_transform(image_size: int) -> Callable[[Image.Image], torch.Tensor]:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def ijepa_transform(image_size: int, augmentation: object) -> Callable[[Image.Image], torch.Tensor]:
    operations: list[Callable] = [
        transforms.RandomResizedCrop(
            image_size,
            scale=tuple(float(v) for v in augmentation.crop_scale),
            ratio=tuple(float(v) for v in augmentation.crop_ratio),
            interpolation=InterpolationMode.BICUBIC,
        )
    ]
    horizontal_flip = float(getattr(augmentation, "horizontal_flip", 0.0))
    if horizontal_flip > 0:
        operations.append(transforms.RandomHorizontalFlip(horizontal_flip))
    color_jitter = float(getattr(augmentation, "color_jitter", 0.0))
    if color_jitter > 0:
        jitter = transforms.ColorJitter(
            brightness=color_jitter,
            contrast=color_jitter,
            saturation=color_jitter,
            hue=min(0.5 * color_jitter, 0.1),
        )
        operations.append(transforms.RandomApply([jitter], p=0.8))
    operations.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return transforms.Compose(operations)


class MultiViewTransform:
    """Apply independently sampled augmentation pipelines to one source image."""

    def __init__(self, image_size: int, view_config: object) -> None:
        self.count = int(view_config.count)
        if self.count < 2:
            raise ValueError("LeJEPA requires at least two views")

        operations: list[Callable] = [
            transforms.RandomResizedCrop(
                image_size,
                scale=tuple(float(v) for v in view_config.crop_scale),
                ratio=tuple(float(v) for v in view_config.crop_ratio),
                interpolation=InterpolationMode.BICUBIC,
            )
        ]
        horizontal_flip = float(getattr(view_config, "horizontal_flip", 0.0))
        if horizontal_flip > 0:
            operations.append(transforms.RandomHorizontalFlip(horizontal_flip))
        color_jitter = float(getattr(view_config, "color_jitter", 0.0))
        if color_jitter > 0:
            jitter = transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=min(0.5 * color_jitter, 0.1),
            )
            operations.append(transforms.RandomApply([jitter], p=0.8))
        grayscale = float(getattr(view_config, "grayscale", 0.0))
        if grayscale > 0:
            operations.append(transforms.RandomGrayscale(grayscale))
        operations.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        self.transform = transforms.Compose(operations)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return torch.stack([self.transform(image) for _ in range(self.count)], dim=0)


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)
