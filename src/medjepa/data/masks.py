from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import default_collate


@dataclass(frozen=True)
class Rectangle:
    height: int
    width: int


class MultiBlockMaskCollator:
    """Generate patch-index context and target masks for an I-JEPA batch.

    Block sizes are shared across a batch so tensors can be stacked. Locations are
    sampled per image. When overlap is disabled, context patches lying inside any
    target block are removed, then all masks are truncated to a common length.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        context_scale: tuple[float, float],
        target_scale: tuple[float, float],
        target_aspect_ratio: tuple[float, float],
        num_context_masks: int,
        num_target_masks: int,
        min_keep: int,
        allow_overlap: bool,
    ) -> None:
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.grid_height = image_size // patch_size
        self.grid_width = image_size // patch_size
        self.context_scale = context_scale
        self.target_scale = target_scale
        self.target_aspect_ratio = target_aspect_ratio
        self.num_context_masks = num_context_masks
        self.num_target_masks = num_target_masks
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap

    @property
    def num_patches(self) -> int:
        return self.grid_height * self.grid_width

    def _sample_rectangle(
        self,
        scale: tuple[float, float],
        aspect_ratio: tuple[float, float],
    ) -> Rectangle:
        area_fraction = torch.empty(()).uniform_(*scale).item()
        ratio = torch.empty(()).uniform_(*aspect_ratio).item()
        area = max(1, round(self.num_patches * area_fraction))
        height = max(1, round(math.sqrt(area * ratio)))
        width = max(1, round(math.sqrt(area / ratio)))
        height = min(height, self.grid_height)
        width = min(width, self.grid_width)
        return Rectangle(height=height, width=width)

    def _indices_for_rectangle(self, rectangle: Rectangle) -> torch.Tensor:
        max_top = self.grid_height - rectangle.height
        max_left = self.grid_width - rectangle.width
        top = int(torch.randint(max_top + 1, (1,)).item())
        left = int(torch.randint(max_left + 1, (1,)).item())
        grid = torch.zeros((self.grid_height, self.grid_width), dtype=torch.bool)
        grid[top : top + rectangle.height, left : left + rectangle.width] = True
        return torch.nonzero(grid.flatten(), as_tuple=False).flatten()

    def __call__(self, batch: list[tuple[torch.Tensor, int]]) -> dict[str, object]:
        images, labels = default_collate(batch)
        batch_size = images.shape[0]

        target_rectangle = self._sample_rectangle(
            self.target_scale, self.target_aspect_ratio
        )
        context_rectangle = self._sample_rectangle(self.context_scale, (1.0, 1.0))

        per_image_contexts: list[list[torch.Tensor]] = []
        per_image_targets: list[list[torch.Tensor]] = []

        for _ in range(batch_size):
            targets = [
                self._indices_for_rectangle(target_rectangle)
                for _ in range(self.num_target_masks)
            ]
            target_union = torch.zeros(self.num_patches, dtype=torch.bool)
            for target in targets:
                target_union[target] = True

            contexts: list[torch.Tensor] = []
            for _ in range(self.num_context_masks):
                context = self._indices_for_rectangle(context_rectangle)
                if not self.allow_overlap:
                    context = context[~target_union[context]]
                if context.numel() < self.min_keep:
                    complement = torch.nonzero(~target_union, as_tuple=False).flatten()
                    if complement.numel() < self.min_keep:
                        raise RuntimeError("Mask settings leave too few visible context patches")
                    permutation = torch.randperm(complement.numel())
                    context = complement[permutation[: max(self.min_keep, context.numel())]]
                contexts.append(context)

            per_image_contexts.append(contexts)
            per_image_targets.append(targets)

        min_context = min(mask.numel() for group in per_image_contexts for mask in group)
        min_target = min(mask.numel() for group in per_image_targets for mask in group)
        context_masks = [
            torch.stack([per_image_contexts[b][m][:min_context] for b in range(batch_size)])
            for m in range(self.num_context_masks)
        ]
        target_masks = [
            torch.stack([per_image_targets[b][m][:min_target] for b in range(batch_size)])
            for m in range(self.num_target_masks)
        ]
        return {
            "images": images,
            "labels": labels.long(),
            "context_masks": context_masks,
            "target_masks": target_masks,
        }
