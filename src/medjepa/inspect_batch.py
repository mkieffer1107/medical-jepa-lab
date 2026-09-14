from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from medjepa.config import load_config
from medjepa.data import build_data_bundle
from medjepa.data.transforms import denormalize


@dataclass(frozen=True)
class LocalEnvironment:
    distributed: bool = False
    rank: int = 0
    world_size: int = 1


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Visualize JEPA training inputs")
    result.add_argument("--config", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--examples", type=int, default=4)
    result.add_argument("--override", action="append", default=[])
    return result


def _show_image(axis, image: torch.Tensor, title: str) -> None:
    axis.imshow(image.permute(1, 2, 0).cpu().numpy())
    axis.set_title(title)
    axis.axis("off")


def _ijepa_figure(cfg: object, batch: dict[str, object], examples: int):
    images = denormalize(batch["images"][:examples])
    context_masks: list[torch.Tensor] = batch["context_masks"]
    target_masks: list[torch.Tensor] = batch["target_masks"]
    labels = batch["labels"]
    patch_size = int(cfg.model.patch_size)
    grid = int(cfg.data.image_size) // patch_size
    figure, axes = plt.subplots(examples, 2, figsize=(10, 4.5 * examples), squeeze=False)
    for row in range(examples):
        _show_image(axes[row, 0], images[row], f"source — label {int(labels[row])}")
        _show_image(axes[row, 1], images[row], "context (blue) and targets (outlined)")
        context_indices = set(context_masks[0][row].tolist())
        for index in context_indices:
            y, x = divmod(index, grid)
            axes[row, 1].add_patch(
                Rectangle(
                    (x * patch_size, y * patch_size),
                    patch_size,
                    patch_size,
                    facecolor="tab:blue",
                    alpha=0.20,
                    linewidth=0,
                )
            )
        for mask_number, mask in enumerate(target_masks):
            for index in mask[row].tolist():
                y, x = divmod(index, grid)
                axes[row, 1].add_patch(
                    Rectangle(
                        (x * patch_size, y * patch_size),
                        patch_size,
                        patch_size,
                        fill=False,
                        edgecolor=f"C{(mask_number + 1) % 10}",
                        linewidth=1.5,
                    )
                )
    figure.suptitle(
        f"I-JEPA batch: {len(context_masks)} context mask(s), {len(target_masks)} target mask(s)"
    )
    figure.tight_layout()
    return figure


def _lejepa_figure(batch: tuple[torch.Tensor, torch.Tensor], examples: int):
    views, labels = batch
    examples = min(examples, views.shape[0])
    view_count = views.shape[1]
    flat = views[:examples].reshape(-1, *views.shape[2:])
    flat = denormalize(flat).reshape(examples, view_count, *views.shape[2:])
    figure, axes = plt.subplots(
        examples, view_count, figsize=(3.2 * view_count, 3.2 * examples), squeeze=False
    )
    for row in range(examples):
        for column in range(view_count):
            _show_image(
                axes[row, column],
                flat[row, column],
                f"label {int(labels[row])} — view {column + 1}",
            )
    figure.suptitle("LeJEPA independently augmented views")
    figure.tight_layout()
    return figure


def main() -> None:
    args = parser().parse_args()
    cfg = load_config(args.config, args.override)
    data = build_data_bundle(cfg, str(cfg.algorithm), LocalEnvironment())
    batch = next(iter(data.train_loader))
    examples = min(args.examples, int(cfg.data.batch_size))
    if cfg.algorithm == "ijepa":
        figure = _ijepa_figure(cfg, batch, examples)
    else:
        figure = _lejepa_figure(batch, examples)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
