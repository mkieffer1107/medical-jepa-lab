"""Plot held-out encoder embeddings using PCA fitted on training embeddings."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from medjepa.config import _as_config, apply_override, validate_config
from medjepa.data import build_evaluation_bundle
from medjepa.evaluate import _load_encoder, _plot_projection, extract_embeddings, resolve_device
from medjepa.training.checkpoint import load_checkpoint
from medjepa.utils import ensure_dir, write_json

LOGGER = logging.getLogger(__name__)


def is_finished(checkpoint: dict) -> bool:
    if "training_complete" in checkpoint:
        return bool(checkpoint["training_complete"])
    epochs = checkpoint.get("config", {}).get("optimization", {}).get("epochs")
    return epochs is not None and int(checkpoint.get("epoch", -1)) >= int(epochs)


def select_checkpoint(root: Path, explicit: str | None) -> tuple[Path, dict]:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path, load_checkpoint(path, torch.device("cpu"))
    candidates = sorted(root.glob("**/latest.pt"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for path in candidates:
        LOGGER.info("Checking checkpoint: %s", path)
        try:
            checkpoint = load_checkpoint(path, torch.device("cpu"))
        except (OSError, RuntimeError, EOFError) as exc:
            LOGGER.warning("Cannot load %s: %s", path, exc)
            continue
        if is_finished(checkpoint):
            return path.resolve(), checkpoint
        LOGGER.info("Skipping unfinished run: %s", path.parent.name)
    raise FileNotFoundError(
        f"No finished training checkpoint found under {root}. "
        "Use --checkpoint outputs/<run>/latest.pt to select one explicitly "
        "(including older runs stopped with --max-steps)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="Explicit checkpoint; otherwise newest finished run")
    parser.add_argument("--checkpoint-dir", default="outputs")
    parser.add_argument("--output-dir")
    parser.add_argument("--ijepa-encoder", choices=["target", "student"], default="target")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--override", action="append", default=[], help="E.g. data.num_workers=0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    for option in ("batch_size", "max_train", "max_test"):
        value = getattr(args, option)
        if value is not None and value <= 0:
            parser.error(f"--{option.replace('_', '-')} must be positive")

    path, checkpoint = select_checkpoint(Path(args.checkpoint_dir), args.checkpoint)
    LOGGER.info("Selected checkpoint: %s | epoch=%s step=%s", path, checkpoint.get("epoch"), checkpoint.get("global_step"))
    cfg = _as_config(checkpoint["config"])
    for override in args.override:
        apply_override(cfg, override)
    validate_config(cfg)
    algorithm = checkpoint.get("algorithm", cfg.algorithm)
    implementation = checkpoint.get("implementation", "solution")
    device = resolve_device(args.device)
    LOGGER.info("Loading %s encoder (%s) on %s", algorithm, args.ijepa_encoder if algorithm == "ijepa" else "backbone", device)
    encoder = _load_encoder(cfg, checkpoint, algorithm, implementation, args.ijepa_encoder, device)
    del checkpoint
    data = build_evaluation_bundle(cfg, batch_size=args.batch_size)
    train_x, _ = extract_embeddings(encoder, data.train_loader, device, args.max_train, "Train embeddings (PCA fit)")
    test_x, test_y = extract_embeddings(encoder, data.test_loader, device, args.max_test, "Test embeddings (plot)")
    if min(train_x.shape) < 2:
        raise ValueError("PCA needs at least two training samples and two embedding dimensions")
    LOGGER.info("Fitting PCA on %d training embeddings; projecting %d test embeddings", len(train_x), len(test_x))
    pca = PCA(n_components=2, random_state=int(cfg.experiment.seed))
    pca.fit(train_x)
    coordinates = pca.transform(test_x)
    output = ensure_dir(Path(args.output_dir or Path("reports") / path.parent.name / "embeddings")).resolve()
    image = output / "pca_test.png"
    explained = float(pca.explained_variance_ratio_.sum())
    _plot_projection(coordinates, test_y, data.info.class_names,
                     f"{algorithm.upper()} | {data.info.name} test embeddings\nPCA — {explained:.1%} variance explained", image)
    np.savez_compressed(output / "test_embeddings.npz", embeddings=test_x, labels=test_y,
                        coordinates=coordinates, class_names=np.asarray(data.info.class_names))
    write_json(output / "projection.json", {
        "checkpoint": str(path), "algorithm": algorithm, "implementation": implementation,
        "encoder": args.ijepa_encoder if algorithm == "ijepa" else "backbone",
        "pca_fit_split": "train", "plot_split": "test", "train_samples": len(train_x),
        "test_samples": len(test_x), "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "image": str(image),
    })
    LOGGER.info("Saved embeddings: %s", output / "test_embeddings.npz")
    print(f"Plot saved to: {image}", flush=True)


if __name__ == "__main__":
    main()
