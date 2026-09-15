"""Export 2D/3D PCA and t-SNE encoder plots and an offline interactive HTML report."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from medjepa.config import _as_config, apply_override, validate_config
from medjepa.data import build_evaluation_bundle
from medjepa.embedding_report import plot_3d, write_interactive_report
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


def tsne_projections(embeddings: np.ndarray, seed: int):
    if len(embeddings) < 4:
        raise ValueError("2D/3D t-SNE needs at least four test samples")
    dimensions = min(50, embeddings.shape[1], len(embeddings) - 1)
    reduced = PCA(n_components=dimensions, random_state=seed).fit_transform(embeddings)
    perplexity = min(30.0, (len(embeddings) - 1) / 3)
    results = []
    for dimension in (2, 3):
        LOGGER.info("Fitting %dD t-SNE on %d test embeddings (perplexity=%.2f)", dimension, len(embeddings), perplexity)
        results.append(TSNE(n_components=dimension, perplexity=perplexity,
                            init="pca", learning_rate="auto", random_state=seed,
                            verbose=1).fit_transform(reduced))
    return *results, perplexity


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
    if min(train_x.shape) < 3:
        raise ValueError("PCA needs at least three training samples and three embedding dimensions")
    LOGGER.info("Fitting PCA on %d training embeddings; projecting %d test embeddings", len(train_x), len(test_x))
    pca = PCA(n_components=3, random_state=int(cfg.experiment.seed))
    pca.fit(train_x)
    coordinates = pca.transform(test_x)
    output = ensure_dir(Path(args.output_dir or Path("reports") / path.parent.name / "embeddings")).resolve()
    image = output / "pca_test.png"
    explained = float(pca.explained_variance_ratio_[:2].sum())
    _plot_projection(coordinates, test_y, data.info.class_names,
                     f"{algorithm.upper()} | {data.info.name} test embeddings\nPCA — {explained:.1%} variance explained", image)
    image3d = output / "pca_test_3d.png"
    plot_3d(coordinates, test_y, data.info.class_names,
            f"{algorithm.upper()} | test embeddings | 3D PCA", image3d)
    tsne2, tsne3, perplexity = tsne_projections(test_x, int(cfg.experiment.seed))
    tsne_image = output / "tsne_test.png"
    tsne_image3 = output / "tsne_test_3d.png"
    _plot_projection(tsne2, test_y, data.info.class_names,
                     f"{algorithm.upper()} | test embeddings | t-SNE", tsne_image)
    plot_3d(tsne3, test_y, data.info.class_names,
            f"{algorithm.upper()} | test embeddings | 3D t-SNE", tsne_image3, prefix="t-SNE")
    report = output / "embeddings.html"
    write_interactive_report(coordinates, tsne2, tsne3, test_y, data.info.class_names,
                             pca.explained_variance_ratio_, path.parent.name, str(path), report)
    np.savez_compressed(output / "test_embeddings.npz", embeddings=test_x, labels=test_y,
                        coordinates=coordinates[:, :2], pca_3d=coordinates,
                        tsne_2d=tsne2, tsne_3d=tsne3, class_names=np.asarray(data.info.class_names))
    write_json(output / "projection.json", {
        "checkpoint": str(path), "algorithm": algorithm, "implementation": implementation,
        "encoder": args.ijepa_encoder if algorithm == "ijepa" else "backbone",
        "pca_fit_split": "train", "plot_split": "test", "train_samples": len(train_x),
        "test_samples": len(test_x), "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "image": str(image), "image_3d": str(image3d), "html": str(report),
        "tsne": {"fit_split": "test", "perplexity": perplexity, "seed": int(cfg.experiment.seed),
                 "preprocessing": "centered PCA to at most 50 dimensions",
                 "image": str(tsne_image), "image_3d": str(tsne_image3)},
    })
    LOGGER.info("Saved embeddings: %s", output / "test_embeddings.npz")
    for artifact in (image, image3d, tsne_image, tsne_image3, report):
        print(f"Saved: {artifact}", flush=True)


if __name__ == "__main__":
    main()
