from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Protocol

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from torch import nn
from tqdm import tqdm

from medjepa.config import load_config
from medjepa.data import build_evaluation_bundle
from medjepa.training.checkpoint import load_checkpoint
from medjepa.utils import ensure_dir, write_json


LOGGER = logging.getLogger(__name__)


class Encoder(Protocol):
    def encode(self, images: torch.Tensor) -> torch.Tensor: ...


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Evaluate frozen JEPA representations")
    result.add_argument("--config", required=True)
    result.add_argument("--checkpoint", default=None)
    result.add_argument(
        "--random-init", action="store_true", help="Evaluate an untrained baseline"
    )
    result.add_argument("--algorithm", choices=["ijepa", "lejepa"], default=None)
    result.add_argument(
        "--implementation", choices=["exercise", "solution"], default=None
    )
    result.add_argument(
        "--ijepa-encoder", choices=["target", "student"], default="target"
    )
    result.add_argument("--reducer", choices=["pca", "umap", "both"], default="both")
    result.add_argument("--output-dir", default=None)
    result.add_argument("--batch-size", type=int, default=None)
    result.add_argument("--max-train", type=int, default=None)
    result.add_argument("--max-val", type=int, default=None)
    result.add_argument("--max-test", type=int, default=None)
    result.add_argument("--silhouette-samples", type=int, default=3000)
    result.add_argument(
        "--umap-fit-samples",
        type=int,
        default=20_000,
        help="Maximum number of training embeddings used to fit UMAP; <=0 uses all",
    )
    result.add_argument("--device", default="auto")
    result.add_argument("--override", action="append", default=[])
    return result


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_encoder(
    cfg: object,
    checkpoint: dict | None,
    algorithm: str,
    implementation: str,
    ijepa_encoder: str,
    device: torch.device,
) -> nn.Module:
    if implementation == "solution":
        if algorithm == "ijepa":
            from solutions.medjepa_solutions.ijepa import build_ijepa

            student, target = build_ijepa(cfg)
        else:
            from solutions.medjepa_solutions.lejepa import build_lejepa

            model = build_lejepa(cfg)
    else:
        if algorithm == "ijepa":
            from medjepa.models.ijepa import build_ijepa

            student, target = build_ijepa(cfg)
        else:
            from medjepa.models.lejepa import build_lejepa

            model = build_lejepa(cfg)

    if algorithm == "ijepa":
        if checkpoint is not None:
            student.load_state_dict(checkpoint["student"])
            target.load_state_dict(checkpoint["target_encoder"])
        encoder: nn.Module = target if ijepa_encoder == "target" else student
    else:
        if checkpoint is not None:
            model.load_state_dict(checkpoint["model"])
        encoder = model
    encoder.to(device)
    encoder.eval()
    return encoder


@torch.inference_mode()
def extract_embeddings(
    encoder: Encoder,
    loader: object,
    device: torch.device,
    maximum: int | None,
    description: str,
) -> tuple[np.ndarray, np.ndarray]:
    feature_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    seen = 0
    for images, labels in tqdm(loader, desc=description, leave=False):
        if maximum is not None and seen >= maximum:
            break
        images = images.to(device, non_blocking=True)
        features = encoder.encode(images).detach().float().cpu().numpy()
        labels_array = labels.detach().cpu().numpy().reshape(-1)
        if maximum is not None and seen + len(features) > maximum:
            keep = maximum - seen
            features = features[:keep]
            labels_array = labels_array[:keep]
        feature_batches.append(features)
        label_batches.append(labels_array)
        seen += len(features)
    if not feature_batches:
        raise RuntimeError(f"No embeddings extracted for {description}")
    return np.concatenate(feature_batches), np.concatenate(label_batches).astype(int)


def _plot_projection(
    coordinates: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    title: str,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 8))
    scatter = axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=labels,
        s=12,
        alpha=0.72,
    )
    axis.set_title(title)
    axis.set_xlabel("component 1")
    axis.set_ylabel("component 2")
    handles, _ = scatter.legend_elements(num=len(class_names))
    if len(handles) == len(class_names):
        axis.legend(
            handles,
            class_names,
            title="class",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, float]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="lbfgs"),
    )
    model.fit(train_x, train_y)
    val_prediction = model.predict(val_x)
    test_prediction = model.predict(test_x)
    return {
        "linear_probe/val_accuracy": float(accuracy_score(val_y, val_prediction)),
        "linear_probe/val_balanced_accuracy": float(
            balanced_accuracy_score(val_y, val_prediction)
        ),
        "linear_probe/test_accuracy": float(accuracy_score(test_y, test_prediction)),
        "linear_probe/test_balanced_accuracy": float(
            balanced_accuracy_score(test_y, test_prediction)
        ),
    }


def _knn_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, float]:
    neighbors = min(5, len(train_x))
    model = make_pipeline(
        Normalizer(),
        KNeighborsClassifier(n_neighbors=neighbors, weights="distance"),
    )
    model.fit(train_x, train_y)
    val_prediction = model.predict(val_x)
    test_prediction = model.predict(test_x)
    return {
        "knn/val_accuracy": float(accuracy_score(val_y, val_prediction)),
        "knn/val_balanced_accuracy": float(
            balanced_accuracy_score(val_y, val_prediction)
        ),
        "knn/test_accuracy": float(accuracy_score(test_y, test_prediction)),
        "knn/test_balanced_accuracy": float(
            balanced_accuracy_score(test_y, test_prediction)
        ),
    }


def _silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray,
    maximum: int,
    seed: int,
) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    rng = np.random.default_rng(seed)
    if len(embeddings) > maximum:
        indices = rng.choice(len(embeddings), size=maximum, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
    normalized = Normalizer().fit_transform(embeddings)
    return float(silhouette_score(normalized, labels, metric="euclidean"))


def _umap_fit_subset(
    train_embeddings: np.ndarray,
    maximum: int,
    seed: int,
) -> np.ndarray:
    """Choose a deterministic subset for fitting a transductive visualizer."""
    if maximum <= 0 or maximum >= len(train_embeddings):
        return train_embeddings
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(train_embeddings), size=maximum, replace=False)
    return train_embeddings[indices]


def _print_metrics(metrics: dict[str, object]) -> None:
    LOGGER.info("Frozen representation results")
    for key, value in metrics.items():
        if isinstance(value, float):
            LOGGER.info("  %-42s %.5f", key, value)


def main() -> None:
    args = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = load_config(args.config, args.override)
    device = resolve_device(args.device)
    if args.random_init and args.checkpoint:
        raise ValueError("Choose either --random-init or --checkpoint, not both")
    if not args.random_init and not args.checkpoint:
        raise ValueError("Provide --checkpoint, or use --random-init for a baseline")
    checkpoint = None if args.random_init else load_checkpoint(args.checkpoint, device)
    algorithm = args.algorithm or str(
        checkpoint.get("algorithm", cfg.algorithm) if checkpoint is not None else cfg.algorithm
    )
    implementation = args.implementation or str(
        checkpoint.get("implementation", "solution")
        if checkpoint is not None
        else "solution"
    )
    if algorithm != cfg.algorithm:
        raise ValueError("Algorithm argument/checkpoint does not match the YAML config")

    encoder = _load_encoder(
        cfg,
        checkpoint,
        algorithm,
        implementation,
        args.ijepa_encoder,
        device,
    )
    data = build_evaluation_bundle(cfg, batch_size=args.batch_size)
    train_x, train_y = extract_embeddings(
        encoder, data.train_loader, device, args.max_train, "train embeddings"
    )
    val_x, val_y = extract_embeddings(
        encoder, data.val_loader, device, args.max_val, "validation embeddings"
    )
    test_x, test_y = extract_embeddings(
        encoder, data.test_loader, device, args.max_test, "test embeddings"
    )

    run_name = str(cfg.experiment.run_name)
    output_dir = Path(args.output_dir or Path("reports") / run_name)
    output_dir = ensure_dir(output_dir)
    np.savez_compressed(
        output_dir / "embeddings.npz",
        train_embeddings=train_x,
        train_labels=train_y,
        val_embeddings=val_x,
        val_labels=val_y,
        test_embeddings=test_x,
        test_labels=test_y,
    )

    metrics: dict[str, object] = {
        "algorithm": algorithm,
        "implementation": implementation,
        "checkpoint": None if checkpoint is None else str(Path(args.checkpoint).resolve()),
        "random_initialization": checkpoint is None,
        "checkpoint_epoch": -1 if checkpoint is None else int(checkpoint.get("epoch", -1)),
        "checkpoint_step": -1 if checkpoint is None else int(checkpoint.get("global_step", -1)),
        "samples/train": int(len(train_x)),
        "samples/val": int(len(val_x)),
        "samples/test": int(len(test_x)),
        "embedding_dimension": int(train_x.shape[1]),
    }
    metrics.update(_linear_probe(train_x, train_y, val_x, val_y, test_x, test_y))
    metrics.update(_knn_probe(train_x, train_y, val_x, val_y, test_x, test_y))
    silhouette = _silhouette(
        test_x, test_y, int(args.silhouette_samples), int(cfg.experiment.seed)
    )
    if silhouette is not None:
        metrics["clustering/test_silhouette"] = silhouette

    if args.reducer in {"pca", "both"}:
        pca = PCA(n_components=2, random_state=int(cfg.experiment.seed))
        pca.fit(train_x)
        test_coordinates = pca.transform(test_x)
        explained = float(pca.explained_variance_ratio_.sum())
        metrics["pca/explained_variance_2d"] = explained
        np.save(output_dir / "pca_test_coordinates.npy", test_coordinates)
        _plot_projection(
            test_coordinates,
            test_y,
            data.info.class_names,
            f"{algorithm.upper()} — {data.info.name} test embeddings — PCA",
            output_dir / "pca_test.png",
        )

    if args.reducer in {"umap", "both"}:
        try:
            import umap
        except ImportError as exc:
            raise RuntimeError("Install `umap-learn` or choose --reducer pca") from exc
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=20,
            min_dist=0.10,
            metric="cosine",
            random_state=int(cfg.experiment.seed),
        )
        umap_fit_x = _umap_fit_subset(
            train_x,
            int(args.umap_fit_samples),
            int(cfg.experiment.seed),
        )
        metrics["umap/fit_samples"] = int(len(umap_fit_x))
        reducer.fit(umap_fit_x)
        test_coordinates = reducer.transform(test_x)
        np.save(output_dir / "umap_test_coordinates.npy", test_coordinates)
        _plot_projection(
            test_coordinates,
            test_y,
            data.info.class_names,
            f"{algorithm.upper()} — {data.info.name} test embeddings — UMAP",
            output_dir / "umap_test.png",
        )

    write_json(output_dir / "metrics.json", metrics)
    _print_metrics(metrics)
    LOGGER.info("Wrote report to %s", output_dir.resolve())


if __name__ == "__main__":
    main()
