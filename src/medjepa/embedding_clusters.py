"""Cluster full encoder features and preserve exact dataset row/image identities."""
from __future__ import annotations

import base64
import io
import logging

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from torch.utils.data import Subset
from tqdm import tqdm

from medjepa.data.datasets import LabelSqueezingDataset, SyntheticMedicalDataset

LOGGER = logging.getLogger(__name__)


def cluster_embeddings(features, labels, class_names, count=9, examples=6, seed=42):
    if count < 1 or examples < 2:
        raise ValueError("Cluster count must be positive and examples must be at least 2")
    count = min(count, len(features))
    LOGGER.info("Fitting K-means: k=%d on %d full encoder embeddings (Euclidean, no scaling)", count, len(features))
    model = KMeans(n_clusters=count, n_init=10, random_state=seed).fit(features)
    assignments = model.labels_.astype(int)
    summaries = []
    for cluster in sorted(np.unique(assignments)):
        members = np.flatnonzero(assignments == cluster)
        values = features[members]
        distances = np.linalg.norm(values - model.cluster_centers_[cluster], axis=1)
        order = np.argsort(distances, kind="stable")
        chosen = [int(order[0])]
        roles = ["Nearest to centroid"]
        if len(members) > 1:
            chosen.append(int(order[-1]))
            roles.append("Farthest from centroid")
        # Greedy farthest-first traversal covers different parts of a cluster,
        # rather than showing only redundant neighbors of its center.
        while len(chosen) < min(examples, len(members)):
            nearest = np.min(np.stack([np.sum((values - values[i]) ** 2, axis=1) for i in chosen]), axis=0)
            nearest[chosen] = -np.inf
            chosen.append(int(np.argmax(nearest)))
            roles.append("Diverse member")
        summaries.append({
            "id": int(cluster), "size": len(members), "members": members.tolist(),
            "class_counts": [{"name": name, "count": int(np.sum(labels[members] == i))}
                             for i, name in enumerate(class_names) if np.any(labels[members] == i)],
            "examples": [{"index": int(members[i]), "role": role,
                          "distance": float(distances[i])} for i, role in zip(chosen, roles)],
        })
    return assignments, {"requested_k": count, "actual_k": len(summaries), "seed": seed,
                         "space": "full encoder embeddings", "metric": "Euclidean",
                         "scaling": "none", "n_init": 10, "clusters": summaries}


def source_image(dataset, index):
    """Return the source pixels and label before image preprocessing."""
    while isinstance(dataset, Subset):
        index = int(dataset.indices[index])
        dataset = dataset.dataset
    if isinstance(dataset, LabelSqueezingDataset):
        raw = dataset.dataset
        image = Image.fromarray(raw.imgs[index]).convert("RGB")
        label = int(np.asarray(raw.labels[index]).reshape(-1)[0])
        return image, label, index
    if isinstance(dataset, SyntheticMedicalDataset):
        label = index % dataset.classes
        return dataset._render(index, label), label, index
    raise TypeError(f"Source image export is not supported for {type(dataset).__name__}")


def export_samples(dataset, labels, dataset_name, image_size):
    """The evaluation loader is sequential; each embedding matches dataset[index]."""
    archive = None
    if dataset_name != "synthetic":
        import medmnist
        metadata = medmnist.INFO[dataset_name]
        archive = metadata.get("url" if image_size == 28 else f"url_{image_size}")
    samples = []
    for position, expected_label in enumerate(tqdm(labels, desc="Embedding source images", leave=False)):
        image, label, row = source_image(dataset, position)
        if label != int(expected_label):
            raise ValueError(f"Sample/embedding label mismatch at position {position}")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        samples.append({"id": f"{dataset_name}-{image_size}-test-{row:06d}",
                        "row": row, "split": "test", "label": label,
                        "png": base64.b64encode(buffer.getvalue()).decode("ascii")})
    return {"dataset": dataset_name, "image_size": image_size, "archive_url": archive,
            "row_definition": "zero-based index in test_images / test_labels of the MedMNIST archive",
            "samples": samples}
