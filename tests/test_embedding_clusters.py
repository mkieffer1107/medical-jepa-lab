import base64
import io

import numpy as np
from PIL import Image
from torch.utils.data import Subset

from medjepa.data.datasets import SyntheticMedicalDataset
from medjepa.embedding_clusters import cluster_embeddings, export_samples


def test_representatives_cover_center_edge_and_diversity():
    rng = np.random.default_rng(12)
    x = np.concatenate([rng.normal(0, .1, (20, 5)), rng.normal(10, .1, (20, 5))])
    labels = np.arange(40) % 2
    assigned, report = cluster_embeddings(x, labels, ['A', 'B'], 2, 6, 7)
    assert len(np.unique(assigned)) == 2
    assert len(set(assigned[:20])) == 1 and assigned[0] != assigned[-1]
    for cluster in report['clusters']:
        members = cluster['members']
        examples = cluster['examples']
        assert sum(c['count'] for c in cluster['class_counts']) == len(members)
        assert len(set(e['index'] for e in examples)) == 6
        assert all(assigned[e['index']] == cluster['id'] for e in examples)
        distances = np.linalg.norm(x[members] - x[members].mean(axis=0), axis=1)
        assert examples[0]['index'] == members[int(distances.argmin())]
        assert examples[1]['index'] == members[int(distances.argmax())]


def test_sample_images_keep_subset_source_row_and_pixels():
    dataset = SyntheticMedicalDataset('test', 8, 2, 32, lambda x: x, 4)
    subset = Subset(dataset, [5, 2, 7])
    report = export_samples(subset, np.array([1, 0, 1]), 'synthetic', 32)
    for sample, row in zip(report['samples'], [5, 2, 7]):
        actual = Image.open(io.BytesIO(base64.b64decode(sample['png'])))
        assert sample['row'] == row
        assert np.array_equal(np.asarray(actual), np.asarray(dataset._render(row, row % 2)))
    assert report['archive_url'] is None


def test_medmnist_wrapper_exports_archive_pixels_without_transform():
    from types import SimpleNamespace

    from medjepa.data.datasets import DatasetInfo, LabelSqueezingDataset
    from medjepa.embedding_clusters import source_image

    pixels = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
    raw = SimpleNamespace(imgs=pixels, labels=np.array([[1], [0]]))
    wrapped = LabelSqueezingDataset(raw, DatasetInfo('bloodmnist', ['A', 'B'], 2))
    image, label, row = source_image(Subset(wrapped, [1]), 0)
    assert row == 1 and label == 0
    assert np.array_equal(np.asarray(image), pixels[1])
