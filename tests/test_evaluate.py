import numpy as np

from medjepa.evaluate import _umap_fit_subset


def test_umap_fit_subset_is_capped_and_deterministic() -> None:
    embeddings = np.arange(200, dtype=np.float32).reshape(100, 2)
    first = _umap_fit_subset(embeddings, maximum=17, seed=9)
    second = _umap_fit_subset(embeddings, maximum=17, seed=9)
    assert first.shape == (17, 2)
    assert np.array_equal(first, second)


def test_umap_fit_subset_nonpositive_limit_uses_all() -> None:
    embeddings = np.arange(20, dtype=np.float32).reshape(10, 2)
    assert _umap_fit_subset(embeddings, maximum=0, seed=1) is embeddings
