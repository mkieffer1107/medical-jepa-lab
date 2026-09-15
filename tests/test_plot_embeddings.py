import os

import pytest
import torch

from medjepa.plot_embeddings import is_finished, select_checkpoint


def test_select_newest_finished_run_and_explicit_override(tmp_path):
    for name, completed, modified in [("old", True, 10), ("finished", True, 20), ("active", False, 30)]:
        folder = tmp_path / name
        folder.mkdir()
        path = folder / "latest.pt"
        torch.save({"training_complete": completed}, path)
        os.utime(path, (modified, modified))
    selected, _ = select_checkpoint(tmp_path, None)
    assert selected.parent.name == "finished"
    selected, _ = select_checkpoint(tmp_path, str(tmp_path / "active" / "latest.pt"))
    assert selected.parent.name == "active"


def test_legacy_completion_and_empty_directory(tmp_path):
    checkpoint = {"config": {"optimization": {"epochs": 25}}, "epoch": 24}
    assert not is_finished(checkpoint)
    checkpoint["epoch"] = 25
    assert is_finished(checkpoint)
    checkpoint["training_complete"] = False
    assert not is_finished(checkpoint)
    with pytest.raises(FileNotFoundError, match="No finished"):
        select_checkpoint(tmp_path, None)


def test_tsne_and_offline_html(tmp_path):
    import json
    import re

    import numpy as np

    from medjepa.embedding_report import write_interactive_report
    from medjepa.plot_embeddings import tsne_projections

    features = np.random.default_rng(4).normal(size=(12, 8))
    xy, xyz, perplexity = tsne_projections(features, 4)
    assert xy.shape == (12, 2)
    assert xyz.shape == (12, 3)
    assert np.isfinite(xy).all() and np.isfinite(xyz).all()
    assert 0 < perplexity < 12
    output = tmp_path / 'report.html'
    write_interactive_report(xyz, xy, xyz, np.arange(12) % 2,
                             ['A', '</script><script>bad()</script>'],
                             np.array([.4, .2, .1]), 'run <1>', '/checkpoint', output)
    page = output.read_text()
    assert '<script src=' not in page
    assert 'run &lt;1&gt;' in page
    assert '</script><script>bad()' not in page
    payload = json.loads(re.search(r'id="embedding-data">(.*?)</script>', page, re.S)[1])
    assert len(payload['groups']) == 2
    assert len(payload['groups'][0]['tsne2'][0]) == 2
    assert len(payload['groups'][0]['tsne3'][0]) == 3
