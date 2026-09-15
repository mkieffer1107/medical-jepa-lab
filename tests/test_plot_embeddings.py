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
