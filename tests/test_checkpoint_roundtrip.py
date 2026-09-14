from pathlib import Path

import torch

from medjepa.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    payload = {"epoch": 3, "tensor": torch.arange(5)}
    save_checkpoint(path, payload)
    restored = load_checkpoint(path, torch.device("cpu"))
    assert restored["epoch"] == 3
    assert torch.equal(restored["tensor"], payload["tensor"])
