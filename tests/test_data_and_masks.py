from dataclasses import dataclass

import torch

from medjepa.config import load_config
from medjepa.data import build_data_bundle


@dataclass(frozen=True)
class LocalEnvironment:
    distributed: bool = False
    rank: int = 0
    world_size: int = 1


def test_ijepa_collator_contract_and_nonoverlap() -> None:
    cfg = load_config("configs/smoke_ijepa_synthetic.yaml")
    data = build_data_bundle(cfg, "ijepa", LocalEnvironment())
    batch = next(iter(data.train_loader))
    images = batch["images"]
    contexts = batch["context_masks"]
    targets = batch["target_masks"]
    assert images.shape == (cfg.data.batch_size, 3, cfg.data.image_size, cfg.data.image_size)
    assert len(contexts) == cfg.mask.num_context_masks
    assert len(targets) == cfg.mask.num_target_masks
    for context in contexts:
        assert context.ndim == 2
        assert context.shape[0] == images.shape[0]
        assert context.shape[1] >= cfg.mask.min_keep
    for target in targets:
        assert target.ndim == 2
        assert target.shape[0] == images.shape[0]
    for batch_index in range(images.shape[0]):
        target_union = set()
        for target in targets:
            target_union.update(target[batch_index].tolist())
        for context in contexts:
            assert set(context[batch_index].tolist()).isdisjoint(target_union)


def test_lejepa_multiview_contract() -> None:
    cfg = load_config("configs/smoke_lejepa_synthetic.yaml")
    data = build_data_bundle(cfg, "lejepa", LocalEnvironment())
    views, labels = next(iter(data.train_loader))
    assert views.shape == (
        cfg.data.batch_size,
        cfg.data.views.count,
        3,
        cfg.data.image_size,
        cfg.data.image_size,
    )
    assert labels.shape == (cfg.data.batch_size,)
    assert not torch.equal(views[:, 0], views[:, 1])
