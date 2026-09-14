import inspect

import pytest
import torch
import torch.nn.functional as F

from medjepa.config import load_config
from medjepa.data.masks import MultiBlockMaskCollator
from medjepa.models.ijepa import IJEPAPredictor as ExercisePredictor
from solutions.medjepa_solutions.ijepa import (
    IJEPAPredictor,
    build_ijepa,
    extract_target_tokens,
)
from solutions.medjepa_solutions.lejepa import build_lejepa


def _mask_batch(cfg, images):
    collator = MultiBlockMaskCollator(
        image_size=int(cfg.data.image_size),
        patch_size=int(cfg.model.patch_size),
        context_scale=tuple(cfg.mask.context_scale),
        target_scale=tuple(cfg.mask.target_scale),
        target_aspect_ratio=tuple(cfg.mask.target_aspect_ratio),
        num_context_masks=int(cfg.mask.num_context_masks),
        num_target_masks=int(cfg.mask.num_target_masks),
        min_keep=int(cfg.mask.min_keep),
        allow_overlap=bool(cfg.mask.allow_overlap),
    )
    batch = [(image, 0) for image in images]
    return collator(batch)


def test_predictor_constructor_contract_and_patch_positions() -> None:
    assert inspect.signature(IJEPAPredictor) == inspect.signature(ExercisePredictor)
    predictor = IJEPAPredictor(
        num_patches=16, encoder_dim=32, predictor_dim=16,
        depth=1, num_heads=4, mlp_ratio=2.0, dropout=0.0,
    )
    assert predictor.position_embedding.shape == (1, 16, 16)
    predictions = predictor(
        torch.randn(2, 3, 32),
        [torch.tensor([[0, 1, 2], [3, 4, 5]])],
        [torch.tensor([[14, 15], [0, 15]])],
    )
    assert predictions.shape == (2, 2, 32)
    predictions.square().mean().backward()
    assert predictor.mask_token.grad is not None


@pytest.mark.parametrize("num_patches", [0, -4, 15])
def test_predictor_rejects_invalid_patch_counts(num_patches: int) -> None:
    with pytest.raises(ValueError, match="positive perfect square"):
        IJEPAPredictor(
            num_patches=num_patches, encoder_dim=32, predictor_dim=16,
            depth=1, num_heads=4, mlp_ratio=2.0, dropout=0.0,
        )


def test_reference_ijepa_shapes_and_backward() -> None:
    cfg = load_config("configs/smoke_ijepa_synthetic.yaml")
    student, target = build_ijepa(cfg)
    images = torch.randn(3, 3, cfg.data.image_size, cfg.data.image_size)
    batch = _mask_batch(cfg, images)
    with torch.no_grad():
        target_tokens = target.forward_tokens(batch["images"])
        targets = extract_target_tokens(
            target_tokens, batch["context_masks"], batch["target_masks"]
        )
    predictions = student(
        batch["images"], batch["context_masks"], batch["target_masks"]
    )
    assert predictions.shape == targets.shape
    loss = F.smooth_l1_loss(predictions, targets)
    loss.backward()
    assert any(parameter.grad is not None for parameter in student.parameters())
    assert all(parameter.grad is None for parameter in target.parameters())
    assert student.encode(images).shape == (3, cfg.model.embed_dim)


def test_reference_lejepa_shapes_and_backward() -> None:
    cfg = load_config("configs/smoke_lejepa_synthetic.yaml")
    model = build_lejepa(cfg)
    views = torch.randn(
        4, cfg.data.views.count, 3, cfg.data.image_size, cfg.data.image_size
    )
    output = model(views)
    assert output["projections"].shape == (
        cfg.data.views.count,
        4,
        cfg.model.projection_dim,
    )
    output["projections"].square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert model.encode(views[:, 0]).shape == (4, cfg.model.embed_dim)
