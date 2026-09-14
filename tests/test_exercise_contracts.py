"""These tests become active as the exercise TODOs are implemented."""

import pytest
import torch

from medjepa.config import load_config


def test_exercise_ijepa_public_contract() -> None:
    from medjepa.models.ijepa import build_ijepa

    cfg = load_config("configs/smoke_ijepa_synthetic.yaml")
    try:
        student, target = build_ijepa(cfg)
    except NotImplementedError as exc:
        pytest.skip(str(exc))
    images = torch.randn(2, 3, cfg.data.image_size, cfg.data.image_size)
    assert student.encode(images).shape == (2, cfg.model.embed_dim)
    assert target.encode(images).shape == (2, cfg.model.embed_dim)


def test_exercise_lejepa_public_contract() -> None:
    from medjepa.models.lejepa import build_lejepa

    cfg = load_config("configs/smoke_lejepa_synthetic.yaml")
    try:
        model = build_lejepa(cfg)
    except NotImplementedError as exc:
        pytest.skip(str(exc))
    views = torch.randn(
        2, cfg.data.views.count, 3, cfg.data.image_size, cfg.data.image_size
    )
    output = model(views)
    assert output["projections"].shape == (
        cfg.data.views.count,
        2,
        cfg.model.projection_dim,
    )
