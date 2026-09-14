import torch

from solutions.medjepa_solutions.sigreg import LeJEPAObjective, SIGReg, invariance_loss


def test_invariance_is_zero_for_identical_views() -> None:
    base = torch.randn(8, 16)
    projections = torch.stack([base, base, base], dim=0)
    assert torch.allclose(invariance_loss(projections), torch.zeros(()), atol=1e-7)


def test_sigreg_is_finite_and_differentiable() -> None:
    regularizer = SIGReg(num_slices=32, t_max=3.0, num_points=17, seed=1)
    projections = torch.randn(3, 64, 16, requires_grad=True)
    loss = regularizer(projections)
    assert torch.isfinite(loss)
    loss.backward()
    assert projections.grad is not None
    assert torch.isfinite(projections.grad).all()


def test_collapsed_features_are_penalized_more_than_gaussian_features() -> None:
    collapsed_regularizer = SIGReg(num_slices=64, t_max=3.0, num_points=17, seed=9)
    random_regularizer = SIGReg(num_slices=64, t_max=3.0, num_points=17, seed=9)
    collapsed = torch.zeros(2, 512, 16)
    gaussian = torch.randn(2, 512, 16)
    collapsed_loss = collapsed_regularizer(collapsed)
    gaussian_loss = random_regularizer(gaussian)
    assert collapsed_loss > gaussian_loss


def test_objective_returns_all_terms() -> None:
    objective = LeJEPAObjective(
        SIGReg(num_slices=16, t_max=3.0, num_points=17, seed=2),
        regularization_weight=0.05,
    )
    losses = objective(torch.randn(3, 32, 8))
    assert set(losses) == {"loss", "invariance", "sigreg"}
