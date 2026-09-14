from __future__ import annotations

import torch
from torch import nn


def invariance_loss(projections: torch.Tensor) -> torch.Tensor:
    """Measure disagreement between augmented views of the same image.

    Args:
        projections: Floating tensor (V, B, D), with views first. Entries
            [:, b, :] must correspond to augmentations of the same image.

    Returns:
        Differentiable scalar tensor (shape ()): mean squared deviation from
        each image's mean across views, averaged over views, images, and
        features. Identical projections across views give zero loss.

    Raises:
        ValueError: projections is not rank 3.
    """
    # TODO 25: Penalize each view's deviation from its image-level view mean.
    raise NotImplementedError("TODO 25: implement invariance_loss")


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization exercise."""

    def __init__(
        self,
        num_slices: int,
        t_max: float,
        num_points: int,
        seed: int,
    ) -> None:
        """Initialize the random-projection Gaussian regularizer.

        Args:
            num_slices: Positive number of random 1D projection directions.
            t_max: Positive upper endpoint of the integration interval [0, t_max].
            num_points: Number Q of evenly spaced integration points; at least 2.
            seed: Integer base seed for reproducible directions shared across ranks.

        Returns:
            None. Store num_slices and seed; register float32 buffers t, normal_cf,
            and weights, each (Q,), and scalar integer global_step initialized to 0.
            They must move with the module and survive state_dict checkpointing.

        Raises:
            ValueError: num_points is less than 2.
        """
        super().__init__()
        # TODO 26: Precompute the characteristic-function integration grid/weights.
        raise NotImplementedError("TODO 26: construct SIGReg")

    def forward(self, projections: torch.Tensor) -> torch.Tensor:
        """Measure projected distribution mismatch with an isotropic Gaussian.

        Args:
            projections: Floating tensor (..., S, D), typically (V, B, D).
                The penultimate axis contains samples; the last contains features.
                Leading groups (such as views) are regularized separately. Under
                DDP, ranks must have equal local S and matching other dimensions.

        Returns:
            Differentiable scalar float32 tensor (shape ()) on the input device,
            averaging the sample-count-scaled integrated discrepancy across random
            slices and leading groups. Compute the empirical characteristic function
            across global samples under DDP with gradients through communication.
            A finite Gaussian sample need not have exactly zero penalty.

        Side effects:
            Advance the checkpointed global_step for the next projection draw;
            use the same directions on every rank for each call.

        Raises:
            ValueError: projections has fewer than two dimensions.
        """
        # TODO 27: Compare random 1D projected distributions with N(0, 1).
        # In DDP, the empirical characteristic-function mean must use all ranks.
        raise NotImplementedError("TODO 27: implement SIGReg.forward")


class LeJEPAObjective(nn.Module):
    def __init__(self, sigreg: SIGReg, regularization_weight: float) -> None:
        """Initialize the weighted LeJEPA pretraining objective.

        Args:
            sigreg: SIGReg module mapping projection features to a scalar penalty.
            regularization_weight: Scalar w in [0, 1], trading invariance against
                SIGReg. w=0 uses only invariance; w=1 uses only SIGReg in total loss.

        Returns:
            None. Store sigreg and regularization_weight for forward/checkpointing.

        Raises:
            ValueError: regularization_weight is outside [0, 1].
        """
        super().__init__()
        # TODO 28: Store the regularizer and validate the tradeoff weight.
        raise NotImplementedError("TODO 28: construct LeJEPAObjective")

    def forward(self, projections: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the combined loss and its unweighted components.

        Args:
            projections: Floating tensor (V, B, D), views first, with aligned
                image identities across views.

        Returns:
            Dictionary of differentiable scalar tensors (shape ()) with keys:
            "loss": (1-w) * invariance + w * sigreg, where w is the stored weight;
            "invariance": the unweighted view-invariance loss;
            "sigreg": the unweighted Gaussian regularization penalty.
            Do not detach these tensors or convert them to Python floats here.
        """
        # TODO 29: Return total, invariance, and SIGReg losses.
        raise NotImplementedError("TODO 29: implement LeJEPAObjective.forward")
