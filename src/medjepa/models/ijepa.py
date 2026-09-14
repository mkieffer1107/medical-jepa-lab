from __future__ import annotations

import torch
from torch import nn

from medjepa.models.blocks import (  # noqa: F401
    TinyVisionTransformer,
    TransformerBlock,
    build_2d_sincos_position_embedding,
    gather_tokens,
)


class IJEPAPredictor(nn.Module):
    def __init__(
        self,
        num_patches: int,
        encoder_dim: int,
        predictor_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        """Initialize a predictor from visible tokens to missing-region features.

        Args:
            num_patches: Full-image patch count N, a perfect square. The position
                grid side is sqrt(N); this argument is NOT the grid side length.
            encoder_dim: Input and output token width D.
            predictor_dim: Internal width P, divisible by 4 and num_heads.
            depth: Number of predictor transformer blocks.
            num_heads: Attention heads per predictor block.
            mlp_ratio: Predictor MLP hidden-width multiplier.
            dropout: Dropout probability in [0, 1].

        Returns:
            None. Initialize projections between D and P, a trainable mask token
            (1, 1, P), fixed positions (1, N, P), transformer blocks, and norm.
            forward returns target predictions in encoder space (last dimension D).

        Raises:
            ValueError: num_patches is not a positive perfect square.
        """
        super().__init__()
        # TODO 11: Construct the latent predictor, target mask token, and positions.
        raise NotImplementedError("TODO 11: construct IJEPAPredictor")

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_masks: list[torch.Tensor],
        target_masks: list[torch.Tensor],
    ) -> torch.Tensor:
        """Predict target tokens for every target-mask/context-mask pair.

        Args:
            context_tokens: Floating tensor (M * B, Kc, D), grouped by context
                mask first, then image, as returned by forward_tokens.
            context_masks: Nonempty list of M integer tensors (B, Kc).
            target_masks: Nonempty list of T integer tensors (B, Kt).
                All masks contain full-image patch indices in [0, num_patches).
                Context lengths must agree with each other, as must target lengths.

        Returns:
            Tensor (T * M * B, Kt, D) of target predictions only, not context
            tokens. Rows are ordered target mask, then context mask, then image:
            row ((t * M + m) * B + b) belongs to target t, context m, image b.
            The Kt axis follows each target mask's index order. D = encoder_dim.

        Raises:
            ValueError: Either mask list is empty or context_tokens has a batch
                dimension different from M * B.
        """
        # TODO 12: Assemble context and target-position tokens and predict targets.
        raise NotImplementedError("TODO 12: implement IJEPAPredictor.forward")


class IJEPAStudent(nn.Module):
    def __init__(self, context_encoder: TinyVisionTransformer, predictor: IJEPAPredictor):
        """Register the two trainable parts of the I-JEPA student.

        Args:
            context_encoder: TinyVisionTransformer producing tokens of width D.
            predictor: IJEPAPredictor with encoder_dim = D and matching patch grid.

        Returns:
            None. Store the supplied modules as context_encoder and predictor so
            their parameters are registered. The target encoder lives separately.
        """
        super().__init__()
        # TODO 13: Store the trainable context encoder and predictor.
        raise NotImplementedError("TODO 13: construct IJEPAStudent")

    def forward(
        self,
        images: torch.Tensor,
        context_masks: list[torch.Tensor],
        target_masks: list[torch.Tensor],
    ) -> torch.Tensor:
        """Predict missing-region representations from visible image patches.

        Args:
            images: Floating tensor (B, C, H, H) matching the context encoder.
            context_masks: Nonempty list of M integer index tensors (B, Kc).
            target_masks: Nonempty list of T integer index tensors (B, Kt).
                Indices address the full image patch grid; lengths are shared
                within each list.

        Returns:
            Tensor (T * M * B, Kt, D), D = context encoder width. Ordering is
            target mask, then context mask, then image, matching IJEPAPredictor
            and extract_target_tokens. Gradients flow through both student modules.
        """
        # TODO 14: Encode visible contexts and invoke the predictor.
        raise NotImplementedError("TODO 14: implement IJEPAStudent.forward")

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Expose full-image features for downstream evaluation.

        Args:
            images: Floating tensor (B, C, H, H) matching the context encoder.

        Returns:
            Tensor (B, D) of pooled context-encoder features. Do not invoke the
            predictor or produce class logits; gradients remain available.
        """
        # TODO 15: Expose a pooled full-image representation for evaluation.
        raise NotImplementedError("TODO 15: implement IJEPAStudent.encode")


def extract_target_tokens(
    full_target_tokens: torch.Tensor,
    context_masks: list[torch.Tensor],
    target_masks: list[torch.Tensor],
) -> torch.Tensor:
    """Select target features in exactly the predictor's output order.

    Args:
        full_target_tokens: Floating tensor (B, N, D) from the full-image
            target encoder, after any normalization required by the caller.
        context_masks: Nonempty list of M integer tensors (B, Kc); the list
            determines how many context predictions each target must match.
        target_masks: Nonempty list of T integer tensors (B, Kt), with shared
            Kt and patch indices in [0, N).

    Returns:
        Tensor (T * M * B, Kt, D), preserving input dtype/device. Row
        ((t * M + m) * B + b) contains full_target_tokens[b] selected by
        target_masks[t][b], repeated for each context m. Preserve index order.
        Detaching teacher features is the training caller's responsibility.
    """
    # TODO 16: Select target regions in the same ordering returned by the predictor.
    raise NotImplementedError("TODO 16: implement extract_target_tokens")


def build_ijepa(cfg: object) -> tuple[IJEPAStudent, TinyVisionTransformer]:
    """Construct matching student and frozen teacher models from config.

    Args:
        cfg: Resolved attribute-access config with data.image_size and model
            fields patch_size, embed_dim, depth, num_heads, mlp_ratio, dropout,
            predictor_dim, predictor_depth, and predictor_heads. Images have
            3 channels. Pass encoder.num_patches to the predictor, matching the
            reference implementation's constructor.

    Returns:
        (student, target_encoder): IJEPAStudent and TinyVisionTransformer.
        The target is an independent copy of the context encoder with equal
        initial weights and requires_grad=False for all parameters. The student
        remains trainable. Both encode (B, 3, H, H) into (B, embed_dim).
        Device placement and DDP wrapping are handled by the caller.
    """
    # TODO 17: Build the student and an initially identical frozen target encoder.
    raise NotImplementedError("TODO 17: implement build_ijepa")


@torch.no_grad()
def update_target_encoder(
    context_encoder: nn.Module,
    target_encoder: nn.Module,
    momentum: float,
) -> None:
    """Update teacher parameters by an exponential moving average in place.

    Args:
        context_encoder: Student encoder (unwrap DDP first if needed).
        target_encoder: Separate teacher with matching parameter order/shapes.
        momentum: Scalar m in [0, 1]. At m=0 copy the student; at m=1 keep
            the teacher unchanged.

    Returns:
        None. Each teacher parameter becomes m * old_target + (1-m) * student.
        Do not modify student parameters or build an autograd graph. This
        operation updates parameters, not buffers such as fixed positions.

    Raises:
        ValueError: momentum is outside [0, 1].
    """
    # TODO 18: Apply an in-place EMA parameter update to the target encoder.
    raise NotImplementedError("TODO 18: implement update_target_encoder")
