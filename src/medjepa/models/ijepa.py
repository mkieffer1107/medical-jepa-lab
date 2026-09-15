from __future__ import annotations

import copy
import math
import torch
from torch import nn

from medjepa.models.blocks import (  
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
        if num_patches <= 0:
            raise ValueError("num_patches must be a positive perfect square")
        grid_size = math.isqrt(num_patches)
        if grid_size * grid_size != num_patches:
            raise ValueError("num_patches must be a positive perfect square")
        
        self.input_projection = nn.Linear(in_features=encoder_dim, out_features=predictor_dim)

        # single mask token that is updated by gradients accumulated from all BxKt targets
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim)) # (1, 1, P)

        positions = build_2d_sincos_position_embedding(grid_size, predictor_dim) # (1, N, P), P=predictor_dim, N=num_patches
        self.register_buffer("position_embedding", positions, persistent=True)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(predictor_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(predictor_dim)
        self.output_projection = nn.Linear(in_features=predictor_dim, out_features=encoder_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)



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

        # context_tokens are the encoder embeddings from the ViT. they are the visible tokens that were not masked.
        # context_masks are integer index tensors that give the visible / unmasked token indices in the original patchified images.
        if not context_masks or not target_masks:
            raise ValueError("At least one context and target mask are required")

        # context tokens are created by applying masks to images. so if M masks are applied
        # to B images, there should be M x B total sets of tokens. each set has multiple tokens within it
        B = context_masks[0].shape[0]
        expected = B * len(context_masks)
        if context_tokens.shape[0] != expected:
            raise ValueError(f"Expected {expected} context batches, got {context_tokens.shape[0]}")

        # pos embs have dim (1, N, P), where N=num_patches, P=predictor_dim
        # we create a view, not copied tensor, so that each set of context toks receive pos embs
        positions = self.position_embedding.expand(B, -1, -1) # (B, N, P)

        predictions: list[torch.Tensor] = [] # we'll predictions for each mask here
        for target_mask in target_masks:
            # positions: (B, N, P), all tokens across all images
            # target_mask: (B, Kt), Kt masked token indices for each of the B images to be used as targets
            # target_positions: (B, Kt, P), the actual pos embs for the target toks in each image
            target_positions = gather_tokens(positions, target_mask)
            
            # context_masks: M integer tensors (B, Kc), the Kc visible tokens that were passed through the encoder, for the B images
            # context masks are not always exact complements of the target masks. subsets of the visible tokens can be used to make more robust encoder/representations
            for context_idx, context_mask in enumerate(context_masks):
                # context_tokens has shape (M * B, Kc, D) because we flattened all visible tokens into batch dim in context encoder forward.
                # we iterate over only one mask at a time and retrieve their encoded context toks, in order to predict corresponding masked toks
                start = context_idx * B
                stop = start + B
                # (B, Kc, D) --> # (B, Kc, P)
                context = self.input_projection(context_tokens[start:stop]) # (B, Kc, P)
        
                # add pos embs to context toks, same ideas as above with target_positions
                context = context + gather_tokens(positions, context_mask) # (B, Kc, P)

                # make a view where every masked target token slot refers to the mask token
                # (1, 1, P) --> (B, Kt, P), Kt=num_targ_toks, P=predictor_dim
                targets = self.mask_token.expand(B, target_mask.shape[1], -1) # (B, Kt, P)
                targets = targets + target_positions # (B, Kt, P)

                # concat all the context and target toks for current mask. all toks will attend to each other.
                # then we'll strip off the masked toks infused with all that good tok info to predict them
                # (B, Kc, P) + (B, Kt, P) --> (B, Kc + Kt, P)
                sequence = torch.cat([context, targets], dim=1) # (B, Kc + Kt, P)
                context_length = context.shape[1]
                
                for block in self.blocks:
                    sequence = block(sequence) # (B, Kc + Kt, P)
                sequence = self.norm(sequence)

                # we only need predictions for the masked tokens in the latter portion of the sequence
                # (B, Kc + Kt, P) --> (B, Kt, P) --> (B, Kt, D)
                predictions.append(self.output_projection(sequence[:, context_length:])) # (B, Kt, D)
        # flatten the list of prediction of each mask
        # for each of the T target masks, we have M context masks over the B images, hence TxMxB
        # [(B, Kt, D), (B, Kt, D), ... TxM times] --> (TxMxB, Kt, D)
        return torch.cat(predictions, dim=0) # (TxMxB, Kt, D)




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
        self.context_encoder = context_encoder
        self.predictor = predictor

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
        context_tokens = self.context_encoder(images, context_masks)
        return self.predictor(context_tokens, context_masks, target_masks)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Expose full-image features for downstream evaluation.

        Args:
            images: Floating tensor (B, C, H, H) matching the context encoder.

        Returns:
            Tensor (B, D) of pooled context-encoder features. Do not invoke the
            predictor or produce class logits; gradients remain available.
        """
        # TODO 15: Expose a pooled full-image representation for evaluation.
        return self.context_encoder.encode(images) # (B, D)


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
    targets: list[torch.Tensor] = []
    # iterate over the T target masks
    for target_mask in target_masks:
        target_toks = gather_tokens(full_target_tokens, target_mask) # (B, Kt, D)
        # each of the T targets can be predicted using M different sets of context toke
        targets.extend(target_toks for _ in context_masks) # list of M elements
    # [(B, Kt, D), (B, Kt, D), ... TxM times] --> (TxMxB, Kt, D)
    return torch.cat(targets, dim=0)

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
    model_cfg = cfg.model
    encoder = TinyVisionTransformer(
        image_size=int(cfg.data.image_size),
        patch_size=int(model_cfg.patch_size),
        in_channels=3,
        embed_dim=int(model_cfg.embed_dim),
        depth=int(model_cfg.depth),
        num_heads=int(model_cfg.num_heads),
        mlp_ratio=float(model_cfg.mlp_ratio),
        dropout=float(model_cfg.dropout),
    )
    target_encoder = copy.deepcopy(encoder)
    for parameter in target_encoder.parameters():
        parameter.requires_grad_(False)
    predictor = IJEPAPredictor(
        num_patches=encoder.num_patches,
        encoder_dim=int(model_cfg.embed_dim),
        predictor_dim=int(model_cfg.predictor_dim),
        depth=int(model_cfg.predictor_depth),
        num_heads=int(model_cfg.predictor_heads),
        mlp_ratio=float(model_cfg.mlp_ratio),
        dropout=float(model_cfg.dropout),
    )
    return IJEPAStudent(encoder, predictor), target_encoder


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
    
    if not 0 <= momentum <= 1:
        raise ValueError("EMA momentum must lie in [0, 1]")

    for context_parameter, target_parameter in zip(
        context_encoder.parameters(), target_encoder.parameters(), strict=True
    ):  
        # in-place op, ensuring that we detach context from graph so grads don't flow
        target_parameter.mul_(momentum).add_(
            context_parameter.detach(), alpha = 1.0 - momentum
        )