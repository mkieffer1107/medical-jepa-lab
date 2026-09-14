from __future__ import annotations

import copy
import math

import torch
from torch import nn

from medjepa.models.blocks import build_2d_sincos_position_embedding
from solutions.medjepa_solutions.blocks import (
    TinyVisionTransformer,
    TransformerBlock,
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
        """Build a predictor for num_patches patches arranged in a square grid."""
        super().__init__()
        if num_patches <= 0:
            raise ValueError("num_patches must be a positive perfect square")
        grid_size = math.isqrt(num_patches)
        if grid_size * grid_size != num_patches:
            raise ValueError("num_patches must be a positive perfect square")
        self.input_projection = nn.Linear(encoder_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        positions = build_2d_sincos_position_embedding(grid_size, predictor_dim)
        self.register_buffer("position_embedding", positions, persistent=True)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(predictor_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(predictor_dim)
        self.output_projection = nn.Linear(predictor_dim, encoder_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_masks: list[torch.Tensor],
        target_masks: list[torch.Tensor],
    ) -> torch.Tensor:
        if not context_masks or not target_masks:
            raise ValueError("At least one context and target mask are required")
        batch_size = context_masks[0].shape[0]
        expected = batch_size * len(context_masks)
        if context_tokens.shape[0] != expected:
            raise ValueError(
                f"Expected {expected} context batches, got {context_tokens.shape[0]}"
            )
        positions = self.position_embedding.expand(batch_size, -1, -1)
        predictions: list[torch.Tensor] = []
        for target_mask in target_masks:
            target_positions = gather_tokens(positions, target_mask)
            for context_number, context_mask in enumerate(context_masks):
                start = context_number * batch_size
                stop = start + batch_size
                context = self.input_projection(context_tokens[start:stop])
                context = context + gather_tokens(positions, context_mask)
                target_queries = self.mask_token.expand(
                    batch_size, target_mask.shape[1], -1
                )
                target_queries = target_queries + target_positions
                sequence = torch.cat([context, target_queries], dim=1)
                context_length = context.shape[1]
                for block in self.blocks:
                    sequence = block(sequence)
                sequence = self.norm(sequence)
                predictions.append(self.output_projection(sequence[:, context_length:]))
        return torch.cat(predictions, dim=0)


class IJEPAStudent(nn.Module):
    def __init__(self, context_encoder: TinyVisionTransformer, predictor: IJEPAPredictor):
        super().__init__()
        self.context_encoder = context_encoder
        self.predictor = predictor

    def forward(
        self,
        images: torch.Tensor,
        context_masks: list[torch.Tensor],
        target_masks: list[torch.Tensor],
    ) -> torch.Tensor:
        context_tokens = self.context_encoder.forward_tokens(images, context_masks)
        return self.predictor(context_tokens, context_masks, target_masks)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.context_encoder.encode(images)


def extract_target_tokens(
    full_target_tokens: torch.Tensor,
    context_masks: list[torch.Tensor],
    target_masks: list[torch.Tensor],
) -> torch.Tensor:
    targets: list[torch.Tensor] = []
    for target_mask in target_masks:
        selected = gather_tokens(full_target_tokens, target_mask)
        targets.extend(selected for _ in context_masks)
    return torch.cat(targets, dim=0)


def build_ijepa(cfg: object) -> tuple[IJEPAStudent, TinyVisionTransformer]:
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
    if not 0 <= momentum <= 1:
        raise ValueError("EMA momentum must lie in [0, 1]")
    for context_parameter, target_parameter in zip(
        context_encoder.parameters(), target_encoder.parameters(), strict=True
    ):
        target_parameter.mul_(momentum).add_(
            context_parameter.detach(), alpha=1.0 - momentum
        )
