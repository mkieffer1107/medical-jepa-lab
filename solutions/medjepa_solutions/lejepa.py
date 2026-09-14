from __future__ import annotations

import torch
from torch import nn

from solutions.medjepa_solutions.blocks import TinyVisionTransformer


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.network(embeddings)


class LeJEPAModel(nn.Module):
    def __init__(self, backbone: TinyVisionTransformer, projector: ProjectionHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.projector = projector

    def forward(self, views: torch.Tensor) -> dict[str, torch.Tensor]:
        if views.ndim != 5:
            raise ValueError("Expected views=(B,V,C,H,W)")
        batch_size, view_count = views.shape[:2]
        flattened = views.flatten(0, 1)
        embeddings = self.backbone.encode(flattened)
        projections = self.projector(embeddings)
        projections = projections.reshape(batch_size, view_count, -1).permute(1, 0, 2)
        return {"embeddings": embeddings, "projections": projections}

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode(images)


def build_lejepa(cfg: object) -> LeJEPAModel:
    model_cfg = cfg.model
    backbone = TinyVisionTransformer(
        image_size=int(cfg.data.image_size),
        patch_size=int(model_cfg.patch_size),
        in_channels=3,
        embed_dim=int(model_cfg.embed_dim),
        depth=int(model_cfg.depth),
        num_heads=int(model_cfg.num_heads),
        mlp_ratio=float(model_cfg.mlp_ratio),
        dropout=float(model_cfg.dropout),
    )
    projector = ProjectionHead(
        input_dim=int(model_cfg.embed_dim),
        hidden_dim=int(model_cfg.projection_hidden_dim),
        output_dim=int(model_cfg.projection_dim),
    )
    return LeJEPAModel(backbone, projector)
