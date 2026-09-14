from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from medjepa.models.blocks import build_2d_sincos_position_embedding


def gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3 or indices.ndim != 2:
        raise ValueError("Expected tokens=(B,N,D) and indices=(B,K)")
    if tokens.shape[0] != indices.shape[0]:
        raise ValueError("Token and index batch dimensions must match")
    expanded = indices.to(device=tokens.device, dtype=torch.long).unsqueeze(-1)
    expanded = expanded.expand(-1, -1, tokens.shape[-1])
    return torch.gather(tokens, dim=1, index=expanded)


class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("Expected images=(B,C,H,W)")
        patches = self.projection(images)
        return patches.flatten(2).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    """Reference for TODOs 4–5: explicit attention and an optional SDPA path."""

    def __init__(
        self, embed_dim: int, num_heads: int, dropout: float = 0.0,
        *, use_sdpa: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads:
            raise ValueError("Positive embed_dim must be divisible by positive num_heads")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must lie in [0, 1]")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.use_sdpa = use_sdpa
        self.attn_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.embed_dim:
            raise ValueError("Expected tokens=(B,N,embed_dim)")
        batch, length, _ = tokens.shape
        # (B,N,3D) -> (3,B,H,N,Dh): Q, K, V share a single linear projection.
        qkv = self.attn_proj(tokens).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        dropout = self.dropout if self.training else 0.0
        if self.use_sdpa:
            attended = F.scaled_dot_product_attention(
                query, key, value, dropout_p=dropout, is_causal=False
            )
        else:
            scores = (query @ key.transpose(-2, -1)) * self.head_dim ** -0.5
            softmax_dtype = (
                torch.float32 if scores.dtype in (torch.float16, torch.bfloat16)
                else scores.dtype
            )
            probabilities = scores.softmax(dim=-1, dtype=softmax_dtype).to(value.dtype)
            probabilities = F.dropout(probabilities, p=dropout, training=self.training)
            attended = probabilities @ value
        merged = attended.transpose(1, 2).reshape(batch, length, self.embed_dim)
        return self.out_proj(merged)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            use_sdpa=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(tokens)
        attended = self.attention(normalized)
        tokens = tokens + self.dropout1(attended)
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class TinyVisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        positions = build_2d_sincos_position_embedding(
            self.patch_embedding.grid_size, embed_dim
        )
        self.register_buffer("position_embedding", positions, persistent=True)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self._initialize_weights()

    @property
    def num_patches(self) -> int:
        return self.patch_embedding.num_patches

    def _initialize_weights(self) -> None:
        nn.init.trunc_normal_(self.patch_embedding.projection.weight, std=0.02)
        if self.patch_embedding.projection.bias is not None:
            nn.init.zeros_(self.patch_embedding.projection.bias)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_tokens(
        self,
        images: torch.Tensor,
        masks: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        tokens = self.patch_embedding(images)
        positions = self.position_embedding.to(dtype=tokens.dtype)
        tokens = tokens + positions
        if masks is not None:
            masks = [masks] if isinstance(masks, torch.Tensor) else masks
            tokens = torch.cat([gather_tokens(tokens, mask) for mask in masks], dim=0)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return self.forward_tokens(images, masks)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(images).mean(dim=1)
