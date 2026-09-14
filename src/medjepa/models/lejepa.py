from __future__ import annotations

import torch
from torch import nn

from medjepa.models.blocks import TinyVisionTransformer


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """Initialize the MLP mapping backbone features into projection space.

        Args:
            input_dim: Backbone feature width D.
            hidden_dim: Hidden-layer width used inside the projection MLP.
            output_dim: Projection width P used by the pretraining objective.

        Returns:
            None. Initialize a projection network whose forward maps (S, D) to
            (S, P). The reference uses two hidden Linear/BatchNorm/GELU stages
            and a final linear projection; BatchNorm training needs S > 1.
        """
        super().__init__()
        # TODO 19: Construct the projection MLP used only during pretraining.
        raise NotImplementedError("TODO 19: construct ProjectionHead")

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project a batch of backbone feature vectors.

        Args:
            embeddings: Floating tensor (S, input_dim); S may be B * V flattened
                image/view pairs. With training BatchNorm, S must exceed 1.

        Returns:
            Tensor (S, output_dim), preserving row order. These are projection
            features for the loss, not normalized probabilities or class logits.
        """
        # TODO 20: Project full-image embeddings.
        raise NotImplementedError("TODO 20: implement ProjectionHead.forward")


class LeJEPAModel(nn.Module):
    def __init__(self, backbone: TinyVisionTransformer, projector: ProjectionHead) -> None:
        """Register the trainable backbone and pretraining projection head.

        Args:
            backbone: TinyVisionTransformer with embedding width D.
            projector: ProjectionHead with input_dim = D and output_dim = P.

        Returns:
            None. Store the modules as backbone and projector so both sets of
            parameters are registered for optimization.
        """
        super().__init__()
        # TODO 21: Store the one trainable encoder and its projection head.
        raise NotImplementedError("TODO 21: construct LeJEPAModel")

    def forward(self, views: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode augmented views and return backbone and projection features.

        Args:
            views: Floating tensor (B, V, C, H, H): B images, V views per image,
                with C and H matching the backbone. Flattening uses image-major
                order: all V views of image 0, then all views of image 1, etc.

        Returns:
            Dictionary with "embeddings": (B * V, D) in that flattened order,
            and "projections": (V, B, P) in view-major layout. D is backbone
            embed_dim; P is projector output_dim. Both retain their gradient graphs.

        Raises:
            ValueError: views is not rank 5.
        """
        # TODO 22: Flatten views, encode once, then restore a view-major projection tensor.
        raise NotImplementedError("TODO 22: implement LeJEPAModel.forward")

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return backbone features without applying the pretraining projector.

        Args:
            images: Floating tensor (B, C, H, H) matching the backbone.

        Returns:
            Tensor (B, D), D = backbone embed_dim, pooled over all patch tokens.
            No view dimension or class predictions; gradients remain available.
        """
        # TODO 23: Return backbone embeddings without the projection head.
        raise NotImplementedError("TODO 23: implement LeJEPAModel.encode")


def build_lejepa(cfg: object) -> LeJEPAModel:
    """Build the LeJEPA backbone and projector from the resolved config.

    Args:
        cfg: Attribute-access config with data.image_size and model fields
            patch_size, embed_dim, depth, num_heads, mlp_ratio, dropout,
            projection_hidden_dim, and projection_dim. Images have 3 channels.

    Returns:
        Trainable LeJEPAModel with a matching backbone/projector. forward
        maps (B, V, 3, H, H) to embeddings (B * V, embed_dim) and projections
        (V, B, projection_dim). Device placement and DDP are caller concerns.
    """
    # TODO 24: Build the shared tiny ViT and projection head from the YAML config.
    raise NotImplementedError("TODO 24: implement build_lejepa")
