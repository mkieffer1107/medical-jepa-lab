from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def build_2d_sincos_position_embedding(grid_size: int, embed_dim: int) -> torch.Tensor:
    """Return a fixed `(1, grid_size**2, embed_dim)` 2D sinusoidal embedding.

    This utility is supplied because implementing Fourier position features is not the
    JEPA-specific part of the exercise.
    """
    if embed_dim % 4:
        raise ValueError("embed_dim must be divisible by 4 for 2D sin/cos positions")
    y, x = torch.meshgrid(
        torch.arange(grid_size, dtype=torch.float32),
        torch.arange(grid_size, dtype=torch.float32),
        indexing="ij",
    )
    quarter = embed_dim // 4
    frequencies = 1.0 / (10_000 ** (torch.arange(quarter, dtype=torch.float32) / quarter))
    x_phase = x.reshape(-1, 1) * frequencies.reshape(1, -1)
    y_phase = y.reshape(-1, 1) * frequencies.reshape(1, -1)
    embedding = torch.cat(
        [x_phase.sin(), x_phase.cos(), y_phase.sin(), y_phase.cos()], dim=-1
    )
    return embedding.unsqueeze(0)


def gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Select patch tokens independently for each image, preserving index order.

    Args:
        tokens: Floating tensor (B, N, D): B images, N patch tokens per image,
            and D features per token.
        indices: Integer tensor (B, K) of patch indices in [0, N). Each row
            selects K tokens from the corresponding image; duplicates are allowed.
            Convert to torch.long on tokens.device for indexing.

    Returns:
        Tensor (B, K, D), with the same dtype/device as tokens.
        output[b, k, :] equals tokens[b, indices[b, k], :]. Gradients must
        flow back to the selected tokens; the input tensors are not modified.

    Example:
        tokens shaped (2, 4, 3) and indices [[2, 0], [1, 3]] produce (2, 2, 3).
        Image 0 contributes tokens 2 then 0; image 1 contributes tokens 1 then 3.

    Raises:
        ValueError: tokens is not rank 3, indices is not rank 2, or their
            batch sizes differ.
    """
    # TODO 1: Implement batched token selection without a Python loop over images.
    if tokens.ndim != 3 or indices.ndim != 2:
        raise ValueError("tokens is not rank 3, indices is not rank 2")
    if tokens.shape[0] != indices.shape[0]:
        raise ValueError(f"batch sizes differ: {tokens.shape[0]=} != {indices.shape[0]}")
    
    # with for loop:
    # img_toks = []
    # for idx, toks in enumerate(indices):
    #     img_toks.append(tokens[idx][toks])
    # return torch.stack(img_toks) # .to(dtype=tokens.dtype) # redundant, since made from tokens tensor
    
    # the idx in the indices tensor tells what img to pull tokens from. the actual content at that idx in indices tells which tokens to pull.
    # so the idea here is to basically make a new tensor, batch_idxs, to enumerate the idxs in indices. then pass that in to grab those
    # images, and then pass indices in to grab the actual tokens from thhe retreived image in the batch.  
    # indices = indices.to(device=tokens.device, dtype=torch.long)
    # B = tokens.shape[0]
    # batch_idxs = torch.arange(B).unsqueeze(-1).to(tokens.device) # reshape (B) -> (B, 1) by adding extra dim to end to make col vec for indexing
    # return tokens[batch_idxs, indices] # grab the idx (batch_idxs) and then pass in the token idxs to select (indices)

    # more efficient in part because expand doesn't copy data, just changes teh view
    expanded = indices.to(device=tokens.device, dtype=torch.long).unsqueeze(dim=-1) # (B, K) --> (B, K, 1)
    expanded = expanded.expand(-1, -1, tokens.shape[-1]) # (B, K, 1) --> (B, K, D) broadcast: (B,K) is stacked D times, so dim=2 is a dummy dim
    
    # from docs: out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
    # this is exactly what we want. input[i] gets the ith image, input[i][index[i][j][k]] gets the corresponding tokens for the 
    # ith image, and then the final [k] index is just a dummy index
    return torch.gather(tokens, dim=1, index=expanded) # (B, K, D)


class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        """Initialize a projection from non-overlapping image patches to tokens.

        Args:
            image_size: Side length H of the square input image, in pixels.
            patch_size: Patch side P in pixels; H must be divisible by P.
            in_channels: Number C of input image channels (3 in this lab).
            embed_dim: Number D of output features per patch.

        Returns:
            None. Initialize the module so forward maps (B, C, H, H) to
            (B, N, D), where grid_size = H // P and N = grid_size ** 2.
            Expose image_size, patch_size, grid_size, num_patches, and projection.

        Raises:
            ValueError: image_size is not divisible by patch_size.
        """
        super().__init__()

        # TODO 2: Create the image-to-patch projection and record the patch count.
        if image_size % patch_size:
            raise ValueError(f"{image_size=} is not divisible by {patch_size=}")

        self.image_size = image_size
        self.patch_size = patch_size

        # split image into grid_size x grid_size patches
        self.grid_size = image_size // patch_size 
        self.num_patches = self.grid_size ** 2

        # kernel_size==stride because we want to jump from patch to patch / have non-overlapping patches
        self.projection = nn.Conv2d(
            in_channels=in_channels, # the number of inputs for each pixel, 3 for rgb
            out_channels=embed_dim, # num outputs per patch is just the embedding dimension
            kernel_size=patch_size, 
            stride=patch_size
        )


    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Convert an image batch into a sequence of learned patch embeddings.

        Args:
            images: Floating tensor (B, C, H, H), with C and H matching the
                constructor's in_channels and image_size.

        Returns:
            Tensor (B, N, D), where N = (H // patch_size) ** 2 and D = embed_dim.
            Patch order is row-major: left to right, then top to bottom.
            No CLS token or positional embedding is added here.

        Raises:
            ValueError: images is not rank 4.
        """

        # TODO 3: Return a patch sequence with shape `(batch, patches, embed_dim)`.
        if images.ndim != 4:
            raise ValueError(f"{images.ndim=} is not rank 4")
        
        # project (B, C, H, H) to (B, D, grid_size, grid_size), an embedding for each patch in the grid.
        # we want to flatten into (B, N, D), where we enumerate the patches into a list, rather than keeping
        # their spatial layout. and we know N = grid_size x grid_size

        # could also supply end_dim, but this does the trick
        out = self.projection(images).flatten(start_dim=2) # (B, D, N)
        return out.transpose(1, 2) # (B, D, N) --> (B, N, D)


class MultiHeadSelfAttention(nn.Module):
    """Non-causal self-attention"""

    def __init__(
        self, embed_dim: int, num_heads: int, dropout: float = 0.0,
        *, use_sdpa: bool = False,
    ) -> None:
        """Initialize projections and head metadata.

        Args:
            embed_dim: Positive token width D, divisible by num_heads.
            num_heads: Positive head count H; each head has width Dh = D // H.
            dropout: Attention-probability dropout in [0, 1].
            use_sdpa: False selects your explicit attention calculation;
                True selects F.scaled_dot_product_attention with your own
                projections and head reshaping.

        Returns:
            None. Expose embed_dim, num_heads, head_dim, dropout, use_sdpa,
            attn_proj (nn.Linear(D, 3*D), with bias, ordered Q then K then V),
            and out_proj (nn.Linear(D, D), with bias).

        Raises:
            ValueError: Dimensions are nonpositive, D is not divisible by H,
                or dropout is outside [0, 1].
        """
        super().__init__()
        # TODO 4: Construct multihead self-attention projections and validate dimensions.
        if embed_dim < 1 or num_heads < 1:
            raise ValueError(f"{embed_dim=} and {num_heads=} must be greater than 0")
        if embed_dim % num_heads:
            raise ValueError(f"{embed_dim=} is not divisible by {num_heads=}")
        if dropout < 0 or dropout > 1:
            raise ValueError(f"{dropout=} must be within [0,1]")

        self.embed_dim = embed_dim # embed_dim = D
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.use_sdpa = use_sdpa

        # wq,wk,wv projection mats have shape (D,D), where D=embed_dim, we combine all 3 into a single mat. project input
        self.attn_proj = nn.Linear(in_features=embed_dim, out_features = 3*embed_dim, bias=True)

        self.out_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix information between all tokens independently within each image.

        Args:
            x: Floating tensor (B, N, D), D = embed_dim. Noncontiguous
                inputs are allowed. There is no causal or padding mask.

        Returns:
            Tensor (B, N, D), preserving token order and gradients through
            inputs and all projections. Return features only, not weights.

        Intermediate contracts:
            Project to (B, N, 3*D), then split Q/K/V into (B, H, N, Dh).
            Scores Q @ K.transpose(-2, -1) / sqrt(Dh) have (B, H, N, N).
            Softmax acts over the final (key-token) axis. Apply dropout to
            these probabilities only in training, then multiply by V.
            Merge heads back to (B, N, D) and apply out_proj. Do not add
            residuals or layer normalization; TransformerBlock handles them.

            For use_sdpa=True, replace only the score/softmax/value operation
            with F.scaled_dot_product_attention(..., is_causal=False).
            Pass dropout_p=0.0 in eval mode; SDPA does not do this for you.
            For the manual path, compute softmax in float32 for float16 or
            bfloat16 scores, then cast probabilities back to the value dtype.

        Raises:
            ValueError: x is not rank 3 or its last dimension is not D.
        """
        # TODO 5: Implement Q/K/V splitting, scaled attention, head merging, and output projection.

        if x.ndim != 3:
            raise ValueError(f"{x.ndim=} must be equal to 3")
        if x.shape[-1] != self.embed_dim:
            raise ValueError(f"{x.shape[-1]} does not have the correct embedding dimension of {self.embed_dim}")

        dropout = self.dropout if self.training else 0.0

        B, N, D = x.shape # D == embed_dim

        # attn_proj has in_features=D, out_features=3D for shape (3D,D). x.shape = (B, N, D)
        # attn_proj(x) --> x @ attn_proj.T --> (B, N, D) @ (D, 3D) --> (B, N, 3D)
        # but then we want to split along the 3D dim to create q,k,v mats. and we know that
        # num_heads = D / head_dim. so we first split into 3 parts for (B, N, 3, D) and then split D into num_heads and head_dim
        qkv = self.attn_proj(x).reshape(
            B, N, 3, self.num_heads, self.head_dim # (B, N, 3, num_heads, head_dim)
        ).permute(2, 0, 3, 1, 4) # (3, B, num_heads, N, head_dim)
        q,k,v = qkv.unbind(dim=0) # (B, num_heads, N, head_dim)

        if self.use_sdpa:
            # (B, num_heads, N, head_dim)
            context_vec = F.scaled_dot_product_attention( 
                q, k, v, dropout_p=dropout, is_causal=False
            )
        else:
            # (B, num_heads, N, head_dim) @ (B, num_heads, N, head_dim).T(-2, -1)
            # --> (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, N).T
            # --> (B, num_heads, N, N), the attn scores of all pairwise tokens
            attn_scores = q @ k.transpose(-2, -1) # (B, num_heads, N, N)

            softmax_dtype = (
                torch.float32 if attn_scores.dtype in (torch.float16, torch.bfloat16)
                else attn_scores.dtype
            )

            # normalize along the rows of the (N, N) attn mat for each head. perform attn op in float32, then cast to values dtype
            attn_weights = torch.softmax(attn_scores / k.shape[-1]**0.5, dim=-1, dtype=softmax_dtype).to(v.dtype) # (B, num_heads, N, N)
            attn_weights = F.dropout(attn_weights, p=dropout, training=self.training)

            # (B, num_heads, N, N) @ (B, num_heads, N, head_dim) 
            context_vec = attn_weights @ v # (B, num_heads, N, head_dim)
        
        # (B, num_heads, N, head_dim)
        #  transpose --> (B, N, num_heads, head_dim)
        #  reshape  -->  (B, N, D)
        # context_vec = context_vec.permute(0, 2, 1, 3).reshape(B, N, -1)
        context_vec = context_vec.transpose(1, 2).reshape(B, N, D) # (B, N, D)

        # (B, N, D) @ (D, D)
        return self.out_proj(context_vec) # (B, N, D)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        """Initialize a transformer block that preserves token shape.

        Args:
            embed_dim: Token width D, divisible by num_heads.
            num_heads: Number of self-attention heads.
            mlp_ratio: Hidden MLP width multiplier; hidden width is int(D * mlp_ratio).
            dropout: Dropout probability in [0, 1].

        Returns:
            None. Initialize pre-attention and pre-MLP normalization, batch-first
            MultiHeadSelfAttention from TODOs 4-5, an MLP, and dropout.
            forward accepts and returns (B, N, D). Use use_sdpa=True for
            the training model after checking the manual implementation.
        """
        super().__init__()
        # TODO 6: Construct a pre-normalization block using MultiHeadSelfAttention (TODOs 4-5)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.hidden_dim = int(embed_dim * mlp_ratio)
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout, use_sdpa=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Update tokens with attention and MLP residual branches.

        Args:
            x: Floating tensor (B, N, D), with D = embed_dim.

        Returns:
            Tensor (B, N, D): updated features with unchanged batch size, token
            count, and feature width. Call self.attention(normalized_tokens):
            our attention takes one tensor and returns one tensor, not a tuple.
        """
        # TODO 7: Apply attention and MLP residual branches.
        x = x + self.dropout1(self.attention(self.norm1(x)))
        return x + self.mlp(self.norm2(x))
        


class TinyVisionTransformer(nn.Module):
    """A no-CLS-token ViT shared by the two exercises."""

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
        """Initialize the shared vision transformer without a CLS token.

        Args:
            image_size: Square image side H in pixels.
            patch_size: Patch side P dividing H.
            in_channels: Image channel count C.
            embed_dim: Token width D; divisible by 4 for fixed 2D positions
                and by num_heads for attention.
            depth: Number of transformer blocks.
            num_heads: Attention heads per block.
            mlp_ratio: MLP hidden-width multiplier.
            dropout: Dropout probability in [0, 1].

        Returns:
            None. Initialize patch_embedding, fixed position_embedding (1, N, D),
            blocks, and final norm, where N = (H // P) ** 2. Expose embed_dim
            and num_patches for downstream model construction. Positions should
            be a buffer that moves with the module, not a trainable parameter.
        """
        super().__init__()
        # TODO 8: Construct patch embedding, fixed positions, blocks, and final norm.
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
                # set to 1 and 0 so that LN initially does pure z-score normalization.
                # once weights update it will scale norm by weights and add on bias ofc
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_tokens(
        self,
        images: torch.Tensor,
        masks: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Encode all patches or only the patches selected by each mask.

        Args:
            images: Floating tensor (B, C, H, H), matching the configured image size.
            masks: None, one integer index tensor (B, K), or a nonempty list of M
                such tensors. List entries must share K. Values are patch indices
                in [0, N), not boolean masks, where N is the full patch count.

        Returns:
            Normalized tokens of width D = embed_dim. Shape is (B, N, D) for
            None, (B, K, D) for one mask, or (M * B, K, D) for a mask list.
            For a list, rows are mask-major: all B images for mask 0, then all B
            for mask 1, etc. Within each row, preserve the mask's index order.
            Selected tokens retain their original full-image positional features.
        """
        # TODO 9: Embed patches, add positions, optionally select masks, then encode.
        # patchify + tokeenize images: (B, C, H, H) -> (B, N, D)
        x = self.patch_embedding(images) # (B, N, D)
        positions = self.position_embedding.to(dtype=x.dtype)
        x = x + positions

        if masks is not None:
            masks = [masks] if isinstance(masks, torch.Tensor) else masks
            
            # select all visible tokens, then concat along the batch dim
            # to give shape (MxB, K, D), where K are the unmasked / visible tokens and M is the num of masks
            x = torch.cat([gather_tokens(x, mask) for mask in masks], dim=0)

        # attention only sees visible tokens
        for block in self.blocks:
            x = block(x)
        return self.norm(x) # (B, N, D) no mask / (MxB, K, D) mask  (K=visible toks, M=num masks)
        

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # (B, C, H, H) --> (B, N, D) no mask / (MxB, K, D) mask  (K=visible toks, M=num masks)
        return self.forward_tokens(images, masks)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return one backbone representation per image using every patch.

        Args:
            images: Floating tensor (B, C, H, H), matching the constructor.

        Returns:
            Tensor (B, D), D = embed_dim, obtained by averaging the full-image
            encoded patch tokens. This is a feature vector, not class logits.
        """
        # TODO 10: Pool full-image patch tokens into one representation per image.
        # (B, C, H, H) --> (B, N, D) --> (B, D) -- no masks provided here, as we encode entire image, so no (MxB,) first dim
        return self.forward_tokens(images).mean(dim=1)

