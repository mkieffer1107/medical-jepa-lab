# Exercise checklist

The numbered TODOs in the code are the assignment. A suggested order follows.

## Part 0 — verify the scaffold

- Run the two synthetic reference smoke tests.
- Run `uv run pytest`.
- Inspect an I-JEPA mask batch and a LeJEPA multi-view batch.

## Part 1 — shared vision transformer

Implement `src/medjepa/models/blocks.py`:

- `gather_tokens`
- `PatchEmbedding.forward`
- `MultiHeadSelfAttention.__init__` and `.forward` (TODOs 4–5)
- `TransformerBlock.forward`
- `TinyVisionTransformer.forward_tokens`
- `TinyVisionTransformer.encode`

Contract tests are in `tests/test_exercise_contracts.py`. They skip unfinished functions; run them after each TODO.

### Multihead self-attention — before the transformer block

Implement TODOs 4–5 without `nn.MultiheadAttention`: learn the Q/K/V
projections, split into heads, scale dot products by `sqrt(head_dim)`, apply
softmax over keys and training-only attention dropout, combine values, merge
heads, and project back to the input width. Input and output are `(B, N, D)`;
each head uses `(B, num_heads, N, D // num_heads)`. Attention is non-causal.
JEPA context masks select tokens before attention; no causal mask belongs here.

First implement the explicit calculation (`use_sdpa=False`), then add
`use_sdpa=True` using `torch.nn.functional.scaled_dot_product_attention` for
the central attention operation. Keep your own projections and head reshaping
in both paths. SDPA needs an explicit `dropout_p=0.0` in evaluation mode.
Use the optimized path inside `TransformerBlock` (TODOs 6–7), whose attention
call now takes one tensor and returns one tensor.

Run `uv run pytest tests/test_attention.py`. It compares outputs, input gradients,
and projection gradients against PyTorch with identical weights and dropout off,
and checks dropout behavior and invalid inputs. Exercise cases skip until their
constructor/forward is implemented; reference cases always run.

## Part 2 — I-JEPA

Implement `src/medjepa/models/ijepa.py`:

- predictor construction
- context/target positional-token assembly
- prediction forward pass
- target-token extraction
- model factory

Then implement `train_one_epoch` in `src/medjepa/train_ijepa.py`. Your loop must include the target no-gradient path, student prediction path, latent loss, backward/optimizer step, and EMA target update.

Expected public contracts:

- full-image encoder tokens: `(batch, patches, encoder_dim)`
- context encoder output: `(batch × context_masks, visible_patches, encoder_dim)`
- predictor and target tensors: same three-dimensional shape
- `encode(images)`: `(batch, encoder_dim)`

## Part 3 — LeJEPA

Implement `src/medjepa/models/lejepa.py` and `src/medjepa/losses/sigreg.py`.

Then implement `train_one_epoch` in `src/medjepa/train_lejepa.py`. Your loop must include all views in one encoder call, the invariance term, SIGReg, backward/optimizer step, and logged collapse diagnostics.

Expected public contracts:

- input views: `(batch, views, channels, height, width)`
- projected representations: `(views, batch, projection_dim)`
- `encode(images)`: `(batch, encoder_dim)`

## Part 4 — distributed training

No second training loop is required. Confirm that your same implementation works with:

```bash
uv run torchrun --standalone --nproc-per-node=2 -m medjepa.train_ijepa \
  --config configs/smoke_ijepa_synthetic.yaml --max-steps 8
```

Check that:

- only rank 0 writes checkpoints and Trackio logs;
- each rank sees a different shard;
- `sampler.set_epoch(epoch)` is called;
- local losses are reduced only for reporting;
- LeJEPA SIGReg computes distribution statistics across the global distributed batch.

## Part 5 — representation report

Train both methods on the same unlabeled training split and run `medjepa-eval`.

Compare:

- frozen linear-probe and 5-NN accuracy;
- balanced accuracy, because medical datasets can be imbalanced;
- silhouette score;
- PCA/UMAP plots;
- feature standard deviation and covariance diagnostics during training;
- wall-clock throughput and peak GPU memory.

Do not tune against the test labels. Use validation results for model choices and generate the test visualization once at the end.
