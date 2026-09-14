# Reference solutions

This folder contains complete implementations matching the public contracts in `TASKS.md`.

Run them without copying anything into the exercise package:

```bash
uv run python -m solutions.train_ijepa --config configs/smoke_ijepa_synthetic.yaml --max-steps 8
uv run python -m solutions.train_lejepa --config configs/smoke_lejepa_synthetic.yaml --max-steps 8
```

Suggested use:

1. Write a TODO yourself.
2. Run the relevant test.
3. Inspect only the corresponding function here if the test fails and the tensor contract is still unclear.
4. Close the file and reimplement it rather than copying line-for-line.

The reference code prioritizes clarity over maximal kernel efficiency.

`medjepa_solutions/blocks.py` includes `MultiHeadSelfAttention` for TODOs 4–5.
Its default path shows the explicit attention math; `use_sdpa=True` uses PyTorch's
optimized attention operation with the same learned projections. Reference
transformer blocks use that optimized path. Run `uv run pytest tests/test_attention.py`
to compare both paths with PyTorch's multihead attention, including gradients.

Attention projection checkpoint keys are now `attention.attn_proj.weight` and
`attention.attn_proj.bias`, replacing `attention.in_proj_weight` and
`attention.in_proj_bias`. Older model checkpoints need those keys renamed before
loading; the tensor shapes and Q/K/V ordering are unchanged.
