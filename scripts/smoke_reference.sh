#!/usr/bin/env bash
set -euo pipefail

uv run python -m solutions.train_ijepa \
  --config configs/smoke_ijepa_synthetic.yaml --max-steps 8
uv run python -m solutions.train_lejepa \
  --config configs/smoke_lejepa_synthetic.yaml --max-steps 8
uv run pytest
