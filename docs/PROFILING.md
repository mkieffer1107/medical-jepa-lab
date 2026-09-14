# Metrics versus profiling

The project includes two complementary tools.

## Trackio / JSONL: experiment-level metrics

The training loops log:

- total loss and objective components;
- learning rate;
- I-JEPA EMA momentum;
- gradient norm;
- end-to-end image and encoded-view throughput;
- slowest-rank optimizer-step time, input-wait time, and input-wait fraction;
- peak allocated GPU memory;
- representation standard deviation, norm, and off-diagonal covariance diagnostics;
- mask sizes for I-JEPA.

Trackio receives rank-zero logs. `metrics.jsonl` is always written locally when the backend is `trackio` or `jsonl`.

Open the local dashboard with:

```bash
uv run trackio show --project medical-jepa-lab
```

## `torch.profiler`: operator-level traces

Use a short run rather than profiling an entire experiment:

```bash
uv run python -m solutions.train_ijepa \
  --config configs/smoke_ijepa_synthetic.yaml \
  --override profiler.enabled=true \
  --max-steps 12
```

For a GPU run, use the real BloodMNIST config but still cap the step count. Traces appear at:

```text
outputs/<run-name>/profiler/rank-<rank>/
```

Useful questions to answer from a trace:

- Is attention or the input pipeline dominant?
- Are there long synchronization gaps between ranks?
- Does the predictor dominate I-JEPA time?
- How much extra encoder work comes from LeJEPA's multiple views?
- Is the SIGReg characteristic-function calculation material relative to the backbone?

Profiler tracing adds overhead, so do not compare its throughput with an ordinary run.

TensorBoard is an optional dependency rather than part of the core environment:

```bash
uv sync --extra dev --extra profile
uv run tensorboard --logdir outputs/<run-name>/profiler
```
