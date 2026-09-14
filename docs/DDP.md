# Single GPU and DDP

PyTorch does not make multi-GPU training quite as implicit as JAX sharding, but a clean DDP design still uses one training implementation.

## What changes under `torchrun`

`torchrun` starts one Python process per GPU and provides environment variables for global rank, local rank, and world size. The scaffold then:

- initializes an NCCL process group on CUDA;
- selects `cuda:LOCAL_RANK`;
- wraps the trainable model in `DistributedDataParallel`;
- uses `DistributedSampler` so ranks receive different examples;
- calls `sampler.set_epoch(epoch)` before each epoch;
- averages detached metrics for logging;
- limits Trackio and checkpoint writes to rank 0.

The model forward, loss, backward pass, optimizer step, and EMA update remain the same.

## DDP rather than `nn.DataParallel`

DDP is the standard choice: one process per GPU, explicit data sharding, and generally better performance. `nn.DataParallel` uses one process and replicates work through a primary device; it is retained mostly for convenience and is not used here.

## Batch and learning-rate conventions

`data.batch_size` is local. For an optimizer step:

```text
effective optimizer batch = local batch × number of GPUs × gradient accumulation
```

For LeJEPA, the SIGReg statistic itself is evaluated on each forward pass using `local batch × number of GPUs` samples **per view**. Gradient accumulation averages gradients from several independently regularized microbatches; it does not concatenate those samples into one larger empirical distribution.

There are two legitimate experiments:

1. **Implementation test:** keep local batch fixed, so global batch grows with GPU count.
2. **Scaling test:** divide local batch by GPU count, keeping global batch and optimizer hyperparameters fixed.

Do not compare loss curves without recording which experiment you ran.

## LeJEPA-specific collective

Ordinary DDP averages parameter gradients, but SIGReg estimates a batch distribution. The solution computes characteristic-function means across all ranks through differentiable collectives, so its statistical batch is the cross-rank batch rather than one rank's shard.

## Choosing only some GPUs

`torchrun` consumes the devices visible to the process. On an eight-GPU node, a three-GPU exercise can be launched with:

```bash
CUDA_VISIBLE_DEVICES=1,3,6 uv run torchrun --standalone --nproc-per-node=3 \
  -m medjepa.train_lejepa --config configs/lejepa_bloodmnist_128.yaml
```

Inside each process, `LOCAL_RANK` is still 0, 1, or 2; CUDA maps those local indices onto the selected physical devices.
