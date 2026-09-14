# A realistic weekend plan

## Friday evening — plumbing and shape contracts

1. `uv sync --extra dev`.
2. Run both synthetic reference smoke tests.
3. Inspect the BloodMNIST I-JEPA masks and LeJEPA views.
4. Implement the shared tiny ViT and pass its contract tests.
5. Read the two objective summaries in `OBJECTIVES.md` while keeping the official papers/repositories nearby.

Stop Friday once a random batch reaches your encoder and every tensor shape is understood.

## Saturday morning — I-JEPA

1. Implement the predictor.
2. Verify that predicted and target latent tensors have identical shapes.
3. Implement a single optimization step without EMA.
4. Add the EMA update and verify target parameters have no gradients.
5. Overfit a 32-image synthetic subset for a few dozen steps.
6. Train BloodMNIST on one GPU.

The useful learning objective is not “make the loss tiny.” Confirm that the predictor cannot see the target pixels, target outputs are detached, and the target encoder moves only through EMA.

## Saturday afternoon — LeJEPA

1. Implement the single-encoder multi-view model.
2. Implement invariance loss.
3. Implement SIGReg first on a normal random tensor, then on model projections.
4. Check that a constant/collapsed tensor receives a larger regularization penalty than approximately Gaussian random features.
5. Train BloodMNIST on one GPU.

Use float32 inside the characteristic-function calculation even when the encoder runs in bfloat16.

## Sunday morning — DDP

1. Run the synthetic config on two GPUs.
2. Run the same checkpointable script on four GPUs.
3. Compare global batch sizes fairly. For a scaling experiment, keep the global batch fixed by lowering the per-GPU batch.
4. Confirm that changing GPU count does not silently change the learning-rate convention.
5. Inspect throughput, data-loader time, and peak memory.

BloodMNIST is small enough that DDP may be educational rather than faster; process startup, synchronization, and input loading can dominate.

## Sunday afternoon — the payoff

1. Evaluate the same architecture at random initialization as a baseline.
2. Freeze each trained encoder.
3. Fit PCA on training embeddings and transform the test embeddings.
4. Run UMAP as a secondary view.
5. Fit a linear probe and 5-NN classifier on training embeddings.
6. Compare the quantitative metrics with the pictures and random-init baseline.
7. Save a short table recording algorithm, global batch, epochs, wall time, linear accuracy, balanced accuracy, k-NN accuracy, silhouette, and peak memory.

A visually separated UMAP with poor held-out probe performance is not a successful representation.
