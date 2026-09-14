# Evaluating the embedding space

The desired final picture is useful, but it is easy to fool yourself with dimensionality reduction. This project creates both pictures and held-out measurements.

## Procedure

1. Freeze the learned encoder.
2. Extract one pooled embedding per training, validation, and test image.
3. Fit PCA on training embeddings only; transform the test set.
4. Optionally fit UMAP on a reproducible training subset; transform the test set. The default cap is 20,000 training embeddings and can be changed with `--umap-fit-samples`.
5. Train a standardized logistic-regression probe on frozen training embeddings.
6. Train a 5-nearest-neighbor classifier on normalized training embeddings.
7. Report test accuracy and balanced accuracy.
8. Compute silhouette score on a capped test subset.

## Interpretation

- **Linear probe:** asks whether classes are linearly accessible in the representation.
- **5-NN:** asks whether local neighborhoods respect labels without a trained nonlinear head.
- **Balanced accuracy:** prevents common classes from dominating the headline.
- **Silhouette:** measures label-cluster compactness/separation, but can punish continuous biological structure.
- **PCA:** deterministic linear summary; often the most honest first plot.
- **UMAP:** useful for local visual structure but highly sensitive to settings and can create apparent islands.

## A fair I-JEPA versus LeJEPA comparison

Keep the dataset split, image resolution, backbone width/depth, optimizer family, global batch, and evaluation protocol fixed. Parameter counts will still differ because I-JEPA has a predictor during pretraining and an EMA encoder copy in memory; the downstream encoder being evaluated should have the same backbone capacity.
