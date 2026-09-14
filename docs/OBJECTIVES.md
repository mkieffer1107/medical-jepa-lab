# The two objectives

## I-JEPA

Given an image split into patch tokens:

1. A **context encoder** receives only selected visible/context patch indices.
2. A **target encoder** processes the full image. Its target-region patch representations are selected and detached from gradient computation.
3. A **predictor** receives context representations plus positional information for the hidden target locations and predicts the target encoder's representations.
4. The context encoder and predictor are optimized using a latent regression loss.
5. The target encoder is updated as an exponential moving average of the context encoder.

The target branch does not reconstruct pixels. It supplies latent targets. The EMA branch changes the training dynamics and prevents both sides of the regression target from chasing one another directly.

This lab follows the official multiblock idea with one context mask and multiple target blocks. The default mask scales are close to the official ImageNet configuration, but the model is drastically smaller and the image size is 128 rather than 224.

## LeJEPA

“Le” refers to a **lean** JEPA formulation. In the image setup used here:

1. Produce several augmented views of each image.
2. Encode all views with one trainable encoder and projection head.
3. Minimize an invariance term that pulls each view projection toward that image's mean projection.
4. Add **Sketched Isotropic Gaussian Regularization (SIGReg)** so the batch distribution of projected representations approaches an isotropic standard Gaussian under many random one-dimensional projections.

There is no EMA target encoder, no stop-gradient target branch, and no teacher/student pair. The projection head is a trainable component, not a teacher.

For a projected tensor `z` with shape `(views, batch, dimension)`, the minimal invariance term is the mean squared deviation from the per-image mean over views. SIGReg samples random unit directions, projects embeddings onto each direction, compares empirical characteristic functions with the standard-normal characteristic function, and averages the discrepancy.

## Why the regularizer matters

Invariance alone has a trivial solution: every image and every view can map to the same vector. SIGReg makes that collapsed distribution expensive by asking the aggregate representation distribution to retain nonzero, approximately isotropic variation.

## What is simplified here

- The backbone is a tiny ViT implemented for readability.
- LeJEPA uses four equal-resolution random crops rather than the larger paper-scale global/local crop recipe.
- BloodMNIST is tiny relative to ImageNet.
- Hyperparameters are educational defaults, not a reproduced benchmark.

The point is to make every mechanism visible in a small codebase, then transfer the implementation pattern to your larger medical or EEG work.
