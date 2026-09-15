# Medical JEPA Lab

A weekend-sized, hands-on PyTorch project for implementing two self-supervised image representation learners. In **LeJEPA**, “Le” means **lean**.

1. **I-JEPA** — a context encoder predicts latent representations of masked target regions produced by an EMA target encoder.
2. **LeJEPA** — one trainable encoder learns view-invariant representations while **SIGReg** regularizes the embedding distribution; there is no target/teacher encoder and no EMA update.

The default experiment uses **BloodMNIST+ at 128×128**. Labels are ignored during self-supervised training and used only afterward for PCA/UMAP plots, k-nearest-neighbor evaluation, a frozen linear probe, and a silhouette score.

The repository is intentionally split into:

- `src/medjepa/`: the **exercise version**. Data, distributed setup, logging, evaluation, checkpointing, and other plumbing are complete. Model and optimization logic contain numbered TODOs.
- `solutions/`: complete reference implementations. Leave this folder closed until you want to check a shape, compare an implementation, or run the known-working baseline.

## Set up

```bash
uv sync --extra dev
```

The archive intentionally has no fabricated `uv.lock`: the build environment could not resolve packages from the network. Your first `uv sync` will create the lock for the platform and PyTorch index available on your GPU machine.

On a CUDA machine, `uv` will install the PyTorch build selected by the package resolver. If your cluster requires a site-specific CUDA wheel/index, install PyTorch according to the cluster instructions first, then run `uv sync`.

## Inspect the data before coding

### Interactive TODO checklist

Run `./scripts/todos.sh` (requires Python 3.9+). Enter a TODO number to open its
source line, or `d 1` to toggle TODO 1 complete. Completed entries show `[x]` and
strikethrough in terminals that support it. Progress is saved locally and ignored
by Git; marking an item complete does not change or validate its implementation.
Use `r` to refresh line locations and `q` to quit. Removed TODO comments remain in
the saved checklist at their last known location.

Full file paths use your IDE terminal's native file links: Cmd-click in Cursor
to open the source line in Cursor (Ctrl-click on Linux/Windows). Run the script
inside your IDE's integrated terminal for this behavior.
Number-based opening uses `TODO_EDITOR`, `VISUAL`, or
`EDITOR`, then detects VS Code, Cursor, or Vim. For example:
`TODO_EDITOR='cursor' ./scripts/todos.sh`. Other editors should accept `+LINE FILE`.
Use `./scripts/todos.sh --list` for a plain listing without saving progress.

```bash
uv run medjepa-inspect \
  --config configs/ijepa_bloodmnist_128.yaml \
  --output reports/ijepa_batch.png

uv run medjepa-inspect \
  --config configs/lejepa_bloodmnist_128.yaml \
  --output reports/lejepa_views.png
```

## Run the exercise implementation

Single GPU:

```bash
uv run medjepa-train-ijepa --config configs/ijepa_bloodmnist_128.yaml
uv run medjepa-train-lejepa --config configs/lejepa_bloodmnist_128.yaml
```

Two, three, or four GPUs use the **same Python module**:

```bash
uv run torchrun --standalone --nproc-per-node=2 \
  -m medjepa.train_ijepa --config configs/ijepa_bloodmnist_128.yaml

uv run torchrun --standalone --nproc-per-node=4 \
  -m medjepa.train_lejepa --config configs/lejepa_bloodmnist_128.yaml
```

Select a subset of an eight-GPU node explicitly:

```bash
CUDA_VISIBLE_DEVICES=2,3,4 uv run torchrun --standalone --nproc-per-node=3 \
  -m medjepa.train_ijepa --config configs/ijepa_bloodmnist_128.yaml
```

Use `--override data.batch_size=16` for a quick memory adjustment. `data.batch_size` is **per process / per GPU**, so the effective optimizer batch is `batch_size × world_size × gradient_accumulation_steps`. LeJEPA's SIGReg distribution statistics are computed on each forward pass across `batch_size × world_size` samples per view; gradient accumulation averages several such losses rather than forming one larger SIGReg sample set.

For the first real pass, run 25 epochs before committing to the 100-epoch configs:

```bash
uv run medjepa-train-ijepa --config configs/ijepa_bloodmnist_128.yaml \
  --override optimization.epochs=25 \
  --override experiment.run_name=ijepa-bloodmnist-25ep
```

## Run the reference solution

These commands do not overwrite the exercise files:

```bash
uv run python -m solutions.train_ijepa \
  --config configs/ijepa_bloodmnist_128.yaml \
  --override experiment.run_name=ijepa-reference

uv run torchrun --standalone --nproc-per-node=2 \
  -m solutions.train_lejepa \
  --config configs/lejepa_bloodmnist_128.yaml \
  --override experiment.run_name=lejepa-reference
```

For an offline plumbing test with no dataset download:

```bash
uv run python -m solutions.train_ijepa \
  --config configs/smoke_ijepa_synthetic.yaml --max-steps 8

uv run python -m solutions.train_lejepa \
  --config configs/smoke_lejepa_synthetic.yaml --max-steps 8
```

## Evaluate the representation

### Plot encoder embeddings

From the repo root, create 2D and 3D PCA and t-SNE plots colored by test-set class:

```bash
uv run medjepa-plot-embeddings
```

This selects the most recently modified `outputs/**/latest.pt` from a finished run,
loads its saved configuration, and uses the I-JEPA EMA target encoder (or the LeJEPA
backbone). It fits PCA on training embeddings and projects test embeddings; class
labels are used only to color points and name the legend. Progress is printed while
extracting embeddings and fitting projections, followed by all absolute output paths.
The encoder is frozen; this command does not train the model. t-SNE fits test
embeddings independently in 2D and 3D, after PCA preprocessing to at most 50
dimensions; its seed and adaptive perplexity are recorded in the metadata.

Outputs under `reports/<run>/embeddings/`:

- `embeddings.html`: a standalone offline page with PCA/t-SNE and 2D/3D switches,
  rotation, zoom, class legend toggles, point tooltips, reset view, and PNG export.
  Copy this one file to your computer and open it in your browser; no server is needed.
- `pca_test.png`, `pca_test_3d.png`, `tsne_test.png`, `tsne_test_3d.png`: static plots.
- `test_embeddings.npz`: raw embeddings, labels, 2D/3D projection coordinates.
- `projection.json`: checkpoint and projection metadata.
- `clusters.json`: K-means assignments, class composition, and representative sample indices.
- `samples.json`: dataset/split/row IDs and source archive URL (without image bytes).

The HTML also includes lossless source PNGs. Hover a point to preview the exact
sample; click it to pin the inspector and download its PNG. BloodMNIST is loaded
from the MedMNIST Zenodo archive, not a verified Hugging Face row mapping. Sample IDs
include dataset, image size, split, and zero-based archive row. Images are embedded
for offline use. Browser Blob URLs are created lazily in memory and revoked when
leaving the page; there is no localStorage, IndexedDB, or persistent image cache.
The downloaded HTML itself retains its embedded images.

K-means runs on full, unscaled test encoder embeddings with Euclidean distance,
independently of PCA/t-SNE and ground-truth labels. The default is nine clusters:

```bash
uv run medjepa-plot-embeddings --clusters 9 --cluster-examples 6
```

Hover/focus a cluster row to gray out the others, click its name to pin selection,
and use **Show all clusters** to clear it. Each row lists its class mixture and
shows the nearest and farthest members from its centroid, followed by greedy
farthest-first examples for diversity. These are examples, not all cluster members;
use the class counts to see the full mixture. Cluster numbers are run-specific and
K-means groups need not match apparent t-SNE islands.


Clusters in these projections are exploratory, not a measure of downstream accuracy.
t-SNE cluster sizes and between-cluster distances may be misleading.

```bash
# Choose any checkpoint explicitly, including an unfinished run:
uv run medjepa-plot-embeddings --checkpoint outputs/ijepa-bloodmnist-128/latest.pt

# Use another checkpoint directory or adjust evaluation loading:
uv run medjepa-plot-embeddings --checkpoint-dir /path/to/outputs --batch-size 32 \
  --override data.num_workers=0
```

New checkpoints record completion at the configured final epoch or `--max-steps`
limit. Older checkpoints are recognized as finished when their saved epoch reaches
the configured epoch count; select older short runs explicitly with `--checkpoint`.
Use `--ijepa-encoder student` for the context encoder instead of the default target.

First record an untrained-backbone baseline. Blood-cell classes can already be separable by color and gross morphology, so this check prevents a random feature map from getting credit for your JEPA training:

```bash
uv run medjepa-eval \
  --config configs/ijepa_bloodmnist_128.yaml \
  --algorithm ijepa --implementation solution --random-init --reducer pca \
  --output-dir reports/ijepa-random-init
```

I-JEPA normally evaluates the EMA target encoder:

```bash
uv run medjepa-eval \
  --config configs/ijepa_bloodmnist_128.yaml \
  --checkpoint outputs/ijepa-reference/latest.pt \
  --algorithm ijepa \
  --implementation solution \
  --reducer both
```

LeJEPA:

```bash
uv run medjepa-eval \
  --config configs/lejepa_bloodmnist_128.yaml \
  --checkpoint outputs/lejepa-reference/latest.pt \
  --algorithm lejepa \
  --implementation solution \
  --reducer both
```

The evaluator writes plots, embeddings, and a `metrics.json` file beneath `reports/<run-name>/`. PCA is fitted on frozen **training** embeddings and applied to the test set. UMAP is treated as a visualization, not evidence by itself; by default it is fit on at most 20,000 training embeddings so the PathMNIST follow-up remains practical. The quantitative checks are frozen linear-probe accuracy, 5-NN accuracy, balanced accuracy, and silhouette score.

## Tracking and profiling

The default tracker is **Trackio**. Every rank writes no dashboard data except rank 0, and every run also receives an append-only local `metrics.jsonl`, so a network/login failure never destroys the experiment record.

Launch the local dashboard after or during a run with:

```bash
uv run trackio show --project medical-jepa-lab
```

To sync a Trackio run to a Hugging Face Space, set:

```yaml
tracking:
  backend: trackio
  space_id: your-user/your-trackio-space
```

Disable remote tracking with `tracking.backend: jsonl`.

Optional PyTorch profiler traces are controlled in each YAML config. Enable `profiler.enabled: true`; traces are written under the run directory and can be opened in TensorBoard or Perfetto-compatible viewers.

Install the optional TensorBoard viewer with:

```bash
uv sync --extra dev --extra profile
```

## Where to start

Read these in order:

1. `docs/WEEKEND_PLAN.md`
2. `docs/OBJECTIVES.md`
3. `docs/PROFILING.md`
4. `TASKS.md`
5. `src/medjepa/models/blocks.py`
6. `src/medjepa/models/ijepa.py` or `src/medjepa/models/lejepa.py`
7. the corresponding training file

This is a representation-learning exercise, not a diagnostic model. MedMNIST is intended for educational benchmarking; do not interpret the resulting clusters as clinical phenotypes.
