# Dataset choices

## Recommended first run: BloodMNIST+

Why it fits this exercise:

- Eight blood-cell categories are visually distinct enough that a useful self-supervised representation often produces an interpretable embedding plot.
- The dataset is large enough to be more meaningful than MNIST but small enough for repeated weekend experiments.
- MedMNIST+ provides 128×128 and 224×224 versions with fixed train/validation/test splits.
- Color and morphology both carry signal, making augmentation choices worth thinking about.

Start with `bloodmnist` at 128×128 and patch size 8. That yields a 16×16 patch grid without the pixel cost of 224×224.

## Good second run: PathMNIST+

PathMNIST is substantially larger and contains multiple colorectal histology tissue classes. It is a better stress test after the code works, but stain/color augmentation can strongly affect what the representation learns. Use medically conservative color jitter first.

Try it without copying a config:

```bash
uv run python -m solutions.train_ijepa \
  --config configs/ijepa_bloodmnist_128.yaml \
  --override data.dataset=pathmnist \
  --override experiment.run_name=ijepa-pathmnist-128
```

The rest of the loader is metadata-driven.

## Good grayscale comparison: OrganAMNIST+ or OCTMNIST+

These datasets let you test whether the same objective works when color is absent or less important. The loader converts inputs to three channels so the same model code can be reused.

## Datasets not recommended for the first weekend

- Very small binary datasets: the plot may look “clean” without teaching much about representation learning.
- Full chest X-ray corpora: larger storage, slower data pipelines, multilabel targets, and patient-level splitting distract from the JEPA mechanisms.
- 3D CT/MRI: patching, anisotropy, and memory management become the project.

## Label discipline

Training loaders return labels because the underlying dataset API does, but the self-supervised loops must not use them. Labels are consumed only by the evaluator and optional visualization tools. The fixed MedMNIST splits should be preserved.
