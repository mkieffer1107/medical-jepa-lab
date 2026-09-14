# Validation performed for this scaffold

The generated project was checked in the build environment with:

- syntax parsing of every Python file in `src/`, `solutions/`, `scripts/`, and `tests/`;
- **16 passing unit tests**, with two exercise-contract tests intentionally skipped until the corresponding TODO constructors are implemented;
- end-to-end, one-process reference I-JEPA and LeJEPA synthetic training runs;
- checkpoint save/reload and frozen-embedding PCA, linear-probe, 5-NN, balanced-accuracy, and silhouette reports;
- two-process `torchrun` reference runs for both algorithms using the CPU/Gloo backend;
- SIGReg forward/backward tests, including a check that collapsed features receive a larger penalty than approximately Gaussian features;
- deterministic tests of the UMAP training-fit sample cap.

The build environment had no external package-download access and no CUDA device. It therefore did **not**:

- download BloodMNIST+ through MedMNIST;
- execute the CUDA/NCCL path;
- launch the Trackio dashboard;
- execute the actual UMAP reducer, because `umap-learn` was not installed in the build environment.

All four packages are declared in `pyproject.toml`, so a normal `uv sync --extra dev` installs the runtime dependencies before those paths are used. The CUDA path shares the same tested one-process/DDP control flow, with device selection and NCCL chosen automatically by `init_distributed()` when CUDA is available.

The build environment also lacked the `ruff` executable, so linting was not claimed as part of validation. The included `make lint` target runs it after the development dependencies are installed. Package resolution was unavailable as well, so the archive does not contain a fabricated `uv.lock`; the first `uv sync` creates one against the user's platform and configured PyTorch index.
