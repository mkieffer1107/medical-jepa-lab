#!/usr/bin/env bash
#SBATCH --job-name=medical-jepa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Adapt module/venv setup to your cluster.
# module load cuda
uv sync

srun uv run torchrun \
  --standalone \
  --nproc-per-node=4 \
  -m medjepa.train_ijepa \
  --config configs/ijepa_bloodmnist_128.yaml
