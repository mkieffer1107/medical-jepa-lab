#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <ijepa|lejepa> <gpu-count> <config.yaml> [extra args...]" >&2
  exit 2
fi

algorithm="$1"
gpus="$2"
config="$3"
shift 3

case "$algorithm" in
  ijepa|lejepa) ;;
  *) echo "algorithm must be ijepa or lejepa" >&2; exit 2 ;;
esac

module="medjepa.train_${algorithm}"
if [[ "$gpus" == "1" ]]; then
  exec uv run python -m "$module" --config "$config" "$@"
else
  exec uv run torchrun --standalone --nproc-per-node="$gpus" \
    -m "$module" --config "$config" "$@"
fi
