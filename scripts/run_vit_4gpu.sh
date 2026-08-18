#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/vit/paper/foam_f20_tau075_epsmax3e-7.yaml}"
shift || true
torchrun --standalone --nproc-per-node=4 -m foam_experiments.train_vit \
  --config "$CONFIG" "$@"
