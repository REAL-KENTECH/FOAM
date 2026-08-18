#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
python -m pytest -q
for optimizer in foam stale_shampoo foam_no_adaptive_epsilon foam_no_evd_refresh dr_shampoo adamw; do
  rm -rf "runs/smoke/${optimizer}"
  python -m foam_experiments.train_vit \
    --config "configs/vit/smoke/${optimizer}.yaml" --cpu
done
