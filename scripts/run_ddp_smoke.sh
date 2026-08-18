#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUTPUT_DIR=${OUTPUT_DIR:-runs/smoke_ddp/foam}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29613}
rm -rf "$OUTPUT_DIR"
torchrun \
  --nnodes=1 \
  --nproc-per-node=2 \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  -m foam_experiments.train_vit \
  --config configs/vit/smoke/foam.yaml \
  --set "output_dir=${OUTPUT_DIR}" \
  --set expected_world_size=0 \
  --cpu
