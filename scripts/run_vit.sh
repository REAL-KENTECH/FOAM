#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CONFIG.yaml [additional train_vit arguments]" >&2
  exit 2
fi

CONFIG=$1
shift
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-29500}

exec torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  -m foam_experiments.train_vit \
  --config "${CONFIG}" \
  "$@"
