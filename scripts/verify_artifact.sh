#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}

MODE=${1:-quick}
case "$MODE" in
  quick|extended) ;;
  *) echo "Usage: $0 [quick|extended]" >&2; exit 2 ;;
esac

make verify

if [[ "$MODE" == "extended" ]]; then
  timeout 120 bash scripts/run_ddp_smoke.sh
  python tools/verify_resume_equivalence.py \
    --output-dir runs/verification_resume_equivalence
  bash scripts/run_smoke_tests.sh
  rm -rf runs/verification_snapshot
  python -m foam_experiments.train_vit \
    --config configs/vit/smoke/foam.yaml \
    --set output_dir=runs/verification_snapshot \
    --set factor_snapshot_interval=1 \
    --set factor_snapshot_max_per_rank=2 \
    --set expected_world_size=0 \
    --cpu
  python tools/profile_snapshots.py \
    runs/verification_snapshot/factor_snapshots/epoch_001_rank_0000.pt \
    --output runs/verification_snapshot/snapshot_profile.csv \
    --warmup 1 --repeats 2
fi
