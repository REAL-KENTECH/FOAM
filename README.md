# FOAM ViT Experiment Suite — Reconstructed Artifact

Korean quick start: [`README_KO.md`](README_KO.md). Paper-to-code mapping: [`EXPERIMENT_MATRIX.md`](EXPERIMENT_MATRIX.md). Release file map: [`ARTIFACT_MANIFEST.md`](ARTIFACT_MANIFEST.md).

This repository reconstructs the ViT/ImageNet experiment code for **FOAM
(Frequency and Operator Error-Based Adaptive Damping for Shampoo)** from the
uploaded public artifact. The reconstruction removes the runtime monkey patch,
uses one canonical refresh controller inside Distributed Shampoo, and adds the
instrumentation required to reproduce the paper's wall-clock, clean full-train
loss, damping, and eigendecomposition-rate analyses.

## What is fixed

- One canonical implementation of the refresh policy in
  `ShampooPreconditionerList`; `vit.py` is now only a stable entry point.
- The FOAM proxy implements
  `h = RC(epsilon) * alpha(epsilon) / p`.
- Adaptive damping is floored at the base damping `epsilon_0`.
- Fixed-cadence stale Shampoo never evaluates the FOAM proxy.
- Supported refresh policies:
  - `stale_shampoo`
  - `foam`
  - `foam_no_adaptive_epsilon`
  - `foam_no_evd_refresh`
  - `dr_shampoo`
- The stale eigenspace, eigenvalues, adaptive damping, and controller counters
  are checkpointed and restored.
- CPU, single-GPU, and DDP execution paths are supported.
- Exact warmup is derived from `warmup_ratio * total_steps` unless an explicit
  number of warmup steps is supplied.
- Evaluation includes a clean, deterministic full-training-set cross entropy
  (`train_full_hard_ce`) rather than the moving online Mixup objective.
- Both cumulative **training compute time** and end-to-end time are logged.
- Per-factor damping, proxy, EVD, reuse, and refresh counters are exported.

The unmodified uploaded files are retained as text under `legacy/` for
provenance.

## Installation

Create an environment appropriate for the installed CUDA driver. The paper
reported PyTorch 2.8 and CUDA 12.8; `environment-paper.yml` records that target.
For a generic environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Optional Hugging Face, W&B, and AlgoPerf dependencies are listed in
`requirements-optional.txt`.

## Dataset layout

The default paper configurations use the standard ImageFolder layout:

```text
data/imagenet/
├── train/
│   ├── n01440764/
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

Change `data_path` in YAML or on the command line.

## Run the main ViT experiment

The paper configuration uses per-device batch size 256. Four processes therefore
produce global batch size 1024.

```bash
bash scripts/run_vit_4gpu.sh \
  configs/vit/paper/foam_f20_tau075_epsmax3e-7.yaml
```

Equivalent command:

```bash
torchrun --standalone --nproc-per-node=4 \
  -m foam_experiments.train_vit \
  --config configs/vit/paper/foam_f20_tau075_epsmax3e-7.yaml
```

Override any flat configuration field without editing the YAML:

```bash
torchrun --standalone --nproc-per-node=4 \
  -m foam_experiments.train_vit \
  --config configs/vit/paper/foam_f20_tau075_epsmax3e-7.yaml \
  --set base_lr=0.0022 \
  --set matrix_root_inv_threshold=0.5 \
  --set output_dir=runs/vit/foam_tau05_lr22e-4
```

The compatibility entry point also works:

```bash
python vit.py --config configs/vit/smoke/foam.yaml --cpu
```

## Matched baselines and ablations

```bash
# Fixed-cadence stale Shampoo; no proxy calculation.
bash scripts/run_vit_4gpu.sh configs/vit/paper/stale_f20.yaml

# AdamW.
bash scripts/run_vit_4gpu.sh configs/vit/paper/adamw.yaml

# Diagonalization-residual refresh rule.
bash scripts/run_vit_4gpu.sh configs/vit/paper/dr_shampoo_f20_tau075.yaml

# FOAM without adaptive damping.
bash scripts/run_vit_4gpu.sh \
  configs/vit/ablations/foam_no_adaptive_epsilon_f20.yaml

# Adaptive damping without cap-triggered EVD refresh.
bash scripts/run_vit_4gpu.sh \
  configs/vit/ablations/foam_no_evd_refresh_f20.yaml
```

The optional SOAP adapter is configured in `configs/vit/paper/soap_f20.yaml`.
SOAP source was not present in the uploaded artifact and is therefore not
bundled; see `third_party/README.md`.

## Hyperparameter sweep

The repository includes paper-derived sweep definitions:

- `configs/sweeps/vit_lr_selection.yaml`: stale-Shampoo LR selection grid;
- `configs/sweeps/vit_figure1.yaml`: Figure-1-style damping/frequency grid;
- `configs/sweeps/vit_table3.yaml`: matched internal baseline comparison;
- `configs/sweeps/vit_ablation.yaml`: reconstructed ablation modes.

Preview commands before launching:

```bash
python tools/run_sweep.py \
  --sweep configs/sweeps/vit_figure1.yaml \
  --dry-run
```

Run all entries sequentially:

```bash
python tools/run_sweep.py --sweep configs/sweeps/vit_figure1.yaml
```

## Output contract

Each run directory contains:

| File | Meaning |
|---|---|
| `resolved_config.yaml` | Exact configuration used |
| `run_manifest.json` | Environment, command, parameter count, batch size, step count |
| `metrics.csv` | Online objective, clean full-train CE, validation metrics, clocks, EVD rate |
| `factor_diagnostics.csv` | Factor-wise epsilon, proxy, EVD/reuse counters and dimensions |
| `summary.json` | Final run summary |
| `checkpoints/last/` | Model plus rank-local optimizer/RNG/controller state |
| `checkpoints/best/` | Best validation model |
| `factor_snapshots/` | Optional factor matrices and stale eigenspaces for profiling |

### Timing fields

- `train_compute_seconds_cumulative`: sum of synchronized training-loop time;
  excludes evaluation and checkpoint writing. This is the preferred x-axis for
  optimizer wall-clock comparisons.
- `end_to_end_wall_clock_seconds`: epoch metric rows record elapsed time after
  evaluation and diagnostics and before the current epoch's checkpoint write;
  `summary.json` records the final elapsed run time after checkpointing.
- `train_full_eval_seconds` and `validation_seconds`: explicit evaluation cost.

This separation prevents clean full-train evaluation from being silently charged
to one optimizer's update cost.

## Plot and summarize runs

```bash
python tools/summarize_runs.py \
  runs/vit/stale_f20 \
  runs/vit/foam_f20_tau075_epsmax3e-7 \
  --output reports/vit_summary.csv

python tools/plot_results.py \
  runs/vit/stale_f20 \
  runs/vit/foam_f20_tau075_epsmax3e-7 \
  --output-dir reports/plots

# Factor-wise epsilon, proxy, and EVD-rate dynamics for one FOAM run.
python tools/plot_diagnostics.py \
  runs/vit/foam_f20_tau075_epsmax3e-7 \
  --output-dir reports/diagnostics

# Wall-clock/quality ablation scatter and compact CSV summary.
python tools/plot_ablation.py \
  runs/vit/stale_f20 \
  runs/vit/foam_f20_tau075_epsmax3e-7 \
  runs/vit/foam_no_adaptive_epsilon_f20 \
  runs/vit/foam_no_evd_refresh_f20 \
  --output-dir reports/ablation
```

## Factor profiling

Enable snapshots in a configuration:

```yaml
factor_snapshot_interval: 10
factor_snapshot_max_per_rank: 32
```

Then compare the proxy and direct EVD on the saved real training factors:

```bash
python tools/profile_snapshots.py \
  runs/vit/foam_profile/factor_snapshots/epoch_010_rank_0000.pt \
  --device cuda \
  --output reports/factor_profile.csv
```

## Checkpoint and resume

Checkpoints are rank-local because the optimizer state may be sharded. Resume
requires the same world size:

```bash
torchrun --standalone --nproc-per-node=4 \
  -m foam_experiments.train_vit \
  --config configs/vit/paper/foam_f20_tau075_epsmax3e-7.yaml \
  --resume runs/vit/foam_f20_tau075_epsmax3e-7/checkpoints/last
```

The checkpoint contains Python, NumPy, PyTorch, CUDA, DataLoader-generator, AMP,
and FOAM controller state. Resume is epoch-boundary exact; mid-epoch replay is
not implemented.

For preemption testing without changing the planned learning-rate schedule,
set the execution-only `stop_after_epoch` field. The run saves a normal
checkpoint and exits after the requested absolute epoch. Exact CPU equivalence
of uninterrupted and resumed model/optimizer tensor states can be checked with:

```bash
python tools/verify_resume_equivalence.py
```

## Validation

```bash
make test
```

The test suite checks:

- the exact proxy formula, including the `1/p` factor;
- the base-damping floor;
- zero proxy calls in stale mode;
- scalar inverse-root API consistency;
- checkpoint restoration of adaptive epsilon and eigenspace state;
- a complete CPU synthetic training run and artifact contract.

The executed checks and remaining unverified large-scale claims are recorded
in `VERIFICATION_REPORT.md`.

For a larger local exercise of every optimizer mode:

```bash
bash scripts/run_smoke_tests.sh
```

Quick and extended packaged verification:

```bash
bash scripts/verify_artifact.sh quick
bash scripts/verify_artifact.sh extended
```

The extended mode additionally runs the two-rank CPU/Gloo DDP smoke test,
exact checkpoint/resume equivalence, factor snapshot export, and offline
proxy/EVD profiling.

## Repository map

```text
foam_experiments/          Reconstructed training, data, model, metrics, checkpointing
optimizers/                Distributed Shampoo with canonical FOAM controller
configs/vit/               Paper, ablation, profiling, and smoke configurations
configs/sweeps/            Reproducible sweep definitions
scripts/                   Launch helpers
tools/                     Sweep, plotting, summarization, and profiling tools
tests/                     Controller and end-to-end regression tests
legacy/                    Original uploaded files retained as text
```

## Scope and remaining limitations

- This artifact was statically validated and exercised with CPU synthetic
  workloads. A full 90-epoch ImageNet-1K run on four A6000 GPUs was not executed
  in the reconstruction environment.
- The custom ViT preserves the uploaded architecture and source initialization
  by default. Set `init_scheme: vit` or use the optional timm implementation
  only as an explicitly different experiment.
- The theory in the paper models a simpler Shampoo update than the empirical
  implementation, which also uses bias correction, momentum, Adam grafting,
  blocking, and delayed preconditioning. Those implementation choices are
  recorded in every resolved configuration and run manifest.
- SOAP remains an external optional dependency because its implementation was
  not part of the uploaded source archive.
- The exact private implementation of the paper's ablations and its
  `full-batch train loss` routine were not in the archive. Their reconstructed
  semantics are documented in `RECONSTRUCTION_REPORT.md` and
  `VERIFICATION_REPORT.md`; no byte-for-byte equivalence is claimed.
