# FOAM ViT reconstructed artifact manifest

This manifest identifies the principal files in the completed experiment suite.
The uploaded source is preserved under `legacy/`; reconstructed or modified files
are documented in `MODIFICATIONS.md` and `RECONSTRUCTION_REPORT.md`.

## Primary entry points

| Path | Purpose |
|---|---|
| `vit.py` | Compatibility launcher for the reconstructed ViT experiment |
| `foam_experiments/train_vit.py` | Canonical training, evaluation, timing, diagnostics, and checkpoint loop |
| `foam_experiments/optim.py` | Matched optimizer factory for AdamW, Shampoo/FOAM modes, and optional SOAP |
| `optimizers/distributed_shampoo/distributed_shampoo.py` | Distributed Shampoo optimizer with refresh-policy plumbing and diagnostics API |
| `optimizers/distributed_shampoo/utils/shampoo_preconditioner_list.py` | Canonical FOAM/DR-Shampoo controller and checkpointed factor state |
| `submission.py` | Reconstructed AlgoPerf-compatible optimizer submission entry point |

## Paper and ablation configurations

| Path | Purpose |
|---|---|
| `configs/vit/paper/foam_f20_tau075_epsmax3e-7.yaml` | Main reconstructed FOAM setting |
| `configs/vit/paper/stale_f20.yaml` | Fixed-cadence stale Shampoo with no proxy work |
| `configs/vit/paper/adamw.yaml` | Matched AdamW baseline |
| `configs/vit/paper/dr_shampoo_f20_tau075.yaml` | Diagonal-residual refresh baseline |
| `configs/vit/paper/soap_f20.yaml` | Optional adapter configuration for externally supplied SOAP source |
| `configs/vit/ablations/` | No-adaptive-epsilon and no-EVD-refresh modes |
| `configs/sweeps/` | LR selection, Figure-1 grid, Table-3 comparison, and ablation sweeps |

## Analysis and verification tools

| Path | Purpose |
|---|---|
| `tools/summarize_runs.py` | Aggregate completed runs to a compact CSV |
| `tools/plot_results.py` | Wall-clock learning curves |
| `tools/plot_diagnostics.py` | Factor-wise epsilon, proxy, and EVD-rate dynamics |
| `tools/plot_ablation.py` | Wall-clock/quality ablation scatter and CSV summary |
| `tools/profile_snapshots.py` | Proxy-versus-EVD profiling on saved real factor states |
| `tools/verify_resume_equivalence.py` | Exact uninterrupted-versus-resumed model/optimizer tensor comparison |
| `scripts/verify_artifact.sh` | Quick or extended packaged verification |
| `tests/` | Controller, checkpoint, matrix-function, tooling, and end-to-end smoke tests |

## Documentation

- `README.md`: full English operation guide.
- `README_KO.md`: Korean quick-start guide.
- `EXPERIMENT_MATRIX.md`: paper-to-code mapping and metric definitions.
- `RECONSTRUCTION_REPORT.md`: source issues and reconstruction decisions.
- `VERIFICATION_REPORT.md`: executed checks and unverified large-scale claims.
- `TEST_RESULTS.txt`: compact recorded validation output.
- `LICENSES.md`: license and attribution boundary.

## Verification status at release

Executed successfully in the reconstruction environment:

- `make verify`: source compilation, 9 pytest tests, and 15 ViT YAML validations;
- all six internal optimizer modes on CPU synthetic data;
- 2-rank CPU/Gloo DDP training with two rank-local optimizer checkpoints;
- exact epoch-boundary resume equivalence: 24 model tensors and 661 optimizer
  tensors, both with maximum absolute difference `0.0`;
- factor snapshot export, proxy/EVD profiling, result plotting, diagnostic
  plotting, and ablation plotting.

Not executed in the reconstruction environment:

- the full 90-epoch ImageNet-1K run on 4×A6000;
- external SOAP training, because SOAP source was absent from the uploaded archive;
- end-to-end AlgoPerf execution, because the AlgoPerf package/workload was absent.
