# Experiment matrix

이 문서는 논문의 ViT 실험 항목과 재구성 코드의 실행 파일을 대응시킨다.

| Paper item | Reconstructed entry | Notes |
|---|---|---|
| AdamW matched baseline | `configs/vit/paper/adamw.yaml` | Same model/data pipeline and LR selected from stale-Shampoo grid |
| Fixed-cadence stale Shampoo | `configs/vit/paper/stale_f20.yaml` | Strict no-proxy path; EVD at every check |
| Figure 1 stale settings | `configs/sweeps/vit_figure1.yaml` | Includes `f=30`, `epsilon0` in `{1e-9,1e-8}` |
| Main FOAM setting | `configs/vit/paper/foam_f20_tau075_epsmax3e-7.yaml` | `f=20`, `tau=.75`, `epsilon_max=3e-7` |
| Figure 1 FOAM grid | `configs/sweeps/vit_figure1.yaml` | `tau` in `{.25,.5,.75}`, `epsilon_max` in `{1e-7,3e-7,5e-7}` |
| DR-Shampoo | `configs/vit/paper/dr_shampoo_f20_tau075.yaml` | EVD trigger from stale-basis diagonalization residual |
| FOAM without adaptive epsilon | `configs/vit/ablations/foam_no_adaptive_epsilon_f20.yaml` | Proxy decides refresh; damping stays at `epsilon0` |
| FOAM without EVD refresh | `configs/vit/ablations/foam_no_evd_refresh_f20.yaml` | Adaptive damping is capped; no scheduled fresh EVD after initialization |
| Figure 4 ablations | `configs/sweeps/vit_ablation.yaml` | Includes `f=20` and `f=50` matched modes |
| LR selection grid | `configs/sweeps/vit_lr_selection.yaml` | `{1.75,2.00,2.20,2.35}e-3`, selected using stale Shampoo |
| Table 3 internal baselines | `configs/sweeps/vit_table3.yaml` | AdamW, stale, DR-Shampoo, FOAM |
| SOAP | `configs/vit/paper/soap_f20.yaml` | Optional adapter; external `soap.py` must be supplied |
| Epsilon dynamics | `factor_diagnostics.csv`, `tools/plot_diagnostics.py` | Factor-level current epsilon, proxy, and EVD-rate trajectories |
| EVD operation rate | `metrics.csv`, `factor_diagnostics.csv` | Total and left/right factor rates |
| Proxy/EVD cost | `configs/vit/profiling/foam_profile.yaml`, `tools/profile_snapshots.py` | Online counters plus offline real-factor benchmark |
| Wall-clock curves | `tools/plot_results.py` | Can plot training-compute or end-to-end time |
| Ablation scatter | `tools/plot_ablation.py` | Wall-clock/quality scatter plus `ablation_summary.csv` |
| Final summary table | `tools/summarize_runs.py` | Best clean train CE, best validation accuracy, time, EVD rate |

## Metric definitions

- `train_online_objective`: minibatch training objective after Mixup/CutMix and
  label smoothing; model parameters change during accumulation.
- `train_full_hard_ce`: deterministic evaluation-mode cross entropy over the
  complete training split with hard labels and evaluation transforms.
- `train_compute_seconds_cumulative`: training-step compute time, excluding
  full-train and validation evaluation.
- `end_to_end_wall_clock_seconds`: elapsed run time including evaluation,
  diagnostics, and checkpointing.
- `evd_rate`: cumulative `evd_calls / check_calls` over locally owned factors,
  aggregated across ranks.

The exact private definition used to generate the paper's reported
“full-batch train loss” was not present in the uploaded archive. The explicit
`train_full_hard_ce` definition prevents the reconstructed result from being
confused with the original online minibatch objective.
