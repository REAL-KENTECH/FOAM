# FOAM ViT 전체 실험 코드 재구성 보고서

## 1. 재구성 목표

업로드된 `FOAM-main(1).zip`의 ViT 실험 코드를 기준으로 다음 조건을
만족하는 하나의 실행 가능한 artifact로 재구성했다.

1. FOAM 수식을 구현하는 canonical code path가 하나만 존재할 것.
2. Stale Shampoo와 FOAM의 wall-clock 비교에서 baseline이 FOAM proxy 비용을
   부담하지 않을 것.
3. 논문의 clean full-batch training loss, EVD operation rate, damping dynamics,
   wall-clock 지표를 코드에서 직접 생성할 수 있을 것.
4. 중단·재시작 시 stale eigenspace와 adaptive damping이 보존될 것.
5. CPU smoke test, single GPU, multi-GPU DDP가 같은 experiment interface를
   사용할 것.

## 2. 확인된 문제와 수정 결과

| 기존 문제 | 재구성 결과 |
|---|---|
| `vit.py`의 종료되지 않은 docstring으로 실행 불가 | `vit.py`를 안정적인 entry point로 교체하고 원본은 `legacy/`에 보존 |
| optimizer library와 `vit.py` monkey patch에 서로 다른 FOAM 구현 존재 | monkey patch 제거, `ShampooPreconditionerList`만 canonical implementation으로 사용 |
| library FOAM에서 `h=RC·alpha`로 구현되어 `1/p` 누락 | `h=RC·alpha/p`로 수정하고 regression test 추가 |
| adaptive epsilon이 `epsilon_0` 아래로 감소 가능 | `max(epsilon_0, epsilon_{t-1} h/tau)` floor 적용 |
| stale baseline에서도 proxy를 계산할 가능성 | `stale` mode는 매 check에서 EVD만 수행하며 proxy counter가 항상 0 |
| FOAM의 `Q`, `D`, `epsilon_t`가 checkpoint state에 없음 | eigenspace, eigenvalues, epsilon, refresh counters를 tensor state로 저장 |
| AlgoPerf `submission.py`에서 FOAM이 비활성화 | refresh policy hyperparameter를 canonical optimizer에 전달하도록 수정 |
| CPU/non-DDP 경로에서 process group 또는 CUDA device 가정 | rank/device-safe allocation과 CPU execution path 적용 |
| scalar factor에서 inverse-root 반환 arity 불일치 | 모든 EVD 경로가 `(inverse, epsilon, eigenvalues, eigenvectors)` 반환 |
| fixed 5,200-step warmup | 기본적으로 실제 total step의 5%를 계산 |
| online Mixup loss를 full-batch loss처럼 사용할 위험 | eval transform, hard label, `model.eval()` 기반 `train_full_hard_ce` 별도 계산 |
| wall-clock, EVD rate, epsilon dynamics logging 부재 | `metrics.csv`, `factor_diagnostics.csv`, profiling counters 추가 |
| baseline/ablation 실행 경로 분리 | YAML에서 optimizer mode만 변경하여 동일 training code 사용 |
| 기존 test suite가 수정된 API를 검증하지 못함 | FOAM-specific unit/integration tests 신설 |

## 3. Canonical controller

지원되는 mode는 다음과 같다.

- `stale`: 고정 check cadence마다 full EVD. Proxy 계산 없음.
- `foam`: proxy 기반 damping 조절, 필요한 damping이 `epsilon_max`를 넘으면
  EVD refresh.
- `foam_no_adaptive_epsilon`: damping은 `epsilon_0`로 고정하고 proxy threshold
  초과 시 EVD.
- `foam_no_evd_refresh`: adaptive damping만 사용하며 `epsilon_max`에서 cap.
- `dr_shampoo`: stale basis의 diagonalization residual로 EVD 여부 결정.

FOAM proxy는 stale basis `Q`, stale eigenvalues `D`, current factor `L_t`에
대해 다음을 계산한다.

```text
RC = ||(D + epsilon I)^(-1/2) (Q^T L_t Q - D)
      (D + epsilon I)^(-1/2)||_F
alpha = ||(D + epsilon I)^(-1/p)||_2 /
        ||(D + epsilon I)^(-1/p)||_F
h = RC * alpha / p
```

Damping update는 다음과 같다.

```text
epsilon_next = max(epsilon_0, epsilon_current * h / tau)
```

`foam` mode에서 `epsilon_next > epsilon_max`이면 fresh EVD를 수행하고
`epsilon_0`로 reset한다.

## 4. 공정한 wall-clock 측정

재구성 코드는 두 clock을 분리한다.

- `train_compute_seconds_cumulative`: synchronized training loop만 누적한다.
  Full-train evaluation, validation, checkpoint writing은 제외한다.
- `end_to_end_wall_clock_seconds`: evaluation과 logging을 포함한 실제 elapsed
  loop time이다.

논문의 optimizer wall-clock comparison을 재구성할 때는 첫 번째 값을
기본 x-axis로 사용한다. Clean full-train loss를 매 epoch 측정하더라도 그
비용이 특정 optimizer의 update cost로 오인되지 않도록 하기 위함이다.

## 5. 생성되는 실험 산출물

각 run directory에는 다음이 생성된다.

- exact resolved YAML;
- environment/run manifest;
- epoch-level metric CSV;
- factor-level epsilon/proxy/EVD CSV;
- rank-local optimizer/RNG/controller checkpoints;
- best model;
- optional factor snapshots;
- final JSON summary.

## 6. 재현 설정

논문의 ViT 설정을 반영한 기본값은 다음과 같다.

- ViT-S/16 계열 custom implementation;
- 90 epochs;
- per-device batch 256, 4 GPU에서 global batch 1024;
- beta `(0.95, 0.995)`;
- weight decay `4.2e-4`;
- 5% warmup + cosine decay;
- Adam grafting;
- inverse-root order `p=2`;
- `max_preconditioner_dim=1024`;
- merge dimensions enabled;
- base damping `1e-9`;
- main FOAM config: `f=20`, `tau=0.75`, `epsilon_max=3e-7`.

## 7. 검증 결과

재구성 환경에서 수행한 검증:

- 전체 Python source static compilation;
- 9개 pytest controller/integration test;
- canonical proxy numerical equality test;
- stale mode proxy-call zero test;
- damping floor test;
- scalar inverse-root test;
- optimizer checkpoint round-trip test;
- CPU synthetic ViT end-to-end training 및 output contract test;
- 2-rank Gloo DDP smoke test;
- epoch-boundary checkpoint/resume smoke test.

전체 90-epoch ImageNet-1K, 4×A6000 실험은 현재 환경에서 실행하지 않았다.
따라서 제공 artifact는 code correctness와 small-scale execution까지
검증되었으며, 논문 표의 수치 자체를 재측정한 결과물은 아니다.

## 8. Source와 reconstruction의 경계

업로드 archive에는 Figure 4 ablation의 exact implementation, full-batch
training-loss 계산 routine, SOAP source가 포함되어 있지 않았다. 따라서
해당 부분은 논문 설명과 공개 코드의 interface를 보존하여 재구성했으며,
원 저자의 비공개 실험 코드와 완전히 동일하다고 주장하지 않는다. 상세한
검증 범위는 `VERIFICATION_REPORT.md`에 기록했다.

## 9. 권장 실행 순서

```bash
make test
bash scripts/run_smoke_tests.sh
python tools/run_sweep.py --sweep configs/sweeps/vit_figure1.yaml --dry-run
bash scripts/run_vit_4gpu.sh configs/vit/paper/foam_f20_tau075_epsmax3e-7.yaml
```
