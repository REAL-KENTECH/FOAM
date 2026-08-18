# FOAM ViT 재구성 코드 검증 보고서

검증 일자: 2026-08-18

## 1. 검증 대상

업로드된 `FOAM-main(1).zip`을 기준으로 다음을 하나의 실행 경로로
재구성했다.

- ViT-S/16 계열 학습 코드;
- canonical Distributed Shampoo/FOAM controller;
- stale Shampoo, FOAM, DR-Shampoo 및 두 ablation mode;
- AdamW matched baseline;
- clean full-train metric, wall-clock, epsilon, proxy, EVD/reuse logging;
- rank-local optimizer/controller checkpoint 및 resume;
- paper/smoke/profiling/sweep configuration;
- optional external SOAP adapter.

## 2. Source-derived와 reconstructed의 경계

### 업로드 소스에서 직접 확인된 부분

- custom ViT architecture와 기본 차원;
- 90 epochs, per-device batch 256, 4 GPU global batch 1024;
- beta `(0.95, 0.995)`, weight decay `4.2e-4`;
- Mixup 0.2, label smoothing 0.1, RandAugment policy;
- Adam grafting, inverse-root order `p=2`;
- `max_preconditioner_dim=1024`, precondition frequency 및 damping 계열 값;
- FOAM proxy `h = RC * alpha / p`와 multiplicative damping controller;
- stale eigenspace reuse와 epsilon-max refresh logic.

### 재구성한 부분

- source에 없던 full experiment harness, output contract, plots, sweeps;
- deterministic hard-label full-train CE 정의;
- DR-Shampoo와 ablation의 standalone mode semantics;
- factor profiling/snapshot pipeline;
- checkpoint metadata, RNG capture, same-world-size resume;
- external SOAP adapter interface.

특히 논문 Figure 4의 ablation에 사용된 exact internal source와 paper의
`full-batch train loss` 계산 코드가 업로드 archive에는 없었다. 따라서
`foam_no_adaptive_epsilon`, `foam_no_evd_refresh`, `train_full_hard_ce`는
논문의 설명과 공개 코드 구조를 일관되게 결합한 재구성 정의이며, 원 저자의
비공개 실행 코드와 byte-for-byte 동일하다고 주장하지 않는다.

## 3. 자동 검증 결과

### Static compilation

```text
python -m compileall -q foam_experiments optimizers tools vit.py submission.py
PASS
```

### Unit/integration tests

```text
python -m pytest -q
9 passed
```

검증 항목:

1. scalar inverse-root API가 항상 4-tuple을 반환하는지;
2. FOAM proxy가 `1/p`를 포함한 논문 식과 수치적으로 일치하는지;
3. stale mode에서 proxy가 한 번도 호출되지 않는지;
4. adaptive epsilon이 base damping 아래로 내려가지 않는지;
5. epsilon cap 초과 시 fresh EVD와 damping reset이 발생하는지;
6. eigenspace/eigenvalue/epsilon/counter가 optimizer checkpoint에 보존되는지;
7. synthetic ViT training이 metric, diagnostics, checkpoint를 생성하는지;
8. complete CPU experiment output contract가 유지되는지.

### 모든 내부 optimizer mode CPU smoke test

다음 mode가 동일 training entry point에서 성공적으로 실행되었다.

```text
foam
stale_shampoo
foam_no_adaptive_epsilon
foam_no_evd_refresh
dr_shampoo
adamw
```

### 2-rank CPU DDP smoke test

```text
torchrun --standalone --nproc-per-node=2 ... --cpu
PASS
world_size = 2
global_step = 4
rank-local optimizer checkpoints = 2
```

Gloo backend에서 distributed data sampling, DDP model, distributed Shampoo
state allocation, diagnostics gather, rank-local checkpoint가 정상 작동했다.

### Checkpoint/resume equivalence test

`stop_after_epoch=1`로 2-epoch schedule을 1 epoch에서 중단한 뒤 같은
checkpoint에서 재개하여 uninterrupted 2-epoch run과 비교했다. 최종 결과:

```text
model tensor states:     24개, max_abs_diff = 0.0
optimizer tensor states: 661개, max_abs_diff = 0.0
global_step:             8
PASS
```

즉 planned total step과 LR schedule을 유지하는 epoch-boundary resume에서는
model뿐 아니라 `Q`, `D`, adaptive epsilon, controller counter를 포함한
optimizer tensor state가 정확히 일치했다. Mid-epoch replay는 구현하지 않았다.

### Config validation

paper, ablation, profiling, smoke YAML 15개를 world size 4 조건으로
검증했다. Figure 1, LR selection, Table 3, ablation sweep definition도 dry-run
command generation이 가능하다.

### Factor snapshot/profiling pipeline

실제 synthetic training factor snapshot을 export한 뒤 동일 factor에서 FOAM
proxy와 direct inverse-root EVD를 비교하는 `tools/profile_snapshots.py`를
실행했다. CPU의 작은 32/64 차원에서는 kernel overhead 때문에 EVD가 더
빠른 경우도 있었으나, snapshot loading, 정확한 proxy 식, EVD timing, CSV
출력 경로가 정상 동작함을 확인했다. 이 smoke 결과를 논문의 GPU 대규모
profiling 수치로 해석해서는 안 된다.

### Model contract

paper custom ViT default configuration의 parameter count:

```text
22,050,664
```

업로드된 ViT implementation에서 확인한 수치와 일치한다.

## 4. 수동 확인한 핵심 invariant

- `stale_shampoo`는 check마다 fresh EVD를 수행하며 `proxy_calls=0`이다.
- `foam`은
  `epsilon_next=max(epsilon_0, epsilon_current*h/tau)`를 사용한다.
- `epsilon_next > epsilon_max`이면 fresh EVD 후 `epsilon_0`로 reset한다.
- controller state는 optimizer state tensor로 저장되므로 resume 후
  `Q`, `D`, `epsilon_t`, EVD/reuse counters가 소실되지 않는다.
- factor diagnostics는 matrix-valued block의 L/R rate와 vector block을
  구분한다.
- profiling mode는 정확한 `h(epsilon)` 계산과 direct inverse-root EVD를
  동일 snapshot에서 비교한다.

## 5. 실행하지 못한 검증

다음은 현재 환경의 자원 또는 source 부재 때문에 수행하지 않았다.

- 4×A6000, ImageNet-1K, 90-epoch full paper run;
- 논문 Table/Figure의 최종 수치 재측정;
- NVIDIA RTX Pro 6000/B200 환경 profiling;
- external SOAP implementation을 사용한 baseline run;
- AlgoPerf package 내부에서의 end-to-end submission execution.

따라서 본 artifact는 code correctness, controller invariants, checkpointing,
CPU/DDP small-scale execution까지 검증되었다. 논문에 보고된 대규모 수치
자체의 재현 성공을 의미하지는 않는다.
