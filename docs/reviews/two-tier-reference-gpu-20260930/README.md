# G1 실제 GPU 실행과 미수용 결과

> 이 문서는 초기 `d5bc3af13`/`c07f80bb8` f32 실패의 역사적 기록이다. 후속 [word 정밀도 및 zero-mass RED→GREEN 기록](word-precision/README.md)은 별도 commit·source·실행 증거로 분리한다. 작은 합성 회귀의 해소를 q121/격리 s1·정식 릴리스 수용으로 확대하지 않는다.

## 판정

**코드·장치 계약 구현은 완료했지만 G1 수치 동등성은 아직 수용하지 않는다.**
동일 합성 입력의 q7/7, Φ=I 적합에서 GPU가 실제 실행되고 CPU와 같은 수렴 횟수를 기록했다. 작은 fixture는 모수 대조도 통과했다. 그러나 64명×16문항 P2/S4 합성 입력의 엄격한 수렴 대조에서는 최대 문항 모수 차이가 0.321로 커졌다. likelihood·EAP가 가까워도 이를 모수 동등성으로 대신하지 않는다. q121/121 대조와 새 head의 격리 s1 실행도 아직 완료하지 않았다.

연구 수치 HOLD를 유지한다. 이 기록에는 합성 결과만 있으며 참가자 자료·개인별 결과·IRB 값은 없다.

## 코드와 빌드

- 코드 commit: `d5bc3af13464ea30990cea94b4d47cb107d160ad`
- PR: [fast-mlsirm#2296](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/2296), #2246 head 기반의 별도 PR
- Python: 3.12.14, macOS arm64
- Python core SHA-256: `be4c05fcdac4ab930e879785f8bd6843aae49ae464aa1436b9e29b582aeca8d3`
- 빌드: `VIRTUAL_ENV="$PWD/.venv" .venv/bin/maturin develop --uv --profile dev -j 1 --locked`
- 실제 GPU: `Apple M1`, wgpu backend `Metal`. 위 시간은 dev 빌드의 로컬 측정이며 다른 하드웨어나 연구 규모의 성능을 보장하지 않는다.

## 합성 대조

| 입력·설정 | CPU | GPU | 판정 |
|---|---:|---:|---|
| tiny, q7/7, Φ=I, tol=1e-4 | 21회 수렴, 0.493초 | 21회 수렴, 0.777초 | 최대 모수 차이 7.83e-6, loglik 차이 7.13e-9; 회귀 검사 통과 |
| 64×16, P2/S4, q7/7, Φ=I, tol=1e-6 | 261회 수렴, 133.50초 | 261회 수렴, 73.94초 | 최대 모수 차이 0.321; **미수용** |

두 번째 입력은 기존 연속 Gaussian 합성 생성기를 사용하며 seed=20260930, n_starts=1, max_iter=1000이다. GPU buffer 예산은 268435456 bytes다.

- 응답 SHA-256: `d829e8fd51d22eb73da4f25f73cbe828f0386ab0c23e1c0e53f5e8d4ac589ee8`
- 배열 shape·응답·지도·설정을 포함한 입력 SHA-256: `5797fcdd50ac7f24d55fda5bd0c85728be0693382c6a7954549749f8013b20b3`
- 모수 최대 차이: primary slope 0.213259, specific slope 0.321231, threshold 0.246188, Φ 0
- loglik 차이: −2.4996e-7; AIC/BIC 차이 약 +4.9992e-7
- EAP 최대 차이: 1.6724e-6; posterior SD 최대 차이: 2.8151e-5
- benchmark exit=0은 두 적합의 수렴과 실행 완료만 뜻한다. 모수 동등성 수용 판정은 아니다.
- 전체 결과: [synthetic-q7.json](synthetic-q7.json)

최악 문항은 0-based item 13이다. CPU/GPU의 specific slope는 82.8062/83.1274, primary slope는 14.9765/15.1898, 첫 threshold는 18.3055/18.5517이었다. 두 적합은 같은 부호이고 Φ=I·latent SD=1을 유지한다.

같은 seed·입력·제약·prior·q7 가중치에서 반복 cap만 바꿔 궤적을 재실행했다. 초기 likelihood 차이는 매번 정확히 0이었다. 최초 갱신의 모수 차이는 약1.5e-7, 50회까지 약1.1e-6이었다. 100회에서 specific slope 차이가 3.76e-4, 200회에서 1.06까지 커졌다. GPU와 CPU의 iteration 순서나 부호 반전으로 설명되는 차이는 아니었다. 261회에서 같은 node의 item13 category 확률 최대 차이는 독립적인 합성 진단 oracle에서 1.74e-7이었다. 이는 극단 모수 주변의 평탄한 방향과 정밀도 민감성을 뒷받침하지만 Hessian 조건이나 식별 실패를 확정하는 증명은 아니다.

[RCA 궤적 로그](trajectory-rca.txt)와 [재현 스크립트](trajectory-rca.py)를 보존한다. 엄격한 문항 모수 대조를 기존 하드웨어 gate에 추가해 실제 **1 failed / exit101**을 재현했다([strict-parity-failure.txt](strict-parity-failure.txt)). 작은 fixture만 통과해 이 실패를 가리지 않는다. 다음 대조는 동일 모수 상태에서 f64 E-step, f32 table만 양자화한 E-step, 실제 GPU E-step의 기대 빈도와 M-step을 분리하는 것이다. 기준 완화·새 식별 제약·모형 변경·CPU fallback은 하지 않는다.

## 검증 로그

- [rust-hardware.txt](rust-hardware.txt): 명시적 device/예산 계약과 실제 GPU 기준 적합 2 passed; provenance merge 1 passed. 처음 작성한 CPU metadata fixture가 9번째 반복에서 기존 monotonicity 검사에 실패한 기록도 보존했다. 해당 metadata 검사는 수렴 증거가 아닌 1회 갱신 검증으로 바꿨으며 [build.txt](build.txt)에서 통과를 확인했다. 기존 monotonicity 검사는 바꾸지 않았다.
- [binding-gpu.txt](binding-gpu.txt): 다집단 bifactor 실제 adapter 기록·기존 focal 적률 대조 2 passed
- [group-pipeline.txt](group-pipeline.txt): 기준 적합부터 초점 적합·채점까지 실제 GPU를 요구하는 파이프라인 1 passed
- 기존 CPU golden·device 경계: 9 passed. 첫 명령의 `-k`로 GPU 두 검사가 deselect돼서, GPU 검사는 별도 정확한 명령으로 다시 실행했다.

## 재현

```bash
cargo test -p mlsirm-core --lib reference_fit_ -j 1 -- --include-ignored --nocapture
FOCAL_GPU_NATIVE=1 .venv/bin/python -m pytest -q -s tests/test_bifactor_gpu.py::test_bifactor_gpu_equivalence_multigroup tests/test_two_tier_focal_gpu_native.py::test_actual_gpu_six_latent_score_and_focal_update
FOCAL_GPU_NATIVE=1 .venv/bin/python -m pytest -q -s tests/test_two_tier_group_pipeline_native.py::test_group_pipeline_preserves_slots_priors_and_nonconvergence
.venv/bin/python scripts/benchmark_two_tier_reference_gpu.py --persons 64 --q-primary 7 --q-specific 7 --max-iter 1000 --tol 1e-6 --n-starts 1 --seed 20260930 --gpu-memory-budget-bytes 268435456 --device both --out synthetic-q7.json
```

q121/121에서는 P2/S4 product grid와 문항 빈도·확률 표의 host 메모리도 커진다. GPU byte 예산은 host 메모리나 물리 VRAM 가용량을 보장하지 않는다. 실행 전 실제 여유와 기존 계산 소유를 확인해야 한다.

## s1과 기존 게이트

#2246의 지정된 rust/python(3.12·3.14·집계)/gpu-smoke/focal-gpu-native 및 strix가 pass였다. `focal-gpu-native`는 43분28초였고, job 109691255755 로그에서 Python hardware 6 passed, s1 bit 고정 1 passed, bifactor parity 11 passed를 확인했다. 이는 #2296 새 head의 통과 증거가 아니다.

s1 PID3974018(user seongho, rehearsal_pipeline.py)은 읽기 전용 확인 때 CPU 약99.6%, elapsed 7시간22분으로 실행 중이었다. 출력 로그의 내용이나 참가자 입력은 열지 않았다. 프로세스를 종료·전환하지 않았다. py-spy native stack 조회는 권한 거부로 실패했으며 sudo나 다른 세션 우회는 하지 않았다. org runner 조회 때 cwlab-s1-gpu-01은 online/busy=false였지만 향후 슬롯 예약이나 새 검증 완료를 뜻하지 않는다.

## 의무와 일정

최종 목표는 late-life 원고 완성과 이슈·PR 해소다. G1→G2→G3 정밀도 검증→불변 릴리스·소비자 수용의 순서를 유지한다. 기존 Coordinator와 late-life Lead에 범위·실패·자원 상태를 공유했다. #251 소비자 수용 검사는 orchestration-lead-latelife-v3-47이 맡으며 이 PR과 중복 구현하지 않는다. #2290은 G1 후속으로 소유 확인을 유지한다.

2026-09-30 KST 기준 코드·native 빌드·작은 GPU 대조는 12:30 전후에 완료하고 PR은 12:38까지 생성했다. 13:30 대조 목표는 **RCA 진행 및 근거 보고**로 갱신한다. 모수 미수용과 q121 검증을 날짜만으로 완료 처리하지 않는다.

Goal cron `facfc906`은 매시17분, 진술서 cron `99b7e2fe`는 매시43분이다. 둘 다 이 세션에만 있고 7일 뒤 만료되며 별도 permalink는 없다. 공개 협업 기준점은 [late-life#258의 G1–G4 계획](https://github.com/ContextualWisdomLab/late-life-anxiety-reanalysis/issues/258#issuecomment-5901989261)이다.

## 논문 근거

Cai, L. (2010). A two-tier full-information item factor analysis model with applications. *Psychometrika, 75*(4), 581–612. https://doi.org/10.1007/s11336-010-9178-0 — pp.608–609 Appendices A/B를 직접 읽고 기존 posterior 기대 빈도·적률을 재사용했다. PDF 재배포 권한은 확인하지 않아 첨부하지 않았다. WGSL 가중치는 f32이며 f64 GPU 계산이라고 주장하지 않는다.
