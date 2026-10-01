# 작은 고정 모수 counts 계약 검증

## 실행 범위

단계1 test-only source는 `3ad7d0dd0639b7239340099e87bb1723399bd062`, parent는 읽기 전용 제안 commit `73ff986171bbc142bf3ef5e86e7a44df1365ceef`다. 변경은 `tests/unit/two_tier_grm_tests.rs` 한 파일뿐이며 production 계산·binding·GPU shader는 parent와 byte가 같다.

기존 owned `reference_gpu_fixed_bank_kernel_profile` 식별자에 명시적 `G1_KERNEL_COUNTS_CONTRACT=1` 로컬 모드를 추가했다. 새 ignored test나 소유 없는 CI 분기를 만들지 않았다. 값이 없거나 0이면 기존 1인 moments-only 검사를 수행하며 19개 timing key 계약을 유지한다. counts 모드가 실행됐다는 판단은 실제 `counts_case`/`counts_rejections` 이벤트로 한다. 테스트 1 passed만 보고 moments와 counts 증거를 서로 대체하지 않는다.

같은 clean source의 q5/5와 q7/7 counts 모드, q5/5 기본 모드를 순차 실행했다. 각각 1 passed / 0 failed / 0 ignored였고 fit·M-step은 실행하지 않았다. compiler는 실제 로컬 rustc/cargo 1.98.1 Homebrew / aarch64-apple-darwin / LLVM 22.1.8이다. CI Rust 1.97.1이나 독립 재빌드 결과로 표시하지 않는다.

## 실제 fixed-state 대조

각 q에서 16문항·4범주·4인 합성 state를 세 가지로 검사했다.

1. P2/S4 고정 known bank, 모든 응답 관측.
2. P2/S4의 specific-free 문항·음수 slope, 일부 cell/전체 block/전체 person missing. masked `usize::MAX` sentinel가 범주 또는 counts index가 되지 않는지도 확인했다.
3. P1/S0 primary-only와 전체 person missing.

같은 모수·GH 좌표/prior에서 CPU streamed와 CPU cached 결과를 대조했다. GPU는 기존 reference word log-product를 실행하고 Rust가 정규화·counts/cross를 누적한다. GPU helper가 같은 CPU 함수로 posterior를 대체하도록 바꾸지 않았다. source의 shared f64 likelihood certification도 그대로다.

모든 item/node/category counts와 cross·likelihood의 bits가 일치했다. 작은 예산으로 GPU batch 3인+마지막 1인을 강제했고 full-budget 결과와 같았다. 다음 고정 state는 threshold 하나를 명시적으로 0.125 바꾼 별도 점이며, 기존 counts와 달라졌고 새로운 CPU oracle와 같았다. Newton 또는 추정 업데이트가 아니다. input·다음 state·GH/prior hash를 원로그에 각각 기록했다.

| 실제 관측 | q5/5 | q7/7 |
|---|---:|---:|
| positive shape 사례 | 3 | 3 |
| counts 최대 절대차 | 0 | 0 |
| cross/likelihood bit 대조 | 같음 | 같음 |
| 관측 category mass 최대 잔차 | 3.552713678800501e-15 | 4.662936703425658e-15 |
| 실제 거부 사례 | 7 | 7 |

mass 잔차를 기록하면서 기존 strict 조건을 완화하지 않았다. 이 작은 node 검사는 연구 q121 적분·fit·recovery 수용을 대신하지 않는다.

## 거부·예산

각 q에서 관측 범주 범위, response shape, mask shape, primary-map shape, specific-map 범위의 validation 5건이 실제 거부됐다. 한 사람의 최소 buffer 정책보다 1 byte 작은 budget도 거부됐다(q5: 72947 bytes, q7: 199107 bytes).

전체 mass 소실 검사는 finite threshold `[1e308,0,-1e308]`와 category 0 응답을 사용했다. 원래 f64 log-product의 유한 덧셈 overflow로 모든 posterior node의 likelihood가 소실되는 작은 입력이다. CPU oracle likelihood는 nonfinite였고 GPU 경로는 `nonfinite GPU-derived f64 likelihood`로 실패했다. floor·node 변경·CPU fallback을 넣지 않았으며 실패 뒤 다음 정상 사례도 실행됐다. 이는 단계2의 zero-count×legal-zero-probability objective 검증이 아니다. gradient/Newton은 실행하지 않았다.

caller GPU budget은 1048576 bytes(1MiB), host budget은 67108864 bytes(64MiB)다. 실제 short-batch budget은 q5의 세 사례에서 89876/83468/3292 bytes, q7에서 244708/225884/4444 bytes다. host source estimate는 q5 1066384/979984/103824 bytes, q7 2802576/2548560/118160 bytes다. source-shaped payload/header 정책이며 실제 peak RSS/VRAM 또는 driver staging 상한이 아니다. GH·table·counts 할당 전에 가장 큰 case의 caller host budget을 검사한다.

## 기존 경로와 컴파일 확인

같은 source의 q5 기본 moments 모드에서도 기존 input hash `5f211b3d5872eeda9e610fb721f633a03fcff7ed787ddf4c7bdd9526a6d9047e`, 1인/no-count/no-update, 19개 unique timing key, malformed receipt 거부 5건, likelihood·moment·SD 차이 0과 실제 1 pass를 확인했다.

CPU-only 및 coverage CPU-only lib/tests check와 Python 실행/소유/toolchain 계약 4파일의 37 passed도 확인했다. 기존 dead-code warning은 남아 있다. Python 전체 suite나 fresh required CI 결과가 아니다.

## 원증거와 미수용

[review packet](review-packet.json)은 exact source·parent·test-only diff·binary·input/GH/prior·원로그 hash와 scalar 결과를 결속한다. preliminary dirty-source 로그와 최종 clean-source 로그는 구분하며, 전자를 clean-head 수용으로 표시하지 않는다. 원로그는 byte 그대로 보존한다.

- [clean q5 counts](q5-counts.txt)
- [clean q7 counts](q7-counts.txt)
- [clean q5 기본 moments](q5-moments.txt)
- CPU/coverage·37 계약 결과가 함께 들어 있는 전체 순차 출력은 세션 scratchpad의 `tasks/bi3efs9cs.output`에 보존했다. 공개 원증거에 복제할 때도 실제 읽은 구간과 범위를 구분한다.

이번 작업은 로컬 Apple M1 / Metal의 작은 fixed-state 통계 검사다. q121 counts·item M-step·공개 fit·340인·bootstrap·큰 CPU fit·remote GPU dispatch는 실행하지 않았다. fit이 없으므로 수렴 판정도 해당하지 않는다. production 수정이 필요한 확정 결함을 주장하지 않으며, 독립 읽기 검토·정식 승인·필수 CI·불변 릴리스·연구 수치 수용은 별도다.

API HOLD 동안 이 commit과 문서는 로컬에만 보존한다. 보호 E·runner·권한 설정을 바꾸지 않았고 참가자 자료·원고 수치·IRB 값은 다루지 않았다. 기존 Cai(2010), 직접 읽은 pp.608–609 Appendices A/B의 expected-count 계약을 검증했으며 새 추정기나 criterion을 넣지 않았다.
