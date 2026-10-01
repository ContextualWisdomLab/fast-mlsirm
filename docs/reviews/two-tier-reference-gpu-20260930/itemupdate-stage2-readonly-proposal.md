# 단계2 고정 counts item 갱신 검증 설계

## Motivation와 범위

기준 checkout은 `90ca45bf7730208e90db7124974de762c20be029`, 단계1 test source는 `3ad7d0dd0639b7239340099e87bb1723399bd062`다. production 계산 소스는 4c 이후 바뀌지 않았다. 이번 단계는 기존 함수·합성 fixture·정지/반환 계약을 읽고 검증 계획을 정리하는 작업이다. 새 test·production 코드·fit·M-step·GPU 실행은 하지 않았다.

단계1은 고정 모수 counts/cross/likelihood의 정합만 검증했다. CPU streamed는 `item_cat_logprob→grm_logprobs`, cached는 `item_primary_base`와 node별 직접 `grm_logprobs`를 사용한다. 공통 primitive 때문에 두 독립 GRM 수학 oracle는 아니다. GPU word product 이후의 Rust 통계 누적도 구분한다. 이 한계와 단계1 독립 보고서의 source/log 검토 결론을 단계2 갱신 수용으로 확대하지 않는다.

## 현재 계산 계약

근거는 `crates/mlsirm-core/src/two_tier_grm.rs:1606–1764,1916–2074`, `crates/mlsirm-core/src/poly.rs:118–145,393–418`이다.

| 구간 | 실제 현재 동작 |
|---|---|
| 모수 packing | 자유 primary 차원의 slope, optional specific slope, K−1 threshold 순서. block counts node는 `g*H+h`, specific-free는 `g` |
| objective/gradient | 현재 좌표의 base와 기존 GRM log probability로 `f=-Σ r logP` 계산. `grm_node_gradient`의 base/threshold derivative를 자유 slope에 chain rule 적용. item objective에 ridge penalty는 넣지 않음 |
| 첫 정지 | `f0` 또는 gradient norm이 nonfinite이거나 norm<1e-9면 그 Newton loop를 break |
| FD Hessian | analytic gradient의 forward 차분, 현재 고정 h=1e-5. source의 in-place 평균화 순서를 적용한 뒤 diagonal에 caller ridge 추가 |
| 선형 solve | partial-pivot Gaussian elimination. pivot<1e-12면 당시의 g를 반환. 중간 elimination이 진행된 뒤의 반환을 항상 원래 gradient라고 가정하지 않음 |
| 방향·크기 | nonfinite step 또는 g·step≤0이면 원래 gradient 사용. 최대 성분이 2보다 크면 현재 방식으로 scale |
| backtracking | candidate=`params−alpha*step`, alpha=1에서 최대25번. finite objective와 현재 Armijo 조건 `candidate_f≤f0−1e-4*alpha*directional`에 맞을 때만 채택, 아니면 alpha 절반 |
| 종료/반환 | 채택 실패 또는 alpha*max_step<1e-9면 break. 반환은 `Vec<f64>`뿐이며 per-item 성공·실패·수렴 상태가 없음 |

위 상수와 caller `ridge/newton_iter`를 바꾸지 않는다. source의 in-place Hessian 평균화에는 순서가 있으므로, 현재 solve 입력과 별도로 복사해 만든 symmetric part를 구분해야 한다. 기존 item13 진단도 현재 행렬 순서를 먼저 재현한 뒤 복사본의 symmetric part만 eigenvalue로 보고한다(`tests/unit/two_tier_grm_tests.rs:1520–1557`). 그 진단을 실제 행렬의 대칭성·SPD·전체 solver condition number나 식별 증명으로 쓰지 않는다. 지금 이 동작을 수정하거나 새 확정 결함으로 보고하지 않는다.

## 반환값과 EM 수렴 표기

`run_single_start`는 E-step observed LL을 먼저 검사하고 상대 변화가 `tol*(1+|previous|)` 이하이면 `tolerance_met`로 끝낸다. 허용된 감소 범위는 현재 `checked_em_loglik_change`의 `32*EPSILON*(1+|previous|)`다. iteration cap과 all-start failure도 기존 그대로다.

item 갱신 함수가 같은 모수 Vec를 반환한 이유는 gradient 정지, nonfinite, step/backtracking 실패 등일 수 있다. caller는 별도 item 상태 없이 다음 E-step으로 진행한다. **변경 없는 Vec 또는 작은 다음 LL 변화만으로 해당 item 갱신이 성공했다고 기록하지 않는다.** 이후 상태가 수렴 조건에 들어갈 수 있다는 source-level 가능성과 실제 전체 fit 재현은 별개다. 이번 계획은 fit을 실행하지 않고 이 경계를 구분한다.

`grm_node_gradient`는 zero count 항을 분리하지만 item objective는 원래 `count*log_probability` 합을 사용한다. legal logP=-inf와 zero count가 만나는 입력, nonfinite objective와 반환값 보존은 반드시 별도 negative case로 기록한다. 논리적 zero contribution을 사용하는 독립 oracle를 production 처리로 몰래 적용하거나 probability floor를 추가하지 않는다.

## Proposal: 작은 fixture와 oracle를 분리

후속 실행 배정 전까지 아래는 설계뿐이다.

### A. 고정 counts의 출처

단계1의 같은 입력·모수·GH/prior hash로 작은 q5/q7 통계를 재생성하는 기존 helper를 재사용한다. count 배열은 반환·archive된 것으로 가정하지 않는다. cached/streamed CPU와 word-derived counts 중 어느 것을 입력했는지 item/node/category layout, input hash와 함께 기록한다.

일반 block·cross-loading block·specific-free·P1/S0에서 대표 item만 고른다. 공개 calibration의 category coverage 요구와 standalone fixed-count 평가를 구분한다. 261회 fit을 다시 실행하거나 q121 counts를 만들지 않는다. 기존 sparse fixture item13/state102 검증은 참조 증거이며 새 실행으로 표시하지 않는다.

### B. objective와 analytic gradient

먼저 packed shape·counts 길이·finite/nonnegative 값·threshold ordering이 원래 계약에 맞는지 확인한다. 중간 범위의 작은 fixture에서는 기존 `grm_logprobs/grm_node_gradient`를 호출하지 않는 직접 cumulative-logistic category probability로 별도 Q objective를 설계한다. 이는 제한된 테스트 oracle이지 새 runtime 구현이 아니다.

양수 count의 logP 항과 zero count 항을 구분해 수학적 기대값을 기록한다. 확률을 clip/floor하거나 임의로 node를 버리지 않는다. extreme tail/zero probability 사례를 중간 범위의 양호한 gradient FD 사례와 섞지 않는다.

기존 GRM unit(`tests/unit/grm_tests.rs:127–173`)의 central objective FD, eps=1e-6 및 1e-4 비교 조건을 참조한다. production의 forward-gradient h=1e-5와 같은 것으로 표시하지 않는다. forward/central 오차·actual residual을 별도로 남기며 기존 G1 strict 조건이나 fit criterion을 완화하지 않는다. threshold 차분이 유효한 순서를 벗어나면 그것도 별도 거부/정의역 결과로 기록한다.

### C. 고정 counts의 item 반환

같은 packed 모수·free dimension 순서·좌표·ridge·newton_iter에서 CPU counts와 word-derived counts를 각각 기존 `m_step_item`에 넣는 parity를 계획한다. 동일 함수에 같은 입력을 넣는 parity만으로 analytic gradient나 Newton의 수학 정합을 입증하지 않는다.

독립 objective, 현재 gradient, 실제 현재 FD matrix/ridge/solve 입력, 방향·scale·backtracking 후보와 최종 Vec의 finite 여부·변화량을 구분한다. 필요한 세부 관측이 기존 반환에 없으면 이후 승인된 test-only 계측으로만 기록하고, 지금 helper·public API를 바꾸지 않는다. successful update, unchanged return, rejected candidate는 각각 다른 결과다.

### D. negative/정지 사례

- 전체 zero counts와 양호한 finite logP: 현재 gradient 정지·변경 없는 반환을 구분한다.
- legal zero probability와 해당 zero count: `0*logP` objective·gradient·returned Vec를 각각 기록한다. 현재 source-level 위험이며 아직 단계2 실행으로 재현하지 않았다.
- 아주 좁지만 strictly ordered threshold: 차분 정의역·nonfinite 후보·backtracking 거부를 구분한다.
- nonfinite objective/gradient와 finite 입력의 overflow: 초기 break인지 candidate 거부인지 확인한다. 독립 oracle 결과를 production 값으로 대체하지 않는다.
- flat/near-singular counts, 첫/뒤 pivot의 singular helper 사례, negative direction/step scale: 현재 helper의 실제 반환과 outer safeguard를 구분한다.
- backtracking 채택 실패 또는 미소 step: 변경 없음이 수렴/성공 상태는 아니다.

raw branch 관측이 없으면 특정 branch가 실제 실행됐다고 추정하지 않는다. 작은 독립 fixture가 필요한 branch를 만들지 못하면 미확인으로 남긴다. malformed private-array shape를 그대로 실행해 panic을 유도하지 않고 원래 validation 계약을 사용한다.

## 기록과 실행 선행조건

실제 실행을 배정받으면 exact base/head/test diff, production byte 불변, item별 counts·packed 입력·GH/prior hash, actual adapter/word 출처, CPU/GPU별 objective·gradient·return, FD residual, 거부·unchanged 결과와 원로그/binary hash를 결속한다. fit이 없으면 수렴은 해당 없음으로 기록한다.

한 번에 작은 local item 단위만 순차 실행하고 caller 예산·RAM 압력·소유 프로세스를 먼저 확인한다. 큰 CPU fit·q121 counts·340인·bootstrap·remote dispatch는 하지 않는다. 현재 API HOLD, 단일 writer, 기존 E·runner·권한 경계를 유지한다. 단계3 공개 fit과 단계4 q121 자원 실측은 별도 배정이다.

## Alternatives와 Non-goals

권장은 기존 expected counts와 CPU item helper를 그대로 두고, cache/word parity와 독립 objective FD·정지 결과를 분리하는 것이다. GPU M-step, solver/Hessian 변경, prior/ridge/criteria 조정, probability floor·node cap·CPU fallback, 새 host-budget API는 이번 설계에 포함하지 않는다. 확정 결함이 실제 재현되면 자동 수정하지 않고 최소 수정 제안을 별도로 제출한다.

기존 expected complete-data 통계의 근거는 직접 읽은 Cai(2010), pp.608–609 Appendices A/B다. 이번에는 그 통계를 소비하는 현행 구현의 검증을 설계하며 새 추정 방법이나 연구 수치 수용을 주장하지 않는다.

참고문헌: Cai, L. (2010). A two-tier full-information item factor analysis model with applications. *Psychometrika, 75*(4), 581–612. https://doi.org/10.1007/s11336-010-9178-0.
