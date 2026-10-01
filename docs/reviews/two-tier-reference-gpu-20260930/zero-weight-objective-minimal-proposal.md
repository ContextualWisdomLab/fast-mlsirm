# Zero-weight objective 경계 최소 수정 제안

## 실제 재현과 범위

production 소스는 `4c09920d0e08f5fcbb8f68cab5ceb140d4c7af1b` 이후 바뀌지 않았다. 단계2 preliminary test는 base `0b561646a2d8ee0f4a3d54d5537b2b04082744fa`의 dirty test-only source SHA-256 `4f5db8604aee43ead67c2c944965e36846b09607ff0536a490181f8878124c30`에서 실행했다. 이것을 clean commit 실행으로 표시하지 않는다.

finite packed 모수 `[1.0,1e-18,0.0]`, 자유 primary `[0]`, primary 좌표 `[1.0]`, specific 없음, node 한 개, counts `[[0,0,0]]`를 기존 item helper에 넣었다. strictly decreasing threshold라도 f64의 `base+1e-18`과 `base+0`이 같아져 middle category log probability가 legal `-inf`가 된다.

- 기존 `item_neg_ll_grad` objective는 실제 NaN이었다.
- 별도 직접 cumulative-logistic objective는 zero count 기여를 0으로 정의해 0이었다.
- 기존 analytic gradient는 `[-0,-0,-0]`, item helper는 입력 Vec 그대로 반환했다.
- fit은 실행하지 않았고 수렴·실제 원고·recovery 영향은 입증하지 않았다. 반환이 같다는 이유로 갱신 성공으로 세지 않는다.

원로그는 세션 scratchpad `item-return-q5-initial.txt` SHA-256 `5e2c2c0daee5bd661882897b31c5a7aafc4475f502c9db871c836a24ca5a70f6`, `item-return-q7-initial.txt` SHA-256 `dc5bbb3a4e836db22887cb434698bc3b63d7d996b5004484c32961f4de62acd6`다. 각 harness 자체는 observation을 기록해 통과했다. 이 기록을 failing regression이나 TDD RED로 꾸미지 않는다.

## 현재 소스 계약과 원인

`two_tier_grm.rs:1637–1639`의 item objective는 모든 category에 `count * log_probability`를 계산한다. IEEE 연산의 `0*(-inf)`는 NaN이다. 반면 `poly.rs:132–141`의 기존 gradient는 zero count 항을 명시적으로 0으로 처리한다. `m_step_item`은 objective 또는 gradient norm이 nonfinite면 초기 break를 하고 별도 status 없이 Vec를 반환한다(`two_tier_grm.rs:1688–1691`).

이 문제는 GPU product의 정밀도나 새로운 모델식 변경이 아니다. CPU item objective의 zero observed weight 기여와 현재 gradient가 따르는 zero-weight 계약의 차이다. 문헌 근거는 기존 직접 읽은 Cai(2010), pp.608–609 Appendices A/B의 expected-count/expected-complete-data 계약이다. zero count 항의 정의된 0 기여를 보존하는 코드 정합 후보로 제안하며 새로운 prior·ridge·확률 모델·정지 기준은 제안하지 않는다.

## 가장 작은 수정 후보: 아직 구현하지 않음

독립 source/log 검토에서 초기 blanket zero-count 후보의 거부 경계를 두 번 좁혔다. 현재 최소 후보는 **item threshold 벡터가 유한하고 모수 공간에서 엄격히 감소하며, count가 정확히 0이고 log probability가 정확히 negative infinity인 경우에만 0을 기여하게 한다.** 나머지는 기존 `count*log_probability`와 합산 순서를 유지한다. probability나 table 자체를 바꾸거나 전체 node를 버리지 않는다.

추가 검토가 발견한 소스 유도 위험은 equal adjacent threshold다. 같은 threshold는 유효 모형이 아니지만 middle category의 `ln(-0.0)`가 NaN이 아닌 `-inf`가 되므로 term 조건만으로는 잘못된 후보를 finite로 만들 수 있다. threshold domain guard는 모수 자체에 적용하며, 유효한 서로 다른 threshold의 base 덧셈 rounding으로 생긴 legal zero mass와 구분한다. 추가 test-only source `27614479b`에서 equal zero/mixed count를 실제 재현했다. 원래 objective와 domain-guard 후보는 NaN이었지만 term-only 후보는 finite였다. [원증거](candidate-domain-boundary/README.md)는 objective 경계의 관측이며 실제 line-search 채택이나 전체 fit 영향은 관측하지 않았다.

양수 count와 logP=-inf는 계속 nonfinite objective여야 한다. zero count여도 NaN/+inf log probability는 기존 multiplication의 nonfinite 결과를 유지한다. Newton/line-search 후보의 unordered threshold가 만든 NaN을 숨겨 finite objective로 통과시키지 않는다. negative/nonfinite count를 zero처럼 통과시키지 않는다. private helper의 malformed shape/count 입력을 새로운 public validation 계약으로 확장하지 않는다.

이 안전 경계의 unordered NaN 검사는 test-only source `d82c0e11`에서, equal threshold domain 검사는 후속 `27614479b`에서 실행했다. 원래 단계2의 세 경계 관측과 구분한다. 제안은 여전히 test-only 비교이며 production 변경·fit·GPU 실행은 하지 않았다. 유한하지 않은 threshold 전체와 모든 invalid candidate에 대한 완전한 보장을 이 작은 사례로 주장하지 않는다.

## 필요한 대조와 미확인

clean test-only source `71fc6d88561ccbc7ed6c5387b4d8180114cb6bcd`의 후속 q5/q7 순차 검증 결과를 회수했다. all-zero와 supported category의 mixed count는 native objective가 NaN, 독립 objective는 finite였고 gradient는 finite·모수 반환은 unchanged였다. zero-probability category의 positive count는 native/독립 objective 모두 positive infinity, gradient는 nonfinite·모수 반환은 unchanged였다. 양수 impossible count를 정상 성공으로 바꾸지 않았음을 확인했다. [clean-source 원증거](itemupdate-stage2/README.md)에 source·입력·원로그·정합 결과를 구분해 보존한다.

invalid private counts는 upstream expected-count 생성이 보장하는 finite nonnegative domain 밖이다. 이번 작은 item 실행에 malformed private counts/shape를 주입해 새 성공·거부 정책을 주장하지 않는다. source 계약 밖 입력의 검증은 별도 public trust-boundary 작업이다.

solver/ridge/FD/backtracking·in-place Hessian, GPU word kernel, mask·budget·posterior mass 소실 거부는 변경하지 않는다. backtracking 내부 branch는 이번 test에서 계측하지 않았으므로 unknown이다. reconstructed 첫 FD matrix는 진단이며 native runtime trace 또는 solver 수학 증명이 아니다.

단계2의 일반 대표 item에서 직접 objective와 central FD가 작은 차이를 보였다는 사실, CPU와 word counts의 같은 함수 반환 parity, 이 zero-weight 경계 재현은 서로 다른 검증 결과다. 전체 fit·261회 궤적·q121 counts·340인·bootstrap·불변 릴리스 수용으로 확대하지 않는다. API HOLD, E·권한 보호, 참가자 자료 접근 금지와 IRB 값 금지를 유지한다.

참고문헌: Cai, L. (2010). A two-tier full-information item factor analysis model with applications. *Psychometrika, 75*(4), 581–612. https://doi.org/10.1007/s11336-010-9178-0.
