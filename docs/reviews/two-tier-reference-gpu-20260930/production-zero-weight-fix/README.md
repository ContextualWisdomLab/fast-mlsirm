# 실제 item objective의 zero-weight 최소 수리

## 정확한 소스와 범위

실행 source는 `ea55bf1ffdf56e007df825207ec8fa4972eb7529`, base는 `5433f59b2eb15a81cd1d57143074a4b87517f3dc`다. 변경은 two-tier item objective의 threshold-domain guard와 zero-count/negative-infinity 기여 예외, 작은 CPU 회귀 및 기존 test-only legal 기대값 갱신뿐이다.

production 규칙은 원모수 threshold가 유한하고 엄격히 감소할 때만 `count==0 && lp==-inf` 항을 0으로 처리한다. 나머지 곱과 category·node 합산 순서를 유지한다. probability floor·threshold 재정렬·새 prior·ridge·criteria·node를 추가하지 않았고 analytic gradient·Newton solver·FD step·backtracking·GPU kernel·binding·공개 API는 바꾸지 않았다.

이 문서는 이전 [test-only 설계·관측](../zero-weight-objective-minimal-proposal.md)의 후보를 좁게 적용한 후속 기록이다. 이전 판본의 미적용 상태와 이번 local source를 구분한다. 수리의 GPU parity·fit·수렴·q121 counts·340인·bootstrap·전체 suite·CI·릴리스·연구 수용은 검증하지 않았다.

## 실제 RED

[수리 전 RED](red.txt)는 base5433의 production 원본에 새 CPU 회귀만 추가한 dirty test 단계다. 원래 production SHA-256 `51c47ceba6cb5eee1ec33000722e417cd1336fc8dd4544faf768adc07ce2866b`, test SHA-256 `6a11426a8903890afa08117728143c254b4054d701fda6ab5e4af17bafa107ed`를 본문에 기록했다.

먼저 equal/unordered/NaN/+inf/-inf threshold 후보 5개가 모두 nonfinite objective·unchanged 반환임을 확인했다. positive count의 legal -inf도 +inf였다. 이후 legal strict zero-count objective가 finite여야 한다는 assertion에서 실제 실패했다: **0 passed / 1 failed / exit101**. 실패 시점에는 mixed-supported legal assertion까지 도달하지 않았으며 그 결과를 RED 실행으로 꾸미지 않는다.

## 같은 clean source의 GREEN

[GREEN](green.txt)은 새 sourceea55의 CPU-only 작은 회귀다. GPU context·GH·fit을 실행하지 않았다. 실제 **1 passed / 0 failed / 0 ignored**, 본문 0.00초다.

| actual production 입력 | 실제 결과 |
|---|---|
| legal packed `[1,1e-18,0]`, counts `[0,0,0]` | objective finite -0, 독립 기대값 0과 일치 |
| 같은 packed, counts `[1,0,0]` | objective finite 1.3132616875182228, 직접 `ln(1+exp(1))`와 일치 |
| 같은 packed, counts `[0,1,0]` | objective +inf·gradient nonfinite 유지 |
| equal/unordered threshold, zero counts | objective NaN·gradient finite·반환 unchanged 유지 |
| NaN/+inf/-inf threshold, zero counts | objective NaN·gradient finite·반환 unchanged 유지 |

유효 threshold 벡터 자체가 엄격히 감소하는지 검사한다. base 덧셈에서 rounded upper/lower가 같아져 생긴 legal zero mass는 invalid equal-threshold parameter state와 혼동하지 않는다.

invalid 후보가 finite로 바뀌는 반례는 이 5개에서 없었다. 모든 가능한 invalid 후보의 완전한 보장은 아니다. internal branch는 계측하지 않았고, invalid 후보의 unchanged 반환을 성공/수렴으로 표시하지 않는다. 수정한 legal mixed 입력의 item-return/fit 동작은 이번 GREEN에서 실행하지 않았다.

## 컴파일·회귀·원증거

같은 source의 CPU-only production check, coverage CPU-only lib/tests check와 기존 Python 실행/소유/toolchain 계약 37개가 통과했다. 기존 dead-code warning은 남아 있다. 실제 rustc/cargo는 1.98.1 Homebrew / aarch64-apple-darwin / LLVM22.1.8이며 CI 1.97.1 실행 증거가 아니다.

- [RED 원로그](red.txt)
- [GREEN 원로그](green.txt)
- [실제 순차 compile·37 계약 출력](compile-contract.txt)
- [source/base/diff·RED/GREEN 입력·binary·원로그 manifest](review-packet.json)

RED와 GREEN은 서로 다른 compiled binary다. 해시는 실제 회수한 GREEN binary에만 부여하고, 그 해시를 RED binary로 소급 사용하지 않는다. RED test source는 dirty stage로 명시하며 clean head 실행으로 표시하지 않는다.

기존 test-only boundary 관측의 legal 기대값은 수리된 objective에 맞게 갱신했지만 이번 단계에서 해당 GPU-feature profile·q5/q7 item suite를 재실행하지 않았다. 이전 clean 결과를 새 ea55 결과로 대신 세지 않는다. source diff와 narrow CPU 회귀만 기존 독립 검토자에게 전달하며 검토 완료도 자동으로 가정하지 않는다.

이 수정은 기존 Cai(2010), 직접 읽은 pp.608–609 Appendices A/B의 zero expected-count 기여와 기존 gradient의 zero 처리에 맞춘 코드 정합이다. 새 추정기·수치 모형·연구 수치를 도입하지 않았다. API HOLD 동안 source/doc은 로컬만 보존하고 원격 query·push·게시·dispatch는 하지 않았다. E·runner·권한 설정·참가자 자료·원고·IRB 값은 변경하지 않았다.
