# 최소 수리 후 CPU candidate·item 반환 확인

## 실행 식별과 범위

실행 head는 `dd2d842acbbc19e474e34dc3e118455961efbeb5`다. 문서 commit이며 계산·test source는 독립 검토된 `ea55bf1ffdf56e007df825207ec8fa4972eb7529`와 byte가 같다. 완료된 CPU-only RED/GREEN 회귀를 반복하지 않고, 수리 후 아직 실행하지 않았던 기존 GPU-feature binary의 CPU candidate 모드만 실행했다.

명시한 모드는 `G1_KERNEL_CANDIDATE_CONTRACT=1`, counts/itemupdate 모드는 각각 0이다. GPU feature로 binary를 빌드했지만 candidate 함수는 GPU context·GH·GPU dispatch·fit을 호출하지 않는다. 실제 1 passed / 0 failed / 0 ignored, 본문 0.00초로 끝났다.

## 실제 관측

| native item 입력 | objective | gradient | 반환 |
|---|---|---|---|
| legal strict zero counts | finite 0, 독립 Q와 일치 | finite zero | unchanged |
| legal strict mixed supported counts | finite 1.3132616875182228, 독립 Q와 일치 | finite | unchanged |
| positive count / legal -inf | +inf | nonfinite | unchanged |
| unordered 후보 / zero count | NaN | finite | unchanged |
| equal 후보 / zero·mixed counts | NaN | finite | unchanged |

같은 모수 Vec를 반환했다는 사실을 갱신 성공이나 수렴으로 해석하지 않는다. 내부 solver/backtracking branch는 미계측이며, legal mixed objective가 finite로 수리됐다는 사실과 실제 accepted step 존재는 다른 문제다. 이번 사례에서의 unchanged 반환은 원로그로 확인한 결과다.

유효한 `[1,1e-18,0]` threshold의 base rounding collapse는 finite·strict parameter-domain 안에서 처리했다. 잘못된 equal/unordered threshold는 유효 모델이 아닌 후보로 기록했다. synthetic zero/+inf, zero/NaN, positive/-inf 세 term도 nonfinite를 유지했고 이를 native 모델의 gradient/return 실행으로 표시하지 않았다.

raw event의 `rule_runtime_connected=false`는 별도 test-only 비교 closure가 runtime에 연결되지 않았다는 뜻이다. 실제 native objective는 ea55의 검토된 좁은 production 수리로 평가했다. 이 값을 production 수리 자체가 미적용이라는 뜻으로 사용하지 않는다. synthetic event의 `production_modified=false`도 scalar 대조가 추가 production 편집을 하지 않았다는 범위로 해석한다.

## 원증거와 미검증

- [실제 원로그](candidate.txt)
- [execution/code head·입력·source/binary/log hash](review-packet.json)

법적 모델 영역 밖 후보의 일반적 검증이나 모든 solver branch를 포괄하지 않는다. NaN/±inf threshold 회귀는 앞선 [production CPU-only 검사](../production-zero-weight-fix/README.md)의 별도 증거다. q5/q7 counts·itemupdate GPU 모드, GPU numerical parity, 261회 fit·q121 counts·340인·bootstrap·전체 fit·수렴·CI·릴리스·연구 수용은 이번 실행에서 검증하지 않았다.

실제 도구체인은 rustc/cargo 1.98.1 Homebrew / aarch64-apple-darwin / LLVM22.1.8이다. CI 1.97.1이나 독립 재실행 결과로 표시하지 않는다. 기존 실행·작성 프로세스가 없음을 확인한 뒤 하나만 시작했고 중복 실행은 하지 않았다. API HOLD 동안 원격 조회·push·게시·dispatch는 하지 않았고 E·runner·권한 설정·참가자 자료·원고·IRB 값은 변경하지 않았다.
