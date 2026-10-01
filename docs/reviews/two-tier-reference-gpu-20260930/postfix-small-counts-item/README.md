# 수리 후 작은 counts·item 모드 정합

## 실행과 중복 방지

사용자의 결과 확인·계속 진행 지시에 따라, 완료된 CPU RED/GREEN·candidate 검사는 반복하지 않았다. 진행 중인 cargo/rustc/test 프로세스가 없고 검토된 source와 기존 binary가 일치함을 확인한 뒤, 수리 후 미실행이던 q5 counts부터 실행했다. 결과를 확인한 다음 q7 counts와 q5/q7 itemupdate 모드를 각각 한 번 순차 실행했다.

실행 head는 `caed17c3e38faf59312f467f4fc18bdddc4815a2`이며 계산·test source는 독립 검토된 `ea55bf1ffdf56e007df825207ec8fa4972eb7529`와 byte가 같다. 기존 test binary SHA-256은 `d8c9308fb66dd508872e14fba6c30cab913a55834a912801c0a854bec874b303`이다. 새 빌드·source 변경·원격 실행은 하지 않았다.

실제 Apple M1 / Metal에서 기존 owned test 식별자의 명시적 모드만 사용했다. 각 검사에서 node는 q5/5 또는 q7/7, 4인·16문항·4범주의 고정 합성 state다. GPU caller budget은 1048576 bytes(1MiB), host budget은 67108864 bytes(64MiB)이며 source-shaped host 추정의 가장 큰 값은 2802576 bytes였다. 예산은 peak RSS/VRAM 측정이나 상한 보장이 아니다.

## 실제 결과

| 실행 모드 | 결과 | 본문 시간 |
|---|---|---:|
| q5 counts | 1 passed / 0 failed / 0 ignored | 0.32초 |
| q7 counts | 1 passed / 0 failed / 0 ignored | 0.37초 |
| q5 itemupdate | 1 passed / 0 failed / 0 ignored | 0.08초 |
| q7 itemupdate | 1 passed / 0 failed / 0 ignored | 0.10초 |

각 counts 모드는 P2/S4의 fully observed state, specific-free·negative slope·missing cell/block/person state, P1/S0 state를 대조한다. streamed/cached CPU와 실제 GPU word-product 경로의 counts 최대 절대차는 0, cross·likelihood bits도 같았다. mass 잔차의 최대값은 q5 `3.552713678800501e-15`, q7 `4.6629367034256575e-15`였다. 3+1 batch, masked sentinel와 다음 고정 모수 state의 정합도 유지했다.

각 모드에서 기존 validation 5종·한 사람 최소 budget보다 1 byte 부족·finite-input overflow에 따른 전체 posterior mass 소실의 7종 거부를 확인했다. 이를 GPU 실패 뒤 CPU fallback으로 대체하지 않았다.

itemupdate 모드의 대표 item은 일반 block, cross-loading block, specific-free, P1/S0다. q별 4개, 총 8개 대표 관측에서 CPU와 word-derived counts의 기존 item 반환이 bitwise 같았다. 독립 직접 cumulative objective 차이는 최대 `7.105427357601002e-15`, central-FD 잔차는 최대 `3.479223020796951e-9`였다. 대표 item 반환은 finite·changed였고 native objective는 증가하지 않았다.

특수 zero/mixed supported-count 사례의 native objective는 이제 finite이며 독립 Q와 일치했다. positive impossible-count는 +inf를 유지했다. 이 특수 사례의 item 반환은 unchanged로 기록됐다. 대표 일반 item의 changed 반환과 이 특수 unchanged 결과를 서로 대체하지 않는다. 어느 쪽도 전체 fit 성공·수렴을 입증하지 않으며 내부 solver/backtracking branch는 미계측이다.

## 연산 담당과 검증 범위

GPU는 기존 reference binary64 word log-product를 계산하고 Rust CPU가 정규화·counts/cross를 누적한다. item objective·gradient·Newton은 CPU다. counts 모드는 item M-step을 실행하지 않고, itemupdate 모드는 작은 대표 item helper만 호출한다. fit·latent parameter EM·261회 궤적을 실행한 것은 아니다.

q5/q7의 구조·고정 통계 검증을 연구 q121 적분 설정이나 recovery 증거로 쓰지 않는다. 이번 실행에서 기본 moments의 19키 시간 검사를 다시 돌리거나 새 elapsed ratio를 speedup으로 주장하지 않았다. 이전 head의 q121 결과를 ea55 수리 후 실행으로 소급하지 않는다.

## 원증거와 미수용

- [q5 counts 원로그](q5-counts.txt)
- [q7 counts 원로그](q7-counts.txt)
- [q5 item 원로그](q5-item.txt)
- [q7 item 원로그](q7-item.txt)
- [source·입력·GH/prior·binary·로그 manifest](review-packet.json)

실행 전 input/GH/prior가 이전 fixed-state 검사와 같음을 대조했고, source·binary는 ea55 수리가 적용된 상태다. 원로그는 마지막 빈 줄까지 byte 그대로 보존한다. 이 때문에 원로그 파일의 `git diff --check`는 EOF 빈 줄 경고를 내며, 해시를 바꿔 경고를 지우지 않았다. 문서와 manifest의 공백 검사는 별도로 통과시킨다. 실제 도구체인은 기존 binary의 rustc/cargo1.98.1 Homebrew / aarch64-apple-darwin / LLVM22.1.8이다. CI1.97.1·독립 rebuild·fresh required CI 결과가 아니다.

GPU parity는 이 작은 고정 통계·대표 item 범위로만 확인했다. 공개 API 전체 fit, 최종 모수·수렴·적합도, 261회 sparse 궤적 재검증, q121 counts·340인·bootstrap·전체 suite·불변 릴리스·연구 수용은 미검증이다. 공식 승인과 독립 offline receipt 대조는 별도다.

API HOLD 동안 GH 조회·push·게시·remote dispatch는 하지 않았다. E·runner·권한 설정·모형·prior·node·criteria·참가자 자료·원고·IRB 값은 변경하지 않았다.

## 별도 후속: 수리 후 기본 q5 moments·19키

위 네 모드의 완료·독립 대조 뒤, 아직 실행하지 않았던 기본 q5/5 모드만 기존 binary로 한 번 확인했다. 실행 head `0856b6e5a3765e78b3f58ecfa199fd69814a63bd`, code/test는 ea55와 byte가 같다. 실제 Apple M1 / Metal에서 1 passed / 0 failed / 0 ignored, 본문 0.24초였다. 1인 고정은행·GPU1MiB/host64MiB·no-count/no-update/no-fit이며 input hash는 `5f211b3d5872eeda9e610fb721f633a03fcff7ed787ddf4c7bdd9526a6d9047e`다.

likelihood·mean·second·marginal SD 차이는 모두 0, 19개 timing key는 unique·finite·nonnegative이고 malformed receipt 5건도 실제로 거부했다. CPU cached+table은 0.000297791초, GPU 전체는 0.003683084초로 이 작은 관측에서도 GPU가 느렸다. parent와 child는 중첩되므로 합산하지 않고, clock/driver/계측 오버헤드를 포함한 host 관측을 hardware GPU/PCIe 시간이나 q121 speedup으로 표시하지 않는다.

[별도 moments 원로그](q5-moments.txt)와 [별도 manifest](moments-receipt.json)에 source·input·binary·원로그 해시를 결속했다. 위 원래 네 실행의 `review-packet.json`과 독립 보고서 해시는 변경하지 않았다. 이 한 번의 기본 경로 통과도 full fit·수렴·q121·CI·릴리스 수용을 뜻하지 않는다.
