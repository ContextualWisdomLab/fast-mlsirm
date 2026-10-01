# 작은 고정 counts item 반환 검증

## 소스와 실제 실행

test-only source `71fc6d88561ccbc7ed6c5387b4d8180114cb6bcd`는 단계2 설계 base `0b561646a2d8ee0f4a3d54d5537b2b04082744fa` 이후 테스트 파일 하나만 바꿨다. 중간 test commit은 `1a98dbe67e0abaab2e5deae16e208ca71ac739c5`다. `crates/` 전체 diff는 비어 있고 production objective·gradient·solver·GPU kernel·binding은 그대로다.

Apple M1 / Metal에서 같은 clean source로 q5와 q7의 고정 counts를 각각 사용해 대표 item 네 개를 검사했다. 일반 block, cross-loading block, specific-free, P1/S0 item이다. 각 실행은 1 passed / 0 failed / 0 ignored, 본문 시간은 q5 0.24초, q7 0.13초였다. fit·261회 궤적·q121 counts·340인·bootstrap·원격 GPU dispatch는 실행하지 않았다.

기존 단계1 모드와 기본 moments 모드도 같은 소스에서 각각 1 pass로 통과했다. CPU-only·coverage CPU-only lib/tests check와 기존 Python 실행/소유/toolchain 계약 37개도 통과했다. 기존 warning은 남아 있으며 전체 suite나 required CI·독립 재빌드 증거는 아니다. 실제 도구체인은 rustc/cargo 1.98.1 Homebrew / aarch64-apple-darwin / LLVM 22.1.8이다.

## 중간 범위 objective·gradient·반환

counts는 단계1의 같은 bank/GH/prior·현재 모수로 streamed/cached CPU와 GPU word-product 경로에서 생성했다. counts·item packed 입력·free dimension·ridge·Newton 횟수를 raw input과 hash로 결속했다.

기존 helper를 호출하지 않는 테스트 전용 직접 cumulative-logistic objective와 central FD(eps=1e-6)를 사용했다. 기존 analytic gradient와의 최대 FD 잔차는 q5 `2.2971059915333214e-9`, q7 `3.479223020796951e-9`였다. objective 최대 절대차는 q5 `5.329070518200751e-15`, q7 `7.105427357601002e-15`였다. 기존 unit의 1e-4 검사 조건을 사용했고 production FD h=1e-5나 fit 정지 기준을 바꾸지 않았다.

CPU와 word-derived counts에서 기존 item helper의 반환은 bitwise 같았다. 대표 8개 관측에서 반환 모수는 finite이고 초기와 달랐으며, 재평가한 native objective는 초기값보다 크지 않았다. 이를 수렴 또는 solver 수학 정합의 단독 증거로 쓰지 않는다. 하나의 현재 함수를 같은 counts에 적용한 parity와 독립 objective FD는 서로 다른 검사다.

첫 forward-FD Hessian과 현재 source의 in-place 평균화·ridge를 재구성해 raw matrix로 기록했다. 이는 진단이다. native 함수 안의 solve step·backtracking 후보·branch는 계측하지 않았으므로 unknown으로 남겼다. 행렬 진단을 runtime trace·SPD·조건수·전체 solver 식별 증명으로 표시하지 않는다. flat/singular·backtracking 실패·비교 정의역 밖 tail의 모든 branch를 실행했다는 주장도 하지 않는다.

## 실제 zero-weight 경계 관측

finite packed `[1.0,1e-18,0.0]`, coords `[1.0]`, specific 없음, 3범주 node 한 개의 item helper를 따로 평가했다. middle category는 f64 rounding으로 legal log probability `-inf`가 된다.

| counts | native objective | 독립 zero-contribution objective | gradient | 반환 |
|---|---|---|---|---|
| `[0,0,0]` | NaN | finite 0 | finite zero | unchanged |
| `[1,0,0]` | NaN | finite | finite | unchanged |
| `[0,1,0]` | positive infinity | positive infinity | nonfinite | unchanged |

첫 두 사례는 `0*(-inf)`의 native objective 문제를 실제로 재현했다. 마지막은 양수 impossible count를 finite 성공으로 통과시키지 않았음을 대조한다. 기존 helper의 초기 nonfinite guard와 일치하는 결과이나, 내부 branch 계측 자체는 하지 않았다. observation을 기록한 harness는 통과했으므로 이를 failing regression/TDD RED로 꾸미지 않는다.

[최소 수정 제안](../zero-weight-objective-minimal-proposal.md)은 zero count 항만 정의된 0 기여를 보존하는 후보다. **production 수리는 아직 적용하지 않았다.** positive count·legal `-inf`, posterior mass 소실·invalid 입력을 통과시키는 floor/clamp·node 제거·nonfinite success는 추가하지 않았다. 이 standalone 경계의 전체 fit·수렴·연구 수치 영향은 미입증이다.

private helper의 invalid count/shape는 expected-count 생성의 finite nonnegative domain 밖이다. malformed private 배열을 실행해 새 public 거부 계약을 주장하지 않았다. 입력 validation은 기존 단계1 경계와 구분한다.

## 예산·기존 경로·원증거

GPU caller budget 1048576 bytes(1MiB), host budget 67108864 bytes(64MiB)와 기존 source-shaped host guard를 사용했다. packed 모수·작은 FD 행렬은 test-only host 객체이며 이 예산을 peak RSS/VRAM 상한으로 표시하지 않는다. source/input/GH·raw log·binary hash는 [review packet](review-packet.json)에 보존했다.

- [q5 clean item 검사](q5-item.txt)
- [q7 clean item 검사](q7-item.txt)
- [기존 단계1 counts](stage1-counts.txt)
- [기본 moments·19키](moments.txt)
- [CPU/coverage·37 계약](compile-contract.txt)

기존 profile의 owned test 식별자를 유지했고 item 모드는 별도 `G1_KERNEL_ITEMUPDATE_CONTRACT=1`로 명시한다. counts와 item 모드를 동시에 선택하면 거부하며 absent/0 기본 모드는 기존 moments 검사다. 새로운 remote/CI 분기나 ignored test를 추가하지 않았다.

preliminary dirty-source 관측과 최종 clean-source 관측은 분리한다. 이 문서 commit은 실행 source와 별개다. 독립 읽기 검토와 production 최소 수정 검토는 별도이며 공식 승인·불변 릴리스 수용은 아니다. API HOLD 동안 source/test/doc은 로컬만 보존한다. E·runner·권한 설정·원고 수치·참가자 자료·IRB 값을 변경하지 않았다.

기존 expected-count 근거는 직접 읽은 Cai(2010), pp.608–609 Appendices A/B다. 기존 CPU item 구현을 검증한 것이며 새 추정기·prior·ridge·수렴 기준을 도입하지 않았다.
