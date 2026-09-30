# Host GRM 표의 node별 1회 계산 검증

## 변경 범위

실행 소스는 `4c09920d0e08f5fcbb8f68cab5ceb140d4c7af1b`, parent는 문서 commit `a141ecbf9`다. [읽기 전용 최소안](../q121-prepare-stages/host-table-minimal-proposal.md)을 검토한 뒤 배정된 범위만 구현했다.

`item_logprob_tables`의 category loop를 같은 `(item,g,h)`에서 기존 `item_primary_base`·specific 항의 동일 덧셈 순서·기존 `grm_logprobs` 1회 호출과 연속 복사로 바꿨다. primary 합을 h 밖으로 이동하지 않았다. 캐시·병렬화·새 추상화, 수식·prior·노드·허용오차, table layout, 크기·할당 guard, CPU on-the-fly·Newton 경로와 GPU word shader는 그대로다.

호출 counter는 test-only module의 opt-in thread-local이다. `poly.rs`에는 `#[cfg(test)]` hook 두 줄만 추가했다. 이를 제거한 production probability 본문은 parent와 동일하다. production API를 추가하지 않았다.

## 실제 RED와 GREEN

순수 bit 동일성 검사는 기존 구현에서도 통과할 수 있으므로 이를 RED로 부르지 않았다. 전체 범주 helper의 실제 호출 수로 중복 계산을 재현했다.

- [CPU-only RED](callcount-red.txt): 기존 표 생성, 이진 범주, node 12개에 helper 24회. 호출 수 assertion이 실제 실패했고 0 passed / 1 failed / exit 101이었다. parent 위에 test-only 계측과 회귀만 추가한 dirty 단계다.
- [CPU-only GREEN](callcount-green.txt) 및 [default feature GREEN](default-bit-green.txt): 12개 구조·모수 갱신 사례에서 helper 호출 수가 node 수와 같고 모든 표 원소의 `to_bits()`가 기존 scalar oracle과 같았다. 두 검사 모두 1 passed / 0 failed / 0 ignored다.

cross-loading, 음수 slope, S0, specific-free 문항, 이진·4범주·좁은 3범주 thresholds, 비대칭 좌표와 새 모수를 포함했다. legal `-inf` 67개를 보존했고 overflow는 probability 평가 전에 거부됐다. 크기·할당 guard의 코드가 그대로인지 별도 source diff로 대조했다. 큰 할당을 강제로 실패시키거나 실제 peak RSS를 측정한 검사는 아니다.

## 같은 clean source의 로컬 GPU 검사

실제 로컬 도구체인은 rustc/cargo 1.98.1 Homebrew, aarch64-apple-darwin, LLVM 22.1.8이다. CI의 Rust 1.97.1 실행으로 표시하지 않는다. 모든 검사는 `CARGO_BUILD_JOBS=1`, `CARGO_INCREMENTAL=0`으로 순차 실행했다.

[q5 고정은행 profile](q5-profile.txt)은 Apple M1 / Metal / IntegratedGpu에서 같은 합성 입력으로 통과했다. 입력 hash는 `5f211b3d5872eeda9e610fb721f633a03fcff7ed787ddf4c7bdd9526a6d9047e`, GPU 예산 1048576 bytes, host 예산 67108864 bytes다. 19개 host-clock 키, timing receipt 거부 5건과 likelihood·평균·2차 적률·SD 차이 0을 확인했다. counts 수집과 모수 갱신은 없는 1인 검사다.

이번 q5의 CPU table 생성은 0.000241750초, CPU table 포함 전체는 0.000284417초, GPU 경로 전체는 0.004224584초였다. 여전히 이 작은 전체 측정에서는 GPU가 느렸다. 다른 source/window의 시간과 나눈 값을 가속률로 제시하지 않는다. parent/child 계측은 겹치므로 합산하지 않는다.

[기존 strict hardware 검사](strict-hardware.txt)는 같은 compiled test binary에서 1 passed / 0 failed / 0 ignored, 45.98초로 통과했다.

- binary64 word addition 1037개, fixed-state full/short/missing/invalid/new-iteration reuse 7개, malformed input 거부 3개 및 overflow zero-mass 보존을 확인했다.
- legal GRM zero-probability node 4개 사례가 통과했다.
- 기존 sparse 합성 fixture는 CPU/GPU 모두 261회였고 전체 궤적의 모수·likelihood 차이가 0이었다. 최종 primary/specific loading·threshold 차이도 0이었다.
- 같은 CPU state 102의 word-product counts·gradient·Newton update 차이는 0이었다. 로그에 함께 남은 기존 f32 quantization 진단은 별도 경로이며 그 오차를 0으로 바꾼 것으로 해석하지 않는다.

[CPU/coverage 및 실행 계약 검사](cpu-coverage-contract.txt)도 통과했다. CPU-only production check, coverage CPU-only lib/tests check와 Python 4개 계약 파일의 37 passed다. 기존 dead-code warning은 남아 있다. Python 전체 suite·PyO3 재빌드·새 required CI 전체 통과를 주장하지 않는다.

## 증거와 남은 수용 범위

[source·diff·binary·로그 manifest](review-packet.json)에 실행 head, parent, 3개 source-file 변경, source hash, local binary hash와 원로그 hash를 결속했다. RED binary의 해시는 당시에 별도로 보존하지 않았으므로 후속 GREEN binary 해시로 대신 표시하지 않는다. 원로그는 끝 공백·빈 줄을 포함해 byte 그대로 보존했다.

이번 변경으로 q121 고정은행의 source상 helper 호출 수는 113379904회에서 28344976회가 된다. 실제 q121 성능 측정은 아직 하지 않았다. [기존 q121 시간 분리 기록](../q121-prepare-stages/README.md)은 이전 source `3dc48dc`의 결과이며 새 구현의 실행 증거로 사용하지 않는다.

독립 검토, fresh runner/owner·예산 guard와 별도 실행 배정 전까지 새 q121 remote run은 하지 않는다. 전체 reference fit·340인 throughput·node sensitivity·peak RSS/VRAM·필수 CI·공식 승인·불변 릴리스는 미수용이다. API HOLD 동안 push·게시·dispatch는 중단한다. 참가자 자료·원고 수치·IRB 값은 다루지 않았으며 E나 권한 설정을 변경하지 않았다.

기존 Cai(2010), 직접 읽은 pp.608–609 Appendices A/B의 posterior 계약을 유지하는 호출 중복 제거다. 이 문서 commit은 별도이며 위 실행 source와 혼동하지 않는다.
