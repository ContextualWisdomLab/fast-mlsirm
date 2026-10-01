# Zero-weight 후보 규칙의 거부 경계 검증

## 정확한 실행 범위

test-only source `d82c0e11e18a6d62ceda9e162b6b73593ea76782`, base `60c831767ded38de2fb733695237aee0ffab7407`에서 기존 owned profile의 명시적 `G1_KERNEL_CANDIDATE_CONTRACT=1` 모드를 실행했다. 변경은 테스트 파일 하나이며 production `crates/` diff는 비어 있다.

GPU feature로 컴파일된 기존 test binary를 사용했지만 이 모드는 GPU context·GH node 생성·GPU dispatch를 하지 않는다. 작은 CPU item 평가 4개와 synthetic scalar term 3개만 수행했다. 실제 결과는 1 passed / 0 failed / 0 ignored, 본문 0.00초다. proposal rule은 test closure에만 있고 production helper/runtime API에 연결하지 않았다.

## 실제 native와 test-only 규칙 대조

규칙은 정확히 `count==0 && lp==-inf`일 때만 0 기여, 나머지는 원래 multiplication과 같은 category sum 순서다. 이 규칙으로 gradient·solver·ridge·FD step·backtracking·prior·node·criteria를 변경하지 않는다.

| item 입력 | 기존 objective | test-only 규칙 objective | 기존 gradient | 기존 반환 |
|---|---|---|---|---|
| legal zero count·legal -inf | NaN | finite 0, 독립 Q 일치 | finite zero | unchanged |
| legal mixed supported count·legal -inf | NaN | finite 1.3132616875182228, 독립 Q 일치 | finite | unchanged |
| positive count·legal -inf | +inf | +inf | nonfinite | unchanged |
| zero count·unordered 후보 NaN | NaN | NaN | finite | unchanged |

unordered 후보는 packed `[1,-0.2,0.2]`와 counts `[[1,0,0]]`다. 잘못된 threshold 순서를 유효한 모형 입력으로 표시하지 않고 line-search candidate 경계로 명시했다. zero middle count가 있어도 해당 NaN objective를 숨기지 않았다. 실제 helper의 unchanged 반환과 nonfinite objective를 관측했으며 내부 branch를 계측했다고 주장하지 않는다. 같은 Vec 반환은 업데이트 성공·수렴을 의미하지 않는다.

합법 zero-mass의 packed `[1,1e-18,0]`와 coords `[1]`는 기존 경계 재현을 재사용한다. 독립 직접 cumulative Q는 production probability helper를 호출하지 않는다. test-only 규칙은 합법 두 사례를 finite로 만들고 독립 Q와 1e-12 안에서 일치했다. 기존 production objective의 문제는 여전히 남아 있으며 수리 완료로 표시하지 않는다.

## +inf·NaN synthetic term의 구분

`lp=+inf`는 합법 GRM log probability가 아니다. 잘못된 native 모수로 실제 +inf category가 생성됐다고 주장하지 않고, 별도의 synthetic scalar 입력으로 원래 곱과 test-only 규칙만 대조했다.

- zero count/+inf: 양쪽 곱 모두 NaN.
- zero count/NaN: 양쪽 곱 모두 NaN.
- positive count/-inf: 양쪽 곱 모두 -inf, 음의 objective로 바꾸면 +inf.

이 세 입력에서 native item objective·gradient·return을 실행했다고 표시하지 않았다. primitive 거부 경계와 unordered native 후보 검사는 서로 다른 관측이다. malformed private counts도 새 유효 입력으로 받아들이지 않았다.

## 컴파일·원증거·미수용

같은 source의 CPU-only 및 coverage CPU-only lib/tests check와 기존 Python 실행/소유/toolchain 계약 37개가 통과했다. 기존 dead-code warning은 남아 있다. 실제 도구체인은 rustc/cargo 1.98.1 Homebrew / aarch64-apple-darwin / LLVM 22.1.8이며 CI 1.97.1·전체 suite·독립 실행 결과가 아니다.

- [native/synthetic 경계 원로그](candidate.txt)
- [컴파일·계약 원로그](compile-contract.txt)
- [exact source/base/diff·입력·binary·원로그 hash](review-packet.json)

raw event는 모수·count·lp/gradient bits, finite/NaN/inf 여부, 반환 unchanged 및 독립 Q 결과를 구분한다. internal branch는 unknown이며 이 검증은 solver 수학 정합·fit·수렴·q121 counts·340인·bootstrap·불변 릴리스·연구 수용을 입증하지 않는다.

생산 수리는 아직 미배정·미적용이다. API HOLD 동안 로컬 증거만 보존했고 GH 조회·push·게시·remote dispatch는 하지 않았다. E·runner·권한 설정과 참가자 자료·원고 수치·IRB 값은 변경하지 않았다. [최소안](../zero-weight-objective-minimal-proposal.md)의 안전 조건 검증만 추가했다.
