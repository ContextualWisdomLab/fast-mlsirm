# Equal-threshold 후보의 domain 경계 대조

## 실행 식별

test-only source `27614479bff357c0aad8f2c27dc7011c8ddad6b0`, base `45578feb5c9d108df951f9691546a4a406fa7218`다. 기존 owned CPU candidate 모드에 equal threshold zero/mixed count 두 사례와 threshold-domain 조건을 추가했다. production crate diff는 비어 있다. 제안 규칙을 production helper/API에 연결하지 않았다.

실제 작은 CPU 검사는 1 passed / 0 failed / 0 ignored, 본문 0.00초로 끝났다. native item 6개와 synthetic scalar term 3개를 수행했으며 GPU context·GH·fit·remote dispatch는 없었다. CPU-only·coverage CPU-only lib/tests check와 기존 Python 실행/소유/toolchain 계약 37개도 통과했다. warning은 남아 있고 전체 suite·CI 결과가 아니다. 실제 도구체인은 rustc/cargo 1.98.1 Homebrew / aarch64-apple-darwin / LLVM 22.1.8이다.

## 실제 equal 후보 관측

packed `[1,0,0]`, coords `[1]`에서 threshold 두 개는 같은 finite 값이다. 이는 모형의 strict decreasing domain을 위반하는 line-search 후보이지 유효한 모형 입력이 아니다. middle log probability는 실제 -inf였다.

| counts | native objective | 기존 term-only 후보 | domain-guard 후보 | native 반환 |
|---|---|---|---|---|
| `[0,0,0]` | NaN | finite | NaN | unchanged |
| `[1,0,0]` | NaN | finite | NaN | unchanged |

이 두 사례의 native gradient는 finite였다. native objective가 nonfinite이고 반환이 unchanged였음을 기록했다. 내부 branch는 계측하지 않았으므로 accepted/rejected Armijo 후보를 직접 관측했다는 주장은 하지 않는다. term-only 후보가 잘못된 equal state를 finite로 만드는 위험은 이제 test-only objective 대조로 실제 재현됐지만, production이 그 제안 규칙을 적용했거나 전체 fit에서 해당 후보를 채택했다는 증거는 아니다.

## 보존한 다른 경계

수정한 test-only 규칙은 **원모수 threshold 벡터의 finite·strict decreasing 조건을 먼저 확인하고**, 그 domain 안에서 `count==0 && lp==-inf`일 때만 0 기여를 허용한다. 나머지 곱·합산 순서는 그대로다.

- 유효한 packed `[1,1e-18,0]`는 원모수 threshold가 엄격히 감소한다. base 덧셈에서 rounded upper/lower가 같아져 생긴 legal zero mass는 계속 허용했고 zero/mixed objective는 finite·독립 Q와 일치했다.
- positive count/-inf는 native·제안 objective 모두 +inf로 남았다.
- unordered threshold/zero count의 NaN은 native·제안 모두 NaN으로 남았다.
- synthetic zero/+inf, zero/NaN, positive/-inf term의 nonfinite도 유지했다. 합법 GRM probability로 표시하거나 synthetic term의 native item gradient/return을 실행했다고 주장하지 않았다.

finite 검사는 제안 코드에 있지만 이 실행은 finite strict/equal/unordered threshold 사례만 다룬다. 비유한 threshold 벡터 전체나 모든 invalid 후보를 동적으로 검증했다고 주장하지 않는다. solver/ridge/FD/backtracking·prior·node·criteria는 변경하지 않았다. unchanged 반환을 갱신 성공이나 수렴으로 세지 않는다.

## 원증거와 한계

- [native/synthetic 원로그](candidate.txt)
- [컴파일·37 계약](compile-contract.txt)
- [exact source/base/test diff·입력·binary·원로그 hash](review-packet.json)

원모수 domain 판정과 log-probability/gradient bits, native·term-only·domain-guard objective의 finite/NaN/inf를 구분해 기록했다. [이전 term-only 검사](../candidate-rule-boundary/README.md)와 [미적용 최소안](../zero-weight-objective-minimal-proposal.md)은 다른 source의 기록이다.

production 결함 수리·fit·수렴·q121 counts·340인·bootstrap·GPU·릴리스·연구 수용은 여전히 미입증이다. 독립 읽기 검토도 실행 재현·GitHub 승인과 구분한다. API HOLD 동안 로컬에만 보존하며 E·runner·권한 설정·참가자 자료·원고·IRB 값을 변경하지 않았다.
