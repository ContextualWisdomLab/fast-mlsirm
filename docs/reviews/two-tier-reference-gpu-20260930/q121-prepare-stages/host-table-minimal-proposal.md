# Host GRM 표 생성 최소안: 읽기 전용 검토

## 범위와 근거

검토 소스는 `3dc48dc28a8529f0a4542f95fbd85e91fcb69fde`다. [실측](README.md)에서 GPU 경로의 host table 생성은 44.548초, 준비 중 map/copy는 3.055초, compute pipeline은 0.319초였다. 이 시간은 고정 합성은행 q121/121·1인·counts 없음·모수 갱신 없음의 한 번 측정이다. 전체 fit의 병목이나 예상 가속률을 단정하지 않는다.

소스 대조로 표 생성의 중복 호출을 확인했다. 아래는 구현 전 제안이며 코드 변경, 캐시, 병렬화 또는 새 실행은 하지 않았다.

## 호출 경로와 기존 구현

`two_tier_grm.rs:1092–1124`의 `item_logprob_tables`는 순차 item → primary node → specific node → category loop다. category마다 `item_cat_logprob`(`:832–848`)를 호출한다. 이 함수는 매번 같은 primary 합과 specific 항을 구하고 `poly.rs:89–110`의 `grm_logprobs(base, thresholds)`로 **전체 범주 Vec**를 만든 뒤 한 범주만 선택한다.

이 표 생성 함수는 GPU posterior 경로(`two_tier_grm.rs:1312`)와 선택적 CPU fixed-bank table 경로(`:2985–2987`)가 함께 사용한다. 기본 CPU on-the-fly 평가나 Newton M-step을 이 제안으로 바꾸지 않는다.

기존 `bifactor_grm.rs:565–599`의 `fill_logprob_tables`는 node마다 `grm_logprobs`를 한 번 호출하고 전체 범주를 연속 slice로 복사한다. 이것이 재사용할 패턴이다. 다만 bifactor의 단일 general 좌표와 다른 `Validated`/`ItemParams` 형태를 two-tier의 다중 primary 좌표 대신 사용할 수는 없다. 직접 helper를 호출하거나 새 공통 추상화를 만드는 대신, 기존 two-tier `item_primary_base`와 기존 `grm_logprobs`를 그대로 사용한다.

대조한 두 표 생성 함수 내부는 순차 loop이며 `par_iter`, thread spawn 또는 worker 인자가 없다. 이 결과는 저장소 전체에 병렬 기능이 없다는 뜻이 아니다. 다른 모형의 worker 경로를 이 표에 연결하는 일은 최소 중복 제거와 별개이고 이번 제안에서 제외한다.

## 최소 변경 후보

`item_logprob_tables`에서 각 `(item, g, h)`의 base를 기존 `item_primary_base`와 기존 `a_s` match로 계산한다. 같은 `grm_logprobs`를 한 번 호출하고 결과를 category 순서대로 `table.extend_from_slice(&probs)`로 복사한다. 기존 checked size 계산, `try_reserve_exact`, `Result` 오류와 외부 loop 순서는 유지한다. 우선 primary 합을 h loop 밖으로 이동하는 추가 최적화도 하지 않는다.

q121/P2/S4 고정은행은 모든 16문항에 specific block이 있다. primary grid는 `121²=14641`이므로 `(item,g,h)` 조합 수는 `16×14641×121=28344976`이다. 현재 전체 범주 helper 호출 수는 `28344976×4=113379904`이고 후보는 28344976회다. 이는 소스로 계산한 호출 수이며 실제 allocator 호출 횟수나 시간 단축 측정이 아니다. table 길이와 약 907MB의 table 데이터는 줄지 않는다.

## 보존할 불변식과 실패 위험

- 자유 primary 차원 순서와 `prim += a_p[dim] * coords[...]`, specific 항의 덧셈 순서를 그대로 유지한다. FMA, BLAS, 벡터화, 좌표 재정렬을 도입하지 않는다.
- `grm_logprobs` 자체는 수정하지 않는다. 동일 base·thresholds의 전체 범주 결과를 한 번 계산해 복사하며 확률 floor, clipping, threshold 변환 또는 새 prior를 넣지 않는다.
- row layout `(g*h_count+h)*n_cat+cat`, specific-free `h_count=1`, mask·누락 응답, legal `-inf` zero mass와 전체 posterior mass 소실 오류를 유지한다.
- 전체 table은 매 E-step의 현재 모수로 새로 만든다. 이전 모수나 다른 device의 표를 캐시하지 않는다. GPU word arithmetic, buffer 수명, readback, CPU likelihood certification과 fail-closed 정책을 건드리지 않는다.
- 같은 helper를 써도 산술 재배치나 base의 부호가 달라지면 bit 동일성을 잃을 수 있다. 기존 scalar 호출 결과를 독립 oracle로 남겨 이를 검사한다. 중복 제거만으로 q121 메모리 문제나 340인 전체 fit 실행 가능성을 해결했다고 주장하지 않는다.

## 구현 배정 뒤 검증 계획

1. 작은 node 배열에서 기존 `item_cat_logprob`의 범주별 결과와 후보 표의 모든 원소를 `to_bits()`로 대조한다. cross-loading·음수 slope·specific-free/S0, 이진 및 4범주, 비대칭 primary 좌표, 엄격하지만 좁은 thresholds와 legal `-inf`를 포함한다. 새 code보다 먼저 실패하는 focused regression을 만든다. 오류/크기 guard를 완화하지 않는다.
2. 기존 q5 고정은행과 strict GPU 회귀를 같은 입력·예산으로 검사한다. posterior만이 아니라 기존 fixed-state counts·gradient·Newton 및 전체 sparse fixture의 수렴·모수 parity를 유지한다. 기존 19개 timing receipt와 5개 malformed receipt 거부도 유지한다.
3. 독립 소스 검토 뒤에만 새 source head와 fresh runner/owner/E·예산 guard에 묶인 q121/121 고정은행 1인 probe를 별도로 배정한다. input hash `5ae65a0d541523ac76e4e472eede55a33bc32298644f5f0cd9f4f0a26a403313`, 같은 1GiB/8GiB, counts 없음·모수 갱신 없음을 유지한다. 이전 실행 시간을 대체하지 않고 새 관측으로 보존한다.

전체 fit, 340인 throughput, 수렴·모수 갱신, peak RSS/VRAM, 적분 민감도와 불변 릴리스 수용은 별도 검증이다. 참가자 자료·원고 수치·IRB 값은 다루지 않는다. 기존 Cai(2010), pp.608–609 Appendices A/B의 posterior 계약을 바꾸지 않는 호출 중복 제거만 제안한다.
