# Counts·M-step 분리 검증 제안

## Motivation

검토 기준은 문서 head `281631b72e409c6322c05e8cdf21fb464ef2d4f1`, 계산 소스 `4c09920d0e08f5fcbb8f68cab5ceb140d4c7af1b`다. 이번 작업은 소스 읽기와 정수 크기 계산뿐이다. production 코드·API·워크플로를 수정하거나 fit·GPU dispatch를 실행하지 않았다.

[q121 재측정](q121-table-once/README.md)은 같은 고정은행의 `counts=false`, 1인 moments-only 검사다. 모수 갱신·전체 수렴을 검증하지 않는다. 그 실행의 독립 offline receipt 대조는 별도 진행 중이며, 이 설계의 완료로 대신하지 않는다.

목표는 현재 계산 담당과 배열 수명을 확인하고 **고정 모수 E-step 통계 → 고정 counts의 item 갱신 → 작은 공개 fit**을 독립 검증 단위로 나누는 것이다. 새 GPU kernel이나 추정기 변경을 먼저 제안하지 않는다.

## 현재 공개 API와 호출 순서

| 경로 | 현재 순서·계산 담당 | 근거 |
|---|---|---|
| Two-tier 기준 fit | Python 입력·mask → PyO3 소유 버퍼 → Rust multi-start → 매 반복 E-step counts/cross → 기존 CPU item Newton → winning 모수의 final EAP → 출력·반사 식별 | `python/fast_mlsirm/two_tier_grm.py:174`, `crates/fast-mlsirm-py/src/lib.rs:1927`, `two_tier_grm.rs:1950,2167,2264` |
| Two-tier grouped pipeline | reference group의 기준 fit을 먼저 수행 → reference부터 group별 처리 → focal은 고정 bank의 Gaussian mean/SD fit → 각 group의 별도 score 노드 → 공통 기대 원점수 곡선 → 원래 row slot 반환 | `python/fast_mlsirm/two_tier_focal.py:375–487` |
| Bifactor 단일 fit | 현재 item table → device E-step counts → CPU item Newton·기존 slope prior/objective → 반복 → winning 모수의 CPU final EAP | `bifactor_grm.rs:691,1114–1438` |
| Bifactor 다집단 | 현재 group mean/SD·specific SD로 node 재생성 → group별 table/counts·moment E-step → anchor는 group 바깥/node 안쪽 순서로 pooling하여 CPU Newton, free item은 group별 Newton → 기존 분포 갱신 | `bifactor_grm.rs:2081,2360–2533` |
| Bifactor FIPC | Python/PyO3/config에 device 인자 없음 → 현재 focal node의 `e_step_multigroup(...Device::Cpu)` → free item만 CPU Newton, anchor 입력은 고정 → 기존 focal 분포 갱신 | `python/fast_mlsirm/bifactor_grm.py:686`, PyO3 `lib.rs:1792`, `bifactor_grm.rs:3196,3435–3523` |

Two-tier 공개 GPU reference fit은 explicit `cpu/gpu`, 양수 `gpu_memory_budget_bytes`, Φ=I를 요구한다. `auto`와 GPU correlated-primary fit은 거부한다. Python binding의 기존 Newton 제어는 `newton_iter=10`, `ridge=1e-8`이며 새 prior가 아니다. 현재 공개 함수에 host budget 인자는 없다. 고정은행 profile의 host budget을 공개 fit에 적용되는 상한으로 표시하면 안 된다.

공개 결과의 `category_counts[J,K]`는 관측 빈도다. 내부 E-step의 node별 expected counts와 다른 배열이므로 이 출력만으로 expected-count 검증을 대신하지 않는다.

Bifactor 다집단의 anchor는 group들 사이에서 공유하며 pooled counts로 추정하는 item이다. FIPC anchor는 입력 모수에 고정하는 item이다. 같은 이름을 이유로 두 갱신 계약을 서로 대체하지 않는다.

Grouped two-tier의 reference E-step은 word-preserving 경로지만 focal Gaussian fit·score는 기존 f32 GPU posterior 경로다. 같은 `device='gpu'` 전달을 전 구간의 동일 정밀도 구현으로 해석하지 않는다. FIPC의 device 전달은 G2의 별도 변경이며 이번 설계에서 구현하지 않는다. 기존 FIPC node 기본값도 바꾸지 않고, 아래 합성 검사는 node 수를 명시한다.

## G1의 실제 CPU/GPU 분담

`two_tier_grm.rs:1158–1572`의 reference-precision 경로를 기준으로 한다.

1. Rust CPU가 현재 모수의 f64 item table을 새로 만든다(`:1316`).
2. `GpuLogProductState`가 이 table/prior의 f64 word를 보존하고, GPU가 general/block log-product를 계산한다. 한 E-step 안에서 person batch에 고정 자원을 재사용한다.
3. readback을 Rust가 f64로 decode한다. Rust CPU가 기존 순서의 LSE/exp 정규화를 수행한다(`:1414–1463`).
4. `statistics=Some`이면 CPU가 nested counts와 `P×P` cross를 먼저 할당한다(`:1331–1337`). person → latent dimension → node → 관측 item 순서로 posterior를 누적한다(`:1479–1531`). 이는 GPU counts kernel이 아니다.
5. 기존 f64 CPU reduction은 `collect_counts=false`로 likelihood를 인증한다(`:1547–1566`). GPU-derived 값과 bit 일치를 검사한다. 실패 뒤 CPU posterior를 재계산해 대체하는 fallback과 다르다.
6. `run_single_start`는 `moments=None, statistics=Some, reference_precision=true`로 E-step을 호출한다. counts를 받은 뒤 Rust CPU가 기존 `item_neg_ll_grad`·FD Hessian·ridge·backtracking의 `m_step_item`을 호출한다(`:1990–2074`, `:1606–1764`). winning final EAP는 반대로 moments만 수집하고 counts는 없다(`:2278–2304`).

현재 19개 profile 키의 contraction 구간은 statistics가 켜지면 counts 작업도 포함한다. 반면 counts allocation은 GPU preparation clock 시작 전에 일어난다. 따라서 moments-only의 19개 구간만으로 counts allocation이나 Newton 시간을 분리했다고 주장할 수 없다.

## Shape와 배열 수명

기호는 persons `N`, items `J`, primary dimensions `P`, specific blocks `S`, categories `K`, primary grid `G=q_primary^P`, specific nodes `H=q_specific`, 실제 batch `B`다. item별 node 수 `L_i`는 block item이면 `GH`, specific-free이면 `G`다.

| 배열 | shape·payload | 위치·수명 |
|---|---|---|
| 응답·mask | `N×J` usize/bool, device staging은 i32 | 입력 보유; batch별 slice |
| primary 좌표·prior | `G×P`, `G`, `H` f64 | fit 공통 좌표/현재 density; 원래 prior 유지 |
| item tables | item별 `L_i×K`, 총 `8KΣL_i` bytes | host f64 contiguous table; E-step마다 현재 모수로 새 생성 |
| word product·posterior | general `B×G`, block/joint `B×S×G×H` f64 | device/readback 및 host batch별 배열 |
| expected counts | `Vec<Vec<Vec<f64>>>`, item별 `L_i×K` | host만; E-step 통계 누적 후 item M-step까지 유지, N에 비례해 크기가 늘지는 않음 |
| cross | `P×P` f64 | host 통계; Φ=I라도 현재 함수는 수집, 임의 생략하지 않음 |
| moments | mean·second 각각 `N×(P+S)` f64 | 요청한 경우 host output |
| item Newton | 자유 primary + optional specific + `K−1` 모수, Hessian은 그 크기의 제곱 | host만; 현재 item의 counts를 참조 |

Bifactor는 primary가 하나이고 `G=q_general`이다. 다집단은 group 수 `A`에 따라 table/counts 한 벌이 group별로 생긴다. GPU f32 counts의 flat stride는 `q_general×q_specific`이고 general-only item의 쓰지 않는 tail도 stride에 포함된다(`gpu_bifactor.rs:87–103,463–485`). anchor M-step은 group별 counts를 clone해 쌓는 별도 host 배열과 node 좌표가 추가된다(`bifactor_grm.rs:2462–2511`).

### q121/P2/S4/J16/K4의 source 계산

64-bit 환경의 f64=8 bytes, Vec header=24 bytes를 사용한 **payload/header 계산**이다. allocator metadata·fragmentation·driver staging의 실측 상한은 아니다.

| 항목 | bytes |
|---|---:|
| `G=14641`, `H=121`, 총 item node `ΣL_i=28344976` | — |
| table 또는 counts 숫자 payload 한 벌 | 907039232 |
| counts의 nested Vec header 한 벌 | 680279832 |
| counts payload+header 한 벌 | 1587319064 |
| CPU/GPU oracle counts 두 벌 동시 보유 | 3174638128 |
| reference word 경로 fixed device policy | 907157572 |
| person 1명의 buffer/readback policy 추가 | 113614224 |
| caller 1GiB에서 source policy가 허용하는 batch | 1명 |

중요한 정정: 기존 moments-only profile의 host 추정 **4480932284 bytes에는 실제 수집하지 않은 counts payload/header 한 벌도 이미 들어 있다**(`tests/unit/two_tier_grm_tests.rs:362–419`). counts가 추정에서 빠졌다고 해석하지 않는다. 그러나 그 한 벌의 실제 할당·누적을 실행한 증거는 아니다. CPU/GPU counts를 동시에 보유하는 비교는 두 번째 counts 한 벌만 더해도 source subtotal이 6068251348 bytes가 된다. 아직 allocator overhead·기존 cgroup 사용량·계측을 합친 실행 가능성을 보장하지 못한다.

사람 수만 줄여도 counts node 배열 자체는 작아지지 않는다. q121의 1인 counts 검사를 moments-only처럼 가벼운 검사로 세면 안 된다. q241/P2/S4의 table payload만 7166730752 bytes, counts payload+header 한 벌만 12541779224 bytes다. 현재 1GiB device policy와 GTX1050 buffer 한도에서 같은 단일 table 경로로 수행할 수 없다. 실패를 node cap·축소·floor·CPU fallback으로 바꾸지 않는다.

## Proposal: 기존 helper로 검증 단위를 나누기

아래는 후속 배정 뒤의 계획이며 지금 실행하지 않았다.

### 1. 작은 고정 모수 counts 검사

기존 known bank 또는 이미 검증된 sparse fixture·map을 재사용한다. caller가 명시한 작은 q5/q7은 구조 검사용이며 연구 q121 수용을 대신하지 않는다. 공개 calibration은 미관측 category를 거부하므로 1인 known-bank helper 검사와 공개 fit 입력을 혼동하지 않는다.

- GPU reference의 `statistics=Some`, `moments=None`와 CPU `e_step_with_moments(collect_counts=true)`를 같은 모수·GH 좌표/prior로 대조한다. CPU streamed/cached 둘도 대조해 table indexing oracle을 분리한다.
- 기존 fixed-state 검증(`tests/unit/two_tier_grm_tests.rs:1179–1283`)의 exact/quantized/f32/word 구분을 재사용한다. shared code 한 번을 양쪽에 호출한 것만 독립 검증이라고 부르지 않는다.
- 모든 item/node/category counts·cross·likelihood의 실제 최대 차이와 bit 일치 여부, 관측 category별 mass residual을 기록한다. 기존 strict 조건을 완화하거나 새 과학적 수용 기준을 넣지 않는다.
- S0·P1·cross-loading·음수 slope·specific-free, 일부 cell/전체 block/전체 person missing, 작은 budget에서 full/short batch, 새 모수의 다음 E-step을 분리한다. missing 응답의 counts는 추가되지 않아야 한다.
- malformed 범주·map·shape·budget, CPU software adapter, GPU unavailable/dispatch failure와 posterior mass 전체 소실의 현재 거부 경계를 확인한다. private CPU helper가 tuple/nonfinite를 반환하는 경우와 공개 fit의 오류 처리를 구분한다. legal `-inf`는 임의 제거하지 않는다.

### 2. 같은 counts에서 item 갱신 검사

같은 packed 모수·free-map·좌표·기존 `newton_iter`/`ridge`로 CPU와 word counts 각각을 현재 `item_neg_ll_grad`·`m_step_item`에 넣는다. 기존 item13 fixed-state gradient/Newton 대조를 재사용하되 실제 objective·gradient·step/갱신값을 기록한다. 작은 fixture의 독립 objective finite-difference 대조를 별도로 사용하고, FD Hessian의 일부 진단을 전체 solver 식별·조건수 증명으로 쓰지 않는다.

negative case는 zero counts와 legal zero probability, 아주 좁은 threshold, 비유한 objective/gradient, backtracking 미수용, 거의 평평한 counts다. 소스의 objective에는 `count * log_prob` 항이 있고 Newton은 nonfinite면 break한다. `0×(-inf)`와 변경 없는 반환을 어떻게 기록할지 작은 재현으로 먼저 확인해야 하며, 반환값이 같다는 이유로 성공한 갱신·수렴으로 표시하지 않는다. 이것은 아직 실행하지 않은 위험 검증 항목이지 새 확정 결함 또는 수정 배정이 아니다.

현재 gradient 계산 안에는 기존 GRM 확률 helper를 다시 호출하는 연산도 있다. 이번에는 측정 대상의 현황으로만 기록하며 algebra·gradient·solver를 최적화하거나 수정하지 않는다.

### 3. 작은 공개 fit 경계 검사

이미 있는 합성 fixture의 category coverage·초기화·seed·starts·Φ=I·tol·max_iter를 그대로 사용해 입력/반환 경계를 별도 검사한다. 한 번의 M-step이나 max_iter 종료를 수렴으로 부르지 않는다. start 선택, all-failed 오류, nonconvergence, final EAP의 actual adapter와 grouped pipeline의 group failure/row slot 보존을 확인한다.

공개 CPU reference fit은 기본 streamed E-step이고, 앞선 1인 비교의 CPU cached oracle과 다르다. 향후 전체 fit 시간 비교는 같은 공개 API와 caller controls를 사용하고 이 차이를 보고해야 한다. grouped pipeline의 focal f32 정밀도와 FIPC/G2는 별도 검증이다.

### 4. q121 counts·M-step 측정 전 선행조건

작은 검사·독립 source 검토 뒤에 별도 실행을 배정한다. 먼저 prospective counts allocation과 CPU oracle 동시 보유 여부를 포함한 source-shaped host manifest를 확정한다. 가능하면 비교값을 순차 검증해 두 대형 counts를 불필요하게 동시에 잡지 않되, 사전에 설계한 독립 대조를 없애거나 순서를 바꾸지 않는다.

host/device 실제 예산, 현재 cgroup 사용량·GPU 활동·intended runner/정상 queue·E 보존·중복 run·exact source를 다시 확인한다. 다른 사용자 private cwd·권한 우회는 제외한다. counts allocation/CPU contraction/item objective·gradient·Newton 시간과 소유 job의 RSS·device 사용량을 관측하는 방법을 먼저 정한다. 기존 19 parent/child 키를 합산하거나 과거 cgroup high-water를 이번 run peak로 표시하지 않는다. 측정할 수 없는 값은 미측정으로 남긴다.

공개 API의 device budget만으로 host 안전성을 보장할 수 없으므로 자원 관측·외부 job 경계가 확정되지 않으면 q121 counts/M-step은 실행하지 않는다. 340인·전체 fit·bootstrap은 이 단위 뒤 별도 자원·정밀도·수렴 검증이다.

## Alternatives와 Non-goals

- 권장: 기존 CPU f64 oracle, word-product E-step, CPU count/gradient/Newton을 분리 검증한다. 가장 작은 변경으로 현재 계약을 확인할 수 있다.
- 이번에 제외: Bifactor f32 counts kernel을 G1에 그대로 이식. 다중 primary/cross shape와 정밀도·CPU certification 계약이 다르며 기존 f32 모수 divergence를 다시 만들 수 있다.
- 이번에 제외: counts flat화, 병렬화, table/pipeline cache, M-step GPU화, allocator 교체, 새 host-budget 공개 API. 실측과 별도 검토 전 production 변경으로 들어가지 않는다.
- 금지: node/prior/criteria 변경, floor·새 ridge·무음 CPU fallback, 큰 CPU fit·새 q121 모수 fit·340인·bootstrap·GPU dispatch, 참가자 자료·원고 수치·IRB 값·E 또는 runner 권한 변경.

기존 posterior와 expected-count 근거는 직접 읽은 Cai(2010), pp.608–609 Appendices A/B다. 여기서는 기존 item Newton 구현의 정합 검증을 설계할 뿐 새 추정기를 제안하지 않는다. FIPC는 현재 소스의 호출 경계만 확인했으며 새 방법 구현이나 해석을 제안하지 않는다.

참고문헌: Cai, L. (2010). A two-tier full-information item factor analysis model with applications. *Psychometrika, 75*(4), 581–612. https://doi.org/10.1007/s11336-010-9178-0.
