# q121 고정은행 1인 E-step 준비 비용 분리

## 확인한 범위

같은 고정 합성은행의 q121/121·1인 posterior E-step이 NVIDIA GeForce GTX 1050 / Vulkan에서 통과했다. CPU/GPU likelihood·평균·2차 적률·marginal SD 차이는 모두 0이었다. 테스트는 1 passed, 0 failed, 0 ignored로 끝났으며 본문 실행 시간은 107.19초다.

counts 수집, 모수 갱신, 전체 fit 수렴, 340인 처리, bootstrap은 실행하지 않았다. 이 기록은 G1 전체 적합 수용, 필수 CI 전체 통과, 공식 승인, 불변 릴리스 또는 연구 수치 채택의 근거가 아니다. 연구 수치 HOLD를 유지한다.

| 실행 정체성 | 값 |
|---|---|
| run | [36773537763](https://github.com/ContextualWisdomLab/fast-mlsirm/actions/runs/36773537763) |
| job | [110085654900](https://github.com/ContextualWisdomLab/fast-mlsirm/actions/runs/36773537763/job/110085654900) |
| actual source head | `3dc48dc28a8529f0a4542f95fbd85e91fcb69fde` |
| source parent | `4b3ec892bfd241eb4b133414102564fdf3cdc9e0` |
| runner | `cwlab-s1-gpu-01`, group `cwl focal gpu` |
| adapter | NVIDIA GeForce GTX 1050 / Vulkan / DiscreteGpu |
| job 시작–종료 | 2026-10-01 05:34:14–05:45:20 KST |

원로그의 source·resource·input·timing·comparison 이벤트와 실제 실행한 테스트 본문을 대조했다. resource-only probe와 전체 fit benchmark 단계는 skipped였다. 성공한 job 이름에 benchmark가 들어 있어도 전체 적합을 수행한 것으로 세지 않는다.

## 입력·예산·소스

입력은 16문항/P2/S4/4범주/Φ=I의 고정 합성은행과 명시한 응답 16개다. seed와 참가자 파일은 없다. q-primary=121, q-specific=121, GPU 예산=1073741824 bytes(1GiB), host 예산=8589934592 bytes(8GiB)를 호출자가 지정했다.

| SHA-256 | 값 |
|---|---|
| 입력 | `5ae65a0d541523ac76e4e472eede55a33bc32298644f5f0cd9f4f0a26a403313` |
| `gpu_bifactor.rs` | `6f4cbb7ed77c33951165eb0038b2cf3ed9cd3dc3c3c71d842a58d3e2f164c532` |
| `two_tier_grm.rs` | `5379799d96c7bb49664d256929b1070c2d27b2b3aa7d2997e6517ead90858c70` |
| `two_tier_grm_tests.rs` | `3a2ca74ba25e7fcab6ca91429847f70d55e2c15dbd97289ff21d625ca44d6497` |
| 검토한 workflow | `a5473d49f694db8389596d1d9556d4263f0891c1aafbb5a7ea209c4082d10581` |

본문의 source-derived host 추정치는 4480932284 bytes, GPU policy minimum은 1020771796 bytes다. 실제 adapter의 buffer binding 한도는 2147483644 bytes, buffer 한도는 4292870144 bytes였다. 예산은 source 정책이고 adapter 한도는 buffer별 제약이다. peak RSS, 물리 VRAM 가용량, driver staging의 최대치를 측정하거나 보장한 값이 아니다.

이번 소스 변경은 test-only host clock 계측이다. 원래 probability/product 수식, 산술 순서, table·buffer 갱신, 오류 거부, 모형, 노드, prior, 허용오차를 바꾸지 않았다. pipeline 캐시나 병렬화를 구현하지 않았다. GPU가 원래 f64 table의 두 u32 word로 log product를 계산하고, Rust CPU가 f64 정규화·적률·likelihood certification을 수행하는 기존 연산 분담도 유지했다. native f64 GPU 또는 전체 GPU 계산이 아니다.

## 시간 실측

| 전체 비교 구간 | 초 |
|---|---:|
| CPU cached E-step | 1.472282539 |
| CPU table 생성 | 43.273341845 |
| CPU table 포함 전체 | 44.745624384 |
| GPU 경로 전체 E-step | 51.081596032 |

**이 1인 측정에서도 table을 포함한 GPU 경로가 CPU보다 느렸다.** 아래 19개 키는 parent와 child를 모두 기록한다. child는 parent 구간의 일부이므로 19개를 합산하지 않는다.

| 계측 키 | 초 |
|---|---:|
| `table_generation_seconds` | 44.548111046 |
| `prepare_validation_metadata_seconds` | 0.329279417 |
| `prepare_error_scope_setup_seconds` | 0.000012667 |
| `prepare_uniform_buffer_seconds` | 0.012515694 |
| `prepare_table_map_copy_seconds` | 3.054670883 |
| `prepare_input_output_buffers_seconds` | 0.000319340 |
| `prepare_layout_bindgroup_seconds` | 0.439176765 |
| `prepare_shader_module_seconds` | 0.001481095 |
| `prepare_compute_pipeline_seconds` | 0.319008588 |
| `prepare_error_receipt_seconds` | 0.000007525 |
| `gpu_state_preparation_seconds` (parent) | 4.156495695 |
| `sweep_queue_write_encode_seconds` | 0.000060839 |
| `sweep_readback_buffer_preparation_seconds` | 0.034256581 |
| `sweep_submit_map_decode_seconds` | 0.764564007 |
| `sweep_error_receipt_seconds` | 0.000003884 |
| `gpu_log_products_and_readback_seconds` (parent) | 0.832076767 |
| `rust_lse_exp_normalization_seconds` | 0.299258353 |
| `posterior_moment_contraction_seconds` | 0.120244594 |
| `f64_likelihood_certification_seconds` | 0.917340135 |

19개 키가 각각 한 번 존재하고 모두 유한·비음수임을 확인했다. 누락·중복·NaN·무한대·음수 기록을 거부하는 5개 negative case도 본문에 명시돼 있다. 기존 strict 1e-3 조건을 완화하지 않았으며 실제 차이는 모두 0이었다.

가장 큰 구간은 host table 생성 44.548초다. 준비 구간에서는 table map/copy 3.055초가 크고, layout/bindgroup은 0.439초, compute pipeline은 0.319초였다. 이 측정만으로 pipeline 캐시를 우선 구현할 근거는 부족하다.

이는 clock·collector 오버헤드를 포함한 host 시간이다. driver의 지연 작업은 submit 구간에 나타날 수 있다. 순수 GPU 실행 시간, hardware timestamp, 분리한 PCIe 전송 시간을 주장하지 않는다. [이전 실행](../q121-one-person/README.md)의 준비 14.377초와 이번 4.156초는 소스·계측·실행 시점과 driver 상태가 다른 관측이다. 이번 세부 구간으로 이전 14.377초의 원인을 소급 설명하거나 성능 개선을 주장하지 않는다.

## 보존한 증거

- [원 kernel artifact](kernel-artifact.txt), SHA-256 `aa9992a068f4ecfd75fa4ab3e1fd43fee066e49561af09763f61699bbd7dfda6`. ANSI escape, 행 끝 공백과 마지막 빈 줄도 원본 그대로 보존했다. 이 파일의 `git diff --check` 공백 경고를 없애려고 증거를 정규화하지 않았다.
- [run/job/step receipt](run-receipt.json), SHA-256 `340860a9b8da2e0ed50f5576a469f48ea0a88a7254ad11e423be429cf51c63d5`
- [대조한 이벤트](verified-events.json): 원로그 이벤트와 원증거 해시. 해시 목록의 경로만 보관 파일명으로 바꿨다.
- 전체 job 로그 SHA-256 `2df3647e0fa01cfa984d925d6d0e9251175a825a519ae13d4999148077d97853`는 세션 scratchpad의 `prepare-stage-q121-job.txt`에 보존했다. 전체 job 로그를 이 문서에 복제하지 않았다.

compiled test binary나 wheel의 해시는 이 원격 probe에서 별도로 측정하지 않았다. 소스·입력 해시를 binary 해시로 대신 표시하지 않는다. 독립 source/log 읽기 검토는 독립 GPU 재빌드, GitHub 승인 또는 연구 수용과 구분한다.

논문 근거와 연산 계약은 [기존 q121 기록](../q121-one-person/README.md)의 Cai(2010), 직접 읽은 pp.608–609 Appendices A/B를 따른다. 이 문서는 같은 계약의 계측 증거만 추가한다. 참가자 자료, 원고 수치, IRB 값은 없다.
