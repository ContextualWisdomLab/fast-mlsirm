# q121 고정은행의 표 중복 제거 후 재측정

## 실행과 판정 범위

검토한 source `4c09920d0e08f5fcbb8f68cab5ceb140d4c7af1b`를 정상 push한 뒤 같은 고정 합성은행 q121/121·1인 posterior E-step을 격리 러너에서 한 번 실행했다. 실제 GTX 1050 / Vulkan에서 **1 passed / 0 failed / 0 ignored**, 본문 39.63초로 통과했다. likelihood·평균·2차 적률·marginal SD의 CPU/GPU 차이는 모두 0이었다.

counts 수집·모수 갱신·전체 fit·수렴·340인 처리·bootstrap은 실행하지 않았다. 이 결과를 전체 G1 적합, 속도 보장, 필수 CI 전체 통과, 공식 승인, 불변 릴리스 또는 연구 수치 수용으로 확대하지 않는다.

| 실행 정체성 | 값 |
|---|---|
| run | [36786084195](https://github.com/ContextualWisdomLab/fast-mlsirm/actions/runs/36786084195) |
| job | [110127695987](https://github.com/ContextualWisdomLab/fast-mlsirm/actions/runs/36786084195/job/110127695987) |
| actual source | `4c09920d0e08f5fcbb8f68cab5ceb140d4c7af1b` |
| source parent | `a141ecbf97b2f6be56d5a09a09f2f230666bfc5e` |
| runner / group | `cwlab-s1-gpu-01` / `cwl focal gpu` |
| adapter | NVIDIA GeForce GTX 1050 / Vulkan / DiscreteGpu |
| job 시작–종료 | 2026-10-01 07:31:46–07:40:29 KST |

resource-only probe와 전체 fit benchmark 단계는 skipped였다. 테스트 exit나 job 이름만 보고 전체 적합을 실행했다고 판단하지 않았다. 새 dispatch는 한 번이며 재POST·기존 run 재실행은 없었다.

## 입력·소스·예산

[이전 시간 분리 실행](../q121-prepare-stages/README.md)의 source `3dc48dc28a8529f0a4542f95fbd85e91fcb69fde`, run `36773537763`과 입력 JSON·hash, adapter, q121/121, 1인/no-count/no-update, 예산과 19개 timing key가 같다. 입력 SHA-256은 `5ae65a0d541523ac76e4e472eede55a33bc32298644f5f0cd9f4f0a26a403313`이다. 참가자 파일과 seed는 없다.

| 컴파일 시 source hash | SHA-256 |
|---|---|
| `gpu_bifactor.rs` | `6f4cbb7ed77c33951165eb0038b2cf3ed9cd3dc3c3c71d842a58d3e2f164c532` |
| `two_tier_grm.rs` | `51c47ceba6cb5eee1ec33000722e417cd1336fc8dd4544faf768adc07ce2866b` |
| `two_tier_grm_tests.rs` | `ec6af054379fba49bf92af16765b5ba8d702cc9a7f17f1fbaac36b55ca750325` |
| 검토·원격 대조한 workflow | `a5473d49f694db8389596d1d9556d4263f0891c1aafbb5a7ea209c4082d10581` |

호출자 예산은 GPU 1073741824 bytes(1GiB), host 8589934592 bytes(8GiB)다. 본문의 source policy minimum은 GPU 1020771796 bytes, host source 추정은 4480932284 bytes로 이전과 같다. 실제 buffer binding 한도 2147483644 bytes와 buffer 한도 4292870144 bytes 안에서 수행했다. 표 데이터 크기를 줄인 변경은 아니다.

실행 전 허용된 정보로 지정 runner의 조직 소유 label·online/busy=false, 접근 가능한 cgroup 8GiB/current 1535270912 bytes, host 가용 23220285440 bytes, GPU 2048MiB/사용 28MiB/compute 활동 없음과 보호 E PID를 확인했다. 11개 대기 run의 실제 job label에서 알려진 호환 경합이 없었다. 이는 예약권이나 실행 중 peak 측정이 아니다. 다른 사용자 runner의 private cwd 접근은 권한 오류 뒤 중단하고 guard에서 제외했으며 sudo·Docker·다른 세션이나 인증으로 우회하지 않았다. 기존 정상 queue 경계를 유지하고 E·러너·권한 설정을 바꾸지 않았다.

## 같은 입력의 관측 비교

각 열은 다른 실행 시점의 한 번 관측이다. CPU cached+table 값은 별도 계측한 두 구간의 합이며, GPU 전체에는 host 표 생성·준비·정규화·적률·likelihood certification이 포함된다.

| 구간 | 이전 3dc (초) | 새 4c (초) |
|---|---:|---:|
| CPU cached E-step | 1.472282539 | 1.724850954 |
| CPU table 생성 | 43.273341845 | 12.268019805 |
| CPU cached+table | 44.745624384 | 13.992870759 |
| GPU 경로 전체 E-step | 51.081596032 | 16.495533761 |
| GPU 경로 host table 생성 | 44.548111046 | 11.686531938 |

node당 전체 범주 helper를 한 번 호출하는 변경과 함께 표 생성 관측 시간이 줄었다. source상 q121 helper 호출 수 113379904→28344976과 작은 실제 call-count 회귀는 [별도 구현 증거](../host-table-once-per-node/README.md)에 있다. 이번 q121 profile은 호출 counter를 활성화해 직접 횟수를 센 실행은 아니다.

**새 1인 전체 관측에서도 GPU는 CPU cached+table보다 느렸다.** 반복 측정·무작위 교차 배치·cold/warm 상태 통제나 대규모 throughput 검증은 없다. 전체 시간 변화 또는 아래 pipeline/map-copy 변화 모두를 helper 중복 제거의 효과로 단정하지 않는다.

## 새 실행의 19개 host-clock 구간

| key | 초 |
|---|---:|
| `table_generation_seconds` | 11.686531938 |
| `prepare_validation_metadata_seconds` | 0.376266986 |
| `prepare_error_scope_setup_seconds` | 0.000012889 |
| `prepare_uniform_buffer_seconds` | 0.000137654 |
| `prepare_table_map_copy_seconds` | 2.268082362 |
| `prepare_input_output_buffers_seconds` | 0.000315572 |
| `prepare_layout_bindgroup_seconds` | 0.124275572 |
| `prepare_shader_module_seconds` | 0.001481786 |
| `prepare_compute_pipeline_seconds` | 0.001365205 |
| `prepare_error_receipt_seconds` | 0.000005629 |
| `gpu_state_preparation_seconds` (parent) | 2.771964580 |
| `sweep_queue_write_encode_seconds` | 0.000057480 |
| `sweep_readback_buffer_preparation_seconds` | 0.019391801 |
| `sweep_submit_map_decode_seconds` | 0.690231433 |
| `sweep_error_receipt_seconds` | 0.000008182 |
| `gpu_log_products_and_readback_seconds` (parent) | 0.793093115 |
| `rust_lse_exp_normalization_seconds` | 0.290989411 |
| `posterior_moment_contraction_seconds` | 0.115618260 |
| `f64_likelihood_certification_seconds` | 0.722226811 |

19개 key가 각각 하나이고 유한·비음수임을 확인했다. 누락·중복·NaN·무한대·음수 receipt 거부 5건도 본문에서 확인했다. strict 1e-3 조건을 유지했고 실제 차이는 0이었다.

parent와 child는 중첩되므로 19개를 합산하지 않는다. clock·collector와 비활성 test counter의 오버헤드, driver의 지연 작업·실행 시점·환경 차이를 포함하는 host 관측이다. 순수 GPU/PCIe 시간이나 hardware timestamp가 아니다. native f64 GPU도 아니며 기존 GPU word product와 Rust f64 정규화·적률·certification 분담을 유지한다. peak RSS/VRAM 및 compiled remote binary·wheel hash는 별도로 측정하지 않았다.

## 원증거와 남은 수용

- [원 artifact](kernel-artifact.txt), SHA-256 `9c911dfc46c27269c15ba6a0b4f1dea50e84c42527d35798c6ccb3f7dcfa337f`
- [run/job/step receipt](run-receipt.json), SHA-256 `a14107f4efa176c77073878a62a5cf43fb8f031411b16440acf4650ca2f2c7cf`
- [새 이벤트·이전 timing·원증거 hash](verified-events.json)
- 전체 job 로그는 세션 scratchpad `host-table-q121-job.txt`에 보존했고 SHA-256은 `48c64a5149dfdd06e18d4ff017d5f795e03ac72e6cada3928f57e9aa6eea4f04`다. 실제 runner/group은 그 로그에서 대조했다. 전체 job 로그를 공개 문서에 복제하지 않았다.

artifact의 ANSI escape·끝 공백·빈 줄은 byte 그대로 보존한다. source 검토와 supplied-log 검토는 독립 GPU 재빌드·공식 승인·필수 CI 전체 수용과 구분한다. 원래 Cai(2010), 직접 읽은 pp.608–609 Appendices A/B의 posterior 계약을 바꾸지 않았다. 연구 수치 HOLD, 참가자 자료 접근 금지, IRB 값 금지를 유지한다. 이 문서 commit과 실행 source 4c는 별개다.
