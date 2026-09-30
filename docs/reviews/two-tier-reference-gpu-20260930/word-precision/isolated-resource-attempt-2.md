# 격리 adapter-only probe 두 번째 시도

## exact source와 실제 실행

- run: https://github.com/ContextualWisdomLab/fast-mlsirm/actions/runs/36702321423
- actual head: `b46ada286316e67c4be332c0c683f731312be514`
- job: `109844407650`, runner 이름 `cwlab-s1-gpu-01`, group `cwl focal gpu`
- reviewed/remote workflow SHA-256: `2b29658df447df18d864fd3fd8798d17b2a88c5e4361a351e2d1424dadd29a59`
- resource-only flag: `G1_GPU_RESOURCE_PROBE=1`
- job: 2026-09-30 19:25:33–19:33:49 KST, 설치·probe 본문 success, 모형 benchmark step skipped

GitHub 집계 success만으로 판정하지 않았다. exact head receipt, checkout SHA, 실제 runner 로그, 본문 및 내려받은 probe artifact의 결과를 따로 확인했다. 첫 attempt의 설치 실패와 구분하며, 19:29:59 KST의 조회403은 실제 job 실패가 아니었다. 그때 API 조회를 중단했고 단독 담당의 정상 통보 뒤 한정 회수했다.

## 실제 wgpu device 한도

본문은 NVIDIA GeForce GTX 1050 / Vulkan / DiscreteGpu를 기록했다.

| 값 | 실제 device 조회 |
|---|---:|
| max_storage_buffer_binding_size | 2147483644 bytes |
| max_buffer_size | 4292870144 bytes |
| max_storage_buffers_per_shader_stage | 524288 |
| max_compute_workgroups_per_dimension | 65535 |

probe는 `resource_probe_only=true`, `no model fit, no quadrature reduction, no numerical acceptance`를 출력했다. Rust 본문은1 passed/0 failed/0 ignored/2.29초였고 job의 grep 실행 확인도 통과했다.

**검증된 것은 해당 source에서의 격리 adapter 접근과 실제 device 한도뿐이다.** q121 table의 실제 할당·GPU 커널·전송/정규화 시간·모수 동등성·수렴·속도·정식 릴리스 수용은 아직 확인하지 않았다. 예상907MB table이 binding 한도보다 작다는 정적 비교는 실제 할당 성공이나 물리VRAM 여유 증명이 아니다. 호스트 Vulkan이나 M1 한도를 이번 격리 wgpu 결과로 대신 사용하지 않았다.

기존 E 프로세스는 실행 전 PID3974018로 살아 있음을 확인했고 종료·전환하지 않았다. runner 설정·권한·보안 정책·system dependency·모형·노드·prior·수렴 기준을 바꾸지 않았다. 추가 dispatch나 같은 run의 반복 조회는 하지 않았다.

## 보존한 원증거

- [run/job/step receipt](isolated-resource-attempt-2-receipt.json), SHA-256 `d9369513e97ff7cbcb82bf072dac6f7b281446f970b2243aec3f89629e9c376c`
- [실제 probe artifact](isolated-resource-attempt-2-probe.txt), SHA-256 `17b66dc935a9b6e18b44940cc32da647864ca66eba831eb73ba3674efc9f97eb`

원 job 로그의 전체 SHA-256은 `18e86e69dfa0ffaddb7e34e6a0fdd7b13f73eec8c2e59cbc0fd9c28803772b3f`이며 동일 세션 scratchpad의 `g1-resource-probe-attempt-2-job.txt`로 보존했다. 공개 근거에는 설치 실행의 무관한 전체로그 대신 직접 읽은 probe artifact와 terminal receipt를 넣었다.

다음 짧은 kernel/reference 검증은 이 adapter 한도와 caller 예산·host 가용량을 기준으로 별도 source/실행 소유/정밀도 조건을 검토한 뒤 진행한다. 큰 paired CPU q121 적합이나 node 상한·확률 floor·허용오차 완화로 대체하지 않는다. 연구 수치 HOLD·participant 접근 금지·IRB 값 금지를 유지한다.
