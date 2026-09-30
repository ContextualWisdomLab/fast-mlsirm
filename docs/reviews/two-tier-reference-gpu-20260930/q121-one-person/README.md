# 격리 q121 고정은행 1인 E-step 검증

## 판정 범위

**같은 고정 합성은행의 q121/121·1인 posterior E-step은 실제 격리 GPU에서 통과했다.** likelihood·평균·2차 적률·marginal SD의 CPU/GPU 차이는 모두0이었다. 모수 적합·갱신·counts 수집·수렴·recovery·340인 처리·bootstrap은 실행하지 않았다. 이 결과는 G1 전체 적합 수용·속도 보장·필수 CI 전체 통과·공식 승인·불변 릴리스·연구 수치 채택을 뜻하지 않는다. 연구 수치 HOLD를 유지한다.

| 정체성 | 값 |
|---|---|
| 실행 | [36765438361](https://github.com/ContextualWisdomLab/fast-mlsirm/actions/runs/36765438361) |
| job | [110058309425](https://github.com/ContextualWisdomLab/fast-mlsirm/actions/runs/36765438361/job/110058309425) |
| actual head | `8a54da3c95f807c96176bb4b2074d523019c6c67` |
| 검토 base | `94f2197917dc9ed7b41360f85533b61823bdf425` |
| runner | `cwlab-s1-gpu-01`, group `cwl focal gpu` |
| 실제 adapter | NVIDIA GeForce GTX 1050 / Vulkan / DiscreteGpu |
| job 시작–종료 | 2026-10-01 04:24:30–04:36:37 KST |
| probe 본문 | 1 passed, 0 failed, 0 ignored; 109.94초; 성공 |

집계success만으로 판정하지 않았다. exactrun head·job/step receipt, actual runner 로그, 컴파일 시 source hash, 입력 JSON과 hash, 실제 adapter/예산, six timing stages, strict대조·테스트 본문을 각각 확인했다. resource-only와 전체fit benchmark는 둘 다 skipped였다. 재POST·중복 실행·source변경은 없었다.

## 입력·연산 계약

고정은행은 기존 합성 fixture와 같은16문항/P2/S4/4범주/Φ=I 지도·known loadings·thresholds를 사용하며 응답16개를 명시했다. seed나 참가자 파일이 없다. q-primary121, q-specific121의 모든 선언 GH 노드·가중치를 그대로 유지했다. 모수·prior·수렴 기준·허용오차·GPUword arithmetic을 바꾸지 않았고 node 축소·cap·확률floor·새ridge·CPUfallback은 추가하지 않았다.

- 입력 SHA-256: `5ae65a0d541523ac76e4e472eede55a33bc32298644f5f0cd9f4f0a26a403313`
- kernel source SHA-256: `6caa9b0024ec2ded5e2c1b0c731f831998fd975d2a25a8fd886872eb6c4c1d1a`
- numerical source SHA-256: `cbc74f27fa366e39062440846f3f0640359993b9e4bf29e0338a6ae5215a797c`
- test source SHA-256: `11999af60959c402e53619444ebfb7fb4321564ea4d0578e64d9cb5ee87cb5d4`
- reviewed remote workflow SHA-256: `a5473d49f694db8389596d1d9556d4263f0891c1aafbb5a7ea209c4082d10581`

GPU는 원래 f64 table 비트를 u32 두 word로 전달받아 binary64 log-product 덧셈을 정수로 에뮬레이션한다. 실제 GPU product의 LSE/exp/log 정규화·적률·likelihood certification은 Rust CPU가 수행한다. native f64 GPU나 전체 GPU 연산이 아니다. CPU reference는 같은bank/node의 cachedtable을 사용하는1인 E-step으로 제한했다. compiled 실행파일·wheel의 binaryhash는 이번 probe에서 별도로 측정하지 않았다.

## 실제 budget 및 수행 근거

caller가 명시한 값은 GPU **1073741824 bytes(1GiB)**, host **8589934592 bytes(8GiB)**다. 1GB와 혼동하지 않는다. 실제 본문이 기록한 sourcepolicy minimum은1020771796bytes이며 caller의여유는52970028bytes였다. hostsource추정은4480932284bytes다.

primary grid14641, itemtable907039232bytes, 1인 rawoutput56807080bytes의 계산과 생성/sweep/readback·정규화 성공은 이번 실제 수행에서 확인됐다. 그러나 peakRSS/물리VRAM/driverstaging의시간별최대치는 측정하지 않았다. source-derived예산을 hardRSS상한이나모든grid의실제할당보장으로해석하지 않는다.

사전 actualisolatedcgroup한도8GiB/current394125312bytes와host가용약21GiB, GTX1050전체2GiB/사용28MiB/compute앱0·유휴러너를 확인했다. 이사전관측을실행중peak로대신세지않았다. 실제device의maxbinding2147483644/maxbuffer4292870144를사용했으며호스트Vulkan또는M1limit으로대신판정하지않았다. 기존 E PID3974018은사전확인때살아있었고종료·전환하지않았다.

## 같은 조건 시간 실측

모든시간은 hostclock이며 GPUpreparation에는pipeline생성과upload가묶이고 GPUproduct/readback에는compute·copy/map·hostdecode가묶인다. 순수PCIe시간이나hardwaretimestamp가아니다. 측정은1회다.

| 구간 | 초 |
|---|---:|
| CPU cached E-step | 1.413817148 |
| CPU table 생성 | 43.106984244 |
| CPU table 포함 전체 | 44.520801392 |
| GPU 전체 E-step | 60.730846804 |
| GPU 경로 host table 생성 | 43.824468356 |
| fixed GPU state 준비·upload/pipeline | 14.377435620 |
| 실제 GPU product·readback | 1.337327432 |
| Rust LSE/exp 정규화 | 0.279993637 |
| posterior 적률 합산 | 0.108341023 |
| f64 likelihood certification | 0.627493957 |

**이1인 전체 측정에서는 GPU경로가 더 느렸다.** 가속완료나대규모throughput을주장하지않는다. table생성과state준비비용이본실측의큰비중이었다. batch재사용으로고정tableupload를줄였다는국소검증과340인의readback/정규화/M-step확장비용은별개다. fixedpoint1회결과를전체fit시간으로외삽하지않는다.

## 정밀도·거부·미수용

strict1e-3회귀조건을완화하지않고실제차이가0인지확인했다. 필수6timing키는각1개, 유한·비음수이고output길이·finite값·variance를검사한다. `collect_counts=false`, `parameter_updates=false`다.

작은q5입력·예산5개negativecontrol과CPU-only/coverage컴파일증거는local단계로별도보존했다. 이번q121긍정사례가임의입력·q241·다른device의거부/정밀도/적분민감도보장을뜻하지않는다. G1의같은입력전체수렴·모수동등성·fit상태·성능수용및G2/G3는여전히미완료다.

## 원증거

- [실제 kernel artifact](kernel-artifact.txt), SHA-256 `6fd0e5f3fd2f6ba87266c1e90e876c5d515f8c0f59d822acd14596a2d24966b1`
- [exact run/job/step receipt](run-receipt.json), SHA-256 `26ac35677f7948c1de1672dd38bd869223138e0873d6d7c8c54632fed574f6f4`
- [추출·대조한 events와source/loghash](verified-events.json)
- 전체job로그SHA-256 `b2ff1edaaeb037423d8f1c9312bb7e025304b9bba6a78cd9d72aedcc52dd4727`는세션scratchpad `q121-isolated-kernel-job.txt`에보존했다. 공개근거에는직접읽은실제artifact와receipt를넣었다.

## 재현 명령

```bash
gh workflow run statistical-studies.yml \
  --ref feat/two-tier-reference-gpu-20260930 \
  -f study=two-tier-reference-gpu \
  -f reference_resource_probe=false \
  -f reference_kernel_probe=true \
  -f 'reference_args=--q-primary 121 --q-specific 121 --gpu-memory-budget-bytes 1073741824 --host-memory-budget-bytes 8589934592'
```

위명령의실제접수시remotehead가8a54였음을확인했다. 재실행때는현재branchidentity/검토source·idle/resource/권한·APIguard를다시확인해야하며이기록을새실행허가로사용하지않는다.

논문basis는기존문서의Cai(2010), 직접읽은pp.608–609 Appendices A/B의동일posterior E-step이다. 이번추가는모형식이아닌동일계약의측정·검증이다. participant자료·개인별결과·원고수치·IRB값은없다.
