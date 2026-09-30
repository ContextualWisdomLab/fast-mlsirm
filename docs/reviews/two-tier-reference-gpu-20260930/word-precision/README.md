# G1 정밀도 및 zero-mass 수리 검증

## 범위와 현재 판정

이 기록은 기존 [f32 실패 기록](../README.md)을 보존하고 후속 수리를 분리한다. 원격 #2296의 당시 head `c07f80bb8`과 로컬 검증 head는 같지 않았다. 작은 합성 입력의 모수 불일치는 아래 실행에서 해소됐지만, **q121/121·새 격리 s1·전체 CI·공식 비작성자 승인·불변 릴리스 수용은 아직 미완료**다. 연구 수치 HOLD와 참가자 자료·IRB 경계를 유지한다.

- 정밀도 코드: `6ac2622b12ca009dd2cd27bc0bcb5e67429c4f2a`, parent `c07f80bb83d9d4bdc3432634f3341c9f6d323e5d`
- zero-mass 호환 수리: `83084118a313978feee8ab72210389cf79bdd223`, parent `6ac2622b12ca009dd2cd27bc0bcb5e67429c4f2a`
- optional 수동 측정·미완료 receipt: `231b4ef11290fa46f74b963fdf7d660688f20e0a`, parent `83084118a313978feee8ab72210389cf79bdd223`; 이 변경은 수학 수리의 검토와 구분한다.

## 실제 연산 분담

원래 Rust f64 item/node table을 f32로 양자화하지 않고 두 u32 word로 GPU에 전달한다. 같은 부호 binary64 덧셈의 significand 정렬·guard/round/sticky·ties-to-even·carry·subnormal을 GPU 정수 연산으로 구현한다. 실제 GPU general/block log product를 반환받은 뒤 기존 Rust f64 LSE/exp 순서로 정규화하고 기대 빈도·적률·EAP를 합산한다. Newton M-step·원래 f64 likelihood 검증·모형·초기화·노드·prior·수렴 기준은 유지한다.

GPU는 native f64 연산을 하는 것이 아니며 전체 적합이 GPU 계산인 것도 아니다. 호스트 정규화는 명시적인 계산 분담이다. GPU 실패 뒤 CPU posterior를 재계산해 결과를 대체하는 fallback은 없다. 기존 focal/bifactor f32 경로는 바꾸지 않았다.

WGSL의 rounding mode·reassociation·FTZ 허용 때문에 단순 f32 보상합을 f64라고 보장하지 않는다. 두 word는 양자화된 입력을 사후 복원하는 것이 아니라 **원래 f64 비트를 처음부터 보존**한다. 지원 범위와 실제 adapter 검증을 넘어 임의 입력·모형·장치의 전체 정밀도를 보장하지 않는다.

## 첫 차이 발생 지점과 수리 효과

같은 합성 입력·seed·Φ=I·prior 없음·q7/7·tol1e-6에서 최초 모수 1e-3 기준 교차는 102→103회였다. 102회 최대 차이는0.000643, 103회0.001763이었다.

동일 CPU 모수 상태 102회를 고정한 분리 결과:

| 경로 | counts 최대 차이 | item13 gradient 차이 | 원래 Newton 갱신 차이 |
|---|---:|---:|---:|
| f32 table만 양자화, f64 sweep | 2.2548e-7 | 1.0880e-7 | 2.8285e-4 |
| 기존 실제 f32 GPU sweep | 2.1705e-6 | 5.5864e-7 | 2.4188e-4 |
| 새 GPU word product + 원래 f64 정규화 | 0 | 0 | 0 |

원래 ridge를 포함한 FD Hessian **대칭 부분** 고유값은 약 `[6.261, 2.295, 1.704, 1.374e-8, -3.257e-7]`이다. 작은·음의 곡률은 이 합성 상태의 진단이며 실제 비대칭 solver 행렬 전체의 조건수나 식별 실패 판정이 아니다. 새 ridge·prior·clipping·제약·허용오차를 추가하지 않았다.

## 같은 조건의 전체 실행

[hardware 및 build 원로그](hardware-build.txt)는 실제 Apple M1/Metal에서 1034 word bit 사례, 5개 당시 거부 경계, 전체 261회 궤적의 모수·likelihood 차이0, 최종 모수·EAP/SD 차이0, `1 passed / exit0`를 기록한다. 이 로그는 **6ac의 finite-only 단계**이며 아래 -inf 수리와 구분한다.

[정확한 clean 6ac benchmark](benchmark-6ac.json):

- CPython 3.12.14, macOS arm64, Rust dev 빌드
- core SHA-256: `e8d81f1ed85998ba73a79322ee897ef11bfca0c287c0f55e837c89052a43c7c6`
- 동일 입력 SHA-256: `5797fcdd50ac7f24d55fda5bd0c85728be0693382c6a7954549749f8013b20b3`
- 64명×16문항, P2/S4, q7/7, seed20260930, n_starts1, max_iter1000, tol1e-6, GPU 예산268435456 bytes
- CPU/GPU 모두261회 수렴; 모수6종·loglik·AIC/BIC·EAP·SD 차이0
- GPU 전송·호스트 정규화·모수 갱신을 포함한 전체 시간: CPU102.90초, GPU50.02초

이 시간은 한 합성 입력의 로컬 dev 빌드 측정이다. 다른 하드웨어·원고 설정·q121 속도나 가속 보장이 아니다. [bindings/group 원로그](bindings-group.txt)는 기존 bifactor/focal 및 기준→초점→채점3 passed, CPU golden·device 경계·출력보호10 passed를 기록한다.

## 독립 검토에서 확인된 -inf 차이의 RED→GREEN

6ac의 finite-only 입력 거부는 기존 CPU의 합법적인 zero-probability node를 모두 지원하지 못했다. 기존 `grm_logprobs`에 finite base1과 strictly ordered thresholds `[1e-18,0]`을 주면 중간 범주가 반올림으로 확률0/log=-inf가 된다. q3에서는 중심의 가능한 node가 있어 CPU likelihood가 finite인데 기존 GPU는 거부했다.

- [RED 원로그](zero-mass-red.txt): 실제 CPU가 수용한 q3/S0 사례에서 GPU Err, `1 failed / exit101`
- [GREEN 원로그](zero-mass-green.txt): -inf 흡수·1037 word bit 사례, NaN/+inf/양의 log3개 거부, finite 합산 overflow의 -inf zero-mass node1개 보존, 같은 finite GRM S0/S1×q3 허용/q2 전체 mass 소실 거부4개 사례, 전체261회 궤적·최종 모수·likelihood 차이0, `1 passed / exit0`
- [수리 commit·parent·3경로 source hash 및 RED/GREEN hash](zero-mass-source.json)

0 확률/-inf는 흡수 원소로 보존한다. NaN/+inf/잘못된 양의 log·전체 posterior mass 소실과 구분하며, 임의 floor·clamp·노드 삭제·저정밀 대체·CPU fallback은 없다. 이 국소 호환 수리의 독립 읽기 전용 검토는 실행을 새로 수행한 GPU 검토나 공식 GitHub 승인과 같지 않다.

## Word 사례·지원 조건

word bit 회귀는 LCG seed20260930의1024 random pair와 명시 edge 벡터를 사용한다. zero/음의zero, normal↔subnormal 및 최소 subnormal, exponent carry·경계, ties-to-even, 큰 exponent 차이, -inf 흡수가 포함된다. 완전한 벡터 생성·모형 fixture는 `tests/unit/two_tier_grm_tests.rs`와 기존 합성 JSON에 들어 있다. 지원하지 않는 NaN/+inf/양의 log를 거부하고 전체 mass 소실을 실패로 처리한다. 1037 통과를 모든 binary64 연산·모형·grid의 보장으로 확대하지 않는다.

버퍼는 원래 f64 요소당8 bytes를 사용한다. caller byte 예산과 adapter의 per-buffer·workgroup·index 범위를 검사하며 fixed input·현재 batch output·readback을 계산한다. 이것은 물리 VRAM·호스트 RSS·driver overhead 보장이 아니다. mapped table에 직접 원래 little-endian bit를 복사하며 호스트 전체 복제본은 만들지 않는다.

## q121 자원 계획 — 실측 아님

[source-derived 자원 계획](q121-resource-plan.json)은 340×16/P2/S4/4범주/q121/121에서 grid14641, person당 joint7086244, 한 table binding907MB, 1GiB GPU 예산에서 batch1을 계산한다. 호스트 table907MB와 count 값907MB 외에 node별 Vec header 추정약680MB가 더 필요하다. 로컬 M1은 16:20 KST free9%/wired약11.56GiB라 이 대형 적합을 시작하지 않았다.

현재 one-shot 구현을 batch마다 부르면 fixed table upload가 E-step당약308GB, readback약19.3GB, 호스트 정규화 exp 호출약4.83bn이다. 이는 소스의 반복 구조·바이트 계산이며 transfer/normalization 시간 실측이 아니다. fixed upload 재사용만으로 readback·호스트 정규화·M-step 병목까지 해소됐다고 선언하지 않는다.

다음은 실제 격리 adapter limit/가용량 probe → 짧은 kernel/reference 검사 및 transfer/정규화 시간 분리 → fixed input 재사용 검증 → 필요하면 정규화/집계 GPU 이관의 별도 정밀도 검토 → 연구 설정의 같은 조건 대조다. 큰 paired CPU q121 적합을 무작정 시작하거나 node 상한·확률 floor·기준 완화로 대신하지 않는다.

optional 수동 workflow는 기존 Statistical Studies의 기본5개 study 및 필수 PR CI·보안·pinned toolchain을 유지한다. caller가 모든 수치 인자를 지정하고, 정상 JSON의 두 수렴 결과·실제 GPU 기록 없이 성공하지 않는다. 미완료 receipt의 `running`/빈 runs/exit_status 부재는 완료 판정이 아니다. 이 경로의 실제 dispatch·S1 실행 결과는 아직 없다.

## 원로그·source 정체성

[source 및 lock/fixture hash](source-6ac.json)와 위 원로그를 보존한다. 독립 검토에서 읽은 source hash와 원로그 hash는 실행 원자료의 정체성을 검증할 뿐 재실행·다른 adapter·정식 릴리스 수용을 뜻하지 않는다.

Cai, L. (2010). A two-tier full-information item factor analysis model with applications. *Psychometrika, 75*(4), 581–612. https://doi.org/10.1007/s11336-010-9178-0 — 직접 읽은 pp.608–609 Appendices A/B의 원래 product/LSE/posterior contraction을 재사용했다.

Higham, N. J. (1993). The accuracy of floating point summation. *SIAM Journal on Scientific Computing, 14*(4), 783–799. https://doi.org/10.1137/0914050 — 전문을 읽고 pp.784–785/790–791의 합산 정밀도·rounding 가정과 p.797의 한계를 참고했다. 이 논문이 GPU 정수 에뮬레이션을 제안했다고 주장하지 않는다. IEEE binary64 형식에서 구현을 별도로 도출해 실제 GPU에서 대조했다. [저자 제공 전문](https://nhigham.com/wp-content/uploads/2023/10/high93s.pdf)은 재배포 권한을 확인하지 않아 PDF를 commit하지 않았다.

W3C. (n.d.). *WebGPU Shading Language*, §§15.7.2,15.7.4,15.7.5. https://www.w3.org/TR/WGSL/ — 원문을 읽어 f32 rounding mode 미지정·reassociation·FTZ 및 native f64와의 차이를 확인했다.
