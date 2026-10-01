# 격리 adapter probe 첫 시도의 설치 실패

## 실행 정체성과 판정

- 실행: https://github.com/ContextualWisdomLab/fast-mlsirm/actions/runs/36700555893
- actual head: `c64cde1225464be724277878df2ea4791753778f`
- job: `109838736284`, runner log 이름 `cwlab-s1-gpu-01`, group `cwl focal gpu`
- 사전 runner API 식별: ID1084343, online/busy=false, focal-gpu-isolated labels
- reviewed/local/remote workflow SHA-256: `f1bd2e8b215f255e7d94b729666dbc907081d88ac2512973b2822c26ecaf2cf2`
- 선택: `study=two-tier-reference-gpu`, `reference_resource_probe=true`. 모형 적합이나 수치 인자는 지정하지 않았다.

**설치 단계 exit1, adapter probe 본문 SKIPPED. 실제 adapter 이름·wgpu device limit은 얻지 못했다.** workflow failure를 GPU 연산 실패로 세지 않으며, artifact step의 success도 probe 증거 생성·수치 수용을 뜻하지 않는다.

원래 raw SHA를 dispatch ref로 지정한 요청은 HTTP422 No ref found로 실행을 만들지 못했다. 원격 branch가 exact c64임을 다시 확인하고 GitHub가 지원하는 branch ref로 요청한 한 실행만 실제 접수됐다. 옛 실행을 blind rerun하거나 중복 dispatch하지 않았다.

## RCA와 최소 수리

requirements/ci.txt의 해시 잠금 의존성을 project-local `.venv`에 설치했지만, PATH에 `.venv/bin`을 넣지 않았다. editable metadata의 maturin PEP517 backend가 `maturin pep517 write-dist-info` subprocess를 실행하면서 `FileNotFoundError: [Errno 2] No such file or directory: 'maturin'`로 실패했다. adapter 초기화는 시작하지 않았다.

설치 전용 commit `b46ada286316e67c4be332c0c683f731312be514`(parent c64)은 기존 focal-gpu-native 잡과 같은 `export PATH="$PWD/.venv/bin:$PATH"` 한 줄을 추가하고 누락을 검출하는 계약 검사 한 줄을 추가한다. 수리 전 해당 검사1 failed, 수리 후 workflow·실행 소유·pinned toolchain26 passed를 확인했다.

system dependency 설치·runner 설정·권한·보안 gate·모형·노드·prior·수렴 기준은 바꾸지 않았다. 이 로컬 수리 검사는 새 격리 실행의 성공이나 q121 수용이 아니다. 다음 attempt는 정상 push 및 exact source·runner idle·기존 E/다른 GPU 충돌·수신함·API 조건을 다시 확인한 뒤 별도 run으로 기록해야 한다.

## 보존한 원자료

- [원 job 로그](isolated-resource-attempt-1-job.txt), SHA-256 `6e03d5dc4703de4ec8c86a6bd2c74208c5b4f5fb24f432be8e4cfe7a890bdce1`
- [run/job/step terminal receipt](isolated-resource-attempt-1-receipt.json), SHA-256 `1d1f25892e9ed6ece9e841b9a6fa173321f37f5254a9310cd74da9b2d7bd54bd`
- [수리 exact source·parent·2파일·실패 근거 hash](isolated-resource-path-repair-source.json)
- [수리 후 경량 검사 로그](isolated-resource-path-repair-tests.txt), SHA-256 `1fbee58dab055071797fb259ac1b2157a18e651e0e141c4829ba14d36821be15`

원자료는 합성·설치·runner 실행 로그이며 participant 입력이나 E 결과를 포함하지 않는다. 기존 E PID3974018은 사전 확인 때 살아 있었고 종료·전환하지 않았다.
