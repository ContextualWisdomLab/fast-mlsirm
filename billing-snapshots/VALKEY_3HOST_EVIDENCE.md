# Valkey 3-host evidence (2026-09-20)

**LABEL: communications/transport race test — NOT psychometric distributed compute.**

## Artifacts
- `valkey_3host_air_s1_m1_20260920.json` — Air+s1+M1 distribute/dedup/aggregate (n=36)
- `valkey_fail_retry_20260920.json` — deliberate crash→XAUTOCLAIM→HSETNX retry
- `valkey_daemon_evidence_20260920.txt` — local 9/9 protocol suite vs disposable Redis
- `host_budget_mba_20260920.txt` — host disk/RAM notes

## Library SHA
Contract under test: PR #2073 / `74ecd9f90cdb2675ae4c858d07200d6edb7ce3a4` (Valkey Streams + committed hash).
Race runtime used PyPI `redis` clients only (no `import fast_mlsirm`, no fit).

## numeric_equality
Committed hash values `[float(unit)*1.5 for unit in 0..n)` vs expected list after drain.

## Library path (mc_replicate / simulate) — 2026-09-20 later

**LABEL: generic fast_mlsirm remote/Valkey path validation — NOT research model complete.**

- Artifact: `valkey_library_3host_mc_replicate_20260920.json`
- FIRST_CMD: `HETERO_AIR_PYTHON=<fit-score-.venv/python> FAST_MLSIRM_VALKEY_URL=redis://192.168.68.3:16381/0 FAST_MLSIRM_LIBRARY_VERSION=0.11.4 PYTHONPATH=python python scripts/run_hetero_library_valkey_evidence.py` (cwd: fmls-2001-valkey-transport)
- API: `SubprocessExecutor` + `ValkeyStreamsOutcomeStore` + `remote_worker.execute_mc_replicate` → `fast_mlsirm.simulate`
- Hosts: air local, s1 `seongho@192.168.68.3`, m1 `seonghobae@10.6.0.3`
- Checks: 9/9 completed, all `library_function=fast_mlsirm.simulate`, fail→reclaim→dedup PASS
