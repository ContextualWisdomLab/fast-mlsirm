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
