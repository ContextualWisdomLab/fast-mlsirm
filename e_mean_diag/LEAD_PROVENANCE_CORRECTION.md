# Lead correction 2026-09-21T03:38Z

MEAN_DRIFT_DIAG concluded consumer loaded 6c30712d because `sha`/`build_source_sha` fields said so.

**Disproven for the binary:** on s1, perf-diag `_core.so` sha16 `a0453b7a` **byte-matches** wheel `fmls-g4w-fipc-s1-wheels/32807ed0/...whl` embedded `.so`. Verify-run HEAD is `32807ed0`.

Therefore numerical observations (100 accepts, LL improving, mean→[0.393,-1.42], max_iter, not overshoot) apply to **32807ed0 tip content**. Harness metadata fields remain wrong and must be fixed separately.

Next owner: `ctx_d9c2b100491f` (fmls-e-mean-next).
