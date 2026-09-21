# TOP PRIORITY: PyO3 GIL release for bootstrap n_jobs (v0.11.4)

Coordinator 20260918T0310Z / 024649Z.

## Verified
- `allow_threads` string is 0 (PyO3 0.29 uses `Python::detach` / `.detach(|| ...)`).
- `origin/main` (a3602747): `.detach(` already wraps `fit_bifactor_grm` + `fit_bifactor_grm_multigroup` only.
- Still holds GIL end-to-end: `bifactor_oakes_se`, `fit_bifactor_grm_fipc`, `fit_two_tier_grm`, `two_tier_oakes_se` (+ audit other heavy entry points used by bootstrap).
- Air late-life checkout e253504b / installed wheel: **zero** `.detach(` — older than main; must ship 0.11.4 with full coverage.
- mlsirm-core: no rayon / par_iter (single-threaded fit) — OK; concurrency comes from Python threads once GIL released.

## Fix
For each heavy binding: copy `PyReadonlyArray*` → owned Rust (`to_vec`/`to_owned_array`) **before** `.detach(|| ...)`, call core inside detach, build Python objects after. Pattern already in `fit_bifactor_grm` (~lib.rs:1428).

## Test
Wall-clock concurrency regression: N identical fits via ThreadPoolExecutor vs serial; assert speedup with measured threshold + ADR-0028 justification (not a magic constant without evidence).

## Release
Fold into v0.11.4 with #1993+#1996. Builds on Air only (no local cargo).
