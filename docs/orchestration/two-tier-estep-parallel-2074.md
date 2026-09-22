# Two-tier E-step person-chunk parallelism (#2074)

## Root cause (aligned with ctx_a9403daada13)

The core does **not** build a Rayon/OpenMP EM pool for two-tier fits. Idle
~`ncpus+1` TIDs on the A4 host are **not** EM workers. Draft #2018 already
added deterministic person-chunk rayon inside bifactor/two-tier Rust E-steps,
but the Python/PyO3 `fit_two_tier_grm` path hard-coded
`e_step_n_chunks=1` / `e_step_n_threads=1`, so callers never scheduled
chunks.

Separately, rootcause notes the M-step hot loop
`item_neg_ll_grad` (node sweep via `run_single_start`) remains serial. That
path is **not** parallelized here; it belongs with #2030/#2031 (analytic
Hessian), not with the Bock–Aitkin pattern-frequency person partition.

## Legal partition (this PR)

Person-axis E-step chunks: caller-owned `e_step_n_chunks` /
`e_step_n_threads`, local rayon pool (never `build_global`), fold in
chunk-index order so the FP association tree does not depend on thread
count.

### Citation limits (APA locators only for texts read in-run)

- **Cited with locators already grounded in-repo:** Gibbons et al. (2007,
  eq. 15); Cai, Yang, & Hansen (2011, p. 221) naming Gibbons & Hedeker’s
  (1992) dimension reduction and Bock & Aitkin (1981) EM.
- **Not re-verified in this dispatch:** Gibbons & Hedeker (1992) page
  locators from the issue text; Higham (2002) section locators from the
  #2018 cherry-pick. Those page/section claims are **not** asserted as
  freshly read here.
- Local PDF present for Bock & Aitkin (1981) under `~/papers/`; page
  numbers from the issue were not re-checked against that PDF in this run.

## Out of scope

- Do **not** restart A4 until an immutable release containing this fix
  remeasures pass.
- Do **not** treat small-study wall/cpu ratios as A4 (q=241, n=340)
  performance.
- M-step `item_neg_ll_grad` parallelization / analytic Hessian: #2030 →
  #2031.
- Bootstrap knob forwarding: #2034.

## Evidence contract

Same input / version / config (no shrinkage):

1. `e_step_n_threads=1` bit-identical floor (issue claims 0).
2. Draft PR linked to #2074.
3. Core utilization + wall-time vs baseline on a **small** study setting
   (not A4 resume; not q=241 n=340 overnight).
4. Numerical equivalence vs threads=1 at fixed chunks (bit-identical).
5. A4 not restarted.

## Measured (this host — small setting only)

From `scripts/evidence_2074_two_tier_estep_parallel.py`
(`n=48`, `q=7`, `max_iter=12`, `e_step_n_chunks=4`). **Not A4.**

| Metric | Value |
|---|---|
| threads=1 wall (run a) | 2.153 s |
| threads=1 bit-identical repeats | True (floor 0) |
| threads=4 wall | 1.892 s |
| wall speedup (small setting) | 1.14× |
| cpu/wall (utilization proxy) | 1.71 |
| bit-identical vs threads=1 | True |

Artifact: `docs/orchestration/evidence-2074/two_tier_estep_parallel_evidence.json`.

## Verified tests (this dispatch)

- `pytest tests/test_two_tier_grm.py` → **8 passed** (exit 0).
- `cargo test --lib fit_same_chunks_bit_identical_across_thread_counts` →
  bifactor + two_tier **ok** (exit 0).
- `cargo test --lib fit_records_e_step_chunk_provenance` → **ok** (exit 0).
- `cargo test --lib ordered_map_is_bit_identical` → **ok** (exit 0).

