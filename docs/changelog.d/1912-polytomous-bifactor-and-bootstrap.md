# GPU-parallel bifactor E-step and joint person bootstrap with caller-controlled stopping

## Added

- Add a GPU-parallel E-step for the Bock-Aitkin bifactor GRM with
  Gibbons-Hedeker dimension reduction
  (`crates/mlsirm-core/src/gpu_bifactor.rs`), covering the single-group
  estimator and the multigroup calibration (including group moment
  accumulators). Kernels accumulate in f32; CPU/GPU fit-level agreement is
  asserted within a documented single-precision tolerance on fixtures
  including reverse-keyed items and multiple groups. CPU fallback when no
  GPU adapter is available; `device` is a validated argument
  (`cpu`/`gpu`/`auto`) on both configs, both PyO3 entry points, and both
  Python wrappers.
- Release the GIL around the Rust bifactor fits (`py.detach`) so the
  bootstrap thread pool parallelizes.
- Add a joint person bootstrap driver
  (`fast_mlsirm.bifactor_bootstrap.run_bifactor_bootstrap`) in which
  replicate count, batch size, Monte Carlo stopping ratio, and compute
  budget are caller arguments with validated ranges. The stopping rule is a
  sequential application of the endpoint-accuracy framework of Andrews and
  Buchinsky (2000, §§ 2–4): the run stops once the maximum
  percentile-interval endpoint movement relative to the interval half-width
  falls below the caller ratio. Per-replicate convergence is reported;
  failed replicates are excluded, never substituted.
- Report bootstrap percentile intervals alongside empirical standard errors.
- Add two-stage Lord-Wingersky score recursion
  (`fast_mlsirm.bifactor_recursion`) matching direct enumeration within
  1e-12.
- Assert stage-5 CPU/GPU fit-level parity at the maintainer-standard
  study-precision quadrature grids (`tests/test_bifactor_gpu_high_q.py`,
  gated behind `STAGE5_HIGH_Q=1` like the Rust `#[ignore]` node-count
  regressions): CPU and GPU E-steps agree within the documented
  single-precision envelope at 121 and 241 nodes per dimension, and the
  CPU fits at 121 vs 241 nodes agree within the 5e-3 numerical band
  (marginal-likelihood integral convergence). Quadrature counts are caller
  arguments with no defaults and no caps: any `n >= 1` resolves via the
  shared arbitrary-`n` Gauss-Hermite rule
  (`quadrature::require_gh_rule`, Golub & Welsch, 1969; #1929/#1945),
  replacing the fixed `SUPPORTED_Q` membership table this branch
  previously enforced.
- Measure the joint person bootstrap at the 121-point study grid
  (`test_joint_bootstrap_cpu_vs_gpu_wall_time_q121`, same gate):
  measured CPU vs GPU wall times are printed for the PR record rather
  than asserted against machine-specific thresholds.
- Measured study-grid evidence (Apple Silicon, `STAGE5_HIGH_Q=1`, tiny
  48-person/6-item/2-specific fixture, `tol=1e-3`): single-group parity
  at `q=121` — CPU 4.544s vs GPU 6.181s, `max|Δslope|=1.897e-07`,
  `max|Δthreshold|=1.138e-07`, `|Δloglik|=2.374e-05` (4 EM iterations,
  both converged); at `q=241` — CPU 21.642s vs GPU 26.561s,
  `max|Δslope|=1.326e-07`, `max|Δthreshold|=1.356e-07`,
  `|Δloglik|=3.232e-05` (4 iterations, both converged); CPU 121-vs-241
  agreement `|Δloglik|=4.829e-09` (integral converged). Joint bootstrap
  at `q=121` (`B=2`, two-group multigroup path): CPU 161.512s
  (80.756s/rep) vs GPU 210.261s (105.130s/rep), replicate-by-replicate
  parity within 1e-3. Small problems stay CPU-faster (per-sweep GPU
  buffer setup dominates), as already disclosed in ADR-0027.

## Changed

- `q_general`/`q_specific` validation in the stage-5 Python surface
  (`bifactor_bootstrap.run_bifactor_bootstrap`) now accepts any integer
  `n >= 1` instead of the removed fixed-table membership set, matching
  the merged estimator contract (#1929/#1945); out-of-range counts still
  fail loudly and are never clamped.

## Fixed

- Remove the crate-wide `clippy::erasing_op` / `clippy::identity_op`
  allowance; no broad lint suppression remains.
