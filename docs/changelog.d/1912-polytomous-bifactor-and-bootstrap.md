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

## Fixed

- Remove the crate-wide `clippy::erasing_op` / `clippy::identity_op`
  allowance; no broad lint suppression remains.
