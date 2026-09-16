# GPU-parallel bifactor E-step and joint person bootstrap with caller-controlled stopping

## Added

- Add a GPU-parallel E-step for the polytomous bifactor GRM QMCEM (`crates/mlsirm-core/src/gpu_bifactor.rs`), covering the merged stage-1 estimator and the stage-2 multigroup calibration. Kernels accumulate in f32; CPU/GPU fit-level agreement is asserted within a documented single-precision tolerance on fixtures including reverse-keyed items and multiple groups. CPU fallback when no GPU adapter is available.
- Add a joint person bootstrap driver (`fast_mlsirm.bifactor_bootstrap.run_bifactor_bootstrap`) in which replicate count, batch size, Monte Carlo stopping ratio, and compute budget are caller arguments with validated ranges. The stopping rule is a sequential application of the endpoint-accuracy framework of Andrews and Buchinsky (2000, §§ 2–4): the run stops once the maximum percentile-interval endpoint movement relative to the interval half-width falls below the caller ratio. Per-replicate convergence is reported; failed replicates are excluded, never substituted.
- Report bootstrap percentile intervals alongside empirical standard errors.

## Fixed

- Build the Oakes observed information from converged posterior expected counts (final E-step at the fitted parameters), replacing the uniform-weight accumulation; eigenvalues and condition numbers are computed from the information matrix, never hard-coded.
- Remove the crate-wide `clippy::erasing_op` / `clippy::identity_op` allowance; no broad lint suppression remains.
