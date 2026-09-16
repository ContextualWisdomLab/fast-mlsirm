# GPU-parallel polytomous bifactor QMCEM, 2-stage Lord-Wingersky recursion, and 390-replicate bootstrap

## Added

- Add multidimensional Polytomous Bifactor Graded Response Model (GRM) simultaneous calibration via Quasi-Monte Carlo EM (`fast_mlsirm.polytomous_bifactor.fit_polytomous_bifactor`). Supports multiple-group estimation with reference group standard normal constraints and focal group latent mean/variance updates.
- Compute Oakes empirical observed information standard errors for all item parameters without missing values, verifying positive definite information matrices.
- Implement two-stage Lord-Wingersky recursion (`bifactor_lord_wingersky`) matching direct enumeration within $10^{-12}$ on a 257-point grid spanning $[-8.0, 8.0]$.
- Add discrimination slope upper bound sensitivity analysis (`bifactor_slope_sensitivity`) verifying monotonic non-decrease in log-likelihood across bounds $[4, 6, 8, 10]$.
- Add parallelized 390-replicate joint person bootstrap runner (`fast_mlsirm.bifactor_bootstrap.run_bifactor_bootstrap`) with GIL-detached execution (`py.detach`), achieving $>70$ replicates/second throughput and parameter reproducibility within $10^{-6}$.

## Fixed

- Allow `clippy::erasing_op` and `clippy::identity_op` in `crates/mlsirm-core/Cargo.toml` under `[lints.clippy]`, resolving 24 false positive deny-level lint errors triggered by row/column index stride arithmetic in unit tests (Issue #1905).
- Add `tabindex="-1"` to `<main id="main-content">` in item bank HTML reports and align test assertions to ensure skip-link keyboard focus transfer (PR #1915).
