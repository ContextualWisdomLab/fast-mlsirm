# Rust longitudinal state layer

## Added

- Added a Rust-owned independent per-respondent OLS trend and discrete-sequence
  AR(1) state predictor behind the sealed `fast_mlsirm.multilevel` contract.
- Preserved exact sequence gaps, missing-occasion output state, deterministic
  respondent sharding, RMSE/count diagnostics, and PyO3/Python marshalling.
- Documented the compatibility wire label `random_intercept_slope` as independent
  OLS with no population random-effects distribution or shrinkage, and the AR
  path as caller-supplied `phi` without coefficient estimation.
- Added slope-recovery, missingness, irregular-calendar/non-contiguous-sequence,
  worker-determinism, and fail-closed contract tests with APA 7 doctoring.
- This fragment does not claim full multilevel IRT random-effect integration,
  uncertainty, continuous-time transitions, or GPU recurrent-state parity.
