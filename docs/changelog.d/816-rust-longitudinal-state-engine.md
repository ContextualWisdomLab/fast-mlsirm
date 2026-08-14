# Rust longitudinal state layer

## Added

- Added a Rust-owned respondent intercept/slope and discrete-sequence AR(1)
  state estimator behind the sealed `fast_mlsirm.multilevel` contract.
- Preserved exact sequence gaps, missing-occasion output state, deterministic
  respondent sharding, RMSE/count diagnostics, and PyO3/Python marshalling.
- Added slope-recovery, missingness, irregular-calendar/non-contiguous-sequence,
  worker-determinism, and fail-closed contract tests with APA 7 doctoring.
- This fragment does not claim full multilevel IRT random-effect integration,
  uncertainty, continuous-time transitions, or GPU recurrent-state parity.
