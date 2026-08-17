# Direct Rust response-time iteration ceiling

## Changed

- Enforce the package-wide `max_iter` ceiling (`1..=100_000`) inside the Rust
  lognormal response-time EM (`fit_rt_lognormal`) so direct PyO3 callers cannot
  bypass the Python `MAX_MAX_ITER` bound.
- Added a fail-closed ownership contract that rejects `MAX_MAX_ITER + 1` at the
  Rust boundary and documented the van der Linden (2007) speed model reference.
