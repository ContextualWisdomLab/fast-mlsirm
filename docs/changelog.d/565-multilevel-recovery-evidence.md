# Multilevel recovery evidence

## Added

- Added a deterministic Rust-owned known-truth recovery gate for the existing
  crossed, weighted multiple-membership contextual-effect MAP estimator.
- The gate reports centered-effect bias, MAE, and RMSE against the generating
  effects. It does not claim standard errors, causal contextual effects, or
  completion of the broader longitudinal and drift recovery program.

## Fixed

- The public Rust crossed/multiple-membership estimator now rejects a person's
  contextual design unless the finite non-negative weights sum to one within
  every declared classification, using the same `1e-12` absolute tolerance as
  the canonical Python membership contract. This preserves direct Rust/Python
  admission parity without changing MAP likelihood or estimator arithmetic.
