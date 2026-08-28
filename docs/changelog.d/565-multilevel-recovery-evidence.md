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
- The direct Rust crossed estimator now rejects declared response matrices above
  20,000,000 logical cells before response-slice validation or Newton work,
  matching the canonical Python public-admission resource ceiling without
  changing response semantics or numerical estimation.
- The public Rust contextual-membership boundary now rejects more than 100,000
  membership edges or 100,001 CSR row-pointer entries before per-row uniqueness,
  referenced-effect, or output-allocation work, matching the canonical Python
  design and PyO3 resource bounds.
- The direct Rust crossed estimator now rejects `worker_count` values above
  10,000 at control admission, matching the canonical Python estimator guard
  before any worker partitioning or iterative estimation begins.
