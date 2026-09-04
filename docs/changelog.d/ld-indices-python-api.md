# Local-dependence Python API

## Fixed

- Expose the existing Rust-owned Chen-Thissen signed X2 and G2 pairwise
  local-dependence indices through `fast_mlsirm.fitstats.ld_indices` and the
  package root. Python performs only established response, shape, and control
  validation plus NumPy marshalling. Pairs with fewer than 20 jointly observed
  responses remain undefined, and the API deliberately supplies no universal
  pass/fail cutoff.
- Reuse package-owned callback-free fit-statistics admission for LD quadrature
  controls. The always-used `q_theta` must be an embedded Gauss-Hermite rule;
  `q_xi` is admitted as an exact positive integer representable by the native
  Rust `usize` before dispatch because MIRT does not consume the latent-space
  rule, while spatial models retain the Rust core as authority for materializing
  and validating that model-dependent rule. Callback-bearing integer subclasses
  and Python integers outside the native `usize` range fail before parameter
  marshalling or native discovery.
- Seal `eps_distance` at the same public boundary as an exact, losslessly
  representable, finite positive Rust `f64` control. Callback-bearing scalar
  subclasses and non-positive or non-finite values fail before parameter
  marshalling or native discovery; the Rust core remains the numerical owner
  of distance and local-dependence arithmetic.
- Replay the Rust/PyO3 LD result postcondition before publication. The signed X2
  and G2 vectors must be one-dimensional binary64 arrays with matching
  cardinality, their cardinality must be a valid upper-triangle pair count and
  match the admitted item count whenever that count is available from an inert
  factor-id carrier, and values may be finite or `NaN` but never infinite.
  Published vectors are copied into package-owned arrays so stale or malformed
  native evidence cannot define a contradictory public pair surface.
- Fail closed on multidimensional trait banks until LD expectations integrate
  independent trait dimensions correctly. The current public path accepts only
  the one-dimension `factor_id == 0` contract rather than aliasing distinct
  traits onto one Gauss-Hermite node.
- Preflight LD probability-table storage, pair-output cardinality, and
  pair-by-person work before native dispatch. This bounds otherwise valid
  quadrature/item combinations that could request multi-gigabyte Rust
  allocations or unbounded quadratic diagnostic work.
