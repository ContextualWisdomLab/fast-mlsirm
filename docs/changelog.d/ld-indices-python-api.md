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
- Replay the raw Rust/PyO3 LD result before any general mapping or NumPy
  conversion protocol can execute. The native carrier must be an exact built-in
  dictionary with exactly `x2_signed` and `g2_signed`; each pair vector must be
  an exact built-in list of binary64 Python floats with matching cardinality.
  The cardinality must be a valid upper-triangle pair count matching the
  admitted item count, and values may be finite or `NaN` but never infinite.
  Only after that carrier is sealed are package-owned NumPy arrays created, so a
  stale or malformed native boundary cannot invoke caller-controlled `keys`,
  iteration, or `__array__` hooks or define a contradictory public pair surface.
- Fail closed on multidimensional trait banks until LD expectations integrate
  independent trait dimensions correctly. The current public path accepts only
  the one-dimension `factor_id == 0` contract rather than aliasing distinct
  traits onto one Gauss-Hermite node.
- Surface the population expectation as a public `population` control. The
  current numerical kernel supports only the standard-normal single population
  (`None` or `{"kind": "single"}`); explicit `singlefree`, multigroup, or
  multilevel population metadata fails closed rather than silently reusing
  zero-mean/unit-SD expectations.
- Make `mlsirm-core` the canonical owner of LD resource admission with the
  versioned `ld-resource-v1` contract. The Rust public boundary now checks the
  item-by-quadrature probability surface, upper-triangle pair outputs,
  pair-by-person work, and pair-by-quadrature work before `icc_nodes` or pair
  allocation, using checked/division-before-multiplication arithmetic. MIRT
  continues to ignore the latent-space rule while spatial GH/QMC/MC node counts
  follow their native model-aware cardinality. Python retains the same ceilings
  as an earlier caller boundary rather than substituting for the Rust invariant.
  `ld_resource_preflight` exposes the admitted native work surface without
  executing the diagnostic so Rust consumers can validate capacity explicitly.
