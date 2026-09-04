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
