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
  `q_xi` is admitted as an exact positive integer before dispatch because MIRT
  does not consume the latent-space rule, while spatial models retain the Rust
  core as authority for materializing and validating that model-dependent rule.
  Callback-bearing integer subclasses fail before parameter marshalling or
  native discovery.
