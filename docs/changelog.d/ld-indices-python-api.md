# Local-dependence Python API

## Fixed

- Expose the existing Rust-owned Chen-Thissen signed X2 and G2 pairwise
  local-dependence indices through `fast_mlsirm.fitstats.ld_indices` and the
  package root. Python performs only established response, shape, and control
  validation plus NumPy marshalling. Pairs with fewer than 20 jointly observed
  responses remain undefined, and the API deliberately supplies no universal
  pass/fail cutoff.
- Reuse the package-owned callback-free fit-statistics quadrature admission for
  `q_theta` and `q_xi`. Only embedded Gauss-Hermite rule sizes cross the public
  Rust boundary; callback-bearing integer subclasses and unsupported positive
  sizes fail before parameter marshalling or native dispatch.
