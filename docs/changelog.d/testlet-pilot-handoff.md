# Generated-item pilot handoff to testlet calibration

## Added

- A factory-sealed, content-addressed `TestletPilotDesign` assembled from the
  existing replay-verified binary pilot design. It preserves exact missingness,
  per-cell rater provenance, item provenance, and governed
  `query_testlet_id` groupings while emitting copied `responses` and
  integer `testlet_id` arrays accepted by the Rust-backed `fit_testlet` API.
- Singleton-only groupings are rejected instead of being labelled as a
  testlet design. Rasch/2PL selection, iteration limits, quadrature size,
  tolerance, variance initialization, and convergence policy are validated
  before calibration arguments are returned.
- The handoff performs no psychometric arithmetic and makes no connectedness,
  convergence, local-dependence, fit, reliability, fairness, scoreability, or
  validity claim. Model comparison, parameter recovery, residual diagnostics,
  DIF/fairness analysis, and human-anchored validity evidence remain required
  before operational use.
