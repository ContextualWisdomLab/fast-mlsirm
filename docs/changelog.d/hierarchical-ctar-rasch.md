# Joint MAP hierarchical continuous-time AR(1) Rasch

## Added

- Added a Rust-owned joint MAP hierarchical continuous-time AR(1) Rasch
  estimator behind `fit_hierarchical_longitudinal_irt`, stacked on the
  `#976` longitudinal design handoff.
- Estimated shared population hyperparameters `(mu, tau, lambda)` and person-
  occasion states from exact millisecond elapsed-day gaps. State intervals
  are Wald intervals from measurement observed information; short series
  leave `lambda` weakly identified under joint MAP.
- Documented the estimand as joint MAP, not independent OLS, not caller-
  supplied discrete AR, not Fox and Glas Gibbs, and not estimated
  multiple-membership `u_h`. GPU parity is reported false because the
  existing wgpu path owns a different MLSIRM objective.
- Added multi-seed true-parameter recovery, irregular-time, missing-response,
  worker-determinism, and fail-closed marshalling tests with APA 7 ADR and
  doctoring.
- Normalized oversized integer and non-finite real execution controls before
  native dispatch so Python callers receive package-owned validation errors.
- Enforced finite, identified sum-zero Rasch item intercepts in the simulator
  before generating recovery data.
