# Governed enterprise issue facets calibration reports

## Added

- Added `fit_enterprise_issue_facets_calibration_reports()` as a bounded governed
  workflow from exact enterprise issue scoring executions to one existing shared
  `ScoringFacetsCalibrationReport` per criterion.
- Reused the enterprise provenance replay and shared calibration bundle assembler,
  then delegated every criterion design to the existing Rust-backed shared report
  helper without adding an enterprise-specific fit, report schema, or statistical
  arithmetic.
- Added package-managed bundle, design, and criterion report provenance,
  deterministic execution-order invariance, one-time review-trigger
  normalization, source-free caller metadata, and reserved-key rejection.
- Added fail-closed batch validation of every derived report identifier before any
  Rust estimator delegation, preventing an overlong prefix-and-criterion
  combination from producing a partially fitted report tuple.
- Added a realistic connected two-issue, two-task-revision, two-rater-family,
  two-criterion Rust fit, complete orchestration and privacy tests, public
  documentation, and APA 7th scientific and governance traceability.
- Aligned shared report and HTML replay validation with the Rust estimator's
  nonconverged trace contract: `n_iter` optimization iterations may be followed
  by one retained terminal post-update likelihood evaluation.
