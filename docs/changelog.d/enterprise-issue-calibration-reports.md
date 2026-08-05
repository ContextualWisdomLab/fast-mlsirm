# Governed enterprise issue calibration reports

## Added

- Added `fit_enterprise_issue_facets_calibration_reports()` as an additive
  orchestration boundary from exact enterprise scoring executions to the existing
  shared criterion-specific calibration report objects.
- Reused the accepted enterprise provenance replay, shared calibration bundle,
  Rust-backed facets estimator, and canonical report schema without adding an
  enterprise-specific fit, report, aggregation, ranking, utility, or decision
  contract.
- Bound every report to the exact shared bundle and criterion-design fingerprints,
  materialized review triggers once, preserved criterion separation, and added
  deterministic delegation, validation-order, public-export, documentation, and
  statement/branch coverage tests for issue #404.
