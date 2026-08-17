# Governed enterprise issue calibration bundle

## Added

- Added `build_enterprise_issue_facets_calibration_bundle()` as a bounded,
  fail-closed assembler from exact enterprise issue scoring execution tuples into
  the existing shared `ScoringFacetsCalibrationBundle` contract.
- Reused the complete issue-owned provenance replay for every execution and
  delegated criterion separation, task-revision and rater identity, category
  support, record budgets, and connectedness to the existing shared bundle
  builder without adding a competing enterprise schema.
- Added deterministic order-invariance, exact tuple-shape, delegation,
  resource-bound, public-export, and connected-design tests while preserving
  Rust-only psychometric arithmetic and conservative validity, fairness, and
  causal limits for issue #404.
