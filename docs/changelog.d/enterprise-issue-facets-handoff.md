# Governed enterprise issue many-facet handoff

## Added

- Added `build_enterprise_issue_facets_rating_records()` as a fail-closed replay
  boundary from exact enterprise issue scoring executions into the existing
  shared `ScoringFacetsRatingRecord` contract.
- Replayed atomic issue, issue-content, respondent, response-revision,
  request-bound evidence, counterevidence, and package-managed observation
  provenance before delegating record projection to the shared calibration
  builder.
- Preserved abstention as a terminal missing rating, separate analytic criteria,
  Rust-only psychometric arithmetic, deterministic order invariance, complete
  statement and branch tests, and conservative validity and causal limits for
  issue #404.
