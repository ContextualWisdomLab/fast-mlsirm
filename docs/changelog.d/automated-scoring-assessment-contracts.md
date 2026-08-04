# Automated scoring assessment contracts

## Added

- A provider-neutral `fast_mlsirm.scoring` contract layer with factory-sealed,
  content-addressed `AssessmentSpec` artifacts that bind declared constructs,
  exact governed-rubric SHA-256 fingerprints, response format, engine allowlists,
  subgroup identifiers, and calibration, validation, adjudication, and monitoring
  policies without duplicating `fast_mlsirm.rubric`.
- Deterministic canonical JSON and SHA-256 identities, descriptive 128-bit public
  assessment handles, deeply immutable bounded metadata, structured fail-closed
  errors, and replay validation for every rubric, construct, engine, and subgroup
  reference. The slice adds no provider SDK or numerical estimator; future
  observation and calibration modules must delegate psychometric arithmetic to
  the existing Rust/PyO3 core.
