# Automated-scoring assessment and policy contracts

## Added

- A provider-neutral `fast_mlsirm.scoring` foundation with factory-sealed,
  content-addressed `AssessmentSpec` and immutable construct, engine,
  calibration, validation, adjudication, and monitoring policy contracts.
  Assessments bind exact `fast_mlsirm.rubric` fingerprints to declared
  constructs and fail closed on undeclared rubric, engine, or subgroup
  references.
- An independently owned `ASSESSMENT_SCHEMA_VERSION` for the assessment wire
  format. Rubric artifacts remain linked by exact content fingerprints instead
  of forcing assessment-schema releases to share the rubric schema lifecycle.
- Deterministic canonical JSON, SHA-256 artifact identities, descriptive public
  handles, strict two-or-more-token identifiers, and recursively copied/frozen
  bounded JSON metadata. Non-finite numbers, unsafe object types, excessive
  nesting/width/node counts, and signed-64-bit integer overflow are rejected
  without reflecting caller content in exceptions.
- The slice records reproducible scoring policy but introduces no provider SDK,
  response-content storage, psychometric arithmetic, universal metric cutoff,
  fairness claim, or high-stakes automation claim. Later scoring, calibration,
  validation, adjudication, monitoring, and domain adapters must reuse these
  exact contracts and the existing Rust-backed numerical implementations.
