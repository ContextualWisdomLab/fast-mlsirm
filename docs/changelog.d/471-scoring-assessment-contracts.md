# Shared assessment and scoring-policy contracts

## Added

- A new provider-neutral `fast_mlsirm.scoring` namespace with immutable
  construct, engine, calibration, validation, adjudication, monitoring, and
  reporting policy contracts. `build_assessment_spec` binds those policies to
  exact `fast_mlsirm.rubric.RubricSpecification` SHA-256 fingerprints without
  defining a parallel rubric schema.
- Factory-sealed, content-addressed `AssessmentSpec` artifacts expose a full
  SHA-256 fingerprint and descriptive 128-bit public handle, normalize ordering
  deterministically, preserve deeply immutable bounded metadata, and fail
  closed on unknown, mismatched, duplicated, unused, or dangling rubric and
  construct references.
- Buyer documentation states the MSA embedding boundary and makes explicit that
  an assessment contract is provenance rather than scoring, validity,
  authorization, fairness, or operational-deployment evidence.
