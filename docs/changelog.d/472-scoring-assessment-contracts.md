# Automated-scoring assessment contracts

## Added

- A new provider-neutral `fast_mlsirm.scoring` namespace defines immutable `ConstructSpec`, `PolicyDocument`, `RubricBinding`, and factory-sealed `AssessmentSpec` contracts for automated-scoring workflows.
- Assessments bind exact rubric semantic versions and complete SHA-256 fingerprints from `fast_mlsirm.rubric`, require one content-addressed engine, calibration, validation, adjudication, and monitoring policy, and expose deterministic 128-bit public handles without duplicating rubric levels.
- Policy settings and assessment metadata use bounded canonical JSON, reject non-finite values and raw response or source-content fields, and return stable redacted validation errors. The slice performs no psychometric arithmetic and makes no scoring-accuracy, fairness, reliability, scoreability, or validity claim.
