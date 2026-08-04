# Shared automated-scoring assessment contracts

## Added

- Added the provider-neutral `fast_mlsirm.scoring` namespace with immutable construct, engine, calibration, validation, adjudication, monitoring, reporting, and factory-built assessment contracts.
- Assessment artifacts bind exact `RubricSpecification` fingerprints without duplicating rubric levels, own an independent scoring wire-schema version, and expose deterministic full SHA-256 fingerprints plus descriptive 128-bit public handles.
- Bounded canonical metadata is deeply immutable, normalizes equivalent floating values, rejects response or source content, and returns structured non-reflective validation errors. This contract layer performs no psychometric arithmetic and does not itself establish scoring accuracy, reliability, fairness, scoreability, or validity.
