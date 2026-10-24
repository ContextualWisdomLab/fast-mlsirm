### Added

- Added the domain-neutral `cwl_governed_rater_observation/v1` published
  language, Rust aggregate invariants, and a strict Draft 2020-12 JSON Schema
  for human, model, and algorithmic rater invocations.
- Added canonical cross-SDK reference fixtures, criterion-keyed serialized
  observations, and matching Rust bounds for 64 evidence references, 32 review
  signals, and 128 criterion observations.
- Added a DDD context map that separates observation creation, numerical
  calibration, assessment operations, temporal monitoring, and measurement
  reference metadata without introducing a shared-kernel repository.
- Explicitly excluded CEFR levels, scores, placement, pass/fail,
  certification, employment decisions, provider payloads, and hosted workflow
  state from the generic observation contract.
