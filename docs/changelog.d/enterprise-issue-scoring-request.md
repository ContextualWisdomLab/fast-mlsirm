# Enterprise issue scoring request compiler

## Added

- Added deterministic compilation from accepted enterprise issue, evidence,
  counterevidence, stakeholder-perspective, and candidate-intervention records
  into the existing shared criterion-level `ScoringRequest` contract.
- Added package-managed exact provenance for source revisions, evidence spans,
  epistemic assertion kinds, perspectives, interventions, task revision, rubric,
  assessment, and engine authorization without retaining raw enterprise text.
- Preserved the existing `ScoringEngine`, `ScoreObservation`, and
  `ScoringResult` execution boundaries without adding a parallel result schema
  or sentiment, calibration, ranking, utility, causal, or routing arithmetic.
- Added fail-closed duplicate-evidence, cross-issue replay, unbound perspective
  source, reserved metadata, sensitive-content, ordering-invariance, and shared
  contract delegation tests for issue #404.
