# Enterprise criterion-level observation adapter

## Added

- Added `build_enterprise_issue_score_observation`, which compiles exact
  request-bound enterprise evidence into the existing shared criterion-level
  `ScoreObservation` contract without introducing a parallel observation schema.
- Added fail-closed request-provenance replay, evidence-subset validation,
  deterministic evidence ordering, managed issue/evidence fingerprints, and
  supporting, counter, and context evidence counts without retaining source text.
- Required supporting evidence for every non-abstained enterprise observation and
  explicit counterevidence representation whenever the issue declares
  counterevidence; insufficient evidence remains an abstention rather than a low
  score.
- Added deterministic provenance, order-invariance, abstention, terminal-state,
  adversarial metadata, evidence-binding, shared-contract delegation, and
  statement/branch coverage tests for the next issue #404 vertical slice.
