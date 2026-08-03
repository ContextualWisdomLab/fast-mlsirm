# Generated-item audit and pilot-admission gate

## Added

- A deterministic, content-addressed `CandidateAuditReport` that inspects only
  parser-validated generated-item candidates and emits bounded redacted
  findings for instruction-override indicators, ambiguity-prone option or
  stem patterns, duplicate normalized option/evidence surfaces, overlapping
  rubric indicators, non-atomic criterion indicators, near-duplicate source
  attributions, declared safety notes, and excessive finding volume.
- An enforced `draft -> audited -> pilot` lifecycle: blocking or
  review-required findings retain the candidate in `draft`; only an exact
  candidate/audit fingerprint match may produce an immutable
  `PilotCandidateRecord` with explicit pilot-study, query/testlet, generator,
  judge-policy, occasion, rubric, blueprint, and item provenance.
- Public pilot admission replays the package's exact named and versioned audit
  policy and compares the complete report fingerprint, preventing a caller
  from constructing a clean-looking report that bypasses real findings or
  relabeling the current implementation as an unsupported policy version.
- The audit is a deterministic screening and governance boundary, not a
  semantic answerability, fairness, scoreability, psychometric validity, or
  operational-deployment declaration. Pilot observation conversion and
  Rust-backed calibration remain follow-up slices of issue #407.
