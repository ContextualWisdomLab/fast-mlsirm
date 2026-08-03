# Pilot-observation handoff to observed-score DIF screening

## Added

- A factory-sealed, content-addressed `DifPilotDesign` assembled by
  `build_dif_pilot_design` from replay-verified pilot observation records.
  It wraps the binary MIRT assembler's fail-closed response, provenance,
  missingness, duplicate-cell, category, observed-support, and
  dense-allocation contracts, and requires descriptive reference/focal
  group identifiers with an exact assignment for every indexed respondent;
  missing, unknown, undeclared, one-group, and normalized-collision group
  contracts are rejected with structured error codes.
- `to_observed_score_dif_kwargs` emits copied `responses` and `group`
  arrays accepted directly by the repository's Rust-backed binary
  observed-score DIF APIs (`mantel_haenszel_dif`, `logistic_dif`, their
  purified variants, and `sibtest`) and refuses incomplete matrices
  instead of silently applying complete-case deletion or imputation. The
  handoff performs no psychometric arithmetic and makes no invariance,
  fairness, or validity claim. The G-theory handoff remains the final
  follow-up slice of issue #407.
