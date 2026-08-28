# Add Rust-owned ordered proficiency posterior summaries

## Added

- Add a framework-neutral Rust API that converts already-calibrated posterior draws and immutable ordered cut scores into normalized level probabilities, a weighted posterior mean and uncertainty, a shortest contiguous credible-level set, and an ambiguity-preserving modal-level decision.
- Preserve ordinal semantics: exact cut scores enter the upper level, tied modal probabilities do not force a reported level, and cut scores are never reordered or repaired.
- Add fail-closed validation for empty or non-finite draws, malformed weights, non-increasing cut scores, invalid credible mass, and non-finite posterior moments, plus joint draw/weight permutation regressions.
- This bounded numerical slice does not estimate ability, validate standard setting or CEFR linking, average ordinal labels, implement CAT, or authorize certification decisions.
