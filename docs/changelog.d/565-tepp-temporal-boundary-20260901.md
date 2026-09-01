# TEPP temporal boundary

## Changed

- Clarify the bounded-context split for time-indexed psychometrics: TEPP owns temporal/event composition and semantics, while `fast-mlsirm` retains reusable Rust numerical psychometric kernels over explicitly supplied occasion/time carriers. The existing CT-AR Rasch estimand is preserved; event ontology, temporal validity, changing-membership history, leakage policy, and EA projection remain foreign-owner concerns behind explicit contracts.
