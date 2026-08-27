# Governed factor-retention result invariants

## Fixed

- Public `FactorRetentionResult` records now replay package-owned evidence invariants and reject decision, retained-count, candidate-range, duplicate-method, or post-construction-mutated evidence states that `govern_factor_retention(...)` could never produce. This changes only Python governance/transport validation; numerical factor-retention and structural-model-selection arithmetic remain outside this surface and Rust-owned where implemented.
