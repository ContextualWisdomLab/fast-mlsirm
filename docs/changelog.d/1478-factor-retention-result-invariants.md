# Governed factor-retention result invariants

## Fixed

- Public `FactorRetentionResult` records now replay package-owned evidence invariants and reject decision, retained-count, candidate-range, duplicate-method, or post-construction-mutated evidence states that `govern_factor_retention(...)` could never produce.
- Valid direct result construction is order-insensitive at the evidence-set boundary: exact package-owned evidence tuples are canonicalized to the same deterministic method ordering returned by `govern_factor_retention(...)` rather than rejecting semantically identical unsorted evidence.
- These changes affect only Python governance/transport validation; numerical factor-retention and structural-model-selection arithmetic remain outside this surface and Rust-owned where implemented.
