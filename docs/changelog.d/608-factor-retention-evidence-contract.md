# Governed factor-retention evidence contract

## Added

- Added a provider-neutral `fast_mlsirm.factor_retention` contract that records already-computed candidate counts from supported retention methods, rejects duplicate method evidence, and reports `consensus`, `disagreement`, or `insufficient_evidence` without forcing a winner when methods disagree.
- Added deterministic conservative candidate ranges, a fixed transport ceiling, closed method identities, complete fail-closed tests, and scientific doctoring while keeping factor-retention and structural model-selection arithmetic Rust-owned and separate.
