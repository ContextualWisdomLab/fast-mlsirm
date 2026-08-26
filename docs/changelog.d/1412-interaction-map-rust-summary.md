# Rust-owned residual interaction-map summaries

## Added

- Extend the Rust residual interaction-map result with the effective numerical rank, complete-case map respondent/item counts, excluded scored-row/column counts, and deterministic closest/farthest retained-cell identities derived from the existing requested-axis distance surface.
- Retain the complete-case observed and expected cell values in the same original-index order as residual/distance/reconstruction evidence so downstream products do not have to reconstruct the Rust calculation inputs.
- Persist the caller-requested axis count and the explicit `lexicographic-first-original-index` closest/farthest tie policy in the versioned Rust envelope so downstream consumers do not infer either contract from coordinate shapes or implementation comments.
- Resolve distance ties by the lexicographically first original `(person_index, item_index)` so downstream products do not invent their own ranking convention.
- Add a domain-neutral versioned Rust envelope that validates the exact `fast-mlsirm.residual-interaction-map.v1` schema before numerical work, rejects duplicate or shape-mismatched opaque person/item identifiers, maps retained/extreme cells back to those caller identities, records a stable algorithm identifier, crate implementation version, calculation-provenance identity, requested axis count, deterministic tie-policy identity, and finite-value status, and fails closed if the numerical owner ever returns a non-finite persisted value.
- Preserve Gabriel symmetric-scaling arithmetic, complete-case missingness, resource bounds, and the existing residual/distance/reconstruction/unexplained/cross-share calculations. This remains a bounded Rust-core step toward issue #1412; public PyO3/Python exposure, cryptographic input-digest provenance, binding/recovery parity evidence, and release pinning remain follow-up work.

Research basis: Gabriel (1971), *Biometrika*, 58(3), 453–467; Jeon, Jin, Schweinberger, and Baugh (2021), *Psychometrika*, 86(2), 378–403, https://doi.org/10.1007/s11336-021-09762-5.
