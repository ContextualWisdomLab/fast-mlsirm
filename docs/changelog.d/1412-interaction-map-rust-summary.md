# Rust-owned residual interaction-map summaries

## Added

- Extend the Rust residual interaction-map result with the effective numerical rank, complete-case map respondent/item counts, excluded scored-row/column counts, and deterministic closest/farthest retained-cell identities derived from the existing requested-axis distance surface.
- Resolve distance ties by the lexicographically first original `(person_index, item_index)` so downstream products do not invent their own ranking convention.
- Preserve Gabriel symmetric-scaling arithmetic, complete-case missingness, resource bounds, and the existing residual/distance/reconstruction/unexplained/cross-share calculations. This is a bounded Rust-core step toward issue #1412; public PyO3 schema/version/provenance exposure remains follow-up work.

Research basis: Gabriel (1971), *Biometrika*, 58(3), 453–467; Jeon, Jin, Schweinberger, and Baugh (2021), *Psychometrika*, 86(2), 378–403, https://doi.org/10.1007/s11336-021-09762-5.
