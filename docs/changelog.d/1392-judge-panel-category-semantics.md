# Judge panel category-generation semantics

## Fixed

- LLM-as-a-Judge construct projection now requires one category-generation mode per persons-by-items panel. A panel may use explicit zero-based criterion categories for every row or derive categories from criterion scores for every row, but it cannot mix those response-generation semantics across respondents. The first row that changes mode is rejected before projection, while all-explicit and all-score-derived panels preserve the existing authoritative criterion order and Rust-owned GRM/GPCM numerical path.
