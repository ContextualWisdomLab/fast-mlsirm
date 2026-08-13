# Conditional Rasch M2 Rust ownership

## Changed

- Public `m2_cmle_rasch()` and `m2(..., estimator="cmle")` fail closed without the compiled Rust core and delegate every result field to `m2_cmle_rasch_stat`.
