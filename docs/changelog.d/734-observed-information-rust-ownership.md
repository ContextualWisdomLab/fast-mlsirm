# Observed-information Rust ownership

## Fixed

- Public `observed_information` assembles finite-difference Hessians in the Rust
  core from evaluated objective samples, and `second_order_test` eigenvalue
  diagnostics are Rust-owned.
