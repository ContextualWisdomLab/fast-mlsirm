# Parallel-analysis control callback boundary

## Fixed

- Validated explicit `n_iterations`, `centile`, and `seed` controls before compiled-core discovery, so malformed controls cannot cross the native-loader boundary.
- Restricted accepted controls to exact built-in Python integers and exact supported NumPy integer scalar types, rejecting caller-defined subclasses before conversion or representation callbacks while preserving the existing Rust-owned Horn/Glorfeld calculation, control domains, and 128 MiB random-workspace ceiling.
