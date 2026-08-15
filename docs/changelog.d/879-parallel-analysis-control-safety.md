# Parallel-analysis control trust hardening

## Security

- Validate `n_iterations`, `centile`, and `seed` before native-core discovery, accepting only exact built-in integers and supported concrete NumPy integer scalars while rejecting caller-defined subclasses and conversion providers without executing their callbacks.
- Preserve the existing positive-iteration, centile `0..99`, Rust `u64` seed, and 128 MiB random-benchmark workspace limits without changing Rust-owned Horn/Glorfeld factor-retention arithmetic.
