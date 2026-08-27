# Parallel-analysis observed data budget

## Fixed

- Bound exact NumPy and built-in persons × items evidence to 20,000,000 logical cells before contiguous `float64` marshalling, while preserving the existing 128 MiB random-eigenvalue workspace ceiling and Rust-owned Horn/Glorfeld factor-retention arithmetic.
- Mirror the same 20,000,000-cell observed-data ceiling in `mlsirm-core` immediately after checked shape arithmetic, before data-length/value validation or package-owned random-data allocation, so direct Rust callers cannot bypass the public resource envelope.
