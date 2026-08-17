# Parallel-analysis input and workspace bounds

## Fixed

- `parallel_analysis()` now rejects booleans, floats, strings, and caller-defined integer-conversion hooks for integer controls instead of silently coercing them before Rust dispatch.
- The public wrapper validates the Rust `u64` seed range and rejects oversized random-eigenvalue benchmark workspaces before PyO3 dispatch.
- `mlsirm-core` independently caps the random-eigenvalue simulation workspace at 128 MiB before allocation while preserving the existing Horn/paran numerical algorithm and deterministic RNG contract.
