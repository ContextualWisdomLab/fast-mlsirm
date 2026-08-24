# Parallel-analysis data admission

## Fixed

- Reject complex and non-real-numeric caller matrices before Horn/Glorfeld parallel-analysis input is narrowed to `float64`, preventing imaginary evidence from being silently discarded or object-element numeric callbacks from running during package-owned admission.
- Preserve existing real numeric input compatibility, integer-control validation, bounded random-eigenvalue workspace policy, and Rust ownership of eigenvalue, random-benchmark, centile, and retention arithmetic.
