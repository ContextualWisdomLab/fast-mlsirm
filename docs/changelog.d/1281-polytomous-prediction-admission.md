# Polytomous prediction evidence admission

## Fixed

- Reject callback-bearing, complex, non-numeric, and lossy GRM/GPCM prediction evidence before NumPy narrowing or compiled-core discovery.
- Preserve trusted exact NumPy and built-in sequence inputs while enforcing the 20,000,000-cell output ceiling before contiguous dense materialization.
- Keep category-probability and expected-score arithmetic in the Rust prediction kernel; Python performs validation, bounded materialization, and marshalling only.
