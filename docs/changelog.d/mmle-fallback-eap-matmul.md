# MMLE fallback EAP matrix-vector projection

## Changed

- Replaced the NumPy fallback MMLE EAP broadcast-and-reduce expression with an equivalent dense matrix-vector product, avoiding the additional posterior-shaped temporary array and permitting optimized numerical-library dispatch when available.
- Added an independently reconstructed missing-data parity fixture, a source-level allocation-path regression, and APA 7th doctoring while preserving the Rust primary backend and all statistical contracts.
