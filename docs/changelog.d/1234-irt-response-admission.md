# IRT response and mask admission integrity

## Fixed

- Reject complex, textual, object-backed, and arbitrary array-provider response evidence before float64 marshalling at the shared IRT response/readiness boundary.
- Preserve exact NumPy real-numeric arrays and ordinary built-in nested response containers, including supported concrete NumPy real scalar cells and NaN missingness.
- Reject callback-bearing mask evidence before Boolean coercion while preserving exact Boolean/real-numeric NumPy arrays and ordinary trusted numeric mask containers.
- Validate IRT family and category-count semantics before `fit_irt_experiment()` can materialize caller response evidence, and apply trusted response/mask admission before any production numerical fitter runs; psychometric arithmetic remains Rust-owned.
