# IRT response admission integrity

## Fixed

- Reject complex, textual, object-backed, and arbitrary array-provider response evidence before float64 marshalling at the shared IRT response/readiness boundary.
- Preserve exact NumPy real-numeric arrays and ordinary built-in nested response containers, including supported concrete NumPy real scalar cells and NaN missingness.
- Apply the same trusted response admission before `fit_irt_experiment()` can invoke a production numerical fitter; psychometric arithmetic remains Rust-owned.
