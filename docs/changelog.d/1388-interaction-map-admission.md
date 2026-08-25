# Interaction-map evidence admission

## Fixed

- Seal `axis_count` before caller matrix work and reject callback-bearing integer identities, booleans, nonpositive controls, and requests above the interaction-map coordinate envelope before scientific evidence is inspected.
- Admit only exact real-numeric NumPy arrays or exact built-in two-dimensional numeric sequences before NumPy materialization, reject complex/non-real storage and infinities, preserve `NaN` as missingness, and require wider integer/floating evidence to survive the Rust `f64` boundary without changing identity.
- Bound public and native interaction-map logical cells and coordinate requests at 20,000,000 cells, and bound the Rust symmetric-eigendecomposition workspace at 128 MiB before dense Gram/eigenvector allocation.

Gabriel symmetric factorization, singular values, coordinates, reconstruction, unexplained residuals, cross-term arithmetic, and other production numerical behavior remain Rust-owned and unchanged.
