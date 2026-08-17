# Bradley-Terry control trust boundary

## Fixed

- `bradley_terry_mm()` now validates and normalizes `alpha`, `max_iter`, and `tol` before caller data materialization or compiled-core discovery. Exact built-in and supported NumPy scalar identities remain compatible; booleans, numeric subclasses, and arbitrary conversion providers are rejected without executing caller callbacks.
- `max_iter` is bounded by the package-wide `MAX_MAX_ITER` resource ceiling. Exact integers that overflow IEEE-754 conversion raise a package-owned `ValueError` before data materialization, matching the ICC adapter. Bradley-Terry MM arithmetic, convergence, normalization, estimates, and result statistics remain Rust-owned.
