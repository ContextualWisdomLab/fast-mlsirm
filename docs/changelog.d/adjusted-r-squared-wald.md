# Adjusted R squared on the Rust regression API

## Added

- `adjusted_r_squared(n, k, sse, sst)` is available from the Rust regression
  core and the Python package. `k` counts every design column, including the
  intercept, and both sums of squares must come from the same response.
- The one-degree-of-freedom contrast keeps `p_chi2` as the two-sided normal
  Wald tail `P(χ²₁ ≥ z²)`. Fixed synthetic checks cover that tail, the
  ten-column contrast order, and the adjusted R² input guards.
