# Complex-valued curvature admission

## Fixed

- Reject complex-valued Hessian and covariance matrices before any `float64` narrowing can discard imaginary components and alter second-order, covariance, or standard-error evidence.
- Keep eigendecomposition, inversion/pseudoinversion, and standard-error arithmetic in the Rust core while preserving existing real square-matrix contracts.
