# Inference covariance ownership

## Changed

- `vcov_from_hessian` and `standard_errors_from_vcov` now own inversion, pseudoinverse, and SE extraction in Rust.
