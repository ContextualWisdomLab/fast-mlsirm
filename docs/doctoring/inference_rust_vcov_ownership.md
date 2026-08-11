# Inference Rust covariance ownership

## Standards

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

Pritikin, J. N. (2017). A comparison of parameter covariance estimation methods for item response models in the presence of multidimensionality. *Applied Psychological Measurement*. (Oakes identity / observed information context)

## Rationale

Observed-information inversion and standard-error extraction are pure numeric kernels. A single Rust owner prevents divergent NumPy inverses and keeps commercial inference paths fail-closed under singular information.

## Implementation

`mlsirm_core::inference::{vcov_from_hessian, standard_errors_from_vcov}` with PyO3 exports; Python wrappers marshal only.
