# Inference Rust covariance ownership

## Standards and primary research

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

Oakes, D. (1999). Direct calculation of the information matrix via the EM algorithm. *Journal of the Royal Statistical Society: Series B (Statistical Methodology), 61*(2), 479–482. https://doi.org/10.1111/1467-9868.00188

Pritikin, J. N. (2017). A comparison of parameter covariance estimation methods for item response models in an expectation-maximization framework. *Cogent Psychology, 4*(1), Article 1279435. https://doi.org/10.1080/23311908.2017.1279435

## Rationale

Observed-information assembly, positive-definiteness diagnostics, covariance inversion, and standard-error extraction are result-affecting numerical kernels. A single Rust owner prevents divergent NumPy implementations and keeps commercial inference paths fail-closed under singular or invalid information.

A second-order diagnostic is specifically a positive-definiteness check. Its tolerance may therefore be zero or positive, but a negative tolerance is not an innocuous numerical setting: it can classify a matrix with a small negative eigenvalue as passing. The Rust boundary rejects negative and non-finite tolerances before eigenvalue classification so callers cannot redefine the scientific meaning of a positive-definite information matrix.

The tolerance is a numerical diagnostic threshold, not evidence that the fitted model is identified, that uncertainty is calibrated, or that coverage is adequate. Parameter-recovery and interval-coverage evidence remain separate acceptance obligations.

## Implementation

`mlsirm_core::inference::{second_order_test, vcov_from_hessian, standard_errors_from_vcov}` owns the numerical diagnostics with PyO3 exports; Python wrappers validate/marshal/report only.
