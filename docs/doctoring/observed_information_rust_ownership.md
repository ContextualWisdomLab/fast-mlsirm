# Doctoring: Rust-owned observed information and second-order tests

## Claim

Dense finite-difference Hessian assembly (coefficients and symmetrisation) and
positive-definiteness eigenvalue diagnostics for observed information are owned
by the compiled Rust numeric core. Python validates shapes, evaluates the scalar
objective at FD offsets, and marshals results.

## Standards and literature (APA 7th)

Pritikin, J. N. (2017). A comparison of parameter covariance estimation methods
for item response models in an expectation-maximization framework. *Cogent
Psychology, 4*(1), Article 1279435.
https://doi.org/10.1080/23311908.2017.1279435

Oakes, D. (1999). Direct calculation of the information matrix via the EM
algorithm. *Journal of the Royal Statistical Society Series B: Statistical
Methodology, 61*(2), 479–482. https://doi.org/10.1111/1467-9868.00188

## Verification

- Rust unit tests for positive-definite / indefinite diagnostics and quadratic
  Hessian recovery.
- Python ownership sentinels for `core.observed_information` and
  `core.second_order_test`.
