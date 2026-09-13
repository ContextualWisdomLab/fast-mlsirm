# Multilevel M2 moment and covariance Rust boundary

## Decision

The public multigroup and multilevel M2 paths delegate population moment
integration and cluster moment-covariance construction to the compiled Rust
core through PyO3. The change preserves the existing simple-structure
parameterization and M2/RMSEA2 estimand; it does not introduce a new
discrimination-vector model or change the degrees-of-freedom contract.

The multilevel moment kernel integrates one shared cluster-intercept node across
all items, while residual trait integration remains factorized according to the
existing simple-structure `factor_id` contract. The covariance kernel retains
the finite-cluster correction and uses compact cluster labels. Missing native
entry points fail closed instead of silently selecting the Python reference
arithmetic.

## Scientific boundary

M2 is a limited-information fit statistic based on univariate and bivariate
response margins. The implementation continues to use the repository's
paper-supported reduction: items assigned to the same factor share a residual
trait node, items on distinct factors are integrated independently conditional
on the common latent-space node, and multilevel cells additionally share one
cluster-intercept node. The Rust migration is therefore an ownership and
parity change, not a new estimator claim.

Cluster covariance is a finite-sample diagnostic for the multilevel M2
projection. It does not establish variance-component identification,
measurement invariance, causal contextual effects, or longitudinal validity.
Those claims require separate recovery and identification evidence.

## Verification

- Rust unit tests check shared-factor products, shared cluster-node integration,
  finite-cluster correction, and fail-closed shape/label validation.
- Python ownership tests reject the NumPy production moment and projection
  paths, record the Rust/PyO3 entry points, and compare native results with the
  explicit NumPy parity reductions.
- Existing multigroup, multilevel, M2, and cluster-covariance tests remain the
  behavioral acceptance suite.

## APA 7 references

Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel IRT
model using Gibbs sampling. *Psychometrika, 66*, 271–288.
https://doi.org/10.1007/BF02294839

Maydeu-Olivares, A., & Joe, H. (2005). Limited- and full-information
estimation and goodness-of-fit testing in 2^n contingency tables. *Journal of
the American Statistical Association, 100*(471), 1009–1020.
https://doi.org/10.1198/016214504000002069

Maydeu-Olivares, A., & Joe, H. (2006). Limited information goodness-of-fit
testing in multidimensional contingency tables. *Psychometrika, 71*, 713–732.
https://doi.org/10.1007/s11336-005-1295-9
