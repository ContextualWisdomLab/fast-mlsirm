# All-fixed bifactor focal calibration with free factor means and variances

`fast_mlsirm.bifactor_grm.fit_bifactor_grm_fipc_full` estimates the general and
specific population means and diagonal variances, keeping all supplied item
slopes and boundary intercepts fixed. Factor covariances remain zero. It is a
separate path: the existing partial-anchor `fit_bifactor_grm_fipc` retains its
zero-specific-mean contract. The new path produces no participant scores and
performs no reference calibration or item maximization.

## Scientific scope

Cai et al. (2011, pp. 230–232, Eqs. 15–17) describe estimable factor means and
variances with common-item scale restrictions and reduced marginal integration.
Kim (2006, pp. 360–363) describes fixed-item calibration with iterated population
updates. On each sweep, normal quadrature nodes are transformed as
`factor_mean + factor_sd * standard_node`; the standard weights do not change.
For complete observed responses, category-summed expected counts for any one
item in a specific block equal that block's joint posterior mass at each
(general, specific) node. Its first and second moments yield the corresponding
population update. The general moments come from the same reduced posterior.

All responses must be observed integer categories. Missing values are refused:
using one item's observed counts as the posterior mass would otherwise be
incorrect. Current structural validation also requires at least two items per
specific factor and every declared category observed in each item. These are
implementation restrictions, not general requirements for all-fixed models.
All means and initial SDs must be supplied; SDs must be finite and positive.
Quadrature counts, iteration budget, stopping tolerance and device are required
caller arguments, with no quadrature defaults, upper caps or silent clamps.

This is deterministic MML-EM. Matching an existing model's loading pattern,
fixed items and free population parameters does **not** make it the same
numerical estimator as a retained stochastic MHRM fit, nor does it prove that
its parameters or likelihood are accurate enough for a study. Original adopted
scores and regression outputs must not be replaced on that basis.

## Stopping and precision

All population updates are full EM steps, without damping, rollback or variance
clamping. Nonpositive or nonfinite variances fail explicitly. Convergence
requires the existing nondecreasing observed-likelihood guard, a likelihood
change no greater than `tol * (1 + abs(previous_loglik))`, and an EM-map
maximum displacement no greater than `tol`. Displacement covers every mean and
log SD at the returned population state. At budget exhaustion, the fit reports
`converged=False` unless those same convergence conditions genuinely hold.

The GPU performs the existing reduced expected-count/moment E-step in binary32.
Its scalar likelihood can round down while the same-support binary64 likelihood
increases. Therefore, when GPU execution is used, the observed likelihood used
for strict monotonicity and stopping is recomputed on the CPU in binary64, on
exactly the same items, affine nodes and weights. This scalar evaluation does
not replace the GPU posterior moments, does not permit fallback and does not
relax the likelihood guard. Explicit `device="gpu"` requires actual GPU dispatch,
nonempty hardware identity and no recorded CPU fallback. `auto` may fall back.
This is a hybrid execution path, not an all-arithmetic-on-GPU claim.

Stopping tolerance is not an accuracy, adjacent-quadrature or CPU/GPU parity
threshold. Study acceptance requires caller-specified scientific criteria and
actual participant-data evidence separately. A synthetic convergence result,
physical device receipt or source review is not study acceptance.

## References

Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information item
bifactor analysis. *Psychological Methods, 16*(3), 221–248.
https://doi.org/10.1037/a0023350

Kim, S. (2006). A comparative study of IRT fixed parameter calibration methods.
*Journal of Educational Measurement, 43*(4), 355–381.
https://doi.org/10.1111/j.1745-3984.2006.00021.x

The cited method sections were read from retained full texts. Copyrighted PDFs
are not redistributed here; these citations and method summary supply the
source basis without copying the articles.
