# ADR-0021: TEPP-posterior topic-context influence

Status: **Accepted**  
Implementation maturity: **influence arithmetic available; estimator unavailable**
Date: 2026-08-25

## Decision

`fast-mlsirm` will estimate the LineageWeave ADR-0210 case-deletion diagnostic

\[
D_{dkl}=(\hat\psi_{kl,-d}-\hat\psi_{kl})^\top
I_{kl}(\hat\psi)(\hat\psi_{kl,-d}-\hat\psi_{kl})
\]

from TEPP `tepp.topic_context_posterior.v1` posterior logistic-normal plausible
values, event time, and explicit time-valid business-unit, PU, team, and person
multiple memberships. Rust owns likelihood, observed information, deletion
refits, posterior-draw combination, influence arithmetic, CPU fixed-worker
execution, and the real GPU path.

The Rust `case_deletion_influence_cpu` primitive implements the diagnostic for
each plausible-value draw and its equal-mass Monte Carlo posterior expectation.
It rejects malformed, non-finite, non-symmetric, or negative-quadratic-form
observed-information inputs. This is owner-side arithmetic only: it does not
claim that deletion refits or observed-information blocks have been produced.

The existing crossed binary MAP estimator is a different estimand. Posterior
coordinates cannot be thresholded into binary responses or collapsed to point
estimates. `fit_topic_context_influence` therefore validates the producer
identity and returns `topic_context_influence_estimator_unavailable` until the
exact deletion-refit estimator, true-parameter recovery, and real GPU parity
are implemented. No fallback, keyword, heuristic, arbitrary weight, or local
Python arithmetic is permitted. GPU remains unavailable rather than silently
using CPU while claiming GPU execution.

## Acceptance

Promotion requires recovery of injected context effects and influential cases
under nested, crossed, weighted multiple-membership, time-varying, sparse,
unbalanced, missing, tied, masked, and jointly influential designs. Report
bias, RMSE, interval coverage, convergence, identification, posterior-draw
coverage, deterministic CPU worker parity, and hardware-proven GPU parity. A
CPU fallback cannot satisfy the GPU claim.

## References

Browne, W. J., Goldstein, H., & Rasbash, J. (2001). Multiple membership
multiple classification (MMMC) models. *Statistical Modelling, 1*(2), 103–124.
https://doi.org/10.1177/1471082X0100100202

Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel IRT
model using Gibbs sampling. *Psychometrika, 66*(2), 271–288.
https://doi.org/10.1007/BF02294839

Shi, L., & Chen, G. (2008). Case deletion diagnostics in multilevel models.
*Journal of Multivariate Analysis, 99*(9), 1860–1877.
https://doi.org/10.1016/j.jmva.2008.01.023
