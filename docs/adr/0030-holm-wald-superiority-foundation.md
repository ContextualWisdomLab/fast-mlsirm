# ADR-0030: Holm–Wald candidate-superiority foundation

Status: Proposed  
Date: 2026-10-02  
Supersedes: none  
Superseded by: none

## Context

`ContextualWisdomLab/contextual-orchestrator` issue dependency
`fast-mlsirm#2315` exposed a decision boundary that the protected `0.11.4`
package cannot satisfy: `predict_proba` returns point estimates, while an
arbitrarily small unique maximum does not establish decision superiority.
`FitResult` also does not expose the joint covariance of the contextual MLSRM
predictions. Therefore a complete calibrated, route-authorizing contract cannot
be honestly added by comparing point estimates or inventing a margin.

The bounded first dependency is a reusable numerical procedure that can decide
whether one candidate is statistically superior when a valid joint estimate
and covariance are available. The remaining estimation, calibration, coverage,
convergence, identification, and provenance work remains open.

## Decision drivers

- No point-estimate maximum, input order, or undocumented margin may select a candidate.
- The full candidate family needs explicit type-I error control.
- Numerical decision arithmetic belongs to Rust; Python validates and marshals.
- A consumer-selected error level requires a separately governed policy and has no package default.

## Ownership and dependency direction

`fast-mlsirm` owns the domain-neutral statistical kernel and typed result.
Consumers may depend on an immutable released contract. They may not copy the
kernel or treat this Proposed ADR/open PR as released authority. Hosted routing,
decision loss, provider choice, and credential policy remain downstream.

## Decision

Add versioned algorithm `holm-wald-one-sided-v1`:

1. accept a finite candidate-estimate vector, its finite symmetric positive-definite covariance, and a caller-supplied family-wise error rate in `(0, 1)`;
2. convert every submitted binary64 covariance entry to a common-scale exact
   dyadic integer, then verify positive definiteness from every leading
   principal-minor sign using fraction-free Bareiss elimination and Sylvester's
   criterion;
3. form every ordered estimate difference and covariance contrast as exact
   dyadic integers, enclose the exact variance, standard error, and Wald ratio
   with directionally conservative binary64 bounds, and fail closed if finite
   bounds cannot be produced;
4. calculate a conservative upper bound for the one-sided Wald normal-tail
   probability in Rust from the Wald-ratio lower bound and the documented
   `erfc` approximation error bound, rounding the final bound outward;
5. apply Holm's sequentially rejective procedure across the complete ordered
   family by exact dyadic cross multiplication rather than a rounded `alpha / m`
   cutoff; and
6. return a winner only when exactly one candidate rejects every outgoing null; otherwise return `indeterminate`.

This foundation does not itself accept fit-status or calibration booleans. That
omission is deliberate: an unverified flag would let a caller relabel point
evidence as calibrated. Route authorization remains unavailable until #2315
adds an owner-produced covariance/calibration artifact and the typed contract
binds its convergence, identification, coverage, failure denominator, and
provenance.

## Invariants / acceptance evidence

1. Equal or statistically unresolved candidates under the supplied covariance
   return `indeterminate`.
2. Candidate permutation cannot change the winner identity.
3. Nonfinite estimates, nonsymmetric covariance, and non-positive-definite covariance fail closed.
4. Rust owns exact-dyadic positive-definiteness and contrast arithmetic,
   directional binary64 bounds, outward-rounded conservative p-value bounds,
   exact Holm critical comparisons, and winner selection.
5. No family-wise error default is provided.

## Non-goals and claims not made

- No MLSRM prediction covariance estimator is added.
- No convergence, identification, or calibration certificate is inferred.
- No posterior probability, practical superiority, utility, or causal claim is made.
- A `superior` kernel result alone does not authorize routing or another operational decision.

## Consequences and trade-offs

### Benefits

- Removes input-order and unique-maximum tie-breaking from the future owner contract.
- Controls family-wise type-I error without requiring independent contrasts.
- Supplies a small Rust primitive that future calibrated estimator work can reuse.

### Costs / risks

- Wald inference depends on a valid asymptotic joint covariance and regular identification.
- Testing every ordered pair is conservative; lower power is accepted in this safety-first foundation.
- Exact covariance certification adds the permissively licensed `num-bigint`
  dependency and cubic integer arithmetic before Wald computation; this bounded
  foundation prioritizes invariant admission over floating-point shortcuts.
- The consumer remains deliberately blocked until the upstream covariance/calibration contract is complete and released.

## Alternatives considered

### Unique maximum or fixed probability margin

Rejected. Neither accounts for estimation uncertainty, and any fixed margin
would be an undocumented application loss threshold.

### Uncorrected pairwise Wald tests

Rejected because the probability of at least one false superiority conclusion
increases with the candidate family.

### Caller-supplied `calibrated=True`

Rejected because a Boolean assertion is not executable calibration or coverage evidence.

### Bayesian posterior dominance

Deferred. The current contextual MLSRM fit is not a posterior sampler, and an
invented prior or posterior-probability cutoff would violate the evidence boundary.

## Failure, degraded, and recovery behavior

Invalid numerical evidence raises a bounded `ValueError`. Valid evidence that
does not reject every outgoing null returns `indeterminate`. Downstream systems
must retain their higher-assurance path. Rollback removes the Proposed API; no
released schema or stored artifact is migrated by this foundation.

## Security and privacy implications

The API accepts only candidate identifiers and numeric aggregates. It does not
accept prompts, provider responses, credentials, or secret-bearing diagnostic
text. Error messages describe package-owned validation failures only.

## Compatibility, migration, and rollback

The change is additive and Proposed. Algorithm identity is explicit so a future
replacement cannot silently change historical meaning. No consumer may pin it
until an immutable release includes the complete #2315 contract.

## Verification and release evidence

Before this ADR may become Accepted, #2315 still requires owner-produced joint
prediction uncertainty, true-parameter bias/RMSE and interval/calibration
coverage, convergence and failed-fit denominators across null/near/separated/
sparse/ill-conditioned designs, exact provenance, Rust/Python and CPU parity,
hosted exact-head checks, independent approval, ordinary merge, and an immutable release.

## Research and standards basis

American Educational Research Association, American Psychological Association,
& National Council on Measurement in Education. (2014). *Standards for
educational and psychological testing*. American Educational Research Association.

Holm, S. (1979). A simple sequentially rejective multiple test procedure.
*Scandinavian Journal of Statistics, 6*(2), 65–70.
https://www.jstor.org/stable/4615733

Bareiss, E. H. (1968). Sylvester's identity and multistep integer-preserving
Gaussian elimination. *Mathematics of Computation, 22*(103), 565–578.
https://doi.org/10.1090/S0025-5718-1968-0226829-0

## Follow-ups

- `fast-mlsirm#2315`: joint contextual-MLSRM prediction uncertainty, calibration/coverage evidence, and the complete versioned result.
- Consumer integration only after immutable owner release and exact-version bump.

## Reversal / supersession conditions

Supersede this ADR if simulation shows Wald coverage is inadequate for the
intended contextual MLSRM design, identification prevents a valid joint
covariance, or a peer-reviewed finite-sample/posterior method with stronger
validated operating characteristics replaces the foundation.
