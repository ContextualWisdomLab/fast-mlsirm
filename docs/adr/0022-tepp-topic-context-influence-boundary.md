# ADR-0022: TEPP posterior topic-context influence boundary

Status: **Proposed**
Date: 2026-08-26

## Context

Downstream products need to identify documents whose removal materially changes
topic conclusions across business-unit, process-unit, team, and person
contexts. Point estimates, binary topic assignments, diagonal normal
approximations, lexical thresholds, and caller-chosen weights discard the
posterior and multiple-membership uncertainty required for that interpretation.
TEPP owns temporal topic inference and publishes
`tepp.topic_context_posterior.v1`; fast-mlsirm owns reusable psychometric
arithmetic, not TEPP topic fitting or downstream persistence/UI.

## Decision

The Rust core admits only the exact TEPP v1 artifact after validating:

- a complete document-by-draw logistic-normal posterior grid;
- stable topic identities, topic activity and producer-fitted topic lineage;
- admitted `event_lineage_precedes` document relations;
- event-clock and historical-cutoff consistency;
- time-covering, provenance-bound business-unit, process-unit, team, and person
  multiple memberships whose source-derived weights are preserved exactly; and
- bounded resources, finite coordinates, unique records, and evidence digests.

The contract labels coordinates as posterior topic coordinates, never document
importance. The joint plausible values added by TEPP PR #253 are draws from the
full-data posterior `p(theta | D)`. They do not identify the deleted-data
posterior `p(theta | D \\ {i})`: v1 supplies neither producer-owned
leave-one-document-out draws nor the per-case likelihood contribution required
to reweight full-data draws. Consequently, the public entry point returns
`CaseDeletionRefitEvidenceUnavailable` even for a valid v1 artifact. Computing
a weighted context mean with one fixed draw removed was rejected because that
is finite-population aggregation leverage, not Bayesian model-refit influence.

A future producer contract may unlock the estimator only after it binds either
exact refit posterior draws or auditable per-case likelihood contributions to
the same model, prior, fit, snapshot, event clock, membership evidence, and
Event Lineage evidence. The Rust consumer must then pass identification,
true-parameter/deletion-effect recovery, interval coverage, and CPU/GPU parity.
Python may marshal the result but may not implement the likelihood,
case-deletion refit, weighting, or ranking.

### Accepted producer contract and final estimand

The preferred next wire artifact is
`tepp.topic_context_case_deletion_posterior.v1`. Its full-fit block binds the
run, immutable snapshot and source digest, cutoff and event clock, admitted
document set, model configuration and prior digests, stable topic order, and
an independent anchor basis containing per-topic anchor-term distributions and
its digest. Every admitted document has exactly one actual `D \\ {i}` refit
under those identical inputs. A refit records the deleted and retained document
sets, the removed incident Event Lineage and membership assertions, a bijective
permutation into the full-fit anchor basis, producer evidence that the
assignment optimum is unique, and a complete refit posterior draw set.
Non-unique or tied anchor alignment is a producer failure, never a consumer tie
break. Full and refit draws use independent, domain-separated randomness while
retaining a common draw-index cardinality for posterior contrasts. CPU/GPU
receipts bind implementation, objective, parameter, and draw digests plus a
method-derived numerical error bound; a caller-selected tolerance is forbidden.

For posterior draw `s`, topic `k`, dimension `l`, and context `h`, let `p_jk^s`
be the simplex probability obtained from the full-fit ALR coordinate and let
`w_jh` be the source-provided, time-valid membership weight. Define

```text
A_hk^s(D)      = sum_j w_jh p_jk^s      / sum_j w_jh
A_hk^s(D \\ i) = sum_(j != i) w_jh p_-i,jk^s / sum_(j != i) w_jh
Delta_ihk^s    = A_hk^s(D) - A_hk^s(D \\ i)
```

This posterior contrast is the final reusable estimand. It is evaluated
separately for business unit, process unit, team, and person contexts; there is
no cross-topic, cross-context, or cross-dimension fusion weight. A zero
post-deletion denominator is structurally unavailable, not zero. Event Lineage
enters only through TEPP's actual refit after removing the deleted document's
incident relations; fast-mlsirm assigns no relation weight and makes no causal
claim.

The consumer preserves signed draw-level contrasts and reports their posterior
mean, variance, and order-statistic interval. Within one identical
`(dimension, context, topic)` comparison set, document ordering may use
`E[abs(Delta)]`; exact equal values receive the same dense rank. The raw signed
posterior and interval remain authoritative, and no threshold converts
uncertainty into a binary important/not-important label.

## Invariants and acceptance evidence

1. Foreign schema versions, incomplete draw grids, missing hierarchy levels,
   non-finite values, temporal contradictions, unsupported relations, and
   malformed provenance fail closed.
2. Membership weights come only from the TEPP evidence artifact; no equal-share
   repair, normalization, threshold, tolerance rule, or locally selected weight
   is allowed.
3. Event Lineage is an observed relation input, not a causal effect claim and
   not a replacement for the topic posterior.
4. Releasing influence requires synthetic recovery of known deletion effects
   and calibrated uncertainty, plus Rust CPU/GPU numerical parity on the same
   objective. Contract validation tests alone do not satisfy that gate.
5. Full-data plausible values are not refit evidence. Importance reweighting is
   unavailable unless the producer supplies each deleted case's joint
   likelihood contribution on the retained draws and the consumer verifies its
   diagnostics; no constant, rank, threshold, or locally reconstructed
   likelihood may substitute for that evidence.
6. The deletion-refit artifact covers every admitted document exactly once,
   and every retained set is exactly the admitted set minus that document.
7. Anchor permutations are bijective and producer-certified unique. A tie
   fails the artifact rather than being resolved by UUID, position, or lexical
   order.
8. Posterior contrasts never pool BU, PU, team, person, context, or topic cells
   through a local weight. Exact scalar ties remain ties.

## Consequences and alternatives

This prerequisite gives consumers an auditable fail-closed boundary without
prematurely naming a pseudo-posterior score “importance.” It delays the
Dashboard ranking until TEPP produces the governed artifact and the estimator
passes its scientific gate. Rank fusion, binary topic membership, diagonal
Laplace substitution, and fixed channel weights were rejected because none
targets the requested posterior multiple-membership deletion estimand.

## Security, privacy, and compatibility

The reusable artifact carries opaque identifiers and digests, not source text
or names. Parsing is size bounded and rejects unknown fields. This is an
additive Rust API. Any TEPP wire change requires a new schema version and a
reviewed compatibility decision; v1 payloads are never guessed or repaired.

## References

Browne, W. J., Goldstein, H., & Rasbash, J. (2001). Multiple membership
multiple classification (MMMC) models. *Statistical Modelling, 1*(2), 103–124.
https://doi.org/10.1177/1471082X0100100202

Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel item
response theory model using Gibbs sampling. *Psychometrika, 66*(2), 271–288.
https://doi.org/10.1007/BF02294839

Bradlow, E. T., & Zaslavsky, A. M. (1997). Case influence analysis in Bayesian
inference. *Journal of Computational and Graphical Statistics, 6*(3), 314–331.
https://doi.org/10.1080/10618600.1997.10474731

Jackson, C. H., Sharples, L. D., Thompson, S. G., Duffy, S. W., & Couto, E.
(2009). Bayesian case influence diagnostics for survival models. *Biometrics,
65*(1), 116–124. https://doi.org/10.1111/j.1541-0420.2008.01049.x

American Educational Research Association, American Psychological
Association, & National Council on Measurement in Education. (2014).
*Standards for educational and psychological testing*. American Educational
Research Association.
