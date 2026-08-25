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
importance. Until a Rust continuous-posterior multiple-membership estimator has
an identification study, true-parameter/deletion-effect recovery, interval
coverage, and exact CPU/GPU parity, the public influence entry point returns
`EstimatorUnavailable`. Python may marshal the future result but may not
implement the likelihood, case-deletion refit, weighting, or ranking.

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

American Educational Research Association, American Psychological
Association, & National Council on Measurement in Education. (2014).
*Standards for educational and psychological testing*. American Educational
Research Association.
