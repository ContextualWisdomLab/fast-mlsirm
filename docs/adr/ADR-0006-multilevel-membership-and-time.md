# ADR-0006 — Multilevel, Multiple-Membership and Temporal Structure

- **Status:** Accepted
- **Date:** 2026-08-09
- **Decision owner:** `fast-mlsirm`
- **Implementation status:** contracts/design direction accepted; numerical estimators require feature-specific recovery evidence before support claims

## Context

Psychometric observations commonly arise inside schools, organizations, raters, prompts, tasks, testlets, repeated occasions, multiple teams or overlapping contexts. Flattening these structures into person-only records can commit an atomistic fallacy, confound contextual effects with individual traits and understate uncertainty. Time is also not interchangeable with an unordered group label.

## Decision

`fast-mlsirm` treats scientifically material hierarchy, cross-classification, weighted multiple membership, testlets/facets and repeated occasions as explicit measurement-design information.

Reusable contracts preserve qualified context dimension + context identity, weights, observation/respondent identity, occasion ordering and revision fingerprints. New numerical multilevel/longitudinal models are accepted only after identification analysis and Rust true-parameter recovery.

Broader event/relationship/trajectory analytics remain TEPP's bounded context. `fast-mlsirm` may hand off exact versioned measurement/occasion artifacts but does not become a competing general temporal-event platform.

## Invariants

- Context labels are dimension-qualified; the same label in two dimensions is not silently treated as one random-effect level.
- Multiple-membership weights are validated explicitly and are not silently normalized unless the public contract says normalization occurs.
- Aggregate contracts replay/verify package-owned child artifacts and reject provenance mutation.
- Time offsets/order are preserved. A discrete occasion-step AR(1) parameter is not described as continuous-time merely because timestamps exist.
- Rater/testlet/occasion structure is not converted into substantive person dimensions without evidence.
- A contract being structurally valid is not proof that its estimator is identified, unbiased or causally interpretable.

## Alternatives considered

1. Aggregate all context into person-level fixed covariates — rejected as the default because it cannot represent all nested/cross-classified dependence.
2. Put every temporal/event concern in `fast-mlsirm` — rejected due TEPP overlap.
3. Explicit reusable measurement-design contracts plus Rust model extensions when justified — accepted.

## Consequences

APIs and simulations require richer design metadata. This increases upfront complexity but prevents false precision and allows future cross-classified, multiple-membership and longitudinal estimators to share auditable inputs.

## Failure / degraded behavior

Disconnected/confounded designs fail closed where the estimator requires connectedness/identification. If continuous-time dynamics are not implemented, return only ordering/discrete-step semantics rather than inventing interval-adjusted parameters. If a broader temporal analysis belongs to TEPP, export a versioned artifact rather than reimplementing it here.

## Security and privacy

Context and temporal identifiers can increase re-identification risk. Public or portable artifacts should prefer opaque identifiers and minimum necessary context; raw organizational/participant identity remains downstream and access-controlled.

## Verification

Future mathematical releases include simulations with realistic cluster sizes, cross-classification, weighted membership imbalance, missingness, irregular occasions and drift, reporting parameter bias/RMSE/coverage, convergence, identification failure and CPU/GPU parity where applicable.

## Sources

Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel IRT model. *Psychometrika, 66*, 271–288. https://doi.org/10.1007/BF02294839

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

## Supersession criteria

Supersede if responsibility for a reusable measurement-design class moves to a separate shared mathematical component with equal contract/version/recovery coverage and without duplicating TEPP or hosted-product ownership.
