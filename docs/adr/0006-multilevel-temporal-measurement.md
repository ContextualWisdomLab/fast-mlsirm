# ADR-0006: Preserve multilevel, multiple-membership, and temporal structure

- Status: **Proposed** for the dedicated governed contract/fitter stack; existing released multilevel-adjacent capabilities remain governed by their current APIs.
- Date: 2026-08-09
- Owner: reusable measurement-model layer

## Context

Respondents, items, raters, departments, clients, projects, prompts, and measurement occasions often form nested, cross-classified, weighted multiple-membership, repeated, or longitudinal structures. Flattening those contexts into independent person/item rows can create atomistic interpretations, underestimated uncertainty, and conflation of stable trait, contextual effect, state change, drift, and local dependence.

An active development branch currently carries a dedicated governed multilevel/longitudinal contract namespace that is not yet accepted on protected `main`. This ADR records the required architecture without claiming that unmerged APIs are released.

## Decision

Reusable measurement contracts and future numerical models must be capable of preserving:

- explicit context dimensions and dimension-qualified context identities;
- one-hot nesting, cross-classification, and positive weighted multiple membership;
- per-observation membership checks required by the declared design;
- repeated respondent/item/rater/task occasions;
- explicit event/occasion order and provenance; and
- distinct parameterizations for discrete occasion-step autoregression, elapsed-time/continuous-time dynamics, lagged-response dependence, and evaluator drift.

These structures are part of measurement design, not labels to be silently discarded before fitting.

## Invariants

1. Context membership is explicit; labels are not guessed into random-effect families.
2. Multiple-membership weights are validated and not silently renormalized when the contract says the caller supplies exact weights.
3. Cross-classified contexts preserve each dimension independently.
4. Timestamps/order are evidence; an AR(1) coefficient over discrete occasions is not described as a continuous-time process unless the likelihood uses elapsed gaps.
5. A structurally valid design is not evidence that the estimator is identified.
6. Numerical support is not released until simulation demonstrates scale-aligned bias/RMSE, coverage, convergence, and backend parity where applicable.
7. TEPP/Gyeot may own adjacent collection/event analytics, but reusable psychometric design and kernels remain domain-neutral in `fast-mlsirm`.

## Alternatives considered

- **Flatten all contextual records and use cluster-robust SE later:** rejected as a default because the latent model itself can be misspecified and contextual effects become uninterpretable.
- **Treat every context as a testlet:** rejected because testlets, hierarchical contexts, multiple membership, and longitudinal states answer different scientific questions.
- **Move all temporal measurement to TEPP:** rejected because psychometric estimators still need an explicit reusable contract; TEPP remains an integration/analytics owner rather than the only measurement representation.

## Failure and recovery

Unsupported/confounded/disconnected designs fail before estimation when detectable. An estimator that cannot represent a required hierarchy/time mechanism must return unsupported rather than flattening the data. Recovery is a different supported model/design, not hidden aggregation.

## Compatibility and rollout

The dedicated contract namespace should enter through a versioned additive API. Existing single-level callers continue to operate unchanged. A later numerical fitter is a separate reviewed slice with its own recovery and resource evidence.

## Verification

- nested, cross-classified, and weighted multiple-membership construction tests;
- order/timezone/occasion invariants;
- disconnected/confounded design checks where identifiable;
- serialization/fingerprint replay;
- true-parameter recovery for each released numerical structure;
- CPU/GPU parity where GPU kernels are claimed.

## Consequences

This increases model/design complexity, but prevents the reusable library from institutionalizing an atomistic fallacy or treating time as unstructured noise.
