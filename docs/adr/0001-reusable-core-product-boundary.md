# ADR-0001: Keep fast-mlsirm a reusable measurement core

- Status: **Accepted**
- Date: 2026-08-09
- Owner: `fast-mlsirm` measurement bounded context

## Context

The ecosystem needs both a reusable scientific library and a hosted assessment product. Mixing participant/session/consent persistence, HTTP APIs, tenant authorization, deployment configuration, and UI into the numerical/measurement package would make the library harder to reuse and would create reverse dependencies from scientific code into a product runtime.

`fast-mlsirm` already exposes domain-neutral assessment, rubric, scoring, calibration, validation, reporting, and psychometric capabilities. `ContextualWisdomLab/psychometrics-commons` is the canonical downstream hosted product.

## Decision

`fast-mlsirm` owns reusable measurement contracts and psychometric computation. Psychometrics Commons owns hosted product lifecycle and product persistence.

Allowed direction:

```text
standalone users / Psychometrics Commons / other consumers
                         |
                         v
                     fast-mlsirm
```

Forbidden direction:

```text
fast-mlsirm -> Psychometrics Commons ORM, HTTP, session, consent, UI, or deployment types
```

Cross-repository integration uses explicit versioned APIs, events, or immutable artifacts. Application databases are not shared.

## Invariants

1. The core package remains independently installable and useful.
2. There is no hosted participant/session/consent database in this repository.
3. Product-specific HTTP routes, identity mappings, deployment composition, and tenant lifecycle do not enter reusable contracts.
4. Domain adapters may project essay/RAG/enterprise evidence into shared measurement contracts, but may not fork independent calibration engines.
5. A future persistence layer in this repository requires a new ADR proving that its state is domain-neutral core state.

## Alternatives considered

### Put the hosted runtime inside fast-mlsirm

Rejected. It couples scientific releases to web/database lifecycle and makes standalone use depend on product infrastructure.

### Put all psychometric computation in the hosted product

Rejected. It duplicates scientific truth, weakens reuse, and makes independent validation harder.

### Share a database across services

Rejected. Ownership and migration authority become ambiguous; versioned contracts provide a cleaner MSA boundary.

## Failure and recovery

If a proposed feature requires a hosted-product type, the change fails the architecture gate and is moved to the owning downstream repository or redesigned as a reusable contract. Existing accidental reverse dependencies must be removed through a compatibility-preserving migration rather than normalized as precedent.

## Compatibility and rollback

Public core contracts are versioned/content-addressed. Downstream consumers can pin exact versions/artifacts. A breaking boundary change requires a superseding ADR and migration plan.

## Verification

Documentation and contract tests should assert the ownership direction and the absence of a recreated hosted runtime such as `services/assessment_runtime`.

## Consequences

This decision increases explicit integration work but preserves scientific independence, modular deployment, separate release cadence, and a clear acquisition/security boundary.
