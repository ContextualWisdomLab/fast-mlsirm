# ADR-0001 — Reusable Core and Hosted Product Boundary

- **Status:** Accepted
- **Date:** 2026-08-09
- **Owners:** fast-mlsirm maintainers; downstream Psychometrics Commons maintainers

## Context

The CWL ecosystem contains both a reusable psychometric library and hosted
product concerns. Combining them would couple mathematical releases to HTTP,
identity, persistence, consent, tenant and UI lifecycles and would make the
measurement core difficult to reuse independently.

## Decision

`fast-mlsirm` remains the domain-neutral measurement/computation component.
`ContextualWisdomLab/psychometrics-commons` is the canonical hosted product and
owns HTTP/admin APIs, participant/session/consent/result lifecycle, product
persistence/migrations, tenant/resource authorization, client applications,
research-release orchestration and deployment composition.

`fast-mlsirm` must not recreate a hosted Assessment Runtime under
`services/assessment_runtime` and must not import downstream product code.

## Invariants

- Downstream products consume explicit versioned contracts or immutable artifacts.
- Product-specific ORM/database/HTTP/UI fields do not enter core contracts merely
  for one consumer.
- Keyverse, TEPP, Gyeot, semantic-data-portal, contextual-orchestrator and
  EgressWeave remain independently owned bounded contexts.
- A consumer can install and test `fast-mlsirm` without running any CWL service.

## Alternatives considered

1. **Hosted monolith in fast-mlsirm** — rejected: reverses dependency direction
   and mixes operational/product release cadence with scientific kernels.
2. **New shared mega-schema repository** — rejected unless a future independently
   justified contract package becomes necessary; current canonical contracts
   already exist here.
3. **Separate hosted product repository consuming fast-mlsirm** — accepted.

## Consequences

Positive: clearer system-of-record ownership, independent releases, easier third-
party reuse, smaller attack surface and more defensible scientific provenance.
Negative: cross-repository compatibility must be tested explicitly and product
features cannot be implemented by directly reaching into library internals.

## Failure / degraded behavior

If a downstream consumer needs a missing field or behavior, first decide whether
it is reusable measurement semantics. If not, keep it downstream. If yes, add it
to fast-mlsirm through a versioned contract with compatibility tests.

## Security and privacy

Hosted identity and tenant authorization do not become ambient library concerns.
This limits credential and PII exposure in the computation layer.

## Migration / rollback

Any product-specific code accidentally introduced here should be extracted behind
an explicit contract. Rollback is a normal library release rollback; downstream
products pin a known compatible contract version/artifact.

## Verification

- Documentation contract tests pin the prohibited reverse dependency.
- Package/release tests prove independent installation.
- Cross-repository integrations use public exports or serialized artifacts.

## Supersession criteria

Supersede only if a documented multi-product need proves that a different
bounded-context split produces lower coupling without making the mathematical
core depend on hosted lifecycle/persistence.
