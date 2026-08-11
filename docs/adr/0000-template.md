# ADR-NNNN: Decision title

Status: Proposed
Date: YYYY-MM-DD  
Supersedes: none  
Superseded by: none

## Context

Describe the observed product/scientific/technical problem and current protected-main behavior. Separate facts from assumptions, active PR work, and future plans.

## Decision drivers

- Driver one.
- Driver two.

## Ownership and dependency direction

State the owning bounded context/repository and which dependencies are allowed or forbidden. Explicitly note whether the decision changes the `fast-mlsirm` reusable-core vs Psychometrics Commons hosted-product boundary.

## Decision

State the decision precisely enough that code/tests/documentation can determine compliance. If only part of the decision is implemented, mark the ADR Proposed or distinguish the accepted invariant from proposed implementation.

## Invariants / acceptance evidence

1. Invariant tied to a test, recovery study, security check, or other exact evidence.
2. Invariant tied to a failure/degraded/recovery rule.

## Non-goals and claims not made

List adjacent capabilities or interpretations this decision does not authorize.

## Consequences and trade-offs

### Benefits

- Benefit.

### Costs / risks

- Cost or risk.

## Alternatives considered

### Alternative A

Why it was considered and rejected/deferred.

### Alternative B

Why it was considered and rejected/deferred.

## Failure, degraded, and recovery behavior

Describe fail-closed/fallback behavior, retry/idempotency if applicable, operator evidence, recovery/rollback, and how a failed migration or rollout is handled.

## Security and privacy implications

Cover new credentials/permissions, data classification, PII/sensitive evidence, native/provider trust, supply-chain surface, retention, auditability, and threat-model changes where applicable.

## Compatibility, migration, and rollback

Define public API/schema/artifact compatibility, migration steps, old-artifact interpretation, rollback and how supersession preserves decision history.

## Verification and release evidence

List required unit/property/fuzz/security tests, Rust↔Python or CPU↔GPU parity, true-parameter recovery/coverage, documentation/traceability, package/release evidence, or downstream contract checks before the implementation may be called Accepted/released.

## Research and standards basis

Use APA 7 references to primary peer-reviewed methods and current official standards/specifications where material. Mark preprints as preprints; do not use legacy software output as the scientific oracle when the original method is available.

## Follow-ups

Record deliberately deferred bounded work, with owning issue/PR only as a tracking aid rather than the decision authority.

## Reversal / supersession conditions

State what evidence, standard, product-boundary change, or implementation failure should trigger a new superseding ADR instead of silently editing this accepted history.
