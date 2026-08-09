# ADR-0003: Use canonical versioned, content-addressed measurement contracts

- Status: **Accepted**
- Date: 2026-08-09
- Owner: assessment/rubric/scoring contract layer

## Context

Assessment, rubric, generated-item, scoring, calibration, and report artifacts pass through several modules and sometimes across services. If each adapter invents its own schema or uses mutable display identifiers as authority, a later result cannot prove which exact measurement specification produced it.

## Decision

`fast-mlsirm` maintains one domain-neutral contract family. `RubricSpecification` is the rubric source of truth; `AssessmentSpec` and scoring policies reference exact rubric fingerprints instead of copying score levels or construct definitions.

Canonical governed artifacts are:

- explicitly schema/versioned;
- immutable after construction;
- deterministically serialized;
- identified by complete content fingerprints for durable provenance;
- allowed to expose bounded descriptive public handles in addition to full fingerprints; and
- replay-validated so an object cannot be rebound to a different rubric, blueprint, response, rater, or calibration artifact without changing its identity.

## Invariants

1. Domain adapters reuse shared contracts and do not create parallel rubric, observation, calibration, or result schemas.
2. A semantic revision changes its version/fingerprint; published artifacts are not edited in place.
3. Cross-object references are checked against package-owned canonical objects before a fit, report, or release artifact is trusted.
4. Caller mutation of nested metadata cannot change a constructed artifact.
5. Human-readable/display identifiers are never accepted as a substitute for exact provenance.
6. Sensitive source text is omitted from durable metadata when a digest/reference is sufficient for reconstruction by the authorized owner.

## Alternatives considered

- **Mutable IDs plus database timestamps:** rejected because identity depends on one hosted persistence implementation and cannot prove content equality.
- **Separate schemas per domain adapter:** rejected because calibration and reporting semantics diverge.
- **Full raw source snapshots in every artifact:** rejected because it unnecessarily expands privacy and storage scope.

## Failure and recovery

Missing, duplicated, unknown, stale, or fingerprint-mismatched references fail before numerical execution or report publication. A compatibility migration creates a new contract version and explicit translator; it does not reinterpret an old fingerprint as new content.

## Compatibility and rollback

Older supported schema versions remain interpretable through documented version adapters. A breaking contract revision requires a migration plan, tests for old artifacts, and a superseding ADR if the identity model changes.

## Verification

Determinism, direct-construction bypass, replay/forgery, mutation, cross-reference, size/depth, public-export, and serialization round-trip tests are required for canonical artifact types.

## Consequences

Content addressing creates more explicit versioning work, but it enables reproducible calibration, audit, distributed MSA integration, and reliable release provenance without requiring a shared database.
