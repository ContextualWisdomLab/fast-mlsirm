# Scoring Observation and Engine Contracts Design

## Objective

Add the second executable slice of the provider-neutral automated-scoring core. The merged `AssessmentSpec` defines what may be scored; this slice defines how one exact response is presented to a human or automated engine and how scored, abstained, failed, or excluded observations return with complete provenance.

The slice performs validation, canonicalization, deterministic fixture execution, and provenance only. It adds no model inference, provider SDK, network call, calibration, aggregation, uncertainty calculation, DIF, or utility arithmetic.

## Product boundary

```text
AssessmentSpec + RubricSpecification + response fingerprint
        ↓
factory-sealed ScoringRequest
        ↓
runtime-checkable ScoringEngine
        ↓
factory-sealed ScoreObservation[]
        ↓
factory-sealed ScoringResult
```

Raw response, prompt, source, and provider-output text stay outside the governed shared contracts. The request retains exact fingerprints, bounded content statistics, task/occasion/respondent identities, and criterion scope.

## Components

### `scoring.execution`

- `EngineKind`: `human_engine` or `automated_engine`.
- `EngineDescriptor`: exact engine/family/provider/version identity, optional model identity, prompt-driven boundary, optional prompt-template fingerprint, bounded metadata, full SHA-256 fingerprint, and 128-bit public handle.
- `ObservationGranularity`: `criterion_level` or `holistic`.
- `EvidenceRole`: `supporting_evidence`, `counter_evidence`, or `context_evidence`.
- `EvidenceReference`: source/span identifiers and exact content fingerprint without source text.
- `ScoringRequest`: exact assessment/rubric/construct/task/response identities, response content fingerprint and bounded statistics, allowed rubric scores, requested criterion set, metadata, and factory seal.
- `ObservationStatus`: `scored`, `abstained`, `failed`, or `excluded`.
- `ScoreObservation`: one governed response for a criterion or holistic request, with exact request/engine identities, optional score, terminal reason, evidence, confidence metadata, and factory seal.
- `ScoringResult`: one request, one engine, complete observation coverage, execution identity, bounded diagnostics, content fingerprint, and factory seal.
- `ScoringEngine`: runtime-checkable protocol accepting a `ScoringRequest` and returning `ScoringResult`.
- `StaticFixtureEngine`: deterministic offline fixture for contract tests and examples only.

## Invariants

1. Requests bind to an exact `AssessmentSpec` fingerprint and an exact `RubricSpecification` fingerprint declared by that assessment.
2. The rubric construct is declared by the assessment and the selected task family belongs to the rubric.
3. Criterion-level requests contain one or more unique descriptive criterion identifiers. Holistic requests contain none.
4. The request granularity must be permitted by the assessment response type; `mixed` assessments permit either explicit request granularity but never silently combine them.
5. Raw response content is represented only by a full SHA-256 fingerprint and bounded signed-64 content statistics.
6. Human engines cannot claim model or prompt-template identities. Automated engines require an exact model identity; prompt-driven engines require an exact prompt-template fingerprint.
7. Scored observations contain exactly one rubric score and no terminal reason. Non-scored observations contain a stable reason and no score.
8. Result observations bind to the same request and engine, are canonically ordered, and cover the request exactly once.
9. Evidence references are unique and canonically ordered; source text is never stored.
10. Equivalent caller ordering yields byte-identical content identities.
11. Public factories fail closed with structured non-reflective `AssessmentSpecError` values.
12. No psychometric arithmetic or provider coupling is introduced in Python.

## Resource and privacy boundaries

All identifiers use descriptive two-or-more-token lower `snake_case`. Fingerprints are complete lower-hexadecimal SHA-256 values. Metadata reuses the merged assessment contract's UTF-8, signed-64, `-0.0`, cycle, depth, node, collection, text, and byte budgets. Reserved raw-content fields remain prohibited. Public errors use caller-independent index paths and never include rejected values.

## Testing

Tests require 100% statement and branch coverage for the added production module and cover:

- exact engine identity and human/automated consistency;
- assessment/rubric/construct/task/granularity graph validation;
- score-category and status/reason invariants;
- evidence uniqueness and ordering;
- complete criterion/holistic result coverage;
- request/engine mismatch rejection;
- factory seals and deterministic identities;
- callback, UTF-8, metadata, signed-integer, and canonicalization safety;
- runtime protocol conformance and deterministic fixture execution;
- explicit public exports and complete docstrings.

## Release governance

The authoritative changelog fragment is rendered into `CHANGELOG.md` on the same head. A package version bump is deferred until the observation and engine contracts are connected to a first end-to-end scoring/calibration vertical; structural contracts alone are not presented as a complete scoring release.
