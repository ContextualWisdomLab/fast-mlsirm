# Architecture Decision Records

This directory records durable architecture decisions for `fast-mlsirm`. An ADR is required when a change alters bounded-context ownership, numerical ownership, provenance/identity rules, model-selection semantics, persistence boundaries, LLM authority, multilevel/temporal modeling assumptions, or release/governance architecture.

## Status vocabulary

- `Proposed` — under review; not yet the governing architecture.
- `Accepted` — current architecture contract.
- `Superseded` — retained for history; a newer ADR governs.
- `Deprecated` — still present for compatibility but should not be used for new work.

## Current decisions

| ADR | Decision | Status |
|---|---|---|
| [ADR-001](ADR-001-domain-boundary.md) | Domain-neutral measurement core; hosted product is downstream | Accepted |
| [ADR-002](ADR-002-rust-first-numerics.md) | Rust-first production numerics with governed Python/PyO3 boundary | Accepted |
| [ADR-003](ADR-003-content-addressed-contracts.md) | Immutable content-addressed Assessment/Rubric/Scoring provenance | Accepted |
| [ADR-004](ADR-004-governed-item-bank.md) | Rubric→item→screening→calibration→bank lifecycle | Accepted |
| [ADR-005](ADR-005-relation-safe-model-selection.md) | Relation-safe factor/model selection and recovery evidence | Accepted |
| [ADR-006](ADR-006-multilevel-temporal-first-class.md) | Multilevel, multiple-membership, and temporal structure are first-class | Accepted |
| [ADR-007](ADR-007-fallible-raters-and-llm-orchestration.md) | Humans/LLMs are fallible raters; provider/model execution remains optional | Accepted |
| [ADR-008](ADR-008-logical-persistence-boundary.md) | Core owns logical contracts, not a physical hosted database | Accepted |

## ADR template

```markdown
# ADR-NNN: Decision title

- Status: Proposed | Accepted | Superseded | Deprecated
- Date: YYYY-MM-DD
- Deciders: repository maintainers / architecture owners

## Context
What problem and constraints require a durable architecture decision?

## Decision
What is the chosen architecture?

## Consequences
What becomes easier/harder? What must downstream work respect?

## Alternatives considered
Which materially different options were rejected and why?

## Evidence and references
Primary standards, research, tests, or operational evidence.
```

## Maintenance rule

An implementation PR that contradicts an accepted ADR must either:

1. update/supersede the ADR in the same reviewed change; or
2. be rejected as architecture drift.

A feature-specific plan is not a substitute for an ADR when it changes a repository-wide invariant.
