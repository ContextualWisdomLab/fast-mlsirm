# Architecture Decision Records

Architecture Decision Records (ADRs) are the durable history of material `fast-mlsirm` design decisions. Implementation plans and doctoring may explain how and why a change was made; an ADR records the decision, alternatives, invariants, ownership boundary, failure/recovery behavior, and conditions under which the decision may be superseded.

## Status values

- **Proposed** — accepted for investigation/implementation, not yet the released architecture.
- **Accepted** — governing decision for protected `main`.
- **Superseded** — replaced by a later ADR; retained as history.
- **Rejected** — considered and intentionally not adopted.
- **Deprecated** — still present for compatibility but no longer the preferred decision.

A code path cannot be declared `Accepted` merely because a Draft PR exists. If an ADR mixes an accepted current boundary and a proposed future extension, each invariant must state which status applies.

## Decision index

| ADR | Status | Decision |
|---|---|---|
| [0001](0001-reusable-core-product-boundary.md) | Accepted | `fast-mlsirm` is the domain-neutral measurement core; Psychometrics Commons owns hosted product lifecycle |
| [0002](0002-rust-numerical-authority.md) | Accepted | Rust is numerical authority; Python orchestrates/validates/reports through a canonical PyO3 boundary |
| [0003](0003-versioned-contracts-provenance.md) | Accepted | Assessment, rubric, scoring, and evidence artifacts are immutable, versioned, and content-addressed |
| [0004](0004-generated-item-trust-boundary.md) | Accepted | Generated provider output is hostile until strict replay/source/format validation succeeds |
| [0005](0005-fallible-raters-model-selection.md) | Accepted | Humans/LLMs are fallible raters; model selection and score interpretation fail closed on unresolved relation/validity evidence |
| [0006](0006-multilevel-temporal-measurement.md) | Proposed | Reusable measurement must preserve multilevel, multiple-membership, cross-classified, testlet, and temporal structure |
| [0007](0007-factor-rotation-selection.md) | Accepted | No universal rotation criterion/global optimum claim; registry + multi-start + stability/recovery/theory evidence |
| [0008](0008-scientific-ci-release-evidence.md) | Accepted | True-parameter recovery, bounded PR smoke, scheduled heavy studies, and exact-head release provenance govern scientific releases |
| [0009](0009-purpose-limited-sensitive-data.md) | Accepted | Preserve legitimate sensitive-data utility through minimization, separated identity/evidence domains, authorization and purpose limitation rather than blanket masking |
| [0010](0010-llm-orchestration-credential-boundary.md) | Accepted | LLM orchestration is optional; NVIDIA NIM model credentials remain separate from independent reviewer and merge authority |

## ADR creation rule

Create or supersede an ADR when a change affects one or more of:

- repository/bounded-context ownership or dependency direction;
- the numerical source of truth or language/runtime boundary;
- public artifact/schema identity or compatibility;
- security, privacy, authorization, credential, or provider trust boundary;
- model identification/selection/score interpretation policy;
- persistence ownership or database schema;
- release/scientific evidence policy;
- cross-repository API/event/artifact authority; or
- a user-visible workflow whose rollback requires architectural coordination.

Routine refactors, local performance optimizations that preserve the accepted formula/API, and ordinary bug fixes usually do not require an ADR.

## Required ADR content

Use [`0000-template.md`](0000-template.md). A material ADR should contain:

1. status/date/context and decision drivers;
2. bounded-context owner and dependency directions;
3. explicit invariants and non-goals;
4. alternatives considered;
5. failure/degraded/recovery behavior;
6. security/privacy implications;
7. compatibility/migration/rollback;
8. test/acceptance evidence;
9. scientific/standards references where material; and
10. supersession/reversal conditions.

## Consistency gate

Accepted ADRs, PRD/TRD, `ARCHITECTURE.md`, code/API contracts, diagrams, tests, and the traceability matrix must not contradict one another. A contradiction is release-blocking documentation debt. Accepted history is superseded with a new ADR; it is not silently rewritten to make current code look intentional.
