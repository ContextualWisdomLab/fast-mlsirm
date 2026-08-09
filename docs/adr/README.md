# Architecture Decision Records

Architecture Decision Records (ADRs) preserve material technical, scientific, product-boundary, security/privacy and release-governance decisions for `fast-mlsirm`.

An accepted ADR is historical evidence as well as current guidance. Do not silently rewrite the decision when the architecture changes: add a superseding ADR, update this index, and keep the superseded record available.

## Status vocabulary

- **Proposed** — under review and not yet normative.
- **Accepted** — normative design for new work; implementation status may still be partial and must be stated separately.
- **Deprecated** — retained for compatibility/history but should not gain new dependencies.
- **Superseded** — replaced by another ADR.
- **Rejected** — evaluated and intentionally not adopted.

## Required contents

Use [`0000-template.md`](0000-template.md). Every material ADR must state:

1. concrete context, drivers and assumptions;
2. decision owner repository/service/bounded context;
3. implementation status separately from decision status;
4. allowed dependency direction and forbidden coupling;
5. public API/schema/identifier/idempotency/ordering/error semantics when affected;
6. numerical/scientific assumptions and source-of-truth ownership when affected;
7. invariants tied to tests or operational controls;
8. fail-closed/degraded/recovery/poison-input behavior;
9. authentication, authorization, tenancy, privacy, residency, encryption, retention and audit consequences where relevant;
10. migration, compatibility window, rollback triggers and rollback mechanics;
11. measurable scientific, security, accessibility and operational acceptance evidence;
12. alternatives, accepted risks, follow-up work and objective supersession/reversal conditions;
13. authoritative standards/primary papers in APA 7 form when the decision relies on external evidence.

A short aspirational note without these decision mechanics is not an ADR.

## Index

| ADR | Status | Implementation | Decision |
|---|---|---|---|
| [ADR-0001](ADR-0001-product-boundaries-and-scientific-governance.md) | Accepted baseline | mixed/current + future gates | Umbrella repository boundaries and scientific governance |
| [ADR-0002](ADR-0002-rust-numerical-source-of-truth.md) | Accepted | active | Rust owns production psychometric arithmetic |
| [ADR-0003](ADR-0003-canonical-contracts-and-provenance.md) | Accepted | active/expanding | Canonical reusable contracts and immutable provenance |
| [ADR-0004](ADR-0004-relation-safe-model-selection.md) | Accepted | partial | Fail closed until model relation/distinguishability evidence is valid |
| [ADR-0005](ADR-0005-rater-aware-ai-evaluation.md) | Accepted | active principle / feature-specific | Human and LLM judges are fallible raters, not truth providers |
| [ADR-0006](ADR-0006-multilevel-membership-and-time.md) | Accepted | contract/design; estimator-specific | Preserve hierarchy, multiple membership and time |
| [ADR-0007](ADR-0007-adaptive-rotation-selection.md) | Accepted | active design | No universal rotation optimum; use multi-start and criterion-neutral evidence |
| [ADR-0008](ADR-0008-statistical-evidence-and-release-gates.md) | Accepted | active | Tier CI without weakening release-bound scientific evidence |
| [ADR-0009](ADR-0009-governed-rubric-item-bank-lifecycle.md) | Accepted | partial / lifecycle feature-gated | Rubric/item generation is a governed measurement lifecycle |

## Deliberately deferred ADRs

Do not pre-accept architecture merely to make the ADR count larger. Create a new ADR when evidence and implementation are stable enough to make the decision concrete. Current triggers include:

- final composed PyO3 registration/public-export architecture after the competing integration approaches settle;
- complete Vuong/boundary-aware model-comparison public contract when formal distinguishability APIs stabilize;
- canonical reference-free RAG observation schema when accepted into the public package;
- estimator-specific multilevel/longitudinal objective/identification/GPU architecture when the first production estimator is accepted;
- cross-repository CSAP/SOC 2 control ownership when hosted/control-plane evidence locations are stable.

## Related authorities

- `ARCHITECTURE.md` — repository/bounded-context and dependency architecture.
- `docs/PRD.md` — user, value and product requirements.
- `docs/TRD.md` — technical realization and acceptance requirements.
- `docs/UML.md` — logical component/sequence/deployment views.
- `docs/ERD.md` — persistence-neutral logical entity relationships; **not** a `fast-mlsirm` physical database schema.
- `docs/documentation_coverage_matrix.md` — documentation sufficiency and residual-gap register.
- `AGENTS.md` / `CLAUDE.md` — contributor/agent operating constraints.
- `docs/doctoring/` and method-specific design docs — scientific equation-to-source and implementation evidence.
