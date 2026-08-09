# Architecture Decision Records

Architecture Decision Records (ADRs) preserve material technical, scientific and
product-boundary decisions for `fast-mlsirm`. Accepted ADRs are not silently
rewritten when the decision changes: create a superseding ADR and update this
index.

## Status vocabulary

- **Proposed** — under active review; not yet normative.
- **Accepted** — normative for new work.
- **Deprecated** — retained for history; avoid new dependencies.
- **Superseded** — replaced by another ADR.
- **Rejected** — evaluated and intentionally not adopted.

## Required contents

Every material ADR records context, decision, invariants, alternatives,
consequences, failure/degraded behavior, security/privacy implications,
compatibility/migration/rollback, verification evidence, sources, and explicit
supersession conditions.

## Index

| ADR | Status | Decision |
|---|---|---|
| [0001](0001-reusable-core-hosted-product-boundary.md) | Accepted | Keep fast-mlsirm reusable; hosted product remains downstream |
| [0002](0002-rust-numerical-source-of-truth.md) | Accepted | Rust owns production psychometric arithmetic |
| [0003](0003-canonical-contracts-and-provenance.md) | Accepted | Reuse canonical Assessment/Rubric/Scoring contracts and immutable provenance |
| [0004](0004-relation-safe-model-selection.md) | Accepted | Model selection fails closed until relation/distinguishability evidence is valid |
| [0005](0005-rater-aware-ai-evaluation.md) | Accepted | Human and LLM judges are fallible raters, not truth providers |
| [0006](0006-multilevel-membership-and-time.md) | Accepted | Preserve hierarchy, multiple membership and time instead of atomistic flattening |
| [0007](0007-adaptive-rotation-selection.md) | Accepted | No universal rotation optimum; use multi-start and criterion-neutral evidence |
| [0008](0008-statistical-evidence-and-release-gates.md) | Accepted | Separate bounded PR smoke from heavy studies without weakening scientific release proof |
| [0009](0009-governed-rubric-item-bank-lifecycle.md) | Accepted | Rubric/item generation is an immutable governed measurement lifecycle |
| [0010](0010-canonical-pyo3-export-registry.md) | Accepted | Numerical feature bindings converge on one maintainable public registry |

## Related authorities

- `ARCHITECTURE.md` — bounded context and dependency architecture.
- `docs/PRD.md` — user/value/product requirements.
- `docs/TRD.md` — technical realization and acceptance requirements.
- `docs/requirements-traceability.md` — requirement-to-code/test/evidence coverage.
- `docs/architecture/` — UML/C4-style diagrams and logical ERD.
- `AGENTS.md` — contributor/agent operating rules and primary method references.
- `docs/doctoring/` — method- and implementation-specific scientific evidence.
