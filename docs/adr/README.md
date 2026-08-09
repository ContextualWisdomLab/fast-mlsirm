# Architecture Decision Records

This directory is the authoritative decision log for `fast-mlsirm`. ADRs record durable architectural/scientific decisions that cannot be reconstructed reliably from chat history, pull-request bodies, handoff notes, or implementation details alone.

## Status values

- `accepted` — governing decision for protected main.
- `proposed` — reviewable decision not yet governing protected main.
- `superseded` — replaced by a newer ADR; retained for history.
- `deprecated` — no longer recommended but retained for compatibility/history.

## ADR index

| ADR | Status | Decision |
|---|---|---|
| [0001](0001-reusable-measurement-core-boundary.md) | accepted | Keep fast-mlsirm as the reusable measurement/scientific core; hosted product concerns stay downstream. |
| [0002](0002-rust-first-numerical-authority.md) | accepted | Rust is the production numerical authority; Python orchestrates/validates/reports and retained NumPy paths are bounded reference/fallback paths. |
| [0003](0003-governed-assessment-rubric-scoring-lifecycle.md) | accepted | Use canonical Assessment/Rubric/Scoring contracts and a governed rubric/item/calibration lifecycle; human and AI judges are fallible raters. |
| [0004](0004-structural-model-selection-and-context.md) | accepted | Choose structural models relation-safely and preserve testlet, multilevel, multiple-membership, and temporal structure before residual latent-space complexity. |

## Required ADR content

Every material ADR should state:

1. context/problem;
2. decision and scope;
3. alternatives considered;
4. invariants and forbidden dependencies;
5. failure/degraded behavior;
6. security/privacy implications;
7. scientific/validation evidence;
8. compatibility/migration/rollback expectations;
9. consequences and follow-up criteria;
10. primary references where the decision is methodological.

## Change policy

A change that contradicts an accepted ADR must either:

- supersede the ADR in the same or prerequisite PR; or
- demonstrate that the change is outside the ADR's stated scope.

Material changes to public contracts, numerical authority, model parameterization, hosted-product boundaries, lifecycle states, or evidence/merge/release authority require an ADR impact review.
