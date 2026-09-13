# fast-mlsirm documentation authority

This index distinguishes governing product/architecture documents from implementation history and method-local evidence.

## Canonical architecture package

| Document | Governing purpose |
|---|---|
| [`../ARCHITECTURE.md`](../ARCHITECTURE.md) | System of interest, bounded contexts, dependency direction, component/data/deployment/scientific views |
| [`PRD.md`](PRD.md) | Product requirements, users, workflows, non-goals, acceptance boundaries |
| [`TRD.md`](TRD.md) | Technical realization, numerical/runtime/security/resource/release requirements |
| [`adr/README.md`](adr/README.md) | Durable architecture/scientific decision log and status history |
| [`standards_watch.md`](standards_watch.md) | Published governing standards versus draft/revision watch items; no certification shortcut |
| [`verification_validation_plan.md`](verification_validation_plan.md) | Software, numerical, scientific, scoring/RAG, recovery, security, packaging and exact-artifact V&V evidence |
| [`uml/README.md`](uml/README.md) | PlantUML component, sequence, lifecycle, model-selection, deployment and reusable domain/public-contract views |
| [`uml/domain-public-contract.puml`](uml/domain-public-contract.puml) | Persistence-neutral reusable domain/public-contract classes and construction rules |
| [`erd/domain-model.puml`](erd/domain-model.puml) | Logical reusable-domain artifact relationships; **not** a hosted ORM schema |
| [`traceability/requirements-matrix.md`](traceability/requirements-matrix.md) | PRD/TRD/ADR -> protected-main implementation/evidence maturity |
| [`traceability/research-basis.md`](traceability/research-basis.md) | Scientific/standards evidence and APA 7 reference mapping |
| [`papers/implemented-literature-map.md`](papers/implemented-literature-map.md) | Paper-to-kernel map for implemented methods |
| [`delta_plot_dif.md`](delta_plot_dif.md) | Angoff TID / delta-plot observed-score DIF screen (not a security control) |
| [`bradley_terry_mm.md`](bradley_terry_mm.md) | Bradley–Terry / Hunter MM pairwise ranking and additive-ties BRATT |
| [`documentation_coverage.md`](documentation_coverage.md) | Documentation completeness states, remaining P0/P1/P2 gaps and maintenance gate |
| [`security/threat-model.md`](security/threat-model.md) | Reusable-core trust/threat model; hosted product threats remain downstream |
| [`doctoring/`](doctoring/) | Method/security/interoperability evidence and conservative implementation boundaries |
| [`../AGENTS.md`](../AGENTS.md), [`../CLAUDE.md`](../CLAUDE.md) | Agent/developer operating rules aligned to this architecture |
| [`../CHANGELOG.md`](../CHANGELOG.md) | User-visible released/unreleased change history |

`prd_trd_summary.md` is historical and must not compete with `PRD.md` and `TRD.md` as a requirements source.

## Authority and status

1. Protected-main source/tests define executable behavior.
2. Accepted ADRs define governing architecture/scientific decisions.
3. PRD/TRD define product/technical requirements and non-claims.
4. `ARCHITECTURE.md`, UML/ERD and the threat model define coherent system views.
5. The standards watch defines which published editions may govern claims and which drafts/revisions are only monitored.
6. The V&V plan defines what evidence is needed to verify software behavior and validate scientific/product interpretations.
7. Method-specific doctoring and primary literature justify local scientific/interoperability details.
8. Proposed ADRs, open PRs/issues and plans describe future/active work and are not released capability merely because they exist.

A conversation or PR body is discovery evidence until the durable decision is captured in the documents above.

## Implementation history

`docs/superpowers/specs/` and `docs/superpowers/plans/` preserve bounded design/implementation history. They do not automatically remain normative after implementation. If a plan creates a durable product/architecture/scientific decision, update the canonical PRD/TRD/ADR/traceability set.

## Completeness gate

A material change is incomplete if it creates a contradiction among code, accepted ADRs, PRD/TRD, architecture diagrams, security/threat model, standards status, V&V evidence, traceability, doctoring or release evidence. The documentation-contract test and `documentation_coverage.md` make these gaps visible; a missing or stale canonical artifact is release-maintenance debt rather than harmless prose drift.

## Cross-repository boundary

`fast-mlsirm` is the standalone reusable measurement/psychometric core. `ContextualWisdomLab/psychometrics-commons` or another owning downstream service is responsible for hosted HTTP/session/consent/tenant/RBAC/UI/database/deployment lifecycle. Architecture documents here may define interoperable reusable artifacts and versioned handoffs without creating a shared application database or reverse product dependency.
