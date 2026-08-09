# fast-mlsirm Documentation Map

Use this index to find the authoritative product, technical, architecture, scientific, and release contracts instead of inferring repository behavior from whichever implementation note or PR handoff is easiest to find.

## Authoritative product and architecture set

| Document | Purpose |
|---|---|
| [`../ARCHITECTURE.md`](../ARCHITECTURE.md) | Repository bounded context, dependency direction, scientific architecture, security/privacy, and release gates. |
| [`PRD.md`](PRD.md) | Current component product requirements, users, buyer jobs, non-goals, quality objectives, and release acceptance. |
| [`TRD.md`](TRD.md) | Technical requirements for contracts, Rust/PyO3/Python ownership, resources, model selection, scoring, privacy, tests, and releases. |
| [`adr/README.md`](adr/README.md) | ADR index, status, supersession rules, and durable cross-cutting decisions. |
| [`architecture/README.md`](architecture/README.md) | Architecture-diagram and logical-data-model index. |
| [`architecture/uml.md`](architecture/uml.md) | Component, sequence, activity, state, release, and ecosystem diagrams. |
| [`architecture/logical-data-model.md`](architecture/logical-data-model.md) | Persistence-neutral logical ERD for canonical measurement artifacts. It is not a hosted-product ORM schema. |
| [`requirements-traceability.md`](requirements-traceability.md) | Conversation/research requirement → PRD/TRD/ADR → protected-main/open-PR/planned/downstream evidence mapping. |
| [`documentation_coverage_matrix.md`](documentation_coverage_matrix.md) | Sufficiency audit, residual documentation gaps, and update triggers. |
| [`prd_trd_summary.md`](prd_trd_summary.md) | Historical compatibility path; it must remain a pointer, not a second PRD/TRD authority. |

## Specialized scientific and implementation evidence

The authoritative baseline intentionally does not duplicate every method equation. Detailed scientific and implementation evidence remains in the relevant specialized records, including:

- `docs/doctoring/` for method, equation, standard, security, and operational traceability;
- model-comparison, bifactor, rotation, MLSIRM/MMLE, recovery, scoring, rubric-generation, automated-essay, and enterprise-issue documents;
- `docs/superpowers/specs/` and `docs/superpowers/plans/` as reviewed design/implementation history;
- release/readiness/buyer-evidence documents for exact-artifact acceptance.

If a specialized document contradicts the authoritative bounded context, numerical ownership, shipped/planned status, or a later Accepted ADR, repair or supersede the conflicting document rather than creating a second architecture.

## Documentation lifecycle

1. **Product requirement or non-goal changes:** update `PRD.md` and traceability.
2. **Technical/API/schema/resource/error/security/release behavior changes:** update `TRD.md` and any affected architecture/data-model diagrams.
3. **Durable cross-cutting decision:** add or supersede an ADR; never edit history to make an old decision appear as though it was always the current one.
4. **Scientific method/formula change:** update method doctoring, primary references, recovery/validation contracts, and the relevant architecture/traceability status.
5. **Logical artifact relationship or lifecycle change:** update the logical ERD and UML/state diagrams.
6. **Release-visible change:** render/update the authoritative changelog material and bind release evidence to the exact artifact.
7. **Repository/ecosystem ownership change:** align `ARCHITECTURE.md`, `AGENTS.md`, `CLAUDE.md`, PRD/TRD, ADRs, diagrams, and downstream integration guidance in the same change.

## Implementation-state rule

Every architecture-facing document must distinguish at least:

- **implemented_on_main** — accepted on the current protected main;
- **active_pr** — represented by open unmerged work;
- **planned** — accepted direction without integrated implementation;
- **research_only** — investigated, not a product commitment;
- **downstream** — owned by another bounded context;
- **out_of_scope** — explicitly not owned here.

Documentation is not implementation evidence. No feature becomes shipped merely because a PRD, ADR, UML, or ERD describes its target architecture.

## Ownership warning

`fast-mlsirm` owns independently installable reusable measurement contracts and psychometric computation. Hosted participant/session/consent/result lifecycle, tenant/resource authorization, product persistence/migrations, HTTP/admin APIs, reference clients, hosted UI, billing, and deployment composition belong to `ContextualWisdomLab/psychometrics-commons` or another explicitly owning downstream service. Do not recreate that hosted runtime here.
