# fast-mlsirm Documentation Index

Use this page to find the **canonical** product, architecture, technical, decision, scientific, and release documentation. Feature-specific design notes remain useful, but they must not contradict the canonical documents below.

## Canonical product and architecture spine

- [`../ARCHITECTURE.md`](../ARCHITECTURE.md) — system of interest, bounded contexts, module/data/deployment architecture, numerical ownership, architecture fitness functions.
- [`PRD.md`](PRD.md) — product requirements, users/jobs, functional requirements, non-goals, product horizons.
- [`TRD.md`](TRD.md) — technical, numerical, psychometric, security, quality, packaging, and verification requirements.
- [`architecture/UML.md`](architecture/UML.md) — component, contract-class, sequence, activity, state-machine, multilevel/temporal, and deployment views.
- [`architecture/ERD.md`](architecture/ERD.md) — logical contract/provenance ERD and downstream persistence ownership boundary.
- [`adr/README.md`](adr/README.md) — Architecture Decision Record index and governance.
- [`requirements_traceability.md`](requirements_traceability.md) — PRD → TRD/ADR → implementation/evidence maturity map.
- [`documentation_coverage.md`](documentation_coverage.md) — completeness audit and remaining documentation debt.

## Core contract documentation

- [`rubric_item_generation.md`](rubric_item_generation.md) — RubricSpecification → Blueprint → GenerationContract.
- [`scoring_assessment_contracts.md`](scoring_assessment_contracts.md) — AssessmentSpec and policy graph.
- [`scoring_execution_contracts.md`](scoring_execution_contracts.md) — provider-neutral scoring request/engine/observation/result contracts.

## Scientific and method documentation

Method-specific documentation and `docs/papers/` or doctoring records provide the equations, identification assumptions, source references, recovery designs, limitations, and validation evidence for individual algorithms. `AGENTS.md` at the repository root remains the contributor-facing paper-first formula and review policy.

Important rule: **method docs prove neither release readiness nor product architecture on their own**. Material changes to model parameterization, bounded-context ownership, contract identity, model-selection semantics, numerical ownership, multilevel/temporal assumptions, or security/privacy boundaries must update the relevant ADR/PRD/TRD/architecture view as well.

## Commercial and release evidence

- [`commercial_readiness.md`](commercial_readiness.md)
- [`enterprise_sales_readiness.md`](enterprise_sales_readiness.md)
- [`release_acceptance.md`](release_acceptance.md)
- other buyer, benchmark, procurement, release-index, and Figma evidence documents referenced from the README and release scripts.

These are evidence and go-to-market artifacts. They must bind to an exact release candidate and must not be used to claim that a planned architecture feature is implemented.

## Documentation ownership rules

1. **PRD** states what the product must do and what it deliberately does not own.
2. **TRD** states enforceable technical/scientific constraints.
3. **ARCHITECTURE/UML/ERD** state how responsibilities, components, information, and flows are structured.
4. **ADRs** explain durable choices and rejected alternatives.
5. **Feature/method docs** explain a particular implementation or research method.
6. **Traceability** identifies implementation/evidence maturity.
7. **Release evidence** proves the exact artifact that was actually tested.

If two documents disagree, fix or explicitly supersede the stale document rather than allowing multiple authoritative interpretations to persist.
