# fast-mlsirm Architecture Documentation Index

This directory is the review entry point for the repository's architecture and
logical data/artifact model.

## Authoritative documents

- [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md) — bounded context, system
  context, internal layers and release architecture.
- [`../PRD.md`](../PRD.md) — reusable component product requirements.
- [`../TRD.md`](../TRD.md) — technical realization and acceptance contracts.
- [`../adr/README.md`](../adr/README.md) — architecture/scientific decision index.
- [`uml.md`](uml.md) — component, sequence, activity and state diagrams.
- [`logical-data-model.md`](logical-data-model.md) — canonical artifact ERD and
  ownership rules.
- [`../requirements-traceability.md`](../requirements-traceability.md) —
  conversation/research-to-code/evidence coverage matrix.

## Interpretation rule

`AGENTS.md` and `CLAUDE.md` are contributor/agent guidance. Method-specific RFCs,
plans and `docs/doctoring/` records explain individual algorithms. They are not a
substitute for the repository-wide PRD/TRD/ADR/architecture authorities above.

When documents disagree:

1. protected-main code/tests are the current implemented behavior;
2. an Accepted ADR controls the intended decision until superseded;
3. `ARCHITECTURE.md` controls bounded-context/dependency ownership;
4. PRD controls product requirement/non-goal;
5. TRD controls the expected technical mechanism/acceptance boundary;
6. the traceability matrix records any resulting gap and must not relabel planned
   or active-PR work as shipped.
