# Documentation map

This directory contains the durable documentation for `fast-mlsirm`. The repository has accumulated substantial research notes, implementation plans, and feature-specific doctoring; this index distinguishes **normative authority** from historical or local implementation evidence.

## Normative documents

| Document | Authority |
|---|---|
| [`../ARCHITECTURE.md`](../ARCHITECTURE.md) | System boundaries, dependency directions, owned bounded contexts |
| [`product_requirements.md`](product_requirements.md) | Product requirements, users, jobs-to-be-done, claims/non-claims, acceptance gates |
| [`technical_requirements.md`](technical_requirements.md) | Technical realization, API/runtime/resource/error/security/release contracts |
| [`adr/README.md`](adr/README.md) | Architecture-decision index and status history |
| [`traceability_matrix.md`](traceability_matrix.md) | Requirement -> architecture -> code -> test/evidence traceability |
| [`architecture/diagrams.md`](architecture/diagrams.md) | Executable system/component/sequence/state/logical-ERD views |
| [`../AGENTS.md`](../AGENTS.md) | Paper-first contributor and autonomous-agent governance |
| [`../CLAUDE.md`](../CLAUDE.md) | Operational developer guide aligned to `AGENTS.md` |
| [`../CHANGELOG.md`](../CHANGELOG.md) | Published and unreleased user-visible changes |

`docs/prd_trd_summary.md` is retained as a compatibility pointer to the canonical PRD and TRD above; it is no longer an independent source of product truth.

## Scientific and standards evidence

`docs/doctoring/` records the primary sources, implementation boundaries, failure modes, verification evidence, and conservative interpretation limits for substantive scientific, security, interoperability, privacy, and accessibility decisions. Doctoring can justify a decision, but a material architecture decision must also appear in an ADR.

`docs/papers/` contains literature maps, method specifications, and redistributed paper material where licensing permits. Primary methodological sources are preferred over legacy software as validation authority.

## Design and implementation history

`docs/superpowers/specs/` and `docs/superpowers/plans/` capture bounded design and implementation history. They are valuable engineering evidence but are **not** automatically normative after their work is merged. When a plan changes architecture, the durable decision belongs in `docs/adr/`, and product/technical consequences belong in the PRD/TRD.

Feature-specific documents such as rubric generation, automated scoring, enterprise issue calibration, model comparison, recovery, factor rotation, and report rendering remain authoritative for their local API or scientific boundary only when they do not conflict with the documents in the normative table.

## Downstream product boundary

`fast-mlsirm` documents reusable measurement capabilities only. Hosted product lifecycle, participant/session/consent persistence, HTTP/admin APIs, resource authorization, hosted UI, and deployment topology belong to `ContextualWisdomLab/psychometrics-commons`. References to those concerns here describe integration contracts or ownership boundaries; they do not create reverse dependencies or a hosted-product database in this repository.

## Change discipline

A material product or architecture change is complete only when the relevant items are synchronized:

1. PRD requirement or explicit non-goal;
2. TRD mechanism and failure/recovery behavior;
3. accepted or superseding ADR when a decision changes;
4. code/API/schema contract;
5. realistic test/CI/scientific evidence;
6. security/privacy/accessibility impact where applicable;
7. diagram/logical model updates;
8. traceability matrix row; and
9. changelog/release evidence when user-visible behavior changes.

Contradiction or staleness is a defect, not harmless documentation drift.
