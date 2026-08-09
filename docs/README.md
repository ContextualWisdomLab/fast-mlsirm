# fast-mlsirm Documentation Map

Use this file to choose the authoritative document instead of inferring architecture from whichever implementation note or PR handoff is easiest to find.

## Authoritative product and architecture set

| Document | Purpose |
|---|---|
| [`../ARCHITECTURE.md`](../ARCHITECTURE.md) | System architecture, bounded contexts, scientific model hierarchy, security and quality gates. |
| [`PRD.md`](PRD.md) | Current product requirements, users, buyer problems, scope, NFRs, release criteria. |
| [`TRD.md`](TRD.md) | Technical requirements, numerical authority, contracts, model selection, validation, security and runtime rules. |
| [`adr/README.md`](adr/README.md) | Architecture decision record index and governance. |
| [`UML.md`](UML.md) | Component, class, sequence, state, model-selection, and deployment diagrams. |
| [`ERD.md`](ERD.md) | Persistence-agnostic logical contract/data model. |
| [`traceability.md`](traceability.md) | Research/conversation requirement → protected-main/open-PR/planned status mapping. |
| [`prd_trd_summary.md`](prd_trd_summary.md) | Historical compatibility path; only redirects to the authoritative set. |

## Specialized design and implementation doctoring

The repository also contains method-specific and feature-specific documentation such as:

- scoring assessment/execution contracts;
- rubric item generation and generation validation;
- automated essay scoring, facets calibration, and validation reports;
- enterprise issue evidence/scoring/calibration handoffs;
- model-comparison and bifactor-scoreability documentation;
- adaptive factor-rotation documentation;
- MLSIRM/MMLE/recovery and paper maps;
- `docs/doctoring/` standards/method records;
- `docs/superpowers/specs/` and `docs/superpowers/plans/` implementation design history.

These files provide detail and historical evidence. When they conflict with the architecture boundary or product ownership, update the conflicting file; do not silently create a second architecture.

## Documentation lifecycle

1. **Material decision:** add/supersede an ADR.
2. **Public product behavior:** update PRD/traceability and specialized docs.
3. **Technical contract or dependency direction:** update TRD/ARCHITECTURE/UML/ERD as applicable.
4. **Scientific method:** update doctoring and primary references plus relevant recovery/validation tests.
5. **Release-visible change:** update the repository's authoritative changelog fragment/render workflow when release policy requires it.
6. **Integration:** ensure `AGENTS.md`/`CLAUDE.md` remain aligned with the authoritative boundary.

## Ownership warning

`fast-mlsirm` documents reusable measurement/scientific contracts. Hosted participant/session/consent/product-database/UI/deployment architecture belongs to `ContextualWisdomLab/psychometrics-commons` and should not be copied into this documentation as if this repository owned it.
