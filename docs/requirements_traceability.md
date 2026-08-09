# Requirements Traceability Matrix — fast-mlsirm

Status: canonical traceability index  
Date: 2026-08-09

This document links product requirements to technical requirements, architecture decisions, implementation/evidence surfaces, and current maturity. It is intentionally filename/contract-oriented rather than PR-number-oriented so the trace remains valid after branches merge or close.

## Maturity vocabulary

- **Implemented** — present on protected `main` and governed by tests/docs.
- **Partial** — meaningful contract/implementation exists but the end-to-end requirement is not complete.
- **Planned** — accepted architecture requirement without a protected-main implementation.
- **Downstream** — intentionally owned by another product/service.

## Traceability matrix

| PRD requirement | TRD / ADR | Main implementation or evidence | Maturity |
|---|---|---|---|
| PRD-CONTRACT-001 Assessment contracts | TRD-CONTRACT-001..004; ADR-003 | `python/fast_mlsirm/scoring/`; `docs/scoring_assessment_contracts.md`; `docs/scoring_execution_contracts.md` | Implemented / evolving |
| PRD-RUBRIC-001 Rubric-centered authoring | TRD-RUBRIC-001..003; ADR-004 | `python/fast_mlsirm/rubric/`; `docs/rubric_item_generation.md` | Implemented foundation |
| PRD-RUBRIC-002 Generated-candidate validation | TRD-RUBRIC-004; ADR-003/004 | provider-output/candidate validation design and tests in active development | Partial |
| PRD-BANK-001 Governed item bank | ADR-004/008 | lifecycle defined in architecture; calibration/linking primitives exist | Planned / partial primitives |
| PRD-SCORING-001 Unified scoring | TRD-SCORE-001/002; ADR-003/007 | `fast_mlsirm.scoring` request/engine/observation/result contracts | Implemented foundation |
| PRD-SCORING-002 Essay calibration/validation | TRD-SCORE-*; ADR-007 | essay validation/reporting modules; many-facet primitives; range-use work in progress | Partial |
| PRD-RAG-001 Reference-free RAG measurement | ADR-004/005/007 | generic rubric/scoring/facet/model-selection primitives; domain adapter not canonical | Planned integration |
| PRD-PSY-001 Core psychometrics | TRD-NUM-*; ADR-002 | `crates/mlsirm-core`; PyO3; fit/diagnostics/linking/CAT/ATA/facets modules | Implemented broad surface |
| PRD-MODEL-001 Factor retention/structure | TRD-PSY-001..004; ADR-005 | dimensionality diagnostics; bifactor/testlet/rotation/model-comparison work | Partial |
| PRD-MODEL-002 Relation-safe comparison | TRD-PSY-002; ADR-005 | model-relation/result contracts and Rust Vuong kernel foundation | Partial; formal distinguishability expansion remains |
| PRD-ROT-001 Adaptive rotation | ADR-002/005 | adaptive rotation design/implementation work; Rust-first requirement | Partial / active |
| PRD-MULTI-001 Multilevel/multiple membership | TRD-PSY-005; ADR-006 | multilevel fit summaries on main; contextual/membership contract work active | Partial |
| PRD-TIME-001 Temporal/longitudinal | TRD-PSY-006; ADR-006 | task/occasion revision provenance on main; richer longitudinal contracts/estimators active/planned | Partial |
| PRD-RECOVERY-001 Recovery | TRD-NUM-005; ADR-002/005/006 | simulation/recovery utilities; Rust literature-recovery studies; GPU parity evidence | Implemented foundation, expanding |
| PRD-REPORT-001 Auditable reports | TRD-QUAL-*; ADR-003 | standalone HTML/JSON reports; accessibility/exact-value regression tests | Implemented, continuously hardened |
| PRD-RELEASE-001 Release evidence | TRD-REL-002/003 | `scripts/release_acceptance.py`, commercial-release/buyer/procurement evidence builders | Implemented evidence framework |

## Architectural concerns trace

### Standalone vs hosted product

- Governing ADR: ADR-001, ADR-008.
- Verification: package import/build tests must not require Psychometrics Commons, Keyverse, a product DB, or model credentials.
- Downstream hosted lifecycle: `ContextualWisdomLab/psychometrics-commons`.

### Rust-first numerical ownership

- Governing ADR: ADR-002.
- Verification: PyO3 delegation tests, Rust/reference parity, recovery, explicit GPU no-skip/parity where applicable.
- Failure mode to prevent: Python convenience/refactor silently becoming the production numerical implementation.

### Reference-free and automated scoring

- Governing ADR: ADR-003, ADR-004, ADR-007.
- Verification: exact rubric/task/engine fingerprints; distinct scoring statuses; rater/facet diagnostics; evidence-grounded rubrics; no ground-truth claim from LLM output alone.

### Model selection and score interpretation

- Governing ADR: ADR-005.
- Verification: model-relation classification, appropriate formal test, predictive evidence, recovery, DIF/invariance, bifactor scoreability.
- Failure mode to prevent: selecting a complex model solely by fit/AIC/BIC or returning a winner when distinguishability is not established.

### Multilevel and temporal structure

- Governing ADR: ADR-006.
- Verification: context-dimension identity, membership-weight contract, connectedness/identification, time ordering, model-specific recovery.
- Failure mode to prevent: atomistic inference from clustered/cross-classified/repeated observations.

## Documentation-to-code change rule

A PR that changes any of the following must update the traceability row and governing docs in the same change or a clearly linked follow-up before release:

- public contract schema/version;
- bounded-context ownership;
- production numerical owner/backend semantics;
- model relation/selection semantics;
- multilevel/temporal parameterization;
- item-bank lifecycle state/approval semantics;
- security/privacy provenance boundary;
- release gate or interpretation claim.

## Release trace rule

The release evidence must bind to the exact integrated protected head. This traceability document describes intended requirements; it is not itself proof that a particular release satisfies them.
