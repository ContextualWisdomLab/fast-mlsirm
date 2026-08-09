# Architecture-governance requirements traceability

## Status

This proposed traceability addendum belongs to the canonical architecture-documentation PR. The canonical top-level matrix remains `docs/requirements_traceability.md`; this file supplies detailed rows for continuous execution, documentation governance, verification/validation, sensitive-data governance, orchestration, and standards watch. On protected merge these rows are **IMPLEMENTED as documentation controls**, not evidence that planned numerical or hosted capabilities exist.

| Requirement ID | Requirement | Maturity | Governing decision/source | Implementation or evidence |
|---|---|---|---|---|
| GOV-EXEC-001 | An invocation continues after each intermediate event while another safe executable action exists. | PROPOSED | ADR 0013 | `fast-mlsirm Commercial Loop`; repository-state evidence per run |
| GOV-LEASE-001 | The loop writes only fast-mlsirm and scopes writer conflicts to the affected branch. | PROPOSED | ADR 0013 | scheduler contract; exact-head/base/blob pre-write checks |
| DOC-CANON-001 | One active canonical branch owns cross-cutting PRD/TRD/Architecture/ADR/UML/ERD/threat/traceability changes. | PROPOSED | ADR 0013 | canonical documentation PR; duplicate closure evidence |
| DOC-MAT-001 | Documentation distinguishes IMPLEMENTED/ACCEPTED, ACTIVE PR, PROPOSED, PLANNED, DOWNSTREAM, and REJECTED/SUPERSEDED. | PROPOSED | ADR 0013 | `tests/test_architecture_documentation_contract.py` |
| DOC-COMP-001 | Substantive contract changes update every applicable requirements, decision, view, traceability, doctoring, validation, migration, and changelog artifact. | PROPOSED | ADR 0013 | documentation completeness review and tests |
| VV-VER-001 | Deterministic verification is distinct from scientific and consequential validation. | PROPOSED | Verification and validation architecture; Testing Standards | `docs/architecture/verification_and_validation.md`; PlantUML evidence flow |
| VV-REC-001 | Parameter recovery reports aligned bias, MAE, RMSE, uncertainty/coverage and convergence; correlation is supplementary. | PROPOSED/partially implemented by method | V&V architecture; method doctoring | method recovery suites; release evidence by changed method |
| VV-SEL-001 | Model selection classifies relations and requires appropriate LR/bootstrap/Vuong/predictive/recovery evidence. | PROPOSED/partial | V&V architecture; model-selection ADRs | relation-safe comparison APIs and planned factor-retention workflow |
| VV-SCORE-001 | Model fit does not authorize score interpretation without scoreability evidence. | PROPOSED/partial | V&V architecture; bifactor research | bifactor scoreability and rater/scale diagnostics |
| DATA-MIN-001 | Core governed artifacts avoid ambient raw response, prompt, source, provider-output and identity content. | PROPOSED/partial | ADR 0014 | contract schemas, failure redaction, hostile-input/privacy tests |
| DATA-LINK-001 | Opaque identifiers and fingerprints preserve lawful psychometric, longitudinal, multiple-membership and audit linkage. | PROPOSED/partial | ADR 0014 | assessment, scoring and multilevel contract fingerprints |
| DATA-AUTH-001 | Fingerprints are never treated as authentication, authorization, consent, signature or anonymity. | PROPOSED | ADR 0014 | documentation contract test and downstream security review |
| DATA-HOST-001 | Tenant identity, consent, legal basis, durable persistence, residency, keys, retention and erasure are downstream responsibilities. | ACCEPTED boundary | AGENTS.md; ADR 0014 | Psychometrics Commons boundary; no reverse dependency |
| LLM-TTC-001 | Workflow stages, model/provider, roles, decomposition, recursion, tools/access, effort, verification and total compute are versioned experimental variables. | PROPOSED | ADR 0015 | provider-neutral contracts and future orchestration policy |
| LLM-ABL-001 | Deeper orchestration must outperform a single/shallow comparable-budget baseline for the target use. | PROPOSED | ADR 0015; primary test-time-compute research watch | comparable-budget ablation evidence |
| LLM-AUTH-001 | Model workers and judgments receive no merge, release, signing, protection or high-stakes decision authority. | PROPOSED/accepted governance | ADR 0015; repository review policy | workflow permissions and independent approval evidence |
| LLM-KEY-001 | Model-backed tests/development use `NVIDIA_NIM_API_KEY`; autonomous development does not use `COPILOT_GITHUB_TOKEN`. | PROPOSED/partially implemented | ADR 0015; AGENTS.md | workflow contract tests and secret-boundary evidence |
| STD-WATCH-001 | Published applicable editions are normative only after official-source verification; drafts remain watch items. | PROPOSED | `docs/standards_watch.md` | `tests/test_standards_watch_contract.py`; release review |
| STD-CLAIM-001 | Citation never implies implementation, certification, conformance, safety, fairness, validity or legal compliance. | PROPOSED | standards watch; commercial-readiness boundary | documentation/release review and buyer evidence |

## Requirement-to-evidence rule

A row is promoted to **IMPLEMENTED / ACCEPTED** only when the governing document is on protected main and its claimed behavior has exact protected-main implementation and evidence. A documentation-control row may be implemented by the documentation and test contract itself; a numerical, scoring, model-selection, privacy, hosted-product, or orchestration row additionally requires the relevant code and scientific/operational validation.

## Downstream boundary

The following remain **DOWNSTREAM** unless a reusable contract is explicitly accepted in fast-mlsirm:

- tenant and resource authorization;
- participant/session/consent/result lifecycle;
- identity mapping and customer-managed keys;
- product persistence and migrations;
- HTTP/admin APIs, deployment, billing and hosted operations;
- domain-specific consequential decisions and impact assessment.

The canonical hosted owner is Psychometrics Commons or another named product bounded context. fast-mlsirm remains independently installable and domain-neutral.
