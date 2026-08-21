# Requirements, decisions, implementation and evidence matrix

Status: **Authoritative traceability baseline**  
Last reviewed: 2026-08-09

This matrix makes the major product requirements discoverable without reconstructing decisions from chat history or PR bodies. It deliberately distinguishes **protected-main implementation**, **active/open work**, **future research**, and **downstream ownership**.

| Requirement family | PRD / TRD IDs | ADR | Protected-main implementation/evidence | State |
|---|---|---|---|---|
| Repository ownership | PRD-PRN-007, TRD-BOUND-001/002 | ADR-0001 | `AGENTS.md`, `CLAUDE.md`; package boundary in `python/fast_mlsirm/` | Accepted |
| Rust numerical ownership | PRD-PRN-002, TRD-NUM-001..006 | ADR-0002 | `crates/mlsirm-core/`, `crates/fast-mlsirm-py/`, backend/parity tests | Accepted |
| Canonical PyO3/public exports | TRD-API / numerical integration | ADR-0011 | current package exports exist; future Rust feature modules must converge on one registry instead of independent initializer/import rewrites | Proposed hardening |
| Assessment/scoring contracts | PRD-FR-001..004, TRD-API/P-ROV/SCR | ADR-0003, ADR-0005 | `python/fast_mlsirm/scoring/contracts.py` and bounded submodules | Accepted |
| Rubric/blueprint/generation | PRD-FR-010..014, TRD-RUB-001..006 | ADR-0003, ADR-0004 | `python/fast_mlsirm/rubric/`: models/compiler/contracts/generation/candidates/audit/pilot modules | Partial / evolving |
| Governed item bank lifecycle | PRD-FR-010, FR-080 | ADR-0004 | Pilot/admission/lifecycle primitives exist; complete approved-bank linking/exposure/monitoring/retirement workflow remains evolving; issue #609 tracks the planned canonical closed loop | Proposed/partial |
| Automated essay scoring | PRD-FR-020..023, TRD-SCR | ADR-0005 | governed essay score, calibration, validation and HTML report modules/tests from v0.7.0-era work | Accepted baseline / evolving diagnostics |
| Enterprise issue evaluation | PRD-FR-020..023 | ADR-0005 | `fast_mlsirm.scoring.enterprise_issue` adapters, governed observations/calibration/reporting | Accepted reusable adapter; causal action/utility policy is downstream |
| Reference-free RAG measurement | PRD-FR-030..033, TRD-RAG | ADR-0005, ADR-0006 | shared measurement primitives exist, but no canonical end-to-end RAG observation/calibration pipeline is accepted on protected main; issue #607 tracks the governed adapter/evidence-regime contract | Proposed |
| Fallible human/LLM raters | PRD-FR-020..033, TRD-SCR/RAG | ADR-0005 | common observation/scoring contracts, facets/agreement/validation primitives | Accepted principle; generalized discrimination/range/drift extensions require separate recovery |
| Model relation/comparison | PRD-FR-040..043, TRD-MOD | ADR-0006 | relation-safe comparison primitives and diagnostics where merged; formal family-wide distinguishability remains work in progress | Partial |
| Bifactor scoreability | PRD-FR-044, TRD-BIF | ADR-0006 | protected-main package exposes bifactor scoreability surfaces; interpretation still depends on the evidence contract and model relation | Accepted bounded capability / evolving evidence |
| Factor retention | PRD-FR-040/050, TRD-MOD-001 | ADR-0006 | dimensionality diagnostics exist; unified retention + structural-selection evidence workflow remains a product gap tracked by issue #608 | Partial / planned integration |
| Latent-space residual interaction | PRD-FR-040..052, TRD-MOD | ADR-0006 | MLSIRM family on protected main | Accepted only after substantive dimension/testlet/facet diagnosis; not a substitute for omitted structure |
| Adaptive rotation | PRD-FR-051/052, TRD-ROT | ADR-0009 | protected main contains `crates/mlsirm-core/src/rotation/`, PyO3 bindings, `python/fast_mlsirm/rotation.py`, `rotation_selection.py`, package-root exports, criterion-neutral selection and rotation regression/doctoring evidence | Accepted CPU baseline / planned GPU and broader recovery extensions |
| True-parameter recovery | PRD-PRN-003, TRD-TEST-003..006 | ADR-0008 | simulation/recovery reports, Rust/NumPy parity, scheduled statistical studies/recovery contracts | Accepted |
| Correlation vs recovery/agreement | PRD-PRN-003, scoring validity requirements | ADR-0008, ADR-0005 | recovery/simulation, agreement/QWK/facets evidence | Accepted: correlation is supplementary association evidence, never sole proof of parameter recovery or interchangeability |
| Multilevel/multiple-membership/temporal | PRD-FR-060..062, TRD-MLT | ADR-0007, ADR-0018, ADR-0019 | contracts plus OLS/AR state layer and a separate joint MAP hierarchical CT-AR Rasch slice exist on the stacked longitudinal PRs; estimated MMMC `u_h` and GPU parity remain excluded | Proposed/partial / active PR |
| Accessible standalone reports | PRD-FR-070..072, NFR-004 | ADR-0005 | report renderers, exact-value exports, WCAG-focused regression/doctoring | Accepted/evolving |
| Sensitive data / PII utility | privacy/security requirements | ADR-0012 | source-free/digest/opaque-id provenance where implemented; provider error redaction; hosted identity/retention downstream | Accepted reusable policy / Downstream operations |
| Continuous execution / documentation governance | TRD-DOC-002 / work-conserving automation | ADR-0013 | single-writer exact branch head; work-conserving when blocked; feasibility-first prioritization | Accepted governance |
| Reusable-core threat model | security/release requirements | ADR-0001/0002/0003/0005/0010/0012 | `docs/security/threat-model.md`, Security Scan/SAST/fuzz/resource tests | Accepted documentation baseline; feature-specific controls evolve |
| LLM credentials/orchestration | TRD-LLM-001..004 | ADR-0010 | repo/org automation contracts; deterministic paths avoid unnecessary model credentials | Accepted governance |
| Scientific vs business/safety criticality | enterprise/automated scoring decision boundary | ADR-0005 | measurement outputs remain separate from causal intervention/cost/utility policy; critical safety/business gates are not derived from psychometric discrimination alone | Accepted boundary |
| Release/provenance | PRD-FR-080..082, TRD release section | ADR-0003, ADR-0008 | release acceptance, commercial evidence, buyer packet, SBOM/provenance/readiness builders | Accepted baseline |
| Documentation architecture completeness | documentation governance | ADR index + `docs/README.md` | `ARCHITECTURE.md`, PRD/TRD, UML, logical ERD, domain/public-contract class view, traceability, threat model, documentation contract | ACTIVE PR; becomes an Accepted baseline only after the canonical set and contract test are protected-main integrated |

## Key source locations

### Canonical public contract composition

- `python/fast_mlsirm/scoring/contracts.py`
- `python/fast_mlsirm/rubric/__init__.py`
- `python/fast_mlsirm/__init__.py`

### Numerical source of truth

- `crates/mlsirm-core/`
- `crates/fast-mlsirm-py/`

### Scientific and product evidence

- `tests/`
- `fuzz/`
- `docs/doctoring/`
- `docs/changelog.d/`
- release/recovery/governance scripts under `scripts/`

### Architecture/security maintenance

- `ARCHITECTURE.md`
- `docs/PRD.md`
- `docs/TRD.md`
- `docs/adr/README.md`
- `docs/uml/`
- `docs/erd/domain-model.puml`
- `docs/security/threat-model.md`
- `docs/documentation_coverage.md`

## Conversation-wide interpretation invariants

These principles are intentionally repeated here because silently losing them would change product behavior even if file names and APIs remained stable:

1. **LLM and human judges are fallible raters.** Rater identity does not make an observation truth; severity, disagreement, bias/range/occasion and drift are measurement evidence where the design supports them.
2. **Correlation is not parameter recovery or absolute agreement.** Scientific estimator claims use aligned bias/MAE/RMSE/coverage/convergence/probability/information recovery as applicable; scorer interchangeability needs agreement/calibration evidence beyond association.
3. **Latent space follows substantive diagnosis.** Multidimensional, bifactor/higher-order, testlet/two-tier and rater/task/occasion structure are represented before residual latent-space interaction is added; latent geometry may not absorb an omitted scientific dimension by default.
4. **Psychometric discrimination is not business/safety criticality.** Item/judge discrimination measures how well observations separate latent levels. Policy-critical failure, causal action value, expected loss or regulatory severity are separate decision/governance layers and may require conjunctive gates or downstream utility models.
5. **Reference-free is not truth-free.** Groundedness to supplied context can be evaluated without a gold answer, but world correctness, completeness and absolute retrieval recall require stronger evidence.
6. **Context and time are part of the design.** Nested/cross-classified/multiple-membership/repeated/temporal structure is not silently flattened when the intended inference depends on it.

## Documentation authority

The documentation authority order is:

1. protected-main source and tests for executable behavior;
2. accepted ADRs for governing architectural/scientific decisions;
3. `docs/PRD.md` and `docs/TRD.md` for product/technical requirements;
4. root `ARCHITECTURE.md`, UML/ERD and reusable-core threat model for system views;
5. method-specific/doctoring documentation and primary literature;
6. proposed ADRs/open issues/PRs for future work.

PR bodies, automation handoffs and conversations are evidence/discovery inputs but are not authoritative after their decisions have been captured here.

## Maintenance rule

A PR that materially changes a public contract, bounded-context ownership, numerical owner, native binding/export authority, model interpretation, lifecycle, trust/privacy boundary, scientific acceptance criterion or release requirement must update this matrix or explicitly demonstrate that the existing mapping remains correct. A maturity row may move to Accepted/implemented only when the corresponding code is on protected main with the required exact-head evidence.
