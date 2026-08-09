# Documentation Completeness Audit — fast-mlsirm

Date: 2026-08-09  
Scope: architecture and product/technical design content accumulated across the fast-mlsirm project discussions, research notes, repository documentation, and current protected-main product boundary.

## Executive finding

The documentation corpus was **not sufficient as a canonical architecture package** before this baseline.

The repository already contained strong feature/method documentation — notably repository/agent guidance, rubric-centered item generation, shared assessment/scoring contracts, release/commercial evidence, and method-specific research/doctoring — but the design intent was distributed across conversations, PR bodies, feature docs, and agent instructions. The repository lacked a single canonical set of:

- product requirements (PRD);
- technical requirements (TRD);
- ISO/IEC/IEEE 42010-style architecture description;
- ADR index and durable architecture decisions;
- UML component/class/sequence/state views;
- logical ERD and persistence-ownership boundary;
- requirements-to-design-to-evidence traceability;
- a completeness audit telling maintainers what is authoritative versus still missing.

This branch adds that baseline. It does **not** claim that every roadmap feature is implemented.

## 1. Pre-baseline audit

| Documentation class | Before baseline | Finding |
|---|---|---|
| README / product scope | Strong | Good operational/product introduction, but not a requirements specification |
| AGENTS / CLAUDE guidance | Strong | Excellent contributor constraints; inappropriate as sole architecture authority |
| Feature design docs | Strong but distributed | Good local designs for rubric/scoring/rotation/recovery/reporting |
| Scientific references/doctoring | Strong and growing | Method evidence exists, but cross-feature architecture trace was fragmented |
| Commercial/release evidence | Strong | Release/buyer evidence framework is ahead of the architecture index |
| PRD | Missing canonical document | Product requirements were scattered across conversations/issues/docs |
| TRD | Missing canonical document | Technical invariants existed in AGENTS/PRs but were not normalized as requirements |
| Architecture description | Missing root canonical file | No top-level architecture viewpoints/bounded-context map |
| ADRs | Missing canonical index | Durable decisions were encoded as prose and repeated instructions |
| UML | Missing coherent suite | Individual ASCII/Mermaid flows existed, not a maintained architecture view set |
| ERD | Missing | No explicit statement that the core owns logical contracts but not physical DB schema |
| Requirements traceability | Missing | Difficult to prove feature→design→test/evidence coverage across the product |
| Documentation completeness policy | Missing | No mechanism to detect architecture documentation drift |

## 2. Conversation/research themes covered by the new baseline

The baseline explicitly incorporates the durable conclusions from the project's major research/development threads.

### Reference-free RAG measurement

Covered in PRD/ADR/architecture:

- groundedness, correctness, completeness/coverage, relevance, robustness, and abstention remain distinguishable constructs;
- LLM judges are fallible raters, not truth;
- multi-judge/facet calibration, anchors, DIF, G-theory, and held-out validation are part of the evidence path;
- candidate-blind or cross-fitted rubric generation is required where benchmark leakage matters.

### Dynamic rubric and item generation

Covered in PRD/ADR/UML/ERD:

```text
Rubric → Blueprint → Generation Contract → Candidate Validation
→ Screening → Pilot → Rust Calibration → Governed Item Bank → Monitoring
```

The architecture distinguishes schema conformance from psychometric validity and preserves immutable rubric/item provenance.

### Automated scoring and essay evaluation

Covered in PRD/TRD/ADR/UML:

- one AssessmentSpec/rubric source of truth;
- human and automated raters under a shared observation contract;
- severity/range-use/drift/fairness evidence;
- agreement/correlation as descriptive evidence rather than sufficient validity;
- adjudication and monitoring as separate lifecycle stages.

### Factor retention and structural models

Covered in PRD/TRD/ADR:

- factor count is separate from model form;
- correlated MIRT, bifactor, higher-order, testlet, two-tier, many-facet, and latent-space structures answer different questions;
- formal relation must be derived from constraints/boundaries rather than names;
- predictive evidence, residual dependence, scoreability, invariance/DIF, and recovery accompany formal model comparison.

### Adaptive factor rotation

Covered in PRD/TRD/ADR-002/005:

- no universal best criterion;
- Rust criterion registry and multi-start optimization;
- best-observed solution rather than global-optimum claim;
- criterion-neutral stability/recovery/theory evidence;
- rotation/sign/permutation alignment before recovery comparisons;
- GPU work requires parity evidence and is not implied by CPU support.

### True-parameter recovery and AI validation

Covered in PRD/TRD:

- scale/alignment before error calculation;
- bias/RMSE/coverage/convergence over correlation-only claims;
- recovery of response/information behavior when that is the decision-relevant target;
- realistic missingness, grouping, rater, and temporal design in simulations.

### Multilevel, multiple membership, and time

Covered in ADR-006, TRD, UML/ERD:

- atomistic fallback is not acceptable when hierarchy is scientifically material;
- context dimension and context identity are distinct;
- cross-classification and multiple membership are explicit;
- exact time/order/revision provenance is preserved;
- discrete occasion-step dynamics are not mislabeled continuous-time dynamics.

### MSA / CWL ecosystem

Covered in ARCHITECTURE and ADR-001/008:

- fast-mlsirm is the reusable measurement core;
- Psychometrics Commons is the downstream hosted product;
- contextual-orchestrator, TEPP, Keyverse, EgressWeave, semantic-data-portal, naruon and other CWL repositories are integrations rather than hidden implementation dependencies;
- the core owns logical measurement contracts, not hosted persistence, auth, sessions, billing, or deployment.

## 3. Post-baseline canonical package

| Artifact | Purpose |
|---|---|
| `ARCHITECTURE.md` | canonical ISO 42010-style architecture description and bounded-context map |
| `docs/PRD.md` | product requirements, personas, functional/non-goal/product-horizon baseline |
| `docs/TRD.md` | enforceable technical, numerical, psychometric, security, quality, release constraints |
| `docs/architecture/UML.md` | component, class, sequence, model-selection, state and deployment views |
| `docs/architecture/ERD.md` | logical contract/provenance ERD and downstream persistence rules |
| `docs/adr/README.md` | ADR governance and index |
| `docs/adr/ADR-001..008` | durable architecture decisions |
| `docs/requirements_traceability.md` | requirement→architecture→implementation/evidence map |
| `docs/documentation_coverage.md` | this completeness and drift audit |

## 4. Remaining documentation gaps after this baseline

The architecture corpus is now sufficient as a **baseline**, but not complete forever. The following remain required as features mature.

### P0 — release-blocking documentation gaps for a feature when applicable

- exact API/schema migration guides for any breaking contract revision;
- method-specific identification/parameterization docs for new estimators;
- primary-source equation/page trace for interpretation-critical formulas;
- recovery/benchmark report tied to the exact implementation;
- security/privacy threat/control delta for new external data/model boundaries;
- release notes and rollback implications.

### P1 — platform maturity

- canonical item-bank lifecycle API reference once implementation lands;
- artificial-crowd orchestration contract and evidence schema;
- formal factor-retention API/decision protocol;
- formal score/information interface for general Vuong distinguishability;
- multilevel/longitudinal Rust estimator architecture and identification ADR;
- continuous-time model ADR if elapsed intervals enter the likelihood;
- RAG observation adapter specification;
- essay calibration/adjudication/monitoring end-to-end operational guide.

### P2 — downstream/product integration

These belong primarily in owning repositories, not here:

- Psychometrics Commons physical ERD and migrations;
- tenant/RBAC/SSO/SCIM diagrams;
- consent/session/data-rights state machines;
- customer data residency and key-management deployment views;
- billing/usage metering;
- hosted SLO/SLA/topology;
- public research catalog deployment.

## 5. Documentation quality gates

A substantive PR should be considered documentation-complete only when all applicable statements below are true:

1. public behavior is described somewhere authoritative;
2. a new architecture invariant has an ADR;
3. the PRD/TRD are updated if the product/technical requirement changed;
4. UML/ERD views are updated when components, flows, cardinalities, or ownership changed;
5. `docs/requirements_traceability.md` maps the new requirement/evidence;
6. method claims cite primary authoritative sources in APA 7 form;
7. implementation status is not overstated — planned contracts are not described as protected-main functionality;
8. exact test/recovery/release evidence exists separately from design prose;
9. changelog/version are updated when the release contract warrants them;
10. outdated docs are removed or explicitly superseded rather than allowed to coexist ambiguously.

## 6. Standards baseline

The documentation framework is aligned to the following current or confirmed standards as of 2026-08-09:

- ISO/IEC/IEEE 42010:2022 — architecture descriptions;
- ISO/IEC/IEEE 29148:2018 — requirements engineering (confirmed current in 2024, revision underway);
- ISO/IEC/IEEE 12207:2026 — software life-cycle processes;
- ISO/IEC/IEEE 15289:2019 — life-cycle information items/documentation (confirmed current in 2025);
- ISO/IEC 25010:2023 — software product quality model;
- ISO/IEC 5338:2023 — AI-system lifecycle processes where AI integrations are involved;
- OMG UML 2.5.1 — diagram vocabulary used conceptually in the UML companion.

## 7. Overall assessment

Before this baseline, documentation maturity was **strong at feature-level evidence and operational governance, weak at canonical system architecture and cross-feature traceability**.

After this baseline, the repository has a coherent product/technical/architecture/decision/diagram/ERD/traceability spine. The remaining work is to keep that spine synchronized with implementation rather than continue adding isolated design documents without updating the canonical model.
