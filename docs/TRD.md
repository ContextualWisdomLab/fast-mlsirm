# Technical Requirements Document — fast-mlsirm

Status: **Canonical technical requirements baseline**  
Version: 0.1  
Date: 2026-08-09

## 1. Purpose

This TRD turns `docs/PRD.md` into enforceable technical constraints for the standalone library, Rust/PyO3 computation core, Python contract/orchestration layer, CI/release evidence, and optional integration boundaries.

It follows current software-life-cycle and documentation guidance from ISO/IEC/IEEE 12207:2026, ISO/IEC/IEEE 15289:2019, ISO/IEC/IEEE 29148:2018, ISO/IEC/IEEE 42010:2022, and the ISO/IEC 25010:2023 product-quality model.

## 2. Technical architecture requirements

### TRD-ARCH-001 — Independent package boundary

`fast-mlsirm` shall remain installable and testable without importing product-specific code from Psychometrics Commons, naruon, Keyverse, TEPP, contextual-orchestrator, EgressWeave, or other CWL repositories.

Allowed cross-repository coupling:

- versioned serialized contracts;
- immutable artifact references;
- optional provider-neutral adapters;
- explicit service APIs owned by downstream integrations.

Prohibited coupling:

- downstream ORM/database models in the core package;
- product HTTP/session/consent/tenant types;
- shared mutable filesystem assumptions;
- hidden model credentials or deployment configuration.

### TRD-ARCH-002 — Module ownership

The source tree shall preserve clear ownership:

```text
python/fast_mlsirm/       public Python API, contracts, validation, orchestration, reports
crates/mlsirm-core/       Rust psychometric/statistical production numerics
crates/fast-mlsirm-py/    PyO3 bindings and transport
scripts/                  deterministic release/evidence tooling
tests/                    public-contract, delegation, integration, recovery and regression evidence
fuzz/                     adversarial parser/config/numeric safety evidence
docs/                     requirements, architecture, ADRs, doctoring, standards, method limitations
```

## 3. Contract requirements

### TRD-CONTRACT-001 — Canonical serialization

Governed contracts shall have one deterministic canonical representation. Order-insensitive fields must normalize deterministically; semantically ordered fields must preserve order.

### TRD-CONTRACT-002 — Identity layers

Where exposed, content-addressed artifacts shall distinguish:

- logical descriptive ID;
- schema version;
- semantic/governance version;
- full SHA-256 fingerprint;
- bounded public handle if an external compact reference is required.

Short display IDs must not substitute for full fingerprints in durable provenance, replay protection, deduplication, or authorization decisions.

### TRD-CONTRACT-003 — Input bounds and exact types

Caller-controlled values shall be validated before expensive allocation, numerical work, canonical hashing, or callback invocation. Validation must reject, where not explicitly supported:

- Boolean-as-integer ambiguity;
- non-finite floating values;
- negative or overflow dimensions;
- unbounded collection materialization;
- malformed or one-token public identifiers when the two-token naming contract applies;
- unexpected object subclasses capable of executing hostile conversion callbacks;
- unknown fields in closed schemas.

Public errors shall expose stable codes/paths and shall not echo rejected secrets or raw source/response/provider content.

### TRD-CONTRACT-004 — Immutability and replay

Factory-created governed artifacts shall be replayable from normalized package-owned values. Aggregate builders shall revalidate child artifact identity/type/provenance rather than trusting post-construction Python object state.

## 4. Numerical requirements

### TRD-NUM-001 — Rust-first production arithmetic

Likelihoods, gradients, Hessians/information, iterative estimation, psychometric scoring/ranking, factor/model numerical kernels, DIF statistics, linking/equating arithmetic, and computationally material optimization introduced as production functionality shall be implemented in Rust.

Python may:

- validate inputs;
- prepare typed/bounded arrays;
- orchestrate calls;
- expose explicitly governed reference implementations for parity/research;
- marshal results and generate reports.

Python reference arithmetic must not silently become the production source of truth for a Rust-owned feature.

### TRD-NUM-002 — CPU execution

Computationally material Rust kernels shall use coarse-grained CPU multithreading where it improves throughput without increasing context switching, nondeterministic reduction error, or memory blow-up beyond the accepted contract.

### TRD-NUM-003 — GPU execution

GPU support is warranted when batch size/algorithm structure can benefit materially. GPU paths shall:

- be explicit about precision and backend/device identity;
- fail or fall back according to a documented contract;
- include numerical/recovery parity appropriate to the method;
- never mark a skipped GPU test as GPU success;
- avoid separate statistical semantics from the CPU implementation.

### TRD-NUM-004 — Resource preflight

Before allocating input-dependent matrices/tensors/workspaces, production and retained reference/fallback paths shall preflight checked dimensions and byte budgets where allocation can be buyer-controlled or adversarially large.

### TRD-NUM-005 — Numerical evidence

Every new estimator/parameterization shall include as applicable:

- independent hand/numerical or finite-difference oracles;
- property/metamorphic invariants;
- true-parameter bias and RMSE;
- interval/SE coverage;
- convergence and failure classification;
- boundary and degeneracy cases;
- scale/rotation/label alignment before recovery comparison;
- CPU/GPU or Rust/reference parity.

A high correlation alone is not sufficient estimator-validation evidence.

## 5. Psychometric structural requirements

### TRD-PSY-001 — Dimensionality and factor retention

Factor retention shall be a distinct operation from confirmatory structural-model selection. Candidate-count evidence may include parallel analysis/MAP/EGA or analogous exploratory evidence, likelihood/information criteria, bootstrap LR, predictive likelihood, residual dependence, and recovery appropriate to the response model.

### TRD-PSY-002 — Structural model relation

Comparison code shall classify actual parameter constraints/boundaries, not model names only. Relation classes include at least:

- regular nested;
- boundary/singular nested;
- nonlinear/constrained nested;
- strictly non-nested;
- overlapping;
- indistinguishable/degenerate;
- unknown.

Unknown relation or missing distinguishability evidence shall suppress winner claims.

### TRD-PSY-003 — Bifactor interpretation

Bifactor model fit and bifactor scoreability are separate gates. General/specific score reporting requires relevant reliability/ECV/PUC/construct-replicability/factor-determinacy evidence and a structurally meaningful declared general factor.

### TRD-PSY-004 — Testlet/local dependence

Shared stimulus/question/source/prompt structures shall be modeled or diagnosed as local-dependence/testlet structures when the design justifies them. Latent-space interaction must not be used merely to absorb omitted substantive/testlet/facet structure.

### TRD-PSY-005 — Multilevel and multiple membership

Contracts and future estimators shall distinguish context dimension from context identity and support nesting, cross-classification, multiple membership, and exact membership weights. Estimators must reject unidentified/confounded/disconnected designs rather than silently fit atomistic approximations.

### TRD-PSY-006 — Temporal structure

Task/rubric/model/occasion revisions shall retain exact ordering/version provenance. Discrete occasion-step state models shall not claim continuous-time dynamics. Interval-dependent likelihoods require a separate model and recovery contract.

## 6. Rubric and item-generation requirements

### TRD-RUBRIC-001 — Rubric source of truth

`fast_mlsirm.rubric.RubricSpecification` and related rubric contracts are the sole internal rubric source of truth. Scoring, essay, RAG, and other adapters reference exact rubric fingerprints instead of defining parallel incompatible rubric schemas.

### TRD-RUBRIC-002 — Candidate-blind benchmark path

Benchmark/evaluation adapters shall support rubric/item generation without seeing the candidate being evaluated when the methodology requires candidate-blindness. Candidate-aware discovery must use explicit cross-fitting or a separately governed training/discovery bank.

### TRD-RUBRIC-003 — Provider isolation

Core rubric/item-generation contracts shall not depend on a hosted LLM SDK. Network/provider execution belongs to optional adapters or an owning service such as contextual-orchestrator.

### TRD-RUBRIC-004 — Screening separation

Structural parsing shall be separate from semantic/evidence screening. Screening results should distinguish at least construct alignment, atomicity, answerability, evidence grounding, ambiguity, redundancy, leakage, bias/fairness risk, anchor recovery, and runtime cost.

## 7. Scoring/LLM requirements

### TRD-SCORE-001 — Unified observation surface

Human, deterministic, and automated raters shall produce compatible typed observations while preserving engine-kind provenance.

### TRD-SCORE-002 — Missing/terminal semantics

`abstained`, `failed`, `excluded`, `not_applicable`, and insufficient-evidence conditions shall not be converted to low scores.

### TRD-SCORE-003 — LLM model tests

Live LLM-dependent tests shall use `NVIDIA_NIM_API_KEY` from GitHub Secrets. They should use contextual-orchestrator when appropriate, while respecting repository writer boundaries. `COPILOT_GITHUB_TOKEN` shall not be used for model execution or autonomous development.

### TRD-SCORE-004 — Orchestration evidence

When a product feature uses model orchestration, test-time compute allocation should be explicitly controlled by workflow stage, decomposition/recursion, role, access list, and reasoning effort. Single-model routing and deeper orchestration shall be compared under defensible budgets when this affects quality. Speed is not the primary optimization target.

## 8. Security and privacy requirements

### TRD-SEC-001 — Least privilege and supply chain

CI/release workflows shall use least privilege, immutable action/workflow references where practical, fail-closed security gates, bounded artifacts, and no PR-controlled self-modifying writer workflow.

### TRD-SEC-002 — Data minimization rather than blanket masking

Governed artifacts shall minimize raw sensitive content while preserving operationally necessary evidence. Raw PII/source/response/prompt data, when business-required, stays under the owning service's purpose-bound authorization, encryption, retention, data-rights, and audit controls instead of being indiscriminately masked inside the measurement core.

### TRD-SEC-003 — Hash semantics

Content fingerprints are provenance identities only. They shall not be presented as encryption, authentication, authorization, anonymization, or digital signatures.

## 9. Reliability and process requirements

### TRD-REL-001 — Bounded subprocesses

Build/recovery/release tooling shall bound subprocess execution with operation-specific deadlines appropriate to the scientific workload. Timeout evidence shall not echo untrusted command/output secrets and, where needed, shall terminate descendant process groups predictably.

### TRD-REL-002 — Exact-head evidence

Any statement that a PR/release passed a gate must name or bind to the exact source head/artifact. Predecessor/stale/synthetic-only evidence does not transfer after source changes.

### TRD-REL-003 — Release integrity

A release shall include or derive deterministic evidence for:

- tests and coverage;
- Rust/PyO3 build/import;
- explicit GPU evidence when GPU is claimed;
- package metadata and artifact digests;
- security/supply-chain checks;
- changelog/version;
- model/feature limitations;
- migration/rollback implications if any;
- provenance/SBOM as applicable.

## 10. Coverage and documentation requirements

### TRD-QUAL-001 — Coverage

Owned production code shall target exact 100% statement and branch coverage, plus function/line/region coverage where the toolchain exposes it. This requirement must not be met by excluding meaningful behavior or writing assertions that do not test observable contracts.

### TRD-QUAL-002 — Docstrings/rustdoc

Public APIs and scientifically material internal kernels shall have beginner-readable documentation describing inputs, outputs, units/scales, assumptions, failure modes, numerical ownership, and interpretation boundaries.

### TRD-QUAL-003 — Architecture traceability

Material product/technical decisions shall be traceable from PRD requirement → TRD requirement → ADR/architecture view → implementation/tests/evidence. `docs/requirements_traceability.md` is the canonical index.

## 11. Database naming and persistence boundary

This repository does not own a required physical database schema. If a test/example/integration introduces logical persistence names, database objects shall use descriptive two-or-more-word `snake_case` by default. Sequential numeric public identifiers are discouraged in favor of opaque descriptive references/UUIDv7/ULID where a downstream service requires durable public identity.

The logical ERD in `docs/architecture/ERD.md` is a contract relationship model, not an instruction to add an ORM to this package.

## 12. Supported environment and packaging

- Python support follows `pyproject.toml`; every advertised interpreter requires build/import/test evidence.
- Source/editable installs require a functioning Rust toolchain for deterministic maturin builds.
- Wheel releases shall include and import the compiled PyO3 extension for features sold as Rust-backed.
- Binding crates omitted from the root workspace shall receive explicit CI invocations.
- Cross-platform claims require evidence on the claimed platforms rather than inference from one runner.

## 13. Verification matrix

| Technical area | Required evidence |
|---|---|
| Contracts | canonical/replay/property/adversarial tests |
| Rust numerics | unit + property + recovery + parity |
| GPU | explicit non-skip + CPU/GPU numerical/recovery parity |
| Model selection | relation classification + correct formal test + CV/recovery |
| Bifactor | structural validation + scoreability evidence |
| Rubric generation | schema + provenance + injection/adversarial validation |
| Scoring | state/provenance/replay + rater calibration evidence |
| Multilevel/time | contract tests + identification/recovery before estimator release |
| Reports | accessibility + exact-value + no unsupported interpretation |
| Release | package/import/security/coverage/provenance exact-head evidence |

## 14. Standards and research basis

ISO/IEC. (2023). *ISO/IEC 25010:2023 Systems and software engineering—Systems and software Quality Requirements and Evaluation (SQuaRE)—Product quality model*.

ISO/IEC. (2023). *ISO/IEC 5338:2023 Information technology—Artificial intelligence—AI system life cycle processes*.

ISO/IEC. (2023). *ISO/IEC 42001:2023 Information technology—Artificial intelligence—Management system*.

ISO/IEC/IEEE. (2018). *ISO/IEC/IEEE 29148:2018 Systems and software engineering—Life cycle processes—Requirements engineering*.

ISO/IEC/IEEE. (2019). *ISO/IEC/IEEE 15289:2019 Systems and software engineering—Content of life-cycle information items (documentation)*.

ISO/IEC/IEEE. (2022). *ISO/IEC/IEEE 42010:2022 Software, systems and enterprise—Architecture description*.

ISO/IEC/IEEE. (2026). *ISO/IEC/IEEE 12207:2026 Systems and software engineering—Software life cycle processes*.

Schneider, L., Chalmers, R. P., Debelak, R., & Merkle, E. C. (2020). Model selection of nested and non-nested item response models using Vuong tests. *Multivariate Behavioral Research, 55*(5), 664–684. https://doi.org/10.1080/00273171.2019.1664280
