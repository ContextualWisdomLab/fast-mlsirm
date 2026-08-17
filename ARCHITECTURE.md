# Architecture — fast-mlsirm

Status: **Authoritative living architecture baseline**
Repository: `ContextualWisdomLab/fast-mlsirm`  
Last reviewed: 2026-08-11

This document describes the current and intended architecture of `fast-mlsirm`
using the concerns and viewpoints of ISO/IEC/IEEE 42010:2022. It is the root
navigation point for product requirements, technical requirements, ADRs,
UML/ERD diagrams, and research-to-code traceability.

## 1. Mission and boundary

`fast-mlsirm` is a reusable, domain-neutral psychometric measurement library
for Multidimensional Latent Space Item Response Models (MLSIRM/MLS2PLM) and
related IRT tooling. It owns scientific and numerical measurement truth and
reusable contracts; it does not own a hosted assessment application's runtime
state. It must stand alone as a local Python package with a Rust numerical core
and compose with ContextualWisdomLab products without importing sibling
repositories at runtime.

```text
Downstream products and services
  Psychometrics Commons / independent callers / research pipelines
                              |
                     versioned public contracts
                              v
+-------------------------------------------------------------------+
|                         fast-mlsirm                               |
|                                                                   |
| assessment/rubric/scoring contracts -> validation/orchestration   |
|              |                                      |             |
|              v                                      v             |
| response/rater evidence -> model selection and recovery evidence |
|              |                                      |             |
|              +---------------> Rust numerical core <+             |
|                                      |                            |
|                            PyO3 / typed results                    |
|                                      |                            |
|               reports / release / audit evidence                  |
+-------------------------------------------------------------------+
                              |
          optional explicit integrations, never hidden coupling
                              v
 contextual-orchestrator / TEPP / Gyeot / semantic-data-portal / ...
```

### Owned bounded context

- assessment, rubric, scoring, item, rater, response, and calibration
  contracts;
- CTT/IRT/MIRT and MLSIRM-family numerical functions;
- testlet, many-facet, factor/model-selection, DIF/invariance, linking,
  equating, G-theory, CAT, ATA, rotation, and recovery primitives;
- automated-scoring and LLM-judge validation primitives;
- governed rubric/item-bank lifecycles and deterministic scientific, audit,
  report, and release evidence.

### Explicitly outside the bounded context

- product HTTP/admin APIs, participant/session/consent/result persistence;
- identity/federation credentials and model-provider secret stores;
- hosted tenant/database migrations, end-user UI, and deployment control planes.

`ContextualWisdomLab/psychometrics-commons` is a downstream hosted assessment
product. The dependency direction is downstream -> `fast-mlsirm`; never the reverse.
Optional integrations are explicit host adapters, not hidden imports or
cross-service database access.

## Rust numeric core

Production psychometric arithmetic is owned by the **Rust numeric core** in
`crates/mlsirm-core` with PyO3 bindings in `crates/fast-mlsirm-py` and a thin
`python/fast_mlsirm` orchestration layer.

## 2. Architecture drivers

1. **Scientific defensibility:** interpretation is tied to identification, fit,
   recovery, invariance, uncertainty, and appropriate limitations.
2. **Reproducibility:** versioned content-addressed contracts and immutable
   revisions make analyses independently reconstructible.
3. **Performance:** production psychometric arithmetic is Rust-first with
   low-context-switch CPU parallelism and parity-verified GPU paths where
   material.
4. **Safety:** untrusted data, unsupported estimators, provider failures, and
   governance uncertainty fail closed rather than becoming silent success.
5. **Composability:** the package remains independently installable while
   exposing stable public contracts to CWL services and third parties.
6. **Explainability:** exact numerical values, provenance, model relation,
   convergence, and interpretation boundaries remain machine-readable.
7. **Evolution:** changed rubrics, items, models, and calibration artifacts are
   new versions or superseding revisions, never silent semantic mutation.

## Recovery evidence path

True-parameter recovery is a first-class **Recovery evidence path**: simulate
→ estimate with the production Rust path → report RMSE/bias-style accuracy
metrics under CI-gated contracts (see ADR 0008 and the verification plan).

## 3. Architectural views

### 3.1 Layered system and component view

See [`docs/uml/component.puml`](docs/uml/component.puml).

```text
Presentation / product surfaces (optional)
  HTML diagnostics reports · CLI · release and buyer evidence
                         |
Python orchestration (python/fast_mlsirm/)
  fit API · validation · simulation · I/O · scoring adapters · contracts
                         |
PyO3 binding registry (crates/fast-mlsirm-py)
                         |
Rust mlsirm-core (production numerical hot path)
  likelihood/gradients · MMLE/multigroup/multilevel · CAT/ATA
  CPU parallel execution · GPU marginal path where supported
```

Pure numeric work lives in Rust. Python owns validation, bounded
materialization, packaging, report rendering, and user-facing contracts. A
NumPy implementation may remain only as an explicit reference/parity backend;
it may not silently become a second production engine. GPU device selection is
explicit and CPU is the portable fallback.

### 3.2 Contract and data-flow view

See [`docs/uml/scoring-sequence.puml`](docs/uml/scoring-sequence.puml).

```text
AssessmentSpec + RubricSpecification
                 -> ScoringRequest
                 -> Human / AI / external engine
                 -> ScoreObservation
                 -> criterion/rater calibration handoff
                 -> Rust calibration
                 -> validation / fairness / adjudication / report
```

Every trust boundary rechecks content identity rather than trusting display
handles or cached parent objects. Host adapters own transport, authentication,
tenancy, persistence, and provider credentials.

### 3.3 Rubric/item-bank and lifecycle view

See [`docs/uml/item-bank-state.puml`](docs/uml/item-bank-state.puml) and
[`docs/uml/item-lifecycle.puml`](docs/uml/item-lifecycle.puml).

```text
approved rubric -> deterministic blueprint -> provider-neutral generation
  -> untrusted candidate -> structural/evidence/semantic screening
  -> pilot -> Rust calibration -> approval -> active monitoring
  -> quarantine/suspension/retirement or a new superseding revision
```

Candidate-blind generation is the default for benchmark/evaluation banks.
Candidate-aware discovery requires cross-fitting or an equivalent anti-leakage
design. Published or approved revisions do not mutate in place; correcting a
quarantined item creates a new draft identity and records supersession.

### 3.4 Model-selection and recovery view

See [`docs/uml/model-selection-sequence.puml`](docs/uml/model-selection-sequence.puml).

Model selection is multi-stage: determine factor-retention candidates,
classify the structural relation, use relation-appropriate inference, compare
cluster-aware held-out prediction, inspect residual dependence and
DIF/invariance, inspect scoreability and rotation stability, confirm realistic
true-parameter recovery, and choose the simplest model meeting interpretation
requirements. Bifactor, higher-order, testlet, two-tier, many-facet, and
latent-space structures are not interchangeable names.

### 3.5 Deployment and composition view

See [`docs/uml/deployment.puml`](docs/uml/deployment.puml).

`fast-mlsirm` is delivered as a Python package with a compiled Rust extension
and may be embedded in a CLI, notebook, batch worker, service, or hosted
product. Explicit CWL integrations include Psychometrics Commons, Keyverse,
Gyeot, TEPP, `contextual-orchestrator`, `pg-llm-batch`,
`semantic-data-portal`, and EgressWeave. Each host remains independently
operable; no service accesses another service's application database through
this library.

## 4. Domain and population model

See [`docs/erd/domain-model.puml`](docs/erd/domain-model.puml) and the
persistence-neutral [`docs/uml/domain-public-contract.puml`](docs/uml/domain-public-contract.puml)
class view.

The ERD documents reusable identity and cardinality, not ownership of a hosted
relational database. It includes assessment and rubric versions, item
blueprints/candidates/revisions, scoring requests/observations/results,
engine/rater descriptors, calibration designs/reports, item-bank history, and
model-comparison/recovery evidence. Calibration design inputs are a versioned
many-to-many association with observations.

The population contract must not force an atomistic analysis when the design is
hierarchical, multiply affiliated, or longitudinal:

- **single:** independent persons;
- **multigroup:** known group membership for DIF/equating contexts;
- **multilevel:** nested cluster effects such as school/class;
- **multiple membership:** weighted membership in more than one cluster;
- **longitudinal:** explicit person, occasion, time origin, and ordering with
  temporal validity rules.

The `fast_mlsirm.multilevel` contracts are content-addressed and fail closed.
The ADR-0018 state layer fits independent OLS trends and caller-supplied
discrete AR predictions. ADR-0019 adds a separate joint MAP hierarchical
continuous-time AR(1) Rasch slice with estimated `(mu, tau, lambda)`, elapsed-day
transitions, and Wald observed-information intervals. That slice excludes
estimated multiple-membership `u_h` and does not claim GPU parity. Remaining
nested/crossed estimators stay paper-scoped until their own Rust
implementation and recovery evidence are complete; the presence of a contract
is not a claim that every estimator is production-ready.

## 5. Numerical and scientific architecture

### 5.1 Rust ownership and parity

The Rust core owns production numerical algorithms. Python performs input
validation, provider/domain orchestration, NumPy marshaling, explicit reference
calculations, and report construction. Parity is checked at the identified
mathematical invariant: raw values where identified, Procrustes-aligned
loadings/coordinates under arbitrary rotation, pairwise distances for latent
geometry, and linked/scaled parameter errors after scale alignment.

### 5.2 Scientific evidence

True-parameter recovery is a release mechanism. Bias, RMSE, coverage,
convergence, information/function recovery, and realistic simulation are the
primary accuracy evidence; correlation is supplementary order-preservation
evidence and is not parameter recovery or absolute agreement.

LLM judges are fallible raters. Model family/version, prompt, order/occasion,
assignment, severity, discrimination, bias, and drift are retained whenever
they affect interpretation. Reference-free evaluation is not truth-free:
faithfulness to supplied context and world correctness require different
evidence regimes.

## 6. Security, privacy, and compliance posture

### 6.1 Trust boundaries

- bound before allocate, read, or materialize;
- use closed schemas, reject duplicate keys and non-finite JSON numbers;
- verify evidence spans against exact source revisions;
- sanitize untrusted exception text and never place secrets or uncontrolled
  source text in identifiers/evidence logs;
- enforce least-privilege, immutable action pins where practical, central
  SAST/dependency gates, exact-head evidence, and stale-head refusal;
- prohibit self-modifying write-capable PR workflows.

### 6.2 PII and assurance

The core library must not require blanket masking that destroys measurement
semantics. Prefer purpose limitation, opaque identifiers, minimal fields,
host-owned encryption and access control, auditable linkage, and separation of
identity-bearing hosted data from reusable measurement artifacts. CSAP and SOC
2 control objectives inform change control, access, logging, supply-chain, and
incident evidence; this document does not claim certification.

LLM automation uses dedicated `NVIDIA_NIM_API_KEY` credentials when a host
authorizes model execution and does not use `COPILOT_GITHUB_TOKEN` for agent
paths. Existing review-agent key schemes are not repurposed.

## 7. Quality attributes and test strategy

The principal ISO/IEC 25010:2023 concerns are functional suitability,
performance efficiency, compatibility, accessibility, reliability, security,
maintainability, flexibility, and safety. The corresponding evidence layers
are:

| Layer | Required evidence |
| --- | --- |
| Rust unit | equations, gradients, backend/device and multilevel edges |
| Recovery | seeded true-parameter recovery with RMSE/bias/coverage sentinels |
| Python API | fail-closed configuration, public fit path, real report behavior |
| GPU | explicit CPU parity smoke, including the CI software device where available |
| Fuzz/security | bounded CSV/report/config inputs and hostile-control rejection |
| CI matrix | complete pytest on CPython 3.12 and 3.14, with the required `python` aggregate |

Realistic tests must measure the software's scientific property: simulated
truth versus estimates for psychometrics, exact expected semantics for reports
and contracts, and parity across supported Rust/CPU/GPU/reference paths. A
green keyword or import-only test is not sufficient evidence of a production
feature.

## 8. Governance documents and conformance

The repository map is:

```text
crates/mlsirm-core/     Rust formulas, GPU marginal, recovery tests
crates/fast-mlsirm-py/  PyO3 bindings
python/fast_mlsirm/     public API and orchestration
tests/                  contract, security, recovery, and integration tests
docs/                   PRD/TRD, ADRs, UML/ERD, doctoring and traceability
scripts/                release acceptance, buyer evidence, changelog rendering
.github/workflows/      CI, security, and governance agents
```

`AGENTS.md` and `CLAUDE.md` define operating rules; `ARCHITECTURE.md` defines
system structure; `CHANGELOG.md` and `docs/changelog.d/` define release notes;
`docs/PRD.md` and `docs/TRD.md` define current requirements; `docs/doctoring/`
contains APA 7th source records; and the threat model, test strategy,
operability, and traceability documents define assurance evidence.

Continuous execution and documentation governance is recorded in
[`docs/adr/0013-continuous-execution-and-documentation-governance.md`](docs/adr/0013-continuous-execution-and-documentation-governance.md)
(**ADR-0013**): work-conserving, feasibility-first loops with a single active
writer per exact branch head.

The ADR index is [`docs/adr/README.md`](docs/adr/README.md). Material changes
conform only when they preserve bounded-context ownership, Rust/Python
numerical ownership, version/provenance and migration evidence, identification
and recovery evidence for model claims, fail-closed trust boundaries, and
updated requirements/ADR/test/release traceability. The machine-checkable
documentation contract is maintained in
`tests/test_architecture_documentation_contract.py`.

## 9. References (APA 7th)

American Educational Research Association, American Psychological Association,
& National Council on Measurement in Education. (2014). *Standards for
educational and psychological testing*. American Educational Research
Association.

Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel IRT
model. *Psychometrika, 66*(2), 271–288. https://doi.org/10.1007/BF02294839

International Organization for Standardization. (2023). *ISO/IEC 25010:2023
Systems and software engineering—Systems and software Quality Requirements and
Evaluation (SQuaRE)—Product quality model*.

International Organization for Standardization. (2023). *ISO/IEC 42001:2023
Information technology—Artificial intelligence—Management system*.

International Organization for Standardization, International Electrotechnical
Commission, & Institute of Electrical and Electronics Engineers. (2022).
*ISO/IEC/IEEE 42010:2022 Software, systems and enterprise—Architecture
description*.

Jeon, M., Jin, I. H., Schweinberger, M., & Baugh, S. (2021). Mapping unobserved
item-respondent interactions: A latent space item response model with
interaction map. *Psychometrika, 86*(2), 378–403.
https://doi.org/10.1007/s11336-021-09762-5

Kang, I., & Jeon, M. (2025). Multidimensional latent space item response
models: A note on the relativity of conditional dependence. *Psychometrika,
90*(2), 799–826. https://doi.org/10.1017/psy.2025.5

Molenaar, D., & Jeon, M. (2026). Regularized joint maximum likelihood
estimation of latent space item response models. *Psychometrika, 91*, 335–359.
https://doi.org/10.1017/psy.2025.10068

World Wide Web Consortium. (2024). *Web Content Accessibility Guidelines
(WCAG) 2.2*.
