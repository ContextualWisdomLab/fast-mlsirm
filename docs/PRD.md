# Product Requirements Document — fast-mlsirm

Status: **Canonical product requirements baseline**  
Version: 0.1  
Date: 2026-08-09

## 1. Product definition

`fast-mlsirm` is a reusable, domain-neutral psychometric measurement and AI-evaluation toolkit. It must support researchers, assessment engineers, AI-evaluation engineers, and downstream products that need auditable measurement contracts, Rust-first estimation, model diagnostics, scoring calibration, item/rater evidence, and reproducible decision support.

It is **not** the hosted assessment product, identity provider, tenant database, billing system, consent/session application, or general-purpose LLM orchestration service.

## 2. Product problem

Existing assessment and LLM-evaluation stacks commonly fail in one or more of these ways:

- treat LLM/human ratings as ground truth rather than fallible measurements;
- average heterogeneous RAG/LLM-judge metrics without item/rater calibration;
- create rubrics and items ad hoc without immutable evidence/task contracts;
- select psychometric models by fit alone without relation-safe comparison, predictive evidence, or recovery;
- report correlations without absolute-error, calibration, uncertainty, DIF, or scoreability evidence;
- ignore multilevel, multiple-membership, rater, testlet, and temporal dependence;
- make scientific behavior difficult to reproduce across versions, providers, environments, and release artifacts.

The product must make these failure modes structurally difficult.

## 3. Product principles

1. **Measurement before decision.** Observations, calibration, validation, and decision policy are separate layers.
2. **Rubric and item provenance are first-class.** Generated content is not trusted merely because it is valid JSON.
3. **LLM and human judges are fallible raters.** Severity, discrimination, range use, drift, and disagreement remain measurable.
4. **Model selection is relation-safe.** Unknown or boundary relationships fail closed.
5. **True-parameter recovery outranks correlation-only validation.** Bias, RMSE, interval coverage, convergence, and decision consequences matter.
6. **Rust owns production numerics.** Python owns orchestration/contracts/reporting and governed references.
7. **Hierarchies and time are first-class when the design contains them.** No atomistic default for clustered/cross-classified/longitudinal data.
8. **Standalone first, MSA-compatible always.** No hidden hosted-product dependency.
9. **Content-minimized auditability.** Preserve provenance without unnecessarily retaining raw response/source/prompt text.
10. **Evidence must bind to exact artifacts.** Stale or predecessor evidence does not become release proof.

## 4. Personas and jobs-to-be-done

### Psychometrician / researcher

- simulate realistic measurement designs;
- fit and compare IRT/MIRT/MLSIRM/faceted/testlet models;
- recover known parameters with bias/RMSE/coverage evidence;
- inspect model fit, local dependence, DIF/invariance, reliability, linking, and uncertainty;
- reproduce results from immutable inputs and model versions.

### AI-evaluation / scoring engineer

- define one versioned `AssessmentSpec` and rubric set;
- represent human and automated scorers under one observation contract;
- calibrate LLM judges instead of averaging their raw scores;
- handle abstention/failure/exclusion distinctly from low scores;
- route uncertain/disagreeing cases to human adjudication;
- monitor rater/model/rubric drift.

### Item/rubric engineer

- compile rubric → blueprint → bounded generation contract;
- validate untrusted generated candidates;
- screen answerability, ambiguity, evidence support, leakage, bias, duplication, and construct alignment;
- pilot with artificial crowds and/or humans;
- publish only calibrated, approved items into a governed item bank.

### Downstream product engineer

- import versioned contracts and results without importing hosted-product code;
- persist logical artifacts in an owning service;
- integrate optional LLM/egress/temporal services without changing core measurement semantics.

## 5. Functional requirements

### PRD-CONTRACT-001 — Assessment contracts

The package shall expose immutable, content-addressed, versioned assessment, construct, rubric, scoring-engine, validation, adjudication, monitoring, and reporting contracts.

Acceptance:

- stable canonical serialization;
- full SHA-256 fingerprint for exact content identity;
- bounded descriptive public handles where required;
- fail-closed cross-reference validation;
- raw content excluded from governed metadata unless a contract explicitly requires it.

### PRD-RUBRIC-001 — Rubric-centered item authoring

The package shall support the governed path:

```text
RubricSpecification → BlueprintPlan → ItemBlueprint → GenerationContract
```

It shall keep schema version distinct from semantic rubric version and invalidate downstream provenance when an authoritative rubric changes.

### PRD-RUBRIC-002 — Generated-candidate validation

The product shall treat provider output as untrusted and validate:

- duplicate JSON keys and non-finite values;
- exact allowed fields and bounded sizes;
- typed answer-key semantics by response format;
- rubric score coverage/order;
- source identities and evidence spans;
- contract/rubric/blueprint provenance replay;
- candidate identifiers/fingerprints.

Structural validity shall not imply psychometric validity.

### PRD-BANK-001 — Governed item-bank lifecycle

The product shall support a logical lifecycle:

```text
draft → audited → screened → piloting → calibrated → approved → active → suspended → retired
```

The lifecycle shall preserve calibration history, linking anchors, exposure/drift/DIF evidence, approval provenance, regeneration triggers, and rollback-safe version identity. Physical persistence belongs to an owning application/service.

### PRD-SCORING-001 — Unified human/automated scoring

The package shall represent human and automated scoring through compatible provider-neutral request/observation/result contracts while preserving distinct engine provenance.

Observation states shall distinguish `scored`, `abstained`, `failed`, and `excluded` without coercing terminal/missing states to low scores.

### PRD-SCORING-002 — Automated essay scoring calibration

The package shall support automated essay-scoring calibration/validation as a measurement problem, including:

- criterion-level ratings;
- human and AI rater facets;
- severity and range-use evidence;
- QWK/exact/adjacent agreement as descriptive evidence, not sole validity proof;
- DIF/subgroup evidence;
- human-review routing;
- model/rubric version drift.

### PRD-RAG-001 — Reference-free RAG measurement

The package shall support a domain-neutral observation schema suitable for reference-free RAG evaluation where groundedness, correctness, completeness/coverage, relevance, robustness, and abstention are distinct constructs.

LLM judge outputs shall be calibration inputs, not truth labels. Candidate-blind evidence-grounded rubric generation and perturbation anchors shall be supported by adapters/orchestrators without embedding a provider SDK into the core.

### PRD-PSY-001 — IRT/MIRT/MLSIRM and faceted measurement

The package shall support domain-appropriate psychometric estimation and diagnostics including existing IRT/MIRT/MLSIRM, many-facet, linking, CAT/ATA, fit, local dependence, DIF/invariance, and reliability/scoreability capabilities.

New statistical features shall be integrated into the same evidence and provenance model rather than exposed as isolated ungoverned utilities.

### PRD-MODEL-001 — Factor retention and structural model selection

The product shall separate factor retention from structural model choice and support evidence for:

- unidimensional and correlated multidimensional structures;
- bifactor and higher-order structures;
- testlet and two-tier structures;
- multifaceted/rater structures;
- latent-space residual interaction;
- adaptive rotation when exploratory interpretation is required.

The package shall not infer formal nestedness from model names alone.

### PRD-MODEL-002 — Relation-safe model comparison

Comparison APIs shall classify candidate relationships and fail closed when relation/distinguishability is unknown.

Selection evidence shall combine the appropriate subset of:

- regular LR;
- boundary-aware/parametric-bootstrap LR;
- formal Vuong distinguishability and selection for non-nested models;
- held-out or clustered predictive likelihood;
- residual dependence;
- parameter/structure recovery;
- scoreability and interpretability.

### PRD-ROT-001 — Adaptive factor rotation

The product shall provide an extensible Rust rotation criterion registry, multi-start optimization, solution diagnostics, and criterion-neutral evidence for selecting a best-observed solution/policy. It shall never claim a universal rotation criterion or finite-multistart global optimality.

### PRD-MULTI-001 — Multilevel and multiple-membership designs

The contract and estimator roadmap shall support explicit context dimensions, nesting, cross-classification, multiple membership, exact membership weights, connectedness/identification diagnostics, and rater/task contexts when required by the data-generating design.

### PRD-TIME-001 — Temporal and longitudinal designs

The product shall preserve exact temporal ordering and version/occasion identity. Discrete-step AR/state contracts shall not be misrepresented as continuous-time models. Continuous-time or interval-adjusted estimators require separate Rust implementations and recovery evidence.

### PRD-RECOVERY-001 — Scientific recovery evidence

New estimators or parameterizations shall include realistic simulations with known truth and report, as applicable:

- bias;
- MAE/RMSE;
- interval coverage / SE calibration;
- convergence/failure rate;
- ICC/category-response/information recovery;
- alignment for rotation/latent-space non-identifiability;
- CPU/GPU parity.

### PRD-REPORT-001 — Auditable reporting

The product shall emit machine-readable and accessible human-readable reports that preserve exact values, limitations, model/provenance identity, and insufficient-evidence states without making unsupported high-stakes or causal claims.

### PRD-RELEASE-001 — Release evidence

A release shall be tied to one exact protected commit and reproducible package artifacts, including tests, security, coverage, package/import, provenance/SBOM where applicable, changelog, model limitations, support boundaries, and release-acceptance evidence.

## 6. Quality requirements

The product shall optimize against ISO/IEC 25010:2023 quality characteristics relevant to the library:

- functional suitability;
- performance efficiency without sacrificing numerical correctness;
- compatibility/interoperability;
- reliability and recoverability;
- security and supply-chain integrity;
- maintainability/modularity/testability;
- portability across supported Python/Rust/wheel environments;
- interaction capability/accessibility for generated human-readable reports.

Quality claims must be backed by evidence, not adjectives.

## 7. Explicit non-goals

The core library does not own:

- user authentication or product authorization;
- tenant/customer persistence and DB migrations;
- assessment session/consent/result-access lifecycle;
- raw PII document storage;
- general LLM routing or model credentials;
- billing/CRM/support portals;
- hosted UI/control plane;
- causal intervention policy engines;
- regulatory certification claims.

Those concerns may consume `fast-mlsirm` outputs through explicit interfaces.

## 8. Product horizons

### Horizon A — existing/near-term measurement platform

- stable Assessment/Rubric/Scoring contracts;
- generated-candidate structural/evidence validation;
- automated-scoring calibration/validation;
- relation-safe model selection;
- bifactor scoreability and factor retention;
- multilevel/temporal contract support;
- adaptive rotation;
- exact release evidence.

### Horizon B — governed evaluation lifecycle

- semantic candidate screening;
- artificial-crowd orchestration contracts;
- calibrated item-bank lifecycle;
- linking, exposure, DIF/drift, quarantine/retirement;
- reference-free RAG adapters;
- essay-scoring adjudication/monitoring workflows;
- buyer-facing workflow reports.

### Horizon C — ecosystem-scale research and operational evidence

- richer multilevel/longitudinal Rust estimators;
- formal score/information interfaces for general Vuong distinguishability;
- mixed response likelihoods where justified;
- parity-verified GPU acceleration for computationally material kernels;
- immutable interoperability artifacts for downstream products.

## 9. Release decision

A feature is not release-ready merely because its PR is mergeable. Release requires the exact integrated protected head to satisfy the technical gates in `docs/TRD.md` and the repository release-acceptance evidence contract.

## References

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

ISO/IEC. (2023). *ISO/IEC 25010:2023 Systems and software engineering—Systems and software Quality Requirements and Evaluation (SQuaRE)—Product quality model*.

ISO/IEC/IEEE. (2018). *ISO/IEC/IEEE 29148:2018 Systems and software engineering—Life cycle processes—Requirements engineering*.

Mislevy, R. J., Almond, R. G., & Lukas, J. F. (2003). A brief introduction to evidence-centered design. *ETS Research Report Series, 2003*(1), i–29. https://doi.org/10.1002/j.2333-8504.2003.tb01908.x

Williamson, D. M., Xi, X., & Breyer, F. J. (2012). A framework for evaluation and use of automated scoring. *Educational Measurement: Issues and Practice, 31*(1), 2–13. https://doi.org/10.1111/j.1745-3992.2011.00223.x
