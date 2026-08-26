# fast-mlsirm Product Requirements Document

Status: **Authoritative product requirements baseline**  
Repository: `ContextualWisdomLab/fast-mlsirm`  
Last reviewed: 2026-08-09

## 1. Product definition

`fast-mlsirm` is the reusable, domain-neutral measurement and psychometric computation layer for ContextualWisdomLab and for independent Python/Rust consumers. It provides governed measurement contracts, psychometric estimation and diagnostics, automated-scoring calibration/validation primitives, item/rubric lifecycle primitives, simulation/recovery evidence, and deterministic reports.

It is **not** the hosted Psychometrics Commons application. HTTP APIs, participant/session/consent/result lifecycle, product databases and migrations, UI, tenancy, deployment composition, and research-release orchestration belong to `ContextualWisdomLab/psychometrics-commons` or another owning bounded context. `fast-mlsirm` must remain independently installable and must not depend on hosted product code.

## 2. Product outcome

The product shall make measurement decisions defensible from the moment a construct/rubric is specified through scoring, psychometric calibration, model selection, validation, reporting, and governed item/model lifecycle.

The product is successful when a technical team can answer, with reproducible evidence:

1. **What construct and scoring contract was used?**
2. **Which exact item, rubric, response, rater/engine, task revision, and software artifact produced each observation?**
3. **Which psychometric model was fit, under which identification assumptions?**
4. **Are scores recoverable, reliable enough for their stated interpretation, invariant enough for the intended comparison, and free of known unresolved model-fit or fairness blockers?**
5. **How much uncertainty remains, and what evidence would change the decision?**
6. **Can the exact analysis be rebuilt and independently checked from immutable provenance?**

## 3. Primary users

### PRD-PER-001 Psychometric researcher

Needs simulation, true-parameter recovery, model comparison, diagnostics, and reproducible research-grade outputs without relying on opaque legacy package behavior.

### PRD-PER-002 Assessment engineer

Needs versioned assessment/rubric/scoring contracts, calibrated item/rater data, linking, CAT/ATA, DIF/invariance, and governed release evidence.

### PRD-PER-003 AI evaluation engineer

Needs to treat LLM judges as fallible raters rather than truth, calibrate judge severity/bias/drift, build evidence-grounded rubrics/items, and compare RAG/LLM systems on measurement-aware scales.

### PRD-PER-004 Automated-scoring validation lead

Needs human/AI/external scorer observations, many-facet calibration, agreement beyond correlation, range-use evidence, fairness/DIF, adjudication routing, and audit reports.

### PRD-PER-005 Downstream product/service team

Needs stable, provider-neutral, content-addressed contracts and Rust-backed outputs that can be composed into hosted applications without importing product-specific persistence or UI assumptions.

## 4. Product principles

### PRD-PRN-001 Measurement before aggregation

Raw scores, RAGAS values, LLM judgments, or human ratings are observations, not truth. The system shall preserve the facets and conditions needed to model measurement error before deriving consequential summaries.

### PRD-PRN-002 Rust owns production psychometric arithmetic

Likelihoods, gradients, Hessians, optimization, information, psychometric scoring/ranking, and other production mathematical kernels are Rust-owned. Python may orchestrate, validate, marshal, report, and retain explicit reference implementations for parity testing where governed.

### PRD-PRN-003 Correlation is not accuracy

Validation shall not treat correlation alone as proof of parameter recovery, agreement, calibration, fairness, or validity. Where true parameters are known, bias, MAE/RMSE, interval/SE coverage, convergence, response/information recovery, and backend parity are first-class evidence.

### PRD-PRN-004 Hierarchy and time are first-class

Scientifically relevant designs shall support or explicitly model multilevel, cross-classified, multiple-membership, testlet/local-dependence, repeated-measurement, temporal, and drift structure rather than flattening observations into an atomistic single level.

### PRD-PRN-005 Fail closed on unidentified interpretation

Unknown model relations, disconnected designs, incomplete provenance, unsupported contract major versions, non-finite results, scoreability failures, or missing required evidence shall not silently produce a preferred model, operational score, or release-ready result.

### PRD-PRN-006 Content-addressed reproducibility

Published/reusable measurement artifacts shall be immutable or superseded by new versions, with deterministic fingerprints for the exact construct/rubric/task/model/provenance content relevant to interpretation.

### PRD-PRN-007 Modular MSA compatibility without hidden coupling

The package shall expose stable versioned interfaces usable independently and by CWL services. It shall not require another service's database, ORM, HTTP type, UI component, deployment manifest, or ambient credential.

## 5. Current product capabilities

The following are implemented on protected `main` as of this baseline unless explicitly marked otherwise:

- MLS2PLM-family binary simulation and point estimation, including `MIRT`, `MLSRM`, `MLS2PLM`, `ULSRM`, and `ULS2PLM` constraints.
- Rust-backed likelihood/gradient/distance kernels through PyO3/maturin, with
  an explicit NumPy reference/parity path and parity tests; `auto` fails closed without the compiled Rust core.
- Missing-response handling, optimization, recovery, fit and dimensionality diagnostics.
- Fixed-item calibration/linking, CAT item-information selection, ATA form assembly.
- Response-process diagnostics, model-fit summaries, multigroup/multilevel-context summaries exposed by current APIs.
- Rubric-centered schemas and deterministic bounded item-blueprint/generation-contract compilation.
- Governed assessment/scoring contracts and provenance-aware automated essay and enterprise-issue adapters added through the v0.7.0-era scoring work.
- Rust-backed criterion many-facet calibration/reporting paths for governed scoring workflows.
- Standalone accessible HTML audit/report artifacts.
- Release, benchmark, procurement, buyer-packet, PR-queue and provenance evidence builders.

Open PRs and issues may contain additional capabilities. They are **not** considered accepted product behavior until protected integration.

## 6. Functional requirements

### 6.1 Canonical measurement contracts

**PRD-FR-001** The package shall own one canonical `AssessmentSpec` family and one canonical `RubricSpecification` family for reusable assessment domains. Duplicate parallel schemas are prohibited.

**PRD-FR-002** Assessment/rubric/scoring contracts shall carry deterministic content identity, schema/version identity, construct identity, scoring/calibration/validation policy references, and bounded metadata.

**PRD-FR-003** Scoring observations shall distinguish `scored`, `abstained`, `failed`, and `excluded` semantics; terminal states shall not be coerced to a low score.

**PRD-FR-004** Rater/engine identity, task identity and task revision, response identity/revision, assessment/rubric identity, and criterion identity shall remain separately auditable.

### 6.2 Rubric-to-item lifecycle

**PRD-FR-010** The package shall support the lifecycle:

`Rubric -> Blueprint -> Generation Contract -> Candidate -> Screening -> Pilot -> Calibration -> Approved Item Bank -> Monitoring -> Revision/Retirement`.

**PRD-FR-011** Benchmark generation shall support candidate-blind evidence-grounded criteria. Candidate-aware criterion discovery, when implemented, shall be isolated through cross-fitting or a separate training/diagnostic bank.

**PRD-FR-012** Canonical criteria shall support atomic, evidence-grounded judgments where appropriate; holistic score descriptions may be compiled compatibility views rather than the sole source of truth.

**PRD-FR-013** Generated provider output shall be treated as untrusted and shall be bounded, closed-schema, replay-resistant, provenance-bound, and checked for duplicate keys, non-finite numbers, answer-key integrity, evidence-span integrity, and response-format consistency before psychometric use.

**PRD-FR-014** Semantic screening shall be able to represent answerability, construct alignment, ambiguity, distractor quality, redundancy, leakage, evidence entailment/support, and content/bias review without conflating them with structural JSON validity.

### 6.3 Automated scoring

**PRD-FR-020** Human, LLM, external-model, deterministic, and future scorer implementations shall map to a shared scoring-engine/rater observation contract.

**PRD-FR-021** The automated-scoring validation path shall support ordinal many-facet calibration, rater severity, criterion-specific bias, range-use diagnostics, drift evidence, agreement, DIF/fairness evidence, and human-review/adjudication routing.

**PRD-FR-022** Human ratings are measurements with error and shall not automatically be treated as error-free true scores.

**PRD-FR-023** Generated feedback is not required for the measurement core. When a downstream system adds feedback, feedback must not silently alter the numerical score or provenance-bound psychometric evidence.

### 6.4 Reference-free RAG and LLM-as-a-Judge measurement

**PRD-FR-030** The product shall distinguish groundedness/faithfulness, world correctness, retrieval relevance/coverage, answer utility/completeness, robustness, abstention/calibration, and citation/evidence attribution where the evidence regime permits them.

**PRD-FR-031** LLM judges shall be representable as rater facets with model/provider/prompt/occasion/version provenance.

**PRD-FR-032** Reference-free evaluation shall not claim world correctness or absolute recall when the available evidence supports only context-grounded or pooled-corpus claims.

**PRD-FR-033** Query-derived probes from the same question/response shall be able to retain query/testlet identity to avoid pseudo-replication.

### 6.5 Measurement models and model selection

**PRD-FR-040** The model portfolio shall distinguish unidimensional, correlated multidimensional, bifactor, higher-order, testlet, two-tier, multifaceted, and latent-space structures by actual parameter constraints rather than names.

**PRD-FR-041** Model relation shall be classified as regular nested, boundary/singular nested, nonlinear-constraint nested, strictly non-nested, overlapping/indistinguishable, or unknown as evidence supports.

**PRD-FR-042** Non-nested preference shall require formal distinguishability evidence before selection. Boundary models shall use boundary-aware/bootstrap procedures when ordinary chi-square likelihood-ratio theory is invalid.

**PRD-FR-043** Model selection shall combine relation-appropriate inferential comparison with held-out cluster-aware predictive evidence, residual dependence, scoreability, DIF/invariance, stability, and true-structure recovery.

**PRD-FR-044** Bifactor model fit shall not imply general or specific-score interpretability. Scoreability evidence shall be reported separately.

**PRD-FR-045** Reusable finite-population proportion sampling designs shall
expose a versioned Rust-owned sample-size, FPC, and stratified-allocation
artifact. Prior/pilot proportions, confidence, precision, costs, design effects,
and response assumptions shall never be invented or hidden in Python.

### 6.6 Factor retention and rotation

**PRD-FR-050** Factor retention shall be a separate decision from structural model selection.

**PRD-FR-051** Exploratory rotation shall not expose a universal-best criterion claim. Rotation solutions shall retain criterion, optimizer, start/basin, convergence/stationarity, sign/permutation alignment, and stability evidence.

**PRD-FR-052** Criterion selection shall compare candidates using criterion-neutral recovery/stability/theory evidence rather than raw objective values from incomparable criteria.

### 6.7 Multilevel, multiple-membership, and longitudinal measurement

**PRD-FR-060** Reusable contracts shall represent explicit context dimensions, context identities, membership weights, repeated occasions, and temporal state specifications without inferring random-effect families from labels.

**PRD-FR-061** Multiple-membership weights and temporal ordering shall be provenance-bound. Elapsed-time effects shall not be claimed unless the fitted model actually parameterizes elapsed-time transitions. OLS may use exact day-scaled offsets as regression covariates, while only the joint MAP hierarchical CT-AR Rasch slice parameterizes elapsed-day transitions; the caller-supplied discrete AR layer uses sequence gaps.

**PRD-FR-062** Future Rust estimators for these contracts shall establish identification and true-parameter recovery before release as production estimators.

### 6.8 Reporting and evidence

**PRD-FR-070** Reports shall separate exact machine-readable values from human-readable summaries and preserve provenance needed to reconstruct the analysis.

**PRD-FR-071** HTML outputs shall be script-free by default where feasible, use restrictive content-security policy where relevant, and meet the repository's accessibility contracts.

**PRD-FR-072** Reliability, validity, fairness, fit, convergence, and deployment readiness shall remain distinct concepts in report language.

### 6.9 Lifecycle and release

**PRD-FR-080** Measurement/rubric/item/model artifacts shall have explicit lifecycle states where lifecycle management is exposed, e.g. `draft -> audited/screened -> pilot -> calibrated -> approved -> active -> suspended/quarantined -> retired`.

**PRD-FR-081** A release shall be created only from an exact integrated protected head with required CI, security, coverage, packaging, provenance/SBOM, reproducibility, compatibility, review, rollback/migration, and release-acceptance evidence.

**PRD-FR-082** Release notes and `CHANGELOG.md` shall match the released artifact and authoritative changelog-fragment workflow.

## 7. Quality requirements

The quality model follows the concerns of ISO/IEC 25010:2023 while adding psychometric/scientific evidence requirements.

### PRD-NFR-001 Functional suitability

Public contracts and numerical outputs shall match documented semantics; unsupported interpretations fail closed.

### PRD-NFR-002 Performance efficiency

Computationally material numerical kernels shall use Rust and low-context-switch CPU parallelism; GPU paths shall be added when beneficial and must have parity evidence. Resource use shall be explicitly bounded for caller-controlled dimensions and expensive fallback workspaces.

### PRD-NFR-003 Compatibility/interoperability

The wheel/package shall work independently. Cross-service composition shall use versioned contracts/artifacts rather than database coupling.

### PRD-NFR-004 Usability/accessibility

CLI, Python API, documentation, and standalone reports shall expose understandable errors and evidence. HTML/report surfaces target WCAG 2.2 AA-relevant semantics without claiming full conformance absent audit.

### PRD-NFR-005 Reliability

Long-running studies/subprocesses shall have operation-appropriate deadlines, fail-closed timeout evidence, deterministic seeds where relevant, idempotent/reproducible artifact generation, and no stale evidence reuse.

### PRD-NFR-006 Security

Untrusted data/provider output shall be bounded and validated. CI uses least privilege, immutable action pins where practical, supply-chain/security scanning, and no secrets in evidence artifacts. Security failures shall not be converted to success merely to unblock a merge.

### PRD-NFR-007 Maintainability

Public Python/Rust docs shall be beginner-readable. Owned production code targets exact 100% statement/branch coverage plus line/function coverage where tooling exposes it. Architecture/ADR/traceability documentation shall change with governing contracts.

### PRD-NFR-008 Scientific validity

Parameter recovery, uncertainty coverage, model identification, model relation, scoreability, DIF/invariance, and local-dependence assumptions shall be tested and reported within the scope supported by evidence.

## 8. Data and privacy requirements

`fast-mlsirm` is not the system of record for participant identity or hosted operational data.

- Public/reusable artifacts should use opaque nonnumeric identifiers where durable identity is required.
- Provider text, response text, source content, and PII shall not be duplicated into audit artifacts unless explicitly required by a reusable contract and appropriately bounded.
- Prefer purpose limitation, authorization, minimization, encryption by the host, restricted linkage, hashes/fingerprints, and retention control rather than blanket PII masking that destroys measurement utility.
- Hosted storage, data residency, DSAR, consent, and tenant isolation are downstream product responsibilities unless a reusable library primitive explicitly owns a local artifact format.

## 9. Non-goals

The following are not product responsibilities unless a future approved ADR changes the boundary:

- hosted participant/session/consent/result APIs;
- hosted tenant databases or migrations;
- SSO/SCIM/identity credentials;
- hosted UI/workbench deployment;
- clinical diagnosis or treatment recommendation;
- employment/admission/credit/insurance/legal automated decision authority;
- unconditional claims that one factor rotation, model, LLM judge, or rubric is universally optimal;
- treating `kaefa`, `aFIPC`, or `nonnest2` as runtime/build/release dependencies or sole scientific oracles;
- claiming SOC 2, CSAP, ISO, WCAG, or regulated-device certification solely from repository controls.

## 10. Product roadmap by bounded capability

Priority is evidence-driven; open PRs and protected-main state override this ordering when dependencies demand it.

1. Drain/merge current PR queue safely and remove obsolete duplicates.
2. Complete canonical documentation/ADR/traceability baseline.
3. Complete multilevel/contextual/longitudinal reusable contracts and move estimators to Rust only after recovery design is accepted.
4. Complete rubric-generation trust boundary and semantic screening.
5. Complete artificial-crowd calibration and governed item-bank lifecycle.
6. Complete automated-scoring range/discrimination/drift/fairness evidence.
7. Complete relation-safe factor retention and structural model selection.
8. Complete bifactor scoreability and adaptive rotation product APIs with primary-source traceability.
9. Add buyer-facing workbench only when public domain contracts stabilize; host UI remains downstream.
10. Mature release/provenance/performance/security evidence for enterprise procurement.

## 11. Acceptance and traceability

Each material requirement must map to at least one of:

- a public API or schema;
- a Rust/PyO3/Python implementation module;
- a realistic test/recovery study;
- an ADR explaining the decision and alternatives;
- a release/readiness gate.

The mapping is maintained under `docs/traceability/`.

## References

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

International Organization for Standardization. (2023). *ISO/IEC 25010:2023 Systems and software engineering—Systems and software Quality Requirements and Evaluation (SQuaRE)—Product quality model*.

International Organization for Standardization. (2023). *ISO/IEC 42001:2023 Information technology—Artificial intelligence—Management system*.

International Organization for Standardization, International Electrotechnical Commission, & Institute of Electrical and Electronics Engineers. (2018). *ISO/IEC/IEEE 29148:2018 Systems and software engineering—Life cycle processes—Requirements engineering*.

World Wide Web Consortium. (2024). *Web Content Accessibility Guidelines (WCAG) 2.2*.
