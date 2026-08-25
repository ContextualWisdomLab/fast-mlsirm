# fast-mlsirm Technical Requirements Document

Status: **Authoritative technical requirements baseline**  
Repository: `ContextualWisdomLab/fast-mlsirm`  
Last reviewed: 2026-08-09

## 1. Purpose

This TRD turns the product requirements in [`PRD.md`](PRD.md) into implementation, interface, evidence, and release constraints for the reusable `fast-mlsirm` package.

The document follows ISO/IEC/IEEE 29148 requirements-engineering principles and the architecture concerns of ISO/IEC/IEEE 42010. It does not replace method-specific design documents or primary methodological literature.

## 2. System boundary

### TRD-BOUND-001 Owned by fast-mlsirm

- versioned domain-neutral `AssessmentSpec`, `RubricSpecification`, scoring/observation and calibration contracts;
- psychometric model configuration and result contracts;
- simulation, recovery, fit and model-selection evidence;
- CTT/IRT/MIRT, MLSIRM/MLS2PLM, testlet, many-facet and related reusable numerical capabilities;
- factor retention, bifactor scoreability, rotation and relation-safe model comparison where implemented;
- DIF/invariance/fairness, linking/equating, CAT/ATA and G-theory primitives;
- automated-scoring and LLM-judge validation primitives;
- governed rubric/item-bank primitives and reusable reports;
- package/release/provenance evidence.

### TRD-BOUND-002 Not owned by fast-mlsirm

- hosted HTTP/admin APIs;
- participant/session/consent/result lifecycle;
- hosted product persistence/migrations and multi-tenant database administration;
- end-user authentication, SSO/SCIM/passkeys;
- UI deployment and product navigation;
- provider credential stores;
- operational research-release catalogs.

These belong to downstream bounded contexts such as `ContextualWisdomLab/psychometrics-commons`, Keyverse, semantic-data-portal, contextual-orchestrator, or another explicitly versioned service.

## 3. Repository architecture

```text
python/fast_mlsirm/       Public Python API, validation/orchestration, reports,
                          governed reference/parity paths
crates/mlsirm-core/       Rust psychometric/numerical source of truth
crates/fast-mlsirm-py/    PyO3 bindings and Python transport
scripts/                  Release, evidence, governance, study runners
fuzz/                     Python/Rust fuzz targets and corpora
tests/                    Public contract, regression, delegation, parity tests
docs/                     PRD/TRD, method docs, ADRs, doctoring, diagrams,
                          traceability, release evidence guidance
```

## 4. Technical requirements

### 4.1 Numerical ownership and precision

**TRD-NUM-001** Production likelihoods, gradients, Hessians/information matrices, psychometric scoring/ranking, optimization, item/factor information, and other mathematically material kernels shall be implemented in Rust before being considered production-owned.

**TRD-NUM-002** Python reference implementations may exist for numerical parity, diagnostics, controlled fallback, or research transparency, but shall not silently diverge into a second production formula.

**TRD-NUM-003** Every formula-contract change shall update parameterization, likelihood, analytic derivatives, simulation, recovery, Python/Rust parity, public documentation, and method citations as one coherent model-design change.

**TRD-NUM-004** The existing simple-structure MLS2PLM specialization is preserved unless a dedicated full-vector discrimination model path is introduced. Local algebra/performance changes shall not silently reinterpret the model.

**TRD-NUM-005** Caller-controlled dimensions, array sizes, item counts, bootstrap counts, JSON sizes, subprocess workloads, and fallback workspaces shall be bounded before allocation or expensive execution.

**TRD-NUM-006** Non-finite caller input and non-finite intermediate/result states shall fail with bounded, non-secret-bearing errors when the mathematical contract requires finiteness.

### 4.2 CPU and GPU execution

**TRD-PERF-001** Computationally material Rust workloads shall use coarse-grained or otherwise low-context-switch CPU parallelism when concurrency improves throughput without violating determinism or memory ceilings.

**TRD-PERF-002** GPU is a Rust device path, not an independent public model/backend. GPU implementations shall have CPU/Rust parity evidence at the level appropriate to the algorithm, including invariant-aware comparisons for non-identifiable coordinates.

**TRD-PERF-003** A GPU fallback shall not be presented as a GPU success. Tests requiring GPU execution shall prove a real adapter/kernel path ran and did not skip when the acceptance contract says GPU is required.

**TRD-PERF-004** f32 GPU arithmetic shall not be treated as evidence-equivalent to f64 CPU arithmetic without explicit error tolerances and recovery/parity studies.

### 4.3 PyO3/public API

**TRD-API-001** PyO3 bindings shall expose typed, bounded domain results rather than unstable error-string parsing.

**TRD-API-002** Secondary extension initializers or feature bindings shall be registered through one canonical Rust/Python export structure so independent feature PRs can coexist without overwriting module initialization.

**TRD-API-003** Public Python APIs shall validate shape, type, bounds, identifiers, schema versions and obvious semantic invariants before delegating numerical work to Rust, while avoiding duplicate numerical computation.

**TRD-API-004** Public error codes/paths are part of the contract. Caller-controlled text and provider exceptions shall not be echoed into durable error evidence unless explicitly safe.

**TRD-API-005** Public durable identifiers shall be descriptive opaque strings where identity must survive serialization; numeric database-style IDs shall not be introduced into reusable public contracts solely for implementation convenience.

### 4.4 Canonical serialization and provenance

**TRD-PROV-001** Reusable immutable artifacts shall have deterministic canonical serialization and full cryptographic fingerprinting where content identity matters.

**TRD-PROV-002** Human-readable handles may be shortened representations but shall never substitute for the authoritative full fingerprint in replay or integrity decisions.

**TRD-PROV-003** Schema version and semantic/domain revision shall remain separate dimensions. Changing a rubric/task/calibration revision shall change content identity even if the wire schema is unchanged.

**TRD-PROV-004** Aggregate objects shall replay/verify package-owned child artifacts at trust boundaries rather than trusting cached display identifiers.

**TRD-PROV-005** Exact task revisions, response-content revisions, assessment/rubric identities, engine/rater identities and source/evidence revisions shall be preserved separately where their conflation changes the interpretation.

### 4.5 Rubric and generated-item trust boundary

**TRD-RUB-001** `RubricSpecification` is the canonical rubric source; scoring or domain adapters shall reference it rather than define competing rubric schemas.

**TRD-RUB-002** Blueprint compilation shall be deterministic, bounded and content-addressed.

**TRD-RUB-003** Generation contracts shall use closed response-format-specific schemas with bounded text/collections and typed answer-key semantics.

**TRD-RUB-004** Provider output shall be parsed as untrusted input. The parser shall reject duplicate object keys, `NaN`/infinities, oversized payloads, unknown/missing fields, provenance mismatches, invalid score order/coverage, option/answer-key inconsistencies, undeclared source IDs, and invalid evidence spans.

**TRD-RUB-005** Structural schema conformance shall not imply psychometric/content acceptance. Separate screening shall record answerability, construct alignment, evidence support, ambiguity, distractor quality, redundancy, leakage and bias/content-review results.

**TRD-RUB-006** Candidate-aware rubric/criterion discovery shall not evaluate the same candidate set used to discover criteria unless cross-fitting or another approved anti-leakage design is used.

### 4.6 Automated scoring and rater measurement

**TRD-SCR-001** Human and automated scoring shall project into shared rater/engine observation contracts.

**TRD-SCR-002** `scored`, `abstained`, `failed`, and `excluded` shall remain distinguishable through calibration and audit boundaries.

**TRD-SCR-003** Many-facet calibration shall bind person/respondent, task/task-revision, and rater/engine axes without substituting response IDs for stable respondent/system identities when the estimator interprets a person effect.

**TRD-SCR-004** Designs shall explicitly validate respondent-task and task-rater connectedness where those effects must be separately identified.

**TRD-SCR-005** Automated-scoring validation shall provide agreement evidence appropriate to ordinal ratings, plus descriptive bias/range-use evidence. Pearson/Spearman correlation may be supplementary only.

**TRD-SCR-006** Future generalized rater discrimination/range-restriction/drift estimators require separate identification and true-parameter recovery contracts before release.

### 4.7 Reference-free RAG/LLM judge measurement

**TRD-RAG-001** Evaluation records shall carry system-run, query/testlet, judge family/model/version, prompt/occasion, evidence regime, and criterion identities when available.

**TRD-RAG-002** Groundedness shall not be relabeled world correctness. Pooled-corpus coverage shall not be relabeled absolute corpus recall. Claims are constrained by the evidence universe.

**TRD-RAG-003** LLM judges are raters; judge severity/bias/discrimination/drift and family dependence are measurement concerns, not assumed truth.

**TRD-RAG-004** Candidate-independent perturbation anchors shall be supported for evaluator validation where feasible, such as unsupported-claim insertion, evidence deletion, distractor/citation swaps, and meaning-preserving paraphrases.

### 4.8 Factor/model structure and relation-safe comparison

**TRD-MOD-001** Factor count/retention and structural model choice are separate workflows.

**TRD-MOD-002** The system shall not infer nestedness solely from model names. Actual loading, variance, proportionality, and boundary constraints determine relation class.

**TRD-MOD-003** Regular nested comparisons may use appropriate LR tests; boundary/singular comparisons require boundary-aware or parametric-bootstrap evidence; strictly non-nested/overlapping comparisons require formal distinguishability before a selection statistic can produce preference.

**TRD-MOD-004** Same-question/probe or same-system repeated observations shall use cluster-aware aggregation/resampling rather than treating all response cells as independent.

**TRD-MOD-005** Final model selection shall include held-out prediction at operationally relevant cluster levels and recovery/model-selection simulations under realistic generating structures.

### 4.9 Bifactor scoreability

**TRD-BIF-001** A declared general factor must satisfy the documented applicability contract before general-factor ECV/item-ECV/scoreability quantities are returned.

**TRD-BIF-002** Standardized loadings and uniquenesses shall satisfy the documented variance identity within a bounded numerical tolerance.

**TRD-BIF-003** PUC shall be returned only for structures for which its interpretation is defined by the implemented contract.

**TRD-BIF-004** Omega derived from latent-response standardization shall be labeled as latent-response reliability and shall not be presented as categorical observed-score reliability.

**TRD-BIF-005** Intermediate sums and denominators shall be checked for finite overflow/underflow before reliability indices are returned.

### 4.10 Rotation

**TRD-ROT-001** Rotation criteria implement a shared Rust criterion interface separated from the optimizer.

**TRD-ROT-002** Orthogonal and oblique optimization shall report convergence/stationarity and reject singular/degenerate transformations.

**TRD-ROT-003** Deterministic multi-start shall report best observed solution and basin/start evidence; it shall not claim mathematical global optimality.

**TRD-ROT-004** Sign/permutation alignment shall preserve semantically privileged columns when the criterion requires them, such as a designated bifactor general factor or target labels.

**TRD-ROT-005** Selection across criteria shall use common recovery/stability/theory/degeneracy evidence, not incomparable raw criterion objective values.

### 4.11 Multilevel/multiple-membership/temporal contracts

**TRD-MLT-001** Context membership shall include explicit context dimension and context identity.

**TRD-MLT-002** Multiple-membership weights shall be validated rather than silently renormalized when the public contract promises exact caller-supplied membership.

**TRD-MLT-003** Cross-classified designs shall maintain dimension-qualified identities.

**TRD-MLT-004** Temporal occasions shall retain ordering/time provenance and explicitly separate discrete occasion-step AR effects from continuous-time parameterizations. OLS may use exact day-scaled offsets as regression covariates; discrete AR uses sequence gaps. Only the hierarchical CT-AR Rasch slice estimates elapsed-day Ornstein–Uhlenbeck transitions.

**TRD-MLT-005** Numerical multilevel/longitudinal estimators shall remain proposed until Rust implementations pass identification and true-parameter recovery studies. The hierarchical CT-AR Rasch slice reports multi-seed state RMSE/coverage as joint MAP evidence, not as Fox–Glas Gibbs or estimated MMMC recovery.

**TRD-MLT-006** The TEPP topic-context influence boundary shall validate the
exact `tepp.topic_context_posterior.v1` schema in Rust, including complete
posterior draws, Event Lineage, historical event time, evidence provenance,
and source-derived BU/PU/team/person multiple memberships. A v1 artifact is a
full-data posterior only; it shall return
`CaseDeletionRefitEvidenceUnavailable` because it contains neither exact
case-deleted refits nor the per-case joint likelihood contributions needed for
reviewed importance reweighting. A future estimator must consume one of those
producer-owned evidence forms and pass deletion-effect recovery, interval
coverage, and CPU/GPU parity; no Python or heuristic fallback is permitted.

**TRD-MLT-007** The accepted deletion-refit successor contract shall bind an
independent full-fit anchor distribution basis and one actual `D \\ {i}` refit
per admitted document. Each refit shall prove identical prior/configuration,
snapshot, cutoff, and event clock; exact retained membership; incident Event
Lineage and membership deletion; unique bijective anchor alignment; complete
posterior draws; and CPU/GPU objective, parameter, and draw parity receipts.
Missing or tied alignment fails closed.

**TRD-MLT-008** Rust shall transform ALR draws to simplex probabilities and
compute the source-weighted context prevalence difference between the full fit
and refit separately for each deleted document, BU/PU/team/person context, and
topic. It shall preserve complete signed posterior draws and their mean and
variance, structurally
unavailable empty cells, and exact dense-rank ties. It shall not fuse levels,
contexts, or topics, infer weights, add Event Lineage weights, or emit a binary
importance decision.

### 4.12 Testing and scientific evidence

**TRD-TEST-001** Owned production Python statement/branch coverage target is 100%; public docstrings and Rust documentation shall be complete and beginner-readable.

**TRD-TEST-002** Tests shall use realistic measurement cases, not only type/shape smoke tests.

**TRD-TEST-003** True-parameter simulations shall report bias, MAE/RMSE, SE/interval coverage, convergence, and relevant model-specific recovery. Scale/rotation/linking alignment occurs before parameter error calculation.

**TRD-TEST-004** Same numerical contract implemented on CPU/GPU/Python-reference paths shall have parity tests using invariants appropriate to identification (e.g. distances/Procrustes rather than raw coordinates where necessary).

**TRD-TEST-005** Heavy literature/recovery studies shall run on scheduled/manual/release paths when too expensive for every PR; bounded smoke/recovery sentinels remain on PRs. Scientific gates are not deleted to reduce CI latency.

**TRD-TEST-006** Monte Carlo acceptance bounds shall be specified prospectively from scientific/statistical reasoning and shall not be fitted to one observed random seed's result.

### 4.13 LLM/provider tests and automation

**TRD-LLM-001** Model-backed GitHub tests/actions use `NVIDIA_NIM_API_KEY` through GitHub Secrets when a model call is materially necessary. `COPILOT_GITHUB_TOKEN` is prohibited for autonomous development scheduling.

**TRD-LLM-002** `contextual-orchestrator` is preferred as a provider-neutral orchestration integration when suitable, but remains a read-only external dependency while its own writer loop is active.

**TRD-LLM-003** Deterministic gates must remain executable without model credentials when the feature being validated does not require a model call.

**TRD-LLM-004** Deep orchestration must be justified with comparable-budget evidence versus simpler routing, including task decomposition, recursion/workflow depth, role-specific reasoning effort, and ablations where relevant.

### 4.14 Security and supply chain

**TRD-SEC-001** PR checks include repository policy and central security scanning; known HIGH/CRITICAL dependency findings shall be remediated or narrowly documented as verified false positives rather than ignored by weakening gates.

**TRD-SEC-002** GitHub Actions shall use least privilege and immutable action pins where practical. Write-capable self-modifying branch workflows are prohibited.

**TRD-SEC-003** Provider/source/response text and secrets shall not leak into exception messages, audit IDs, logs, test artifacts, or generated reports unless the contract explicitly permits the content.

**TRD-SEC-004** Persistence/PII controls that belong to hosted applications shall not be emulated in the core library. Reusable artifacts prefer fingerprints, opaque identities and data minimization.

### 4.15 Documentation and architecture governance

**TRD-DOC-001** Canonical documents are `docs/PRD.md`, `docs/TRD.md`, root `ARCHITECTURE.md`, `docs/adr/README.md`, the accepted/proposed ADR corpus, `docs/uml/`, `docs/erd/`, and `docs/traceability/`.

**TRD-DOC-002** Continuous commercial-hardening loops follow **ADR-0013** (continuous execution and documentation governance): feasibility-first prioritization, work-conserving progress when a step is non-actionable under current authority, and a single-writer rule on each exact branch head so parallel authority is prohibited.

**TRD-DOC-002** `docs/prd_trd_summary.md` is a historical summary and shall point to the canonical PRD/TRD rather than remain an independent authority.

**TRD-DOC-003** Material ownership/API/model/lifecycle/security/release changes require architecture/ADR impact review in the same change or an explicit no-impact assertion.

**TRD-DOC-004** Documentation structure shall be machine-checked for required canonical files, valid ADR statuses and core ownership-boundary consistency.

## 5. Release acceptance

A release candidate is accepted only when the exact integrated protected head provides evidence for:

1. required Python/Rust/PyO3 tests and 100% owned coverage policy;
2. package build and clean reinstall/import;
3. explicit Rust-primary backend assertion;
4. GPU-required tests when the released claim requires GPU;
5. fuzz/security/SAST/supply-chain gates;
6. realistic recovery or parity studies for changed psychometric kernels;
7. release artifact digests/SBOM/provenance and reproducibility evidence;
8. documentation/changelog/version consistency;
9. zero valid unresolved review/security findings;
10. repository approval/branch-protection policy;
11. migration/compatibility/rollback evidence for changed serialized contracts.

## 6. Standards and primary evidence baseline

- AERA, APA, & NCME (2014), *Standards for Educational and Psychological Testing*.
- ISO/IEC/IEEE 29148:2018, requirements engineering.
- ISO/IEC/IEEE 42010:2022, architecture description.
- ISO/IEC 25010:2023, product quality model.
- ISO/IEC 42001:2023, AI management-system controls where AI lifecycle governance is relevant.
- W3C WCAG 2.2, report/accessibility concerns.
- Method-specific psychometric primary literature recorded in `AGENTS.md`, doctoring records, and relevant ADRs.
