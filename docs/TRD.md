# fast-mlsirm Technical Requirements Document

**Status:** Authoritative technical requirements baseline  
**Snapshot basis:** protected `main` `4d910ed650f384ff882c8b5fba6a8b08fd532236`

## 1. Technical objective

Provide a Rust-first, Python-accessible, independently installable measurement toolkit whose public contracts can be reused by assessment, AI-evaluation, and research products without importing product-specific persistence, HTTP, identity, or UI concerns.

## 2. Repository topology

```text
python/fast_mlsirm/        Public Python API, contracts, validation, orchestration, reports
crates/mlsirm-core/        Numerical source of truth
crates/fast-mlsirm-py/     PyO3 bindings and marshaling
tests/                     Public-contract, parity, integration, security, report tests
fuzz/                      Atheris and Rust fuzzing
scripts/                   Release, governance, scientific-study, buyer-evidence tooling
docs/                      Authoritative specs, ADRs, doctoring, research, handoff evidence
```

## 3. Layering constraints

### 3.1 Allowed dependency direction

```text
Python public API/contracts
        ↓
Python validation/orchestration
        ↓
PyO3 binding
        ↓
Rust numerical core
        ↓
CPU/GPU implementation details
```

Provider adapters may be invoked by orchestration, but they cannot become numerical authorities. Downstream hosted products depend on `fast-mlsirm`; `fast-mlsirm` must not depend on downstream product runtime code.

### 3.2 Numerical authority

- Rust is authoritative for new production mathematical/psychometric arithmetic.
- NumPy reference/fallback code may exist only for bounded compatibility, reference comparison, or degraded operation already supported by the package.
- Equation-changing work must update simulation, objective/likelihood, gradients, fitters, tests, documentation, and Rust/Python parity as one model-design change.
- Float precision is method-specific; f32 GPU execution cannot silently replace an f64 scientific contract without parity/error evidence.

## 4. Public contract architecture

### 4.1 Assessment and scoring

Public reusable contracts should express, at minimum:

- assessment/construct identity and version;
- rubric identity/revision/fingerprint;
- response type and allowed score categories;
- scoring engine/rater identity and version;
- observation status (`scored`, `abstained`, `failed`, `excluded`, etc. where defined);
- criterion/task/item identity;
- rater/judge/model/prompt/occasion identity;
- context/testlet/temporal identifiers needed by later estimators;
- provenance/evidence references without requiring raw sensitive payload storage.

### 4.2 Rubric and item generation

A generated-item request must bind the exact rubric, blueprint, generation contract, source/evidence digests where applicable, provider/model identity, and bounded response contract. Provider output is untrusted and must be parsed with closed schemas.

Required parser properties include:

- maximum payload sizes before full materialization where practical;
- duplicate JSON key rejection;
- `NaN`/infinity rejection;
- top-level type enforcement;
- unknown/missing field rejection;
- response-format-specific typed answer keys;
- option/reference integrity;
- score-order preservation;
- evidence/source identity and span validation where the contract claims it;
- replay/forgery protection through recomputed identities, not merely echoed strings.

### 4.3 Immutable lifecycle artifacts

Operational/published rubric, assessment, bank, calibration, and result-like reusable artifacts must not be mutated in place after publication. Corrections create a new version/superseding artifact and preserve linkability.

## 5. Measurement model requirements

### 5.1 Model families

The library may expose multiple model families, but every implementation must define:

- response distribution/link;
- parameter shapes;
- identification/scaling constraints;
- missingness semantics;
- likelihood/objective;
- gradient/optimizer or estimation algorithm;
- diagnostic applicability;
- simulation/recovery contract;
- backend/device behavior.

### 5.2 Structural selection

The model-comparison API must represent relation explicitly, including the practical categories:

- regular nested;
- boundary nested;
- nonlinear-constraint nested;
- strictly non-nested;
- overlapping;
- indistinguishable;
- unknown.

A model is not selected merely from AIC/BIC or a raw Vuong statistic. Selection evidence should combine relation-appropriate tests, cluster-aware held-out marginal likelihood, residual dependence, scoreability, invariance/DIF, and simulation recovery.

### 5.3 Bifactor and higher-order

Bifactor scoreability diagnostics are distinct from model selection. General-factor indices must fail closed when the declared general-factor assumptions do not hold. Higher-order vs bifactor relations depend on actual loading/proportionality constraints rather than names.

### 5.4 Testlet and two-tier

Known shared-stimulus/query groups are testlets or secondary dimensions, not automatically substantive traits. Two-tier structures are appropriate only when multiple primary traits and secondary local-dependence/method dimensions are both identified.

### 5.5 Latent-space terms

Latent-space interaction is a residual interaction model, not a substitute for missing substantive dimensions, rater facets, or testlet structure. Add it only after the main factor/facet structure has been tested and residual evidence supports it. Coordinates require appropriate alignment; distance-based recovery should use invariant quantities where raw coordinates are not identified.

## 6. Multilevel, multiple-membership, and time

### 6.1 Context contracts

Context membership must be dimension-qualified. Weighted multiple memberships must preserve caller-supplied weights and validate the declared normalization rule rather than silently renormalizing.

### 6.2 Design identification

Future numerical estimators must detect disconnected/confounded designs when the requested random effects or facets cannot be separated. A structurally valid schema is not proof that the requested model is estimable.

### 6.3 Temporal semantics

At minimum distinguish:

- observed/occasion ordering;
- rater/model/prompt version drift;
- discrete-step transitions;
- actual elapsed-time models.

Timestamps used only for ordering/provenance must not be claimed as continuous-time likelihood inputs.

## 7. Rater, LLM judge, and automated-scoring requirements

### 7.1 Observation model

Human and machine judgments enter as observations from raters. The package may estimate or diagnose, where supported:

- global severity;
- criterion/dimension-specific severity;
- discrimination/consistency;
- range restriction/category use;
- drift/occasion effects;
- rater × criterion or rater × group interactions.

### 7.2 Validation evidence

No single metric is a release gate for all automated scoring. Depending on score type and intended use, support:

- QWK and exact/adjacent agreement;
- MAE/RMSE/bias if a defensible criterion score exists;
- calibration intercept/slope or probability calibration where relevant;
- rater-facet estimates and fit;
- subgroup error and DIF/invariance;
- leave-task/domain/rater-family validation;
- human-human context rather than treating one human score as true by definition.

## 8. Reference-free evaluation requirements

Reference-free evaluation must record its evidence regime. A criterion generated from the target retriever's own context cannot be interpreted as absolute retrieval recall. Contracts should distinguish `prompt_only`, `retrieved_context`, `pooled_corpus`, `authoritative_corpus`, or `human_anchor`-like regimes where such modes exist.

Candidate-aware criterion discovery used for diagnosis/training must not evaluate the same candidate without an independent split/cross-fit unless explicitly labeled exploratory and non-comparative.

## 9. Rotation and factor-retention requirements

### 9.1 Factor retention

Candidate dimensionality can be informed by multiple evidence sources; no single eigenvalue or fit cutoff is universally authoritative. For MIRT/IRT production decisions, combine appropriate retention heuristics with marginal-likelihood/CV/residual/recovery evidence.

### 9.2 Rotation

Rotation criterion implementations must share a clear Rust criterion/gradient contract where possible. Optimization shall use deterministic seeds for reproducible multi-start behavior when promised. A finite multi-start search returns a **best observed solution**, not a proof of global optimality.

Different criterion objective values are not directly comparable across criterion families. Criterion selection uses criterion-neutral evidence such as recovery, stability, loading complexity, degeneracy, theory/target alignment, or split-sample/bootstrap reproducibility.

## 10. Recovery and scientific CI

For simulated true parameters or structures, report method-appropriate:

- bias;
- MAE/RMSE;
- standard-error bias;
- interval coverage;
- convergence/failure rate;
- item/person probability or information recovery;
- factor/loading recovery after sign/permutation/rotation alignment;
- latent-distance recovery using invariant distances when appropriate;
- CPU/GPU parity.

Correlation is auxiliary rank/association evidence only.

Heavy Monte Carlo studies may be scheduled/manual/release workflows, but PR smoke must still exercise the same code path with bounded workloads. Do not tune pass thresholds post hoc to a realized random seed.

## 11. Concurrency and resource safety

- Bound untrusted dimensions, arrays, file sizes, provider responses, and subprocess deadlines.
- Avoid allocating an unnecessary `N × J × D`-like tensor where a bounded blocked/reused-buffer algorithm exists.
- CPU parallel work should minimize context switching and retain deterministic aggregation where required.
- GPU jobs must prove a real adapter/path executed when the gate claims GPU evidence; silent skip is not success.
- Scientific-study subprocesses must terminate descendants safely on timeout and preserve machine-readable failure classification.

## 12. Error contract

Public exceptions/results should be typed or structured where callers must branch on the condition. Python adapters must not parse unstable Rust error text to infer scientific state. Error messages must not echo credentials, full uncontrolled provider payloads, or sensitive child-process output when a stable code/path is sufficient.

## 13. Security and supply-chain requirements

- Central Security Scan, SAST, dependency review, and fuzzing remain fail-closed where configured.
- Immutable action pins are preferred for privileged automation.
- Autonomous development uses OpenCode Agent with `NVIDIA_NIM_API_KEY` when model-backed development is genuinely required; do not use `COPILOT_GITHUB_TOKEN`.
- Review-agent identities/credentials remain independent from writer/merge authority.
- Temporary self-modifying or write-capable repair workflows are not valid final-state architecture.
- Release evidence should include package/build provenance and SBOM/signing where repository release tooling supports it.

## 14. Privacy requirements

The reusable library should avoid storing raw PII when identifiers/digests/provenance are sufficient, but it must not destroy measurement utility with blanket masking. Identity/tenant linkage, encryption keys, data residency, retention, deletion/export, consent, and privileged-access workflows belong to the hosted product/service that owns those data.

## 15. API and compatibility requirements

- Public Python APIs are documented and typed where practical.
- Rust/PyO3 bindings must remain installable by the package's supported build path.
- New secondary bindings/export surfaces must compose with existing bindings and package-root exports.
- Breaking contract changes require explicit versioning/migration guidance.
- Existing formula semantics cannot be altered through a performance-only PR.

## 16. Test requirements

Each source change should select tests from the following set as applicable:

- deterministic unit tests;
- property-based tests;
- Rust/Python delegation or parity tests;
- known numerical oracle tests;
- true-parameter recovery simulation;
- malformed/untrusted-input/security regressions;
- fuzzing;
- concurrency/resource-bound tests;
- package/wheel/reinstall tests;
- accessibility/report serialization tests;
- migration/version replay tests;
- end-to-end contract pipeline tests.

Tests must fail for the intended production reason before a defect fix where test-first repair is feasible.

## 17. Documentation and traceability requirements

Material changes must update, as applicable:

- root `ARCHITECTURE.md`;
- `docs/PRD.md`;
- this TRD;
- ADRs;
- UML/ERD;
- doctoring/primary references;
- public docstrings/rustdoc;
- `CHANGELOG.md` via the repository's authoritative fragment renderer;
- `docs/traceability.md` when status/ownership changes materially.

A narrow historical MVP summary is not authoritative once contradicted by protected-main functionality.
