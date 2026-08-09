# fast-mlsirm Product Requirements Document

**Status:** Authoritative product requirements baseline  
**Snapshot basis:** protected `main` `4d910ed650f384ff882c8b5fba6a8b08fd532236`  
**Audience:** psychometricians, assessment engineers, AI-evaluation teams, researchers, downstream product teams, enterprise technical buyers.

## 1. Product vision

`fast-mlsirm` is a reusable measurement infrastructure package for turning complex human/AI judgments and assessment responses into scientifically defensible, auditable measurement evidence. Its differentiator is not one IRT estimator: it combines Rust-first psychometric computation with reusable assessment/rubric/scoring contracts, rater calibration, structural model selection, recovery evidence, generated-item governance, and modular downstream integration.

It remains a library/toolkit rather than a hosted product. The hosted end-user product is owned downstream by Psychometrics Commons.

## 2. Buyer problems

The product must solve the following recurring problems.

### P1. Raw scores hide measurement error

Organizations routinely average rubric scores, RAG metrics, human ratings, or LLM judge outputs although items differ in difficulty/discrimination and raters differ in severity, consistency, range use, and drift.

**Required outcome:** expose calibrated latent estimates and diagnostics rather than treating raw averages as ground truth.

### P2. One structural model is not universally correct

Unidimensional, correlated MIRT, bifactor, higher-order, testlet, two-tier, many-facet, multilevel, temporal, and latent-space models answer different scientific questions.

**Required outcome:** relation-safe model comparison, residual diagnostics, held-out evidence, scoreability evidence, and true-structure recovery must govern model choice.

### P3. LLM-as-a-Judge is often treated as an oracle

Reference-free RAG evaluation, automated essay scoring, and enterprise issue assessment frequently conflate judge outputs with truth.

**Required outcome:** represent AI judges as fallible raters with explicit model/version/prompt/occasion identity; preserve evidence, uncertainty, disagreement, DIF, and drift.

### P4. Evaluation items and rubrics are not governed assets

A measurement engine is incomplete if it assumes high-quality items already exist.

**Required outcome:** support a closed loop from versioned rubric/assessment contracts to blueprints, bounded generation contracts, untrusted candidate validation, pilot observations, psychometric calibration, governed item-bank lifecycle, and rubric revision.

### P5. Scientific evidence and production code drift apart

A statistically plausible implementation can still be unusable if the public API, Rust/Python parity, provenance, tests, or release evidence are incomplete.

**Required outcome:** scientific claims are tied to executable recovery/parity/fit tests and authoritative documentation.

## 3. Product scope

### 3.1 Core measurement

The package shall provide reusable implementations and contracts for:

- CTT and IRT/MIRT workflows used by the current package.
- MLSIRM/MLS2PLM and related simple-structure constraints.
- Binary and supported polytomous/ordinal response models.
- Missing-response handling with explicit semantics.
- Item/person/model fit diagnostics.
- DIF/invariance/fairness evidence.
- Fixed-item linking/equating and calibration diagnostics.
- CAT/ATA and item-information workflows where implemented.
- Generalizability-theory evidence where implemented.
- Many-facet/rater calibration.
- Bifactor scoreability and structural diagnostics.
- Factor retention, model comparison, and rotation evidence.
- True-parameter recovery and simulation.

### 3.2 Contextual, multiple-membership, and temporal measurement

The architecture shall support contracts and, as estimators are added, identified numerical models for:

- nested and cross-classified contexts;
- weighted multiple membership;
- testlets and shared-stimulus local dependence;
- respondent/system-run longitudinal occasions;
- rater/model/prompt drift;
- irregular time ordering where explicitly modeled;
- multilevel/multiple-membership structures that prevent atomistic interpretation errors.

A timestamp alone must never be represented as a validated continuous-time model.

### 3.3 Assessment, rubric, scoring, and observations

The package shall maintain one reusable canonical contract layer for:

- `AssessmentSpec` and construct definitions;
- `RubricSpecification` and immutable rubric revisions;
- scoring engine/observation contracts;
- human, LLM, and external scorer observations;
- evidence/provenance references;
- calibration/model-selection/reporting contracts.

Duplicate domain schemas must not proliferate across essay, RAG, enterprise-issue, or downstream hosted products.

### 3.4 Rubric-to-item lifecycle

The target buyer workflow is:

```text
Rubric / Assessment contract
→ Blueprint
→ Generation contract
→ Untrusted provider output
→ Structural/evidence/semantic screening
→ Artificial crowd or human/AI pilot
→ Rust calibration
→ Governed item bank
→ Monitoring / linking / retirement
→ Rubric revision
```

The benchmark path should prefer candidate-blind evidence-grounded criteria. Candidate-aware rubric discovery is allowed only with explicit discovery/scoring separation such as cross-fitting.

### 3.5 Automated scoring

The product shall support calibration and validation of automated scoring without assuming either AI or one human rater is perfect. Required evidence families include:

- exact/adjacent agreement and QWK for ordinal ratings;
- absolute error/calibration where a defensible criterion exists;
- many-facet rater severity and criterion-specific behavior;
- range-use/compression evidence;
- subgroup bias, DIF, and invariance;
- prompt/model/version drift;
- human-review/adjudication routing based on governed criteria;
- deterministic JSON/HTML evidence reports.

Correlation may be reported as descriptive association but is not sufficient evidence of absolute agreement, parameter recovery, fairness, or validity.

### 3.6 Reference-free RAG evaluation

The reusable measurement layer shall distinguish:

- groundedness/faithfulness to retrieved evidence;
- world correctness when an independent authority exists;
- retrieval relevance/precision and evidence sufficiency;
- answer relevance/utility;
- completeness or obligation coverage only under an explicit evidence universe;
- robustness to distractors/paraphrase/order;
- abstention/answerability;
- citation attribution.

RAGAS/LLM-judge values are observations, not latent truth.

### 3.7 Enterprise issue measurement

The package may provide reusable measurement contracts for structured issue evidence and rater calibration, but final organizational priority/utility is not a psychometric score. Product-specific causal uplift, costs, stakeholder weights, and expected net intervention value belong in a downstream decision layer unless they are implemented as domain-neutral reusable decision primitives.

## 4. Users and jobs to be done

| User | Job |
|---|---|
| Psychometrician | Fit and compare defensible measurement models, diagnose fit/DIF, and recover parameters. |
| AI evaluation engineer | Calibrate multiple LLM judges, prompts, rubrics, and generated probes. |
| Assessment engineer | Define reusable assessment/rubric/scoring contracts and governed item banks. |
| Researcher | Reproduce simulation and recovery evidence and compare scientific hypotheses. |
| Downstream product team | Consume stable, versioned contracts/results without importing internal numerical implementation details. |
| Enterprise reviewer | Inspect deterministic provenance, scientific boundaries, security posture, and release evidence. |

## 5. Functional requirements

### FR-1 Contract identity and immutability

Published/operational contracts must be versioned and immutable. Durable public identifiers should be opaque/nonnumeric where appropriate; content-addressed identities are preferred for audit-critical artifacts.

### FR-2 Fail-closed scientific boundaries

The package shall reject or return an explicit unsupported/indeterminate state rather than manufacture a result for:

- unidentified/disconnected designs;
- unknown model relation where relation determines the valid test;
- non-finite numerical results;
- incomplete provenance at a replay-protected boundary;
- invalid scoreability assumptions;
- unsupported response/model combinations.

### FR-3 Rust numerical authority

New mathematical/psychometric production arithmetic shall be implemented in Rust. Python may orchestrate and validate. Any retained Python numerical reference/fallback must be bounded and parity-tested.

### FR-4 Realistic validation

Scientific algorithms must include tests against known truth or known invariants. Appropriate evidence may include bias, MAE/RMSE, interval coverage, convergence, response-probability/information recovery, factor/loading recovery after correct alignment, and CPU/GPU parity.

### FR-5 Model selection

The model-comparison system shall classify relationship/nestedness from actual constraints where feasible and route to the appropriate comparison procedure. It shall not select a model solely because it has better in-sample fit.

### FR-6 Rater and judge calibration

Human and AI judges shall be represented with explicit identity/version/occasion metadata. Where the data and estimator support it, severity, discrimination, criterion bias, range restriction, and drift must be separable from target quality.

### FR-7 Hierarchy/time preservation

Observation contracts shall preserve context dimension, multiple membership, testlet/query grouping, temporal occasion, and system-run identity needed for later identified models.

### FR-8 Governed generated-item trust boundary

Provider output is untrusted. Candidate parsing/validation shall be bounded and reject malformed/duplicate/non-finite/unknown fields, provenance replay, invalid answer-key references, invalid evidence references, and contradictory response-format semantics.

### FR-9 Reports

Reports shall distinguish observation, estimate, diagnostic, model-selection evidence, uncertainty, interpretation boundary, and release/provenance metadata. HTML reports shall remain accessible and script-free where that is the existing contract.

### FR-10 Modular interoperability

`fast-mlsirm` shall communicate with downstream/adjacent CWL systems through explicit versioned contracts or immutable artifacts. Direct cross-service database access is prohibited.

## 6. Non-functional requirements

### NFR-1 Correctness

- Exact, reproducible public contract validation.
- Deterministic seeds where deterministic simulation/selection is promised.
- No silent fallback that changes the scientific model.
- Numerical tolerance documented by method and precision.

### NFR-2 Performance

- Rust-first kernels.
- Low-context-switch CPU parallelization for computationally material work.
- GPU acceleration only where workload and parity evidence justify it.
- Bounded memory and input sizes for untrusted or fallback paths.
- No universal speedup claim without reproducible benchmark evidence.

### NFR-3 Reliability

- Bounded subprocess/runtime behavior for scheduled scientific studies.
- Reproducible packaging and explicit failure classification.
- Heavy Monte Carlo studies separated from bounded PR smoke tests without deleting scientific evidence.

### NFR-4 Security and privacy

- Least privilege and immutable workflow/action pinning where practical.
- No `COPILOT_GITHUB_TOKEN` for autonomous development scheduling.
- `NVIDIA_NIM_API_KEY` only for genuine model-backed work, preferably via the owning orchestration boundary.
- Purpose limitation and identity separation rather than indiscriminate PII masking.
- No raw credentials or uncontrolled provider output in audit logs.

### NFR-5 Quality

- Beginner-readable public docstrings/rustdoc.
- Exact 100% owned production statement and branch coverage where repository policy enforces it, with meaningful tests.
- Security Scan/SAST/fuzz/package gates remain fail-closed.

### NFR-6 Documentation

PRD, TRD, root architecture, ADRs, UML, logical ERD, traceability, AGENTS, CLAUDE, doctoring, and CHANGELOG must track material contract changes.

## 7. Product status taxonomy

Each major capability shall be labeled in traceability/documentation as one of:

- **implemented_on_main** — available on the referenced protected-main snapshot;
- **open_pr** — concrete implementation exists but is not integrated;
- **planned** — accepted direction without integrated implementation;
- **research_only** — evidence/hypothesis exists but no product contract is promised;
- **out_of_scope** — belongs to another bounded context.

No PR-body claim or conversation note is promoted to `implemented_on_main` without protected-main evidence.

## 8. Out of scope

- Clinical diagnosis/treatment claims.
- Employment/admission/insurance/credit/legal consequential decision authority.
- Hosted tenant/session/consent databases and UI.
- Identity credential storage.
- Generic LLM provider routing.
- Claims of SOC 2/CSAP certification absent actual certification.
- Treating a statistically selected model as causal truth.

## 9. Release criteria

A release is allowed only from the exact integrated protected head after all required repository gates pass, documentation/changelog are rendered and synchronized, and scientific/recovery evidence appropriate to the changed methods is present. Version bumps must follow repository policy; release artifacts must be verified after publication.

## 10. Success measures

The product should be evaluated by evidence such as:

- parameter/structure recovery accuracy and uncertainty coverage;
- reproducible model-selection accuracy under realistic simulation;
- reduction in raw-rater/judge bias after calibration;
- stable cross-version/linking behavior;
- bounded runtime/memory behavior;
- end-to-end rubric-to-calibration provenance completeness;
- defect escape rate and exact-head CI/security reliability;
- downstream adoption without reverse dependency or schema duplication.

Commercial value claims require actual customer outcomes; repository quality targets are not a valuation claim.
