# Product Requirements Document — fast-mlsirm

Status: candidate authoritative product requirements baseline; becomes normative when merged to protected `main`. Requirements describe intended product behavior and are **not** evidence that every capability is already shipped. Current implementation maturity and requirement-to-evidence traceability live in `docs/architecture/capability_maturity.md` and `docs/requirements_traceability.md`.

## 1. Product definition

`fast-mlsirm` is a reusable psychometric measurement engine for researchers, assessment engineers, AI-evaluation teams, and downstream products that need defensible latent-variable measurement rather than raw score aggregation.

The product value is not limited to MLSIRM estimation. The product should support a governed lifecycle from assessment/rubric definition through observations, calibration, diagnostics, model selection, scoreability, fairness, longitudinal/multilevel structure, evidence reporting, and release-quality provenance.

## 2. Primary users and jobs to be done

### Psychometric researcher

Needs to simulate, fit, diagnose, compare, recover, and report latent-variable models with reproducible scientific evidence.

### Assessment engineer

Needs stable AssessmentSpec/Rubric/Scoring contracts, item-generation blueprints, calibrated item/rater evidence, DIF/fairness diagnostics, linking, CAT/ATA, and versioned serving artifacts.

### AI evaluation engineer

Needs human and LLM judgments treated as fallible rater observations, reference-free RAG evaluation, evaluator drift/bias diagnostics, artificial-crowd calibration, and evidence-grounded reports.

### Automated-scoring operator

Needs one schema for human/AI/external scores, criterion-level evidence, many-facet calibration, agreement/fairness/drift monitoring, and human-review routing.

### Downstream product / MSA integrator

Needs an independently installable library with explicit versioned contracts, no hidden dependence on hosted product repositories, and clean composition with Psychometrics Commons, naruon, contextual-orchestrator, Keyverse, EgressWeave, TEPP, Gyeot, and other services.

## 3. Product outcomes

The product shall:

1. produce psychometric estimates and reports whose claims are bounded by the implemented model and available evidence;
2. distinguish relative association from absolute recovery, agreement, calibration, validity, fairness, and decision utility;
3. make human/LLM/rule judgments auditable observations rather than uncalibrated truth;
4. preserve multilevel, cross-classified, multiple-membership, repeated-measurement, and temporal context where scientifically relevant;
5. support end-to-end rubric/item/scoring workflows without turning provider-specific LLM calls into the core architecture;
6. support research reproducibility and acquisition-grade due diligence through deterministic provenance and release evidence;
7. remain usable standalone and as a modular component of a larger MSA system.

## 4. Functional requirements

### FR-01 — Core psychometric computation

Support canonical binary MLSIRM/MLS2PLM and related MIRT constraints, response-process diagnostics, polytomous/testlet/facet functionality where implemented, linking, CAT/ATA, recovery, information/uncertainty, and model-fit diagnostics.

### FR-02 — Rust-first numerical ownership

Production likelihood, gradient, Hessian/curvature, optimization, psychometric scoring/ranking, model-comparison kernels, and other numerical arithmetic shall be owned by Rust when introduced. Python may validate, orchestrate, marshal, and report.

### FR-03 — Multilevel and time-aware measurement

When a domain contains contextual membership or repeated observations, contracts shall be able to represent nested, cross-classified, weighted multiple-membership, respondent/occasion identity, revision provenance, and explicit time semantics. No atomistic or continuous-time interpretation may be inferred from missing structure.

### FR-04 — Measurement-model selection

Factor retention and structural-model choice shall be separate. Model relationships must be classified from actual constraints/boundaries. Unknown or observationally indistinguishable relations fail closed. Predictive validation, residual dependence, scoreability, invariance/DIF, and recovery must supplement fit statistics.

### FR-05 — Bifactor scoreability

If a bifactor model is used for reporting, the system shall expose scoreability evidence such as ECV/PUC/omega-H/omega-HS/construct replicability/factor determinacy as appropriate to the implemented scale and shall not authorize general/specific score interpretation merely because fit is improved.

### FR-06 — Adaptive rotation

Rotation shall use an extensible criterion registry, a common Rust optimizer, deterministic multi-start, stability diagnostics, and criterion-neutral evidence. Finite multi-start output shall be described as the best observed solution, not a certified global optimum. No universal rotation criterion shall be claimed.

### FR-07 — Rubric and item-generation lifecycle

Provide versioned RubricSpecification/Blueprint/Generation contracts, bounded structured-output schemas, provenance fingerprints, candidate validation/screening interfaces, artificial-crowd/human observation pathways, Rust calibration, governed item-bank lifecycle, linking, exposure, drift, quarantine, retirement, and rubric revision.

### FR-08 — AssessmentSpec and scoring contracts

All scoring workflows shall share stable assessment, rubric, score-observation, rater/engine, provenance, fairness, adjudication, and reporting contracts. Competing rubric or score schemas are prohibited unless a versioned migration is explicitly justified.

### FR-09 — Automated essay scoring validation

Support essay/prompt/criterion observations, human and AI raters, many-facet ordinal calibration, agreement, rater severity/fit/range behavior, subgroup SMD/DIF, uncertainty and human-review routing, with deterministic JSON/HTML evidence. Correlation with raw human scores is descriptive only.

### FR-10 — Reference-free RAG measurement

Represent question/context/response/atomic-claim/criterion/judge/system-run/testlet evidence so groundedness, retrieval relevance, answer utility, abstention, robustness, judge severity/discrimination, testlet dependence, and model-family sensitivity can be calibrated without treating LLM judgments as truth.

### FR-11 — Enterprise issue measurement

Represent evidence, counterevidence, stakeholder perspective, criterion observations, measurement uncertainty, and candidate interventions separately. Measurement scores must not be silently equated with business priority. Consequential decision layers require explicit utility/cost/action and identified causal or human-validation assumptions.

### FR-12 — Reports and audit evidence

Emit deterministic, accessible, source-bounded JSON/HTML artifacts with stable identifiers, complete provenance, explicit interpretation boundaries, and no credential/raw-source leakage. Reports shall support machine audit and human review.

### FR-13 — Release evidence

Provide reproducible package, benchmark, recovery, security, SBOM/provenance, buyer-evidence, due-diligence, and rollback evidence sufficient to evaluate the exact candidate artifact.

## 5. Non-functional requirements

- **Correctness:** realistic domain tests plus true-parameter recovery; correlation-only evidence is insufficient.
- **Coverage:** 100% owned production statement/branch coverage and full public docstrings/rustdoc where technically meaningful.
- **Performance:** deterministic low-contention CPU parallelism; GPU only with measured value and parity.
- **Security:** fail closed, least privilege, bounded inputs/outputs, immutable workflow sources/pins where practical, supply-chain evidence, no self-modifying branch writers.
- **Privacy:** purpose-bound access, tenant isolation in hosts, pseudonymous/opaque identifiers, selective disclosure and encryption rather than blanket masking that destroys measurement utility.
- **Accessibility:** buyer/operator HTML evidence follows applicable WCAG/WAI-ARIA semantics.
- **Interoperability:** provider-neutral schemas and versioned contracts; standalone operation remains first-class.
- **Auditability:** exact-head/source/model/rubric/evidence fingerprints and stable non-reflective errors.

## 6. Explicit non-goals of this repository

- hosted participant/session/consent management;
- identity federation or SSO;
- tenant application database/ORM ownership;
- organization-wide deployment control plane;
- provider-specific LLM gateway ownership;
- unreviewed autonomous high-stakes decisions;
- claims of regulatory certification or universal psychometric validity.

Those concerns belong to downstream hosted products or bounded service repositories.

## 7. Product acceptance gates

A vertical slice is product-ready only when:

- the end-to-end public contract is implemented, not merely an internal kernel or design document;
- `docs/architecture/capability_maturity.md` can truthfully promote the capability based on protected-main evidence;
- `docs/requirements_traceability.md` identifies the owner, governing ADR/architecture and verification evidence;
- the exact current head passes required tests/security/package/provenance gates;
- documentation and APA 7 equation-to-source traceability match the implementation;
- realistic recovery/agreement/fairness/compatibility evidence exists as applicable;
- migration/rollback and MSA compatibility are explicit;
- `CHANGELOG.md` is rendered and versioning is changed only when a release is genuinely ready.

## 8. Commercial-readiness principle

The acquisition-quality target is a prioritization bar, not a valuation claim. Evidence of scientific defensibility, operational reliability, enterprise governance, interoperability, supportability, measurable buyer value, and durable product differentiation must accumulate before any large valuation assertion is supportable.
