# Product Requirements Document — fast-mlsirm

Status: **Authoritative**  
Scope: reusable `fast-mlsirm` library, CLI, contracts, scientific evidence, and package artifacts  
Last reconciled: **2026-08-09**

## 1. Product thesis

`fast-mlsirm` turns assessment, human-rating, LLM-as-a-Judge, and other structured evaluation evidence into auditable psychometric measurement rather than treating raw scores, correlations, or model verdicts as truth. Its differentiator is the combination of reusable versioned evaluation contracts, Rust-first psychometric computation, calibration of fallible raters, relation-safe model selection, true-parameter recovery, and governed evidence for score interpretation.

The product is a **reusable measurement platform component**, not the hosted assessment application. `ContextualWisdomLab/psychometrics-commons` is the canonical downstream hosted product.

## 2. Primary users and consumers

1. **Psychometricians and measurement scientists** who need reproducible simulation, estimation, model diagnostics, linking, DIF/invariance, recovery, and scoreability evidence.
2. **Assessment and automated-scoring engineers** who need one versioned contract for rubrics, human/AI raters, calibration, validation, adjudication, monitoring, and reports.
3. **AI evaluation/RAG engineers** who need reference-free or partially referenced evaluation without assuming that an LLM judge is ground truth.
4. **Research engineers** constructing and calibrating new rubric/item banks, including artificial-crowd pilots and governed item generation.
5. **Downstream products and services**, especially Psychometrics Commons, that consume immutable contracts and scientific results through stable interfaces.

## 3. Jobs to be done

A user should be able to:

- define what is being measured through immutable `AssessmentSpec` and `RubricSpecification` artifacts;
- compile rubrics into reproducible item blueprints and bounded generation contracts;
- ingest provider-generated candidates only after strict provenance and source checks;
- represent human, LLM, and external scoring engines through common observation contracts without confusing abstention, failure, missingness, and low scores;
- fit appropriate psychometric models and estimate uncertainty/diagnostics through Rust-owned numerical paths;
- distinguish rater severity and other evaluator effects from the latent construct;
- compare dimensional/model structures without forcing a winner when nestedness, distinguishability, or identification is unresolved;
- determine whether general/subscale scores are scoreable rather than assuming that good model fit licenses all interpretations;
- inspect local dependence, testlet/context effects, DIF/invariance, fairness, agreement, drift, and recovery;
- prove numerical accuracy with bias/MAE/RMSE/coverage and backend parity instead of correlation alone;
- link versions and maintain auditable item/model evidence across time; and
- export accessible, deterministic evidence reports suitable for downstream review and release governance.

## 4. Product requirements

### PRD-001 — Canonical assessment and rubric contracts

The package shall provide versioned, immutable, content-addressed assessment, rubric, scoring-policy, and construct contracts. Rubric score levels and construct definitions have one source of truth; domain adapters reference those artifacts instead of redefining them.

**Acceptance:** mutation/replay/cross-reference tests, deterministic fingerprints, bounded metadata, stable public API, and backward-compatible migration guidance for contract changes.

### PRD-002 — Governed rubric-to-item construction

The package shall support the upstream path from rubric to blueprint to generation contract. It shall preserve construct, difficulty/evidence/task constraints, response format, score semantics, and immutable provenance.

A generated item is a **candidate**, not an accepted operational item.

### PRD-003 — Untrusted provider-output boundary

Provider output shall be treated as hostile input. The package shall reject duplicate JSON keys, non-finite JSON numbers, unknown/missing fields, over-budget payloads, provenance replay, invalid answer-key semantics, invalid score order, invalid source references, and false evidence spans before canonical candidate construction.

Semantic answerability, construct alignment, ambiguity, distractor quality, leakage, fairness, and psychometric quality remain later gates.

### PRD-004 — Human and automated raters as observations

Human raters, LLM judges, and external scoring engines shall share domain-neutral observation contracts. The system shall preserve evaluator identity/version, task/rubric revision, criterion, occasion/prompt provenance, evidence, and terminal outcome semantics.

The product shall never describe an LLM judge as an oracle merely because it agrees with another rater.

### PRD-005 — Psychometric calibration and diagnostics

The package shall provide reusable fitting and diagnostic capabilities for the supported IRT/MLSIRM, polytomous, many-facet, testlet, linking, fit, agreement, DIF/invariance, G-theory, and related measurement families exposed by the current package.

New model families require explicit identification, simulation/recovery, numerical accuracy, documentation, and public API evidence before being described as supported.

### PRD-006 — Multidimensional structure and score interpretation

The package shall distinguish the questions answered by unidimensional, correlated multidimensional, bifactor, higher-order, testlet/two-tier, multifaceted, and latent-space structures. A more complex model shall be selected only when relation-aware statistical comparison, residual/local-dependence evidence, held-out prediction, stability, and recovery support it.

Bifactor fit and bifactor scoreability are separate product decisions. General/subscale scores require appropriate ECV/PUC/omega/H/factor-determinacy evidence where scientifically applicable.

### PRD-007 — Relation-safe model comparison

Model-comparison APIs shall fail closed when the relation between models is unknown, overlapping without a distinguishability result, or boundary/nonlinear-constraint nested without the required likelihood-ratio/parametric-bootstrap treatment. No API may manufacture a preferred model solely from a positive sample variance or a generic fit statistic.

### PRD-008 — Factor retention and rotation

Factor-retention workflows shall treat the number of substantive factors as a model-order problem supported by multiple evidence sources rather than a single universal cutoff. Factor rotation shall expose an extensible criterion registry and stable selection evidence without claiming one universally optimal criterion or a mathematically proven global optimum from finite multi-start search.

### PRD-009 — Contextual, multiple-membership, and temporal measurement

Where observations are nested, cross-classified, multiple-membership, repeated, or temporally ordered, the product shall represent that structure rather than silently collapsing it into individual-level data. Reusable contracts and future numerical models must distinguish discrete occasion order, elapsed-time effects, rater drift, testlet dependence, and contextual membership.

**Current boundary:** protected `main` contains extensive multilevel-adjacent functionality, but the dedicated governed contextual/longitudinal contract slice is still under active PR work. The PRD records the required product direction without claiming that unmerged APIs are released.

### PRD-010 — Reference-free RAG evaluation as measurement

The product shall support domain-neutral contracts enabling RAG evaluation that separates at least groundedness/faithfulness, response relevance/utility, retrieval/context relevance, robustness, abstention/calibration, and citation/evidence dimensions when the task requires them.

Reference-free must not be represented as truth-free: context-grounded support, world correctness, and completeness are distinct claims. LLM judge outputs remain calibratable rater observations.

### PRD-011 — Automated scoring calibration and validation

The product shall support automated-scoring workflows in which people and machines can be jointly evaluated for agreement, severity, bias, range use, drift, DIF/fairness, and human-review triggers. Correlation alone is insufficient evidence of interchangeability or validity.

Essay evaluation and enterprise-issue evaluation are domain adapters over the same reusable scoring/calibration contracts, not separate psychometric engines.

### PRD-012 — Governed item/measurement-bank lifecycle

The target platform contract shall support an immutable lifecycle:

`draft -> audited -> screened -> pilot -> calibrated -> approved -> active -> suspended/quarantined -> retired`.

An operational implementation must retain calibration history, linking anchors, exposure/usage evidence, DIF/drift findings, approvals, rollback/supersession, and immutable provenance. The library may provide contracts and scientific utilities without owning hosted persistence.

### PRD-013 — Scientific recovery as a release property

For estimators and model-selection procedures, release evidence shall prefer true-parameter bias, MAE/RMSE, interval/SE coverage, convergence, probability/ICC/information recovery, invariance/DIF, and CPU/GPU parity as applicable. Pearson/Spearman correlation can be reported as association evidence but cannot stand in for parameter recovery or absolute agreement.

### PRD-014 — Standalone and modular operation

The package shall remain installable and useful without Psychometrics Commons, Keyverse, TEPP, Gyeot, contextual-orchestrator, semantic-data-portal, or other CWL services. Optional integrations use explicit versioned contracts/artifacts and preserve the owning repository's authority boundary.

### PRD-015 — Security, privacy, and governance

The package shall fail closed on malformed/stale/replayed evidence, use least privilege in CI, preserve immutable action/source pinning where practical, and design for auditable SOC 2/CSAP control evidence without claiming certification.

PII shall not be blanket-masked when doing so destroys measurement utility. Instead the architecture shall prefer purpose limitation, data minimization, separate identity domains, authorization, encryption, bounded retention, selective disclosure, and audited access in the systems that own the sensitive data.

### PRD-016 — Accessible, deterministic reporting

Machine-readable JSON and human-readable reports shall preserve exact scientific evidence, clear insufficient-evidence states, accessible semantics, deterministic rendering, and explicit interpretation boundaries. Presentation shall not silently mutate numerical results.

### PRD-017 — Release-grade packaging and provenance

A release shall be cut only from an exact protected integration head with current CI, security, owned coverage/docstrings, package/reinstall acceptance, compatibility, scientific recovery where relevant, SBOM/provenance, independent review, and release-acceptance evidence. Version and `CHANGELOG.md` are updated only when that release is genuinely ready.

## 5. Buyer-visible workflows

### Measurement authoring

`AssessmentSpec -> Rubric -> Blueprint -> generated/curated candidates -> screening -> pilot -> calibration -> bank approval`

### Automated scoring

`response -> human/AI/external observations -> connected rating design -> Rust calibration -> validity/fairness/rater evidence -> adjudication trigger -> deterministic report`

### Scientific model selection

`candidate dimensions/structures -> relation classification -> appropriate LR/Vuong/predictive comparison -> residual dependence -> recovery -> scoreability -> accepted model artifact`

### Reference-free AI evaluation

`question/evidence/response -> atomic criteria -> multi-judge observations -> many-facet/multidimensional calibration -> testlet/local interaction diagnostics -> uncertainty/validity report`

## 6. Non-goals and claims not made

The reusable package does not claim to be:

- a hosted multi-tenant assessment platform;
- a clinical diagnostic/treatment device;
- an autonomous employment, admission, insurance, credit, or legal decision maker;
- a guarantee that any LLM judgment is correct;
- a guarantee that model fit establishes construct validity;
- a general causal inference platform for deciding interventions;
- a guarantee of universal GPU availability or universal performance improvement;
- a Bayesian HMC/NUTS platform unless and until a separately validated implementation is added; or
- a persistence/ORM owner for hosted participant, identity, consent, or session data.

## 7. Success measures

Product readiness is demonstrated by evidence, not by a valuation label. The durable scorecard is:

- exact owned production statement/branch coverage and beginner-readable public documentation;
- realistic RED->GREEN regressions for product defects;
- true-parameter recovery and uncertainty coverage for scientific estimators;
- Rust/Python and CPU/GPU parity where a second execution path is claimed;
- stable versioned contracts with replay/forgery protection;
- predictable resource ceilings and failure semantics;
- model-selection decisions that can return indeterminate rather than false certainty;
- DIF/invariance/fairness and rater-drift evidence where score use requires it;
- deterministic packaging/SBOM/provenance/release acceptance; and
- downstream workflows that can consume the package without hidden repository coupling.

Commercial valuation or certification is explicitly outside this PRD; acquisition readiness must be supported by independently verifiable product, scientific, operational, adoption, and customer-value evidence.
