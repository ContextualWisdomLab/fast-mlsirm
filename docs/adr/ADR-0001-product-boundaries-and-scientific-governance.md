# ADR-0001: Product Boundaries and Scientific Governance

- **Status:** Accepted baseline
- **Date:** 2026-08-09
- **Decision owners:** fast-mlsirm maintainers
- **Scope:** repository ownership, numerical authority, model claims, evidence, privacy, MSA composition, and release governance

## Context

`fast-mlsirm` has evolved beyond its original narrow MLS2PLM prototype. The repository now includes or supports broader psychometric diagnostics, rubric/item-generation contracts, automated scoring/essay validation, enterprise measurement, release evidence, and cross-repository integration. The old combined PRD/TRD described a NumPy-first MVP and no longer represented the product or its scientific constraints.

At the same time, the ContextualWisdomLab ecosystem now has a canonical hosted product boundary: `psychometrics-commons` owns hosted application concerns, while `fast-mlsirm` must remain reusable and independently installable.

The project also has unusually strict scientific obligations. Flexible models can overfit; human or LLM raters can be biased; latent spaces are non-identifiable up to transformations; finite multi-start does not prove a global optimum; multilevel/temporal structure can be lost through flattened schemas; and high raw agreement/correlation can coexist with serious bias or invalid score interpretation.

## Decision 1 — Repository ownership

`fast-mlsirm` owns reusable measurement contracts, psychometric computation, diagnostics, calibration, model selection, recovery, fairness/invariance evidence, item/rater/scoring contracts, and deterministic reports.

`psychometrics-commons` owns hosted participant/session/consent/result lifecycle, persistence/migrations, HTTP/admin APIs, tenant/resource authorization, reference clients, deployment composition, and hosted workflow UX.

Other CWL repositories remain bounded integrations. `fast-mlsirm` must not depend on their implementation details.

### Consequence

No hosted runtime, product ORM, tenant database schema, identity service, or deployment configuration is recreated inside `fast-mlsirm`. Integration occurs through explicit versioned contracts or immutable artifacts.

## Decision 2 — Rust is the production numerical authority

Production psychometric mathematics is implemented in Rust. Python may validate, marshal, orchestrate, expose typed APIs, render deterministic reports, and retain bounded numerical references for parity/fallback only.

### Consequence

A feature is not product-complete merely because a Rust kernel exists; it also requires a stable PyO3/Python product path when Python is the supported public interface. Conversely, Python must not duplicate a second independent production likelihood, optimizer, score, rank, or utility engine.

## Decision 3 — CPU/GPU performance claims require parity

CPU implementations favor deterministic, low-contention multithreading with minimal unnecessary synchronization/context switching. GPU acceleration is used only for computationally material kernels and requires explicit CPU/GPU objective/result parity.

### Consequence

No feature may advertise a GPU backend that silently skips to CPU or produces unverified estimates.

## Decision 4 — Factor retention and structural model choice are separate

The software treats correlated MIRT, bifactor, higher-order, testlet, two-tier, multifaceted, and latent-space structures as distinct scientific hypotheses whose relationships depend on actual constraints and boundary conditions.

### Consequence

- regular nested comparisons use appropriate LR procedures;
- boundary/singular extensions require boundary-aware or parametric-bootstrap approaches;
- non-nested selection requires formal distinguishability before preference;
- held-out/cluster-aware prediction, residual dependence, DIF/invariance, scoreability, and recovery supplement in-sample fit;
- unknown relation fails closed rather than defaulting to a winner.

## Decision 5 — Bifactor fit does not authorize score interpretation

Bifactor general/specific scores require scoreability evidence such as applicable ECV/PUC, omega-H/omega-HS, factor determinacy and construct replicability, with the exact interpretation matched to the latent-response or observed-score scale used by the implementation.

### Consequence

The API and docs must distinguish model fit from score interpretation. A flexible bifactor model cannot win merely because it has a higher in-sample fit index.

## Decision 6 — No universal rotation criterion or global-optimum claim

Rotation criteria have condition-dependent behavior. Finite deterministic multi-start identifies the best observed solution among attempted starts, not a mathematically certified global optimum.

### Consequence

Rotation selection uses criterion-neutral stability/recovery/interpretability evidence rather than comparing raw objective values from different criteria.

## Decision 7 — True-parameter recovery outranks correlation-only validation

Parameter recovery requires scale/rotation/sign alignment before error calculation and reports bias, MAE, RMSE, uncertainty/coverage, convergence and classified failures. Correlation may be reported as supplementary rank/association evidence but never as proof of accurate recovery or agreement.

### Consequence

Simulation/recovery tests must include realistic designs and known truth. Automated scoring validation must also distinguish agreement, calibration, fairness, DIF and generalization from correlation.

## Decision 8 — Human and AI judges are fallible raters

Human, LLM, rules-based and external scoring engines emit comparable rater observations and provenance. No rater is implicitly treated as an infallible ground truth.

### Consequence

Many-facet calibration, rater severity/fit/range behavior, criterion-specific effects, drift and human adjudication are first-class concerns where relevant.

## Decision 9 — Rubrics and item banks are governed measurement artifacts

Rubrics, blueprints, generation contracts, candidates, calibrations and item-bank versions are versioned, bounded, provenance-bound artifacts. Operational versions are immutable; changes create new versions and require linking/anchor evidence when scores are compared across versions.

### Consequence

The product lifecycle is not `prompt → score`; it is `rubric → blueprint → generation → validation/screening → pilot observations → Rust calibration → governed item bank → monitoring/revision`.

## Decision 10 — Multilevel, multiple-membership and temporal context are explicit

Where data are contextual or repeated, the software preserves contextual dimension, weighted membership, respondent/occasion identity, ordering/time provenance and revision identity. It does not infer random-effect or temporal semantics from identifiers.

### Consequence

- atomistic interpretations are rejected when required contextual structure is absent;
- a discrete occasion-step autoregressive parameter is not silently reinterpreted as continuous time;
- future estimators must detect disconnected/confounded/under-linked designs.

## Decision 11 — Evidence measurement and consequential decisions are separate

Reference-free RAG evaluation distinguishes groundedness from world correctness/completeness according to the evidence regime. Enterprise issue measurement distinguishes latent issue state from intervention priority.

### Consequence

Causal or high-stakes automation requires explicit action/outcome assumptions, costs, expected intervention value/urgency/VOI as appropriate, and identified human or experimental validation. A psychometric score alone is not a causal recommendation.

## Decision 12 — PII is protected by architecture, not blanket masking

Blanket PII masking can destroy longitudinal linkage, multiple-membership structure, adjudication provenance and legitimate assessment operations. The preferred pattern is purpose-bound authorization, least privilege, pseudonymous/opaque core IDs, isolated identity mapping, selective disclosure, field/envelope encryption, KMS-backed keys, retention/export/residency controls and tamper-evident audit evidence.

### Consequence

Raw source content and direct identity data are retained only where an authorized host needs them. Core artifacts prefer references and fingerprints. Privacy tests must include cross-tenant/re-identification/replay/retention scenarios where applicable.

## Decision 13 — LLM development and test orchestration

LLM-backed tests/development use the GitHub Secret `NVIDIA_NIM_API_KEY`, preferably through contextual-orchestrator. `COPILOT_GITHUB_TOKEN` is not used for autonomous development scheduling. Review-agent identities and credential boundaries remain independent.

When orchestration affects scientific/product outcomes, shallow single-model routing and deeper test-time-compute/multi-agent strategies are compared under comparable budgets with workflow/decomposition/reasoning-effort ablations; latency is not the primary optimization criterion.

## Decision 14 — Documentation and release evidence are product interfaces

PRD, TRD, architecture, ADR, UML/ERD, method-specific doctoring, changelog and release evidence must describe the same current product boundary. Documentation drift that misstates the numerical backend, supported models, product ownership or scientific claims is a release defect.

### Consequence

The repository maintains an explicit documentation coverage matrix and updates architecture documents when governing contracts change.

## Standards and research basis

The baseline is informed by:

- ISO/IEC. (2023). *ISO/IEC 25010:2023 Systems and software engineering — Systems and software Quality Requirements and Evaluation (SQuaRE) — Product quality model*.
- ISO/IEC. (2023). *ISO/IEC 42001:2023 Information technology — Artificial intelligence — Management system*.
- Tabassi, E. (2023). *Artificial Intelligence Risk Management Framework (AI RMF 1.0)* (NIST AI 100-1). National Institute of Standards and Technology. https://doi.org/10.6028/NIST.AI.100-1
- Autio, C., Schwartz, R., Dunietz, J., Jain, S., Stanley, M., Tabassi, E., Hall, P., & Roberts, K. (2024). *Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile* (NIST AI 600-1). National Institute of Standards and Technology. https://doi.org/10.6028/NIST.AI.600-1
- American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.
- Jeon, M., Jin, I. H., Schweinberger, M., & Baugh, S. (2021). Mapping unobserved item-respondent interactions: A latent space item response model with interaction map. *Psychometrika, 86*(2), 378–403. https://doi.org/10.1007/s11336-021-09762-5
- Kang, I., & Jeon, M. (2025). Multidimensional latent space item response models: A note on the relativity of conditional dependence. *Psychometrika, 90*(2), 799–826. https://doi.org/10.1017/psy.2025.5
- Schneider, L., Chalmers, R. P., Debelak, R., & Merkle, E. C. (2020). Model selection of nested and non-nested item response models using Vuong tests. *Multivariate Behavioral Research, 55*, 664–684.
- Rodriguez, A., Reise, S. P., & Haviland, M. G. (2016). Evaluating bifactor models: Calculating and interpreting statistical indices. *Psychological Methods, 21*, 137–150.

Method-specific documents may add newer primary papers. A citation is not a substitute for implementation/recovery evidence.

## Rejected alternatives

### Recreate the hosted product inside this repository

Rejected because it couples reusable psychometrics to product persistence, identity, UI and deployment concerns and duplicates Psychometrics Commons.

### Keep Python as a co-equal production numerical backend

Rejected because two independent production numerical authorities increase drift and weaken CPU/GPU/recovery evidence. Python references remain useful for parity.

### Treat LLM/human scores as truth and average them

Rejected because severity, discrimination, criterion bias, drift, local dependence and construct-irrelevant shortcuts can be hidden by a raw average.

### Use one universal model/rotation criterion

Rejected because model and rotation adequacy are data-generating-structure and use-case dependent.

### Blanket-mask PII

Rejected because it can make valid measurement and audit workflows unusable; purpose-bound isolation and disclosure controls are more appropriate.
