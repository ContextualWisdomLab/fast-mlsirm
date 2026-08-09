# Technical Requirements Document — fast-mlsirm

Status: authoritative technical baseline for implementation and integration.

## 1. Technology architecture

### 1.1 Rust numerical core

`crates/mlsirm-core` is the production owner of psychometric mathematics. New likelihoods, gradients, Hessians/curvature, optimizers, information matrices, calibration arithmetic, scoring/ranking, rotation objectives, recovery kernels, and decision-utility arithmetic must be implemented in Rust unless a documented exception exists.

Requirements:

- checked dimensions and bounded workspace sizes before allocation;
- deterministic reductions where reproducibility is required;
- multithreading that minimizes unnecessary synchronization/context switching;
- stable finite-value validation and non-reflective errors;
- no silent fallback that changes model identity;
- explicit algorithm and parameterization identifiers in evidence outputs.

### 1.2 PyO3 product boundary

`crates/fast-mlsirm-py` exposes Rust functionality to Python. Secondary PyO3 modules may exist only through one composed registry/initialization design so feature PRs cannot overwrite each other's exported symbols.

Every Rust-backed Python feature requires:

- bounded Python input validation;
- direct Rust delegation tests;
- typed immutable Python result objects where appropriate;
- wheel/reinstall/import evidence;
- platform compatibility evidence before support claims.

### 1.3 Python layer

`python/fast_mlsirm` owns public contracts, orchestration, validation, marshalling, compatibility adapters, and deterministic reporting. Python reference numerics are permitted for parity/research/fallback only when clearly labeled and prevented from becoming a conflicting numerical authority.

## 2. Core data contracts

All public domain objects should be immutable or effectively immutable, bounded, versioned, content-addressed where provenance matters, and backed by stable descriptive identifiers.

Identifier requirements:

- descriptive opaque string identifiers for public/audit objects;
- database/persistence field and object names use two-or-more-word `snake_case` by default;
- no semantic parsing of IDs into latent-factor meaning;
- revisions and artifacts use cryptographic fingerprints where replay resistance matters.

Key contract families:

- AssessmentSpec / assessment configuration;
- RubricSpecification / RubricLevel / response format;
- Blueprint and generation contracts;
- item, rater, criterion, score-observation and evidence records;
- respondent/context/occasion membership and longitudinal structures;
- calibration/model-comparison/scoreability/recovery results;
- validation/fairness/adjudication/monitoring/report contracts.

## 3. Psychometric numerical requirements

### 3.1 MLSIRM/MLS2PLM

The existing simple-structure MLS2PLM contract remains distinct from the fully general discrimination-vector form. Any full-vector MLS2PLM extension must be a complete model path covering simulation, parameter shapes, likelihood, gradients, fitting, recovery, diagnostics, docs, and Rust/Python parity.

### 3.2 Multidimensional and structural models

Candidate structures may include correlated MIRT, bifactor, higher-order, testlet, two-tier, multifaceted, and latent-space variants. Their relationship must be determined from actual parameter constraints and variance boundaries.

### 3.3 Model comparison

Required model-comparison pipeline:

1. classify relation (`regular_nested`, `boundary_nested`, `nonlinear_constraint_nested`, `strictly_non_nested`, `overlapping`, `indistinguishable`, `unknown`);
2. use regular LR only for regular nested models;
3. use boundary-aware/parametric-bootstrap LR for singular or boundary extensions;
4. require formal distinguishability before non-nested preference;
5. add cluster-aware held-out prediction and bootstrap uncertainty;
6. require recovery/model-selection simulation for consequential interpretation.

No API may infer a winner from a positive numerical variance proxy alone.

### 3.4 Bifactor scoreability

The scoreability layer must distinguish descriptive latent-response indices from observed categorical reliability. General-factor applicability, standardized-loading identities, ECV/PUC/omega-H/omega-HS/construct replicability/factor determinacy and interpretation limits must be explicit.

### 3.5 Rotation

Rotation criteria implement a common Rust value/gradient contract. Optimizers and criteria are separate. Orthogonal/oblique transforms must preserve the correct matrix convention and chain rule. Multi-start results expose convergence, stationarity, basin support and equivalent-solution normalization. Criterion selection uses neutral evidence rather than comparing incomparable criterion objective values directly.

### 3.6 Recovery and validation

For simulated truth, minimum evidence includes scale/rotation/alignment before comparison plus bias, MAE, RMSE, SE bias, confidence/credible interval coverage where available, convergence and failure classification. Distance/latent-space structures require invariant comparison such as distance recovery and/or Procrustes-aligned coordinates.

## 4. Multilevel and temporal requirements

Contextual membership must preserve explicit dimensions and may represent nesting, cross-classification and weighted multiple membership. Weights normalize according to the declared dimension-level contract rather than a flattened global context.

Longitudinal observations require respondent, occasion, ordering/time provenance, revision identity, and explicit state semantics. A discrete occasion-step autoregressive coefficient may not be scaled by elapsed time unless a continuous-time transition model with defined units and recovery evidence is implemented.

Future Rust estimators must detect identification failures such as disconnected designs, context/time confounding, insufficient level linkage, random slopes without within-person variation, and unanchored drift.

## 5. Rubric, generation and item-bank requirements

The governed lifecycle is:

`Rubric → Blueprint → Generation Contract → Untrusted Output → Validation/Screening → Pilot Observations → Rust Calibration → Item Bank → Serving → Monitoring/Revision`.

Requirements:

- candidate-blind evidence-grounded benchmark mode by default;
- candidate-aware discovery isolated with cross-fitting when used;
- atomic criteria preferred for calibration/audit;
- common/domain anchors for linking;
- typed response-format-specific answer keys;
- duplicate-key/nonfinite/oversize JSON rejection;
- evidence-reference integrity and source-bounded attribution;
- immutable operational rubric/item versions;
- lifecycle states and signed/hashed provenance;
- DIF, drift, exposure, quarantine and retirement hooks.

Provider SDKs must remain adapters, not the canonical domain model.

## 6. Automated scoring requirements

All scoring engines implement one provider-neutral protocol and emit the same criterion-level observation contract. Required provenance includes rater/engine/model version, rubric version, prompt/configuration identity, score status, evidence spans or evidence references, and abstain/fail/exclude states where applicable.

Calibration may model person/essay, item/prompt, criterion, rater, occasion and other facets. Human review routing is based on evidence, uncertainty, disagreement, fit/fairness/drift and policy—not merely a raw confidence scalar.

## 7. Reference-free RAG requirements

Canonical observations distinguish:

- query and system run;
- retrieved evidence and provenance;
- generated response identity;
- atomic claims/obligations;
- criterion/probe and construct;
- judge family/model/prompt/occasion;
- testlet/query grouping;
- perturbation/anchor expectations.

Groundedness cannot be labeled world correctness unless the evidence regime supports that claim. Context recall/completeness cannot be claimed without an explicit reference/obligation universe.

## 8. Enterprise issue measurement requirements

Evidence extraction and measurement are separate from intervention decisions. Deterministic parsers should own explicit dates, amounts, frequencies and identifiers where possible; LLMs handle semantic inference behind provider-neutral contracts.

Measurement reports retain evidence/counterevidence, stakeholder perspective and uncertainty. A decision layer, if supplied by a downstream module, must explicitly represent candidate actions, outcome assumptions, cost, expected net intervention value, urgency from delay and value of information.

## 9. Security and privacy requirements

### 9.1 General

- fail closed on malformed or unauthenticated evidence;
- least-privilege GitHub/workflow permissions;
- exact-head and immutable-source binding for review/release evidence;
- no self-modifying or branch-writing validation workflow as final product state;
- bounded external input, log and artifact sizes;
- no credentials or rejected raw values in errors;
- SBOM, dependency scanning, SAST and provenance in release gates.

### 9.2 PII without blanket masking

PII is handled through architecture, not destructive masking:

- purpose-bound RBAC/ABAC and tenant boundaries in hosts;
- pseudonymous/opaque core identifiers with identity mapping isolated outside the psychometric core;
- field-level/envelope encryption with KMS-backed key management in persistence hosts;
- selective disclosure to model adapters;
- explicit retention/erasure/export and residency policy;
- access-purpose and tamper-evident audit trails;
- tests for cross-tenant access, re-identification boundaries, replay and retention behavior.

## 10. LLM and autonomous-agent requirements

LLM-backed repository tests/development use `NVIDIA_NIM_API_KEY` from GitHub Secrets; `COPILOT_GITHUB_TOKEN` is prohibited for development scheduling. Prefer contextual-orchestrator as a provider-neutral routing layer while respecting repository writer leases.

Where orchestration materially affects product quality, compare shallow routing and deeper multi-agent/test-time-compute strategies under comparable budgets. Record workflow stages, decomposition, recursion depth, access lists, role-specific reasoning effort and ablation evidence. Optimize correctness, evidence quality and control rather than latency alone.

## 11. CI/CD requirements

For every PR:

- exact current head/base evidence;
- current review/thread state;
- test-first fixes for valid defects;
- focused and complete relevant suites;
- Rust/PyO3/package/import validation;
- 100% owned-production statement/branch and public-docstring gates;
- security/SAST/dependency/supply-chain checks;
- no stale-head evidence promoted to pass;
- merge only through repository policy/branch protection.

Long Monte Carlo studies may run scheduled/manual/release lanes when PR execution cost is disproportionate, but PRs still require bounded scientifically representative smoke/recovery contracts. Scheduled development agents use immutable OpenCode workflow sources and NVIDIA NIM credentials.

## 12. Release requirements

A release is permitted only from one exact integrated protected head with:

- all required CI/security/review gates successful;
- package/wheel reinstall and compatibility evidence;
- scientific recovery/validation appropriate to changed models;
- provenance/SBOM/reproducibility evidence;
- migration and rollback plans for versioned contracts;
- rendered `CHANGELOG.md`;
- version bump and release artifact hashes bound to the accepted source commit.

## 13. Standards baseline

- ISO/IEC 25010:2023 — product quality model.
- ISO/IEC 42001:2023 — AI management-system requirements.
- NIST AI RMF 1.0 and NIST AI 600-1 Generative AI Profile — risk-management and TEVV guidance; note that AI RMF 1.0 is under revision as of 2026.
- WCAG 2.2 / WAI-ARIA for human-readable HTML evidence surfaces.
- Standards for Educational and Psychological Testing and method-specific primary psychometric papers for validity, fairness and score-use claims.

Method-specific doctoring must contain APA 7 references and enough equation-to-source traceability for an independent reviewer to reconstruct the intended implementation.
