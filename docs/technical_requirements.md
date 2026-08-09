# Technical Requirements Document — fast-mlsirm

Status: **Authoritative**  
Last reconciled: **2026-08-09**

This TRD maps the product requirements in `docs/product_requirements.md` to the reusable package's technical boundaries. It does not define hosted-product persistence or HTTP/session architecture; those concerns belong downstream in Psychometrics Commons.

## 1. Runtime and module boundaries

### TRD-001 — Three-layer implementation boundary

The supported architecture is:

```text
Python public API / contracts / validation / orchestration / reports
                         |
                         v
                 PyO3 binding registry
                         |
                         v
              Rust psychometric core
                         |
                 +-------+-------+
                 |               |
              CPU f64        GPU device path
```

- `python/fast_mlsirm/` owns public APIs, immutable contract types, validation, orchestration, reference/fallback implementations explicitly retained for parity, and reports.
- `crates/mlsirm-core/` owns production mathematical/statistical/psychometric arithmetic.
- `crates/fast-mlsirm-py/` owns the PyO3/numpy boundary.
- New numerical features shall not be independently reimplemented in Python when Rust owns the calculation. Python reference formulas are allowed as explicit independent oracles/parity paths, not competing production truth.

### TRD-002 — Canonical PyO3/export registry

All Rust-backed public feature modules shall coexist through one reviewed binding/export structure. Feature PRs may not introduce mutually incompatible secondary extension initializers, import hacks, runtime compilation, or source rewriting. Package-root exports shall be deterministic and tested.

## 2. Contract architecture

### TRD-003 — Canonical assessment/rubric/scoring identities

`RubricSpecification` is the rubric source of truth. `AssessmentSpec` and scoring policies reference exact rubric fingerprints rather than copying rubric levels or construct definitions. Canonical artifacts use:

- schema/version identifiers;
- deterministic canonical serialization;
- complete SHA-256 content fingerprints for durable identity;
- bounded public audit handles where exposed;
- immutable nested collections; and
- factory/replay validation for cross-object references.

Direct construction must not permit bypass of invariants that a factory claims to enforce.

### TRD-004 — Generated-item request boundary

The item-generation flow shall preserve the exact chain:

`RubricSpecification -> ItemBlueprint -> GenerationContract -> GenerationRequest -> GeneratedItemCandidate -> GenerationExecution`.

Provider output is untrusted. At minimum the parser must enforce:

- raw payload and nesting limits;
- RFC 8259 finite/unique-member JSON behavior;
- closed required-field schemas;
- exact echoed rubric/blueprint/contract provenance;
- response-format-specific option and typed answer-key invariants;
- declared score order and complete rubric-level coverage;
- source cardinality by evidence mode;
- exact referenced source identifiers and bounded verbatim evidence spans;
- redacted provider/parser errors; and
- deterministic candidate/execution fingerprints.

Semantic entailment, answerability, ambiguity, distractor quality, leakage, bias/fairness, and psychometric quality are separate screening stages and must not be implied by structural acceptance.

### TRD-005 — Scoring observation semantics

Shared scoring contracts shall distinguish at least:

- scored observation;
- abstained / insufficient evidence;
- failed evaluation;
- excluded/missing observation.

Each usable observation binds assessment/rubric/task revision, criterion, rater/engine identity and version, response/evidence identity, and provenance. Domain adapters (essay, RAG, enterprise issue) project into the shared contracts instead of introducing independent calibration arithmetic.

## 3. Numerical architecture

### TRD-006 — Rust numerical authority and precision

Likelihoods, gradients, estimators, factor/model diagnostics, calibration, linking, recovery statistics, rotation arithmetic, and other owned psychometric kernels execute in Rust when exposed as production functionality.

- CPU scientific reference arithmetic uses f64 unless the method explicitly requires another type.
- GPU kernels may use a device-appropriate type only with documented tolerances, realistic parity tests, and explicit fallback semantics.
- Non-finite intermediate/final results fail closed unless a published method explicitly defines a non-finite diagnostic representation.
- Resource products (array dimensions, bytes, iterations, candidates) are checked before allocation or multiplication overflow.

### TRD-007 — CPU/GPU execution

CPU parallelism should be coarse-grained and minimize synchronization/context switching. GPU paths are device sub-options of Rust computation, never a separate scientific model. The product may claim GPU execution only when an actual adapter/kernel executed and parity evidence exists; CPU fallback cannot be relabeled GPU success.

### TRD-008 — Simple-structure MLS2PLM compatibility

The existing canonical formula remains a valid simple-structure specialization:

```text
eta_pi = exp(alpha_i) * theta_p,d(i) + b_i - exp(tau) * r_pi
r_pi = sqrt(sum_k (xi_pk - zeta_ik)^2 + eps)
```

A full discrimination-vector MLS2PLM is a separate model path requiring coordinated parameter shapes, simulation, likelihood, gradients, tests, documentation, recovery, and Rust/Python evidence. Local algebra/performance PRs must not silently redefine the model.

## 4. Structural measurement requirements

### TRD-009 — Relation-aware model metadata

Candidate fit objects and comparison APIs shall preserve model-family and relation metadata sufficient to classify comparisons as regular nested, boundary nested, nonlinear-constraint nested, strictly non-nested, overlapping, indistinguishable, or unknown where evidence permits.

An `unknown` relation is not treated as strictly non-nested by default.

### TRD-010 — Model comparison

- Regular nested: use appropriate likelihood-ratio/robust procedures.
- Boundary/singularity/nonidentified-under-null extensions: prefer parametric bootstrap or another justified boundary-aware method.
- Strictly non-nested/overlapping: require the formal distinguishability evidence needed by the implemented Vuong procedure before returning a preferred model.
- Held-out likelihood uses leakage-safe grouping (person/query/testlet/rater/domain as appropriate), not arbitrary response-cell splits that share latent/evidence context across train/test.
- If models are practically indistinguishable, APIs return indeterminate and recommend the simpler interpretation rather than fabricating certainty.

### TRD-011 — Bifactor scoreability

Bifactor diagnostics require a valid declared general factor and the input assumptions for each index. PUC is returned only where its strict-structure assumptions hold. Logistic latent-response transformations must not be described as categorical observed-score reliability. Public APIs expose units/assumptions and reject malformed/non-finite/overflowing inputs.

### TRD-012 — Factor retention and rotation

Factor retention combines candidate-count evidence, ML/MML fit, boundary-aware adjacent-model tests where appropriate, predictive performance, residual dependence, stability, and recovery. Rotation uses a criterion registry and shared optimizer, deterministic multi-start, sign/permutation canonicalization, stationarity/basin evidence, and criterion-neutral selector evidence. Finite multi-start means **best observed**, not proven global optimum. Target/PST and SPD algebra semantics are explicitly tested.

### TRD-013 — Testlet, contextual, multiple-membership, and temporal structures

Reusable contracts must preserve:

- observation-to-context dimension identity;
- one-hot or positive weighted membership with a validated sum-to-one contract when the model requires it;
- cross-classified/multiple-membership assignments without collapsing them into one label;
- respondent/task/rater/testlet grouping identities;
- event/occasion ordering and explicit temporal parameter semantics; and
- separation between discrete-step autoregression and continuous-time/elapsed-gap models.

Numerical fitters for these structures are not considered released until true-parameter recovery, identification, uncertainty coverage, and CPU/GPU parity (when applicable) are proven.

## 5. Reliability, validity, fairness, and recovery

### TRD-014 — Recovery evidence

Scientific estimator releases use simulated known-truth designs representative of product use. Required evidence, as applicable:

- parameter bias and MAE/RMSE after required scale/rotation/Procrustes/linking alignment;
- convergence and invalid-fit rates;
- standard-error bias and interval coverage;
- response/category probability or ICC recovery;
- test/item information recovery;
- DIF/invariance recovery; and
- CPU/GPU/Rust/reference parity.

Correlation may supplement, never replace, absolute recovery evidence.

### TRD-015 — Rater evidence

Many-facet/automated-scoring workflows preserve connectedness and evaluate severity, agreement, criterion/task effects, range use, drift, and subgroup/fairness evidence as the implemented model permits. Human scores are observations with error, not automatically true scores.

### TRD-016 — Consequential use boundary

The reusable library reports measurement and diagnostic evidence. It must not silently convert validity/fit/rater diagnostics into high-stakes policy decisions. Downstream products own approved decision policy, consent, authorization, and human governance.

## 6. Resource and error contracts

### TRD-017 — Bounded input/materialization

Every API that materializes caller/provider-controlled iterables, arrays, strings, JSON, logs, model candidates, subprocess output, or source packets has an explicit size/count/depth/time ceiling before unbounded allocation. Error messages expose stable codes/paths or bounded diagnostics without echoing credentials, raw PII, untrusted provider output, or arbitrary source content.

### TRD-018 — Timeouts and long studies

PR smoke tests are bounded. Heavy Monte Carlo recovery, long fuzzing, and large scientific studies belong in scheduled/manual/release workflows unless a small deterministic PR-scale regression is required. Subprocess and external-provider operations use operation-specific deadlines and process-tree cleanup rather than one unrealistic global timeout.

### TRD-019 — Concurrency and determinism

Scientific randomization accepts explicit seeds; multithreaded start/shard ordering remains deterministic where product output depends on it. Concurrent artifact/release/job paths use idempotency or immutable identifiers when introduced. No CI workflow may rewrite the PR branch to generate the implementation it is supposed to test.

## 7. Security and privacy

### TRD-020 — Least privilege and supply-chain controls

CI uses least-privilege permissions and immutable action pins where practical. Security gates remain fail closed. Dependency, source, artifact, SBOM, and release provenance are bound to the exact tested head/artifact. No model/review credential is repurposed to manufacture approval or merge authority.

### TRD-021 — LLM credential boundary

When a live LLM is genuinely required, development/test automation uses `NVIDIA_NIM_API_KEY` via GitHub Secrets and preferably the versioned contextual-orchestrator integration. `COPILOT_GITHUB_TOKEN` is not used for autonomous development. Existing independent reviewer identities and credential chains are preserved.

Deterministic parsing, validation, psychometric arithmetic, security policy, and model-selection gates shall not be delegated to an LLM merely because a model is available.

### TRD-022 — Sensitive-data handling

Canonical durable artifacts avoid raw PII where the scientific contract does not require it. Sensitive source/evidence content may cross an explicit provider or scoring boundary only under the caller's authorized workflow; durable metadata prefer exact digests, opaque references, and bounded counts. Hosted retention, residency, encryption keys, participant linkage, and data-rights workflows are downstream-owner concerns.

## 8. Persistence and logical data model

### TRD-023 — No hosted database in the reusable core

There is no canonical product ORM/database schema in `fast-mlsirm`. The logical ERD in `docs/architecture/diagrams.md` describes immutable artifact relationships and can be implemented in memory, files, object storage, a service database, or another downstream persistence layer.

If a reusable bank/catalog persistence layer is later added here, its ADR must prove:

1. the state is domain-neutral and independently useful;
2. no participant/session/consent/identity/product-tenant lifecycle leaks into the core;
3. migrations and rollback are supported; and
4. all database object names use descriptive multi-word `snake_case` by default.

## 9. Interoperability and MSA contracts

### TRD-024 — Dependency directions

Allowed:

`standalone user / hosted consumer -> fast-mlsirm`

`fast-mlsirm -> optional versioned client/protocol abstractions` only when the core remains functional without the external service.

Forbidden:

`fast-mlsirm -> psychometrics-commons ORM/HTTP/session/UI/deployment`

Cross-service application DB reads/writes are forbidden. Interoperation uses versioned APIs/events/immutable artifacts.

## 10. Testing and CI

### TRD-025 — Quality gates

Owned production code targets exact 100% statement and branch coverage, plus function/line/region coverage where tooling exposes it, using meaningful tests rather than exclusions. Public Python and Rust APIs include beginner-readable docs/docstrings/rustdoc.

Required test classes, where relevant:

- unit and property tests;
- Rust/Python delegation/parity;
- true-parameter/recovery tests;
- hostile input/resource-bound tests;
- realistic domain fixtures;
- CPU/GPU no-skip parity tests;
- fuzzing;
- package/reinstall/CLI smoke;
- accessibility/report semantics;
- documentation/contract trace tests; and
- release/provenance acceptance.

### TRD-026 — Documentation contract

The repository shall maintain current PRD, TRD, `ARCHITECTURE.md`, ADR index/template/material decisions, logical ERD and sequence/state/component diagrams, traceability matrix, doctoring, AGENTS/CLAUDE guidance, and release/changelog evidence. Scattered implementation plans do not satisfy this requirement by themselves.

## 11. Release and rollback

### TRD-027 — Release gate

Release only from the exact protected integrated head after current CI, security/SAST, owned coverage/docstrings, Rust/PyO3/wheel/package, compatibility, scientific recovery for changed estimators, provenance/SBOM, independent review, and release-acceptance gates pass. Then bump version according to policy, render/update `CHANGELOG.md`, publish signed/provenanced artifacts, and verify the released artifact.

A release does not claim SOC 2, CSAP, clinical, or consequential-decision certification unless that claim is independently obtained and explicitly scoped.

## 12. Observability and auditability

### TRD-028 — Reconstructable evidence

Scientific/public result artifacts preserve enough immutable identifiers and exact values to reconstruct which assessment/rubric/model/calibration/rater/package artifact produced them without persisting uncontrolled raw source text. Reports expose insufficient-evidence and degraded states rather than replacing them with successful-looking defaults.

The traceability matrix is part of release evidence: every material PRD requirement must point to a TRD mechanism, ADR or explicit no-decision state, code/API owner, and test/doctoring evidence or be marked planned.
