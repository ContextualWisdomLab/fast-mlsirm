# Technical Requirements Document — fast-mlsirm

**Status:** Authoritative component TRD  
**Last reviewed:** 2026-08-09

## 1. Scope

This TRD implements the reusable-component requirements in `docs/PRD.md`.
Hosted product/session/database/identity/UI concerns are explicitly out of scope
and belong downstream. `ARCHITECTURE.md` is the bounded-context authority.

## 2. Technology and execution layers

```text
consumer
  ↓
Python package / CLI
  ├─ immutable assessment/rubric/scoring contracts
  ├─ validation, orchestration, serialization, reporting
  └─ NumPy reference/fallback paths where retained
  ↓
PyO3 + numpy binding layer
  ↓
Rust psychometric core
  ├─ likelihood / gradients / estimation
  ├─ fit / diagnostics / linking / DIF / comparison
  ├─ simulation / recovery
  └─ CPU fixed-pool / bounded parallel kernels
  ↓
GPU device kernels only for parity-proven workloads
```

### TR-001 — Dependency direction

- Rust core MUST NOT depend on Python product logic.
- `fast-mlsirm` MUST NOT depend on Psychometrics Commons hosted code.
- Python contract modules MAY depend on domain-neutral validation utilities but
  MUST NOT import provider SDKs merely to define an artifact.
- Optional provider/orchestrator adapters MUST remain separable from the core
  package contract when practical.

### TR-002 — Canonical public API registry

Public Python symbols MUST be exported from one intentional package-root registry
or a clearly documented subordinate namespace. PyO3 numerical surfaces MUST use
one maintainable registration architecture so independent feature PRs do not
create mutually overwriting secondary module initialization paths.

## 3. Canonical contracts

The canonical reusable flow is:

```text
RubricSpecification
→ AssessmentSpec
→ ScoringRequest
→ ScoreObservation[]
→ ScoringResult
```

Rubric/item-generation work extends rather than replaces that flow:

```text
RubricSpecification
→ Blueprint / GenerationContract
→ GenerationRequest
→ GeneratedItemCandidate
→ screening evidence
→ pilot observations
→ calibration evidence
→ item-bank release
```

### TR-003 — Identity and versioning

Material artifacts MUST carry stable schema identity plus content/version
provenance sufficient for replay. Human-readable handles MAY be truncated for
display but MUST NOT replace full collision-resistant fingerprints where exact
identity matters.

### TR-004 — Serialization

JSON contracts MUST follow RFC 8259 interoperability expectations and the
repository's strict-validation policy. Structured generated output SHOULD use
JSON Schema Draft 2020-12 where schema exchange is required. Duplicate keys,
NaN/Infinity where forbidden by the contract, unknown fields in closed objects,
invalid UTF-8 and over-budget payloads MUST fail closed.

### TR-005 — Construction integrity

Package-managed immutable/factory-sealed artifacts MUST be replay-validated at
trust boundaries. A caller MUST NOT be able to forge a trusted object by direct
constructor mutation, stale fingerprint echoing or cross-request reuse.

## 4. Rust numerical requirements

### TR-006 — Source of truth

Production psychometric arithmetic belongs in `crates/mlsirm-core` or another
explicitly approved reusable Rust crate. Python performs no shadow production
likelihood merely to bypass a missing binding.

### TR-007 — Numeric safety

Every public or binding-reachable numerical kernel MUST validate shape,
dimensional compatibility, finite inputs where mathematically required, integer
ranges and bounded resource products before allocating large workspaces.

Overflow-prone products MUST use checked arithmetic or division-before-
multiplication guards. Resource ceilings MUST use explicit units (elements,
bytes, iterations, observations) and MUST NOT be described as process-RSS
limits unless that is actually measured.

### TR-008 — CPU multithreading

Parallel CPU kernels SHOULD use coarse work units and a fixed/bounded worker pool
appropriate to the operation. Avoid nested BLAS/Rayon/thread-pool
oversubscription and unnecessary context switching. Thread-count behavior MUST be
observable or reproducible where it materially affects benchmarks or results.

### TR-009 — GPU

GPU is a device implementation of an owned Rust numerical contract, not a new
statistical model. Any accepted GPU path MUST:

1. have a CPU/reference oracle;
2. define precision/tolerance semantics;
3. execute in CI/recovery evidence without being silently skipped when the gate
   claims GPU coverage;
4. preserve model/parameter/serialization identity;
5. fail or fall back according to an explicit documented policy.

## 5. Measurement-model requirements

### TR-010 — Current MLS2PLM scope

The existing implemented response term is the documented simple-structure
specialization:

```text
eta_pi = exp(alpha_i) * theta_p,d(i) + b_i - exp(tau) * r_pi
```

A full discrimination-vector MLS2PLM is a separate model-design change requiring
parameter-shape, likelihood, gradient, simulation, recovery, API and documentation
changes together.

### TR-011 — Model relation before selection

Model-comparison APIs MUST classify the actual relationship as one of at least:
regular nested, boundary nested, nonlinear-constraint nested, strictly
non-nested, overlapping/indistinguishable, or unknown. The relation MUST be based
on parameter constraints/identified distributions rather than model-name folklore.

### TR-012 — Relation-appropriate inference

- regular nested → appropriate LR/scaled-LR where regularity holds;
- boundary/singular nested → bootstrap or other boundary-aware reference;
- strictly non-nested/overlapping → formal distinguishability before selection;
- unknown/indistinguishable → no preferred model.

Held-out predictive evidence, residual dependence, DIF/invariance, scoreability
and recovery MUST supplement formal pairwise inference for product selection.

### TR-013 — Bifactor

Bifactor diagnostics MUST verify applicability of the declared general factor,
loading/uniqueness scale contracts and finite denominators. PUC MUST only be
returned for structures where its assumptions are satisfied. Any omega computed
from a latent-response/logit-slope transformation MUST be labelled as such and
MUST NOT be sold as observed categorical-score reliability.

### TR-014 — Rotation

Rotation criteria MUST implement common value/gradient contracts and keep
criterion semantics explicit. Orthogonal/oblique optimization, multi-start,
sign/permutation alignment, stationarity, basin evidence and target semantics
MUST be tested independently. Criterion objective values with unrelated scales
MUST NOT be directly ranked as if commensurate.

### TR-015 — Multilevel/multiple membership/time

Reusable contracts MUST preserve explicit context dimension, context identity,
membership weights, respondent/observation identity and occasion/version
identity. Weighted multiple memberships MUST be validated rather than silently
renormalized unless a particular API explicitly documents normalization.

A structurally valid contract does not imply an identified estimator. Before a
new multilevel/longitudinal likelihood becomes production, add Rust estimation,
realistic true-parameter bias/RMSE/coverage, convergence, missingness and
CPU/GPU evidence.

## 6. Automated scoring and LLM-as-a-Judge

### TR-016 — Shared observation contract

Human, LLM and external scorers SHOULD emit the same reusable scoring observation
contract with exact engine/rater identity, request/assessment/rubric identity,
criterion/task/occasion identity, terminal status and evidence/provenance.

Terminal states such as abstained, failed or excluded MUST remain missing/terminal
states and MUST NOT be converted to low scores.

### TR-017 — Judge diagnostics

The reusable validation layer SHOULD support, as data permit:

- severity and criterion-specific bias;
- rater discrimination/consistency;
- observed rating-range use/range restriction evidence;
- exact/adjacent agreement and QWK;
- absolute/relative agreement metrics where appropriate;
- subgroup SMD/DIF/invariance evidence;
- prompt/model/occasion drift;
- human-review triggers and uncertainty.

Pearson/Spearman correlation MAY be reported descriptively but MUST NOT be the
sole accuracy/validity gate.

### TR-018 — AI provider boundary

Provider/model SDK calls belong behind explicit protocols/adapters or in
contextual-orchestrator. Model-backed tests/actions use `NVIDIA_NIM_API_KEY` when
that is the approved organization path; `COPILOT_GITHUB_TOKEN` MUST NOT be used as
model authentication. Reviewer-agent credentials and model-execution credentials
remain separate.

## 7. Rubric and governed item bank

### TR-019 — Candidate-blind benchmark mode

For fair benchmark evaluation, rubric/criterion generation MUST be candidate-
blind with respect to the systems being scored, or use cross-fitting when
candidate-aware discovery is explicitly studied. The evidence regime must state
whether criteria are prompt-only, retrieved-context, pooled-corpus,
authoritative-corpus or human-anchor based.

### TR-020 — Screening

Generated candidates pass separate layers:

1. structural/schema/provenance validation;
2. evidence/source-span validation;
3. semantic answerability/ambiguity/alignment/distractor/redundancy/leakage
   screening;
4. pilot administration;
5. psychometric calibration and fit/DIF/local-dependence evaluation;
6. governance approval/version release.

A structurally valid generated JSON object is not a calibrated item.

### TR-021 — Lifecycle

Operational item/rubric versions are immutable. Revision creates a new version
and requires linking/anchor evidence for cross-version score comparison.
Calibration, exposure, DIF/drift, approval, suspension/quarantine, retirement and
rollback histories remain auditable.

## 8. Errors, privacy and security

### TR-022 — Stable errors

Public exceptions/errors SHOULD expose stable code and bounded caller-independent
path/context. They MUST NOT echo untrusted source text, provider output, secrets,
opaque sensitive identifiers or entire malformed payloads.

### TR-023 — Privacy controls

Avoid blanket PII masking where it prevents the intended measurement workflow.
Use minimization/purpose limitation, explicit access authorization, encryption,
selective disclosure, retention/deletion policy, pseudonymous/public identifiers,
auditable access and controlled egress. Raw sensitive data MUST NOT enter logs or
telemetry merely for convenience.

### TR-024 — Security/supply chain

Required gates include dependency/supply-chain scanning, SAST, fuzz/property
tests at parser boundaries, pinned/verified Actions and dependencies where
repository policy requires them, reproducible/SBOM/provenance evidence as the
release process matures, and no PR-controlled self-modifying write-capable CI.

## 9. Observability

Library and CLI evidence SHOULD expose:

- operation/model/backend/device identity;
- version/fingerprint/release identity;
- deterministic request/run correlation identity;
- iteration/convergence/failure state;
- worker/device configuration when relevant;
- bounded timing/allocation evidence for performance claims;
- no raw sensitive response text in metrics by default.

Operational telemetry owned by hosted products remains downstream.

## 10. Testing architecture

### TR-025 — Test types

- deterministic unit/contract tests;
- property/fuzz tests at untrusted boundaries;
- Rust/Python delegation and numerical parity tests;
- true-parameter recovery and known-oracle tests;
- model-selection true-structure simulations;
- realistic missingness/DIF/multilevel/longitudinal conditions;
- package/wheel reinstall/public-export tests;
- explicit GPU parity/no-skip tests for GPU gates;
- documentation/traceability contract tests for critical architecture boundaries.

### TR-026 — Coverage

Owned production Python and Rust surfaces target exact 100% meaningful
statement/branch/function/line-region coverage as applicable, plus complete public
docstrings/rustdoc. Tests MUST assert real behavior, not execute lines solely to
satisfy a percentage.

### TR-027 — Heavy studies

Expensive Monte Carlo/recovery studies MAY be separated from per-PR smoke into
scheduled/manual/release evidence, but no scientific claim may disappear merely
to shorten PR latency. Heavy-study acceptance criteria must be prospective or
methodologically justified, not tuned to one observed seed outcome.

## 11. Packaging, compatibility and release

### TR-028 — Package contract

A candidate release MUST build and reinstall the actual wheel/source artifact,
prove the compiled Rust backend/API imports, and exercise a release acceptance
smoke using the installed artifact rather than relying solely on an editable
source tree.

### TR-029 — Compatibility

Breaking schema/public-API changes require an explicit migration/deprecation or
major-version decision. Internal names that materially diverge from the current
product/domain terminology SHOULD be migrated with compatibility evidence rather
than retained indefinitely.

### TR-030 — Release proof

One release proof binds the exact artifact hashes to CI/security/scientific
acceptance, changelog, SBOM/provenance where available, buyer evidence and
rollback information. Stale/predecessor/synthetic evidence never substitutes for
that proof.

## 12. Architecture diagrams and data model

`docs/architecture/uml.md` contains component, sequence and state diagrams.
`docs/architecture/logical-data-model.md` is the canonical logical ERD. Because
this repository does not own a hosted DB, neither document authorizes product ORM
or persistence by itself.

## 13. Standards and primary technical references

- American Educational Research Association, American Psychological Association,
  & National Council on Measurement in Education. (2014). *Standards for
  educational and psychological testing*.
- National Institute of Standards and Technology. (2023). *Artificial
  intelligence risk management framework (AI RMF 1.0)* (NIST AI 100-1).
- National Institute of Standards and Technology. (2024). *Artificial
  intelligence risk management framework: Generative artificial intelligence
  profile* (NIST AI 600-1).
- International Organization for Standardization. (2023). *ISO/IEC 42001:2023
  Information technology—Artificial intelligence—Management system*.
- Bray, T. (2017). *The JavaScript Object Notation (JSON) data interchange
  format* (RFC 8259). RFC Editor. https://doi.org/10.17487/RFC8259
- JSON Schema. (2022). *JSON Schema Draft 2020-12*.

Method-specific mathematical references remain in `AGENTS.md` and authoritative
`docs/doctoring/` records rather than being duplicated incompletely here.
