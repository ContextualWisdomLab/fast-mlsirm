# Product Requirements Document — fast-mlsirm

**Status:** Authoritative component PRD  
**Product:** `ContextualWisdomLab/fast-mlsirm`  
**Last reviewed:** 2026-08-09

## 1. Product definition

`fast-mlsirm` is a reusable psychometric measurement and AI-evaluation
infrastructure component. It converts versioned measurement contracts and
observations into scientifically auditable calibration, fit, recovery, fairness,
model-selection and release evidence through Rust-first numerical kernels and a
stable Python API.

The product is **not** the hosted Psychometrics Commons application. Hosted
identity, participant sessions, consent, result lifecycle, product databases,
HTTP/admin APIs, billing, tenant authorization, reference web clients and
deployment composition are downstream concerns.

## 2. Vision

Make rigorous measurement infrastructure as reusable and operationally reliable
as a production database or ML serving library, while preserving the distinctions
between:

- observed scores and latent constructs;
- association and absolute estimation accuracy;
- model fit and score interpretability;
- human/LLM agreement and rater calibration;
- individual effects and multilevel/contextual effects;
- cross-sectional effects and change over time;
- generated content and validated measurement evidence;
- statistical information and business/policy criticality.

## 3. Primary users and consumers

1. **Psychometricians and quantitative researchers** who need Rust-accelerated
   IRT/MIRT/MLSIRM and adjacent measurement evidence with reproducible recovery.
2. **Assessment platform engineers** who need versioned assessment, rubric,
   scoring and calibration contracts without implementing psychometric kernels.
3. **AI/RAG evaluation teams** who need to treat LLM judges as fallible raters
   rather than ground truth and to calibrate multidimensional evaluation evidence.
4. **Automated-scoring teams** who need human/AI scoring observations,
   rater/facet analysis, validity/fairness evidence and review routing inputs.
5. **Item/rubric-development teams** who need a governed path from construct and
   rubric to blueprint, candidate, pilot, calibration and item-bank evidence.
6. **Downstream CWL products and independent third parties** that consume explicit
   versioned contracts and portable evidence artifacts.

## 4. Jobs to be done

### JTBD-1 — Fit and diagnose a measurement model

Given a validated response design, estimate an appropriate psychometric model,
quantify uncertainty/fit and export deterministic evidence without requiring the
consumer to own Rust or optimizer implementation details.

### JTBD-2 — Prove estimation accuracy

Given simulated or known-parameter data, show bias, MAE/RMSE, coverage,
convergence, response/information recovery and alignment-aware latent recovery.
Correlation alone is insufficient evidence.

### JTBD-3 — Compare competing structures safely

Given candidate dimensional, bifactor, higher-order, testlet, many-facet,
latent-space or related models, determine the actual relation between models,
apply relation-appropriate inference, validate out-of-sample behavior and fail
closed when evidence cannot identify a winner.

### JTBD-4 — Calibrate human and AI raters

Given ratings from human, LLM or other scoring engines, preserve exact rater,
criterion, task and occasion identities; estimate relevant rater effects where
supported; quantify agreement, range use, drift and subgroup behavior; and avoid
treating raw human/AI correlation as validity.

### JTBD-5 — Build governed assessment items from rubrics

Given a construct/rubric and approved evidence regime, compile bounded item
blueprints and generation contracts, validate untrusted candidates, pilot with
artificial/human crowds, calibrate items and retain lifecycle/linking evidence for
a governed item bank.

### JTBD-6 — Represent contextual and temporal measurement designs

When observations are nested, cross-classified, multiply affiliated, repeated or
rater/testlet structured, preserve that design so downstream estimation cannot
silently commit an atomistic fallacy. Broader event/trajectory analytics may be
handed to TEPP through explicit artifacts.

### JTBD-7 — Produce acquisition/procurement-grade release evidence

Given one exact release candidate, package the software and its acceptance,
benchmark, scientific, security, provenance and buyer-review evidence without
promoting stale/predecessor/synthetic results to release proof.

## 5. Buyer-visible workflows

### 5.1 Core psychometrics

```text
validated data/design
→ fit/simulate
→ diagnostics/recovery
→ model-selection evidence
→ portable result/report
```

### 5.2 Automated scoring / LLM-as-a-Judge

```text
AssessmentSpec + RubricSpecification
→ ScoringRequest
→ human/automated ScoreObservation[]
→ ScoringResult
→ many-facet/multidimensional calibration evidence
→ validity/fairness/review evidence
```

### 5.3 Rubric/item development

```text
construct + evidence regime
→ rubric
→ blueprint
→ generation contract
→ untrusted candidate validation
→ semantic screening
→ pilot
→ Rust calibration
→ governed item-bank release
→ linking/drift/exposure/retirement
```

## 6. Product requirements

### PR-001 — Independent installability

The package MUST remain independently installable and usable without
Psychometrics Commons or any other hosted product.

### PR-002 — Rust-first mathematical ownership

Production psychometric arithmetic MUST have a Rust source of truth. Python MAY
provide validated reference implementations for parity and fallback but MUST NOT
silently diverge from the owned mathematical contract.

### PR-003 — Reproducible versioned contracts

Assessment, rubric, scoring, generated-item and scientific evidence artifacts
MUST carry sufficient version/content identity to reproduce their interpretation.
Material contract revisions MUST create a new identity rather than mutate an
operational artifact in place.

### PR-004 — Scientific evidence beyond correlation

Any estimation/recovery claim MUST be supportable with scale/alignment-aware
absolute error and uncertainty evidence. Any automated-scoring validity claim
MUST distinguish association, agreement, calibration, rater effects, subgroup
behavior and intended score use.

### PR-005 — Relation-safe model selection

The product MUST NOT return a model winner from a test whose assumptions do not
match the actual nesting/boundary/overlap relationship. Unknown relation or
missing distinguishability evidence MUST produce an indeterminate outcome.

### PR-006 — Bifactor scoreability is not model selection

Bifactor fit and bifactor score interpretability MUST remain distinct decisions.
General/specific scores require applicable scoreability, determinacy and recovery
contracts; a better fit index alone does not authorize reporting a score.

### PR-007 — No universal rotation winner

Rotation APIs MUST describe finite multi-start solutions as best observed rather
than mathematically global unless globality is proved. Criterion selection MUST
use criterion-neutral recovery/stability/theory evidence rather than comparing
unrelated objective scales directly.

### PR-008 — Rater-aware AI evaluation

LLM judges and human raters MUST be representable as fallible measurement
instruments with explicit model/version/prompt/occasion provenance. Automated
judging MUST support abstention/failure and human-review boundaries.

### PR-009 — Context and time

Reusable contracts MUST support scientifically material contextual and temporal
structure or fail closed rather than silently flattening it. Estimation extensions
MUST demonstrate identification and true-parameter recovery before product claims.

### PR-010 — Untrusted artifact safety

Provider output, JSON, files and other external artifacts MUST be bounded,
strictly validated and safe against replay/provenance confusion, malformed
serialization, path/resource abuse and data reflection in stable errors.

### PR-011 — Governed item lifecycle

The target item/rubric system MUST support an immutable lifecycle equivalent to:
`draft → audited/screened → pilot → calibrated → approved → active → suspended
or quarantined → retired`, with linking, calibration history, drift/exposure and
rollback evidence.

### PR-012 — Exact-head software quality

Release evidence MUST come from one unchanged integrated head/artifact and cover
meaningful tests, public documentation, package/API compatibility, Rust/Python
integration, security, fuzzing and applicable CPU/GPU/scientific evidence.

### PR-013 — Modular ecosystem interoperability

CWL and third-party integrations MUST use explicit contracts or immutable
artifacts. No consumer MAY require a hidden direct database join or internal
implementation dependency.

### PR-014 — Documentation as a product contract

Material behavior MUST be traceable through PRD, TRD, ADR, architecture,
logical data model/ERD, diagrams, scientific doctoring, tests and release history.
Scattered implementation notes are not a substitute for an authoritative
baseline.

## 7. Non-goals and prohibited claims

The reusable component does not by itself provide or claim:

- hosted participant/session/consent/tenant lifecycle;
- identity/federation or product authorization;
- clinical diagnosis, treatment recommendation or other unvalidated high-stakes
  consequential decisions;
- validity merely because an AI score correlates with a human score;
- causal business utility from a psychometric score alone;
- a universally best measurement model, factor count or rotation criterion;
- Bayesian posterior inference unless a separately validated path is added;
- SOC 2, CSAP, ISO/IEC 42001 or other certification merely because controls are
  designed with those frameworks in mind.

## 8. Scientific quality objectives

For mathematical features, choose design-specific acceptance criteria before
looking at outcome seeds where feasible and record at least:

- bias and MAE/RMSE of recoverable parameters;
- interval/standard-error coverage where uncertainty is claimed;
- convergence/failure rates and failure modes;
- response-probability/information recovery where relevant;
- alignment/linking/rotation-aware latent recovery;
- realistic missingness, DIF, subgroup and structural conditions;
- CPU/reference parity and GPU parity where GPU is used;
- false-selection/model-selection recovery for comparison procedures.

## 9. Product and operational KPIs

KPIs are evidence families rather than universal hard-coded thresholds:

- scientific recovery error and interval coverage;
- reproducibility across seed/platform/device within declared tolerance;
- fraction of released APIs covered by exact contract and public documentation;
- statement/branch/docstring gate status;
- number of unresolved security/scientific release blockers;
- exact-head CI/package/security/recovery pass rate;
- calibration/item-development cycle time for supported workflows;
- item-bank version/linking/drift audit completeness;
- rater/LLM review coverage and adjudication rate;
- buyer evidence packet reproducibility from the released artifact.

Business valuation or a specific acquisition price is NOT inferred from a test
pass. Commercial value additionally requires adoption, recurring value, switching
cost, defensible IP, operational trust and measured customer outcomes.

## 10. Release acceptance

A releasable increment MUST have:

1. exact protected-head or exact release-artifact identity;
2. no unresolved P0 scientific/security correctness issue;
3. complete required CI/security/package/provenance gates;
4. true-parameter/recovery evidence for changed mathematical behavior;
5. current PRD/TRD/ADR/architecture/traceability and method doctoring;
6. compatible public API/schema evidence or an explicit migration/major-version
   decision;
7. changelog/release notes reflecting the shipped behavior;
8. independent review required by the repository's current policy.

## 11. Current product gaps to converge

Priority remains evidence-driven and is re-evaluated against protected main:

- complete Dynamic Evidence-Grounded Rubric / Governed Item Bank lifecycle;
- semantic candidate screening and artificial-crowd calibration orchestration;
- fuller joint many-facet/multidimensional/bifactor/testlet/latent-space pathways
  only where recovery and model comparison justify them;
- formal model-relation and distinguishability evidence across supported model
  families;
- reusable contextual/multiple-membership/longitudinal numerical models with
  Rust recovery evidence;
- canonical modular PyO3/export registry as numerical feature surfaces expand;
- more parity-proven GPU kernels for workloads where acceleration is material;
- stronger portable validation/adjudication/drift evidence for automated scoring;
- complete acquisition/procurement traceability without ambiguous valuation
  naming.

## 12. Research and standards basis

The product requirements follow the validity, fairness, reliability and intended-
use principles of the *Standards for Educational and Psychological Testing*
(AERA, APA, & NCME, 2014). Method-level implementations additionally cite their
primary psychometric sources in `AGENTS.md` and `docs/doctoring/`.

GenAI evaluation and automation controls should be mapped, where applicable, to
NIST AI RMF 1.0 (NIST, 2023), the Generative AI Profile NIST AI 600-1 (NIST,
2024), and ISO/IEC 42001:2023 as governance references without implying
certification.
