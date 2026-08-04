# Scoring Observation and Engine Execution Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the provider-neutral request, evidence, score-observation, engine, and execution contracts needed to connect merged `AssessmentSpec` artifacts to later Rust-backed calibration workflows.

**Architecture:** Add focused `observations.py`, `engines.py`, and `executions.py` modules under the existing `fast_mlsirm.scoring` namespace. Every artifact is immutable, factory-sealed, content-addressed, and bound to the exact assessment/rubric identity graph; Python validates and marshals only, with no psychometric arithmetic.

**Tech Stack:** Python 3.10+, frozen dataclasses, standard-library `enum`, `typing.Protocol`, deterministic JSON/SHA-256 helpers from `fast_mlsirm.scoring`, pytest, existing GitHub Actions quality gates.

## Global Constraints

- Work only inside `ContextualWisdomLab/fast-mlsirm`; create no new repository or distribution.
- `fast_mlsirm.rubric.RubricSpecification` remains the only rubric source of truth.
- All persisted identifiers are descriptive, nonnumeric, two-or-more-token lower `snake_case`.
- Public artifacts use complete SHA-256 fingerprints and descriptive 128-bit handles.
- New public functions/classes require complete docstrings.
- Added code requires 100% statement and branch coverage.
- Ordinary callback/provider exceptions are redacted as structured domain errors; `BaseException` is not swallowed.
- No raw response, essay, source, prompt, API-key, or provider-error payload is stored or reflected.
- No likelihood, gradient, Hessian, reliability, DIF, utility, or other psychometric arithmetic is implemented in Python.
- `CHANGELOG.md` must be rendered from the authoritative fragment before readiness.
- Same-head Python, Rust, PyO3, package, GPU-no-skip, fuzz, Security Scan, SAST, review, and unresolved-thread gates are mandatory.

---

### Task 1: Establish the public RED contract

**Files:**
- Create: `tests/scoring_execution_fixtures.py`
- Create: `tests/test_scoring_observation_contracts.py`
- Create: `tests/test_scoring_engine_contracts.py`
- Create: `tests/test_scoring_execution_contracts.py`
- Modify: `python/fast_mlsirm/scoring/__init__.py`

**Interfaces:**
- Consumes: merged `AssessmentSpec`, `AssessmentResponseType`, `RubricSpecification`, `AssessmentSpecError`, `artifact_digest`.
- Produces: failing imports for all issue #480 public names and reusable real assessment/rubric fixtures.

- [ ] **Step 1: Write the fixture module**

Create a real two-level binary rubric, a five-level ordinal rubric, and a merged `AssessmentSpec` containing both constructs and exact fingerprints. Provide factory functions rather than mutable module-level containers.

- [ ] **Step 2: Write failing import and enum tests**

Assert the package root exposes:

```python
from fast_mlsirm.scoring import (
    BooleanScoreValue,
    ContinuousScoreValue,
    EngineDescriptor,
    EvidenceReference,
    EvidenceRole,
    ExecutionState,
    IntegerScoreValue,
    NominalScoreValue,
    ObservationState,
    PairwiseOutcome,
    PairwiseScoreValue,
    ScoreObservation,
    ScoringContractError,
    ScoringEngine,
    ScoringExecution,
    ScoringRequest,
    StaticScoringEngine,
    build_engine_descriptor,
    build_evidence_reference,
    build_score_observation,
    build_scoring_execution,
    build_scoring_request,
    execute_scoring_request,
)
```

- [ ] **Step 3: Run the focused tests and prove RED**

Run:

```bash
pytest tests/test_scoring_observation_contracts.py \
       tests/test_scoring_engine_contracts.py \
       tests/test_scoring_execution_contracts.py -q
```

Expected: import failures for the missing public contract.

- [ ] **Step 4: Commit the RED contract**

```bash
git add tests/scoring_execution_fixtures.py \
        tests/test_scoring_observation_contracts.py \
        tests/test_scoring_engine_contracts.py \
        tests/test_scoring_execution_contracts.py
git commit -m "test(scoring): define observation and execution contracts"
```

### Task 2: Implement evidence and score-value primitives

**Files:**
- Create: `python/fast_mlsirm/scoring/observations.py`
- Modify: `python/fast_mlsirm/scoring/__init__.py`
- Test: `tests/test_scoring_observation_contracts.py`

**Interfaces:**
- Consumes: shared canonical helpers and structured assessment errors.
- Produces: `ObservationState`, `EvidenceRole`, `PairwiseOutcome`, `EvidenceReference`, five score-value types, and sealed builders.

- [ ] **Step 1: Add RED truth-table and evidence tests**

Cover:

- all enum values;
- exact-type Boolean and integer behavior;
- full signed-64-bit integer range;
- finite continuous values and negative-zero normalization;
- descriptive nominal categories;
- pairwise tie/winner consistency;
- offset all-or-none and ordering rules;
- evidence fingerprint format, duplicate identity, immutability, and deterministic digest;
- hostile conversion, iterator, Unicode, and oversize inputs with non-reflective errors.

- [ ] **Step 2: Run tests and verify RED**

```bash
pytest tests/test_scoring_observation_contracts.py -q
```

- [ ] **Step 3: Implement immutable primitives**

Use frozen dataclasses with private factory tokens. Every builder must validate before allocation and compute full canonical identity from complete content. Do not accept caller serializer hooks.

- [ ] **Step 4: Run focused coverage**

```bash
pytest tests/test_scoring_observation_contracts.py \
  --cov=fast_mlsirm.scoring.observations \
  --cov-branch --cov-fail-under=100 -q
```

- [ ] **Step 5: Commit**

```bash
git add python/fast_mlsirm/scoring/observations.py \
        python/fast_mlsirm/scoring/__init__.py \
        tests/test_scoring_observation_contracts.py
git commit -m "feat(scoring): add evidence and score observation primitives"
```

### Task 3: Implement assessment-bound requests and observations

**Files:**
- Modify: `python/fast_mlsirm/scoring/observations.py`
- Test: `tests/test_scoring_observation_contracts.py`
- Test: `tests/test_scoring_contract_graph_replay.py`

**Interfaces:**
- Consumes: `AssessmentSpec`, exact rubric registry, primitive score values and evidence references.
- Produces: `ScoringRequest`, `ScoreObservation`, `build_scoring_request`, and `build_score_observation`.

- [ ] **Step 1: Write RED graph-binding tests**

Test exact rejection of:

- unknown assessment fingerprint;
- undeclared construct;
- rubric not assigned to construct;
- changed respondent/item/rater/occasion identity;
- modified request content with copied request ID;
- observed state without score;
- non-observed state with score;
- missing or forbidden reason code by state;
- failed state carrying evidence;
- score category not declared by the rubric;
- pairwise score without request option identities;
- criterion-level versus holistic mismatch.

- [ ] **Step 2: Run and verify RED**

```bash
pytest tests/test_scoring_observation_contracts.py \
       tests/test_scoring_contract_graph_replay.py -q
```

- [ ] **Step 3: Implement sealed builders**

Builders receive the actual `AssessmentSpec` and rubric registry. They recompute all identities and copy only exact fingerprints into artifacts. Reason/failure codes use descriptive identifiers. Evidence collection is bounded before materialization and deduplicated by complete fingerprint.

- [ ] **Step 4: Run focused tests and coverage**

```bash
pytest tests/test_scoring_observation_contracts.py \
       tests/test_scoring_contract_graph_replay.py \
  --cov=fast_mlsirm.scoring.observations \
  --cov-branch --cov-fail-under=100 -q
```

- [ ] **Step 5: Commit**

```bash
git add python/fast_mlsirm/scoring/observations.py \
        tests/test_scoring_observation_contracts.py \
        tests/test_scoring_contract_graph_replay.py
git commit -m "feat(scoring): bind observations to assessment provenance"
```

### Task 4: Implement engine descriptors and deterministic fixture engine

**Files:**
- Create: `python/fast_mlsirm/scoring/engines.py`
- Modify: `python/fast_mlsirm/scoring/__init__.py`
- Test: `tests/test_scoring_engine_contracts.py`

**Interfaces:**
- Consumes: `ScoringRequest`, `ScoreObservation`, shared canonical helpers.
- Produces: `EngineDescriptor`, `ScoringEngine`, `StaticScoringEngine`, and descriptor builder.

- [ ] **Step 1: Add RED descriptor/protocol tests**

Cover:

- exact engine/model/prompt/code/data provenance;
- immutable metadata;
- invalid or missing fingerprints;
- independent engine identity under mapping input order;
- runtime protocol conformance;
- static fixture lookup by request fingerprint;
- unknown request rejection;
- mutation of fixture mappings after construction;
- callback/iterator failures and resource limits.

- [ ] **Step 2: Run and verify RED**

```bash
pytest tests/test_scoring_engine_contracts.py -q
```

- [ ] **Step 3: Implement provider-neutral engine contracts**

`ScoringEngine.score` returns an iterable. `StaticScoringEngine` freezes request-fingerprint mappings and returns immutable observation tuples. It contains no network or provider dependency.

- [ ] **Step 4: Run focused coverage**

```bash
pytest tests/test_scoring_engine_contracts.py \
  --cov=fast_mlsirm.scoring.engines \
  --cov-branch --cov-fail-under=100 -q
```

- [ ] **Step 5: Commit**

```bash
git add python/fast_mlsirm/scoring/engines.py \
        python/fast_mlsirm/scoring/__init__.py \
        tests/test_scoring_engine_contracts.py
git commit -m "feat(scoring): add provider-neutral engine contracts"
```

### Task 5: Implement execution validation and redacted failure artifacts

**Files:**
- Create: `python/fast_mlsirm/scoring/executions.py`
- Modify: `python/fast_mlsirm/scoring/__init__.py`
- Test: `tests/test_scoring_execution_contracts.py`
- Test: `tests/test_scoring_execution_security.py`

**Interfaces:**
- Consumes: `ScoringEngine`, `EngineDescriptor`, `ScoringRequest`, `ScoreObservation`.
- Produces: `ScoringExecution`, `build_scoring_execution`, and `execute_scoring_request`.

- [ ] **Step 1: Write RED execution tests**

Test:

- completed execution with deterministic observation ordering;
- empty success result rejection;
- maximum-result bound enforced during iteration;
- duplicate observation/rater-item rejection;
- cross-request, cross-assessment, cross-rubric, cross-construct, and wrong-rater results;
- provider exception before output;
- provider exception after partial output;
- iterator creation/advance failure;
- failed execution contains only stable failure code, no exception text;
- request, engine, observation, and execution replay/forgery mutation rejection;
- exact execution fingerprint and handle determinism.

- [ ] **Step 2: Run and verify RED**

```bash
pytest tests/test_scoring_execution_contracts.py \
       tests/test_scoring_execution_security.py -q
```

- [ ] **Step 3: Implement fail-closed execution orchestration**

Catch ordinary `Exception` around engine invocation and result iteration, but re-raise existing structured scoring errors unchanged. Never publish partial output as completed evidence. Failed executions bind the original request and engine descriptors and carry only package-owned reason codes.

- [ ] **Step 4: Run focused coverage**

```bash
pytest tests/test_scoring_execution_contracts.py \
       tests/test_scoring_execution_security.py \
  --cov=fast_mlsirm.scoring.executions \
  --cov-branch --cov-fail-under=100 -q
```

- [ ] **Step 5: Commit**

```bash
git add python/fast_mlsirm/scoring/executions.py \
        python/fast_mlsirm/scoring/__init__.py \
        tests/test_scoring_execution_contracts.py \
        tests/test_scoring_execution_security.py
git commit -m "feat(scoring): add fail-closed engine executions"
```

### Task 6: Complete adversarial contract QA

**Files:**
- Create: `tests/test_scoring_execution_resource_bounds.py`
- Create: `tests/test_scoring_execution_callbacks.py`
- Modify: observation/engine/execution modules only where RED tests require.

**Interfaces:**
- Consumes: complete public slice.
- Produces: hostile-protocol, Unicode, cycle, identity, and resource-bound evidence.

- [ ] **Step 1: Add hostile-protocol RED tests**

Include custom objects whose `__iter__`, next step, mapping inspection, `__index__`, equality, or string callbacks raise private payload text. Assert stable codes/paths and non-reflection. Include dishonest `__len__`, infinite iterators, cycles, lone surrogates, oversized evidence/result sets, integer overflow, NaN/infinity, and sensitive metadata key variants.

- [ ] **Step 2: Run and verify RED where any boundary remains open**

```bash
pytest tests/test_scoring_execution_resource_bounds.py \
       tests/test_scoring_execution_callbacks.py -q
```

- [ ] **Step 3: Implement the minimum GREEN hardening**

Use package-owned exact-type checks, cycle-safe bounded materialization, and caller-independent index paths. Do not broaden catches to `BaseException`.

- [ ] **Step 4: Run all scoring tests with 100% coverage**

```bash
pytest tests/test_scoring_contract_*.py \
       tests/test_scoring_observation_contracts.py \
       tests/test_scoring_engine_contracts.py \
       tests/test_scoring_execution_contracts.py \
       tests/test_scoring_execution_security.py \
       tests/test_scoring_execution_resource_bounds.py \
       tests/test_scoring_execution_callbacks.py \
  --cov=fast_mlsirm.scoring --cov-branch --cov-fail-under=100 -q
```

- [ ] **Step 5: Commit**

```bash
git add python/fast_mlsirm/scoring tests/test_scoring_*.py
git commit -m "test(scoring): harden observation and execution boundaries"
```

### Task 7: Document the MSA and scientific boundary

**Files:**
- Create: `docs/scoring_observation_execution_contracts.md`
- Create: `docs/changelog.d/480-scoring-observation-execution.md`
- Modify: `CHANGELOG.md`
- Modify: PR body.

**Interfaces:**
- Consumes: final public APIs and supported/deferred boundary.
- Produces: buyer-facing contract guide and authoritative release note.

- [ ] **Step 1: Write buyer documentation**

Document the identity graph, state truth table, evidence provenance, static engine example, MSA boundary, and downstream calibration requirements. Include APA 7th references already used by the scoring assessment design: the 2014 testing standards and Williamson, Xi, and Breyer (2012).

- [ ] **Step 2: Add authoritative changelog fragment**

Use a level-one title and allowed release sections. State explicitly that the slice adds provenance/execution semantics but no scoring-validity claim.

- [ ] **Step 3: Render and verify changelog parity**

```bash
python scripts/render_changelog_fragments.py --update CHANGELOG.md
python scripts/render_changelog_fragments.py --check CHANGELOG.md
```

- [ ] **Step 4: Run docs and public-docstring tests**

```bash
pytest tests/test_changelog_fragment_contract.py \
       tests/test_scoring_observation_contracts.py \
       tests/test_scoring_engine_contracts.py \
       tests/test_scoring_execution_contracts.py -q
```

- [ ] **Step 5: Commit**

```bash
git add docs/scoring_observation_execution_contracts.md \
        docs/changelog.d/480-scoring-observation-execution.md \
        CHANGELOG.md
git commit -m "docs(scoring): document observation and execution contracts"
```

### Task 8: Exact-head integration, review, and merge readiness

**Files:**
- Modify only files required by verified failing checks or actionable review.

**Interfaces:**
- Consumes: complete PR head.
- Produces: one unchanged reviewed head eligible for merge.

- [ ] **Step 1: Run complete local verification**

```bash
pytest
cargo test --workspace
cargo test --manifest-path crates/fast-mlsirm-py/Cargo.toml
python -m build --no-isolation
```

- [ ] **Step 2: Push one stable head and inspect every required GitHub check**

Require CI, Security Scan, SAST, package, explicit GPU parity without skip, and fuzzing to succeed on the same head.

- [ ] **Step 3: Request final current-head review**

```text
@coderabbitai review
```

Inspect all human, CodeRabbit, security, and CI feedback. Address only valid findings with tests first. Resolve addressed threads.

- [ ] **Step 4: Re-run exact-head checks after any review fix**

Do not merge evidence from different heads.

- [ ] **Step 5: Merge only after policy is satisfied**

Use the repository's accepted squash-merge policy. Verify the returned merge SHA, `merged_at`, closure of #480, and a zero open-PR inventory before selecting the next product slice.