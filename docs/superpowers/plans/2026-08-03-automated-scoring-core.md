# Automated Scoring Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the operational automated-scoring foundation to the existing `fast-mlsirm` package under `fast_mlsirm.scoring` without creating a new repository or distribution.

**Architecture:** Introduce immutable, serializable scoring contracts and thin orchestration modules that reuse the repository's existing psychometric estimators. Provider-specific scorers implement a protocol; validation, adjudication, monitoring, and audit artifacts remain deterministic domain values. Any new numerical kernel must be implemented in Rust and exposed through the existing PyO3 backend.

**Tech Stack:** Python 3, dataclasses and typing protocols, NumPy, existing `fast_mlsirm` psychometric APIs, Rust/PyO3 where numerical computation is required, pytest and existing coverage tooling.

## Global Constraints

- Do not create a new repository or separately versioned package.
- The installation unit remains `fast-mlsirm`; public APIs use `fast_mlsirm.scoring`.
- Public functions, classes, protocols, and fields require complete docstrings.
- New code requires 100% statement and branch coverage.
- New numerical computation belongs in Rust with CPU multithreading and GPU support where applicable; Python only orchestrates.
- Identifiers are descriptive non-numeric strings.
- Missing, abstained, failed, and excluded observations remain distinct.
- Response text must not appear in exceptions, digests, or default logs.

---

### Task 1: Assessment and Rubric Contracts

**Files:**
- Create: `fast_mlsirm/scoring/__init__.py`
- Create: `fast_mlsirm/scoring/contracts.py`
- Create: `fast_mlsirm/scoring/rubric.py`
- Create: `tests/test_scoring_contracts.py`
- Modify: `fast_mlsirm/__init__.py`

**Interfaces:**
- Produces: `AssessmentSpec`, `ConstructSpec`, `RubricSpec`, `ScoreLevel`, policy dataclasses, `canonical_json()`, and `artifact_digest()`.

- [ ] **Step 1: Write failing contract tests**

Cover valid construction, non-numeric identifier rejection, duplicate score-level rejection, dangling construct references, non-finite metadata rejection, canonical serialization stability, and digest stability.

- [ ] **Step 2: Run the focused tests**

Run: `pytest tests/test_scoring_contracts.py -q`

Expected: collection or import failure because `fast_mlsirm.scoring` does not exist.

- [ ] **Step 3: Implement immutable contracts and validation**

Use frozen dataclasses and explicit `__post_init__` validation. Canonical JSON must sort mapping keys, preserve tuple order, encode enums by value, encode UTC timestamps in ISO 8601 form, and reject NaN and infinity.

- [ ] **Step 4: Export the public contract API**

Export names from `fast_mlsirm.scoring` and expose the namespace from the root package without importing optional providers.

- [ ] **Step 5: Verify coverage and commit**

Run: `pytest tests/test_scoring_contracts.py --cov=fast_mlsirm.scoring.contracts --cov=fast_mlsirm.scoring.rubric --cov-branch --cov-fail-under=100`

Commit: `feat(scoring): add assessment and rubric contracts`

---

### Task 2: Observation and Evidence Model

**Files:**
- Create: `fast_mlsirm/scoring/observations.py`
- Create: `tests/test_scoring_observations.py`
- Modify: `fast_mlsirm/scoring/__init__.py`

**Interfaces:**
- Consumes: `AssessmentSpec`, `RubricSpec`.
- Produces: `ScoreObservation`, `EvidenceSpan`, `ObservationStatus`, `RaterKind`, and `validate_observations()`.

- [ ] **Step 1: Write failing observation tests**

Cover human and automated observations, mandatory engine identity for automated ratings, forbidden engine identity for human ratings, explicit abstention/failure states, score-level validation, UTC timestamp validation, evidence offsets, and safe exception content.

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `pytest tests/test_scoring_observations.py -q`

- [ ] **Step 3: Implement the lossless observation schema**

Do not coerce missing or non-scored states into numeric values. Ensure `validate_observations()` checks assessment and rubric references while returning observations in input order.

- [ ] **Step 4: Verify full coverage and commit**

Run: `pytest tests/test_scoring_observations.py --cov=fast_mlsirm.scoring.observations --cov-branch --cov-fail-under=100`

Commit: `feat(scoring): add rating observation schema`

---

### Task 3: Scoring Engine Protocol and Deterministic Fixture

**Files:**
- Create: `fast_mlsirm/scoring/engines.py`
- Create: `tests/test_scoring_engines.py`
- Modify: `fast_mlsirm/scoring/__init__.py`

**Interfaces:**
- Produces: `ScoringEngine`, `EngineDescriptor`, `ScoringRequest`, `ScoringResult`, `DeterministicRuleEngine`.

- [ ] **Step 1: Write protocol contract tests**

Test deterministic descriptor serialization, declared capability checks, structured evidence, abstention, failure normalization, unsupported rubric rejection, and absence of provider-specific objects in outputs.

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_scoring_engines.py -q`

- [ ] **Step 3: Implement the protocol and fixture**

`DeterministicRuleEngine` must accept injected pure rules for tests and return only `ScoringResult`; rule exceptions become a failed result with a machine-readable code and no response text.

- [ ] **Step 4: Run coverage and commit**

Run: `pytest tests/test_scoring_engines.py --cov=fast_mlsirm.scoring.engines --cov-branch --cov-fail-under=100`

Commit: `feat(scoring): add scoring engine adapter contract`

---

### Task 4: Calibration Dataset Adapter

**Files:**
- Create: `fast_mlsirm/scoring/calibration.py`
- Create: `tests/test_scoring_calibration.py`
- Modify: `fast_mlsirm/scoring/__init__.py`

**Interfaces:**
- Consumes: `AssessmentSpec`, `ScoreObservation`, existing `fit_facets`, `fit`, linking, DIF, fit-diagnostic, and G-theory APIs.
- Produces: `CalibrationDataset`, `CalibrationModel`, `CalibrationResult`, `build_facets_dataset()`, and `calibrate_scores()`.

- [ ] **Step 1: Write hand-calculated matrix tests**

Construct a small crossed design and assert exact person, item, and rater index maps; exact 3D matrix placement; preservation of `NaN` missing cells; exclusion reasons; and connectedness metadata.

- [ ] **Step 2: Run focused tests to verify failure**

Run: `pytest tests/test_scoring_calibration.py -q`

- [ ] **Step 3: Implement dataset construction without a new estimator**

The first `calibrate_scores()` implementation dispatches only to existing stable public estimators. Reject unsupported combinations rather than silently approximating them.

- [ ] **Step 4: Add estimator monkeypatch tests**

Verify arguments sent to existing estimators, backend provenance, input digest, version fields, convergence propagation, and safe failure wrapping.

- [ ] **Step 5: Run coverage and commit**

Run: `pytest tests/test_scoring_calibration.py --cov=fast_mlsirm.scoring.calibration --cov-branch --cov-fail-under=100`

Commit: `feat(scoring): add calibration orchestration`

---

### Task 5: Validation and Fairness Gates

**Files:**
- Create: `fast_mlsirm/scoring/validation.py`
- Create: `tests/test_scoring_validation.py`
- Modify: `fast_mlsirm/scoring/__init__.py`

**Interfaces:**
- Consumes: validated observations, `CalibrationResult`, `ValidationPolicy`, and existing judge-validation, agreement, DIF, and fit functions.
- Produces: `ValidationStatus`, `MetricResult`, `ValidationReport`, and `validate_scoring_system()`.

- [ ] **Step 1: Write failing metric-state tests**

Cover `pass`, `fail`, and `insufficient_evidence`; exact and adjacent agreement; QWK; correlations; overall and subgroup SMD; human-human degradation; abstention/failure rates; and disconnected or undersized subgroup handling.

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_scoring_validation.py -q`

- [ ] **Step 3: Implement evidence-aware gate evaluation**

Each metric records its numerator, denominator, threshold, direction, status, and reason. Missing evidence cannot produce a pass.

- [ ] **Step 4: Add deterministic report serialization tests**

Ensure stable ordering by policy declaration, canonical digest stability, and no response content in reports.

- [ ] **Step 5: Run coverage and commit**

Run: `pytest tests/test_scoring_validation.py --cov=fast_mlsirm.scoring.validation --cov-branch --cov-fail-under=100`

Commit: `feat(scoring): add validation and fairness gates`

---

### Task 6: Human-Machine Adjudication

**Files:**
- Create: `fast_mlsirm/scoring/adjudication.py`
- Create: `tests/test_scoring_adjudication.py`
- Modify: `fast_mlsirm/scoring/__init__.py`

**Interfaces:**
- Consumes: observations, validation results, calibration uncertainty, and `AdjudicationPolicy`.
- Produces: `AdjudicationTrigger`, `AdjudicationDecision`, and `route_for_adjudication()`.

- [ ] **Step 1: Write failing routing tests**

Cover score-distance, uncertainty, abstention, failure, missing evidence, sensitive subgroup, engine-version disagreement, multiple simultaneous triggers, and no-trigger decisions.

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_scoring_adjudication.py -q`

- [ ] **Step 3: Implement transparent policy evaluation**

Return all triggering rules in policy order. Never mutate or replace source observations.

- [ ] **Step 4: Run coverage and commit**

Run: `pytest tests/test_scoring_adjudication.py --cov=fast_mlsirm.scoring.adjudication --cov-branch --cov-fail-under=100`

Commit: `feat(scoring): add adjudication routing`

---

### Task 7: Drift Monitoring

**Files:**
- Create: `fast_mlsirm/scoring/monitoring.py`
- Create: `tests/test_scoring_monitoring.py`
- Modify: `fast_mlsirm/scoring/__init__.py`

**Interfaces:**
- Consumes: time-ordered observations, validation reports, calibration summaries, and `MonitoringPolicy`.
- Produces: `MonitoringWindow`, `DriftMetric`, `MonitoringAlert`, and `monitor_scoring_system()`.

- [ ] **Step 1: Write failing monitoring tests**

Cover score-distribution shifts, abstention and failure-rate shifts, engine-version transitions, rater-severity changes, subgroup drift, contract mismatches, minimum-window evidence, and stable-data no-alert behavior.

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_scoring_monitoring.py -q`

- [ ] **Step 3: Implement policy-declared deterministic detectors**

Use only formulas already available in the repository or simple descriptive comparisons. Any advanced detector requiring a new estimator must be deferred to a separately reviewed Rust task.

- [ ] **Step 4: Run coverage and commit**

Run: `pytest tests/test_scoring_monitoring.py --cov=fast_mlsirm.scoring.monitoring --cov-branch --cov-fail-under=100`

Commit: `feat(scoring): add scoring drift monitoring`

---

### Task 8: Audit Provenance and End-to-End Workflow

**Files:**
- Create: `fast_mlsirm/scoring/audit.py`
- Create: `tests/test_scoring_audit.py`
- Create: `tests/test_scoring_workflow.py`
- Modify: `fast_mlsirm/scoring/__init__.py`
- Modify: `README.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: every scoring artifact.
- Produces: `AuditEvent`, `AuditTrail`, and end-to-end examples from assessment definition through monitoring.

- [ ] **Step 1: Write append-only audit tests**

Cover actor, operation, timestamp, input/output digests, software versions, reason codes, event ordering, deterministic serialization, and rejection of sensitive payload fields.

- [ ] **Step 2: Write an end-to-end workflow test**

Create a specification, generate human and deterministic-engine observations, build a facets dataset, monkeypatch the estimator, validate gates, route one disagreement, monitor two windows, and assert linked digests across every artifact.

- [ ] **Step 3: Implement audit values and workflow exports**

Keep persistence outside library scope. Provide values suitable for JSON storage without adding a database dependency.

- [ ] **Step 4: Document the public workflow and limitations**

Document that the first release is an operational framework, not a bundled NLP, speech, handwriting, or multimodal scorer and not a regulated decision product.

- [ ] **Step 5: Update the changelog**

Add an unreleased entry listing assessment contracts, observation schema, engine protocol, calibration orchestration, validation gates, adjudication, monitoring, and audit provenance.

- [ ] **Step 6: Run complete verification**

Run:

```bash
pytest --cov=fast_mlsirm --cov-branch
cargo test --workspace
python -m compileall fast_mlsirm
```

Also run the repository's configured formatter, linter, type checker, documentation checks, and release acceptance commands.

- [ ] **Step 7: Commit**

Commit: `feat(scoring): complete automated scoring core workflow`
