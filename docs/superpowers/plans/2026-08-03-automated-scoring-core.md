# Automated Scoring Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` or `superpowers:executing-plans` to
> implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for
> tracking.

**Goal:** Add the operational automated-scoring foundation to the existing
`fast-mlsirm` package under `fast_mlsirm.scoring` without creating a new
repository or distribution.

**Architecture:** Reuse `fast_mlsirm.rubric` as the only rubric and score-level
source of truth. Add immutable scoring contracts and thin orchestration modules
that reuse existing psychometric estimators. Provider-specific scorers
implement a protocol; validation, adjudication, monitoring, and audit artifacts
remain deterministic domain values. Any new numerical kernel must be
implemented in Rust and exposed through the existing PyO3 backend.

**Tech Stack:** Python 3.10+ under `python/fast_mlsirm`, immutable dataclasses,
typing protocols, NumPy only where existing orchestration requires it, the
existing Rust/PyO3 workspace, pytest, and repository coverage and release
gates.

## Global Constraints

- Do not create a new repository or separately versioned package.
- The installation unit remains `fast-mlsirm`; public APIs use
  `fast_mlsirm.scoring`.
- Do not create `fast_mlsirm.scoring.rubric` or a duplicate `RubricSpec`.
  Consume `fast_mlsirm.rubric.RubricSpecification`, `RubricLevel`,
  `ResponseFormat`, and their fingerprints.
- Public functions, classes, protocols, and modules require complete
  docstrings.
- New code requires 100% statement and branch coverage.
- New numerical computation belongs in Rust with CPU multithreading and GPU
  support where applicable; Python only validates, marshals, and orchestrates.
- Identifiers use descriptive two-or-more-token lower `snake_case`; numeric-only
  identifiers are rejected.
- Missing, abstained, failed, and excluded observations remain distinct.
- Response text must not appear in exceptions, digests, or default logs.
- Exact rubric fingerprints are carried through requests, observations,
  calibration, validation, adjudication, monitoring, and audit artifacts.

---

### Task 1: Assessment and Policy Contracts

**Files:**
- Create: `python/fast_mlsirm/scoring/__init__.py`
- Create: `python/fast_mlsirm/scoring/contracts.py`
- Create: `tests/test_scoring_contracts.py`
- Modify: `python/fast_mlsirm/__init__.py` only when a root-level namespace
  export is required and does not import optional providers.

**Interfaces:**
- Consumes: `fast_mlsirm.rubric.RubricSpecification` and its fingerprint.
- Produces: `AssessmentSpec`, `ConstructSpec`, policy dataclasses,
  `canonical_json()`, and `artifact_digest()`.

- [ ] **Step 1: Write failing contract tests**

Cover valid construction, descriptive identifier enforcement, unknown rubric
fingerprints, rubric-to-construct mismatches, dangling policy references,
non-finite metadata rejection, canonical serialization stability, and digest
stability.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest tests/test_scoring_contracts.py -q
```

Expected: import failure because `fast_mlsirm.scoring` does not exist.

- [ ] **Step 3: Implement immutable contracts and validation**

Use frozen dataclasses and explicit `__post_init__` validation. Canonical JSON
sorts mapping keys, preserves tuple order, encodes enums by value, encodes UTC
timestamps in ISO 8601 form, and rejects NaN and infinity. An assessment stores
rubric fingerprints and validates them against an explicit rubric registry.

- [ ] **Step 4: Export the public contract API**

Export names from `fast_mlsirm.scoring`. Do not import hosted-provider SDKs or
optional provider adapters.

- [ ] **Step 5: Verify and commit**

```bash
pytest tests/test_scoring_contracts.py \
  --cov=fast_mlsirm.scoring.contracts \
  --cov-branch --cov-fail-under=100
```

Commit: `feat(scoring): add assessment and policy contracts`

---

### Task 2: Observation and Evidence Model

**Files:**
- Create: `python/fast_mlsirm/scoring/observations.py`
- Create: `tests/test_scoring_observations.py`
- Modify: `python/fast_mlsirm/scoring/__init__.py`

**Interfaces:**
- Consumes: `AssessmentSpec` and a registry of
  `fast_mlsirm.rubric.RubricSpecification` values.
- Produces: `ScoreObservation`, `EvidenceSpan`, `ObservationStatus`,
  `RaterKind`, and `validate_observations()`.

- [ ] **Step 1: Write failing observation tests**

Cover human and automated observations, mandatory engine identity for automated
ratings, forbidden engine identity for human ratings, exact rubric fingerprint
validation, explicit abstention/failure/exclusion states, score-level
validation, UTC timestamp validation, evidence offsets, input-order
preservation, and safe exception content.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_scoring_observations.py -q
```

- [ ] **Step 3: Implement the lossless observation schema**

Do not coerce missing or non-scored states into numeric values. A scored event
must use a level allowed by the exact rubric fingerprint; a non-scored event
must not carry a score. Exceptions must identify the failing field without
including response text.

- [ ] **Step 4: Verify and commit**

```bash
pytest tests/test_scoring_observations.py \
  --cov=fast_mlsirm.scoring.observations \
  --cov-branch --cov-fail-under=100
```

Commit: `feat(scoring): add rating observation schema`

---

### Task 3: Scoring Engine Protocol and Deterministic Fixture

**Files:**
- Create: `python/fast_mlsirm/scoring/engines.py`
- Create: `tests/test_scoring_engines.py`
- Modify: `python/fast_mlsirm/scoring/__init__.py`

**Interfaces:**
- Produces: `ScoringEngine`, `EngineDescriptor`, `ScoringRequest`,
  `ScoringResult`, and `DeterministicRuleEngine`.

- [ ] **Step 1: Write protocol contract tests**

Test deterministic descriptor serialization, declared response-format and
rubric-fingerprint capabilities, structured evidence, abstention, failure
normalization, unsupported-rubric rejection, and absence of provider-specific
objects in outputs.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_scoring_engines.py -q
```

- [ ] **Step 3: Implement the protocol and fixture**

`DeterministicRuleEngine` accepts injected pure rules for tests and returns only
`ScoringResult`. Rule exceptions become a failed result with a bounded
machine-readable code and no response text.

- [ ] **Step 4: Verify and commit**

```bash
pytest tests/test_scoring_engines.py \
  --cov=fast_mlsirm.scoring.engines \
  --cov-branch --cov-fail-under=100
```

Commit: `feat(scoring): add scoring engine adapter contract`

---

### Task 4: Calibration Dataset Adapter

**Files:**
- Create: `python/fast_mlsirm/scoring/calibration.py`
- Create: `tests/test_scoring_calibration.py`
- Modify: `python/fast_mlsirm/scoring/__init__.py`

**Interfaces:**
- Consumes: `AssessmentSpec`, exact rubric registry, `ScoreObservation`, and
  existing `fit_facets`, `fit`, linking, DIF, fit-diagnostic, and G-theory APIs.
- Produces: `CalibrationDataset`, `CalibrationModel`, `CalibrationResult`,
  `build_facets_dataset()`, and `calibrate_scores()`.

- [ ] **Step 1: Write hand-constructed matrix tests**

Build a small crossed design and assert exact response, item, and rater index
maps; exact 3D matrix placement; preservation of `NaN` missing cells; exclusion
reasons; rubric fingerprint provenance; and connectedness metadata.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_scoring_calibration.py -q
```

- [ ] **Step 3: Implement dataset construction without a new estimator**

The first `calibrate_scores()` dispatches only to existing stable public
estimators. Reject unsupported model/response combinations instead of silently
approximating them. Python constructs validated matrices; all statistical
estimation remains Rust-backed.

- [ ] **Step 4: Add estimator delegation tests**

Verify exact arguments sent to existing estimators, backend provenance, input
and rubric digests, package and Rust-core versions, convergence propagation,
and safe failure wrapping.

- [ ] **Step 5: Verify and commit**

```bash
pytest tests/test_scoring_calibration.py \
  --cov=fast_mlsirm.scoring.calibration \
  --cov-branch --cov-fail-under=100
```

Commit: `feat(scoring): add calibration orchestration`

---

### Task 5: Validation and Fairness Gates

**Files:**
- Create: `python/fast_mlsirm/scoring/validation.py`
- Create: `tests/test_scoring_validation.py`
- Modify: `python/fast_mlsirm/scoring/__init__.py`

**Interfaces:**
- Consumes: validated observations, `CalibrationResult`, `ValidationPolicy`,
  and existing judge-validation, agreement, DIF, and fit functions.
- Produces: `ValidationStatus`, `MetricResult`, `ValidationReport`, and
  `validate_scoring_system()`.

- [ ] **Step 1: Write failing metric-state tests**

Cover `pass`, `fail`, and `insufficient_evidence`; exact and adjacent agreement;
QWK; correlations; overall and subgroup SMD; human-human degradation;
abstention/failure rates; rubric mismatch; and disconnected or undersized
subgroup handling.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_scoring_validation.py -q
```

- [ ] **Step 3: Implement evidence-aware gate evaluation**

Each metric records its numerator, denominator, threshold, direction, status,
and reason. Missing or non-identifiable evidence cannot produce a pass. Reuse
existing Rust-backed statistical functions rather than recoding formulas in
Python.

- [ ] **Step 4: Add deterministic report tests**

Ensure stable ordering by policy declaration, canonical digest stability, exact
rubric and calibration provenance, and no response content in reports.

- [ ] **Step 5: Verify and commit**

```bash
pytest tests/test_scoring_validation.py \
  --cov=fast_mlsirm.scoring.validation \
  --cov-branch --cov-fail-under=100
```

Commit: `feat(scoring): add validation and fairness gates`

---

### Task 6: Human-Machine Adjudication

**Files:**
- Create: `python/fast_mlsirm/scoring/adjudication.py`
- Create: `tests/test_scoring_adjudication.py`
- Modify: `python/fast_mlsirm/scoring/__init__.py`

**Interfaces:**
- Consumes: observations, validation results, calibration uncertainty, exact
  rubric provenance, and `AdjudicationPolicy`.
- Produces: `AdjudicationTrigger`, `AdjudicationDecision`, and
  `route_for_adjudication()`.

- [ ] **Step 1: Write failing routing tests**

Cover score distance, uncertainty, abstention, failure, missing evidence,
sensitive subgroup, rubric mismatch, engine-version disagreement, multiple
simultaneous triggers, and no-trigger decisions.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_scoring_adjudication.py -q
```

- [ ] **Step 3: Implement transparent policy evaluation**

Return all triggering rules in policy order. Never mutate, replace, or collapse
source observations.

- [ ] **Step 4: Verify and commit**

```bash
pytest tests/test_scoring_adjudication.py \
  --cov=fast_mlsirm.scoring.adjudication \
  --cov-branch --cov-fail-under=100
```

Commit: `feat(scoring): add adjudication routing`

---

### Task 7: Drift Monitoring

**Files:**
- Create: `python/fast_mlsirm/scoring/monitoring.py`
- Create: `tests/test_scoring_monitoring.py`
- Modify: `python/fast_mlsirm/scoring/__init__.py`

**Interfaces:**
- Consumes: time-ordered observations, validation reports, calibration
  summaries, exact assessment/rubric/engine versions, and `MonitoringPolicy`.
- Produces: `MonitoringWindow`, `DriftMetric`, `MonitoringAlert`, and
  `monitor_scoring_system()`.

- [ ] **Step 1: Write failing monitoring tests**

Cover score-distribution shifts, abstention and failure-rate shifts,
engine-version transitions, rubric-fingerprint changes, rater-severity changes,
subgroup drift, contract mismatches, minimum-window evidence, and stable-data
no-alert behavior.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_scoring_monitoring.py -q
```

- [ ] **Step 3: Implement policy-declared deterministic detectors**

Use only formulas already available in the repository or simple declared
comparisons. Any detector requiring a new estimator is deferred to a separately
reviewed Rust task with recovery and CPU/GPU parity evidence where applicable.

- [ ] **Step 4: Verify and commit**

```bash
pytest tests/test_scoring_monitoring.py \
  --cov=fast_mlsirm.scoring.monitoring \
  --cov-branch --cov-fail-under=100
```

Commit: `feat(scoring): add scoring drift monitoring`

---

### Task 8: Audit Provenance and End-to-End Workflow

**Files:**
- Create: `python/fast_mlsirm/scoring/audit.py`
- Create: `tests/test_scoring_audit.py`
- Create: `tests/test_scoring_workflow.py`
- Modify: `python/fast_mlsirm/scoring/__init__.py`
- Modify: `README.md`
- Add: `docs/changelog.d/<issue>-automated-scoring-core.md`

**Interfaces:**
- Consumes: every scoring artifact.
- Produces: `AuditEvent`, `AuditTrail`, and an end-to-end example from
  assessment definition through monitoring.

- [ ] **Step 1: Write append-only audit tests**

Cover actor, operation, UTC timestamp, input/output digests, software versions,
rubric fingerprints, reason codes, event ordering, deterministic serialization,
and rejection of sensitive payload fields.

- [ ] **Step 2: Write an end-to-end workflow test**

Create a rubric and assessment, generate human and deterministic-engine
observations, build a facets dataset, delegate to the estimator, validate gates,
route one disagreement, monitor two windows, and assert linked digests across
every artifact.

- [ ] **Step 3: Implement audit values and workflow exports**

Keep persistence outside library scope. Provide JSON-compatible values without
adding a database dependency.

- [ ] **Step 4: Document the public workflow and limitations**

Document that the first delivery is an operational framework, not a bundled
NLP, speech, handwriting, or multimodal scorer and not a regulated decision
product.

- [ ] **Step 5: Add the changelog fragment**

Record assessment contracts, observation schema, engine protocol, calibration
orchestration, validation gates, adjudication, monitoring, and audit provenance.

- [ ] **Step 6: Run complete same-head verification**

```bash
pytest --cov=fast_mlsirm --cov-branch
cargo test --workspace
python -m compileall python/fast_mlsirm
python -m build
```

Also run the repository's formatter, linter, type checker, docstring,
security/SAST, release-acceptance, CPU multithreading, and GPU parity gates on
the exact unchanged head.

- [ ] **Step 7: Review, merge, and re-inventory**

Resolve every review thread, merge only after all required checks pass, then
re-inventory open PRs and issues before selecting the next product gap.

Commit: `feat(scoring): complete automated scoring core workflow`
