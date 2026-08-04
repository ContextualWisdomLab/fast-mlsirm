# Task-Revision Calibration Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent changed task or essay-prompt content from being silently pooled under one calibration item while preserving logical task identifiers for reporting.

**Architecture:** Advance only the scoring-request wire contract to schema `1.1`, require an exact task-revision digest at the shared boundary, and use that digest as the many-facet item axis. Add an explicit legacy request migration helper that reconstructs and verifies schema `1.0` artifacts but never guesses the missing revision.

**Tech Stack:** Python 3.11+, frozen dataclass contracts, canonical JSON/SHA-256 identities, NumPy tensor marshalling, existing Rust/PyO3 many-facet estimator, pytest, GitHub Actions.

## Global Constraints

- Python performs validation, provenance, migration, and tensor marshalling only; psychometric arithmetic remains in the existing Rust backend.
- All new public types, constants, functions, and tests have complete docstrings.
- Stable error codes use two-or-more-token lower snake_case and never reflect caller-controlled values.
- Exact task revisions are 64-character lowercase SHA-256 fingerprints.
- One logical task may have multiple revisions; one revision may map to only one logical task and family.
- No cross-revision linking is performed without an explicit governed linking policy.
- Documentation citations use APA 7th edition.
- The authoritative changelog fragment is rendered into `CHANGELOG.md`.

---

### Task 1: Pin the shared request revision contract

**Files:**
- Modify: `python/fast_mlsirm/scoring/execution.py`
- Modify: `python/fast_mlsirm/scoring/authorization.py`
- Modify: `python/fast_mlsirm/scoring/contracts.py`
- Modify: `python/fast_mlsirm/scoring/__init__.py`
- Modify: `tests/scoring_execution_fixtures.py`
- Modify: `tests/test_scoring_execution_contracts.py`
- Modify: `tests/test_scoring_execution_authorization.py`
- Modify: `tests/test_scoring_execution_authorization_integrity.py`
- Create: `tests/test_scoring_task_revision_identity.py`

**Interfaces:**
- Produces: `ScoringRequest.task_revision_fingerprint: str`
- Produces: `SCORING_REQUEST_SCHEMA_VERSION = "1.1"`
- Produces: `LEGACY_SCORING_REQUEST_SCHEMA_VERSION = "1.0"`
- Extends: both `build_scoring_request(...)` layers with required `task_revision_fingerprint: str`

- [ ] **Step 1: Write failing request-contract tests**

Add tests that assert the exact revision is serialized, affects `request_fingerprint`, malformed digests fail with `invalid_task_revision_fingerprint`, and direct construction remains factory-sealed.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
pytest -q tests/test_scoring_task_revision_identity.py tests/test_scoring_execution_contracts.py tests/test_scoring_execution_authorization.py tests/test_scoring_execution_authorization_integrity.py
```

Expected: failures because the new field and builder parameter do not exist.

- [ ] **Step 3: Implement the request schema and call-site propagation**

Add the field immediately after `task_id`, validate it with `fingerprint`, include it in canonical content, and pass it through the authorization wrapper. Keep assessment and result schema constants unchanged.

- [ ] **Step 4: Run focused request tests and confirm GREEN**

Use the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/fast_mlsirm/scoring tests

git commit -m "feat(scoring): bind requests to exact task revisions"
```

### Task 2: Propagate essay prompt revisions

**Files:**
- Modify: `python/fast_mlsirm/scoring/essay/contracts.py`
- Modify: `tests/test_scoring_essay_contracts.py`
- Modify: `tests/test_scoring_task_revision_identity.py`

**Interfaces:**
- Consumes: `ScoringRequest.task_revision_fingerprint`
- Produces: `build_essay_scoring_request(...).scoring_request.task_revision_fingerprint == prompt.prompt_fingerprint`

- [ ] **Step 1: Write the metamorphic essay test**

Build two `EssayPrompt` values with the same `prompt_id` and different normalized content. Assert their shared requests retain the same logical `task_id` but have different revision and request fingerprints.

- [ ] **Step 2: Run the essay tests and confirm RED**

```bash
pytest -q tests/test_scoring_essay_contracts.py tests/test_scoring_task_revision_identity.py
```

Expected: revision propagation assertion fails.

- [ ] **Step 3: Pass the exact prompt fingerprint to the shared builder**

Set `task_revision_fingerprint=prompt.prompt_fingerprint` in `build_essay_scoring_request` while retaining existing domain metadata.

- [ ] **Step 4: Run the essay tests and confirm GREEN**

Use the Step 2 command.

- [ ] **Step 5: Commit**

```bash
git add python/fast_mlsirm/scoring/essay tests/test_scoring_essay_contracts.py tests/test_scoring_task_revision_identity.py

git commit -m "feat(scoring): propagate essay prompt revisions"
```

### Task 3: Make revisions the calibration item axis

**Files:**
- Modify: `python/fast_mlsirm/scoring/calibration.py`
- Modify: `python/fast_mlsirm/scoring/_calibration_validation.py`
- Modify: `tests/test_scoring_facets_calibration.py`
- Modify: `tests/test_scoring_facets_calibration_review.py`
- Modify: `tests/test_scoring_facets_fit_replay.py`
- Modify: `tests/test_scoring_facets_projection_replay.py`
- Modify: `tests/test_scoring_facets_respondent_indexing.py`
- Modify: `tests/test_scoring_facets_response_revision.py`
- Modify: `tests/test_scoring_task_revision_identity.py`

**Interfaces:**
- Produces: `ScoringFacetsRatingRecord.task_revision_fingerprint`
- Produces: `ScoringFacetsRatingRecord.task_family_id`
- Produces: `ScoringFacetsDesign.task_revision_fingerprints`
- Produces: aligned `task_ids`, `task_family_ids`, and `response_task_revision_fingerprints`
- Changes: tensor item index and graph edges to exact revision fingerprints

- [ ] **Step 1: Write failing axis and provenance tests**

Assert two prompt revisions sharing one logical ID occupy separate item-axis positions, logical metadata remains aligned, revision-to-logical collisions fail, and post-construction mutation is rejected before Rust delegation.

- [ ] **Step 2: Run focused calibration tests and confirm RED**

```bash
pytest -q tests/test_scoring_task_revision_identity.py tests/test_scoring_facets_*.py
```

Expected: missing record/design fields and old task-axis behavior.

- [ ] **Step 3: Implement record, design, graph, and replay changes**

Use `(respondent_id, task_revision_fingerprint, engine_fingerprint)` for cells. Build a deterministic revision map to logical task/family, retain response-level revision audit fields, and replay every new field in `_calibration_validation.py`.

- [ ] **Step 4: Run focused calibration tests and confirm GREEN**

Use the Step 2 command. Confirm no Python test invokes replacement numeric arithmetic.

- [ ] **Step 5: Commit**

```bash
git add python/fast_mlsirm/scoring/calibration.py python/fast_mlsirm/scoring/_calibration_validation.py tests

git commit -m "feat(scoring): index facets by exact task revision"
```

### Task 4: Add explicit schema 1.0 migration

**Files:**
- Create: `python/fast_mlsirm/scoring/migrations.py`
- Modify: `python/fast_mlsirm/scoring/contracts.py`
- Modify: `python/fast_mlsirm/scoring/__init__.py`
- Modify: `tests/test_scoring_task_revision_identity.py`

**Interfaces:**
- Produces: `migrate_scoring_request_v1(artifact, *, assessment, rubric, task_revision_fingerprint) -> ScoringRequest`

- [ ] **Step 1: Write failing migration tests**

Create a genuine schema `1.0` artifact fixture by removing the revision field from normalized `1.1` content, setting schema `1.0`, and recomputing the legacy digest. Test successful migration, explicit revision requirement, field/fingerprint/handle tamper rejection, and changed request identity.

- [ ] **Step 2: Run the migration tests and confirm RED**

```bash
pytest -q tests/test_scoring_task_revision_identity.py -k migration
```

Expected: import or missing-function failure.

- [ ] **Step 3: Implement bounded fail-closed migration**

Validate exact mapping shape, separate and verify package-managed authorization metadata, rebuild through the current public factory, derive the legacy canonical content by dropping the revision field and restoring schema `1.0`, verify legacy fingerprint and handle, and return the new request.

- [ ] **Step 4: Run migration and full focused tests**

```bash
pytest -q tests/test_scoring_task_revision_identity.py tests/test_scoring_execution_*.py tests/test_scoring_essay_contracts.py tests/test_scoring_facets_*.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add python/fast_mlsirm/scoring/migrations.py python/fast_mlsirm/scoring/contracts.py python/fast_mlsirm/scoring/__init__.py tests/test_scoring_task_revision_identity.py

git commit -m "feat(scoring): migrate legacy task identities explicitly"
```

### Task 5: Document, render changelog, and verify release gates

**Files:**
- Create: `docs/scoring_task_revision_identity.md`
- Create: `docs/changelog.d/scoring-task-revision-identity.md`
- Modify: `docs/scoring_execution_contracts.md`
- Modify: `docs/scoring_facets_calibration_handoff.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Documents: shared field, migration, estimator axis, non-linking boundary, and APA 7th evidence.

- [ ] **Step 1: Write product and migration documentation**

Explain that a digest prevents accidental equality but does not establish comparability. Include Standards 5.7, 5.12–5.15 implications and the empirical references from the design.

- [ ] **Step 2: Render and check the authoritative changelog**

```bash
python3 scripts/render_changelog_fragments.py --update CHANGELOG.md
python3 scripts/render_changelog_fragments.py --check CHANGELOG.md
```

- [ ] **Step 3: Run exact verification**

```bash
pytest -q
cargo test --workspace --all-targets
python3 scripts/check_docstring_coverage.py
```

Also run repository CI, Security Scan, SAST, Rust/PyO3, and explicit GPU-no-skip workflows on the exact head.

- [ ] **Step 4: Request review and address only verified findings**

Request CodeRabbit review, inspect every unresolved thread, implement valid corrections, rerun exact-head gates, and preserve the one-revision-one-item boundary.

- [ ] **Step 5: Commit and merge under repository policy**

```bash
git add docs CHANGELOG.md

git commit -m "docs(scoring): explain task revision calibration identity"
```

Open a draft PR, mark ready only after exact-head checks and review are green, then squash merge with `Closes #499`.
