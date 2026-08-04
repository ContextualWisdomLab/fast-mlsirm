# Automated-Scoring Assessment Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add content-addressed, factory-sealed automated-scoring assessment and policy contracts under `fast_mlsirm.scoring`.

**Architecture:** Reuse `fast_mlsirm.rubric` as the sole rubric source of truth. Add bounded JSON normalization, redacted structured errors, immutable policy/construct/rubric-binding values, and an `AssessmentSpec` factory. The slice performs no numerical estimation.

**Tech Stack:** Python 3.10+, frozen dataclasses, enums, standard-library JSON/SHA-256, existing rubric validators, pytest, branch coverage, and repository changelog tooling.

## Global Constraints

- Remain inside `ContextualWisdomLab/fast-mlsirm`; do not create another repository or distribution.
- Use descriptive two-or-more-token lower `snake_case` identifiers.
- Reuse exact `RubricSpecification` fingerprints and do not duplicate rubric levels.
- Keep all psychometric arithmetic in existing Rust-backed APIs.
- Add complete public docstrings and 100% statement/branch coverage.
- Reject response/source text in metadata and expose only bounded redacted errors.
- Render the authoritative changelog fragment into `CHANGELOG.md` on the same branch.

---

### Task 1: Write RED contract tests

**Files:**
- Create: `tests/test_scoring_contracts.py`

**Interfaces:**
- Expects: `fast_mlsirm.scoring` with `AssessmentSpec`, `ConstructSpec`, `PolicyDocument`, `PolicyKind`, `RubricBinding`, `ScoringContractError`, `build_assessment_spec`, and `build_policy_document`.
- Proves: deterministic identities, exact rubric bindings, policy completeness, resource bounds, seals, error safety, and public exports.

- [ ] Add valid construct, rubric, policy, and assessment fixtures.
- [ ] Add ordering/fingerprint/fresh-copy tests.
- [ ] Add invalid identifiers, versions, construct references, duplicate IDs, policy-family, and collection-budget tests.
- [ ] Add JSON depth/node/string/collection/encoded-size/non-finite/sensitive-field tests.
- [ ] Add factory-seal and corrupted-internal-state tests.
- [ ] Run `pytest tests/test_scoring_contracts.py -q` and confirm import failure because the package does not exist.
- [ ] Commit as `test(scoring): define assessment contract behavior`.

### Task 2: Add structured errors and bounded JSON

**Files:**
- Create: `python/fast_mlsirm/scoring/errors.py`
- Create: `python/fast_mlsirm/scoring/_json.py`

**Interfaces:**
- Produces: `ScoringContractError`, `contract_error`, `canonical_object_json`, and `decode_object_json`.

- [ ] Implement bounded machine-readable error metadata.
- [ ] Implement finite and resource-bounded canonical JSON normalization.
- [ ] Reject descriptive-key violations and raw response/source-content fields.
- [ ] Run the focused tests and confirm remaining failures are missing contracts.
- [ ] Commit as `feat(scoring): add bounded contract primitives`.

### Task 3: Add assessment and policy contracts

**Files:**
- Create: `python/fast_mlsirm/scoring/contracts.py`
- Create: `python/fast_mlsirm/scoring/__init__.py`

**Interfaces:**
- Produces: `ConstructSpec`, `PolicyKind`, `PolicyDocument`, `RubricBinding`, `AssessmentSpec`, `build_policy_document`, and `build_assessment_spec`.

- [ ] Implement construct normalization and content identity.
- [ ] Implement factory-sealed policy documents with immutable canonical settings.
- [ ] Implement factory-issued exact rubric bindings.
- [ ] Implement canonical assessment assembly and complete policy-family checks.
- [ ] Export only the documented package surface.
- [ ] Run `pytest tests/test_scoring_contracts.py --cov=fast_mlsirm.scoring --cov-branch --cov-fail-under=100 -q` and confirm 100% statement/branch coverage.
- [ ] Commit as `feat(scoring): add assessment and policy contracts`.

### Task 4: Add buyer documentation and changelog evidence

**Files:**
- Create: `docs/automated_scoring_assessment_contracts.md`
- Create: `docs/changelog.d/472-scoring-assessment-contracts.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Documents construction, canonical identities, trust boundaries, MSA embedding, and explicit non-goals.

- [ ] Add a minimal provider-neutral example using exact rubric fingerprints.
- [ ] Document that hashes are not authorization and that passing validation is not psychometric validity.
- [ ] Add and render the authoritative changelog fragment.
- [ ] Run `python scripts/render_changelog_fragments.py --check CHANGELOG.md`.
- [ ] Commit as `docs(scoring): describe assessment contract boundary`.

### Task 5: Exact-head verification and PR

**Files:**
- Verify all changed files.

- [ ] Run focused scoring tests with 100% coverage.
- [ ] Run the full Python test suite.
- [ ] Run `cargo test --workspace` and the PyO3 crate tests.
- [ ] Verify package import and changelog render parity.
- [ ] Open a draft PR referencing issue #472.
- [ ] Inspect CodeRabbit/human review and required checks; address every actionable finding.
- [ ] Mark ready and merge only after the exact head satisfies repository policy.
