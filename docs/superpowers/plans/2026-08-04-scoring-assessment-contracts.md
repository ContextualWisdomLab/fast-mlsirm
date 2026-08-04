# Automated-Scoring Assessment Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add content-addressed, factory-sealed automated-scoring assessment and policy contracts under `fast_mlsirm.scoring`.

**Architecture:** Reuse `fast_mlsirm.rubric` as the sole rubric source of truth. Add bounded JSON normalization, redacted structured errors, immutable policy/construct/rubric-binding values, and an `AssessmentSpec` factory. The slice performs no numerical estimation.

**Tech Stack:** Python 3.10+, frozen dataclasses, enums, standard-library JSON/SHA-256, existing rubric validators, pytest, branch coverage, and repository release gates.

## Global Constraints

- Remain inside `ContextualWisdomLab/fast-mlsirm`; do not create another repository or distribution.
- Use descriptive two-or-more-token lower `snake_case` identifiers.
- Reuse exact `RubricSpecification` fingerprints and do not duplicate rubric levels.
- Keep all psychometric arithmetic in existing Rust-backed APIs.
- Add complete public docstrings and 100% statement/branch coverage.
- Reject response/source text in metadata and expose only bounded redacted errors.
- Defer a version bump and authoritative release note until observation and scoring-engine protocol slices form a coherent shared-scoring-core release.

---

### Task 1: Write RED contract tests

**Files:**
- Create: `tests/test_scoring_contracts.py`

**Interfaces:**
- Expects: `fast_mlsirm.scoring` with `AssessmentSpec`, `ConstructSpec`, `PolicyDocument`, `PolicyKind`, `RubricBinding`, `ScoringContractError`, `build_assessment_spec`, and `build_policy_document`.
- Proves: deterministic identities, exact rubric bindings, policy completeness, resource bounds, seals, error safety, and public exports.

- [x] Add valid construct, rubric, policy, and assessment fixtures.
- [x] Add ordering/fingerprint/fresh-copy tests.
- [x] Add invalid identifiers, versions, construct references, duplicate IDs, policy-family, and collection-budget tests.
- [x] Add JSON depth/node/string/collection/encoded-size/non-finite/sensitive-field tests.
- [x] Add factory-seal and corrupted-internal-state tests.
- [x] Confirm the initial test commit was RED because `fast_mlsirm.scoring` did not exist.
- [x] Commit as `test(scoring): define assessment contract behavior`.

### Task 2: Add structured errors and bounded JSON

**Files:**
- Create: `python/fast_mlsirm/scoring/errors.py`
- Create: `python/fast_mlsirm/scoring/_json.py`

**Interfaces:**
- Produces: `ScoringContractError`, `contract_error`, `canonical_object_json`, and `decode_object_json`.

- [x] Implement bounded machine-readable error metadata.
- [x] Implement finite and resource-bounded canonical JSON normalization.
- [x] Reject descriptive-key violations and raw response/source-content fields.
- [x] Commit the bounded primitives as ordinary reviewed source.

### Task 3: Add assessment and policy contracts

**Files:**
- Create: `python/fast_mlsirm/scoring/contracts.py`
- Create: `python/fast_mlsirm/scoring/__init__.py`

**Interfaces:**
- Produces: `ConstructSpec`, `PolicyKind`, `PolicyDocument`, `RubricBinding`, `AssessmentSpec`, `build_policy_document`, and `build_assessment_spec`.

- [x] Implement construct normalization and content identity.
- [x] Implement factory-sealed policy documents with immutable canonical settings.
- [x] Implement factory-issued exact rubric bindings.
- [x] Implement canonical assessment assembly and complete policy-family checks.
- [x] Export only the documented package surface.
- [x] Prove 100% focused statement and branch coverage in the contract suite.

### Task 4: Add buyer documentation

**Files:**
- Create: `docs/automated_scoring_assessment_contracts.md`

**Interfaces:**
- Documents construction, canonical identities, trust boundaries, MSA embedding, and explicit non-goals.

- [x] Add a minimal provider-neutral example using exact rubric fingerprints.
- [x] Document that hashes are not authorization and that passing validation is not psychometric validity.
- [x] Document why the foundation does not claim an independent release.

### Task 5: Exact-head verification and PR

**Files:**
- Verify all changed files.

- [ ] Confirm focused scoring tests and repository-wide 100% coverage on the GitHub head.
- [ ] Confirm the full Python test suite.
- [ ] Confirm `cargo test --workspace` and PyO3 crate tests.
- [ ] Confirm package import, Security Scan, SAST, fuzz, and release acceptance.
- [x] Open a draft PR referencing issue #472.
- [ ] Inspect CodeRabbit/human review and required checks; address every actionable finding.
- [ ] Mark ready and merge only after the exact head satisfies repository policy.
