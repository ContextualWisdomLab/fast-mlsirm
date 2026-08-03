# Rubric Blueprint Compiler Implementation Plan

> **For agentic workers:** Use reviewable TDD commits. Do not mark the pull request ready until the exact head passes all required checks and review findings are resolved.

**Goal:** Build a provider-neutral, auditable compiler that transforms strict versioned scoring rubrics into deterministic item-generation blueprints and canonical generation contracts.

**Architecture:** Add an isolated `fast_mlsirm.rubric` package with immutable schema objects, bounded deterministic compilation, complete content provenance, and strict provider-neutral structured-output contracts. Python owns orchestration, validation, serialization, and hashing only; psychometric estimation remains in the Rust backend.

**Tech stack:** Python 3.10+ standard library, pytest, and the repository's existing maturin/Rust workspace and CI gates.

## Global constraints

- Public identifiers contain at least two lower `snake_case` tokens.
- `schema_version` and human-governed `rubric_version` are independent.
- This slice accepts schema version `1.0` and canonical numeric rubric semantic versions.
- Ordinal scores are contiguous integers beginning at zero.
- `items_per_cell` is 1–100; total compiled blueprints do not exceed 10,000.
- Seeds fit an unsigned 64-bit integer.
- Rubric, blueprint, and contract outputs expose complete SHA-256 fingerprints; short ids are display handles only.
- Output is deterministic canonical UTF-8 JSON.
- No provider SDK, network call, credential handling, or runtime dependency is added.
- No psychometric scoring, calibration, fit, DIF, or information arithmetic is implemented in Python.
- Added Python scope requires complete docstrings and 100% statement and branch coverage.
- Scientific references use APA 7th formatting.

---

## Task 1: Specify the public contract with RED tests

**Files:** `tests/test_rubric_authoring.py` and focused edge/provenance/schema tests.

- [x] Specify valid immutable rubric and level objects.
- [x] Specify deterministic rubric fingerprints.
- [x] Specify bounded plan compilation, ordering, ids, and provider seeds.
- [x] Specify full blueprint and contract fingerprints.
- [x] Specify independent rubric governance and wire-schema versions.
- [x] Specify fail-closed replay guards, including rubric-version mismatch.
- [x] Specify JSON Schema Draft 2020-12, typed answer keys, ordered score-level entries, and bounded provider output.
- [x] Preserve a RED test-only commit before production implementation.

## Task 2: Implement immutable rubric and blueprint models

**Files:** `python/fast_mlsirm/rubric/models.py`, `python/fast_mlsirm/rubric/__init__.py`.

- [x] Implement bounded text, collection, enum, integer, identifier, locale, fingerprint, schema-version, and semantic-version normalization.
- [x] Implement `RubricLevel`.
- [x] Implement `RubricSpecification` with `rubric_version`, `schema_version`, canonical serialization, and full fingerprint.
- [x] Implement `BlueprintPlan` with explicit work limits.
- [x] Implement `ItemBlueprint` with the exact rubric revision and full blueprint fingerprint.

## Task 3: Implement deterministic bounded compilation

**File:** `python/fast_mlsirm/rubric/compiler.py`.

- [x] Calculate the design matrix size before allocation and reject requests above 10,000 cells.
- [x] Preserve declared task-family, difficulty, evidence-mode, and replicate ordering.
- [x] Derive deterministic provider seeds from the complete design identity.
- [x] Derive the full blueprint fingerprint from normalized blueprint content.
- [x] Derive `item_blueprint_<16 hex>` from the full fingerprint without treating it as the complete audit identity.

## Task 4: Implement canonical generation contracts

**File:** `python/fast_mlsirm/rubric/contracts.py`.

- [x] Fail closed on rubric id, rubric version, fingerprint, response format, scoring levels, evidence requirements, or prohibited-pattern mismatch.
- [x] Build a closed, bounded JSON Schema Draft 2020-12 output contract.
- [x] Use response-format-specific typed answer-key objects with rationale.
- [x] Require every rubric score once in order using `prefixItems`.
- [x] Return full `contract_fingerprint` and readable `contract_id`.
- [x] Produce byte-stable canonical JSON and a provider-neutral prompt boundary.

## Task 5: Document the governed product boundary

**Files:** `docs/rubric_item_generation.md`, design specification, README, changelog fragment.

- [x] Document rubric governance versions and downstream invalidation.
- [x] Document full fingerprints versus short display ids.
- [x] Document response-format-specific answer-key objects and required cross-field validation.
- [x] Document that structural conformance is not psychometric validity.
- [x] Document the Python orchestration/Rust psychometric boundary.
- [x] Document the provider, screening, calibration, item-bank, and reporting roadmap.

## Task 6: Same-head verification and merge gate

- [ ] Run all focused rubric tests on the final unchanged head.
- [ ] Run the full Python suite, 100% coverage gate, and docstring gate.
- [ ] Run `cargo test --workspace` and the PyO3 crate suite.
- [ ] Run packaging/import verification.
- [ ] Require CI, Security Scan, SAST Semgrep, and all branch-protection checks to succeed on the same head.
- [ ] Inspect all review threads and current-head comments; add a test-first correction for every actionable issue.
- [ ] Mark the PR ready only after the full gate is clean.
- [ ] Enable native auto-merge and confirm `merged_at` before reporting completion.
- [ ] Re-inventory the queue and begin the next highest-value buyer gap: bounded candidate screening and provider protocol, unless a check or review failure takes precedence.
