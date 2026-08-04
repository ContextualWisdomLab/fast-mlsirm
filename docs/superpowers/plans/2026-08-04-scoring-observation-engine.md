# Scoring Observation and Engine Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add governed scoring requests, observations, results, and a provider-neutral engine protocol on top of the merged `AssessmentSpec`.

**Architecture:** Reuse exact `AssessmentSpec` and `RubricSpecification` fingerprints, the merged fail-closed validation helpers, and immutable content-addressed contracts. Keep raw content, providers, and psychometric arithmetic outside this slice.

**Tech Stack:** Python 3.10+, frozen dataclasses, enums, `typing.Protocol`, standard-library JSON/SHA-256, existing scoring/rubric validators, pytest, branch coverage, and repository changelog tooling.

## Global Constraints

- Stay inside the existing `fast-mlsirm` package and repository.
- Use descriptive two-or-more-token lower `snake_case` identifiers.
- Bind exact assessment and rubric fingerprints rather than copying rubric definitions.
- Store no raw response, prompt, provider-output, essay, or source text.
- Keep all psychometric arithmetic in existing Rust-backed APIs.
- Add complete public docstrings and 100% statement/branch coverage.
- Reuse structured non-reflective errors and bounded canonicalization.
- Render the authoritative changelog fragment before readiness.

---

### Task 1: Add RED engine and observation tests

**Files:**
- Create: `tests/test_scoring_execution_contracts.py`
- Create: `tests/scoring_execution_fixtures.py`

- [ ] Define valid assessment, rubric, engine, request, observation, and result fixtures.
- [ ] Cover human/automated engine consistency and content identities.
- [ ] Cover request graph references, granularity, criterion sets, task family, score levels, content statistics, and metadata.
- [ ] Cover observation status/score/reason/evidence invariants and factory seals.
- [ ] Cover result coverage, request/engine mismatch, deterministic ordering, diagnostics, and fixture engine behavior.
- [ ] Cover runtime protocol, callback failures, UTF-8, signed-64, `-0.0`, and public exports.
- [ ] Confirm initial import failure before implementation.

### Task 2: Implement engine, request, observation, and result contracts

**Files:**
- Create: `python/fast_mlsirm/scoring/execution.py`
- Modify: `python/fast_mlsirm/scoring/contracts.py`
- Modify: `python/fast_mlsirm/scoring/__init__.py`

- [ ] Implement enums and `EvidenceReference`.
- [ ] Implement `EngineDescriptor` and factory.
- [ ] Implement factory-sealed `ScoringRequest` bound to exact assessment/rubric graphs.
- [ ] Implement factory-sealed `ScoreObservation` with status/score/reason invariants.
- [ ] Implement factory-sealed `ScoringResult` with exact observation coverage.
- [ ] Implement runtime-checkable `ScoringEngine` and deterministic `StaticFixtureEngine`.
- [ ] Export the documented public surface.
- [ ] Reach 100% focused statement and branch coverage.

### Task 3: Add buyer documentation and authoritative changelog evidence

**Files:**
- Create: `docs/scoring_execution_contracts.md`
- Create: `docs/changelog.d/477-scoring-observation-engine.md`
- Modify: `CHANGELOG.md`

- [ ] Document provider-neutral human/automated engine use.
- [ ] Document raw-content exclusion, evidence fingerprints, status semantics, and non-goals.
- [ ] Add a deterministic fixture example.
- [ ] Render the authoritative changelog fragment and verify parity.

### Task 4: Exact-head review and merge

- [ ] Run focused execution-contract tests with 100% statement/branch coverage.
- [ ] Run the full Python suite, Rust workspace/PyO3 tests, package build, GPU parity, fuzz, Security Scan, and SAST.
- [ ] Review every human, CodeRabbit, and security finding.
- [ ] Mark ready and merge only when the unchanged exact head satisfies repository policy.
