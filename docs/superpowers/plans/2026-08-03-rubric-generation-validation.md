# Governed Item Generation and Candidate Validation Implementation Plan

> **For agentic workers:** Use the Superpowers TDD and verification workflows. Preserve a failing test-only commit before production code.

**Goal:** Turn a compiled rubric blueprint plus bounded sources into a content-addressed provider request, then convert one untrusted JSON response into a strict, auditable item candidate without provider dependencies or Python psychometric arithmetic.

**Architecture:** Extend `fast_mlsirm.rubric` with immutable generation/source models, a runtime-checkable provider protocol and fixture adapter, strict candidate models/parser, and an exactly-once executor. Build on PR #396 and keep the branch stacked until its base merges.

**Tech stack:** Python 3.10+ standard library (`dataclasses`, `hashlib`, `json`, `typing`, `re`), pytest, existing maturin/Rust workspace.

## Global constraints

- All public ids use two-or-more-token lower `snake_case`.
- Schema version is exactly `1.0`.
- Source text ≤262,144 characters each, ≤32 sources, ≤1,048,576 aggregate characters.
- Raw provider JSON ≤262,144 characters.
- Duplicate JSON keys and non-finite constants are rejected.
- Errors never echo raw source/candidate/provider text.
- No network, SDK, credentials, URL fetching, filesystem access, retry loop, or async runtime.
- No scoring/calibration/fit/DIF/information arithmetic in Python.
- 100% statement/branch coverage and 100% docstring coverage for added code.

---

### Task 1: RED behavioral contract

**Create:** `tests/test_rubric_generation.py`, `tests/test_rubric_generation_edge_cases.py`

- [ ] Test `SourceDocument` normalization, content digest, redacted metadata, provider payload, id/media/locale/type/size guards.
- [ ] Test evidence-mode source cardinality, duplicate source ids, aggregate budget, wrong boundary types, request determinism, and source-content sensitivity.
- [ ] Test runtime protocol, valid static fixture provider, invalid provider metadata, non-protocol objects, non-string output, exception redaction, and exactly-one invocation.
- [ ] Test safe JSON parsing: size, duplicate keys at every nesting level, NaN/Infinity, syntax, top-level non-object, missing/unknown fields.
- [ ] Test every candidate field, nested object, collection, identifier, and size guard.
- [ ] Test exact score-guide/alignment coverage and duplicate/missing score rejection.
- [ ] Test source-id existence, attribution uniqueness, exact evidence-span presence, and closed-book/source-backed attribution rules.
- [ ] Test all five response-format structural branches.
- [ ] Test deterministic candidate/execution fingerprints and raw-text redaction.
- [ ] Commit tests only, open a draft PR based on `agent/rubric-blueprint-compiler`, and retain expected import failure as RED evidence.

---

### Task 2: Source and generation-request models

**Create:** `python/fast_mlsirm/rubric/generation.py`

- [ ] Implement bounded helper reuse from `rubric.models` without copying validation formulas.
- [ ] Add `SourceDocument` with canonical metadata/provider serialization and SHA-256 content digest.
- [ ] Add immutable `GenerationRequest` with redacted `to_metadata_dict()` and explicit `to_provider_dict()`.
- [ ] Add `build_generation_request` with exact rubric/blueprint compatibility, source uniqueness, evidence-mode cardinality, aggregate budget, and deterministic request id.
- [ ] Add `@runtime_checkable ItemGenerationProvider` and validated `StaticFixtureProvider`.
- [ ] Run focused tests; commit.

---

### Task 3: Strict candidate parser

**Create:** `python/fast_mlsirm/rubric/candidates.py`

- [ ] Implement `CandidateValidationError(code, path, message)` with redacted formatting.
- [ ] Implement immutable option, score-guide, alignment, attribution, and candidate models.
- [ ] Add bounded JSON decode with duplicate-key and non-finite-constant rejection.
- [ ] Validate exact top-level/nested fields, scalar types, ids, text and collection bounds.
- [ ] Enforce exact rubric score coverage and normalize score entries into ascending order.
- [ ] Validate source attributions against request sources and evidence spans.
- [ ] Enforce response-format structural contracts.
- [ ] Compute deterministic candidate fingerprint from canonical normalized content.
- [ ] Run focused tests; commit.

---

### Task 4: Exactly-once executor and provenance

**Modify:** `python/fast_mlsirm/rubric/generation.py`, `python/fast_mlsirm/rubric/__init__.py`

- [ ] Add redacted `GenerationProviderError`.
- [ ] Add immutable `GenerationExecution` metadata result.
- [ ] Implement `execute_generation(provider, request)` with provider metadata validation, exactly one call, non-string guard, parser delegation, raw-response SHA-256 digest, and deterministic execution id.
- [ ] Export the supported public API explicitly.
- [ ] Run focused tests; commit.

---

### Task 5: Documentation and changelog fragment

**Create:** `docs/rubric_generation_validation.md`, `docs/changelog.d/406-rubric-generation-validation.md`

**Modify:** `README.md`, `docs/rubric_item_generation.md`

- [ ] Document complete offline example, integration boundary, cardinality rules, redaction guarantees, validation failures, and provider-adapter roadmap.
- [ ] Add RFC 8259 and JSON Schema Draft 2020-12 APA 7 references.
- [ ] Add README capability/link and update the rubric workflow guide.
- [ ] Add changelog fragment; do not bump version.
- [ ] Test public exports and documentation links; commit.

---

### Task 6: Verification, review, retarget, merge

- [ ] Run focused tests and 100% statement/branch coverage for added modules.
- [ ] Run full Python suite, docstring gate, Rust workspace tests, packaging build, security, and SAST.
- [ ] Inspect every review and inline thread; fix concrete findings test-first.
- [ ] After #396 merges, rebase/retarget this PR to `main` without losing RED evidence.
- [ ] Require same-head checks and approval; merge using repository policy.
- [ ] Re-inventory PRs/issues and start semantic screening or address a higher-value blocker.