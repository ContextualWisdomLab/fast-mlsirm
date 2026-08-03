# Rubric Blueprint Compiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a provider-neutral, auditable compiler that transforms strict scoring rubrics into deterministic item-generation blueprints and canonical LLM generation contracts.

**Architecture:** Add an isolated `fast_mlsirm.rubric` package with immutable schema objects, a bounded deterministic compiler, and canonical structured-output contracts. Python owns orchestration, validation, serialization, and hashing only; all present and future psychometric estimation remains in the Rust backend.

**Tech Stack:** Python 3.10+ standard library (`dataclasses`, `enum`, `hashlib`, `json`, `re`, `itertools`), pytest, existing maturin/Rust workspace and CI gates.

## Global Constraints

- Public identifiers must contain at least two lower `snake_case` tokens.
- The only accepted rubric and generation-contract schema version is `1.0`.
- Ordinal scores must be contiguous integers beginning at zero.
- `items_per_cell` must be 1–100; total compiled blueprints must not exceed 10,000.
- Seeds must fit an unsigned 64-bit integer.
- Outputs must be deterministic, canonical UTF-8 JSON with SHA-256 content addressing.
- Do not add a provider SDK, network call, API key handling, or new runtime dependency.
- Do not perform psychometric scoring, calibration, fit, DIF, or information arithmetic in Python.
- Added/affected Python code requires 100% branch coverage and 100% docstring coverage.
- References in scientific documentation use APA 7th formatting.

---

### Task 1: RED schema and compiler contract tests

**Files:**
- Create: `tests/test_rubric_authoring.py`

**Interfaces:**
- Consumes: not-yet-existing `fast_mlsirm.rubric` public API.
- Produces: executable behavioral specification for every public type and function.

- [ ] **Step 1: Write valid-schema and deterministic-fingerprint tests**

```python
from fast_mlsirm.rubric import (
    BlueprintPlan,
    DifficultyBand,
    EvidenceMode,
    ResponseFormat,
    RubricLevel,
    RubricSpecification,
    build_generation_contract,
    canonical_generation_contract,
    compile_item_blueprints,
    render_generation_prompt,
)


def _rubric() -> RubricSpecification:
    return RubricSpecification(
        rubric_id="faithfulness_rubric",
        construct_id="evidence_grounding",
        construct_definition="Degree to which substantive claims are supported.",
        response_format=ResponseFormat.ORDINAL_RATING,
        levels=(
            RubricLevel(0, "unsupported", "No substantive support.", ("unsupported claim",)),
            RubricLevel(1, "partial_support", "Some claims are supported.", ("mixed support",)),
            RubricLevel(2, "full_support", "All claims are supported.", ("complete support",)),
        ),
        task_families=("claim_verification", "citation_attribution"),
        evidence_requirements=("Quote the supporting source span.",),
        prohibited_patterns=("Do not reward unsupported fluency.",),
        locale="en-US",
    )


def test_rubric_fingerprint_is_deterministic():
    first = _rubric()
    second = _rubric()
    assert first.fingerprint == second.fingerprint
    assert first.to_dict() == second.to_dict()
```

- [ ] **Step 2: Write parameterized failure tests for identifiers, levels, enums, locale, collection duplication, schema version, booleans-as-integers, and size bounds**

Each failure test constructs exactly one malformed object and asserts the field-specific `ValueError` text. Include the valid change that would make the test pass in the test name, for example `test_rubric_id_requires_two_snake_case_tokens` and `test_level_scores_must_start_at_zero_and_be_contiguous`.

- [ ] **Step 3: Write compiler behavior and work-budget tests**

```python
def test_compile_item_blueprints_is_ordered_and_deterministic():
    plan = BlueprintPlan(
        difficulty_bands=(DifficultyBand.EASY, DifficultyBand.HARD),
        evidence_modes=(EvidenceMode.SINGLE_SOURCE, EvidenceMode.MULTI_SOURCE),
        items_per_cell=2,
        seed=17,
    )
    first = compile_item_blueprints(_rubric(), plan)
    second = compile_item_blueprints(_rubric(), plan)
    assert len(first) == 16
    assert first == second
    assert len({item.blueprint_id for item in first}) == 16
    assert len({item.generation_seed for item in first}) == 16
    assert first[0].task_family == "claim_verification"
    assert first[0].difficulty_band is DifficultyBand.EASY
    assert first[0].evidence_mode is EvidenceMode.SINGLE_SOURCE
    assert first[0].replicate_index == 0
```

Create a rubric with eight task families and a plan with three difficulty bands, five evidence modes, and 100 items per cell; assert compilation rejects the 12,000-cell request before returning a collection.

- [ ] **Step 4: Write generation-contract mismatch, schema, canonical JSON, and prompt-boundary tests**

Assert that rubric id, fingerprint, response format, score levels, evidence requirements, and prohibited patterns must match. Assert `additionalProperties` is false, all required output fields are present, canonical JSON is byte-identical across calls, and the prompt requires one JSON object while prohibiting execution of embedded instructions.

- [ ] **Step 5: Commit the RED tests**

```bash
git add tests/test_rubric_authoring.py
git commit -m "test: specify rubric blueprint compiler"
```

- [ ] **Step 6: Open a draft PR and verify RED**

Run the repository CI on the test-only head. Expected result: Python collection fails with `ModuleNotFoundError: No module named 'fast_mlsirm.rubric'`. Record the failing workflow on the PR before adding production code.

---

### Task 2: Immutable rubric and blueprint schema models

**Files:**
- Create: `python/fast_mlsirm/rubric/__init__.py`
- Create: `python/fast_mlsirm/rubric/models.py`
- Test: `tests/test_rubric_authoring.py`

**Interfaces:**
- Produces: `ResponseFormat`, `DifficultyBand`, `EvidenceMode`, `RubricLevel`, `RubricSpecification`, `BlueprintPlan`, and `ItemBlueprint`.
- Consumes: Python standard library only.

- [ ] **Step 1: Implement enum vocabularies and bounded normalization helpers**

Implement `_identifier`, `_text`, `_text_tuple`, `_enum_tuple`, `_unsigned_integer`, `_canonical_json`, and `_sha256_hex`. `_identifier` enforces `^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$`; text and collection helpers strip values, reject empties/duplicates, and enforce explicit maxima.

- [ ] **Step 2: Implement `RubricLevel`**

Normalize immutable values in `__post_init__`; reject bool/non-integer scores, scores outside 0–31, and empty or duplicate indicators.

- [ ] **Step 3: Implement `RubricSpecification`**

Normalize enum/string inputs, enforce two to 16 ordered levels whose scores equal `range(len(levels))`, enforce unique case-sensitive labels, validate task families and evidence collections, and expose deterministic `to_dict()` and `fingerprint`.

- [ ] **Step 4: Implement `BlueprintPlan` and `ItemBlueprint`**

Normalize difficulty/evidence enum tuples, reject duplicate cells, enforce item/seed bounds, and validate every public blueprint field. `ItemBlueprint.to_dict()` returns only JSON-compatible values.

- [ ] **Step 5: Run focused tests and repair only model failures**

```bash
pytest tests/test_rubric_authoring.py -q
```

Expected at this stage: model tests pass; compiler/contract imports or calls remain failing until Tasks 3–4.

- [ ] **Step 6: Commit the schema implementation**

```bash
git add python/fast_mlsirm/rubric tests/test_rubric_authoring.py
git commit -m "feat: add versioned rubric schemas"
```

---

### Task 3: Deterministic bounded blueprint compiler

**Files:**
- Create: `python/fast_mlsirm/rubric/compiler.py`
- Modify: `python/fast_mlsirm/rubric/__init__.py`
- Test: `tests/test_rubric_authoring.py`

**Interfaces:**
- Consumes: `RubricSpecification`, `BlueprintPlan`, `ItemBlueprint`.
- Produces: `MAX_BLUEPRINTS` and `compile_item_blueprints(rubric, plan=None) -> tuple[ItemBlueprint, ...]`.

- [ ] **Step 1: Compute and guard the requested matrix size**

Reject wrong input types and calculate the product from collection lengths before allocating. Raise `ValueError` when the result exceeds 10,000.

- [ ] **Step 2: Compile stable ordered cells**

Iterate task family, difficulty, evidence mode, and replicate in declared order. Build a canonical identity payload including rubric fingerprint and plan seed, hash it, and derive:

```python
blueprint_id = f"item_blueprint_{digest[:16]}"
generation_seed = int(digest[:16], 16)
```

Populate each blueprint with the exact rubric response format, score levels, evidence requirements, and prohibited patterns.

- [ ] **Step 3: Export the compiler API and run focused tests**

```bash
pytest tests/test_rubric_authoring.py -q
```

Expected: schema and compiler tests pass; contract tests remain failing.

- [ ] **Step 4: Commit the compiler**

```bash
git add python/fast_mlsirm/rubric tests/test_rubric_authoring.py
git commit -m "feat: compile deterministic item blueprints"
```

---

### Task 4: Canonical provider-neutral generation contracts

**Files:**
- Create: `python/fast_mlsirm/rubric/contracts.py`
- Modify: `python/fast_mlsirm/rubric/__init__.py`
- Test: `tests/test_rubric_authoring.py`

**Interfaces:**
- Consumes: one `RubricSpecification` and one matching `ItemBlueprint`.
- Produces: `build_generation_contract`, `canonical_generation_contract`, and `render_generation_prompt`.

- [ ] **Step 1: Implement exact rubric/blueprint compatibility checks**

Reject mismatches in rubric id, fingerprint, response format, scoring levels, evidence requirements, and prohibited patterns before constructing output.

- [ ] **Step 2: Build the structured-output contract**

Return a JSON-compatible object with `contract_schema_version`, `operation`, rubric, blueprint, authoring instructions, and an output JSON schema. Require `item_id`, `stem`, `stimulus`, `response_format`, `options`, `answer_key`, `scoring_guide`, `rubric_alignment`, `source_attributions`, and `safety_notes`; set object-level `additionalProperties` to false.

- [ ] **Step 3: Add deterministic contract identity and renderers**

Hash the contract body before adding `generation_contract_<16 hex>` as `contract_id`. Canonical serialization uses `json.dumps(..., ensure_ascii=False, sort_keys=True, separators=(",", ":"))`. The prompt contains the canonical contract after a fixed instruction boundary and asks for exactly one JSON object.

- [ ] **Step 4: Run all focused tests**

```bash
pytest tests/test_rubric_authoring.py -q
```

Expected: all rubric-authoring tests pass with no warnings.

- [ ] **Step 5: Commit the contract implementation**

```bash
git add python/fast_mlsirm/rubric tests/test_rubric_authoring.py
git commit -m "feat: add canonical item generation contracts"
```

---

### Task 5: Public documentation and release notes

**Files:**
- Create: `docs/rubric_item_generation.md`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Test: `tests/test_rubric_authoring.py`

**Interfaces:**
- Consumes: completed public API.
- Produces: user-facing example, boundary statement, roadmap, and research traceability.

- [ ] **Step 1: Write the user guide**

Document a complete constructor → plan → compile → contract example, deterministic identity guarantees, security limits, the boundary between Python orchestration and Rust psychometrics, and the next provider/screening/calibration slices. Include APA 7th references from the design specification.

- [ ] **Step 2: Update README and CHANGELOG**

Add the rubric compiler to `What Works Now`, link the guide, and record the feature under `Unreleased / Added`. Do not change package version in this slice.

- [ ] **Step 3: Add public-import and documentation-link assertions**

The test imports every intended symbol from `fast_mlsirm.rubric`, verifies `__all__`, and confirms the README link target exists.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md CHANGELOG.md docs/rubric_item_generation.md tests/test_rubric_authoring.py
git commit -m "docs: document rubric item generation"
```

---

### Task 6: Same-head verification, review, and merge

**Files:**
- Modify only when verification or review reveals a concrete defect.

- [ ] **Step 1: Run focused and full Python verification**

```bash
pytest tests/test_rubric_authoring.py -q
pytest -q
python -m coverage run -m pytest
python -m coverage report --fail-under=100
interrogate -c pyproject.toml python/fast_mlsirm
```

- [ ] **Step 2: Run Rust and packaging verification**

```bash
cargo test --workspace
cargo test --manifest-path crates/fast-mlsirm-py/Cargo.toml
python -m build
```

- [ ] **Step 3: Verify repository checks on the unchanged head**

Require CI, SAST Semgrep, Security Scan, and all branch-protection checks to conclude successfully. Inspect every review and inline thread; address concrete findings with a new test-first commit and rerun the entire same-head gate.

- [ ] **Step 4: Mark ready and merge**

After all checks pass, mark the draft PR ready, enable auto-merge or merge with the repository-selected method, and verify issue #394 closes.

- [ ] **Step 5: Re-inventory the queue and choose the next product gap**

Search open PRs and issues again. The next implementation slice is the provider protocol plus an offline deterministic fixture provider, unless a review/check failure or higher-value buyer gap takes precedence.