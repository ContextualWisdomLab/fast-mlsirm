# Auditable Rubric-to-Item Blueprint Compiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline, provider-neutral compiler that validates versioned rubric specifications and emits bounded, deterministic, content-addressed item blueprints and structured-output generation contracts.

**Architecture:** Frozen standard-library dataclasses define the evidence-centered rubric, task, evidence, and difficulty schemas. A pure compiler forms a caller-bounded Cartesian product in canonical axis order. A separate canonical-contract module serializes a restricted JSON value set, computes SHA-256 identities, and builds immutable provider-neutral contracts without network calls or psychometric arithmetic.

**Tech Stack:** Python 3.10+, standard-library `dataclasses`, `json`, `hashlib`, `itertools`, `re`, pytest; existing maturin/Rust project remains unchanged.

## Global Constraints

- Use no hosted model, network client, provider SDK, or new runtime dependency.
- Keep all psychometric estimation and scoring arithmetic outside this Python authoring slice.
- Use two-or-more-token lower-case `snake_case` domain identifiers.
- Require supported semantic schema versions, contiguous ordinal score levels, and contiguous zero-based difficulty order indices.
- Reject unsupported canonical JSON types, including every floating-point value.
- Calculate the full blueprint product before allocation and never truncate silently.
- Keep every added Python branch and every public docstring at 100% coverage.
- Preserve standalone package use and submodule/MSA use through provider-neutral contracts.
- Cite Evidence-Centered Design literature in APA 7th style.

---

### Task 1: Establish RED schema and compiler contracts

**Files:**
- Create: `tests/test_rubric_engine.py`

**Interfaces:**
- Consumes: the design in `docs/superpowers/specs/2026-08-03-rubric-blueprint-compiler-design.md`.
- Produces: executable contracts for `RubricLevel`, `RubricCriterion`, `TaskFamily`, `EvidenceMode`, `DifficultyBand`, `RubricSpecification`, `canonical_json`, `sha256_fingerprint`, `compile_item_blueprints`, `generated_item_output_schema`, and `build_generation_contract`.

- [ ] **Step 1: Add a complete valid rubric fixture**

```python
from fast_mlsirm.rubric_engine import (
    DifficultyBand,
    EvidenceMode,
    RubricCriterion,
    RubricLevel,
    RubricSpecification,
    TaskFamily,
)


def _valid_specification(**overrides):
    values = {
        "schema_version": "1.0.0",
        "rubric_id": "customer_issue_priority",
        "rubric_version": "1.0.0",
        "title": "Customer issue priority",
        "purpose": "Measure evidence-conditioned business issue priority.",
        "locale": "ko-KR",
        "response_format": "structured_json_record",
        "criteria": (
            RubricCriterion(
                criterion_id="business_materiality",
                title="Business materiality",
                construct="decision_consequence",
                description="Expected consequence if the issue is realized.",
                levels=(
                    RubricLevel(0, "No consequence", "No material effect.", ("No decision consequence is supported.",)),
                    RubricLevel(1, "Local consequence", "A bounded effect is supported.", ("A reversible local consequence is supported.",)),
                    RubricLevel(2, "Enterprise consequence", "A broad effect is supported.", ("A contract, revenue, legal, or operational consequence is supported.",)),
                ),
                prohibited_patterns=("emotion_only_rationale",),
            ),
        ),
        "task_families": (
            TaskFamily(
                task_family_id="evidence_issue_analysis",
                description="Analyze one atomic issue against supplied evidence.",
                expected_response_format="structured_json_record",
                required_reasoning=("identify_support", "identify_counterevidence"),
            ),
        ),
        "evidence_modes": (
            EvidenceMode(
                evidence_mode_id="bounded_source_packet",
                description="Use only the supplied source packet.",
                evidence_requirement="Every material claim names a supplied source.",
                minimum_sources=1,
                maximum_sources=3,
                allowed_source_types=("enterprise_report", "customer_record"),
            ),
        ),
        "difficulty_bands": (
            DifficultyBand(
                difficulty_band_id="direct_evidence_band",
                order_index=0,
                description="Direct, explicit evidence.",
                constraints=("single_issue", "explicit_consequence"),
            ),
        ),
        "prohibited_patterns": ("invented_evidence",),
        "generation_rules": ("preserve_counterevidence",),
    }
    values.update(overrides)
    return RubricSpecification(**values)
```

- [ ] **Step 2: Add schema-validation tests**

Cover invalid identifiers, unsupported versions, empty text, invalid locale,
boolean numeric fields, duplicate axis identifiers, missing axes, noncontiguous
scores, missing observable indicators, noncontiguous difficulty indices,
negative or reversed source bounds, empty constraints, and dataclass
immutability.

- [ ] **Step 3: Add canonicalization and fingerprint tests**

```python
def test_canonical_json_is_sorted_utf8_and_rejects_floats():
    assert canonical_json({"z_key": 2, "a_key": "한글"}) == '{"a_key":"한글","z_key":2}'
    with pytest.raises(ValueError, match="floating-point"):
        canonical_json({"bad_value": 0.5})


def test_semantically_unordered_axes_have_the_same_fingerprint():
    left = _multi_axis_specification(reverse=False)
    right = _multi_axis_specification(reverse=True)
    assert sha256_fingerprint(left) == sha256_fingerprint(right)
```

Also reject bytes, sets, non-string mapping keys, and arbitrary objects; verify
the `sha256:` prefix and 64 lowercase hexadecimal digits.

- [ ] **Step 4: Add compiler and contract tests**

```python
def test_compiler_emits_complete_deterministic_cartesian_product():
    blueprints = compile_item_blueprints(_multi_axis_specification(), max_blueprints=16)
    assert len(blueprints) == 16
    assert blueprints[0].blueprint_id.startswith("item_blueprint_")
    assert blueprints == compile_item_blueprints(_multi_axis_specification(), max_blueprints=16)


def test_compiler_fails_before_silent_truncation():
    with pytest.raises(ValueError, match="16.*15"):
        compile_item_blueprints(_multi_axis_specification(), max_blueprints=15)
```

Verify canonical axis order, fingerprint sensitivity to every material
constraint, generation-contract tamper detection, JSON Schema Draft 2020-12,
`additionalProperties: false`, and required evidence/rubric traceability fields.

- [ ] **Step 5: Commit the failing tests**

```bash
git add tests/test_rubric_engine.py
git commit -m "test: define rubric blueprint compiler contract"
```

- [ ] **Step 6: Run CI and record the expected RED evidence**

Expected failure: test collection cannot import `fast_mlsirm.rubric_engine`
because no production package exists yet. Confirm the failure is caused by the
missing feature rather than syntax or fixture errors.

---

### Task 2: Implement strict immutable rubric schemas

**Files:**
- Create: `python/fast_mlsirm/rubric_engine/rubric_schema.py`
- Create: `python/fast_mlsirm/rubric_engine/__init__.py`
- Test: `tests/test_rubric_engine.py`

**Interfaces:**
- Produces: `SUPPORTED_RUBRIC_SCHEMA_VERSION`, `RubricLevel`, `RubricCriterion`, `TaskFamily`, `EvidenceMode`, `DifficultyBand`, and `RubricSpecification`.
- Consumes: only Python standard-library modules.

- [ ] **Step 1: Implement reusable validators**

```python
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
_SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_LOCALE = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")
```

Add field-naming `ValueError` guards for non-empty strings, tuples of non-empty
strings, integer-but-not-boolean values, unique identifiers, and supported
versions.

- [ ] **Step 2: Implement frozen leaf schemas**

Use `@dataclass(frozen=True)` and `object.__setattr__` only to replace incoming
iterables with validated tuples. Validate all leaf fields in `__post_init__`.

- [ ] **Step 3: Implement and normalize `RubricSpecification`**

Sort criteria, task families, and evidence modes by identifier; sort difficulty
bands by `(order_index, difficulty_band_id)`; preserve score order after sorting
by score. Reject duplicate identifiers and incomplete axes. Require scores to
be `range(min_score, max_score + 1)` and difficulty indices to equal
`range(len(difficulty_bands))`.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_rubric_engine.py -q`
Expected: schema tests pass; canonical/compiler tests remain RED because their
modules do not exist.

- [ ] **Step 5: Commit**

```bash
git add python/fast_mlsirm/rubric_engine tests/test_rubric_engine.py
git commit -m "feat: add immutable rubric schemas"
```

---

### Task 3: Implement canonical JSON and content-addressed envelopes

**Files:**
- Create: `python/fast_mlsirm/rubric_engine/canonical_contract.py`
- Modify: `python/fast_mlsirm/rubric_engine/__init__.py`
- Test: `tests/test_rubric_engine.py`

**Interfaces:**
- Produces: `canonical_json(value) -> str`, `sha256_fingerprint(value) -> str`, `BlueprintPayload`, `CompiledBlueprint`, `GenerationContractPayload`, and `GenerationContract`.
- Consumes: the frozen schemas from Task 2.

- [ ] **Step 1: Implement restricted canonical-data conversion**

Recursively allow only dataclasses, string-keyed mappings, list/tuple, string,
integer, boolean, and `None`. Check boolean before integer and reject every
float with `ValueError("floating-point values are outside the canonical contract")`.

- [ ] **Step 2: Implement canonical serialization and SHA-256**

```python
def canonical_json(value: object) -> str:
    data = _canonical_data(value)
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_fingerprint(value: object) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
```

- [ ] **Step 3: Implement frozen payload and envelope dataclasses**

`CompiledBlueprint` contains `blueprint_id`, `blueprint_fingerprint`, and
`payload`. `GenerationContract` contains `contract_id`,
`contract_fingerprint`, and `payload`. Envelope constructors validate
identifier and fingerprint syntax; payload fingerprints deliberately exclude
their own envelope identity to avoid self-reference.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_rubric_engine.py -q`
Expected: canonical and schema tests pass; compiler and generation-contract
builder tests remain RED.

- [ ] **Step 5: Commit**

```bash
git add python/fast_mlsirm/rubric_engine tests/test_rubric_engine.py
git commit -m "feat: add canonical rubric contracts"
```

---

### Task 4: Compile bounded deterministic item blueprints

**Files:**
- Create: `python/fast_mlsirm/rubric_engine/blueprint_compiler.py`
- Modify: `python/fast_mlsirm/rubric_engine/__init__.py`
- Test: `tests/test_rubric_engine.py`

**Interfaces:**
- Produces: `compile_item_blueprints(specification: RubricSpecification, *, max_blueprints: int) -> tuple[CompiledBlueprint, ...]`.
- Consumes: all schemas and content-addressing helpers from Tasks 2-3.

- [ ] **Step 1: Validate the caller-controlled bound**

Reject boolean, non-integer, values below one, and values above
`MAX_COMPILED_BLUEPRINTS = 100_000`.

- [ ] **Step 2: Compute product size before materialization**

```python
product_size = (
    len(specification.criteria)
    * len(specification.task_families)
    * len(specification.evidence_modes)
    * len(specification.difficulty_bands)
)
if product_size > max_blueprints:
    raise ValueError(
        f"blueprint product size {product_size} exceeds max_blueprints {max_blueprints}"
    )
```

- [ ] **Step 3: Build canonical payloads and envelopes**

For each canonical Cartesian combination, include the rubric fingerprint,
criterion score levels, task reasoning, evidence requirements, difficulty
constraints, locale, response format, combined prohibited patterns, and
combined generation rules. Compute the full payload fingerprint and use its
64-character digest in `item_blueprint_<digest>`.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_rubric_engine.py -q`
Expected: compiler tests pass; generation-contract builder tests remain RED.

- [ ] **Step 5: Commit**

```bash
git add python/fast_mlsirm/rubric_engine tests/test_rubric_engine.py
git commit -m "feat: compile content-addressed item blueprints"
```

---

### Task 5: Build provider-neutral structured generation contracts

**Files:**
- Create: `python/fast_mlsirm/rubric_engine/generation_contract.py`
- Modify: `python/fast_mlsirm/rubric_engine/__init__.py`
- Test: `tests/test_rubric_engine.py`

**Interfaces:**
- Produces: `generated_item_output_schema() -> dict[str, object]` and `build_generation_contract(blueprint: CompiledBlueprint) -> GenerationContract`.
- Consumes: `CompiledBlueprint`, `GenerationContractPayload`, `GenerationContract`, `canonical_json`, and `sha256_fingerprint`.

- [ ] **Step 1: Implement a strict Draft 2020-12 output schema**

The root and every nested object use `additionalProperties: false`. Require:
`item_id`, `item_text`, `response_format`, `evidence_references`,
`rubric_alignment`, `difficulty_rationale`, `scoring_guidance`, and
`prohibited_pattern_checks`.

- [ ] **Step 2: Implement tamper-evident contract construction**

Recompute the blueprint payload fingerprint and expected full-digest identifier;
raise `ValueError` if either differs from the envelope. Serialize the output
schema canonically into `structured_output_schema_json`.

- [ ] **Step 3: Build the contract payload**

Include fixed generation instructions:

```python
(
    "use_only_supplied_blueprint_evidence",
    "emit_exactly_one_generated_item_record",
    "do_not_add_unspecified_properties",
    "preserve_rubric_and_evidence_traceability",
    "report_prohibited_pattern_checks",
)
```

Fingerprint the payload and use `generation_contract_<digest>` as the contract
identifier.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_rubric_engine.py -q`
Expected: all rubric-engine tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/fast_mlsirm/rubric_engine tests/test_rubric_engine.py
git commit -m "feat: add provider-neutral generation contracts"
```

---

### Task 6: Document and expose the vertical slice

**Files:**
- Modify: `python/fast_mlsirm/rubric_engine/__init__.py`
- Create: `docs/rubric_blueprint_compiler.md`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Test: `tests/test_rubric_engine.py`

**Interfaces:**
- Produces: stable imports from `fast_mlsirm.rubric_engine` and a copy-pasteable offline example.
- Consumes: every public type/function from Tasks 2-5.

- [ ] **Step 1: Define the explicit package export list**

Export only supported constants, schema types, envelopes, canonicalization
helpers, compiler, output-schema builder, and generation-contract builder.

- [ ] **Step 2: Add scientific and operational documentation**

Document the ECD mapping, identifier/version constraints, Cartesian compilation,
canonicalization scope, trust boundary, JSON Schema output, no-network behavior,
and deferred provider/calibration slices. Cite the three ECD sources in APA 7th
format.

- [ ] **Step 3: Add README and changelog entries**

Add a compact `Rubric blueprint compiler` example to README and an `Unreleased / Added`
entry to CHANGELOG. Do not change the package version in this feature PR.

- [ ] **Step 4: Verify public documentation**

Run:

```bash
python -m interrogate -c pyproject.toml python/fast_mlsirm
pytest tests/test_rubric_engine.py -q
```

Expected: 100% public docstring coverage and all focused tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/fast_mlsirm/rubric_engine docs/rubric_blueprint_compiler.md README.md CHANGELOG.md tests/test_rubric_engine.py
git commit -m "docs: publish rubric blueprint compiler contract"
```

---

### Task 7: Full same-head verification, review, and merge

**Files:**
- Modify only files required by valid review or CI findings.

**Interfaces:**
- Consumes: GitHub review threads and all required repository checks.
- Produces: a mergeable PR closing issue #394.

- [ ] **Step 1: Run the complete local verification contract**

```bash
pytest --cov=fast_mlsirm --cov-branch --cov-report=term-missing
python -m interrogate -c pyproject.toml python/fast_mlsirm
cargo test --workspace
cargo test --manifest-path crates/fast-mlsirm-py/Cargo.toml
python -m build
```

Expected: zero test failures; Python branch coverage 100%; public docstrings
100%; Rust tests pass; source and wheel distributions build.

- [ ] **Step 2: Inspect the final diff for scope and secrets**

Confirm no provider credential, network call, new dependency, psychometric
arithmetic, generated artifact, or unrelated refactor is present.

- [ ] **Step 3: Mark the draft PR ready and request review**

Resolve every actionable review thread. Keep the PR head unchanged while final
checks run.

- [ ] **Step 4: Verify same-head GitHub checks**

Require CI, Security Scan, SAST Semgrep, ClusterFuzzLite where configured,
coverage, docstrings, package build, and branch protection to succeed on the
exact final head SHA.

- [ ] **Step 5: Enable auto-merge or merge**

Merge only after all required checks and reviews succeed. Confirm issue #394 is
closed by the merged PR.
