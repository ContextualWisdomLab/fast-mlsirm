# Scoring Observation and Engine Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add lossless scoring observations, evidence references, and a provider-neutral scoring-engine protocol that preserve exact assessment/rubric provenance for future essay, RAG, and enterprise-issue adapters.

**Architecture:** Extend the merged `fast_mlsirm.scoring` contract layer without adding psychometric arithmetic or a second rubric schema. Immutable observation and engine artifacts reuse the existing bounded validation, canonical JSON, SHA-256 identity, and `AssessmentSpecError` boundary; future calibration code can consume these artifacts and delegate estimation to the Rust-backed facet APIs.

**Tech Stack:** Python 3.10+, frozen dataclasses, `Enum`, `typing.Protocol`/generics, existing `fast_mlsirm.scoring` validation helpers, `fast_mlsirm.rubric.RubricSpecification`, pytest, repository coverage/docstring/security/release gates.

## Global Constraints

- Keep one repository and distribution: `ContextualWisdomLab/fast-mlsirm` / `fast-mlsirm`.
- `fast_mlsirm.rubric.RubricSpecification` remains the sole rubric source of truth.
- Public identifiers use descriptive two-or-more-token lower `snake_case`; numeric-only identifiers are rejected.
- Database-facing object names, when later persisted by consumers, use two-or-more-token `snake_case`.
- Preserve `scored`, `missing`, `abstained`, `failed`, and `excluded` as distinct states.
- Preserve criterion-level and holistic observations as distinct levels; never average or derive one from the other in this slice.
- Raw response text, source text, provider exception text, and rejected caller values must not appear in public errors, fingerprints, or default serialization.
- All public construction/validation failures use `AssessmentSpecError` with stable codes, caller-independent paths, and non-reflective messages.
- Reuse canonical UTF-8, metadata, integer, iterable, negative-zero, and output-size limits from the merged assessment-contract layer.
- Add no hosted-provider dependency and no Python psychometric arithmetic.
- New public classes, enums, protocols, functions, and modules require complete docstrings.
- New module code requires 100% statement and branch coverage.
- Update an authoritative `docs/changelog.d` fragment and render it into `CHANGELOG.md` before readiness.

---

## File Structure

- Create `python/fast_mlsirm/scoring/observations.py`: observation enums, evidence references, immutable `ScoreObservation`, factories, and batch validation.
- Create `python/fast_mlsirm/scoring/engines.py`: immutable engine descriptor/result, generic protocol, and deterministic offline fixture.
- Modify `python/fast_mlsirm/scoring/__init__.py`: public exports only; no optional provider imports.
- Modify `python/fast_mlsirm/scoring/contracts.py`: stable compatibility exports for future internal modules.
- Create `tests/scoring_observation_fixtures.py`: deterministic approved assessment/rubric/observation fixtures.
- Create `tests/test_scoring_observations.py`: status, provenance, evidence, canonicalization, and batch-validation contracts.
- Create `tests/test_scoring_observation_safety.py`: resource-bound, hostile-callback, UTF-8, redaction, and direct-construction contracts.
- Create `tests/test_scoring_engines.py`: descriptor/protocol/result/fixture behavior.
- Create `tests/test_scoring_engine_safety.py`: rule-failure redaction, capability mismatches, and resource limits.
- Create `docs/scoring_observation_engine_contracts.md`: user-facing architecture, status semantics, provenance, MSA boundary, and non-claims.
- Create `docs/changelog.d/482-scoring-observation-engine-contracts.md`: authoritative release note fragment.

---

### Task 1: Lossless Observation and Evidence Contracts

**Files:**
- Create: `python/fast_mlsirm/scoring/observations.py`
- Create: `tests/scoring_observation_fixtures.py`
- Create: `tests/test_scoring_observations.py`
- Create: `tests/test_scoring_observation_safety.py`
- Modify: `python/fast_mlsirm/scoring/__init__.py`
- Modify: `python/fast_mlsirm/scoring/contracts.py`

**Interfaces:**
- Consumes: `AssessmentSpec`, `RubricSpecification`, `AssessmentSpecError`, `canonical_json()`, `artifact_digest()`, and the merged bounded validation helpers.
- Produces:

```python
class ObservationStatus(str, Enum):
    SCORED = "scored"
    MISSING = "missing"
    ABSTAINED = "abstained"
    FAILED = "failed"
    EXCLUDED = "excluded"

class ObservationLevel(str, Enum):
    CRITERION_LEVEL = "criterion_level"
    HOLISTIC = "holistic"

class RaterKind(str, Enum):
    HUMAN = "human"
    AUTOMATED = "automated"

@dataclass(frozen=True)
class EvidenceSpan(CanonicalContract):
    source_id: str
    start_offset: int
    end_offset: int
    evidence_label: str | None = None
    content_digest: str | None = None

@dataclass(frozen=True)
class ScoreObservation(CanonicalContract):
    observation_id: str
    assessment_fingerprint: str
    rubric_fingerprint: str
    response_id: str
    rater_id: str
    rater_kind: RaterKind
    engine_id: str | None
    construct_id: str
    observation_level: ObservationLevel
    criterion_id: str | None
    status: ObservationStatus
    score_category: int | None
    reason_code: str | None
    confidence: float | None
    evidence_spans: tuple[EvidenceSpan, ...]
    occasion_id: str
    scorer_family: str
    scorer_version: str
    prompt_template_version: str | None
    metadata: Mapping[str, JSONValue]
    observation_fingerprint: str
    observation_handle: str

def build_score_observation(
    *,
    assessment: AssessmentSpec,
    rubrics: Iterable[RubricSpecification],
    observation_id: str,
    rubric_fingerprint: str,
    response_id: str,
    rater_id: str,
    rater_kind: RaterKind | str,
    engine_id: str | None,
    construct_id: str,
    observation_level: ObservationLevel | str,
    criterion_id: str | None,
    status: ObservationStatus | str,
    score_category: int | None,
    reason_code: str | None,
    confidence: float | None = None,
    evidence_spans: Iterable[EvidenceSpan] = (),
    occasion_id: str,
    scorer_family: str,
    scorer_version: str,
    prompt_template_version: str | None = None,
    metadata: Mapping[str, JSONValue] | None = None,
) -> ScoreObservation: ...

def validate_observations(
    observations: Iterable[ScoreObservation],
    *,
    assessment: AssessmentSpec,
    rubrics: Iterable[RubricSpecification],
    minimum: int = 1,
    maximum: int = 1_000_000,
) -> tuple[ScoreObservation, ...]: ...
```

- [ ] **Step 1: Write RED tests for enum, evidence, and scored-observation behavior**

Create fixture helpers that build one two-level criterion rubric, one five-level holistic rubric, and one merged-#473 `AssessmentSpec`. Add tests equivalent to:

```python
def test_automated_scored_observation_binds_exact_provenance() -> None:
    observation = build_score_observation(
        assessment=approved_assessment(),
        rubrics=approved_rubrics(),
        observation_id="argument_score_observation",
        rubric_fingerprint=argument_rubric().fingerprint,
        response_id="essay_response_alpha",
        rater_id="automated_rater_alpha",
        rater_kind=RaterKind.AUTOMATED,
        engine_id="deterministic_engine",
        construct_id="argument_quality",
        observation_level=ObservationLevel.CRITERION_LEVEL,
        criterion_id="evidence_alignment",
        status=ObservationStatus.SCORED,
        score_category=1,
        reason_code=None,
        confidence=0.75,
        evidence_spans=(
            EvidenceSpan(
                source_id="essay_response_alpha",
                start_offset=4,
                end_offset=12,
                evidence_label="supporting_span",
            ),
        ),
        occasion_id="scoring_occasion_alpha",
        scorer_family="deterministic_fixture",
        scorer_version="1.0.0",
        prompt_template_version="1.0.0",
    )
    assert observation.status is ObservationStatus.SCORED
    assert observation.score_category == 1
    assert observation.assessment_fingerprint == approved_assessment().assessment_fingerprint
    assert len(observation.observation_fingerprint) == 64
    assert observation.observation_handle.startswith("score_observation_")
```

Add tests that score category `2` fails for a two-level rubric, offsets reject `end_offset <= start_offset`, and evidence serialization contains no raw content field.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
pytest tests/test_scoring_observations.py -q
```

Expected: collection/import failure because `fast_mlsirm.scoring.observations` does not exist.

- [ ] **Step 3: Implement enums, `EvidenceSpan`, canonical content, and sealed construction**

Use the merged internal helpers from `_validation.py` and `_contract_safety.py`. Set explicit limits:

```python
MAX_EVIDENCE_SPANS = 64
MAX_OBSERVATIONS = 1_000_000
MAX_EVIDENCE_OFFSET = (1 << 63) - 1
```

`EvidenceSpan._content_dict()` must return exactly:

```python
{
    "source_id": self.source_id,
    "start_offset": self.start_offset,
    "end_offset": self.end_offset,
    "evidence_label": self.evidence_label,
    "content_digest": self.content_digest,
}
```

Use a package-owned factory token, as `AssessmentSpec` does, so direct public dataclass construction cannot forge derived fingerprints.

- [ ] **Step 4: Write RED tests for every status and observation level**

Add parameterized tests covering:

```python
@pytest.mark.parametrize(
    ("status", "score", "reason"),
    [
        (ObservationStatus.MISSING, None, None),
        (ObservationStatus.ABSTAINED, None, "insufficient_evidence"),
        (ObservationStatus.FAILED, None, "engine_execution_failed"),
        (ObservationStatus.EXCLUDED, None, "response_out_of_scope"),
    ],
)
def test_non_scored_states_remain_distinct(status, score, reason): ...
```

Also assert:

- `SCORED` requires `score_category` and forbids `reason_code`.
- `MISSING` forbids score and reason.
- `ABSTAINED`, `FAILED`, and `EXCLUDED` forbid score and require a descriptive reason code.
- Criterion-level requires `criterion_id`.
- Holistic forbids `criterion_id`.
- Human forbids `engine_id`.
- Automated requires `engine_id` declared by `assessment.engine_policy.engine_ids`.

- [ ] **Step 5: Implement factory cross-reference and status invariants**

The factory must:

1. materialize the rubric registry within `MAX_ASSESSMENT_RUBRICS`;
2. reject duplicate rubric IDs/fingerprints;
3. verify the assessment fingerprint exactly;
4. map the selected rubric fingerprint to one assessment construct;
5. validate `score_category` against `tuple(level.score for level in rubric.levels)`;
6. validate rater/engine consistency against `EnginePolicy`;
7. sort evidence spans deterministically by `(source_id, start_offset, end_offset, evidence_label or "", content_digest or "")` without merging overlaps;
8. freeze bounded metadata;
9. derive full SHA-256 content identity and `score_observation_<32 hex>` handle.

- [ ] **Step 6: Write RED safety and adversarial tests**

Cover:

- a generator that yields more than `MAX_EVIDENCE_SPANS`;
- a generator that raises `RuntimeError("secret response")` during iteration;
- a custom `__index__` offset that raises arbitrary `Exception`;
- lone-surrogate identifiers/text;
- `-0.0` confidence canonicalization;
- NaN/infinite/out-of-range confidence;
- metadata keys such as `Response_Text`, `RAW_RESPONSE`, and `source_content`;
- duplicate observation IDs and fingerprints;
- direct dataclass construction without the factory token;
- validation errors that do not contain caller values or callback exception text.

- [ ] **Step 7: Implement confidence, hostile-callback, and batch-validation hardening**

Confidence is optional scorer-reported metadata only:

```python
if confidence is not None:
    confidence = finite_float(confidence, "confidence", minimum=0.0, maximum=1.0)
    if confidence == 0.0:
        confidence = 0.0
```

`validate_observations()` must preserve input order, reject non-`ScoreObservation` values with index-only paths, reject duplicates, and verify every observation's assessment/rubric/construct cross-reference without recomputing or changing the observation.

- [ ] **Step 8: Verify Task 1 and commit**

Run:

```bash
pytest \
  tests/test_scoring_observations.py \
  tests/test_scoring_observation_safety.py \
  --cov=fast_mlsirm.scoring.observations \
  --cov-branch --cov-fail-under=100 -q
```

Expected: all tests pass and the new module reports 100% statement and branch coverage.

Commit:

```bash
git add python/fast_mlsirm/scoring tests/scoring_observation_fixtures.py \
  tests/test_scoring_observations.py tests/test_scoring_observation_safety.py
git commit -m "feat(scoring): add lossless score observations"
```

---

### Task 2: Provider-Neutral Engine Protocol and Deterministic Fixture

**Files:**
- Create: `python/fast_mlsirm/scoring/engines.py`
- Create: `tests/test_scoring_engines.py`
- Create: `tests/test_scoring_engine_safety.py`
- Modify: `python/fast_mlsirm/scoring/__init__.py`
- Modify: `python/fast_mlsirm/scoring/contracts.py`

**Interfaces:**
- Consumes: `AssessmentSpec`, `ObservationLevel`, `RaterKind`, `ScoreObservation`, `build_score_observation()`, and exact rubric fingerprints.
- Produces:

```python
RequestT = TypeVar("RequestT", contravariant=True)

@dataclass(frozen=True)
class EngineDescriptor(CanonicalContract):
    engine_id: str
    engine_family: str
    engine_version: str
    rater_kind: RaterKind
    supported_rubric_fingerprints: tuple[str, ...]
    supported_observation_levels: tuple[ObservationLevel, ...]
    descriptor_fingerprint: str
    descriptor_handle: str

@dataclass(frozen=True)
class ScoringResult(CanonicalContract):
    result_id: str
    assessment_fingerprint: str
    engine_descriptor_fingerprint: str
    request_fingerprint: str
    observations: tuple[ScoreObservation, ...]
    result_fingerprint: str
    result_handle: str

@runtime_checkable
class ScoringEngine(Protocol[RequestT]):
    @property
    def descriptor(self) -> EngineDescriptor: ...

    def score(self, request: RequestT) -> ScoringResult: ...

class DeterministicFixtureEngine(Generic[RequestT]):
    def __init__(
        self,
        *,
        descriptor: EngineDescriptor,
        rule: Callable[[RequestT], Iterable[ScoreObservation]],
        request_fingerprint: Callable[[RequestT], str],
        assessment_fingerprint: Callable[[RequestT], str],
    ) -> None: ...

    @property
    def descriptor(self) -> EngineDescriptor: ...

    def score(self, request: RequestT) -> ScoringResult: ...
```

- [ ] **Step 1: Write RED descriptor and protocol tests**

Add tests that descriptor input order cannot change identity, capabilities are non-empty and unique, `RaterKind.HUMAN` descriptors cannot represent an automated engine, and `isinstance(fixture, ScoringEngine)` succeeds for the runtime-checkable protocol.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
pytest tests/test_scoring_engines.py -q
```

Expected: import failure because `fast_mlsirm.scoring.engines` does not exist.

- [ ] **Step 3: Implement immutable `EngineDescriptor` and protocol**

Build descriptors through `build_engine_descriptor(...)`; seal direct construction; normalize supported fingerprints and levels into sorted unique tuples; derive SHA-256 identity and `scoring_engine_<32 hex>` handle. The first fixture engine supports automated raters only; a future human adapter is separate and must not pretend to execute an engine.

- [ ] **Step 4: Write RED result and fixture-engine tests**

Create a tiny request fixture:

```python
@dataclass(frozen=True)
class FixtureRequest:
    request_id: str
    assessment_fingerprint: str
    request_fingerprint: str
```

Test:

- deterministic rules produce byte-identical canonical results;
- every observation uses the request assessment fingerprint;
- every observation's engine ID matches the descriptor;
- unsupported rubric fingerprints or observation levels fail closed;
- empty results are rejected;
- result order is deterministic by observation identity rather than callback iteration order;
- scored, abstained, and failed observations all survive unchanged.

- [ ] **Step 5: Implement `ScoringResult` and deterministic fixture execution**

`DeterministicFixtureEngine.score()` must:

1. obtain request and assessment fingerprints through injected pure accessors;
2. call the rule once;
3. bounded-materialize at most `MAX_RESULT_OBSERVATIONS = 4_096` observations;
4. validate descriptor capabilities and engine/assessment provenance;
5. sort observations by `observation_id` for deterministic execution evidence;
6. create `ScoringResult` through a factory-sealed constructor.

It must not infer or alter scores.

- [ ] **Step 6: Write RED arbitrary-rule failure tests**

Add a rule that raises `RuntimeError("raw essay secret")` and assert `score()` returns one `FAILED` observation only when the engine was created with an injected `failure_observation` factory:

```python
engine = DeterministicFixtureEngine(
    descriptor=descriptor,
    rule=hostile_rule,
    request_fingerprint=lambda request: request.request_fingerprint,
    assessment_fingerprint=lambda request: request.assessment_fingerprint,
    failure_observation=lambda request: build_failed_observation(
        request,
        reason_code="engine_execution_failed",
    ),
)
```

Assert canonical JSON and exception messages do not contain `"raw essay secret"`. If no failure factory is supplied, raise `AssessmentSpecError(code="engine_execution_failed", path="$.engine_execution", ...)` without reflecting the callback exception.

- [ ] **Step 7: Implement failure normalization and capability safety**

Catch arbitrary `Exception` from descriptor accessors and rule execution while re-raising `AssessmentSpecError`; never catch `BaseException`. Validate returned objects before reading caller-controlled properties. Reject hostile iterators with stable `invalid_engine_result` errors.

- [ ] **Step 8: Verify Task 2 and commit**

Run:

```bash
pytest \
  tests/test_scoring_engines.py \
  tests/test_scoring_engine_safety.py \
  --cov=fast_mlsirm.scoring.engines \
  --cov-branch --cov-fail-under=100 -q
```

Expected: all tests pass and the engine module reports 100% statement and branch coverage.

Commit:

```bash
git add python/fast_mlsirm/scoring tests/test_scoring_engines.py \
  tests/test_scoring_engine_safety.py
git commit -m "feat(scoring): add provider-neutral engine contracts"
```

---

### Task 3: Documentation, Release Evidence, and Full Verification

**Files:**
- Create: `docs/scoring_observation_engine_contracts.md`
- Create: `docs/changelog.d/482-scoring-observation-engine-contracts.md`
- Modify: `CHANGELOG.md`
- Modify: PR body after verification

**Interfaces:**
- Documents all public Task 1-2 interfaces and the boundary consumed later by #397 and #404.

- [ ] **Step 1: Write the user-facing contract guide**

Document:

1. state semantics and why non-scored states are not numeric missing values;
2. criterion-level versus holistic separation;
3. assessment/rubric/engine/rater/request/occasion provenance;
4. evidence offsets without raw content;
5. automated versus human rater boundaries;
6. deterministic fixture usage;
7. modular/standalone and MSA use;
8. explicit non-claims: schema validity is not construct validity, reliability, fairness, calibration, authorization, or high-stakes readiness;
9. next handoff to Rust-backed faceted calibration.

Include APA 7th references already governing the shared scoring layer:

- American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*.
- Williamson, D. M., Xi, X., & Breyer, F. J. (2012). A framework for evaluation and use of automated scoring. *Educational Measurement: Issues and Practice, 31*(1), 2-13. https://doi.org/10.1111/j.1745-3992.2011.00223.x

- [ ] **Step 2: Add authoritative changelog material and render it**

Create a fragment with `## Added` and a descriptive heading. Run:

```bash
python scripts/render_changelog_fragments.py --update CHANGELOG.md
python scripts/render_changelog_fragments.py --check CHANGELOG.md
```

Expected: update changes `CHANGELOG.md`; check exits successfully without modifying files.

- [ ] **Step 3: Run focused public-API, coverage, and docstring gates**

Run:

```bash
pytest \
  tests/test_scoring_observations.py \
  tests/test_scoring_observation_safety.py \
  tests/test_scoring_engines.py \
  tests/test_scoring_engine_safety.py \
  --cov=fast_mlsirm.scoring.observations \
  --cov=fast_mlsirm.scoring.engines \
  --cov-branch --cov-fail-under=100 -q
```

Run the repository docstring checker used by CI and confirm every new public symbol is documented.

- [ ] **Step 4: Run complete repository verification**

Run or obtain exact-head evidence for:

```bash
pytest
cargo test --workspace
cargo test --manifest-path crates/fast-mlsirm-py/Cargo.toml
python -m build --no-isolation
```

Also require the existing CI jobs for explicit GPU parity, fuzzing, package/release acceptance, Security Scan, and SAST Semgrep to succeed on the same head.

- [ ] **Step 5: Final scope and safety review**

Confirm:

- no psychometric formula or estimator was added in Python;
- no provider SDK or network call was introduced;
- raw content and callback error text cannot enter errors, digests, or default serialization;
- the observation and engine contracts remain generic enough for essay, RAG, and issue-intelligence adapters;
- all issue #482 acceptance criteria map to tests and documentation;
- no temporary write-capable or self-modifying workflow exists.

- [ ] **Step 6: Commit final evidence**

```bash
git add docs CHANGELOG.md python/fast_mlsirm/scoring

git commit -m "docs(scoring): document observation and engine contracts"
```

- [ ] **Step 7: Publish a draft PR**

Use title:

```text
feat(scoring): add lossless observations and engine contracts
```

The body must state:

- product gap and vertical slice;
- exact status/provenance invariants;
- no-scoring/no-validity architectural boundary;
- exact-head verification gate;
- review focus;
- `Closes #482`, `Advances #397 and #404`.

Keep the PR draft until all exact-head checks and final current-head review pass.
