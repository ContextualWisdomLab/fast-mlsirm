# Automated Scoring Core Design

## Objective

Implement a trustworthy automated-scoring operational core inside the existing
`fast-mlsirm` repository. The work must not create a new repository or a
separately versioned package. The installation unit remains `fast-mlsirm`, the
public orchestration API is exposed under `fast_mlsirm.scoring`, and
computationally intensive psychometric routines continue to use the existing
Rust/PyO3 path.

This design depends on the provider-neutral rubric contracts delivered under
`fast_mlsirm.rubric`. It must **reuse** `RubricSpecification`, `RubricLevel`,
`ResponseFormat`, canonical generation contracts, and their content-addressed
identities. A second rubric schema under `fast_mlsirm.scoring` would create
incompatible fingerprints and is prohibited.

## Scope

The first delivery covers the operational foundation required for trustworthy
automated scoring:

1. assessment design and reproducible scoring contracts;
2. reuse of immutable rubric and score-level definitions;
3. human-rater and automated-rater observations with assignment metadata;
4. automated-scoring engine adapters;
5. score calibration through existing IRT, many-facet, linking, DIF, fit, and
   G-theory capabilities;
6. validation and fairness gates;
7. human-machine adjudication;
8. score and evaluator drift monitoring; and
9. immutable audit records and version provenance.

The first delivery does not implement domain-specific NLP, speech,
handwriting, or multimodal models. Those systems integrate through a stable
adapter interface.

## Architecture

```text
python/fast_mlsirm/
  rubric/                         # existing source of truth
    models.py
    compiler.py
    contracts.py
  scoring/
    __init__.py
    contracts.py                  # AssessmentSpec + policies
    observations.py
    engines.py
    calibration.py
    validation.py
    adjudication.py
    monitoring.py
    audit.py
```

No independent microservice is required for the initial implementation. The
modules are bounded components with serializable contracts so they can later be
embedded in services without changing the core domain model.

### Dependency direction

```text
fast_mlsirm.rubric <- scoring.contracts
scoring.contracts <- scoring.observations
scoring.contracts <- scoring.engines
scoring.contracts + scoring.observations <- scoring.calibration
scoring.contracts + scoring.calibration <- scoring.validation
scoring.contracts + scoring.validation <- scoring.adjudication
scoring.contracts + scoring.observations + scoring.validation <- scoring.monitoring
all scoring modules -> scoring.audit
```

`scoring` may call stable public functions from the existing psychometric
package. Existing numerical modules and `fast_mlsirm.rubric` must not import
`scoring`.

## Core domain contracts

### `AssessmentSpec`

A frozen, versioned specification describing what is scored and how evidence is
interpreted.

Required fields:

- `assessment_id: str`
- `assessment_version: str`
- `constructs: tuple[ConstructSpec, ...]`
- `rubric_fingerprints: tuple[str, ...]`
- `response_type: ResponseType`
- `engine_policy: EnginePolicy`
- `calibration_policy: CalibrationPolicy`
- `validation_policy: ValidationPolicy`
- `adjudication_policy: AdjudicationPolicy`
- `monitoring_policy: MonitoringPolicy`
- `metadata: Mapping[str, JsonValue]`

An `AssessmentSpec` is constructed against an explicit registry of
`fast_mlsirm.rubric.RubricSpecification` values. It records their immutable
fingerprints rather than copying a second rubric representation.

Validation rules:

- identifiers use descriptive two-or-more-token lower `snake_case` values;
- all referenced construct identifiers and rubric fingerprints exist;
- every rubric's `construct_id` resolves to a declared construct;
- policies cannot reference an engine, group, rubric, or threshold that is not
  declared;
- canonical JSON serialization is deterministic; and
- a SHA-256 digest identifies the exact contract.

### Rubric source of truth

Automated scoring consumes, but does not redefine:

- `fast_mlsirm.rubric.RubricSpecification`;
- `fast_mlsirm.rubric.RubricLevel`;
- `fast_mlsirm.rubric.ResponseFormat`; and
- canonical generation-contract and blueprint fingerprints.

A scoring observation records both `rubric_id` and `rubric_fingerprint` so a
rating cannot be replayed under a modified rubric with the same display name.

### `ScoreObservation`

One human or automated rating event.

Required fields:

- `observation_id: str`
- `assessment_id: str`
- `assessment_version: str`
- `response_id: str`
- `item_id: str`
- `rubric_id: str`
- `rubric_fingerprint: str`
- `rater_id: str`
- `rater_kind: human | automated`
- `engine_id: str | None`
- `engine_version: str | None`
- `score: int | float | None`
- `status: scored | abstained | failed | excluded`
- `evidence: tuple[EvidenceSpan, ...]`
- `created_at: datetime`
- `metadata: Mapping[str, JsonValue]`

Missing, abstained, failed, and excluded observations remain distinguishable.
They must never be silently converted to zero. A `scored` observation must use
a score allowed by the exact rubric version, while a non-scored status must not
carry a numeric score.

## Engine adapter

```python
class ScoringEngine(Protocol):
    @property
    def descriptor(self) -> EngineDescriptor: ...

    def score(self, request: ScoringRequest) -> ScoringResult: ...
```

The adapter contract requires:

- deterministic engine identity and version;
- declared supported response formats and rubric fingerprints;
- structured evidence output;
- explicit abstention and failure states;
- latency and token or compute usage metadata when available; and
- no provider-specific types in public scoring contracts.

The first implementation supplies a deterministic rule-engine fixture for
tests. External LLM adapters remain optional integrations.

## Calibration integration

`CalibrationDataset` converts observations into validated matrices and metadata
accepted by existing functions such as `fit_facets`, `fit`, linking, DIF, fit
diagnostics, and G-theory routines.

The initial calibration API is orchestration rather than a new estimator:

```python
def calibrate_scores(
    spec: AssessmentSpec,
    rubrics: Sequence[RubricSpecification],
    observations: Sequence[ScoreObservation],
    *,
    model: CalibrationModel,
) -> CalibrationResult:
    ...
```

The result records:

- selected estimator and backend;
- exact rubric and input-observation digests;
- person, item, and rater mappings;
- fitted parameters;
- connectedness and convergence diagnostics;
- excluded observations and reasons; and
- package, Rust-core, rubric, engine, and model versions.

Python may validate and marshal matrices, but it must not duplicate likelihood,
gradient, fitting, DIF, linking, uncertainty, or utility calculations already
owned by Rust.

## Validation and fairness

A validation run evaluates a candidate engine against a declared policy.
Supported initial metrics are those already available or derivable without
introducing an unreviewed statistical estimator:

- exact agreement and adjacent agreement;
- quadratic weighted kappa;
- Pearson and Spearman association;
- standardized mean difference overall and by declared subgroup;
- human-human versus human-machine degradation;
- item and person fit summaries;
- DIF results where group sample requirements are met;
- missingness, abstention, and failure rates;
- rater severity and connectedness diagnostics; and
- reproducibility under repeated deterministic fixtures.

Every metric returns `pass`, `fail`, or `insufficient_evidence`. Small,
disconnected, or non-identifiable samples must not be reported as passing.

## Adjudication

Adjudication is policy-driven. A response is routed to human review when one or
more declared conditions hold:

- human-machine score distance exceeds a threshold;
- calibrated uncertainty exceeds a threshold;
- an engine abstains or fails;
- evidence requirements are not satisfied;
- subgroup or DIF policy marks the case as sensitive;
- rubric fingerprints do not match; or
- engine versions disagree beyond the allowed tolerance.

The result records all triggering rules and never overwrites original
observations.

## Monitoring

Monitoring consumes time-ordered observations and validation summaries.
Initial detectors cover:

- score-distribution drift;
- abstention and failure-rate drift;
- engine-version changes;
- rubric-fingerprint or assessment-contract changes;
- rater-severity drift;
- subgroup metric drift; and
- contract-version mismatch.

A monitor emits evidence-bearing alerts. Thresholds and windows are defined in
`MonitoringPolicy`; defaults are not silently invented.

## Audit and provenance

Every artifact has a stable descriptive string identifier and canonical digest.
Audit events are append-only values containing actor, operation, UTC timestamp,
input digests, output digests, software versions, rubric fingerprints, and
decision reasons. The library does not provide user authentication or external
persistence in this phase.

## Error handling

Public APIs use domain-specific exceptions derived from
`AutomatedScoringError`:

- `InvalidAssessmentSpecError`
- `UnsupportedScoringRequestError`
- `ObservationValidationError`
- `CalibrationDataError`
- `ValidationEvidenceError`
- `AdjudicationPolicyError`
- `MonitoringConfigurationError`

Exceptions include machine-readable codes and bounded context without embedding
sensitive response text.

## Testing and quality gates

- Public modules and members require complete docstrings.
- New Python code must maintain 100% branch and statement coverage.
- Serialization tests verify deterministic canonical output and digest
  stability.
- Contract tests verify engine substitutability.
- Property tests cover ordering, identifier, rubric-fingerprint, and
  missing-state invariants.
- Calibration adapter tests compare generated matrices with hand-constructed
  expected matrices.
- Any new numerical kernel is implemented in Rust; Python may orchestrate but
  must not duplicate computational estimators.
- Existing test suites, Rust workspace tests, formatting, linting,
  documentation, security, and release checks must pass on the same head.

## Security and privacy

- response text is excluded from exceptions, digests, and default logs;
- metadata values are treated as untrusted input;
- adapters cannot execute arbitrary code from a rubric, assessment, or source
  contract;
- canonical serialization rejects non-finite numeric values;
- identifiers use descriptive strings rather than sequential numeric IDs; and
- artifact digests are content identities, not authentication or authorization
  controls.

## Initial acceptance criteria

1. A user can define and serialize an `AssessmentSpec` bound to exact
   `fast_mlsirm.rubric` fingerprints.
2. Human and automated observations share one lossless schema.
3. A deterministic engine fixture can score a request and emit evidence,
   abstention, or failure.
4. Observations can be converted into a many-facet calibration dataset with
   explicit mappings and missing values.
5. A validation report evaluates declared gates and distinguishes insufficient
   evidence from success.
6. Adjudication returns transparent rule triggers while preserving source
   observations.
7. Monitoring detects configured changes and records assessment, rubric, and
   engine versions.
8. All new public APIs are documented and all new code has 100% branch and
   statement coverage.

## References

American Educational Research Association, American Psychological Association,
& National Council on Measurement in Education. (2014). *Standards for
educational and psychological testing*. American Educational Research
Association.

Williamson, D. M., Xi, X., & Breyer, F. J. (Eds.). (2012). *Automated scoring of
complex tasks in computer-based testing*. Routledge.
