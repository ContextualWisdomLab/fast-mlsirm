# Automated Scoring Core Design

## Objective

Implement the operational core of the *Handbook of Automated Scoring* inside the existing `fast-mlsirm` repository. The work must not create a new repository or a separately versioned package. The installation unit remains `fast-mlsirm`, while the public Python API is exposed under `fast_mlsirm.scoring` and computationally intensive psychometric routines continue to use the existing Rust/PyO3 path.

## Scope

The first delivery covers the operational foundation required for trustworthy automated scoring:

1. assessment design and reproducible scoring contracts;
2. rubric and score-level definitions;
3. human-rater observations and assignment metadata;
4. automated-scoring engine adapters;
5. score calibration through existing IRT, many-facet, linking, DIF, fit, and G-theory capabilities;
6. validation and fairness gates;
7. human-machine adjudication;
8. score and evaluator drift monitoring;
9. immutable audit records and version provenance.

The first delivery does not implement domain-specific NLP, speech, handwriting, or multimodal models. Those systems integrate through a stable adapter interface.

## Architecture

```text
fast_mlsirm/
  scoring/
    __init__.py
    contracts.py
    rubric.py
    observations.py
    engines.py
    calibration.py
    validation.py
    adjudication.py
    monitoring.py
    audit.py
```

No independent microservice is required for the initial implementation. The modules are designed as bounded components with serializable contracts so they can later be embedded in services without changing the core domain model.

### Dependency direction

```text
contracts <- rubric
contracts <- observations
contracts <- engines
contracts + observations <- calibration
contracts + calibration <- validation
contracts + validation <- adjudication
contracts + observations + validation <- monitoring
all modules -> audit
```

`scoring` may call stable public functions from the existing psychometric package. Existing numerical modules must not import `scoring`.

## Core domain contracts

### `AssessmentSpec`

A frozen, versioned specification describing what is scored and how evidence is interpreted.

Required fields:

- `assessment_id: str`
- `assessment_version: str`
- `constructs: tuple[ConstructSpec, ...]`
- `rubrics: tuple[RubricSpec, ...]`
- `response_type: ResponseType`
- `engine_policy: EnginePolicy`
- `calibration_policy: CalibrationPolicy`
- `validation_policy: ValidationPolicy`
- `adjudication_policy: AdjudicationPolicy`
- `monitoring_policy: MonitoringPolicy`
- `metadata: Mapping[str, JsonValue]`

Validation rules:

- identifiers are non-empty non-numeric strings;
- all referenced construct and rubric identifiers exist;
- score levels are ordered and unique;
- policies cannot reference an engine, group, or threshold that is not declared;
- canonical JSON serialization is deterministic;
- a SHA-256 digest identifies the exact contract.

### `RubricSpec`

A rubric defines observable evidence, ordered score levels, and anchors. It does not contain executable model logic.

Required fields:

- `rubric_id: str`
- `construct_id: str`
- `levels: tuple[ScoreLevel, ...]`
- `evidence_requirements: tuple[EvidenceRequirement, ...]`
- `anchors: tuple[AnchorExample, ...]`

### `ScoreObservation`

One human or automated rating event.

Required fields:

- `observation_id: str`
- `assessment_id: str`
- `assessment_version: str`
- `response_id: str`
- `item_id: str`
- `rubric_id: str`
- `rater_id: str`
- `rater_kind: human | automated`
- `engine_id: str | None`
- `engine_version: str | None`
- `score: int | float | None`
- `status: scored | abstained | failed | excluded`
- `evidence: tuple[EvidenceSpan, ...]`
- `created_at: datetime`
- `metadata: Mapping[str, JsonValue]`

Missing, abstained, failed, and excluded observations remain distinguishable. They must never be silently converted to zero.

## Engine adapter

```python
class ScoringEngine(Protocol):
    @property
    def descriptor(self) -> EngineDescriptor: ...

    def score(self, request: ScoringRequest) -> ScoringResult: ...
```

The adapter contract requires:

- deterministic engine identity and version;
- declared supported response and rubric types;
- structured evidence output;
- explicit abstention and failure states;
- latency and token or compute usage metadata when available;
- no provider-specific types in public scoring contracts.

The first implementation supplies a deterministic rule-engine fixture for tests. External LLM adapters remain optional integrations.

## Calibration integration

`CalibrationDataset` converts observations into validated matrices and metadata accepted by existing functions such as `fit_facets`, `fit`, linking, DIF, fit diagnostics, and G-theory routines.

The initial calibration API is orchestration rather than a new estimator:

```python
def calibrate_scores(
    spec: AssessmentSpec,
    observations: Sequence[ScoreObservation],
    *,
    model: CalibrationModel,
) -> CalibrationResult:
    ...
```

The result records:

- selected estimator and backend;
- exact input-observation digest;
- person, item, and rater mappings;
- fitted parameters;
- connectedness and convergence diagnostics;
- excluded observations and reasons;
- package, Rust-core, and model versions.

## Validation and fairness

A validation run evaluates a candidate engine against a declared policy. Supported initial metrics are those already available or derivable without introducing a new statistical estimator:

- exact agreement and adjacent agreement;
- quadratic weighted kappa;
- Pearson and Spearman association;
- standardized mean difference overall and by declared subgroup;
- human-human versus human-machine degradation;
- item and person fit summaries;
- DIF results where group sample requirements are met;
- missingness, abstention, and failure rates;
- rater severity and connectedness diagnostics;
- reproducibility under repeated deterministic fixtures.

Every metric returns `pass`, `fail`, or `insufficient_evidence`. Small or disconnected samples must not be reported as passing.

## Adjudication

Adjudication is policy-driven. A response is routed to human review when one or more declared conditions hold:

- human-machine score distance exceeds a threshold;
- calibrated uncertainty exceeds a threshold;
- an engine abstains or fails;
- evidence requirements are not satisfied;
- subgroup or DIF policy marks the case as sensitive;
- engine versions disagree beyond the allowed tolerance.

The result records the triggering rules and never overwrites original observations.

## Monitoring

Monitoring consumes time-ordered observations and validation summaries. Initial detectors cover:

- score-distribution drift;
- abstention and failure-rate drift;
- engine-version changes;
- rater-severity drift;
- subgroup metric drift;
- contract-version mismatch.

A monitor emits evidence-bearing alerts. Thresholds and windows are defined in `MonitoringPolicy`; defaults are not silently invented.

## Audit and provenance

Every artifact has a stable string identifier and canonical digest. Audit events are append-only values containing actor, operation, timestamp, input digests, output digests, software versions, and decision reason. The library does not provide user authentication or external persistence in this phase.

## Error handling

Public APIs use domain-specific exceptions derived from `AutomatedScoringError`:

- `InvalidAssessmentSpecError`
- `UnsupportedScoringRequestError`
- `ObservationValidationError`
- `CalibrationDataError`
- `ValidationEvidenceError`
- `AdjudicationPolicyError`
- `MonitoringConfigurationError`

Exceptions include machine-readable codes and context without embedding sensitive response text.

## Testing and quality gates

- Public modules and public members require complete docstrings.
- New Python code must maintain 100% branch and statement coverage.
- Serialization tests verify deterministic canonical output and digest stability.
- Contract tests verify engine substitutability.
- Property tests cover ordering, identifier, and missing-state invariants.
- Calibration adapter tests compare generated matrices with hand-constructed expected matrices.
- Rust is used for any new numerical kernels; Python may orchestrate but must not duplicate computational estimators.
- Existing test suites, Rust workspace tests, formatting, linting, and documentation checks must pass.

## Security and privacy

- response text is excluded from exceptions, digests, and default logs;
- metadata values are treated as untrusted input;
- adapters cannot execute arbitrary code from a rubric or assessment contract;
- canonical serialization rejects non-finite numeric values;
- identifiers use descriptive strings rather than sequential numeric IDs.

## Initial acceptance criteria

1. A user can define and serialize an `AssessmentSpec` with rubric, engine, calibration, validation, adjudication, and monitoring policies.
2. Human and automated observations share one lossless schema.
3. A deterministic engine fixture can score a request and emit evidence, abstention, or failure.
4. Observations can be converted into a many-facet calibration dataset with explicit mappings and missing values.
5. A validation report evaluates declared gates and distinguishes insufficient evidence from success.
6. Adjudication returns transparent rule triggers while preserving source observations.
7. Monitoring detects configured changes and records contract and engine versions.
8. All new public APIs are documented and all new code has 100% coverage.
