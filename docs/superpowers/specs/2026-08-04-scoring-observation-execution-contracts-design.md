# Scoring Observation and Engine Execution Contracts Design

## Status and authority

This design implements issue #480 as the second provider-neutral `fast_mlsirm.scoring` slice. It extends the merged `AssessmentSpec` and policy contracts from PR #473. It also supplies the common execution boundary required by automated essay scoring issue #397 and enterprise issue intelligence issue #404.

The design is intentionally limited to immutable workflow contracts. It performs no scoring likelihood, psychometric estimation, reliability calculation, fairness analysis, causal inference, utility calculation, or provider-specific network access.

## Problem statement

The package can now describe an assessment and bind exact rubric fingerprints, but it cannot yet record a trustworthy scoring event. Without a common contract, every domain vertical would invent different representations for:

- respondents, items, criteria, raters, and occasions;
- engine/model/prompt versions;
- observed categories, abstentions, failures, exclusions, and non-applicability;
- evidence and counterevidence provenance;
- request and execution replay protection;
- provider exceptions and partial result sets.

That duplication would break calibration interoperability, audit reconstruction, MSA composition, and version linking.

## Chosen approach

### Recommended: one common analytic-observation graph

The implementation adds one exact identity graph:

```text
AssessmentSpec
    │ assessment_fingerprint
    ▼
ScoringRequest
    │ request_fingerprint
    ▼
ScoringEngine / EngineDescriptor
    │
    ▼
ScoreObservation[]
    │ observation_fingerprint
    ▼
ScoringExecution
    ├── execution_fingerprint
    └── execution_handle
```

Essay, RAG, enterprise-issue, and future adapters may supply domain payloads outside these artifacts, but they must return common `ScoreObservation` records. Raw response or source content never enters the common configuration metadata.

### Rejected alternatives

1. **Domain-specific observation types only.** This gives each vertical convenient fields but produces incompatible calibration and audit handoffs.
2. **One opaque holistic score result.** This hides criterion-level evidence, rater behavior, abstention states, and local failure modes; it cannot support many-facet calibration safely.
3. **Provider SDK objects as public contracts.** This couples the psychometric package to mutable hosted-provider schemas and prevents deterministic offline validation.

## Module boundaries

```text
python/fast_mlsirm/scoring/
├── observations.py   # evidence, values, states, and ScoreObservation
├── engines.py        # EngineDescriptor, ScoringRequest, protocol, fixture engine
├── executions.py     # ScoringExecution and execution orchestration
└── __init__.py       # stable public exports
```

The existing files retain their current ownership:

- `assessment.py`: assessment and rubric-policy graph;
- `contracts.py`: shared canonical artifact helpers and errors;
- `_contract_safety.py` / `_validation.py`: resource and canonicalization boundary;
- `policies.py`: lifecycle policy declarations.

No new repository, package, database, queue, web framework, or hosted SDK is introduced.

## Public types

### Enumerations

`ObservationState`

- `observed`
- `abstained`
- `insufficient_evidence`
- `not_applicable`
- `failed`
- `excluded`

`ExecutionState`

- `completed`
- `failed`

`EvidenceRole`

- `support`
- `counterevidence`
- `context`
- `rationale`

`PairwiseOutcome`

- `left_option`
- `right_option`
- `tie`

### EvidenceReference

Fields:

- `source_id`
- `span_id`
- `content_fingerprint`
- `role`
- `start_offset: int | None`
- `end_offset: int | None`

Offsets are either both absent or both present. Present offsets satisfy `0 <= start_offset < end_offset <= MAX_EVIDENCE_OFFSET`. Identity includes every field. The artifact stores no source text.

### EngineDescriptor

Fields:

- `engine_id`
- `engine_family`
- `engine_version`
- `prompt_template_version`
- `code_fingerprint`
- optional `model_fingerprint`
- optional `data_fingerprint`
- immutable bounded metadata

The descriptor is content-addressed. Provider names may be descriptive metadata, but no provider SDK type crosses the boundary.

### Score values

The canonical observation stores exactly one of these package-owned value objects when `state == observed`:

- `BooleanScoreValue(value: bool)`
- `IntegerScoreValue(value: int)`
- `ContinuousScoreValue(value: float)`
- `NominalScoreValue(value: str)`
- `PairwiseScoreValue(outcome: PairwiseOutcome, preferred_option_id: str | None)`

Values remain distinct; callers cannot silently treat an integer category as a continuous score or aggregate criterion-level values into a holistic score.

Pairwise invariants:

- `tie` requires `preferred_option_id is None`;
- `left_option` or `right_option` requires the matching declared option identifier;
- presentation order remains in the request identity so A/B and B/A are not conflated.

### ScoringRequest

Fields:

- assessment identity and fingerprint;
- construct identifier;
- exact rubric fingerprint;
- respondent identifier;
- item/criterion identifier;
- occasion identifier;
- requested rater identifier;
- optional left/right option identifiers for pairwise work;
- immutable bounded metadata.

Construction requires the actual `AssessmentSpec`. The builder verifies construct and rubric membership before sealing the request. The request stores no raw response, essay, prompt, or source payload; domain adapters retain those in their own bounded request layer and bind them by fingerprints.

### ScoreObservation

Fields:

- request identity and fingerprint;
- assessment, construct, and rubric provenance copied from the request;
- rater identifier;
- observation state;
- score value or reason/failure code according to the state truth table;
- evidence references;
- immutable bounded metadata;
- full observation fingerprint and descriptive public handle.

State truth table:

| State | Score | Reason/failure code | Evidence |
|---|---|---|---|
| observed | required | forbidden | optional |
| abstained | forbidden | required | optional |
| insufficient_evidence | forbidden | required | optional |
| not_applicable | forbidden | required | optional |
| failed | forbidden | required | forbidden |
| excluded | forbidden | required | optional |

No state is converted to `NaN`, zero, empty category, or missing category. Downstream calibration builders must make the conversion policy explicit while preserving the state matrix.

### ScoringEngine and StaticScoringEngine

`ScoringEngine` is a runtime-checkable protocol:

```python
class ScoringEngine(Protocol):
    @property
    def descriptor(self) -> EngineDescriptor: ...

    def score(self, request: ScoringRequest) -> Iterable[ScoreObservation]: ...
```

`StaticScoringEngine` accepts a sealed descriptor and an immutable request-fingerprint-to-observation mapping. It supports deterministic offline tests and examples only.

### ScoringExecution

The execution builder validates the complete engine result before publishing:

- request fingerprint matches;
- engine descriptor is exactly the declared descriptor;
- observation rater/request/assessment/rubric/construct identities match;
- result count is bounded before full materialization;
- duplicate observation fingerprints and duplicate rater-item cells fail;
- completed executions contain at least one valid observation;
- provider exceptions become a failed execution with a stable `engine_execution_failed` code and no reflected exception text;
- partial output followed by provider failure is discarded rather than published as completed evidence.

Fields:

- execution state;
- request fingerprint;
- engine fingerprint;
- ordered observation fingerprints;
- redacted failure code for failed executions;
- immutable metadata;
- full execution fingerprint and descriptive public handle.

The initial slice executes one request at a time. Batch orchestration, retries, rate limits, costs, and network transport remain later infrastructure concerns.

## Response-type validation

The assessment contract declares `criterion_level` or `holistic`. This slice does not infer a new response schema from arbitrary values. It validates observed score values against the referenced `RubricSpecification`:

- Boolean values are accepted only for a two-level rubric whose scores correspond to the declared binary boundary.
- Integer values must be an exact declared rubric score.
- Continuous and nominal values require an explicit request metadata declaration that is validated against package-owned identifiers; they are preserved as observations but are not eligible for ordinal calibration until a later handoff validates the model contract.
- Pairwise values require the request's left/right option identifiers and a pairwise rubric/assessment policy declaration.

Where the merged rubric schema cannot prove a mode safely, construction fails with `unsupported_observation_mode`; it never guesses.

## Identity and canonicalization

Every public artifact is factory-sealed and derives its identity from deterministic strict UTF-8 JSON:

- sorted mapping keys;
- preserved declaration order only where order has semantic meaning;
- full signed-64-bit integer support;
- finite floats only;
- `-0.0` canonicalized to `0.0`;
- full SHA-256 authoritative fingerprint;
- descriptive 128-bit public handle.

Convenience identifiers do not authorize replay. Builders recompute all dependent fingerprints from complete current bodies.

## Error model

`ScoringContractError` extends the shared structured scoring error behavior and exposes:

- two-or-more-token lower `snake_case` code;
- bounded caller-independent JSON-style path;
- bounded non-reflective message.

Ordinary `Exception` from iterators, mappings, numeric conversion, engine code, or package protocol callbacks is redacted. `BaseException` subclasses are not swallowed. Existing structured scoring errors are re-raised unchanged.

## Resource and security controls

The implementation defines explicit finite limits for:

- evidence references per observation;
- observations per execution;
- identifiers, reason codes, metadata text, canonical JSON, and offsets;
- iterator materialization and mapping traversal;
- metadata depth, node count, and collection width.

Cycle detection and iteration-time bounds apply before recursive copying. Sensitive metadata keys are rejected case-insensitively. Artifacts never contain API keys, raw provider output, source text, response text, essays, or prompts.

Python is not a sandbox: callers can execute code in their own object methods. The contract guarantees fail-closed type boundaries and non-reflective errors; it does not claim to neutralize arbitrary code already executing in the caller process.

## Testing strategy

1. Import tests establish the RED public surface.
2. State/value truth-table tests cover every branch.
3. Real `AssessmentSpec` and `RubricSpecification` fixtures prove graph binding.
4. Replay tests mutate one field at every provenance layer and require rejection.
5. Static-engine tests cover completed, failed, partial-then-failed, duplicate, extra, missing, and cross-request results.
6. Hostile Python protocol tests cover iterator creation/advance, mapping inspection, conversion, Unicode, cycles, dishonest lengths, and resource limits.
7. Mutation tests prove deep immutability.
8. Canonicalization tests prove ordering invariance where appropriate and order sensitivity for pairwise presentation/evidence order when semantic.
9. Added code must reach 100% statement and branch coverage and public docstring coverage.
10. Complete exact-head Python, Rust, PyO3, package, GPU-no-skip, fuzz, Security Scan, SAST, and changelog parity remain mandatory.

## Documentation and release boundary

Buyer documentation must state that these artifacts provide provenance and execution-state semantics only. They do not establish scoring validity, reliability, fairness, connectedness, calibration, or fitness for high-stakes use.

The PR adds and renders an authoritative changelog fragment. It does not bump the package version. A release becomes appropriate after the observation/execution layer is connected to at least one Rust-backed calibration and validation workflow.