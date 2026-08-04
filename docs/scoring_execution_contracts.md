# Scoring Execution Contracts

`fast_mlsirm.scoring` provides one provider-neutral request and observation boundary for human raters, deterministic fixtures, and later external automated-scoring adapters. It is the next layer after `AssessmentSpec`: the assessment declares the allowed constructs, rubrics, engines, policies, and reporting boundary; a `ScoringRequest` declares one exact response instance without retaining its raw text.

## Execution flow

```text
AssessmentSpec + RubricSpecification
        ↓
build_scoring_request(...)
        ↓
ScoringEngine.score(request)
        ↓
ScoreObservation[]
        ↓
ScoringResult
```

## Example

```python
from fast_mlsirm.scoring import (
    EngineKind,
    FixtureOutcome,
    ObservationGranularity,
    ObservationStatus,
    StaticFixtureEngine,
    build_engine_descriptor,
    build_scoring_request,
)

engine = build_engine_descriptor(
    engine_id="fixture_engine",
    engine_family_id="fixture_family",
    provider_id="local_provider",
    engine_version="1.0.0",
    engine_kind=EngineKind.AUTOMATED,
    model_id="fixture_model",
    prompt_driven=True,
    prompt_template_fingerprint="a" * 64,
    metadata={"deterministic_mode": True},
)

request = build_scoring_request(
    request_id="evidence_request",
    assessment=assessment_spec,
    rubric=rubric_specification,
    granularity=ObservationGranularity.CRITERION_LEVEL,
    respondent_id="sample_respondent",
    response_id="sample_response",
    task_id="sample_task",
    task_revision_fingerprint=task_sha256,
    task_family_id="evidence_review",
    occasion_id="initial_occasion",
    criterion_ids=("claim_support", "source_alignment"),
    response_content_fingerprint=response_sha256,
    response_character_count=128,
    response_unit_count=8,
)

fixture = StaticFixtureEngine(
    descriptor=engine,
    outcomes=(
        FixtureOutcome(
            criterion_id="claim_support",
            status=ObservationStatus.SCORED,
            score_category=2,
        ),
        FixtureOutcome(
            criterion_id="source_alignment",
            status=ObservationStatus.ABSTAINED,
            reason_code="insufficient_evidence",
        ),
    ),
)

result = fixture.score(request)
```

The fixture engine exists only for deterministic tests, examples, and offline contract demonstrations. It is not a model and does not infer a score.

## Status semantics

- `scored`: contains one score category declared by the exact bound rubric and no terminal reason.
- `abstained`: contains no score and explains why the engine declined to judge.
- `failed`: contains no score and records a stable execution-failure reason.
- `excluded`: contains no score and records a stable policy or data-exclusion reason.

Missing, not applicable, insufficient-evidence, execution-failure, and exclusion states are never coerced into low scores.

## Evidence provenance

`EvidenceReference` stores only:

- descriptive source identity;
- descriptive span identity;
- complete content SHA-256 fingerprint;
- supporting, counter, or context evidence role.

Source text remains outside the governed observation. Duplicate evidence references fail closed and ordering does not change identity.

## Human and automated engines

`EngineDescriptor` keeps human and automated provenance distinct.

- Human engines cannot claim a model or prompt-template identity.
- Automated engines require an exact model identity.
- Prompt-driven engines require an exact prompt-template fingerprint.
- Compiled or deterministic non-prompt engines must not fabricate prompt provenance.

A runtime-checkable `ScoringEngine` protocol allows optional integration packages and services to implement execution without adding provider SDKs to the psychometric core.

## Trust and privacy boundaries

- Raw essay, response, prompt, source, and provider-output text is not stored in governed request, observation, result, or metadata fields.
- Full SHA-256 fingerprints identify exact normalized content; they are not signatures, permissions, or authentication credentials.
- Factory seals are API-governance controls, not a security boundary against arbitrary in-process Python memory mutation.
- Public failures expose stable codes and caller-independent paths without rejected text.
- A structurally valid observation does not establish reliability, fairness, model fit, scoreability, or validity.
- No likelihood, gradient, calibration, DIF, uncertainty, linking, or utility arithmetic is implemented by this module.

## Modular deployment

The same contracts can be used in-process by the standalone package or serialized across an MSA boundary. Assessment, request, engine, observation, evidence, and result fingerprints should be retained in event envelopes, audit stores, model registries, and later Rust-backed calibration handoffs.

## Exact task revisions

Schema `1.1` scoring requests require `task_revision_fingerprint`, a complete SHA-256 identity for the exact normalized task content. `task_id` remains a logical display and administration identifier. Callers must never derive the revision from the logical ID. Legacy schema `1.0` artifacts require `migrate_scoring_request_v1` with an authoritative explicit revision; observations and results bound to the legacy request fingerprint must be produced again.
