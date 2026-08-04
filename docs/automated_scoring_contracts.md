# Automated Scoring Assessment Contracts

`fast_mlsirm.scoring` defines the provider-neutral contract boundary that
connects governed rubrics to human and automated scoring workflows. The first
slice deliberately contains no scoring model, hosted-provider SDK, database, or
new statistical estimator. It establishes immutable assessment and policy
artifacts that later observation, calibration, validation, adjudication, and
monitoring modules can reuse without creating incompatible rubric identities.

## Why this boundary exists

A score is not reproducible when the construct, rubric revision, engine
allowlist, calibration model, validation thresholds, review policy, and drift
policy are scattered across prompts or deployment configuration. An
`AssessmentSpec` binds those decisions to:

- exact `fast_mlsirm.rubric.RubricSpecification` SHA-256 fingerprints;
- declared construct, engine, and subgroup identifiers;
- an explicit response format;
- immutable engine, calibration, validation, adjudication, and monitoring
  policies;
- bounded JSON-compatible metadata; and
- a deterministic full fingerprint plus a descriptive 128-bit public handle.

The contract does **not** treat a rubric fingerprint as authorization. Identity,
authentication, tenant isolation, and persistence remain deployment-layer
responsibilities.

## Example

```python
from fast_mlsirm.rubric import ResponseFormat
from fast_mlsirm.scoring import (
    AdjudicationPolicy,
    CalibrationModel,
    CalibrationPolicy,
    ConstructSpec,
    EnginePolicy,
    GateComparison,
    MonitoringPolicy,
    ValidationGate,
    ValidationPolicy,
    build_assessment_spec,
)

assessment = build_assessment_spec(
    assessment_id="essay_assessment",
    assessment_version="1.0.0",
    constructs=(
        ConstructSpec(
            construct_id="evidence_grounding",
            construct_definition=(
                "Degree to which claims are supported by declared evidence."
            ),
            reporting_label="Evidence grounding",
        ),
    ),
    rubric_fingerprints=(grounding_rubric.fingerprint,),
    response_format=ResponseFormat.ORDINAL_RATING,
    declared_engine_ids=("human_rater", "fixture_engine"),
    declared_group_ids=("all_respondents",),
    engine_policy=EnginePolicy(
        allowed_engine_ids=("fixture_engine",),
        require_evidence=True,
        maximum_attempts=2,
    ),
    calibration_policy=CalibrationPolicy(
        model=CalibrationModel.FACETS,
        minimum_raters=2,
        require_connected_design=True,
        allow_missing_observations=True,
    ),
    validation_policy=ValidationPolicy(
        gates=(
            ValidationGate(
                metric_id="quadratic_weighted_kappa",
                comparison=GateComparison.MINIMUM,
                threshold=0.80,
                minimum_observations=100,
                group_id="all_respondents",
            ),
            ValidationGate(
                metric_id="engine_failure_rate",
                comparison=GateComparison.MAXIMUM,
                threshold=0.02,
                minimum_observations=100,
            ),
        ),
        required_group_ids=("all_respondents",),
    ),
    adjudication_policy=AdjudicationPolicy(
        trigger_codes=("engine_abstention", "score_disagreement"),
        maximum_score_distance=1.0,
        maximum_uncertainty=0.5,
        require_evidence=True,
    ),
    monitoring_policy=MonitoringPolicy(
        window_size=500,
        minimum_observations=100,
        monitored_group_ids=("all_respondents",),
        alert_on_rubric_change=True,
        alert_on_engine_change=True,
    ),
    rubrics=(grounding_rubric,),
    metadata={"owner_team": "psychometrics"},
)
```

The builder replays every reference before constructing the sealed artifact.
Direct `AssessmentSpec(...)` construction fails. In particular:

- every selected rubric fingerprint must exist in the supplied registry;
- every selected rubric construct must be declared;
- every selected rubric must use the assessment response format;
- engine-policy references must be in the declared engine set; and
- validation and monitoring group references must be declared.

## Canonical identity

`canonical_json(assessment)` serializes the content fields with sorted mapping
keys and stable tuple order. `artifact_digest(assessment)` and
`assessment.assessment_fingerprint` are the SHA-256 digest of that content.
`assessment.assessment_handle` is a descriptive public handle containing the
first 128 bits:

```text
assessment_spec_<32 lowercase hexadecimal characters>
```

The full fingerprint remains the authoritative content identity.

Metadata is deeply frozen and bounded before identity calculation. Non-finite
numbers, unsupported objects, oversized collections, excessive nesting, and
invalid mapping keys fail closed without copying rejected values into exception
messages.

## Separation of concerns

This slice intentionally separates five concerns.

1. **Rubric semantics** remain owned by `fast_mlsirm.rubric`.
2. **Assessment policy** is owned by `fast_mlsirm.scoring`.
3. **Rating observations** will preserve scored, missing, abstained, failed, and
   excluded states in the next scoring slice.
4. **Psychometric arithmetic** remains in the existing Rust/PyO3 kernels.
5. **Enterprise identity and persistence** remain outside the library.

`CalibrationPolicy` declares a model family; it does not fit a model.
`ValidationPolicy` declares thresholds; it does not claim that the thresholds
are universally valid. `AdjudicationPolicy` declares review triggers; it does
not overwrite evidence. `MonitoringPolicy` declares windows and provenance
transitions; it does not infer drift from an undeclared estimator.

## Security and privacy boundary

The public validation errors expose stable codes and field names but never
include rejected metadata or response text. Assessment metadata is treated as
untrusted input. Rubric text is data and is never executed. Content fingerprints
provide replay detection and audit identity, not access control.

## MSA compatibility

The contracts are immutable JSON-compatible values with no provider or database
dependency. A standalone Python process can use them directly, while a service
can serialize the same values across an MSA boundary. Database projections
should preserve descriptive two-or-more-token `snake_case` object names and the
full content fingerprints.

## Scientific interpretation boundary

An assessment contract makes a scoring process reproducible; it does not by
itself establish reliability, fairness, validity, scoreability, or fitness for
high-stakes use. Those claims require connected rating designs, estimator
convergence, parameter-recovery evidence, human-anchored validation,
subgroup/DIF analysis, generalization studies, and decision-policy evaluation.

## References

American Educational Research Association, American Psychological Association,
& National Council on Measurement in Education. (2014). *Standards for
educational and psychological testing*. American Educational Research
Association.

Williamson, D. M., Xi, X., & Breyer, F. J. (2012). A framework for evaluation
and use of automated scoring. *Educational Measurement: Issues and Practice,
31*(1), 2–13. https://doi.org/10.1111/j.1745-3992.2011.00223.x
