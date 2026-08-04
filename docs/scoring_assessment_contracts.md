# Shared assessment and scoring-policy contracts

`fast_mlsirm.scoring` begins the operational automated-scoring layer with one
immutable contract that binds exact rubric versions to scoring-engine,
calibration, validation, adjudication, monitoring, and reporting policies.
The contract is intentionally domain-neutral so essay evaluation, RAG
evaluation, enterprise issue intelligence, and future assessment adapters can
reuse the same provenance boundary without creating incompatible rubric or
policy schemas.

## Product boundary

The assessment contract is a **governed specification artifact**. It establishes
what constructs and rubric fingerprints an assessment intends to use and which
policy identifiers must control the downstream workflow. It does not itself:

- score a response;
- fit a psychometric model;
- establish construct validity, reliability, fairness, or regulatory fitness;
- authenticate a caller or authorize data access; or
- replace human governance for high-stakes score use.

Subsequent orchestration layers must retain this contract's fingerprint and
delegate numerical estimation to the existing Rust-backed APIs.

## Contract graph

```text
RubricSpecification[]
        │ exact SHA-256 fingerprints
        ▼
ConstructSpec[]
        │
        ├── EnginePolicy
        ├── CalibrationPolicy
        ├── ValidationPolicy
        ├── AdjudicationPolicy
        ├── MonitoringPolicy
        └── ReportingPolicy
        │
        ▼
AssessmentSpec
        ├── assessment_fingerprint  # full SHA-256 identity
        └── assessment_handle       # descriptive 128-bit public handle
```

`fast_mlsirm.rubric.RubricSpecification` remains the sole rubric source of
truth. `fast_mlsirm.scoring` stores rubric fingerprints rather than copying or
redefining score levels, response formats, or construct definitions.

## Example

```python
from fast_mlsirm.rubric import RubricSpecification
from fast_mlsirm.scoring import (
    AdjudicationPolicy,
    AssessmentResponseType,
    CalibrationPolicy,
    ConstructSpec,
    EnginePolicy,
    MonitoringPolicy,
    ReportingPolicy,
    ValidationPolicy,
    build_assessment_spec,
)

rubrics: tuple[RubricSpecification, ...] = load_approved_rubrics()
argument_rubric = next(
    rubric for rubric in rubrics if rubric.rubric_id == "argument_rubric"
)

assessment = build_assessment_spec(
    assessment_id="essay_assessment",
    assessment_version="1.0.0",
    constructs=(
        ConstructSpec(
            construct_id="argument_quality",
            construct_definition="Quality of the response argument.",
            rubric_fingerprints=(argument_rubric.fingerprint,),
        ),
    ),
    rubrics=(argument_rubric,),
    response_type=AssessmentResponseType.CRITERION_LEVEL,
    engine_policy=EnginePolicy(
        policy_id="engine_policy",
        engine_ids=("production_engine",),
        allow_human_raters=True,
        allow_automated_raters=True,
        minimum_raters_per_response=2,
    ),
    calibration_policy=CalibrationPolicy(
        policy_id="calibration_policy",
        model_id="facets_ordinal",
        construct_ids=("argument_quality",),
    ),
    validation_policy=ValidationPolicy(
        policy_id="validation_policy",
        metric_ids=("quadratic_weighted_kappa",),
        construct_ids=("argument_quality",),
    ),
    adjudication_policy=AdjudicationPolicy(
        policy_id="adjudication_policy",
        trigger_ids=("scorer_disagreement",),
        construct_ids=("argument_quality",),
    ),
    monitoring_policy=MonitoringPolicy(
        policy_id="monitoring_policy",
        metric_ids=("severity_drift",),
        construct_ids=("argument_quality",),
    ),
    reporting_policy=ReportingPolicy(
        policy_id="reporting_policy",
        format_ids=("json_report", "html_report"),
        construct_ids=("argument_quality",),
        include_exact_values=True,
    ),
    metadata={"study_name": "Connected sparse pilot"},
)

print(assessment.assessment_handle)
print(assessment.assessment_fingerprint)
```

## Fail-closed invariants

The builder rejects an assessment before storage when:

- an identifier is numeric, one-token, malformed, duplicated, or ambiguous;
- a rubric fingerprint is unknown, duplicated, unused, or attached to a
  different construct;
- one rubric identifier names multiple content fingerprints;
- a calibration, validation, adjudication, monitoring, or reporting policy
  references an undeclared construct;
- automated raters are enabled without a declared engine;
- no human or automated rater kind is available;
- metadata contains non-finite numbers, unsafe keys, unsupported objects,
  excessive depth, excessive node counts, or oversized collections; or
- direct construction attempts to bypass the cross-reference-validating
  factory.

Input order does not change the resulting assessment identity. Nested metadata
is copied into immutable mappings and tuples so later caller mutation cannot
alter a signed or stored contract.

## Modular and MSA use

The library object is serializable and content-addressed, which allows a service
to store or transmit it as a domain contract. Authentication, authorization,
tenant isolation, persistence, workflow state, and message delivery remain
outside this package slice. A standalone Python user and a distributed service
therefore consume the same assessment representation without coupling the
psychometric core to a particular database, queue, hosted model provider, or
web framework.

## Required downstream evidence

A valid `AssessmentSpec` is only the first evidence link. Operational use still
requires:

1. lossless human and automated observation contracts;
2. connected rating-design checks;
3. Rust-backed calibration and parameter-recovery evidence;
4. validation and fairness gates with explicit insufficient-evidence states;
5. transparent human-review routing;
6. evaluator, rubric, and score drift monitoring; and
7. reconstructable reports and audit records.

## References

American Educational Research Association, American Psychological Association,
& National Council on Measurement in Education. (2014). *Standards for
educational and psychological testing*. American Educational Research
Association.

Williamson, D. M., Xi, X., & Breyer, F. J. (2012). A framework for evaluation
and use of automated scoring. *Educational Measurement: Issues and Practice,
31*(1), 2–13. https://doi.org/10.1111/j.1745-3992.2011.00223.x
