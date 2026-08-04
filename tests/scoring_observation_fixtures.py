"""Shared deterministic fixtures for scoring-observation contract tests."""

from __future__ import annotations

from fast_mlsirm.rubric import ResponseFormat, RubricLevel, RubricSpecification
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


def argument_rubric() -> RubricSpecification:
    """Return one deterministic ordinal rubric with contiguous score categories."""
    return RubricSpecification(
        rubric_id="argument_rubric",
        construct_id="argument_quality",
        construct_definition="Quality of the response argument.",
        response_format=ResponseFormat.ORDINAL_RATING,
        levels=(
            RubricLevel(
                score=0,
                label="not_demonstrated",
                descriptor="The construct is not demonstrated.",
                observable_indicators=("Required evidence is absent.",),
            ),
            RubricLevel(
                score=1,
                label="partially_demonstrated",
                descriptor="The construct is partially demonstrated.",
                observable_indicators=("Some required evidence is present.",),
            ),
            RubricLevel(
                score=2,
                label="fully_demonstrated",
                descriptor="The construct is fully demonstrated.",
                observable_indicators=("All required evidence is present.",),
            ),
        ),
        task_families=("analytic_response",),
        evidence_requirements=("Cite the exact response evidence used.",),
        rubric_version="1.0.0",
    )


def assessment_spec(*, allow_human: bool = True, allow_automated: bool = True):
    """Return one assessment allowing the requested scorer kinds."""
    rubric = argument_rubric()
    engine_ids = ("fixture_engine",) if allow_automated else ()
    return build_assessment_spec(
        assessment_id="essay_assessment",
        assessment_version="1.0.0",
        constructs=(
            ConstructSpec(
                construct_id="argument_quality",
                construct_definition="Quality of the response argument.",
                rubric_fingerprints=(rubric.fingerprint,),
            ),
        ),
        rubrics=(rubric,),
        response_type=AssessmentResponseType.CRITERION_LEVEL,
        engine_policy=EnginePolicy(
            policy_id="engine_policy",
            engine_ids=engine_ids,
            allow_human_raters=allow_human,
            allow_automated_raters=allow_automated,
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
            format_ids=("json_report",),
            construct_ids=("argument_quality",),
            include_exact_values=True,
        ),
    )
