"""Contracts for the provider-neutral automated-scoring assessment foundation."""

from __future__ import annotations

import json
from types import MappingProxyType

import pytest

import fast_mlsirm.scoring as scoring
from fast_mlsirm.rubric import ResponseFormat, RubricLevel, RubricSpecification
from fast_mlsirm.scoring import (
    AdjudicationPolicy,
    AdjudicationRule,
    AssessmentSpec,
    CalibrationPolicy,
    ConstructSpec,
    EnginePolicy,
    InvalidAssessmentSpecError,
    MetricDirection,
    MetricGate,
    MonitoringPolicy,
    MonitoringRule,
    ResponseType,
    ValidationPolicy,
    build_assessment_spec,
)


def _rubric(rubric_id: str, construct_id: str) -> RubricSpecification:
    """Return one deterministic two-level rubric fixture."""
    return RubricSpecification(
        rubric_id=rubric_id,
        construct_id=construct_id,
        construct_definition=f"Observable evidence for {construct_id}.",
        response_format=ResponseFormat.ORDINAL_RATING,
        levels=(
            RubricLevel(
                score=0,
                label="Not demonstrated",
                descriptor="Required evidence is absent.",
                observable_indicators=("No supported claim is present.",),
            ),
            RubricLevel(
                score=1,
                label="Demonstrated",
                descriptor="Required evidence is present.",
                observable_indicators=("A supported claim is present.",),
            ),
        ),
        task_families=("evidence_review",),
        evidence_requirements=("Cite the exact supporting span.",),
    )


def _rubrics() -> tuple[RubricSpecification, RubricSpecification]:
    """Return two rubrics bound to distinct declared constructs."""
    return (
        _rubric("evidence_quality_rubric", "evidence_quality_construct"),
        _rubric("reasoning_quality_rubric", "reasoning_quality_construct"),
    )


def _constructs(
    rubrics: tuple[RubricSpecification, RubricSpecification],
) -> tuple[ConstructSpec, ConstructSpec]:
    """Return construct declarations covering the exact rubric registry."""
    by_construct = {rubric.construct_id: rubric.fingerprint for rubric in rubrics}
    return (
        ConstructSpec(
            construct_id="evidence_quality_construct",
            rubric_fingerprints=(by_construct["evidence_quality_construct"],),
        ),
        ConstructSpec(
            construct_id="reasoning_quality_construct",
            rubric_fingerprints=(by_construct["reasoning_quality_construct"],),
        ),
    )


def _policies(
    rubrics: tuple[RubricSpecification, RubricSpecification],
) -> tuple[
    EnginePolicy,
    CalibrationPolicy,
    ValidationPolicy,
    AdjudicationPolicy,
    MonitoringPolicy,
]:
    """Return internally connected policy fixtures."""
    fingerprints = tuple(rubric.fingerprint for rubric in rubrics)
    engine_policy = EnginePolicy(
        engine_ids=("human_anchor_engine", "automated_fixture_engine"),
        required_engine_ids=("human_anchor_engine",),
        require_evidence=True,
        allow_abstention=True,
    )
    calibration_policy = CalibrationPolicy(
        model_id="many_facet_baseline",
        rubric_fingerprints=fingerprints,
        minimum_observations_per_item=2,
        minimum_observations_per_rater=2,
        require_connected_design=True,
    )
    validation_policy = ValidationPolicy(
        declared_group_ids=("reference_group", "focal_group"),
        metric_gates=(
            MetricGate(
                metric_id="exact_agreement_gate",
                direction=MetricDirection.AT_LEAST,
                threshold=0.80,
                minimum_evidence_count=10,
                rubric_fingerprints=fingerprints,
            ),
        ),
    )
    adjudication_policy = AdjudicationPolicy(
        rules=(
            AdjudicationRule(
                rule_id="engine_disagreement_rule",
                threshold=1.0,
                engine_ids=("automated_fixture_engine",),
                group_ids=("focal_group",),
                rubric_fingerprints=fingerprints,
            ),
        )
    )
    monitoring_policy = MonitoringPolicy(
        rules=(
            MonitoringRule(
                rule_id="severity_drift_rule",
                direction=MetricDirection.AT_MOST,
                threshold=0.25,
                window_size=50,
                engine_ids=("automated_fixture_engine",),
                group_ids=("focal_group",),
                rubric_fingerprints=fingerprints,
            ),
        )
    )
    return (
        engine_policy,
        calibration_policy,
        validation_policy,
        adjudication_policy,
        monitoring_policy,
    )


def _assessment(
    *,
    rubrics: tuple[RubricSpecification, RubricSpecification] | None = None,
    constructs: tuple[ConstructSpec, ...] | None = None,
    metadata: object | None = None,
    engine_policy: EnginePolicy | None = None,
    calibration_policy: CalibrationPolicy | None = None,
    validation_policy: ValidationPolicy | None = None,
    adjudication_policy: AdjudicationPolicy | None = None,
    monitoring_policy: MonitoringPolicy | None = None,
) -> AssessmentSpec:
    """Build one complete assessment contract with optional overrides."""
    registry = rubrics or _rubrics()
    policies = _policies(registry)
    return build_assessment_spec(
        assessment_id="enterprise_issue_assessment",
        assessment_version="1.0.0",
        constructs=constructs or _constructs(registry),
        rubrics=registry,
        response_type=ResponseType.TEXT_RESPONSE,
        engine_policy=engine_policy or policies[0],
        calibration_policy=calibration_policy or policies[1],
        validation_policy=validation_policy or policies[2],
        adjudication_policy=adjudication_policy or policies[3],
        monitoring_policy=monitoring_policy or policies[4],
        metadata={} if metadata is None else metadata,
    )


def test_assessment_contract_is_deterministic_content_addressed_and_exported():
    """Equivalent registries produce one canonical public artifact."""
    rubrics = _rubrics()
    first = _assessment(rubrics=rubrics)
    second = _assessment(
        rubrics=tuple(reversed(rubrics)),
        constructs=tuple(reversed(_constructs(rubrics))),
    )

    assert first == second
    assert first.rubric_fingerprints == tuple(sorted(first.rubric_fingerprints))
    assert tuple(item.construct_id for item in first.constructs) == (
        "evidence_quality_construct",
        "reasoning_quality_construct",
    )
    assert len(first.artifact_digest()) == 64
    assert first.assessment_handle.startswith("assessment_spec_")
    assert first.assessment_handle.endswith(first.artifact_digest()[:32])
    assert first.canonical_json() == second.canonical_json()
    assert json.loads(first.canonical_json()) == first.to_dict()

    assert scoring.AssessmentSpec is AssessmentSpec
    assert scoring.build_assessment_spec is build_assessment_spec
    assert set(scoring.__all__) >= {
        "AssessmentSpec",
        "ConstructSpec",
        "EnginePolicy",
        "build_assessment_spec",
    }


def test_metadata_is_deeply_immutable_bounded_and_copied():
    """Caller mutation cannot change assessment identity after construction."""
    metadata = {
        "deployment": {
            "regions": ["primary_region", "recovery_region"],
            "high_stakes": False,
        },
        "revision_count": 2,
        "tolerance": 0.125,
        "optional_note": None,
    }
    assessment = _assessment(metadata=metadata)
    original_digest = assessment.artifact_digest()

    metadata["deployment"]["regions"].append("late_region")
    metadata["revision_count"] = 99

    assert isinstance(assessment.metadata, MappingProxyType)
    assert assessment.to_dict()["metadata"]["deployment"]["regions"] == [
        "primary_region",
        "recovery_region",
    ]
    assert assessment.artifact_digest() == original_digest
    with pytest.raises(TypeError):
        assessment.metadata["new_key"] = "forbidden"  # type: ignore[index]


@pytest.mark.parametrize(
    ("metadata", "code"),
    [
        ({"bad_float": float("nan")}, "non_finite_json_number"),
        ({"bad_float": float("inf")}, "non_finite_json_number"),
        ({1: "bad_key"}, "invalid_json_key"),
        ({"oversized_integer": 1 << 63}, "json_integer_out_of_range"),
        ({"binary_payload": b"secret"}, "unsupported_json_value"),
    ],
)
def test_metadata_rejects_noncanonical_or_unsafe_values(metadata, code):
    """Non-canonical JSON cannot enter an assessment digest."""
    with pytest.raises(InvalidAssessmentSpecError) as error:
        _assessment(metadata=metadata)
    assert error.value.code == code
    assert "secret" not in str(error.value)


def test_metadata_depth_width_and_node_budgets_fail_closed():
    """Attacker-controlled metadata cannot cause unbounded traversal."""
    deep: object = "leaf"
    for _ in range(10):
        deep = {"nested_value": deep}
    with pytest.raises(InvalidAssessmentSpecError) as depth_error:
        _assessment(metadata={"root_value": deep})
    assert depth_error.value.code == "json_depth_exceeded"

    with pytest.raises(InvalidAssessmentSpecError) as width_error:
        _assessment(metadata={f"field_{index}": index for index in range(65)})
    assert width_error.value.code == "json_collection_too_large"

    with pytest.raises(InvalidAssessmentSpecError) as nodes_error:
        _assessment(
            metadata={
                f"field_{index}": list(range(64))
                for index in range(20)
            }
        )
    assert nodes_error.value.code == "json_node_budget_exceeded"


def test_factory_rejects_unknown_construct_and_incomplete_registry_coverage():
    """Every exact rubric must map once to its declared construct."""
    rubrics = _rubrics()
    unknown = (
        ConstructSpec(
            construct_id="unknown_quality_construct",
            rubric_fingerprints=(rubrics[0].fingerprint,),
        ),
        ConstructSpec(
            construct_id="reasoning_quality_construct",
            rubric_fingerprints=(rubrics[1].fingerprint,),
        ),
    )
    with pytest.raises(InvalidAssessmentSpecError) as construct_error:
        _assessment(rubrics=rubrics, constructs=unknown)
    assert construct_error.value.code == "rubric_construct_mismatch"

    incomplete = (
        ConstructSpec(
            construct_id="evidence_quality_construct",
            rubric_fingerprints=(rubrics[0].fingerprint,),
        ),
    )
    with pytest.raises(InvalidAssessmentSpecError) as coverage_error:
        _assessment(rubrics=rubrics, constructs=incomplete)
    assert coverage_error.value.code == "rubric_registry_not_covered"

    duplicate = (
        ConstructSpec(
            construct_id="evidence_quality_construct",
            rubric_fingerprints=(rubrics[0].fingerprint,),
        ),
        ConstructSpec(
            construct_id="reasoning_quality_construct",
            rubric_fingerprints=(
                rubrics[0].fingerprint,
                rubrics[1].fingerprint,
            ),
        ),
    )
    with pytest.raises(InvalidAssessmentSpecError) as duplicate_error:
        _assessment(rubrics=rubrics, constructs=duplicate)
    assert duplicate_error.value.code == "rubric_assigned_multiple_times"


def test_factory_rejects_duplicate_rubrics_and_non_rubric_registry_values():
    """The exact rubric registry is type-safe and fingerprint-unique."""
    rubric = _rubrics()[0]
    policies = _policies((rubric, rubric))
    with pytest.raises(InvalidAssessmentSpecError) as duplicate_error:
        build_assessment_spec(
            assessment_id="duplicate_rubric_assessment",
            assessment_version="1.0.0",
            constructs=(
                ConstructSpec(
                    construct_id=rubric.construct_id,
                    rubric_fingerprints=(rubric.fingerprint,),
                ),
            ),
            rubrics=(rubric, rubric),
            response_type=ResponseType.TEXT_RESPONSE,
            engine_policy=policies[0],
            calibration_policy=policies[1],
            validation_policy=policies[2],
            adjudication_policy=policies[3],
            monitoring_policy=policies[4],
        )
    assert duplicate_error.value.code == "duplicate_rubric_fingerprint"

    with pytest.raises(InvalidAssessmentSpecError) as type_error:
        build_assessment_spec(
            assessment_id="invalid_registry_assessment",
            assessment_version="1.0.0",
            constructs=(),
            rubrics=(object(),),  # type: ignore[arg-type]
            response_type=ResponseType.TEXT_RESPONSE,
            engine_policy=EnginePolicy(),
            calibration_policy=CalibrationPolicy(
                model_id="many_facet_baseline",
                rubric_fingerprints=("0" * 64,),
            ),
            validation_policy=ValidationPolicy(
                metric_gates=(
                    MetricGate(
                        metric_id="evidence_gate_metric",
                        direction=MetricDirection.AT_LEAST,
                        threshold=0.0,
                    ),
                )
            ),
            adjudication_policy=AdjudicationPolicy(
                rules=(AdjudicationRule(rule_id="manual_review_rule"),)
            ),
            monitoring_policy=MonitoringPolicy(
                rules=(
                    MonitoringRule(
                        rule_id="distribution_drift_rule",
                        direction=MetricDirection.AT_MOST,
                        threshold=0.1,
                        window_size=10,
                    ),
                )
            ),
        )
    assert type_error.value.code == "invalid_rubric_registry_value"


def test_cross_policy_references_fail_closed():
    """Engines, groups, and rubric fingerprints must be declared centrally."""
    rubrics = _rubrics()
    fingerprints = tuple(rubric.fingerprint for rubric in rubrics)
    policies = _policies(rubrics)

    bad_calibration = CalibrationPolicy(
        model_id="many_facet_baseline",
        rubric_fingerprints=("f" * 64,),
    )
    with pytest.raises(InvalidAssessmentSpecError) as calibration_error:
        _assessment(calibration_policy=bad_calibration)
    assert calibration_error.value.code == "unknown_policy_rubric"

    bad_validation = ValidationPolicy(
        declared_group_ids=("reference_group",),
        metric_gates=(
            MetricGate(
                metric_id="subgroup_agreement_gate",
                direction=MetricDirection.AT_LEAST,
                threshold=0.80,
                group_ids=("focal_group",),
            ),
        ),
    )
    with pytest.raises(InvalidAssessmentSpecError) as group_error:
        _assessment(validation_policy=bad_validation)
    assert group_error.value.code == "unknown_policy_group"

    bad_adjudication = AdjudicationPolicy(
        rules=(
            AdjudicationRule(
                rule_id="unknown_engine_rule",
                engine_ids=("undeclared_engine",),
                rubric_fingerprints=fingerprints,
            ),
        )
    )
    with pytest.raises(InvalidAssessmentSpecError) as engine_error:
        _assessment(adjudication_policy=bad_adjudication)
    assert engine_error.value.code == "unknown_policy_engine"

    bad_monitoring = MonitoringPolicy(
        rules=(
            MonitoringRule(
                rule_id="unknown_rubric_rule",
                direction=MetricDirection.AT_MOST,
                threshold=0.25,
                window_size=20,
                rubric_fingerprints=("e" * 64,),
            ),
        )
    )
    with pytest.raises(InvalidAssessmentSpecError) as monitor_error:
        _assessment(monitoring_policy=bad_monitoring)
    assert monitor_error.value.code == "unknown_policy_rubric"

    assert policies[0].required_engine_ids == ("human_anchor_engine",)


def test_policy_local_invariants_reject_ambiguous_configuration():
    """Each policy validates its own bounded, typed contract before composition."""
    with pytest.raises(InvalidAssessmentSpecError) as required_engine_error:
        EnginePolicy(
            engine_ids=("declared_engine",),
            required_engine_ids=("missing_engine",),
        )
    assert required_engine_error.value.code == "unknown_required_engine"

    with pytest.raises(InvalidAssessmentSpecError) as boolean_integer_error:
        CalibrationPolicy(
            model_id="many_facet_baseline",
            rubric_fingerprints=("0" * 64,),
            minimum_observations_per_item=True,  # type: ignore[arg-type]
        )
    assert boolean_integer_error.value.code == "invalid_integer"

    with pytest.raises(InvalidAssessmentSpecError) as non_finite_error:
        MetricGate(
            metric_id="invalid_threshold_gate",
            direction=MetricDirection.AT_LEAST,
            threshold=float("inf"),
        )
    assert non_finite_error.value.code == "invalid_finite_number"

    gate = MetricGate(
        metric_id="duplicate_metric_gate",
        direction=MetricDirection.AT_LEAST,
        threshold=0.5,
    )
    with pytest.raises(InvalidAssessmentSpecError) as duplicate_gate_error:
        ValidationPolicy(metric_gates=(gate, gate))
    assert duplicate_gate_error.value.code == "duplicate_metric_id"

    rule = AdjudicationRule(rule_id="duplicate_review_rule")
    with pytest.raises(InvalidAssessmentSpecError) as duplicate_rule_error:
        AdjudicationPolicy(rules=(rule, rule))
    assert duplicate_rule_error.value.code == "duplicate_rule_id"

    with pytest.raises(InvalidAssessmentSpecError) as window_error:
        MonitoringRule(
            rule_id="invalid_window_rule",
            direction=MetricDirection.AT_MOST,
            threshold=0.1,
            window_size=0,
        )
    assert window_error.value.code == "integer_out_of_range"


def test_assessment_is_factory_sealed_and_versioned():
    """Direct construction cannot bypass exact-rubric registry validation."""
    rubrics = _rubrics()
    policies = _policies(rubrics)
    fingerprints = tuple(rubric.fingerprint for rubric in rubrics)

    with pytest.raises(InvalidAssessmentSpecError) as direct_error:
        AssessmentSpec(
            assessment_id="direct_assessment_spec",
            assessment_version="1.0.0",
            constructs=_constructs(rubrics),
            rubric_fingerprints=fingerprints,
            response_type=ResponseType.TEXT_RESPONSE,
            engine_policy=policies[0],
            calibration_policy=policies[1],
            validation_policy=policies[2],
            adjudication_policy=policies[3],
            monitoring_policy=policies[4],
        )
    assert direct_error.value.code == "factory_required"

    with pytest.raises(InvalidAssessmentSpecError) as version_error:
        build_assessment_spec(
            assessment_id="invalid_version_assessment",
            assessment_version="01.0.0",
            constructs=_constructs(rubrics),
            rubrics=rubrics,
            response_type=ResponseType.TEXT_RESPONSE,
            engine_policy=policies[0],
            calibration_policy=policies[1],
            validation_policy=policies[2],
            adjudication_policy=policies[3],
            monitoring_policy=policies[4],
        )
    assert version_error.value.code == "invalid_semantic_version"

    with pytest.raises(InvalidAssessmentSpecError) as identifier_error:
        build_assessment_spec(
            assessment_id="singleword",
            assessment_version="1.0.0",
            constructs=_constructs(rubrics),
            rubrics=rubrics,
            response_type=ResponseType.TEXT_RESPONSE,
            engine_policy=policies[0],
            calibration_policy=policies[1],
            validation_policy=policies[2],
            adjudication_policy=policies[3],
            monitoring_policy=policies[4],
        )
    assert identifier_error.value.code == "invalid_identifier"


def test_wrong_component_types_are_rejected_without_attribute_errors():
    """Public composition errors remain stable domain errors."""
    rubrics = _rubrics()
    policies = _policies(rubrics)
    with pytest.raises(InvalidAssessmentSpecError) as construct_error:
        build_assessment_spec(
            assessment_id="invalid_component_assessment",
            assessment_version="1.0.0",
            constructs=(object(),),  # type: ignore[arg-type]
            rubrics=rubrics,
            response_type=ResponseType.TEXT_RESPONSE,
            engine_policy=policies[0],
            calibration_policy=policies[1],
            validation_policy=policies[2],
            adjudication_policy=policies[3],
            monitoring_policy=policies[4],
        )
    assert construct_error.value.code == "invalid_construct_value"

    with pytest.raises(InvalidAssessmentSpecError) as policy_error:
        build_assessment_spec(
            assessment_id="invalid_policy_assessment",
            assessment_version="1.0.0",
            constructs=_constructs(rubrics),
            rubrics=rubrics,
            response_type=ResponseType.TEXT_RESPONSE,
            engine_policy=object(),  # type: ignore[arg-type]
            calibration_policy=policies[1],
            validation_policy=policies[2],
            adjudication_policy=policies[3],
            monitoring_policy=policies[4],
        )
    assert policy_error.value.code == "invalid_policy_value"
