"""Contracts for shared assessment and scoring-policy specifications."""

from __future__ import annotations

from dataclasses import replace
import math
from types import MappingProxyType

import pytest

import fast_mlsirm.scoring as scoring
from fast_mlsirm.rubric import ResponseFormat, RubricLevel, RubricSpecification
from fast_mlsirm.scoring import (
    MAX_METADATA_COLLECTION_VALUES,
    MAX_METADATA_DEPTH,
    MAX_METADATA_NODES,
    AdjudicationPolicy,
    AssessmentResponseType,
    AssessmentSpec,
    AssessmentSpecError,
    CalibrationPolicy,
    ConstructSpec,
    EnginePolicy,
    MonitoringPolicy,
    ReportingPolicy,
    ValidationPolicy,
    artifact_digest,
    build_assessment_spec,
    canonical_json,
)


def _rubric(
    rubric_id: str,
    construct_id: str,
    *,
    rubric_version: str = "1.0.0",
) -> RubricSpecification:
    """Return one deterministic ordinal rubric for assessment-contract tests."""
    return RubricSpecification(
        rubric_id=rubric_id,
        construct_id=construct_id,
        construct_definition=f"Definition for {construct_id}.",
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
        rubric_version=rubric_version,
    )


def _policies(
    construct_ids: tuple[str, ...] = ("argument_quality", "evidence_use"),
) -> tuple[
    EnginePolicy,
    CalibrationPolicy,
    ValidationPolicy,
    AdjudicationPolicy,
    MonitoringPolicy,
    ReportingPolicy,
]:
    """Return one complete deterministic scoring-policy family."""
    return (
        EnginePolicy(
            policy_id="engine_policy",
            engine_ids=("fixture_engine", "human_adapter"),
            allow_human_raters=True,
            allow_automated_raters=True,
            minimum_raters_per_response=2,
        ),
        CalibrationPolicy(
            policy_id="calibration_policy",
            model_id="facets_ordinal",
            construct_ids=construct_ids,
        ),
        ValidationPolicy(
            policy_id="validation_policy",
            metric_ids=("exact_agreement", "quadratic_weighted_kappa"),
            construct_ids=construct_ids,
        ),
        AdjudicationPolicy(
            policy_id="adjudication_policy",
            trigger_ids=("scorer_disagreement", "insufficient_evidence"),
            construct_ids=construct_ids,
        ),
        MonitoringPolicy(
            policy_id="monitoring_policy",
            metric_ids=("severity_drift", "failure_rate_drift"),
            construct_ids=construct_ids,
        ),
        ReportingPolicy(
            policy_id="reporting_policy",
            format_ids=("json_report", "html_report"),
            construct_ids=construct_ids,
            include_exact_values=True,
        ),
    )


def _assessment(
    *,
    constructs: tuple[ConstructSpec, ...] | None = None,
    rubrics: tuple[RubricSpecification, ...] | None = None,
    policies: tuple[
        EnginePolicy,
        CalibrationPolicy,
        ValidationPolicy,
        AdjudicationPolicy,
        MonitoringPolicy,
        ReportingPolicy,
    ]
    | None = None,
    metadata: object | None = None,
):
    """Build one complete assessment with two exact rubric bindings."""
    argument_rubric = _rubric("argument_rubric", "argument_quality")
    evidence_rubric = _rubric("evidence_rubric", "evidence_use")
    selected_rubrics = rubrics or (argument_rubric, evidence_rubric)
    selected_constructs = constructs or (
        ConstructSpec(
            construct_id="argument_quality",
            construct_definition="Quality of the response argument.",
            rubric_fingerprints=(argument_rubric.fingerprint,),
        ),
        ConstructSpec(
            construct_id="evidence_use",
            construct_definition="Quality of cited supporting evidence.",
            rubric_fingerprints=(evidence_rubric.fingerprint,),
        ),
    )
    selected_policies = policies or _policies()
    return build_assessment_spec(
        assessment_id="essay_assessment",
        assessment_version="1.0.0",
        constructs=selected_constructs,
        rubrics=selected_rubrics,
        response_type=AssessmentResponseType.CRITERION_LEVEL,
        engine_policy=selected_policies[0],
        calibration_policy=selected_policies[1],
        validation_policy=selected_policies[2],
        adjudication_policy=selected_policies[3],
        monitoring_policy=selected_policies[4],
        reporting_policy=selected_policies[5],
        metadata=(
            {
                "study_name": "Connected sparse pilot",
                "nested_metadata": {
                    "fold_count": 5,
                    "threshold_values": [0.1, 0.2],
                },
                "enabled_flag": True,
                "optional_value": None,
            }
            if metadata is None
            else metadata
        ),
    )


def test_assessment_spec_is_deterministic_content_addressed_and_deeply_immutable():
    """Input ordering cannot change identity and nested metadata cannot mutate."""
    first = _assessment()

    argument_rubric = _rubric("argument_rubric", "argument_quality")
    evidence_rubric = _rubric("evidence_rubric", "evidence_use")
    second = _assessment(
        constructs=tuple(reversed(first.constructs)),
        rubrics=(evidence_rubric, argument_rubric),
        policies=_policies(tuple(reversed(first.construct_ids))),
        metadata={
            "optional_value": None,
            "enabled_flag": True,
            "nested_metadata": {
                "threshold_values": (0.1, 0.2),
                "fold_count": 5,
            },
            "study_name": "Connected sparse pilot",
        },
    )

    assert first == second
    assert first.construct_ids == ("argument_quality", "evidence_use")
    assert first.rubric_fingerprints == tuple(sorted(first.rubric_fingerprints))
    assert len(first.assessment_fingerprint) == 64
    assert first.assessment_handle == (
        f"assessment_spec_{first.assessment_fingerprint[:32]}"
    )
    assert first.to_dict()["assessment_fingerprint"] == first.assessment_fingerprint
    assert canonical_json(first) == canonical_json(second)
    assert artifact_digest(first) == first.assessment_fingerprint

    assert isinstance(first.metadata, MappingProxyType)
    nested = first.metadata["nested_metadata"]
    assert isinstance(nested, MappingProxyType)
    assert nested["threshold_values"] == (0.1, 0.2)
    with pytest.raises(TypeError):
        first.metadata["study_name"] = "mutated"  # type: ignore[index]
    with pytest.raises(TypeError):
        nested["fold_count"] = 10  # type: ignore[index]


def test_canonical_json_and_digest_are_stable_and_reject_nonfinite_values():
    """Canonical serialization sorts mappings and never emits NaN or infinity."""
    left = {"beta_value": [2, 3], "alpha_value": {"finite_value": 1.5}}
    right = {
        "alpha_value": {"finite_value": 1.5},
        "beta_value": (2, 3),
    }
    assert canonical_json(left) == canonical_json(right)
    assert artifact_digest(left) == artifact_digest(right)
    assert len(artifact_digest(left)) == 64

    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            canonical_json({"invalid_value": value})
        with pytest.raises(ValueError, match="finite"):
            artifact_digest({"invalid_value": value})


def test_contract_types_normalize_identifiers_collections_and_enum_values():
    """Public policy contracts expose sorted unique immutable values."""
    construct = ConstructSpec(
        construct_id="argument_quality",
        construct_definition="Quality of the response argument.",
        rubric_fingerprints=("a" * 64, "b" * 64),
    )
    assert construct.rubric_fingerprints == ("a" * 64, "b" * 64)

    engine = EnginePolicy(
        policy_id="engine_policy",
        engine_ids=("human_adapter", "fixture_engine"),
        allow_human_raters=True,
        allow_automated_raters=True,
        minimum_raters_per_response=2,
    )
    assert engine.engine_ids == ("fixture_engine", "human_adapter")

    calibration = CalibrationPolicy(
        policy_id="calibration_policy",
        model_id="facets_ordinal",
        construct_ids=("evidence_use", "argument_quality"),
    )
    assert calibration.construct_ids == ("argument_quality", "evidence_use")

    validation = ValidationPolicy(
        policy_id="validation_policy",
        metric_ids=("quadratic_weighted_kappa", "exact_agreement"),
        construct_ids=("evidence_use", "argument_quality"),
    )
    assert validation.metric_ids == (
        "exact_agreement",
        "quadratic_weighted_kappa",
    )

    adjudication = AdjudicationPolicy(
        policy_id="adjudication_policy",
        trigger_ids=("scorer_disagreement", "insufficient_evidence"),
        construct_ids=("evidence_use", "argument_quality"),
    )
    assert adjudication.trigger_ids == (
        "insufficient_evidence",
        "scorer_disagreement",
    )

    monitoring = MonitoringPolicy(
        policy_id="monitoring_policy",
        metric_ids=("severity_drift", "failure_rate_drift"),
        construct_ids=("evidence_use", "argument_quality"),
    )
    assert monitoring.metric_ids == ("failure_rate_drift", "severity_drift")

    reporting = ReportingPolicy(
        policy_id="reporting_policy",
        format_ids=("json_report", "html_report"),
        construct_ids=("evidence_use", "argument_quality"),
        include_exact_values=1,
    )
    assert reporting.format_ids == ("html_report", "json_report")
    assert reporting.include_exact_values is True

    spec = _assessment()
    assert spec.response_type is AssessmentResponseType.CRITERION_LEVEL
    assert spec.to_dict()["response_type"] == "criterion_level"


def test_assessment_rejects_unknown_mismatched_duplicate_and_unused_rubrics():
    """Exact rubric provenance cannot be missing, repurposed, duplicated, or unused."""
    argument_rubric = _rubric("argument_rubric", "argument_quality")
    evidence_rubric = _rubric("evidence_rubric", "evidence_use")

    unknown_constructs = (
        ConstructSpec(
            construct_id="argument_quality",
            construct_definition="Quality of the response argument.",
            rubric_fingerprints=("f" * 64,),
        ),
    )
    with pytest.raises(AssessmentSpecError) as unknown_error:
        _assessment(constructs=unknown_constructs, rubrics=(argument_rubric,))
    assert unknown_error.value.code == "unknown_rubric_fingerprint"
    assert unknown_error.value.path == "$.constructs[0].rubric_fingerprints[0]"

    mismatched_constructs = (
        ConstructSpec(
            construct_id="evidence_use",
            construct_definition="Quality of cited supporting evidence.",
            rubric_fingerprints=(argument_rubric.fingerprint,),
        ),
    )
    with pytest.raises(AssessmentSpecError) as mismatch_error:
        _assessment(constructs=mismatched_constructs, rubrics=(argument_rubric,))
    assert mismatch_error.value.code == "rubric_construct_mismatch"

    with pytest.raises(AssessmentSpecError) as duplicate_error:
        _assessment(
            constructs=(
                ConstructSpec(
                    construct_id="argument_quality",
                    construct_definition="Quality of the response argument.",
                    rubric_fingerprints=(argument_rubric.fingerprint,),
                ),
            ),
            rubrics=(argument_rubric, argument_rubric),
        )
    assert duplicate_error.value.code == "duplicate_rubric_fingerprint"

    with pytest.raises(AssessmentSpecError) as unused_error:
        _assessment(
            constructs=(
                ConstructSpec(
                    construct_id="argument_quality",
                    construct_definition="Quality of the response argument.",
                    rubric_fingerprints=(argument_rubric.fingerprint,),
                ),
            ),
            rubrics=(argument_rubric, evidence_rubric),
            policies=_policies(("argument_quality",)),
        )
    assert unused_error.value.code == "unused_rubric_fingerprint"

    argument_revision = _rubric(
        "argument_rubric",
        "argument_quality",
        rubric_version="2.0.0",
    )
    with pytest.raises(AssessmentSpecError) as identifier_error:
        _assessment(
            constructs=(
                ConstructSpec(
                    construct_id="argument_quality",
                    construct_definition="Quality of the response argument.",
                    rubric_fingerprints=(
                        argument_rubric.fingerprint,
                        argument_revision.fingerprint,
                    ),
                ),
            ),
            rubrics=(argument_rubric, argument_revision),
            policies=_policies(("argument_quality",)),
        )
    assert identifier_error.value.code == "duplicate_rubric_id"


def test_assessment_rejects_duplicate_constructs_and_dangling_policy_references():
    """Construct and policy references must resolve inside one assessment graph."""
    argument_rubric = _rubric("argument_rubric", "argument_quality")
    construct = ConstructSpec(
        construct_id="argument_quality",
        construct_definition="Quality of the response argument.",
        rubric_fingerprints=(argument_rubric.fingerprint,),
    )
    with pytest.raises(AssessmentSpecError) as duplicate_error:
        _assessment(
            constructs=(construct, construct),
            rubrics=(argument_rubric,),
            policies=_policies(("argument_quality",)),
        )
    assert duplicate_error.value.code == "duplicate_construct_id"

    base = _policies(("argument_quality",))
    dangling_policies = (
        replace(base[1], construct_ids=("unknown_construct",)),
        replace(base[2], construct_ids=("unknown_construct",)),
        replace(base[3], construct_ids=("unknown_construct",)),
        replace(base[4], construct_ids=("unknown_construct",)),
        replace(base[5], construct_ids=("unknown_construct",)),
    )
    for index, dangling in enumerate(dangling_policies, start=1):
        policies = list(base)
        policies[index] = dangling
        with pytest.raises(AssessmentSpecError) as reference_error:
            _assessment(
                constructs=(construct,),
                rubrics=(argument_rubric,),
                policies=tuple(policies),  # type: ignore[arg-type]
            )
        assert reference_error.value.code == "unknown_policy_construct"
        assert reference_error.value.path.startswith("$.")


def test_engine_policy_requires_declared_rater_kinds_and_consistent_engines():
    """A policy cannot advertise an unusable or contradictory rater boundary."""
    with pytest.raises(ValueError, match="at least one rater kind"):
        EnginePolicy(
            policy_id="engine_policy",
            engine_ids=(),
            allow_human_raters=False,
            allow_automated_raters=False,
        )
    with pytest.raises(ValueError, match="at least one engine"):
        EnginePolicy(
            policy_id="engine_policy",
            engine_ids=(),
            allow_human_raters=True,
            allow_automated_raters=True,
        )
    with pytest.raises(ValueError, match="must be empty"):
        EnginePolicy(
            policy_id="engine_policy",
            engine_ids=("fixture_engine",),
            allow_human_raters=True,
            allow_automated_raters=False,
        )
    for value in (True, 0, 65, 1.5):
        with pytest.raises(ValueError, match="minimum_raters_per_response"):
            EnginePolicy(
                policy_id="engine_policy",
                engine_ids=(),
                allow_human_raters=True,
                allow_automated_raters=False,
                minimum_raters_per_response=value,  # type: ignore[arg-type]
            )


def test_public_contracts_reject_invalid_and_duplicate_identifiers():
    """Identifiers and bounded reference collections fail closed before storage."""
    constructors = (
        lambda: ConstructSpec(
            construct_id="invalid",
            construct_definition="Definition.",
            rubric_fingerprints=("a" * 64,),
        ),
        lambda: EnginePolicy(
            policy_id="invalid",
            engine_ids=(),
            allow_human_raters=True,
            allow_automated_raters=False,
        ),
        lambda: CalibrationPolicy(
            policy_id="invalid",
            model_id="facets_ordinal",
            construct_ids=("argument_quality",),
        ),
        lambda: ValidationPolicy(
            policy_id="invalid",
            metric_ids=("exact_agreement",),
            construct_ids=("argument_quality",),
        ),
        lambda: AdjudicationPolicy(
            policy_id="invalid",
            trigger_ids=("scorer_disagreement",),
            construct_ids=("argument_quality",),
        ),
        lambda: MonitoringPolicy(
            policy_id="invalid",
            metric_ids=("severity_drift",),
            construct_ids=("argument_quality",),
        ),
        lambda: ReportingPolicy(
            policy_id="invalid",
            format_ids=("json_report",),
            construct_ids=("argument_quality",),
        ),
    )
    for constructor in constructors:
        with pytest.raises(ValueError, match="two-or-more-token"):
            constructor()

    with pytest.raises(ValueError, match="duplicates"):
        ConstructSpec(
            construct_id="argument_quality",
            construct_definition="Definition.",
            rubric_fingerprints=("a" * 64, "a" * 64),
        )
    with pytest.raises(ValueError, match="64-character"):
        ConstructSpec(
            construct_id="argument_quality",
            construct_definition="Definition.",
            rubric_fingerprints=("not_a_digest",),
        )
    with pytest.raises(ValueError, match="duplicates"):
        ValidationPolicy(
            policy_id="validation_policy",
            metric_ids=("exact_agreement", "exact_agreement"),
            construct_ids=("argument_quality",),
        )
    with pytest.raises(ValueError, match="boolean"):
        ReportingPolicy(
            policy_id="reporting_policy",
            format_ids=("json_report",),
            construct_ids=("argument_quality",),
            include_exact_values="yes",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="response_type"):
        build_assessment_spec(
            assessment_id="essay_assessment",
            assessment_version="1.0.0",
            constructs=(),
            rubrics=(),
            response_type="not_supported",  # type: ignore[arg-type]
            engine_policy=_policies(())[0],
            calibration_policy=_policies(())[1],
            validation_policy=_policies(())[2],
            adjudication_policy=_policies(())[3],
            monitoring_policy=_policies(())[4],
            reporting_policy=_policies(())[5],
        )


def test_metadata_validation_is_bounded_safe_and_preserves_no_mutable_aliases():
    """Untrusted metadata cannot amplify work, carry unsafe keys, or retain aliases."""
    source = {
        "study_name": "Pilot",
        "nested_metadata": {"threshold_values": [0.1]},
    }
    spec = _assessment(metadata=source)
    source["study_name"] = "Mutated"
    source["nested_metadata"]["threshold_values"].append(0.2)  # type: ignore[index,union-attr]
    assert spec.metadata["study_name"] == "Pilot"
    assert spec.metadata["nested_metadata"]["threshold_values"] == (0.1,)

    unsafe_values = (
        {"invalid\nkey": "value"},
        {"unsupported_value": object()},
        {"oversized_values": list(range(MAX_METADATA_COLLECTION_VALUES + 1))},
        {f"metadata_key_{index}": index for index in range(MAX_METADATA_NODES + 1)},
    )
    for value in unsafe_values:
        with pytest.raises(ValueError):
            _assessment(metadata=value)

    nested: object = "leaf"
    for index in range(MAX_METADATA_DEPTH + 1):
        nested = {f"nested_level_{index}": nested}
    with pytest.raises(ValueError, match="depth"):
        _assessment(metadata=nested)


def test_assessment_is_factory_sealed_and_validates_component_types():
    """Only the builder may create a cross-reference-validated assessment artifact."""
    spec = _assessment()
    with pytest.raises(ValueError, match="build_assessment_spec"):
        AssessmentSpec(
            assessment_id=spec.assessment_id,
            assessment_version=spec.assessment_version,
            constructs=spec.constructs,
            rubric_fingerprints=spec.rubric_fingerprints,
            response_type=spec.response_type,
            engine_policy=spec.engine_policy,
            calibration_policy=spec.calibration_policy,
            validation_policy=spec.validation_policy,
            adjudication_policy=spec.adjudication_policy,
            monitoring_policy=spec.monitoring_policy,
            reporting_policy=spec.reporting_policy,
            metadata=spec.metadata,
        )

    argument_rubric = _rubric("argument_rubric", "argument_quality")
    valid_policies = _policies(("argument_quality",))
    invalid_values = (
        ("constructs", (object(),)),
        ("rubrics", (object(),)),
        ("engine_policy", object()),
        ("calibration_policy", object()),
        ("validation_policy", object()),
        ("adjudication_policy", object()),
        ("monitoring_policy", object()),
        ("reporting_policy", object()),
    )
    base_kwargs = {
        "assessment_id": "essay_assessment",
        "assessment_version": "1.0.0",
        "constructs": (
            ConstructSpec(
                construct_id="argument_quality",
                construct_definition="Definition.",
                rubric_fingerprints=(argument_rubric.fingerprint,),
            ),
        ),
        "rubrics": (argument_rubric,),
        "response_type": AssessmentResponseType.CRITERION_LEVEL,
        "engine_policy": valid_policies[0],
        "calibration_policy": valid_policies[1],
        "validation_policy": valid_policies[2],
        "adjudication_policy": valid_policies[3],
        "monitoring_policy": valid_policies[4],
        "reporting_policy": valid_policies[5],
    }
    for name, value in invalid_values:
        kwargs = dict(base_kwargs)
        kwargs[name] = value
        with pytest.raises((TypeError, ValueError)):
            build_assessment_spec(**kwargs)  # type: ignore[arg-type]


def test_scoring_contract_public_exports_are_explicit_and_complete():
    """The new namespace exposes only the documented provider-neutral contract surface."""
    expected = {
        "MAX_METADATA_COLLECTION_VALUES",
        "MAX_METADATA_DEPTH",
        "MAX_METADATA_NODES",
        "AdjudicationPolicy",
        "AssessmentResponseType",
        "AssessmentSpec",
        "AssessmentSpecError",
        "CalibrationPolicy",
        "ConstructSpec",
        "EnginePolicy",
        "MonitoringPolicy",
        "ReportingPolicy",
        "ValidationPolicy",
        "artifact_digest",
        "build_assessment_spec",
        "canonical_json",
    }
    assert set(scoring.__all__) == expected
    for name in expected:
        assert getattr(scoring, name) is globals()[name]
