"""Behavioral contracts for automated-scoring assessment specifications."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from types import MappingProxyType

import pytest

from fast_mlsirm.rubric import ResponseFormat, RubricLevel, RubricSpecification
from fast_mlsirm.scoring import (
    SCORING_SCHEMA_VERSION,
    AdjudicationPolicy,
    AssessmentSpec,
    AutomatedScoringError,
    CalibrationModel,
    CalibrationPolicy,
    ConstructSpec,
    EnginePolicy,
    GateComparison,
    InvalidAssessmentSpecError,
    MonitoringPolicy,
    ValidationGate,
    ValidationPolicy,
    artifact_digest,
    build_assessment_spec,
    canonical_json,
)
from fast_mlsirm.scoring import __all__ as scoring_exports


def _levels() -> tuple[RubricLevel, ...]:
    """Return a compact three-level rubric scale."""
    return (
        RubricLevel(0, "unsupported", "No support.", ("no evidence",)),
        RubricLevel(1, "partial_support", "Partial support.", ("mixed evidence",)),
        RubricLevel(2, "full_support", "Full support.", ("complete evidence",)),
    )


def _rubric(
    *,
    rubric_id: str = "faithfulness_rubric",
    construct_id: str = "evidence_grounding",
    response_format: ResponseFormat = ResponseFormat.ORDINAL_RATING,
) -> RubricSpecification:
    """Return one exact rubric source-of-truth fixture."""
    return RubricSpecification(
        rubric_id=rubric_id,
        construct_id=construct_id,
        construct_definition="Degree to which claims are supported.",
        response_format=response_format,
        levels=_levels(),
        task_families=("claim_verification",),
        evidence_requirements=("Quote the supporting evidence.",),
        locale="en-US",
    )


def _construct(
    construct_id: str = "evidence_grounding",
) -> ConstructSpec:
    """Return one valid assessment construct."""
    return ConstructSpec(
        construct_id=construct_id,
        construct_definition="Evidence-conditioned response quality.",
        reporting_label="Evidence grounding",
    )


def _engine_policy(
    allowed_engine_ids: tuple[str, ...] = ("fixture_engine",),
) -> EnginePolicy:
    """Return one valid engine policy."""
    return EnginePolicy(
        allowed_engine_ids=allowed_engine_ids,
        require_evidence=True,
        maximum_attempts=2,
    )


def _calibration_policy() -> CalibrationPolicy:
    """Return one valid many-facet calibration policy."""
    return CalibrationPolicy(
        model=CalibrationModel.FACETS,
        minimum_raters=2,
        require_connected_design=True,
        allow_missing_observations=True,
    )


def _validation_policy(
    *,
    gate_group: str | None = "all_respondents",
    required_group_ids: tuple[str, ...] = ("all_respondents",),
) -> ValidationPolicy:
    """Return one ordered validation policy."""
    return ValidationPolicy(
        gates=(
            ValidationGate(
                metric_id="quadratic_weighted_kappa",
                comparison=GateComparison.MINIMUM,
                threshold=0.8,
                minimum_observations=10,
                group_id=gate_group,
            ),
            ValidationGate(
                metric_id="engine_failure_rate",
                comparison=GateComparison.MAXIMUM,
                threshold=0.05,
                minimum_observations=10,
            ),
        ),
        required_group_ids=required_group_ids,
    )


def _adjudication_policy() -> AdjudicationPolicy:
    """Return one valid human-review policy."""
    return AdjudicationPolicy(
        trigger_codes=("engine_abstention", "score_disagreement"),
        maximum_score_distance=1.0,
        maximum_uncertainty=0.5,
        require_evidence=True,
    )


def _monitoring_policy(
    monitored_group_ids: tuple[str, ...] = ("all_respondents",),
) -> MonitoringPolicy:
    """Return one valid drift-monitoring policy."""
    return MonitoringPolicy(
        window_size=100,
        minimum_observations=20,
        monitored_group_ids=monitored_group_ids,
        alert_on_rubric_change=True,
        alert_on_engine_change=True,
    )


def _build(
    *,
    rubric: RubricSpecification | None = None,
    constructs: tuple[ConstructSpec, ...] | None = None,
    rubric_fingerprints: tuple[str, ...] | None = None,
    response_format: ResponseFormat | str = ResponseFormat.ORDINAL_RATING,
    declared_engine_ids: tuple[str, ...] = ("fixture_engine",),
    declared_group_ids: tuple[str, ...] = ("all_respondents",),
    engine_policy: object | None = None,
    calibration_policy: object | None = None,
    validation_policy: object | None = None,
    adjudication_policy: object | None = None,
    monitoring_policy: object | None = None,
    rubrics: object | None = None,
    metadata: object | None = None,
    assessment_id: str = "essay_assessment",
    assessment_version: str = "1.0.0",
    schema_version: str = SCORING_SCHEMA_VERSION,
) -> AssessmentSpec:
    """Build one valid assessment unless a test supplies a mutated field."""
    selected = _rubric() if rubric is None else rubric
    return build_assessment_spec(
        assessment_id=assessment_id,
        assessment_version=assessment_version,
        constructs=(_construct(),) if constructs is None else constructs,
        rubric_fingerprints=(selected.fingerprint,)
        if rubric_fingerprints is None
        else rubric_fingerprints,
        response_format=response_format,
        declared_engine_ids=declared_engine_ids,
        declared_group_ids=declared_group_ids,
        engine_policy=_engine_policy() if engine_policy is None else engine_policy,
        calibration_policy=_calibration_policy()
        if calibration_policy is None
        else calibration_policy,
        validation_policy=_validation_policy()
        if validation_policy is None
        else validation_policy,
        adjudication_policy=_adjudication_policy()
        if adjudication_policy is None
        else adjudication_policy,
        monitoring_policy=_monitoring_policy()
        if monitoring_policy is None
        else monitoring_policy,
        rubrics=(selected,) if rubrics is None else rubrics,
        schema_version=schema_version,
        metadata={
            "owner_team": "psychometrics",
            "execution_flags": [True, None, 3, 1.25],
            "nested_value": {"run_count": 2},
        }
        if metadata is None
        else metadata,
    )


def _direct_kwargs() -> dict[str, object]:
    """Return all public fields required by direct construction."""
    rubric = _rubric()
    return {
        "assessment_id": "essay_assessment",
        "assessment_version": "1.0.0",
        "constructs": (_construct(),),
        "rubric_fingerprints": (rubric.fingerprint,),
        "response_format": ResponseFormat.ORDINAL_RATING,
        "declared_engine_ids": ("fixture_engine",),
        "declared_group_ids": ("all_respondents",),
        "engine_policy": _engine_policy(),
        "calibration_policy": _calibration_policy(),
        "validation_policy": _validation_policy(),
        "adjudication_policy": _adjudication_policy(),
        "monitoring_policy": _monitoring_policy(),
        "metadata": {},
    }


def test_public_exports_are_explicit() -> None:
    """The scoring package exposes only the reviewed contract surface."""
    assert set(scoring_exports) == {
        "SCORING_SCHEMA_VERSION",
        "AdjudicationPolicy",
        "AssessmentSpec",
        "AutomatedScoringError",
        "CalibrationModel",
        "CalibrationPolicy",
        "ConstructSpec",
        "EnginePolicy",
        "GateComparison",
        "InvalidAssessmentSpecError",
        "MonitoringPolicy",
        "ValidationGate",
        "ValidationPolicy",
        "artifact_digest",
        "build_assessment_spec",
        "canonical_json",
    }


def test_build_assessment_spec_is_deterministic_and_deeply_immutable() -> None:
    """Registry-validated content is normalized, frozen, and content-addressed."""
    first = _build()
    second = _build(
        metadata={
            "nested_value": {"run_count": 2},
            "execution_flags": (True, None, 3, 1.25),
            "owner_team": "psychometrics",
        }
    )
    assert first == second
    assert first.assessment_id == "essay_assessment"
    assert first.assessment_version == "1.0.0"
    assert first.response_format is ResponseFormat.ORDINAL_RATING
    assert first.metadata == second.metadata
    assert isinstance(first.metadata, MappingProxyType)
    assert first.metadata["execution_flags"] == (True, None, 3, 1.25)
    assert isinstance(first.metadata["nested_value"], MappingProxyType)
    assert first.assessment_fingerprint == artifact_digest(first)
    assert first.assessment_handle == (
        f"assessment_spec_{first.assessment_fingerprint[:32]}"
    )
    assert len(first.assessment_fingerprint) == 64
    assert json.loads(canonical_json(first))["assessment_id"] == "essay_assessment"
    payload = first.to_dict()
    assert payload["assessment_fingerprint"] == first.assessment_fingerprint
    assert payload["assessment_handle"] == first.assessment_handle
    assert payload["metadata"]["nested_value"] == {"run_count": 2}
    with pytest.raises(TypeError):
        first.metadata["owner_team"] = "changed"
    with pytest.raises(FrozenInstanceError):
        first.assessment_id = "changed_assessment"


def test_canonical_json_and_digest_support_every_public_contract() -> None:
    """Canonical serialization covers all policy values and raw JSON content."""
    values = (
        _construct(),
        _engine_policy(),
        _calibration_policy(),
        _validation_policy(),
        _validation_policy().gates[0],
        _adjudication_policy(),
        _monitoring_policy(),
        CalibrationModel.FACETS,
        {"z_value": (1, 2), "a_value": True},
    )
    for value in values:
        encoded = canonical_json(value)
        assert json.loads(encoded) is not None
        assert len(artifact_digest(value)) == 64
    assert canonical_json({"z_value": 1, "a_value": 2}) == (
        '{"a_value":2,"z_value":1}'
    )


def test_automated_scoring_error_preserves_machine_fields() -> None:
    """Domain exceptions expose bounded codes and fields."""
    error = AutomatedScoringError(
        "stable_error_code",
        "Stable public message.",
        field="assessment_id",
    )
    assert error.code == "stable_error_code"
    assert error.field == "assessment_id"
    assert str(error) == "Stable public message."


def test_assessment_requires_the_registry_validating_factory() -> None:
    """Direct construction cannot bypass rubric and policy replay."""
    with pytest.raises(
        InvalidAssessmentSpecError,
        match="must be created by build_assessment_spec",
    ) as caught:
        AssessmentSpec(**_direct_kwargs())
    assert caught.value.code == "factory_required"
    assert caught.value.field == "assessment_spec"


@pytest.mark.parametrize(
    ("factory_kwargs", "code", "match"),
    [
        (
            {"assessment_id": "single"},
            "invalid_identifier",
            "two-or-more-token",
        ),
        (
            {"assessment_version": "01.0.0"},
            "invalid_semantic_version",
            "canonical semantic version",
        ),
        (
            {"response_format": "unknown_format"},
            "invalid_enum_value",
            "response_format must be one of",
        ),
        (
            {"rubric_fingerprints": ("not-a-digest",)},
            "invalid_fingerprint",
            "64-character",
        ),
        (
            {
                "rubric_fingerprints": (
                    _rubric().fingerprint,
                    _rubric().fingerprint,
                )
            },
            "duplicate_rubric_fingerprint",
            "must not contain duplicates",
        ),
        (
            {"constructs": (_construct(), _construct())},
            "duplicate_construct",
            "must not repeat",
        ),
    ],
)
def test_assessment_scalar_and_identity_contracts_fail_closed(
    factory_kwargs: dict[str, object],
    code: str,
    match: str,
) -> None:
    """Invalid assessment identities and duplicate declarations are rejected."""
    with pytest.raises(InvalidAssessmentSpecError, match=match) as caught:
        _build(**factory_kwargs)
    assert caught.value.code == code


@pytest.mark.parametrize(
    ("constructs", "match"),
    [
        ((), "constructs must contain at least 1"),
        (("not-a-construct",), r"constructs\[0\] must be a ConstructSpec"),
        (tuple(_construct(f"construct_{index}") for index in range(33)), "at most 32"),
        ("not-a-collection", "constructs must be a collection"),
        (7, "constructs must be a collection"),
    ],
)
def test_construct_collection_is_bounded_and_typed(
    constructs: object,
    match: str,
) -> None:
    """Construct inputs cannot trigger unbounded or ill-typed materialization."""
    with pytest.raises(InvalidAssessmentSpecError, match=match):
        _build(constructs=constructs)


@pytest.mark.parametrize(
    ("rubric_fingerprints", "match"),
    [
        ((), "rubric_fingerprints must contain at least 1"),
        (tuple("a" * 64 for _ in range(65)), "at most 64"),
        ("not-a-collection", "rubric_fingerprints must be a collection"),
    ],
)
def test_rubric_fingerprint_collection_is_bounded(
    rubric_fingerprints: object,
    match: str,
) -> None:
    """Assessment rubric identities are finite collections."""
    with pytest.raises(InvalidAssessmentSpecError, match=match):
        _build(rubric_fingerprints=rubric_fingerprints)


def test_unknown_rubric_fingerprint_is_rejected() -> None:
    """A well-formed digest absent from the exact registry cannot be replayed."""
    with pytest.raises(
        InvalidAssessmentSpecError,
        match="absent from rubrics",
    ) as caught:
        _build(rubric_fingerprints=("a" * 64,))
    assert caught.value.code == "unknown_rubric_fingerprint"


def test_rubric_construct_and_response_format_must_match() -> None:
    """Selected rubrics resolve to declared constructs and one response format."""
    unknown_construct = _rubric(construct_id="answer_relevance")
    with pytest.raises(
        InvalidAssessmentSpecError,
        match="undeclared construct",
    ) as caught:
        _build(rubric=unknown_construct)
    assert caught.value.code == "unknown_rubric_construct"

    wrong_format = _rubric(response_format=ResponseFormat.BINARY_JUDGMENT)
    with pytest.raises(
        InvalidAssessmentSpecError,
        match="must use response_format",
    ) as caught:
        _build(rubric=wrong_format)
    assert caught.value.code == "rubric_response_format_mismatch"


@pytest.mark.parametrize(
    ("rubrics", "match", "code"),
    [
        ((), "rubrics must contain at least 1", "collection_too_small"),
        ("not-a-collection", "rubrics must be a collection", "invalid_collection"),
        (7, "rubrics must be a collection", "invalid_collection"),
        (("not-a-rubric",), r"rubrics\[0\] must be", "invalid_rubric"),
        (
            (_rubric(), _rubric()),
            "duplicate fingerprints",
            "duplicate_rubric",
        ),
        (
            tuple(
                _rubric(rubric_id=f"rubric_{index}")
                for index in range(65)
            ),
            "at most 64",
            "collection_too_large",
        ),
    ],
)
def test_rubric_registry_is_bounded_typed_and_unique(
    rubrics: object,
    match: str,
    code: str,
) -> None:
    """The external rubric registry is replayed under a strict work budget."""
    with pytest.raises(InvalidAssessmentSpecError, match=match) as caught:
        _build(rubrics=rubrics)
    assert caught.value.code == code


def test_multiple_registry_entries_can_select_one_exact_version() -> None:
    """Extra registry versions do not alter the requested assessment identity."""
    first = _rubric()
    second = _rubric(rubric_id="relevance_rubric", construct_id="answer_relevance")
    built = _build(rubric=first, rubrics=(second, first))
    assert built.rubric_fingerprints == (first.fingerprint,)


def test_engine_and_group_policy_references_must_be_declared() -> None:
    """Policies cannot silently reference undeclared engines or subgroups."""
    with pytest.raises(
        InvalidAssessmentSpecError,
        match="undeclared engine",
    ) as caught:
        _build(engine_policy=_engine_policy(("other_engine",)))
    assert caught.value.code == "unknown_engine_reference"

    variants = (
        _validation_policy(required_group_ids=("other_group",)),
        _validation_policy(gate_group="other_group", required_group_ids=()),
        _validation_policy(gate_group=None, required_group_ids=()),
    )
    monitoring = (
        _monitoring_policy(),
        _monitoring_policy(),
        _monitoring_policy(("other_group",)),
    )
    for validation_policy, monitoring_policy in zip(variants, monitoring):
        if validation_policy is variants[2]:
            with pytest.raises(
                InvalidAssessmentSpecError,
                match="undeclared group",
            ):
                _build(
                    validation_policy=validation_policy,
                    monitoring_policy=monitoring_policy,
                )
        else:
            with pytest.raises(
                InvalidAssessmentSpecError,
                match="undeclared group",
            ) as caught:
                _build(validation_policy=validation_policy)
            assert caught.value.code == "unknown_group_reference"


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("engine_policy", object(), "EnginePolicy"),
        ("calibration_policy", object(), "CalibrationPolicy"),
        ("validation_policy", object(), "ValidationPolicy"),
        ("adjudication_policy", object(), "AdjudicationPolicy"),
        ("monitoring_policy", object(), "MonitoringPolicy"),
    ],
)
def test_assessment_rejects_wrong_policy_types(
    field: str,
    value: object,
    expected: str,
) -> None:
    """Every policy boundary requires its exact reviewed value type."""
    with pytest.raises(
        InvalidAssessmentSpecError,
        match=f"{field} must be a {expected}",
    ) as caught:
        _build(**{field: value})
    assert caught.value.code == "invalid_policy"


def test_schema_version_is_explicit_and_fixed() -> None:
    """Unsupported scoring contract schemas cannot enter an assessment."""
    with pytest.raises(
        InvalidAssessmentSpecError,
        match="schema_version must be '1.0'",
    ) as caught:
        _build(schema_version="2.0")
    assert caught.value.code == "unsupported_schema_version"
    assert SCORING_SCHEMA_VERSION == "1.0"


@pytest.mark.parametrize(
    ("factory", "args", "match", "code"),
    [
        (
            ConstructSpec,
            (7, "Definition.", "Label"),
            "construct_id must be a string",
            "invalid_text_type",
        ),
        (
            ConstructSpec,
            ("  ", "Definition.", "Label"),
            "construct_id must not be empty",
            "empty_text",
        ),
        (
            ConstructSpec,
            ("valid_construct", "", "Label"),
            "construct_definition must not be empty",
            "empty_text",
        ),
        (
            ConstructSpec,
            ("valid_construct", "x" * 4097, "Label"),
            "construct_definition must contain at most",
            "text_too_long",
        ),
        (
            ConstructSpec,
            ("valid_construct", "Definition.", "x" * 257),
            "reporting_label must contain at most 256",
            "text_too_long",
        ),
        (
            ConstructSpec,
            ("InvalidConstruct", "Definition.", "Label"),
            "two-or-more-token",
            "invalid_identifier",
        ),
    ],
)
def test_construct_text_and_identifier_validation(
    factory: object,
    args: tuple[object, ...],
    match: str,
    code: str,
) -> None:
    """Construct values are bounded and safely normalized."""
    with pytest.raises(InvalidAssessmentSpecError, match=match) as caught:
        factory(*args)
    assert caught.value.code == code


def test_construct_normalizes_surrounding_whitespace() -> None:
    """Construct values store canonical trimmed text."""
    construct = ConstructSpec(
        " evidence_grounding ",
        " Definition. ",
        " Evidence grounding ",
    )
    assert construct.to_dict() == {
        "construct_id": "evidence_grounding",
        "construct_definition": "Definition.",
        "reporting_label": "Evidence grounding",
    }


@pytest.mark.parametrize(
    ("value", "match", "code"),
    [
        ("fixture_engine", "allowed_engine_ids must be a collection", "invalid_collection"),
        (7, "allowed_engine_ids must be a collection", "invalid_collection"),
        (("fixture_engine", "fixture_engine"), "must not contain duplicates", "duplicate_identifier"),
        (tuple(f"engine_{index}" for index in range(65)), "at most 64", "collection_too_large"),
        (("invalid",), "two-or-more-token", "invalid_identifier"),
    ],
)
def test_engine_allowlist_is_bounded_unique_and_descriptive(
    value: object,
    match: str,
    code: str,
) -> None:
    """Engine identities cannot be unbounded, duplicated, or numeric-like."""
    with pytest.raises(InvalidAssessmentSpecError, match=match) as caught:
        EnginePolicy(value)
    assert caught.value.code == code


@pytest.mark.parametrize("value", [1, "true", None])
def test_policy_boolean_fields_require_booleans(value: object) -> None:
    """Boolean policy switches reject truthy and falsey substitutes."""
    constructors = (
        lambda: EnginePolicy(require_evidence=value),
        lambda: CalibrationPolicy(require_connected_design=value),
        lambda: CalibrationPolicy(allow_missing_observations=value),
        lambda: AdjudicationPolicy(("engine_abstention",), require_evidence=value),
        lambda: MonitoringPolicy(10, 2, alert_on_rubric_change=value),
        lambda: MonitoringPolicy(10, 2, alert_on_engine_change=value),
    )
    for constructor in constructors:
        with pytest.raises(
            InvalidAssessmentSpecError,
            match="must be a boolean",
        ) as caught:
            constructor()
        assert caught.value.code == "invalid_boolean"


@pytest.mark.parametrize(
    ("constructor", "match", "code"),
    [
        (
            lambda: EnginePolicy(maximum_attempts=True),
            "maximum_attempts must be an integer",
            "invalid_integer",
        ),
        (
            lambda: EnginePolicy(maximum_attempts=1.5),
            "maximum_attempts must be an integer",
            "invalid_integer",
        ),
        (
            lambda: EnginePolicy(maximum_attempts=0),
            "maximum_attempts must be between 1 and 16",
            "integer_out_of_range",
        ),
        (
            lambda: CalibrationPolicy(minimum_raters=0),
            "minimum_raters must be between 1 and 64",
            "integer_out_of_range",
        ),
        (
            lambda: ValidationGate(
                "metric_value", "minimum", 0.0, minimum_observations=1
            ),
            "minimum_observations must be between 2",
            "integer_out_of_range",
        ),
        (
            lambda: MonitoringPolicy(True, 2),
            "window_size must be an integer",
            "invalid_integer",
        ),
        (
            lambda: MonitoringPolicy(10, 11),
            "must not exceed window_size",
            "monitoring_sample_exceeds_window",
        ),
    ],
)
def test_policy_integer_contracts(
    constructor: object,
    match: str,
    code: str,
) -> None:
    """Integer policy controls are exact and bounded."""
    with pytest.raises(InvalidAssessmentSpecError, match=match) as caught:
        constructor()
    assert caught.value.code == code


@pytest.mark.parametrize(
    ("constructor", "match", "code"),
    [
        (
            lambda: CalibrationPolicy(model="unknown"),
            "model must be one of",
            "invalid_enum_value",
        ),
        (
            lambda: ValidationGate("metric_value", "unknown", 0.0),
            "comparison must be one of",
            "invalid_enum_value",
        ),
        (
            lambda: ValidationGate("metric_value", "minimum", True),
            "threshold must be a finite number",
            "invalid_number",
        ),
        (
            lambda: ValidationGate("metric_value", "minimum", object()),
            "threshold must be a finite number",
            "invalid_number",
        ),
        (
            lambda: ValidationGate("metric_value", "minimum", float("nan")),
            "threshold must be finite",
            "non_finite_number",
        ),
        (
            lambda: AdjudicationPolicy(
                ("engine_abstention",),
                maximum_score_distance=-0.1,
            ),
            "maximum_score_distance must be at least 0.0",
            "number_below_minimum",
        ),
        (
            lambda: AdjudicationPolicy(
                ("engine_abstention",),
                maximum_uncertainty=float("inf"),
            ),
            "maximum_uncertainty must be finite",
            "non_finite_number",
        ),
    ],
)
def test_policy_enum_and_numeric_contracts(
    constructor: object,
    match: str,
    code: str,
) -> None:
    """Enum and floating-point policies reject ambiguous or undefined values."""
    with pytest.raises(InvalidAssessmentSpecError, match=match) as caught:
        constructor()
    assert caught.value.code == code


def test_optional_adjudication_thresholds_and_empty_engine_allowlist_are_valid() -> None:
    """Human-only assessments may omit engines and optional review thresholds."""
    engine = EnginePolicy((), False, 1)
    adjudication = AdjudicationPolicy(("engine_failure",), None, None, False)
    assert engine.to_dict()["allowed_engine_ids"] == []
    assert adjudication.to_dict()["maximum_score_distance"] is None
    assert adjudication.to_dict()["maximum_uncertainty"] is None


@pytest.mark.parametrize(
    ("gates", "match", "code"),
    [
        ((), "gates must contain at least 1", "collection_too_small"),
        ("not-a-collection", "gates must be a collection", "invalid_collection"),
        ((object(),), r"gates\[0\] must be a ValidationGate", "invalid_validation_gate"),
        (
            tuple(
                ValidationGate(f"metric_{index}", "minimum", 0.0)
                for index in range(65)
            ),
            "at most 64",
            "collection_too_large",
        ),
    ],
)
def test_validation_gate_collection_is_bounded_and_typed(
    gates: object,
    match: str,
    code: str,
) -> None:
    """Validation policies cannot contain unbounded or opaque gate values."""
    with pytest.raises(InvalidAssessmentSpecError, match=match) as caught:
        ValidationPolicy(gates)
    assert caught.value.code == code


def test_validation_gate_identity_includes_optional_group() -> None:
    """Duplicate metric/group decisions fail while distinct groups remain valid."""
    first = ValidationGate("metric_value", "minimum", 0.5, group_id="group_alpha")
    duplicate = ValidationGate("metric_value", "maximum", 0.7, group_id="group_alpha")
    with pytest.raises(
        InvalidAssessmentSpecError,
        match="must not repeat",
    ) as caught:
        ValidationPolicy((first, duplicate))
    assert caught.value.code == "duplicate_validation_gate"
    distinct = ValidationGate("metric_value", "minimum", 0.5, group_id="group_beta")
    policy = ValidationPolicy((first, distinct))
    assert [gate["group_id"] for gate in policy.to_dict()["gates"]] == [
        "group_alpha",
        "group_beta",
    ]


@pytest.mark.parametrize(
    ("constructor", "match", "code"),
    [
        (
            lambda: AdjudicationPolicy(()),
            "trigger_codes must contain at least 1",
            "collection_too_small",
        ),
        (
            lambda: MonitoringPolicy(10, 2, monitored_group_ids=("single",)),
            "two-or-more-token",
            "invalid_identifier",
        ),
    ],
)
def test_adjudication_and_monitoring_identifiers_are_governed(
    constructor: object,
    match: str,
    code: str,
) -> None:
    """Review triggers and monitored groups use descriptive identifiers."""
    with pytest.raises(InvalidAssessmentSpecError, match=match) as caught:
        constructor()
    assert caught.value.code == code


@pytest.mark.parametrize(
    ("metadata", "match", "code"),
    [
        ([], "metadata must be a mapping", "invalid_metadata"),
        ({"bad-key": 1}, "keys must use lower snake_case", "invalid_metadata_key"),
        ({1: "value"}, "keys must use lower snake_case", "invalid_metadata_key"),
        ({"bad_value": object()}, "JSON-compatible", "unsupported_metadata_type"),
        ({"bad_value": float("nan")}, "must be finite", "non_finite_metadata"),
        ({"long_value": "x" * 4097}, "at most 4096", "metadata_text_too_long"),
        (
            {"many_values": tuple(range(65))},
            "at most 64 values",
            "collection_too_large",
        ),
        (
            {f"key_{index}": index for index in range(65)},
            "at most 64 entries",
            "metadata_mapping_too_large",
        ),
    ],
)
def test_metadata_rejects_unsafe_or_unbounded_content(
    metadata: object,
    match: str,
    code: str,
) -> None:
    """Metadata is deeply bounded and cannot carry non-JSON payloads."""
    with pytest.raises(InvalidAssessmentSpecError, match=match) as caught:
        _build(metadata=metadata)
    assert caught.value.code == code


def test_metadata_depth_is_bounded() -> None:
    """Nested metadata cannot exceed the documented depth budget."""
    value: object = "leaf"
    for _ in range(10):
        value = {"nested_value": value}
    with pytest.raises(
        InvalidAssessmentSpecError,
        match="maximum nesting depth",
    ) as caught:
        _build(metadata={"root_value": value})
    assert caught.value.code == "metadata_too_deep"


def test_metadata_none_uses_an_empty_mapping() -> None:
    """The public builder supplies an immutable empty mapping when omitted."""
    rubric = _rubric()
    spec = build_assessment_spec(
        assessment_id="essay_assessment",
        assessment_version="1.0.0",
        constructs=(_construct(),),
        rubric_fingerprints=(rubric.fingerprint,),
        response_format=ResponseFormat.ORDINAL_RATING,
        declared_engine_ids=("fixture_engine",),
        declared_group_ids=("all_respondents",),
        engine_policy=_engine_policy(),
        calibration_policy=_calibration_policy(),
        validation_policy=_validation_policy(),
        adjudication_policy=_adjudication_policy(),
        monitoring_policy=_monitoring_policy(),
        rubrics=(rubric,),
    )
    assert dict(spec.metadata) == {}


def test_canonical_json_rejects_unsupported_root_content() -> None:
    """Standalone serialization follows the same bounded metadata contract."""
    with pytest.raises(
        InvalidAssessmentSpecError,
        match="JSON-compatible",
    ):
        canonical_json({1, 2})
