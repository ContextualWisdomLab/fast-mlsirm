"""RED contracts for shared scoring observations and engine provenance."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import runpy

import pytest

from fast_mlsirm.scoring import (
    AssessmentResponseType,
    AssessmentSpecError,
    EnginePolicy,
    EvidenceReference,
    ObservationState,
    RaterKind,
    ScoreObservation,
    ScoringEngineDescriptor,
    ScoringExecution,
    build_evidence_reference,
    build_score_observation,
    build_scoring_engine_descriptor,
    build_scoring_execution,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_contract_fixtures.py"))
)
assessment = _FIXTURES["assessment"]
policies = _FIXTURES["policies"]
rubric = _FIXTURES["rubric"]


def _argument_rubric(*, rubric_version: str = "1.0.0"):
    """Return the exact argument-quality rubric used by the assessment fixture."""
    return rubric(
        "argument_rubric",
        "argument_quality",
        rubric_version=rubric_version,
    )


def _engine(selected_assessment=None, *, kind=RaterKind.AUTOMATED_RATER):
    """Build one valid engine descriptor for a selected assessment."""
    resolved_assessment = assessment() if selected_assessment is None else selected_assessment
    if kind is RaterKind.HUMAN_RATER:
        return build_scoring_engine_descriptor(
            assessment=resolved_assessment,
            engine_id="human_adapter",
            engine_family="human_panel",
            engine_version="1.0.0",
            rater_kind=kind,
            prompt_template_version=None,
            configuration={"panel_name": "calibration_panel"},
        )
    return build_scoring_engine_descriptor(
        assessment=resolved_assessment,
        engine_id="fixture_engine",
        engine_family="fixture_provider",
        engine_version="1.0.0",
        rater_kind=kind,
        prompt_template_version="1.0.0",
        configuration={
            "temperature_value": -0.0,
            "nested_configuration": {"seed_value": 7},
        },
    )


def _evidence(reference_id: str = "evidence_reference") -> EvidenceReference:
    """Return one exact bounded source-span reference."""
    return build_evidence_reference(
        reference_id=reference_id,
        source_id="response_source",
        start_offset=0,
        end_offset=17,
        content_digest="a" * 64,
        metadata={"span_kind": "response_evidence"},
    )


def _observation(
    *,
    selected_assessment=None,
    selected_engine=None,
    selected_rubric=None,
    state=ObservationState.OBSERVED_SCORE,
    score_category=2,
    criterion_id="claim_support",
    evidence_references=None,
    reason_ids=(),
):
    """Build one valid criterion-level observation."""
    resolved_assessment = assessment() if selected_assessment is None else selected_assessment
    resolved_engine = (
        _engine(resolved_assessment) if selected_engine is None else selected_engine
    )
    resolved_rubric = (
        _argument_rubric() if selected_rubric is None else selected_rubric
    )
    resolved_evidence = (
        (_evidence(),) if evidence_references is None else evidence_references
    )
    return build_score_observation(
        assessment=resolved_assessment,
        rubric=resolved_rubric,
        engine=resolved_engine,
        observation_id="score_observation",
        respondent_id="essay_response",
        item_id="essay_prompt",
        rater_id="automated_rater",
        construct_id="argument_quality",
        criterion_id=criterion_id,
        state=state,
        score_category=score_category,
        evidence_references=resolved_evidence,
        reason_ids=reason_ids,
        uncertainty_metadata={"confidence_value": 0.75},
    )


def test_engine_descriptor_is_canonical_content_addressed_and_factory_sealed() -> None:
    """Equivalent engine configuration has one identity and rejects direct replay."""
    selected_assessment = assessment()
    first = build_scoring_engine_descriptor(
        assessment=selected_assessment,
        engine_id="fixture_engine",
        engine_family="fixture_provider",
        engine_version="1.0.0",
        rater_kind=RaterKind.AUTOMATED_RATER,
        prompt_template_version="1.0.0",
        configuration={"beta_value": 2, "alpha_value": -0.0},
    )
    second = build_scoring_engine_descriptor(
        assessment=selected_assessment,
        engine_id="fixture_engine",
        engine_family="fixture_provider",
        engine_version="1.0.0",
        rater_kind="automated_rater",
        prompt_template_version="1.0.0",
        configuration={"alpha_value": 0.0, "beta_value": 2},
    )

    assert first == second
    assert first.engine_fingerprint == second.engine_fingerprint
    assert len(first.engine_fingerprint) == 64
    assert first.engine_handle == f"scoring_engine_{first.engine_fingerprint[:32]}"
    with pytest.raises(AssessmentSpecError, match="build_scoring_engine_descriptor"):
        replace(first, engine_id="forged_engine")


def test_engine_descriptor_enforces_assessment_rater_policy() -> None:
    """Human, automated, disabled, and unknown engine paths fail closed."""
    base_policies = policies()
    human_only = assessment(
        selected_policies=(
            EnginePolicy(
                policy_id="engine_policy",
                engine_ids=("human_adapter",),
                allow_human_raters=True,
                allow_automated_raters=False,
                minimum_raters_per_response=1,
            ),
            *base_policies[1:],
        )
    )
    automated_only = assessment(
        selected_policies=(
            EnginePolicy(
                policy_id="engine_policy",
                engine_ids=("fixture_engine",),
                allow_human_raters=False,
                allow_automated_raters=True,
                minimum_raters_per_response=1,
            ),
            *base_policies[1:],
        )
    )

    assert _engine(human_only, kind=RaterKind.HUMAN_RATER).rater_kind is RaterKind.HUMAN_RATER
    assert _engine(automated_only).rater_kind is RaterKind.AUTOMATED_RATER

    with pytest.raises(AssessmentSpecError) as human_error:
        _engine(automated_only, kind=RaterKind.HUMAN_RATER)
    assert human_error.value.code == "human_rater_disabled"
    with pytest.raises(AssessmentSpecError) as automated_error:
        _engine(human_only)
    assert automated_error.value.code == "automated_rater_disabled"
    with pytest.raises(AssessmentSpecError) as unknown_error:
        build_scoring_engine_descriptor(
            assessment=assessment(),
            engine_id="unknown_engine",
            engine_family="fixture_provider",
            engine_version="1.0.0",
            rater_kind=RaterKind.AUTOMATED_RATER,
            prompt_template_version="1.0.0",
        )
    assert unknown_error.value.code == "unknown_scoring_engine"


def test_evidence_reference_is_reference_only_bounded_and_factory_sealed() -> None:
    """Evidence stores offsets and digests, never response or source content."""
    reference = _evidence()
    assert reference.to_dict()["start_offset"] == 0
    assert reference.to_dict()["end_offset"] == 17
    assert "content" not in reference.to_dict()
    assert len(reference.evidence_fingerprint) == 64
    assert reference.evidence_handle.startswith("evidence_reference_")
    with pytest.raises(AssessmentSpecError, match="build_evidence_reference"):
        replace(reference, start_offset=1)

    for start_offset, end_offset, code in (
        (-1, 2, "invalid_start_offset"),
        (3, 3, "invalid_evidence_range"),
        (4, 3, "invalid_evidence_range"),
    ):
        with pytest.raises(AssessmentSpecError) as error:
            build_evidence_reference(
                reference_id="evidence_reference",
                source_id="response_source",
                start_offset=start_offset,
                end_offset=end_offset,
                content_digest="a" * 64,
            )
        assert error.value.code == code

    with pytest.raises(AssessmentSpecError) as content_error:
        build_evidence_reference(
            reference_id="evidence_reference",
            source_id="response_source",
            start_offset=0,
            end_offset=1,
            content_digest="a" * 64,
            metadata={"Response_Text": "private_payload"},
        )
    assert content_error.value.code == "sensitive_metadata_field"
    assert "private_payload" not in str(content_error.value)


def test_observation_is_deterministic_factory_sealed_and_replay_bound() -> None:
    """Observation identity binds assessment, rubric, engine, evidence, and state."""
    first = _observation(
        evidence_references=(_evidence("evidence_beta"), _evidence("evidence_alpha")),
        reason_ids=("secondary_reason", "primary_reason"),
    )
    second = _observation(
        evidence_references=(_evidence("evidence_alpha"), _evidence("evidence_beta")),
        reason_ids=("primary_reason", "secondary_reason"),
    )

    assert first == second
    assert len(first.observation_fingerprint) == 64
    assert first.observation_handle == (
        f"score_observation_{first.observation_fingerprint[:32]}"
    )
    assert first.assessment_fingerprint == assessment().assessment_fingerprint
    assert first.rubric_fingerprint == _argument_rubric().fingerprint
    assert first.engine_fingerprint == _engine().engine_fingerprint
    with pytest.raises(AssessmentSpecError, match="build_score_observation"):
        replace(first, observation_id="forged_observation")


def test_observation_requires_exact_assessment_rubric_and_construct_bindings() -> None:
    """Changed rubrics, constructs, assessments, and engines cannot replay."""
    changed_rubric = _argument_rubric(rubric_version="2.0.0")
    with pytest.raises(AssessmentSpecError) as rubric_error:
        _observation(selected_rubric=changed_rubric)
    assert rubric_error.value.code == "unknown_observation_rubric"

    with pytest.raises(AssessmentSpecError) as construct_error:
        build_score_observation(
            assessment=assessment(),
            rubric=_argument_rubric(),
            engine=_engine(),
            observation_id="score_observation",
            respondent_id="essay_response",
            item_id="essay_prompt",
            rater_id="automated_rater",
            construct_id="evidence_use",
            criterion_id="claim_support",
            state=ObservationState.OBSERVED_SCORE,
            score_category=2,
            evidence_references=(_evidence(),),
        )
    assert construct_error.value.code == "observation_construct_mismatch"

    changed_assessment = assessment(metadata={"study_name": "changed_study"})
    with pytest.raises(AssessmentSpecError) as engine_error:
        _observation(selected_assessment=changed_assessment, selected_engine=_engine())
    assert engine_error.value.code == "engine_assessment_mismatch"


def test_response_type_controls_criterion_identifier_cardinality() -> None:
    """Criterion-level, holistic, and mixed assessments keep distinct semantics."""
    with pytest.raises(AssessmentSpecError) as missing_error:
        _observation(criterion_id=None)
    assert missing_error.value.code == "criterion_id_required"

    holistic = assessment(response_type=AssessmentResponseType.HOLISTIC)
    holistic_engine = _engine(holistic)
    with pytest.raises(AssessmentSpecError) as holistic_error:
        _observation(
            selected_assessment=holistic,
            selected_engine=holistic_engine,
            criterion_id="claim_support",
        )
    assert holistic_error.value.code == "criterion_id_not_allowed"

    mixed = assessment(response_type=AssessmentResponseType.MIXED)
    mixed_engine = _engine(mixed)
    assert _observation(
        selected_assessment=mixed,
        selected_engine=mixed_engine,
        criterion_id=None,
    ).criterion_id is None
    assert _observation(
        selected_assessment=mixed,
        selected_engine=mixed_engine,
        criterion_id="claim_support",
    ).criterion_id == "claim_support"


@pytest.mark.parametrize(
    ("state", "score_category", "evidence_references", "reason_ids", "code"),
    (
        (ObservationState.OBSERVED_SCORE, None, (_evidence(),), (), "observed_score_required"),
        (ObservationState.OBSERVED_SCORE, 2, (), (), "observed_evidence_required"),
        (ObservationState.ABSTAINED_SCORE, 1, (), ("insufficient_evidence",), "score_not_allowed"),
        (ObservationState.FAILED_SCORE, 1, (), ("engine_failure",), "score_not_allowed"),
        (ObservationState.EXCLUDED_SCORE, 1, (), ("policy_exclusion",), "score_not_allowed"),
        (ObservationState.NOT_APPLICABLE_SCORE, 1, (), ("criterion_not_applicable",), "score_not_allowed"),
        (ObservationState.ABSTAINED_SCORE, None, (), (), "observation_reason_required"),
        (ObservationState.FAILED_SCORE, None, (), (), "observation_reason_required"),
        (ObservationState.EXCLUDED_SCORE, None, (), (), "observation_reason_required"),
        (ObservationState.NOT_APPLICABLE_SCORE, None, (), (), "observation_reason_required"),
    ),
)
def test_observation_state_machine_fails_closed(
    state,
    score_category,
    evidence_references,
    reason_ids,
    code,
) -> None:
    """Observed and non-observed states cannot be silently interchanged."""
    with pytest.raises(AssessmentSpecError) as error:
        _observation(
            state=state,
            score_category=score_category,
            evidence_references=evidence_references,
            reason_ids=reason_ids,
        )
    assert error.value.code == code


@pytest.mark.parametrize("score_category", (True, -1, 3, 2.0, "2"))
def test_observed_score_must_be_one_exact_rubric_category(score_category) -> None:
    """No Boolean, numeric, string, clipping, or coercion shortcut is accepted."""
    with pytest.raises(AssessmentSpecError) as error:
        _observation(score_category=score_category)
    assert error.value.code == "invalid_score_category"


@pytest.mark.parametrize(
    ("state", "reason_id"),
    (
        (ObservationState.ABSTAINED_SCORE, "insufficient_evidence"),
        (ObservationState.FAILED_SCORE, "engine_failure"),
        (ObservationState.EXCLUDED_SCORE, "policy_exclusion"),
        (ObservationState.NOT_APPLICABLE_SCORE, "criterion_not_applicable"),
    ),
)
def test_non_observed_states_remain_explicit_non_numeric_records(state, reason_id) -> None:
    """Every non-observed state remains serializable without a surrogate score."""
    value = _observation(
        state=state,
        score_category=None,
        evidence_references=(),
        reason_ids=(reason_id,),
    )
    assert value.state is state
    assert value.score_category is None
    assert value.reason_ids == (reason_id,)


def test_scoring_execution_binds_exact_observation_engine_and_assessment() -> None:
    """Execution provenance cannot bless a changed or foreign observation."""
    selected_assessment = assessment()
    selected_engine = _engine(selected_assessment)
    selected_observation = _observation(
        selected_assessment=selected_assessment,
        selected_engine=selected_engine,
    )
    execution = build_scoring_execution(
        assessment=selected_assessment,
        engine=selected_engine,
        observation=selected_observation,
        execution_id="scoring_execution",
        execution_metadata={"attempt_count": 1},
    )

    assert len(execution.execution_fingerprint) == 64
    assert execution.execution_handle == (
        f"scoring_execution_{execution.execution_fingerprint[:32]}"
    )
    assert execution.observation_fingerprint == (
        selected_observation.observation_fingerprint
    )
    with pytest.raises(AssessmentSpecError, match="build_scoring_execution"):
        replace(execution, execution_id="forged_execution")

    changed_assessment = assessment(metadata={"study_name": "changed_study"})
    with pytest.raises(AssessmentSpecError) as assessment_error:
        build_scoring_execution(
            assessment=changed_assessment,
            engine=selected_engine,
            observation=selected_observation,
            execution_id="scoring_execution",
        )
    assert assessment_error.value.code == "execution_assessment_mismatch"

    human_engine = _engine(selected_assessment, kind=RaterKind.HUMAN_RATER)
    with pytest.raises(AssessmentSpecError) as engine_error:
        build_scoring_execution(
            assessment=selected_assessment,
            engine=human_engine,
            observation=selected_observation,
            execution_id="scoring_execution",
        )
    assert engine_error.value.code == "execution_engine_mismatch"


def test_public_contracts_are_exported_and_documented() -> None:
    """The shared MSA boundary has stable documented package exports."""
    for value in (
        EvidenceReference,
        ScoreObservation,
        ScoringEngineDescriptor,
        ScoringExecution,
        build_evidence_reference,
        build_score_observation,
        build_scoring_engine_descriptor,
        build_scoring_execution,
    ):
        assert value.__doc__
