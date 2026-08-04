"""Adversarial and fail-closed tests for scoring observations."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from fast_mlsirm.scoring import (
    AssessmentResponseType,
    AssessmentSpecError,
    EvidenceSpan,
    ObservationLevel,
    ObservationStatus,
    RaterKind,
    build_score_observation,
    validate_observations,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_observation_fixtures.py"))
)
_SHARED = runpy.run_path(str(Path(__file__).with_name("scoring_contract_fixtures.py")))
approved_assessment = _FIXTURES["approved_assessment"]
approved_rubrics = _FIXTURES["approved_rubrics"]
argument_rubric = _FIXTURES["argument_rubric"]
assessment_fixture = _SHARED["assessment"]
policies = _SHARED["policies"]


def _values(**overrides):
    values = {
        "assessment": approved_assessment(),
        "rubrics": approved_rubrics(),
        "observation_id": "argument_score_observation",
        "rubric_fingerprint": argument_rubric().fingerprint,
        "response_id": "essay_response_alpha",
        "rater_id": "automated_rater_alpha",
        "rater_kind": RaterKind.AUTOMATED,
        "engine_id": "fixture_engine",
        "construct_id": "argument_quality",
        "observation_level": ObservationLevel.CRITERION_LEVEL,
        "criterion_id": "evidence_alignment",
        "status": ObservationStatus.SCORED,
        "score_category": 1,
        "reason_code": None,
        "occasion_id": "scoring_occasion_alpha",
        "scorer_family": "deterministic_fixture",
        "scorer_version": "1.0.0",
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    ("start", "end", "code"),
    [
        (-1, 2, "invalid_start_offset"),
        (2, 2, "invalid_evidence_offsets"),
        (3, 2, "invalid_evidence_offsets"),
        (0, 1 << 63, "invalid_end_offset"),
        (True, 2, "invalid_start_offset"),
    ],
)
def test_evidence_offsets_are_bounded_and_half_open(start, end, code: str) -> None:
    """Evidence references reject invalid, Boolean, and oversized offsets."""
    with pytest.raises(AssessmentSpecError) as captured:
        EvidenceSpan(
            source_id="essay_response_alpha",
            start_offset=start,
            end_offset=end,
        )
    assert captured.value.code == code


def test_evidence_digest_and_identifiers_fail_closed() -> None:
    """Evidence references require descriptive identifiers and exact digests."""
    with pytest.raises(AssessmentSpecError) as source_error:
        EvidenceSpan(source_id="1", start_offset=0, end_offset=1)
    assert source_error.value.code == "invalid_source_id"

    with pytest.raises(AssessmentSpecError) as digest_error:
        EvidenceSpan(
            source_id="essay_response_alpha",
            start_offset=0,
            end_offset=1,
            content_digest="not_a_digest",
        )
    assert digest_error.value.code == "invalid_content_digest"


class HostileIndex:
    """Integer-like object that attempts to leak content from ``__index__``."""

    def __index__(self):
        """Raise a caller-controlled exception instead of an integer."""
        raise RuntimeError("secret essay response")


class HostileIterable:
    """Iterable that attempts to leak content during materialization."""

    def __iter__(self):
        """Yield once and then raise a caller-controlled exception."""
        yield EvidenceSpan(
            source_id="essay_response_alpha",
            start_offset=0,
            end_offset=1,
        )
        raise RuntimeError("raw response secret")


def test_hostile_numeric_and_iterable_callbacks_are_redacted() -> None:
    """Callback failures become stable non-reflective scoring errors."""
    with pytest.raises(AssessmentSpecError) as offset_error:
        EvidenceSpan(
            source_id="essay_response_alpha",
            start_offset=HostileIndex(),
            end_offset=2,
        )
    assert offset_error.value.code == "invalid_start_offset"
    assert "secret essay response" not in str(offset_error.value)

    with pytest.raises(AssessmentSpecError) as iterable_error:
        build_score_observation(**_values(evidence_spans=HostileIterable()))
    assert iterable_error.value.code == "invalid_evidence_spans"
    assert "raw response secret" not in str(iterable_error.value)


def test_evidence_span_budget_is_enforced_before_unbounded_materialization() -> None:
    """An overlong span stream fails at the declared contract limit."""
    spans = (
        EvidenceSpan(
            source_id="essay_response_alpha",
            start_offset=index * 2,
            end_offset=index * 2 + 1,
        )
        for index in range(65)
    )
    with pytest.raises(AssessmentSpecError) as captured:
        build_score_observation(**_values(evidence_spans=spans))
    assert captured.value.code == "invalid_evidence_spans"


@pytest.mark.parametrize("confidence", [True, float("nan"), float("inf"), -0.1, 1.1])
def test_confidence_is_optional_finite_and_bounded(confidence) -> None:
    """Scorer-reported confidence cannot be Boolean, non-finite, or unbounded."""
    with pytest.raises(AssessmentSpecError) as captured:
        build_score_observation(**_values(confidence=confidence))
    assert captured.value.code == "invalid_confidence"


def test_negative_zero_confidence_has_one_canonical_identity() -> None:
    """Floating negative zero normalizes to canonical positive zero."""
    negative = build_score_observation(**_values(confidence=-0.0))
    positive = build_score_observation(**_values(confidence=0.0))
    assert negative == positive
    assert negative.confidence == 0.0
    assert negative.observation_fingerprint == positive.observation_fingerprint


@pytest.mark.parametrize("metadata_key", ["Response_Text", "RAW_RESPONSE", "source_content"])
def test_sensitive_metadata_names_are_rejected_case_insensitively(
    metadata_key: str,
) -> None:
    """Metadata cannot smuggle response or source content under case variants."""
    with pytest.raises(AssessmentSpecError) as captured:
        build_score_observation(**_values(metadata={metadata_key: "secret"}))
    assert captured.value.code == "sensitive_metadata_field"
    assert metadata_key not in str(captured.value)


def test_invalid_utf8_and_huge_numeric_conversions_do_not_escape() -> None:
    """Invalid UTF-8 and resource-amplifying numerics use stable domain errors."""
    with pytest.raises(AssessmentSpecError) as utf8_error:
        build_score_observation(**_values(observation_id="invalid_\ud800_identifier"))
    assert utf8_error.value.code in {"invalid_observation_id", "invalid_utf8_text"}

    with pytest.raises(AssessmentSpecError) as score_error:
        build_score_observation(**_values(score_category=10**400))
    assert score_error.value.code == "unknown_score_category"


def test_exact_rubric_registry_and_construct_binding_are_required() -> None:
    """Stale, partial, and mismatched rubric registries cannot create observations."""
    with pytest.raises(AssessmentSpecError) as partial_registry:
        build_score_observation(**_values(rubrics=(argument_rubric(),)))
    assert partial_registry.value.code == "rubric_registry_mismatch"

    with pytest.raises(AssessmentSpecError) as mismatch:
        build_score_observation(**_values(construct_id="evidence_use"))
    assert mismatch.value.code == "rubric_construct_mismatch"


def test_assessment_response_type_restricts_observation_level() -> None:
    """A criterion-only assessment cannot accept a holistic observation."""
    assessment = assessment_fixture(response_type=AssessmentResponseType.CRITERION_LEVEL)
    with pytest.raises(AssessmentSpecError) as captured:
        build_score_observation(
            **_values(
                assessment=assessment,
                observation_level=ObservationLevel.HOLISTIC,
                criterion_id=None,
            )
        )
    assert captured.value.code == "unsupported_observation_level"


def test_disabled_rater_kinds_are_rejected() -> None:
    """Observation rater kinds must be enabled by the assessment engine policy."""
    base_policies = policies()
    human_only_policy = type(base_policies[0])(
        policy_id="human_engine_policy",
        engine_ids=(),
        allow_human_raters=True,
        allow_automated_raters=False,
        minimum_raters_per_response=1,
    )
    human_only = assessment_fixture(
        selected_policies=(human_only_policy, *base_policies[1:]),
        response_type=AssessmentResponseType.MIXED,
    )
    with pytest.raises(AssessmentSpecError) as automated_error:
        build_score_observation(**_values(assessment=human_only))
    assert automated_error.value.code == "disabled_automated_rater"

    automated_only_policy = type(base_policies[0])(
        policy_id="automated_engine_policy",
        engine_ids=("fixture_engine",),
        allow_human_raters=False,
        allow_automated_raters=True,
        minimum_raters_per_response=1,
    )
    automated_only = assessment_fixture(
        selected_policies=(automated_only_policy, *base_policies[1:]),
        response_type=AssessmentResponseType.MIXED,
    )
    with pytest.raises(AssessmentSpecError) as human_error:
        build_score_observation(
            **_values(
                assessment=automated_only,
                rater_kind=RaterKind.HUMAN,
                rater_id="human_rater_alpha",
                engine_id=None,
            )
        )
    assert human_error.value.code == "disabled_human_rater"


def test_batch_validation_rejects_invalid_entries_and_bounds() -> None:
    """Batch validation rejects non-observations and invalid resource bounds."""
    with pytest.raises(AssessmentSpecError) as entry_error:
        validate_observations(
            (object(),),
            assessment=approved_assessment(),
            rubrics=approved_rubrics(),
        )
    assert entry_error.value.code == "invalid_score_observation"

    with pytest.raises(AssessmentSpecError) as bounds_error:
        validate_observations(
            (),
            assessment=approved_assessment(),
            rubrics=approved_rubrics(),
            minimum=2,
            maximum=1,
        )
    assert bounds_error.value.code == "invalid_observation_bounds"

    with pytest.raises(AssessmentSpecError) as maximum_error:
        validate_observations(
            (),
            assessment=approved_assessment(),
            rubrics=approved_rubrics(),
            minimum=0,
            maximum=1_000_001,
        )
    assert maximum_error.value.code == "invalid_maximum"
