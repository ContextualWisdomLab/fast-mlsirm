"""Contracts for content-addressed scoring-task revisions and calibration axes."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

import fast_mlsirm.scoring as scoring
from fast_mlsirm.scoring import (
    AssessmentSpecError,
    FixtureOutcome,
    ObservationStatus,
    StaticFixtureEngine,
    build_scoring_facets_calibration_bundle,
    build_scoring_facets_rating_records,
)

_EXECUTION_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)
_ESSAY_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_essay_contracts.py"))
)
assessment = _EXECUTION_FIXTURES["assessment"]
automated_engine = _EXECUTION_FIXTURES["automated_engine"]
criterion_request = _EXECUTION_FIXTURES["criterion_request"]
human_engine = _EXECUTION_FIXTURES["human_engine"]
rubric = _EXECUTION_FIXTURES["rubric"]
essay_evidence = _ESSAY_FIXTURES["essay_evidence"]
essay_request = _ESSAY_FIXTURES["essay_request"]
prompt = _ESSAY_FIXTURES["prompt"]
submission = _ESSAY_FIXTURES["submission"]


def assert_error(code: str, callback) -> AssessmentSpecError:
    """Assert one stable task-revision contract error and return it."""
    with pytest.raises(AssessmentSpecError) as caught:
        callback()
    assert caught.value.code == code
    return caught.value


def test_shared_request_keeps_logical_task_and_exact_revision_separate() -> None:
    """One request exposes both display identity and content revision identity."""
    revision = "d" * 64
    request = criterion_request(task_revision_fingerprint=revision)

    assert request.task_id == "sample_task"
    assert request.task_revision_fingerprint == revision
    assert request.to_dict()["task_revision_fingerprint"] == revision
    assert request.request_fingerprint != criterion_request(
        task_revision_fingerprint="e" * 64
    ).request_fingerprint


def test_shared_request_rejects_missing_or_malformed_task_revision() -> None:
    """The public builder never derives task equality from a logical identifier."""
    missing = assert_error(
        "invalid_task_revision_fingerprint",
        lambda: criterion_request(task_revision_fingerprint=None),
    )
    malformed = assert_error(
        "invalid_task_revision_fingerprint",
        lambda: criterion_request(task_revision_fingerprint="not_a_digest"),
    )

    assert missing.path == malformed.path == "$.task_revision_fingerprint"
    assert "not_a_digest" not in str(malformed)


def test_essay_adapter_uses_exact_prompt_fingerprint_as_task_revision() -> None:
    """Essay prompts project their complete revision, not only prompt content text."""
    prompt_value = prompt(prompt_content_fingerprint="7" * 64)
    request = essay_request(prompt=prompt_value)

    assert request.scoring_request.task_id == prompt_value.prompt_id
    assert (
        request.scoring_request.task_revision_fingerprint
        == prompt_value.prompt_fingerprint
    )


def test_changed_prompt_content_creates_distinct_calibration_item_identity() -> None:
    """The same logical prompt ID cannot silently pool changed prompt content."""
    first_prompt = prompt(prompt_content_fingerprint="1" * 64)
    second_prompt = prompt(prompt_content_fingerprint="2" * 64)
    first = essay_request(prompt=first_prompt)
    second = essay_request(prompt=second_prompt)

    assert first_prompt.prompt_id == second_prompt.prompt_id
    assert first.scoring_request.task_id == second.scoring_request.task_id
    assert (
        first.scoring_request.task_revision_fingerprint
        != second.scoring_request.task_revision_fingerprint
    )
    assert (
        first.scoring_request.request_fingerprint
        != second.scoring_request.request_fingerprint
    )


def _revision_records():
    """Return a fully linked two-revision, two-respondent, two-rater pilot."""
    records = []
    prompts = (
        prompt(prompt_content_fingerprint="3" * 64),
        prompt(prompt_content_fingerprint="4" * 64),
    )
    raters = (automated_engine(), human_engine())
    for respondent_index, respondent_id in enumerate(
        ("respondent_alpha", "respondent_beta")
    ):
        for prompt_index, prompt_value in enumerate(prompts):
            submission_value = submission(
                prompt_value,
                submission_id=(
                    f"submission_{respondent_index}_{prompt_index}"
                ),
                respondent_id=respondent_id,
                response_id=f"response_{respondent_index}_{prompt_index}",
            )
            evidence_value = essay_evidence(prompt_value, submission_value)
            request = essay_request(
                request_id=f"request_{respondent_index}_{prompt_index}",
                prompt=prompt_value,
                submission=submission_value,
                evidence_record=evidence_value,
            )
            for rater_index, descriptor in enumerate(raters):
                engine = StaticFixtureEngine(
                    descriptor=descriptor,
                    outcomes=(
                        FixtureOutcome(
                            criterion_id="claim_support",
                            status=ObservationStatus.SCORED,
                            score_category=(
                                respondent_index + prompt_index + rater_index
                            )
                            % 3,
                        ),
                        FixtureOutcome(
                            criterion_id="source_alignment",
                            status=ObservationStatus.SCORED,
                            score_category=(
                                2 + respondent_index + prompt_index + rater_index
                            )
                            % 3,
                        ),
                    ),
                )
                result = engine.score(request.scoring_request)
                records.extend(
                    build_scoring_facets_rating_records(
                        request=request.scoring_request,
                        result=result,
                        engine=descriptor,
                    )
                )
    return tuple(records), prompts


def test_facets_item_axis_uses_revision_and_retains_logical_task_labels() -> None:
    """Changed revisions become separate Rust item levels with auditable labels."""
    records, prompts = _revision_records()
    bundle = build_scoring_facets_calibration_bundle(records)
    expected_revisions = tuple(sorted(value.prompt_fingerprint for value in prompts))

    for design in bundle.designs:
        assert design.task_revision_fingerprints == expected_revisions
        assert design.task_revision_task_ids == (
            "argument_prompt",
            "argument_prompt",
        )
        assert design.responses_array().shape == (2, 2, 2)
        assert {
            record.task_revision_fingerprint for record in design.rating_records
        } == set(expected_revisions)
        serialized = design.to_dict()
        assert serialized["task_revision_fingerprints"] == list(expected_revisions)
        assert serialized["task_revision_task_ids"] == [
            "argument_prompt",
            "argument_prompt",
        ]


def test_one_task_revision_cannot_rebind_to_another_logical_task() -> None:
    """A revision digest maps to exactly one logical task across calibration data."""
    records, _prompts = _revision_records()
    target = records[0]
    conflicting = scoring.calibration.ScoringFacetsRatingRecord(
        **{
            **target._content_dict(),
            "task_id": "conflicting_task",
            "_rating_token": scoring.calibration._RATING_TOKEN,
        }
    )
    error = assert_error(
        "task_revision_provenance_conflict",
        lambda: build_scoring_facets_calibration_bundle((target, conflicting)),
    )

    assert error.path.endswith(".task_revision_fingerprint")
    assert target.task_revision_fingerprint not in str(error)


def test_v1_request_migration_requires_an_explicit_revision() -> None:
    """Legacy request migration is explicit and never hashes a logical task ID."""
    modern = criterion_request(task_revision_fingerprint="8" * 64)
    legacy = modern.to_dict()
    for key in (
        "request_handle",
        "request_fingerprint",
        "task_revision_fingerprint",
    ):
        legacy.pop(key)
    legacy["schema_version"] = "1.0"

    migrated = scoring.migrate_scoring_request_v1(
        legacy,
        assessment=assessment(),
        rubric=rubric(),
        task_revision_fingerprint="9" * 64,
    )

    assert migrated.task_id == modern.task_id
    assert migrated.task_revision_fingerprint == "9" * 64
    assert migrated.respondent_id == modern.respondent_id
    assert migrated.response_content_fingerprint == modern.response_content_fingerprint
    assert migrated.metadata == modern.metadata

    assert_error(
        "invalid_task_revision_fingerprint",
        lambda: scoring.migrate_scoring_request_v1(
            legacy,
            assessment=assessment(),
            rubric=rubric(),
            task_revision_fingerprint=None,
        ),
    )
