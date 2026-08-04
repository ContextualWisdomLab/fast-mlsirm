"""Regression contracts for exact task revisions across scoring and calibration."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import runpy

import pytest

import fast_mlsirm.scoring.calibration as calibration
from fast_mlsirm.scoring import (
    AssessmentSpecError,
    LEGACY_SCORING_REQUEST_SCHEMA_VERSION,
    SCORING_REQUEST_SCHEMA_VERSION,
    artifact_digest,
    build_scoring_facets_calibration_bundle,
    canonical_json,
    fit_scoring_facets_design,
    migrate_scoring_request_v1,
)

_EXECUTION = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)
assessment = _EXECUTION["assessment"]
criterion_request = _EXECUTION["criterion_request"]
rubric = _EXECUTION["rubric"]

_ESSAY = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_essay_contracts.py"))
)
essay_request = _ESSAY["essay_request"]
prompt = _ESSAY["prompt"]
submission = _ESSAY["submission"]

_FACETS = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_facets_calibration.py"))
)
connected_records = _FACETS["connected_records"]


def _assert_error(code: str, callback) -> AssessmentSpecError:
    """Return one stable structured error after asserting its public code."""
    with pytest.raises(AssessmentSpecError) as caught:
        callback()
    assert caught.value.code == code
    return caught.value


def _legacy_request_artifact(metadata=None):
    """Return a genuine schema-1.0 artifact derived from shared content."""
    request = criterion_request(metadata={} if metadata is None else metadata)
    artifact = request.to_dict()
    artifact.pop("task_revision_fingerprint")
    artifact.pop("request_handle")
    artifact.pop("request_fingerprint")
    artifact["schema_version"] = LEGACY_SCORING_REQUEST_SCHEMA_VERSION
    legacy_fingerprint = hashlib.sha256(
        canonical_json(artifact).encode("utf-8")
    ).hexdigest()
    artifact["request_handle"] = f"scoring_request_{legacy_fingerprint[:32]}"
    artifact["request_fingerprint"] = legacy_fingerprint
    return artifact


def test_shared_request_requires_one_exact_task_revision() -> None:
    """The shared wire contract serializes a required schema-1.1 task digest."""
    request = criterion_request()
    assert request.schema_version == SCORING_REQUEST_SCHEMA_VERSION == "1.1"
    assert request.task_revision_fingerprint == "d" * 64
    assert request.to_dict()["task_revision_fingerprint"] == "d" * 64
    for invalid_value in ("not_a_digest", None):
        caught = _assert_error(
            "invalid_task_revision_fingerprint",
            lambda invalid_value=invalid_value: criterion_request(
                task_revision_fingerprint=invalid_value
            ),
        )
        assert caught.path == "$.task_revision_fingerprint"
        assert str(invalid_value) not in str(caught)


def test_task_revision_changes_request_identity_without_changing_logical_task() -> None:
    """Changed task content remains a distinct request under one display ID."""
    first = criterion_request(task_revision_fingerprint="1" * 64)
    second = criterion_request(task_revision_fingerprint="2" * 64)
    assert first.task_id == second.task_id == "sample_task"
    assert first.task_revision_fingerprint != second.task_revision_fingerprint
    assert first.request_fingerprint != second.request_fingerprint


def test_essay_prompt_revision_is_the_authoritative_shared_task_revision() -> None:
    """Changed prompt content propagates beyond adapter-specific metadata."""
    first_prompt = prompt(prompt_content_fingerprint="1" * 64)
    second_prompt = prompt(prompt_content_fingerprint="9" * 64)
    first_submission = submission(first_prompt, response_id="first_response")
    second_submission = submission(second_prompt, response_id="second_response")
    first = essay_request(prompt=first_prompt, submission=first_submission)
    second = essay_request(prompt=second_prompt, submission=second_submission)
    assert first.scoring_request.task_id == second.scoring_request.task_id
    assert first.scoring_request.task_revision_fingerprint == first_prompt.prompt_fingerprint
    assert second.scoring_request.task_revision_fingerprint == second_prompt.prompt_fingerprint
    assert (
        first.scoring_request.task_revision_fingerprint
        != second.scoring_request.task_revision_fingerprint
    )
    assert first.scoring_request.request_fingerprint != second.scoring_request.request_fingerprint


def test_facets_axis_separates_revisions_of_one_logical_task() -> None:
    """Calibration indexes exact revisions while retaining aligned display IDs."""
    bundle = build_scoring_facets_calibration_bundle(
        connected_records(shared_task_id=True)
    )
    design = bundle.designs[0]
    assert len(design.task_revision_fingerprints) == 2
    assert design.task_ids == ("shared_prompt", "shared_prompt")
    assert design.task_family_ids == ("evidence_review", "evidence_review")
    assert design.responses_array().shape == (2, 2, 2)
    serialized = design.to_dict()
    assert [value["task_id"] for value in serialized["task_revisions"]] == [
        "shared_prompt",
        "shared_prompt",
    ]
    assert [
        value["task_revision_fingerprint"]
        for value in serialized["task_revisions"]
    ] == list(design.task_revision_fingerprints)


def test_one_revision_cannot_claim_two_logical_task_identities() -> None:
    """A revision-to-logical collision fails before tensor allocation."""
    records = list(connected_records())
    first_revision = records[0].task_revision_fingerprint
    conflicting_index = next(
        index
        for index, record in enumerate(records[1:], start=1)
        if record.task_revision_fingerprint != first_revision
    )
    records[conflicting_index] = replace(
        records[conflicting_index],
        task_revision_fingerprint=first_revision,
        _rating_token=calibration._RATING_TOKEN,
    )
    caught = _assert_error(
        "task_revision_provenance_conflict",
        lambda: build_scoring_facets_calibration_bundle(records),
    )
    assert caught.path.endswith(".task_id")


def test_design_replay_rejects_a_mutated_revision_axis_before_rust(monkeypatch) -> None:
    """Estimator authorization replays the exact revision-indexed design."""
    design = build_scoring_facets_calibration_bundle(connected_records()).designs[0]
    object.__setattr__(
        design,
        "task_revision_fingerprints",
        tuple(reversed(design.task_revision_fingerprints)),
    )
    called = False

    def unexpected_fit(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("Rust delegation must not occur")

    monkeypatch.setattr("fast_mlsirm.facets.fit_facets", unexpected_fit)
    _assert_error(
        "facets_design_replay_mismatch",
        lambda: fit_scoring_facets_design(design),
    )
    assert called is False


def test_legacy_request_migration_requires_an_explicit_revision() -> None:
    """A verified v1.0 artifact migrates only with supplied task content."""
    artifact = _legacy_request_artifact()
    migrated = migrate_scoring_request_v1(
        artifact,
        assessment=assessment(),
        rubric=rubric(),
        task_revision_fingerprint="e" * 64,
    )
    assert migrated.schema_version == SCORING_REQUEST_SCHEMA_VERSION
    assert migrated.task_revision_fingerprint == "e" * 64
    assert migrated.task_id == artifact["task_id"]
    assert migrated.request_fingerprint != artifact["request_fingerprint"]
    for invalid_value in ("missing_revision", None):
        caught = _assert_error(
            "invalid_task_revision_fingerprint",
            lambda invalid_value=invalid_value: migrate_scoring_request_v1(
                artifact,
                assessment=assessment(),
                rubric=rubric(),
                task_revision_fingerprint=invalid_value,
            ),
        )
        assert caught.path == "$.task_revision_fingerprint"
        assert str(invalid_value) not in str(caught)


def test_legacy_migration_preserves_caller_metadata_and_replays_authorization() -> None:
    """Migration retains caller metadata but rebuilds package policy fields."""
    artifact = _legacy_request_artifact(
        metadata={
            "language_code": "ko",
            "deployment_context": {"region_name": "seoul_region"},
        }
    )
    selected_assessment = assessment()
    migrated = migrate_scoring_request_v1(
        artifact,
        assessment=selected_assessment,
        rubric=rubric(),
        task_revision_fingerprint="e" * 64,
    )
    metadata = migrated.to_dict()["metadata"]
    assert metadata["language_code"] == "ko"
    assert metadata["deployment_context"] == {"region_name": "seoul_region"}
    assert metadata["engine_policy_fingerprint"] == artifact_digest(
        selected_assessment.engine_policy
    )
    assert metadata["permitted_engine_ids"] == list(
        selected_assessment.engine_policy.engine_ids
    )


def test_legacy_request_migration_rejects_payload_and_identity_tampering() -> None:
    """Migration verifies canonical content identity and public handle."""
    changed_payload = _legacy_request_artifact()
    changed_payload["task_id"] = "changed_task"
    _assert_error(
        "legacy_request_fingerprint_mismatch",
        lambda: migrate_scoring_request_v1(
            changed_payload,
            assessment=assessment(),
            rubric=rubric(),
            task_revision_fingerprint="e" * 64,
        ),
    )
    changed_handle = _legacy_request_artifact()
    changed_handle["request_handle"] = "scoring_request_" + "0" * 32
    _assert_error(
        "legacy_request_handle_mismatch",
        lambda: migrate_scoring_request_v1(
            changed_handle,
            assessment=assessment(),
            rubric=rubric(),
            task_revision_fingerprint="e" * 64,
        ),
    )


def test_legacy_migration_rejects_unknown_or_incomplete_artifact_shapes() -> None:
    """Unversioned, incomplete, or extended legacy mappings are not guessed."""
    missing = _legacy_request_artifact()
    missing.pop("request_id")
    _assert_error(
        "invalid_legacy_scoring_request",
        lambda: migrate_scoring_request_v1(
            missing,
            assessment=assessment(),
            rubric=rubric(),
            task_revision_fingerprint="e" * 64,
        ),
    )
    extended = _legacy_request_artifact()
    extended["essay_prompt_fingerprint"] = "f" * 64
    _assert_error(
        "invalid_legacy_scoring_request",
        lambda: migrate_scoring_request_v1(
            extended,
            assessment=assessment(),
            rubric=rubric(),
            task_revision_fingerprint="e" * 64,
        ),
    )
