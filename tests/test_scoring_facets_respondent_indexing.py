"""Respondent-indexed identification contracts for scoring-facets handoffs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import runpy

import pytest

import fast_mlsirm.scoring.calibration as calibration
from fast_mlsirm.scoring import (
    AssessmentSpecError,
    build_scoring_facets_calibration_bundle,
)

_BASE = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_facets_calibration.py"))
)
automated_engine = _BASE["automated_engine"]
human_engine = _BASE["human_engine"]
execution = _BASE["execution"]
project = _BASE["project"]


def _project_rows(rows, *, rater_selector=None):
    """Project deterministic respondent-task rows through selected raters."""
    records = []
    engines = (automated_engine(), human_engine())
    for row_index, (respondent_id, task_id, base_score) in enumerate(rows):
        selected = engines if rater_selector is None else rater_selector(task_id, engines)
        for engine_index, engine in enumerate(selected):
            rater_index = engines.index(engine)
            response_id = f"response_{respondent_id}_{task_id}"
            records.extend(
                project(
                    execution(
                        request_id=(
                            f"request_{respondent_id}_{task_id}_{rater_index}"
                        ),
                        response_id=response_id,
                        respondent_id=respondent_id,
                        task_id=task_id,
                        engine=engine,
                        claim_score=(base_score + rater_index) % 3,
                        source_score=(2 - base_score + rater_index) % 3,
                    )
                )
            )
    return tuple(records)


def respondent_connected_records():
    """Return two respondents linked across two tasks and two raters."""
    return _project_rows(
        (
            ("respondent_alpha", "prompt_alpha", 0),
            ("respondent_alpha", "prompt_beta", 1),
            ("respondent_beta", "prompt_alpha", 2),
            ("respondent_beta", "prompt_beta", 0),
        )
    )


def test_person_axis_is_unique_respondent_not_response_identity() -> None:
    """Repeated task performances for one respondent share one person parameter."""
    bundle = build_scoring_facets_calibration_bundle(respondent_connected_records())

    for design in bundle.designs:
        assert design.respondent_ids == ("respondent_alpha", "respondent_beta")
        assert design.task_ids == ("prompt_alpha", "prompt_beta")
        assert design.responses_array().shape == (2, 2, 2)
        assert design.connected is True
        assert {
            (record.respondent_id, record.task_id, record.response_id)
            for record in design.rating_records
        } == {
            (
                "respondent_alpha",
                "prompt_alpha",
                "response_respondent_alpha_prompt_alpha",
            ),
            (
                "respondent_alpha",
                "prompt_beta",
                "response_respondent_alpha_prompt_beta",
            ),
            (
                "respondent_beta",
                "prompt_alpha",
                "response_respondent_beta_prompt_alpha",
            ),
            (
                "respondent_beta",
                "prompt_beta",
                "response_respondent_beta_prompt_beta",
            ),
        }


def test_one_respondent_across_tasks_is_not_an_identified_person_design() -> None:
    """Multiple response IDs from one respondent cannot fake person support."""
    records = _project_rows(
        (
            ("respondent_alpha", "prompt_alpha", 0),
            ("respondent_alpha", "prompt_beta", 2),
        )
    )

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_calibration_bundle(records)

    assert caught.value.code == "insufficient_facets_respondents"
    assert caught.value.path == "$.records"


def test_one_task_with_multiple_respondents_is_not_an_identified_task_design() -> None:
    """Task difficulty requires at least two linked task levels."""
    records = _project_rows(
        (
            ("respondent_alpha", "prompt_alpha", 0),
            ("respondent_beta", "prompt_alpha", 2),
        )
    )

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_calibration_bundle(records)

    assert caught.value.code == "insufficient_facets_tasks"
    assert caught.value.path == "$.records"


def test_diagonal_respondent_task_design_fails_before_fitting() -> None:
    """One-task-per-respondent data cannot separate persons from tasks."""
    records = _project_rows(
        (
            ("respondent_alpha", "prompt_alpha", 0),
            ("respondent_beta", "prompt_beta", 2),
        )
    )

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_calibration_bundle(records)

    assert caught.value.code == "disconnected_facets_design"
    assert caught.value.path == "$.records"


def test_task_rater_graph_must_also_be_connected() -> None:
    """Respondent-task linking does not substitute for task-rater linking."""

    def task_specific_rater(task_id, engines):
        return (engines[0],) if task_id == "prompt_alpha" else (engines[1],)

    records = _project_rows(
        (
            ("respondent_alpha", "prompt_alpha", 0),
            ("respondent_alpha", "prompt_beta", 1),
            ("respondent_beta", "prompt_alpha", 2),
            ("respondent_beta", "prompt_beta", 0),
        ),
        rater_selector=task_specific_rater,
    )

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_calibration_bundle(records)

    assert caught.value.code == "disconnected_facets_design"
    assert caught.value.path == "$.records"


def test_respondent_task_cell_binds_one_response_revision() -> None:
    """Different response artifacts cannot occupy one respondent-task cell."""
    records = list(respondent_connected_records())
    target = records[0]
    conflicting_indexes = [
        index
        for index, record in enumerate(records)
        if record.respondent_id == target.respondent_id
        and record.task_id == target.task_id
        and record.engine_fingerprint != target.engine_fingerprint
    ]
    assert len(conflicting_indexes) == 2
    private_digest = "f" * 64
    for index in conflicting_indexes:
        records[index] = replace(
            records[index],
            response_id="conflicting_response_revision",
            response_content_fingerprint=private_digest,
            _rating_token=calibration._RATING_TOKEN,
        )

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_calibration_bundle(records)

    assert caught.value.code == "respondent_task_provenance_conflict"
    assert caught.value.path.endswith(".response_id")
    assert private_digest not in str(caught.value)


def test_multiple_raters_share_one_exact_respondent_task_revision() -> None:
    """Rater-specific requests may score the same governed response revision."""
    records = respondent_connected_records()
    target_records = tuple(
        record
        for record in records
        if record.respondent_id == "respondent_alpha"
        and record.task_id == "prompt_alpha"
        and record.criterion_id == "claim_support"
    )

    assert len(target_records) == 2
    assert len({record.engine_fingerprint for record in target_records}) == 2
    assert len({record.request_fingerprint for record in target_records}) == 2
    assert len({record.response_id for record in target_records}) == 1
    assert len({record.response_content_fingerprint for record in target_records}) == 1
    bundle = build_scoring_facets_calibration_bundle(records)
    assert all(design.connected for design in bundle.designs)
