"""Identification regressions for respondent-indexed scoring-facets designs."""

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
connected_records = _BASE["connected_records"]


def _replace_record(record, **changes):
    """Return one factory-authorized rating-record replacement for a test."""
    return replace(
        record,
        **changes,
        _rating_token=calibration._RATING_TOKEN,
    )


def respondent_connected_records():
    """Return two respondents observed on both tasks by both raters."""
    respondent_by_response = {
        "response_alpha_one": "respondent_one",
        "response_beta_one": "respondent_one",
        "response_alpha_two": "respondent_two",
        "response_beta_two": "respondent_two",
    }
    return tuple(
        _replace_record(
            record,
            respondent_id=respondent_by_response[record.response_id],
        )
        for record in connected_records()
    )


def respondent_diagonal_records():
    """Return an unidentified design with every respondent on only one task."""
    return tuple(
        _replace_record(
            record,
            respondent_id=f"respondent_for_{record.response_id}",
        )
        for record in connected_records()
    )


def test_design_uses_respondents_as_people_and_responses_as_provenance() -> None:
    """Repeated task responses share one respondent person parameter."""
    bundle = build_scoring_facets_calibration_bundle(
        respondent_connected_records()
    )

    for design in bundle.designs:
        assert design.respondent_ids == ("respondent_one", "respondent_two")
        assert design.responses_array().shape == (2, 2, 2)
        assert design.original_scores_array().shape == (2, 2, 2)
        assert len(design.response_states()) == 2
        assert design.to_dict()["respondent_ids"] == [
            "respondent_one",
            "respondent_two",
        ]
        assert {
            record.response_id for record in design.rating_records
        } == {
            "response_alpha_one",
            "response_alpha_two",
            "response_beta_one",
            "response_beta_two",
        }


def test_one_task_per_respondent_diagonal_fails_identification() -> None:
    """Rater overlap cannot identify task effects without respondent bridging."""
    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_calibration_bundle(respondent_diagonal_records())

    assert caught.value.code == "disconnected_facets_design"
    assert caught.value.path == "$.records"


def test_one_respondent_task_cell_cannot_pool_two_response_revisions() -> None:
    """Different raters cannot bind one respondent-task cell to two responses."""
    records = list(respondent_connected_records())
    first = records[0]
    conflicting_engine = next(
        record.engine_fingerprint
        for record in records
        if record.response_id == first.response_id
        and record.engine_fingerprint != first.engine_fingerprint
    )
    private_response = "conflicting_response"
    private_digest = "f" * 64
    records = [
        _replace_record(
            record,
            response_id=private_response,
            response_content_fingerprint=private_digest,
        )
        if record.response_id == first.response_id
        and record.engine_fingerprint == conflicting_engine
        else record
        for record in records
    ]

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_calibration_bundle(records)

    assert caught.value.code == "respondent_task_provenance_conflict"
    assert caught.value.path.endswith(".response_id")
    assert private_response not in str(caught.value)
    assert private_digest not in str(caught.value)
