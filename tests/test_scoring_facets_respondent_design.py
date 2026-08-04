"""Identification contracts for respondent-indexed scoring-facets designs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import runpy

import numpy as np
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
execution = _BASE["execution"]
human_engine = _BASE["human_engine"]
project = _BASE["project"]


def _cell_records(
    *,
    respondent_id: str,
    task_id: str,
    response_id: str,
    content_fingerprint: str,
    engine,
    engine_index: int,
    base_score: int,
):
    """Return two criterion records for one governed respondent-task rating."""
    records = project(
        execution(
            request_id=f"request_{respondent_id}_{task_id}_{engine_index}",
            response_id=response_id,
            respondent_id=respondent_id,
            task_id=task_id,
            engine=engine,
            claim_score=(base_score + engine_index) % 3,
            source_score=(base_score + engine_index + 1) % 3,
        )
    )
    return tuple(
        replace(
            record,
            response_content_fingerprint=content_fingerprint,
            _rating_token=calibration._RATING_TOKEN,
        )
        for record in records
    )


def _fully_linked_records():
    """Return two respondents crossing two tasks and two exact raters."""
    records = []
    engines = (automated_engine(), human_engine())
    respondents = ("respondent_alpha", "respondent_beta")
    tasks = ("prompt_alpha", "prompt_beta")
    digests = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)
    for respondent_index, respondent_id in enumerate(respondents):
        for task_index, task_id in enumerate(tasks):
            cell_index = respondent_index * len(tasks) + task_index
            response_id = f"response_{respondent_id}_{task_id}"
            for engine_index, engine in enumerate(engines):
                records.extend(
                    _cell_records(
                        respondent_id=respondent_id,
                        task_id=task_id,
                        response_id=response_id,
                        content_fingerprint=digests[cell_index],
                        engine=engine,
                        engine_index=engine_index,
                        base_score=cell_index,
                    )
                )
    return tuple(records)


def test_design_person_axis_is_respondent_not_response() -> None:
    """Repeated task responses for one respondent share one proficiency axis row."""
    bundle = build_scoring_facets_calibration_bundle(_fully_linked_records())

    assert bundle.criterion_ids == ("claim_support", "source_alignment")
    for design in bundle.designs:
        assert design.respondent_ids == ("respondent_alpha", "respondent_beta")
        assert design.task_ids == ("prompt_alpha", "prompt_beta")
        assert design.responses_array().shape == (2, 2, 2)
        assert design.original_scores_array().shape == (2, 2, 2)
        assert not np.isnan(design.responses_array()).any()
        assert design.connected is True
        assert len({record.response_id for record in design.rating_records}) == 4
        assert len(
            {record.response_content_fingerprint for record in design.rating_records}
        ) == 4


def test_diagonal_respondent_task_design_fails_before_fitting() -> None:
    """Common raters cannot identify task effects when respondents never bridge tasks."""
    records = []
    for respondent_id, task_id, response_id, digest, base_score in (
        (
            "respondent_alpha",
            "prompt_alpha",
            "response_alpha",
            "a" * 64,
            0,
        ),
        (
            "respondent_beta",
            "prompt_beta",
            "response_beta",
            "b" * 64,
            1,
        ),
    ):
        for engine_index, engine in enumerate((automated_engine(), human_engine())):
            records.extend(
                _cell_records(
                    respondent_id=respondent_id,
                    task_id=task_id,
                    response_id=response_id,
                    content_fingerprint=digest,
                    engine=engine,
                    engine_index=engine_index,
                    base_score=base_score,
                )
            )

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_calibration_bundle(records)

    assert caught.value.code == "disconnected_respondent_task_design"
    assert caught.value.path == "$.records"


def test_disconnected_task_rater_design_has_a_distinct_failure() -> None:
    """Respondent bridging cannot identify rater severity confounded with task."""
    records = []
    for respondent_index, respondent_id in enumerate(
        ("respondent_alpha", "respondent_beta")
    ):
        for task_index, (task_id, engine) in enumerate(
            (("prompt_alpha", automated_engine()), ("prompt_beta", human_engine()))
        ):
            cell_index = respondent_index * 2 + task_index
            records.extend(
                _cell_records(
                    respondent_id=respondent_id,
                    task_id=task_id,
                    response_id=f"response_{respondent_id}_{task_id}",
                    content_fingerprint=("a", "b", "c", "d")[cell_index] * 64,
                    engine=engine,
                    engine_index=task_index,
                    base_score=cell_index,
                )
            )

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_calibration_bundle(records)

    assert caught.value.code == "disconnected_task_rater_design"
    assert caught.value.path == "$.records"


def test_one_respondent_task_cell_cannot_bind_two_response_ids() -> None:
    """Raters of one cell must consume the same governed response revision."""
    records = []
    for engine_index, (engine, response_id) in enumerate(
        (
            (automated_engine(), "response_revision_one"),
            (human_engine(), "response_revision_two"),
        )
    ):
        records.extend(
            _cell_records(
                respondent_id="respondent_alpha",
                task_id="prompt_alpha",
                response_id=response_id,
                content_fingerprint="a" * 64,
                engine=engine,
                engine_index=engine_index,
                base_score=engine_index,
            )
        )
    records.extend(
        _cell_records(
            respondent_id="respondent_alpha",
            task_id="prompt_beta",
            response_id="response_bridge",
            content_fingerprint="b" * 64,
            engine=automated_engine(),
            engine_index=0,
            base_score=2,
        )
    )
    records.extend(
        _cell_records(
            respondent_id="respondent_beta",
            task_id="prompt_alpha",
            response_id="response_beta_alpha",
            content_fingerprint="c" * 64,
            engine=human_engine(),
            engine_index=1,
            base_score=0,
        )
    )
    records.extend(
        _cell_records(
            respondent_id="respondent_beta",
            task_id="prompt_beta",
            response_id="response_beta_beta",
            content_fingerprint="d" * 64,
            engine=automated_engine(),
            engine_index=0,
            base_score=1,
        )
    )

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_calibration_bundle(records)

    assert caught.value.code == "respondent_task_response_conflict"
    assert caught.value.path.endswith(".response_id")


def test_one_respondent_task_cell_cannot_bind_two_content_revisions() -> None:
    """A reused response ID cannot hide different content across raters."""
    records = list(_fully_linked_records())
    target = next(
        index
        for index, record in enumerate(records)
        if record.criterion_id == "claim_support"
        and record.respondent_id == "respondent_alpha"
        and record.task_id == "prompt_alpha"
        and record.engine_id == "human_engine"
    )
    records[target] = replace(
        records[target],
        response_content_fingerprint="f" * 64,
        _rating_token=calibration._RATING_TOKEN,
    )

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_calibration_bundle(records)

    assert caught.value.code == "respondent_task_response_conflict"
    assert caught.value.path.endswith(".response_content_fingerprint")
