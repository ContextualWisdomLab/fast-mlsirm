"""Identification contracts for governed criterion-level facets handoffs."""

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
    fit_scoring_facets_design,
)

_BASE = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_facets_calibration.py"))
)
automated_engine = _BASE["automated_engine"]
execution = _BASE["execution"]
human_engine = _BASE["human_engine"]
project = _BASE["project"]


def identified_records():
    """Return two respondents linked across two tasks and two raters."""
    records = []
    engines = (
        ("automated", automated_engine()),
        ("human", human_engine()),
    )
    score_plan = {
        ("alpha", "alpha", "automated"): (0, 2),
        ("alpha", "alpha", "human"): (1, 0),
        ("alpha", "beta", "automated"): (2, 1),
        ("alpha", "beta", "human"): (0, 2),
        ("beta", "alpha", "automated"): (1, 0),
        ("beta", "alpha", "human"): (2, 1),
        ("beta", "beta", "automated"): (0, 2),
        ("beta", "beta", "human"): (1, 0),
    }
    for respondent_name in ("alpha", "beta"):
        for task_name in ("alpha", "beta"):
            response_id = f"response_{respondent_name}_{task_name}"
            respondent_id = f"respondent_{respondent_name}"
            task_id = f"prompt_{task_name}"
            for engine_name, engine in engines:
                claim_score, source_score = score_plan[
                    (respondent_name, task_name, engine_name)
                ]
                records.extend(
                    project(
                        execution(
                            request_id=(
                                f"request_{respondent_name}_{task_name}_{engine_name}"
                            ),
                            response_id=response_id,
                            respondent_id=respondent_id,
                            task_id=task_id,
                            engine=engine,
                            claim_score=claim_score,
                            source_score=source_score,
                        )
                    )
                )
    return tuple(records)


def diagonal_records():
    """Return a task-rater-linked but respondent-task-disconnected design."""
    records = []
    rows = (
        ("alpha", "alpha", 0, 1),
        ("beta", "beta", 2, 0),
    )
    for respondent_name, task_name, automated_score, human_score in rows:
        for engine_name, engine, score in (
            ("automated", automated_engine(), automated_score),
            ("human", human_engine(), human_score),
        ):
            records.extend(
                project(
                    execution(
                        request_id=(
                            f"diagonal_{respondent_name}_{task_name}_{engine_name}"
                        ),
                        response_id=f"diagonal_response_{respondent_name}_{task_name}",
                        respondent_id=f"diagonal_respondent_{respondent_name}",
                        task_id=f"diagonal_prompt_{task_name}",
                        engine=engine,
                        claim_score=score,
                        source_score=(2 - score),
                    )
                )
            )
    return tuple(records)


def _criterion(records, criterion_id="claim_support"):
    """Return one criterion-specific record collection."""
    return tuple(record for record in records if record.criterion_id == criterion_id)


def test_identified_design_uses_respondents_as_person_axis() -> None:
    """Distinct task responses for one respondent share one person parameter."""
    bundle = build_scoring_facets_calibration_bundle(identified_records())
    design = bundle.design_by_criterion()["claim_support"]

    assert design.respondent_ids == ("respondent_alpha", "respondent_beta")
    assert design.task_ids == ("prompt_alpha", "prompt_beta")
    assert design.responses_array().shape == (2, 2, 2)
    assert design.original_scores_array().shape == (2, 2, 2)
    assert design.connected is True

    bindings: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for record in design.rating_records:
        bindings.setdefault((record.respondent_id, record.task_id), set()).add(
            (record.response_id, record.response_content_fingerprint)
        )
    assert set(bindings) == {
        ("respondent_alpha", "prompt_alpha"),
        ("respondent_alpha", "prompt_beta"),
        ("respondent_beta", "prompt_alpha"),
        ("respondent_beta", "prompt_beta"),
    }
    assert all(len(values) == 1 for values in bindings.values())


def test_identified_tensor_and_identities_are_input_order_invariant() -> None:
    """Permuting ratings does not change the governed design or dense tensor."""
    records = identified_records()
    first = build_scoring_facets_calibration_bundle(records)
    second = build_scoring_facets_calibration_bundle(reversed(records))

    assert first.bundle_fingerprint == second.bundle_fingerprint
    for criterion_id in first.criterion_ids:
        left = first.design_by_criterion()[criterion_id]
        right = second.design_by_criterion()[criterion_id]
        assert left.design_fingerprint == right.design_fingerprint
        np.testing.assert_array_equal(left.responses_array(), right.responses_array())


def test_diagonal_person_task_design_fails_identification_before_fit() -> None:
    """Rater overlap cannot identify respondents observed on disjoint tasks."""
    records = diagonal_records()
    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_calibration_bundle(records)
    assert caught.value.code == "disconnected_facets_design"

    diagnostic = build_scoring_facets_calibration_bundle(
        records,
        require_connected=False,
    )
    design = diagnostic.designs[0]
    assert design.connected is False
    for allow_disconnected in (False, True):
        with pytest.raises(AssessmentSpecError) as fit_error:
            fit_scoring_facets_design(
                design,
                allow_disconnected=allow_disconnected,
            )
        assert fit_error.value.code == "disconnected_facets_design"


def test_respondent_task_cell_rejects_conflicting_response_id() -> None:
    """Raters cannot score different logical responses as one person-task cell."""
    records = list(_criterion(identified_records()))
    first = records[0]
    conflict_index = next(
        index
        for index, record in enumerate(records[1:], start=1)
        if record.respondent_id == first.respondent_id
        and record.task_id == first.task_id
        and record.engine_fingerprint != first.engine_fingerprint
    )
    records[conflict_index] = replace(
        records[conflict_index],
        response_id="conflicting_response_revision",
        _rating_token=calibration._RATING_TOKEN,
    )

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_calibration_bundle(records)
    assert caught.value.code == "response_provenance_conflict"
    assert caught.value.path.endswith(".response_id")


def test_respondent_task_cell_rejects_conflicting_content_revision() -> None:
    """Raters cannot pool changed content under one respondent-task response."""
    records = list(_criterion(identified_records()))
    first = records[0]
    conflict_index = next(
        index
        for index, record in enumerate(records[1:], start=1)
        if record.respondent_id == first.respondent_id
        and record.task_id == first.task_id
        and record.engine_fingerprint != first.engine_fingerprint
    )
    private_digest = "f" * 64
    records[conflict_index] = replace(
        records[conflict_index],
        response_content_fingerprint=private_digest,
        _rating_token=calibration._RATING_TOKEN,
    )

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_calibration_bundle(records)
    assert caught.value.code == "response_provenance_conflict"
    assert caught.value.path.endswith(".response_content_fingerprint")
    assert private_digest not in str(caught.value)


def test_fit_delegates_respondent_indexed_tensor_to_rust(monkeypatch) -> None:
    """The Python handoff only validates and delegates the identified tensor."""
    design = build_scoring_facets_calibration_bundle(
        identified_records()
    ).designs[0]
    calls = []

    def fake_fit_facets(**kwargs):
        calls.append(kwargs)
        return kwargs["responses"].shape

    monkeypatch.setattr("fast_mlsirm.facets.fit_facets", fake_fit_facets)
    assert fit_scoring_facets_design(design) == (2, 2, 2)
    assert len(calls) == 1
    assert calls[0]["responses"].shape == (2, 2, 2)
    assert calls[0]["n_cat"] == 3
