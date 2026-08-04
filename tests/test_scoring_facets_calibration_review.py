"""Regression tests for reviewed scoring-facets calibration invariants."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import runpy

import pytest

import fast_mlsirm.scoring.calibration as calibration
from fast_mlsirm.scoring import (
    ObservationStatus,
    build_scoring_facets_calibration_bundle,
    build_scoring_facets_rating_records,
    fit_scoring_facets_bundle,
)

_BASE = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_facets_calibration.py"))
)
assert_error = _BASE["assert_error"]
connected_records = _BASE["connected_records"]
execution = _BASE["execution"]


def test_bundle_fit_forwards_bundle_specific_tuning_values(monkeypatch) -> None:
    """Every criterion fit receives the bundle call's explicit tuning values."""
    bundle = build_scoring_facets_calibration_bundle(connected_records())
    calls = []

    def fake_fit_facets(**kwargs):
        calls.append(kwargs)
        return kwargs["responses"].shape

    monkeypatch.setattr("fast_mlsirm.facets.fit_facets", fake_fit_facets)
    fitted = fit_scoring_facets_bundle(
        bundle,
        q_theta=15,
        max_iter=66,
        tol=1e-4,
    )

    assert set(fitted) == set(bundle.criterion_ids)
    assert len(calls) == len(bundle.designs)
    assert all(call["q_theta"] == 15 for call in calls)
    assert all(call["max_iter"] == 66 for call in calls)
    assert all(call["tol"] == 1e-4 for call in calls)
    assert all(call["n_cat"] == 3 for call in calls)
    response_objects = [id(call["responses"]) for call in calls]
    assert len(set(response_objects)) == len(response_objects)


def test_all_declared_categories_must_be_observed_before_fitting() -> None:
    """Unused threshold categories cannot enter the Rust rating-scale model."""
    records = tuple(
        replace(
            record,
            allowed_scores=(0, 1, 2, 3),
            _rating_token=calibration._RATING_TOKEN,
        )
        for record in connected_records()
    )
    assert_error(
        "unobserved_facets_category",
        lambda: build_scoring_facets_calibration_bundle(records),
    )


def test_rating_fingerprint_is_cached_after_normalization(monkeypatch) -> None:
    """Repeated audit and sorting access does not reserialize rating content."""
    record = connected_records()[0]
    expected = record.rating_fingerprint

    def fail_digest(_value):
        raise AssertionError("rating digest was recomputed")

    monkeypatch.setattr(calibration, "artifact_digest", fail_digest)
    assert record.rating_fingerprint == expected
    assert record.to_dict()["rating_fingerprint"] == expected


def test_missing_criterion_has_a_specific_fail_closed_error() -> None:
    """A corrupted criterion-level result cannot degrade to an identifier error."""
    request, result, engine = execution(
        request_id="missing_criterion_request",
        response_id="missing_criterion_response",
        respondent_id="missing_criterion_respondent",
        task_id="missing_criterion_prompt",
        engine=_BASE["automated_engine"](),
        claim_score=0,
        source_score=1,
    )
    object.__setattr__(result.observations[0], "criterion_id", None)
    assert_error(
        "missing_observation_criterion",
        lambda: build_scoring_facets_rating_records(
            request=request,
            result=result,
            engine=engine,
        ),
    )


def test_bundle_rejects_cross_criterion_response_provenance_conflict() -> None:
    """One response identity cannot be rebound only in another criterion group."""
    records = list(connected_records())
    claim = next(record for record in records if record.criterion_id == "claim_support")
    source_index = next(
        index
        for index, record in enumerate(records)
        if record.criterion_id == "source_alignment"
        and record.response_id != claim.response_id
    )
    records[source_index] = replace(
        records[source_index],
        response_id=claim.response_id,
        respondent_id="conflicting_respondent",
        _rating_token=calibration._RATING_TOKEN,
    )
    assert_error(
        "response_provenance_conflict",
        lambda: build_scoring_facets_calibration_bundle(records),
    )


def test_bundle_rejects_cross_criterion_rater_provenance_conflict() -> None:
    """One engine fingerprint cannot name another rater in a second criterion."""
    records = list(connected_records())
    claim = next(record for record in records if record.criterion_id == "claim_support")
    source_index = next(
        index
        for index, record in enumerate(records)
        if record.criterion_id == "source_alignment"
        and record.engine_fingerprint != claim.engine_fingerprint
    )
    records[source_index] = replace(
        records[source_index],
        engine_fingerprint=claim.engine_fingerprint,
        engine_id="conflicting_engine",
        _rating_token=calibration._RATING_TOKEN,
    )
    assert_error(
        "rater_provenance_conflict",
        lambda: build_scoring_facets_calibration_bundle(records),
    )


def test_terminal_records_do_not_satisfy_category_observation() -> None:
    """A terminal state cannot make an unobserved score category identifiable."""
    records = []
    for record in connected_records():
        if record.score_category == 2:
            records.append(
                replace(
                    record,
                    status=ObservationStatus.ABSTAINED,
                    score_category=None,
                    _rating_token=calibration._RATING_TOKEN,
                )
            )
        else:
            records.append(record)
    assert_error(
        "unobserved_facets_category",
        lambda: build_scoring_facets_calibration_bundle(records),
    )
