"""Tests for governed criterion-level many-facet calibration handoffs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import runpy
from typing import Any

import numpy as np
import pytest

import fast_mlsirm.scoring as scoring
import fast_mlsirm.scoring.calibration as calibration
from fast_mlsirm.scoring import (
    AssessmentSpecError,
    FixtureOutcome,
    ObservationStatus,
    StaticFixtureEngine,
    build_scoring_facets_calibration_bundle,
    build_scoring_facets_rating_records,
    fit_scoring_facets_bundle,
    fit_scoring_facets_design,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)
assessment = _FIXTURES["assessment"]
automated_engine = _FIXTURES["automated_engine"]
criterion_request = _FIXTURES["criterion_request"]
human_engine = _FIXTURES["human_engine"]


def assert_error(code: str, callback) -> None:
    """Assert one stable scoring-contract error code."""
    with pytest.raises(AssessmentSpecError) as caught:
        callback()
    assert caught.value.code == code


def governed_execution(
    *,
    request_id: str,
    response_id: str,
    respondent_id: str,
    task_id: str,
    engine,
    claim_score: int | None,
    source_score: int | None,
    claim_status: ObservationStatus = ObservationStatus.SCORED,
    source_status: ObservationStatus = ObservationStatus.SCORED,
    occasion_id: str = "initial_occasion",
):
    """Return one governed request/result/engine execution fixture."""
    request = criterion_request(
        request_id=request_id,
        response_id=response_id,
        respondent_id=respondent_id,
        task_id=task_id,
        occasion_id=occasion_id,
    )
    outcomes = []
    for criterion_id, status, score in (
        ("claim_support", claim_status, claim_score),
        ("source_alignment", source_status, source_score),
    ):
        outcomes.append(
            FixtureOutcome(
                criterion_id=criterion_id,
                status=status,
                score_category=score,
                reason_code=(
                    None
                    if status is ObservationStatus.SCORED
                    else "insufficient_evidence"
                ),
            )
        )
    fixture = StaticFixtureEngine(descriptor=engine, outcomes=tuple(outcomes))
    return request, fixture.score(request), engine


def connected_records(*, terminal_cell: bool = False):
    """Return a connected two-task, two-rater criterion-level design."""
    engines = (automated_engine(), human_engine())
    rows = (
        ("request_alpha_one", "response_alpha_one", "respondent_alpha_one", "prompt_alpha", 0),
        ("request_alpha_two", "response_alpha_two", "respondent_alpha_two", "prompt_alpha", 1),
        ("request_beta_one", "response_beta_one", "respondent_beta_one", "prompt_beta", 2),
        ("request_beta_two", "response_beta_two", "respondent_beta_two", "prompt_beta", 0),
    )
    records = []
    for row_index, (request_id, response_id, respondent_id, task_id, base_score) in enumerate(rows):
        for engine_index, engine in enumerate(engines):
            claim_status = ObservationStatus.SCORED
            claim_score: int | None = (base_score + engine_index) % 3
            if terminal_cell and row_index == 0 and engine_index == 0:
                claim_status = ObservationStatus.ABSTAINED
                claim_score = None
            execution = governed_execution(
                request_id=f"{request_id}_{engine_index}",
                response_id=response_id,
                respondent_id=respondent_id,
                task_id=task_id,
                engine=engine,
                claim_score=claim_score,
                source_score=(2 - base_score + engine_index) % 3,
                claim_status=claim_status,
            )
            records.extend(
                build_scoring_facets_rating_records(
                    request=execution[0],
                    result=execution[1],
                    engine=execution[2],
                )
            )
    return tuple(records)


def disconnected_records():
    """Return two task-rater components with no observed linking edge."""
    executions = (
        governed_execution(
            request_id="request_component_one",
            response_id="response_component_one",
            respondent_id="respondent_component_one",
            task_id="prompt_component_one",
            engine=automated_engine(),
            claim_score=0,
            source_score=0,
        ),
        governed_execution(
            request_id="request_component_two",
            response_id="response_component_two",
            respondent_id="respondent_component_two",
            task_id="prompt_component_two",
            engine=human_engine(),
            claim_score=2,
            source_score=2,
        ),
    )
    records = []
    for request, result, engine in executions:
        records.extend(
            build_scoring_facets_rating_records(
                request=request,
                result=result,
                engine=engine,
            )
        )
    return tuple(records)


def first_record():
    """Return one valid projected rating record."""
    return connected_records()[0]


def direct_record(**overrides: Any):
    """Construct a rating with the private token for invariant branch tests."""
    source = first_record()
    values: dict[str, Any] = {
        "assessment_fingerprint": source.assessment_fingerprint,
        "rubric_fingerprint": source.rubric_fingerprint,
        "construct_id": source.construct_id,
        "request_fingerprint": source.request_fingerprint,
        "result_fingerprint": source.result_fingerprint,
        "observation_fingerprint": source.observation_fingerprint,
        "respondent_id": source.respondent_id,
        "response_id": source.response_id,
        "task_id": source.task_id,
        "occasion_id": source.occasion_id,
        "criterion_id": source.criterion_id,
        "engine_id": source.engine_id,
        "engine_family_id": source.engine_family_id,
        "engine_fingerprint": source.engine_fingerprint,
        "status": source.status,
        "score_category": source.score_category,
        "allowed_scores": source.allowed_scores,
        "_rating_token": calibration._RATING_TOKEN,
    }
    values.update(overrides)
    return calibration.ScoringFacetsRatingRecord(**values)


def test_public_surface_is_explicit_without_changing_star_import_contract() -> None:
    """Calibration APIs are explicit attributes while pinned ``__all__`` stays stable."""
    names = {
        "MAX_SCORING_FACETS_CELLS",
        "MAX_SCORING_FACETS_RATINGS",
        "ScoringFacetsCalibrationBundle",
        "ScoringFacetsDesign",
        "ScoringFacetsRatingRecord",
        "build_scoring_facets_calibration_bundle",
        "build_scoring_facets_rating_records",
        "fit_scoring_facets_bundle",
        "fit_scoring_facets_design",
    }
    assert all(hasattr(scoring, name) for name in names)
    assert names.isdisjoint(scoring.__all__)
    assert all(getattr(scoring, name).__doc__ for name in names if name[0].isupper())


def test_rating_projection_retains_exact_governed_provenance() -> None:
    """One governed result becomes sorted immutable criterion rating records."""
    request, result, engine = governed_execution(
        request_id="projection_request",
        response_id="projection_response",
        respondent_id="projection_respondent",
        task_id="projection_prompt",
        engine=automated_engine(),
        claim_score=2,
        source_score=None,
        source_status=ObservationStatus.ABSTAINED,
    )
    records = build_scoring_facets_rating_records(
        request=request,
        result=result,
        engine=engine,
    )

    assert tuple(record.criterion_id for record in records) == tuple(
        sorted(("claim_support", "source_alignment"))
    )
    scored = next(record for record in records if record.criterion_id == "claim_support")
    terminal = next(
        record for record in records if record.criterion_id == "source_alignment"
    )
    assert scored.score_category == 2
    assert terminal.status is ObservationStatus.ABSTAINED
    assert terminal.score_category is None
    assert scored.request_fingerprint == request.request_fingerprint
    assert scored.result_fingerprint == result.result_fingerprint
    assert scored.engine_fingerprint == engine.engine_fingerprint
    assert scored.rating_handle == f"scoring_facets_rating_{scored.rating_fingerprint[:32]}"
    assert scored.to_dict()["allowed_scores"] == [0, 1, 2]


def test_bundle_is_order_invariant_and_preserves_sparse_states() -> None:
    """Record order cannot change identities, arrays, or terminal-state semantics."""
    records = connected_records(terminal_cell=True)
    first = build_scoring_facets_calibration_bundle(records)
    second = build_scoring_facets_calibration_bundle(reversed(records))

    assert first.bundle_fingerprint == second.bundle_fingerprint
    assert first.bundle_handle.startswith("scoring_facets_bundle_")
    assert first.criterion_ids == ("claim_support", "source_alignment")
    assert first.to_dict()["criterion_ids"] == list(first.criterion_ids)
    assert set(first.design_by_criterion()) == set(first.criterion_ids)

    design = first.design_by_criterion()["claim_support"]
    normalized = design.responses_array()
    original = design.original_scores_array()
    states = design.response_states()
    assert normalized.shape == original.shape == (4, 2, 2)
    assert np.isnan(normalized).sum() == 9
    assert np.isnan(original).sum() == 9
    assert set(normalized[np.isfinite(normalized)]) <= {0.0, 1.0, 2.0}
    assert set(original[np.isfinite(original)]) <= {0.0, 1.0, 2.0}
    assert any(
        state is ObservationStatus.ABSTAINED
        for response_states in states
        for task_states in response_states
        for state in task_states
    )
    assert any(
        state is None
        for response_states in states
        for task_states in response_states
        for state in task_states
    )
    normalized[0, 0, 0] = 99
    assert 99 not in design.responses_array()
    assert design.to_fit_facets_kwargs()["n_cat"] == 3
    assert design.design_handle.startswith("scoring_facets_design_")
    assert design.to_dict()["connected"] is True


def test_noncontiguous_scores_map_only_at_estimator_boundary() -> None:
    """Original ordinal labels remain auditable while Rust receives zero-based values."""
    records = []
    for index, record in enumerate(connected_records()):
        score = None if record.score_category is None else (1, 3, 5)[record.score_category]
        records.append(
            replace(
                record,
                score_category=score,
                allowed_scores=(1, 3, 5),
                _rating_token=calibration._RATING_TOKEN,
            )
        )
    design = build_scoring_facets_calibration_bundle(records).designs[0]
    assert design.category_values == (1, 3, 5)
    assert set(design.original_scores_array()[np.isfinite(design.original_scores_array())]) <= {
        1.0,
        3.0,
        5.0,
    }
    assert set(design.responses_array()[np.isfinite(design.responses_array())]) <= {
        0.0,
        1.0,
        2.0,
    }


def test_fit_helpers_delegate_without_reimplementing_numerics(monkeypatch) -> None:
    """Fit helpers pass copied tensors and tuning values to ``fit_facets``."""
    bundle = build_scoring_facets_calibration_bundle(connected_records())
    calls = []

    def fake_fit_facets(**kwargs):
        calls.append(kwargs)
        return {"shape": kwargs["responses"].shape, "n_cat": kwargs["n_cat"]}

    monkeypatch.setattr("fast_mlsirm.facets.fit_facets", fake_fit_facets)
    one = fit_scoring_facets_design(
        bundle.designs[0], q_theta=21, max_iter=77, tol=1e-5
    )
    all_fits = fit_scoring_facets_bundle(
        bundle, q_theta=15, max_iter=66, tol=1e-4
    )

    assert one == {"shape": (4, 2, 2), "n_cat": 3}
    assert set(all_fits) == set(bundle.criterion_ids)
    assert calls[0]["q_theta"] == 21
    assert calls[0]["max_iter"] == 77
    assert calls[0]["tol"] == 1e-5
    assert all(call["n_cat"] == 3 for call in calls)


def test_disconnected_design_requires_two_explicit_opt_ins(monkeypatch) -> None:
    """Disconnected diagnostic artifacts cannot be fitted accidentally."""
    records = disconnected_records()
    assert_error(
        "disconnected_facets_design",
        lambda: build_scoring_facets_calibration_bundle(records),
    )
    bundle = build_scoring_facets_calibration_bundle(
        records,
        require_connected=False,
    )
    assert all(not design.connected for design in bundle.designs)
    assert_error(
        "disconnected_facets_design",
        lambda: fit_scoring_facets_design(bundle.designs[0]),
    )
    monkeypatch.setattr(
        "fast_mlsirm.facets.fit_facets",
        lambda **kwargs: kwargs["responses"].shape,
    )
    assert fit_scoring_facets_design(
        bundle.designs[0], allow_disconnected=True
    ) == (2, 2, 2)
    assert_error(
        "invalid_require_connected",
        lambda: build_scoring_facets_calibration_bundle(
            records,
            require_connected=1,
        ),
    )
    assert_error(
        "invalid_allow_disconnected",
        lambda: fit_scoring_facets_design(
            bundle.designs[0],
            allow_disconnected=1,
        ),
    )


def test_projection_rejects_invalid_types_granularity_and_bindings() -> None:
    """Projection accepts only matched criterion-level governed artifacts."""
    request, result, engine = governed_execution(
        request_id="validation_request",
        response_id="validation_response",
        respondent_id="validation_respondent",
        task_id="validation_prompt",
        engine=automated_engine(),
        claim_score=0,
        source_score=1,
    )
    assert_error(
        "invalid_scoring_request",
        lambda: build_scoring_facets_rating_records(
            request=object(), result=result, engine=engine
        ),
    )
    assert_error(
        "invalid_scoring_result",
        lambda: build_scoring_facets_rating_records(
            request=request, result=object(), engine=engine
        ),
    )
    assert_error(
        "invalid_engine_descriptor",
        lambda: build_scoring_facets_rating_records(
            request=request, result=result, engine=object()
        ),
    )
    holistic = _FIXTURES["holistic_request"](
        request_id="holistic_validation_request",
        response_id="holistic_validation_response",
        respondent_id="holistic_validation_respondent",
        task_id="holistic_validation_prompt",
    )
    holistic_result = StaticFixtureEngine(
        descriptor=engine,
        outcomes=(
            FixtureOutcome(
                criterion_id=None,
                status=ObservationStatus.SCORED,
                score_category=1,
            ),
        ),
    ).score(holistic)
    assert_error(
        "unsupported_calibration_granularity",
        lambda: build_scoring_facets_rating_records(
            request=holistic,
            result=holistic_result,
            engine=engine,
        ),
    )
    other_request, other_result, _ = governed_execution(
        request_id="other_validation_request",
        response_id="other_validation_response",
        respondent_id="other_validation_respondent",
        task_id="other_validation_prompt",
        engine=engine,
        claim_score=1,
        source_score=2,
    )
    assert_error(
        "calibration_request_result_mismatch",
        lambda: build_scoring_facets_rating_records(
            request=request,
            result=other_result,
            engine=engine,
        ),
    )
    assert_error(
        "calibration_engine_result_mismatch",
        lambda: build_scoring_facets_rating_records(
            request=other_request,
            result=other_result,
            engine=human_engine(),
        ),
    )


def test_rating_factory_guards_score_status_scale_and_direct_construction() -> None:
    """Private-token invariant tests cover malformed score and status combinations."""
    source = first_record()
    with pytest.raises(AssessmentSpecError) as caught:
        replace(source)
    assert caught.value.code == "unverified_facets_rating"
    assert_error(
        "invalid_observation_status",
        lambda: direct_record(status="unknown_status"),
    )
    assert_error(
        "missing_rating_score",
        lambda: direct_record(score_category=None),
    )
    assert_error(
        "unexpected_rating_score",
        lambda: direct_record(
            status=ObservationStatus.ABSTAINED,
            score_category=1,
        ),
    )
    assert_error(
        "rating_score_out_of_range",
        lambda: direct_record(score_category=9),
    )
    assert_error(
        "invalid_allowed_scores",
        lambda: direct_record(allowed_scores=(0, True, 2)),
    )
    assert_error(
        "invalid_allowed_scores",
        lambda: direct_record(allowed_scores=(0, 2, 1)),
    )


def test_bundle_rejects_invalid_records_duplicates_and_mixed_contracts() -> None:
    """Collection assembly fails closed before dense allocation."""
    records = connected_records()
    assert_error(
        "invalid_facets_rating_record",
        lambda: build_scoring_facets_calibration_bundle((*records, object())),
    )
    assert_error(
        "duplicate_facets_rating_record",
        lambda: build_scoring_facets_calibration_bundle((*records, records[0])),
    )
    mixed = replace(
        records[0],
        occasion_id="secondary_occasion",
        _rating_token=calibration._RATING_TOKEN,
    )
    assert_error(
        "mixed_facets_calibration_contract",
        lambda: build_scoring_facets_calibration_bundle((mixed, *records[1:])),
    )
    assert_error(
        "invalid_facets_rating_record",
        lambda: build_scoring_facets_calibration_bundle("not_records"),
    )


def test_design_rejects_duplicate_and_conflicting_provenance() -> None:
    """Response, rater, and cell identities cannot be rebound within a criterion."""
    records = list(connected_records())
    criterion = records[0].criterion_id
    group = [record for record in records if record.criterion_id == criterion]

    duplicate_cell = replace(
        group[1],
        observation_fingerprint="e" * 64,
        result_fingerprint="f" * 64,
        _rating_token=calibration._RATING_TOKEN,
    )
    duplicate_cell = replace(
        duplicate_cell,
        response_id=group[0].response_id,
        respondent_id=group[0].respondent_id,
        task_id=group[0].task_id,
        engine_id=group[0].engine_id,
        engine_family_id=group[0].engine_family_id,
        engine_fingerprint=group[0].engine_fingerprint,
        _rating_token=calibration._RATING_TOKEN,
    )
    assert_error(
        "duplicate_facets_rating_cell",
        lambda: calibration._build_criterion_design(
            (group[0], duplicate_cell), require_connected=False
        ),
    )

    response_conflict = replace(
        group[1],
        response_id=group[0].response_id,
        respondent_id="conflicting_respondent",
        _rating_token=calibration._RATING_TOKEN,
    )
    assert_error(
        "response_provenance_conflict",
        lambda: calibration._build_criterion_design(
            (group[0], response_conflict), require_connected=False
        ),
    )

    rater_conflict = replace(
        group[1],
        engine_fingerprint=group[0].engine_fingerprint,
        engine_id="conflicting_engine",
        _rating_token=calibration._RATING_TOKEN,
    )
    assert_error(
        "rater_provenance_conflict",
        lambda: calibration._build_criterion_design(
            (group[0], rater_conflict), require_connected=False
        ),
    )


def test_design_rejects_insufficient_support_and_resource_amplification(monkeypatch) -> None:
    """Calibration requires estimable support and bounds the full dense tensor."""
    records = connected_records()
    criterion = records[0].criterion_id
    group = tuple(record for record in records if record.criterion_id == criterion)

    one_response = tuple(
        record for record in group if record.response_id == group[0].response_id
    )
    assert_error(
        "insufficient_facets_responses",
        lambda: calibration._build_criterion_design(
            one_response, require_connected=False
        ),
    )

    one_rater = tuple(
        record
        for record in group
        if record.engine_fingerprint == group[0].engine_fingerprint
    )
    assert_error(
        "insufficient_facets_raters",
        lambda: calibration._build_criterion_design(
            one_rater, require_connected=False
        ),
    )

    terminal = replace(
        group[0],
        status=ObservationStatus.ABSTAINED,
        score_category=None,
        _rating_token=calibration._RATING_TOKEN,
    )
    remaining_same_response = [
        record
        for record in group
        if record.response_id == group[0].response_id and record is not group[0]
    ]
    all_terminal_response = [terminal]
    all_terminal_response.extend(
        replace(
            record,
            status=ObservationStatus.ABSTAINED,
            score_category=None,
            _rating_token=calibration._RATING_TOKEN,
        )
        for record in remaining_same_response
    )
    without_original = tuple(
        record for record in group if record.response_id != group[0].response_id
    )
    assert_error(
        "unobserved_facets_level",
        lambda: calibration._build_criterion_design(
            (*without_original, *all_terminal_response),
            require_connected=False,
        ),
    )

    one_category = tuple(
        replace(
            record,
            score_category=0,
            _rating_token=calibration._RATING_TOKEN,
        )
        for record in group
    )
    assert_error(
        "single_observed_category",
        lambda: calibration._build_criterion_design(
            one_category, require_connected=False
        ),
    )

    monkeypatch.setattr(calibration, "MAX_SCORING_FACETS_CELLS", 1)
    assert_error(
        "facets_cell_budget_exceeded",
        lambda: calibration._build_criterion_design(
            group, require_connected=False
        ),
    )


def test_design_bundle_factory_and_fit_type_guards() -> None:
    """Public design, bundle, and fitting entry points reject unverified values."""
    bundle = build_scoring_facets_calibration_bundle(connected_records())
    with pytest.raises(AssessmentSpecError) as design_error:
        replace(bundle.designs[0])
    assert design_error.value.code == "unverified_facets_design"
    with pytest.raises(AssessmentSpecError) as bundle_error:
        replace(bundle)
    assert bundle_error.value.code == "unverified_facets_bundle"
    assert_error(
        "invalid_facets_design",
        lambda: fit_scoring_facets_design(object()),
    )
    assert_error(
        "invalid_facets_bundle",
        lambda: fit_scoring_facets_bundle(object()),
    )
