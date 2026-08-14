"""Coverage for the legacy calibration module retained behind the replay wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import runpy
import sys

import pytest

from fast_mlsirm.scoring import AssessmentSpecError, ObservationStatus, StaticFixtureEngine


_BASE = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_facets_calibration.py"))
)
_LEGACY_MODULE_NAME = "fast_mlsirm.scoring._legacy_calibration_coverage"


def _legacy_module():
    """Load the pre-replay calibration source without changing the public module."""
    cached = sys.modules.get(_LEGACY_MODULE_NAME)
    if cached is not None:
        return cached
    source = Path(__file__).parents[1] / "python/fast_mlsirm/scoring/calibration.py"
    spec = importlib.util.spec_from_file_location(_LEGACY_MODULE_NAME, source)
    if spec is None or spec.loader is None:
        raise AssertionError("legacy calibration module spec is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_LEGACY_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _error(action):
    """Return one package-owned calibration error."""
    with pytest.raises(AssessmentSpecError) as captured:
        action()
    return captured.value


def _execution():
    """Return one complete shared execution fixture."""
    return _BASE["execution"](
        request_id="legacy_coverage_request",
        response_id="legacy_coverage_response",
        respondent_id="legacy_coverage_respondent",
        task_id="legacy_coverage_task",
        engine=_BASE["automated_engine"](),
        claim_score=1,
        source_score=2,
    )


def _legacy_records(legacy, records=None):
    """Re-seal fixture records with the isolated legacy module token."""
    fields = (
        "assessment_fingerprint",
        "rubric_fingerprint",
        "construct_id",
        "request_fingerprint",
        "result_fingerprint",
        "observation_fingerprint",
        "respondent_id",
        "response_id",
        "response_content_fingerprint",
        "task_id",
        "task_revision_fingerprint",
        "task_family_id",
        "occasion_id",
        "criterion_id",
        "engine_id",
        "engine_family_id",
        "engine_fingerprint",
        "status",
        "score_category",
        "allowed_scores",
        "schema_version",
    )
    source_records = _BASE["connected_records"]() if records is None else records
    return tuple(
        legacy.ScoringFacetsRatingRecord(
            **{field: getattr(record, field) for field in fields},
            _rating_token=legacy._RATING_TOKEN,
        )
        for record in source_records
    )


def test_legacy_observation_replay_and_projection_guards() -> None:
    """The superseded module still has explicit fail-closed guards for compatibility."""
    legacy = _legacy_module()
    request, result, engine = _execution()

    object.__setattr__(result, "observations", (object(),))
    assert _error(
        lambda: legacy._validated_result_observations(request=request, result=result)
    ).code == "invalid_score_observation"

    request, result, _engine = _execution()
    object.__setattr__(result.observations[0], "criterion_id", None)
    assert _error(
        lambda: legacy._validated_result_observations(request=request, result=result)
    ).code == "missing_observation_criterion"

    request, result, _engine = _execution()
    object.__setattr__(result.observations[0], "criterion_id", "undeclared_criterion")
    assert _error(
        lambda: legacy._validated_result_observations(request=request, result=result)
    ).code == "calibration_observation_criterion_mismatch"

    request, result, _engine = _execution()
    object.__setattr__(result, "observations", (result.observations[0],) * 2)
    assert _error(
        lambda: legacy._validated_result_observations(request=request, result=result)
    ).code == "duplicate_observation_criterion"

    request, result, _engine = _execution()
    object.__setattr__(result.observations[1], "observation_id", result.observations[0].observation_id)
    assert _error(
        lambda: legacy._validated_result_observations(request=request, result=result)
    ).code == "duplicate_observation_id"

    request, result, _engine = _execution()
    object.__setattr__(result, "requested_criterion_ids", ("claim_support",))
    assert _error(
        lambda: legacy._validated_result_observations(request=request, result=result)
    ).code == "calibration_result_criteria_mismatch"

    request, result, _engine = _execution()
    object.__setattr__(result, "observations", (result.observations[0],))
    assert _error(
        lambda: legacy._validated_result_observations(request=request, result=result)
    ).code == "calibration_observation_coverage_mismatch"

    request, result, engine = _execution()
    assert _error(
        lambda: legacy.build_scoring_facets_rating_records(
            request=object(), result=result, engine=engine
        )
    ).code == "invalid_scoring_request"
    assert _error(
        lambda: legacy.build_scoring_facets_rating_records(
            request=request, result=object(), engine=engine
        )
    ).code == "invalid_scoring_result"
    assert _error(
        lambda: legacy.build_scoring_facets_rating_records(
            request=request, result=result, engine=object()
        )
    ).code == "invalid_engine_descriptor"

    holistic = _BASE["holistic_request"](
        request_id="legacy_holistic_request",
        response_id="legacy_holistic_response",
        respondent_id="legacy_holistic_respondent",
        task_id="legacy_holistic_task",
        task_revision_fingerprint="8" * 64,
    )
    holistic_result = StaticFixtureEngine(
        descriptor=engine,
        outcomes=(
            _BASE["FixtureOutcome"](
                criterion_id=None,
                status=ObservationStatus.SCORED,
                score_category=1,
            ),
        ),
    ).score(holistic)
    assert _error(
        lambda: legacy.build_scoring_facets_rating_records(
            request=holistic, result=holistic_result, engine=engine
        )
    ).code == "unsupported_calibration_granularity"
    assert _error(
        lambda: legacy.build_scoring_facets_rating_records(
            request=request, result=holistic_result, engine=engine
        )
    ).code == "unsupported_calibration_granularity"

    other_request, other_result, _ = _BASE["execution"](
        request_id="legacy_other_request",
        response_id="legacy_other_response",
        respondent_id="legacy_other_respondent",
        task_id="legacy_other_task",
        task_revision_fingerprint="6" * 64,
        engine=engine,
        claim_score=1,
        source_score=2,
    )
    assert _error(
        lambda: legacy.build_scoring_facets_rating_records(
            request=request, result=other_result, engine=engine
        )
    ).code == "calibration_request_result_mismatch"
    assert _error(
        lambda: legacy.build_scoring_facets_rating_records(
            request=other_request,
            result=other_result,
            engine=_BASE["human_engine"](),
        )
    ).code == "calibration_engine_result_mismatch"

    request, result, engine = _execution()
    object.__setattr__(result.observations[0], "request_fingerprint", "f" * 64)
    assert _error(
        lambda: legacy.build_scoring_facets_rating_records(
            request=request, result=result, engine=engine
        )
    ).code == "calibration_observation_request_mismatch"


def test_legacy_bundle_and_fit_guards_delegate_once(monkeypatch) -> None:
    """Legacy fitting remains covered while the public API uses replay validation."""
    legacy = _legacy_module()
    records = _legacy_records(legacy)
    assert _error(
        lambda: legacy.build_scoring_facets_calibration_bundle((*records, object()))
    ).code == "invalid_facets_rating_record"
    bundle = legacy.build_scoring_facets_calibration_bundle(records)

    disconnected_records = _legacy_records(
        legacy,
        _BASE["disconnected_records"](),
    )
    disconnected = legacy.build_scoring_facets_calibration_bundle(
        disconnected_records,
        require_connected=False,
    )
    assert _error(lambda: legacy.fit_scoring_facets_design(object())).code == "invalid_facets_design"
    assert _error(
        lambda: legacy.fit_scoring_facets_design(disconnected.designs[0])
    ).code == "unidentified_respondent_task_design"
    design = bundle.designs[0]
    object.__setattr__(design, "task_rater_connected", False)
    assert _error(
        lambda: legacy.fit_scoring_facets_design(design)
    ).code == "disconnected_task_rater_design"
    assert _error(lambda: legacy.fit_scoring_facets_bundle(object())).code == "invalid_facets_bundle"

    object.__setattr__(design, "task_rater_connected", True)
    monkeypatch.setattr("fast_mlsirm.facets.fit_facets", lambda **kwargs: kwargs["responses"].shape)
    assert legacy.fit_scoring_facets_design(bundle.designs[0]) == (2, 2, 2)
    fitted = legacy.fit_scoring_facets_bundle(bundle)
    assert set(fitted) == set(bundle.criterion_ids)
