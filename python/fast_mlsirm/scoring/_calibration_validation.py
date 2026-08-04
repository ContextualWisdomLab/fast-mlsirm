"""Fail-closed replay for governed scoring-facets calibration artifacts."""

from __future__ import annotations

from typing import Any

from . import calibration as _base
from ._contract_safety import bounded_values
from ._observation_validation import validate_score_observation
from ._validation import assessment_error, strict_boolean
from .execution import (
    EngineDescriptor,
    MAX_REQUEST_CRITERIA,
    ObservationGranularity,
    ScoreObservation,
    ScoringRequest,
    ScoringResult,
)

_ORIGINAL_BUILD_RECORDS = _base.build_scoring_facets_rating_records
_ORIGINAL_BUILD_BUNDLE = _base.build_scoring_facets_calibration_bundle


def _same_value(actual: Any, expected: Any) -> bool:
    """Return whether normalized package values have identical concrete shape."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, tuple):
        return len(actual) == len(expected) and all(
            _same_value(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _validate_result_observations(
    *,
    request: ScoringRequest,
    result: ScoringResult,
    engine: EngineDescriptor,
) -> tuple[ScoreObservation, ...]:
    """Bound and replay exact criterion observations before digest access."""
    observations = bounded_values(
        result.observations,
        "observations",
        minimum=1,
        maximum=MAX_REQUEST_CRITERIA,
    )
    for index, observation in enumerate(observations):
        if type(observation) is not ScoreObservation:
            raise assessment_error(
                "invalid_score_observation",
                f"$.result.observations[{index}]",
                "result observations must be exact ScoreObservation values",
            )
        if observation.criterion_id is None:
            raise assessment_error(
                "missing_observation_criterion",
                f"$.result.observations[{index}].criterion_id",
                "criterion-level observations require a criterion identifier",
            )
        if observation.criterion_id not in request.criterion_ids:
            raise assessment_error(
                "calibration_observation_criterion_mismatch",
                f"$.result.observations[{index}].criterion_id",
                "observation criterion is not declared by the scoring request",
            )
    criterion_ids = tuple(value.criterion_id for value in observations)
    if len(set(criterion_ids)) != len(criterion_ids):
        raise assessment_error(
            "duplicate_observation_criterion",
            "$.result.observations",
            "criterion-level observations must be unique",
        )
    observation_ids = tuple(value.observation_id for value in observations)
    if len(set(observation_ids)) != len(observation_ids):
        raise assessment_error(
            "duplicate_observation_id",
            "$.result.observations",
            "observation identifiers must be unique",
        )
    if result.requested_criterion_ids != request.criterion_ids:
        raise assessment_error(
            "calibration_result_criteria_mismatch",
            "$.result.requested_criterion_ids",
            "result criterion scope does not match the supplied scoring request",
        )
    if tuple(sorted(criterion_ids)) != request.criterion_ids:
        raise assessment_error(
            "calibration_observation_coverage_mismatch",
            "$.result.observations",
            "observations must cover every requested criterion exactly once",
        )

    expected_fields = (
        (
            "request_fingerprint",
            request.request_fingerprint,
            "calibration_observation_request_mismatch",
        ),
        (
            "engine_fingerprint",
            engine.engine_fingerprint,
            "calibration_observation_engine_mismatch",
        ),
        (
            "assessment_fingerprint",
            request.assessment_fingerprint,
            "calibration_observation_assessment_mismatch",
        ),
        (
            "rubric_fingerprint",
            request.rubric_fingerprint,
            "calibration_observation_rubric_mismatch",
        ),
        (
            "construct_id",
            request.construct_id,
            "calibration_observation_construct_mismatch",
        ),
        (
            "granularity",
            request.granularity,
            "calibration_observation_granularity_mismatch",
        ),
    )
    validated: list[ScoreObservation] = []
    for index, observation in enumerate(observations):
        path = f"$.result.observations[{index}]"
        for field_name, expected, code in expected_fields:
            if getattr(observation, field_name) != expected:
                raise assessment_error(
                    code,
                    f"{path}.{field_name}",
                    "observation provenance does not match the supplied execution",
                )
        validated.append(
            validate_score_observation(
                observation,
                request=request,
                engine=engine,
                path=path,
            )
        )
    return tuple(validated)


def build_scoring_facets_rating_records(
    *,
    request: ScoringRequest,
    result: ScoringResult,
    engine: EngineDescriptor,
) -> tuple[_base.ScoringFacetsRatingRecord, ...]:
    """Project one execution only after complete bounded observation replay."""
    if not isinstance(request, ScoringRequest):
        raise assessment_error(
            "invalid_scoring_request",
            "$.request",
            "request must be a ScoringRequest",
        )
    if not isinstance(result, ScoringResult):
        raise assessment_error(
            "invalid_scoring_result",
            "$.result",
            "result must be a ScoringResult",
        )
    if not isinstance(engine, EngineDescriptor):
        raise assessment_error(
            "invalid_engine_descriptor",
            "$.engine",
            "engine must be an EngineDescriptor",
        )
    if request.granularity is not ObservationGranularity.CRITERION_LEVEL:
        raise assessment_error(
            "unsupported_calibration_granularity",
            "$.request.granularity",
            "facets calibration requires criterion-level observations",
        )
    if result.granularity is not ObservationGranularity.CRITERION_LEVEL:
        raise assessment_error(
            "unsupported_calibration_granularity",
            "$.result.granularity",
            "facets calibration requires criterion-level observations",
        )
    if result.request_fingerprint != request.request_fingerprint:
        raise assessment_error(
            "calibration_request_result_mismatch",
            "$.result.request_fingerprint",
            "result is not bound to the supplied scoring request",
        )
    if result.engine_fingerprint != engine.engine_fingerprint:
        raise assessment_error(
            "calibration_engine_result_mismatch",
            "$.result.engine_fingerprint",
            "result is not bound to the supplied engine descriptor",
        )
    observations = _validate_result_observations(
        request=request,
        result=result,
        engine=engine,
    )
    object.__setattr__(result, "observations", observations)
    return _ORIGINAL_BUILD_RECORDS(
        request=request,
        result=result,
        engine=engine,
    )


def _replay_rating_record(
    value: Any,
    *,
    path: str,
) -> _base.ScoringFacetsRatingRecord:
    """Reconstruct one package rating before design assembly or fitting."""
    if type(value) is not _base.ScoringFacetsRatingRecord:
        raise assessment_error(
            "invalid_facets_rating_record",
            path,
            "records must contain exact ScoringFacetsRatingRecord values",
        )
    rebuilt = _base.ScoringFacetsRatingRecord(
        assessment_fingerprint=value.assessment_fingerprint,
        rubric_fingerprint=value.rubric_fingerprint,
        construct_id=value.construct_id,
        request_fingerprint=value.request_fingerprint,
        result_fingerprint=value.result_fingerprint,
        observation_fingerprint=value.observation_fingerprint,
        respondent_id=value.respondent_id,
        response_id=value.response_id,
        response_content_fingerprint=value.response_content_fingerprint,
        task_id=value.task_id,
        occasion_id=value.occasion_id,
        criterion_id=value.criterion_id,
        engine_id=value.engine_id,
        engine_family_id=value.engine_family_id,
        engine_fingerprint=value.engine_fingerprint,
        status=value.status,
        score_category=value.score_category,
        allowed_scores=value.allowed_scores,
        schema_version=value.schema_version,
        _rating_token=_base._RATING_TOKEN,
    )
    if not _same_value(value._content_dict(), rebuilt._content_dict()):
        raise assessment_error(
            "facets_rating_replay_mismatch",
            path,
            "rating record no longer matches its normalized factory contract",
        )
    return value


def build_scoring_facets_calibration_bundle(
    records: Any,
    *,
    require_connected: bool = True,
) -> _base.ScoringFacetsCalibrationBundle:
    """Assemble a bundle only from bounded replayed rating records."""
    raw = bounded_values(
        records,
        "records",
        minimum=1,
        maximum=_base.MAX_SCORING_FACETS_RATINGS,
    )
    replayed = tuple(
        _replay_rating_record(value, path=f"$.records[{index}]")
        for index, value in enumerate(raw)
    )
    return _ORIGINAL_BUILD_BUNDLE(
        replayed,
        require_connected=require_connected,
    )


def _design_fields(design: _base.ScoringFacetsDesign) -> tuple[Any, ...]:
    """Return the complete normalized design identity without derived handles."""
    return (
        design.schema_version,
        design.assessment_fingerprint,
        design.rubric_fingerprint,
        design.construct_id,
        design.occasion_id,
        design.criterion_id,
        design.category_values,
        design.respondent_ids,
        design.task_ids,
        design.response_ids,
        design.response_respondent_ids,
        design.response_task_ids,
        design.response_content_fingerprints,
        design.rater_engine_ids,
        design.rater_engine_family_ids,
        design.rater_engine_fingerprints,
        tuple(record.rating_fingerprint for record in design.rating_records),
        design.respondent_task_connected,
        design.task_rater_connected,
        design.connected,
    )


def _replay_design(value: Any) -> _base.ScoringFacetsDesign:
    """Rebuild one criterion design from replayed records with identification."""
    if type(value) is not _base.ScoringFacetsDesign:
        raise assessment_error(
            "invalid_facets_design",
            "$.design",
            "design must be an exact ScoringFacetsDesign",
        )
    raw = bounded_values(
        value.rating_records,
        "rating_records",
        minimum=1,
        maximum=_base.MAX_SCORING_FACETS_RATINGS,
    )
    records = tuple(
        _replay_rating_record(record, path=f"$.design.rating_records[{index}]")
        for index, record in enumerate(raw)
    )
    rebuilt = _base._build_criterion_design(records, require_connected=True)
    if not _same_value(_design_fields(value), _design_fields(rebuilt)):
        raise assessment_error(
            "facets_design_replay_mismatch",
            "$.design",
            "design no longer matches its replayed rating collection",
        )
    return rebuilt


def _replay_bundle(value: Any) -> _base.ScoringFacetsCalibrationBundle:
    """Rebuild one bundle and reject duplicate or divergent criterion coverage."""
    if type(value) is not _base.ScoringFacetsCalibrationBundle:
        raise assessment_error(
            "invalid_facets_bundle",
            "$.bundle",
            "bundle must be an exact ScoringFacetsCalibrationBundle",
        )
    raw = bounded_values(
        value.designs,
        "designs",
        minimum=1,
        maximum=MAX_REQUEST_CRITERIA,
    )
    for index, design in enumerate(raw):
        if type(design) is not _base.ScoringFacetsDesign:
            raise assessment_error(
                "invalid_facets_design",
                f"$.bundle.designs[{index}]",
                "bundle designs must be exact ScoringFacetsDesign values",
            )
    criterion_ids = tuple(design.criterion_id for design in raw)
    if len(set(criterion_ids)) != len(criterion_ids):
        raise assessment_error(
            "duplicate_facets_bundle_criterion",
            "$.bundle.designs",
            "each criterion may appear only once in a calibration bundle",
        )
    designs = tuple(_replay_design(design) for design in raw)
    all_records = tuple(
        record
        for design in designs
        for record in design.rating_records
    )
    rebuilt = _ORIGINAL_BUILD_BUNDLE(all_records, require_connected=True)
    actual_fields = (
        value.schema_version,
        value.assessment_fingerprint,
        value.rubric_fingerprint,
        value.construct_id,
        value.occasion_id,
        value.category_values,
        value.criterion_ids,
        tuple(design.design_fingerprint for design in value.designs),
    )
    rebuilt_fields = (
        rebuilt.schema_version,
        rebuilt.assessment_fingerprint,
        rebuilt.rubric_fingerprint,
        rebuilt.construct_id,
        rebuilt.occasion_id,
        rebuilt.category_values,
        rebuilt.criterion_ids,
        tuple(design.design_fingerprint for design in rebuilt.designs),
    )
    if not _same_value(actual_fields, rebuilt_fields):
        raise assessment_error(
            "facets_bundle_replay_mismatch",
            "$.bundle",
            "bundle no longer matches its replayed design collection",
        )
    return rebuilt


def fit_scoring_facets_design(
    design: _base.ScoringFacetsDesign,
    *,
    q_theta: int = 41,
    max_iter: int = 500,
    tol: float = 1e-6,
    allow_disconnected: bool = False,
):
    """Fit one replayed identified design through the compiled Rust estimator."""
    strict_boolean(allow_disconnected, "allow_disconnected")
    replayed = _replay_design(design)
    from fast_mlsirm.facets import fit_facets

    return fit_facets(
        **replayed.to_fit_facets_kwargs(),
        q_theta=q_theta,
        max_iter=max_iter,
        tol=tol,
    )


def fit_scoring_facets_bundle(
    bundle: _base.ScoringFacetsCalibrationBundle,
    *,
    q_theta: int = 41,
    max_iter: int = 500,
    tol: float = 1e-6,
    allow_disconnected: bool = False,
) -> dict[str, Any]:
    """Fit every replayed criterion without trusting a mutable result mapping."""
    strict_boolean(allow_disconnected, "allow_disconnected")
    replayed = _replay_bundle(bundle)
    from fast_mlsirm.facets import fit_facets

    return {
        design.criterion_id: fit_facets(
            **design.to_fit_facets_kwargs(),
            q_theta=q_theta,
            max_iter=max_iter,
            tol=tol,
        )
        for design in replayed.designs
    }


def install(module: Any) -> None:
    """Install validated public entry points on the loaded calibration module."""
    module.build_scoring_facets_rating_records = build_scoring_facets_rating_records
    module.build_scoring_facets_calibration_bundle = build_scoring_facets_calibration_bundle
    module.fit_scoring_facets_design = fit_scoring_facets_design
    module.fit_scoring_facets_bundle = fit_scoring_facets_bundle


__all__: list[str] = []
