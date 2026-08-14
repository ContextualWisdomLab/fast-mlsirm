"""Direct replay-boundary coverage for scoring-facets calibration artifacts."""

from __future__ import annotations

from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from fast_mlsirm.scoring import (
    AssessmentSpecError,
    EvidenceReference,
    build_scoring_facets_calibration_bundle,
)
import fast_mlsirm.scoring._calibration_validation as validation
import fast_mlsirm.scoring._observation_validation as observation_validation


_BASE = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_facets_calibration.py"))
)


def _execution():
    """Return one complete criterion-level execution fixture."""
    return _BASE["execution"](
        request_id="validation_edge_request",
        response_id="validation_edge_response",
        respondent_id="validation_edge_respondent",
        task_id="validation_edge_task",
        engine=_BASE["automated_engine"](),
        claim_score=1,
        source_score=2,
    )


def _error(action):
    """Return one structured replay error."""
    with pytest.raises(AssessmentSpecError) as captured:
        action()
    return captured.value


def test_same_value_and_observation_budget_boundaries_are_explicit():
    """Replay uses exact concrete shapes and rejects oversized observation tuples."""
    assert validation._same_value(1, True) is False
    request, result, engine = _execution()
    object.__setattr__(
        result,
        "observations",
        result.observations * (validation.MAX_REQUEST_CRITERIA + 1),
    )
    assert _error(
        lambda: validation._validated_result_observations(
            request=request,
            result=result,
            engine=engine,
        )
    ).code == "invalid_observations"


def test_result_and_rating_replay_detect_stale_content():
    """Replayed result and rating identities cannot accept stale in-memory content."""
    request, result, engine = _execution()
    object.__setattr__(result, "schema_version", "9.9")
    assert _error(
        lambda: validation._replay_result(
            request=request,
            result=result,
            engine=engine,
            observations=result.observations,
        )
    ).code == "scoring_result_replay_mismatch"

    record = _BASE["connected_records"]()[0]
    object.__setattr__(record, "_content_dict", lambda: {"tampered": True})
    assert _error(
        lambda: validation._replay_rating_record(record, path="$.record")
    ).code == "facets_rating_replay_mismatch"


def test_design_replay_wraps_nested_factory_errors_and_preserves_type_guards():
    """Design replay distinguishes tuple, cardinality, and nested factory failures."""
    bundle = build_scoring_facets_calibration_bundle(_BASE["connected_records"]())
    design = bundle.designs[0]
    object.__setattr__(design, "rating_records", list(design.rating_records))
    assert _error(
        lambda: validation._replay_design(design, path="$.nested_design")
    ).code == "facets_design_replay_mismatch"

    bundle = build_scoring_facets_calibration_bundle(_BASE["connected_records"]())
    design = bundle.designs[0]
    object.__setattr__(design, "rating_records", ())
    assert _error(
        lambda: validation._replay_design(design, path="$.nested_design")
    ).code == "invalid_rating_records"

    disconnected = build_scoring_facets_calibration_bundle(
        _BASE["disconnected_records"](),
        require_connected=False,
    )
    assert _error(
        lambda: validation._replay_design(
            disconnected.designs[0],
            path="$.nested_design",
        )
    ).code == "unidentified_respondent_task_design"


def test_bundle_replay_rejects_budget_tuple_and_child_type_boundaries(monkeypatch):
    """Bundle replay bounds flattened records and validates exact child containers."""
    bundle = build_scoring_facets_calibration_bundle(_BASE["connected_records"]())
    monkeypatch.setattr(validation._base, "MAX_SCORING_FACETS_RATINGS", 1)
    assert _error(
        lambda: validation._bounded_bundle_records(
            [SimpleNamespace(rating_records=(object(), object()))]
        )
    ).code == "invalid_records"

    object.__setattr__(bundle, "designs", list(bundle.designs))
    assert _error(lambda: validation._replay_bundle(bundle)).code == "facets_bundle_replay_mismatch"

    object.__setattr__(bundle, "designs", ())
    assert _error(lambda: validation._replay_bundle(bundle)).code == "invalid_designs"

    object.__setattr__(bundle, "designs", (object(),))
    assert _error(lambda: validation._replay_bundle(bundle)).code == "invalid_facets_design"


def test_bundle_replay_rejects_duplicate_criteria_and_stale_identity():
    """A bundle cannot duplicate criterion coverage or retain stale metadata."""
    bundle = build_scoring_facets_calibration_bundle(_BASE["connected_records"]())
    object.__setattr__(
        bundle.designs[1],
        "criterion_id",
        bundle.designs[0].criterion_id,
    )
    assert _error(lambda: validation._replay_bundle(bundle)).code == "duplicate_facets_bundle_criterion"

    bundle = build_scoring_facets_calibration_bundle(_BASE["connected_records"]())
    object.__setattr__(bundle, "schema_version", "9.9")
    assert _error(lambda: validation._replay_bundle(bundle)).code == "facets_bundle_replay_mismatch"


def test_observation_replay_covers_nested_type_and_normalization_mismatches():
    """Observation replay validates exact child types and normalized enum values."""
    request, result, engine = _execution()
    assert observation_validation._same_concrete_value(1, True) is False
    assert _error(
        lambda: observation_validation.validate_score_observation(
            object(),
            request=request,
            engine=engine,
            path="$.observation",
        )
    ).code == "invalid_score_observation"

    evidence = EvidenceReference(
        source_id="source_record",
        span_id="source_span",
        content_fingerprint="e" * 64,
    )
    observation = result.observations[0]
    object.__setattr__(observation, "evidence_references", (evidence,))
    object.__setattr__(evidence, "evidence_role", "supporting_evidence")
    assert _error(
        lambda: observation_validation.validate_score_observation(
            observation,
            request=request,
            engine=engine,
            path="$.observation",
        )
    ).code == "evidence_reference_validation_mismatch"

    object.__setattr__(observation, "evidence_references", ())
    object.__setattr__(observation, "status", "scored")
    assert _error(
        lambda: observation_validation.validate_score_observation(
            observation,
            request=request,
            engine=engine,
            path="$.observation",
        )
    ).code == "score_observation_validation_mismatch"
