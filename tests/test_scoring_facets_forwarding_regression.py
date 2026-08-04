"""Regression coverage for scoring-facets estimator delegation."""

from __future__ import annotations

from pathlib import Path
import runpy

from fast_mlsirm.scoring import (
    build_scoring_facets_calibration_bundle,
    fit_scoring_facets_bundle,
    fit_scoring_facets_design,
)

_HELPERS = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_facets_calibration.py"))
)
connected_records = _HELPERS["connected_records"]


def test_bundle_forwards_tuning_values_and_fresh_response_tensors(monkeypatch) -> None:
    """Bundle fitting forwards its own controls and never reuses dense arrays."""
    bundle = build_scoring_facets_calibration_bundle(connected_records())
    calls: list[dict[str, object]] = []

    def fake_fit_facets(**kwargs):
        calls.append(kwargs)
        return {"shape": kwargs["responses"].shape, "n_cat": kwargs["n_cat"]}

    monkeypatch.setattr("fast_mlsirm.facets.fit_facets", fake_fit_facets)
    fit_scoring_facets_design(
        bundle.designs[0],
        q_theta=21,
        max_iter=77,
        tol=1e-5,
    )
    fitted = fit_scoring_facets_bundle(
        bundle,
        q_theta=15,
        max_iter=66,
        tol=1e-4,
    )

    assert set(fitted) == set(bundle.criterion_ids)
    assert calls[0]["q_theta"] == 21
    assert calls[0]["max_iter"] == 77
    assert calls[0]["tol"] == 1e-5

    bundle_calls = calls[1:]
    assert len(bundle_calls) == len(bundle.designs)
    assert all(call["q_theta"] == 15 for call in bundle_calls)
    assert all(call["max_iter"] == 66 for call in bundle_calls)
    assert all(call["tol"] == 1e-4 for call in bundle_calls)
    assert all(call["n_cat"] == 3 for call in calls)

    response_objects = [id(call["responses"]) for call in calls]
    assert len(set(response_objects)) == len(response_objects)
