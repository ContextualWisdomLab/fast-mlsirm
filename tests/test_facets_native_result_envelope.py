"""Fail-closed tests for the Rust-to-Python many-facet result envelope."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

import fast_mlsirm.facets as facets
import fast_mlsirm.fitstats as fitstats


class _FakeCore:
    """Return one controlled native-shaped payload without numerical work."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def fit_facets(self, *args: object, **kwargs: object) -> dict[str, object]:
        return self.payload


def _responses() -> np.ndarray:
    return np.array([[[0], [1]], [[1], [0]]], dtype=np.float64)


def _valid_payload() -> dict[str, object]:
    return {
        "item_difficulty": np.array([-0.25, 0.25], dtype=np.float64),
        "rater_severity": np.array([0.0], dtype=np.float64),
        "thresholds": np.array([0.0], dtype=np.float64),
        "theta": np.array([-0.5, 0.5], dtype=np.float64),
        "loglik_trace": np.array([-5.0, -4.5], dtype=np.float64),
        "n_iter": 1,
        "converged": True,
        "connected": True,
        "n_parameters": 4,
    }


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("item_difficulty", [0.0]),
        ("rater_severity", [0.0, 1.0]),
        ("thresholds", [0.0, 1.0]),
        ("theta", [0.0]),
        ("loglik_trace", [float("nan")]),
        ("n_iter", True),
        ("n_iter", "1"),
        ("converged", 1),
        ("connected", "true"),
        ("n_parameters", 4.5),
    ],
)
def test_fit_facets_rejects_malformed_native_result(
    monkeypatch: pytest.MonkeyPatch, field: str, invalid: object
) -> None:
    payload = _valid_payload()
    payload[field] = invalid
    monkeypatch.setattr(fitstats, "_core_module", lambda: _FakeCore(payload))

    with pytest.raises((TypeError, ValueError), match="native fit_facets result"):
        facets.fit_facets(_responses(), n_cat=2, max_iter=5)


def test_fit_facets_rejects_missing_native_result_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_payload()
    del payload["connected"]
    monkeypatch.setattr(fitstats, "_core_module", lambda: _FakeCore(payload))

    with pytest.raises(ValueError, match="native fit_facets result"):
        facets.fit_facets(_responses(), n_cat=2, max_iter=5)


def test_fit_facets_returns_owned_arrays_from_valid_native_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_payload()
    expected = deepcopy(payload)
    monkeypatch.setattr(fitstats, "_core_module", lambda: _FakeCore(payload))

    result = facets.fit_facets(_responses(), n_cat=2, max_iter=5)

    assert np.array_equal(result.item_difficulty, expected["item_difficulty"])
    assert np.array_equal(result.rater_severity, expected["rater_severity"])
    assert np.array_equal(result.thresholds, expected["thresholds"])
    assert np.array_equal(result.theta, expected["theta"])
    assert np.array_equal(result.loglik_trace, expected["loglik_trace"])
    payload["item_difficulty"][0] = 99.0  # type: ignore[index]
    assert result.item_difficulty[0] == pytest.approx(-0.25)
