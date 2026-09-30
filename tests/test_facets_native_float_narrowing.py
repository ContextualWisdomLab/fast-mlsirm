"""Fail-first coverage for lossless native floating result marshalling."""

from __future__ import annotations

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
        "n_iter": 2,
        "converged": True,
        "connected": True,
        "n_parameters": 2,
    }


def _require_extended_longdouble() -> None:
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("platform longdouble does not exceed float64 precision")


def test_fit_facets_rejects_lossy_extended_float_vector_narrowing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_extended_longdouble()
    payload = _valid_payload()
    extended = np.longdouble(1.0) + np.finfo(np.longdouble).eps
    assert np.longdouble(np.float64(extended)) != extended
    payload["item_difficulty"] = np.array(
        [extended, np.longdouble(0.0)], dtype=np.longdouble
    )
    monkeypatch.setattr(fitstats, "_core_module", lambda: _FakeCore(payload))

    with pytest.raises(
        ValueError,
        match=r"native fit_facets result item_difficulty floating values must be exactly representable as float64",
    ):
        facets.fit_facets(_responses(), n_cat=2, max_iter=5)


def test_fit_facets_allows_exact_extended_float_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_extended_longdouble()
    payload = _valid_payload()
    payload["item_difficulty"] = np.array(
        [np.longdouble(-0.25), np.longdouble(0.25)], dtype=np.longdouble
    )
    monkeypatch.setattr(fitstats, "_core_module", lambda: _FakeCore(payload))

    result = facets.fit_facets(_responses(), n_cat=2, max_iter=5)

    assert result.item_difficulty.dtype == np.dtype(np.float64)
    assert result.item_difficulty.tolist() == [-0.25, 0.25]
