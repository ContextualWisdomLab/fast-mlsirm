"""Response-integrity regressions for the public mixture-IRT boundary."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.mixture import fit_mixture


def _unexpected_core() -> object:
    """Fail if native discovery happens before response rejection."""

    raise AssertionError("compiled core discovered before mixture response validation")


@pytest.mark.parametrize(
    "responses",
    [
        np.array([[0.0 + 1.0j, 1.0], [1.0, 0.0]], dtype=np.complex128),
        np.array([[0.0, 1.0], [1.0 + 0.5j, 0.0]], dtype=np.complex64),
    ],
)
def test_fit_mixture_rejects_complex_responses_before_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
    responses: np.ndarray,
) -> None:
    """Imaginary response evidence must never be projected onto real categories."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    with pytest.raises(ValueError, match="responses must be real-valued"):
        fit_mixture(responses)


@pytest.mark.parametrize("bad_value", [float("inf"), float("-inf")])
def test_fit_mixture_rejects_infinite_responses_before_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
    bad_value: float,
) -> None:
    """Infinity is invalid evidence, not an undocumented missing response."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    responses = np.array([[0.0, 1.0], [bad_value, 0.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="responses must be finite where not missing"):
        fit_mixture(responses)


def test_fit_mixture_preserves_nan_missingness_and_real_marshalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Documented NaN missingness survives the hardened Python-to-Rust boundary."""

    captured: dict[str, tuple[object, ...]] = {}

    class _Core:
        @staticmethod
        def fit_mixture(*args: object) -> dict[str, object]:
            captured["args"] = args
            return {
                "model": "rasch",
                "n_classes": 2,
                "a": [1.0, 1.0, 1.0, 1.0],
                "b": [0.0, 0.0, 0.0, 0.0],
                "pi": [0.5, 0.5],
                "class_posterior": [0.5, 0.5, 0.5, 0.5],
                "map_class": [0, 0],
                "theta": [0.0, 0.0],
                "loglik_trace": [0.0],
                "n_iter": 1,
                "converged": True,
                "n_parameters": 5,
            }

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())
    fit_mixture(np.array([[0.0, np.nan], [1.0, 0.0]], dtype=np.float32))

    args = captured["args"]
    np.testing.assert_array_equal(np.asarray(args[0]), np.array([0.0, 0.0, 1.0, 0.0]))
    np.testing.assert_array_equal(np.asarray(args[1]), np.array([True, False, True, True]))
