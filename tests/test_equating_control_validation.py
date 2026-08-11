"""Fail-first public validation contracts for observed-score equating controls."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.equating import equate_observed_scores


class _HostileMethod:
    """Semantic control that must never receive caller-controlled stringification."""

    def __str__(self) -> str:
        raise RuntimeError("METHOD_STR_SENTINEL")

    def __repr__(self) -> str:
        raise RuntimeError("METHOD_REPR_SENTINEL")


class _HostileCeiling:
    """Score ceiling that must never receive caller-controlled integer coercion."""

    def __int__(self) -> int:
        raise RuntimeError("CEILING_INT_SENTINEL")

    def __index__(self) -> int:
        raise RuntimeError("CEILING_INDEX_SENTINEL")

    def __repr__(self) -> str:
        raise RuntimeError("CEILING_REPR_SENTINEL")


def _scores() -> tuple[np.ndarray, np.ndarray]:
    """Return tiny otherwise-valid observed-score vectors."""
    return (
        np.array([0.0, 1.0, 2.0], dtype=np.float64),
        np.array([0.0, 1.0, 2.0], dtype=np.float64),
    )


def _result_payload() -> dict[str, object]:
    """Return the minimum successful Rust-shaped equating result payload."""
    return {
        "x_scores": [0.0, 1.0, 2.0],
        "y_equivalents": [0.0, 1.0, 2.0],
        "mu_x": 1.0,
        "sigma_x": 1.0,
        "mu_y": 1.0,
        "sigma_y": 1.0,
        "mu_eq": 1.0,
        "sigma_eq": 1.0,
        "slope": 1.0,
        "intercept": 0.0,
        "n_x": 3,
        "n_y": 3,
    }


def test_equating_rejects_hostile_method_before_rust(monkeypatch) -> None:
    """A non-string method fails closed without representation callbacks or Rust."""
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class Core:
        def equate_observed_scores(self, *args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("RUST_MUST_NOT_RUN")

    monkeypatch.setattr(fitstats, "_core_module", lambda: Core())
    x_scores, y_scores = _scores()

    with pytest.raises(ValueError, match="method"):
        equate_observed_scores(
            x_scores,
            y_scores,
            method=_HostileMethod(),
            k_x=2,
            k_y=2,
        )

    assert calls == []


@pytest.mark.parametrize("ceiling_name", ["k_x", "k_y"])
def test_equating_rejects_hostile_explicit_ceiling_before_rust(
    monkeypatch,
    ceiling_name: str,
) -> None:
    """Explicit ceilings fail closed without conversion callbacks or Rust execution."""
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class Core:
        def equate_observed_scores(self, *args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("RUST_MUST_NOT_RUN")

    monkeypatch.setattr(fitstats, "_core_module", lambda: Core())
    x_scores, y_scores = _scores()
    kwargs: dict[str, object] = {"method": "mean", "k_x": 2, "k_y": 2}
    kwargs[ceiling_name] = _HostileCeiling()

    with pytest.raises(ValueError, match=ceiling_name):
        equate_observed_scores(x_scores, y_scores, **kwargs)

    assert calls == []


def test_equating_preserves_trusted_numpy_integer_ceilings(monkeypatch) -> None:
    """Genuine NumPy integer ceilings preserve the existing accepted Rust call."""
    calls: list[tuple[int, int, str]] = []

    class Core:
        def equate_observed_scores(
            self,
            x_scores,
            y_scores,
            k_x,
            k_y,
            *,
            method,
        ):
            calls.append((k_x, k_y, method))
            return _result_payload()

    monkeypatch.setattr(fitstats, "_core_module", lambda: Core())
    x_scores, y_scores = _scores()

    result = equate_observed_scores(
        x_scores,
        y_scores,
        method="mean",
        k_x=np.int64(2),
        k_y=np.int32(2),
    )

    assert calls == [(2, 2, "mean")]
    assert result.method == "mean"
    assert np.array_equal(result.x_scores, x_scores)
    assert np.array_equal(result.y_equivalents, y_scores)
