"""Precision-boundary regressions for Mokken semantic controls."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import mokken


def _lossy_longdouble_half() -> np.longdouble:
    """Return an in-domain longdouble that cannot round-trip through float64."""
    candidate = np.nextafter(np.longdouble("0.5"), np.longdouble("1.0"))
    if np.longdouble(float(candidate)) == candidate:
        pytest.skip("platform longdouble does not exceed float64 precision")
    return candidate


@pytest.mark.parametrize("control_name", ["lower_bound", "alpha"])
@pytest.mark.parametrize("carrier", ["scalar", "zero_dimensional_array"])
def test_mokken_rejects_lossy_extended_precision_control_before_scores(
    monkeypatch: pytest.MonkeyPatch,
    control_name: str,
    carrier: str,
) -> None:
    """Rust f64 controls must not be silently rounded before score traversal."""
    candidate = _lossy_longdouble_half()
    control: object
    if carrier == "scalar":
        control = candidate
    else:
        control = np.array(candidate, dtype=np.longdouble)

    def _unexpected_scores(responses: object) -> tuple[np.ndarray, int, int]:
        raise AssertionError("response traversal executed before control rejection")

    monkeypatch.setattr(mokken, "_validated_scores", _unexpected_scores)
    kwargs = {control_name: control}

    with pytest.raises(
        ValueError,
        match=rf"{control_name} must be exactly representable as float64",
    ):
        mokken.mokken_analysis(object(), **kwargs)


@pytest.mark.parametrize("carrier", ["scalar", "zero_dimensional_array"])
def test_mokken_keeps_exact_extended_precision_control_admissible(carrier: str) -> None:
    """Exactly representable trusted extended controls retain existing semantics."""
    exact = np.longdouble("0.5")
    control: object
    if carrier == "scalar":
        control = exact
    else:
        control = np.array(exact, dtype=np.longdouble)

    assert mokken._real_control("lower_bound", control) == 0.5
