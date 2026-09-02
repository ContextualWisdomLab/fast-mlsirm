"""Precision-boundary regressions for Mokken semantic controls."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import mokken


def _lossy_longdouble_half() -> np.longdouble | None:
    """Return one lossy wider-than-f64 value, or prove the platform lacks it."""
    candidate = np.nextafter(np.longdouble("0.5"), np.longdouble("1.0"))
    if np.longdouble(float(candidate)) == candidate:
        assert np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant
        return None
    return candidate


@pytest.mark.parametrize("control_name", ["lower_bound", "alpha"])
@pytest.mark.parametrize("carrier", ["scalar", "zero_dimensional_array"])
def test_mokken_rejects_lossy_extended_precision_control_before_scores(
    monkeypatch: pytest.MonkeyPatch,
    control_name: str,
    carrier: str,
) -> None:
    """Rust f64 controls reject loss where this platform can represent it."""
    candidate = _lossy_longdouble_half()
    if candidate is None:
        return

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


@pytest.mark.parametrize("control_name", ["lower_bound", "alpha"])
def test_mokken_normalizes_overflowing_integer_control_before_scores(
    monkeypatch: pytest.MonkeyPatch,
    control_name: str,
) -> None:
    """Exact integers beyond f64 range fail through package-owned validation."""
    def _unexpected_scores(responses: object) -> tuple[np.ndarray, int, int]:
        raise AssertionError("response traversal executed before control rejection")

    monkeypatch.setattr(mokken, "_validated_scores", _unexpected_scores)

    with pytest.raises(ValueError, match=rf"{control_name} must be finite"):
        mokken.mokken_analysis(object(), **{control_name: 10**400})


@pytest.mark.parametrize(
    ("control_name", "value", "message"),
    [
        ("lower_bound", -0.01, r"lower_bound must be in \[0, 1\)"),
        ("lower_bound", 1.0, r"lower_bound must be in \[0, 1\)"),
        ("alpha", 0.0, r"alpha must be in \(0, 1\)"),
        ("alpha", 1.0, r"alpha must be in \(0, 1\)"),
    ],
)
def test_mokken_rejects_explicit_out_of_range_control_before_scores(
    monkeypatch: pytest.MonkeyPatch,
    control_name: str,
    value: float,
    message: str,
) -> None:
    """Explicit invalid decision controls fail before rejected response work."""
    def _unexpected_scores(responses: object) -> tuple[np.ndarray, int, int]:
        raise AssertionError("response traversal executed before control rejection")

    monkeypatch.setattr(mokken, "_validated_scores", _unexpected_scores)

    with pytest.raises(ValueError, match=message):
        mokken.mokken_analysis(object(), **{control_name: value})
