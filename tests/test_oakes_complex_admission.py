"""Regression coverage for lossless Oakes uncertainty input admission."""

from __future__ import annotations

from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from fast_mlsirm import _core
from fast_mlsirm.inference import oakes_standard_errors


def _result():
    """Return the smallest converged marginal-fit record accepted by the wrapper."""
    return SimpleNamespace(
        model="MLS2PLM",
        optimizer="mmle_marginal_em/rust",
        convergence_status="converged",
        population={},
        params=SimpleNamespace(
            alpha=np.array([0.0], dtype=np.float64),
            b=np.array([0.0], dtype=np.float64),
            zeta=np.zeros((1, 1), dtype=np.float64),
            tau=1.0,
        ),
    )


def _unexpected_oakes_dispatch(*args, **kwargs):
    """Fail if an invalid input reaches Rust-owned Oakes arithmetic."""
    raise AssertionError("invalid Oakes input reached native uncertainty arithmetic")


def test_oakes_rejects_complex_responses_before_native_arithmetic(monkeypatch):
    """Imaginary response components must not be discarded before Oakes SEs."""
    monkeypatch.setattr(_core, "oakes_standard_errors", _unexpected_oakes_dispatch)
    responses = np.array([[0.0 + 1.0j], [1.0 + 0.0j]])

    with pytest.raises(ValueError, match="responses must be real-valued"):
        oakes_standard_errors(_result(), responses, np.array([0], dtype=np.int64))


def test_oakes_rejects_complex_factor_id_before_native_arithmetic(monkeypatch):
    """Imaginary factor assignments must not be discarded before Oakes SEs."""
    monkeypatch.setattr(_core, "oakes_standard_errors", _unexpected_oakes_dispatch)
    responses = np.array([[0.0], [1.0]])
    factor_id = np.array([0.0 + 1.0j])

    with pytest.raises(ValueError, match="factor_id must be real-valued integers"):
        oakes_standard_errors(_result(), responses, factor_id)


@pytest.mark.parametrize("factor_value", [2**63, np.iinfo(np.uint64).max])
def test_oakes_rejects_factor_id_signed64_overflow_before_native_arithmetic(
    monkeypatch, factor_value
):
    """Unsigned factor labels must not wrap while narrowing to Rust int64 IDs."""
    monkeypatch.setattr(_core, "oakes_standard_errors", _unexpected_oakes_dispatch)
    responses = np.array([[0.0], [1.0]])
    factor_id = np.array([factor_value], dtype=np.uint64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="factor_id must fit signed 64-bit integers"):
            oakes_standard_errors(_result(), responses, factor_id)

    assert not [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]


def test_oakes_preserves_signed64_max_until_structural_dimension_validation(monkeypatch):
    """The valid signed-int64 boundary must not be mistaken for narrowing overflow."""
    monkeypatch.setattr(_core, "oakes_standard_errors", _unexpected_oakes_dispatch)
    responses = np.array([[0.0], [1.0]])
    factor_id = np.array([np.iinfo(np.int64).max], dtype=np.int64)

    with pytest.raises(ValueError, match="factor_id implies more dimensions than items"):
        oakes_standard_errors(_result(), responses, factor_id)


def test_oakes_preserves_real_response_missingness_and_factor_assignment(monkeypatch):
    """Binary values, NaN/-1 missingness, and integer factors retain their contract."""
    captured: dict[str, np.ndarray] = {}

    def fake_oakes(*args, **kwargs):
        captured["responses"] = np.asarray(args[0])
        captured["observed"] = np.asarray(args[1])
        captured["factors"] = np.asarray(args[2])
        return {"ok": True}

    monkeypatch.setattr(_core, "oakes_standard_errors", fake_oakes)
    responses = np.array([[0.0], [np.nan], [-1.0], [1.0]], dtype=np.float32)

    result = oakes_standard_errors(_result(), responses, np.array([0], dtype=np.int32))

    assert result == {"ok": True}
    assert np.array_equal(captured["responses"], np.array([0.0, 0.0, 0.0, 1.0]))
    assert np.array_equal(captured["observed"], np.array([True, False, False, True]))
    assert np.array_equal(captured["factors"], np.array([0], dtype=np.int64))