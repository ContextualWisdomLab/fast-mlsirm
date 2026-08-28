"""Regression tests for the subscore Rust-result binding boundary."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm.subscores import (
    SubscoreResult,
    _native_float_scalar,
    _native_float_vector,
    subscore_analysis,
)


class _HostileArray:
    """Array provider that must never run during native-result admission."""

    calls = 0

    def __len__(self) -> int:
        return 2

    def __array__(self, dtype=None, copy=None):  # noqa: ANN001
        type(self).calls += 1
        raise AssertionError("caller/native conversion callback executed")


class _FakeCore:
    def __init__(self, result: object) -> None:
        self._result = result

    def subscore_analysis(self, *_args: object) -> object:
        return self._result


def _inputs() -> tuple[np.ndarray, np.ndarray]:
    responses = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    groups = np.array([0, 0, 1, 1], dtype=np.int64)
    return responses, groups


def _valid_native_result() -> dict[str, object]:
    return {
        "alpha": [0.8, 0.9],
        "alpha_total": 0.85,
        "corr": [1.0, 0.2, 0.7, 0.2, 1.0, 0.8, 0.7, 0.8, 1.0],
        "disattenuated_corr": [float("nan"), 0.3, 0.3, float("nan")],
        "prmse_s": [0.8, 0.9],
        "prmse_x": [0.6, 0.7],
        "prmse_sx": [0.81, 0.91],
        "tau": [0.1, 0.2],
        "beta": [0.3, 0.4],
        "gamma": [0.5, 0.6],
        "added_value_s": [True, True],
        "added_value_sx": [False, False],
        "observed": [1.0, 1.0, 1.0, 1.0, 2.0, 0.0],
        "total": [2.0, 2.0, 2.0],
        "subscore_s": [0.9, 1.1, 1.0, 1.0, 1.8, 0.2],
        "subscore_x": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "subscore_sx": [0.95, 1.05, 1.0, 1.0, 1.7, 0.3],
    }


def _self_consistent_k3_native_result() -> dict[str, object]:
    k = 3
    n_persons = 3
    diagonal = {0, 4, 8}
    return {
        "alpha": [0.8, 0.9, 0.85],
        "alpha_total": 0.86,
        "corr": [1.0] * ((k + 1) * (k + 1)),
        "disattenuated_corr": [
            float("nan") if index in diagonal else 0.2 for index in range(k * k)
        ],
        "prmse_s": [0.8, 0.9, 0.85],
        "prmse_x": [0.6, 0.7, 0.65],
        "prmse_sx": [0.81, 0.91, 0.86],
        "tau": [0.1, 0.2, 0.15],
        "beta": [0.3, 0.4, 0.35],
        "gamma": [0.5, 0.6, 0.55],
        "added_value_s": [True, True, True],
        "added_value_sx": [False, False, False],
        "observed": [1.0] * (n_persons * k),
        "total": [3.0] * n_persons,
        "subscore_s": [1.0] * (n_persons * k),
        "subscore_x": [1.0] * (n_persons * k),
        "subscore_sx": [1.0] * (n_persons * k),
    }


def _run_with_result(result: object) -> SubscoreResult:
    responses, groups = _inputs()
    with patch("fast_mlsirm.fitstats._core_module", return_value=_FakeCore(result)):
        return subscore_analysis(responses, groups)


def test_subscore_analysis_rejects_hostile_native_array_before_callback() -> None:
    result = _valid_native_result()
    hostile = _HostileArray()
    _HostileArray.calls = 0
    result["alpha"] = hostile

    with pytest.raises(RuntimeError, match="invalid subscore Rust result payload"):
        _run_with_result(result)

    assert _HostileArray.calls == 0


def test_subscore_analysis_rejects_native_cardinality_mismatch_before_reshape() -> None:
    result = _valid_native_result()
    result["corr"] = [1.0] * 8

    with pytest.raises(RuntimeError, match="invalid subscore Rust result payload"):
        _run_with_result(result)


def test_subscore_analysis_rejects_asymmetric_native_correlation_matrix() -> None:
    result = _valid_native_result()
    corr = list(result["corr"])
    corr[1] = 0.25
    result["corr"] = corr

    with pytest.raises(RuntimeError, match="invalid subscore Rust result payload"):
        _run_with_result(result)


def test_subscore_analysis_rejects_asymmetric_disattenuated_correlation() -> None:
    result = _valid_native_result()
    disattenuated = list(result["disattenuated_corr"])
    disattenuated[1] = 0.4
    result["disattenuated_corr"] = disattenuated

    with pytest.raises(RuntimeError, match="invalid subscore Rust result payload"):
        _run_with_result(result)


def test_subscore_analysis_rejects_nonfinite_native_values() -> None:
    result = _valid_native_result()
    result["alpha"] = [float("nan"), 0.9]

    with pytest.raises(RuntimeError, match="invalid subscore Rust result payload"):
        _run_with_result(result)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("alpha", [0.0, 0.9]),
        ("alpha_total", 1.01),
        ("prmse_s", [-0.01, 0.9]),
        ("prmse_x", [0.6, 1.000000002]),
        ("prmse_sx", [0.81, 1.000000002]),
    ],
)
def test_subscore_analysis_rejects_native_values_outside_rust_domains(
    field: str, invalid_value: object
) -> None:
    result = _valid_native_result()
    result[field] = invalid_value

    with pytest.raises(RuntimeError, match="invalid subscore Rust result payload"):
        _run_with_result(result)


def test_subscore_analysis_rejects_native_subscale_count_not_in_group_evidence() -> None:
    with pytest.raises(RuntimeError, match="invalid subscore Rust result payload"):
        _run_with_result(_self_consistent_k3_native_result())


def test_subscore_analysis_rejects_prmse_s_not_equal_to_alpha() -> None:
    result = _valid_native_result()
    result["prmse_s"] = [0.79, 0.9]

    with pytest.raises(RuntimeError, match="invalid subscore Rust result payload"):
        _run_with_result(result)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("added_value_s", [False, True]),
        ("added_value_sx", [True, False]),
    ],
)
def test_subscore_analysis_rejects_decisions_inconsistent_with_prmse(
    field: str, invalid_value: object
) -> None:
    result = _valid_native_result()
    result[field] = invalid_value

    with pytest.raises(RuntimeError, match="invalid subscore Rust result payload"):
        _run_with_result(result)


def test_native_float_admission_avoids_numpy_scalar_ufunc_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("NumPy scalar ufunc dispatch executed")

    monkeypatch.setattr("fast_mlsirm.subscores.np.isnan", forbidden)
    monkeypatch.setattr("fast_mlsirm.subscores.np.isfinite", forbidden)

    assert _native_float_vector([0.25, 0.75]) == [0.25, 0.75]
    assert _native_float_scalar(0.5) == 0.5


def test_subscore_analysis_accepts_current_rust_shaped_payload() -> None:
    result = _run_with_result(_valid_native_result())

    assert result.alpha.shape == (2,)
    assert result.corr.shape == (3, 3)
    assert result.disattenuated_corr.shape == (2, 2)
    assert np.isnan(np.diag(result.disattenuated_corr)).all()
    assert result.observed.shape == (3, 2)
    assert result.subscore_sx.shape == (3, 2)
    assert result.total.shape == (3,)
