"""Trust-boundary regressions for compensatory 2PL native result admission."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np
import pytest

from fast_mlsirm._twopl_result_safety import validate_twopl_native_result
from fast_mlsirm.twopl import fit_2pl


class _HostileNativeResult(Mapping[str, object]):
    """Mapping-shaped provider object whose callbacks must never execute."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getitem__(self, _key: str) -> object:
        self.calls.append("getitem")
        raise AssertionError("native result mapping callback executed")

    def __iter__(self) -> Iterator[str]:
        self.calls.append("iter")
        raise AssertionError("native result iteration callback executed")

    def __len__(self) -> int:
        self.calls.append("len")
        raise AssertionError("native result length callback executed")


class _HostileInt(int):
    """Integer subclass whose conversion protocol must not be invoked."""

    calls = 0

    def __int__(self) -> int:
        type(self).calls += 1
        raise AssertionError("native scalar conversion callback executed")


class _Core:
    def __init__(self, result: object) -> None:
        self.result = result

    def fit_2pl(self, *_args: object) -> object:
        return self.result


def _valid_native_result() -> dict[str, object]:
    """Return the exact built-in carrier emitted by a one-dimensional fake core."""

    return {
        "loading": np.array([1.0, 1.25], dtype=np.float64),
        "intercept": np.array([-0.25, 0.25], dtype=np.float64),
        "theta": np.array([-0.5, 0.5], dtype=np.float64),
        "n_dims": 1,
        "corr": np.array([1.0], dtype=np.float64),
        "loglik_trace": np.array([-2.0, -1.5], dtype=np.float64),
        "n_iter": 1,
        "converged": False,
        "n_parameters": 4,
        "termination_reason": "max_iter_reached",
        "final_loglik_change": 0.5,
    }


def _validate(result: object) -> tuple[object, ...]:
    """Apply the two-person/two-item/one-dimension native result contract."""

    return validate_twopl_native_result(
        result,
        n_persons=2,
        n_items=2,
        n_dims=1,
        max_iter=1,
    )


def test_fit_2pl_rejects_non_builtin_native_result_without_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Foreign mapping protocols cannot run at the Rust-result trust boundary."""

    result = _HostileNativeResult()
    import fast_mlsirm.fitstats as fitstats

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core(result))

    with pytest.raises(ValueError, match="native fit_2pl result must be a built-in dict"):
        fit_2pl(np.array([[0.0, 1.0], [1.0, 0.0]]), max_iter=1)

    assert result.calls == []


def test_native_result_matches_real_binding_dimension_field() -> None:
    """The PyO3 result's required n_dims field is admitted and identity checked."""

    admitted = _validate(_valid_native_result())
    assert admitted[0].tolist() == [1.0, 1.25]

    result = _valid_native_result()
    result["n_dims"] = 2
    with pytest.raises(ValueError, match="native fit_2pl result n_dims must equal 1"):
        _validate(result)


def test_native_result_arrays_are_cardinality_bound_and_package_owned() -> None:
    """Public vectors are independent snapshots with model-derived cardinalities."""

    result = _valid_native_result()
    loading_source = result["loading"]
    assert type(loading_source) is np.ndarray

    admitted = _validate(result)
    loading = admitted[0]
    assert type(loading) is np.ndarray
    assert loading.tolist() == [1.0, 1.25]

    loading_source[0] = 99.0
    assert loading.tolist() == [1.0, 1.25]

    result = _valid_native_result()
    result["loading"] = np.array([1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="native fit_2pl result loading must have length 2"):
        _validate(result)


def test_native_result_rejects_nonfinite_and_lossy_float64_vectors() -> None:
    """Published binary64 vectors preserve finite native numerical identity."""

    result = _valid_native_result()
    result["theta"] = np.array([0.0, np.inf], dtype=np.float64)
    with pytest.raises(ValueError, match="native fit_2pl result theta must contain only finite"):
        _validate(result)

    result = _valid_native_result()
    result["corr"] = np.array([2**53 + 1], dtype=np.uint64)
    with pytest.raises(
        ValueError,
        match="native fit_2pl result corr integer values must be exactly representable as float64",
    ):
        _validate(result)


def test_native_result_rejects_extended_float_rounding_when_platform_has_it() -> None:
    """Extended native floats cannot be silently rounded into public binary64 evidence."""

    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("platform longdouble is not wider than float64")

    extra = np.longdouble(1.0) + np.finfo(np.longdouble).eps
    assert np.longdouble(float(extra)) != extra
    result = _valid_native_result()
    result["corr"] = np.array([extra], dtype=np.longdouble)
    with pytest.raises(
        ValueError,
        match="native fit_2pl result corr floating values must be exactly representable as float64",
    ):
        _validate(result)


def test_native_result_rejects_scalar_subclass_without_conversion_callback() -> None:
    """Metadata type identity wins before caller-defined integer conversion."""

    result = _valid_native_result()
    result["n_iter"] = _HostileInt(1)
    _HostileInt.calls = 0

    with pytest.raises(ValueError, match="native fit_2pl result n_iter must be an integer"):
        _validate(result)

    assert _HostileInt.calls == 0
