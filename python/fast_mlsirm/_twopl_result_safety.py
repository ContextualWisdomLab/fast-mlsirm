"""Fail-closed admission for Rust compensatory-2PL fit results.

This module performs representation, cardinality, finiteness, and identity checks
only. It does not recompute any psychometric/statistical quantity; the Rust core
remains the sole production numerical owner.
"""

from __future__ import annotations

import math

import numpy as np


_NUMPY_INTEGER_TYPES = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.intp,
    np.longlong,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.uintp,
    np.ulonglong,
)
_NUMPY_FLOAT_TYPES = (np.float16, np.float32, np.float64, np.longdouble)
_RESULT_KEYS = (
    "loading",
    "intercept",
    "theta",
    "n_dims",
    "corr",
    "loglik_trace",
    "n_iter",
    "converged",
    "n_parameters",
    "termination_reason",
    "final_loglik_change",
)


def _result_error(detail: str) -> ValueError:
    return ValueError(f"native fit_2pl result {detail}")


def _native_float_vector(
    result: dict[str, object],
    key: str,
    *,
    exact_length: int | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
) -> np.ndarray:
    """Admit one native real vector and return independent float64 evidence."""

    value = result[key]
    value_type = type(value)
    if value_type is np.ndarray:
        array = value
    elif value_type is list or value_type is tuple:
        admitted_length = len(value)
        if exact_length is not None and admitted_length != exact_length:
            raise _result_error(f"{key} must have length {exact_length}")
        if min_length is not None and admitted_length < min_length:
            raise _result_error(f"{key} must contain at least {min_length} value")
        if max_length is not None and admitted_length > max_length:
            raise _result_error(f"{key} exceeds length limit {max_length}")

        sequence: tuple[object, ...]
        if value_type is list:
            snapshot_limit = (
                exact_length
                if exact_length is not None
                else max_length
                if max_length is not None
                else admitted_length
            )
            sequence = tuple(value[: snapshot_limit + 1])
            if len(sequence) != admitted_length or len(value) != admitted_length:
                raise _result_error(f"{key} changed during validation")
        else:
            sequence = value

        for entry in sequence:
            entry_type = type(entry)
            integer_entry = entry_type is int or entry_type in _NUMPY_INTEGER_TYPES
            floating_entry = entry_type is float or entry_type in _NUMPY_FLOAT_TYPES
            if not integer_entry and not floating_entry:
                raise _result_error(f"{key} must be a real numeric vector")
            if integer_entry:
                integer_value = int(entry)
                try:
                    float_value = float(integer_value)
                except OverflowError:
                    raise _result_error(
                        f"{key} integer values must be exactly representable as float64"
                    ) from None
                if not math.isfinite(float_value) or int(float_value) != integer_value:
                    raise _result_error(
                        f"{key} integer values must be exactly representable as float64"
                    )
        array = np.asarray(sequence)
    else:
        raise _result_error(f"{key} must be a real numeric vector")

    if array.ndim != 1 or np.iscomplexobj(array) or array.dtype.kind not in ("i", "u", "f"):
        raise _result_error(f"{key} must be a real numeric 1-D vector")
    length = int(array.size)
    if exact_length is not None and length != exact_length:
        raise _result_error(f"{key} must have length {exact_length}")
    if min_length is not None and length < min_length:
        raise _result_error(f"{key} must contain at least {min_length} value")
    if max_length is not None and length > max_length:
        raise _result_error(f"{key} exceeds length limit {max_length}")

    snapshot = np.array(array, copy=True)
    if snapshot.ndim != 1 or np.iscomplexobj(snapshot) or snapshot.dtype.kind not in ("i", "u", "f"):
        raise _result_error(f"{key} must be a real numeric 1-D vector")
    snapshot_length = int(snapshot.size)
    if exact_length is not None and snapshot_length != exact_length:
        raise _result_error(f"{key} must have length {exact_length}")
    if min_length is not None and snapshot_length < min_length:
        raise _result_error(f"{key} must contain at least {min_length} value")
    if max_length is not None and snapshot_length > max_length:
        raise _result_error(f"{key} exceeds length limit {max_length}")
    if not np.all(np.isfinite(snapshot)):
        raise _result_error(f"{key} must contain only finite values")

    owned = snapshot.astype(np.float64, copy=False)
    if snapshot.dtype.kind in ("i", "u"):
        with np.errstate(invalid="ignore", over="ignore"):
            recovered = owned.astype(snapshot.dtype, copy=False)
        if not np.array_equal(recovered, snapshot):
            raise _result_error(
                f"{key} integer values must be exactly representable as float64"
            )
    elif snapshot.dtype.kind == "f" and snapshot.dtype.itemsize > np.dtype(np.float64).itemsize:
        recovered = owned.astype(snapshot.dtype, copy=False)
        if not np.array_equal(recovered, snapshot):
            raise _result_error(
                f"{key} floating values must be exactly representable as float64"
            )
    if not np.all(np.isfinite(owned)):
        raise _result_error(f"{key} must contain only finite values")
    return np.array(owned, copy=True)


def _finite_f64_scalar(value: object, key: str) -> float:
    """Admit one exact built-in/trusted NumPy scalar without caller callbacks."""

    value_type = type(value)
    if value_type is int or value_type in _NUMPY_INTEGER_TYPES:
        integer = int(value)
        try:
            numeric = float(integer)
        except OverflowError:
            raise _result_error(f"{key} must be exactly representable as float64") from None
        if not math.isfinite(numeric) or int(numeric) != integer:
            raise _result_error(f"{key} must be exactly representable as float64")
        return numeric
    if value_type is float:
        numeric = value
    elif value_type in _NUMPY_FLOAT_TYPES:
        numeric = float(value)
        if value_type(numeric) != value:
            raise _result_error(f"{key} must be exactly representable as float64")
    else:
        raise _result_error(f"{key} must be a real number")
    if not math.isfinite(numeric):
        raise _result_error(f"{key} must be finite")
    return numeric


def validate_twopl_native_result(
    value: object,
    *,
    n_persons: int,
    n_items: int,
    n_dims: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, bool, int, str, float]:
    """Seal and validate the PyO3 result envelope before public construction."""

    if type(value) is not dict:
        raise _result_error("must be a built-in dict")
    if len(value) != len(_RESULT_KEYS):
        raise _result_error(f"must contain exactly {len(_RESULT_KEYS)} keys")
    if any(type(key) is not str for key in value):
        raise _result_error("keys must be exact strings")

    result = value.copy()
    if len(result) != len(_RESULT_KEYS):
        raise _result_error(f"must contain exactly {len(_RESULT_KEYS)} keys")
    if any(type(key) is not str for key in result):
        raise _result_error("keys must be exact strings")
    missing = [key for key in _RESULT_KEYS if key not in result]
    if missing:
        raise _result_error(f"is missing required key {missing[0]!r}")

    result_n_dims = result["n_dims"]
    if type(result_n_dims) is not int or result_n_dims != n_dims:
        raise _result_error(f"n_dims must equal {n_dims}")

    loading = _native_float_vector(result, "loading", exact_length=n_items * n_dims)
    intercept = _native_float_vector(result, "intercept", exact_length=n_items)
    theta = _native_float_vector(result, "theta", exact_length=n_persons * n_dims)
    corr = _native_float_vector(result, "corr", exact_length=n_dims * n_dims)
    loglik_trace = _native_float_vector(
        result,
        "loglik_trace",
        min_length=1,
        max_length=max_iter + 1,
    )

    n_iter = result["n_iter"]
    if type(n_iter) is not int or not 1 <= n_iter <= max_iter:
        raise _result_error(f"n_iter must be an integer in 1..{max_iter}")
    converged = result["converged"]
    if type(converged) is not bool:
        raise _result_error("converged must be a boolean")
    n_parameters = result["n_parameters"]
    max_parameters = n_items * (n_dims + 1) + n_dims * (n_dims - 1) // 2
    if type(n_parameters) is not int or not 1 <= n_parameters <= max_parameters:
        raise _result_error(f"n_parameters must be an integer in 1..{max_parameters}")
    termination_reason = result["termination_reason"]
    if type(termination_reason) is not str or termination_reason not in {
        "converged",
        "max_iter_reached",
    }:
        raise _result_error("termination_reason must be 'converged' or 'max_iter_reached'")
    if converged != (termination_reason == "converged"):
        raise _result_error("converged must agree with termination_reason")
    final_loglik_change = _finite_f64_scalar(
        result["final_loglik_change"], "final_loglik_change"
    )

    return (
        loading,
        intercept,
        theta,
        corr,
        loglik_trace,
        n_iter,
        converged,
        n_parameters,
        termination_reason,
        final_loglik_change,
    )
