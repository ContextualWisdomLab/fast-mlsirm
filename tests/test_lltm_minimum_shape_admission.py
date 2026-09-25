"""Minimum-shape admission regressions for the public LLTM boundary."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import lltm


_MINIMUM_SHAPE_ERROR = "n_persons, n_items and n_basic must be >= 1"


def _unexpected_core() -> object:
    """Fail if compiled-core discovery happens before structural admission."""

    raise AssertionError("compiled core discovered before LLTM minimum-shape admission")


def _minimal_result() -> dict[str, object]:
    """Return one trusted-core result for a 1-person by 1-item LLTM fixture."""

    return {
        "eta": [0.0],
        "intercept": float("nan"),
        "b": [0.0],
        "theta": [0.0],
        "loglik_trace": [0.0],
        "n_iter": 1,
        "converged": True,
        "n_parameters": 1,
        "loglik_rasch": float("nan"),
        "lr_stat": float("nan"),
        "lr_df": 0,
        "lr_p": float("nan"),
    }


def test_zero_numpy_dimensions_fail_before_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact 2-D matrices with a zero Rust dimension are rejected in Python."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    cases = (
        (
            np.empty((0, 1), dtype=np.float64),
            np.ones((1, 1), dtype=np.float64),
        ),
        (
            np.empty((1, 0), dtype=np.float64),
            np.empty((0, 1), dtype=np.float64),
        ),
        (
            np.zeros((1, 1), dtype=np.float64),
            np.empty((1, 0), dtype=np.float64),
        ),
    )

    for responses, q_design in cases:
        with pytest.raises(ValueError, match=_MINIMUM_SHAPE_ERROR):
            lltm.fit_lltm(
                responses,
                q_design,
                fit_intercept=False,
                compute_lr=False,
                max_iter=1,
                tol=0.0,
            )


def test_zero_width_builtin_row_fails_before_dense_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A known-zero built-in row width fails before NumPy matrix construction."""

    def unexpected_materialization(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("dense LLTM matrix materialization ran for zero-width evidence")

    monkeypatch.setattr(lltm, "_materialize_real_matrix", unexpected_materialization)
    with pytest.raises(ValueError, match=_MINIMUM_SHAPE_ERROR):
        lltm._trusted_real_matrix([[]], "q_design")


def test_minimum_valid_shape_still_reaches_rust_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 1x1 response and 1x1 design preserve existing native marshalling."""

    captured: dict[str, tuple[object, ...]] = {}

    class CapturingCore:
        def fit_lltm(self, *args: object) -> dict[str, object]:
            captured["args"] = args
            return _minimal_result()

    monkeypatch.setattr(fitstats, "_core_module", lambda: CapturingCore())
    fitted = lltm.fit_lltm(
        [[1.0]],
        [[1.0]],
        fit_intercept=False,
        compute_lr=False,
        max_iter=1,
        tol=0.0,
    )

    args = captured["args"]
    assert args[3:6] == (1, 1, 1)
    np.testing.assert_array_equal(args[0], np.array([1.0]))
    np.testing.assert_array_equal(args[2], np.array([1.0]))
    assert fitted.n_iter == 1
