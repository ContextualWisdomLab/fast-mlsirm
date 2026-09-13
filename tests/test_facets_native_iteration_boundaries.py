"""Boundary evidence for many-facet native iteration metadata."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.facets as facets


def _payload(*, n_iter: int, converged: bool, trace: list[float]) -> dict[str, object]:
    return {
        "item_difficulty": np.array([-0.25, 0.25], dtype=np.float64),
        "rater_severity": np.array([0.0], dtype=np.float64),
        "thresholds": np.array([0.0], dtype=np.float64),
        "theta": np.array([-0.5, 0.5], dtype=np.float64),
        "loglik_trace": np.array(trace, dtype=np.float64),
        "n_iter": n_iter,
        "converged": converged,
        "connected": True,
        "n_parameters": 2,
    }


def _validate(payload: dict[str, object]) -> tuple[object, ...]:
    return facets._validate_native_fit_result(
        payload,
        n_persons=2,
        n_items=2,
        n_raters=1,
        n_cat=2,
        max_iter=5,
    )


def test_native_result_allows_nonconverged_terminal_evaluation_at_max_iter() -> None:
    result = _validate(
        _payload(
            n_iter=5,
            converged=False,
            trace=[-5.0, -4.8, -4.6, -4.4, -4.2, -4.1],
        )
    )

    assert result[5] == 5
    assert result[6] is False
    assert np.asarray(result[4]).tolist() == [-5.0, -4.8, -4.6, -4.4, -4.2, -4.1]


def test_native_result_rejects_iteration_count_above_requested_max() -> None:
    with pytest.raises(
        ValueError,
        match=r"native fit_facets result n_iter must be an integer in 1\.\.5",
    ):
        _validate(
            _payload(
                n_iter=6,
                converged=False,
                trace=[-5.0, -4.8, -4.6, -4.4, -4.2, -4.1],
            )
        )
