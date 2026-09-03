"""Evaluation-grid resource regressions for the public KSIRT boundary."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import ksirt


def test_ksirt_rejects_oversized_evaluation_work_before_value_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Person-item-grid work is bounded before response values or Rust are observed."""
    monkeypatch.setattr(ksirt, "_MAX_KSIRT_EVALUATION_TERMS", 6, raising=False)
    responses = np.array([[np.nan, 0.0], [1.0, 0.0]], dtype=np.float64)

    def _unexpected_asarray(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("response values were materialized")

    def _unexpected_core() -> object:
        raise AssertionError("compiled core was discovered")

    monkeypatch.setattr(ksirt.np, "asarray", _unexpected_asarray)
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    with pytest.raises(
        ValueError,
        match=r"KSIRT evaluation work exceeds 6 person-item-grid terms",
    ):
        ksirt.ksirt_analysis(responses, nevalpoints=2)


def test_ksirt_evaluation_work_budget_accepts_exact_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The configured evaluation ceiling is inclusive rather than off by one."""
    monkeypatch.setattr(ksirt, "_MAX_KSIRT_EVALUATION_TERMS", 8)

    ksirt._require_evaluation_work_budget(2, 2, 2)
