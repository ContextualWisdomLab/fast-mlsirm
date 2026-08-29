"""Resource-admission regression tests for confirmatory loading patterns."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.models as models
from fast_mlsirm.irt_contract import MAX_IRT_RESPONSE_CELLS


_RESOURCE_ERROR = "confirmatory loading_pattern exceeds the supported cell budget"


def _unexpected_full_scan(*_args: object, **_kwargs: object) -> object:
    raise AssertionError("oversized loading_pattern reached an O(n) value scan")


def _unexpected_scalar_normalization(_value: object) -> int:
    raise AssertionError("oversized loading_pattern reached scalar normalization")


def test_broadcast_ndarray_is_rejected_before_full_value_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    huge = np.broadcast_to(
        np.array([[1.0]], dtype=np.float64),
        (1, MAX_IRT_RESPONSE_CELLS + 1),
    )
    monkeypatch.setattr(models.np, "isfinite", _unexpected_full_scan)

    with pytest.raises(ValueError, match=_RESOURCE_ERROR):
        models.ConfirmatoryModel(huge)


def test_exact_sequence_is_rejected_before_scalar_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models, "_MAX_CONFIRMATORY_LOADING_CELLS", 4, raising=False)
    monkeypatch.setattr(models, "_confirmatory_scalar", _unexpected_scalar_normalization)

    with pytest.raises(ValueError, match=_RESOURCE_ERROR):
        models.ConfirmatoryModel([[1, 0, 1], [0, 1, 0]])


def test_replay_rejects_rebound_oversize_before_binary_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = models.ConfirmatoryModel(np.array([[1]], dtype=np.int64))
    rebound = np.ones((2, 3), dtype=np.int64)
    rebound.setflags(write=False)
    object.__setattr__(model, "loading_pattern", rebound)

    monkeypatch.setattr(models, "_MAX_CONFIRMATORY_LOADING_CELLS", 4, raising=False)
    monkeypatch.setattr(models.np, "all", _unexpected_full_scan)

    with pytest.raises(ValueError, match="confirmatory model loading_pattern is not canonical"):
        _ = model.n_dims


def test_confirmatory_loading_cell_budget_boundary_remains_admissible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models, "_MAX_CONFIRMATORY_LOADING_CELLS", 4, raising=False)

    model = models.ConfirmatoryModel([[1, 0], [0, 1]])

    assert model.n_dims == 2
    assert np.array_equal(model.loading_pattern, np.array([[1, 0], [0, 1]], dtype=np.int64))
