"""Resource regressions for KSIRT response admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import ksirt


def test_ksirt_rejects_oversized_response_before_dense_conversion_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Logical response size is bounded before NumPy float64 materialization."""
    monkeypatch.setattr(ksirt, "_MAX_KSIRT_RESPONSE_CELLS", 2, raising=False)
    responses = np.broadcast_to(np.array([[0.0]], dtype=np.float32), (3, 1))

    def _unexpected_asarray(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("NumPy materialization occurred")

    def _unexpected_core() -> object:
        raise AssertionError("compiled core was discovered")

    monkeypatch.setattr(ksirt.np, "asarray", _unexpected_asarray)
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    with pytest.raises(ValueError, match=r"responses exceed 2 logical cells"):
        ksirt.ksirt_analysis(responses)


def test_ksirt_preserves_valid_response_at_logical_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid 2x1 response matrix at the ceiling still reaches Rust unchanged."""
    monkeypatch.setattr(ksirt, "_MAX_KSIRT_RESPONSE_CELLS", 2, raising=False)
    captured: dict[str, object] = {}

    class _Core:
        def ksirt_occ(
            self,
            responses: np.ndarray,
            n_persons: int,
            n_items: int,
            kernel: str,
            nevalpoints: int,
            bandwidth: list[float] | None,
        ) -> dict[str, object]:
            captured.update(
                responses=responses.copy(),
                n_persons=n_persons,
                n_items=n_items,
                kernel=kernel,
                nevalpoints=nevalpoints,
                bandwidth=bandwidth,
            )
            return {
                "theta": [-0.5, 0.5],
                "grid": [-1.0, 0.0, 1.0],
                "bandwidth": [0.5],
                "options": [[0.0, 1.0]],
                "occ": [[0.8, 0.5, 0.2, 0.2, 0.5, 0.8]],
                "expected": [[0.2, 0.5, 0.8]],
                "expected_total": [0.2, 0.5, 0.8],
            }

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())

    result = ksirt.ksirt_analysis(
        [[np.int16(0)], [np.float32(1.0)]],
        nevalpoints=3,
        bandwidth=(np.float32(0.5),),
    )

    assert captured["n_persons"] == 2
    assert captured["n_items"] == 1
    assert isinstance(captured["responses"], np.ndarray)
    assert captured["responses"].tolist() == [0.0, 1.0]
    assert result.grid.shape == (3,)
