"""Pre-copy admission regressions for nested ndarray response rows."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import mokken


def test_nested_ndarray_rejects_unsupported_dtype_before_outer_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inert row dtype metadata must reject before duplicating the outer list."""
    responses = [
        np.array([b"0", b"1"], dtype="S32"),
        np.array([b"1", b"0"], dtype="S32"),
    ]

    def _unexpected_islice(*args: object, **kwargs: object) -> object:
        raise AssertionError("unsupported nested ndarray reached outer-list snapshot")

    monkeypatch.setattr(mokken, "islice", _unexpected_islice)

    with pytest.raises(ValueError, match=r"responses must be a numeric array"):
        mokken._validated_scores(responses)
