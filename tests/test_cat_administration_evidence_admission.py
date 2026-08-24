"""Trust-boundary regressions for dichotomous CAT administration evidence."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm
from fast_mlsirm.cat import estimate_ability_eap
from fast_mlsirm.types import MLSIRMParams


class _ArrayProvider:
    def __init__(self, payload: np.ndarray) -> None:
        self.payload = payload
        self.calls = 0

    def __array__(self, dtype: object | None = None) -> np.ndarray:
        self.calls += 1
        return np.asarray(self.payload, dtype=dtype)


class _CoreSentinel:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"compiled CAT core must not be reached for invalid evidence: {name}")


def _bank() -> MLSIRMParams:
    return MLSIRMParams(
        theta=np.zeros((1, 1), dtype=np.float64),
        alpha=np.zeros(2, dtype=np.float64),
        b=np.zeros(2, dtype=np.float64),
        xi=np.zeros((1, 1), dtype=np.float64),
        zeta=np.zeros((2, 1), dtype=np.float64),
        tau=0.0,
    )


def test_cat_rejects_administered_array_provider_before_callback_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _ArrayProvider(np.array([0], dtype=np.int64))
    monkeypatch.setattr(fast_mlsirm, "_core", _CoreSentinel(), raising=False)

    with pytest.raises(ValueError, match="administered item indices"):
        estimate_ability_eap(
            _bank(),
            np.zeros(2, dtype=np.int64),
            provider,  # type: ignore[arg-type]
            [1],
        )

    assert provider.calls == 0


def test_cat_rejects_response_array_provider_before_callback_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _ArrayProvider(np.array([1.0], dtype=np.float64))
    monkeypatch.setattr(fast_mlsirm, "_core", _CoreSentinel(), raising=False)

    with pytest.raises(ValueError, match="responses must be real-valued"):
        estimate_ability_eap(
            _bank(),
            np.zeros(2, dtype=np.int64),
            [0],
            provider,  # type: ignore[arg-type]
        )

    assert provider.calls == 0
