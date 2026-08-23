"""Callback-free response-container regressions for Mokken analysis."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import mokken


class _HostileArrayProvider:
    """Array provider whose NumPy callback must never execute during admission."""

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("response array callback executed")


def test_mokken_rejects_array_provider_before_protocol_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Untrusted response providers fail before NumPy or Rust discovery."""

    def _unexpected_core() -> object:
        raise AssertionError("compiled core was discovered")

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    with pytest.raises(ValueError, match="responses must be a numeric array"):
        mokken.mokken_analysis(_HostileArrayProvider())
