"""Callback-safety regressions for public Benjamini-Hochberg admission."""

from __future__ import annotations

import fast_mlsirm
import fast_mlsirm.fitstats as fitstats_module
import numpy as np
import pytest


class _HostileArray:
    """Array provider that records caller-dispatchable conversion."""

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def __array__(self, *_args: object, **_kwargs: object) -> np.ndarray:
        """Fail if BH admission dispatches caller array conversion."""
        self.calls.append("array")
        raise AssertionError("caller array callback executed")


class _HostileFloat(float):
    """Floating subclass that records caller-dispatchable conversion."""

    def __new__(cls, value: float, calls: list[str]) -> "_HostileFloat":
        instance = super().__new__(cls, value)
        instance.calls = calls
        return instance

    def __float__(self) -> float:
        """Fail if BH admission dispatches caller floating conversion."""
        self.calls.append("float")
        raise AssertionError("caller floating callback executed")


def _bomb_core() -> object:
    """Fail if invalid evidence/control reaches compiled-core discovery."""
    raise AssertionError("unsafe BH admission reached native-core discovery")


def test_bh_rejects_q_before_caller_evidence_and_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A callback-bearing FDR control fails before p-value work and Rust."""
    calls: list[str] = []
    monkeypatch.setattr(fitstats_module, "_core_module", _bomb_core)

    with pytest.raises(ValueError, match="q"):
        fitstats_module.benjamini_hochberg(
            _HostileArray(calls),
            q=_HostileFloat(0.05, calls),
        )

    assert calls == []


def test_bh_rejects_array_provider_before_callback_and_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Callback-bearing p-value carriers fail before array protocols and Rust."""
    calls: list[str] = []
    monkeypatch.setattr(fitstats_module, "_core_module", _bomb_core)

    with pytest.raises(ValueError, match="p_values"):
        fitstats_module.benjamini_hochberg(_HostileArray(calls), q=0.05)

    assert calls == []


@pytest.mark.parametrize(
    "p_values",
    (
        np.array([0.01, np.nan], dtype=np.float64),
        np.array([0.01, np.inf], dtype=np.float64),
        np.array([-0.01, 0.5], dtype=np.float64),
        np.array([0.5, 1.01], dtype=np.float64),
    ),
)
def test_bh_rejects_nonfinite_or_out_of_range_p_values_before_core(
    monkeypatch: pytest.MonkeyPatch,
    p_values: np.ndarray,
) -> None:
    """Invalid probability evidence fails before Rust BH arithmetic."""
    monkeypatch.setattr(fitstats_module, "_core_module", _bomb_core)

    with pytest.raises(ValueError, match="p_values"):
        fitstats_module.benjamini_hochberg(p_values, q=0.05)


def test_bh_rejects_oversized_builtin_integer_as_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scalar dtype overflow stays inside the package-owned p-value contract."""
    monkeypatch.setattr(fitstats_module, "_core_module", _bomb_core)

    with pytest.raises(ValueError, match="p_values"):
        fitstats_module.benjamini_hochberg(10**400, q=0.05)


def test_bh_trusted_evidence_is_normalized_and_public_alias_is_hardened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trusted inputs preserve shape and reach Rust as package-owned primitives."""
    captured: dict[str, object] = {}

    class _Core:
        def benjamini_hochberg(self, p_values: np.ndarray, q: float) -> list[bool]:
            captured["p_values"] = p_values
            captured["q"] = q
            return [True, False, True, False]

    monkeypatch.setattr(fitstats_module, "_core_module", lambda: _Core())
    result = fitstats_module.benjamini_hochberg(
        [[0.01, np.float32(0.20)], [np.float64(0.03), 1]],
        q=np.float32(0.05),
    )

    rust_p = captured["p_values"]
    assert type(rust_p) is np.ndarray
    assert rust_p.dtype == np.float64
    assert rust_p.flags.c_contiguous
    assert rust_p.shape == (4,)
    assert type(captured["q"]) is float
    assert result.dtype == np.bool_
    assert result.shape == (2, 2)
    assert fast_mlsirm.benjamini_hochberg is fitstats_module.benjamini_hochberg
