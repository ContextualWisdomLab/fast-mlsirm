"""Trust-boundary regressions for public delta-plot controls."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.config import MAX_MAX_ITER
from fast_mlsirm.deltaplot import delta_plot


class _DataSentinel:
    """Fail if rejected controls permit response/group materialization."""

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Raise on any attempt to materialize caller-owned data."""
        raise AssertionError("data must not be materialized for invalid controls")


def _unexpected_core_discovery():
    """Fail if rejected controls permit compiled-core discovery."""
    raise AssertionError("compiled core must not be discovered for invalid controls")


class _HostileStr(str):
    """String subclass whose comparison/hash callbacks must stay inert."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        """Reset the callback counter."""
        cls.calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        return str.__eq__(self, other)

    def __hash__(self) -> int:
        type(self).calls += 1
        return str.__hash__(self)


class _HostileFloat(float):
    """Float subclass whose conversion/comparison callbacks must stay inert."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        """Reset the callback counter."""
        cls.calls = 0

    def __float__(self):
        type(self).calls += 1
        return float.__float__(self)

    def __lt__(self, other: object) -> bool:
        type(self).calls += 1
        return float.__lt__(self, other)


class _HostileInt(int):
    """Integer subclass whose conversion/comparison callbacks must stay inert."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        """Reset the callback counter."""
        cls.calls = 0

    def __int__(self):
        type(self).calls += 1
        return int.__int__(self)

    def __index__(self):
        type(self).calls += 1
        return int.__index__(self)

    def __lt__(self, other: object) -> bool:
        type(self).calls += 1
        return int.__lt__(self, other)


class _RangeTuple(tuple):
    """Tuple subclass whose indexing callback must never be invoked."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        """Reset the callback counter."""
        cls.calls = 0

    def __getitem__(self, index: object) -> object:
        type(self).calls += 1
        return tuple.__getitem__(self, index)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("threshold", _HostileStr("norm"), "threshold must be an exact string"),
        ("extreme", _HostileStr("constraint"), "extreme must be an exact string"),
        ("purify", _HostileStr("IPP1"), "purify must be None or an exact string"),
        ("alpha", _HostileFloat(0.05), "alpha must be a real number"),
        ("fixed_threshold", _HostileFloat(1.5), "fixed_threshold must be a real number"),
        ("nr_add", _HostileInt(1), "nr_add must be an integer"),
        ("max_iter", _HostileInt(10), "max_iter must be an integer"),
    ),
)
def test_rejects_executable_controls_before_callbacks_data_or_core(
    monkeypatch, field, value, message
):
    """Executable control providers fail before callbacks, data, and Rust discovery."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    type(value).reset()
    kwargs: dict[str, object] = {field: value}
    if field == "fixed_threshold":
        kwargs["threshold"] = "fixed"
    if field == "nr_add":
        kwargs["extreme"] = "add"

    with pytest.raises(ValueError, match=message):
        delta_plot(_DataSentinel(), _DataSentinel(), **kwargs)

    assert type(value).calls == 0


def test_rejects_tuple_subclass_before_indexing_data_or_core(monkeypatch):
    """Constraint ranges require a package-trusted tuple identity."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    value = _RangeTuple((0.001, 0.999))
    value.reset()

    with pytest.raises(ValueError, match="const_range must be an exact 2-tuple"):
        delta_plot(_DataSentinel(), _DataSentinel(), const_range=value)

    assert value.calls == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"threshold": "other"}, "threshold must be 'norm' or 'fixed'"),
        ({"extreme": "other"}, "extreme must be 'constraint' or 'add'"),
        ({"purify": "other"}, "purify must be None, 'IPP1', 'IPP2', or 'IPP3'"),
        ({"alpha": 0.0}, "alpha must be finite and in \(0, 1\)"),
        ({"alpha": float("nan")}, "alpha must be finite and in \(0, 1\)"),
        ({"threshold": "fixed", "fixed_threshold": float("inf")}, "fixed_threshold must be finite"),
        ({"extreme": "add", "nr_add": 0}, "nr_add must be a positive integer >= 1"),
        ({"max_iter": 0}, f"max_iter must be between 1 and {MAX_MAX_ITER}"),
        ({"max_iter": MAX_MAX_ITER + 1}, f"max_iter must be between 1 and {MAX_MAX_ITER}"),
        ({"const_range": (-0.1, 0.9)}, "constraint range must satisfy 0 <= lo < hi <= 1"),
        ({"const_range": (0.9, 0.1)}, "constraint range must satisfy 0 <= lo < hi <= 1"),
        ({"const_range": (0.1, float("inf"))}, "constraint range must satisfy 0 <= lo < hi <= 1"),
    ),
)
def test_invalid_exact_controls_fail_before_data_and_core(monkeypatch, kwargs, message):
    """Type-correct invalid controls fail before any data/native side effect."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)

    with pytest.raises(ValueError, match=message):
        delta_plot(_DataSentinel(), _DataSentinel(), **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"alpha": True},
        {"alpha": np.bool_(True)},
        {"threshold": "fixed", "fixed_threshold": False},
        {"extreme": "add", "nr_add": True},
        {"max_iter": np.bool_(True)},
        {"const_range": (np.bool_(False), 0.9)},
    ),
)
def test_boolean_numeric_controls_fail_before_data_and_core(monkeypatch, kwargs):
    """Boolean scalar identities are not numeric control identities."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)

    with pytest.raises(ValueError):
        delta_plot(_DataSentinel(), _DataSentinel(), **kwargs)


class _FakeCore:
    """Capture the trusted PyO3 payload for one valid delta-plot call."""

    def __init__(self) -> None:
        """Initialize an empty call ledger."""
        self.calls: list[tuple[Any, ...]] = []

    def py_delta_plot(self, *args: Any) -> dict[str, Any]:
        """Record dispatch and return a structurally valid Rust-like result."""
        self.calls.append(args)
        return {
            "props": [0.5, 0.5, 0.5, 0.5],
            "adj_props": [0.5, 0.5, 0.5, 0.5],
            "deltas": [13.0, 13.0, 13.0, 13.0],
            "dist": [0.0, 0.0],
            "axis_par": [0.0, 1.0],
            "thresholds": [1.0],
            "dif_items": [],
            "n_iter": 1,
            "converged": True,
        }


def _responses() -> np.ndarray:
    """Return a small valid two-item response matrix."""
    return np.array([[0, 1], [1, 0]], dtype=np.int64)


def _group() -> np.ndarray:
    """Return a valid reference/focal group vector."""
    return np.array([0, 1], dtype=np.int64)


def test_genuine_numpy_controls_dispatch_as_exact_builtins(monkeypatch):
    """Supported NumPy scalars normalize once before the PyO3 call."""
    core = _FakeCore()
    monkeypatch.setattr(fitstats, "_core_module", lambda: core)

    result = delta_plot(
        _responses(),
        _group(),
        threshold="norm",
        alpha=np.float32(0.05),
        extreme="constraint",
        const_range=(np.float32(0.001), np.float64(0.999)),
        nr_add=np.int64(2),
        purify="IPP1",
        max_iter=np.int64(20),
    )

    assert result.n_iter == 1
    assert len(core.calls) == 1
    (_xf, _gu, n, ni, extreme, ea, eb, threshold, tv, purify, max_iter) = core.calls[0]
    assert type(n) is int
    assert type(ni) is int
    assert type(extreme) is str
    assert type(ea) is float
    assert type(eb) is float
    assert type(threshold) is str
    assert type(tv) is float
    assert type(purify) is str
    assert type(max_iter) is int
    assert (n, ni, extreme, threshold, purify, max_iter) == (2, 2, "constraint", "norm", "IPP1", 20)
