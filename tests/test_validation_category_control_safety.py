"""Security regressions for judge category-count validation."""

from __future__ import annotations

import builtins
from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm
from fast_mlsirm import validation


class _HostileInt(int):
    """Integer subclass whose conversion callback records execution."""

    calls = 0

    def __int__(self) -> int:
        type(self).calls += 1
        return 2


class _HostileNumpyInt(np.int64):
    """NumPy integer subclass whose conversion callback records execution."""

    calls = 0

    def __int__(self) -> int:
        type(self).calls += 1
        return 2


class _HostileScalarMeta(type):
    """Metaclass proving scalar-type admission cannot hash caller types."""

    calls = 0

    def __hash__(cls) -> int:
        type(cls).calls += 1
        return type.__hash__(np.int64)

    def __eq__(cls, other: object) -> bool:
        type(cls).calls += 1
        return type.__eq__(cls, other)


class _MetaclassHostileNumpyInt(np.int64, metaclass=_HostileScalarMeta):
    """NumPy scalar subclass with caller-controlled type hash/equality hooks."""


class _IntegerProtocolProvider:
    """Arbitrary integer protocol provider that must never be invoked."""

    calls = 0

    def __int__(self) -> int:
        type(self).calls += 1
        return 2


def _fake_core(calls: list[tuple[int, type[int]]]) -> SimpleNamespace:
    """Return a deterministic Rust-boundary stand-in for marshalling tests."""

    def validate_scoring(judge, human, k, **kwargs):
        del judge, human, kwargs
        calls.append((k, type(k)))
        return {
            "gates": [],
            "exact_agreement": 1.0,
            "adjacent_agreement": 1.0,
            "pass": True,
        }

    return SimpleNamespace(validate_scoring=validate_scoring)


@pytest.mark.parametrize(
    "control_type",
    [_HostileInt, _HostileNumpyInt, _IntegerProtocolProvider],
)
def test_validate_judge_rejects_executable_integer_controls_without_callbacks(
    monkeypatch: pytest.MonkeyPatch,
    control_type: type,
) -> None:
    """Rejected category controls cannot run conversion code or Rust dispatch."""
    control_type.calls = 0
    rust_calls: list[tuple[int, type[int]]] = []
    monkeypatch.setattr(fast_mlsirm, "_core", _fake_core(rust_calls), raising=False)

    value = control_type(2) if control_type is not _IntegerProtocolProvider else control_type()
    with pytest.raises(ValueError):
        validation.validate_judge(np.array([0, 1]), np.array([0, 1]), k=value)

    assert control_type.calls == 0
    assert rust_calls == []


def test_validate_judge_rejects_hostile_scalar_metaclass_without_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scalar-type admission cannot dispatch caller metaclass hash/equality hooks."""
    _HostileScalarMeta.calls = 0
    rust_calls: list[tuple[int, type[int]]] = []
    monkeypatch.setattr(fast_mlsirm, "_core", _fake_core(rust_calls), raising=False)

    with pytest.raises(ValueError):
        validation.validate_judge(
            np.array([0, 1]),
            np.array([0, 1]),
            k=_MetaclassHostileNumpyInt(2),
        )

    assert _HostileScalarMeta.calls == 0
    assert rust_calls == []


def test_validate_judge_rejects_invalid_k_before_core_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid exact category count fails before compiled-core discovery."""
    real_import = builtins.__import__
    core_import_calls = 0

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        nonlocal core_import_calls
        if "_core" in fromlist:
            core_import_calls += 1
            raise AssertionError("compiled core discovered before k validation")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ValueError, match="must be >= 2"):
        validation.validate_judge(np.array([0, 1]), np.array([0, 1]), k=1)

    assert core_import_calls == 0


@pytest.mark.parametrize(
    "k",
    [np.int8(2), np.uint16(2), np.int32(2), np.uint64(2)],
)
def test_validate_judge_normalizes_genuine_numpy_category_counts(
    monkeypatch: pytest.MonkeyPatch,
    k: np.integer,
) -> None:
    """Supported concrete NumPy integer scalars marshal as exact Python ints."""
    rust_calls: list[tuple[int, type[int]]] = []
    monkeypatch.setattr(fast_mlsirm, "_core", _fake_core(rust_calls), raising=False)

    verdict = validation.validate_judge(np.array([0, 1]), np.array([0, 1]), k=k)

    assert verdict.passed is True
    assert rust_calls == [(2, int)]