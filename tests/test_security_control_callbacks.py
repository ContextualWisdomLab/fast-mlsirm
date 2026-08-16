"""Fail-first contracts for answer-copying scalar control marshalling."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
import fast_mlsirm.security as security


_RESPONSES = np.array(
    [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ],
    dtype=np.int64,
)
_WOLLACK_COPIER = np.array([0, 1], dtype=np.int64)
_WOLLACK_SOURCE = np.array([0, 1], dtype=np.int64)
_WOLLACK_PROBS = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=np.float64)


def _reject_core_discovery(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Make native-core discovery observable and forbidden for invalid controls."""
    calls: list[bool] = []

    def discover_core() -> None:
        calls.append(True)
        raise AssertionError("native core discovered before control validation")

    monkeypatch.setattr(fitstats, "_core_module", discover_core)
    return calls


def _hostile_integer(kind: str, callbacks: list[str]) -> int | np.integer:
    """Return an integer subclass whose comparison/conversion hooks are executable."""

    class HostilePythonInt(int):
        def __lt__(self, other: object) -> bool:
            callbacks.append("lt")
            raise AssertionError("caller-controlled __lt__ executed")

        def __le__(self, other: object) -> bool:
            callbacks.append("le")
            raise AssertionError("caller-controlled __le__ executed")

        def __int__(self) -> int:
            callbacks.append("int")
            raise AssertionError("caller-controlled __int__ executed")

        def __repr__(self) -> str:
            callbacks.append("repr")
            raise AssertionError("caller-controlled __repr__ executed")

    class HostileNumpyInt(np.int64):
        def __lt__(self, other: object) -> bool:
            callbacks.append("lt")
            raise AssertionError("caller-controlled NumPy __lt__ executed")

        def __le__(self, other: object) -> bool:
            callbacks.append("le")
            raise AssertionError("caller-controlled NumPy __le__ executed")

        def __int__(self) -> int:
            callbacks.append("int")
            raise AssertionError("caller-controlled NumPy __int__ executed")

        def __repr__(self) -> str:
            callbacks.append("repr")
            raise AssertionError("caller-controlled NumPy __repr__ executed")

    if kind == "python":
        return HostilePythonInt(1)
    return HostileNumpyInt(1)


def _call_wollack(value: object) -> object:
    return security.wollack_omega(
        _WOLLACK_COPIER,
        _WOLLACK_SOURCE,
        _WOLLACK_PROBS,
        n_options=value,  # type: ignore[arg-type]
    )


def _call_k_index(value: object) -> object:
    return security.k_index(
        _RESPONSES,
        copier=value,  # type: ignore[arg-type]
        source=1,
    )


def _call_k_variants(value: object) -> object:
    return security.k_variants(
        _RESPONSES,
        copier=value,  # type: ignore[arg-type]
        source=1,
    )


@pytest.mark.parametrize("kind", ["python", "numpy"])
@pytest.mark.parametrize(
    ("call", "message"),
    [
        (_call_wollack, r"^n_options must be an integer$"),
        (_call_k_index, r"^copier must be an integer row index$"),
        (_call_k_variants, r"^copier must be an integer row index$"),
    ],
)
def test_integer_subclasses_fail_without_callbacks_or_core_discovery(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    call: Callable[[object], object],
    message: str,
) -> None:
    """Executable integer subclasses are data, not trusted control authority."""
    discovery_calls = _reject_core_discovery(monkeypatch)
    callbacks: list[str] = []
    value = _hostile_integer(kind, callbacks)

    with pytest.raises(ValueError, match=message):
        call(value)

    assert callbacks == []
    assert discovery_calls == []


@pytest.mark.parametrize(
    ("call", "expected_message"),
    [
        (
            lambda: security.wollack_omega(
                _WOLLACK_COPIER,
                _WOLLACK_SOURCE,
                _WOLLACK_PROBS,
                n_options=np.int64(2),
            ),
            "wollack_omega requires the compiled Rust core",
        ),
        (
            lambda: security.k_index(
                _RESPONSES,
                copier=np.int64(0),
                source=np.int64(1),
            ),
            "k_index requires the compiled Rust core",
        ),
        (
            lambda: security.k_variants(
                _RESPONSES,
                copier=np.int64(0),
                source=np.int64(1),
            ),
            "k_variants requires the compiled Rust core",
        ),
    ],
)
def test_exact_numpy_integer_controls_reach_core_only_after_marshalling(
    monkeypatch: pytest.MonkeyPatch,
    call: Callable[[], object],
    expected_message: str,
) -> None:
    """Genuine NumPy integer scalars remain supported at the Python boundary."""
    discovery_calls: list[bool] = []

    def absent_core() -> None:
        discovery_calls.append(True)
        return None

    monkeypatch.setattr(fitstats, "_core_module", absent_core)

    with pytest.raises(RuntimeError, match=expected_message):
        call()

    assert discovery_calls == [True]
