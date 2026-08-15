"""Fail-first contracts for parallel-analysis control and workspace validation."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.parallel_analysis import parallel_analysis


_DATA = np.array(
    [
        [0.1, 1.0],
        [0.4, 0.5],
        [0.9, -0.2],
        [1.2, -0.7],
    ],
    dtype=np.float64,
)

_NUMPY_INTEGER_TYPES = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.intp,
    np.longlong,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.uintp,
    np.ulonglong,
)


class _RecordingCore:
    """Capture accepted controls without running the expensive numerical kernel."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def parallel_analysis(self, *args):
        self.calls.append(args)
        return {
            "retained": 1,
            "eigenvalues": [1.5, 0.5],
            "random_eigenvalues": [1.1, 0.9],
            "bias": [0.1, -0.1],
            "adjusted_eigenvalues": [1.4, 0.6],
        }


class _HostilePythonInt(int):
    """Integer subclass whose conversion/representation hooks must stay inert."""

    def __int__(self) -> int:
        raise AssertionError("caller-controlled integer conversion executed")

    def __repr__(self) -> str:
        raise AssertionError("caller-controlled representation executed")


class _HostileNumpyInt(np.int64):
    """NumPy integer subclass whose callbacks must never authorize a control."""

    def __int__(self) -> int:
        raise AssertionError("caller-controlled integer conversion executed")

    def __repr__(self) -> str:
        raise AssertionError("caller-controlled representation executed")


def _install_core(monkeypatch: pytest.MonkeyPatch, core: object) -> None:
    """Replace the package core loader used by the public wrapper."""
    import fast_mlsirm.fitstats as fitstats

    monkeypatch.setattr(fitstats, "_core_module", lambda: core)


def _forbid_core_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if public validation crosses the native-loader boundary."""
    import fast_mlsirm.fitstats as fitstats

    def unexpected_core_discovery() -> object:
        raise AssertionError("native core discovery executed")

    monkeypatch.setattr(fitstats, "_core_module", unexpected_core_discovery)


@pytest.mark.parametrize(
    ("name", "bad_value"),
    [
        ("n_iterations", True),
        ("n_iterations", 2.5),
        ("n_iterations", "2"),
        ("centile", True),
        ("centile", 50.5),
        ("centile", "50"),
        ("seed", True),
        ("seed", 1.5),
        ("seed", "1"),
    ],
)
def test_noninteger_controls_fail_before_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    bad_value: object,
) -> None:
    """Booleans, floats, and strings fail before native-core discovery."""
    _forbid_core_discovery(monkeypatch)
    kwargs: dict[str, object] = {"n_iterations": 2, "centile": 0, "seed": 1}
    kwargs[name] = bad_value

    with pytest.raises(ValueError, match=rf"^{name} "):
        parallel_analysis(_DATA, **kwargs)


def test_hostile_integer_conversion_is_not_executed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control validation must use admitted types rather than caller conversion hooks."""
    _forbid_core_discovery(monkeypatch)

    class HostileInt:
        def __int__(self) -> int:
            raise RuntimeError("caller-controlled conversion executed")

    with pytest.raises(ValueError, match=r"^n_iterations "):
        parallel_analysis(_DATA, n_iterations=HostileInt())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("name", "hostile_type", "raw_value"),
    [
        ("n_iterations", _HostilePythonInt, 2),
        ("n_iterations", _HostileNumpyInt, 2),
        ("centile", _HostilePythonInt, 50),
        ("centile", _HostileNumpyInt, 50),
        ("seed", _HostilePythonInt, 7),
        ("seed", _HostileNumpyInt, 7),
    ],
    ids=[
        "iterations-python-subclass",
        "iterations-numpy-subclass",
        "centile-python-subclass",
        "centile-numpy-subclass",
        "seed-python-subclass",
        "seed-numpy-subclass",
    ],
)
def test_integer_subclasses_fail_before_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    hostile_type: type,
    raw_value: int,
) -> None:
    """Untrusted integer subclasses fail before callbacks or native discovery."""
    _forbid_core_discovery(monkeypatch)
    bad_value = hostile_type(raw_value)
    kwargs: dict[str, object] = {"n_iterations": 2, "centile": 0, "seed": 1}
    kwargs[name] = bad_value

    with pytest.raises(ValueError, match=rf"^{name} "):
        parallel_analysis(_DATA, **kwargs)


def test_oversized_random_benchmark_workspace_fails_before_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oversized simulation storage is rejected before native-core discovery."""
    _forbid_core_discovery(monkeypatch)

    with pytest.raises(ValueError, match="workspace"):
        parallel_analysis(_DATA, n_iterations=2**62)


def test_seed_must_fit_rust_u64_before_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The PyO3 integer transport range is validated before core discovery."""
    _forbid_core_discovery(monkeypatch)

    with pytest.raises(ValueError, match=r"^seed "):
        parallel_analysis(_DATA, n_iterations=2, seed=2**64)


@pytest.mark.parametrize("scalar_type", _NUMPY_INTEGER_TYPES)
def test_numpy_integer_controls_remain_accepted(
    monkeypatch: pytest.MonkeyPatch,
    scalar_type: type[np.integer],
) -> None:
    """Every genuine supported NumPy integer scalar retains the public contract."""
    core = _RecordingCore()
    _install_core(monkeypatch, core)

    result = parallel_analysis(
        _DATA,
        n_iterations=scalar_type(2),
        centile=scalar_type(50),
        seed=scalar_type(7),
    )

    assert result.retained == 1
    assert len(core.calls) == 1
    call = core.calls[0]
    assert call[-3:] == (2, 50, 7)
