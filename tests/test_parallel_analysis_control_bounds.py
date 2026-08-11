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


class _TrapCore:
    """Fail if an invalid public control reaches compiled numerical dispatch."""

    def parallel_analysis(self, *args, **kwargs):
        raise AssertionError("invalid control reached Rust dispatch")


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


def _install_core(monkeypatch: pytest.MonkeyPatch, core: object) -> None:
    """Replace the package core loader used by the public wrapper."""
    import fast_mlsirm.fitstats as fitstats

    monkeypatch.setattr(fitstats, "_core_module", lambda: core)


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
def test_noninteger_controls_fail_before_rust_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    bad_value: object,
) -> None:
    """Booleans, floats, and strings cannot be silently coerced to controls."""
    _install_core(monkeypatch, _TrapCore())
    kwargs: dict[str, object] = {"n_iterations": 2, "centile": 0, "seed": 1}
    kwargs[name] = bad_value

    with pytest.raises(ValueError, match=rf"^{name} "):
        parallel_analysis(_DATA, **kwargs)


def test_hostile_integer_conversion_is_not_executed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control validation must use admitted types rather than caller conversion hooks."""
    _install_core(monkeypatch, _TrapCore())

    class HostileInt:
        def __int__(self) -> int:
            raise RuntimeError("caller-controlled conversion executed")

    with pytest.raises(ValueError, match=r"^n_iterations "):
        parallel_analysis(_DATA, n_iterations=HostileInt())  # type: ignore[arg-type]


def test_oversized_random_benchmark_workspace_fails_before_rust_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller-controlled iteration counts cannot request unbounded simulation storage."""
    _install_core(monkeypatch, _TrapCore())

    with pytest.raises(ValueError, match="workspace"):
        parallel_analysis(_DATA, n_iterations=2**62)


def test_seed_must_fit_rust_u64_before_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Python validates the PyO3 integer transport range with a stable package error."""
    _install_core(monkeypatch, _TrapCore())

    with pytest.raises(ValueError, match=r"^seed "):
        parallel_analysis(_DATA, n_iterations=2, seed=2**64)


def test_numpy_integer_controls_remain_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exact NumPy integer scalars retain the documented public contract."""
    core = _RecordingCore()
    _install_core(monkeypatch, core)

    result = parallel_analysis(
        _DATA,
        n_iterations=np.int64(2),
        centile=np.int64(50),
        seed=np.int64(7),
    )

    assert result.retained == 1
    assert len(core.calls) == 1
    call = core.calls[0]
    assert call[-3:] == (2, 50, 7)
