"""Trust-boundary regressions for public Angoff delta-plot controls."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.config import MAX_MAX_ITER
from fast_mlsirm.deltaplot import delta_plot


class _ResponsesSentinel:
    """Fail if rejected controls allow response-matrix materialization."""

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Raise when NumPy is asked to materialize this sentinel."""
        raise AssertionError("responses must not be materialized for invalid controls")


class _GroupSentinel:
    """Fail if rejected controls allow group-vector materialization."""

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Raise when NumPy is asked to materialize this sentinel."""
        raise AssertionError("group must not be materialized for invalid controls")


class _ExplosiveControl:
    """Fail on every caller-owned conversion or comparison callback."""

    def __str__(self) -> str:
        """Reject string conversion."""
        raise AssertionError("__str__ must not run")

    def __repr__(self) -> str:
        """Reject representation conversion."""
        raise AssertionError("__repr__ must not run")

    def __int__(self) -> int:
        """Reject integer conversion."""
        raise AssertionError("__int__ must not run")

    def __index__(self) -> int:
        """Reject index conversion."""
        raise AssertionError("__index__ must not run")

    def __float__(self) -> float:
        """Reject floating-point conversion."""
        raise AssertionError("__float__ must not run")

    def __eq__(self, other: object) -> bool:
        """Reject equality comparison."""
        raise AssertionError("__eq__ must not run")

    def __hash__(self) -> int:
        """Reject hashing."""
        raise AssertionError("__hash__ must not run")

    def __lt__(self, other: object) -> bool:
        """Reject ordering comparison."""
        raise AssertionError("__lt__ must not run")

    def __le__(self, other: object) -> bool:
        """Reject ordering comparison."""
        raise AssertionError("__le__ must not run")

    def __gt__(self, other: object) -> bool:
        """Reject ordering comparison."""
        raise AssertionError("__gt__ must not run")

    def __ge__(self, other: object) -> bool:
        """Reject ordering comparison."""
        raise AssertionError("__ge__ must not run")

    def __getitem__(self, key: object) -> object:
        """Reject sequence access on a pair-shaped provider."""
        raise AssertionError("__getitem__ must not run")


class _ExplosiveString(str):
    """A built-in string subclass whose conversion must never execute."""

    def __str__(self) -> str:
        """Reject normalization through ``str``."""
        raise AssertionError("string subclass __str__ must not run")

    def __eq__(self, other: object) -> bool:
        """Reject equality comparison on the subclass instance."""
        raise AssertionError("string subclass __eq__ must not run")

    __hash__ = str.__hash__


class _ExplosiveFloat(float):
    """A built-in float subclass whose conversion must never execute."""

    def __float__(self) -> float:
        """Reject normalization through ``float``."""
        raise AssertionError("float subclass __float__ must not run")


class _ExplosiveInt(int):
    """A built-in integer subclass whose conversion must never execute."""

    def __int__(self) -> int:
        """Reject normalization through ``int``."""
        raise AssertionError("int subclass __int__ must not run")

    def __index__(self) -> int:
        """Reject index conversion."""
        raise AssertionError("int subclass __index__ must not run")

    def __float__(self) -> float:
        """Reject normalization through ``float``."""
        raise AssertionError("int subclass __float__ must not run")


class _ExplosiveNumpyFloat(np.float64):
    """A NumPy floating subclass whose conversion must never execute."""

    def __float__(self) -> float:
        """Reject normalization through ``float``."""
        raise AssertionError("NumPy subclass __float__ must not run")


class _ExplosiveNumpyInt(np.int64):
    """A NumPy integer subclass whose conversion must never execute."""

    def __int__(self) -> int:
        """Reject normalization through ``int``."""
        raise AssertionError("NumPy subclass __int__ must not run")

    def __index__(self) -> int:
        """Reject index conversion."""
        raise AssertionError("NumPy subclass __index__ must not run")


def _unexpected_core_discovery():
    """Fail if rejected controls reach compiled-core discovery."""
    raise AssertionError("compiled core must not be discovered for invalid controls")


def _call_with_sentinels(**kwargs: object) -> None:
    """Invoke the public adapter with materialization sentinels."""
    delta_plot(_ResponsesSentinel(), _GroupSentinel(), **kwargs)


@pytest.mark.parametrize("field", ("threshold", "extreme", "purify"))
def test_executable_string_controls_fail_before_data_or_core(monkeypatch, field):
    """Selector controls reject protocol providers before any side effect."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    with pytest.raises(ValueError, match=field):
        _call_with_sentinels(**{field: _ExplosiveControl()})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("threshold", _ExplosiveString("norm")),
        ("extreme", _ExplosiveString("constraint")),
        ("purify", _ExplosiveString("IPP1")),
    ),
)
def test_string_subclasses_fail_before_data_or_core(monkeypatch, field, value):
    """Selector controls require exact built-in string identity."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    with pytest.raises(ValueError, match=field):
        _call_with_sentinels(**{field: value})


@pytest.mark.parametrize("field", ("alpha", "fixed_threshold", "nr_add", "max_iter"))
@pytest.mark.parametrize(
    "value",
    (
        _ExplosiveControl(),
        _ExplosiveFloat(0.5),
        _ExplosiveInt(1),
        _ExplosiveNumpyFloat(0.5),
        _ExplosiveNumpyInt(1),
    ),
)
def test_executable_numeric_controls_fail_before_data_or_core(monkeypatch, field, value):
    """Numeric controls reject subclasses and protocol providers before coercion."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    with pytest.raises(ValueError, match=field):
        _call_with_sentinels(**{field: value})


def test_executable_const_range_fails_before_data_or_core(monkeypatch):
    """Constraint-range pairs reject hostile items before sequence coercion."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    with pytest.raises(ValueError, match="const_range"):
        _call_with_sentinels(const_range=_ExplosiveControl())
    with pytest.raises(ValueError, match="const_range"):
        _call_with_sentinels(const_range=(_ExplosiveFloat(0.001), 0.999))
    with pytest.raises(ValueError, match="const_range"):
        _call_with_sentinels(const_range=(0.001, _ExplosiveControl()))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"threshold": "bogus"}, "threshold"),
        ({"threshold": True}, "threshold"),
        ({"extreme": "bogus"}, "extreme"),
        ({"extreme": False}, "extreme"),
        ({"purify": "IPP9"}, "purify"),
        ({"purify": True}, "purify"),
        ({"alpha": True}, "alpha"),
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": 1.0}, "alpha"),
        ({"alpha": float("nan")}, "alpha"),
        ({"alpha": float("inf")}, "alpha"),
        ({"fixed_threshold": True}, "fixed_threshold"),
        ({"fixed_threshold": float("nan")}, "fixed_threshold"),
        ({"fixed_threshold": float("inf")}, "fixed_threshold"),
        ({"nr_add": True}, "nr_add"),
        ({"nr_add": 0}, "nr_add"),
        ({"nr_add": -1}, "nr_add"),
        ({"max_iter": True}, "max_iter"),
        ({"max_iter": 0}, "max_iter"),
        ({"max_iter": MAX_MAX_ITER + 1}, "max_iter"),
        ({"const_range": [0.001, 0.999]}, "const_range"),
        ({"const_range": (0.0, 0.999)}, "const_range"),
        ({"const_range": (0.001, 1.0)}, "const_range"),
        ({"const_range": (0.8, 0.2)}, "const_range"),
        ({"const_range": (0.5, 0.5)}, "const_range"),
    ),
)
def test_invalid_exact_controls_fail_before_data_and_core(monkeypatch, kwargs, message):
    """Type-correct but invalid controls fail before data and native boundaries."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    with pytest.raises(ValueError, match=message):
        _call_with_sentinels(**kwargs)


@pytest.mark.parametrize("field", ("alpha", "fixed_threshold"))
def test_huge_builtin_integer_reals_fail_as_local_value_errors(monkeypatch, field):
    """Reject exact integers that cannot normalize to a finite float64."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    with pytest.raises(ValueError, match=field):
        _call_with_sentinels(**{field: 10**10000})


class _FakeDeltaPlotCore:
    """Capture the trusted PyO3 payload without running Delta-plot arithmetic."""

    def __init__(self) -> None:
        """Initialize an empty dispatch ledger."""
        self.calls: list[tuple[Any, ...]] = []

    def py_delta_plot(self, *args: Any) -> dict[str, Any]:
        """Record one dispatch and return a structurally valid Rust-like result."""
        self.calls.append(args)
        return {
            "props": [0.5, 0.5, 0.5, 0.5],
            "adj_props": [0.5, 0.5, 0.5, 0.5],
            "deltas": [13.0, 13.0, 13.0, 13.0],
            "dist": [0.0, 0.0],
            "axis_par": [0.0, 1.0],
            "thresholds": [1.5],
            "dif_items": [],
            "n_iter": 1,
            "converged": True,
        }


def _valid_data() -> tuple[np.ndarray, np.ndarray]:
    """Return the smallest ordinary person-by-item matrix the adapter accepts."""
    responses = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    group = np.array([0, 0, 1, 1], dtype=np.int64)
    return responses, group


def test_genuine_numpy_controls_dispatch_as_exact_builtins(monkeypatch):
    """Supported NumPy scalars and exact strings normalize once before PyO3."""
    core = _FakeDeltaPlotCore()
    monkeypatch.setattr(fitstats, "_core_module", lambda: core)
    responses, group = _valid_data()

    result = delta_plot(
        responses,
        group,
        threshold="fixed",
        alpha=np.float64(0.05),
        fixed_threshold=np.float32(1.5),
        extreme="add",
        const_range=(np.float64(0.001), np.float32(0.999)),
        nr_add=np.int64(2),
        purify="IPP1",
        max_iter=np.int32(4),
    )

    assert result.n_iter == 1
    assert result.converged is True
    assert len(core.calls) == 1
    (
        _flat_responses,
        _group,
        n_persons,
        n_items,
        extreme,
        extreme_a,
        extreme_b,
        threshold,
        threshold_value,
        purify,
        max_iter,
    ) = core.calls[0]
    assert n_persons == 4
    assert n_items == 2
    assert extreme == "add"
    assert type(extreme) is str
    assert type(extreme_a) is float
    assert type(extreme_b) is float
    assert extreme_a == 2.0
    assert extreme_b == 0.0
    assert threshold == "fixed"
    assert type(threshold) is str
    assert type(threshold_value) is float
    assert threshold_value == pytest.approx(float(np.float32(1.5)))
    assert purify == "IPP1"
    assert type(purify) is str
    assert type(max_iter) is int
    assert max_iter == 4
