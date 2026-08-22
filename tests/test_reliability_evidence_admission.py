"""Regression tests for reliability evidence admission before Rust dispatch."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import reliability


class _HostileArrayProvider:
    """Array provider that records any caller-controlled protocol execution."""

    def __init__(self) -> None:
        self.calls = 0

    def __array__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("caller-controlled __array__ callback executed")


def _native_discovery_must_not_run():
    raise AssertionError("compiled-core discovery ran before evidence admission")


def _core_that_must_not_dispatch() -> SimpleNamespace:
    def _dispatch(*args, **kwargs):
        raise AssertionError("Rust dispatch ran for rejected evidence")

    return SimpleNamespace(
        guttman_lambdas=_dispatch,
        tenberge_mu=_dispatch,
        cronbach_alpha=_dispatch,
        separation_reliability=_dispatch,
    )


@pytest.mark.parametrize(
    "invoke",
    [
        lambda: reliability.guttman_lambdas(
            np.array([[1.0 + 0.5j, 0.0], [0.0, 1.0]], dtype=np.complex128)
        ),
        lambda: reliability.tenberge_mu(
            np.array([[1.0, 0.0 + 0.25j], [0.0, 1.0]], dtype=np.complex128)
        ),
        lambda: reliability.cronbach_alpha(
            np.array([[1.0 + 1.0j, 0.0], [0.0, 1.0]], dtype=np.complex128)
        ),
        lambda: reliability.separation_reliability(
            np.array([0.1 + 0.2j, 0.3], dtype=np.complex128),
            np.array([0.01, 0.02], dtype=np.float64),
        ),
        lambda: reliability.separation_reliability(
            np.array([0.1, 0.3], dtype=np.float64),
            np.array([0.01 + 0.02j, 0.02], dtype=np.complex128),
        ),
    ],
)
def test_reliability_rejects_complex_evidence_before_native_discovery(
    monkeypatch, invoke
):
    """Imaginary evidence must not be projected onto a different real dataset."""
    monkeypatch.setattr(fitstats, "_core_module", _native_discovery_must_not_run)

    with pytest.raises(ValueError, match="real numeric"):
        invoke()


@pytest.mark.parametrize(
    "invoke",
    [
        lambda value: reliability.guttman_lambdas(value),
        lambda value: reliability.tenberge_mu(value),
        lambda value: reliability.cronbach_alpha(value),
        lambda value: reliability.separation_reliability(
            value, np.array([0.01, 0.02], dtype=np.float64)
        ),
        lambda value: reliability.separation_reliability(
            np.array([0.1, 0.2], dtype=np.float64), value
        ),
    ],
)
def test_reliability_rejects_arbitrary_array_provider_without_callback(
    monkeypatch, invoke
):
    """Package validation must not invoke a caller-defined array protocol."""
    hostile = _HostileArrayProvider()
    monkeypatch.setattr(fitstats, "_core_module", _core_that_must_not_dispatch)

    with pytest.raises(ValueError, match="real numeric"):
        invoke(hostile)

    assert hostile.calls == 0


def test_cronbach_alpha_preserves_plain_sequence_numeric_compatibility(monkeypatch):
    """Exact built-in sequences with trusted real scalars remain valid evidence."""
    seen: dict[str, object] = {}

    def _cronbach_alpha(flat, n_persons, n_items):
        seen["flat"] = flat
        seen["shape"] = (n_persons, n_items)
        return 0.75

    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: SimpleNamespace(cronbach_alpha=_cronbach_alpha),
    )

    result = reliability.cronbach_alpha(
        [
            [1, np.float32(2.0)],
            [np.int16(2), 3.0],
            [3, np.float64(4.0)],
        ]
    )

    assert result == pytest.approx(0.75)
    assert seen["shape"] == (3, 2)
    flat = seen["flat"]
    assert isinstance(flat, np.ndarray)
    assert flat.dtype == np.float64


def test_separation_reliability_preserves_plain_sequence_numeric_compatibility(
    monkeypatch,
):
    """Trusted vector sequences are marshalled as ordinary contiguous floats."""
    seen: dict[str, object] = {}

    def _separation(measures, se):
        seen["measures"] = measures
        seen["se"] = se
        return {"sep_rel": 0.5, "ssd": 2.0, "mse": 1.0, "sep_index": 1.0}

    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: SimpleNamespace(separation_reliability=_separation),
    )

    result = reliability.separation_reliability(
        (np.float32(0.1), 0.2, np.int16(1)),
        [np.float64(0.01), 0.02, np.uint8(1)],
    )

    assert result.sep_rel == pytest.approx(0.5)
    assert isinstance(seen["measures"], np.ndarray)
    assert isinstance(seen["se"], np.ndarray)
    assert seen["measures"].dtype == np.float64
    assert seen["se"].dtype == np.float64
