"""Regression tests for ICC ratings admission before native discovery."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm
import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import reliability


class _HostileArrayProvider:
    def __init__(self) -> None:
        self.calls = 0

    def __array__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("caller-controlled __array__ callback executed")


def _native_discovery_must_not_run():
    raise AssertionError("compiled-core discovery ran before ICC ratings admission")


def test_icc_rejects_array_provider_without_callback_or_native(monkeypatch):
    hostile = _HostileArrayProvider()
    monkeypatch.setattr(fitstats, "_core_module", _native_discovery_must_not_run)

    with pytest.raises(ValueError, match="real numeric"):
        reliability.icc(hostile)

    assert hostile.calls == 0


def test_icc_rejects_complex_ratings_before_native(monkeypatch):
    monkeypatch.setattr(fitstats, "_core_module", _native_discovery_must_not_run)

    with pytest.raises(ValueError, match="real numeric"):
        reliability.icc(
            np.array(
                [[1.0 + 0.5j, 2.0], [2.0, 1.0], [3.0, 4.0]],
                dtype=np.complex128,
            )
        )


def test_icc_preserves_boolean_ratings_rejection_before_native(monkeypatch):
    monkeypatch.setattr(fitstats, "_core_module", _native_discovery_must_not_run)

    with pytest.raises(ValueError, match="real numeric"):
        reliability.icc(np.array([[True, False], [False, True]], dtype=np.bool_))


def test_icc_preserves_trusted_sequence_compatibility(monkeypatch):
    seen: dict[str, object] = {}

    def _icc(flat, n_subjects, n_raters, model, kind, unit, r0, conf_level):
        seen["flat"] = flat
        seen["shape"] = (n_subjects, n_raters)
        seen["controls"] = (model, kind, unit, r0, conf_level)
        return {
            "value": 0.5,
            "subjects": n_subjects,
            "raters": n_raters,
            "fvalue": 2.0,
            "df1": 2.0,
            "df2": 3.0,
            "p_value": 0.2,
            "lbound": 0.1,
            "ubound": 0.8,
        }

    monkeypatch.setattr(fitstats, "_core_module", lambda: SimpleNamespace(icc=_icc))

    result = reliability.icc(
        [
            [np.float32(1.0), 2],
            [2.0, np.int16(1)],
            [3, np.float64(4.0)],
        ],
        model="twoway",
        type="agreement",
        unit="average",
        r0=np.float32(0.1),
        conf_level=np.float64(0.9),
    )

    assert result.value == pytest.approx(0.5)
    assert seen["shape"] == (3, 2)
    assert seen["controls"] == ("twoway", "agreement", "average", 0.1, 0.9)
    assert isinstance(seen["flat"], np.ndarray)
    assert seen["flat"].dtype == np.float64


def test_top_level_icc_export_uses_hardened_adapter():
    assert fast_mlsirm.icc is reliability.icc
