"""Native-layout provenance regressions for EBDIF Rust results."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.ebdif as ebdif
import fast_mlsirm.fitstats as fitstats


def _result() -> dict[str, object]:
    """Return an otherwise-valid two-item EBDIF native payload."""

    return {
        "mu": 0.0,
        "tau2": 0.0,
        "tau2_raw": 0.0,
        "weight": np.zeros(2, dtype=np.float64),
        "post_mean": np.zeros(2, dtype=np.float64),
        "post_var": np.zeros(2, dtype=np.float64),
        "cat_probs": np.array([0.0, 0.0, 1.0, 0.0, 0.0] * 2, dtype=np.float64),
    }


def test_native_result_vector_must_match_contiguous_pyo3_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale core cannot publish a strided view instead of PyArray1 output."""

    result = _result()
    strided = np.zeros(4, dtype=np.float64)[::2]
    assert type(strided) is np.ndarray
    assert strided.dtype == np.dtype(np.float64)
    assert strided.shape == (2,)
    assert not strided.flags.c_contiguous
    result["weight"] = strided

    class Core:
        @staticmethod
        def py_eb_mh_dif(mh: np.ndarray, se: np.ndarray) -> dict[str, object]:
            del mh, se
            return result

    monkeypatch.setattr(fitstats, "_core_module", lambda: Core())

    with pytest.raises(RuntimeError, match="invalid EBDIF Rust result payload"):
        ebdif.eb_mh_dif([0.1, -0.2], [0.3, 0.4])


def test_native_result_vector_must_own_pyo3_backing_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale core cannot publish a C-contiguous view backed elsewhere."""

    result = _result()
    backing = np.zeros(3, dtype=np.float64)
    foreign_backed = backing[:2]
    assert type(foreign_backed) is np.ndarray
    assert foreign_backed.dtype == np.dtype(np.float64)
    assert foreign_backed.shape == (2,)
    assert foreign_backed.flags.c_contiguous
    assert not foreign_backed.flags.owndata
    result["weight"] = foreign_backed

    class Core:
        @staticmethod
        def py_eb_mh_dif(mh: np.ndarray, se: np.ndarray) -> dict[str, object]:
            del mh, se
            return result

    monkeypatch.setattr(fitstats, "_core_module", lambda: Core())

    with pytest.raises(RuntimeError, match="invalid EBDIF Rust result payload"):
        ebdif.eb_mh_dif([0.1, -0.2], [0.3, 0.4])
