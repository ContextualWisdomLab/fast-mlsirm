"""Resource-bound regressions for Empirical Bayes DIF evidence admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.ebdif as ebdif
import fast_mlsirm.fitstats as fitstats


def _unexpected_core() -> object:
    """Fail if compiled-core discovery happens before resource admission."""

    raise AssertionError("compiled core discovered before EBDIF resource admission")


def _result() -> dict[str, object]:
    """Return a trusted-core result for two admitted items."""

    return {
        "mu": 0.0,
        "tau2": 0.0,
        "tau2_raw": 0.0,
        "weight": np.zeros(2),
        "post_mean": np.zeros(2),
        "post_var": np.zeros(2),
        "cat_probs": np.array([[0.0, 0.0, 1.0, 0.0, 0.0]] * 2),
    }


def test_oversized_exact_numpy_vector_fails_before_contiguous_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Logical NumPy length is bounded before a tiny broadcast view becomes dense."""

    monkeypatch.setattr(ebdif, "_MAX_EBDIF_ITEMS", 2, raising=False)
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    original = np.ascontiguousarray

    def guarded_contiguous(value, *args, **kwargs):  # noqa: ANN001, ANN202
        if type(value) is np.ndarray and value.ndim == 1 and value.shape[0] > 2:
            raise AssertionError("oversized NumPy evidence reached contiguous allocation")
        return original(value, *args, **kwargs)

    monkeypatch.setattr(ebdif.np, "ascontiguousarray", guarded_contiguous)
    oversized = np.broadcast_to(np.array([0.1], dtype=np.float64), (3,))

    with pytest.raises(ValueError, match="mh exceeds the 2-item resource limit"):
        ebdif.eb_mh_dif(oversized, [0.3, 0.4])


def test_oversized_builtin_vector_fails_before_numpy_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Built-in length is bounded before NumPy materializes its scalar evidence."""

    monkeypatch.setattr(ebdif, "_MAX_EBDIF_ITEMS", 2, raising=False)
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    original = np.asarray

    def guarded_asarray(value, *args, **kwargs):  # noqa: ANN001, ANN202
        if type(value) in (list, tuple) and len(value) > 2:
            raise AssertionError("oversized built-in evidence reached NumPy materialization")
        return original(value, *args, **kwargs)

    monkeypatch.setattr(ebdif.np, "asarray", guarded_asarray)

    with pytest.raises(ValueError, match="se exceeds the 2-item resource limit"):
        ebdif.eb_mh_dif([0.1, -0.2], [0.3, 0.4, 0.5])


def test_exact_resource_boundary_preserves_rust_marshalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two items at a reduced ceiling still reach Rust as contiguous float64 arrays."""

    monkeypatch.setattr(ebdif, "_MAX_EBDIF_ITEMS", 2, raising=False)
    captured: dict[str, np.ndarray] = {}

    class Core:
        @staticmethod
        def py_eb_mh_dif(mh: np.ndarray, se: np.ndarray) -> dict[str, object]:
            captured["mh"] = mh.copy()
            captured["se"] = se.copy()
            return _result()

    monkeypatch.setattr(fitstats, "_core_module", lambda: Core())
    mh = np.broadcast_to(np.array([0.1], dtype=np.float64), (2,))
    result = ebdif.eb_mh_dif(mh, (0.3, 0.4))

    assert captured["mh"].dtype == np.float64
    assert captured["se"].dtype == np.float64
    assert captured["mh"].flags.c_contiguous
    assert captured["se"].flags.c_contiguous
    np.testing.assert_allclose(captured["mh"], [0.1, 0.1])
    np.testing.assert_allclose(captured["se"], [0.3, 0.4])
    assert result.cat_probs.shape == (2, 5)
