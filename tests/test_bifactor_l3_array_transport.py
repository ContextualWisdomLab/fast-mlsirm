"""L3 array transport to a recording binding, not native numerical execution.

The inspected numpy 0.29.0 PyReadonlyArray.as_slice requires contiguous
storage. NumPy require's C and A requirements ensure contiguous and aligned
storage while copying only when needed (NumPy Developers, n.d.). Tests preserve
logical item order, caller storage, and explicit precision/device controls.

References:
    NumPy Developers. (n.d.). numpy.require. NumPy reference guide,
        Parameters (C_CONTIGUOUS and ALIGNED) and Notes.
        https://numpy.org/doc/stable/reference/generated/numpy.require.html
    rust-numpy contributors. (n.d.). PyReadonlyArray::as_slice.
        numpy 0.29.0 API documentation.
        https://docs.rs/numpy/0.29.0/numpy/borrow/struct.PyReadonlyArray.html
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

bifactor = importlib.import_module("fast_mlsirm.bifactor_grm")
fitstats = importlib.import_module("fast_mlsirm.fitstats")


@pytest.mark.parametrize("entry", ["fit", "oakes", "fipc"])
@pytest.mark.parametrize(
    "layout",
    ["contiguous", "positive_stride", "negative_stride", "single_column", "unaligned", "readonly"],
)
def test_specific_map_transport_preserves_values_with_slice_safe_storage(
    monkeypatch: pytest.MonkeyPatch, layout: str, entry: str,
) -> None:
    """Observe contiguous/aligned transport, not executed Rust slice admission."""
    expected = np.array([0, 0, 1, 1], dtype=np.int64)
    if layout == "positive_stride":
        backing = np.repeat(expected, 2)
        specific_map = backing[::2]
    elif layout == "negative_stride":
        backing = expected[::-1].copy()
        specific_map = backing[::-1]
    elif layout == "single_column":
        backing = np.column_stack([expected, np.full(4, 99, dtype=np.int64)])
        specific_map = backing[:, :1].reshape(-1)
    elif layout == "unaligned":
        backing = bytearray(expected.nbytes + 1)
        specific_map = np.ndarray(expected.shape, dtype=np.int64, buffer=backing, offset=1)
        specific_map[:] = expected
        assert not specific_map.flags.aligned
    else:
        backing = expected.copy()
        specific_map = backing
        if layout == "readonly":
            specific_map.flags.writeable = False
    y = np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.int64)
    original_bytes = specific_map.tobytes()
    original_flags = (specific_map.strides, specific_map.flags.aligned, specific_map.flags.writeable)
    storage_bytes = bytes(backing) if isinstance(backing, bytearray) else backing.tobytes()
    calls = []

    def binding(*args: object) -> dict[str, object]:
        """Record actual wrapper arguments without loading an extension."""
        calls.append(args)
        yy, observed, item_map = args[3:6] if entry == "oakes" else args[:3]
        np.testing.assert_array_equal(yy, y.reshape(-1))
        np.testing.assert_array_equal(observed, np.ones(y.size, dtype=bool))
        np.testing.assert_array_equal(item_map, expected)
        assert item_map.dtype == np.dtype(np.int64)
        assert item_map.flags.c_contiguous, "specific_map transport is not contiguous"
        assert item_map.flags.aligned, "specific_map transport is not aligned"
        if entry == "fit":
            assert args[3:] == (2, 4, 2, 2, 121, 241, 1, 1e-6, 1, 2**53 + 1, "cpu", None)
        elif entry == "oakes":
            assert args[6:] == (2, 4, 2, 2, 121, 241, 1e-4)
            return {"labels": ["fixture"], "information": [1.0], "vcov": None,
                    "se": None, "positive_definite": False, "non_pd_reason": "fixture"}
        else:
            assert args[3:7] == (2, 4, 2, 2)
            assert args[11:] == (121, 241, 1, 1e-6, 10, 1e-8, False)
        return {
            "a_general": [1.0] * 4, "a_specific": [1.0] * 4,
            "threshold": [0.0] * 4, "theta_g_eap": [0.0] * 2,
            "theta_g_sd": [1.0] * 2, "category_counts": [1] * 8,
            "loglik_trace": [-4.0, -3.0], "n_iter": 1,
            "converged": False, "termination_reason": "max_iter_reached",
            "final_loglik_change": 1.0, "best_start": 0, "n_parameters": 12,
            "effective_device": "cpu", "estep_shards": [],
            "general_mean": 0.0, "general_sd": 1.0, "specific_sd": [1.0, 1.0],
        }

    monkeypatch.setattr(
        fitstats, "_core_module",
        lambda: SimpleNamespace(fit_bifactor_grm=binding, bifactor_oakes_se=binding, fit_bifactor_grm_fipc=binding),
    )
    try:
        common = dict(n_cat=2, n_specific=2, q_general=121, q_specific=241)
        if entry == "fit":
            result = bifactor.fit_bifactor_grm(
                y, specific_map, **common, max_iter=1, tol=1e-6,
                n_starts=1, seed=2**53 + 1, device="cpu",
            )
            assert result.converged is False
        elif entry == "oakes":
            result = bifactor.bifactor_oakes_se(
                np.ones(4), np.ones(4), np.zeros((4, 1)), y, specific_map,
                **common, fd_step=1e-4,
            )
            assert result.positive_definite is False
        else:
            result = bifactor.fit_bifactor_grm_fipc(
                y, specific_map, **common, anchor=np.ones(4, dtype=bool),
                fixed_a_general=np.ones(4), fixed_a_specific=np.ones(4),
                fixed_threshold=np.zeros((4, 1)), max_iter=1, tol=1e-6,
            )
            assert result.converged is False
    finally:
        assert len(calls) == 1
        assert specific_map.tobytes() == original_bytes
        assert (specific_map.strides, specific_map.flags.aligned, specific_map.flags.writeable) == original_flags
        assert (bytes(backing) if isinstance(backing, bytearray) else backing.tobytes()) == storage_bytes
        np.testing.assert_array_equal(specific_map, expected)
