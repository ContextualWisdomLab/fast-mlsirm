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


@pytest.mark.parametrize(
    "entry, field",
    [
        ("oakes", "a_general"), ("oakes", "a_specific"), ("oakes", "threshold"),
        ("fipc", "anchor"), ("fipc", "fixed_a_general"),
        ("fipc", "fixed_a_specific"), ("fipc", "fixed_threshold"),
    ],
)
@pytest.mark.parametrize(
    "layout", ["contiguous", "positive_stride", "negative_stride", "column", "readonly"],
)
def test_item_parameter_transport_preserves_caller_storage(
    monkeypatch: pytest.MonkeyPatch, entry: str, field: str, layout: str,
) -> None:
    """Check item/anchor buffers at the same documented C/A transport boundary.

    Basis: NumPy Developers (n.d., numpy.require, Parameters and Notes) and
    rust-numpy contributors (n.d., PyReadonlyArray::as_slice), referenced in
    this module. This records wrapper arguments, not native numerical output.
    """
    _check_item_parameter_transport(monkeypatch, entry, field, layout)


@pytest.mark.parametrize(
    "entry, field",
    [
        ("oakes", "a_general"), ("oakes", "a_specific"), ("oakes", "threshold"),
        ("fipc", "fixed_a_general"), ("fipc", "fixed_a_specific"),
        ("fipc", "fixed_threshold"),
    ],
)
def test_item_parameter_transport_aligns_unaligned_float_buffer(
    monkeypatch: pytest.MonkeyPatch, entry: str, field: str,
) -> None:
    """Observe C/A storage per module API references, without native admission."""
    _check_item_parameter_transport(monkeypatch, entry, field, "unaligned")


def _check_item_parameter_transport(
    monkeypatch: pytest.MonkeyPatch, entry: str, field: str, layout: str,
) -> None:
    """Use actual wrapper source and a recording seam with byte-preservation checks.

    Basis: NumPy Developers (n.d., numpy.require, Parameters and Notes),
    cited in the module. No estimator, information matrix or fit is computed.
    """
    ag = np.array([1.0, -1.0, 1.25, -0.75], dtype=np.float64)
    as_ = np.array([0.5, 0.625, -0.5, -0.625], dtype=np.float64)
    th = np.array([[1.5, -0.5], [1.25, -0.75], [1.0, -1.0], [0.75, -1.25]])
    if entry == "oakes":
        expected = {"a_general": ag, "a_specific": as_, "threshold": th}
        slots = {"a_general": 0, "a_specific": 1, "threshold": 2}
    else:
        expected = {"anchor": np.array([True, False, True, False]),
                    "fixed_a_general": ag, "fixed_a_specific": as_, "fixed_threshold": th}
        slots = {"anchor": 7, "fixed_a_general": 8, "fixed_a_specific": 9, "fixed_threshold": 10}
    inputs = {name: value.copy() for name, value in expected.items()}
    target = expected[field]
    if layout == "positive_stride":
        backing = np.repeat(target.reshape(-1), 2)
        inputs[field] = backing[::2].reshape(target.shape)
    elif layout == "negative_stride":
        backing = target.reshape(-1)[::-1].copy()
        inputs[field] = backing[::-1].reshape(target.shape)
    elif layout == "column":
        backing = np.column_stack([target.reshape(-1), target.reshape(-1)])
        inputs[field] = backing[:, :1].reshape(target.shape)
    elif layout == "unaligned":
        backing = bytearray(target.nbytes + 1)
        inputs[field] = np.ndarray(target.shape, dtype=target.dtype, buffer=backing, offset=1)
        inputs[field][:] = target
        assert not inputs[field].flags.aligned
    else:
        backing = inputs[field]
        if layout == "readonly":
            inputs[field].flags.writeable = False
    snapshots = {name: (value.tobytes(), value.shape, value.strides,
                        value.flags.aligned, value.flags.writeable)
                 for name, value in inputs.items()}
    backing_bytes = bytes(backing) if isinstance(backing, bytearray) else backing.tobytes()
    calls = []

    def binding(*args: object) -> dict[str, object]:
        """Record slice-safe bytes and controls; return explicit test-only outcome."""
        calls.append(args)
        for name, slot in slots.items():
            observed = args[slot]
            assert observed.dtype == expected[name].dtype
            assert observed.ndim == 1
            assert observed.tobytes() == expected[name].reshape(-1).tobytes(), name
            assert observed.flags.c_contiguous, f"{name} transport is not contiguous"
            assert observed.flags.aligned, f"{name} transport is not aligned"
        if entry == "oakes":
            assert args[6:] == (2, 4, 2, 3, 121, 241, 1e-4)
            return {"labels": ["fixture"], "information": [1.0], "vcov": None,
                    "se": None, "positive_definite": False, "non_pd_reason": "fixture"}
        assert args[3:7] == (2, 4, 2, 3)
        assert args[11:] == (121, 241, 1, 1e-6, 10, 1e-8, False)
        return {"a_general": ag.tolist(), "a_specific": as_.tolist(),
                "threshold": th.reshape(-1).tolist(), "general_mean": 0.0,
                "general_sd": 1.0, "specific_sd": [1.0, 1.0],
                "theta_g_eap": [0.0, 0.0], "theta_g_sd": [1.0, 1.0],
                "category_counts": [1] * 12, "loglik_trace": [-4.0, -3.0],
                "n_iter": 1, "converged": False, "termination_reason": "max_iter_reached",
                "final_loglik_change": 1.0, "n_parameters": 12}

    monkeypatch.setattr(
        fitstats, "_core_module",
        lambda: SimpleNamespace(bifactor_oakes_se=binding, fit_bifactor_grm_fipc=binding),
    )
    y = np.array([[0, 1, 2, 0], [1, 2, 0, 1]])
    smap = np.array([0, 0, 1, 1])
    try:
        common = dict(n_cat=3, n_specific=2, q_general=121, q_specific=241)
        if entry == "oakes":
            result = bifactor.bifactor_oakes_se(
                **inputs, responses=y, specific_map=smap, **common, fd_step=1e-4,
            )
            assert result.positive_definite is False
            assert result.vcov is None and result.se is None
        else:
            result = bifactor.fit_bifactor_grm_fipc(
                responses=y, specific_map=smap, **inputs, **common, max_iter=1, tol=1e-6,
            )
            assert result.converged is False
            assert result.a_general.tobytes() == ag.tobytes()
            assert result.a_specific.tobytes() == as_.tobytes()
            assert result.threshold.tobytes() == th.tobytes()
    finally:
        assert len(calls) == 1
        for name, value in inputs.items():
            assert (value.tobytes(), value.shape, value.strides,
                    value.flags.aligned, value.flags.writeable) == snapshots[name]
        assert (bytes(backing) if isinstance(backing, bytearray) else backing.tobytes()) == backing_bytes
