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


@pytest.mark.parametrize("entry", ["fit", "oakes", "fipc"])
@pytest.mark.parametrize("invalid", [0.5, -0.5, np.nan, np.inf])
def test_specific_map_rejects_noninteger_float_before_binding(
    monkeypatch: pytest.MonkeyPatch, entry: str, invalid: float,
) -> None:
    """Reject lossy casts before loader discovery; no numerical fit is executed.

    NumPy Developers (n.d., ndarray.astype, Examples) illustrates fractional
    truncation; numpy.isfinite (Parameters) identifies NaN/infinity. References:
    NumPy Developers. (n.d.). numpy.ndarray.astype. NumPy reference guide.
    NumPy Developers. (n.d.). numpy.isfinite. NumPy reference guide.
    """
    specific_map = np.array([0.0, invalid, 1.0, 1.0])
    original = specific_map.tobytes()
    loader_calls = []

    def unexpected_loader():
        loader_calls.append(True)
        raise RuntimeError("invalid specific_map reached core discovery")

    monkeypatch.setattr(fitstats, "_core_module", unexpected_loader)
    try:
        with pytest.raises(ValueError, match="specific_map entries must be"):
            _call_specific_map_entry(entry, specific_map)
    finally:
        assert specific_map.tobytes() == original
        assert not loader_calls


@pytest.mark.parametrize("entry", ["fit", "oakes", "fipc"])
def test_specific_map_accepts_integer_valued_float_without_changing_order(
    monkeypatch: pytest.MonkeyPatch, entry: str,
) -> None:
    """Preserve valid float maps at the recording boundary, not native output."""
    specific_map = np.array([0.0, 0.0, 1.0, 1.0])
    original = specific_map.tobytes()
    calls = []

    class BindingReached(Exception):
        """Stop after observing arguments without returning a fitted result."""

    def binding(*args):
        calls.append(args)
        item_map = args[5] if entry == "oakes" else args[2]
        np.testing.assert_array_equal(item_map, [0, 0, 1, 1])
        assert item_map.dtype == np.dtype(np.int64)
        assert item_map.flags.c_contiguous and item_map.flags.aligned
        raise BindingReached

    monkeypatch.setattr(
        fitstats, "_core_module",
        lambda: SimpleNamespace(fit_bifactor_grm=binding, bifactor_oakes_se=binding,
                                fit_bifactor_grm_fipc=binding),
    )
    try:
        with pytest.raises(BindingReached):
            _call_specific_map_entry(entry, specific_map)
    finally:
        assert len(calls) == 1
        assert specific_map.tobytes() == original


@pytest.mark.parametrize("entry", ["fit", "oakes", "fipc"])
@pytest.mark.parametrize("invalid", [float(2**63), float(-(2**63) - 2048)])
def test_specific_map_rejects_out_of_int64_float_before_binding(monkeypatch, entry, invalid):
    """Reject integral-valued floats outside the native signed-map range."""
    smap = np.array([0.0, 0.0, 1.0, invalid], dtype=np.float64)
    calls = []
    monkeypatch.setattr(fitstats, "_core_module", lambda: calls.append(True))
    with pytest.raises(ValueError, match="specific_map entries must"):
        _call_specific_map_entry(entry, smap)
    assert calls == []


@pytest.mark.parametrize("entry", ["fit", "oakes", "fipc"])
def test_specific_map_rejects_complex_dtype_before_binding(monkeypatch, entry):
    """A complex array is not a real-valued item-to-specific map."""
    smap = np.array([0, 0, 1, 1 + 0.5j], dtype=np.complex128)
    calls = []
    monkeypatch.setattr(fitstats, "_core_module", lambda: calls.append(True))
    with pytest.raises(ValueError, match="specific_map entries must"):
        _call_specific_map_entry(entry, smap)
    assert calls == []


@pytest.mark.parametrize("entry", ["fit", "oakes", "fipc"])
def test_specific_map_rejects_unsigned_wrap_before_binding(monkeypatch, entry):
    """Reject a uint64 value that would wrap to the valid -1 map sentinel.

    Basis: NumPy Developers (n.d., ndarray.astype, casting parameter) and
    NumPy Developers (n.d., numpy.iinfo, integer limits). References:
    NumPy Developers. (n.d.). numpy.ndarray.astype. NumPy reference guide.
    NumPy Developers. (n.d.). numpy.iinfo. NumPy reference guide.
    """
    specific_map = np.array([0, 0, 1, 2**64 - 1], dtype=np.uint64)
    original = specific_map.tobytes()
    loader_calls = []

    def unexpected_loader():
        loader_calls.append(True)
        raise RuntimeError("unsigned overflow reached core discovery")

    monkeypatch.setattr(fitstats, "_core_module", unexpected_loader)
    try:
        with pytest.raises(ValueError, match="specific_map entries must be"):
            _call_specific_map_entry(entry, specific_map)
    finally:
        assert specific_map.tobytes() == original
        assert not loader_calls


@pytest.mark.parametrize("entry", ["fit", "oakes", "fipc"])
def test_specific_map_preserves_valid_unsigned_values(monkeypatch, entry):
    """Keep valid unsigned maps lossless at the recording core boundary."""
    specific_map = np.array([0, 0, 1, 1], dtype=np.uint64)
    original = specific_map.tobytes()
    calls = []

    class BindingReached(Exception):
        """Stop after argument capture without numerical execution."""

    def binding(*args):
        calls.append(args)
        item_map = args[5] if entry == "oakes" else args[2]
        np.testing.assert_array_equal(item_map, [0, 0, 1, 1])
        assert item_map.dtype == np.dtype(np.int64)
        assert item_map.flags.c_contiguous and item_map.flags.aligned
        raise BindingReached

    monkeypatch.setattr(
        fitstats, "_core_module",
        lambda: SimpleNamespace(fit_bifactor_grm=binding, bifactor_oakes_se=binding,
                                fit_bifactor_grm_fipc=binding),
    )
    try:
        with pytest.raises(BindingReached):
            _call_specific_map_entry(entry, specific_map)
    finally:
        assert len(calls) == 1
        assert specific_map.tobytes() == original


def _call_specific_map_entry(entry: str, specific_map: np.ndarray) -> None:
    """Reach the actual wrapper using explicit controls and a recording core."""
    y = np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.int64)
    common = dict(n_cat=2, n_specific=2, q_general=121, q_specific=241)
    if entry == "fit":
        bifactor.fit_bifactor_grm(
            y, specific_map, **common, max_iter=1, tol=1e-6,
            n_starts=1, seed=20261004, device="cpu",
        )
    elif entry == "oakes":
        bifactor.bifactor_oakes_se(
            np.ones(4), np.ones(4), np.zeros((4, 1)), y, specific_map,
            **common, fd_step=1e-4,
        )
    else:
        bifactor.fit_bifactor_grm_fipc(
            y, specific_map, **common, anchor=np.ones(4, dtype=bool),
            fixed_a_general=np.ones(4), fixed_a_specific=np.ones(4),
            fixed_threshold=np.zeros((4, 1)), max_iter=1, tol=1e-6,
        )


@pytest.mark.parametrize('entry', ['fit', 'oakes', 'fipc'])
@pytest.mark.parametrize('invalid', [0.5, -0.5, float('nan'), float('inf'), -float('inf'), float(2**63), -float(2**64)])
def test_boxed_fraction_rejected_before_loader(monkeypatch, entry, invalid):
    """Reject a real fraction even when NumPy stores it as a boxed scalar."""
    values = np.array([0, invalid, 1, 1], dtype=object)
    before = tuple(id(v) for v in values)
    calls = []
    def loader():
        calls.append(True)
        raise RuntimeError('boxed fractional map reached core discovery')
    monkeypatch.setattr(fitstats, '_core_module', loader)
    try:
        with pytest.raises(ValueError, match='specific_map entries must'):
            _call_specific_map_entry(entry, values)
    finally:
        assert tuple(id(v) for v in values) == before
        assert calls == [], 'boxed fractional map reached core discovery'

@pytest.mark.parametrize('entry', ['fit', 'oakes', 'fipc'])
@pytest.mark.parametrize('dtype', ['integer', 'integral_float'])
def test_valid_boxed_numeric_map_transport_preserved(monkeypatch, entry, dtype):
    """Keep supported boxed integer values and integral floats lossless."""
    values = np.array([0, 0, 1, 1] if dtype == 'integer' else [0., 0., 1., 1.], dtype=object)
    before = tuple(id(v) for v in values)
    calls = []
    class BindingReached(Exception):
        """Terminate at a recording binding before any numerical execution."""
    def binding(*args):
        transported = args[5] if entry == 'oakes' else args[2]
        calls.append(transported.copy())
        np.testing.assert_array_equal(transported, [0, 0, 1, 1])
        assert transported.dtype == np.dtype(np.int64)
        assert transported.flags.c_contiguous and transported.flags.aligned
        raise BindingReached
    monkeypatch.setattr(fitstats, '_core_module', lambda: SimpleNamespace(
        fit_bifactor_grm=binding, bifactor_oakes_se=binding,
        fit_bifactor_grm_fipc=binding))
    try:
        with pytest.raises(BindingReached):
            _call_specific_map_entry(entry, values)
    finally:
        assert len(calls) == 1
        assert tuple(id(v) for v in values) == before


from decimal import Decimal
from fractions import Fraction

EXACT_NUMBER_OBSERVATIONS = []

@pytest.mark.parametrize('entry', ['fit', 'oakes', 'fipc'])
@pytest.mark.parametrize('invalid', [Decimal('0.5'), Decimal('-0.5'), Fraction(1, 2), Fraction(-1, 2)], ids=['decimal-positive-half', 'decimal-negative-half', 'fraction-positive-half', 'fraction-negative-half'])
def test_exact_boxed_fraction_rejects_before_loader(monkeypatch, entry, invalid):
    values = np.array([0, invalid, 1, 1], dtype=object)
    original_ids = tuple(id(v) for v in values)
    calls = []
    def loader():
        calls.append('core-discovery')
        raise RuntimeError('exact boxed fraction reached excluded native loader')
    monkeypatch.setattr(fitstats, '_core_module', loader)
    try:
        with pytest.raises(ValueError, match='specific_map entries must'):
            _call_specific_map_entry(entry, values)
    finally:
        EXACT_NUMBER_OBSERVATIONS.append({'entry': entry, 'kind': type(invalid).__name__, 'value': str(invalid), 'invalid': True, 'loader_calls': len(calls), 'caller_identity_preserved': tuple(id(v) for v in values) == original_ids})
        assert tuple(id(v) for v in values) == original_ids
        assert not calls, 'fraction was narrowed before map admission'

@pytest.mark.parametrize('entry', ['fit', 'oakes', 'fipc'])
@pytest.mark.parametrize('kind', ['decimal', 'fraction'])
def test_exact_boxed_integral_control_reaches_binding(monkeypatch, entry, kind):
    constructor = Decimal if kind == 'decimal' else Fraction
    values = np.array([constructor(0), constructor(0), constructor(1), constructor(1)], dtype=object)
    original_ids = tuple(id(v) for v in values)
    calls = []
    class BindingReached(Exception):
        pass
    def binding(*args):
        transported = args[5] if entry == 'oakes' else args[2]
        calls.append(transported.tolist())
        np.testing.assert_array_equal(transported, [0, 0, 1, 1])
        assert transported.dtype == np.dtype(np.int64)
        assert transported.flags.c_contiguous and transported.flags.aligned
        raise BindingReached
    monkeypatch.setattr(fitstats, '_core_module', lambda: SimpleNamespace(fit_bifactor_grm=binding, bifactor_oakes_se=binding, fit_bifactor_grm_fipc=binding))
    try:
        with pytest.raises(BindingReached):
            _call_specific_map_entry(entry, values)
    finally:
        EXACT_NUMBER_OBSERVATIONS.append({'entry': entry, 'kind': kind, 'invalid': False, 'binding_calls': len(calls), 'transported': calls, 'caller_identity_preserved': tuple(id(v) for v in values) == original_ids})
        assert len(calls) == 1
        assert tuple(id(v) for v in values) == original_ids
