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
        "cat_probs": np.array([[0.0, 0.0, 1.0, 0.0, 0.0]] * 2).reshape(-1),
    }


def _core_returning(result: object) -> object:
    """Return a fake native module exposing one deterministic payload."""

    class Core:
        @staticmethod
        def py_eb_mh_dif(mh: np.ndarray, se: np.ndarray) -> object:
            del mh, se
            return result

    return Core()


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


def test_too_few_items_fail_before_dense_float64_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one-item domain is known from inert metadata before dense conversion."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    def unexpected_contiguous(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("one-item evidence reached contiguous float64 conversion")

    monkeypatch.setattr(ebdif.np, "ascontiguousarray", unexpected_contiguous)
    mh = np.broadcast_to(np.array([0.1], dtype=np.float64), (1,))
    se = np.broadcast_to(np.array([0.3], dtype=np.float64), (1,))

    with pytest.raises(ValueError, match="need at least 2 items"):
        ebdif.eb_mh_dif(mh, se)


def test_mismatched_lengths_fail_before_dense_float64_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unequal exact carrier lengths fail before either vector is densely converted."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    def unexpected_contiguous(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("length-mismatched evidence reached contiguous float64 conversion")

    monkeypatch.setattr(ebdif.np, "ascontiguousarray", unexpected_contiguous)
    mh = np.broadcast_to(np.array([0.1], dtype=np.float64), (2,))
    se = np.broadcast_to(np.array([0.3], dtype=np.float64), (3,))

    with pytest.raises(ValueError, match="mh and se must have the same length"):
        ebdif.eb_mh_dif(mh, se)


def test_malformed_first_carrier_keeps_type_precedence_before_length_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structural count preflight must not hide the first carrier's type error."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    malformed = np.array([object(), object()], dtype=object)
    longer = np.broadcast_to(np.array([0.3], dtype=np.float64), (3,))

    with pytest.raises(ValueError, match="mh must be a numeric array"):
        ebdif.eb_mh_dif(malformed, longer)


def test_lossy_integer_evidence_fails_before_core_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scientific evidence must not change when narrowed to the Rust f64 boundary."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    lossy = 2**53 + 1

    with pytest.raises(ValueError, match="mh entries must be exactly representable as float64"):
        ebdif.eb_mh_dif([lossy, 0], [0.3, 0.4])

    with pytest.raises(ValueError, match="se entries must be exactly representable as float64"):
        ebdif.eb_mh_dif([0.1, -0.2], np.array([lossy, 2**53], dtype=np.int64))


def test_wider_numpy_float_evidence_is_lossless_or_rejected_before_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wider concrete NumPy float may reach Rust only when binary64 preserves it."""

    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("platform longdouble does not exceed binary64 precision")

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    lossy = np.nextafter(np.longdouble(1.0), np.longdouble(2.0))

    with pytest.raises(ValueError, match="mh entries must be exactly representable as float64"):
        ebdif.eb_mh_dif(np.array([lossy, np.longdouble(0.0)]), [0.3, 0.4])


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


def test_exact_large_integer_boundary_preserves_rust_marshalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exactly representable large integer evidence remains supported."""

    captured: dict[str, np.ndarray] = {}

    class Core:
        @staticmethod
        def py_eb_mh_dif(mh: np.ndarray, se: np.ndarray) -> dict[str, object]:
            captured["mh"] = mh.copy()
            captured["se"] = se.copy()
            return _result()

    monkeypatch.setattr(fitstats, "_core_module", lambda: Core())
    exact = 2**53
    ebdif.eb_mh_dif([exact, 0], np.array([exact, 1], dtype=np.int64))

    np.testing.assert_array_equal(captured["mh"], np.array([float(exact), 0.0]))
    np.testing.assert_array_equal(captured["se"], np.array([float(exact), 1.0]))


def test_native_result_mapping_subclass_fails_without_mapping_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A foreign mapping carrier cannot execute keyed-access callbacks."""

    callbacks = 0

    class HostileDict(dict):
        def __getitem__(self, key):  # noqa: ANN001, ANN204
            nonlocal callbacks
            callbacks += 1
            return super().__getitem__(key)

    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: _core_returning(HostileDict(_result())),
    )

    with pytest.raises(RuntimeError, match="invalid EBDIF Rust result payload"):
        ebdif.eb_mh_dif([0.1, -0.2], [0.3, 0.4])

    assert callbacks == 0


def test_native_result_scalar_protocol_fails_without_conversion_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A foreign scalar cannot execute __float__ during public marshalling."""

    callbacks = 0

    class HostileFloat:
        def __float__(self) -> float:
            nonlocal callbacks
            callbacks += 1
            return 0.0

    result = _result()
    result["mu"] = HostileFloat()
    monkeypatch.setattr(fitstats, "_core_module", lambda: _core_returning(result))

    with pytest.raises(RuntimeError, match="invalid EBDIF Rust result payload"):
        ebdif.eb_mh_dif([0.1, -0.2], [0.3, 0.4])

    assert callbacks == 0


def test_native_result_array_protocol_fails_without_array_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A foreign vector cannot execute __array__ during public marshalling."""

    callbacks = 0

    class HostileArray:
        def __array__(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN204
            nonlocal callbacks
            callbacks += 1
            return np.zeros(2)

    result = _result()
    result["weight"] = HostileArray()
    monkeypatch.setattr(fitstats, "_core_module", lambda: _core_returning(result))

    with pytest.raises(RuntimeError, match="invalid EBDIF Rust result payload"):
        ebdif.eb_mh_dif([0.1, -0.2], [0.3, 0.4])

    assert callbacks == 0


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("weight", np.zeros(3, dtype=np.float64)),
        ("post_mean", np.zeros(1, dtype=np.float64)),
        ("post_var", np.zeros((2, 1), dtype=np.float64)),
        ("cat_probs", np.zeros(9, dtype=np.float64)),
        ("mu", float("nan")),
        ("weight", np.array([0.0, np.inf], dtype=np.float64)),
    ],
)
def test_native_result_structure_and_finiteness_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    """Malformed or non-finite native evidence never reaches public marshalling."""

    result = _result()
    result[field] = replacement
    monkeypatch.setattr(fitstats, "_core_module", lambda: _core_returning(result))

    with pytest.raises(RuntimeError, match="invalid EBDIF Rust result payload"):
        ebdif.eb_mh_dif([0.1, -0.2], [0.3, 0.4])
