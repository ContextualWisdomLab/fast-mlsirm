"""Ownership regressions for many-facet response admission."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import fitstats
import fast_mlsirm.facets as facets


class _RecordingCore:
    """Record the exact package-owned evidence dispatched to the Rust boundary."""

    def __init__(self) -> None:
        self.yy: np.ndarray | None = None
        self.observed: np.ndarray | None = None

    def fit_facets(
        self,
        yy: np.ndarray,
        observed: np.ndarray,
        n_persons: int,
        n_items: int,
        n_raters: int,
        n_cat: int,
        q_theta: int,
        max_iter: int,
        tol: float,
    ) -> dict[str, object]:
        self.yy = yy.copy()
        self.observed = observed.copy()
        return {
            "item_difficulty": np.zeros(n_items),
            "rater_severity": np.zeros(n_raters),
            "thresholds": np.zeros(n_cat - 1),
            "theta": np.zeros(n_persons),
            "loglik_trace": np.array([0.0]),
            "n_iter": 1,
            "converged": True,
            "connected": True,
            "n_parameters": n_items + (n_raters - 1) + (n_cat - 2),
        }


def test_response_array_seals_exact_float64_ndarray() -> None:
    """Caller mutation after admission cannot redefine the admitted ndarray."""

    source = np.array(
        [
            [[0.0], [1.0]],
            [[1.0], [0.0]],
        ],
        dtype=np.float64,
    )
    expected = source.copy()

    admitted = facets._response_array(source)
    source[...] = 7.0

    assert admitted is not source
    np.testing.assert_array_equal(admitted, expected)


def test_response_array_seals_builtin_sequence_before_numpy_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NumPy materialization cannot reread a retained mutable response tree."""

    source = [
        [[0.0], [1.0]],
        [[1.0], [0.0]],
    ]
    expected = np.asarray(source).copy()
    original_asarray = facets.np.asarray

    def mutate_source_at_materialization(
        value: object, *args: object, **kwargs: object
    ) -> np.ndarray:
        if value is source:
            source[0][0][0] = 1.0
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(facets.np, "asarray", mutate_source_at_materialization)

    admitted = facets._response_array(source)

    np.testing.assert_array_equal(admitted, expected)


@pytest.mark.parametrize(
    ("source", "n_persons", "n_items", "n_raters", "message"),
    [
        (
            [[[0.0]]],
            2,
            1,
            1,
            "responses must be a 3-D persons x items x raters array",
        ),
        (
            [0.0],
            1,
            1,
            1,
            "responses must be a 3-D persons x items x raters array",
        ),
        (
            [[[0.0]]],
            1,
            2,
            1,
            "responses must be a 3-D persons x items x raters array",
        ),
        (
            [[0.0]],
            1,
            1,
            1,
            "responses must be a 3-D persons x items x raters array",
        ),
        (
            [[[0.0]]],
            1,
            1,
            2,
            "responses must be a 3-D persons x items x raters array",
        ),
        (
            [[[object()]]],
            1,
            1,
            1,
            "responses must be a numeric array",
        ),
    ],
)
def test_builtin_response_snapshot_replays_admitted_structure(
    source: list[object],
    n_persons: int,
    n_items: int,
    n_raters: int,
    message: str,
) -> None:
    """The owned tree rejects structural or scalar drift before NumPy use."""

    with pytest.raises(ValueError, match=message):
        facets._snapshot_builtin_response_tree(
            source,
            n_persons=n_persons,
            n_items=n_items,
            n_raters=n_raters,
        )


@pytest.mark.parametrize(
    "source",
    [
        [[[0.0], [1.0]]],
        (((0.0,), (1.0,)),),
    ],
)
def test_builtin_response_snapshot_is_independent_of_mutable_rows(
    source: list[object] | tuple[object, ...],
) -> None:
    """A valid exact-container tree is frozen into package-owned tuples."""

    snapshot = facets._snapshot_builtin_response_tree(
        source,
        n_persons=1,
        n_items=2,
        n_raters=1,
    )

    assert snapshot == (((0.0,), (1.0,)),)
    if type(source) is list:
        source[0][0][0] = 7.0
        assert snapshot == (((0.0,), (1.0,)),)


def test_fit_facets_dispatches_pre_mutation_ndarray_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation after response admission cannot change evidence sent to Rust."""

    source = np.array(
        [
            [[0.0], [1.0]],
            [[1.0], [0.0]],
        ],
        dtype=np.float64,
    )
    expected = source.copy()
    original_response_array = facets._response_array

    def mutate_source_after_admission(value: object) -> np.ndarray:
        admitted = original_response_array(value)
        source[...] = 7.0
        return admitted

    core = _RecordingCore()
    monkeypatch.setattr(facets, "_response_array", mutate_source_after_admission)
    monkeypatch.setattr(fitstats, "_core_module", lambda: core)

    result = facets.fit_facets(source, n_cat=2, q_theta=41, max_iter=10, tol=1e-6)

    assert result.converged is True
    assert core.yy is not None
    assert core.observed is not None
    np.testing.assert_array_equal(core.yy, expected.astype(np.int64).reshape(-1))
    np.testing.assert_array_equal(
        core.observed,
        np.ones(expected.size, dtype=np.bool_),
    )


@pytest.mark.parametrize(
    ("dtype", "message"),
    [
        (np.dtype(np.complex128), "responses must be real-valued"),
        (np.dtype("V4096"), "responses must be a numeric array"),
    ],
)
def test_response_array_rejects_invalid_ndarray_storage_before_copy(
    monkeypatch: pytest.MonkeyPatch,
    dtype: np.dtype,
    message: str,
) -> None:
    """Invalid exact ndarray storage fails before package copy allocation."""

    source = np.empty((1, 1, 1), dtype=dtype)
    copy_calls = 0

    def unexpected_copy(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal copy_calls
        copy_calls += 1
        raise AssertionError("invalid ndarray storage reached ownership copy")

    monkeypatch.setattr(facets.np, "array", unexpected_copy)

    with pytest.raises(ValueError, match=message):
        facets._response_array(source)

    assert copy_calls == 0


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((20_000_000,), "responses must be a 3-D persons x items x raters array"),
        ((0, 1, 1), "responses must contain at least one person, one item and one rater"),
        ((1, 0, 1), "responses must contain at least one person, one item and one rater"),
        ((1, 1, 0), "responses must contain at least one person, one item and one rater"),
    ],
)
def test_response_array_rejects_impossible_ndarray_shape_before_copy(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, ...],
    message: str,
) -> None:
    """Inert invalid shape metadata fails before package copy allocation."""

    source = np.broadcast_to(np.float64(0.0), shape)
    copy_calls = 0

    def unexpected_copy(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal copy_calls
        copy_calls += 1
        raise AssertionError("invalid ndarray shape reached ownership copy")

    monkeypatch.setattr(facets.np, "array", unexpected_copy)

    with pytest.raises(ValueError, match=message):
        facets._response_array(source)

    assert copy_calls == 0
