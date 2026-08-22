"""Trust-boundary regressions for Lord flexilevel wrappers."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import exposure


class _ArrayTrap:
    """Fail if a rejected control reaches caller array materialization."""

    def __array__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("caller array materialization executed")


class _FloatTrap:
    """Fail if object storage reaches per-element numeric conversion."""

    callbacks = 0

    def __float__(self) -> float:
        type(self).callbacks += 1
        raise AssertionError("caller float conversion executed")


class _CoreTrap:
    """Fail if invalid flexilevel evidence reaches native dispatch."""

    @staticmethod
    def py_flexilevel_administer(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("native flexilevel administration executed")

    @staticmethod
    def py_flexilevel_score_distribution(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("native flexilevel distribution executed")


@pytest.mark.parametrize(
    ("n_persons", "n_items", "match"),
    [
        (1, 4, "n_items must be odd"),
        (np.iinfo(np.uintp).max + 1, 3, "n_persons out of range"),
        (np.iinfo(np.uintp).max, 3, "n_persons \* n_items overflows"),
    ],
)
def test_flexilevel_invalid_size_controls_fail_before_response_work(
    n_persons: int,
    n_items: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        exposure.flexilevel_administer(
            _ArrayTrap(), n_persons=n_persons, n_items=n_items
        )


def test_flexilevel_response_size_mismatch_fails_before_native_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fast_mlsirm

    monkeypatch.setattr(fast_mlsirm, "_core", _CoreTrap())
    with pytest.raises(ValueError, match="responses has 2 entries, expected 3"):
        exposure.flexilevel_administer(
            np.array([0, 1], dtype=np.uint8), n_persons=1, n_items=3
        )


@pytest.mark.parametrize(
    "responses",
    [
        np.array([0.0 + 1.0j, 1.0, 0.0]),
        np.array(["0", "1", "0"]),
        np.array([0.0, _FloatTrap(), 1.0], dtype=object),
    ],
)
def test_flexilevel_invalid_response_storage_fails_before_native_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    responses: np.ndarray,
) -> None:
    import fast_mlsirm

    _FloatTrap.callbacks = 0
    monkeypatch.setattr(fast_mlsirm, "_core", _CoreTrap())
    with pytest.raises(ValueError, match="responses must be a real numeric array"):
        exposure.flexilevel_administer(responses, n_persons=1, n_items=3)
    assert _FloatTrap.callbacks == 0


def test_flexilevel_preserves_flat_and_matrix_binary_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fast_mlsirm

    calls: list[tuple[np.ndarray, int, int]] = []

    class _Core:
        @staticmethod
        def py_flexilevel_administer(resp, n_persons, n_items):  # noqa: ANN001
            calls.append((resp.copy(), n_persons, n_items))
            return {
                "n_administered": 2,
                "items": [1, 2, 1, 0],
                "number_right": [2, 0],
                "is_red": [0, 1],
                "score": [2.0, 0.5],
            }

    monkeypatch.setattr(fast_mlsirm, "_core", _Core())
    expected = np.array([0, 1, 1, 1, 0, 0], dtype=np.uint8)
    for responses in (
        expected.astype(np.float32),
        expected.reshape(2, 3).astype(bool),
    ):
        exposure.flexilevel_administer(responses, n_persons=2, n_items=3)
        payload, n_persons, n_items = calls[-1]
        np.testing.assert_array_equal(payload, expected)
        assert payload.dtype == np.uint8
        assert payload.flags.c_contiguous
        assert (n_persons, n_items) == (2, 3)


@pytest.mark.parametrize(
    ("p", "match"),
    [
        (np.array([0.2, 0.8]), "p length must be odd and >= 3"),
        (np.array([0.2 + 0.1j, 0.5, 0.8]), "p must be a real numeric array"),
        (np.array(["0.2", "0.5", "0.8"]), "p must be a real numeric array"),
        (np.array([0.2, _FloatTrap(), 0.8], dtype=object), "p must be a real numeric array"),
        (np.array([0.2, np.inf, 0.8]), "p must contain finite probabilities in \\[0, 1\\]"),
        (np.array([0.2, -0.1, 0.8]), "p must contain finite probabilities in \\[0, 1\\]"),
        (np.array([0.2, 1.1, 0.8]), "p must contain finite probabilities in \\[0, 1\\]"),
    ],
)
def test_flexilevel_probability_admission_fails_before_native_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    p: np.ndarray,
    match: str,
) -> None:
    import fast_mlsirm

    _FloatTrap.callbacks = 0
    monkeypatch.setattr(fast_mlsirm, "_core", _CoreTrap())
    with pytest.raises(ValueError, match=match):
        exposure.flexilevel_score_distribution(p)
    assert _FloatTrap.callbacks == 0


def test_flexilevel_probability_payload_preserves_real_numeric_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fast_mlsirm

    calls: list[np.ndarray] = []

    class _Core:
        @staticmethod
        def py_flexilevel_score_distribution(p):  # noqa: ANN001
            calls.append(p.copy())
            return {
                "scores": [0.5, 1.0, 1.5, 2.0],
                "probs": [0.1, 0.2, 0.3, 0.4],
                "mean": 1.5,
                "variance": 0.25,
            }

    monkeypatch.setattr(fast_mlsirm, "_core", _Core())
    exposure.flexilevel_score_distribution(np.array([0.2, 0.5, 0.8], dtype=np.float32))
    assert len(calls) == 1
    assert calls[0].dtype == np.float64
    assert calls[0].flags.c_contiguous
    np.testing.assert_allclose(calls[0], [0.2, 0.5, 0.8], rtol=1e-6)
