"""Fail-closed tests for the Rust-to-Python many-facet result envelope."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

import fast_mlsirm.facets as facets
import fast_mlsirm.fitstats as fitstats


class _FakeCore:
    """Return one controlled native-shaped payload without numerical work."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def fit_facets(self, *args: object, **kwargs: object) -> dict[str, object]:
        return self.payload


class _CallbackKey(str):
    """Expose accidental result-key protocol execution after native return."""

    callbacks = 0

    def __hash__(self) -> int:
        type(self).callbacks += 1
        return super().__hash__()

    def __eq__(self, other: object) -> bool:
        type(self).callbacks += 1
        return super().__eq__(other)


def _responses() -> np.ndarray:
    return np.array([[[0], [1]], [[1], [0]]], dtype=np.float64)


def _valid_payload() -> dict[str, object]:
    return {
        "item_difficulty": np.array([-0.25, 0.25], dtype=np.float64),
        "rater_severity": np.array([0.0], dtype=np.float64),
        "thresholds": np.array([0.0], dtype=np.float64),
        "theta": np.array([-0.5, 0.5], dtype=np.float64),
        "loglik_trace": np.array([-5.0, -4.5], dtype=np.float64),
        "n_iter": 2,
        "converged": True,
        "connected": True,
        "n_parameters": 2,
    }


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("item_difficulty", [0.0]),
        ("rater_severity", [0.0, 1.0]),
        ("thresholds", [0.0, 1.0]),
        ("theta", [0.0]),
        ("loglik_trace", [float("nan")]),
        ("n_iter", True),
        ("n_iter", "1"),
        ("converged", 1),
        ("connected", "true"),
        ("n_parameters", 4.5),
    ],
)
def test_fit_facets_rejects_malformed_native_result(
    monkeypatch: pytest.MonkeyPatch, field: str, invalid: object
) -> None:
    payload = _valid_payload()
    payload[field] = invalid
    monkeypatch.setattr(fitstats, "_core_module", lambda: _FakeCore(payload))

    with pytest.raises((TypeError, ValueError), match="native fit_facets result"):
        facets.fit_facets(_responses(), n_cat=2, max_iter=5)


def test_fit_facets_rejects_missing_native_result_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_payload()
    del payload["connected"]
    monkeypatch.setattr(fitstats, "_core_module", lambda: _FakeCore(payload))

    with pytest.raises(ValueError, match="native fit_facets result"):
        facets.fit_facets(_responses(), n_cat=2, max_iter=5)


def test_fit_facets_rejects_extra_native_result_key_before_value_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_payload()
    payload["unexpected"] = None
    payload["item_difficulty"] = [object(), 0.0, 0.0]
    monkeypatch.setattr(fitstats, "_core_module", lambda: _FakeCore(payload))

    with pytest.raises(
        ValueError, match=r"native fit_facets result must contain exactly 9 keys"
    ):
        facets.fit_facets(_responses(), n_cat=2, max_iter=5)


def test_fit_facets_rejects_callback_bearing_native_key_without_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = _valid_payload()
    hostile_key = _CallbackKey("item_difficulty")
    payload: dict[str, object] = {hostile_key: valid["item_difficulty"]}
    for key, value in valid.items():
        if key != "item_difficulty":
            payload[key] = value
    _CallbackKey.callbacks = 0
    monkeypatch.setattr(fitstats, "_core_module", lambda: _FakeCore(payload))

    with pytest.raises(
        ValueError, match=r"native fit_facets result keys must be exact strings"
    ):
        facets.fit_facets(_responses(), n_cat=2, max_iter=5)

    assert _CallbackKey.callbacks == 0


def test_fit_facets_rejects_wrong_fixed_vector_length_before_entry_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_payload()
    payload["item_difficulty"] = [object(), 0.0, 0.0]
    monkeypatch.setattr(fitstats, "_core_module", lambda: _FakeCore(payload))

    with pytest.raises(
        ValueError, match=r"native fit_facets result item_difficulty must have length 2"
    ):
        facets.fit_facets(_responses(), n_cat=2, max_iter=5)


def test_fit_facets_rejects_overlong_loglik_trace_before_entry_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_payload()
    payload["loglik_trace"] = [object()] * 7
    monkeypatch.setattr(fitstats, "_core_module", lambda: _FakeCore(payload))

    with pytest.raises(
        ValueError, match=r"native fit_facets result loglik_trace exceeds length limit 6"
    ):
        facets.fit_facets(_responses(), n_cat=2, max_iter=5)


def test_fit_facets_rejects_lossy_native_integer_vector_narrowing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_payload()
    payload["item_difficulty"] = np.array([2**53 + 1, 0], dtype=np.int64)
    monkeypatch.setattr(fitstats, "_core_module", lambda: _FakeCore(payload))

    with pytest.raises(
        ValueError,
        match=r"native fit_facets result item_difficulty integer values must be exactly representable as float64",
    ):
        facets.fit_facets(_responses(), n_cat=2, max_iter=5)


def test_fit_facets_allows_large_exact_native_integer_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_payload()
    payload["item_difficulty"] = np.array([2**60, 0], dtype=np.int64)
    monkeypatch.setattr(fitstats, "_core_module", lambda: _FakeCore(payload))

    result = facets.fit_facets(_responses(), n_cat=2, max_iter=5)

    assert result.item_difficulty.tolist() == [float(2**60), 0.0]


def test_fit_facets_rejects_lossy_mixed_native_integer_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_payload()
    payload["item_difficulty"] = [np.uint64(2**53 + 1), np.int64(-1)]
    monkeypatch.setattr(fitstats, "_core_module", lambda: _FakeCore(payload))

    with pytest.raises(
        ValueError,
        match=r"native fit_facets result item_difficulty integer values must be exactly representable as float64",
    ):
        facets.fit_facets(_responses(), n_cat=2, max_iter=5)


def test_fit_facets_allows_exact_mixed_native_integer_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_payload()
    payload["item_difficulty"] = [np.uint64(2**60), np.int64(-1)]
    monkeypatch.setattr(fitstats, "_core_module", lambda: _FakeCore(payload))

    result = facets.fit_facets(_responses(), n_cat=2, max_iter=5)

    assert result.item_difficulty.tolist() == [float(2**60), -1.0]


@pytest.mark.parametrize(
    ("n_iter", "converged", "trace"),
    [
        (0, False, [-5.0]),
        (2, True, [-5.0]),
        (2, True, [-5.0, -4.5, -4.25]),
        (2, False, [-5.0]),
    ],
)
def test_fit_facets_rejects_impossible_iteration_evidence(
    monkeypatch: pytest.MonkeyPatch,
    n_iter: int,
    converged: bool,
    trace: list[float],
) -> None:
    payload = _valid_payload()
    payload["n_iter"] = n_iter
    payload["converged"] = converged
    payload["loglik_trace"] = trace
    monkeypatch.setattr(fitstats, "_core_module", lambda: _FakeCore(payload))

    with pytest.raises(ValueError, match="native fit_facets result"):
        facets.fit_facets(_responses(), n_cat=2, max_iter=5)


def test_fit_facets_allows_nonconverged_terminal_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_payload()
    payload["n_iter"] = 2
    payload["converged"] = False
    payload["loglik_trace"] = [-5.0, -4.5, -4.25]
    monkeypatch.setattr(fitstats, "_core_module", lambda: _FakeCore(payload))

    result = facets.fit_facets(_responses(), n_cat=2, max_iter=5)

    assert result.n_iter == 2
    assert not result.converged
    assert result.loglik_trace.tolist() == [-5.0, -4.5, -4.25]


def test_fit_facets_rejects_wrong_native_parameter_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_payload()
    payload["n_parameters"] = 3
    monkeypatch.setattr(fitstats, "_core_module", lambda: _FakeCore(payload))

    with pytest.raises(
        ValueError, match=r"native fit_facets result n_parameters must equal 2"
    ):
        facets.fit_facets(_responses(), n_cat=2, max_iter=5)


def test_fit_facets_returns_owned_arrays_from_valid_native_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_payload()
    expected = deepcopy(payload)
    monkeypatch.setattr(fitstats, "_core_module", lambda: _FakeCore(payload))

    result = facets.fit_facets(_responses(), n_cat=2, max_iter=5)

    assert np.array_equal(result.item_difficulty, expected["item_difficulty"])
    assert np.array_equal(result.rater_severity, expected["rater_severity"])
    assert np.array_equal(result.thresholds, expected["thresholds"])
    assert np.array_equal(result.theta, expected["theta"])
    assert np.array_equal(result.loglik_trace, expected["loglik_trace"])
    payload["item_difficulty"][0] = 99.0  # type: ignore[index]
    assert result.item_difficulty[0] == pytest.approx(-0.25)
