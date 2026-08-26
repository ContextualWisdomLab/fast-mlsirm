"""Fail-closed payload contract for the Rust interaction-map envelope."""

from __future__ import annotations

from importlib.metadata import version as distribution_version

import numpy as np
import pytest

import fast_mlsirm.interaction_map_envelope as envelope_module


class _FakeCore:
    def __init__(self, result: dict[str, object]) -> None:
        self._result = result

    def residual_interaction_map_envelope(self, *_args: object) -> dict[str, object]:
        result = dict(self._result)
        result.setdefault("input_digest", _args[1])
        return result


class _HostileInt:
    def __init__(self) -> None:
        self.calls = 0

    def __int__(self) -> int:
        self.calls += 1
        raise AssertionError("caller conversion must not run")


def _valid_payload() -> dict[str, object]:
    return {
        "schema_version": "fast-mlsirm.residual-interaction-map.v1",
        "algorithm_id": "gabriel-complete-case-symmetric-residual-map.v1",
        "implementation_version": distribution_version("fast-mlsirm"),
        "calculation_provenance": "mlsirm-core::interaction_map::residual_interaction_map",
        "requested_axis_count": 1,
        "cell_extrema_tie_policy": "lexicographic-first-original-index",
        "finite_value_status": True,
        "retained_person_ids": ["person-a"],
        "retained_item_ids": ["item-a"],
        "closest_cell_ids": ("person-a", "item-a"),
        "farthest_cell_ids": ("person-a", "item-a"),
        "person_indices": [0],
        "item_indices": [0],
        "scored_person_count": 1,
        "scored_item_count": 1,
        "effective_rank": 1,
        "map_person_count": 1,
        "map_item_count": 1,
        "incomplete_person_count": 0,
        "incomplete_item_count": 0,
        "closest_cell": (0, 0),
        "farthest_cell": (0, 0),
        "person_coordinates": [1.0],
        "item_coordinates": [1.0],
        "singular_values": [1.0],
        "axis_shares": [1.0],
        "observed": [1.0],
        "expected": [0.0],
        "residual": [1.0],
        "distance": [0.0],
        "reconstruction": [1.0],
        "unexplained": [0.0],
        "cross_share": [0.0],
    }


def _call(monkeypatch: pytest.MonkeyPatch, payload: dict[str, object]):
    monkeypatch.setattr(envelope_module, "interaction_map_core", lambda: _FakeCore(payload))
    return envelope_module.residual_interaction_map_envelope(
        np.ones((1, 1), dtype=np.float64),
        np.zeros((1, 1), dtype=np.float64),
        person_ids=["person-a"],
        item_ids=["item-a"],
        axis_count=1,
    )


def test_current_rust_payload_shape_remains_marshallable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _call(monkeypatch, _valid_payload())

    assert result.map_person_count == 1
    assert result.map_item_count == 1
    assert len(result.input_digest) == 64
    assert result.retained_person_ids == ("person-a",)
    assert result.retained_item_ids == ("item-a",)
    np.testing.assert_array_equal(result.person_indices, [0])
    np.testing.assert_array_equal(result.item_indices, [0])
    np.testing.assert_allclose(result.residual, [[1.0]])


def test_foreign_count_object_is_rejected_without_int_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_payload()
    hostile = _HostileInt()
    payload["map_person_count"] = hostile

    with pytest.raises(RuntimeError, match="map_person_count"):
        _call(monkeypatch, payload)

    assert hostile.calls == 0


def test_nonfinite_rust_payload_is_rejected_before_public_marshalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_payload()
    payload["residual"] = [float("nan")]

    with pytest.raises(RuntimeError, match="residual.*non-finite"):
        _call(monkeypatch, payload)


def test_rust_payload_length_mismatch_has_stable_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_payload()
    payload["distance"] = []

    with pytest.raises(RuntimeError, match="distance.*length"):
        _call(monkeypatch, payload)