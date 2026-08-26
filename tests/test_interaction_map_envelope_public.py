"""Public binding contract for the Rust-owned residual interaction-map envelope."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import (
    RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
    ResidualInteractionMapEnvelope,
    residual_interaction_map,
    residual_interaction_map_envelope,
)


class _HostileArrayProvider:
    """Caller-owned array protocol that must not execute during control admission."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __array__(self, *_args: object, **_kwargs: object) -> np.ndarray:
        self.calls.append("array")
        raise AssertionError("caller __array__ must not execute")


def test_public_envelope_exposes_rust_owned_identity_and_map_evidence() -> None:
    """One public call returns opaque identities and the complete Rust result envelope."""
    result = residual_interaction_map_envelope(
        np.array([[2.0, np.nan], [1.0, 2.0]], dtype=np.float64),
        np.ones((2, 2), dtype=np.float64),
        person_ids=["person-a", "person-b"],
        item_ids=("item-a", "item-b"),
        axis_count=1,
    )

    assert isinstance(result, ResidualInteractionMapEnvelope)
    assert result.schema_version == RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION
    assert result.algorithm_id == "gabriel-complete-case-symmetric-residual-map.v1"
    assert result.calculation_provenance == "mlsirm-core::interaction_map::residual_interaction_map"
    assert result.requested_axis_count == 1
    assert result.cell_extrema_tie_policy == "lexicographic-first-original-index"
    assert result.finite_value_status is True
    assert result.retained_person_ids == ("person-b",)
    assert result.retained_item_ids == ("item-a", "item-b")
    assert result.scored_person_count == 2
    assert result.scored_item_count == 2
    assert result.map_person_count == 1
    assert result.map_item_count == 2
    assert result.incomplete_person_count == 1
    assert result.incomplete_item_count == 0
    np.testing.assert_array_equal(result.person_indices, [1])
    np.testing.assert_array_equal(result.item_indices, [0, 1])
    np.testing.assert_array_equal(result.observed, [[1.0, 2.0]])
    np.testing.assert_array_equal(result.expected, [[1.0, 1.0]])
    assert result.person_coordinates.shape == (1, 1)
    assert result.item_coordinates.shape == (2, 1)
    assert result.residual.shape == (1, 2)
    assert result.distance.shape == (1, 2)


def test_public_envelope_matches_existing_rust_map_numerics() -> None:
    """The new binding marshals the existing Rust map without numerical recomputation."""
    observed = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    expected = np.ones((2, 2), dtype=np.float64)

    existing = residual_interaction_map(observed, expected, axis_count=2)
    envelope = residual_interaction_map_envelope(
        observed,
        expected,
        person_ids=["person-a", "person-b"],
        item_ids=["item-a", "item-b"],
        axis_count=2,
    )

    np.testing.assert_array_equal(envelope.person_indices, existing.person_indices)
    np.testing.assert_array_equal(envelope.item_indices, existing.item_indices)
    np.testing.assert_allclose(envelope.person_coordinates, existing.person_coordinates)
    np.testing.assert_allclose(envelope.item_coordinates, existing.item_coordinates)
    np.testing.assert_allclose(envelope.singular_values, existing.singular_values)
    np.testing.assert_allclose(envelope.axis_shares, existing.axis_shares)
    np.testing.assert_allclose(envelope.residual, existing.residual)
    np.testing.assert_allclose(envelope.distance, existing.distance)
    np.testing.assert_allclose(envelope.reconstruction, existing.reconstruction)
    np.testing.assert_allclose(envelope.unexplained, existing.unexplained)
    np.testing.assert_allclose(envelope.cross_share, existing.cross_share, equal_nan=True)


def test_schema_and_identifier_carriers_fail_before_caller_matrix_protocols() -> None:
    """Public controls and opaque IDs are sealed before any caller array protocol can execute."""
    hostile = _HostileArrayProvider()

    with pytest.raises(ValueError, match="schema version"):
        residual_interaction_map_envelope(
            hostile,  # type: ignore[arg-type]
            hostile,  # type: ignore[arg-type]
            person_ids=["person-a"],
            item_ids=["item-a"],
            axis_count=1,
            schema_version="fast-mlsirm.residual-interaction-map.v2",
        )
    assert hostile.calls == []

    with pytest.raises(ValueError, match="person_ids"):
        residual_interaction_map_envelope(
            hostile,  # type: ignore[arg-type]
            hostile,  # type: ignore[arg-type]
            person_ids=("person-a", object()),  # type: ignore[arg-type]
            item_ids=["item-a"],
            axis_count=1,
        )
    assert hostile.calls == []
