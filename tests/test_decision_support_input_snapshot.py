"""Regression coverage for package-owned decision evidence snapshots."""

import numpy as np
import pytest

import fast_mlsirm.decision_support as decision_support


def test_real_array_snapshot_does_not_alias_caller_owned_float64_ndarray() -> None:
    """Native-bound evidence must not remain writable through the caller source."""
    source = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    admitted = decision_support._real_array(
        source,
        name="action_utilities",
        ndim=2,
        max_axis0=decision_support.MAX_DECISION_ACTIONS,
        axis0_label="actions",
        expected_axis1=2,
    )

    source[0, 0] = 99.0

    assert admitted[0, 0] == 1.0


def test_real_array_rejects_shape_change_during_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The preflighted shape remains authoritative through materialization."""
    source = [[1.0, 0.0], [0.0, 1.0]]
    original_trusted_real_array = decision_support._trusted_real_array

    def mutate_before_materialization(value: object, name: str) -> np.ndarray:
        if value is source:
            source.append([0.5, 0.5])
        return original_trusted_real_array(value, name)

    monkeypatch.setattr(
        decision_support,
        "_trusted_real_array",
        mutate_before_materialization,
    )

    with pytest.raises(
        ValueError,
        match="action_utilities changed shape during materialization",
    ):
        decision_support._real_array(
            source,
            name="action_utilities",
            ndim=2,
            max_axis0=decision_support.MAX_DECISION_ACTIONS,
            axis0_label="actions",
            expected_axis1=2,
        )


def test_real_array_rejects_boolean_introduced_during_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late Boolean mutation cannot be normalized into numeric evidence."""
    source = [0.5, 0.5]
    original_trusted_real_array = decision_support._trusted_real_array

    def mutate_before_materialization(value: object, name: str) -> np.ndarray:
        if value is source:
            source[0] = True
        return original_trusted_real_array(value, name)

    monkeypatch.setattr(
        decision_support,
        "_trusted_real_array",
        mutate_before_materialization,
    )

    with pytest.raises(
        ValueError,
        match="state_probabilities must contain real numeric values, not booleans",
    ):
        decision_support._real_array(
            source,
            name="state_probabilities",
            ndim=1,
            max_axis0=decision_support.MAX_DECISION_STATES,
            axis0_label="states",
        )
