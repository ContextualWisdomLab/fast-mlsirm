"""Tests for expected total graded scale score composition."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from fast_mlsirm.estimators.mmle import equal_probability_normal_nodes
from fast_mlsirm.graded_item import (
    compute_expected_graded_item_score,
    expected_graded_scale_score,
)

N_NODES = 2
THETA = np.array([0.30, 0.0, 0.0, 1.10])
FACTOR_VARIANCES = np.array([1.0, 2.25, 4.0, 1.0])
SLOPES = np.array(
    [
        [1.20, 0.85, 0.40, -0.25],
        [0.95, 0.70, 0.55, -0.15],
    ]
)
THRESHOLDS = np.array(
    [
        [1.05, 0.10, -0.70],
        [0.90, -0.05, -0.85],
    ]
)
INTEGRATE_COLUMNS = [(1,), (1, 2)]


def _scaled_nodes_weights(column: int) -> tuple[np.ndarray, np.ndarray]:
    base_nodes, base_weights = equal_probability_normal_nodes(N_NODES)
    scale = float(np.sqrt(FACTOR_VARIANCES[column]))
    return base_nodes * scale, base_weights


def _per_item_sum(theta: np.ndarray = THETA) -> float:
    total = 0.0
    for item_index, columns in enumerate(INTEGRATE_COLUMNS):
        nodes = tuple(_scaled_nodes_weights(column)[0] for column in columns)
        weights = tuple(_scaled_nodes_weights(column)[1] for column in columns)
        total += compute_expected_graded_item_score(
            SLOPES[item_index],
            THRESHOLDS[item_index],
            theta,
            columns,
            nodes,
            weights,
        )
    return total


def test_total_equals_sum_of_per_item_calls() -> None:
    reference = _per_item_sum()
    result = expected_graded_scale_score(
        SLOPES,
        THRESHOLDS,
        THETA,
        INTEGRATE_COLUMNS,
        FACTOR_VARIANCES,
        n_nodes=N_NODES,
    )
    assert result == reference


def test_scaled_nodes_not_unit_variance() -> None:
    unscaled_reference = 0.0
    for item_index, columns in enumerate(INTEGRATE_COLUMNS):
        base_nodes, base_weights = equal_probability_normal_nodes(N_NODES)
        nodes = tuple(base_nodes for _ in columns)
        weights = tuple(base_weights for _ in columns)
        unscaled_reference += compute_expected_graded_item_score(
            SLOPES[item_index],
            THRESHOLDS[item_index],
            THETA,
            columns,
            nodes,
            weights,
        )

    scaled_result = expected_graded_scale_score(
        SLOPES,
        THRESHOLDS,
        THETA,
        INTEGRATE_COLUMNS,
        FACTOR_VARIANCES,
        n_nodes=N_NODES,
    )
    assert scaled_result != unscaled_reference


def test_fixed_general_factor_preserved() -> None:
    baseline = expected_graded_scale_score(
        SLOPES,
        THRESHOLDS,
        THETA,
        INTEGRATE_COLUMNS,
        FACTOR_VARIANCES,
        n_nodes=N_NODES,
    )
    shifted_theta = THETA.copy()
    shifted_theta[0] = 1.75
    shifted = expected_graded_scale_score(
        SLOPES,
        THRESHOLDS,
        shifted_theta,
        INTEGRATE_COLUMNS,
        FACTOR_VARIANCES,
        n_nodes=N_NODES,
    )
    assert shifted != baseline
    assert shifted == _per_item_sum(shifted_theta)


def test_one_and_two_column_items_integrated() -> None:
    one_column = expected_graded_scale_score(
        SLOPES[:1],
        THRESHOLDS[:1],
        THETA,
        INTEGRATE_COLUMNS[:1],
        FACTOR_VARIANCES,
        n_nodes=N_NODES,
    )
    two_column_only = expected_graded_scale_score(
        SLOPES[1:],
        THRESHOLDS[1:],
        THETA,
        INTEGRATE_COLUMNS[1:],
        FACTOR_VARIANCES,
        n_nodes=N_NODES,
    )
    combined = expected_graded_scale_score(
        SLOPES,
        THRESHOLDS,
        THETA,
        INTEGRATE_COLUMNS,
        FACTOR_VARIANCES,
        n_nodes=N_NODES,
    )
    assert combined == one_column + two_column_only
