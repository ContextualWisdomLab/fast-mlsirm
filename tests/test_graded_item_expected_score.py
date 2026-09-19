"""Hand-checked integration for one graded item's expected category score."""

from __future__ import annotations

import numpy as np

from fast_mlsirm.estimators.marginal import compute_grm_category_logprobs
from fast_mlsirm.graded_item import compute_expected_graded_item_score

SLOPE = np.array([1.20, 0.85, 0.40, -0.25])
THETA = np.array([0.30, 0.0, 0.0, 1.10])
THRESHOLDS = np.array([1.05, 0.10, -0.70])
NODES = np.array([-0.80, 0.60])
WEIGHTS = np.array([0.35, 0.65])


def _hand_expected(theta_eff: np.ndarray) -> float:
    eta = float(SLOPE @ theta_eff)
    log_probs = compute_grm_category_logprobs(np.array([eta]), THRESHOLDS)[0]
    probs = np.exp(log_probs)
    return float((probs * np.arange(probs.size)).sum())


def _hand_one_column(column: int, theta: np.ndarray = THETA) -> float:
    total = 0.0
    for node, weight in zip(NODES, WEIGHTS, strict=True):
        theta_eff = theta.copy()
        theta_eff[column] = node
        total += weight * _hand_expected(theta_eff)
    return total


def _hand_two_columns(columns: tuple[int, int]) -> float:
    total = 0.0
    for j, node_j in enumerate(NODES):
        for k, node_k in enumerate(NODES):
            theta_eff = THETA.copy()
            theta_eff[columns[0]] = node_j
            theta_eff[columns[1]] = node_k
            total += WEIGHTS[j] * WEIGHTS[k] * _hand_expected(theta_eff)
    return total


def test_one_integrated_column_matches_hand_computation() -> None:
    reference = _hand_one_column(1)
    result = compute_expected_graded_item_score(
        SLOPE,
        THRESHOLDS,
        THETA,
        integrate_columns=(1,),
        integration_nodes=(NODES,),
        integration_weights=(WEIGHTS,),
    )
    assert result == reference


def test_two_integrated_columns_use_product_weights() -> None:
    reference = _hand_two_columns((1, 2))
    result = compute_expected_graded_item_score(
        SLOPE,
        THRESHOLDS,
        THETA,
        integrate_columns=(1, 2),
        integration_nodes=(NODES, NODES),
        integration_weights=(WEIGHTS, WEIGHTS),
    )
    assert result == reference


def test_non_integrated_column_stays_at_fixed_value() -> None:
    baseline_theta = THETA.copy()
    shifted_theta = THETA.copy()
    shifted_theta[3] = 2.25

    baseline = compute_expected_graded_item_score(
        SLOPE,
        THRESHOLDS,
        baseline_theta,
        integrate_columns=(1,),
        integration_nodes=(NODES,),
        integration_weights=(WEIGHTS,),
    )
    shifted = compute_expected_graded_item_score(
        SLOPE,
        THRESHOLDS,
        shifted_theta,
        integrate_columns=(1,),
        integration_nodes=(NODES,),
        integration_weights=(WEIGHTS,),
    )

    assert shifted != baseline
    assert shifted == _hand_one_column(1, shifted_theta)
