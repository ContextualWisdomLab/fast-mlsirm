"""Expected graded-item scores with caller-controlled factor integration.

Supports marginalizing one graded item's expected category score over a
caller-supplied subset of latent dimensions while the remaining coordinates
(including a plugged general factor) stay at fixed values.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import product as iterproduct

import numpy as np

from .estimators.marginal import compute_grm_category_logprobs

__all__ = ["compute_expected_graded_item_score"]

_INTEGRATION_WEIGHT_TOLERANCE = 1e-12


def _expected_category_score_from_linear_predictor(
    linear_predictor: float,
    thresholds: np.ndarray,
) -> float:
    """Return ``E[Y | eta]`` for one Samejima GRM item at a scalar predictor."""
    log_probs = compute_grm_category_logprobs(
        np.array([linear_predictor], dtype=np.float64),
        thresholds,
    )[0]
    probs = np.exp(log_probs)
    scores = np.arange(probs.size, dtype=np.float64)
    return float((probs * scores).sum())


def _validate_integration_axis_weights(weights: np.ndarray, axis_index: int) -> None:
    """Reject integration weights that are not a probability measure on one axis."""
    label = f"integration_weights[{axis_index}]"
    if np.any(weights < 0.0):
        raise ValueError(f"{label} must be non-negative")
    weight_sum = math.fsum(float(weight) for weight in weights)
    if not math.isfinite(weight_sum):
        raise ValueError(f"{label} must sum to one")
    if not math.isclose(
        weight_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=_INTEGRATION_WEIGHT_TOLERANCE,
    ):
        raise ValueError(f"{label} must sum to one within tolerance")


def compute_expected_graded_item_score(
    slope: np.ndarray,
    thresholds: np.ndarray,
    theta: np.ndarray,
    integrate_columns: Sequence[int],
    integration_nodes: Sequence[np.ndarray],
    integration_weights: Sequence[np.ndarray],
) -> float:
    """Return the expected category score of one graded item after integration.

    The compensatory linear predictor is ``eta = slope @ theta_eff``. Entries
    of ``theta`` that are not listed in ``integrate_columns`` remain fixed
    (for example a plugged general factor). Each integrated column is replaced
    in turn by the corresponding caller-supplied nodes; when more than one
    column is integrated, the quadrature runs on the Cartesian product with
    product weights. Each integrated axis carries its own probability measure:
    weights must be finite, non-negative, and sum to one within a tight
    absolute tolerance; invalid axes fail closed with no silent renormalization.
    Nodes are used as given — the caller is responsible for any prior scaling.

    Category probabilities use :func:`~fast_mlsirm.estimators.marginal.compute_grm_category_logprobs`
    on the item linear predictor.

    Parameters
    ----------
    slope:
        Item discrimination vector, one entry per latent dimension.
    thresholds:
        ``K - 1`` strictly ordered cumulative boundary intercepts for the
        graded item.
    theta:
        Fixed latent vector with non-integrated coordinates already set
        (including the general factor when it is not integrated).
    integrate_columns:
        Column indices to replace with quadrature nodes.
    integration_nodes:
        One finite 1-D node array per integrated column, in the same order
        as ``integrate_columns``.
    integration_weights:
        One finite 1-D weight array per integrated column, aligned with
        ``integration_nodes``. Each array is an independent discrete
        probability measure over that axis (non-negative entries summing to
        one). Multi-axis integration uses the product measure on the Cartesian
        product of node grids.

    Returns
    -------
    float
        The weighted expected category score ``sum_k k * P(Y = k)`` after
        integrating over the requested columns.
    """
    slope_arr = np.asarray(slope, dtype=np.float64)
    if slope_arr.ndim != 1 or slope_arr.size == 0:
        raise ValueError("slope must be a non-empty 1-D array")
    if not np.all(np.isfinite(slope_arr)):
        raise ValueError("slope must be finite")

    theta_arr = np.asarray(theta, dtype=np.float64)
    if theta_arr.ndim != 1 or theta_arr.shape != slope_arr.shape:
        raise ValueError("theta must be a 1-D array with the same length as slope")
    if not np.all(np.isfinite(theta_arr)):
        raise ValueError("theta must be finite")

    thresholds_arr = np.asarray(thresholds, dtype=np.float64)
    if thresholds_arr.ndim != 1 or thresholds_arr.size < 1:
        raise ValueError("thresholds must be a 1-D array of length K-1 >= 1")
    if not np.all(np.isfinite(thresholds_arr)):
        raise ValueError("thresholds must be finite")

    columns = [int(column) for column in integrate_columns]
    if len(columns) != len(set(columns)):
        raise ValueError("integrate_columns must not contain duplicates")
    for column in columns:
        if not 0 <= column < slope_arr.size:
            raise ValueError("integrate_columns entries must index slope/theta")

    if len(columns) != len(integration_nodes) or len(columns) != len(integration_weights):
        raise ValueError(
            "integration_nodes and integration_weights must align with integrate_columns"
        )

    node_arrays: list[np.ndarray] = []
    weight_arrays: list[np.ndarray] = []
    for index, column in enumerate(columns):
        nodes = np.asarray(integration_nodes[index], dtype=np.float64)
        weights = np.asarray(integration_weights[index], dtype=np.float64)
        if nodes.ndim != 1 or nodes.size == 0:
            raise ValueError(f"integration_nodes[{index}] must be a non-empty 1-D array")
        if weights.ndim != 1 or weights.shape != nodes.shape:
            raise ValueError(
                f"integration_weights[{index}] must be a 1-D array matching its nodes"
            )
        if not np.all(np.isfinite(nodes)):
            raise ValueError("integration nodes must be finite")
        if not np.all(np.isfinite(weights)):
            raise ValueError(f"integration_weights[{index}] must be finite")
        _validate_integration_axis_weights(weights, index)
        node_arrays.append(nodes)
        weight_arrays.append(weights)

    if not columns:
        base = float(slope_arr @ theta_arr)
        return _expected_category_score_from_linear_predictor(base, thresholds_arr)

    expected = 0.0
    index_ranges = [range(nodes.size) for nodes in node_arrays]
    for combo in iterproduct(*index_ranges):
        effective_theta = theta_arr.copy()
        weight = 1.0
        for dim_index, node_index in enumerate(combo):
            column = columns[dim_index]
            effective_theta[column] = node_arrays[dim_index][node_index]
            weight *= weight_arrays[dim_index][node_index]
        base = float(slope_arr @ effective_theta)
        expected += weight * _expected_category_score_from_linear_predictor(
            base,
            thresholds_arr,
        )
    return expected
