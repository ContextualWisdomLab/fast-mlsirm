"""Expected graded-item scores with caller-controlled factor integration.

Supports marginalizing one graded item's expected category score over a
caller-supplied subset of latent dimensions while the remaining coordinates
(including a plugged general factor) stay at fixed values.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product as iterproduct

import numpy as np

from .estimators.marginal import compute_grm_category_logprobs
from .estimators.mmle import equal_probability_normal_nodes

__all__ = ["compute_expected_graded_item_score", "expected_graded_scale_score"]


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
    product weights. Nodes are used as given — the caller is responsible for
    any prior scaling.

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
        ``integration_nodes``.

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
        if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(weights)):
            raise ValueError("integration nodes and weights must be finite")
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


def expected_graded_scale_score(
    slopes: np.ndarray,
    thresholds: np.ndarray,
    theta: np.ndarray,
    integrate_columns: Sequence[Sequence[int]],
    factor_variances: np.ndarray,
    *,
    n_nodes: int,
) -> float:
    """Return the expected total graded scale score ``E[T | G]`` after integration.

    Builds equal-probability standard-normal quadrature with
    :func:`~fast_mlsirm.estimators.mmle.equal_probability_normal_nodes`,
    scales integrated dimensions by ``sqrt(factor_variances[d])`` (mean zero),
    and sums per-item expectations from
    :func:`~fast_mlsirm.graded_item.compute_expected_graded_item_score`.
    Non-integrated coordinates of ``theta`` (for example a plugged general
    factor) remain fixed at caller-supplied values.

    Parameters
    ----------
    slopes:
        ``(n_items, n_dims)`` item discrimination matrix.
    thresholds:
        ``(n_items, K - 1)`` cumulative boundary intercepts; every row must
        have the same number of thresholds.
    theta:
        Fixed latent vector with non-integrated coordinates already set.
    integrate_columns:
        Per-item column indices to replace with scaled quadrature nodes.
    factor_variances:
        ``(n_dims,)`` latent factor **variances** (not standard deviations).
        Integrated columns use nodes ``sqrt(factor_variances[d]) * z`` where
        ``z`` are the standard-normal equal-probability nodes.
    n_nodes:
        Number of equal-probability quadrature nodes per integrated dimension.
        Required with no default; callers choose precision for their study.

    Returns
    -------
    float
        ``sum_i E[Y_i | G]`` after integrating the requested columns for
        each item.
    """
    slopes_arr = np.asarray(slopes, dtype=np.float64)
    if slopes_arr.ndim != 2 or slopes_arr.shape[0] == 0:
        raise ValueError("slopes must be a non-empty 2-D array")
    if not np.all(np.isfinite(slopes_arr)):
        raise ValueError("slopes must be finite")

    n_items, n_dims = slopes_arr.shape

    theta_arr = np.asarray(theta, dtype=np.float64)
    if theta_arr.ndim != 1 or theta_arr.shape != (n_dims,):
        raise ValueError("theta must be a 1-D array with length n_dims")
    if not np.all(np.isfinite(theta_arr)):
        raise ValueError("theta must be finite")

    thresholds_arr = np.asarray(thresholds, dtype=np.float64)
    if thresholds_arr.ndim != 2 or thresholds_arr.shape[0] != n_items:
        raise ValueError("thresholds must be a 2-D array with shape (n_items, K-1)")
    if thresholds_arr.shape[1] < 1:
        raise ValueError("thresholds must have at least one boundary per item")
    if not np.all(np.isfinite(thresholds_arr)):
        raise ValueError("thresholds must be finite")

    variances = np.asarray(factor_variances, dtype=np.float64)
    if variances.ndim != 1 or variances.shape != (n_dims,):
        raise ValueError("factor_variances must be a 1-D array with length n_dims")
    if not np.all(np.isfinite(variances)):
        raise ValueError("factor_variances must be finite")
    if np.any(variances < 0.0):
        raise ValueError("factor_variances must be non-negative")

    if len(integrate_columns) != n_items:
        raise ValueError("integrate_columns must have one entry per item")

    if isinstance(n_nodes, (bool, np.bool_)) or not isinstance(n_nodes, (int, np.integer)):
        raise ValueError("n_nodes must be a positive integer")
    validated_nodes = int(n_nodes)
    if validated_nodes < 1:
        raise ValueError("n_nodes must be >= 1")

    base_nodes, base_weights = equal_probability_normal_nodes(validated_nodes)

    scaled_by_column: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for column in range(n_dims):
        scale = float(np.sqrt(variances[column]))
        scaled_by_column[column] = (base_nodes * scale, base_weights.copy())

    total = 0.0
    for item_index in range(n_items):
        columns = [int(column) for column in integrate_columns[item_index]]
        item_nodes = tuple(scaled_by_column[column][0] for column in columns)
        item_weights = tuple(scaled_by_column[column][1] for column in columns)
        total += compute_expected_graded_item_score(
            slopes_arr[item_index],
            thresholds_arr[item_index],
            theta_arr,
            columns,
            item_nodes,
            item_weights,
        )
    return total
