"""MMLE (marginal maximum likelihood) via EM — robust to missing data.

Why this exists
---------------
Penalized JMLE estimates every person's theta as a free parameter jointly with
item parameters. Under missing / sparse responses that joint problem is weakly
identified and item parameters (especially discrimination) become biased.

MMLE marginalizes the person ability theta out of the likelihood by integrating
over a fixed population distribution (Gauss-Hermite quadrature). Each person's
contribution is a product over their **observed** items only, so missingness is
handled *by construction* — no imputation, no bias from unanswered items
(missing-at-random). This is the standard, statistically-sound way to calibrate
item parameters when data are incomplete.

Scope
-----
Unidimensional 2PL (matches ULS2PLM / the 2PL slice of MLS2PLM's measurement
part). Item parameters (a, b) are estimated by EM; person ability is returned as
the EAP posterior mean. Multidimensional / spatial (xi, zeta, tau) and polytomous
GRM are separate follow-up milestones.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite_e import hermegauss


MAX_GAUSS_HERMITE_NODES = 100
MAX_MMLE_FALLBACK_WORKSPACE_BYTES = 512 * 1024 * 1024


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic sigmoid with the exponent clipped to ``[-35, 35]`` for stability."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def _validate_quadrature_node_count(n_nodes: int) -> int:
    """Return a node count within NumPy's documented tested quadrature range."""
    if isinstance(n_nodes, (bool, np.bool_)) or not isinstance(
        n_nodes, (int, np.integer)
    ):
        raise ValueError(
            f"n_nodes must be an integer in [1, {MAX_GAUSS_HERMITE_NODES}]"
        )
    validated = int(n_nodes)
    if validated < 1 or validated > MAX_GAUSS_HERMITE_NODES:
        raise ValueError(f"n_nodes must be in [1, {MAX_GAUSS_HERMITE_NODES}]")
    return validated


def _estimate_mmle_workspace_bytes(
    n_persons: int,
    n_items: int,
    n_nodes: int,
) -> int:
    """Conservatively estimate owned NumPy fallback workspace in bytes.

    The estimate covers response-sized conversion and masking arrays,
    person-by-node posterior arrays, item-by-node Newton arrays, and small
    one-dimensional state. It deliberately overestimates repository-owned
    arrays but does not claim to include caller-owned inputs or hidden BLAS
    workspace.
    """
    response_cells = n_persons * n_items
    person_node_cells = n_persons * n_nodes
    item_node_cells = n_items * n_nodes
    float64_cells = (
        5 * response_cells
        + 6 * person_node_cells
        + 14 * item_node_cells
        + 12 * (n_persons + n_items + n_nodes)
    )
    boolean_cells = response_cells + 2 * n_items
    return (
        float64_cells * np.dtype(np.float64).itemsize
        + boolean_cells * np.dtype(np.bool_).itemsize
    )


def _validate_mmle_workspace(
    n_persons: int,
    n_items: int,
    n_nodes: int,
) -> None:
    """Reject fallback problems whose estimated workspace exceeds the safe cap."""
    estimated_bytes = _estimate_mmle_workspace_bytes(
        n_persons,
        n_items,
        n_nodes,
    )
    if estimated_bytes > MAX_MMLE_FALLBACK_WORKSPACE_BYTES:
        raise ValueError(
            "NumPy MMLE fallback workspace estimate "
            f"{estimated_bytes} bytes exceeds the "
            f"{MAX_MMLE_FALLBACK_WORKSPACE_BYTES}-byte safe limit; "
            "use the Rust backend or reduce the response matrix or quadrature nodes"
        )


def gauss_hermite_nodes(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes/weights for a standard-normal ability prior N(0, 1).

    ``hermegauss`` gives the probabilists' Hermite rule (weight exp(-x^2/2));
    normalizing the weights to sum to 1 turns them into N(0,1) quadrature.
    """
    validated_nodes = _validate_quadrature_node_count(n_nodes)
    nodes, raw_weights = hermegauss(validated_nodes)
    weights = raw_weights / raw_weights.sum()
    return nodes, weights


def fit_mmle_2pl(
    y: np.ndarray,
    observed: np.ndarray,
    *,
    n_nodes: int = 41,
    max_iter: int = 500,
    tol: float = 1e-6,
    ridge_a: float = 1e-3,
    ridge_b: float = 1e-3,
    seed: int = 1,
) -> dict[str, object]:
    """Calibrate a unidimensional 2PL by MMLE-EM under missing data.

    Parameters
    ----------
    y : (n_persons, n_items) float array of 0/1 responses. Missing cells may hold
        NaN / any value; they are ignored wherever ``observed`` is False.
    observed : (n_persons, n_items) bool array; True where a response is present.

    Returns
    -------
    dict with keys: ``a`` (discrimination), ``b`` (difficulty/intercept, so that
    logit = a*theta + b), ``theta`` (EAP ability), ``loglik_trace``, ``n_iter``,
    ``status``.
    """
    validated_nodes = _validate_quadrature_node_count(n_nodes)
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(
        max_iter, (int, np.integer)
    ):
        raise ValueError("max_iter must be an integer >= 1")
    validated_max_iter = int(max_iter)
    if validated_max_iter < 1:
        raise ValueError("max_iter must be an integer >= 1")

    y_array = np.asarray(y)
    observed_array = np.asarray(observed)
    if y_array.shape != observed_array.shape or y_array.ndim != 2:
        raise ValueError("y and observed must be 2D and identically shaped")

    n_persons, n_items = y_array.shape
    _validate_mmle_workspace(n_persons, n_items, validated_nodes)

    y = np.asarray(y_array, dtype=np.float64)
    observed = np.asarray(observed_array, dtype=bool)
    if not observed.any():
        raise ValueError("no observed responses")

    # Zero-fill missing so array math is finite; the observed mask nullifies them.
    y_filled = np.where(observed, y, 0.0)
    obs_f = observed.astype(np.float64)

    nodes, weights = gauss_hermite_nodes(validated_nodes)  # (Q,), (Q,)
    log_weights = np.log(weights)

    rng = np.random.default_rng(seed)
    # Init: a=1, b from observed item log-odds of endorsement.
    p_item = (y_filled * obs_f).sum(0) / np.clip(obs_f.sum(0), 1.0, None)
    p_item = np.clip(p_item, 0.02, 0.98)
    a = np.ones(n_items) + 0.01 * rng.standard_normal(n_items)
    b = np.log(p_item / (1.0 - p_item))

    nodes_sq = nodes * nodes
    loglik_trace: list[float] = []
    status = "max_iter_reached"

    for iteration in range(validated_max_iter):
        # ---- E-step: posterior over quadrature nodes per person ----
        # logit_{q,i} = a_i * node_q + b_i  ->  (Q, n_items)
        logit = nodes[:, None] * a[None, :] + b[None, :]
        log_p1 = -np.logaddexp(0.0, -logit)  # log sigmoid
        log_p0 = -np.logaddexp(0.0, logit)  # log(1 - sigmoid)
        # Per person, per node: sum over OBSERVED items of log P(y_pi | node_q)
        # log_lik[p, q] = sum_i obs_pi * (y_pi*log_p1_qi + (1-y_pi)*log_p0_qi)
        # Compute without forming n_persons * n_items * Q arrays via explicit broadcast products
        pos = (y_filled * obs_f) @ log_p1.T
        neg = ((1.0 - y_filled) * obs_f) @ log_p0.T
        log_joint = pos + neg + log_weights[None, :]  # + log prior weight
        # Normalize across nodes (log-sum-exp)
        max_lj = log_joint.max(axis=1, keepdims=True)
        stab = np.exp(log_joint - max_lj)
        denom = stab.sum(axis=1, keepdims=True)
        posterior = stab / denom  # (n_persons, Q)
        person_loglik = max_lj[:, 0] + np.log(denom[:, 0])
        total_loglik = float(person_loglik.sum())
        loglik_trace.append(total_loglik)

        # ---- M-step: update a, b by weighted logistic regression per item ----
        # Expected counts at each node: n_iq = sum_p obs_pi * posterior_pq  (Q per item)
        # r_iq = sum_p obs_pi * y_pi * posterior_pq
        # Compute without forming n_items * Q arrays via explicit broadcast products
        n_iq = obs_f.T @ posterior
        r_iq = (obs_f * y_filled).T @ posterior

        a_new = a.copy()
        b_new = b.copy()

        # Process active items together to remove the per-item Python control-flow loop.
        active = np.ones(n_items, dtype=bool)

        for _ in range(25):
            if not active.any():
                break

            ai = a_new[active]
            bi = b_new[active]
            n_iq_active = n_iq[active]
            r_iq_active = r_iq[active]

            eta_m = ai[:, None] * nodes[None, :] + bi[:, None]
            p_m = _sigmoid(eta_m)
            w = n_iq_active * p_m * (1.0 - p_m)
            resid = r_iq_active - n_iq_active * p_m

            # Optimization: Replace element-wise multiply and axis reduction with dense matrix multiplication
            # to avoid large intermediate array allocations
            g_a = resid @ nodes - ridge_a * ai
            g_b = resid.sum(axis=1) - ridge_b * bi

            h_aa = -(w @ nodes_sq) - ridge_a
            h_bb = -w.sum(axis=1) - ridge_b
            h_ab = -(w @ nodes)

            det = h_aa * h_bb - h_ab * h_ab

            valid = np.abs(det) >= 1e-12

            da = np.zeros_like(ai)
            db = np.zeros_like(bi)

            if valid.any():
                da[valid] = (h_bb[valid] * g_a[valid] - h_ab[valid] * g_b[valid]) / det[
                    valid
                ]
                db[valid] = (h_aa[valid] * g_b[valid] - h_ab[valid] * g_a[valid]) / det[
                    valid
                ]

            ai -= da
            bi -= db
            ai = np.clip(ai, 1e-3, 10.0)

            converged = (np.abs(da) + np.abs(db)) < 1e-8
            done = converged | ~valid

            a_new[active] = ai
            b_new[active] = bi

            active[active] = ~done

        a, b = a_new, b_new

        if iteration > 0 and abs(loglik_trace[-1] - loglik_trace[-2]) < tol:
            status = "converged"
            break

    # ---- EAP ability for each person ----
    # Optimization: Replace element-wise multiply and axis reduction with dense matrix multiplication
    # to avoid intermediate array allocation
    theta = posterior @ nodes

    return {
        "a": a,
        "b": b,
        "theta": theta,
        "loglik_trace": loglik_trace,
        "n_iter": len(loglik_trace),
        "status": status,
    }
