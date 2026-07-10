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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def gauss_hermite_nodes(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes/weights for a standard-normal ability prior N(0, 1).

    ``hermegauss`` gives the probabilists' Hermite rule (weight exp(-x^2/2));
    normalizing the weights to sum to 1 turns them into N(0,1) quadrature.
    """
    nodes, raw_weights = hermegauss(n_nodes)
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
    y = np.asarray(y, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    if y.shape != observed.shape or y.ndim != 2:
        raise ValueError("y and observed must be 2D and identically shaped")
    if not observed.any():
        raise ValueError("no observed responses")

    n_persons, n_items = y.shape
    # Zero-fill missing so array math is finite; the observed mask nullifies them.
    y_filled = np.where(observed, y, 0.0)
    obs_f = observed.astype(np.float64)

    nodes, weights = gauss_hermite_nodes(n_nodes)  # (Q,), (Q,)
    log_weights = np.log(weights)

    rng = np.random.default_rng(seed)
    # Init: a=1, b from observed item log-odds of endorsement.
    p_item = (y_filled * obs_f).sum(0) / np.clip(obs_f.sum(0), 1.0, None)
    p_item = np.clip(p_item, 0.02, 0.98)
    a = np.ones(n_items) + 0.01 * rng.standard_normal(n_items)
    b = np.log(p_item / (1.0 - p_item))

    loglik_trace: list[float] = []
    status = "max_iter_reached"

    for iteration in range(max_iter):
        # ---- E-step: posterior over quadrature nodes per person ----
        # logit_{q,i} = a_i * node_q + b_i  ->  (Q, n_items)
        logit = nodes[:, None] * a[None, :] + b[None, :]
        log_p1 = -np.logaddexp(0.0, -logit)  # log sigmoid
        log_p0 = -np.logaddexp(0.0, logit)  # log(1 - sigmoid)
        # Per person, per node: sum over OBSERVED items of log P(y_pi | node_q)
        # log_lik[p, q] = sum_i obs_pi * (y_pi*log_p1_qi + (1-y_pi)*log_p0_qi)
        # Compute via matrix products: (n_persons, Q)
        pos = (y_filled * obs_f) @ log_p1.T  # (n_persons, Q)
        neg = ((1.0 - y_filled) * obs_f) @ log_p0.T  # (n_persons, Q)
        log_joint = pos + neg + log_weights[None, :]  # + log prior weight
        # Normalize across nodes (log-sum-exp)
        max_lj = log_joint.max(axis=1, keepdims=True)
        stab = np.exp(log_joint - max_lj)
        denom = stab.sum(axis=1, keepdims=True)
        posterior = stab / denom  # (n_persons, Q)
        person_loglik = (max_lj[:, 0] + np.log(denom[:, 0]))
        total_loglik = float(person_loglik.sum())
        loglik_trace.append(total_loglik)

        # ---- M-step: update a, b by weighted logistic regression per item ----
        # Expected counts at each node: n_iq = sum_p obs_pi * posterior_pq  (Q per item)
        # r_iq = sum_p obs_pi * y_pi * posterior_pq
        n_iq = obs_f.T @ posterior  # (n_items, Q)
        r_iq = (obs_f * y_filled).T @ posterior  # (n_items, Q)

        a_new = a.copy()
        b_new = b.copy()
        for i in range(n_items):
            ai, bi = a[i], b[i]
            # Newton steps on the item's expected log-likelihood over nodes.
            for _ in range(25):
                eta = ai * nodes + bi
                p = _sigmoid(eta)
                w = n_iq[i] * p * (1.0 - p)
                resid = r_iq[i] - n_iq[i] * p
                g_a = float((resid * nodes).sum()) - ridge_a * ai
                g_b = float(resid.sum()) - ridge_b * bi
                h_aa = -float((w * nodes * nodes).sum()) - ridge_a
                h_bb = -float(w.sum()) - ridge_b
                h_ab = -float((w * nodes).sum())
                det = h_aa * h_bb - h_ab * h_ab
                if abs(det) < 1e-12:
                    break
                da = (h_bb * g_a - h_ab * g_b) / det
                db = (h_aa * g_b - h_ab * g_a) / det
                ai -= da
                bi -= db
                ai = float(np.clip(ai, 1e-3, 10.0))
                if abs(da) + abs(db) < 1e-8:
                    break
            a_new[i], b_new[i] = ai, bi

        a, b = a_new, b_new

        if iteration > 0 and abs(loglik_trace[-1] - loglik_trace[-2]) < tol:
            status = "converged"
            break

    # ---- EAP ability for each person ----
    theta = (posterior * nodes[None, :]).sum(axis=1)

    return {
        "a": a,
        "b": b,
        "theta": theta,
        "loglik_trace": loglik_trace,
        "n_iter": len(loglik_trace),
        "status": status,
    }
