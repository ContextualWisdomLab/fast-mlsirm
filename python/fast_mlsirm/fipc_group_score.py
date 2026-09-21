# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Focal-prior person EAP and plug-in expected raw scores for 1-D GRM FIPC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .polytomous import (
    PolyFipcFit,
    PolytomousFit,
    fit_poly_fipc,
    predict_category_probabilities_polytomous,
    predict_expected_response_polytomous,
)


def _normal_rule(q: int) -> tuple[np.ndarray, np.ndarray]:
    """Standard-normal Jacobi rule (Golub & Welsch, 1969, pp. 221–223).

    Reference: Golub, G. H., & Welsch, J. H. (1969). Calculation of Gauss
    quadrature rules. *Mathematics of Computation, 23*(106), 221–230.
    https://doi.org/10.1090/S0025-5718-69-99647-1
    """
    off_diagonal = np.sqrt(np.arange(1, q, dtype=np.float64))
    jacobi = np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    nodes, vectors = np.linalg.eigh(jacobi)
    weights = vectors[0] ** 2
    return nodes, weights / weights.sum()


def _as_poly_fit(fipc: PolyFipcFit) -> PolytomousFit:
    """Wrap FIPC item parameters as a GRM :class:`PolytomousFit` for scoring.

    Does **not** carry ``mu`` / ``sigma``; callers must pass the focal prior
    explicitly into :func:`_score_poly_eap_gaussian_prior`.
    """
    return PolytomousFit(
        model="grm",
        slope=np.asarray(fipc.slope, dtype=np.float64),
        cat_params=np.asarray(fipc.cat_params, dtype=np.float64),
        loglik=float(fipc.loglik),
        n_iter=int(fipc.n_iter),
        converged=bool(fipc.converged),
        termination_reason=str(fipc.termination_reason),
        loglik_trace=np.asarray(fipc.loglik_trace, dtype=np.float64),
        final_delta=float(fipc.final_delta),
        stopping_tolerance=float(fipc.stopping_tolerance),
        thresholds=None,
    )


def _anchor_item_bank(
    anchor: np.ndarray,
    anchor_slope: np.ndarray,
    anchor_cat_params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return slope/cat_params rows for ``anchor==True`` items only."""
    mask = np.asarray(anchor, dtype=bool)
    if mask.ndim != 1 or mask.size == 0 or not bool(np.any(mask)):
        raise ValueError("anchor must be a non-empty 1-D bool mask with >=1 True")
    slope = np.asarray(anchor_slope, dtype=np.float64)
    cat = np.asarray(anchor_cat_params, dtype=np.float64)
    if slope.ndim != 1 or slope.shape[0] != mask.shape[0]:
        raise ValueError("anchor_slope must be length n_items matching anchor")
    if cat.ndim != 2 or cat.shape[0] != mask.shape[0]:
        raise ValueError("anchor_cat_params must be n_items x (n_cat - 1)")
    return slope[mask].copy(), cat[mask].copy()


def _score_poly_eap_gaussian_prior(
    responses: np.ndarray,
    bank: PolytomousFit,
    *,
    mu: float,
    sigma: float,
    q_theta: int,
) -> dict[str, np.ndarray]:
    """EAP under ``theta ~ N(mu, sigma^2)`` on a 1-D GRM bank.

    Scales standard-normal Jacobi nodes to the prior and evaluates category
    probabilities at those nodes (Bock & Mislevy, 1982, pp. 432–434). Missing
    responses marked ``NaN`` or ``-1`` are skipped.

    Reference: Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation
    of ability in a microcomputer environment. *Applied Psychological
    Measurement, 6*(4), 431–444. https://doi.org/10.1177/014662168200600405
    """
    if type(q_theta) is not int:
        raise TypeError("q_theta must be an int")
    if q_theta < 1:
        raise ValueError("q_theta must be >= 1")
    mu_f = float(mu)
    sigma_f = float(sigma)
    if not np.isfinite(mu_f) or not np.isfinite(sigma_f) or sigma_f <= 0.0:
        raise ValueError("mu must be finite and sigma must be finite and > 0")

    y = np.asarray(responses, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    slope = np.asarray(bank.slope, dtype=np.float64)
    if slope.shape[0] != n_items:
        raise ValueError("responses column count must match the fitted item count")

    nodes, weights = _normal_rule(q_theta)
    theta = mu_f + sigma_f * nodes
    cat_probs = predict_category_probabilities_polytomous(bank, theta)
    # cat_probs: (q, n_items, n_cat)
    log_w = np.full(q_theta, -np.inf, dtype=np.float64)
    np.log(weights, out=log_w, where=weights > 0)
    n_cat = int(cat_probs.shape[2])

    observed = ~np.isnan(y) & (y != -1.0)
    if np.any(observed & (~np.isfinite(y) | (y != np.floor(y)))):
        raise ValueError("observed responses must be finite integer categories")
    y_int = np.zeros_like(y, dtype=np.int64)
    y_int[observed] = np.rint(y[observed]).astype(np.int64)
    if np.any(observed & ((y_int < 0) | (y_int >= n_cat))):
        raise ValueError(f"observed responses must be integer categories in 0..{n_cat - 1}")

    log_cat = np.log(np.clip(cat_probs, 1e-300, 1.0))
    theta_eap = np.empty(n_persons, dtype=np.float64)
    theta_sd = np.empty(n_persons, dtype=np.float64)
    for p in range(n_persons):
        log_post = log_w.copy()
        for i in range(n_items):
            if not bool(observed[p, i]):
                continue
            k = int(y_int[p, i])
            log_post += log_cat[:, i, k]
        m = float(np.max(log_post))
        w = np.exp(log_post - m)
        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            raise RuntimeError("EAP posterior weights degenerate")
        w /= w_sum
        mean = float(np.sum(w * theta))
        var = float(np.sum(w * (theta - mean) ** 2))
        if var < 0.0 and var > -1e-12:
            var = 0.0
        if var < 0.0:
            raise RuntimeError("EAP posterior variance computed negative")
        theta_eap[p] = mean
        theta_sd[p] = float(np.sqrt(var))
    return {"theta_eap": theta_eap, "theta_sd": theta_sd}


@dataclass(frozen=True)
class PolyFipcGroupPersonScores:
    """Person scores for one focal group after 1-D poly-GRM FIPC.

    ``theta_eap`` / ``theta_sd`` are on the reference metric fixed by anchors,
    under the fitted focal prior ``N(fipc.mu, fipc.sigma^2)``.
    ``expected_raw`` is the plug-in expected total at each person's EAP
    (full FIPC bank: anchors + free items).
    """

    theta_eap: np.ndarray
    theta_sd: np.ndarray
    expected_raw: np.ndarray
    fipc: PolyFipcFit


def score_poly_fipc_group_persons(
    responses: np.ndarray,
    n_cat: int,
    anchor: np.ndarray,
    anchor_slope: np.ndarray,
    anchor_cat_params: np.ndarray,
    *,
    q_theta: int,
    max_iter: int,
    tol: float,
) -> PolyFipcGroupPersonScores:
    """FIPC-calibrate one focal group, then EAP + expected-raw person scores.

    Parameters mirror :func:`fit_poly_fipc`. ``q_theta`` is required (no
    default; #1929). Domain group labels (age bands, waves, sites) are the
    caller's responsibility — pass one group's response matrix per call.

    EAP uses the fitted focal prior ``N(mu, sigma^2)`` from the FIPC result
    (Kim, 2006), not a dropped ``N(0, 1)`` via :func:`score_polytomous`.

    References
    ----------
    Kim, S. (2006). A comparative study of IRT fixed parameter calibration
    methods. *Journal of Educational Measurement, 43*(4), 355–381.
    https://doi.org/10.1111/j.1745-3984.2006.00021.x

    Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability
    in a microcomputer environment. *Applied Psychological Measurement,
    6*(4), 431–444. https://doi.org/10.1177/014662168200600405

    Lord, F. M. (1980). *Applications of item response theory to practical
    testing problems*. Chapter 4 (test characteristic / number-right true
    score).
    """
    fipc = fit_poly_fipc(
        responses,
        n_cat,
        anchor,
        anchor_slope,
        anchor_cat_params,
        q_theta=q_theta,
        max_iter=max_iter,
        tol=tol,
    )
    bank = _as_poly_fit(fipc)
    scored = _score_poly_eap_gaussian_prior(
        responses,
        bank,
        mu=float(fipc.mu),
        sigma=float(fipc.sigma),
        q_theta=q_theta,
    )
    theta_eap = np.asarray(scored["theta_eap"], dtype=np.float64)
    theta_sd = np.asarray(scored["theta_sd"], dtype=np.float64)
    expected_items = predict_expected_response_polytomous(bank, theta_eap)
    expected_raw = np.asarray(expected_items.sum(axis=1), dtype=np.float64)
    return PolyFipcGroupPersonScores(
        theta_eap=theta_eap,
        theta_sd=theta_sd,
        expected_raw=expected_raw,
        fipc=fipc,
    )
