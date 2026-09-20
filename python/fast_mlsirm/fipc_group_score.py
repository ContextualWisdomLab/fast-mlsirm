# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""1-D poly-GRM FIPC group person scoring and reference expected-score moments.

Library-generic API for **unidimensional** poly-GRM fixed-item parameter
calibration of one focal group, then person EAP / expected-raw scores.
Callers map research labels (e.g. age bands) onto group batches externally —
this module never encodes domain polarity, age semantics, or G+4+W structure.

Scope (honest contract)
-----------------------
- Item model: 1-D GRM via :class:`PolyFipcFit` / :class:`PolytomousFit` only.
- Not two-tier / bifactor / G+4+W FIPC. Not PR #2077
  ``E[T|theta_focal]`` nuisance-integrated curves
  (``expected_total_score_two_tier_given_primary``).

Identification (Kim, 2006)
--------------------------
Anchors pin the item bank to the reference calibration metric; focal
``N(mu, sigma^2)`` is free. Person EAP uses that **fitted focal prior** on the
reference metric (Bock & Mislevy, 1982), not a silent ``N(0, 1)`` drop through
:func:`score_polytomous`. Expected raw totals are plug-in sums of
:func:`predict_expected_response_polytomous` at those EAPs (Lord, 1980, ch. 4).

Reference expected-score moments integrate ``E[T|theta]`` under a caller
``N(mu_ref, sigma_ref^2)`` using the **anchor-item bank only** (reference
parameters for ``anchor==True`` items) — never free-item / focal-estimated
slopes from the FIPC fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .estimators.mmle import gauss_hermite_nodes
from .polytomous import (
    PolyFipcFit,
    PolytomousFit,
    fit_poly_fipc,
    predict_category_probabilities_polytomous,
    predict_expected_response_polytomous,
)


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

    Scales probabilists' Gauss-Hermite nodes to the prior and evaluates
    category probabilities at those nodes (Bock & Mislevy, 1982). Missing
    responses marked ``NaN`` or ``-1`` are skipped.
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

    nodes, weights = gauss_hermite_nodes(q_theta)
    theta = mu_f + sigma_f * nodes
    cat_probs = predict_category_probabilities_polytomous(bank, theta)
    # cat_probs: (q, n_items, n_cat)
    log_w = np.log(np.asarray(weights, dtype=np.float64))
    qn = int(theta.shape[0])
    n_cat = int(cat_probs.shape[2])

    observed = np.isfinite(y) & (y >= 0.0)
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
        second = float(np.sum(w * theta * theta))
        var = second - mean * mean
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


@dataclass(frozen=True)
class PolyReferenceExpectedScoreMoments:
    """Moments of ``E[T|theta]`` when ``theta ~ N(mu_ref, sigma_ref^2)``.

    ``T`` is the sum over the **anchor-item** bank only. This is the reference
    expected-score distribution used to place focal group means after FIPC
    linking — not an Orlando–Thissen item-fit table and not a free-item TCC.
    """

    mu_ref: float
    sigma_ref: float
    q_theta: int
    mean: float
    second_moment: float
    variance: float
    n_anchor_items: int


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


def poly_reference_expected_score_moments(
    slope: np.ndarray,
    cat_params: np.ndarray,
    *,
    mu_ref: float,
    sigma_ref: float,
    q_theta: int,
) -> PolyReferenceExpectedScoreMoments:
    """Gauss-Hermite moments of expected total score under a reference prior.

    ``slope`` / ``cat_params`` must already be the **anchor-only** bank (caller
    slices with :func:`_anchor_item_bank` or equivalent). Transforms
    probabilists' Gauss-Hermite nodes to ``N(mu_ref, sigma_ref^2)`` and
    evaluates ``E[T|theta] = sum_i E[Y_i|theta]`` at each node (Lord, 1980,
    ch. 4). ``q_theta`` is required (#1929).

    References
    ----------
    Lord, F. M. (1980). *Applications of item response theory to practical
    testing problems*. Chapter 4.

    Golub, G. H., & Welsch, J. H. (1969). Calculation of Gauss quadrature
    rules. *Mathematics of Computation, 23*(106), 221–230.
    https://doi.org/10.1090/S0025-5718-69-99647-1
    """
    if type(q_theta) is not int:
        raise TypeError("q_theta must be an int")
    if q_theta < 1:
        raise ValueError("q_theta must be >= 1")
    mu = float(mu_ref)
    sigma = float(sigma_ref)
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("mu_ref must be finite and sigma_ref must be finite and > 0")

    slope_arr = np.asarray(slope, dtype=np.float64)
    cat_arr = np.asarray(cat_params, dtype=np.float64)
    if slope_arr.ndim != 1 or slope_arr.size == 0:
        raise ValueError("slope must be a non-empty 1-D array")
    if cat_arr.ndim != 2 or cat_arr.shape[0] != slope_arr.size:
        raise ValueError("cat_params must be n_items x (n_cat - 1)")
    if not np.all(np.isfinite(slope_arr)) or not np.all(np.isfinite(cat_arr)):
        raise ValueError("slope and cat_params must be finite")

    nodes, weights = gauss_hermite_nodes(q_theta)
    theta = mu + sigma * nodes
    bank = PolytomousFit(
        model="grm",
        slope=slope_arr,
        cat_params=cat_arr,
        loglik=0.0,
        n_iter=0,
        converged=True,
        termination_reason="reference_anchor_bank",
    )
    expected_total = predict_expected_response_polytomous(bank, theta).sum(axis=1)
    mean = float(np.sum(weights * expected_total))
    second = float(np.sum(weights * expected_total * expected_total))
    var = second - mean * mean
    if var < 0.0 and var > -1e-12:
        var = 0.0
    if var < 0.0:
        raise RuntimeError("reference expected-score variance computed negative")
    return PolyReferenceExpectedScoreMoments(
        mu_ref=mu,
        sigma_ref=sigma,
        q_theta=q_theta,
        mean=mean,
        second_moment=second,
        variance=var,
        n_anchor_items=int(slope_arr.size),
    )


def execute_fipc_group_person_score_payload(payload: dict[str, object]) -> dict[str, object]:
    """Remote/worker JSON contract for :func:`score_poly_fipc_group_persons`.

    Required keys: ``responses``, ``n_cat``, ``anchor``, ``anchor_slope``,
    ``anchor_cat_params``, ``q_theta``, ``max_iter``, ``tol``.
    Optional: ``reference_mu`` / ``reference_sigma`` (defaults 0 / 1) to also
    emit reference expected-score moments from the **anchor-item bank only**
    (``anchor_slope`` / ``anchor_cat_params`` rows where ``anchor`` is True),
    never free-item parameters from the focal FIPC fit.
    """
    if type(payload) is not dict:
        raise TypeError("payload must be a dict")
    required = (
        "responses",
        "n_cat",
        "anchor",
        "anchor_slope",
        "anchor_cat_params",
        "q_theta",
        "max_iter",
        "tol",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"payload missing required keys: {missing}")

    anchor = np.asarray(payload["anchor"], dtype=bool)
    anchor_slope = np.asarray(payload["anchor_slope"], dtype=np.float64)
    anchor_cat = np.asarray(payload["anchor_cat_params"], dtype=np.float64)

    result = score_poly_fipc_group_persons(
        np.asarray(payload["responses"]),
        int(payload["n_cat"]),
        anchor,
        anchor_slope,
        anchor_cat,
        q_theta=int(payload["q_theta"]),
        max_iter=int(payload["max_iter"]),
        tol=float(payload["tol"]),
    )
    out: dict[str, object] = {
        "family": "fipc_group_person_score",
        "library_function": "fast_mlsirm.fipc_group_score.score_poly_fipc_group_persons",
        "model_scope": "unidimensional_poly_grm_fipc",
        "converged": bool(result.fipc.converged),
        "focal_mu": float(result.fipc.mu),
        "focal_sigma": float(result.fipc.sigma),
        "theta_eap": result.theta_eap.tolist(),
        "theta_sd": result.theta_sd.tolist(),
        "expected_raw": result.expected_raw.tolist(),
        "n_persons": int(result.theta_eap.shape[0]),
    }
    if "reference_mu" in payload or "reference_sigma" in payload:
        mu_ref = float(payload.get("reference_mu", 0.0))
        sigma_ref = float(payload.get("reference_sigma", 1.0))
        slope_a, cat_a = _anchor_item_bank(anchor, anchor_slope, anchor_cat)
        moments = poly_reference_expected_score_moments(
            slope_a,
            cat_a,
            mu_ref=mu_ref,
            sigma_ref=sigma_ref,
            q_theta=int(payload["q_theta"]),
        )
        out["reference_expected_score"] = {
            "mu_ref": moments.mu_ref,
            "sigma_ref": moments.sigma_ref,
            "mean": moments.mean,
            "variance": moments.variance,
            "q_theta": moments.q_theta,
            "n_anchor_items": moments.n_anchor_items,
            "bank": "anchor_items_only",
        }
    return out
