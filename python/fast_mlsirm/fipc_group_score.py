# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Generic FIPC group person scoring and reference expected-score distribution.

Library-generic API for focal-group fixed-item calibration followed by person
EAP / expected-raw scores on the reference metric identified by anchors.
Callers map research labels (e.g. age bands) onto group batches externally —
this module never encodes domain polarity or age semantics.

Identification (Kim, 2006): anchors pin the item bank to the reference
calibration scale; focal ``N(mu, sigma^2)`` is free. Person EAP uses the
standard reference-metric quadrature in :func:`score_polytomous` (Bock &
Mislevy, 1982). Expected raw totals are plug-in sums of
:func:`predict_expected_response_polytomous` at those EAPs (Lord, 1980,
test characteristic / number-right true score).

The companion reference distribution returns Gauss-Hermite moments of the
expected total score under a caller-supplied reference prior
``N(mu_ref, sigma_ref^2)`` — the population against which focal group means
are compared after FIPC linking.

Not the #2077 ``E[T|theta_focal]`` nuisance-integrated curve API; that path
stays on the two-tier expected-raw PR. This module is the poly-GRM FIPC
person-score + reference-distribution contract for remote consumers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .estimators.mmle import gauss_hermite_nodes
from .polytomous import (
    PolyFipcFit,
    PolytomousFit,
    fit_poly_fipc,
    predict_expected_response_polytomous,
    score_polytomous,
)


def _as_poly_fit(fipc: PolyFipcFit) -> PolytomousFit:
    """Wrap FIPC item parameters as a GRM :class:`PolytomousFit` for scoring."""
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


@dataclass(frozen=True)
class PolyFipcGroupPersonScores:
    """Person scores for one focal group after poly-GRM FIPC.

    ``theta_eap`` / ``theta_sd`` are on the reference metric fixed by anchors.
    ``expected_raw`` is the plug-in expected total at each person's EAP.
    ``fipc`` holds the focal calibration (``mu`` / ``sigma`` = focal prior).
    """

    theta_eap: np.ndarray
    theta_sd: np.ndarray
    expected_raw: np.ndarray
    fipc: PolyFipcFit


@dataclass(frozen=True)
class PolyReferenceExpectedScoreMoments:
    """Moments of ``E[T|theta]`` when ``theta ~ N(mu_ref, sigma_ref^2)``.

    This is the reference expected-score distribution used to place focal
    group means after FIPC linking — not an Orlando–Thissen item-fit table.
    """

    mu_ref: float
    sigma_ref: float
    q_theta: int
    mean: float
    second_moment: float
    variance: float


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
    scored = score_polytomous(responses, bank, q_theta=q_theta)
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

    Transforms probabilists' Gauss-Hermite nodes to ``N(mu_ref, sigma_ref^2)``
    and evaluates ``E[T|theta] = sum_i E[Y_i|theta]`` at each node (Lord, 1980,
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
    # probabilists' GH integrates f(x) e^{-x^2/2} / sqrt(2π); scale to N(mu, σ²)
    theta = mu + sigma * nodes
    bank = PolytomousFit(
        model="grm",
        slope=slope_arr,
        cat_params=cat_arr,
        loglik=0.0,
        n_iter=0,
        converged=True,
        termination_reason="reference_bank",
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
    )


def execute_fipc_group_person_score_payload(payload: dict[str, object]) -> dict[str, object]:
    """Remote/worker JSON contract for :func:`score_poly_fipc_group_persons`.

    Required keys: ``responses``, ``n_cat``, ``anchor``, ``anchor_slope``,
    ``anchor_cat_params``, ``q_theta``, ``max_iter``, ``tol``.
    Optional: ``reference_mu`` / ``reference_sigma`` (defaults 0 / 1) to also
    emit reference expected-score moments from the **anchor** bank (reference
    identification), not the focal free items.
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

    result = score_poly_fipc_group_persons(
        np.asarray(payload["responses"]),
        int(payload["n_cat"]),
        np.asarray(payload["anchor"], dtype=bool),
        np.asarray(payload["anchor_slope"], dtype=np.float64),
        np.asarray(payload["anchor_cat_params"], dtype=np.float64),
        q_theta=int(payload["q_theta"]),
        max_iter=int(payload["max_iter"]),
        tol=float(payload["tol"]),
    )
    out: dict[str, object] = {
        "family": "fipc_group_person_score",
        "library_function": "fast_mlsirm.fipc_group_score.score_poly_fipc_group_persons",
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
        # Reference moments use the anchor-pinned bank on the reference metric.
        moments = poly_reference_expected_score_moments(
            result.fipc.slope,
            result.fipc.cat_params,
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
        }
    return out
