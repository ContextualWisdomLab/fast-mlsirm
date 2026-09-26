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
    predict_expected_response_polytomous,
    score_polytomous,
)


def _as_poly_fit(fipc: PolyFipcFit) -> PolytomousFit:
    """Wrap FIPC item parameters as a GRM :class:`PolytomousFit` for scoring.

    Callers pass the focal prior explicitly to :func:`score_polytomous`.
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
    An unconverged calibration raises before any interpretation-facing score
    is computed.

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
    if not fipc.converged:
        raise RuntimeError(
            "FIPC calibration did not converge: "
            f"termination_reason={fipc.termination_reason}, n_iter={fipc.n_iter}"
        )
    bank = _as_poly_fit(fipc)
    scored = score_polytomous(
        responses, bank, q_theta=q_theta, prior_mean=fipc.mu, prior_sd=fipc.sigma
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
