"""Continuous Response Model (Samejima, 1973): item response theory for a
continuous bounded response, estimated by marginal-ML EM in the Rust core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CrmFit:
    """Fitted continuous response model (Samejima, 1973).

    The logit of the response is conditionally normal and linear in the trait:
    ``logit(Z_ij) | theta_j ~ N(slope_i * theta_j + intercept_i, resid_sd_i^2)`` with
    ``theta ~ N(0, 1)``. ``slope``/``intercept``/``resid_sd`` are the working item
    parameters; ``discrimination = slope / resid_sd`` and
    ``difficulty = -intercept / slope`` are the classic Samejima ``(alpha, b)``.
    ``theta`` is the per-person EAP trait score."""

    slope: np.ndarray
    intercept: np.ndarray
    resid_sd: np.ndarray
    discrimination: np.ndarray
    difficulty: np.ndarray
    theta: np.ndarray
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    n_parameters: int


def fit_crm(
    responses: np.ndarray,
    q_theta: int = 41,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> CrmFit:
    """Fit the continuous response model (compute in Rust; Samejima, 1973).

    Samejima's CRM is the limit of the graded response model as the number of ordered
    categories grows without bound, for an item scored on a *continuous* bounded scale.
    Operationally (Wang & Zeng, 1998), the logit of a response ``Z in (0, 1)`` is
    conditionally normal and linear in the latent trait:
    ``logit(Z_ij) | theta_j ~ N(a_i theta_j + d_i, sigma_i^2)``, ``theta ~ N(0, 1)``.
    The item slope ``a_i``, intercept ``d_i``, and residual sd ``sigma_i`` map to the
    classic ``(discrimination alpha_i = a_i/sigma_i, difficulty b_i = -d_i/a_i,
    scale gamma_i = a_i)``. Estimated by marginal-ML EM with a Gauss-Hermite
    quadrature over the trait and a closed-form weighted-least-squares item M-step.

    ``responses`` is a persons x items array of values in the open interval ``(0, 1)``
    (values are clamped to ``[eps, 1-eps]`` before the logit transform; ``NaN`` marks a
    missing cell, dropped under a missing-at-random assumption). The trait is
    identified up to a global sign, resolved so the mean slope is non-negative.

    References (APA 7th ed.):
        Samejima, F. (1973). Homogeneous case of the continuous response model.
            *Psychometrika, 38*(2), 203-219. https://doi.org/10.1007/BF02291114
        Wang, T., & Zeng, L. (1998). Item parameter estimation for a continuous
            response model using an EM algorithm. *Applied Psychological Measurement,
            22*(4), 333-344. https://doi.org/10.1177/014662169802200402
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_crm"):
        raise RuntimeError("fit_crm requires the compiled Rust core")

    y = np.asarray(responses, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape

    observed = np.isfinite(y)
    yy = np.where(observed, y, 0.5).reshape(-1)
    res = core.fit_crm(
        yy,
        observed.reshape(-1),
        int(n_persons),
        int(n_items),
        int(q_theta),
        int(max_iter),
        float(tol),
    )
    return CrmFit(
        slope=np.asarray(res["slope"], dtype=np.float64),
        intercept=np.asarray(res["intercept"], dtype=np.float64),
        resid_sd=np.asarray(res["resid_sd"], dtype=np.float64),
        discrimination=np.asarray(res["discrimination"], dtype=np.float64),
        difficulty=np.asarray(res["difficulty"], dtype=np.float64),
        theta=np.asarray(res["theta"], dtype=np.float64),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        n_parameters=int(res["n_parameters"]),
    )
