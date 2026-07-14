"""Lognormal response-time model (van der Linden, 2007): the speed-side analogue
of the 2PL for item response *times*, estimated by marginal-ML EM in the Rust
core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RtFit:
    """Fitted lognormal response-time model. ``alpha``/``beta`` are the per-item
    time discriminations and time intensities; ``sigma_tau`` the estimated speed
    SD (``mu_tau`` is pinned to 0 for identification); ``tau_eap``/``tau_sd`` the
    per-person EAP speed and its posterior SD."""

    alpha: np.ndarray
    beta: np.ndarray
    mu_tau: float
    sigma_tau: float
    tau_eap: np.ndarray
    tau_sd: np.ndarray
    loglik: float
    n_iter: int
    converged: bool


def fit_response_times(
    times: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-6,
    var_floor: float = 1e-4,
    sigma_floor: float = 1e-4,
    fix_sigma_tau: float | None = None,
) -> RtFit:
    """Fit the lognormal response-time measurement model (compute in Rust; van der
    Linden, 2007): ``ln(T_ij) ~ Normal(beta_i - tau_j, 1/alpha_i^2)`` for person
    ``j`` (latent speed ``tau_j``) and item ``i`` (time intensity ``beta_i``, time
    discrimination ``alpha_i``). Item parameters and the speed SD are estimated by
    marginal-ML EM with ``tau ~ Normal(0, sigma_tau^2)``, and speed is scored by
    EAP. ``times`` is a persons x items array of raw response times; non-positive
    or ``NaN`` entries are treated as missing (marginalized per person). By default
    ``sigma_tau`` is estimated (the log-time metric identifies the speed scale);
    pass ``fix_sigma_tau`` only to impose a deliberately standardized metric.

    References (APA 7th ed.):
        van der Linden, W. J. (2007). A hierarchical framework for modeling speed
            and accuracy on test items. *Psychometrika, 72*(3), 287–308.
            https://doi.org/10.1007/s11336-006-1478-z
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_rt_lognormal"):
        raise RuntimeError("fit_response_times requires the compiled Rust core")
    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 2:
        raise ValueError("times must be a 2-D persons x items array")
    n_persons, n_items = t.shape
    observed = np.isfinite(t) & (t > 0)
    obs_arg = None if observed.all() else observed.reshape(-1)
    tt = np.where(observed, t, 1.0).reshape(-1)  # masked entries get a valid placeholder
    res = core.fit_rt_lognormal(
        tt, obs_arg, int(n_persons), int(n_items),
        int(max_iter), float(tol), float(var_floor), float(sigma_floor),
        None if fix_sigma_tau is None else float(fix_sigma_tau),
    )
    return RtFit(
        alpha=np.asarray(res["alpha"], dtype=np.float64),
        beta=np.asarray(res["beta"], dtype=np.float64),
        mu_tau=float(res["mu_tau"]),
        sigma_tau=float(res["sigma_tau"]),
        tau_eap=np.asarray(res["tau_eap"], dtype=np.float64),
        tau_sd=np.asarray(res["tau_sd"], dtype=np.float64),
        loglik=float(res["loglik"]),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
    )


def fit_speed_accuracy(
    responses: np.ndarray,
    times: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    q: int = 21,
    max_iter: int = 500,
    tol: float = 1e-6,
    fix_sigma_tau: float | None = None,
) -> dict:
    """Estimate a two-stage marginal-ML adaptation of the joint speed-accuracy
    person covariance in van der Linden (2007) (compute in Rust) -- the
    ability-speed correlation ``rho`` and speed SD ``sigma_tau`` -- over a 2-D
    Gauss-Hermite grid with item parameters held fixed. The original article uses
    a normal-ogive response model and Bayesian MCMC; the fixed-bank logistic 2PL
    estimator here is a repository-specific adaptation, not an estimator reported
    in that article. ``responses`` (0/1) and ``times`` (> 0) are persons x items
    arrays sharing a missingness mask (``NaN``/non-positive = missing); ``a``/``b``
    are the accuracy 2PL raw slope/intercept (``eta = a_i*theta + b_i``);
    ``alpha``/``beta`` are the lognormal time discrimination/intensity (e.g. from
    :func:`fit_response_times`). Returns a dict with ``rho``, ``sigma_tau``,
    ``s_theta2`` (a theta-metric diagnostic ~1), joint ``theta_eap``/``tau_eap``,
    ``loglik``, ``n_iter``, ``converged``.

    ``rho`` here is the consistent marginal-ML correlation, NOT the attenuated
    correlation of the two separately-scored EAPs (which shrinks toward 0).

    References (APA 7th ed.):
        van der Linden, W. J. (2007). A hierarchical framework for modeling speed
            and accuracy on test items. *Psychometrika, 72*(3), 287–308.
            https://doi.org/10.1007/s11336-006-1478-z
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_speed_accuracy_covariance"):
        raise RuntimeError("fit_speed_accuracy requires the compiled Rust core")
    u = np.asarray(responses, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    if u.ndim != 2 or t.shape != u.shape:
        raise ValueError("responses and times must be matching 2-D persons x items arrays")
    n_persons, n_items = u.shape
    observed = np.isfinite(u) & np.isfinite(t) & (t > 0)
    obs_arg = None if observed.all() else observed.reshape(-1)
    uu = np.where(observed, u, 0.0).reshape(-1)
    tt = np.where(observed, t, 1.0).reshape(-1)
    res = core.fit_speed_accuracy_covariance(
        uu, tt, obs_arg,
        np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64),
        np.asarray(alpha, dtype=np.float64), np.asarray(beta, dtype=np.float64),
        int(n_persons), int(n_items),
        int(q), int(max_iter), float(tol),
        None if fix_sigma_tau is None else float(fix_sigma_tau),
    )
    return {
        "rho": float(res["rho"]),
        "sigma_tau": float(res["sigma_tau"]),
        "s_theta2": float(res["s_theta2"]),
        "theta_eap": np.asarray(res["theta_eap"], dtype=np.float64),
        "tau_eap": np.asarray(res["tau_eap"], dtype=np.float64),
        "loglik": float(res["loglik"]),
        "n_iter": int(res["n_iter"]),
        "converged": bool(res["converged"]),
    }


def rt_person_fit(
    times: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    alpha_level: float = 0.05,
    z_fast: float = 1.645,
) -> dict:
    """Sinharay's (2018) frequentist response-time person-fit statistic (computed
    in Rust) under a fitted lognormal RT model. It profiles each person's speed by
    ML, so the sum of squared standardized log-time residuals ``W = sum_i z_i^2`` is exactly
    ``chi2(n_j - 1)`` under the model (a clean one-df correction for the estimated
    speed, the RT analogue of ``l_z*``). Detects speed *inconsistency across items*
    -- rapid guessing or item preknowledge, which appear as clusters of strongly
    negative residuals -- but not a uniform speed level (the profile absorbs it).
    ``times`` is a persons x items array of raw response times (``NaN``/non-positive
    = missing); ``alpha``/``beta`` come from :func:`fit_response_times`. Returns a
    dict with per-person ``w``, ``df``, ``l_t`` (an API-compatible field containing
    the Wilson-Hilferty standardization, approximately ``N(0,1)``), ``p_value``
    (upper-tail chi-square), ``flagged`` (``p < alpha_level``),
    ``tau_ml`` (profiled speed), and persons x items ``z_resid`` (studentized
    residuals; strongly negative = too fast) and ``item_flag`` (one-sided too-fast).
    The item residuals are a fixed-bank diagnostic in this package. Van der Linden
    and Guo (2008) motivate the aberrant-fast-response interpretation, but their
    Bayesian leave-one-out procedure is not implemented here.

    References (APA 7th ed.):
        van der Linden, W. J., & Guo, F. (2008). Bayesian procedures for
            identifying aberrant response-time patterns in adaptive testing.
            *Psychometrika, 73*(3), 365–384.
            https://doi.org/10.1007/s11336-007-9046-8
        Sinharay, S. (2018). A new person-fit statistic for the lognormal model for
            response times. *Journal of Educational Measurement, 55*(4), 457–476.
            https://doi.org/10.1111/jedm.12188
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "rt_person_fit"):
        raise RuntimeError("rt_person_fit requires the compiled Rust core")
    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 2:
        raise ValueError("times must be a 2-D persons x items array")
    n_persons, n_items = t.shape
    observed = np.isfinite(t) & (t > 0)
    obs_arg = None if observed.all() else observed.reshape(-1)
    tt = np.where(observed, t, 1.0).reshape(-1)
    res = core.rt_person_fit(
        tt, obs_arg, int(n_persons), int(n_items),
        np.asarray(alpha, dtype=np.float64), np.asarray(beta, dtype=np.float64),
        float(alpha_level), float(z_fast),
    )
    return {
        "w": np.asarray(res["w"], dtype=np.float64),
        "df": np.asarray(res["df"], dtype=np.int64),
        "l_t": np.asarray(res["l_t"], dtype=np.float64),
        "p_value": np.asarray(res["p_value"], dtype=np.float64),
        "flagged": np.asarray(res["flagged"], dtype=bool),
        "tau_ml": np.asarray(res["tau_ml"], dtype=np.float64),
        "z_resid": np.asarray(res["z_resid"], dtype=np.float64).reshape(n_persons, n_items),
        "item_flag": np.asarray(res["item_flag"], dtype=bool).reshape(n_persons, n_items),
    }
