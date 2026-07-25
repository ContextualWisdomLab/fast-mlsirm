"""Testlet response model (Bradlow, Wainer, & Wang, 1999): a random-effects IRT model
for the local dependence induced when items share a common stimulus (a passage), fit
by marginal-ML EM in the Rust core."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .config import MAX_MAX_ITER


MAX_TESTLET_RESPONSE_CELLS = 20_000_000
_SUPPORTED_Q_GAMMA = (7, 11, 15, 21, 31, 41)


@dataclass
class TestletFit:
    """Fitted testlet model (Bradlow, Wainer, & Wang, 1999).

    ``a``/``b`` are the per-item discriminations and difficulties (``a`` is all ones
    for the Rasch model); ``beta = -a*b`` the intercept metric; ``sigma2`` the
    per-testlet variances ``sigma^2_d`` — the local-dependence estimand, one per
    testlet, where a large value flags strong within-testlet dependence and all zero
    is ordinary conditional-independence 2PL/Rasch. ``theta`` is the per-person EAP
    ability. Singleton testlets (one item) have ``sigma^2_d`` pinned to 0."""

    model: str
    a: np.ndarray
    b: np.ndarray
    beta: np.ndarray
    sigma2: np.ndarray
    theta: np.ndarray
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    n_parameters: int
    termination_reason: str = "unknown"
    final_loglik_change: float = np.nan


def fit_testlet(
    responses: np.ndarray,
    testlet_id: np.ndarray,
    model: str = "rasch",
    max_iter: int = 500,
    tol: float = 1e-6,
    q_gamma: int = 21,
    estimate_sigma: bool = True,
    init_sigma2: float = 0.5,
    require_convergence: bool = False,
) -> TestletFit:
    """Fit the testlet response model (compute in Rust; Bradlow, Wainer, & Wang, 1999).

    A testlet is a bundle of items sharing a stimulus; each item ``i`` in testlet
    ``d(i)`` gets a person-specific random effect ``gamma_{j,d(i)} ~ N(0, sigma^2_d)``,
    so ``P(X_ij=1) = sigmoid(a_i*(theta_j - b_i - gamma_{j,d(i)}))`` (Rasch fixes
    ``a_i=1``). The per-testlet variance ``sigma^2_d`` measures within-testlet local
    dependence; ``sigma^2_d = 0`` for every testlet is the ordinary 2PL/Rasch model,
    to which this reduces exactly (``estimate_sigma=False, init_sigma2=0``). Estimated
    by marginal-ML EM with a theta-outer / per-testlet-gamma-inner nested Gauss-Hermite
    quadrature (cost independent of the number of testlets), accelerated with SQUAREM.

    ``responses`` is a persons x items 0/1 array (``NaN`` = missing, dropped under MAR);
    ``testlet_id`` is a length-items integer array assigning each item to a testlet.
    Use ``model="rasch"`` for the well-identified case; in the 2PL testlet the
    discrimination ``a_i`` and the testlet SD ``sigma_d`` both scale the dependence via
    ``a_i*sigma_d`` and separate only weakly. The variance-component EM converges
    linearly, so a large ``sigma^2_d`` may want a generous ``max_iter``.
    Non-convergence emits ``RuntimeWarning`` and is recorded in
    ``termination_reason``; set ``require_convergence=True`` to raise instead.
    The repository-specific execution policy limits ``max_iter`` to 100,000 and
    the response matrix to 20,000,000 cells; these are resource guards, not
    properties of the testlet model.

    References (APA 7th ed.):
        Bradlow, E. T., Wainer, H., & Wang, X. (1999). A Bayesian random effects model
            for testlets. *Psychometrika, 64*(2), 153-168.
            https://doi.org/10.1007/BF02294533
        Wang, X., Bradlow, E. T., & Wainer, H. (2002). A general Bayesian model for
            testlets. *Applied Psychological Measurement, 26*(1), 109-128.
            https://doi.org/10.1177/0146621602026001007
    """
    raw_y = np.asarray(responses)
    if raw_y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    raw_tid = np.asarray(testlet_id)
    if raw_tid.ndim != 1:
        raise ValueError("testlet_id must be a 1-D array")
    n_persons, n_items = raw_y.shape
    if n_items < 1:
        raise ValueError("responses and testlet_id must describe a non-empty item bank")
    if n_persons < 1:
        raise ValueError("responses must contain at least one person")
    if raw_y.size > MAX_TESTLET_RESPONSE_CELLS:
        raise ValueError(
            "response matrix exceeds the "
            f"{MAX_TESTLET_RESPONSE_CELLS}-cell testlet-calibration limit"
        )
    try:
        y = np.asarray(raw_y, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("responses must contain numeric 0/1 values or NaN") from exc
    if not np.all(np.isnan(y) | (np.isfinite(y) & ((y == 0.0) | (y == 1.0)))):
        raise ValueError("responses must be 0/1 or NaN (missing)")
    if raw_tid.shape[0] != n_items:
        raise ValueError("testlet_id must have length n_items")
    if not np.issubdtype(raw_tid.dtype, np.integer) or np.issubdtype(
        raw_tid.dtype, np.bool_
    ):
        raise ValueError("testlet_id entries must be integers")
    if not np.all((raw_tid >= 0) & (raw_tid < n_items)):
        raise ValueError("testlet_id entries must be between 0 and n_items - 1")
    tid = raw_tid.astype(np.int64, copy=False)
    n_testlets = int(tid.max()) + 1
    if (
        isinstance(max_iter, (bool, np.bool_))
        or not isinstance(max_iter, (int, np.integer))
        or not 1 <= int(max_iter) <= MAX_MAX_ITER
    ):
        raise ValueError(f"max_iter must be an integer between 1 and {MAX_MAX_ITER}")
    if isinstance(tol, (bool, np.bool_)) or not isinstance(
        tol, (int, float, np.integer, np.floating)
    ):
        raise ValueError("tol must be a finite non-negative number")
    tol_value = float(tol)
    if not np.isfinite(tol_value) or tol_value < 0.0:
        raise ValueError("tol must be a finite non-negative number")
    if (
        isinstance(q_gamma, (bool, np.bool_))
        or not isinstance(q_gamma, (int, np.integer))
        or int(q_gamma) not in _SUPPORTED_Q_GAMMA
    ):
        raise ValueError(f"q_gamma must be one of {_SUPPORTED_Q_GAMMA}")
    if isinstance(init_sigma2, (bool, np.bool_)) or not isinstance(
        init_sigma2, (int, float, np.integer, np.floating)
    ):
        raise ValueError("init_sigma2 must be a finite non-negative number")
    init_sigma2_value = float(init_sigma2)
    if not np.isfinite(init_sigma2_value) or init_sigma2_value < 0.0:
        raise ValueError("init_sigma2 must be a finite non-negative number")

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_testlet"):
        raise RuntimeError("fit_testlet requires the compiled Rust core")

    observed = ~np.isnan(y)
    yy = np.where(observed, y, 0.0).reshape(-1)
    res = core.fit_testlet(
        yy,
        observed.reshape(-1),
        tid,
        int(n_persons),
        int(n_items),
        int(n_testlets),
        str(model),
        int(max_iter),
        tol_value,
        int(q_gamma),
        bool(estimate_sigma),
        init_sigma2_value,
    )
    fit = TestletFit(
        model=str(res["model"]),
        a=np.asarray(res["a"], dtype=np.float64),
        b=np.asarray(res["b"], dtype=np.float64),
        beta=np.asarray(res["beta"], dtype=np.float64),
        sigma2=np.asarray(res["sigma2"], dtype=np.float64),
        theta=np.asarray(res["theta"], dtype=np.float64),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        n_parameters=int(res["n_parameters"]),
        termination_reason=str(res["termination_reason"]),
        final_loglik_change=float(res["final_loglik_change"]),
    )
    if not fit.converged:
        message = (
            "testlet calibration did not converge: "
            f"reason={fit.termination_reason}, iterations={fit.n_iter}/{max_iter}, "
            "final_loglik_change="
            f"{fit.final_loglik_change:.12g}, tolerance={tol_value:.12g}"
        )
        if require_convergence:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return fit
