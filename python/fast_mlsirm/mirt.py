"""Confirmatory compensatory multidimensional 2PL (MIRT).

Reckase (2009) / Bock, Gibbons & Muraki (1988) full-information item factor model, in
which an item may load freely on several latent dimensions that trade off additively in the
logit. Factors are orthogonal by default (``estimate_corr=False``, ``Sigma = I``) or their
correlation matrix is estimated (``estimate_corr=True``). Estimated in the Rust core over a
product Gauss-Hermite grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_SUPPORTED_Q = (7, 11, 15, 21, 31, 41)
_MAX_DIMS = 3


@dataclass
class CompMirtFit:
    """Fitted confirmatory compensatory MIRT (Reckase, 2009).

    ``loading`` is the items x dimensions matrix of free loadings ``a_id`` (exactly ``0``
    where the ``loading_pattern`` is ``0``); ``intercept`` the per-item ``b_i``; ``theta``
    the persons x dimensions trait EAP; ``corr`` the ``n_dims x n_dims`` latent correlation
    matrix (identity when ``estimate_corr=False``, estimated off-diagonals otherwise). The
    model is ``P(X_ij=1 | theta_j) = sigmoid(sum_d a_id theta_jd + b_i)`` with
    ``theta_j ~ MVN(0, Sigma)``, ``Sigma`` a unit-diagonal correlation matrix.
    ``termination_reason`` is either ``"converged"`` or ``"max_iter_reached"``;
    ``final_loglik_change`` is the absolute difference between the final two evaluated
    marginal log-likelihoods."""

    loading: np.ndarray
    intercept: np.ndarray
    theta: np.ndarray
    n_dims: int
    corr: np.ndarray
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    n_parameters: int
    termination_reason: str = "unknown"
    final_loglik_change: float = np.nan


def fit_compensatory_mirt(
    responses: np.ndarray,
    loading_pattern: np.ndarray,
    q: int = 21,
    estimate_corr: bool = False,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> CompMirtFit:
    """Fit the confirmatory compensatory MIRT (compute in Rust; Reckase, 2009;
    Bock, Gibbons & Muraki, 1988).

    A general COMPENSATORY multidimensional 2PL: an item may load freely on several latent
    dimensions, which trade off ADDITIVELY inside a single logit,
    ``P(X_ij=1 | theta_j) = sigmoid(sum_{d in S_i} a_id theta_jd + b_i)`` with
    ``theta_j ~ MVN(0, I_D)``. ``S_i`` is item ``i``'s loading set from the 0/1 confirmatory
    ``loading_pattern`` (items x dimensions); ``a_id`` is a free loading for ``d in S_i``
    (zero otherwise). This is distinct from the simple-structure MIRT (one dimension per
    item) and the orthogonal bifactor (one primary + one general per item): arbitrary
    within-item cross-loadings are allowed, which is why it needs the full ``q**n_dims``
    product quadrature (``n_dims <= 3``).

    Identification: unit trait variances fix the loading scale; the confirmatory pattern
    labels the dimensions PROVIDED every dimension has at least one PURE single-loading
    anchor item (rotationally-degenerate patterns such as all-ones are rejected); the
    per-dimension sign is fixed by a reflection anchor. Loadings are NOT constrained
    non-negative — reverse-keyed and suppressor cross-loadings are representable.

    **Latent traits.** With ``estimate_corr=False`` (default) the factors are ORTHOGONAL
    (``theta ~ MVN(0, I)``). With ``estimate_corr=True`` the inter-factor CORRELATION matrix
    ``Sigma`` (unit diagonal) is estimated by an ECM step (the standard GH grid is mapped
    through ``chol(Sigma)`` and the correlations ascend the Gaussian-prior objective with a
    positive-definite, monotone guard). ``n_dims > 3`` (which would need coarser GH or QMC) is
    a deferred extension.

    ``responses`` is a persons x items 0/1 array (``NaN`` = missing, dropped under MAR);
    ``loading_pattern`` is an items x dimensions 0/1 array; ``q`` is the Gauss-Hermite nodes
    per dimension (one of ``7, 11, 15, 21, 31, 41``). Convergence requires the absolute
    change between consecutive evaluated marginal log-likelihoods to be less than ``tol``;
    the returned fit exposes that value as ``final_loglik_change`` and the terminal state as
    ``termination_reason``.

    References (APA 7th ed.):
        Reckase, M. D. (2009). *Multidimensional item response theory*. Springer.
            https://doi.org/10.1007/978-0-387-89976-3
        Bock, R. D., Gibbons, R., & Muraki, E. (1988). Full-information item factor
            analysis. *Applied Psychological Measurement, 12*(3), 261-280.
            https://doi.org/10.1177/014662168801200305
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_compensatory_mirt"):
        raise RuntimeError("fit_compensatory_mirt requires the compiled Rust core")

    y = np.asarray(responses, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    pat = np.asarray(loading_pattern)
    if pat.ndim != 2:
        raise ValueError("loading_pattern must be a 2-D items x dimensions array")
    n_persons, n_items = y.shape
    if pat.shape[0] != n_items:
        raise ValueError("loading_pattern must have one row per item")
    if not np.issubdtype(pat.dtype, np.number) or np.iscomplexobj(pat):
        raise ValueError("loading_pattern entries must be numeric 0 or 1")
    if not np.all(np.isfinite(pat)) or not np.all((pat == 0) | (pat == 1)):
        raise ValueError("loading_pattern entries must be finite and exactly 0 or 1")
    n_dims = pat.shape[1]
    if not 1 <= n_dims <= _MAX_DIMS:
        raise ValueError(
            f"loading_pattern dimensions must be between 1 and {_MAX_DIMS}"
        )
    if np.isinf(y).any():
        raise ValueError("responses must be 0, 1, or NaN (missing)")

    def _finite_integer(value: int, name: str) -> int:
        scalar = np.asarray(value)
        if (
            scalar.ndim != 0
            or not np.issubdtype(scalar.dtype, np.number)
            or np.iscomplexobj(scalar)
        ):
            raise ValueError(f"{name} must be a finite integer")
        numeric = float(scalar)
        if not np.isfinite(numeric) or numeric != np.floor(numeric):
            raise ValueError(f"{name} must be a finite integer")
        return int(numeric)

    q_int = _finite_integer(q, "q")
    max_iter_int = _finite_integer(max_iter, "max_iter")
    if q_int not in _SUPPORTED_Q:
        raise ValueError(f"q must be one of {_SUPPORTED_Q}")

    observed = ~np.isnan(y)
    yy = np.where(observed, y, 0.0).reshape(-1)
    res = core.fit_compensatory_mirt(
        yy,
        observed.reshape(-1),
        pat.astype(np.int64).reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_dims),
        q_int,
        bool(estimate_corr),
        max_iter_int,
        float(tol),
    )
    return CompMirtFit(
        loading=np.asarray(res["loading"], dtype=np.float64).reshape(n_items, n_dims),
        intercept=np.asarray(res["intercept"], dtype=np.float64),
        theta=np.asarray(res["theta"], dtype=np.float64).reshape(n_persons, n_dims),
        n_dims=int(res["n_dims"]),
        corr=np.asarray(res["corr"], dtype=np.float64).reshape(n_dims, n_dims),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        n_parameters=int(res["n_parameters"]),
        termination_reason=str(res["termination_reason"]),
        final_loglik_change=float(res["final_loglik_change"]),
    )
