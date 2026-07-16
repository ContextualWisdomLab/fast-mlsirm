"""Confirmatory MULTIDIMENSIONAL graded response model (Samejima, 1969; Muraki & Carlson, 1995).

Ordered polytomous categories with a single multidimensional discrimination vector per item and
ordered category boundaries: ``P(Y>=k|theta) = sigmoid(sum_d a_id theta_d + beta_i,{k-1})``. The
ordered counterpart of :func:`fast_mlsirm.fit_nominal_mirt` and the polytomous generalization of the
compensatory MIRT; reduces to the unidimensional GRM (``fit_poly_unidim``) at ``n_dims = 1``.
Estimated in the Rust core by Bock-Aitkin marginal MLE over a Gauss-Hermite (``n_dims <= 3``) or
Halton quasi-Monte-Carlo (``n_dims = 4..6``) grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_SUPPORTED_Q = (7, 11, 15, 21, 31, 41)
_MAX_DIMS_GH = 3
_MAX_DIMS_QMC = 6


@dataclass
class GrmMirtFit:
    """Fitted multidimensional graded response model (Samejima, 1969; Muraki & Carlson, 1995).

    ``slope`` is the ``n_items x n_dims`` discrimination matrix ``a_id`` (exactly ``0`` for
    dimensions not in the item's loading pattern), per-dimension reflection-canonicalized so each
    dimension's largest pure anchor is positive; ``threshold`` the ``n_items x (n_cat-1)`` ordered
    boundary intercepts ``beta_ik`` (strictly decreasing within each item); ``theta`` the
    ``n_persons x n_dims`` trait EAP. The model is
    ``P(Y_ij >= k | theta_j) = sigmoid(sum_d a_id theta_jd + beta_i,{k-1})`` with
    ``theta_j ~ MVN(0, I)``. ``termination_reason`` is ``"tolerance_met"`` or ``"max_iter_reached"``;
    ``final_loglik_change`` the SIGNED change ``ll_final - ll_prev`` (non-negative up to a tiny
    monotone-guard band)."""

    slope: np.ndarray
    threshold: np.ndarray
    theta: np.ndarray
    n_dims: int
    n_cat: int
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    termination_reason: str
    final_loglik_change: float
    n_parameters: int


def fit_grm_mirt(
    responses: np.ndarray,
    loading_pattern: np.ndarray,
    n_cat: int,
    q: int = 21,
    max_iter: int = 500,
    tol: float = 1e-6,
    node_rule: str = "gh",
    xi_points: int = 4000,
    xi_seed: int = 0x9E37_79B9_7F4A_7C15,
) -> GrmMirtFit:
    """Fit the confirmatory multidimensional graded response model (compute in Rust; Samejima, 1969;
    Muraki & Carlson, 1995).

    Ordered polytomous categories with a SINGLE multidimensional discrimination vector per item and
    ordered boundary intercepts: for category boundary ``k`` of item ``i``,
    ``P(Y >= k | theta) = sigmoid(sum_{d in S_i} a_id theta_d + beta_i,{k-1})``, where ``S_i`` is the
    item's loading set from the 0/1 ``loading_pattern`` (items x dimensions) and the ``n_cat-1``
    thresholds ``beta_i`` are strictly decreasing (Samejima's graded model). ``theta ~ MVN(0, I)``.
    Reduces to the unidimensional GRM at ``n_dims = 1``.

    Identification: unit trait variances + ordered thresholds + a PURE single-dimension anchor item
    per dimension fix rotation; the per-dimension reflection is CANONICALIZED (each dimension flipped
    so its largest pure anchor loads positive, leaving thresholds unchanged). Slopes are UNCONSTRAINED
    so reverse-keyed / negative cross-loadings are representable.

    **Integration nodes (``node_rule``).** ``"gh"`` (default) uses the ``q**n_dims`` Gauss-Hermite
    product grid and caps ``n_dims <= 3``. For ``n_dims = 4, 5, 6`` use ``"qmc"`` (Halton, Jank 2005)
    or ``"mc"`` with ``xi_points`` prior draws. ``q`` applies only to ``"gh"``; ``xi_points``/
    ``xi_seed`` only to ``"qmc"``/``"mc"``.

    ``responses`` is a persons x items integer-category array (``0..n_cat-1``; ``NaN`` or negative =
    missing, dropped MAR); ``loading_pattern`` an items x dimensions 0/1 array. Every declared
    category must be observed for each item, and every dimension needs a pure anchor item.

    References (APA 7th ed.):
        Samejima, F. (1969). Estimation of latent ability using a response pattern of graded
            scores. *Psychometrika Monograph Supplement, 34*(4, Pt. 2).
            https://doi.org/10.1007/BF03372160
        Muraki, E., & Carlson, J. E. (1995). Full-information factor analysis for polytomous item
            responses. *Applied Psychological Measurement, 19*(1), 73-90.
            https://doi.org/10.1177/014662169501900109
        Reckase, M. D. (2009). *Multidimensional item response theory*. Springer.
            https://doi.org/10.1007/978-0-387-89976-3
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_grm_mirt"):
        raise RuntimeError("fit_grm_mirt requires the compiled Rust core")

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
    _gh = str(node_rule).lower() in ("gh", "gauss-hermite", "gausshermite")
    _max_dims = _MAX_DIMS_GH if _gh else _MAX_DIMS_QMC
    if not 1 <= n_dims <= _max_dims:
        raise ValueError(
            f"loading_pattern dimensions must be between 1 and {_max_dims} (node_rule={node_rule!r})"
        )

    def _finite_int(value, name: str) -> int:
        scalar = np.asarray(value)
        if scalar.ndim != 0 or not np.issubdtype(scalar.dtype, np.number) or np.iscomplexobj(scalar):
            raise ValueError(f"{name} must be a finite integer")
        numeric = float(scalar)
        if not np.isfinite(numeric) or numeric != np.floor(numeric):
            raise ValueError(f"{name} must be a finite integer")
        return int(numeric)

    n_cat_int = _finite_int(n_cat, "n_cat")
    if n_cat_int < 2:
        raise ValueError("n_cat must be >= 2")
    q_int = _finite_int(q, "q")
    if _gh and q_int not in _SUPPORTED_Q:
        raise ValueError(f"q must be one of {_SUPPORTED_Q}")
    max_iter_int = _finite_int(max_iter, "max_iter")
    xi_points_int = _finite_int(xi_points, "xi_points")
    if isinstance(xi_seed, bool) or not isinstance(xi_seed, (int, np.integer)):
        raise ValueError("xi_seed must be a non-negative integer")
    xi_seed_int = int(xi_seed)
    if not 0 <= xi_seed_int < 2**64:
        raise ValueError("xi_seed must be in [0, 2**64)")

    observed = np.isfinite(y) & (y >= 0)
    if np.any(observed):
        observed_y = y[observed]
        if np.any(observed_y != np.floor(observed_y)) or observed_y.max() >= n_cat_int:
            raise ValueError("responses must be integer categories in 0..n_cat-1 where observed")
    yy = np.where(observed, y, 0.0).astype(np.int64).reshape(-1)

    res = core.fit_grm_mirt(
        yy,
        observed.reshape(-1),
        pat.astype(np.int64).reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_dims),
        n_cat_int,
        q_int,
        max_iter_int,
        float(tol),
        str(node_rule),
        xi_points_int,
        xi_seed_int,
    )
    return GrmMirtFit(
        slope=np.asarray(res["slope"], dtype=np.float64).reshape(n_items, n_dims),
        threshold=np.asarray(res["threshold"], dtype=np.float64).reshape(n_items, n_cat_int - 1),
        theta=np.asarray(res["theta"], dtype=np.float64).reshape(n_persons, n_dims),
        n_dims=int(res["n_dims"]),
        n_cat=int(res["n_cat"]),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        termination_reason=str(res["termination_reason"]),
        final_loglik_change=float(res["final_loglik_change"]),
        n_parameters=int(res["n_parameters"]),
    )
