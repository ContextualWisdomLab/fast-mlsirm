"""Confirmatory MULTIDIMENSIONAL nominal response model (Bock, 1972; Thissen, Cai & Bock, 2010).

Each item's unordered categories get a free multidimensional discrimination and intercept; the
category probability is a softmax of ``sum_d a_ikd theta_d + c_ik`` with the baseline category
pinned to zero. Generalizes the unidimensional :func:`fast_mlsirm.fit_nominal` to ``n_dims`` latent
dimensions (reducing to it at ``n_dims = 1``). Estimated in the Rust core by Bock-Aitkin marginal
MLE over a Gauss-Hermite (``n_dims <= 3``) or Halton quasi-Monte-Carlo (``n_dims = 4..6``) grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_SUPPORTED_Q = (7, 11, 15, 21, 31, 41)
_MAX_DIMS_GH = 3
_MAX_DIMS_QMC = 6


@dataclass
class NominalMirtFit:
    """Fitted multidimensional nominal response model (Bock, 1972).

    ``slope`` is the ``n_items x n_cat x n_dims`` category-slope tensor ``a_ikd`` (exactly ``0`` for
    the baseline category ``k = 0`` and for dimensions not in the item's loading pattern);
    ``intercept`` the ``n_items x n_cat`` category intercepts ``c_ik`` (baseline ``0``); ``theta``
    the ``n_persons x n_dims`` trait EAP. The model is
    ``P(Y_ij = k | theta_j) = softmax_k(sum_d a_ikd theta_jd + c_ik)`` with ``theta_j ~ MVN(0, I)``,
    identified up to a per-dimension reflection ``(a_i.d, theta_d) -> (-a_i.d, -theta_d)``.
    ``termination_reason`` is ``"tolerance_met"`` or ``"max_iter_reached"``; ``final_loglik_change``
    the SIGNED change ``ll_final - ll_prev`` between the final two evaluated marginal
    log-likelihoods (non-negative up to a tiny monotone-guard band)."""

    slope: np.ndarray
    intercept: np.ndarray
    theta: np.ndarray
    n_dims: int
    n_cat: int
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    termination_reason: str
    final_loglik_change: float
    n_parameters: int


def fit_nominal_mirt(
    responses: np.ndarray,
    loading_pattern: np.ndarray,
    n_cat: int,
    q: int = 21,
    max_iter: int = 500,
    tol: float = 1e-6,
    node_rule: str = "gh",
    xi_points: int = 4000,
    xi_seed: int = 0x9E37_79B9_7F4A_7C15,
) -> NominalMirtFit:
    """Fit the confirmatory multidimensional nominal response model (compute in Rust; Bock, 1972;
    Thissen, Cai & Bock, 2010).

    Unordered polytomous categories with CATEGORY-SPECIFIC multidimensional discrimination: for
    category ``k`` of item ``i`` the linear predictor is ``eta_ik = sum_{d in S_i} a_ikd theta_d +
    c_ik`` and ``P(Y=k | theta) = softmax_k(eta_ik)``, with the baseline category ``0`` pinned
    ``a_i0 = 0, c_i0 = 0``. ``S_i`` is item ``i``'s loading set from the 0/1 ``loading_pattern``
    (items x dimensions): a slope ``a_ikd`` is free only for ``d in S_i``. ``theta ~ MVN(0, I)``.
    At ``n_dims = 1`` this reduces to :func:`fast_mlsirm.fit_nominal` (the same general free-``a_k``
    parametrization).

    Identification: baseline category + unit trait variances + a PURE single-dimension anchor item
    per dimension (an item loading exactly one dimension) fix the rotation; parameters are identified
    up to a per-dimension reflection (not canonicalized, as in :func:`fit_nominal`).

    **Integration nodes (``node_rule``).** ``"gh"`` (default) uses the ``q**n_dims`` Gauss-Hermite
    product grid and caps ``n_dims <= 3``. For ``n_dims = 4, 5, 6`` use ``"qmc"`` (Halton
    quasi-Monte-Carlo, Jank 2005) or ``"mc"`` (Monte-Carlo) with ``xi_points`` prior draws. ``q``
    applies only to ``"gh"``; ``xi_points``/``xi_seed`` only to ``"qmc"``/``"mc"``.

    ``responses`` is a persons x items integer-category array (``0..n_cat-1``; ``NaN`` or negative =
    missing, dropped MAR); ``loading_pattern`` an items x dimensions 0/1 array. Every declared
    category must be observed for each item, and every dimension needs a pure anchor item.

    References (APA 7th ed.):
        Bock, R. D. (1972). Estimating item parameters and latent ability when responses are
            scored in two or more nominal categories. *Psychometrika, 37*(1), 29-51.
            https://doi.org/10.1007/BF02291411
        Thissen, D., Cai, L., & Bock, R. D. (2010). The nominal categories item response model.
            In *Handbook of polytomous item response theory models* (pp. 43-75). Routledge.
        Reckase, M. D. (2009). *Multidimensional item response theory*. Springer.
            https://doi.org/10.1007/978-0-387-89976-3
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_nominal_mirt"):
        raise RuntimeError("fit_nominal_mirt requires the compiled Rust core")

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
    # xi_seed is a full-range u64: validate as an exact integer, no float64 round-trip.
    if isinstance(xi_seed, bool) or not isinstance(xi_seed, (int, np.integer)):
        raise ValueError("xi_seed must be a non-negative integer")
    xi_seed_int = int(xi_seed)
    if not 0 <= xi_seed_int < 2**64:
        raise ValueError("xi_seed must be in [0, 2**64)")

    # missing = NaN or negative; the core takes a categories array + an observed mask.
    observed = np.isfinite(y) & (y >= 0)
    if np.any(observed):
        maxc = y[observed].max()
        if maxc >= n_cat_int:
            raise ValueError("responses must be integer categories in 0..n_cat-1 where observed")
    yy = np.where(observed, y, 0.0).astype(np.int64).reshape(-1)

    res = core.fit_nominal_mirt(
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
    return NominalMirtFit(
        slope=np.asarray(res["slope"], dtype=np.float64).reshape(n_items, n_cat_int, n_dims),
        intercept=np.asarray(res["intercept"], dtype=np.float64).reshape(n_items, n_cat_int),
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
