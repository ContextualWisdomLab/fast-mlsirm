"""Dimension-agnostic generalized partial credit model (Muraki, 1992).

Ordered categories share one discrimination vector and use INTEGER category
scores with free (unordered) adjacent-category step intercepts. The public
``model=`` argument selects the one-factor model or a confirmatory
multidimensional loading specification; the numerical estimation runs in Rust."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MAX_MAX_ITER, MAX_POLYTOMOUS_CATEGORIES
from .models import ConfirmatoryModel, ExploratoryModel, IrtModel, _resolve_model

_SUPPORTED_Q = (7, 11, 15, 21, 31, 41)
_MAX_DIMS_GH = 3
_MAX_DIMS_QMC = 6
_MAX_NODES = 200_000


@dataclass
class GpcmFit:
    """Fitted multidimensional generalized partial credit model (Muraki, 1992).

    ``slope`` is the ``n_items x n_dims`` discrimination matrix ``a_id`` (exactly ``0`` for
    dimensions not in the item's loading pattern), per-dimension reflection-canonicalized so each
    dimension's largest pure anchor is positive; ``step`` the ``n_items x (n_cat-1)`` category step
    intercepts ``step_ik`` (UNORDERED — the GPCM softmax is valid for any values); ``theta`` the
    ``n_persons x n_dims`` trait EAP. The model is
    ``P(Y_ij = k | theta_j) = softmax_k(k * sum_d a_id theta_jd + step_ik)`` with
    ``theta_j ~ MVN(0, I)``. ``termination_reason`` is ``"tolerance_met"`` or ``"max_iter_reached"``;
    ``final_loglik_change`` the SIGNED change ``ll_final - ll_prev`` (non-negative up to a tiny
    monotone-guard band)."""

    model: IrtModel
    slope: np.ndarray
    step: np.ndarray
    theta: np.ndarray
    n_cat: int
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    termination_reason: str
    final_loglik_change: float
    n_parameters: int

    @property
    def n_dims(self) -> int:
        """Latent dimension count derived from :attr:`model`."""

        return self.model.n_dims


def fit_gpcm(
    responses: np.ndarray,
    n_cat: int,
    model: int | ExploratoryModel | ConfirmatoryModel = 1,
    q: int = 21,
    max_iter: int = 500,
    tol: float = 1e-6,
    node_rule: str = "gh",
    xi_points: int = 4000,
    xi_seed: int = 0x9E37_79B9_7F4A_7C15,
) -> GpcmFit:
    """Fit the generalized partial credit model (compute in Rust; Muraki, 1992).

    Ordered polytomous categories with a SINGLE multidimensional discrimination vector per item and
    INTEGER category scores: for category ``k`` of item ``i``, ``psi_ik = k * sum_{d in S_i} a_id
    theta_d + step_ik`` and ``P(Y = k | theta) = softmax_k(psi_ik)``, where ``S_i`` is the item's
    loading set from the confirmatory model specification and the ``n_cat-1`` steps ``step_i`` are free
    and UNORDERED (the softmax is valid for any values). ``theta ~ MVN(0, I)``. This is the
    ``a_ikd = k * a_id`` integer-scoring restriction of the nominal model, in a single-slope
    parametrization; it reduces to the unidimensional GPCM at ``n_dims = 1``.

    Identification: unit trait variances + a PURE single-dimension anchor item per dimension fix
    rotation; the per-dimension reflection is CANONICALIZED (each dimension flipped so its largest pure
    anchor loads positive, leaving steps unchanged). Slopes are UNCONSTRAINED so reverse-keyed /
    negative cross-loadings are representable.

    **Integration nodes (``node_rule``).** ``"gh"`` (default) uses the ``q**n_dims`` Gauss-Hermite
    product grid and caps ``n_dims <= 3``. For ``n_dims = 4, 5, 6`` use ``"qmc"`` (Halton, Jank 2005)
    or ``"mc"`` with ``xi_points`` prior draws. ``q`` applies only to ``"gh"``; ``xi_points``/
    ``xi_seed`` only to ``"qmc"``/``"mc"``.

    ``responses`` is a persons x items integer-category array (``0..n_cat-1``; ``NaN`` or negative =
    missing, dropped MAR); For ``model=1``, all item slopes on the single factor are free. A
    multidimensional confirmatory structure is supplied with
    ``model=models.confirmatory(loading_pattern)``; a numeric exploratory model greater than
    one is rejected until unrestricted loading rotation and identification are implemented.
    ``n_cat`` is limited to 2..64, ``max_iter`` to 1..100,000, and Monte Carlo/QMC
    ``xi_points`` to 1..200,000. Every declared category must be observed for each item, and every
    dimension needs a pure anchor item.

    References (APA 7th ed.):
        Muraki, E. (1992). A generalized partial credit model: Application of an EM algorithm.
            *Applied Psychological Measurement, 16*(2), 159–176.
            https://doi.org/10.1177/014662169201600206
        Reckase, M. D. (2009). *Multidimensional item response theory*. Springer.
            https://doi.org/10.1007/978-0-387-89976-3
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_gpcm"):
        raise RuntimeError("fit_gpcm requires the compiled Rust core")

    y = np.asarray(responses, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    resolved_model, pat = _resolve_model(model, n_items)
    n_dims = pat.shape[1]
    _gh = str(node_rule).lower() in ("gh", "gauss-hermite", "gausshermite")
    _max_dims = _MAX_DIMS_GH if _gh else _MAX_DIMS_QMC
    if not 1 <= n_dims <= _max_dims:
        raise ValueError(
            f"loading_pattern dimensions must be between 1 and {_max_dims} (node_rule={node_rule!r})"
        )

    def _finite_int(value, name: str) -> int:
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

    n_cat_int = _finite_int(n_cat, "n_cat")
    if not 2 <= n_cat_int <= MAX_POLYTOMOUS_CATEGORIES:
        raise ValueError(f"n_cat must be between 2 and {MAX_POLYTOMOUS_CATEGORIES}")
    q_int = _finite_int(q, "q")
    if _gh and q_int not in _SUPPORTED_Q:
        raise ValueError(f"q must be one of {_SUPPORTED_Q}")
    max_iter_int = _finite_int(max_iter, "max_iter")
    xi_points_int = _finite_int(xi_points, "xi_points")
    if not 1 <= max_iter_int <= MAX_MAX_ITER:
        raise ValueError(f"max_iter must be between 1 and {MAX_MAX_ITER}")
    if not 1 <= xi_points_int <= _MAX_NODES:
        raise ValueError(f"xi_points must be between 1 and {_MAX_NODES}")
    if isinstance(xi_seed, bool) or not isinstance(xi_seed, (int, np.integer)):
        raise ValueError("xi_seed must be a non-negative integer")
    xi_seed_int = int(xi_seed)
    if not 0 <= xi_seed_int < 2**64:
        raise ValueError("xi_seed must be in [0, 2**64)")

    observed = np.isfinite(y) & (y >= 0)
    if np.any(observed):
        observed_y = y[observed]
        if np.any(observed_y != np.floor(observed_y)) or observed_y.max() >= n_cat_int:
            raise ValueError(
                "responses must be integer categories in 0..n_cat-1 where observed"
            )
    yy = np.where(observed, y, 0.0).astype(np.int64).reshape(-1)

    res = core.fit_gpcm(
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
    return GpcmFit(
        model=resolved_model,
        slope=np.asarray(res["slope"], dtype=np.float64).reshape(n_items, n_dims),
        step=np.asarray(res["step"], dtype=np.float64).reshape(n_items, n_cat_int - 1),
        theta=np.asarray(res["theta"], dtype=np.float64).reshape(n_persons, n_dims),
        n_cat=int(res["n_cat"]),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        termination_reason=str(res["termination_reason"]),
        final_loglik_change=float(res["final_loglik_change"]),
        n_parameters=int(res["n_parameters"]),
    )
