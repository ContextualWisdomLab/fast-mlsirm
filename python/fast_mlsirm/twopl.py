"""Dimension-agnostic compensatory 2PL item response model.

Reckase (2009) / Bock, Gibbons & Muraki (1988) full-information item factor model, in
which an item may load freely on several latent dimensions that trade off additively in the
logit. Factors are orthogonal by default (``estimate_corr=False``, ``Sigma = I``) or their
correlation matrix is estimated (``estimate_corr=True``). Estimated in the Rust core over a
product Gauss-Hermite grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MAX_MAX_ITER, MAX_XI_POINTS
from .models import ConfirmatoryModel, ExploratoryModel, IrtModel, _resolve_model


_SUPPORTED_Q = (7, 11, 15, 21, 31, 41)
_MAX_DIMS = 3


@dataclass
class TwoPlFit:
    """Fitted compensatory 2PL item response model (Reckase, 2009).

    ``loading`` is the items x dimensions matrix of free loadings ``a_id`` (exactly ``0``
    where the confirmatory model loading pattern is ``0``); ``intercept`` the per-item ``b_i``; ``theta``
    the persons x dimensions trait EAP; ``corr`` the ``n_dims x n_dims`` latent correlation
    matrix (identity when ``estimate_corr=False``, estimated off-diagonals otherwise). The
    model is ``P(X_ij=1 | theta_j) = sigmoid(sum_d a_id theta_jd + b_i)`` with
    ``theta_j ~ MVN(0, Sigma)``, ``Sigma`` a unit-diagonal correlation matrix.
    ``termination_reason`` is either ``"converged"`` or ``"max_iter_reached"``;
    ``final_loglik_change`` is the absolute difference between the final two evaluated
    marginal log-likelihoods."""

    model: IrtModel
    loading: np.ndarray
    intercept: np.ndarray
    theta: np.ndarray
    corr: np.ndarray
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    n_parameters: int
    termination_reason: str = "unknown"
    final_loglik_change: float = np.nan

    @property
    def n_dims(self) -> int:
        """Latent dimension count derived from :attr:`model`."""

        return self.model.n_dims


def fit_2pl(
    responses: np.ndarray,
    model: int | ExploratoryModel | ConfirmatoryModel = 1,
    q: int = 21,
    estimate_corr: bool = False,
    max_iter: int = 500,
    tol: float = 1e-6,
    node_rule: str = "gh",
    xi_points: int = 4000,
    xi_seed: int = 0x9E37_79B9_7F4A_7C15,
) -> TwoPlFit:
    """Fit the compensatory 2PL item response model (compute in Rust; Reckase, 2009;
    Bock, Gibbons & Muraki, 1988).

    A general COMPENSATORY multidimensional 2PL: an item may load freely on several latent
    dimensions, which trade off ADDITIVELY inside a single logit,
    ``P(X_ij=1 | theta_j) = sigmoid(sum_{d in S_i} a_id theta_jd + b_i)`` with
    ``theta_j ~ MVN(0, I_D)``. ``S_i`` is item ``i``'s loading set from the confirmatory model specification;
    ``a_id`` is a free loading for ``d in S_i``
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
    ``Sigma`` (unit diagonal) is estimated by an ECM step (the standard grid is mapped
    through ``chol(Sigma)`` and the correlations ascend the Gaussian-prior objective with a
    positive-definite, monotone guard).

    **Integration nodes (``node_rule``).** ``"gh"`` (default) uses the exact ``q**n_dims``
    Gauss-Hermite product grid and caps ``n_dims <= 3``. For ``n_dims = 4, 5, 6`` use
    ``"qmc"`` (Halton quasi-Monte-Carlo, Jank 2005) or ``"mc"`` (plain Monte-Carlo): the E-step
    integral is evaluated at ``xi_points`` points drawn from the prior (equal weights) instead
    of the product grid, leaving the item and ``Sigma`` M-steps unchanged. QMC carries an
    ``O(N**-1 (log N)**D)`` finite-node bias that grows with the dimension, so ``n_dims = 5, 6``
    need materially larger ``xi_points``; ``xi_seed`` (nonzero by default) applies a
    Cranley-Patterson random shift that de-correlates the higher Halton axes. ``q`` is used only
    by ``"gh"``; ``xi_points``/``xi_seed`` only by ``"qmc"``/``"mc"``.

    ``responses`` is a persons x items 0/1 array (``NaN`` = missing, dropped under MAR);
    For ``model=1``, all item loadings on the single factor are free. A
    multidimensional confirmatory structure is supplied with
    ``model=models.confirmatory(loading_pattern)``; a numeric exploratory model greater than
    one is rejected until unrestricted loading rotation and identification are implemented.
    ``q`` is the Gauss-Hermite node count per dimension (one of ``7, 11, 15, 21, 31, 41``). Convergence requires the absolute
    change between consecutive evaluated marginal log-likelihoods to be less than ``tol``;
    the returned fit exposes that value as ``final_loglik_change`` and the terminal state as
    ``termination_reason``.

    References (APA 7th ed.):
        Reckase, M. D. (2009). *Multidimensional item response theory*. Springer.
            https://doi.org/10.1007/978-0-387-89976-3
        Bock, R. D., Gibbons, R., & Muraki, E. (1988). Full-information item factor
            analysis. *Applied Psychological Measurement, 12*(3), 261-280.
            https://doi.org/10.1177/014662168801200305
        Jank, W. (2005). Quasi-Monte Carlo sampling to improve the efficiency of Monte
            Carlo EM. *Computational Statistics & Data Analysis, 48*(4), 685-701.
            https://doi.org/10.1016/j.csda.2004.03.019
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_2pl"):
        raise RuntimeError("fit_2pl requires the compiled Rust core")

    y = np.asarray(responses, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    resolved_model, pat = _resolve_model(model, n_items)
    n_dims = pat.shape[1]
    # The Gauss-Hermite product grid caps D <= _MAX_DIMS; the QMC/MC rules reach D <= 6 (the Halton
    # prime axes). The core does the authoritative rule-dependent check; this mirrors it up front.
    _gh = str(node_rule).lower() in ("gh", "gauss-hermite", "gausshermite")
    _max_dims = _MAX_DIMS if _gh else 6
    if not 1 <= n_dims <= _max_dims:
        raise ValueError(
            f"loading_pattern dimensions must be between 1 and {_max_dims} "
            f"(node_rule={node_rule!r})"
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
    # q is used only by the Gauss-Hermite rule; the QMC/MC rules ignore it (matching the core).
    if _gh and q_int not in _SUPPORTED_Q:
        raise ValueError(f"q must be one of {_SUPPORTED_Q}")
    xi_points_int = _finite_integer(xi_points, "xi_points")
    if not (1 <= max_iter_int <= MAX_MAX_ITER):
        raise ValueError(f"max_iter must be in 1..{MAX_MAX_ITER}")
    if not _gh and not (1 <= xi_points_int <= MAX_XI_POINTS):
        raise ValueError(f"xi_points must be in 1..{MAX_XI_POINTS}")
    # xi_seed is a full-range u64 (default 0x9E37_79B9_7F4A_7C15): validate it as an EXACT integer
    # WITHOUT a float64 round-trip. _finite_integer casts through float(), which silently rounds any
    # value >= 2^53 (the default drifts, breaking Rust<->Python parity) and overflows u64 near the
    # top of the range (raising OverflowError in the PyO3 conversion).
    if isinstance(xi_seed, bool) or not isinstance(xi_seed, (int, np.integer)):
        raise ValueError("xi_seed must be a non-negative integer")
    xi_seed_int = int(xi_seed)
    if not 0 <= xi_seed_int < 2**64:
        raise ValueError("xi_seed must be in [0, 2**64)")

    observed = ~np.isnan(y)
    yy = np.where(observed, y, 0.0).reshape(-1)
    res = core.fit_2pl(
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
        str(node_rule),
        xi_points_int,
        xi_seed_int,
    )
    return TwoPlFit(
        model=resolved_model,
        loading=np.asarray(res["loading"], dtype=np.float64).reshape(n_items, n_dims),
        intercept=np.asarray(res["intercept"], dtype=np.float64),
        theta=np.asarray(res["theta"], dtype=np.float64).reshape(n_persons, n_dims),
        corr=np.asarray(res["corr"], dtype=np.float64).reshape(n_dims, n_dims),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        n_parameters=int(res["n_parameters"]),
        termination_reason=str(res["termination_reason"]),
        final_loglik_change=float(res["final_loglik_change"]),
    )
