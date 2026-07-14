"""Unidimensional polytomous item-response fitting (GRM / GPCM).

Thin orchestration over the Rust compute path (``mlsirm_core::poly``): all
numerical work — the Bock-Aitkin marginal-EM loop, the category cells, and the
Newton M-step — runs in Rust. This is the classic (no latent-space) polytomous
model; the latent-space polytomous LSIRM extension slots the same category cell
into the marginal (theta, xi) quadrature and is the next milestone (see
``docs/papers/gpcm-nominal-design-spec.md``).

``GRM`` (Samejima cumulative logit) is the default; ``GPCM`` (Muraki
adjacent-category) is available for partial-credit scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PolytomousFit", "fit_polytomous"]

VALID_POLY_MODELS = {"grm", "gpcm"}


@dataclass
class PolytomousFit:
    """Result of :func:`fit_polytomous`.

    ``slope`` is the per-item discrimination ``a_i``. ``cat_params`` is
    ``n_items x (n_cat - 1)``: GPCM additive category intercepts, or GRM
    cumulative thresholds ``beta_{i,k}`` (ordered decreasing). ``thresholds``
    is the GPCM Muraki step reparametrization ``b_{i,k} = c_{i,k-1} - c_{i,k}``
    (``None`` for GRM, whose ``cat_params`` are already thresholds).
    """

    model: str
    slope: np.ndarray
    cat_params: np.ndarray
    loglik: float
    n_iter: int
    thresholds: np.ndarray | None = None


def _core_module():
    try:
        from . import _core  # type: ignore

        return _core
    except Exception:  # pragma: no cover - core built in CI
        return None


def fit_polytomous(
    responses: np.ndarray,
    n_cat: int,
    model: str = "grm",
    q_theta: int = 21,
    max_iter: int = 80,
    tol: float = 1e-6,
) -> PolytomousFit:
    """Fit a unidimensional GRM or GPCM by marginal MLE (compute in Rust).

    ``responses`` is a persons x items array of integer categories
    ``0..n_cat-1`` (complete data). ``model`` is ``"grm"`` (default) or
    ``"gpcm"``. ``theta ~ N(0, 1)`` on a ``q_theta``-node Gauss-Hermite grid.
    """
    m = str(model).lower()
    if m not in VALID_POLY_MODELS:
        raise ValueError(f"model must be one of {sorted(VALID_POLY_MODELS)}")
    if not isinstance(n_cat, int) or n_cat < 2:
        raise ValueError("n_cat must be an integer >= 2")
    if q_theta not in {7, 11, 15, 21, 31, 41}:
        raise ValueError("q_theta must be one of 7, 11, 15, 21, 31, 41")

    y = np.asarray(responses)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    yf = y.astype(np.float64)
    if not np.all(np.isfinite(yf)) or np.any(yf != np.floor(yf)):
        raise ValueError("responses must be integer categories")
    if y.min() < 0 or y.max() >= n_cat:
        raise ValueError(f"responses must be in 0..{n_cat - 1}")

    core = _core_module()
    if core is None or not hasattr(core, "fit_poly_unidim"):
        raise RuntimeError("fit_polytomous requires the compiled Rust core")

    n_persons, n_items = y.shape
    res = core.fit_poly_unidim(
        y.reshape(-1).astype(np.int64),
        int(n_persons),
        int(n_items),
        int(n_cat),
        m,
        int(q_theta),
        int(max_iter),
        float(tol),
    )
    slope = np.asarray(res["slope"], dtype=np.float64)
    cat_params = np.asarray(res["cat_params"], dtype=np.float64)
    thresholds = None
    if m == "gpcm":
        # Muraki step difficulties from additive intercepts (baseline 0 prepended)
        c = np.concatenate([np.zeros((n_items, 1)), cat_params], axis=1)
        thresholds = c[:, :-1] - c[:, 1:]
    return PolytomousFit(
        model=m,
        slope=slope,
        cat_params=cat_params,
        loglik=float(res["loglik"]),
        n_iter=int(res["n_iter"]),
        thresholds=thresholds,
    )
