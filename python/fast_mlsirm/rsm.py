"""Rating Scale Model (Andrich, 1978): a Rasch-family polytomous model whose
category thresholds are shared across items, estimated by marginal-ML EM in the
Rust core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RsmFit:
    """Fitted rating scale model (Andrich, 1978).

    ``item_location`` is the per-item location ``delta_i``; ``thresholds`` the
    ``n_cat-1`` common category thresholds ``tau_k`` (shared across all items,
    centered so they sum to 0); ``theta`` the per-person EAP trait. The
    adjacent-category log-odds are ``ln[P(k)/P(k-1)] = theta - delta_i - tau_k``."""

    item_location: np.ndarray
    thresholds: np.ndarray
    theta: np.ndarray
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    n_parameters: int


def fit_rsm(
    responses: np.ndarray,
    n_cat: int | None = None,
    q_theta: int = 41,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> RsmFit:
    """Fit the rating scale model (compute in Rust; Andrich, 1978).

    The RSM is the Rasch-family polytomous model for items on a common rating scale
    (e.g. Likert): every item has its own location ``delta_i``, but the ``K-1``
    category thresholds ``tau_k`` are *shared across all items*. The
    adjacent-category log-odds are ``ln[P(X=k)/P(X=k-1)] = theta - delta_i - tau_k``,
    ``theta ~ N(0,1)``. This is a constrained partial-credit model (the PCM has
    item-specific thresholds); at ``K=2`` it reduces to the Rasch model. Estimated by
    marginal-ML EM with a Gauss-Hermite trait grid; the item locations and the shared
    thresholds are updated by a monotone ECM step and the thresholds are centered to
    sum to zero.

    ``responses`` is a persons x items array of integer category indices
    ``0..n_cat-1`` (``NaN`` marks a missing cell, dropped under a missing-at-random
    assumption). ``n_cat`` defaults to ``max(responses) + 1``.

    References (APA 7th ed.):
        Andrich, D. (1978). A rating formulation for ordered response categories.
            *Psychometrika, 43*(4), 561-573. https://doi.org/10.1007/BF02293814
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_rsm"):
        raise RuntimeError("fit_rsm requires the compiled Rust core")

    y = np.asarray(responses, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    observed = np.isfinite(y)
    if n_cat is None:
        if not observed.any():
            raise ValueError("responses has no observed values")
        n_cat = int(np.nanmax(y)) + 1
    yy = np.where(observed, y, 0.0).astype(np.int64).reshape(-1)
    res = core.fit_rsm(
        yy,
        observed.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_cat),
        int(q_theta),
        int(max_iter),
        float(tol),
    )
    return RsmFit(
        item_location=np.asarray(res["item_location"], dtype=np.float64),
        thresholds=np.asarray(res["thresholds"], dtype=np.float64),
        theta=np.asarray(res["theta"], dtype=np.float64),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        n_parameters=int(res["n_parameters"]),
    )
