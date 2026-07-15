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

    if not isinstance(n_cat, (int, type(None))) or isinstance(n_cat, bool):
        raise ValueError("n_cat must be an integer >= 2")
    if n_cat is not None and n_cat < 2:
        raise ValueError("n_cat must be an integer >= 2")
    if q_theta not in {7, 11, 15, 21, 31, 41}:
        raise ValueError("q_theta must be one of 7, 11, 15, 21, 31, 41")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError("max_iter must be an integer >= 1")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and > 0")

    y = np.asarray(responses, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    if n_persons < 1 or n_items < 1:
        raise ValueError("responses must contain at least one person and one item")
    missing = np.isnan(y)
    if np.any(~missing & ~np.isfinite(y)):
        raise ValueError("observed responses must be finite integer categories")
    observed = ~missing
    obs_values = y[observed]
    if obs_values.size and (
        np.any(obs_values != np.floor(obs_values)) or np.any(obs_values < 0)
    ):
        raise ValueError("observed responses must be non-negative integer categories")
    if n_cat is None:
        if obs_values.size == 0:
            raise ValueError("responses has no observed values")
        n_cat = int(obs_values.max()) + 1
        if n_cat < 2:
            raise ValueError("responses must contain at least two categories")
    if obs_values.size and np.any(obs_values >= n_cat):
        raise ValueError(
            f"observed responses must be integer categories in 0..{n_cat - 1}"
        )
    missing_items = np.flatnonzero(~observed.any(axis=0))
    if missing_items.size:
        raise ValueError(f"item {int(missing_items[0])} has no observed responses")
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
