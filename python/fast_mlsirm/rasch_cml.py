"""Rasch conditional maximum likelihood (CML) estimation and Andersen's (1973) LR test.

Conditioning each response pattern on its raw score -- the sufficient statistic for ability -- removes
the person parameters, so the Rasch item difficulties are estimated without any assumption on the
ability distribution (specific objectivity) and consistently at fixed test length, unlike joint or
marginal ML. The numerical computation runs in Rust."""

from __future__ import annotations

import numpy as np


def _binary_matrix(responses: np.ndarray) -> tuple[np.ndarray, int, int]:
    y = np.asarray(responses)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    if n_items < 2:
        raise ValueError("need at least 2 items")
    yf = np.asarray(y, dtype=np.float64)
    if not np.all(np.isin(yf, (0.0, 1.0))):
        raise ValueError("responses must be complete 0/1 (Rasch CML has no missing-data path)")
    return yf.astype(np.int64).reshape(-1), n_persons, n_items


def fit_rasch_cml(
    responses: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Fit the dichotomous Rasch model by conditional maximum likelihood (compute in Rust; Andersen,
    1970, 1972).

    ``responses`` is a persons x items complete ``0/1`` array; persons scoring ``0`` or ``n_items`` (no
    conditional information) are dropped. Returns ``beta`` (the ``n_items`` item difficulties, centered
    to sum zero), ``se`` (standard errors from the pseudoinverse of the conditional information),
    ``loglik`` (conditional log-likelihood), ``n_iter``, ``converged``, and ``n_used`` (retained
    persons). The estimates are person-distribution-free: they do not depend on the shape of the ability
    distribution.

    References (APA 7th ed.):
        Andersen, E. B. (1970). Asymptotic properties of conditional maximum-likelihood estimators.
            *Journal of the Royal Statistical Society: Series B, 32*(2), 283-301.
        Andersen, E. B. (1972). The numerical solution of a set of conditional estimation equations.
            *Journal of the Royal Statistical Society: Series B, 34*(1), 42-54.
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_rasch_cml"):
        raise RuntimeError("fit_rasch_cml requires the compiled Rust core")
    yy, n_persons, n_items = _binary_matrix(responses)
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    res = core.fit_rasch_cml(yy, int(n_persons), int(n_items), int(max_iter), float(tol))
    return {
        "beta": np.asarray(res["beta"], dtype=np.float64),
        "se": np.asarray(res["se"], dtype=np.float64),
        "loglik": float(res["loglik"]),
        "n_iter": int(res["n_iter"]),
        "converged": bool(res["converged"]),
        "n_used": int(res["n_used"]),
    }


def andersen_lr_test(
    responses: np.ndarray,
    group: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> dict[str, float]:
    """Andersen's (1973) conditional likelihood-ratio test of Rasch fit (compute in Rust).

    Partitions the persons by ``group`` (integer labels ``0..n_groups``), fits CML within each group and
    over the pooled sample, and refers ``LR = 2[sum_g llc_g - llc_pooled]`` to
    ``chi2((n_groups - 1)(n_items - 1))``. A significant ``LR`` rejects the invariance of the item
    difficulties across the split (Rasch misfit); splitting on the raw-score median tests the model's
    core sufficiency assumption, and splitting on an external covariate tests for DIF. ``responses`` is a
    persons x items complete ``0/1`` array. Returns ``lr``, ``df``, ``p_value``, ``n_used`` (per-group
    retained counts), and ``converged`` (``False`` if the pooled or any group fit stalled, in which case
    the statistic is untrustworthy — do not read a clamped ``lr = 0`` as a clean non-rejection).

    Reference (APA 7th ed.):
        Andersen, E. B. (1973). A goodness of fit test for the Rasch model. *Psychometrika, 38*(1),
            123-140. https://doi.org/10.1007/BF02291180
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "andersen_lr_test"):
        raise RuntimeError("andersen_lr_test requires the compiled Rust core")
    yy, n_persons, n_items = _binary_matrix(responses)
    g = np.asarray(group)
    if g.ndim != 1 or g.shape[0] != n_persons:
        raise ValueError("group must be a length-n_persons 1-D array")
    gf = np.asarray(g, dtype=np.float64)
    if np.any(gf != np.floor(gf)) or np.any(gf < 0):
        raise ValueError("group labels must be non-negative integers")
    # densify labels so n_groups counts only populated groups
    _, gid = np.unique(gf.astype(np.int64), return_inverse=True)
    n_groups = int(gid.max()) + 1
    if n_groups < 2:
        raise ValueError("the Andersen LR test needs at least 2 groups")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    res = core.andersen_lr_test(
        yy, gid.astype(np.int64), int(n_groups), int(n_persons), int(n_items), int(max_iter), float(tol)
    )
    return {
        "lr": float(res["lr"]),
        "df": int(res["df"]),
        "p_value": float(res["p_value"]),
        "n_used": np.asarray(res["n_used"], dtype=np.int64),
        "converged": bool(res["converged"]),
    }
