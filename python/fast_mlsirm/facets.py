"""Many-Facet Rasch Model (Linacre, 1989): the rating-scale Rasch model with a
rater-severity facet, estimated by marginal-ML EM in the Rust core. All numeric
work happens in Rust; this module only validates and marshals arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MAX_MAX_ITER, MAX_POLYTOMOUS_CATEGORIES


@dataclass
class FacetsFit:
    """Fitted many-facet Rasch model (Linacre, 1989).

    ``item_difficulty`` is the per-item ``d_i``; ``rater_severity`` the per-rater
    ``c_j`` (centered to sum 0; higher = harsher); ``thresholds`` the ``n_cat-1``
    common category thresholds (centered to sum 0); ``theta`` the per-person EAP
    trait. The adjacent-category log-odds are
    ``ln[P(k)/P(k-1)] = theta - d_i - c_j - f_k``. ``connected`` is False when the
    item-rater co-observation design splits into disconnected components — then
    severity/difficulty comparisons across components rest solely on the shared
    ``theta ~ N(0,1)`` assumption rather than on the rating design (Linacre's
    connectedness requirement)."""

    item_difficulty: np.ndarray
    rater_severity: np.ndarray
    thresholds: np.ndarray
    theta: np.ndarray
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    connected: bool
    n_parameters: int


def fit_facets(
    responses: np.ndarray,
    n_cat: int | None = None,
    q_theta: int = 41,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> FacetsFit:
    """Fit the many-facet Rasch model (compute in Rust; Linacre, 1989).

    The MFRM extends the rating scale model (Andrich, 1978) with a rater facet:
    the rating of person ``p`` on item ``i`` by rater ``j`` follows the
    adjacent-category log-odds
    ``ln[P(Y=k)/P(Y=k-1)] = theta_p - d_i - c_j - f_k``, where ``d_i`` is item
    difficulty, ``c_j`` rater severity, and ``f_k`` the category thresholds
    shared across items and raters. ``theta ~ N(0,1)`` fixes the scale;
    severities and thresholds are centered to sum to zero. Estimation is
    marginal-ML EM (Bock & Aitkin, 1981) on a Gauss-Hermite trait grid — not
    Linacre's JMLE, so estimates match Facets output only up to the JMLE-vs-MMLE
    difference.

    In LLM-as-a-Judge calibration, raters are judges: ``rater_severity``
    estimates each judge's harshness on a common logit scale, adjusted for item
    difficulty and respondent ability.

    ``responses`` is a ``persons x items x raters`` array of integer category
    indices ``0..n_cat-1``; ``NaN`` marks unscored cells (sparse judging plans),
    dropped under a missing-at-random assumption. ``n_cat`` defaults to
    ``max(responses) + 1``. Every item and every rater needs at least one
    observed rating.

    References (APA 7th ed.):
        Linacre, J. M. (1989). *Many-facet Rasch measurement*. MESA Press.
        Eckes, T. (2015). *Introduction to many-facet Rasch measurement*
            (2nd ed.). Peter Lang. https://doi.org/10.3726/978-3-653-04844-5
        Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation
            of item parameters: Application of an EM algorithm. *Psychometrika,
            46*(4), 443-459. https://doi.org/10.1007/BF02293801
        Andrich, D. (1978). A rating formulation for ordered response
            categories. *Psychometrika, 43*(4), 561-573.
            https://doi.org/10.1007/BF02293814
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_facets"):
        raise RuntimeError("fit_facets requires the compiled Rust core")

    if not isinstance(n_cat, (int, type(None))) or isinstance(n_cat, bool):
        raise ValueError("n_cat must be an integer >= 2")
    if n_cat is not None and not (2 <= n_cat <= MAX_POLYTOMOUS_CATEGORIES):
        raise ValueError(f"n_cat must be an integer in 2..{MAX_POLYTOMOUS_CATEGORIES}")
    if q_theta not in {7, 11, 15, 21, 31, 41}:
        raise ValueError("q_theta must be one of 7, 11, 15, 21, 31, 41")
    if (
        not isinstance(max_iter, int)
        or isinstance(max_iter, bool)
        or not (1 <= max_iter <= MAX_MAX_ITER)
    ):
        raise ValueError(f"max_iter must be an integer in 1..{MAX_MAX_ITER}")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and > 0")

    y = np.asarray(responses, dtype=np.float64)
    if y.ndim != 3:
        raise ValueError("responses must be a 3-D persons x items x raters array")
    n_persons, n_items, n_raters = y.shape
    if n_persons < 1 or n_items < 1 or n_raters < 1:
        raise ValueError(
            "responses must contain at least one person, one item and one rater"
        )
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
        if n_cat > MAX_POLYTOMOUS_CATEGORIES:
            raise ValueError(
                f"responses imply more than {MAX_POLYTOMOUS_CATEGORIES} categories"
            )
    if obs_values.size and np.any(obs_values >= n_cat):
        raise ValueError(
            f"observed responses must be integer categories in 0..{n_cat - 1}"
        )
    missing_items = np.flatnonzero(~observed.any(axis=(0, 2)))
    if missing_items.size:
        raise ValueError(f"item {int(missing_items[0])} has no observed responses")
    missing_raters = np.flatnonzero(~observed.any(axis=(0, 1)))
    if missing_raters.size:
        raise ValueError(f"rater {int(missing_raters[0])} has no observed responses")
    yy = np.where(observed, y, 0.0).astype(np.int64).reshape(-1)
    res = core.fit_facets(
        yy,
        observed.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_raters),
        int(n_cat),
        int(q_theta),
        int(max_iter),
        float(tol),
    )
    return FacetsFit(
        item_difficulty=np.asarray(res["item_difficulty"], dtype=np.float64),
        rater_severity=np.asarray(res["rater_severity"], dtype=np.float64),
        thresholds=np.asarray(res["thresholds"], dtype=np.float64),
        theta=np.asarray(res["theta"], dtype=np.float64),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        connected=bool(res["connected"]),
        n_parameters=int(res["n_parameters"]),
    )
