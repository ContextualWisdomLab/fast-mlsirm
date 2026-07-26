"""Paired-comparison scaling: Thurstone (1927) Case V and Bradley-Terry.

Thin wrappers over the Rust core (``mlsirm_core::scaling``). The Thurstone
algorithm follows the ``thurstone()`` function of the psych R package
(Revelle, 2025; source READ), which implements Thurstone's (1927) law of
comparative judgment under Case V (equal discriminal dispersions, zero
correlations). The Bradley-Terry maximum-likelihood fit uses the
minorization-maximization (MM) algorithm as implemented by choix 0.4.1
(Maystre, 2015-2020; ``opt.mm`` pairwise path, source READ), which choix
attributes to Hunter (2004). Thurstone (1927), Bradley and Terry (1952), and
Hunter (2004) themselves were NOT read; they are cited as the origins of the
models/algorithm as described by the read sources.

References
----------
Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block
    designs: I. The method of paired comparisons. Biometrika, 39(3/4),
    324-345. [NOT READ; cited as described in choix source]
Hunter, D. R. (2004). MM algorithms for generalized Bradley-Terry models.
    The Annals of Statistics, 32(1), 384-406. [NOT READ; cited as described
    in choix source]
Maystre, L. (2020). choix: Inference algorithms for models based on Luce's
    choice axiom (Python package, version 0.4.1).
    https://github.com/lucasmaystre/choix [READ]
Revelle, W. (2025). psych: Procedures for psychological, psychometric, and
    personality research (R package). https://CRAN.R-project.org/package=psych
Thurstone, L. L. (1927). A law of comparative judgment. Psychological
    Review, 34(4), 273-286. [NOT READ; cited as described in psych source]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ThurstoneResult:
    """Case V scaling result: ``scale[j]`` is the scale value of object j
    (minimum shifted to exactly 0), ``gof`` the psych goodness of fit
    ``1 - sse/ssc`` over the FULL model matrix (including the diagonal --
    this pins the psych *code* behavior; the .Rd prose saying "lower off
    diagonal" is stale), ``model`` the fitted choice probabilities
    ``Phi(scale[j] - scale[i])`` and ``residual = model - choice``, both
    shaped (n, n)."""

    scale: np.ndarray
    gof: float
    model: np.ndarray
    residual: np.ndarray


def thurstone_case_v(choice) -> ThurstoneResult:
    """Scale n objects from an n x n choice-probability matrix.

    ``choice[i, j]`` is the proportion of judges preferring object *j* over
    object *i* (psych convention: column beats row). All entries must be
    strictly in (0, 1) -- a deliberate safety divergence from psych, whose
    direct path lets ``qnorm(0)/qnorm(1)`` produce infinities.
    """
    from .fitstats import _core_module

    arr = np.asarray(choice)
    if np.iscomplexobj(arr):
        raise ValueError("thurstone_case_v: choice must be real-valued")
    if arr.dtype == object:
        try:
            arr = arr.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "thurstone_case_v: choice must be numeric"
            ) from exc
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("thurstone_case_v: choice must be a square 2-D matrix")
    n = arr.shape[0]
    core = _core_module()
    res = core.thurstone_case_v(arr.ravel(), n)
    return ThurstoneResult(
        scale=np.asarray(res["scale"], dtype=np.float64),
        gof=float(res["gof"]),
        model=np.asarray(res["model"], dtype=np.float64).reshape(n, n),
        residual=np.asarray(res["residual"], dtype=np.float64).reshape(n, n),
    )


@dataclass
class BradleyTerryResult:
    """Bradley-Terry MM fit: ``params[i]`` is the centered log-worth of item
    i (mean exactly 0), ``weights[i] = n * softmax(params)[i]`` the exp-scale
    worths (sum exactly n, choix convention), ``iterations`` the number of MM
    updates performed when the L1 convergence criterion
    ``sum |new - prev| <= tol * n`` first fired (the check compares two
    consecutive *updates*, so at least two are always performed)."""

    params: np.ndarray
    weights: np.ndarray
    iterations: int


def bradley_terry_mm(wins, alpha=0.0, max_iter=10000, tol=1e-8):
    """Fit Bradley-Terry worths from an n x n win-count matrix.

    ``wins[i, j]`` is the number of times item *i* beat item *j* (row beats
    column). Counts must be finite and non-negative with a zero diagonal;
    non-integer counts are accepted (weighted comparisons -- a DERIVED
    extension of choix's pair-list interface). ``alpha`` is choix's
    regularization added to both numerator and denominator of each MM update;
    with ``alpha = 0`` an item with zero wins has no finite log-worth and a
    ValueError is raised. An all-zero matrix is rejected for every alpha
    (deliberate divergence from choix, which returns the uniform solution
    when alpha > 0). Non-convergence within ``max_iter`` raises (e.g. when an
    item never loses, violating Ford's (1957) strong-connectivity condition
    as cited by choix; Ford 1957 NOT READ).
    """
    from .fitstats import _core_module

    arr = np.asarray(wins)
    if np.iscomplexobj(arr):
        raise ValueError("bradley_terry_mm: wins must be real-valued")
    if arr.dtype == object:
        try:
            arr = arr.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("bradley_terry_mm: wins must be numeric") from exc
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("bradley_terry_mm: wins must be a square 2-D matrix")
    n = arr.shape[0]
    core = _core_module()
    res = core.bradley_terry_mm(
        arr.ravel(), n, float(alpha), int(max_iter), float(tol)
    )
    return BradleyTerryResult(
        params=np.asarray(res["params"], dtype=np.float64),
        weights=np.asarray(res["weights"], dtype=np.float64),
        iterations=int(res["iterations"]),
    )
