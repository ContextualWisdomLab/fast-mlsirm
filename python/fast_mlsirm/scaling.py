"""Thurstone (1927) Case V paired-comparison scaling.

Thin wrapper over the Rust core (``mlsirm_core::scaling``). The algorithm
follows the ``thurstone()`` function of the psych R package (Revelle, 2025;
source READ), which implements Thurstone's (1927) law of comparative judgment
under Case V (equal discriminal dispersions, zero correlations). Thurstone
(1927) itself was NOT read; it is cited as the origin of the model as
described by the psych source.

References
----------
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
