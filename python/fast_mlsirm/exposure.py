"""Sympson-Hetter item-exposure control for computerized adaptive testing.
All numeric work happens in the Rust core (``mlsirm_core::exposure``); this
module only validates and marshals.

Source status: the algorithm was confirmed from secondary sources that were
READ in full — Georgiadou, Triantafillou, and Economides (2007), which
describes the Sympson-Hetter probabilistic gate and its iterative
calibration, and Barrada, Olea, and Ponsoda (2007), whose Equations 1-3 give
the exact update rule ``k_i <- min(1, r_max / P(S_i))`` implemented here.
The original conference papers (Sympson & Hetter, 1985; Hetter & Sympson,
1997) were NOT read; they are cited as the method's origin as attributed by
the read sources. Convergence is NOT guaranteed (van der Linden, 2003,
abstract); the stopping rule ``max P(A) <= r_max + tol`` is a practical
criterion, not a theorem.

REDUCED SCOPE (spec decision): dichotomous 3PL max-information CAT with an
interim EAP trait estimate only. No theta-stratified (conditional) variants,
no forced-administration fallback when the pool is exhausted (an error is
raised instead — a repository policy, not a classical prescription), and no
claim of a "classical" iteration count.

References (APA 7th ed.):
    Barrada, J. R., Olea, J., & Ponsoda, V. (2007). Methods for restricting
        maximum exposure rate in computerized adaptive testing.
        *Methodology, 3*(1), 14-23. https://doi.org/10.1027/1614-2241.3.1.14
    Georgiadou, E., Triantafillou, E., & Economides, A. A. (2007). A review
        of item exposure control strategies for computerized adaptive
        testing developed from 1983 to 2005. *The Journal of Technology,
        Learning, and Assessment, 5*(8).
    Hetter, R. D., & Sympson, J. B. (1997). Item exposure control in
        CAT-ASVAB. In W. A. Sands, B. K. Waters, & J. R. McBride (Eds.),
        *Computerized adaptive testing: From inquiry to operation*
        (pp. 141-144). American Psychological Association. (As cited in
        Georgiadou et al.; not read.)
    Sympson, J. B., & Hetter, R. D. (1985, October). Controlling
        item-exposure rates in computerized adaptive testing. *Proceedings
        of the 27th Annual Meeting of the Military Testing Association*
        (pp. 973-977). Navy Personnel Research and Development Center.
        (As cited in Georgiadou et al.; not read.)
    van der Linden, W. J. (2003). Some alternatives to Sympson-Hetter
        item-exposure control in computerized adaptive testing.
        *Journal of Educational and Behavioral Statistics, 28*(3),
        249-265. (Abstract only.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SympsonHetterResult:
    """Sympson-Hetter calibration output.

    ``k`` holds the exposure-control parameters ``k_i = P(A_i | S_i)`` in
    ``(0, 1]``; ``exposure``/``selection`` the administration and selection
    rates ``P(A_i)``/``P(S_i)`` from the final simulation cycle;
    ``max_exposure`` their maximum; ``history_max_exposure`` the per-cycle
    trace. The returned ``k`` is always the vector that produced the
    reported rates (the update is skipped after the final cycle)."""

    k: np.ndarray
    exposure: np.ndarray
    selection: np.ndarray
    max_exposure: float
    n_iter: int
    converged: bool
    history_max_exposure: np.ndarray


def sympson_hetter(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    *,
    r_max: float = 0.25,
    test_length: int = 20,
    n_simulees: int = 1000,
    max_iter: int = 20,
    tol: float = 0.02,
    seed: int = 20250724,
    q_theta: int = 41,
) -> SympsonHetterResult:
    """Calibrate Sympson-Hetter exposure-control parameters by simulation.

    ``a``, ``b``, ``c`` are 3PL item parameters (``c=None`` gives 2PL);
    ``r_max`` the target maximum exposure rate in ``(0, 1]`` (must satisfy
    ``r_max >= test_length / n_items``, a counting-identity feasibility
    bound derived in the Rust core); ``tol`` the Monte-Carlo tolerance on
    the stopping rule. ``r_max = 1`` reduces exactly to unconstrained
    max-information CAT (no exposure randomization is consumed).
    """
    from . import _core

    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if c is None:
        c = np.zeros_like(a)
    c = np.ascontiguousarray(c, dtype=np.float64)

    def _as_int(name: str, value, minimum: int = 0, maximum: int | None = None) -> int:
        if isinstance(value, bool) or not isinstance(
            value, (int, np.integer, float, np.floating)
        ):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        iv = int(value)
        if iv != value:
            raise ValueError(f"{name} must be an integer, got {value!r}")
        if iv < minimum or (maximum is not None and iv > maximum):
            raise ValueError(f"{name} out of range: {iv}")
        return iv

    r = _core.py_sympson_hetter(
        a,
        b,
        c,
        float(r_max),
        _as_int("test_length", test_length),
        _as_int("n_simulees", n_simulees),
        _as_int("max_iter", max_iter),
        float(tol),
        _as_int("seed", seed, maximum=2**64 - 1),
        _as_int("q_theta", q_theta),
    )
    return SympsonHetterResult(
        k=np.asarray(r["k"]),
        exposure=np.asarray(r["exposure"]),
        selection=np.asarray(r["selection"]),
        max_exposure=float(r["max_exposure"]),
        n_iter=int(r["n_iter"]),
        converged=bool(r["converged"]),
        history_max_exposure=np.asarray(r["history_max_exposure"]),
    )
