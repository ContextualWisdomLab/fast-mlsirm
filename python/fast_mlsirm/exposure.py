"""Item-exposure control designs for computerized adaptive testing:
Sympson-Hetter calibration and the a-stratified multistage design.
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

    _usize_max = int(np.iinfo(np.uintp).max)
    r = _core.py_sympson_hetter(
        a,
        b,
        c,
        float(r_max),
        _as_int("test_length", test_length, maximum=_usize_max),
        _as_int("n_simulees", n_simulees, maximum=_usize_max),
        _as_int("max_iter", max_iter, maximum=_usize_max),
        float(tol),
        _as_int("seed", seed, maximum=2**64 - 1),
        _as_int("q_theta", q_theta, maximum=_usize_max),
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

@dataclass
class AStratifiedResult:
    """a-stratified multistage CAT simulation output.

    ``exposure`` holds the administration rates ``P(A_i)``; ``stratum`` the
    0-based stratum index per item (ascending discrimination);
    ``stage_lengths`` the number of items administered per stage
    (``sum == test_length``); ``theta_rmse``/``theta_bias`` the final-EAP
    recovery against the simulated true thetas."""

    exposure: np.ndarray
    max_exposure: float
    stratum: np.ndarray
    stage_lengths: np.ndarray
    theta_rmse: float
    theta_bias: float


def a_stratified(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    *,
    n_strata: int = 4,
    test_length: int = 20,
    n_simulees: int = 1000,
    seed: int = 20250724,
    q_theta: int = 41,
) -> AStratifiedResult:
    """Simulate the a-stratified multistage CAT design (Chang & Ying, 1999).

    The pool is sorted ascending by discrimination and split into
    ``n_strata`` contiguous strata; stage ``k`` administers items only from
    stratum ``k``, choosing the item minimizing ``|b_i - theta_hat|``
    (b-matching — the selection rule as restated by Barrada, Mazuela, &
    Olea, 2006; the Chang & Ying full text was not read). The near-equal
    stratum/stage partitions (first strata one larger), the interim EAP
    estimator, and the initial ``theta_hat = 0`` are repository
    implementation choices, not claims from the paper. b-blocking (Chang,
    Qian, & Ying, 2001) is out of scope.

    References (APA 7th ed.):
        Barrada, J. R., Mazuela, P., & Olea, J. (2006). Maximum information
            stratification method for controlling item exposure in
            computerized adaptive testing. *Psicothema, 18*(1), 156-159.
            (Read in full.)
        Chang, H.-H., & Ying, Z. (1999). a-Stratified multistage
            computerized adaptive testing. *Applied Psychological
            Measurement, 23*(3), 211-222.
            https://doi.org/10.1177/01466219922031338 (Abstract only.)
        Chang, H.-H., Qian, J., & Ying, Z. (2001). a-Stratified multistage
            computerized adaptive testing with b blocking. *Applied
            Psychological Measurement, 25*(4), 333-341. (Not read; cited
            only as the deferred b-blocking extension.)
    """
    from . import _core

    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if c is None:
        c = np.zeros_like(a)
    c = np.ascontiguousarray(c, dtype=np.float64)

    _usize_max = int(np.iinfo(np.uintp).max)
    r = _core.py_a_stratified(
        a,
        b,
        c,
        _as_int("n_strata", n_strata, maximum=_usize_max),
        _as_int("test_length", test_length, maximum=_usize_max),
        _as_int("n_simulees", n_simulees, maximum=_usize_max),
        _as_int("seed", seed, maximum=2**64 - 1),
        _as_int("q_theta", q_theta, maximum=_usize_max),
    )
    return AStratifiedResult(
        exposure=np.asarray(r["exposure"]),
        max_exposure=float(r["max_exposure"]),
        stratum=np.asarray(r["stratum"], dtype=np.intp),
        stage_lengths=np.asarray(r["stage_lengths"], dtype=np.intp),
        theta_rmse=float(r["theta_rmse"]),
        theta_bias=float(r["theta_bias"]),
    )

def kl_information(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    *,
    theta0: float,
    delta: float,
) -> np.ndarray:
    """Chang-Ying (1996) Kullback-Leibler information index per item.

    Returns the UNNORMALIZED area of the pointwise Bernoulli KL divergence
    ``K_i(theta || theta0) = P_i(theta0) ln[P_i(theta0)/P_i(theta)] +
    Q_i(theta0) ln[Q_i(theta0)/Q_i(theta)]`` over
    ``[theta0 - delta, theta0 + delta]`` for each 3PL item (``c = 0`` gives
    2PL) — Chang and Ying's Equation 17 interval index in area form (for a
    common ``delta`` the argmax is identical to the interval average). As
    ``delta -> 0`` the area approaches ``I_i(theta0) * delta**3 / 3`` with
    ``I_i`` the Fisher information, the paper's connection between global and
    local information (verified independently by the adversarial spec
    review; anchored by a crate test). All numeric work happens in the Rust
    core (``mlsirm_core::exposure::kl_information``); this wrapper only
    validates and marshals. Inputs must be 1-D.

    Source status: the pointwise Bernoulli form and expectation-under-theta0
    direction were confirmed against Chang and Ying (1996, Definitions
    2.1-2.2, Eq. 17-18) and the catR implementation (Magis & Raiche, 2012,
    ``KL.R``).

    References (APA 7th ed.):
        Chang, H.-H., & Ying, Z. (1996). A global information approach to
            computerized adaptive testing. *Applied Psychological
            Measurement, 20*(3), 213-229.
            https://doi.org/10.1177/014662169602000303
        Magis, D., & Raiche, G. (2012). Random generation of response
            patterns under computerized adaptive testing with the R package
            catR. *Journal of Statistical Software, 48*(8), 1-31.
            https://doi.org/10.18637/jss.v048.i08
    """
    from . import _core

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a and b must be 1-D arrays")
    if c is None:
        c = np.zeros_like(a)
    c = np.asarray(c, dtype=np.float64)
    if c.ndim != 1:
        raise ValueError("c must be a 1-D array")
    return np.asarray(
        _core.py_kl_information(
            np.ascontiguousarray(a),
            np.ascontiguousarray(b),
            np.ascontiguousarray(c),
            float(theta0),
            float(delta),
        )
    )


def kl_select(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    *,
    administered: np.ndarray,
    theta0: float,
    n_administered: int,
    r: float = 3.0,
) -> dict:
    """Select the next CAT item by the Chang-Ying (1996) KL criterion.

    Computes ``argmax_i KL_i(theta0)`` over items where ``administered`` is
    False, with half-width ``delta = r / sqrt(n_administered)`` (the paper's
    Equation 18 shrinking-interval rule; their Study 1 uses ``r = 3``).
    Requires ``n_administered >= 1`` — the rule is undefined at ``n = 0``;
    for the first item call :func:`kl_information` with an explicit
    ``delta``. Returns ``{"index", "selected", "delta"}`` where ``index`` is
    the full per-item KL vector (administered items keep their value; masking
    applies to selection only). All numeric work happens in the Rust core
    (``mlsirm_core::exposure::kl_select``). See :func:`kl_information` for
    sources and references.
    """
    from . import _core

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a and b must be 1-D arrays")
    if c is None:
        c = np.zeros_like(a)
    c = np.asarray(c, dtype=np.float64)
    mask = np.asarray(administered, dtype=np.bool_)
    if c.ndim != 1 or mask.ndim != 1:
        raise ValueError("c and administered must be 1-D arrays")
    res = _core.py_kl_select(
        np.ascontiguousarray(a),
        np.ascontiguousarray(b),
        np.ascontiguousarray(c),
        np.ascontiguousarray(mask),
        float(theta0),
        _as_int("n_administered", n_administered, minimum=1),
        float(r),
    )
    return {
        "index": np.asarray(res["index"]),
        "selected": int(res["selected"]),
        "delta": float(res["delta"]),
    }
