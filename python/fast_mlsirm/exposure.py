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


def owen_update(
    a: float,
    b: float,
    c: float = 0.0,
    *,
    correct: bool,
    mu: float,
    sig2: float,
) -> tuple[float, float]:
    """Owen (1975) approximate Bayesian posterior update for one 3PNO item.

    Given a normal prior ``theta ~ N(mu, sig2)`` and a response to a
    three-parameter normal-ogive item ``P(X=1|theta) = c +
    (1-c) Phi(a (theta - b))``, returns the updated normal-approximation
    posterior moments ``(mu', sig2')`` from Owen's closed-form truncated-
    normal moment matching. All numeric work happens in the Rust core
    (``mlsirm_core::exposure::owen_update``); this wrapper only validates
    and marshals.

    Source status: Owen (1975) itself was NOT read (paywalled). The update
    formulas are implemented as reproduced by van der Linden (1998,
    Appendix Eqs. A.1-A.6) and cross-checked against the R ``irt`` package
    ``src/est_ability_owen.cpp``; the adversarial spec review additionally
    verified three pinned oracle cases against high-precision numerical
    integration of the exact posterior (~1e-13 agreement).

    References (APA 7th ed.):
        Owen, R. J. (1975). A Bayesian sequential procedure for quantal
            response in the context of adaptive mental testing. *Journal of
            the American Statistical Association, 70*(350), 351-356.
            https://doi.org/10.1080/01621459.1975.10479871
        van der Linden, W. J. (1998). *Bayesian item selection criteria for
            adaptive testing* (Research Report 98-01). University of Twente.
        Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of
            ability in a microcomputer environment. *Applied Psychological
            Measurement, 6*(4), 431-444.
            https://doi.org/10.1177/014662168200600405
    """
    from . import _core

    return _core.py_owen_update(
        float(a), float(b), float(c), bool(correct), float(mu), float(sig2)
    )


def owen_cat(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    *,
    responses: np.ndarray,
    mu0: float = 0.0,
    sig2_0: float = 1.0,
    test_length: int,
    sig2_stop: float | None = None,
) -> dict:
    """Owen (1975) approximate Bayesian sequential CAT.

    Selects items by Owen's b-matching rule ``argmin_i |b_i - mu|`` over the
    unadministered pool (ties to the lowest index), applies the
    :func:`owen_update` posterior step after each response, and stops when
    the posterior variance drops to ``sig2_stop`` (Owen's stopping rule) or
    after ``test_length`` items. ``responses`` is a full-pool 0/1 vector
    consulted for whichever item is selected — a fixed-person simulation
    contract, not item-by-item elicitation. Returns a dict with
    ``administered``, ``mu_trace``, ``sig2_trace``, ``mu`` and ``sig2``, all
    computed by the Rust core (``mlsirm_core::exposure::owen_cat``).

    Source status: see :func:`owen_update` (Owen 1975 NOT read; formulas per
    van der Linden 1998 with R ``irt`` cross-check). The b-matching
    selection and variance-threshold stopping rules are Owen's per the same
    secondary sources; ``test_length`` is an implementation cap.

    References (APA 7th ed.):
        Owen, R. J. (1975). A Bayesian sequential procedure for quantal
            response in the context of adaptive mental testing. *Journal of
            the American Statistical Association, 70*(350), 351-356.
            https://doi.org/10.1080/01621459.1975.10479871
        van der Linden, W. J. (1998). *Bayesian item selection criteria for
            adaptive testing* (Research Report 98-01). University of Twente.
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
    responses = np.asarray(responses)
    if responses.ndim != 1:
        raise ValueError("responses must be a 1-D array")
    # Validate BEFORE the uint8 cast: astype(np.uint8) silently truncates
    # (1.2 -> 1, 0.9 -> 0) and wraps negatives (-1 -> 255), and complex
    # inputs would silently drop the imaginary part, laundering invalid
    # inputs past the Rust-side 0/1 check.
    if np.iscomplexobj(responses):
        raise ValueError("responses must contain only 0 or 1")
    if not np.isin(responses.astype(np.float64), (0.0, 1.0)).all():
        raise ValueError("responses must contain only 0 or 1")
    r = _core.py_owen_cat(
        np.ascontiguousarray(a),
        np.ascontiguousarray(b),
        np.ascontiguousarray(c),
        np.ascontiguousarray(responses, dtype=np.uint8),
        float(mu0),
        float(sig2_0),
        _as_int("test_length", test_length, minimum=1),
        None if sig2_stop is None else float(sig2_stop),
    )
    return {
        "administered": list(r["administered"]),
        "mu_trace": np.asarray(r["mu_trace"]),
        "sig2_trace": np.asarray(r["sig2_trace"]),
        "mu": float(r["mu"]),
        "sig2": float(r["sig2"]),
    }


def ccat_select(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    *,
    groups: np.ndarray,
    targets: np.ndarray,
    administered: np.ndarray,
    theta0: float,
) -> dict:
    """Kingsbury & Zara (1989) constrained CAT (CCAT) content balancing.

    Single-step item selection under content-area constraints: any eligible
    content group (one with at least one unadministered item) that has zero
    administered items has priority; otherwise the eligible group with the
    maximal discrepancy ``targets[g] - k_g / k`` (target minus empirical
    proportion of administered items) is chosen; within the chosen group the
    unadministered item with maximal logistic 3PL Fisher information
    ``a^2 (Q/P) ((P - c) / (1 - c))^2`` at ``theta0`` is selected. Ties go
    to the lowest index (documented deterministic deviation from catR's
    random tie-break). Returns a dict with ``selected``, ``group``,
    ``discrepancy`` (per group) and ``info`` (per item; computed for the
    whole pool, masking applies to selection only), all computed by the
    Rust core (``mlsirm_core::exposure::ccat_select``).

    Source status: Kingsbury & Zara (1989) itself was NOT read (paywalled).
    The rule is implemented as reproduced by the R catR package
    (``nextItem.R``, ``cbControl`` branch; READ), and the Fisher-information
    formula was verified against catR ``Ii.R``/``Pi.R``.

    ``targets`` must be strictly positive and sum to 1; ``groups`` maps each
    item to a group index ``0..len(targets)``.

    References (APA 7th ed.):
        Kingsbury, G. G., & Zara, A. R. (1989). Procedures for selecting
            items for computerized adaptive tests. *Applied Measurement in
            Education, 2*(4), 359-375.
            https://doi.org/10.1207/s15324818ame0204_6
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
    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError("groups must be a 1-D array")
    # Validate BEFORE the uintp cast: casting would silently truncate
    # non-integers, wrap negatives, and drop imaginary parts.
    if np.iscomplexobj(groups):
        raise ValueError("groups must contain non-negative integers")
    gf = groups.astype(np.float64)
    if not np.isfinite(gf).all() or (gf < 0).any() or (gf != np.floor(gf)).any():
        raise ValueError("groups must contain non-negative integers")
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 1:
        raise ValueError("targets must be a 1-D array")
    administered = np.asarray(administered)
    if administered.ndim != 1:
        raise ValueError("administered must be a 1-D array")
    if administered.dtype != np.bool_:
        raise ValueError("administered must be a boolean array")
    r = _core.py_ccat_select(
        np.ascontiguousarray(a),
        np.ascontiguousarray(b),
        np.ascontiguousarray(c),
        np.ascontiguousarray(groups, dtype=np.uintp),
        np.ascontiguousarray(targets),
        np.ascontiguousarray(administered),
        float(theta0),
    )
    return {
        "selected": int(r["selected"]),
        "group": int(r["group"]),
        "discrepancy": np.asarray(r["discrepancy"]),
        "info": np.asarray(r["info"]),
    }


def epv_select(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    *,
    administered: np.ndarray,
    mu: float,
    sig2: float,
) -> dict:
    """Owen-approximate posterior-predictive EPV item selection.

    This is NOT van der Linden's (1998) exact minimum expected posterior
    variance (MEPV) criterion. The posterior over theta is Owen's (1975)
    normal approximation ``N(mu, sig2)``; the predictive success probability
    is ``p*_i = c_i + (1 - c_i) * Phi((mu - b_i) / sqrt(1/a_i^2 + sig2))``
    computed exactly as in ``owen_update``; and the two outcome posterior
    variances ``sig2_i^+``/``sig2_i^-`` come from ``owen_update`` rather
    than exact numerical posteriors. The unadministered item minimizing
    ``EPV_i = p*_i sig2_i^+ + (1 - p*_i) sig2_i^-`` is selected; ties go to
    the lowest index. ``epv`` and ``predictive`` are returned for the whole
    pool (masking applies to selection only), all computed by the Rust core
    (``mlsirm_core::exposure::epv_select``).

    Source status: van der Linden (1998) was read as the ERIC ED424235
    research report (Research Report 96-01); the exact-MEPV contract was
    additionally verified against R catR ``EPV.R`` and mirtCAT
    ``selection_criteria.R`` (both READ). Owen (1975) itself was NOT read;
    the update formulas follow the crate's ``owen_update``.

    References (APA 7th ed.):
        van der Linden, W. J. (1998). Bayesian item selection criteria for
            adaptive testing. *Psychometrika, 63*(2), 201-216.
            https://doi.org/10.1007/BF02294775
        Owen, R. J. (1975). A Bayesian sequential procedure for quantal
            response in the context of adaptive mental testing. *Journal of
            the American Statistical Association, 70*(350), 351-356.
            https://doi.org/10.1080/01621459.1975.10479871
        Magis, D., & Raiche, G. (2012). Random generation of response
            patterns under computerized adaptive testing with the R package
            catR. *Journal of Statistical Software, 48*(8), 1-31.
            https://doi.org/10.18637/jss.v048.i08
    """
    from . import _core

    # Reject complex input BEFORE the float64 casts: the casts would silently
    # discard imaginary parts (complex laundering).
    for name, arr in (("a", a), ("b", b), ("c", c)):
        if arr is not None and np.iscomplexobj(np.asarray(arr)):
            raise ValueError(f"{name} must be real-valued")
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a and b must be 1-D arrays")
    if c is None:
        c = np.zeros_like(a)
    c = np.asarray(c, dtype=np.float64)
    if c.ndim != 1:
        raise ValueError("c must be a 1-D array")
    administered = np.asarray(administered)
    if administered.ndim != 1:
        raise ValueError("administered must be a 1-D array")
    if administered.dtype != np.bool_:
        raise ValueError("administered must be a boolean array")
    r = _core.py_epv_select(
        np.ascontiguousarray(a),
        np.ascontiguousarray(b),
        np.ascontiguousarray(c),
        np.ascontiguousarray(administered),
        float(mu),
        float(sig2),
    )
    return {
        "selected": int(r["selected"]),
        "epv": np.asarray(r["epv"]),
        "predictive": np.asarray(r["predictive"]),
    }

def sprt_classify(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    *,
    responses: np.ndarray,
    theta_cut: float,
    delta: float,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> dict:
    """Single-cut binary-response Wald SPRT classification for CAT.

    Compares the point hypotheses ``theta0 = theta_cut - delta`` and
    ``theta1 = theta_cut + delta`` through the cumulative binary
    log-likelihood ratio under the D = 1 logistic 3PL
    ``P_i(theta) = c_i + (1 - c_i) / (1 + exp(-a_i (theta - b_i)))``
    against the log Wald boundaries ``A = ln((1 - beta) / alpha)`` and
    ``B = ln(beta / (1 - alpha))``. Responses are walked in order and the
    FIRST inclusive crossing decides: ``LLR_k >= A`` -> ``"above"``,
    ``LLR_k <= B`` -> ``"below"`` (``n_used = k``, 1-based); no crossing ->
    ``"continue"`` with ``n_used = len(responses)``. All numerics run in the
    Rust core (``mlsirm_core::exposure::sprt_classify``).

    ``llr_trace`` is returned for ALL supplied responses as an offline
    diagnostic; entries past ``n_used`` are counterfactual replay values (a
    live CAT would stop at ``n_used`` and never administer later items).
    Parameters calibrated on the D = 1.7 normal-ogive metric must be
    rescaled by the caller (``a_D1 = 1.7 * a_D17``) before use.

    Source status: the boundary and likelihood-ratio forms were verified
    against R catIrt ``termSPRT.R``/``logLik.brm.R``/``p.brm.R`` (READ) and
    Thompson (2007, p. 7; READ). Reckase (1983) and Eggen (1999) were NOT
    read and are cited as historical origins via Thompson.

    References (APA 7th ed.):
        Thompson, N. A. (2007). A practitioner's guide for variable-length
            computerized classification testing. *Practical Assessment,
            Research & Evaluation, 12*(1).
            https://doi.org/10.7275/fq3r-zz60
        Eggen, T. J. H. M. (1999). Item selection in adaptive testing with
            the sequential probability ratio test. *Applied Psychological
            Measurement, 23*(3), 249-261.
            https://doi.org/10.1177/01466219922031365
        Reckase, M. D. (1983). A procedure for decision making using
            tailored testing. In D. J. Weiss (Ed.), *New horizons in
            testing* (pp. 237-255). Academic Press.
        Wald, A. (1947). *Sequential analysis*. Wiley.
    """
    from . import _core

    # Reject complex input BEFORE the dtype casts: the casts would silently
    # discard imaginary parts (complex laundering).
    for name, arr in (("a", a), ("b", b), ("c", c), ("responses", responses)):
        if arr is not None and np.iscomplexobj(np.asarray(arr)):
            raise ValueError(f"{name} must be real-valued")
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a and b must be 1-D arrays")
    if c is None:
        c = np.zeros_like(a)
    c = np.asarray(c, dtype=np.float64)
    if c.ndim != 1:
        raise ValueError("c must be a 1-D array")
    # Validate responses BEFORE the uint8 cast (casts truncate/wrap).
    resp = np.asarray(responses)
    if resp.ndim != 1:
        raise ValueError("responses must be a 1-D array")
    if resp.dtype == np.bool_:
        resp = resp.astype(np.uint8)
    else:
        resp_f = np.asarray(resp, dtype=np.float64)
        if not np.all(np.isin(resp_f, (0.0, 1.0))):
            raise ValueError("responses must contain only 0 and 1")
        resp = resp_f.astype(np.uint8)
    r = _core.py_sprt_classify(
        np.ascontiguousarray(a),
        np.ascontiguousarray(b),
        np.ascontiguousarray(c),
        np.ascontiguousarray(resp),
        float(theta_cut),
        float(delta),
        float(alpha),
        float(beta),
    )
    return {
        "decision": str(r["decision"]),
        "n_used": int(r["n_used"]),
        "llr": float(r["llr"]),
        "llr_trace": np.asarray(r["llr_trace"]),
    }