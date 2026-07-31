"""Paired-comparison scaling: Thurstone (1927) Case V, Bradley-Terry, and
Luce Spectral Ranking (LSR / I-LSR).

Thin wrappers over the Rust core (``mlsirm_core::scaling``). The Thurstone
algorithm follows the ``thurstone()`` function of the psych R package
(Revelle, 2025; source READ), which implements Thurstone's (1927) law of
comparative judgment under Case V (equal discriminal dispersions, zero
correlations). The Bradley-Terry maximum-likelihood fit uses the
minorization-maximization (MM) algorithm as implemented by choix 0.4.1
(Maystre, 2015-2020; ``opt.mm`` pairwise path, source READ), which choix
attributes to Hunter (2004). The LSR and I-LSR estimators follow choix's
``lsr.py`` dense pairwise path, which choix attributes to Maystre and
Grossglauser (2015). Thurstone (1927), Bradley and Terry (1952), Hunter
(2004), and Maystre and Grossglauser (2015) themselves were NOT read; they
are cited as the origins of the models/algorithms as described by the read
sources.

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
Maystre, L., & Grossglauser, M. (2015). Fast and accurate inference of
    Plackett-Luce models. Advances in Neural Information Processing
    Systems, 28, 172-180. [NOT READ; cited as described in choix source]
Revelle, W. (2025). psych: Procedures for psychological, psychometric, and
    personality research (R package). https://CRAN.R-project.org/package=psych
Thurstone, L. L. (1927). A law of comparative judgment. Psychological
    Review, 34(4), 273-286. [NOT READ; cited as described in psych source]

The circular-triads consistency test and Kendall's coefficient of agreement
follow eba 1.10-0 (Wickelmaier & Schmid; ``circular.R`` and ``kendall.u.R``,
source READ). Kendall and Babington Smith (1940) and Alway (1962, exact null
tables) were NOT read; they are cited as the origins as described by the eba
source and manual pages.

Kendall, M. G., & Babington Smith, B. (1940). On the method of paired
    comparisons. Biometrika, 31(3/4), 324-345. [NOT READ; cited as described
    in eba source]
Wickelmaier, F., & Schmid, C. (2004). A Matlab function to estimate choice
    model parameters from paired-comparison data. Behavior Research Methods,
    Instruments, and Computers, 36(1), 29-40. [eba package source READ]

The Elo rating system follows the CRAN PlayerRatings 1.1-0 package's
``elo()`` (Stephenson & Sonas, 2020; ``R/ratings.R`` and the ``elo_c`` C
kernel, source READ): batch-per-period updates where all expected scores
within a period use the period-start ratings. Elo (1978) was NOT read; it
is cited as the origin of the method as described by PlayerRatings.

The Glicko rating system follows the same package's ``glicko()`` (``glicko_c``
kernel, source READ) and Glickman's technical note *The Glicko system*
(READ; http://www.glicko.net/glicko/glicko.pdf), whose worked example the
implementation reproduces. Glickman (1999), the derivation paper, was NOT
read; it is cited as the method's origin as described by both READ sources.

Elo, A. E. (1978). The rating of chessplayers, past and present. Arco.
    [NOT READ; cited as described in PlayerRatings source]
Glickman, M. E. (n.d.). The Glicko system [Technical note]. Harvard
    University. http://www.glicko.net/glicko/glicko.pdf [READ]
Glickman, M. E. (1999). Parameter estimation in large dynamic paired
    comparison experiments. Applied Statistics, 48(3), 377-394. [NOT READ;
    cited as described in the READ sources]
Stephenson, A., & Sonas, J. (2020). PlayerRatings: Dynamic updating methods
    for player ratings estimation (R package, version 1.1-0).
    https://CRAN.R-project.org/package=PlayerRatings [READ]
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


@dataclass
class LsrResult:
    """Luce Spectral Ranking fit: ``params[i]`` is the centered log-worth of
    item i (mean exactly 0), ``weights[i]`` the stationary distribution of
    the LSR Markov chain scaled to sum exactly n (choix ``statdist``
    convention), ``iterations`` the number of LSR passes (1 for the one-shot
    spectral estimator; the pass count when the L1 criterion
    ``sum |new - prev| <= tol * n`` first fired for I-LSR, which always
    performs at least two passes)."""

    params: np.ndarray
    weights: np.ndarray
    iterations: int


def _lsr_validate(name, wins):
    arr = np.asarray(wins)
    if np.iscomplexobj(arr):
        raise ValueError(f"{name}: wins must be real-valued")
    if arr.dtype == object:
        try:
            arr = arr.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: wins must be numeric") from exc
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name}: wins must be a square 2-D matrix")
    return arr


def lsr_pairwise(wins, alpha=0.0):
    """One-shot Luce Spectral Ranking from an n x n win-count matrix.

    ``wins[i, j]`` is the number of times item *i* beat item *j* (row beats
    column; zero diagonal; non-integer counts accepted -- same DERIVED
    extension as :func:`bradley_terry_mm`). Builds the LSR Markov chain
    (rate ``c / (w_i + w_j)`` on each loser-to-winner edge under uniform
    initial worths, plus ``alpha`` everywhere as a regularizer) and returns
    the centered log of its stationary distribution (choix
    ``lsr_pairwise_dense``). Raises ValueError when the stationary
    distribution does not exist or is not unique -- e.g. a disconnected
    comparison graph at ``alpha = 0`` (choix raises there too) -- and on
    overflow from huge counts or alpha. NOTE: ``alpha`` regularizes the
    chain rates, which is NOT the Dirichlet-MAP ``alpha`` of
    :func:`bradley_terry_mm`; the two disagree for ``alpha > 0``.
    """
    from .fitstats import _core_module

    arr = _lsr_validate("lsr_pairwise", wins)
    res = _core_module().lsr_pairwise(arr.ravel(), arr.shape[0], float(alpha))
    return LsrResult(
        params=np.asarray(res["params"], dtype=np.float64),
        weights=np.asarray(res["weights"], dtype=np.float64),
        iterations=int(res["iterations"]),
    )


def ilsr_pairwise(wins, alpha=0.0, max_iter=100, tol=1e-8):
    """Iterative Luce Spectral Ranking (Bradley-Terry MLE) from an n x n
    win-count matrix.

    Repeats the LSR pass of :func:`lsr_pairwise`, feeding each pass the
    worths from the previous one, until the L1 change is ``<= tol * n``
    (choix ``ilsr_pairwise_dense``; default ``max_iter=100`` as in choix).
    At ``alpha = 0`` the fixed point is the Bradley-Terry maximum-likelihood
    estimate and agrees with :func:`bradley_terry_mm`; for ``alpha > 0`` the
    two regularization paths deliberately differ (each follows its source).
    Raises ValueError on invalid input, a disconnected comparison graph at
    ``alpha = 0``, overflow, or non-convergence within ``max_iter`` passes.
    """
    from .fitstats import _core_module

    arr = _lsr_validate("ilsr_pairwise", wins)
    res = _core_module().ilsr_pairwise(
        arr.ravel(), arr.shape[0], float(alpha), int(max_iter), float(tol)
    )
    return LsrResult(
        params=np.asarray(res["params"], dtype=np.float64),
        weights=np.asarray(res["weights"], dtype=np.float64),
        iterations=int(res["iterations"]),
    )

def rank_centrality(wins, alpha=0.0):
    """Rank Centrality: spectral ranking from the *ratios* of pairwise wins.

    ``wins[i, j]`` is the number of times item *i* beat item *j* (row beats
    column; zero diagonal). The Markov chain accrues, on each
    loser-to-winner edge, the win *ratio*
    ``(alpha + c_win) / (2 * alpha + c_win + c_lose)`` -- choix
    ``rank_centrality``, which ports Negahban, Oh, & Shah's (2017)
    algorithm as a continuous-time chain (the paper's discrete-time
    max-degree walk is NOT what choix, or therefore this port, computes;
    only the choix variant is implemented and verified). Returns the
    centered log of the stationary distribution as :class:`LsrResult`
    (``iterations`` always 1).

    At ``alpha = 0`` the result is exactly invariant under a global
    rescaling of all counts; for fixed ``alpha > 0`` it is not. Raises
    ValueError on invalid input, a disconnected comparison graph at
    ``alpha = 0``, or overflowing intermediates (e.g. ``alpha = 1e308``,
    where the ratio denominator overflows -- choix instead silently
    produces near-zero ratios). An all-zero wins matrix is rejected even
    when ``alpha > 0`` (documented divergence from choix, which would
    regularize it to a uniform chain).
    """
    from .fitstats import _core_module

    arr = _lsr_validate("rank_centrality", wins)
    res = _core_module().rank_centrality(arr.ravel(), arr.shape[0], float(alpha))
    return LsrResult(
        params=np.asarray(res["params"], dtype=np.float64),
        weights=np.asarray(res["weights"], dtype=np.float64),
        iterations=int(res["iterations"]),
    )

def _rankings_to_csr(name, rankings, n):
    """Validate a list of rankings (best first) and CSR-flatten to u64.

    Rejects, BEFORE any unsigned cast: non-integer entries, negative
    indices (a documented divergence -- Python's negative indices would
    silently wrap in choix), booleans, complex/object dtypes, rankings
    shorter than 2 items (choix silently no-ops those), and out-of-range
    items. Duplicate detection within a ranking is enforced by the Rust
    core (documented divergence: choix accepts duplicates whenever the
    chain stays connected).
    """
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or int(n) < 2:
        raise ValueError(f"{name}: n must be an integer >= 2")
    n = int(n)
    if n > 10_000:
        # Mirrors the Rust dense-chain cap BEFORE any uint64/usize cast,
        # so a huge n raises ValueError, never a raw OverflowError.
        raise ValueError(f"{name}: n = {n} exceeds the 10000-item cap")
    flat = []
    starts = [0]
    for r, ranking in enumerate(rankings):
        items = list(ranking)
        if len(items) < 2:
            raise ValueError(
                f"{name}: ranking {r} has fewer than 2 items "
                "(choix silently ignores such rankings; this port rejects them)"
            )
        for x in items:
            if isinstance(x, (bool, np.bool_)) or np.iscomplexobj(np.asarray(x)):
                raise ValueError(f"{name}: ranking {r} has a non-integer item")
            try:
                xi = int(x)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(f"{name}: ranking {r} has a non-integer item")
            if xi != x:
                raise ValueError(f"{name}: ranking {r} has a non-integer item")
            if xi < 0:
                raise ValueError(
                    f"{name}: ranking {r} has a negative item index "
                    "(negative indices do not wrap here)"
                )
            if xi >= n:
                raise ValueError(f"{name}: ranking {r} has item {xi} >= n = {n}")
            flat.append(xi)
        starts.append(len(flat))
    if len(starts) < 2:
        raise ValueError(f"{name}: at least one ranking is required")
    return (
        np.asarray(flat, dtype=np.uint64),
        np.asarray(starts, dtype=np.uint64),
        n,
    )


def lsr_rankings(rankings, n, alpha=0.0):
    """Luce Spectral Ranking for full or partial rankings (one shot).

    ``rankings`` is an iterable of rankings, each an iterable of item
    indices in ``0..n-1`` ordered best first (partial rankings -- any
    length >= 2 -- are allowed and are what distinguishes the
    Plackett-Luce subset denominator from a naive all-items one). Ports
    choix 0.4.1 ``lsr_rankings``: each ranking is a sequence of Luce
    choices; position *i* accrues rate ``1 / (sum of remaining ranked
    worths)`` on every loser-to-winner edge, plus ``alpha`` everywhere.
    Returns the centered log stationary distribution as
    :class:`LsrResult` (``iterations`` always 1). Raises ValueError on
    invalid input, within-ranking duplicates, a disconnected item graph
    at ``alpha = 0``, or overflow.
    """
    from .fitstats import _core_module

    rk, st, n = _rankings_to_csr("lsr_rankings", rankings, n)
    res = _core_module().lsr_rankings(rk, st, n, float(alpha))
    return LsrResult(
        params=np.asarray(res["params"], dtype=np.float64),
        weights=np.asarray(res["weights"], dtype=np.float64),
        iterations=int(res["iterations"]),
    )


def ilsr_rankings(rankings, n, alpha=0.0, max_iter=100, tol=1e-8):
    """Iterative LSR for rankings (Plackett-Luce MLE at ``alpha = 0``).

    Repeats the :func:`lsr_rankings` pass, feeding each pass the worths
    from the previous one, until the L1 parameter change is
    ``<= tol * n`` (choix 0.4.1 ``ilsr_rankings``; defaults match choix).
    Raises ValueError on invalid input, a disconnected item graph at
    ``alpha = 0``, overflow, or non-convergence within ``max_iter``.
    """
    from .fitstats import _core_module

    rk, st, n = _rankings_to_csr("ilsr_rankings", rankings, n)
    res = _core_module().ilsr_rankings(
        rk, st, n, float(alpha), int(max_iter), float(tol)
    )
    return LsrResult(
        params=np.asarray(res["params"], dtype=np.float64),
        weights=np.asarray(res["weights"], dtype=np.float64),
        iterations=int(res["iterations"]),
    )

def _top1_to_csr(name, data, n):
    """Validate top-1 choice data and CSR-flatten to u64 arrays.

    ``data`` is an iterable of ``(winner, losers)`` pairs. Rejects,
    BEFORE any unsigned cast: non-integer entries (bool/np.bool_,
    complex, non-integral floats, int() overflow), negative indices (a
    documented divergence -- Python's negative indices would silently
    wrap in choix), empty loser sets (choix silently no-ops those), and
    out-of-range indices. Winner-in-losers and duplicate-loser detection
    is enforced by the Rust core (documented divergences: choix accepts
    both, silently corrupting the denominator).
    """
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or int(n) < 2:
        raise ValueError(f"{name}: n must be an integer >= 2")
    n = int(n)
    if n > 10_000:
        # Mirrors the Rust dense-chain cap BEFORE any uint64/usize cast,
        # so a huge n raises ValueError, never a raw OverflowError.
        raise ValueError(f"{name}: n = {n} exceeds the 10000-item cap")

    def _index(x, what, r):
        if isinstance(x, (bool, np.bool_)) or np.iscomplexobj(np.asarray(x)):
            raise ValueError(f"{name}: observation {r} has a non-integer {what}")
        try:
            xi = int(x)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name}: observation {r} has a non-integer {what}")
        if xi != x:
            raise ValueError(f"{name}: observation {r} has a non-integer {what}")
        if xi < 0:
            raise ValueError(
                f"{name}: observation {r} has a negative {what} index "
                "(negative indices do not wrap here)"
            )
        if xi >= n:
            raise ValueError(f"{name}: observation {r} has {what} {xi} >= n = {n}")
        return xi

    winners = []
    flat = []
    starts = [0]
    for r, obs in enumerate(data):
        winner, losers = obs
        winners.append(_index(winner, "winner", r))
        losers = list(losers)
        if not losers:
            raise ValueError(
                f"{name}: observation {r} has an empty loser set "
                "(choix silently ignores such observations; this port rejects them)"
            )
        for x in losers:
            flat.append(_index(x, "loser", r))
        starts.append(len(flat))
    if len(starts) < 2:
        raise ValueError(f"{name}: at least one observation is required")
    return (
        np.asarray(winners, dtype=np.uint64),
        np.asarray(flat, dtype=np.uint64),
        np.asarray(starts, dtype=np.uint64),
        n,
    )


def lsr_top1(data, n, alpha=0.0):
    """Luce Spectral Ranking for top-1 choice data (one shot).

    ``data`` is an iterable of ``(winner, losers)`` pairs: the winner
    was chosen out of the choice set ``{winner} | set(losers)``. Ports
    choix 0.4.1 ``lsr_top1``: each observation accrues rate
    ``1 / (sum of choice-set worths)`` on every loser-to-winner edge,
    plus ``alpha`` everywhere off-diagonal. Returns the centered log
    stationary distribution as :class:`LsrResult` (``iterations``
    always 1). Raises ValueError on invalid input, empty loser sets,
    winner-in-losers, duplicate losers, a disconnected item graph at
    ``alpha = 0``, or overflow.
    """
    from .fitstats import _core_module

    wn, ls, st, n = _top1_to_csr("lsr_top1", data, n)
    res = _core_module().lsr_top1(wn, ls, st, n, float(alpha))
    return LsrResult(
        params=np.asarray(res["params"], dtype=np.float64),
        weights=np.asarray(res["weights"], dtype=np.float64),
        iterations=int(res["iterations"]),
    )


def ilsr_top1(data, n, alpha=0.0, max_iter=100, tol=1e-8):
    """Iterative LSR for top-1 choice data (Luce-choice MLE at alpha=0).

    Repeats the :func:`lsr_top1` pass, feeding each pass the worths from
    the previous one, until the L1 parameter change is ``<= tol * n``
    (choix 0.4.1 ``ilsr_top1``; defaults match choix). Raises ValueError
    on invalid input, a disconnected item graph at ``alpha = 0``,
    overflow, or non-convergence within ``max_iter``.
    """
    from .fitstats import _core_module

    wn, ls, st, n = _top1_to_csr("ilsr_top1", data, n)
    res = _core_module().ilsr_top1(
        wn, ls, st, n, float(alpha), int(max_iter), float(tol)
    )
    return LsrResult(
        params=np.asarray(res["params"], dtype=np.float64),
        weights=np.asarray(res["weights"], dtype=np.float64),
        iterations=int(res["iterations"]),
    )


@dataclass
class CircularTriadsResult:
    """Circular-triads consistency test (Kendall & Babington Smith, 1940,
    as implemented by eba's ``circular()``): ``t`` the number of circular
    triads T, ``t_max`` its maximum, ``t_exp`` its null expectation
    C(n,3)/4, ``zeta = 1 - t/t_max`` the consistency coefficient, and the
    test of the null "preferences are random". For ``n <= 10`` the p-value
    is EXACT (``exact=True``, ``chi2``/``df`` are NaN); for ``n >= 11`` a
    chi-square approximation is used and ``chi2`` / ``df`` are filled."""

    t: float
    t_max: float
    t_exp: float
    zeta: float
    chi2: float
    df: float
    p_value: float
    exact: bool


def _kendall_matrix(name, mat):
    arr = np.asarray(mat)
    if np.iscomplexobj(arr):
        raise ValueError(f"{name}: mat must be real-valued")
    if arr.dtype == object:
        try:
            arr = arr.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: mat must be numeric") from exc
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name}: mat must be a square 2-D matrix")
    return arr, arr.shape[0]


def circular_triads(mat, alternative="two.sided", correct=True):
    """Test one judge's paired comparisons for intransitivity.

    ``mat[i, j] = 1`` means object *i* was preferred over object *j*; the
    tournament must be complete (``mat[i, j] + mat[j, i] == 1`` for every
    pair, zero diagonal). ``alternative`` is one of ``"two.sided"``,
    ``"less"`` (fewer circular triads than chance, i.e. consistency), or
    ``"greater"``. ``correct`` applies the +-0.5 continuity correction on
    the chi-square path (``n >= 11``) only. Deliberate divergences from
    eba's ``circular()``: ``n = 2`` is rejected (T_max = 0 makes zeta
    undefined) and malformed tournaments raise instead of returning
    garbage. Raises ValueError on invalid input.
    """
    from .fitstats import _core_module

    arr, n = _kendall_matrix("circular_triads", mat)
    res = _core_module().circular_triads(
        arr.ravel(), n, str(alternative), bool(correct)
    )
    return CircularTriadsResult(
        t=float(res["t"]),
        t_max=float(res["t_max"]),
        t_exp=float(res["t_exp"]),
        zeta=float(res["zeta"]),
        chi2=float(res["chi2"]),
        df=float(res["df"]),
        p_value=float(res["p_value"]),
        exact=bool(res["exact"]),
    )


@dataclass
class KendallUResult:
    """Kendall's coefficient of agreement u between m judges (Kendall &
    Babington Smith, 1940, as implemented by eba's ``kendall.u()``):
    ``sigma`` the number of agreeing judge pairs, ``u`` the agreement
    coefficient (1 = perfect agreement; minimum ``min_u`` is -1/m for odd
    m, -1/(m-1) for even m), and the chi-square test of the null "agreement
    is by chance". ``chi2`` is returned RAW and can be negative under
    strong disagreement with the continuity correction; only the p-value
    clamps (a negative statistic gives p = 1)."""

    sigma: float
    u: float
    min_u: float
    chi2: float
    df: float
    p_value: float


def kendall_u(mat, correct=True):
    """Agreement between m judges from an n x n frequency matrix.

    ``mat[i, j]`` counts the judges who preferred object *i* over object
    *j*. Every pair must have the same number of observations
    ``m = mat[i, j] + mat[j, i] >= 3`` (a documented STRICTER check than
    eba, which reads m from the first pair only); entries must be
    nonnegative integers and the diagonal zero. ``correct`` applies the
    continuity correction (subtract 1 from Sigma). Raises ValueError on
    invalid input.
    """
    from .fitstats import _core_module

    arr, n = _kendall_matrix("kendall_u", mat)
    res = _core_module().kendall_u(arr.ravel(), n, bool(correct))
    return KendallUResult(
        sigma=float(res["sigma"]),
        u=float(res["u"]),
        min_u=float(res["min_u"]),
        chi2=float(res["chi2"]),
        df=float(res["df"]),
        p_value=float(res["p_value"]),
    )

@dataclass
class EloResult:
    """Elo ratings and bookkeeping (Elo, 1978, as implemented by the CRAN
    PlayerRatings package's ``elo()``; batch-per-period update). Wins,
    draws, and losses count only scores exactly 1, 0.5, and 0; other
    fractional scores count a game without a W/D/L. ``lag`` is the number
    of rating periods since the player's last appearance (0 for players
    appearing in the final period or never playing)."""

    ratings: "np.ndarray"
    games: "np.ndarray"
    wins: "np.ndarray"
    draws: "np.ndarray"
    losses: "np.ndarray"
    lag: "np.ndarray"


def elo_rating(games, n_players, init=2200.0, kfac=27.0, gamma=None):
    """Elo ratings from a (g, 4) game schedule.

    Each row of ``games`` is ``[period, white, black, score]``: an integer
    rating-period label, integer player indices in ``0..n_players``, and
    white's score in [0, 1]. Rows are grouped by ascending period value
    (matching R ``split()`` ordering); within a period all expected scores
    use the period-start ratings (batch update). All players start at
    ``init``; ``kfac`` is the scalar K factor (any finite value, including
    0); ``gamma`` is white's per-game advantage — a scalar (broadcast) or a
    length-g array. Defaults ``init=2200, kfac=27`` are the PlayerRatings
    defaults. Raises ValueError on invalid input.
    """
    import numpy as np

    from .fitstats import _core_module

    if np.iscomplexobj(np.asarray(games)):
        raise ValueError("elo_rating: games must be real, not complex")
    raw = np.asarray(games)
    try:
        arr = np.asarray(games, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"elo_rating: games is not numeric: {exc}") from None
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(
            f"elo_rating: games must be (g, 4) [period, white, black, score], got {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise ValueError("elo_rating: at least one game is required")
    if not np.all(np.isfinite(arr)):
        raise ValueError("elo_rating: games contains non-finite values")
    periods, white, black, score = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    for name, col in (("period", periods), ("white", white), ("black", black)):
        if np.any(col != np.floor(col)):
            raise ValueError(f"elo_rating: {name} column must be integral")
    if np.any(periods < 0):
        raise ValueError("elo_rating: period labels must be nonnegative")
    # Period labels are u64 in the core; a float column loses integer
    # fidelity above 2**53 (distinct labels would silently merge into one
    # rating period). Take periods losslessly from an integer input array,
    # and otherwise reject labels the float path cannot represent exactly.
    if raw.ndim == 2 and raw.dtype.kind in "iu":
        if raw.dtype.kind == "i" and np.any(raw[:, 0] < 0):
            raise ValueError("elo_rating: period labels must be nonnegative")
        periods_u64 = raw[:, 0].astype(np.uint64)
    else:
        # Reject labels at/above the source dtype's exact-integer bound
        # (2**mantissa_bits + 1 is the first non-representable integer, so a
        # value equal to the bound is already ambiguous): float64/others
        # 2**53, float32 2**24, float16 2**11. np.finfo(...).nmant excludes
        # the implicit leading bit, so the exact-integer bound is nmant + 1.
        if raw.dtype.kind == "f":
            fidelity = 2.0 ** (np.finfo(raw.dtype).nmant + 1)
        else:
            fidelity = 2.0**53
        if np.any(periods >= fidelity):
            raise ValueError(
                f"elo_rating: period labels at or above {int(fidelity)} are not "
                f"reliably representable in {raw.dtype if raw.dtype.kind == 'f' else 'float64'}; "
                "pass games as an integer array"
            )
        periods_u64 = periods.astype(np.uint64)
    if np.any(white < 0) or np.any(black < 0):
        raise ValueError("elo_rating: player indices must be nonnegative")
    n = int(n_players)
    if n != n_players:
        raise ValueError("elo_rating: n_players must be an integer")
    g = arr.shape[0]
    if gamma is None:
        gamma_arr = np.zeros(g)
    else:
        gamma_arr = np.asarray(gamma, dtype=float)
        if np.iscomplexobj(np.asarray(gamma)):
            raise ValueError("elo_rating: gamma must be real, not complex")
        if gamma_arr.ndim == 0:
            gamma_arr = np.full(g, float(gamma_arr))
        elif gamma_arr.shape != (g,):
            raise ValueError(
                f"elo_rating: gamma must be a scalar or length-{g} array, got {gamma_arr.shape}"
            )
    init = float(init)
    kfac = float(kfac)
    res = _core_module().elo_rating(
        np.ascontiguousarray(periods_u64),
        np.ascontiguousarray(white, dtype=np.uint64),
        np.ascontiguousarray(black, dtype=np.uint64),
        np.ascontiguousarray(score),
        np.ascontiguousarray(gamma_arr),
        n,
        init,
        kfac,
    )
    return EloResult(
        ratings=np.asarray(res["ratings"]),
        games=np.asarray(res["games"]),
        wins=np.asarray(res["wins"]),
        draws=np.asarray(res["draws"]),
        losses=np.asarray(res["losses"]),
        lag=np.asarray(res["lag"]),
    )


@dataclass
class GlickoResult:
    """Glicko ratings, deviations, and bookkeeping (Glickman, n.d., as
    implemented by the CRAN PlayerRatings package's ``glicko()``;
    batch-per-period update with participant-only deviation inflation
    ``RD = min(sqrt(RD^2 + (lag+1) c^2), rdmax)``). Wins, draws, and losses
    count only scores exactly 1, 0.5, and 0; other fractional scores count
    a game without a W/D/L. ``lag`` is the number of rating periods since
    the player's last appearance."""

    ratings: "np.ndarray"
    deviations: "np.ndarray"
    games: "np.ndarray"
    wins: "np.ndarray"
    draws: "np.ndarray"
    losses: "np.ndarray"
    lag: "np.ndarray"


def glicko_rating(
    games, n_players, init=(2200.0, 300.0), gamma=None, cval=15.0, rdmax=350.0
):
    """Glicko ratings from a (g, 4) game schedule.

    Each row of ``games`` is ``[period, white, black, score]`` exactly as in
    :func:`elo_rating`. ``init`` is either a ``(rating, deviation)`` scalar
    pair broadcast to all players or a pair of length-``n_players`` arrays
    (per-player starting ratings/deviations); ``cval`` is the per-period
    uncertainty growth constant and ``rdmax`` the deviation ceiling
    (Glickman's Step 1b: ``RD = min(sqrt(RD^2 + (lag+1) c^2), rdmax)``).
    Defaults ``init=(2200, 300), cval=15, rdmax=350`` are the PlayerRatings
    defaults. Unlike R's fresh-start path, results cover ALL players in
    ``0..n_players``: never-playing players keep their init rating and
    deviation with zero tallies. Raises ValueError on invalid input.
    """
    import numpy as np

    from .fitstats import _core_module

    if np.iscomplexobj(np.asarray(games)):
        raise ValueError("glicko_rating: games must be real, not complex")
    raw = np.asarray(games)
    try:
        arr = np.asarray(games, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"glicko_rating: games is not numeric: {exc}") from None
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(
            f"glicko_rating: games must be (g, 4) [period, white, black, score], got {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise ValueError("glicko_rating: at least one game is required")
    if not np.all(np.isfinite(arr)):
        raise ValueError("glicko_rating: games contains non-finite values")
    periods, white, black, score = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    for name, col in (("period", periods), ("white", white), ("black", black)):
        if np.any(col != np.floor(col)):
            raise ValueError(f"glicko_rating: {name} column must be integral")
    if np.any(periods < 0):
        raise ValueError("glicko_rating: period labels must be nonnegative")
    # Period labels are u64 in the core; a float column loses integer
    # fidelity above the dtype's exact-integer bound (distinct labels would
    # silently merge into one rating period). Same contract as elo_rating.
    if raw.ndim == 2 and raw.dtype.kind in "iu":
        if raw.dtype.kind == "i" and np.any(raw[:, 0] < 0):
            raise ValueError("glicko_rating: period labels must be nonnegative")
        periods_u64 = raw[:, 0].astype(np.uint64)
    else:
        # np.finfo(...).nmant excludes the implicit leading bit, so the
        # exact-integer bound is nmant + 1 (float64 2**53, float32 2**24).
        if raw.dtype.kind == "f":
            fidelity = 2.0 ** (np.finfo(raw.dtype).nmant + 1)
        else:
            fidelity = 2.0**53
        if np.any(periods >= fidelity):
            raise ValueError(
                f"glicko_rating: period labels at or above {int(fidelity)} are not "
                f"reliably representable in {raw.dtype if raw.dtype.kind == 'f' else 'float64'}; "
                "pass games as an integer array"
            )
        periods_u64 = periods.astype(np.uint64)
    if np.any(white < 0) or np.any(black < 0):
        raise ValueError("glicko_rating: player indices must be nonnegative")
    n = int(n_players)
    if n != n_players:
        raise ValueError("glicko_rating: n_players must be an integer")
    g = arr.shape[0]
    if gamma is None:
        gamma_arr = np.zeros(g)
    else:
        if np.iscomplexobj(np.asarray(gamma)):
            raise ValueError("glicko_rating: gamma must be real, not complex")
        gamma_arr = np.asarray(gamma, dtype=float)
        if gamma_arr.ndim == 0:
            gamma_arr = np.full(g, float(gamma_arr))
        elif gamma_arr.shape != (g,):
            raise ValueError(
                f"glicko_rating: gamma must be a scalar or length-{g} array, got {gamma_arr.shape}"
            )
    try:
        init_r_in, init_d_in = init
    except (TypeError, ValueError):
        raise ValueError(
            "glicko_rating: init must be a (rating, deviation) pair"
        ) from None
    if np.iscomplexobj(np.asarray(init_r_in)) or np.iscomplexobj(np.asarray(init_d_in)):
        raise ValueError("glicko_rating: init must be real, not complex")
    try:
        init_r = np.asarray(init_r_in, dtype=float)
        init_d = np.asarray(init_d_in, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"glicko_rating: init is not numeric: {exc}") from None
    if init_r.ndim == 0 and init_d.ndim == 0:
        init_r = np.full(n, float(init_r))
        init_d = np.full(n, float(init_d))
    elif init_r.shape != (n,) or init_d.shape != (n,):
        raise ValueError(
            "glicko_rating: init must be a scalar pair or a pair of "
            f"length-{n} arrays, got shapes {init_r.shape} and {init_d.shape}"
        )
    res = _core_module().glicko_rating(
        np.ascontiguousarray(periods_u64),
        np.ascontiguousarray(white, dtype=np.uint64),
        np.ascontiguousarray(black, dtype=np.uint64),
        np.ascontiguousarray(score),
        np.ascontiguousarray(gamma_arr),
        np.ascontiguousarray(init_r),
        np.ascontiguousarray(init_d),
        float(cval),
        float(rdmax),
    )
    return GlickoResult(
        ratings=np.asarray(res["ratings"]),
        deviations=np.asarray(res["deviations"]),
        games=np.asarray(res["games"]),
        wins=np.asarray(res["wins"]),
        draws=np.asarray(res["draws"]),
        losses=np.asarray(res["losses"]),
        lag=np.asarray(res["lag"]),
    )


@dataclass
class Glicko2Result:
    """Glicko-2 ratings, deviations, volatilities, and bookkeeping
    (Glickman, 2022, as implemented by the CRAN PlayerRatings package's
    ``glicko2()``; batch-per-period update with participant-only variance
    inflation ``phi^2 <- min(phi^2 + lag * sigma^2, (q * rdmax)^2)`` and the
    Illinois volatility iteration of Glickman's Step 5). Wins, draws, and
    losses count only scores exactly 1, 0.5, and 0; other fractional scores
    count a game without a W/D/L. ``lag`` is the number of rating periods
    since the player's last appearance."""

    ratings: "np.ndarray"
    deviations: "np.ndarray"
    volatilities: "np.ndarray"
    games: "np.ndarray"
    wins: "np.ndarray"
    draws: "np.ndarray"
    losses: "np.ndarray"
    lag: "np.ndarray"


def glicko2_rating(
    games, n_players, init=(2200.0, 300.0, 0.15), gamma=None, tau=1.2, rdmax=350.0
):
    """Glicko-2 ratings from a (g, 4) game schedule.

    Each row of ``games`` is ``[period, white, black, score]`` exactly as in
    :func:`elo_rating`. ``init`` is either a ``(rating, deviation,
    volatility)`` scalar triple broadcast to all players or a triple of
    length-``n_players`` arrays; ``tau`` constrains per-period volatility
    change (``tau=0`` freezes volatility) and ``rdmax`` is the deviation
    ceiling (volatility is capped at ``ln(10)/400 * rdmax``). Defaults
    ``init=(2200, 300, 0.15), tau=1.2, rdmax=350`` are the PlayerRatings
    defaults. Unlike R's fresh-start path, results cover ALL players in
    ``0..n_players``: never-playing players keep their init state with zero
    tallies. Raises ValueError on invalid input.
    """
    import numpy as np

    from .fitstats import _core_module

    if np.iscomplexobj(np.asarray(games)):
        raise ValueError("glicko2_rating: games must be real, not complex")
    raw = np.asarray(games)
    try:
        arr = np.asarray(games, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"glicko2_rating: games is not numeric: {exc}") from None
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(
            f"glicko2_rating: games must be (g, 4) [period, white, black, score], got {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise ValueError("glicko2_rating: at least one game is required")
    if not np.all(np.isfinite(arr)):
        raise ValueError("glicko2_rating: games contains non-finite values")
    periods, white, black, score = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    for name, col in (("period", periods), ("white", white), ("black", black)):
        if np.any(col != np.floor(col)):
            raise ValueError(f"glicko2_rating: {name} column must be integral")
    if np.any(periods < 0):
        raise ValueError("glicko2_rating: period labels must be nonnegative")
    # Period labels are u64 in the core; a float column loses integer
    # fidelity above the dtype's exact-integer bound (distinct labels would
    # silently merge into one rating period). Same contract as elo_rating.
    if raw.ndim == 2 and raw.dtype.kind in "iu":
        if raw.dtype.kind == "i" and np.any(raw[:, 0] < 0):
            raise ValueError("glicko2_rating: period labels must be nonnegative")
        periods_u64 = raw[:, 0].astype(np.uint64)
    else:
        # np.finfo(...).nmant excludes the implicit leading bit, so the
        # exact-integer bound is nmant + 1 (float64 2**53, float32 2**24).
        if raw.dtype.kind == "f":
            fidelity = 2.0 ** (np.finfo(raw.dtype).nmant + 1)
        else:
            fidelity = 2.0**53
        if np.any(periods >= fidelity):
            raise ValueError(
                f"glicko2_rating: period labels at or above {int(fidelity)} are not "
                f"reliably representable in {raw.dtype if raw.dtype.kind == 'f' else 'float64'}; "
                "pass games as an integer array"
            )
        periods_u64 = periods.astype(np.uint64)
    if np.any(white < 0) or np.any(black < 0):
        raise ValueError("glicko2_rating: player indices must be nonnegative")
    try:
        n = int(n_players)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"glicko2_rating: n_players is not an integer: {exc}") from None
    if n != n_players:
        raise ValueError("glicko2_rating: n_players must be an integer")
    g = arr.shape[0]
    if gamma is None:
        gamma_arr = np.zeros(g)
    else:
        if np.iscomplexobj(np.asarray(gamma)):
            raise ValueError("glicko2_rating: gamma must be real, not complex")
        try:
            gamma_arr = np.asarray(gamma, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"glicko2_rating: gamma is not numeric: {exc}") from None
        if gamma_arr.ndim == 0:
            gamma_arr = np.full(g, float(gamma_arr))
        elif gamma_arr.shape != (g,):
            raise ValueError(
                f"glicko2_rating: gamma must be a scalar or length-{g} array, got {gamma_arr.shape}"
            )
    try:
        init_r_in, init_d_in, init_v_in = init
    except (TypeError, ValueError):
        raise ValueError(
            "glicko2_rating: init must be a (rating, deviation, volatility) triple"
        ) from None
    if (
        np.iscomplexobj(np.asarray(init_r_in))
        or np.iscomplexobj(np.asarray(init_d_in))
        or np.iscomplexobj(np.asarray(init_v_in))
    ):
        raise ValueError("glicko2_rating: init must be real, not complex")
    try:
        init_r = np.asarray(init_r_in, dtype=float)
        init_d = np.asarray(init_d_in, dtype=float)
        init_v = np.asarray(init_v_in, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"glicko2_rating: init is not numeric: {exc}") from None
    if init_r.ndim == 0 and init_d.ndim == 0 and init_v.ndim == 0:
        init_r = np.full(n, float(init_r))
        init_d = np.full(n, float(init_d))
        init_v = np.full(n, float(init_v))
    elif init_r.shape != (n,) or init_d.shape != (n,) or init_v.shape != (n,):
        raise ValueError(
            "glicko2_rating: init must be a scalar triple or a triple of "
            f"length-{n} arrays, got shapes {init_r.shape}, {init_d.shape}, "
            f"and {init_v.shape}"
        )
    try:
        tau_f = float(tau)
        rdmax_f = float(rdmax)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"glicko2_rating: tau/rdmax is not numeric: {exc}") from None
    res = _core_module().glicko2_rating(
        np.ascontiguousarray(periods_u64),
        np.ascontiguousarray(white, dtype=np.uint64),
        np.ascontiguousarray(black, dtype=np.uint64),
        np.ascontiguousarray(score),
        np.ascontiguousarray(gamma_arr),
        np.ascontiguousarray(init_r),
        np.ascontiguousarray(init_d),
        np.ascontiguousarray(init_v),
        tau_f,
        rdmax_f,
    )
    return Glicko2Result(
        ratings=np.asarray(res["ratings"]),
        deviations=np.asarray(res["deviations"]),
        volatilities=np.asarray(res["volatilities"]),
        games=np.asarray(res["games"]),
        wins=np.asarray(res["wins"]),
        draws=np.asarray(res["draws"]),
        losses=np.asarray(res["losses"]),
        lag=np.asarray(res["lag"]),
    )


@dataclass
class StephensonResult:
    """Stephenson ratings, deviations, and bookkeeping (the CRAN
    PlayerRatings package's ``steph()``; no journal paper exists for this
    system, so the R driver and C kernel of PlayerRatings 1.1.0 are the
    normative sources -- READ, lines cited in the Rust core). Extends
    Glicko with a per-game neighborhood term ``hval``, a bonus ``bval``
    added to each played game's score, and a lambda drift toward
    opponents' ratings. Wins, draws, and losses count only scores exactly
    1, 0.5, and 0 in the CURRENT run; ``lag`` is the number of rating
    periods since the player's last appearance."""

    ratings: "np.ndarray"
    deviations: "np.ndarray"
    games: "np.ndarray"
    wins: "np.ndarray"
    draws: "np.ndarray"
    losses: "np.ndarray"
    lag: "np.ndarray"


def stephenson_rating(
    games,
    n_players,
    init=(2200.0, 300.0),
    gamma=None,
    init_games=None,
    init_lag=None,
    cval=10.0,
    hval=10.0,
    bval=0.0,
    lambda_=2.0,
    rdmax=350.0,
):
    """Stephenson ratings from a (g, 4) game schedule.

    Each row of ``games`` is ``[period, white, black, score]`` exactly as in
    :func:`elo_rating`. ``init`` is either a ``(rating, deviation)`` scalar
    pair broadcast to all players or a pair of length-``n_players`` arrays;
    ``init_games``/``init_lag`` optionally continue a prior run (length-
    ``n_players`` nonnegative integer arrays, default all-zero). ``cval`` is
    the per-period deviation inflation, ``hval`` the per-game neighborhood
    inflation, ``bval`` the per-game bonus (added to each played game's
    score as ``bval/100``), ``lambda_`` the drift toward opponents' ratings
    (``lambda_=0`` disables drift), and ``rdmax`` the deviation ceiling.
    Defaults ``init=(2200, 300), cval=10, hval=10, bval=0, lambda_=2,
    rdmax=350`` are the PlayerRatings defaults. Results cover ALL players
    in ``0..n_players``: never-playing players keep their init state with
    zero current-run tallies. Raises ValueError on invalid input.
    """
    import numpy as np

    from .fitstats import _core_module

    if np.iscomplexobj(np.asarray(games)):
        raise ValueError("stephenson_rating: games must be real, not complex")
    raw = np.asarray(games)
    try:
        arr = np.asarray(games, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"stephenson_rating: games is not numeric: {exc}") from None
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(
            f"stephenson_rating: games must be (g, 4) [period, white, black, score], got {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise ValueError("stephenson_rating: at least one game is required")
    if not np.all(np.isfinite(arr)):
        raise ValueError("stephenson_rating: games contains non-finite values")
    periods, white, black, score = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    for name, col in (("period", periods), ("white", white), ("black", black)):
        if np.any(col != np.floor(col)):
            raise ValueError(f"stephenson_rating: {name} column must be integral")
    if np.any(periods < 0):
        raise ValueError("stephenson_rating: period labels must be nonnegative")
    # Period labels are u64 in the core; a float column loses integer
    # fidelity above the dtype's exact-integer bound (distinct labels would
    # silently merge into one rating period). Same contract as elo_rating.
    if raw.ndim == 2 and raw.dtype.kind in "iu":
        if raw.dtype.kind == "i" and np.any(raw[:, 0] < 0):
            raise ValueError("stephenson_rating: period labels must be nonnegative")
        periods_u64 = raw[:, 0].astype(np.uint64)
        white_u64 = raw[:, 1].astype(np.uint64)
        black_u64 = raw[:, 2].astype(np.uint64)
    else:
        # np.finfo(...).nmant excludes the implicit leading bit, so the
        # exact-integer bound is nmant + 1 (float64 2**53, float32 2**24).
        if raw.dtype.kind == "f":
            fidelity = 2.0 ** (np.finfo(raw.dtype).nmant + 1)
        else:
            fidelity = 2.0**53
        for name, col in (("period", periods), ("white", white), ("black", black)):
            if np.any(col >= fidelity):
                raise ValueError(
                    f"stephenson_rating: {name} labels at or above {int(fidelity)} are not "
                    f"reliably representable in {raw.dtype if raw.dtype.kind == 'f' else 'float64'}; "
                    "pass games as an integer array"
                )
        periods_u64 = periods.astype(np.uint64)
        white_u64 = white.astype(np.uint64)
        black_u64 = black.astype(np.uint64)
    if np.any(white < 0) or np.any(black < 0):
        raise ValueError("stephenson_rating: player indices must be nonnegative")
    try:
        n = int(n_players)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"stephenson_rating: n_players is not an integer: {exc}"
        ) from None
    if n != n_players:
        raise ValueError("stephenson_rating: n_players must be an integer")
    # Enforce the core's cap BEFORE any length-n allocation below.
    if n < 2:
        raise ValueError("stephenson_rating: at least two players are required")
    if n > 10_000:
        raise ValueError(
            f"stephenson_rating: n = {n} exceeds the supported cap of 10000"
        )
    g = arr.shape[0]
    if gamma is None:
        gamma_arr = np.zeros(g)
    else:
        if np.iscomplexobj(np.asarray(gamma)):
            raise ValueError("stephenson_rating: gamma must be real, not complex")
        try:
            gamma_arr = np.asarray(gamma, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"stephenson_rating: gamma is not numeric: {exc}"
            ) from None
        if gamma_arr.ndim == 0:
            gamma_arr = np.full(g, float(gamma_arr))
        elif gamma_arr.shape != (g,):
            raise ValueError(
                f"stephenson_rating: gamma must be a scalar or length-{g} array, got {gamma_arr.shape}"
            )
    try:
        init_r_in, init_d_in = init
    except (TypeError, ValueError):
        raise ValueError(
            "stephenson_rating: init must be a (rating, deviation) pair"
        ) from None
    if np.iscomplexobj(np.asarray(init_r_in)) or np.iscomplexobj(np.asarray(init_d_in)):
        raise ValueError("stephenson_rating: init must be real, not complex")
    try:
        init_r = np.asarray(init_r_in, dtype=float)
        init_d = np.asarray(init_d_in, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"stephenson_rating: init is not numeric: {exc}") from None
    if init_r.ndim == 0 and init_d.ndim == 0:
        init_r = np.full(n, float(init_r))
        init_d = np.full(n, float(init_d))
    elif init_r.shape != (n,) or init_d.shape != (n,):
        raise ValueError(
            "stephenson_rating: init must be a scalar pair or a pair of "
            f"length-{n} arrays, got shapes {init_r.shape} and {init_d.shape}"
        )

    def _count_vec(name, val):
        if val is None:
            return np.zeros(n, dtype=np.uint64)
        if np.iscomplexobj(np.asarray(val)):
            raise ValueError(f"stephenson_rating: {name} must be real, not complex")
        try:
            v = np.asarray(val, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"stephenson_rating: {name} is not numeric: {exc}"
            ) from None
        if v.shape != (n,):
            raise ValueError(
                f"stephenson_rating: {name} must be a length-{n} array, got {v.shape}"
            )
        if not np.all(np.isfinite(v)) or np.any(v < 0) or np.any(v != np.floor(v)):
            raise ValueError(
                f"stephenson_rating: {name} must be nonnegative integers"
            )
        if np.any(v >= 2.0**53):
            raise ValueError(
                f"stephenson_rating: {name} values at or above 2**53 are not "
                "reliably representable; pass smaller counts"
            )
        return v.astype(np.uint64)

    init_g = _count_vec("init_games", init_games)
    init_l = _count_vec("init_lag", init_lag)
    try:
        cval_f = float(cval)
        hval_f = float(hval)
        bval_f = float(bval)
        lambda_f = float(lambda_)
        rdmax_f = float(rdmax)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"stephenson_rating: cval/hval/bval/lambda_/rdmax is not numeric: {exc}"
        ) from None
    res = _core_module().stephenson_rating(
        np.ascontiguousarray(periods_u64),
        np.ascontiguousarray(white_u64),
        np.ascontiguousarray(black_u64),
        np.ascontiguousarray(score),
        np.ascontiguousarray(gamma_arr),
        np.ascontiguousarray(init_r),
        np.ascontiguousarray(init_d),
        np.ascontiguousarray(init_g),
        np.ascontiguousarray(init_l),
        cval_f,
        hval_f,
        bval_f,
        lambda_f,
        rdmax_f,
    )
    return StephensonResult(
        ratings=np.asarray(res["ratings"]),
        deviations=np.asarray(res["deviations"]),
        games=np.asarray(res["games"]),
        wins=np.asarray(res["wins"]),
        draws=np.asarray(res["draws"]),
        losses=np.asarray(res["losses"]),
        lag=np.asarray(res["lag"]),
    )
