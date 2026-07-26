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