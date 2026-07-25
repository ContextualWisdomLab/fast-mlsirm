"""Kernel-smoothing nonparametric IRT: option characteristic curves by
Nadaraya-Watson regression on rank-based ordinal ability estimates (Ramsay,
1991, as cited in Mazza et al., 2014). All numeric work happens in the Rust
core; this module only validates and marshals arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KsirtResult:
    """Kernel-smoothed option characteristic curves.

    ``theta`` are the rank-based ordinal ability estimates
    ``Phi^-1(rank(total_i)/(n+1))`` in subject order; ``grid`` the
    equally-spaced evaluation points; ``bandwidth`` the per-item bandwidths
    used. ``options[j]`` lists item ``j``'s distinct observed scores
    (ascending), ``occ[j]`` is the matching ``m_j x len(grid)`` option
    characteristic curve matrix, ``expected[j]`` the expected item score
    curve, and ``expected_total`` their sum over items."""

    theta: np.ndarray
    grid: np.ndarray
    bandwidth: np.ndarray
    options: list[np.ndarray]
    occ: list[np.ndarray]
    expected: list[np.ndarray]
    expected_total: np.ndarray


def ksirt_analysis(
    responses: np.ndarray,
    kernel: str = "gaussian",
    nevalpoints: int = 51,
    bandwidth: np.ndarray | None = None,
) -> KsirtResult:
    """Kernel smoothing of option characteristic curves (compute in Rust;
    Ramsay, 1991, as cited in Mazza et al., 2014).

    Estimates each item's option characteristic curves by Nadaraya-Watson
    kernel regression of the option indicators on ordinal ability estimates
    ``Phi^-1(rank(total score)/(n+1))`` (ties broken by subject order),
    evaluated on an equally-spaced grid from ``Phi^-1(1/(n+1))`` to
    ``Phi^-1(n/(n+1))``. The default bandwidth is Silverman's rule
    ``1.06 * n^(-1/5)`` on the standard-normal ability metric. Formulas
    follow Mazza et al. (2014, Sections 2-2.3) and the KernSmoothIRT R
    package source (both read); Ramsay (1991) itself is cited only through
    Mazza et al. (2014). Standard errors and cross-validation bandwidths
    are deliberately not implemented (see the Rust module docs).

    In LLM-as-a-Judge item-quality management, nonparametric OCCs reveal
    non-monotone or poorly discriminating evaluation items without assuming
    a parametric response model.

    ``responses`` is a complete ``persons x items`` array of pre-scored
    numeric responses; each column's distinct values form that item's
    options. ``kernel`` is ``"gaussian"``, ``"quadratic"``, or
    ``"uniform"``. ``bandwidth`` optionally gives one positive value per
    item.

    References (APA 7th ed.):
        Mazza, A., Punzo, A., & McGuire, B. (2014). KernSmoothIRT: An R
            package for kernel smoothing in item response theory. *Journal
            of Statistical Software, 58*(6), 1-34.
            https://doi.org/10.18637/jss.v058.i06
        Ramsay, J. O. (1991). Kernel smoothing approaches to nonparametric
            item characteristic curve estimation. *Psychometrika, 56*(4),
            611-630. https://doi.org/10.1007/BF02294494 (as cited in Mazza
            et al., 2014)
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "ksirt_occ"):
        raise RuntimeError("ksirt_analysis requires the compiled Rust core")

    if kernel not in ("gaussian", "quadratic", "uniform"):
        raise ValueError("kernel must be gaussian, quadratic, or uniform")
    nevalpoints = int(nevalpoints)
    if nevalpoints < 2:
        raise ValueError("nevalpoints must be at least 2")
    if nevalpoints > 100_000:
        # trust boundary: nevalpoints drives Rust-side allocations
        raise ValueError("nevalpoints must be at most 100000")

    y = np.asarray(responses, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    if n_persons < 2 or n_items < 1:
        raise ValueError("responses needs at least 2 persons and 1 item")
    if not np.all(np.isfinite(y)):
        raise ValueError("responses must be complete (no missing values)")

    bw = None
    if bandwidth is not None:
        bw_arr = np.asarray(bandwidth, dtype=np.float64).reshape(-1)
        if bw_arr.shape[0] != n_items:
            raise ValueError("bandwidth must supply one value per item")
        if not np.all(np.isfinite(bw_arr)) or np.any(bw_arr <= 0.0):
            raise ValueError("bandwidths must be finite and positive")
        bw = [float(v) for v in bw_arr]

    res = core.ksirt_occ(
        y.reshape(-1), int(n_persons), int(n_items), kernel, nevalpoints, bw
    )
    grid = np.asarray(res["grid"], dtype=np.float64)
    q = grid.shape[0]
    options = [np.asarray(o, dtype=np.float64) for o in res["options"]]
    occ = [
        np.asarray(flat, dtype=np.float64).reshape(len(opts), q)
        for flat, opts in zip(res["occ"], options)
    ]
    return KsirtResult(
        theta=np.asarray(res["theta"], dtype=np.float64),
        grid=grid,
        bandwidth=np.asarray(res["bandwidth"], dtype=np.float64),
        options=options,
        occ=occ,
        expected=[np.asarray(e, dtype=np.float64) for e in res["expected"]],
        expected_total=np.asarray(res["expected_total"], dtype=np.float64),
    )
