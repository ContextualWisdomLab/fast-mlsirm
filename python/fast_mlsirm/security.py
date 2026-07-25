"""Wollack-style omega answer-copying statistic. All numeric work happens in
the Rust core; this module only validates and marshals arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WollackOmegaResult:
    """Omega answer-copying statistic for one (copier, source) pair.

    ``observed_matches`` is the count of identical responses; under no
    copying ``omega`` is approximately standard normal, and ``p_value`` is
    the one-sided upper-tail probability (small values suggest copying)."""

    observed_matches: int
    expected_matches: float
    variance: float
    omega: float
    p_value: float


def _index_vector(arr: np.ndarray, name: str, n_options: int) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if a.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if np.iscomplexobj(a):
        raise ValueError(f"{name} must be real-valued")
    if a.dtype.kind not in ("b", "i", "u", "f"):
        raise ValueError(f"{name} must be numeric")
    af = a.astype(np.float64)
    if not np.all(np.isfinite(af)):
        raise ValueError(f"{name} must be finite")
    ai = af.astype(np.int64)
    if not np.array_equal(af, ai.astype(np.float64)):
        raise ValueError(f"{name} entries must be integers")
    if np.any(ai < 0) or np.any(ai >= n_options):
        raise ValueError(f"{name} entries must be option indices in [0, {n_options})")
    return ai.astype(np.uint64)


def wollack_omega(
    copier: np.ndarray,
    source: np.ndarray,
    probs: np.ndarray,
    n_options: int,
) -> WollackOmegaResult:
    """Omega answer-copying statistic (compute in Rust; Wollack, 1997, as
    implemented by the CRAN CopyDetect and aberrance packages).

    ``h`` counts items where copier and source chose the same option;
    ``p_i = probs[i, source_i]`` is the COPIER's model-implied probability of
    the SOURCE's observed option; ``omega = (h - sum p_i) /
    sqrt(sum p_i (1 - p_i))`` with a one-sided upper-tail normal p-value.
    Formula verified against two READ implementations: the CopyDetect R
    sources (``similarity1.r``/``similarity2.r``) and the independent
    aberrance package (``compute_OMG``). NOT READ: Wollack (1997, *Applied
    Psychological Measurement, 21*(4), 307-320) itself (access blocked); it
    is cited only as implemented by those sources. CopyDetect's printed
    docs flip the sign, but both source files use ``(h - E)/sqrt(V)``; the
    source convention is implemented. No continuity correction, no missing
    responses, no g2/GBT/K-index; the caller supplies the copier's fitted
    option probabilities (e.g. from a nominal response model).

    In LLM-as-a-Judge quality management this flags judge pairs whose
    agreement exceeds what one judge's response model predicts (e.g. one
    model plagiarizing another's outputs).

    ``probs`` is an ``n_items x n_options`` matrix of the copier's option
    probabilities (rows sum to 1); ``copier``/``source`` are observed
    option indices in ``[0, n_options)``.

    References
    ----------
    Wollack, J. A. (1997). A nominal response model approach for detecting
    answer copying. *Applied Psychological Measurement, 21*(4), 307-320.
    (NOT READ; cited as implemented by CopyDetect and aberrance.)
    Zopluoglu, C. (2018). *CopyDetect: Computing response similarity indices
    for multiple-choice tests* (R package). (READ: R sources.)
    Man, K., & Harring, J. R. (2023). *aberrance* (R package, compute_OMG).
    (READ: R sources.)
    """
    if not isinstance(n_options, (int, np.integer)) or isinstance(n_options, bool):
        raise ValueError("n_options must be an integer")
    if n_options <= 0:
        raise ValueError("n_options must be positive")
    n_options = int(n_options)

    c = _index_vector(copier, "copier", n_options)
    s = _index_vector(source, "source", n_options)
    if c.size != s.size:
        raise ValueError("copier and source must have the same length")

    p = np.asarray(probs)
    if np.iscomplexobj(p):
        raise ValueError("probs must be real-valued")
    if p.dtype.kind not in ("b", "i", "u", "f"):
        raise ValueError("probs must be numeric")
    if p.ndim == 2:
        if p.shape != (c.size, n_options):
            raise ValueError(
                f"probs must have shape ({c.size}, {n_options}), got {p.shape}"
            )
        p = np.ascontiguousarray(p, dtype=np.float64).ravel()
    elif p.ndim == 1:
        if p.size != c.size * n_options:
            raise ValueError(
                "flattened probs must have length n_items * n_options"
            )
        p = np.ascontiguousarray(p, dtype=np.float64)
    else:
        raise ValueError("probs must be a 2-D matrix or flattened 1-D array")

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "py_wollack_omega"):
        raise RuntimeError("wollack_omega requires the compiled Rust core")
    res = core.py_wollack_omega(list(map(int, c)), list(map(int, s)), p, n_options)
    return WollackOmegaResult(
        observed_matches=int(res["observed_matches"]),
        expected_matches=float(res["expected_matches"]),
        variance=float(res["variance"]),
        omega=float(res["omega"]),
        p_value=float(res["p_value"]),
    )