"""Generalizability theory G/D-study analyses for crossed designs
(Huebner & Lucht, 2019). All numeric work happens in the Rust core
(``mlsirm_core::gtheory``); this module only validates and marshals.

Source status: Huebner & Lucht (2019) READ in full, including the worked
p x i and p x i x o examples (Tables 3-6) that the Rust tests reproduce.
Brennan (2001) and Shavelson & Webb (1991) are cited by that paper for the
EMS derivations and were NOT read; the EMS-to-variance-component inversions
are hand-derived and numerically verified against the paper's published
tables (see the Rust module docs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

_REFERENCES = """References (APA 7th ed.):
        Huebner, A., & Lucht, M. (2019). Generalizability theory in R.
            *Practical Assessment, Research, and Evaluation, 24*, Article 5.
            https://doi.org/10.7275/5065-gc10
        Brennan, R. L. (2001). *Generalizability theory*. Springer.
            (As cited in Huebner & Lucht, 2019; not read.)
        Shavelson, R. J., & Webb, N. M. (1991). *Generalizability theory:
            A primer*. Sage. (As cited in Huebner & Lucht, 2019; not read.)
    """


@dataclass
class GTheoryDStudyRow:
    """One D-study column: proposed facet sizes with the resulting error
    variances and coefficients (Huebner & Lucht, 2019, Tables 4 and 6).
    ``n_o_prime`` is 1 and unused for the one-facet design."""

    n_i_prime: int
    n_o_prime: int
    rel_error_var: float
    abs_error_var: float
    generalizability: float
    dependability: float


@dataclass
class GTheoryResult:
    """G-study ANOVA table plus D-study rows.

    Component order is ``(p, i, pi)`` for the one-facet design and
    ``(p, i, o, pi, po, io, pio)`` for the two-facet design. ``var_raw``
    holds the raw ANOVA estimates (may be negative); ``var`` is the
    component-wise ``max(., 0)`` used for all D-study quantities
    (clamped-ANOVA policy — an implementation choice, not a
    paper-prescribed estimator)."""

    df: list[float]
    ss: list[float]
    ms: list[float]
    var_raw: list[float]
    var: list[float]
    d_study: list[GTheoryDStudyRow]


def _to_result(res: dict) -> GTheoryResult:
    return GTheoryResult(
        df=[float(v) for v in res["df"]],
        ss=[float(v) for v in res["ss"]],
        ms=[float(v) for v in res["ms"]],
        var_raw=[float(v) for v in res["var_raw"]],
        var=[float(v) for v in res["var"]],
        d_study=[
            GTheoryDStudyRow(
                n_i_prime=int(r["n_i_prime"]),
                n_o_prime=int(r["n_o_prime"]),
                rel_error_var=float(r["rel_error_var"]),
                abs_error_var=float(r["abs_error_var"]),
                generalizability=float(r["generalizability"]),
                dependability=float(r["dependability"]),
            )
            for r in res["d_study"]
        ],
    )


def _core_or_raise(name: str):
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, name):
        raise RuntimeError(f"{name} requires the compiled Rust core")
    return core


def gtheory_pi(
    data: np.ndarray,
    n_i_prime: Sequence[int] = (5, 10, 15, 20),
) -> GTheoryResult:
    """One-facet crossed ``p x i`` generalizability analysis (compute in
    Rust; Huebner & Lucht, 2019, "One-facet crossed design" section and
    Tables 3-4).

    ``data`` is a complete, balanced ``n_persons x n_items`` score array.
    Variance components are the ANOVA (EMS) estimators; negative raw
    estimates are reported in ``var_raw`` and clamped to zero in ``var``
    for the D study. D-study rows give sigma^2(delta), sigma^2(Delta), the
    generalizability coefficient E-rho^2 (eq. 6) and the dependability
    index Phi (eq. 7) at each proposed ``n_i'``; coefficients are NaN when
    their denominator is <= 1e-12. In LLM-as-a-Judge quality management
    this asks how many judge items are needed for a dependable rating.

    """ + _REFERENCES
    core = _core_or_raise("gtheory_pi")
    x = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    if x.ndim != 2:
        raise ValueError("data must be a 2-D persons x items array")
    n_p, n_i = x.shape
    primes = [int(v) for v in n_i_prime]
    return _to_result(core.gtheory_pi(x.reshape(-1), int(n_p), int(n_i), primes))


def gtheory_pio(
    data: np.ndarray,
    n_prime: Sequence[tuple[int, int]] = ((5, 2), (10, 2), (15, 2), (20, 2)),
) -> GTheoryResult:
    """Two-facet crossed ``p x i x o`` generalizability analysis (compute
    in Rust; Huebner & Lucht, 2019, "Two-facet crossed design" section and
    Tables 5-6).

    ``data`` is a complete, balanced ``n_persons x n_items x n_occasions``
    score array. Component order everywhere is
    ``(p, i, o, pi, po, io, pio)``. ``n_prime`` lists proposed
    ``(n_i', n_o')`` D-study pairs; the clamped-ANOVA and NaN-denominator
    policies match :func:`gtheory_pi`.

    """ + _REFERENCES
    core = _core_or_raise("gtheory_pio")
    x = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    if x.ndim != 3:
        raise ValueError("data must be a 3-D persons x items x occasions array")
    n_p, n_i, n_o = x.shape
    pairs = [(int(a), int(b)) for a, b in n_prime]
    return _to_result(
        core.gtheory_pio(x.reshape(-1), int(n_p), int(n_i), int(n_o), pairs)
    )
