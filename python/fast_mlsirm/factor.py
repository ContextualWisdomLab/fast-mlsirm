"""Minres (ULS) exploratory factor analysis and McDonald's omega_total for
the unidimensional case. All numeric work happens in the Rust core
(``mlsirm_core::factor``); this module only validates and marshals.

Source status: the minres algorithm is a line-by-line transcription of the
CRAN psych package's ``fa.R`` (Revelle, 2025 — READ: ``fit.residuals``,
``fit``, ``FAgr.minres``, ``FAout.wls``, smc start values). McDonald (1999)
was NOT read; the omega_total formula is hand-derived from the standardized
1-factor model (derivation in the Rust module docs) and matches what
secondary sources attribute to McDonald. Tests pin parity against an
independent scipy L-BFGS-B transcription oracle (same optimizer family as
R's ``optim``; not claimed bit-identical to any R run).

REDUCED SCOPE (spec decision): no rotation (loadings are unrotated), no
Schmid-Leiman, no omega_hierarchical, no ML/WLS/GLS methods, no factor
scores. The public omega name is ``omega_total_1f`` to make the 1-factor
restriction explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_REFERENCES = """References (APA 7th ed.):
        Revelle, W. (2025). *psych: Procedures for psychological,
            psychometric, and personality research* (Version 2.6.5)
            [R package]. https://CRAN.R-project.org/package=psych
        McDonald, R. P. (1999). *Test theory: A unified treatment*.
            Erlbaum. (As cited in Revelle, 2025; not read.)
    """


@dataclass
class MinresFaResult:
    """Minres factor-analysis output.

    ``loadings`` is a ``(p, n_factors)`` array (unrotated, columns in
    descending-eigenvalue order, column sums >= 0). ``kkt_violation`` is
    the maximum finite-difference box-KKT violation of the minres
    objective at the solution; ``converged`` means it is below the crate
    tolerance (1e-6)."""

    loadings: np.ndarray
    uniquenesses: np.ndarray
    communalities: np.ndarray
    objective: float
    kkt_violation: float
    n_iter: int
    converged: bool


@dataclass
class OmegaResult:
    """McDonald's omega_total from a 1-factor minres fit:
    ``(sum lambda)^2 / ((sum lambda)^2 + sum psi)``."""

    omega_total: float
    fa: MinresFaResult


def _fa_from_dict(d: dict, p: int, n_factors: int) -> MinresFaResult:
    return MinresFaResult(
        loadings=np.asarray(d["loadings"], dtype=np.float64).reshape(p, n_factors),
        uniquenesses=np.asarray(d["uniquenesses"], dtype=np.float64),
        communalities=np.asarray(d["communalities"], dtype=np.float64),
        objective=float(d["objective"]),
        kkt_violation=float(d["kkt_violation"]),
        n_iter=int(d["n_iter"]),
        converged=bool(d["converged"]),
    )


def minres_fa(corr: np.ndarray, n_factors: int) -> MinresFaResult:
    """Minres factor analysis of a ``(p, p)`` correlation matrix
    (psych fa.R transcription; Revelle, 2025).

    %s""" % _REFERENCES
    from . import _core

    r = np.ascontiguousarray(np.asarray(corr, dtype=np.float64))
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError("corr must be a square (p, p) matrix")
    p = int(r.shape[0])
    out = _core.minres_fa(r.reshape(-1), p, int(n_factors))
    return _fa_from_dict(out, p, int(n_factors))


def minres_fa_from_data(data: np.ndarray, n_factors: int) -> MinresFaResult:
    """:func:`minres_fa` from a complete ``(n, p)`` data matrix (Pearson
    correlations computed in the Rust core).

    %s""" % _REFERENCES
    from . import _core

    x = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    if x.ndim != 2:
        raise ValueError("data must be a 2-D (n, p) matrix")
    n, p = map(int, x.shape)
    out = _core.minres_fa_from_data(x.reshape(-1), n, p, int(n_factors))
    return _fa_from_dict(out, p, int(n_factors))


def omega_total_1f(corr: np.ndarray) -> OmegaResult:
    """McDonald's omega_total for the unidimensional case from a ``(p, p)``
    correlation matrix (1-factor minres fit; McDonald, 1999, as cited in
    Revelle, 2025 — formula hand-derived, see Rust module docs).

    %s""" % _REFERENCES
    from . import _core

    r = np.ascontiguousarray(np.asarray(corr, dtype=np.float64))
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError("corr must be a square (p, p) matrix")
    p = int(r.shape[0])
    out = _core.omega_total_1f(r.reshape(-1), p)
    return OmegaResult(
        omega_total=float(out["omega_total"]), fa=_fa_from_dict(out["fa"], p, 1)
    )


def omega_total_1f_from_data(data: np.ndarray) -> OmegaResult:
    """:func:`omega_total_1f` from a complete ``(n, p)`` data matrix.

    %s""" % _REFERENCES
    from . import _core

    x = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    if x.ndim != 2:
        raise ValueError("data must be a 2-D (n, p) matrix")
    n, p = map(int, x.shape)
    out = _core.omega_total_1f_from_data(x.reshape(-1), n, p)
    return OmegaResult(
        omega_total=float(out["omega_total"]), fa=_fa_from_dict(out["fa"], p, 1)
    )

@dataclass
class GlbFaResult:
    """Factor-analytic greatest lower bound to reliability (psych glb.fa
    transcription). ``nf`` is the fitted factor count after psych's df
    adjustment; ``communalities`` come from that nf-factor minres fit.

    REDUCED SCOPE: this is NOT the algebraic glb of Jackson & Agunwamba
    (as computed by psych ``glb.algebraic``, which needs an SDP solver);
    it is the factor-analytic approximation from psych ``glbs.R``
    (Revelle, 2025 — READ). Sijtsma (2009) was NOT read; cite the psych
    transcription only. Correlation-matrix or complete-data input only
    (no cov2cor / pairwise-deletion branches)."""

    glb: float
    communalities: np.ndarray
    nf: int


def _glbfa_from_dict(d: dict) -> GlbFaResult:
    return GlbFaResult(
        glb=float(d["glb"]),
        communalities=np.asarray(d["communalities"], dtype=np.float64),
        nf=int(d["nf"]),
    )


def glb_fa(corr: np.ndarray) -> GlbFaResult:
    """Factor-analytic glb from a ``(p, p)`` correlation matrix (psych
    ``glb.fa`` transcription; Revelle, 2025). See :class:`GlbFaResult`
    for the scope reduction (not the algebraic glb).

    %s""" % _REFERENCES
    from . import _core

    r = np.ascontiguousarray(np.asarray(corr, dtype=np.float64))
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError("corr must be a square (p, p) matrix")
    return _glbfa_from_dict(_core.glb_fa(r.reshape(-1), int(r.shape[0])))


def glb_fa_from_data(data: np.ndarray) -> GlbFaResult:
    """:func:`glb_fa` from a complete ``(n, p)`` data matrix.

    %s""" % _REFERENCES
    from . import _core

    x = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    if x.ndim != 2:
        raise ValueError("data must be a 2-D (n, p) matrix")
    n, p = map(int, x.shape)
    return _glbfa_from_dict(_core.glb_fa_from_data(x.reshape(-1), n, p))
