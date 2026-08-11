"""Nonparametric person-fit statistics. All numeric work happens in the
Rust core; this module only validates and marshals arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PersonFitNpResult:
    """Per-person nonparametric person-fit statistics (PerFit port).

    ``g`` Guttman error counts; ``gnormed`` normed Guttman errors;
    ``nci`` norm conformity index (``1 - 2*gnormed`` for non-perfect
    rows; 0 for all-0s/all-1s rows, source-faithful); ``u3`` and ``zu3``
    van der Flier's U3 and standardized U3; ``c_sato`` Sato's caution
    index; ``cstar`` the modified caution index. ``u3``, ``zu3``,
    ``c_sato``, and ``cstar`` are NaN for perfect (all-0s/all-1s) rows
    and for degenerate data (e.g. all item proportions equal), mirroring
    the R package's NA semantics."""

    g: np.ndarray
    gnormed: np.ndarray
    nci: np.ndarray
    u3: np.ndarray
    zu3: np.ndarray
    c_sato: np.ndarray
    cstar: np.ndarray


def person_fit_np(x) -> PersonFitNpResult:
    """Nonparametric person-fit statistics
    (``mlsirm_core::personfit_np::person_fit_np``), a complete-data port
    of seven dichotomous statistics as IMPLEMENTED by the CRAN PerFit R
    package (READ: ``R/G.R``, ``R/Gnormed.R``, ``R/NCI.R``, ``R/U3.R``,
    ``R/ZU3.R``, ``R/C.Sato.R``, ``R/Cstar.R`` at cran/PerFit commit
    c9df433). NOT READ (cited only as referenced by the PerFit sources):
    van der Flier (1977, 1982); Meijer (1994); Tatsuoka & Tatsuoka
    (1982, 1983); Sato (1975); Harnisch & Linn (1981).

    ``x`` is a 2-D person-by-item matrix of exactly 0s and 1s. Missing
    data is out of scope (PerFit's imputation paths were not ported):
    any other value, including NaN, raises ``ValueError``. Higher G,
    Gnormed, U3, ZU3, C, and C* mean MORE aberrant responding; higher
    NCI means more norm-conforming.

    In LLM-as-a-Judge quality management these flag judges whose
    pass/fail patterns invert the panel-wide item difficulty ordering
    (e.g. failing easy checks while passing hard ones) without needing a
    fitted IRT model.

    References
    ----------
    Tendeiro, J. N., Meijer, R. R., & Niessen, A. S. M. (2016). PerFit:
    An R package for person-fit analysis in IRT. *Journal of
    Statistical Software, 74*(5), 1-27.
    https://doi.org/10.18637/jss.v074.i05 (package paper; the R sources
    listed above were READ and ported.)
    """
    xa = np.asarray(x)
    if xa.ndim != 2:
        raise ValueError("x must be a 2-D person-by-item matrix")
    n, ni = xa.shape
    if n < 1:
        raise ValueError("x must have at least 1 person")
    if ni < 2:
        raise ValueError("x must have at least 2 items")
    if np.iscomplexobj(xa):
        raise ValueError("x must be real-valued")
    if xa.dtype.kind == "b":
        xa = xa.astype(np.float64)
    if xa.dtype.kind not in ("i", "u", "f"):
        raise ValueError("x must be a numeric array")
    xf = np.ascontiguousarray(xa, dtype=np.float64)
    bad = ~((xf == 0.0) | (xf == 1.0))
    if bad.any():
        p, i = np.argwhere(bad)[0]
        raise ValueError(
            f"x[{p}, {i}]: responses must be exactly 0 or 1 "
            "(missing data is out of scope)"
        )

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "py_person_fit_np"):
        raise RuntimeError("Rust core with py_person_fit_np is required")
    res = core.py_person_fit_np(xf.ravel(), int(n), int(ni))
    return PersonFitNpResult(
        g=np.asarray(res["g"], dtype=np.float64),
        gnormed=np.asarray(res["gnormed"], dtype=np.float64),
        nci=np.asarray(res["nci"], dtype=np.float64),
        u3=np.asarray(res["u3"], dtype=np.float64),
        zu3=np.asarray(res["zu3"], dtype=np.float64),
        c_sato=np.asarray(res["c_sato"], dtype=np.float64),
        cstar=np.asarray(res["cstar"], dtype=np.float64),
    )
