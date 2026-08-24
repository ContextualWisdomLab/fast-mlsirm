"""Classical selection utility analysis (Taylor-Russell, Naylor-Shine,
Brogden-Cronbach-Gleser). All numeric work happens in the Rust core
(``mlsirm_core::utility``); this module only validates and marshals.

Source status: the formulas are a transcription of the CRAN iopsych 0.90.1
package (Goebl, Jones, & Beatty — READ: ``R/utility.R`` ``utilityBcg``/``trModel``/
``.utilitySwitch`` and ``R/ai.R`` ``ux``), independently re-derived from the
standard bivariate-normal selection model and verified against a scipy
oracle (``scipy.stats.norm``, ``scipy.stats.multivariate_normal``). The
original Taylor & Russell (1939), Naylor & Shine (1965), and Cronbach &
Gleser (1965) sources were NOT read; they are cited as the models'
origins as attributed by iopsych, and every implemented equation was
verified numerically rather than taken on faith from the citations.

REDUCED SCOPE (spec decision): only the three classical models. No
Boudreau (1983) financial extension, no Raju-Burke-Normand (1990), no
multiple-hurdle or banded selection. ``|rxy| >= 1`` is rejected.

References (APA 7th ed.):
    Cronbach, L. J., & Gleser, G. C. (1965). *Psychological tests and
        personnel decisions* (2nd ed.). University of Illinois Press.
        (As cited in iopsych; not read.)
    Goebl, A., Jones, J., & Beatty, A. (2016). *iopsych: Methods for industrial/
        organizational psychology* (Version 0.90.1) [R package].
        https://CRAN.R-project.org/package=iopsych
    Naylor, J. C., & Shine, L. C. (1965). A table for determining the
        increase in mean criterion score obtained by using a selection
        device. *Journal of Industrial Psychology, 3*(2), 33-42.
        (As cited in iopsych; not read.)
    Taylor, H. C., & Russell, J. T. (1939). The relationship of validity
        coefficients to the practical effectiveness of tests in selection:
        Discussion and tables. *Journal of Applied Psychology, 23*(5),
        565-578. https://doi.org/10.1037/h0057079 (As cited in iopsych;
        not read.)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SelectionUtilityResult:
    """Brogden-Cronbach-Gleser / Naylor-Shine utility output.

    ``xc`` is the standard-normal predictor cutoff ``Phi^-1(1 - sr)``;
    ``ux`` the selection intensity ``phi(xc)/sr`` (mean standardized
    predictor of those selected); ``pux = rxy * ux`` the Naylor-Shine mean
    standardized criterion of those selected; ``utility_gain`` the BCG
    gain ``n * period * sdy * pux - cost_total``."""

    xc: float
    ux: float
    pux: float
    utility_gain: float


@dataclass
class TaylorRussellResult:
    """Taylor-Russell (1939) output: ``success_ratio`` is
    ``P(Y > yc | X > xc)`` — the proportion of selected applicants who
    succeed — under the standard bivariate-normal model; ``q_joint`` is
    the joint tail ``P(X > xc, Y > yc)``."""

    success_ratio: float
    base_rate: float
    q_joint: float


def selection_utility(
    n: float,
    sdy: float,
    rxy: float,
    sr: float,
    cost_total: float = 0.0,
    period: float = 1.0,
) -> SelectionUtilityResult:
    """Brogden-Cronbach-Gleser utility of a top-down selection system.

    Parameters mirror iopsych ``utilityBcg``: ``n`` selectees per period
    (>= 1), ``sdy`` monetary SD of job performance (>= 0), ``rxy``
    validity coefficient in (-1, 1), ``sr`` selection ratio in (0, 1),
    ``cost_total`` TOTAL selection cost (iopsych labels its ``cost`` "per
    applicant" but never multiplies by ``n``; we document the actual
    semantics), ``period`` expected tenure (>= 1).
    """
    from . import _core

    r = _core.selection_utility(
        float(n), float(sdy), float(rxy), float(sr), float(cost_total), float(period)
    )
    return SelectionUtilityResult(
        xc=r["xc"], ux=r["ux"], pux=r["pux"], utility_gain=r["utility_gain"]
    )


def taylor_russell(rxy: float, sr: float, br: float) -> TaylorRussellResult:
    """Taylor-Russell (1939) success ratio for a dichotomous criterion.

    ``rxy`` validity in (-1, 1), ``sr`` selection ratio in (0, 1), ``br``
    base rate of success in (0, 1). At ``rxy = 0`` the success ratio
    equals ``br`` (no selection information).
    """
    from . import _core

    r = _core.taylor_russell(float(rxy), float(sr), float(br))
    return TaylorRussellResult(
        success_ratio=r["success_ratio"],
        base_rate=r["base_rate"],
        q_joint=r["q_joint"],
    )
