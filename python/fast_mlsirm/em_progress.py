"""Shared EM progress report for long Bock–Aitkin MML fits.

The observed-data marginal log-likelihood is already evaluated each E-step
for convergence monitoring; exporting it requires no extra quadrature
(Bock & Aitkin, 1981, pp. 445, 447–448).

References (APA 7th ed.)
------------------------
Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of
    item parameters: Application of an EM algorithm. *Psychometrika, 46*(4),
    443–459. https://doi.org/10.1007/BF02293801 (full text read: p. 445
    eqs. 5–6 define the marginal log-likelihood; p. 447 E-step recomputes
    pattern marginals each cycle; p. 448 notes the procedure satisfies the
    marginal likelihood equations)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmIterationProgress:
    """One EM E-step's already-computed observed-data marginal log-likelihood.

    ``iteration`` is the 0-based index of completed E-step evaluations within
    the current multi-start run (matches ``loglik_trace[iteration]``).
    ``delta_loglik`` is ``None`` on the first evaluation of a start and
    ``loglik - previous`` thereafter — the same change used for the
    tolerance check. ``start`` is the multi-start index in ``0..n_starts-1``.
    """

    iteration: int
    loglik: float
    delta_loglik: float | None
    start: int
