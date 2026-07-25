"""Guttman lambda reliability coefficients (Guttman, 1945, as implemented by
CRAN psych 2.6.5; Revelle, 2025). All numeric work happens in the Rust core;
this module only validates and marshals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GuttmanResult:
    """Guttman lambda coefficients plus split-half summaries.

    ``lambda4`` is the best (maximum) split-half over the enumerated or
    sampled splits, ``beta`` the worst (minimum, floored at 0), and
    ``mean_split`` the average; ``n_splits`` reports how many splits were
    evaluated and ``exhaustive`` whether all C(p, floor(p/2)) subsets were
    enumerated."""

    lambda1: float
    lambda2: float
    lambda3: float
    lambda4: float
    lambda5: float
    lambda6: float
    beta: float
    mean_split: float
    n_splits: int
    exhaustive: bool


_REFERENCES = """References (APA 7th ed.):
        Guttman, L. (1945). A basis for analyzing test-retest reliability.
            *Psychometrika, 10*(4), 255-282.
            https://doi.org/10.1007/BF02288892 (as cited in Revelle, 2025)
        Revelle, W. (2025). *psych: Procedures for psychological,
            psychometric, and personality research* (Version 2.6.5)
            [R package]. https://CRAN.R-project.org/package=psych
    """


def guttman_lambdas(
    data: np.ndarray,
    n_sample_splits: int = 15000,
    seed: int = 1,
) -> GuttmanResult:
    """Guttman's lambda reliability coefficients (compute in Rust; algorithm
    transcribed from the psych 2.6.5 R sources ``guttman.R``, ``splitHalf.R``
    and ``smc.R``, read line by line; Guttman, 1945, not read — attribution
    as cited in Revelle, 2025).

    Computed on the Pearson correlation matrix of ``data`` (an
    ``n_persons x n_items`` array, complete and finite): lambda1-lambda3
    (lambda3 is coefficient alpha), lambda5 (best single covariance column),
    lambda6 (squared multiple correlations), and split-half summaries
    lambda4 (best split), beta (worst split), and the mean split. All
    ``floor(p/2)``-subsets are enumerated when their count fits within
    ``n_sample_splits`` (psych's brute-force cutoff is 15000); otherwise
    that many splits are sampled with this crate's deterministic LCG, so
    sampled results are psych-inspired but not bit-identical to any R run.
    Declared divergences from psych (documented in the Rust module): no
    ``check.keys`` auto-reversal, absolute split-half correlations in both
    branches, plain matrix inverse with an error on singular correlation
    matrices instead of a pseudoinverse. In LLM-as-a-Judge quality
    management these bound the internal consistency of a judge rubric.

    """ + _REFERENCES
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "guttman_lambdas"):
        raise RuntimeError("guttman_lambdas requires the compiled Rust core")
    x = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    if x.ndim != 2:
        raise ValueError("data must be a 2-D persons x items array")
    n_persons, n_items = x.shape
    if int(n_sample_splits) < 1:
        raise ValueError("n_sample_splits must be >= 1")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    res = core.guttman_lambdas(
        x.reshape(-1), int(n_persons), int(n_items), int(n_sample_splits), int(seed)
    )
    return GuttmanResult(
        lambda1=float(res["lambda1"]),
        lambda2=float(res["lambda2"]),
        lambda3=float(res["lambda3"]),
        lambda4=float(res["lambda4"]),
        lambda5=float(res["lambda5"]),
        lambda6=float(res["lambda6"]),
        beta=float(res["beta"]),
        mean_split=float(res["mean_split"]),
        n_splits=int(res["n_splits"]),
        exhaustive=bool(res["exhaustive"]),
    )


@dataclass
class TenBergeResult:
    """ten Berge & Zegers mu reliability lower bounds.

    ``mu0`` equals coefficient alpha (Guttman lambda3) and ``mu1`` equals
    Guttman lambda2 exactly; the series satisfies
    ``mu0 <= mu1 <= mu2 <= mu3``."""

    mu0: float
    mu1: float
    mu2: float
    mu3: float


_TENBERGE_REFERENCES = """References (APA 7th ed.):
        Revelle, W. (2025). *psych: Procedures for psychological,
            psychometric, and personality research* (Version 2.6.5)
            [R package]. https://CRAN.R-project.org/package=psych
        ten Berge, J. M. F., & Zegers, F. E. (1978). A series of lower
            bounds to the reliability of a test. *Psychometrika, 43*(4),
            575-579. https://doi.org/10.1007/BF02293811 (as cited in
            Revelle, 2025)
    """


def tenberge_mu(data: np.ndarray) -> TenBergeResult:
    """ten Berge & Zegers mu0-mu3 reliability lower bounds (compute in
    Rust; algorithm transcribed from the psych 2.6.5 R source
    ``tenberge.R``, read line by line; ten Berge & Zegers, 1978, not read —
    attribution as cited in Revelle, 2025).

    Computed on the Pearson correlation matrix of ``data`` (an
    ``n_persons x n_items`` array, complete and finite) with ``Vt = sum(R)``,
    off-diagonal power sums ``S_k``, and ``c = p/(p-1)`` on the innermost
    radical only: ``mu0 = c*S_1/Vt`` (= alpha), ``mu1 = (S_1 +
    sqrt(c*S_2))/Vt``, ``mu2`` and ``mu3`` nest one and two more radicals.
    Divergences from psych (documented in the Rust module): raw-data input
    only and hard errors on degenerate input. In LLM-as-a-Judge quality
    management the series tightens the lower bound on rubric internal
    consistency beyond alpha.

    """ + _TENBERGE_REFERENCES
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "tenberge_mu"):
        raise RuntimeError("tenberge_mu requires the compiled Rust core")
    x = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    if x.ndim != 2:
        raise ValueError("data must be a 2-D persons x items array")
    n_persons, n_items = x.shape
    res = core.tenberge_mu(x.reshape(-1), int(n_persons), int(n_items))
    return TenBergeResult(
        mu0=float(res["mu0"]),
        mu1=float(res["mu1"]),
        mu2=float(res["mu2"]),
        mu3=float(res["mu3"]),
    )


@dataclass
class AlphaCiResult:
    """Feldt (1965) exact-F confidence interval for coefficient alpha.

    ``lower``/``upper`` are not clamped and may be negative (alpha itself
    can be negative). ``r_bar`` is the average inter-item correlation
    implied by alpha via the Spearman-Brown inversion."""

    alpha: float
    lower: float
    upper: float
    r_bar: float
    df1: float
    df2: float


_FELDT_REFERENCES = """References (APA 7th ed.):
        Cronbach, L. J. (1951). Coefficient alpha and the internal
            structure of tests. *Psychometrika, 16*(3), 297-334.
            https://doi.org/10.1007/BF02310555 (covariance form verified
            against Revelle, 2025, not re-read)
        Feldt, L. S. (1965). The approximate sampling distribution of
            Kuder-Richardson reliability coefficient twenty.
            *Psychometrika, 30*(3), 357-370.
            https://doi.org/10.1007/BF02289499 (as cited in Revelle, 2025)
        Revelle, W. (2025). *psych: Procedures for psychological,
            psychometric, and personality research* (Version 2.6.5)
            [R package]. https://CRAN.R-project.org/package=psych
    """


def cronbach_alpha(data: np.ndarray) -> float:
    """Cronbach's coefficient alpha from raw data (compute in Rust).

    Covariance form ``alpha = p/(p-1) * (1 - tr(C)/sum(C))`` on the sample
    covariance matrix of ``data`` (an ``n_persons x n_items`` array,
    complete and finite). Divergences from psych::alpha (documented in the
    Rust module): raw-data input only (no reverse-keying), zero-variance
    items rejected, hard errors instead of NA.
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "cronbach_alpha"):
        raise RuntimeError("cronbach_alpha requires the compiled Rust core")
    x = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    if x.ndim != 2:
        raise ValueError("data must be a 2-D persons x items array")
    n_persons, n_items = x.shape
    return float(core.cronbach_alpha(x.reshape(-1), int(n_persons), int(n_items)))


cronbach_alpha.__doc__ += "\n" + _FELDT_REFERENCES


def feldt_alpha_ci(
    alpha: float, n_persons: int, n_items: int, level: float = 0.95
) -> AlphaCiResult:
    """Feldt (1965) exact-F confidence interval for alpha (compute in Rust;
    bound mapping transcribed from the psych 2.6.5 R source ``alpha.ci``,
    read line by line; Feldt, 1965, not read — attribution as cited in
    Revelle, 2025).

    The pivot ``(1-alpha)/(1-alpha_hat)`` is approximately
    ``F(n-1, (n-1)(p-1))``, giving
    ``lower = 1 - (1-alpha_hat) * F^-1(1-delta/2)`` and
    ``upper = 1 - (1-alpha_hat) * F^-1(delta/2)`` with ``delta = 1-level``.
    In LLM-as-a-Judge quality management this quantifies the sampling
    uncertainty of a rubric's internal-consistency estimate.
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "feldt_alpha_ci"):
        raise RuntimeError("feldt_alpha_ci requires the compiled Rust core")
    if n_persons < 0 or n_items < 0:
        raise ValueError("n_persons and n_items must be non-negative")
    res = core.feldt_alpha_ci(float(alpha), int(n_persons), int(n_items), float(level))
    return AlphaCiResult(
        alpha=float(res["alpha"]),
        lower=float(res["lower"]),
        upper=float(res["upper"]),
        r_bar=float(res["r_bar"]),
        df1=float(res["df1"]),
        df2=float(res["df2"]),
    )


feldt_alpha_ci.__doc__ += "\n" + _FELDT_REFERENCES
