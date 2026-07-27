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

@dataclass
class SeparationReliabilityResult:
    """Person separation reliability (eRm ``SepRel``).

    ``sep_rel = (ssd - mse) / ssd`` is unclamped (negative when
    ``mse > ssd``, NaN when ``ssd`` is ~0). ``sep_index`` is the
    hand-derived separation index ``G = sqrt((ssd - mse) / mse)``
    (adjusted true SD over RMSE; not in the read source), NaN when
    ``mse`` is ~0 or ``ssd < mse``."""

    sep_rel: float
    ssd: float
    mse: float
    sep_index: float


def separation_reliability(
    measures: np.ndarray, se: np.ndarray
) -> SeparationReliabilityResult:
    """Person separation reliability (compute in Rust; formula transcribed
    from CRAN eRm ``R/SepRel.R``, read in full: ``SSD = var(measures)``
    with n-1 denominator, ``MSE = mean(se**2)``,
    ``R = (SSD - MSE) / SSD``). eRm's docs attribute the statistic to
    Wright and Stone (1999), not read.

    Callers must pass already-cleaned vectors: eRm drops persons with
    extreme raw scores (interpolated thetas) and missing estimates before
    applying the formula; that filtering is not reproduced here. In
    LLM-as-a-Judge quality management this measures how reliably judge
    severity estimates separate the judged units given their standard
    errors.

    References (APA 7th ed.):
        Mair, P., Hatzinger, R., & Maier, M. J. (2025). *eRm: Extended
            Rasch modeling* [R package]. https://CRAN.R-project.org/package=eRm
        Wright, B. D., & Stone, M. H. (1999). *Measurement essentials*
            (2nd ed.). Wide Range. (as cited in Mair et al., 2025; not read)
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "separation_reliability"):
        raise RuntimeError("separation_reliability requires the compiled Rust core")
    x = np.ascontiguousarray(np.asarray(measures, dtype=np.float64))
    s = np.ascontiguousarray(np.asarray(se, dtype=np.float64))
    if x.ndim != 1 or s.ndim != 1:
        raise ValueError("measures and se must be 1-D arrays")
    res = core.separation_reliability(x, s)
    return SeparationReliabilityResult(
        sep_rel=float(res["sep_rel"]),
        ssd=float(res["ssd"]),
        mse=float(res["mse"]),
        sep_index=float(res["sep_index"]),
    )


@dataclass
class IccResult:
    """Intraclass correlation coefficient (irr ``icc``).

    ``value`` is the ICC estimate; ``fvalue``/``df1``/``df2``/``p_value``
    test H0: icc = ``r0`` (upper tail); ``lbound``/``ubound`` are the
    two-sided ``conf_level`` interval (unclamped; can drop below -1 for
    the one-way average variant). ``subjects`` counts the complete rows
    actually used after listwise NaN deletion."""

    value: float
    subjects: int
    raters: int
    fvalue: float
    df1: float
    df2: float
    p_value: float
    lbound: float
    ubound: float


def icc(
    ratings,
    model: str = "oneway",
    type: str = "consistency",
    unit: str = "single",
    r0: float = 0.0,
    conf_level: float = 0.95,
) -> IccResult:
    """Intraclass correlation coefficients for inter-rater reliability
    (compute in Rust; transcribed line by line from the CRAN irr 0.85 R
    source ``icc.R``, read in full; Shrout & Fleiss, 1979, McGraw & Wong,
    1996, and Bartko, 1966, not read — attribution as cited in Gamer et
    al., 2019). Covers the Shrout-Fleiss taxonomy: ``model`` in
    {"oneway", "twoway"}, ``type`` in {"consistency", "agreement"},
    ``unit`` in {"single", "average"}.

    ``ratings`` is a 2-D subjects x raters array of continuous scores.
    Rows containing NaN are dropped listwise (R ``na.omit``); infinities
    are rejected. The two-way agreement F test uses the Satterthwaite
    approximation with the null value ``r0``; its confidence bounds plug
    the estimate back into the nr-scaled Satterthwaite form for both
    units, matching the R source (icc.R lines 139-141) exactly. In
    LLM-as-a-Judge quality management this quantifies how consistently
    multiple judge models (raters) score the same responses (subjects).

    Verified against an exact-Fraction re-derivation of icc.R executed on
    the Shrout-Fleiss Table 2 data (all six variants, plus r0 and
    conf_level sweeps); the Spearman-Brown identity between single and
    average units was verified for all three families.

    References (APA 7th ed.):
        Gamer, M., Lemon, J., Fellows, I., & Singh, P. (2019). *irr:
            Various coefficients of interrater reliability and agreement*
            (Version 0.84.1; source read at 0.85) [R package].
            https://CRAN.R-project.org/package=irr
        Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations:
            Uses in assessing rater reliability. *Psychological Bulletin,
            86*(2), 420-428. https://doi.org/10.1037/0033-2909.86.2.420
            (as cited in Gamer et al., 2019)
        McGraw, K. O., & Wong, S. P. (1996). Forming inferences about
            some intraclass correlation coefficients. *Psychological
            Methods, 1*(1), 30-46. https://doi.org/10.1037/1082-989X.1.1.30
            (as cited in Gamer et al., 2019)
        Bartko, J. J. (1966). The intraclass correlation coefficient as a
            measure of reliability. *Psychological Reports, 19*(1), 3-11.
            https://doi.org/10.2466/pr0.1966.19.1.3 (as cited in Gamer et
            al., 2019)
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "icc"):
        raise RuntimeError("icc requires the compiled Rust core")
    if isinstance(ratings, np.ma.MaskedArray):
        raise ValueError("masked arrays are not supported; use NaN for missing")
    arr = np.asarray(ratings)
    if arr.dtype == object:
        # Rounds 1-5 of adversarial review showed per-element vetting of
        # object arrays is an unwinnable arms race (bool, 0-D wrappers,
        # np.void, timedelta64, self-referential arrays, __float__-lying
        # subclasses). Numeric input never needs object dtype, so reject it.
        raise ValueError(
            "object-dtype arrays are not supported; pass a numeric array"
        )
    if np.iscomplexobj(arr):
        raise ValueError("ratings must be real-valued")
    if arr.dtype.kind == "b":
        raise ValueError("ratings must be numeric, not boolean")
    if arr.dtype.kind not in "fiu":
        raise ValueError("ratings must be a numeric array")
    if arr.ndim != 2:
        raise ValueError("ratings must be a 2-D subjects x raters array")
    x = np.ascontiguousarray(arr, dtype=np.float64)
    ns, nr = x.shape
    for name, val in (("r0", r0), ("conf_level", conf_level)):
        if isinstance(val, bool) or not isinstance(val, (int, float, np.floating, np.integer)):
            raise ValueError(f"{name} must be a real number")
    res = core.icc(
        x.reshape(-1), int(ns), int(nr), str(model), str(type), str(unit),
        float(r0), float(conf_level),
    )
    return IccResult(
        value=float(res["value"]),
        subjects=int(res["subjects"]),
        raters=int(res["raters"]),
        fvalue=float(res["fvalue"]),
        df1=float(res["df1"]),
        df2=float(res["df2"]),
        p_value=float(res["p_value"]),
        lbound=float(res["lbound"]),
        ubound=float(res["ubound"]),
    )
@dataclass
class KrippResult:
    """Krippendorff's alpha (irr ``kripp.alpha``).

    ``value`` is the alpha estimate; ``subjects``/``raters`` echo the
    matrix dimensions as given; ``levels`` counts the distinct observed
    rating values; ``nmatchval`` is the total coincidence-matrix mass
    (R ``nmatchval``). No SE or CI is produced (the R source computes
    none)."""

    value: float
    subjects: int
    raters: int
    levels: int
    nmatchval: float


def kripp_alpha(ratings, method: str = "nominal") -> KrippResult:
    """Krippendorff's alpha for a raters x subjects matrix (compute in
    Rust; transcribed from CRAN irr 0.85 ``R/kripp.alpha.R``, read in
    full). ``ratings`` rows are raters ("classifiers"), columns are
    subjects; NaN marks missing. ``method`` selects the distance metric:
    "nominal", "ordinal", "interval", or "ratio".

    The irr source divides each column's pair counts by
    ``#nonmissing - 1`` only when the matrix contains at least one
    missing value, and by 1 otherwise; that quirk is preserved verbatim,
    so complete-data alpha differs from the ``m - 1`` convention.
    Documented deviations: an all-missing matrix, infinite ratings, and
    a ratio-metric level pair summing to zero raise ``ValueError`` (R
    would return alpha = 1, propagate, or emit Inf/NaN respectively).
    In LLM-as-a-Judge quality management this estimates chance-corrected
    agreement among judges over the same units.

    References (APA 7th ed.):
        Gamer, M., Lemon, J., Fellows, I., & Singh, P. (2019). *irr:
            Various coefficients of interrater reliability and agreement*
            [R package]. https://CRAN.R-project.org/package=irr
        Krippendorff, K. (1980). *Content analysis: An introduction to
            its methodology*. Sage. (as cited in Gamer et al., 2019;
            not read)
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "kripp_alpha"):
        raise RuntimeError("kripp_alpha requires the compiled Rust core")
    if isinstance(ratings, np.ma.MaskedArray):
        raise ValueError("masked arrays are not supported; use NaN for missing")
    arr = np.asarray(ratings)
    if arr.dtype == object:
        raise ValueError(
            "object-dtype arrays are not supported; pass a numeric array"
        )
    if np.iscomplexobj(arr):
        raise ValueError("ratings must be real-valued")
    if arr.dtype.kind == "b":
        raise ValueError("ratings must be numeric, not boolean")
    if arr.dtype.kind not in "fiu":
        raise ValueError("ratings must be a numeric array")
    if arr.ndim != 2:
        raise ValueError("ratings must be a 2-D raters x subjects array")
    if arr.dtype.kind in "iu" and arr.size:
        # Levels are defined by exact f64 value identity; integers beyond
        # 2**53 are not exactly representable and distinct rating labels
        # would silently collapse during the float64 conversion.
        lo, hi = int(arr.min()), int(arr.max())
        if hi > 2**53 or lo < -(2**53):
            raise ValueError(
                "integer ratings beyond 2**53 cannot be represented "
                "exactly as float64; distinct levels would collapse"
            )
    x = np.ascontiguousarray(arr, dtype=np.float64)
    nr, ns = x.shape
    res = core.kripp_alpha(x.reshape(-1), int(nr), int(ns), str(method))
    return KrippResult(
        value=float(res["value"]),
        subjects=int(res["subjects"]),
        raters=int(res["raters"]),
        levels=int(res["levels"]),
        nmatchval=float(res["nmatchval"]),
    )
@dataclass
class FinnResult:
    """Finn (1970) reliability coefficient (irr ``finn``).

    ``value`` is the coefficient ``1 - MS/MSexp``; ``statistic`` is the F
    value ``MSexp/MS`` (``+inf`` for the documented perfect-agreement
    ``MS == 0`` case, in which ``p_value`` is 0); ``df2`` is the
    denominator degrees of freedom ``ns*(nr-1)`` (a convenience field --
    the R return encodes it only inside ``stat.name``); ``p_value`` is
    the upper tail ``pf(F, Inf, df2, lower.tail=FALSE)``. ``subjects``
    counts the complete rows used after listwise NaN deletion."""

    value: float
    statistic: float
    df2: float
    p_value: float
    subjects: int
    raters: int


def finn_coefficient(ratings, s_levels: int, model: str = "oneway") -> FinnResult:
    """Finn (1970) coefficient of reliability for a subjects x raters
    matrix of ratings on a discrete scale with ``s_levels`` levels
    (compute in Rust; transcribed from the CRAN irr 0.85 R source
    ``finn.R``, read in full). ``model`` is "oneway" or "twoway"; rows
    containing NaN are dropped listwise (R ``na.omit``).

    The coefficient compares the observed within-subject (oneway ``MSw``)
    or residual (twoway ``MSe``) mean square against the variance of a
    discrete uniform on 1..s, ``MSexp = (s^2 - 1)/12``. Both models use
    ``df2 = ns*(nr-1)`` -- the R source applies ``ns*(nr-1)`` to twoway
    as well; that quirk is preserved verbatim. The p-value uses the
    limiting identity ``pf(F, Inf, df2, lower.tail=FALSE) =
    pchisq(df2/F, df2)`` (hand-derived from the F-ratio construction and
    convergence-verified against scipy; valid for F > 0).

    Documented deviations from R (deliberate, stricter-than-R): explicit
    errors for fewer than 2 complete rows, fewer than 2 raters,
    ``s_levels < 2`` (or bool), infinities, and negative mean squares
    from floating cancellation. In LLM-as-a-Judge quality management
    this measures how far judge scores depart from a random-uniform
    rating process on the same discrete scale.

    References (APA 7th ed.):
        Gamer, M., Lemon, J., Fellows, I., & Singh, P. (2019). *irr:
            Various coefficients of interrater reliability and agreement*
            [R package]. https://CRAN.R-project.org/package=irr
        Finn, R. H. (1970). A note on estimating the reliability of
            categorical data. *Educational and Psychological Measurement,
            30*(1), 71-76. https://doi.org/10.1177/001316447003000106
            (as cited in Gamer et al., 2019; not read)
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "finn_coefficient"):
        raise RuntimeError("finn_coefficient requires the compiled Rust core")
    if isinstance(ratings, np.ma.MaskedArray):
        raise ValueError("masked arrays are not supported; use NaN for missing")
    arr = np.asarray(ratings)
    if arr.dtype == object:
        raise ValueError(
            "object-dtype arrays are not supported; pass a numeric array"
        )
    if np.iscomplexobj(arr):
        raise ValueError("ratings must be real-valued")
    if arr.dtype.kind == "b":
        raise ValueError("ratings must be numeric, not boolean")
    if arr.dtype.kind not in "fiu":
        raise ValueError("ratings must be a numeric array")
    if arr.ndim != 2:
        raise ValueError("ratings must be a 2-D subjects x raters array")
    # bool is an int subclass; True would silently become s_levels=1.
    if isinstance(s_levels, bool) or not isinstance(s_levels, (int, np.integer)):
        raise ValueError("s_levels must be an integer")
    if int(s_levels) < 2:
        raise ValueError("s_levels must be at least 2")
    x = np.ascontiguousarray(arr, dtype=np.float64)
    ns, nr = x.shape
    res = core.finn_coefficient(x.reshape(-1), int(ns), int(nr), int(s_levels), str(model))
    return FinnResult(
        value=float(res["value"]),
        statistic=float(res["statistic"]),
        df2=float(res["df2"]),
        p_value=float(res["p_value"]),
        subjects=int(res["subjects"]),
        raters=int(res["raters"]),
    )
@dataclass
class MaxwellResult:
    """Maxwell's RE agreement coefficient (irr ``maxwell``).

    ``value`` is ``2*A/ns - 1`` where ``A`` counts subjects with exactly
    equal ratings from the two raters; ``subjects`` counts the complete
    rows used after listwise NaN deletion; ``raters`` is always 2."""

    value: float
    subjects: int
    raters: int


def maxwell_re(ratings) -> MaxwellResult:
    """Maxwell's RE agreement coefficient for a subjects x 2 matrix of
    binary ratings (computed in Rust; transcribed from the CRAN irr
    0.84.1 R source ``maxwell.R``, read in full). Rows containing NaN
    are dropped listwise (R ``na.omit``); the distinct-value union
    across BOTH columns must have at most 2 levels (any two numeric
    labels are accepted; a single level yields RE = 1).

    The R source computes ``2*sum(diag(table(r1, r2)))/ns - 1``; because
    both columns are refactored with the same level vector, the diagonal
    sum equals the exact match count regardless of level ordering
    (hand-derived and verified against an executed exact-arithmetic
    oracle). Documented deviations from R (deliberate, stricter-than-R):
    explicit errors for ``nr != 2`` (R only stops for ``nr > 2`` and
    fails accidentally for one column), infinities, and empty input. In
    LLM-as-a-Judge quality management this measures chance-corrected
    agreement between two judges on a binary criterion.

    References (APA 7th ed.):
        Gamer, M., Lemon, J., Fellows, I., & Singh, P. (2019). *irr:
            Various coefficients of interrater reliability and agreement*
            [R package]. https://CRAN.R-project.org/package=irr
        Maxwell, A. E. (1977). Coefficients of agreement between
            observers and their interpretation. *British Journal of
            Psychiatry, 130*(1), 79-83. https://doi.org/10.1192/bjp.130.1.79
            (as cited in Gamer et al., 2019; not read)
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "maxwell_re"):
        raise RuntimeError("maxwell_re requires the compiled Rust core")
    if isinstance(ratings, np.ma.MaskedArray):
        raise ValueError("masked arrays are not supported; use NaN for missing")
    arr = np.asarray(ratings)
    if arr.dtype == object:
        raise ValueError(
            "object-dtype arrays are not supported; pass a numeric array"
        )
    if np.iscomplexobj(arr):
        raise ValueError("ratings must be real-valued")
    if arr.dtype.kind == "b":
        raise ValueError("ratings must be numeric, not boolean")
    if arr.dtype.kind not in "fiu":
        raise ValueError("ratings must be a numeric array")
    if arr.ndim != 2:
        raise ValueError("ratings must be a 2-D subjects x 2 array")
    # min/max comparisons instead of np.abs: abs overflows on np.int64 min
    # (-2**63), which would silently pass the fidelity guard.
    if arr.dtype.kind in "iu" and arr.size and (
        int(arr.min()) < -(2**53) or int(arr.max()) > 2**53
    ):
        # Exact-label equality must survive the f64 conversion.
        raise ValueError("integer ratings exceed exact float64 range (2**53)")
    x = np.ascontiguousarray(arr, dtype=np.float64)
    ns, nr = x.shape
    res = core.maxwell_re(x.reshape(-1), int(ns), int(nr))
    return MaxwellResult(
        value=float(res["value"]),
        subjects=int(res["subjects"]),
        raters=int(res["raters"]),
    )
@dataclass
class RobinsonResult:
    """Robinson's A coefficient of agreement (irr ``robinson``).

    ``value`` is ``SSb / (SSb + SSr)`` in ``[0, 1]``; ``subjects`` counts
    the complete rows used after listwise NaN deletion; ``raters`` is the
    number of columns."""

    value: float
    subjects: int
    raters: int


def robinson_a(ratings) -> RobinsonResult:
    """Robinson's A coefficient of agreement for a subjects x raters
    matrix of interval-scale ratings (computed in Rust; transcribed from
    the CRAN irr 0.84.1 R source ``robinson.R``, read in full). Rows
    containing NaN are dropped listwise (R ``na.omit``).

    The R source computes sample variances whose ``(n - 1)`` factors
    cancel, giving ``SSb = nr * sum_i (rowmean_i - grand)**2`` and
    ``A = SSb / (SSb + SSr)`` where ``SSr`` is the two-way interaction
    sum of squares ``sum_ij (x_ij - rowmean_i - colmean_j + grand)**2``
    (hand-derived; the identity with R's subtractive form was verified
    exactly against an executed exact-arithmetic oracle). ``SSr`` is
    computed directly as the sum of squared terms, so it is nonnegative
    in floating point. Documented deviation from R (deliberate,
    stricter-than-R): degenerate inputs with no subject variance
    (``SSb + SSr == 0``, e.g. identical rows or a constant matrix), where
    R silently returns NaN from 0/0, raise ``ValueError``, as do
    infinities, a single rater, and fewer than 2 complete rows. In
    LLM-as-a-Judge quality management this measures how much of the
    rating variance is attributable to the items being judged rather
    than judge disagreement.

    References (APA 7th ed.):
        Gamer, M., Lemon, J., Fellows, I., & Singh, P. (2019). *irr:
            Various coefficients of interrater reliability and agreement*
            [R package]. https://CRAN.R-project.org/package=irr
        Robinson, W. S. (1957). The statistical measurement of agreement.
            *American Sociological Review, 22*(1), 17-25.
            https://doi.org/10.2307/2088760 (as cited in Gamer et al.,
            2019; not read)
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "robinson_a"):
        raise RuntimeError("robinson_a requires the compiled Rust core")
    if isinstance(ratings, np.ma.MaskedArray):
        raise ValueError("masked arrays are not supported; use NaN for missing")
    arr = np.asarray(ratings)
    if arr.dtype == object:
        raise ValueError(
            "object-dtype arrays are not supported; pass a numeric array"
        )
    if np.iscomplexobj(arr):
        raise ValueError("ratings must be real-valued")
    if arr.dtype.kind == "b":
        raise ValueError("ratings must be numeric, not boolean")
    if arr.dtype.kind not in "fiu":
        raise ValueError("ratings must be a numeric array")
    if arr.ndim != 2:
        raise ValueError("ratings must be a 2-D subjects x raters array")
    # No 2**53 integer fidelity guard: values enter continuous sums of
    # squares (icc/finn precedent), not exact-label equality.
    x = np.ascontiguousarray(arr, dtype=np.float64)
    ns, nr = x.shape
    res = core.robinson_a(x.reshape(-1), int(ns), int(nr))
    return RobinsonResult(
        value=float(res["value"]),
        subjects=int(res["subjects"]),
        raters=int(res["raters"]),
    )


@dataclass
class MeanCorResult:
    """Mean pairwise Pearson correlation of raters (irr ``meancor``).

    ``value`` is the mean pairwise correlation (Fisher back-transformed
    when ``fisher``); ``statistic``/``p_value`` are the Fisher z test and
    two-sided p, or ``None`` when ``fisher=False``; ``dropped`` counts
    perfectly correlated pairs excluded from the Fisher average;
    ``subjects`` counts complete rows after listwise NaN deletion."""

    value: float
    statistic: float | None
    p_value: float | None
    dropped: int
    subjects: int
    raters: int


def mean_pairwise_cor(ratings, fisher: bool = True) -> MeanCorResult:
    """Mean of the pairwise Pearson correlations between rater columns
    of a subjects x raters matrix (computed in Rust; transcribed from
    the CRAN irr 0.84.1 R source ``meancor.R``, read in full). Rows
    containing NaN are dropped listwise (R ``na.omit``).

    With ``fisher=True`` (the R default), pairs with ``r`` exactly
    ``+/-1`` are dropped (their count is reported in ``dropped``; R only
    appends a warning string), the remaining correlations are averaged
    on the Fisher z scale and back-transformed
    (``value = tanh(mean(atanh(r)))``), and a z test is reported with
    ``SE = sqrt(1/(m-3))`` and ``p = 2*(1 - Phi(|z|))`` (hand-derived
    identity ``= erfc(|z|/sqrt(2))``, verified against an executed
    exact-arithmetic oracle). With ``fisher=False`` the plain mean of
    all pairwise correlations is returned, perfect pairs included, and
    ``statistic``/``p_value`` are ``None``. The Fisher z transformation
    is conventionally attributed to Fisher (1925), not read; only the
    irr source was used as the contract.

    Documented deviations from R (deliberate, stricter-than-R):
    constant rater columns (R ``cor`` NA), fewer than 4 complete rows
    with ``fisher`` (R's SE is infinite or undefined), all pairs perfect
    under ``fisher`` (R mean of an empty vector), infinities, a single
    rater, and fewer than 2 complete rows all raise ``ValueError``.
    ``fisher`` must be a real ``bool`` (or ``numpy.bool_``); truthy
    stand-ins like ``1`` or ``"false"`` are rejected — a deliberately
    stricter policy than older wrappers in this package. In
    LLM-as-a-Judge quality management this summarizes how consistently
    judges rank the same responses.

    References (APA 7th ed.):
        Gamer, M., Lemon, J., Fellows, I., & Singh, P. (2019). *irr:
            Various coefficients of interrater reliability and agreement*
            [R package]. https://CRAN.R-project.org/package=irr
        Fisher, R. A. (1925). *Statistical methods for research workers*.
            Oliver & Boyd. (as cited in Gamer et al., 2019; not read)
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "mean_pairwise_cor"):
        raise RuntimeError("mean_pairwise_cor requires the compiled Rust core")
    if not isinstance(fisher, (bool, np.bool_)):
        raise TypeError("fisher must be a bool")
    if isinstance(ratings, np.ma.MaskedArray):
        raise ValueError("masked arrays are not supported; use NaN for missing")
    arr = np.asarray(ratings)
    if arr.dtype == object:
        raise ValueError(
            "object-dtype arrays are not supported; pass a numeric array"
        )
    if np.iscomplexobj(arr):
        raise ValueError("ratings must be real-valued")
    if arr.dtype.kind == "b":
        raise ValueError("ratings must be numeric, not boolean")
    if arr.dtype.kind not in "fiu":
        raise ValueError("ratings must be a numeric array")
    if arr.ndim != 2:
        raise ValueError("ratings must be a 2-D subjects x raters array")
    # No 2**53 integer fidelity guard: values enter continuous centered
    # sums (icc/finn/robinson precedent), not exact-label equality.
    x = np.ascontiguousarray(arr, dtype=np.float64)
    ns, nr = x.shape
    res = core.mean_pairwise_cor(x.reshape(-1), int(ns), int(nr), bool(fisher))
    stat = float(res["statistic"])
    p = float(res["p_value"])
    return MeanCorResult(
        value=float(res["value"]),
        statistic=None if stat != stat else stat,  # NaN -> None
        p_value=None if p != p else p,
        dropped=int(res["dropped"]),
        subjects=int(res["subjects"]),
        raters=int(res["raters"]),
    )
