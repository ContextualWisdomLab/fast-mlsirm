"""IRT classification accuracy and consistency for cut-score decisions
(Rudner, 2001, 2005; Lee, 2010, as implemented in CRAN cacIRT). All numeric
work happens in the Rust core; this module only validates and marshals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class ClassificationResult:
    """Classification accuracy/consistency for ``m`` cuts and ``n`` points.

    ``per_cut_*`` treat each cut as its own two-category problem;
    ``simultaneous_*`` score the full ``m + 1``-category classification.
    ``conditional_*`` are per evaluation point (``m x n`` arrays for the
    per-cut versions). Marginals are weighted means over points using the
    normalized input weights. Unlike cacIRT, the simultaneous outputs are
    always populated; with one cut they equal the per-cut values."""

    per_cut_accuracy: np.ndarray
    per_cut_consistency: np.ndarray
    simultaneous_accuracy: float
    simultaneous_consistency: float
    conditional_accuracy: np.ndarray
    conditional_consistency: np.ndarray
    conditional_simultaneous_accuracy: np.ndarray
    conditional_simultaneous_consistency: np.ndarray


@dataclass
class LivingstonLewisResult:
    """Livingston-Lewis single-administration classification results.

    ``p_tp``/``p_fp``/``p_tf``/``p_ff`` are the accuracy cells (pass =
    observed score at or above the cut; ``t``/``f`` = true pass/fail);
    ``p_ii``/``p_ij``/``p_ji``/``p_jj`` are consistency cells over two
    hypothetical parallel forms, with ``p_ij == p_ji`` by construction
    (single rounded threshold in both blocks — betafunctions' round/floor
    mix makes its cells asymmetric; divergence documented in the Rust
    core)."""

    effective_test_length: float
    etl_rounded: int
    lower: float
    upper: float
    alpha: float
    beta: float
    used_two_parameter: bool
    p_tp: float
    p_fp: float
    p_tf: float
    p_ff: float
    accuracy: float
    sensitivity: float
    specificity: float
    p_ii: float
    p_ij: float
    p_ji: float
    p_jj: float
    consistency: float
    chance_consistency: float
    kappa: float


_REFERENCES = """References (APA 7th ed.):
        Lathrop, Q. N. (2015). *cacIRT: Classification accuracy and
            consistency under item response theory* (Version 1.4)
            [R package]. https://CRAN.R-project.org/package=cacIRT
        Lee, W.-C. (2010). Classification consistency and accuracy for
            complex assessments using item response theory. *Journal of
            Educational Measurement, 47*(1), 1-17. (as cited in
            Lathrop, 2015)
        Rudner, L. M. (2001). Computing the expected proportions of
            misclassified examinees. *Practical Assessment, Research &
            Evaluation, 7*(14). https://doi.org/10.7275/an9m-2035
        Rudner, L. M. (2005). Expected classification accuracy. *Practical
            Assessment, Research & Evaluation, 10*(13).
            https://doi.org/10.7275/56a5-6b14
    """


def _to_result(res: dict, m: int, n: int) -> ClassificationResult:
    def arr(key: str) -> np.ndarray:
        return np.asarray(res[key], dtype=np.float64)

    return ClassificationResult(
        per_cut_accuracy=arr("per_cut_accuracy"),
        per_cut_consistency=arr("per_cut_consistency"),
        simultaneous_accuracy=float(res["simultaneous_accuracy"]),
        simultaneous_consistency=float(res["simultaneous_consistency"]),
        conditional_accuracy=arr("conditional_accuracy").reshape(m, n),
        conditional_consistency=arr("conditional_consistency").reshape(m, n),
        conditional_simultaneous_accuracy=arr(
            "conditional_simultaneous_accuracy"
        ),
        conditional_simultaneous_consistency=arr(
            "conditional_simultaneous_consistency"
        ),
    )


def _core_or_raise(name: str):
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, name):
        raise RuntimeError(f"{name} requires the compiled Rust core")
    return core


def rudner_classification(
    theta: np.ndarray,
    sem: np.ndarray,
    cutscores: Sequence[float],
    weights: np.ndarray | None = None,
) -> ClassificationResult:
    """Rudner normal-approximation classification accuracy/consistency
    (compute in Rust; Rudner, 2001, 2005, both read in full).

    The observed score of a point with ability ``theta[i]`` is modeled as
    normal with mean ``theta[i]`` and standard deviation ``sem[i]`` (Rudner,
    2001, eqs. 1-3; 2005, eq. 1). Accuracy at a cut is the normal mass on
    the true side of the cut; consistency is the sum of squared category
    masses — a formula that appears in neither Rudner paper and follows the
    cacIRT source (Lathrop, 2015), which attributes it to Lee (2010).
    Category intervals are left-closed (``theta`` exactly on a cut belongs
    to the upper category). ``weights`` defaults to uniform (cacIRT's
    person-level ``Rud.P``); quadrature weights give the distribution-level
    ``Rud.D`` (normalized internally). In LLM-as-a-Judge quality management
    this quantifies how reliably a judge's cut score separates pass from
    fail given the calibration's standard errors.

    """ + _REFERENCES
    core = _core_or_raise("rudner_classification")
    t = np.ascontiguousarray(np.asarray(theta, dtype=np.float64).reshape(-1))
    s = np.ascontiguousarray(np.asarray(sem, dtype=np.float64).reshape(-1))
    w = (
        np.ones_like(t)
        if weights is None
        else np.ascontiguousarray(
            np.asarray(weights, dtype=np.float64).reshape(-1)
        )
    )
    cuts = [float(c) for c in cutscores]
    res = core.rudner_classification(t, s, w, cuts)
    return _to_result(res, len(cuts), t.shape[0])


def lee_classification(
    probs: np.ndarray,
    cutscores: Sequence[float],
    weights: np.ndarray | None = None,
) -> ClassificationResult:
    """Lee summed-score classification accuracy/consistency for dichotomous
    items (compute in Rust; Lee, 2010, as cited in Lathrop, 2015; mechanics
    transcribed from the cacIRT R sources, read line by line).

    ``probs`` is an ``n_points x n_items`` array of correct-response
    probabilities strictly inside (0, 1) — model-agnostic: any binary IRF
    evaluated at persons or quadrature nodes works. The summed-score
    distribution per point comes from the Lord-Wingersky (1984) recursion;
    raw cut ``c`` splits scores at ``ceil(c)`` and a point's true category
    is the raw-score interval containing its expected true score
    (left-closed; cacIRT's ``Lee.D`` alone is right-closed — divergence
    documented in the Rust core). ``weights`` defaults to uniform.

    """ + _REFERENCES
    core = _core_or_raise("lee_classification")
    p = np.ascontiguousarray(np.asarray(probs, dtype=np.float64))
    if p.ndim != 2:
        raise ValueError("probs must be a 2-D points x items array")
    n_points, n_items = p.shape
    w = (
        np.ones(n_points)
        if weights is None
        else np.ascontiguousarray(
            np.asarray(weights, dtype=np.float64).reshape(-1)
        )
    )
    cuts = [float(c) for c in cutscores]
    res = core.lee_classification(
        p.reshape(-1), int(n_points), int(n_items), w, cuts
    )
    return _to_result(res, len(cuts), n_points)


def livingston_lewis(
    scores: np.ndarray,
    reliability: float,
    min_score: float,
    max_score: float,
    cut: float,
) -> LivingstonLewisResult:
    """Livingston-Lewis classification accuracy/consistency from a single
    test administration (compute in Rust; Livingston & Lewis, 1995, as
    implemented in CRAN betafunctions 1.9.0 ``LL.CA``, read line by line —
    the original article was not consulted directly).

    Proportional true scores are modeled as a four-parameter beta fitted by
    the method of moments (Hanson, 1991, as cited in Haakstad, 2022), with
    a two-parameter [0, 1] fail-safe when the four-parameter fit is out of
    bounds or numerically invalid; the observed-score model is binomial
    with ``N = round(effective test length)``. Pass = observed score >=
    ``cut``; sensitivity/specificity follow this pass-positive orientation
    (betafunctions labels *fail* as positive, so its sensitivity is this
    function's specificity). ``reliability`` is any single-administration
    estimate (e.g. alpha). In LLM-as-a-Judge quality management this
    estimates how accurately and repeatably a judge's cut score classifies
    outputs given the score reliability. Sensitivity, specificity, and
    kappa are ``NaN`` when their margin or chance denominator vanishes
    (e.g. a cut outside the fitted beta support).

    """ + _REFERENCES + """
        Haakstad, H. (2022). *betafunctions: Functions for working with
            two- and four-parameter beta probability distributions and
            psychometric analysis of classifications* (Version 1.9.0)
            [R package]. https://CRAN.R-project.org/package=betafunctions
        Hanson, B. A. (1991). *Method of moments estimates for the
            four-parameter beta compound binomial model and the calculation
            of classification consistency indexes* (ACT Research Report
            91-5). (as cited in Haakstad, 2022)
        Livingston, S. A., & Lewis, C. (1995). Estimating the consistency
            and accuracy of classifications based on test scores. *Journal
            of Educational Measurement, 32*(2), 179-197. (as cited in
            Haakstad, 2022)
    """
    core = _core_or_raise("livingston_lewis")
    x = np.ascontiguousarray(np.asarray(scores, dtype=np.float64).reshape(-1))
    res = core.livingston_lewis(
        x, float(reliability), float(min_score), float(max_score), float(cut)
    )
    return LivingstonLewisResult(
        effective_test_length=float(res["effective_test_length"]),
        etl_rounded=int(res["etl_rounded"]),
        lower=float(res["lower"]),
        upper=float(res["upper"]),
        alpha=float(res["alpha"]),
        beta=float(res["beta"]),
        used_two_parameter=bool(res["used_two_parameter"]),
        p_tp=float(res["p_tp"]),
        p_fp=float(res["p_fp"]),
        p_tf=float(res["p_tf"]),
        p_ff=float(res["p_ff"]),
        accuracy=float(res["accuracy"]),
        sensitivity=float(res["sensitivity"]),
        specificity=float(res["specificity"]),
        p_ii=float(res["p_ii"]),
        p_ij=float(res["p_ij"]),
        p_ji=float(res["p_ji"]),
        p_jj=float(res["p_jj"]),
        consistency=float(res["consistency"]),
        chance_consistency=float(res["chance_consistency"]),
        kappa=float(res["kappa"]),
    )
