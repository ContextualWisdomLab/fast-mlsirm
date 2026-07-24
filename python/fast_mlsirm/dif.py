"""Observed-score differential item functioning (Mantel-Haenszel; Holland & Thayer, 1988).

The calibration-free complement to the parametric IRT likelihood-ratio DIF
(:func:`fast_mlsirm.dif_polytomous`): examinees are matched on the number-correct total score and a
common odds ratio is estimated per item across the resulting ``2 x 2`` (group by response) tables. No
item response model is fitted. The numerical computation runs in Rust."""

from __future__ import annotations

import numpy as np


def mantel_haenszel_dif(
    responses: np.ndarray,
    group: np.ndarray,
    exclude_studied_item: bool = False,
    fdr_q: float = 0.05,
) -> dict[str, np.ndarray]:
    """Mantel-Haenszel DIF sweep for dichotomous items (compute in Rust; Holland & Thayer, 1988).

    Examinees are stratified by an observed matching score (the number-correct total, including the
    studied item by default -- the ETS standard, less biased than the rest score per Donoghue, Holland
    & Thayer, 1993; set ``exclude_studied_item=True`` to match on the rest score). For each item, at
    every matching level a ``2 x 2`` table of group (reference/focal) by response (correct/incorrect) is
    formed, and over the DIF-informative strata (all four marginal totals positive):

    - ``alpha_mh`` is the Mantel-Haenszel common odds ratio;
    - ``chi2_mh`` is the continuity-corrected MH chi-square, referred to ``chi2(1)`` for ``p_value``;
    - ``mh_d_dif`` is the ETS delta-metric statistic ``-2.35 ln(alpha_mh)`` (negative = harder for the
      focal group) with the Robins-Breslow-Greenland (1986) standard error ``se_d_dif``;
    - ``ets_class`` is the ETS ``"A"`` (negligible) / ``"B"`` (moderate) / ``"C"`` (large) severity
      classification (Zieky, 1993), or ``"U"`` when the statistic is undefined (no DIF-informative
      strata or a degenerate odds ratio);
    - ``std_p_dif`` is the standardized P-DIF (Dorans & Kulick, 1986), the focal-minus-reference
      focal-weighted proportion-correct difference (an effect size whose sign agrees with ``mh_d_dif``);
    - ``flagged_bh`` is the Benjamini-Hochberg FDR rejection at ``fdr_q`` on ``p_value``.

    Because MH is an observed-score procedure, its chi-square is over-powered at large N and the studied
    item's presence in the matching total mildly contaminates the criterion; the ``ets_class`` A/B/C
    rule (which requires ``|mh_d_dif| >= 1.0`` for a non-A flag) is the practical-significance guard
    against spuriously flagging clean items. MH detects *uniform* DIF and can miss crossing
    (non-uniform) DIF that the parametric IRT-LR test catches.

    ``responses`` is a persons x items ``0/1`` array (no missing data; drop or impute beforehand).
    ``group`` is a length-persons array with ``0`` = reference and ``1`` = focal (both must be present).
    Returns per-item NumPy arrays keyed as above; NaN statistics / ``"U"`` mark items with no
    DIF-informative strata or a degenerate common odds ratio.

    References (APA 7th ed.):
        Dorans, N. J., & Kulick, E. (1986). Demonstrating the utility of the standardization approach
            to assessing unexpected differential item performance on the Scholastic Aptitude Test.
            *Journal of Educational Measurement, 23*(4), 355-368.
            https://doi.org/10.1111/j.1745-3984.1986.tb00255.x
        Holland, P. W., & Thayer, D. T. (1988). Differential item performance and the Mantel-Haenszel
            procedure. In H. Wainer & H. I. Braun (Eds.), *Test validity* (pp. 129-145). Erlbaum.
        Robins, J., Breslow, N., & Greenland, S. (1986). Estimators of the Mantel-Haenszel variance
            consistent in both sparse data and large-strata limiting models. *Biometrics, 42*(2),
            311-323. https://doi.org/10.2307/2531052
        Zieky, M. (1993). Practical questions in the use of DIF statistics in test development. In P. W.
            Holland & H. Wainer (Eds.), *Differential item functioning* (pp. 337-347). Erlbaum.
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "mantel_haenszel_dif"):
        raise RuntimeError("mantel_haenszel_dif requires the compiled Rust core")

    y = np.asarray(responses)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    if n_persons == 0 or n_items == 0:
        raise ValueError("responses must contain at least one person and one item")
    yf = np.asarray(y, dtype=np.float64)
    if not np.all(np.isin(yf, (0.0, 1.0))):
        raise ValueError("responses must be 0 or 1 (Mantel-Haenszel is for dichotomous items)")
    g = np.asarray(group)
    if g.ndim != 1 or g.shape[0] != n_persons:
        raise ValueError("group must be a length-n_persons 1-D array")
    gf = np.asarray(g, dtype=np.float64)
    if not np.all(np.isin(gf, (0.0, 1.0))):
        raise ValueError("group labels must be 0 (reference) or 1 (focal)")
    if not np.isfinite(fdr_q) or not 0 < fdr_q <= 1:
        raise ValueError("fdr_q must be finite and in (0, 1]")

    res = core.mantel_haenszel_dif(
        yf.astype(np.int64).reshape(-1),
        gf.astype(np.int64),
        int(n_persons),
        int(n_items),
        bool(exclude_studied_item),
        float(fdr_q),
    )
    return {
        "item": np.asarray(res["item"], dtype=np.int64),
        "alpha_mh": np.asarray(res["alpha_mh"], dtype=np.float64),
        "chi2_mh": np.asarray(res["chi2_mh"], dtype=np.float64),
        "p_value": np.asarray(res["p_value"], dtype=np.float64),
        "mh_d_dif": np.asarray(res["mh_d_dif"], dtype=np.float64),
        "se_d_dif": np.asarray(res["se_d_dif"], dtype=np.float64),
        "std_p_dif": np.asarray(res["std_p_dif"], dtype=np.float64),
        "ets_class": np.asarray(res["ets_class"]),
        "flagged_bh": np.asarray(res["flagged_bh"], dtype=bool),
    }


def _dif_inputs(responses: np.ndarray, group: np.ndarray, fdr_q: float):
    """Validation shared by the two PURIFIED entry points.

    The unpurified :func:`mantel_haenszel_dif` and :func:`logistic_dif` predate this helper and still
    inline the equivalent checks; the Rust core re-validates everything either way, so this is a
    duplicate-message concern, not a hole.
    """
    y = np.asarray(responses)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    if n_persons == 0 or n_items == 0:
        raise ValueError("responses must contain at least one person and one item")
    yf = np.asarray(y, dtype=np.float64)
    if not np.all(np.isin(yf, (0.0, 1.0))):
        raise ValueError("responses must be 0 or 1 (observed-score DIF is for dichotomous items)")
    g = np.asarray(group)
    if g.ndim != 1 or g.shape[0] != n_persons:
        raise ValueError("group must be a length-n_persons 1-D array")
    gf = np.asarray(g, dtype=np.float64)
    if not np.all(np.isin(gf, (0.0, 1.0))):
        raise ValueError("group labels must be 0 (reference) or 1 (focal)")
    if not np.isfinite(fdr_q) or not 0 < fdr_q <= 1:
        raise ValueError("fdr_q must be finite and in (0, 1]")
    return yf.astype(np.int64).reshape(-1), gf.astype(np.int64), int(n_persons), int(n_items)


def mantel_haenszel_dif_purified(
    responses: np.ndarray,
    group: np.ndarray,
    exclude_studied_item: bool = False,
    fdr_q: float = 0.05,
    max_rounds: int = 3,
    min_anchor_items: int = 4,
) -> dict[str, np.ndarray]:
    """Mantel-Haenszel DIF with an ITERATIVELY PURIFIED matching criterion (compute in Rust; Candell &
    Drasgow, 1988; Clauser, Mazor & Hambleton, 1993).

    :func:`mantel_haenszel_dif` matches on the raw number-correct total, which contains the very items
    under test, so items with DIF contaminate the criterion. Purification rebuilds the criterion from the
    currently unflagged (anchor) items — an item is scored against ``anchor UNION {itself}`` — and
    re-runs the sweep until the flagged set stabilises or ``max_rounds`` is reached. Items are removed
    from the anchor on PRACTICAL significance (ETS class B or C), not raw significance, since the MH
    chi-square is over-powered at large N.

    Returns everything :func:`mantel_haenszel_dif` returns, plus ``anchor`` (bool per item), ``n_anchor``,
    ``rounds`` (purification rounds after the initial full-test sweep; ``0`` means none were applied),
    ``purify_converged``, and ``purify_termination_reason`` (``stable_flag_set``,
    ``max_rounds_reached``, or ``insufficient_anchor_items``).

    IMPORTANT — the anchor is selected from the SAME data that is then tested against it, so the returned
    p-values are conditional on a data-dependent selection: they are not guaranteed super-uniform under
    the null and Benjamini-Hochberg does NOT control the FDR at ``fdr_q`` for a purified sweep. Treat
    ``flagged_bh`` here as a screening device, not an error-rate guarantee. Purification reduces rather
    than removes criterion contamination and can fail outright when DIF is unbalanced in direction
    (Wang & Su, 2004).

    Mantel-Haenszel's crossing-DIF blind spot is inherited and purification cannot repair it: an item MH
    does not flag stays in the anchor every round and keeps contaminating the criterion. The blindness is
    a property of the SIGNED AREA between the two curves over the matched ability distribution, not of
    non-uniform DIF as such — a crossing at the centre of that distribution cancels and is invisible,
    while the same item with its crossing off centre is detected. Prefer :func:`logistic_dif_purified`
    when non-uniform DIF is plausible.

    References (APA 7th ed.):
        Candell, G. L., & Drasgow, F. (1988). An iterative procedure for linking metrics and assessing
            item bias in item response theory. *Applied Psychological Measurement, 12*(3), 253-260.
            https://doi.org/10.1177/014662168801200304
        Clauser, B., Mazor, K., & Hambleton, R. K. (1993). The effects of purification of the matching
            criterion on the identification of DIF using the Mantel-Haenszel procedure. *Applied
            Measurement in Education, 6*(4), 269-279. https://doi.org/10.1207/s15324818ame0604_2
        Wang, W.-C., & Su, Y.-H. (2004). Effects of average signed area between two item characteristic
            curves and test purification procedures on the DIF detection via the Mantel-Haenszel method.
            *Applied Measurement in Education, 17*(2), 113-144.
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "mantel_haenszel_dif_purified"):
        raise RuntimeError("mantel_haenszel_dif_purified requires the compiled Rust core")
    yy, gg, n_persons, n_items = _dif_inputs(responses, group, fdr_q)
    res = core.mantel_haenszel_dif_purified(
        yy, gg, n_persons, n_items, bool(exclude_studied_item), float(fdr_q),
        int(max_rounds), int(min_anchor_items),
    )
    return _mh_rows(res) | _purify_meta(res)


def logistic_dif_purified(
    responses: np.ndarray,
    group: np.ndarray,
    exclude_studied_item: bool = False,
    fdr_q: float = 0.05,
    max_iter: int = 50,
    max_rounds: int = 3,
    min_anchor_items: int = 4,
) -> dict[str, np.ndarray]:
    """Zumbo logistic-regression DIF with an ITERATIVELY PURIFIED matching criterion (compute in Rust).

    The same purification loop as :func:`mantel_haenszel_dif_purified`, with the anchor decided by
    ``jg_class`` (the Jodoin-Gierl class of the 2-df omnibus test). Unlike the Mantel-Haenszel variant
    this detects crossing DIF, so a non-uniform item is removed from the criterion too.

    Returns everything :func:`logistic_dif` returns — including its PER-ITEM ``converged`` array, one
    flag per item's IRLS fit — plus ``anchor``, ``n_anchor``, ``rounds``, and the scalar
    ``purify_converged`` and ``purify_termination_reason`` for the purification loop itself. The
    per-item and loop-level diagnostics are deliberately named differently because they answer
    different questions.

    IMPORTANT — the anchor is selected from the SAME data that is then tested against it, so the returned
    p-values are conditional on a data-dependent selection: they are not guaranteed super-uniform under
    the null and Benjamini-Hochberg does NOT control the FDR at ``fdr_q`` for a purified sweep. Treat
    ``flagged_bh`` here as a screening device, not an error-rate guarantee. Purification reduces rather
    than removes criterion contamination and can fail outright when DIF is unbalanced in direction
    (Wang & Su, 2004).

    Reference (APA 7th ed.):
        French, B. F., & Maller, S. J. (2007). Iterative purification and effect size use with logistic
            regression for differential item functioning detection. *Educational and Psychological
            Measurement, 67*(3), 373-393. https://doi.org/10.1177/0013164406294781
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "logistic_dif_purified"):
        raise RuntimeError("logistic_dif_purified requires the compiled Rust core")
    yy, gg, n_persons, n_items = _dif_inputs(responses, group, fdr_q)
    res = core.logistic_dif_purified(
        yy, gg, n_persons, n_items, bool(exclude_studied_item), float(fdr_q),
        int(max_iter), int(max_rounds), int(min_anchor_items),
    )
    return _logistic_rows(res) | _purify_meta(res)


def _purify_meta(res) -> dict[str, np.ndarray]:
    # `purify_converged`, not `converged`: the logistic rows already carry a PER-ITEM `converged` array
    # and this dict is merged over them, so the loop's scalar flag must not share the name.
    return {
        "anchor": np.asarray(res["anchor"], dtype=bool),
        "n_anchor": int(res["n_anchor"]),
        "rounds": int(res["rounds"]),
        "purify_converged": bool(res["purify_converged"]),
        "purify_termination_reason": str(res["purify_termination_reason"]),
    }


def _mh_rows(res) -> dict[str, np.ndarray]:
    return {
        "item": np.asarray(res["item"], dtype=np.int64),
        "alpha_mh": np.asarray(res["alpha_mh"], dtype=np.float64),
        "chi2_mh": np.asarray(res["chi2_mh"], dtype=np.float64),
        "p_value": np.asarray(res["p_value"], dtype=np.float64),
        "mh_d_dif": np.asarray(res["mh_d_dif"], dtype=np.float64),
        "se_d_dif": np.asarray(res["se_d_dif"], dtype=np.float64),
        "std_p_dif": np.asarray(res["std_p_dif"], dtype=np.float64),
        "ets_class": np.asarray(res["ets_class"]),
        "flagged_bh": np.asarray(res["flagged_bh"], dtype=bool),
    }


def sibtest(
    responses: np.ndarray,
    group: np.ndarray,
    fdr_q: float = 0.05,
    j_min: int = 5,
) -> dict[str, np.ndarray]:
    """Uniform SIBTEST for dichotomous items (compute in Rust; Shealy & Stout, 1993).

    The third observed-score DIF procedure in this module, and the only one that corrects the MATCHING
    CRITERION itself. :func:`mantel_haenszel_dif` and :func:`logistic_dif` both match on an observed
    number-correct score, which is unreliable: under IMPACT (a genuine group difference in ability) two
    examinees from different groups with the same OBSERVED score do not have the same expected TRUE
    score, because each regresses toward their own group's mean. Matching on the raw score therefore
    compares non-equivalent examinees and manufactures DIF for items that have none. Item purification
    cannot substitute for this — it changes which items are in the criterion, not the regression of true
    score on observed score, so even a perfectly purified criterion stays biased.

    SIBTEST transports each group's conditional mean from that group's own Kelley-regressed true score
    to a common target (the unweighted midpoint of the two) before comparing. Each item in turn is the
    studied subtest; the valid subtest is every OTHER item, always disjoint — that is a property of the
    estimator, not an option.

    Returns per-item arrays: ``beta_uni`` (the regression-corrected weighted mean difference),
    ``se_beta``, ``b_uni``, ``p_value`` (``b_uni**2`` referred to ``chi2(1)``), ``alpha_ref`` and
    ``alpha_focal`` (the per-group reliabilities the correction divides by — a low or unstable alpha
    inflates the correction, so they are reported rather than hidden), ``n_strata_used``, ``flagged_bh``.

    **SIGN WARNING.** ``beta_uni > 0`` means the item is harder for the FOCAL group. This is the
    OPPOSITE orientation to ``mh_d_dif`` and ``std_p_dif`` from :func:`mantel_haenszel_dif`, which go
    negative in that same situation. The orientation is kept rather than harmonised because published
    ``|beta_uni|`` cut-offs assume it; flip one of the two when comparing across procedures.

    **When to prefer it — rarely, on the evidence measured here.** This implementation was compared
    against :func:`mantel_haenszel_dif` on identical simulated data, 500 replications per cell, 2PL,
    with NO DIF planted so every rejection is a false positive::

        impact  n per group  items   MH Type I   SIBTEST Type I
        0.0     1000         5       .044        .056
        1.0     1000         5       .046        .086

    These are the exact cells the shipped Monte-Carlo regression test runs and the rates it prints, so
    the table is regenerable from the repository rather than quoted from a study that no longer exists.

    SIBTEST over-rejects in both cells and by roughly DOUBLE under impact — the opposite of the ordering
    the motivation above might suggest. The cause is the standard error, below. This is not a
    transcription error (the closed-form anchors reproduce the reference implementation exactly); it is
    a property of the 1993 estimator, and it is what Jiang and Stout's (1998) paper — "Improved Type I
    error control and reduced estimation bias for DIF detection using SIBTEST" — exists to fix. Prefer
    :func:`mantel_haenszel_dif` or :func:`logistic_dif` for routine screening; reach for this when you
    specifically want the regression-corrected estimand, and read ``beta_uni`` as an effect size rather
    than trusting ``p_value`` as a calibrated test.

    **Limitations, none of them silent.** ``se_beta`` treats the regression correction as FIXED, so it
    does not propagate the correction's own estimation error; it is optimistic and the test
    over-rejects, as measured above. The shipped correction is the single linear one; Jiang and Stout's
    (1998) two-segment version is a different, later estimator and is not implemented. No guessing
    correction. No effect-size letter class: published cut-offs disagree and none was verified against a
    primary source, so the raw effect size ships uncalibrated.
    Crossing (non-uniform) DIF is NOT covered — Chalmers (2018) shows the Li and Stout (1996) crossing
    test is insufficient and no normal-theory referral is valid, so use :func:`logistic_dif`, whose
    ``S x G`` interaction tests crossing directly.

    **Provenance.** The formulas are transcribed from the reference implementation (Chalmers, 2012;
    the ``SIBTEST`` routine of the ``mirt`` package), which attributes them to Shealy and Stout (1993).
    The primary text was not consulted, and ``mirt`` was not executed — the transcription was made by
    reading its source, so the closed-form tests verify this code against the transcribed formulas and
    no agreement with ``mirt`` output is claimed.

    ``responses`` is a persons x items ``0/1`` array (no missing data) with at least 3 items; ``group``
    is length-persons with ``0`` = reference and ``1`` = focal. ``j_min`` (default 5) is the number of
    examinees per group per matching level that must be STRICTLY exceeded for that level to count.

    References (APA 7th ed.):
        Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R
            environment. *Journal of Statistical Software, 48*(6), 1-29.
            https://doi.org/10.18637/jss.v048.i06
        Chalmers, R. P. (2018). Improving the crossing-SIBTEST statistic for detecting non-uniform DIF.
            *Psychometrika, 83*(2), 376-386. https://doi.org/10.1007/s11336-017-9583-8
        DeMars, C. E. (2009). Modification of the Mantel-Haenszel and logistic regression DIF procedures
            to incorporate the SIBTEST regression correction. *Journal of Educational and Behavioral
            Statistics, 34*(2), 149-170. https://doi.org/10.3102/1076998607313923
        Jiang, H., & Stout, W. (1998). Improved Type I error control and reduced estimation bias for DIF
            detection using SIBTEST. *Journal of Educational and Behavioral Statistics, 23*(4), 291-322.
            https://doi.org/10.3102/10769986023004291
        Li, H.-H., & Stout, W. (1996). A new procedure for detection of crossing DIF. *Psychometrika,
            61*(4), 647-677. https://doi.org/10.1007/BF02294041
        Shealy, R., & Stout, W. (1993). A model-based standardization approach that separates true
            bias/DIF from group ability differences and detects test bias/DTF as well as item bias/DIF.
            *Psychometrika, 58*(2), 159-194. https://doi.org/10.1007/BF02294572
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "sibtest"):
        raise RuntimeError("sibtest requires the compiled Rust core")
    yy, gg, n_persons, n_items = _dif_inputs(responses, group, fdr_q)
    if not isinstance(j_min, (int, np.integer)) or j_min < 2:
        raise ValueError("j_min must be an integer >= 2")
    res = core.sibtest(yy, gg, n_persons, n_items, float(fdr_q), int(j_min))
    return {
        "item": np.asarray(res["item"], dtype=np.int64),
        "beta_uni": np.asarray(res["beta_uni"], dtype=np.float64),
        "se_beta": np.asarray(res["se_beta"], dtype=np.float64),
        "b_uni": np.asarray(res["b_uni"], dtype=np.float64),
        "p_value": np.asarray(res["p_value"], dtype=np.float64),
        "alpha_ref": np.asarray(res["alpha_ref"], dtype=np.float64),
        "alpha_focal": np.asarray(res["alpha_focal"], dtype=np.float64),
        "n_strata_used": np.asarray(res["n_strata_used"], dtype=np.int64),
        "flagged_bh": np.asarray(res["flagged_bh"], dtype=bool),
    }


def _logistic_rows(res) -> dict[str, np.ndarray]:
    return {
        "item": np.asarray(res["item"], dtype=np.int64),
        "chi2_uniform": np.asarray(res["chi2_uniform"], dtype=np.float64),
        "p_uniform": np.asarray(res["p_uniform"], dtype=np.float64),
        "chi2_nonuniform": np.asarray(res["chi2_nonuniform"], dtype=np.float64),
        "p_nonuniform": np.asarray(res["p_nonuniform"], dtype=np.float64),
        "chi2_total": np.asarray(res["chi2_total"], dtype=np.float64),
        "p_total": np.asarray(res["p_total"], dtype=np.float64),
        "delta_r2": np.asarray(res["delta_r2"], dtype=np.float64),
        "delta_r2_uniform": np.asarray(res["delta_r2_uniform"], dtype=np.float64),
        "jg_class": np.asarray(res["jg_class"]),
        "flagged_bh": np.asarray(res["flagged_bh"], dtype=bool),
        "converged": np.asarray(res["converged"], dtype=bool),
    }


def logistic_dif(
    responses: np.ndarray,
    group: np.ndarray,
    exclude_studied_item: bool = False,
    fdr_q: float = 0.05,
    max_iter: int = 50,
) -> dict[str, np.ndarray]:
    """Zumbo (1999) logistic-regression DIF for dichotomous items (compute in Rust; Swaminathan &
    Rogers, 1990).

    Each item response is regressed on the observed matching score ``S`` (number-correct total, studied
    item included by default), the group ``G``, and their interaction, in three NESTED logistic models:
    ``M0: b0 + b1 S``; ``M1: + b2 G``; ``M2: + b3 (S x G)``. This separates UNIFORM from NON-UNIFORM
    (crossing) DIF — the latter is invisible to :func:`mantel_haenszel_dif`, whose stratified odds-ratio
    test can only detect a consistent group advantage.

    - ``chi2_total`` / ``p_total`` (2 df) is the PRIMARY Swaminathan-Rogers/Zumbo omnibus DIF test and is
      the value Benjamini-Hochberg adjusts (``flagged_bh``).
    - ``chi2_nonuniform`` / ``p_nonuniform`` (1 df) tests the interaction ``b3``.
    - ``chi2_uniform`` / ``p_uniform`` (1 df) tests ``b2`` *assuming* ``b3 = 0``; it is a descriptive
      follow-up, is NOT the group term of the full model, and is not interpretable when non-uniform DIF
      is present. Component p-values are unadjusted.
    - ``delta_r2`` is the Nagelkerke pseudo-R² change ``R2(M2) - R2(M0)`` (Zumbo's effect size), and
      ``jg_class`` classifies it by Jodoin & Gierl (2001): ``"A"`` negligible (< 0.035), ``"B"`` moderate,
      ``"C"`` large (>= 0.070) — forced to ``"A"`` when the omnibus test is not BH-significant, and
      ``"U"`` when undefined. ``delta_r2_uniform`` is an uncalibrated descriptive value with no class.
      (The older Zumbo & Thomas, 1997 cut-offs of 0.13/0.26 are much more conservative.)

    Items whose fits fail (separation, a rank-deficient design, no convergence) report ``NaN``
    statistics with ``converged=False`` and are never flagged. As with Mantel-Haenszel, the studied item
    is included in the matching score and this function does not purify it (see
    :func:`logistic_dif_purified`); logistic-regression DIF
    additionally assumes the logit is linear in ``S``, so a non-uniform flag is not by itself proof of
    crossing item characteristic curves.

    ``responses`` is a persons x items ``0/1`` array (no missing data); ``group`` is length-persons with
    ``0`` = reference and ``1`` = focal. Returns per-item NumPy arrays keyed as above.

    References (APA 7th ed.):
        Jodoin, M. G., & Gierl, M. J. (2001). Evaluating Type I error and power rates using an effect
            size measure with the logistic regression procedure for DIF detection. *Applied Measurement
            in Education, 14*(4), 329-349. https://doi.org/10.1207/S15324818AME1404_2
        Swaminathan, H., & Rogers, H. J. (1990). Detecting differential item functioning using logistic
            regression procedures. *Journal of Educational Measurement, 27*(4), 361-370.
            https://doi.org/10.1111/j.1745-3984.1990.tb00754.x
        Zumbo, B. D. (1999). *A handbook on the theory and methods of differential item functioning
            (DIF)*. Directorate of Human Resources Research and Evaluation.
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "logistic_dif"):
        raise RuntimeError("logistic_dif requires the compiled Rust core")

    y = np.asarray(responses)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    if n_persons == 0 or n_items == 0:
        raise ValueError("responses must contain at least one person and one item")
    yf = np.asarray(y, dtype=np.float64)
    if not np.all(np.isin(yf, (0.0, 1.0))):
        raise ValueError("responses must be 0 or 1 (logistic-regression DIF is for dichotomous items)")
    g = np.asarray(group)
    if g.ndim != 1 or g.shape[0] != n_persons:
        raise ValueError("group must be a length-n_persons 1-D array")
    gf = np.asarray(g, dtype=np.float64)
    if not np.all(np.isin(gf, (0.0, 1.0))):
        raise ValueError("group labels must be 0 (reference) or 1 (focal)")
    if not np.isfinite(fdr_q) or not 0 < fdr_q <= 1:
        raise ValueError("fdr_q must be finite and in (0, 1]")

    res = core.logistic_dif(
        yf.astype(np.int64).reshape(-1),
        gf.astype(np.int64),
        int(n_persons),
        int(n_items),
        bool(exclude_studied_item),
        float(fdr_q),
        int(max_iter),
    )
    return {
        "item": np.asarray(res["item"], dtype=np.int64),
        "chi2_uniform": np.asarray(res["chi2_uniform"], dtype=np.float64),
        "p_uniform": np.asarray(res["p_uniform"], dtype=np.float64),
        "chi2_nonuniform": np.asarray(res["chi2_nonuniform"], dtype=np.float64),
        "p_nonuniform": np.asarray(res["p_nonuniform"], dtype=np.float64),
        "chi2_total": np.asarray(res["chi2_total"], dtype=np.float64),
        "p_total": np.asarray(res["p_total"], dtype=np.float64),
        "delta_r2": np.asarray(res["delta_r2"], dtype=np.float64),
        "delta_r2_uniform": np.asarray(res["delta_r2_uniform"], dtype=np.float64),
        "jg_class": np.asarray(res["jg_class"]),
        "flagged_bh": np.asarray(res["flagged_bh"], dtype=bool),
        "converged": np.asarray(res["converged"], dtype=bool),
    }
