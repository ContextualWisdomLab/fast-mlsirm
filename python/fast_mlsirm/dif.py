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
