//! Observed-score differential item functioning by the Mantel-Haenszel procedure.
//!
//! The Mantel-Haenszel (MH) DIF statistic (Holland & Thayer, 1988) tests whether a dichotomous item
//! functions differently for a *reference* and a *focal* group after matching examinees on an observed
//! proficiency score (the number-correct total). It is the observed-score, calibration-free complement
//! to the parametric IRT likelihood-ratio DIF ([`crate::poly::poly_dif_sweep`]): no item response model
//! is fitted; examinees are stratified by matching score and a common odds ratio is estimated across the
//! resulting `2 x 2` (group by response) tables.
//!
//! For a studied item, at each matching level `m` the `2 x 2` table is
//!
//! |            | correct | incorrect | total  |
//! |------------|---------|-----------|--------|
//! | reference  | `A_m`   | `B_m`     | `n_Rm` |
//! | focal      | `C_m`   | `D_m`     | `n_Fm` |
//! | total      | `m1_m`  | `m0_m`    | `T_m`  |
//!
//! and, summing over the DIF-informative strata (all four marginal totals positive, so the
//! hypergeometric variance is positive):
//!
//! - **Common odds ratio** `alpha_MH = (sum_m A_m D_m / T_m) / (sum_m B_m C_m / T_m)`.
//! - **MH chi-square** (continuity-corrected, referred to `chi^2(1)`)
//!   `chi2_MH = max(0, |sum_m A_m - sum_m E(A_m)| - 0.5)^2 / sum_m Var(A_m)`, with
//!   `E(A_m) = n_Rm m1_m / T_m` and `Var(A_m) = n_Rm n_Fm m1_m m0_m / (T_m^2 (T_m - 1))`.
//! - **ETS delta metric** `MH_D-DIF = -2.35 ln(alpha_MH)` (negative = harder for the focal group),
//!   with the Robins-Breslow-Greenland (1986) standard error `SE = 2.35 sqrt(Var(ln alpha_MH))`.
//! - **ETS A/B/C classification** (Zieky, 1993; Dorans & Holland, 1993): A (negligible) if `chi2_MH` is
//!   not significant at .05 or `|MH_D-DIF| < 1.0`; C (large) if `|MH_D-DIF| >= 1.5` and `|MH_D-DIF|` is
//!   significantly above 1.0 (`|MH_D-DIF| - 1.645 SE > 1.0`); B otherwise.
//! - **Standardized P-DIF** (Dorans & Kulick, 1986) `STD_P-DIF = sum_m n_Fm (P_Fm - P_Rm) / sum_m n_Fm`
//!   (focal minus reference, so its sign agrees with `MH_D-DIF`) as a companion effect size.
//!
//! Matching uses the total number-correct *including* the studied item by default (thin matching, the
//! ETS standard; Donoghue, Holland & Thayer, 1993, show it is less biased than the rest score); an
//! option matches on the rest score (studied item excluded), recomputing the strata per item. MH is a
//! *uniform*-DIF detector and is known to miss crossing (non-uniform) DIF that the IRT-LR test catches.
//!
//! # References (APA 7th ed.)
//!
//! Donoghue, J. R., Holland, P. W., & Thayer, D. T. (1993). A Monte Carlo study of factors that affect
//!     the Mantel-Haenszel and standardization measures of differential item functioning. In P. W.
//!     Holland & H. Wainer (Eds.), *Differential item functioning* (pp. 137-166). Erlbaum.
//! Dorans, N. J., & Holland, P. W. (1993). DIF detection and description: Mantel-Haenszel and
//!     standardization. In P. W. Holland & H. Wainer (Eds.), *Differential item functioning* (pp.
//!     35-66). Erlbaum.
//! Dorans, N. J., & Kulick, E. (1986). Demonstrating the utility of the standardization approach to
//!     assessing unexpected differential item performance on the Scholastic Aptitude Test. *Journal of
//!     Educational Measurement, 23*(4), 355-368. https://doi.org/10.1111/j.1745-3984.1986.tb00255.x
//! Holland, P. W., & Thayer, D. T. (1988). Differential item performance and the Mantel-Haenszel
//!     procedure. In H. Wainer & H. I. Braun (Eds.), *Test validity* (pp. 129-145). Erlbaum.
//! Robins, J., Breslow, N., & Greenland, S. (1986). Estimators of the Mantel-Haenszel variance
//!     consistent in both sparse data and large-strata limiting models. *Biometrics, 42*(2), 311-323.
//!     https://doi.org/10.2307/2531052
//! Zieky, M. (1993). Practical questions in the use of DIF statistics in test development. In P. W.
//!     Holland & H. Wainer (Eds.), *Differential item functioning* (pp. 337-347). Erlbaum.

use crate::fitstats::{benjamini_hochberg, chi2_sf};
use crate::lltm::{gram_full_rank, solve_small_checked};
use crate::mmle::{log_sigmoid, sigmoid_stable};

/// ETS delta-metric transform constant (`4 / 1.7`): `MH_D-DIF = -DELTA_SCALE * ln(alpha_MH)`.
pub const DELTA_SCALE: f64 = 2.35;
/// Two-sided significance level for the MH chi-square in the ETS A/B/C rule.
const ALPHA_SIG: f64 = 0.05;
/// One-sided normal critical value for the C-boundary test that `|MH_D-DIF|` exceeds 1.0.
const Z_ONE_SIDED_05: f64 = 1.645;
/// Maximum `n_persons * n_items` cells (denial-of-service guard for a service boundary).
const MAX_CELLS: usize = 200_000_000;

/// ETS DIF severity classification (Zieky, 1993).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EtsClass {
    /// Negligible DIF.
    A,
    /// Moderate DIF.
    B,
    /// Large DIF.
    C,
    /// Undefined — no DIF-informative strata, or a degenerate common odds ratio (`0`/`inf`) so the
    /// delta statistic cannot be computed. NOT the same as A (negligible), which is an affirmative
    /// claim requiring data.
    Undefined,
}

impl EtsClass {
    /// Single-letter code (`"A"`/`"B"`/`"C"`/`"U"`) for serialization.
    pub fn as_str(self) -> &'static str {
        match self {
            EtsClass::A => "A",
            EtsClass::B => "B",
            EtsClass::C => "C",
            EtsClass::Undefined => "U",
        }
    }
}

/// One studied item's Mantel-Haenszel DIF result. `NaN` statistics / `Undefined` class mean the item
/// had no DIF-informative strata or a degenerate common odds ratio (perfect group-by-response
/// separation).
pub struct MhDifRow {
    pub item: usize,
    /// MH common odds ratio (`NaN` if degenerate).
    pub alpha_mh: f64,
    /// Continuity-corrected MH chi-square, `chi^2(1)` (`NaN` if no informative strata).
    pub chi2_mh: f64,
    /// Upper-tail `p`-value of `chi2_mh` (`NaN` if no informative strata).
    pub p_value: f64,
    /// ETS delta-metric DIF `-2.35 ln(alpha_MH)`; negative = harder for the focal group (`NaN` if
    /// degenerate).
    pub mh_d_dif: f64,
    /// Robins-Breslow-Greenland standard error of `mh_d_dif` (`NaN` if degenerate).
    pub se_d_dif: f64,
    /// Standardized P-DIF (Dorans & Kulick, 1986), focal minus reference (`NaN` if no stratum has both
    /// groups present).
    pub std_p_dif: f64,
    /// ETS A/B/C (or Undefined) classification.
    pub ets_class: EtsClass,
    /// Benjamini-Hochberg FDR rejection flag on `p_value` across the swept items.
    pub flagged_bh: bool,
}

/// Configuration for [`mantel_haenszel_dif`].
#[derive(Clone, Copy)]
pub struct MhDifConfig {
    /// Match on the rest score (studied item excluded) instead of the total including the studied item.
    /// The item-included default is the ETS standard (Donoghue, Holland & Thayer, 1993).
    pub exclude_studied_item: bool,
    /// Benjamini-Hochberg FDR level for the across-item flag.
    pub fdr_q: f64,
}

impl Default for MhDifConfig {
    fn default() -> Self {
        Self {
            exclude_studied_item: false,
            fdr_q: 0.05,
        }
    }
}

/// Per-item MH statistics computed from a stratified sample (the calibration-free core, exposed for the
/// deterministic anchor). `resp` and `group` are length `n_persons` (`group`: `0` reference, `1` focal);
/// `matching[p]` is examinee `p`'s matching level in `0..n_levels`.
pub(crate) struct MhItemStats {
    pub alpha_mh: f64,
    pub chi2_mh: f64,
    pub p_value: f64,
    pub mh_d_dif: f64,
    pub se_d_dif: f64,
    pub std_p_dif: f64,
    pub ets_class: EtsClass,
}

pub(crate) fn mh_item_stats(
    resp: &[u8],
    group: &[u8],
    matching: &[usize],
    n_levels: usize,
) -> MhItemStats {
    // Per-stratum 2x2 cell counts: a=ref-correct, b=ref-incorrect, c=focal-correct, d=focal-incorrect.
    let (mut a, mut b, mut c, mut d) = (
        vec![0u64; n_levels],
        vec![0u64; n_levels],
        vec![0u64; n_levels],
        vec![0u64; n_levels],
    );
    for p in 0..resp.len() {
        let m = matching[p];
        match (group[p], resp[p]) {
            (0, 1) => a[m] += 1,
            (0, _) => b[m] += 1,
            (_, 1) => c[m] += 1,
            (_, _) => d[m] += 1,
        }
    }

    // MH accumulators over DIF-informative strata (all four marginals > 0 => Var(A_m) > 0), plus the
    // Robins-Breslow-Greenland variance terms; STD-P-DIF accumulates over the weaker "both groups
    // present" gate (an all-correct/all-incorrect stratum contributes 0 to the difference but still
    // carries focal weight in the Dorans-Kulick standardization).
    let (mut sum_ad, mut sum_bc, mut sum_a, mut sum_e, mut sum_var) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut spr, mut spsqr, mut sqs) = (0.0, 0.0, 0.0);
    let (mut sum_w, mut sum_wdiff) = (0.0, 0.0);
    let mut any_mh = false;
    for m in 0..n_levels {
        let (am, bm, cm, dm) = (a[m] as f64, b[m] as f64, c[m] as f64, d[m] as f64);
        let n_rm = am + bm;
        let n_fm = cm + dm;
        if n_rm > 0.0 && n_fm > 0.0 {
            // STD-P-DIF: focal-minus-reference proportion correct, focal-weighted.
            sum_w += n_fm;
            sum_wdiff += n_fm * (cm / n_fm - am / n_rm);
        }
        let m1 = am + cm;
        let m0 = bm + dm;
        let t = n_rm + n_fm;
        if n_rm > 0.0 && n_fm > 0.0 && m1 > 0.0 && m0 > 0.0 {
            // t >= 2 here (both group totals >= 1), so t - 1 >= 1 and Var(A_m) > 0.
            any_mh = true;
            let r = am * dm / t;
            let s = bm * cm / t;
            sum_ad += r;
            sum_bc += s;
            sum_a += am;
            sum_e += n_rm * m1 / t;
            sum_var += n_rm * n_fm * m1 * m0 / (t * t * (t - 1.0));
            let pp = (am + dm) / t;
            let qq = (bm + cm) / t;
            spr += pp * r;
            spsqr += pp * s + qq * r;
            sqs += qq * s;
        }
    }

    let (chi2, pval) = if any_mh && sum_var > 0.0 {
        let num = ((sum_a - sum_e).abs() - 0.5).max(0.0);
        let chi2 = num * num / sum_var;
        (chi2, chi2_sf(chi2, 1.0))
    } else {
        (f64::NAN, f64::NAN)
    };
    // alpha_MH degenerate (0 or inf) when either running sum is 0 -> delta undefined.
    let alpha = if sum_ad > 0.0 && sum_bc > 0.0 {
        sum_ad / sum_bc
    } else {
        f64::NAN
    };
    let d_dif = if alpha.is_finite() && alpha > 0.0 {
        -DELTA_SCALE * alpha.ln()
    } else {
        f64::NAN
    };
    // RBG denominators sum_ad (= sum R_m) and sum_bc (= sum S_m) are both > 0 exactly when alpha is
    // finite and positive, so a finite d_dif always has a finite SE.
    let se = if sum_ad > 0.0 && sum_bc > 0.0 {
        let var_ln = spr / (2.0 * sum_ad * sum_ad)
            + spsqr / (2.0 * sum_ad * sum_bc)
            + sqs / (2.0 * sum_bc * sum_bc);
        DELTA_SCALE * var_ln.sqrt()
    } else {
        f64::NAN
    };
    let std_p = if sum_w > 0.0 {
        sum_wdiff / sum_w
    } else {
        f64::NAN
    };
    let ets_class = classify(d_dif, se, pval);

    MhItemStats {
        alpha_mh: alpha,
        chi2_mh: chi2,
        p_value: pval,
        mh_d_dif: d_dif,
        se_d_dif: se,
        std_p_dif: std_p,
        ets_class,
    }
}

/// ETS A/B/C rule (Zieky, 1993). Undefined when the delta statistic could not be computed.
fn classify(d_dif: f64, se: f64, p: f64) -> EtsClass {
    if !d_dif.is_finite() {
        return EtsClass::Undefined;
    }
    let significant = p.is_finite() && p < ALPHA_SIG;
    let abs_d = d_dif.abs();
    if !significant || abs_d < 1.0 {
        EtsClass::A
    } else if abs_d >= 1.5 && se.is_finite() && abs_d - Z_ONE_SIDED_05 * se > 1.0 {
        EtsClass::C
    } else {
        EtsClass::B
    }
}

/// Shared input validation for the observed-score DIF entry points: shapes, the `0/1` response and
/// group domains, both groups present, and the FDR level.
fn validate_dif_inputs(
    y: &[u8],
    group: &[u8],
    n_persons: usize,
    n_items: usize,
    cfg: &MhDifConfig,
) -> Result<(), String> {
    if n_persons < 1 || n_items < 1 {
        return Err("n_persons and n_items must be >= 1".into());
    }
    let cells = n_persons
        .checked_mul(n_items)
        .ok_or("n_persons * n_items overflow")?;
    if cells > MAX_CELLS {
        return Err(format!(
            "n_persons * n_items = {cells} exceeds the cap {MAX_CELLS}"
        ));
    }
    if y.len() != cells {
        return Err(format!("y has {} entries; expected {cells}", y.len()));
    }
    if group.len() != n_persons {
        return Err(format!(
            "group has {} entries; expected {n_persons}",
            group.len()
        ));
    }
    if y.iter().any(|&v| v > 1) {
        return Err("y responses must be 0 or 1".into());
    }
    if group.iter().any(|&g| g > 1) {
        return Err("group labels must be 0 (reference) or 1 (focal)".into());
    }
    let (mut has_ref, mut has_focal) = (false, false);
    for &g in group {
        if g == 0 {
            has_ref = true;
        } else {
            has_focal = true;
        }
    }
    if !has_ref || !has_focal {
        return Err("both a reference (0) and a focal (1) group must be present".into());
    }
    if !cfg.fdr_q.is_finite() || cfg.fdr_q <= 0.0 || cfg.fdr_q > 1.0 {
        return Err("fdr_q must be in (0, 1]".into());
    }
    Ok(())
}

/// Mantel-Haenszel DIF sweep (Holland & Thayer, 1988) over the dichotomous items of a two-group sample.
///
/// `y` is a row-major `n_persons * n_items` `0/1` response array; `group` is length `n_persons` with
/// `0` = reference and `1` = focal (both must be present). Every item is swept against the
/// total-score matching variable; Benjamini-Hochberg controls the FDR at `cfg.fdr_q`. Returns one
/// [`MhDifRow`] per item.
pub fn mantel_haenszel_dif(
    y: &[u8],
    group: &[u8],
    n_persons: usize,
    n_items: usize,
    cfg: &MhDifConfig,
) -> Result<Vec<MhDifRow>, String> {
    mh_sweep(y, group, n_persons, n_items, cfg, None)
}

/// Per-person base score for the matching criterion: the full number-correct total when `anchor` is
/// `None` (the shipped behaviour), else the anchor-subtest total.
fn base_scores(y: &[u8], n_persons: usize, n_items: usize, anchor: Option<&[bool]>) -> Vec<usize> {
    (0..n_persons)
        .map(|p| {
            (0..n_items)
                .filter(|&j| anchor.map_or(true, |m| m[j]))
                .map(|j| y[p * n_items + j] as usize)
                .sum()
        })
        .collect()
}

/// Matching score for studied item `i` from its `base` (see [`base_scores`]). The criterion is the sum
/// over `anchor UNION {i}`: the studied item is added back only when it is NOT itself an anchor item,
/// so an anchor item is never double-counted. `exclude_studied_item` then removes it again, giving the
/// pure rest/anchor score. With `anchor = None` this reduces exactly to the shipped total-score rule.
#[inline]
fn matching_for_item(base: usize, yi: usize, in_anchor: bool, exclude_studied: bool) -> usize {
    let mut s = base;
    if !in_anchor {
        s += yi;
    }
    if exclude_studied {
        s -= yi;
    }
    s
}

/// The Mantel-Haenszel sweep, optionally against a purified (anchor-only) matching criterion.
/// `anchor = None` reproduces [`mantel_haenszel_dif`] exactly.
fn mh_sweep(
    y: &[u8],
    group: &[u8],
    n_persons: usize,
    n_items: usize,
    cfg: &MhDifConfig,
    anchor: Option<&[bool]>,
) -> Result<Vec<MhDifRow>, String> {
    validate_dif_inputs(y, group, n_persons, n_items, cfg)?;

    let base = base_scores(y, n_persons, n_items, anchor);
    // Reusable per-item response and matching-level buffers.
    let mut resp = vec![0u8; n_persons];
    let mut matching = vec![0usize; n_persons];
    // Any anchor-based score is bounded by n_items (an item added back was excluded from the anchor),
    // and empty strata are skipped by the marginal gates in `mh_item_stats`, so one level count serves
    // every item and every anchor.
    let n_levels = n_items + 1;

    let mut rows: Vec<MhDifRow> = Vec::with_capacity(n_items);
    for i in 0..n_items {
        let in_anchor = anchor.map_or(true, |m| m[i]);
        for p in 0..n_persons {
            let yi = y[p * n_items + i];
            resp[p] = yi;
            matching[p] =
                matching_for_item(base[p], yi as usize, in_anchor, cfg.exclude_studied_item);
        }
        let st = mh_item_stats(&resp, group, &matching, n_levels);
        rows.push(MhDifRow {
            item: i,
            alpha_mh: st.alpha_mh,
            chi2_mh: st.chi2_mh,
            p_value: st.p_value,
            mh_d_dif: st.mh_d_dif,
            se_d_dif: st.se_d_dif,
            std_p_dif: st.std_p_dif,
            ets_class: st.ets_class,
            flagged_bh: false,
        });
    }

    let pvals: Vec<f64> = rows.iter().map(|r| r.p_value).collect();
    let flags = benjamini_hochberg(&pvals, cfg.fdr_q);
    for (r, &f) in rows.iter_mut().zip(&flags) {
        r.flagged_bh = f;
    }
    Ok(rows)
}

// ===================== Zumbo (1999) logistic regression DIF ==========================
//
// The logistic-regression DIF procedure (Swaminathan & Rogers, 1990; Zumbo, 1999) regresses the item
// response on the observed matching score, the group, and their interaction, in three NESTED models:
//
// ```text
//   M0: logit P(Y=1) = b0 + b1 S
//   M1: logit P(Y=1) = b0 + b1 S + b2 G                 (adds the group main effect)
//   M2: logit P(Y=1) = b0 + b1 S + b2 G + b3 (S x G)    (adds the interaction)
// ```
//
// The 2-df omnibus `chi2_total = 2[ll(M2) - ll(M0)]` is the PRIMARY Swaminathan-Rogers / Zumbo DIF
// decision. The 1-df components are DESCRIPTIVE follow-ups: `chi2_nonuniform = 2[ll(M2) - ll(M1)]` is
// the unambiguous test of the interaction `b3`, but `chi2_uniform = 2[ll(M1) - ll(M0)]` tests `b2` in a
// model that ASSUMES `b3 = 0` — it is *not* the test of the group term in the full model M2, and it is
// not interpretable as "uniform DIF" when non-uniform DIF is present (under a crossing item, M1 is
// misspecified and `b2` absorbs a data-dependent mixture of the two effects). The hierarchical entry
// order `S -> G -> S x G` is therefore load-bearing: reversing it changes the uniform component.
// Benjamini-Hochberg is applied to the omnibus `p_total` only; the component p-values are unadjusted.
//
// Effect size: the Nagelkerke (1991) pseudo-`R^2` change `delta_r2 = R2_N(M2) - R2_N(M0)`, with
// `R2_CS(M) = 1 - exp(2(ll_null - ll(M))/n)` and `R2_N(M) = R2_CS(M) / (1 - exp(2 ll_null / n))`
// (`ll_null` is the INTERCEPT-ONLY fit, not M0). Items are classified by the Jodoin & Gierl (2001)
// thresholds on that quantity — A (negligible) `< 0.035`, B `0.035..0.070`, C (large) `>= 0.070` — and
// only when the omnibus test is BH-significant (a non-significant item is A by definition). The older
// Zumbo & Thomas (1997) cut-offs (`0.13` / `0.26`) are considerably more conservative on the same
// quantity. The uniform-only `delta_r2_uniform` is reported as an UNCALIBRATED descriptive number and
// carries no letter class: the Jodoin-Gierl cut-offs were calibrated on the 2-df quantity.
//
// Caveats, same as the Mantel-Haenszel path above: the studied item is INCLUDED in the matching score
// by default, and this entry point does no purification, so its criterion carries the same
// contamination (see `logistic_dif_purified` and the purification notes further down).
// Logistic-regression DIF additionally assumes the logit is LINEAR in the matching score — curvature in
// the true regression, or group differences in the score distribution interacting with that curvature,
// can be absorbed by the `S x G` term, so a non-uniform flag is not by itself evidence of crossing ICCs.
//
// # References (APA 7th ed.)
//
// Jodoin, M. G., & Gierl, M. J. (2001). Evaluating Type I error and power rates using an effect size
//     measure with the logistic regression procedure for DIF detection. *Applied Measurement in
//     Education, 14*(4), 329-349. https://doi.org/10.1207/S15324818AME1404_2
// Nagelkerke, N. J. D. (1991). A note on a general definition of the coefficient of determination.
//     *Biometrika, 78*(3), 691-692. https://doi.org/10.1093/biomet/78.3.691
// Swaminathan, H., & Rogers, H. J. (1990). Detecting differential item functioning using logistic
//     regression procedures. *Journal of Educational Measurement, 27*(4), 361-370.
//     https://doi.org/10.1111/j.1745-3984.1990.tb00754.x
// Zumbo, B. D. (1999). *A handbook on the theory and methods of differential item functioning (DIF)*.
//     Directorate of Human Resources Research and Evaluation, Department of National Defense.
// Zumbo, B. D., & Thomas, D. R. (1997). *A measure of effect size for a model-based approach for
//     studying DIF*. Prince George, Canada: University of Northern British Columbia, Edgeworth
//     Laboratory for Quantitative Behavioral Science.

/// Jodoin & Gierl (2001) `delta_r2` boundary between negligible (A) and moderate (B) DIF.
pub const JG_MODERATE: f64 = 0.035;
/// Jodoin & Gierl (2001) `delta_r2` boundary between moderate (B) and large (C) DIF.
pub const JG_LARGE: f64 = 0.070;
/// Coefficient-magnitude bound; exceeding it signals (quasi-)separation rather than a real fit.
const LOGIT_COEF_BOUND: f64 = 30.0;
/// Minimum persons for a logistic DIF fit: the full model M2 has four parameters, and the conventional
/// floor of ~5 observations per parameter keeps small-sample separation from masquerading as a result.
const MIN_LOGIT_N: usize = 20;

/// One studied item's logistic-regression DIF result. All statistics are `NaN` and `converged` is
/// `false` when the item's nested fits failed (separation, a rank-deficient design, or no convergence);
/// such an item is never BH-flagged and is classified `Undefined`.
pub struct LogisticDifRow {
    pub item: usize,
    /// `2[ll(M1) - ll(M0)]`, `chi^2(1)`. DESCRIPTIVE: tests `b2` assuming `b3 = 0`.
    pub chi2_uniform: f64,
    /// Unadjusted upper-tail `p` for `chi2_uniform`.
    pub p_uniform: f64,
    /// `2[ll(M2) - ll(M1)]`, `chi^2(1)`: the test of the interaction `b3` (non-uniform DIF).
    pub chi2_nonuniform: f64,
    /// Unadjusted upper-tail `p` for `chi2_nonuniform`.
    pub p_nonuniform: f64,
    /// `2[ll(M2) - ll(M0)]`, `chi^2(2)`: the PRIMARY omnibus DIF test.
    pub chi2_total: f64,
    /// Upper-tail `p` for `chi2_total` (the value Benjamini-Hochberg adjusts).
    pub p_total: f64,
    /// Nagelkerke `R2_N(M2) - R2_N(M0)`: the Zumbo (1999) DIF effect size.
    pub delta_r2: f64,
    /// Nagelkerke `R2_N(M1) - R2_N(M0)`: UNCALIBRATED descriptive value, carries no letter class.
    pub delta_r2_uniform: f64,
    /// Jodoin & Gierl (2001) A/B/C class on `delta_r2`; forced to `A` when the omnibus test is not
    /// BH-significant, and `Undefined` when the fits failed.
    pub jg_class: EtsClass,
    /// Benjamini-Hochberg rejection on `p_total` across the swept items.
    pub flagged_bh: bool,
    /// `true` only if all four nested fits converged.
    pub converged: bool,
}

/// Configuration for [`logistic_dif`].
#[derive(Clone, Copy)]
pub struct LogisticDifConfig {
    /// Match on the rest score (studied item excluded) instead of the item-included total.
    pub exclude_studied_item: bool,
    /// Benjamini-Hochberg FDR level applied to `p_total`.
    pub fdr_q: f64,
    /// Maximum IRLS/Newton iterations per nested model.
    pub max_iter: usize,
}

impl Default for LogisticDifConfig {
    fn default() -> Self {
        Self {
            exclude_studied_item: false,
            fdr_q: 0.05,
            max_iter: 50,
        }
    }
}

/// Logistic log-likelihood `sum_p [y log sigmoid(eta) + (1-y) log sigmoid(-eta)]`, computed with the
/// stable `log_sigmoid` so a separated fit saturates smoothly instead of producing `NaN`.
fn logit_loglik(x: &[f64], y: &[f64], n: usize, m: usize, b: &[f64]) -> f64 {
    let mut ll = 0.0;
    for p in 0..n {
        let mut eta = 0.0;
        for c in 0..m {
            eta += x[p * m + c] * b[c];
        }
        ll += if y[p] > 0.5 {
            log_sigmoid(eta)
        } else {
            log_sigmoid(-eta)
        };
    }
    ll
}

/// IRLS / Newton logistic regression with step-halving. `x` is the row-major `n x m` design, `init` the
/// warm start. Returns `(coefficients, loglik)`, or `None` on a singular information matrix, a
/// coefficient blow-up (separation), or failure to converge within `max_iter`.
fn logit_fit(
    x: &[f64],
    y: &[f64],
    n: usize,
    m: usize,
    init: &[f64],
    max_iter: usize,
) -> Option<(Vec<f64>, f64)> {
    let mut b = init.to_vec();
    let mut ll = logit_loglik(x, y, n, m, &b);
    if !ll.is_finite() {
        return None;
    }
    let mut ll_prev = f64::NEG_INFINITY;
    for _ in 0..max_iter {
        // score X'(y - mu) and information X'WX (the NEGATIVE Hessian, positive definite)
        let mut grad = vec![0.0f64; m];
        let mut info = vec![vec![0.0f64; m]; m];
        for p in 0..n {
            let mut eta = 0.0;
            for c in 0..m {
                eta += x[p * m + c] * b[c];
            }
            let mu = sigmoid_stable(eta);
            let w = mu * (1.0 - mu);
            let r = y[p] - mu;
            for a in 0..m {
                let xa = x[p * m + a];
                grad[a] += xa * r;
                for c in 0..m {
                    info[a][c] += w * xa * x[p * m + c];
                }
            }
        }
        // Newton ascent: b += (X'WX)^{-1} X'(y - mu); break (never fall back to a gradient step) on a
        // singular information matrix, which is what separation produces.
        let step = solve_small_checked(info, grad)?;
        let mut scale = 1.0f64;
        let mut advanced = false;
        for _ in 0..40 {
            let cand: Vec<f64> = (0..m).map(|c| b[c] + scale * step[c]).collect();
            if cand
                .iter()
                .any(|v| !v.is_finite() || v.abs() > LOGIT_COEF_BOUND)
            {
                scale *= 0.5;
                continue;
            }
            let ll_c = logit_loglik(x, y, n, m, &cand);
            if ll_c.is_finite() && ll_c >= ll - 1e-12 {
                b = cand;
                ll = ll_c;
                advanced = true;
                break;
            }
            scale *= 0.5;
        }
        if !advanced {
            return None; // could not ascend: (quasi-)separation
        }
        // Standard GLM convergence on the RELATIVE log-likelihood change. A gradient tolerance cannot be
        // used here: near the optimum the per-step likelihood gain falls below the f64 resolution of
        // `ll` itself, so the attainable score floor is ~sqrt(info * eps * |ll|) — for n in the
        // thousands that is O(1e-5), far above any absolute threshold worth calling "converged".
        if (ll - ll_prev).abs() <= 1e-10 * (ll.abs() + 0.1) {
            // ... but a fit pinned against the coefficient bound is SEPARATED, not converged: its
            // likelihood also stops changing as the coefficients run away, so the relative criterion
            // alone would certify a non-existent MLE.
            if b.iter().any(|v| v.abs() >= 0.99 * LOGIT_COEF_BOUND) {
                return None;
            }
            return Some((b, ll));
        }
        ll_prev = ll;
    }
    None // iterations exhausted without meeting the convergence criterion
}

/// Per-item nested-model statistics (before the BH-dependent classification).
struct LogitStats {
    chi2_uniform: f64,
    chi2_nonuniform: f64,
    chi2_total: f64,
    delta_r2: f64,
    delta_r2_uniform: f64,
    converged: bool,
}

const LOGIT_UNDEFINED: LogitStats = LogitStats {
    chi2_uniform: f64::NAN,
    chi2_nonuniform: f64::NAN,
    chi2_total: f64::NAN,
    delta_r2: f64::NAN,
    delta_r2_uniform: f64::NAN,
    converged: false,
};

/// Fit the four nested models (intercept-only null, M0, M1, M2) for one item on ONE identical
/// person subsample, warm-starting each from the previous fit so the nested log-likelihoods are
/// monotone, and return the LR components and Nagelkerke effect sizes.
fn logistic_item_stats(
    resp: &[f64],
    score: &[f64],
    group: &[f64],
    n: usize,
    max_iter: usize,
) -> LogitStats {
    // M2 has four parameters. At `n` barely above that the data are (quasi-)separated with high
    // probability, and a separated fit that happens to terminate would be reported as a converged,
    // significant result; require the conventional minimum of ~5 observations per parameter.
    if n < MIN_LOGIT_N {
        return LOGIT_UNDEFINED;
    }
    // An item with no response variation (everyone correct or everyone incorrect) carries no DIF
    // information at all: every model is separated, the intercept diverges, and `ll_null -> 0` makes the
    // Nagelkerke normalizer degenerate. Reject it up front rather than letting a bounded, half-converged
    // separated fit produce a meaningful-looking result.
    let n_correct = resp.iter().filter(|&&v| v > 0.5).count();
    if n_correct == 0 || n_correct == n {
        return LOGIT_UNDEFINED;
    }
    // Mean-center the matching score: the chi-squares are invariant to this affine reparameterization,
    // but the raw total puts the S x G column on a J^2 scale against the ones-column and leaves the
    // 4 x 4 Gram near-singular.
    let sbar = score.iter().sum::<f64>() / n as f64;
    let mut x = vec![0.0f64; n * 4];
    for p in 0..n {
        let s = score[p] - sbar;
        x[p * 4] = 1.0;
        x[p * 4 + 1] = s;
        x[p * 4 + 2] = group[p];
        x[p * 4 + 3] = s * group[p];
    }
    // Design rank check on the full M2 design (constant score column, a group with no variation, or an
    // S x G column collinear with G all show up here) rather than relying on the solver failing later.
    let mut gram = vec![vec![0.0f64; 4]; 4];
    let mut maxg = 0.0f64;
    for a in 0..4 {
        for c in 0..4 {
            let mut s = 0.0;
            for p in 0..n {
                s += x[p * 4 + a] * x[p * 4 + c];
            }
            gram[a][c] = s;
            maxg = maxg.max(s.abs());
        }
    }
    if !gram_full_rank(&mut gram, 4, 1e-9 * maxg.max(1e-300)) {
        return LOGIT_UNDEFINED;
    }
    // Column-sliced designs for the nested models (M0/M1/M2 take the first 2/3/4 columns).
    let sub = |m: usize| -> Vec<f64> {
        let mut d = vec![0.0f64; n * m];
        for p in 0..n {
            d[p * m..(p + 1) * m].copy_from_slice(&x[p * 4..p * 4 + m]);
        }
        d
    };
    let (x1, x2c, x3, x4) = (sub(1), sub(2), sub(3), sub(4));
    let warm = |prev: &[f64], m: usize| -> Vec<f64> {
        let mut v = vec![0.0f64; m];
        v[..prev.len()].copy_from_slice(prev);
        v
    };
    let (b_null, ll_null) = match logit_fit(&x1, resp, n, 1, &[0.0], max_iter) {
        Some(r) => r,
        None => return LOGIT_UNDEFINED,
    };
    let (b0, ll0) = match logit_fit(&x2c, resp, n, 2, &warm(&b_null, 2), max_iter) {
        Some(r) => r,
        None => return LOGIT_UNDEFINED,
    };
    let (b1, ll1) = match logit_fit(&x3, resp, n, 3, &warm(&b0, 3), max_iter) {
        Some(r) => r,
        None => return LOGIT_UNDEFINED,
    };
    let (_b2, ll2) = match logit_fit(&x4, resp, n, 4, &warm(&b1, 4), max_iter) {
        Some(r) => r,
        None => return LOGIT_UNDEFINED,
    };
    // Nagelkerke normalizer: 1 - exp(2 ll_null / n). An item answered identically by everyone has
    // ll_null = 0, making this 0 and R2_CS a 0/0 - report undefined rather than a spurious 0.
    // A successful intercept-only fit contains both response classes. Together
    // with the public cell cap, that makes this Nagelkerke normalizer strictly
    // positive and bounded away from floating-point zero.
    let denom = 1.0 - (2.0 * ll_null / n as f64).exp();
    let r2n = |ll: f64| (1.0 - (2.0 * (ll_null - ll) / n as f64).exp()) / denom;
    LogitStats {
        chi2_uniform: (2.0 * (ll1 - ll0)).max(0.0),
        chi2_nonuniform: (2.0 * (ll2 - ll1)).max(0.0),
        chi2_total: (2.0 * (ll2 - ll0)).max(0.0),
        delta_r2: r2n(ll2) - r2n(ll0),
        delta_r2_uniform: r2n(ll1) - r2n(ll0),
        converged: true,
    }
}

/// Jodoin & Gierl (2001) classification, conditional on omnibus significance.
fn jg_classify(delta_r2: f64, significant: bool) -> EtsClass {
    if !delta_r2.is_finite() {
        return EtsClass::Undefined;
    }
    if !significant {
        return EtsClass::A; // non-significant => negligible by definition
    }
    if delta_r2 >= JG_LARGE {
        EtsClass::C
    } else if delta_r2 >= JG_MODERATE {
        EtsClass::B
    } else {
        EtsClass::A
    }
}

/// Zumbo (1999) logistic-regression DIF sweep over the dichotomous items of a two-group sample.
///
/// Unlike [`mantel_haenszel_dif`], which is a stratified odds-ratio test sensitive only to UNIFORM DIF,
/// this procedure separates uniform from NON-UNIFORM (crossing) DIF through the `score x group`
/// interaction. `y` is a row-major `n_persons * n_items` `0/1` array; `group` is length `n_persons`
/// with `0` = reference and `1` = focal. Returns one [`LogisticDifRow`] per item; see the module notes
/// above for the interpretation of the 1-df components and the effect-size classification.
pub fn logistic_dif(
    y: &[u8],
    group: &[u8],
    n_persons: usize,
    n_items: usize,
    cfg: &LogisticDifConfig,
) -> Result<Vec<LogisticDifRow>, String> {
    logistic_sweep(y, group, n_persons, n_items, cfg, None)
}

/// The logistic-regression sweep, optionally against a purified (anchor-only) matching criterion.
/// `anchor = None` reproduces [`logistic_dif`] exactly.
fn logistic_sweep(
    y: &[u8],
    group: &[u8],
    n_persons: usize,
    n_items: usize,
    cfg: &LogisticDifConfig,
    anchor: Option<&[bool]>,
) -> Result<Vec<LogisticDifRow>, String> {
    if cfg.max_iter == 0 {
        return Err("max_iter must be >= 1".into());
    }
    let mh_cfg = MhDifConfig {
        exclude_studied_item: cfg.exclude_studied_item,
        fdr_q: cfg.fdr_q,
    };
    validate_dif_inputs(y, group, n_persons, n_items, &mh_cfg)?;

    let base = base_scores(y, n_persons, n_items, anchor);
    let gf: Vec<f64> = group.iter().map(|&g| g as f64).collect();
    let mut resp = vec![0.0f64; n_persons];
    let mut score = vec![0.0f64; n_persons];

    let mut rows: Vec<LogisticDifRow> = Vec::with_capacity(n_items);
    for i in 0..n_items {
        let in_anchor = anchor.map_or(true, |m| m[i]);
        for p in 0..n_persons {
            let yi = y[p * n_items + i];
            resp[p] = yi as f64;
            score[p] =
                matching_for_item(base[p], yi as usize, in_anchor, cfg.exclude_studied_item) as f64;
        }
        let st = logistic_item_stats(&resp, &score, &gf, n_persons, cfg.max_iter);
        // A failed fit must yield a NaN p-value, NOT 1.0. `chi2_sf` maps a NaN statistic to 1.0
        // (`f64::max` ignores NaN, so `NaN.max(0.0) == 0.0` and the survival function is 1 there), which
        // would both contradict the NaN contract and — because 1.0 is finite — make Benjamini-Hochberg
        // COUNT the unfittable item in `m`, shrinking the threshold and costing power on real DIF items.
        let sf = |c: f64, df: f64| {
            if c.is_finite() {
                chi2_sf(c, df)
            } else {
                f64::NAN
            }
        };
        rows.push(LogisticDifRow {
            item: i,
            chi2_uniform: st.chi2_uniform,
            p_uniform: sf(st.chi2_uniform, 1.0),
            chi2_nonuniform: st.chi2_nonuniform,
            p_nonuniform: sf(st.chi2_nonuniform, 1.0),
            chi2_total: st.chi2_total,
            p_total: sf(st.chi2_total, 2.0),
            delta_r2: st.delta_r2,
            delta_r2_uniform: st.delta_r2_uniform,
            jg_class: EtsClass::Undefined,
            flagged_bh: false,
            converged: st.converged,
        });
    }
    // BH on the omnibus p only; NaN p-values (failed fits) are skipped by benjamini_hochberg.
    let pvals: Vec<f64> = rows.iter().map(|r| r.p_total).collect();
    let flags = benjamini_hochberg(&pvals, cfg.fdr_q);
    for (r, &f) in rows.iter_mut().zip(&flags) {
        r.flagged_bh = f;
        r.jg_class = jg_classify(r.delta_r2, f);
    }
    Ok(rows)
}

// ===================== Iterative item purification ==========================
//
// Both sweeps above match on the number-correct total, which CONTAINS the items being tested. Items
// with DIF therefore contaminate the matching criterion, inflating the Type I error rate for clean
// items and attenuating power for genuine ones. Purification rebuilds the criterion from the currently
// UNFLAGGED (anchor) items and re-runs the sweep, iterating until the flagged set stops changing:
// Candell and Drasgow (1988) proposed iterating to stability, while the single-pass "two-stage"
// procedure traces to Lord (1980) and Holland and Thayer's (1988) recommendation of one re-run. Gains
// past the first round or two are small (Clauser, Mazor & Hambleton, 1993; Fidalgo, Mellenbergh &
// Muniz, 2000), hence a small default round cap.
//
// The criterion for a studied item is the sum over `anchor UNION {studied}` — item-included matching is
// what makes the null-DIF condition hold (Holland & Thayer, 1988; Zwick, 1990) — so a flagged item is
// removed from every OTHER item's criterion but still scored against itself plus the anchor.
//
// IMPORTANT LIMITS, none of which purification removes:
//
// - The final anchor is SELECTED FROM THE SAME DATA that is then tested against it, so the reported
//   p-values are conditional on a data-dependent selection. They are not guaranteed super-uniform under
//   the null, and Benjamini-Hochberg does NOT control the FDR at `fdr_q` for a purified sweep. Purified
//   flags are a SCREENING device, not an error-rate guarantee.
// - Purification REDUCES rather than removes contamination, and can fail outright when DIF is
//   unbalanced in direction (Wang & Su, 2004): anchor quality dominates.
// - Mantel-Haenszel's blind spot for crossing DIF is inherited, and purification cannot repair it: an
//   item MH does not flag stays in the anchor every round and keeps contaminating the "purified"
//   criterion. The blindness is NOT a property of non-uniform DIF as such but of the signed area
//   between the two curves over the matched ability distribution (Wang & Su, 2004): a crossing at the
//   centre of that distribution cancels and is invisible, while the same item with its crossing off
//   centre leaves a net signed difference that MH detects (empirically: `a_ref = 2.0` vs `a_foc = 0.4`
//   with equal `b`, standard-normal ability in both groups, crossing at `theta = 0` gives `D-DIF` about
//   0.00 and class `A`; moving the crossing to `theta = +0.8` gives `D-DIF` about +1.80 and class `C`).
//   So MH purification is unreliable, not uniformly blind, whenever non-uniform DIF is plausible — use
//   the logistic variant, whose interaction term tests the crossing directly.
// - The two procedures do not inherit matched Type I control: `MhDifRow::ets_class` is conditioned on
//   the raw .05 MH p-value while `LogisticDifRow::jg_class` is conditioned on the BH flag.
//
// # References (APA 7th ed.)
//
// Candell, G. L., & Drasgow, F. (1988). An iterative procedure for linking metrics and assessing item
//     bias in item response theory. *Applied Psychological Measurement, 12*(3), 253-260.
//     https://doi.org/10.1177/014662168801200304
// Clauser, B., Mazor, K., & Hambleton, R. K. (1993). The effects of purification of the matching
//     criterion on the identification of DIF using the Mantel-Haenszel procedure. *Applied Measurement
//     in Education, 6*(4), 269-279. https://doi.org/10.1207/s15324818ame0604_2
// Fidalgo, A. M., Mellenbergh, G. J., & Muniz, J. (2000). Effects of amount of DIF, test length, and
//     purification type on robustness and power of Mantel-Haenszel procedures. *Methods of Psychological
//     Research Online, 5*(3), 43-53.
// Lord, F. M. (1980). *Applications of item response theory to practical testing problems*. Erlbaum.
// Zwick, R. (1990). When do item response function and Mantel-Haenszel definitions of differential item
//     functioning coincide? *Journal of Educational Statistics, 15*(3), 185-197.
//     https://doi.org/10.3102/10769986015003185
// Wang, W.-C., & Su, Y.-H. (2004). Effects of average signed area between two item characteristic
//     curves and test purification procedures on the DIF detection via the Mantel-Haenszel method.
//     *Applied Measurement in Education, 17*(2), 113-144.
//     https://doi.org/10.1207/s15324818ame1702_2

/// Configuration for the purification loop wrapping a DIF sweep.
#[derive(Clone, Copy)]
pub struct PurifyConfig {
    /// Maximum purification rounds after the initial full-test sweep. Gains past one or two rounds are
    /// small, so this is deliberately small rather than open-ended.
    pub max_rounds: usize,
    /// Minimum anchor items required to keep purifying. A short number-correct criterion is coarse and
    /// unreliable, which itself inflates Mantel-Haenszel Type I error (Donoghue, Holland & Thayer,
    /// 1993); there is no canonical minimum in the literature, so this is a guard, not a standard.
    pub min_anchor_items: usize,
}

impl Default for PurifyConfig {
    fn default() -> Self {
        Self {
            max_rounds: 3,
            min_anchor_items: 4,
        }
    }
}

/// Outcome of a purified DIF sweep: the final per-item rows plus what the purification actually did.
///
/// The `p_value`/`flagged_bh` fields of `rows` are conditional on `anchor`, which was selected from the
/// same data — see the module notes: they do NOT carry an FDR guarantee and are a screening device.
pub struct PurifiedDif<R> {
    /// Per-item rows from the final sweep (against the `anchor` criterion below).
    pub rows: Vec<R>,
    /// The anchor mask the final rows were computed against (`true` = used in the matching criterion).
    pub anchor: Vec<bool>,
    /// Anchor items in `anchor`.
    pub n_anchor: usize,
    /// Purification rounds performed after the initial full-test sweep (`0` = no purification applied).
    pub rounds: usize,
    /// `true` when the flagged set stabilised; `false` when the round cap was hit (including an
    /// oscillating flag set) or purification stopped on the anchor guards, in which case `rows` are
    /// simply the last computed round.
    pub converged: bool,
    /// `stable_flag_set`, `max_rounds_reached`, or `insufficient_anchor_items`.
    pub termination_reason: &'static str,
}

/// Generic purification loop. `sweep` runs a DIF sweep against an anchor mask (`None` = the whole
/// test); `is_flagged` decides which rows are removed from the next round's criterion.
///
/// Round 0 passes `None`, i.e. it dispatches to the *same* code path as the unpurified entry point
/// rather than to an all-`true` mask that merely evaluates the same way. That is also why nothing is
/// sized from a caller-supplied item count before the first sweep: `n_items` is untrusted at the FFI
/// boundary, and every dimension check lives inside the sweep. The anchor mask is allocated from the
/// returned row count, so it can only be as large as a validated sweep.
fn purify_loop<R>(
    cfg: &PurifyConfig,
    mut sweep: impl FnMut(Option<&[bool]>) -> Result<Vec<R>, String>,
    is_flagged: impl Fn(&R) -> bool,
) -> Result<PurifiedDif<R>, String> {
    if cfg.max_rounds == 0 {
        return Err("max_rounds must be >= 1".into());
    }
    let mut rows = sweep(None)?;
    let mut flagged: Vec<bool> = rows.iter().map(&is_flagged).collect();
    let mut anchor = vec![true; flagged.len()];

    let mut rounds = 0usize;
    let mut converged = false;
    let mut termination_reason = "max_rounds_reached";
    while rounds < cfg.max_rounds {
        let next: Vec<bool> = flagged.iter().map(|&f| !f).collect();
        if next == anchor {
            converged = true; // flagged set stable: the criterion would not change
            termination_reason = "stable_flag_set";
            break;
        }
        let n_anchor = next.iter().filter(|&&a| a).count();
        // Guard BEFORE sweeping: never match on an empty or uselessly short criterion. A constant
        // anchor total would also put everyone in one stratum.
        // ponytail: a 1-2 item anchor is equally meaningless but only the configured floor is checked.
        if n_anchor < cfg.min_anchor_items.max(1) {
            termination_reason = "insufficient_anchor_items";
            break; // keep the last usable rows; converged stays false
        }
        anchor = next;
        rows = sweep(Some(&anchor))?;
        rounds += 1;
        let new_flagged: Vec<bool> = rows.iter().map(&is_flagged).collect();
        if new_flagged == flagged {
            converged = true;
            termination_reason = "stable_flag_set";
            break;
        }
        flagged = new_flagged;
    }
    let n_anchor = anchor.iter().filter(|&&a| a).count();
    Ok(PurifiedDif {
        rows,
        anchor,
        n_anchor,
        rounds,
        converged,
        termination_reason,
    })
}

/// An item is removed from the purified criterion only on PRACTICAL significance (`B` or `C`).
/// Deliberately not `class != A`: `Undefined` is also `!= A`, and an unfittable item carries no evidence
/// of DIF, so purging it would shrink the anchor for free. Deliberately not the raw BH flag either: the
/// Mantel-Haenszel chi-square is over-powered at large N (see [`EtsClass`], whose whole purpose is that
/// practical-significance screen), which is exactly the regime where purification matters.
#[inline]
fn purify_flagged(class: EtsClass) -> bool {
    matches!(class, EtsClass::B | EtsClass::C)
}

/// [`mantel_haenszel_dif`] with an iteratively purified matching criterion (see the module notes on
/// purification, including why the resulting p-values carry no FDR guarantee).
pub fn mantel_haenszel_dif_purified(
    y: &[u8],
    group: &[u8],
    n_persons: usize,
    n_items: usize,
    cfg: &MhDifConfig,
    purify: &PurifyConfig,
) -> Result<PurifiedDif<MhDifRow>, String> {
    purify_loop(
        purify,
        |anchor| mh_sweep(y, group, n_persons, n_items, cfg, anchor),
        |r: &MhDifRow| purify_flagged(r.ets_class),
    )
}

/// [`logistic_dif`] with an iteratively purified matching criterion. The purification flag is taken
/// from `jg_class`, i.e. from the 2-df omnibus test that Benjamini-Hochberg already targets.
pub fn logistic_dif_purified(
    y: &[u8],
    group: &[u8],
    n_persons: usize,
    n_items: usize,
    cfg: &LogisticDifConfig,
    purify: &PurifyConfig,
) -> Result<PurifiedDif<LogisticDifRow>, String> {
    purify_loop(
        purify,
        |anchor| logistic_sweep(y, group, n_persons, n_items, cfg, anchor),
        |r: &LogisticDifRow| purify_flagged(r.jg_class),
    )
}

// ============================ SIBTEST (uniform) ==============================
//
// The third observed-score DIF procedure in this module, and the only one that corrects the MATCHING
// CRITERION ITSELF rather than the comparison built on top of it.
//
// Mantel-Haenszel and the logistic sweep both match on an observed number-correct score. That score is
// unreliable, and under IMPACT (a real group difference in ability) two examinees from different groups
// with the same OBSERVED score do not have the same EXPECTED TRUE score — each regresses toward its own
// group mean. Matching on the raw observed score therefore compares non-equivalent examinees and
// produces DIF statistics for items that have none. Item purification cannot repair this: the defect is
// the regression of true score on observed score, not which items are in the sum, so a perfectly
// purified criterion is still biased. SIBTEST transports each group's conditional mean from its own
// estimated true score to a common target before comparing, which is the entire point of the procedure.
//
// The correction, per retained matching level `k` (`R` = reference/group 0, `F` = focal/group 1):
//
// - `V*_Gk = [Xbar_G + alpha_G (k - Xbar_G)] / n_valid` — Kelley's regressed estimate of group `G`'s
//   true valid-subtest score at observed level `k`, with `alpha_G` that group's coefficient alpha
//   (KR-20) on the valid subtest and `Xbar_G` its mean valid score.
// - `V*_k = (V*_Rk + V*_Fk) / 2` — the common target, an UNWEIGHTED midpoint.
// - `M_Gj = (Ybar_G[j+1] - Ybar_G[j-1]) / (V*_G[j+1] - V*_G[j-1])` — a per-level central difference over
//   the group's OWN true-score scale, taken at adjacent OBSERVED level positions (not at `k +/- 1`:
//   levels with no examinees are absent, so arithmetic on `k` would silently use the wrong spacing).
// - `Ybar*_Gk = Ybar_Gk + M_Gj (V*_k - V*_Gk)` — BOTH endpoints of the transport are true-score
//   quantities. Subtracting the OBSERVED mean here instead would collapse the whole correction to
//   `(M_R - M_F)(V*_k - k)`, which vanishes exactly in the null-DIF-with-impact case the method exists
//   to fix.
// - `beta_uni = sum_k p_k (Ybar*_Rk - Ybar*_Fk)` with `p_k` the COMBINED-sample proportion at level `k`,
//   renormalized over the retained strata.
// - `se_beta = sqrt(sum_k p_k^2 [s2_Fk/J_Fk + s2_Rk/J_Rk])`, `b_uni = beta_uni / se_beta`, referred to
//   `chi^2(1)` as `b_uni^2` (identical to the two-sided normal test, and reuses `chi2_sf`).
//
// SIGN, and it is the OPPOSITE of the rest of this module: `beta_uni > 0` means the item is HARDER FOR
// THE FOCAL GROUP, because the estimator is reference-minus-focal. `mh_d_dif` and `std_p_dif` above are
// focal-oriented and go NEGATIVE in that same situation. The orientation is kept rather than harmonised
// because published `|beta_uni|` cut-offs assume it; a differently-signed quantity carrying the name
// `beta_uni` would be the larger error. Cross-method comparisons must flip one of the two.
//
// WHEN TO PREFER IT — and the honest answer is "rarely, on the evidence measured here". This
// implementation was compared against `mantel_haenszel_dif` on the same simulated data, 500
// replications per cell, 2PL, no DIF planted, so every rejection is a false positive. These are the
// exact cells `sibtest_type_i_error_exceeds_mantel_haenszel_under_impact` runs and the rates it
// prints, so the table is regenerable from the repo rather than quoted from a vanished study:
//
//     impact  n(per group)  items   MH Type I   SIBTEST Type I
//     0.0     1000          5       .044        .056
//     1.0     1000          5       .046        .086
//
// SIBTEST's Type I error is ABOVE nominal in both cells and roughly DOUBLE Mantel-Haenszel's under
// impact — the opposite of the ordering one might expect from the motivation above. The cause is the
// next bullet: the correction is estimated but its variance is not propagated, so `se_beta` is
// optimistic and the correction's own noise is charged to the signal. This is a property of the 1993
// estimator rather than a transcription slip — the closed-form anchors reproduce the TRANSCRIBED
// FORMULAS in exact rational arithmetic (`mirt` itself was never executed; see the provenance note
// below) — and it is precisely what Jiang and Stout's (1998) paper, titled "Improved Type I error
// control and reduced estimation bias for DIF detection using SIBTEST", was written to fix. Prefer
// `mantel_haenszel_dif` or `logistic_dif` for routine screening. Reach for this when you specifically
// want the regression-corrected estimand, and read `beta_uni` as an effect size rather than trusting
// `p_value` as a calibrated test.
//
// KNOWN LIMITATIONS, none of them silent:
// - `se_beta` treats the regression correction as FIXED. It is estimated (through `alpha_G` and the
//   local slopes) and that estimation error is NOT propagated, so the standard error is optimistic and
//   the test over-rejects — see the measured table above.
// - The shipped correction is the single linear one. Jiang and Stout's (1998) two-segment piecewise
//   correction is a different, later estimator and is not implemented.
// - No guessing correction.
// - Dichotomous only (`validate_dif_inputs` rejects responses above 1).
// - No effect-size letter class. Published cut-offs disagree across sources and none was verified
//   against its primary text, so the raw effect size ships uncalibrated — the same decision already
//   taken for `delta_r2_uniform` above.
// - No purified variant. `purify_loop` composes with this in a few lines, but purification needs a
//   PRACTICAL-significance predicate and there is no verified cut-off to build one from; flagging on
//   the BH flag alone would contradict this module's own reasoning that the test is over-powered at
//   large N. Purification would also shorten the valid subtest, lowering the very `alpha_G` the
//   correction divides by. Deliberately omitted, not overlooked.
// - Crossing (non-uniform) DIF is NOT covered. Crossing-SIBTEST was evaluated and deliberately not
//   built: Chalmers (2018) shows Li and Stout's (1996) hypothesis test is insufficient, and no
//   normal-theory referral for a crossing statistic is valid. Use `logistic_dif`, whose `S x G`
//   interaction tests crossing directly against a standard 1-df null.
//
// PROVENANCE OF THE FORMULAS ABOVE — stated plainly because it bounds what this code may claim. Every
// equation is transcribed from the reference implementation (Chalmers, 2012; the `SIBTEST` routine of
// the `mirt` package), which attributes them to Shealy and Stout (1993). THE PRIMARY TEXT WAS NOT
// CONSULTED, and neither was `mirt` EXECUTED: the transcription was made by reading its source. The
// closed-form anchors therefore verify this code against the transcribed formulas, NOT against `mirt`
// output — no cross-implementation agreement is claimed anywhere, and the empty-neighbour divergence
// below means exact agreement would not hold on all inputs even if it were tested. Where this
// implementation diverges from that reference it is marked in the code.
//
// # References (APA 7th ed.)
//
// Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R environment.
//     *Journal of Statistical Software, 48*(6), 1-29. https://doi.org/10.18637/jss.v048.i06
// Chalmers, R. P. (2018). Improving the crossing-SIBTEST statistic for detecting non-uniform DIF.
//     *Psychometrika, 83*(2), 376-386. https://doi.org/10.1007/s11336-017-9583-8
// DeMars, C. E. (2009). Modification of the Mantel-Haenszel and logistic regression DIF procedures to
//     incorporate the SIBTEST regression correction. *Journal of Educational and Behavioral Statistics,
//     34*(2), 149-170. https://doi.org/10.3102/1076998607313923
// Jiang, H., & Stout, W. (1998). Improved Type I error control and reduced estimation bias for DIF
//     detection using SIBTEST. *Journal of Educational and Behavioral Statistics, 23*(4), 291-322.
//     https://doi.org/10.3102/10769986023004291
// Li, H.-H., & Stout, W. (1996). A new procedure for detection of crossing DIF. *Psychometrika, 61*(4),
//     647-677. https://doi.org/10.1007/BF02294041
// Shealy, R., & Stout, W. (1993). A model-based standardization approach that separates true
//     bias/DIF from group ability differences and detects test bias/DTF as well as item bias/DIF.
//     *Psychometrika, 58*(2), 159-194. https://doi.org/10.1007/BF02294572

/// Configuration for [`sibtest`].
#[derive(Clone, Copy)]
pub struct SibtestConfig {
    /// Benjamini-Hochberg FDR level for the across-item flag.
    pub fdr_q: f64,
    /// Minimum examinees per group per matching level, STRICTLY exceeded for a level to be retained.
    pub j_min: usize,
}

impl Default for SibtestConfig {
    fn default() -> Self {
        Self {
            fdr_q: 0.05,
            j_min: 5,
        }
    }
}

/// One studied item's uniform SIBTEST result. `NaN` statistics mean the item carried no usable strata
/// or a degenerate reliability — never a silent `0.0` (which would read as an affirmative no-DIF claim).
pub struct SibtestRow {
    pub item: usize,
    /// `beta_uni`, the regression-corrected weighted mean difference. **Positive = harder for the FOCAL
    /// group** (reference minus focal), the OPPOSITE of `mh_d_dif`/`std_p_dif` in this same module.
    pub beta_uni: f64,
    /// Standard error of `beta_uni` (treats the regression correction as fixed; see the module notes).
    pub se_beta: f64,
    /// `beta_uni / se_beta`; its square is referred to `chi^2(1)`.
    pub b_uni: f64,
    /// Upper-tail `p`-value of `b_uni^2` on `chi^2(1)`.
    pub p_value: f64,
    /// Coefficient alpha of the valid subtest in the reference group. Exposed because the correction
    /// DIVIDES by it: a low or unstable alpha inflates the local slope and hence the correction.
    pub alpha_ref: f64,
    /// Coefficient alpha of the valid subtest in the focal group.
    pub alpha_focal: f64,
    /// Matching levels that survived the retention rule.
    pub n_strata_used: usize,
    /// Benjamini-Hochberg FDR rejection flag on `p_value` across the swept items.
    pub flagged_bh: bool,
}

/// One matching level's sufficient statistics for [`sibtest_stats`]. Raw moments rather than
/// correct-counts, so the core computes its own unbiased within-cell variance.
pub(crate) struct SibCell {
    pub level: usize,
    pub j_r: u64,
    pub sum_y_r: f64,
    pub sum_y2_r: f64,
    pub j_f: u64,
    pub sum_y_f: f64,
    pub sum_y2_f: f64,
}

pub(crate) struct SibtestStats {
    pub beta_uni: f64,
    pub se_beta: f64,
    pub b_uni: f64,
    pub p_value: f64,
    pub n_strata_used: usize,
}

const SIB_UNDEFINED: SibtestStats = SibtestStats {
    beta_uni: f64::NAN,
    se_beta: f64::NAN,
    b_uni: f64::NAN,
    p_value: f64::NAN,
    n_strata_used: 0,
};

/// Coefficient alpha (KR-20) of the valid subtest within one group:
/// `alpha = k/(k-1) * (1 - sum_j var(y_j) / var(sum_j y_j))`.
///
/// Biased and unbiased variances cancel in the ratio, so both arguments must merely use the SAME
/// convention. Returns a non-finite value on a degenerate subtest, which the caller gates on.
#[inline]
fn coefficient_alpha(item_var_sum: f64, total_var: f64, n_valid: usize) -> f64 {
    let k = n_valid as f64;
    (k / (k - 1.0)) * (1.0 - item_var_sum / total_var)
}

/// Uniform SIBTEST from one item's per-level sufficient statistics (the calibration-free core, exposed
/// for the deterministic anchors). `cells` must be sorted by ascending `level` and contain only levels
/// OBSERVED in the combined sample.
pub(crate) fn sibtest_stats(
    cells: &[SibCell],
    alpha_r: f64,
    alpha_f: f64,
    n_valid: usize,
    j_min: u64,
) -> SibtestStats {
    // The correction divides by alpha through the local slope, so a non-positive or non-finite
    // reliability makes the whole statistic meaningless. Reject rather than clamp.
    if !(alpha_r.is_finite() && alpha_r > 0.0 && alpha_f.is_finite() && alpha_f > 0.0) {
        return SIB_UNDEFINED;
    }
    // Central differences need an interior level, so at least three observed levels must exist.
    if n_valid < 2 || cells.len() < 3 {
        return SIB_UNDEFINED;
    }

    let (mut n_r, mut n_f, mut sx_r, mut sx_f) = (0u64, 0u64, 0.0f64, 0.0f64);
    for c in cells {
        n_r += c.j_r;
        n_f += c.j_f;
        sx_r += c.level as f64 * c.j_r as f64;
        sx_f += c.level as f64 * c.j_f as f64;
    }
    if n_r == 0 || n_f == 0 {
        return SIB_UNDEFINED;
    }
    let (xbar_r, xbar_f) = (sx_r / n_r as f64, sx_f / n_f as f64);

    let nv = n_valid as f64;
    // Kelley's regressed true-score estimate, on the proportion-correct scale of the valid subtest.
    let vstar = |xbar: f64, alpha: f64, level: usize| (xbar + alpha * (level as f64 - xbar)) / nv;
    let ybar = |sum: f64, j: u64| sum / j as f64;
    // Unbiased within-cell variance from raw moments.
    let cell_var = |sum: f64, sum2: f64, j: u64| {
        let jf = j as f64;
        (sum2 - sum * sum / jf) / (jf - 1.0)
    };

    // A level is retained when both groups are populated well enough for a variance AND both
    // neighbouring positions can supply a slope. `j_min` is exceeded STRICTLY.
    let retained = |j: usize| -> bool {
        // Endpoints have no two-sided neighbours; this also keeps `j - 1` and `j + 1` in range below.
        if j == 0 || j + 1 == cells.len() {
            return false;
        }
        let c = &cells[j];
        if !(c.j_r > j_min && c.j_f > j_min) {
            return false;
        }
        if !(cell_var(c.sum_y_r, c.sum_y2_r, c.j_r) > 0.0
            && cell_var(c.sum_y_f, c.sum_y2_f, c.j_f) > 0.0)
        {
            return false;
        }
        // DIVERGENCE from the reference implementation, deliberate: it imputes an empty group-by-level
        // cell's mean to 0.0 and feeds that fabricated zero into the neighbouring central difference,
        // producing a finite but meaningless slope. Drop the level instead.
        let (lo, hi) = (&cells[j - 1], &cells[j + 1]);
        lo.j_r >= 1 && lo.j_f >= 1 && hi.j_r >= 1 && hi.j_f >= 1
    };

    // Pass 1: the renormalizing constant for the combined-sample weights.
    let mut w_total = 0.0f64;
    let mut n_strata_used = 0usize;
    for j in 0..cells.len() {
        if retained(j) {
            w_total += (cells[j].j_r + cells[j].j_f) as f64;
            n_strata_used += 1;
        }
    }
    if n_strata_used == 0 || !(w_total > 0.0) {
        return SIB_UNDEFINED;
    }

    // Pass 2: accumulate the estimate and its variance.
    let (mut beta, mut var_beta) = (0.0f64, 0.0f64);
    for j in 0..cells.len() {
        if !retained(j) {
            continue;
        }
        let (c, lo, hi) = (&cells[j], &cells[j - 1], &cells[j + 1]);
        let p_k = (c.j_r + c.j_f) as f64 / w_total;

        let vr = vstar(xbar_r, alpha_r, c.level);
        let vf = vstar(xbar_f, alpha_f, c.level);
        let target = 0.5 * (vr + vf);

        // Central difference over each group's OWN true-score scale, at adjacent OBSERVED positions.
        // The denominator is `alpha_G * (level[j+1] - level[j-1]) / n_valid`, which is NOT
        // `2 * alpha_G / n_valid` unless the levels happen to be contiguous.
        let m_r = (ybar(hi.sum_y_r, hi.j_r) - ybar(lo.sum_y_r, lo.j_r))
            / (vstar(xbar_r, alpha_r, hi.level) - vstar(xbar_r, alpha_r, lo.level));
        let m_f = (ybar(hi.sum_y_f, hi.j_f) - ybar(lo.sum_y_f, lo.j_f))
            / (vstar(xbar_f, alpha_f, hi.level) - vstar(xbar_f, alpha_f, lo.level));

        let ystar_r = ybar(c.sum_y_r, c.j_r) + m_r * (target - vr);
        let ystar_f = ybar(c.sum_y_f, c.j_f) + m_f * (target - vf);
        beta += p_k * (ystar_r - ystar_f);

        let s2_r = cell_var(c.sum_y_r, c.sum_y2_r, c.j_r);
        let s2_f = cell_var(c.sum_y_f, c.sum_y2_f, c.j_f);
        var_beta += p_k * p_k * (s2_f / c.j_f as f64 + s2_r / c.j_r as f64);
    }

    let se_beta = var_beta.sqrt();
    if !beta.is_finite() || !(se_beta > 0.0) || !se_beta.is_finite() {
        return SIB_UNDEFINED;
    }
    let b_uni = beta / se_beta;
    // Guard BEFORE squaring: a finite-but-huge ratio overflows to `inf`, and `chi2_sf(inf, .)` is NaN.
    let p_value = if b_uni.is_finite() {
        chi2_sf(b_uni * b_uni, 1.0)
    } else {
        f64::NAN
    };
    SibtestStats {
        beta_uni: beta,
        se_beta,
        b_uni,
        p_value,
        n_strata_used,
    }
}

/// Uniform SIBTEST sweep (Shealy & Stout, 1993, as implemented in Chalmers, 2012 — see the module
/// notes, including the provenance statement and why `beta_uni`'s sign is the opposite of
/// `mh_d_dif`'s).
///
/// Each item in turn is the studied subtest `S = {i}`; the valid subtest `V` is every OTHER item, so
/// `V` and `S` are DISJOINT by construction. That is a property of the estimator rather than a
/// configuration option, and it is the opposite of the item-included Mantel-Haenszel default above
/// (Donoghue, Holland & Thayer, 1993), which is equally deliberate there.
pub fn sibtest(
    y: &[u8],
    group: &[u8],
    n_persons: usize,
    n_items: usize,
    cfg: &SibtestConfig,
) -> Result<Vec<SibtestRow>, String> {
    validate_dif_inputs(
        y,
        group,
        n_persons,
        n_items,
        &MhDifConfig {
            exclude_studied_item: false,
            fdr_q: cfg.fdr_q,
        },
    )?;
    // The valid subtest is every other item, and coefficient alpha needs at least two of them.
    if n_items < 3 {
        return Err("sibtest requires n_items >= 3 (the valid subtest needs >= 2 items)".into());
    }
    if cfg.j_min < 2 {
        return Err("j_min must be >= 2 (a within-level variance needs two examinees)".into());
    }

    let base = base_scores(y, n_persons, n_items, None);
    let n_valid = n_items - 1;
    let j_min = cfg.j_min as u64;

    // Per-group per-item response moments, for the coefficient-alpha numerator. One pass, reused by
    // every item: the numerator for item `i` is the total minus item `i`'s own contribution.
    let mut n_g = [0u64; 2];
    let mut item_sum = [vec![0.0f64; n_items], vec![0.0f64; n_items]];
    for p in 0..n_persons {
        let g = group[p] as usize;
        n_g[g] += 1;
        for i in 0..n_items {
            item_sum[g][i] += y[p * n_items + i] as f64;
        }
    }
    if n_g[0] == 0 || n_g[1] == 0 {
        return Err("both groups must be present".into());
    }
    // Population (biased) variance throughout; the convention cancels in alpha's ratio.
    let item_var: Vec<[f64; 2]> = (0..n_items)
        .map(|i| {
            let mut v = [0.0f64; 2];
            for g in 0..2 {
                // A 0/1 item's mean square equals its mean, so var = mean - mean^2.
                let m = item_sum[g][i] / n_g[g] as f64;
                v[g] = m - m * m;
            }
            v
        })
        .collect();
    let item_var_total: [f64; 2] = [
        item_var.iter().map(|v| v[0]).sum(),
        item_var.iter().map(|v| v[1]).sum(),
    ];

    let mut rows: Vec<SibtestRow> = Vec::with_capacity(n_items);
    // `level = base - y_i` lies in `0..n_items`, so one dense accumulator serves every item.
    let n_levels = n_items;
    let mut acc: Vec<[f64; 6]> = vec![[0.0; 6]; n_levels];
    for i in 0..n_items {
        for slot in acc.iter_mut() {
            *slot = [0.0; 6];
        }
        // Rest-score moments for alpha's denominator, accumulated in the same pass as the cells.
        let mut rest_sum = [0.0f64; 2];
        let mut rest_sq = [0.0f64; 2];
        for p in 0..n_persons {
            let g = group[p] as usize;
            let yi = y[p * n_items + i] as f64;
            let level = base[p] - y[p * n_items + i] as usize;
            let s = &mut acc[level];
            let off = if g == 0 { 0 } else { 3 };
            s[off] += 1.0;
            s[off + 1] += yi;
            s[off + 2] += yi * yi;
            let rest = level as f64;
            rest_sum[g] += rest;
            rest_sq[g] += rest * rest;
        }
        let cells: Vec<SibCell> = (0..n_levels)
            .filter(|&l| acc[l][0] + acc[l][3] > 0.0)
            .map(|l| SibCell {
                level: l,
                j_r: acc[l][0] as u64,
                sum_y_r: acc[l][1],
                sum_y2_r: acc[l][2],
                j_f: acc[l][3] as u64,
                sum_y_f: acc[l][4],
                sum_y2_f: acc[l][5],
            })
            .collect();

        let mut alpha = [f64::NAN; 2];
        for g in 0..2 {
            let n = n_g[g] as f64;
            let mean = rest_sum[g] / n;
            let total_var = rest_sq[g] / n - mean * mean;
            alpha[g] = coefficient_alpha(item_var_total[g] - item_var[i][g], total_var, n_valid);
        }

        let st = sibtest_stats(&cells, alpha[0], alpha[1], n_valid, j_min);
        rows.push(SibtestRow {
            item: i,
            beta_uni: st.beta_uni,
            se_beta: st.se_beta,
            b_uni: st.b_uni,
            p_value: st.p_value,
            alpha_ref: alpha[0],
            alpha_focal: alpha[1],
            n_strata_used: st.n_strata_used,
            flagged_bh: false,
        });
    }

    let pvals: Vec<f64> = rows.iter().map(|r| r.p_value).collect();
    for (r, f) in rows.iter_mut().zip(benjamini_hochberg(&pvals, cfg.fdr_q)) {
        r.flagged_bh = f;
    }
    Ok(rows)
}

#[cfg(test)]
#[path = "../../../tests/unit/dif_tests.rs"]
mod tests;
