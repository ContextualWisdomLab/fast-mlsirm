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
    validate_dif_inputs(y, group, n_persons, n_items, cfg)?;

    // Number-correct total per examinee (item-included matching).
    let totals: Vec<usize> = (0..n_persons)
        .map(|p| (0..n_items).map(|j| y[p * n_items + j] as usize).sum())
        .collect();
    // Reusable per-item response and matching-level buffers.
    let mut resp = vec![0u8; n_persons];
    let mut matching = vec![0usize; n_persons];
    // Item-included matching has levels 0..=n_items; the rest score has 0..=n_items-1.
    let n_levels = if cfg.exclude_studied_item {
        n_items // 0..=n_items-1
    } else {
        n_items + 1 // 0..=n_items
    };

    let mut rows: Vec<MhDifRow> = Vec::with_capacity(n_items);
    for i in 0..n_items {
        for p in 0..n_persons {
            let yi = y[p * n_items + i];
            resp[p] = yi;
            matching[p] = if cfg.exclude_studied_item {
                totals[p] - yi as usize
            } else {
                totals[p]
            };
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
// by default and item purification is out of scope, so the criterion carries the same contamination.
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
    if cfg.max_iter == 0 {
        return Err("max_iter must be >= 1".into());
    }
    let mh_cfg = MhDifConfig {
        exclude_studied_item: cfg.exclude_studied_item,
        fdr_q: cfg.fdr_q,
    };
    validate_dif_inputs(y, group, n_persons, n_items, &mh_cfg)?;

    let totals: Vec<f64> = (0..n_persons)
        .map(|p| (0..n_items).map(|j| y[p * n_items + j] as f64).sum())
        .collect();
    let gf: Vec<f64> = group.iter().map(|&g| g as f64).collect();
    let mut resp = vec![0.0f64; n_persons];
    let mut score = vec![0.0f64; n_persons];

    let mut rows: Vec<LogisticDifRow> = Vec::with_capacity(n_items);
    for i in 0..n_items {
        for p in 0..n_persons {
            let yi = y[p * n_items + i] as f64;
            resp[p] = yi;
            score[p] = if cfg.exclude_studied_item {
                totals[p] - yi
            } else {
                totals[p]
            };
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

#[cfg(test)]
#[path = "../../../tests/unit/dif_tests.rs"]
mod tests;
