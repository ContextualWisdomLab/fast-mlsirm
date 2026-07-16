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
    let std_p = if sum_w > 0.0 { sum_wdiff / sum_w } else { f64::NAN };
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal LCG + Box-Muller normal (crate PRNG idiom) for the simulation anchors.
    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn normal(&mut self) -> f64 {
            let u1 = self.next_f64().max(1e-12);
            let u2 = self.next_f64();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    /// Build a stratified `(resp, group, matching)` sample from explicit per-stratum `(A,B,C,D)` cells
    /// (ref-correct, ref-incorrect, focal-correct, focal-incorrect), with `matching[p]` = the stratum
    /// index. Lets the deterministic anchor pin the arithmetic without engineering total scores.
    fn build(cells: &[(usize, u64, u64, u64, u64)], n_levels: usize) -> (Vec<u8>, Vec<u8>, Vec<usize>) {
        let (mut resp, mut group, mut matching) = (Vec::new(), Vec::new(), Vec::new());
        let mut push = |g: u8, r: u8, m: usize, n: u64| {
            for _ in 0..n {
                resp.push(r);
                group.push(g);
                matching.push(m);
            }
        };
        for &(m, a, b, c, d) in cells {
            push(0, 1, m, a);
            push(0, 0, m, b);
            push(1, 1, m, c);
            push(1, 0, m, d);
        }
        assert!(n_levels > cells.iter().map(|c| c.0).max().unwrap());
        (resp, group, matching)
    }

    /// Deterministic anchor: two strata hand-computed off Holland & Thayer (1988), the RBG (1986)
    /// variance, and the ETS delta/classification. Pins alpha_MH, the CONTINUITY-CORRECTED chi-square,
    /// MH D-DIF, SE, STD-P-DIF (focal minus reference), and the C label. A dropped `-0.5`, a wrong
    /// variance denominator, a sign flip, or a reference-minus-focal STD-P-DIF all fail here.
    #[test]
    fn mh_two_stratum_hand_anchor() {
        // Stratum 1: A=80 B=20 C=40 D=60; Stratum 2: A=60 B=40 C=30 D=70.
        let (resp, group, matching) =
            build(&[(1, 80, 20, 40, 60), (2, 60, 40, 30, 70)], 3);
        let st = mh_item_stats(&resp, &group, &matching, 3);
        // alpha = (80*60/200 + 60*70/200) / (20*40/200 + 40*30/200) = 45 / 10 = 4.5
        assert!((st.alpha_mh - 4.5).abs() < 1e-12, "alpha {}", st.alpha_mh);
        // D-DIF = -2.35 ln(4.5)
        assert!(
            (st.mh_d_dif - (-2.35 * 4.5_f64.ln())).abs() < 1e-10,
            "d_dif {}",
            st.mh_d_dif
        );
        // chi2 = (|140 - 105| - 0.5)^2 / 24.497487... = 34.5^2 / 24.497487 = 48.5865...
        assert!((st.chi2_mh - 48.58647).abs() < 1e-3, "chi2 {}", st.chi2_mh);
        // SE = 2.35 * sqrt(0.04762963) = 0.512869...
        assert!((st.se_d_dif - 0.512869).abs() < 1e-5, "se {}", st.se_d_dif);
        // STD-P-DIF = (100*(0.4-0.8) + 100*(0.3-0.6)) / 200 = -0.35  (focal - reference, negative)
        assert!((st.std_p_dif - (-0.35)).abs() < 1e-12, "std_p {}", st.std_p_dif);
        assert!(st.p_value < 1e-6, "p {}", st.p_value);
        // |D-DIF|=3.53 >= 1.5 and 3.53 - 1.645*0.5129 = 2.69 > 1.0 and significant -> C
        assert_eq!(st.ets_class, EtsClass::C);
        // sign agreement: both effect sizes negative (against the focal group)
        assert!(st.mh_d_dif < 0.0 && st.std_p_dif < 0.0);
    }

    /// No-DIF symmetry: identical reference/focal conditional response rates within every stratum give
    /// alpha_MH = 1, MH D-DIF = 0, STD-P-DIF = 0, and class A.
    #[test]
    fn mh_no_dif_symmetry() {
        // Each stratum: A/n_R == C/n_F exactly, so every 2x2 has odds ratio 1.
        let (resp, group, matching) =
            build(&[(1, 60, 40, 60, 40), (2, 30, 70, 30, 70)], 3);
        let st = mh_item_stats(&resp, &group, &matching, 3);
        assert!((st.alpha_mh - 1.0).abs() < 1e-12, "alpha {}", st.alpha_mh);
        assert!(st.mh_d_dif.abs() < 1e-10, "d_dif {}", st.mh_d_dif);
        assert!(st.std_p_dif.abs() < 1e-12, "std_p {}", st.std_p_dif);
        assert!(st.chi2_mh < 1e-9, "chi2 {}", st.chi2_mh);
        assert_eq!(st.ets_class, EtsClass::A);
    }

    /// Degenerate guard: a single-group stratum (focal absent) contributes nothing, and a perfectly
    /// separated table (no informative stratum) yields NaN statistics and an Undefined class — NOT A.
    #[test]
    fn mh_degenerate_is_undefined_not_a() {
        // Only a reference group present at level 1 (no focal anywhere) -> no informative strata.
        let (resp, group, matching) = build(&[(1, 30, 20, 0, 0)], 2);
        let st = mh_item_stats(&resp, &group, &matching, 2);
        assert!(st.alpha_mh.is_nan(), "alpha {}", st.alpha_mh);
        assert!(st.mh_d_dif.is_nan(), "d_dif {}", st.mh_d_dif);
        assert!(st.se_d_dif.is_nan(), "se {}", st.se_d_dif);
        assert!(st.chi2_mh.is_nan() && st.p_value.is_nan());
        assert_eq!(st.ets_class, EtsClass::Undefined);

        // Perfect separation: reference always correct, focal always incorrect (sum B_m C_m = 0 ->
        // alpha_MH = +inf). Both groups present and both responses present across strata, so chi2 is
        // defined, but the delta metric is undefined.
        let (resp2, group2, matching2) = build(&[(1, 50, 0, 0, 50)], 2);
        let st2 = mh_item_stats(&resp2, &group2, &matching2, 2);
        assert!(st2.mh_d_dif.is_nan(), "sep d_dif {}", st2.mh_d_dif);
        assert_eq!(st2.ets_class, EtsClass::Undefined);
    }

    /// Simulation anchor: a 2PL DGP with a uniform (b-shift) DIF planted on one item, no group impact.
    /// MH flags the planted item as large (class B/C, BH-significant) with the delta sign matching the
    /// shift (item harder for the focal group -> negative D-DIF, negative STD-P-DIF), and classifies the
    /// clean items as A (negligible). The clean items are asserted by the ETS practical-significance
    /// CLASS, not by the raw BH flag: MH chi-square is over-powered at large N and the DIF item's
    /// presence in the number-correct total mildly contaminates the matching criterion, so a clean
    /// item's chi-square can be BH-significant while its effect size stays negligible (the A/B/C
    /// classification is exactly the guard against this; item purification is the standard remedy and is
    /// out of scope here). The parametric IRT-LR DIF, which does not match on the observed total, is
    /// checked on the planted item plus one clean item for cross-method agreement.
    #[test]
    fn mh_flags_planted_uniform_dif_and_agrees_with_irt_lr() {
        use crate::poly::{poly_dif_sweep, PolyModel};
        let (n, n_items) = (3000usize, 12usize);
        let a = vec![1.2f64; n_items];
        let mut b = vec![0.0f64; n_items];
        for (i, bi) in b.iter_mut().enumerate() {
            *bi = -0.8 + 0.14 * i as f64;
        }
        let dif_item = 6usize;
        let clean_item = 0usize;
        let b_focal_shift = 0.7; // item dif_item is HARDER for the focal group (uniform DIF)
        let mut rng = Lcg(0xD1F);
        let mut y = vec![0u8; n * n_items];
        let mut group = vec![0u8; n];
        for p in 0..n {
            let g = if p % 2 == 0 { 0u8 } else { 1u8 };
            group[p] = g;
            // equal ability distribution across groups (no impact) so DIF is isolated
            let theta = rng.normal();
            for i in 0..n_items {
                let mut bi = b[i];
                if i == dif_item && g == 1 {
                    bi += b_focal_shift;
                }
                let pr = 1.0 / (1.0 + (-(a[i] * (theta - bi))).exp());
                y[p * n_items + i] = if rng.next_f64() < pr { 1 } else { 0 };
            }
        }
        let rows = mantel_haenszel_dif(&y, &group, n, n_items, &MhDifConfig::default()).unwrap();
        // the planted item is flagged and large, harder-for-focal (negative delta + std_p)
        let dr = &rows[dif_item];
        assert!(dr.flagged_bh, "planted item not BH-flagged (p={})", dr.p_value);
        assert!(dr.mh_d_dif < -0.8, "planted delta not large-negative: {}", dr.mh_d_dif);
        assert!(dr.std_p_dif < 0.0, "planted std_p sign: {}", dr.std_p_dif);
        assert!(
            matches!(dr.ets_class, EtsClass::B | EtsClass::C),
            "planted class {:?}",
            dr.ets_class
        );
        // clean items are class A (negligible) by the practical-significance classification
        for (i, r) in rows.iter().enumerate() {
            if i != dif_item {
                assert_eq!(r.ets_class, EtsClass::A, "clean item {i} class {:?}", r.ets_class);
                assert!(r.mh_d_dif.abs() < 1.0, "clean item {i} |delta| {}", r.mh_d_dif);
            }
        }
        // agreement with the parametric IRT-LR DIF (uniform DIF, which MH is designed to catch): both
        // flag the planted item and leave a clean item unflagged. Scoped to two studied items to keep
        // the (per-item multigroup EM) cost bounded.
        let yl: Vec<usize> = y.iter().map(|&v| v as usize).collect();
        let gl: Vec<usize> = group.iter().map(|&v| v as usize).collect();
        let studied = [dif_item, clean_item];
        let lr = poly_dif_sweep(
            &yl, None, &gl, 2, n, n_items, 2, PolyModel::Gpcm, Some(&studied), 21, 200, 1e-5, 0.05,
        )
        .unwrap();
        let lr_dif = lr.iter().find(|r| r.item == dif_item).unwrap();
        let lr_clean = lr.iter().find(|r| r.item == clean_item).unwrap();
        assert!(lr_dif.flagged_bh, "IRT-LR missed the planted item (p={})", lr_dif.p_value);
        assert!(!lr_clean.flagged_bh, "IRT-LR spuriously flagged the clean item");
    }

    /// Validation guards trip non-vacuously.
    #[test]
    fn mh_validates() {
        let n = 20usize;
        let n_items = 4usize;
        let y = vec![1u8; n * n_items];
        let mut group = vec![0u8; n];
        for p in 0..n {
            group[p] = (p % 2) as u8;
        }
        let cfg = MhDifConfig::default();
        // ok baseline (degenerate everywhere but valid input -> Undefined rows, not an error)
        assert!(mantel_haenszel_dif(&y, &group, n, n_items, &cfg).is_ok());
        // response > 1
        let mut ybad = y.clone();
        ybad[0] = 2;
        assert!(mantel_haenszel_dif(&ybad, &group, n, n_items, &cfg).is_err());
        // group label > 1
        let mut gbad = group.clone();
        gbad[0] = 2;
        assert!(mantel_haenszel_dif(&y, &gbad, n, n_items, &cfg).is_err());
        // only one group present
        let gone = vec![0u8; n];
        assert!(mantel_haenszel_dif(&y, &gone, n, n_items, &cfg).is_err());
        // y length mismatch
        assert!(mantel_haenszel_dif(&y[..n * n_items - 1], &group, n, n_items, &cfg).is_err());
        // fdr_q out of range
        let badq = MhDifConfig { fdr_q: 0.0, ..cfg };
        assert!(mantel_haenszel_dif(&y, &group, n, n_items, &badq).is_err());
    }

    /// Rest-score matching (`exclude_studied_item=true`) puts persons in different strata than the
    /// item-included total, so the studied item's MH statistics differ between the two modes and the
    /// rest-score path runs without an out-of-bounds level. A mutation dropping the `- y_i` (leaving the
    /// rest score equal to the total) would make the two modes identical.
    #[test]
    fn mh_rest_score_matching_differs_from_item_included() {
        let (n, n_items) = (1200usize, 6usize);
        let a = 1.2f64;
        let b = [-0.6, -0.3, 0.0, 0.3, 0.6, 0.9];
        let dif_item = 2usize;
        let mut rng = Lcg(0x5E5);
        let mut y = vec![0u8; n * n_items];
        let mut group = vec![0u8; n];
        for p in 0..n {
            let g = (p % 2) as u8;
            group[p] = g;
            let theta = rng.normal();
            for i in 0..n_items {
                let mut bi = b[i];
                if i == dif_item && g == 1 {
                    bi += 1.0;
                }
                let pr = 1.0 / (1.0 + (-(a * (theta - bi))).exp());
                y[p * n_items + i] = if rng.next_f64() < pr { 1 } else { 0 };
            }
        }
        let incl = mantel_haenszel_dif(
            &y,
            &group,
            n,
            n_items,
            &MhDifConfig { exclude_studied_item: false, fdr_q: 0.05 },
        )
        .unwrap();
        let excl = mantel_haenszel_dif(
            &y,
            &group,
            n,
            n_items,
            &MhDifConfig { exclude_studied_item: true, fdr_q: 0.05 },
        )
        .unwrap();
        // rest-score path completes (n_levels correct) and still flags the planted item
        assert!(incl[dif_item].flagged_bh && excl[dif_item].flagged_bh);
        // the studied item's strata genuinely change between the two matching schemes
        assert!(
            (incl[dif_item].chi2_mh - excl[dif_item].chi2_mh).abs() > 1e-6,
            "rest-score identical to item-included: {} vs {}",
            incl[dif_item].chi2_mh,
            excl[dif_item].chi2_mh
        );
    }

    /// ETS A/B/C/Undefined boundaries pinned directly, including the ONE-SIDED 1.645 critical value for
    /// the C rule: at `|D|=1.5, SE=0.28` the `|D| - 1.645 SE = 1.039 > 1.0` test passes (C) but the
    /// `1.96` mutant (`0.951`) would fail (B).
    #[test]
    fn mh_classify_boundaries() {
        assert_eq!(classify(f64::NAN, 0.3, 0.001), EtsClass::Undefined); // undefined delta
        assert_eq!(classify(-3.0, 0.4, 0.20), EtsClass::A); // not significant -> A
        assert_eq!(classify(-0.8, 0.2, 0.001), EtsClass::A); // |D| < 1.0 -> A
        assert_eq!(classify(-1.3, 0.2, 0.001), EtsClass::B); // 1.0 <= |D| < 1.5 -> B
        assert_eq!(classify(-1.5, 0.28, 0.001), EtsClass::C); // C via the 1.645 test (1.96 -> B)
        assert_eq!(classify(-1.6, 1.0, 0.001), EtsClass::B); // |D|>=1.5 but not sig. above 1.0 -> B
    }

    /// STD-P-DIF uses the WIDER "both groups present" stratum gate, not the MH 4-marginal gate: an
    /// all-correct stratum (`m0 = 0`, not MH-informative) still contributes focal weight to the
    /// Dorans-Kulick standardization denominator. Under the stricter gate |STD-P-DIF| would inflate from
    /// `40/150` to `40/100`.
    #[test]
    fn mh_std_p_dif_includes_all_correct_stratum_weight() {
        // Stratum 1 informative (DIF); stratum 2 both-groups all-correct (m0 = 0).
        let (resp, group, matching) = build(&[(1, 80, 20, 40, 60), (2, 50, 0, 50, 0)], 3);
        let st = mh_item_stats(&resp, &group, &matching, 3);
        // STD-P-DIF = (100*(0.4-0.8) + 50*(1.0-1.0)) / (100 + 50) = -40/150
        assert!(
            (st.std_p_dif - (-40.0 / 150.0)).abs() < 1e-12,
            "std_p {}",
            st.std_p_dif
        );
        // MH uses only the informative stratum 1: alpha = (80*60/200)/(20*40/200) = 6
        assert!((st.alpha_mh - 6.0).abs() < 1e-12, "alpha {}", st.alpha_mh);
    }
}
