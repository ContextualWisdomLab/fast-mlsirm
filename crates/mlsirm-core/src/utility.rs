//! Selection utility analysis for personnel decisions.
//!
//! Implements the three classical selection-utility results under the standard
//! bivariate-normal predictor-criterion model:
//!
//! - Taylor-Russell success ratio (Taylor & Russell, 1939),
//! - Naylor-Shine mean standardized criterion of the selected group
//!   (Naylor & Shine, 1965),
//! - Brogden-Cronbach-Gleser linear utility gain (Cronbach & Gleser, 1965).
//!
//! # Model
//!
//! Predictor `X` and criterion `Y` are standard bivariate normal with
//! correlation `rxy` (the validity coefficient). Top-down selection retains
//! `X > xc` with selection ratio `sr`, so `xc = Phi^-1(1 - sr)`. For
//! Taylor-Russell, "success" is `Y > yc` with base rate `br`, so
//! `yc = Phi^-1(1 - br)`.
//!
//! Verified formulas (adversarially checked against a scipy oracle and the
//! CRAN iopsych 0.90.1 `R/utility.R` + `R/ai.R` source, which was read in
//! full; see the crate test module for the pinned oracle values):
//!
//! - selection intensity `ux(sr) = phi(xc) / sr` (truncated-normal mean
//!   `E[X | X > xc]`; iopsych `ux`, ai.R line 221),
//! - Naylor-Shine `pux = rxy * ux(sr) = E[Y | X > xc]` (from
//!   `Y = rxy X + sqrt(1-rxy^2) Z` with `Z` independent),
//! - BCG utility gain `n * period * sdy * pux - cost_total` (iopsych
//!   `utilityBcg`; note iopsych labels `cost` "per applicant" but never
//!   multiplies by `n`, so the argument here is documented as a TOTAL cost),
//! - Taylor-Russell success ratio `P(Y > yc | X > xc) = Q(xc, yc, rxy) / sr`,
//!   where `Q(h, k, rho) = P(X > h, Y > k)`. iopsych's `trModel` computes the
//!   equivalent `qa / (qa + qb)`; the identity `qa + qb = sr` was verified
//!   both algebraically and numerically during spec review.
//!
//! `Q` is evaluated by the elementary conditional-normal representation
//! `Q(h, k, rho) = int_h^inf phi(x) Phi((rho x - k)/sqrt(1-rho^2)) dx`
//! with composite Gauss-Legendre quadrature. Accuracy vs the scipy oracle
//! (`tests/oracles/oracle_utility.py`): ~1e-15 at moderate `|rho|`
//! (spec-review fixtures, asymmetric points incl. negative rho), and
//! better than 1e-6 across the whole accepted `|rho|` range
//! (regression-tested at `|rho| = 0.999999`; observed error ~1e-11 there).
//!
//! # Scope
//!
//! REDUCED SCOPE relative to the broader utility literature: only the three
//! models above are implemented. The Boudreau (1983) financial extension and
//! Raju-Burke-Normand (1990) model in iopsych are NOT implemented here.
//! `|rxy| >= 1` is rejected (degenerate BVN); use the analytic limits
//! documented in the spec if needed.
//!
//! # References
//!
//! Cronbach, L. J., & Gleser, G. C. (1965). *Psychological tests and
//! personnel decisions* (2nd ed.). University of Illinois Press.
//!
//! Naylor, J. C., & Shine, L. C. (1965). A table for determining the increase
//! in mean criterion score obtained by using a selection device. *Journal of
//! Industrial Psychology, 3*(2), 33-42.
//!
//! Taylor, H. C., & Russell, J. T. (1939). The relationship of validity
//! coefficients to the practical effectiveness of tests in selection:
//! Discussion and tables. *Journal of Applied Psychology, 23*(5), 565-578.
//! <https://doi.org/10.1037/h0057079>
//!
//! (Citation note: the Naylor-Shine and Taylor-Russell papers' formulas are
//! standard truncated-normal/BVN results that were independently re-derived
//! and verified numerically; the implementation transcribes the iopsych
//! source, which was actually read. The Cronbach-Gleser page-level formula
//! was confirmed against iopsych's `utilityBcg`, not the book itself.)

use crate::fitstats::erfc;
use crate::nodes::inv_normal_cdf;

/// Result of a Brogden-Cronbach-Gleser / Naylor-Shine utility analysis.
#[derive(Debug, Clone, Copy)]
pub struct SelectionUtilityResult {
    /// Predictor cutoff `xc = Phi^-1(1 - sr)`.
    pub xc: f64,
    /// Selection intensity `ux = phi(xc) / sr = E[X | X > xc]`.
    pub ux: f64,
    /// Naylor-Shine mean standardized criterion of the selected group,
    /// `pux = rxy * ux`.
    pub pux: f64,
    /// Brogden-Cronbach-Gleser utility gain
    /// `n * period * sdy * pux - cost_total`.
    pub utility_gain: f64,
}

/// Result of a Taylor-Russell (1939) analysis.
#[derive(Debug, Clone, Copy)]
pub struct TaylorRussellResult {
    /// Success ratio `P(Y > yc | X > xc)` among those selected.
    pub success_ratio: f64,
    /// The base rate `br` supplied (echoed for reporting).
    pub base_rate: f64,
    /// Joint upper-tail probability `Q(xc, yc, rxy) = P(X > xc, Y > yc)`.
    pub q_joint: f64,
}

/// Standard normal pdf.
#[inline]
fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Standard normal CDF via the crate's `erfc` approximation.
#[inline]
fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Upper-tail bivariate-normal probability `Q(h, k, rho) = P(X > h, Y > k)`
/// for standard margins, via the conditional-normal representation
/// `int_h^inf phi(x) Phi((rho x - k)/sqrt(1 - rho^2)) dx` (elementary result;
/// ~1e-15 vs scipy's BVN CDF at moderate `|rho|` during spec review, and
/// regression-tested to better than 1e-6 at the accepted `|rho|` extremes —
/// see `tests/oracles/oracle_utility.py`).
///
/// Composite Gauss-Legendre quadrature (16-point panels) over
/// `[h, max(h, 8) + 2]`; beyond that upper limit `phi(x) < 1e-22` and the
/// truncation error is negligible relative to the `erfc` approximation error.
/// The panel width is `min(0.25, sqrt(1 - rho^2) / 2)` so that the
/// conditional-CDF transition (scale `sqrt(1 - rho^2)`) is always resolved;
/// with a fixed 0.25 width, `|rho|` near 1 was inaccurate by ~1e-3
/// (impl-review finding). Callers must reject `sqrt(1 - rho^2) < 1e-4`
/// (`|rho| > ~0.999999995`) before calling; below that the panel count
/// needed for accuracy is unbounded.
fn bvn_upper(h: f64, k: f64, rho: f64) -> f64 {
    // 16-point Gauss-Legendre nodes/weights on [-1, 1] (Abramowitz & Stegun
    // Table 25.4, symmetric halves).
    const GL_X: [f64; 8] = [
        0.0950125098376374,
        0.2816035507792589,
        0.4580167776572274,
        0.6178762444026438,
        0.7554044083550030,
        0.8656312023878318,
        0.9445750230732326,
        0.9894009349916499,
    ];
    const GL_W: [f64; 8] = [
        0.1894506104550685,
        0.1826034150449236,
        0.1691565193950025,
        0.1495959888165767,
        0.1246289712555339,
        0.0951585116824928,
        0.0622535239386479,
        0.0271524594117541,
    ];
    let s = (1.0 - rho * rho).sqrt();
    let upper = h.max(8.0) + 2.0;
    let panel_w = 0.25f64.min(0.5 * s);
    let n_panels = ((upper - h) / panel_w).ceil() as usize;
    let mut q = 0.0f64;
    for p in 0..n_panels {
        let a = h + p as f64 * panel_w;
        let b = (a + panel_w).min(upper);
        let c = 0.5 * (a + b);
        let half = 0.5 * (b - a);
        for j in 0..8 {
            for sign in [-1.0f64, 1.0] {
                let x = c + sign * half * GL_X[j];
                q += GL_W[j] * half * norm_pdf(x) * norm_cdf((rho * x - k) / s);
            }
        }
    }
    q.clamp(0.0, 1.0)
}

fn check_unit_open(v: f64, name: &str) -> Result<(), String> {
    if !v.is_finite() || v <= 0.0 || v >= 1.0 {
        return Err(format!("{name} must be strictly inside (0, 1), got {v}"));
    }
    // Sub-ulp guard: for v below ~half an ulp at 1.0, `1.0 - v` rounds to
    // exactly 1.0 and `inv_normal_cdf(1.0)` returns NaN, which previously
    // leaked as a NaN/silent-zero Ok result (impl-review HIGH finding).
    if 1.0 - v == 1.0 {
        return Err(format!(
            "{name} = {v} is too close to 0 to resolve in f64 (1.0 - {name} rounds to 1.0)"
        ));
    }
    Ok(())
}

fn check_validity(rxy: f64) -> Result<(), String> {
    if !rxy.is_finite() || rxy <= -1.0 || rxy >= 1.0 {
        return Err(format!(
            "validity rxy must be strictly inside (-1, 1), got {rxy}"
        ));
    }
    Ok(())
}

/// Brogden-Cronbach-Gleser utility of a selection system with Naylor-Shine
/// selected-group mean.
///
/// See the module docs for the model and references. Arguments:
/// `n` applicants selected per period (>= 1), `sdy` monetary SD of
/// performance (>= 0), `rxy` validity in (-1, 1), `sr` selection ratio in
/// (0, 1), `cost_total` total selection cost (finite), `period` expected
/// tenure (>= 1).
pub fn selection_utility(
    n: f64,
    sdy: f64,
    rxy: f64,
    sr: f64,
    cost_total: f64,
    period: f64,
) -> Result<SelectionUtilityResult, String> {
    if !n.is_finite() || n < 1.0 {
        return Err(format!("n must be finite and >= 1, got {n}"));
    }
    if !sdy.is_finite() || sdy < 0.0 {
        return Err(format!("sdy must be finite and >= 0, got {sdy}"));
    }
    check_validity(rxy)?;
    check_unit_open(sr, "selection ratio sr")?;
    if !cost_total.is_finite() {
        return Err(format!("cost_total must be finite, got {cost_total}"));
    }
    if !period.is_finite() || period < 1.0 {
        return Err(format!("period must be finite and >= 1, got {period}"));
    }
    let xc = inv_normal_cdf(1.0 - sr);
    let ux = norm_pdf(xc) / sr;
    let pux = rxy * ux;
    let utility_gain = n * period * sdy * pux - cost_total;
    if !xc.is_finite() || !utility_gain.is_finite() {
        return Err(format!(
            "selection_utility produced non-finite output for sr={sr}"
        ));
    }
    Ok(SelectionUtilityResult {
        xc,
        ux,
        pux,
        utility_gain,
    })
}

/// Taylor-Russell (1939) success ratio for a dichotomous criterion.
///
/// `rxy` validity in (-1, 1), `sr` selection ratio in (0, 1), `br` base rate
/// in (0, 1). Returns `P(Y > yc | X > xc)` under the standard BVN model.
pub fn taylor_russell(rxy: f64, sr: f64, br: f64) -> Result<TaylorRussellResult, String> {
    check_validity(rxy)?;
    check_unit_open(sr, "selection ratio sr")?;
    check_unit_open(br, "base rate br")?;
    // The quadrature in `bvn_upper` resolves the conditional-CDF transition
    // only while sqrt(1 - rxy^2) >= 1e-4; reject nearer-degenerate rxy
    // instead of returning an inaccurate value (impl-review MEDIUM finding).
    if (1.0 - rxy * rxy).sqrt() < 1e-4 {
        return Err(format!(
            "validity rxy = {rxy} is too close to +/-1 for accurate BVN quadrature \
             (requires sqrt(1 - rxy^2) >= 1e-4)"
        ));
    }
    let xc = inv_normal_cdf(1.0 - sr);
    let yc = inv_normal_cdf(1.0 - br);
    // Probability bound: Q = P(X > xc, Y > yc) <= min(sr, br); quadrature
    // round-off could otherwise overshoot slightly (impl-review LOW finding).
    let q_joint = bvn_upper(xc, yc, rxy).min(sr.min(br));
    let success_ratio = (q_joint / sr).clamp(0.0, 1.0);
    if !success_ratio.is_finite() || !q_joint.is_finite() {
        return Err(format!(
            "taylor_russell produced non-finite output for rxy={rxy}, sr={sr}, br={br}"
        ));
    }
    Ok(TaylorRussellResult {
        success_ratio,
        base_rate: br,
        q_joint,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/utility_tests.rs"]
mod tests;
