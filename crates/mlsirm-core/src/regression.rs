//! Ordinary least squares with heteroskedasticity-consistent (sandwich)
//! covariance estimators HC0–HC3, linear contrasts, and exact upper-tail
//! p-values for χ²(1), F, and Student-t.
//!
//! # Estimators
//!
//! For the linear model `y = Xβ + u` (MacKinnon & White, 1985, eq. 1), OLS is
//! `β̂ = (X'X)^{-1} X'y` (eq. 2). The sandwich form
//! `(X'X)^{-1} X' Ω̂ X (X'X)^{-1}` (eq. 5) is specialized as:
//!
//! - **HC0** (White HC): `Ω̂ = diag(û_t²)` (eq. 5).
//! - **HC1**: `(n/(n-k))` times HC0 (eq. 6; Hinkley, 1977).
//! - **HC2**: `Ω̃ = diag(û_t² / (1 - k_tt))` (eqs. 7–9).
//! - **HC3**: `Ω* = diag((û_t / (1 - k_tt))²)`, the diagonal jackknife meat
//!   from `u*_t = û_t/(1-k_tt)` (eqs. 10–12), omitting the optional
//!   `(n-1)/n` scale and rank-1 correction MacKinnon & White note as
//!   asymptotically negligible (p. 4). This matches the late-life
//!   `fit_ols_hc3` / `sandwich::vcovHC(type="HC3")` contract.
//!
//! The labels HC0–HC3 follow the naming popularized by Long and Ervin (2000).
//!
//! # Distribution tails
//!
//! Upper tails use the regularized incomplete beta `I_x(a,b)` via continued
//! fraction (Press, Teukolsky, Vetterling, & Flannery, 2007, §§6.1–6.4;
//! cf. DiDonato & Morris, 1992, Algorithm 708). Relationships:
//! `P(F_{d1,d2} > f) = I_{d2/(d2+d1 f)}(d2/2, d1/2)`;
//! `P(T_ν > t) = (1/2) I_{ν/(ν+t²)}(ν/2, 1/2)` for `t ≥ 0` (and the
//! reflection for `t < 0`). χ²(1) reuses [`crate::fitstats::chi2_sf`].
//!
//! # Moderated (simple) slopes
//!
//! For the H1–H5 study parameterization
//! `Y ~ X*W*Z + X*E` with mean-centered predictors before products, the
//! pick-a-point / simple-slope weights follow Aiken and West (1991, ch. 2)
//! and Hayes (2018, ch. 7–8):
//!
//! ```text
//! dY/dX | (W,Z,E) = b_X + b_XW W + b_XZ Z + b_XE E + b_XWZ W Z
//! dY/dZ | (X,W)   = b_Z + b_XZ X + b_WZ W + b_XWZ X W
//! ```
//!
//! Standard errors are `sqrt(c' V c)` via [`linear_contrast`], never
//! post-hoc SE synthesis. Slope differences use one contrast equal to the
//! difference of two simple-slope weight vectors (Hayes, 2018, conditional-
//! effect pairwise comparison pattern).
//!
//! # References
//!
//! Aiken, L. S., & West, S. G. (1991). *Multiple regression: Testing and
//!     interpreting interactions*. Sage. (Simple-slope / pick-a-point
//!     algebra, ch. 2.)
//!
//! DiDonato, A. R., & Morris, A. H., Jr. (1992). Algorithm 708: Significant
//!     digit computation of the incomplete beta function ratios.
//!     *ACM Transactions on Mathematical Software, 18*(3), 360–373.
//!     <https://doi.org/10.1145/131766.131776>
//!
//! Hayes, A. F. (2018). *Introduction to mediation, moderation, and
//!     conditional process analysis: A regression-based approach* (2nd ed.).
//!     Guilford Press. (Conditional effects and pairwise slope comparisons,
//!     ch. 7–8.)
//!
//! Long, J. S., & Ervin, L. H. (2000). Using heteroscedasticity consistent
//!     standard errors in the linear regression model. *The American
//!     Statistician, 54*(3), 217–224.
//!     <https://doi.org/10.1080/00031305.2000.10474549>
//!
//! MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
//!     covariance matrix estimators with improved finite sample properties.
//!     *Journal of Econometrics, 29*(3), 305–325.
//!     <https://doi.org/10.1016/0304-4076(85)90158-7>
//!     (Working-paper text used for equation locators: QED WP 537.)
//!
//! Pennsylvania State University, Department of Statistics. (n.d.).
//!     *Lesson 13: Weighted least squares & logistic regressions*. STAT 501:
//!     Regression methods. <https://online.stat.psu.edu/stat501/Lesson13>
//!
//! Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P.
//!     (2007). *Numerical recipes: The art of scientific computing* (3rd ed.).
//!     Cambridge University Press.
//!
//! R Core Team. (n.d.). *Standard deviation*. R stats manual.
//!     <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/sd.html>

use crate::fitstats::{chi2_sf, ln_gamma};

/// Heteroskedasticity-consistent sandwich estimator family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HcType {
    /// White (1980) / MacKinnon & White HC (eq. 5); Long & Ervin HC0.
    HC0,
    /// Degrees-of-freedom scaled HC0 (MacKinnon & White eq. 6).
    HC1,
    /// Leveraged residual squares (MacKinnon & White eqs. 7–9).
    HC2,
    /// Squared delete-one residual weights (MacKinnon & White eqs. 10–12 meat).
    HC3,
}

impl HcType {
    /// Parse a case-insensitive label (`"HC0"` … `"HC3"`).
    pub fn parse(label: &str) -> Result<Self, String> {
        match label.trim().to_ascii_uppercase().as_str() {
            "HC0" => Ok(Self::HC0),
            "HC1" => Ok(Self::HC1),
            "HC2" => Ok(Self::HC2),
            "HC3" => Ok(Self::HC3),
            other => Err(format!("unknown HC type {other:?}; expected HC0..HC3")),
        }
    }
}

/// OLS point fit plus quantities shared by all HC sandwich estimators.
#[derive(Clone, Debug)]
pub struct OlsFit {
    /// Number of observations `n`.
    pub n: usize,
    /// Number of columns `k` in `X`.
    pub k: usize,
    /// Coefficient vector `β̂` (`k`).
    pub beta: Vec<f64>,
    /// Residuals `û = y - Xβ̂` (`n`).
    pub residuals: Vec<f64>,
    /// Hat diagonal `k_tt = x_t' (X'X)^{-1} x_t` (`n`).
    pub hat_diagonal: Vec<f64>,
    /// Bread `(X'X)^{-1}` in row-major `k × k`.
    pub xtx_inv: Vec<f64>,
    /// Classical residual variance `û'û / (n - k)`.
    pub sigma2: f64,
}

/// Linear contrast `c'β` under a supplied covariance matrix.
#[derive(Clone, Debug)]
pub struct ContrastResult {
    /// Point estimate `c'β`.
    pub estimate: f64,
    /// Standard error `sqrt(c' V c)`.
    pub se: f64,
    /// Wald χ²(1) = `(c'β)² / (c' V c)`.
    pub wald_chi2: f64,
    /// Upper-tail p-value under χ²(1).
    pub p_chi2: f64,
    /// Quasi-t = `(c'β) / se` (same sign as the estimate).
    pub t_stat: f64,
    /// Upper-tail Student-t p-value with `df` residual degrees of freedom.
    pub p_t: f64,
    /// Equivalent F(1, df) = t².
    pub f_stat: f64,
    /// Upper-tail F(1, df) p-value.
    pub p_f: f64,
    /// Residual degrees of freedom `n - k` used for t/F tails.
    pub df: f64,
}

/// Finite sample mean and sample SD (`n - 1` denominator).
///
/// R Core Team (n.d.), *R stats: Standard Deviation*, Details.
pub fn sample_mean_sd(values: &[f64]) -> Result<(f64, f64), String> {
    if values.len() < 2 || values.iter().any(|value| !value.is_finite()) {
        return Err("sample moments need at least two finite values".to_owned());
    }
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for (i, &value) in values.iter().enumerate() {
        let delta = value - mean;
        mean += delta / (i + 1) as f64;
        m2 += delta * (value - mean);
    }
    let sd = (m2 / (values.len() - 1) as f64).sqrt();
    if !mean.is_finite() || !sd.is_finite() {
        return Err("sample moments are not finite".to_owned());
    }
    Ok((mean, sd))
}

/// Normal-Wald interval for an estimate and supplied standard error.
///
/// Pennsylvania State University (n.d.), *STAT 501*, Lesson 13,
/// coefficient confidence interval equation. `confidence_level` is explicit;
/// this asymptotic interval does not substitute for bootstrap uncertainty.
pub fn normal_wald_interval(
    estimate: f64,
    se: f64,
    confidence_level: f64,
) -> Result<(f64, f64), String> {
    if !estimate.is_finite() || !se.is_finite() || se < 0.0 {
        return Err("estimate and nonnegative SE must be finite".to_owned());
    }
    if !confidence_level.is_finite() || !(0.0..1.0).contains(&confidence_level) {
        return Err("confidence_level must be finite and in (0, 1)".to_owned());
    }
    let critical = crate::mokken::normal_upper_quantile((1.0 - confidence_level) / 2.0);
    let margin = critical * se;
    let bounds = (estimate - margin, estimate + margin);
    if !bounds.0.is_finite() || !bounds.1.is_finite() {
        return Err("normal-Wald interval endpoints are not finite".to_owned());
    }
    Ok(bounds)
}

/// Fit OLS by normal equations and compute the hat diagonal.
///
/// `x` is row-major `n × k`, `y` length `n`. Fails closed on rank deficiency,
/// non-finite inputs, or hat values outside `[0, 1)`.
pub fn fit_ols(x: &[f64], y: &[f64], n: usize, k: usize) -> Result<OlsFit, String> {
    validate_design(x, y, n, k)?;
    if n <= k {
        return Err(format!("need n > k for OLS (got n={n}, k={k})"));
    }

    let mut xtx = vec![0.0_f64; k * k];
    let mut xty = vec![0.0_f64; k];
    for i in 0..n {
        let row = &x[i * k..(i + 1) * k];
        let yi = y[i];
        for a in 0..k {
            xty[a] += row[a] * yi;
            for b in 0..k {
                xtx[a * k + b] += row[a] * row[b];
            }
        }
    }

    let xtx_inv = invert_square(&xtx, k)
        .ok_or_else(|| "X'X is singular; design is rank deficient".to_owned())?;

    let mut beta = vec![0.0_f64; k];
    for a in 0..k {
        let mut s = 0.0;
        for b in 0..k {
            s += xtx_inv[a * k + b] * xty[b];
        }
        beta[a] = s;
    }

    let mut residuals = vec![0.0_f64; n];
    let mut hat_diagonal = vec![0.0_f64; n];
    let mut sse = 0.0_f64;
    for i in 0..n {
        let row = &x[i * k..(i + 1) * k];
        let mut fitted = 0.0;
        let mut h = 0.0;
        for a in 0..k {
            fitted += row[a] * beta[a];
            let mut xa = 0.0;
            for b in 0..k {
                xa += xtx_inv[a * k + b] * row[b];
            }
            h += row[a] * xa;
        }
        if !h.is_finite() || !(0.0..1.0).contains(&h) {
            return Err(format!("hat diagonal out of range at row {i}: {h}"));
        }
        let e = y[i] - fitted;
        if !e.is_finite() {
            return Err(format!("non-finite residual at row {i}"));
        }
        residuals[i] = e;
        hat_diagonal[i] = h;
        sse += e * e;
    }

    let df = (n - k) as f64;
    let sigma2 = sse / df;
    if !sigma2.is_finite() || sigma2 < 0.0 {
        return Err("non-finite classical residual variance".to_owned());
    }

    Ok(OlsFit {
        n,
        k,
        beta,
        residuals,
        hat_diagonal,
        xtx_inv,
        sigma2,
    })
}

/// Sandwich covariance `V` for the requested HC type (row-major `k × k`).
pub fn sandwich_vcov(fit: &OlsFit, x: &[f64], hc: HcType) -> Result<Vec<f64>, String> {
    let n = fit.n;
    let k = fit.k;
    if x.len() != n * k {
        return Err(format!(
            "x length {} does not match n*k = {}",
            x.len(),
            n * k
        ));
    }
    let scale = match hc {
        HcType::HC0 => 1.0,
        HcType::HC1 => (n as f64) / ((n - k) as f64),
        HcType::HC2 | HcType::HC3 => 1.0,
    };

    let mut meat = vec![0.0_f64; k * k];
    for i in 0..n {
        let row = &x[i * k..(i + 1) * k];
        let e = fit.residuals[i];
        let h = fit.hat_diagonal[i];
        let omega = match hc {
            HcType::HC0 | HcType::HC1 => e * e,
            HcType::HC2 => {
                let denom = 1.0 - h;
                if denom <= 0.0 {
                    return Err(format!("non-positive 1-h at row {i}"));
                }
                (e * e) / denom
            }
            HcType::HC3 => {
                let denom = 1.0 - h;
                if denom <= 0.0 {
                    return Err(format!("non-positive 1-h at row {i}"));
                }
                let u_star = e / denom;
                u_star * u_star
            }
        };
        let w = scale * omega;
        if !w.is_finite() || w < 0.0 {
            return Err(format!("non-finite sandwich weight at row {i}"));
        }
        for a in 0..k {
            for b in 0..k {
                meat[a * k + b] += w * row[a] * row[b];
            }
        }
    }

    // V = bread * meat * bread
    let bread = &fit.xtx_inv;
    let mut tmp = vec![0.0_f64; k * k];
    for a in 0..k {
        for b in 0..k {
            let mut s = 0.0;
            for c in 0..k {
                s += bread[a * k + c] * meat[c * k + b];
            }
            tmp[a * k + b] = s;
        }
    }
    let mut vcov = vec![0.0_f64; k * k];
    for a in 0..k {
        for b in 0..k {
            let mut s = 0.0;
            for c in 0..k {
                s += tmp[a * k + c] * bread[c * k + b];
            }
            if !s.is_finite() {
                return Err("non-finite sandwich covariance entry".to_owned());
            }
            vcov[a * k + b] = s;
        }
    }
    Ok(vcov)
}

/// Fit OLS and return the requested HC sandwich covariance in one call.
pub fn fit_ols_hc(
    x: &[f64],
    y: &[f64],
    n: usize,
    k: usize,
    hc: HcType,
) -> Result<(OlsFit, Vec<f64>), String> {
    let fit = fit_ols(x, y, n, k)?;
    let vcov = sandwich_vcov(&fit, x, hc)?;
    Ok((fit, vcov))
}

/// Linear contrast under `vcov` with Wald χ²(1) and t/F tails at `df = n - k`.
pub fn linear_contrast(
    beta: &[f64],
    vcov: &[f64],
    contrast: &[f64],
    df: f64,
) -> Result<ContrastResult, String> {
    let k = beta.len();
    if contrast.len() != k {
        return Err(format!(
            "contrast length {} does not match beta length {k}",
            contrast.len()
        ));
    }
    if vcov.len() != k * k {
        return Err(format!(
            "vcov length {} does not match k*k = {}",
            vcov.len(),
            k * k
        ));
    }
    if !(df.is_finite() && df > 0.0) {
        return Err(format!("residual df must be positive finite, got {df}"));
    }

    let mut estimate = 0.0_f64;
    for i in 0..k {
        estimate += contrast[i] * beta[i];
    }
    let mut var = 0.0_f64;
    for i in 0..k {
        let mut row = 0.0;
        for j in 0..k {
            row += vcov[i * k + j] * contrast[j];
        }
        var += contrast[i] * row;
    }
    if !var.is_finite() || var <= 0.0 {
        return Err(format!("contrast variance is not finite positive: {var}"));
    }
    let se = var.sqrt();
    let wald_chi2 = (estimate * estimate) / var;
    let p_chi2 = chi2_sf(wald_chi2, 1.0);
    let t_stat = estimate / se;
    let p_t = t_sf(t_stat, df);
    let f_stat = t_stat * t_stat;
    let p_f = f_sf(f_stat, 1.0, df);
    if ![estimate, se, wald_chi2, p_chi2, t_stat, p_t, f_stat, p_f]
        .iter()
        .all(|v| v.is_finite())
    {
        return Err("non-finite contrast statistics".to_owned());
    }
    Ok(ContrastResult {
        estimate,
        se,
        wald_chi2,
        p_chi2,
        t_stat,
        p_t,
        f_stat,
        p_f,
        df,
    })
}

/// Number of columns in the H1–H5 design `Y ~ X*W*Z + X*E`.
pub const XWZ_E_K: usize = 10;

/// Design row for `Y ~ X*W*Z + X*E` at centered probe values `(x,w,z,e)`.
///
/// Column order: `(Intercept), X, W, Z, E, X:W, X:Z, W:Z, X:E, X:W:Z`
/// (Aiken & West, 1991, ch. 2 product terms; Hayes, 2018, ch. 7).
pub fn xwz_e_design_row(x: f64, w: f64, z: f64, e: f64) -> [f64; XWZ_E_K] {
    [
        1.0,
        x,
        w,
        z,
        e,
        x * w,
        x * z,
        w * z,
        x * e,
        x * w * z,
    ]
}

/// Dot product of a design row with `β` (predicted mean at probes).
pub fn design_row_dot(row: &[f64], beta: &[f64]) -> Result<f64, String> {
    if row.len() != beta.len() {
        return Err(format!(
            "design row length {} does not match beta length {}",
            row.len(),
            beta.len()
        ));
    }
    let mut s = 0.0_f64;
    for i in 0..row.len() {
        let v = row[i] * beta[i];
        if !v.is_finite() {
            return Err("non-finite design_row_dot contribution".to_owned());
        }
        s += v;
    }
    if !s.is_finite() {
        return Err("non-finite design_row_dot".to_owned());
    }
    Ok(s)
}

/// Weight vector `c` such that `c'β = dY/d(focal)` at probe values.
///
/// `focal` is `"X"` or `"Z"` for the H1–H5 parameterization
/// (Aiken & West, 1991, ch. 2; Hayes, 2018, ch. 7–8).
pub fn conditional_slope_weights(
    focal: &str,
    x: f64,
    w: f64,
    z: f64,
    e: f64,
) -> Result<Vec<f64>, String> {
    let mut c = vec![0.0_f64; XWZ_E_K];
    match focal.trim().to_ascii_uppercase().as_str() {
        // dY/dX = bX + bXW*W + bXZ*Z + bXE*E + bXWZ*W*Z
        "X" => {
            c[1] = 1.0;
            c[5] = w;
            c[6] = z;
            c[8] = e;
            c[9] = w * z;
        }
        // dY/dZ = bZ + bXZ*X + bWZ*W + bXWZ*X*W
        "Z" => {
            c[3] = 1.0;
            c[6] = x;
            c[7] = w;
            c[9] = x * w;
        }
        other => {
            return Err(format!(
                "unknown focal {other:?}; expected \"X\" or \"Z\" for XWZ+E"
            ));
        }
    }
    if !c.iter().all(|v| v.is_finite()) {
        return Err("non-finite conditional slope weights".to_owned());
    }
    Ok(c)
}

/// Simple slope estimate + Wald/t/F via [`linear_contrast`].
pub fn conditional_slope(
    beta: &[f64],
    vcov: &[f64],
    focal: &str,
    x: f64,
    w: f64,
    z: f64,
    e: f64,
    df: f64,
) -> Result<ContrastResult, String> {
    if beta.len() != XWZ_E_K {
        return Err(format!(
            "conditional_slope expects beta length {XWZ_E_K}, got {}",
            beta.len()
        ));
    }
    let weights = conditional_slope_weights(focal, x, w, z, e)?;
    linear_contrast(beta, vcov, &weights, df)
}

/// Difference of two simple slopes (same focal, two probe tuples `(x,w,z,e)`).
pub fn slope_difference(
    beta: &[f64],
    vcov: &[f64],
    focal: &str,
    probes_a: (f64, f64, f64, f64),
    probes_b: (f64, f64, f64, f64),
    df: f64,
) -> Result<ContrastResult, String> {
    if beta.len() != XWZ_E_K {
        return Err(format!(
            "slope_difference expects beta length {XWZ_E_K}, got {}",
            beta.len()
        ));
    }
    let wa = conditional_slope_weights(focal, probes_a.0, probes_a.1, probes_a.2, probes_a.3)?;
    let wb = conditional_slope_weights(focal, probes_b.0, probes_b.1, probes_b.2, probes_b.3)?;
    let diff: Vec<f64> = wa.iter().zip(wb.iter()).map(|(a, b)| a - b).collect();
    linear_contrast(beta, vcov, &diff, df)
}

/// Upper-tail `P(Chi2_1 >= q)` (exact via [`chi2_sf`]).
pub fn chi2_sf_df1(q: f64) -> f64 {
    chi2_sf(q, 1.0)
}

/// Upper-tail `P(F_{df1,df2} >= f)` via the regularized incomplete beta.
pub fn f_sf(f: f64, df1: f64, df2: f64) -> f64 {
    if !(f.is_finite() && df1.is_finite() && df2.is_finite()) || df1 <= 0.0 || df2 <= 0.0 {
        return f64::NAN;
    }
    if f <= 0.0 {
        return 1.0;
    }
    let x = df2 / (df2 + df1 * f);
    betai(df2 / 2.0, df1 / 2.0, x)
}

/// Upper-tail `P(T_df >= t)` via the regularized incomplete beta.
pub fn t_sf(t: f64, df: f64) -> f64 {
    if !(t.is_finite() && df.is_finite()) || df <= 0.0 {
        return f64::NAN;
    }
    let x = df / (df + t * t);
    let two_sided = betai(df / 2.0, 0.5, x);
    if t >= 0.0 {
        0.5 * two_sided
    } else {
        1.0 - 0.5 * two_sided
    }
}

/// Regularized incomplete beta `I_x(a, b)` (Press et al., 2007, §6.4).
pub fn betai(a: f64, b: f64, x: f64) -> f64 {
    if !(a.is_finite() && b.is_finite() && x.is_finite()) || a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x == 0.0 || x == 1.0 {
        return x;
    }
    // bt = B_x(a,b) / B(a,b) * continued-fraction factor (Numerical Recipes §6.4).
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    let ix = if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    };
    ix.clamp(0.0, 1.0)
}

/// Continued-fraction evaluator for the incomplete beta (Press et al. §6.4).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    // Use the numerically stable Lentz method on the standard expansion.
    const MAX_IT: usize = 200;
    const EPS: f64 = 3.0e-14;
    const FPMIN: f64 = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_IT {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;
        let mut aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

fn validate_design(x: &[f64], y: &[f64], n: usize, k: usize) -> Result<(), String> {
    if n == 0 || k == 0 {
        return Err("n and k must be positive".to_owned());
    }
    if y.len() != n {
        return Err(format!("y length {} does not match n={n}", y.len()));
    }
    if x.len() != n * k {
        return Err(format!(
            "x length {} does not match n*k = {}",
            x.len(),
            n * k
        ));
    }
    if !y.iter().all(|v| v.is_finite()) || !x.iter().all(|v| v.is_finite()) {
        return Err("x and y must be finite".to_owned());
    }
    Ok(())
}

fn invert_square(matrix: &[f64], k: usize) -> Option<Vec<f64>> {
    let mut m = matrix.to_vec();
    let mut inv = vec![0.0_f64; k * k];
    for i in 0..k {
        inv[i * k + i] = 1.0;
    }
    for col in 0..k {
        let mut piv = col;
        for r in (col + 1)..k {
            if m[r * k + col].abs() > m[piv * k + col].abs() {
                piv = r;
            }
        }
        if m[piv * k + col].abs() < 1e-12 {
            return None;
        }
        if piv != col {
            for c in 0..k {
                m.swap(col * k + c, piv * k + c);
                inv.swap(col * k + c, piv * k + c);
            }
        }
        let d = m[col * k + col];
        for c in 0..k {
            m[col * k + c] /= d;
            inv[col * k + c] /= d;
        }
        for r in 0..k {
            if r != col {
                let f = m[r * k + col];
                if f != 0.0 {
                    for c in 0..k {
                        m[r * k + c] -= f * m[col * k + c];
                        inv[r * k + c] -= f * inv[col * k + c];
                    }
                }
            }
        }
    }
    Some(inv)
}

#[cfg(test)]
#[path = "../../../tests/unit/regression_tests.rs"]
mod tests;
