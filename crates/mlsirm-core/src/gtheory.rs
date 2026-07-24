//! Generalizability theory G-study / D-study for balanced complete crossed
//! designs: one-facet `p x i` and two-facet `p x i x o`.
//!
//! # Verified sources
//!
//! - **READ IN FULL**: Huebner, A., & Lucht, M. (2019). Generalizability
//!   theory in R. *Practical Assessment, Research, and Evaluation*, 24,
//!   Article 5. https://doi.org/10.7275/5065-gc10 — provides the model
//!   equations (their eqs. 1-2), the D-study component scaling (their
//!   Table 1), the relative/absolute error variances sigma^2(delta) and
//!   sigma^2(Delta) (their Table 2), the coefficients E-rho^2 and Phi
//!   (their eqs. 6-7 with sigma^2(tau) = sigma^2(p) for fully random
//!   designs), and complete worked examples with data (their Appendix and
//!   Tables 3-6) against which this implementation is tested.
//! - **NOT read**: Brennan (2001) and Shavelson & Webb (1991), the classic
//!   texts to which Huebner & Lucht defer the ANOVA/EMS derivations. They
//!   are cited below only "as cited in Huebner & Lucht (2019)".
//!
//! # Derivation status of the EMS inversions
//!
//! Huebner & Lucht (2019) do not print the mean-square-to-variance-component
//! formulas. The inversions used here were HAND-DERIVED from the standard
//! expected mean squares of the fully random balanced two-/three-way ANOVA
//! without replication, e.g. for `p x i`:
//!
//! ```text
//! E[MS_p]  = sigma2_pi + n_i * sigma2_p
//! E[MS_i]  = sigma2_pi + n_p * sigma2_i
//! E[MS_pi] = sigma2_pi
//! ```
//!
//! and for `p x i x o`:
//!
//! ```text
//! E[MS_p]   = s_pio + n_o*s_pi + n_i*s_po + n_i*n_o*s_p   (and symmetric)
//! E[MS_pi]  = s_pio + n_o*s_pi                            (and symmetric)
//! E[MS_pio] = s_pio
//! ```
//!
//! These inversions were verified numerically against every published table
//! of Huebner & Lucht (2019): Table 3/4 (`p x i`, pi_dat) and Table 5/6
//! (`p x i x o`, pio_cross_dat) reproduce to the papers' printed rounding;
//! see the tests. No claim is made that the derivation matches the unread
//! Brennan (2001) text beyond this numerical agreement.
//!
//! # Formulas (Huebner & Lucht, 2019)
//!
//! D study for proposed sizes `n_i'` (and `n_o'`): each G-study component is
//! divided by the product of the primed sizes of the facets it involves
//! (their Table 1). Then (their Table 2, eqs. 6-7):
//!
//! - `p x i`: `sigma2(delta) = s_pi/n_i'`,
//!   `sigma2(Delta) = s_i/n_i' + s_pi/n_i'`.
//! - `p x i x o`: `sigma2(delta) = s_pi/n_i' + s_po/n_o' + s_pio/(n_i' n_o')`,
//!   `sigma2(Delta) = sigma2(delta) + s_i/n_i' + s_o/n_o' + s_io/(n_i' n_o')`.
//! - `E-rho^2 = s_p / (s_p + sigma2(delta))`,
//!   `Phi = s_p / (s_p + sigma2(Delta))`.
//!
//! # Negative-variance policy (implementation policy, NOT from the paper)
//!
//! ANOVA variance-component estimators can be negative. The paper's examples
//! avoid this by construction. This module reports BOTH the raw ANOVA
//! estimates and component-wise clamped-at-zero copies; all D-study
//! quantities are computed from the CLAMPED components. This "clamped-ANOVA"
//! policy is not a joint constrained estimator; for data with negative raw
//! components the D-study outputs are policy outputs, not reproductions of
//! any published pipeline. `E-rho^2`/`Phi` are `NaN` when their denominator
//! is <= 1e-12.
//!
//! # Scope
//!
//! Balanced complete data only (`NaN` input is an error). Nested designs
//! (`i:p`, `p x (i:o)`), unbalanced data, fixed/mixed facets, and standard
//! errors of the components are out of scope.
//!
//! # References
//!
//! Huebner, A., & Lucht, M. (2019). Generalizability theory in R.
//! *Practical Assessment, Research, and Evaluation*, 24, Article 5.
//! https://doi.org/10.7275/5065-gc10
//!
//! Brennan, R. L. (2001). *Generalizability theory*. Springer. (As cited in
//! Huebner & Lucht, 2019; not read.)
//!
//! Shavelson, R. J., & Webb, N. M. (1991). *Generalizability theory: A
//! primer*. Sage. (As cited in Huebner & Lucht, 2019; not read.)

/// One D-study column: proposed facet sizes with the resulting error
/// variances and coefficients (Huebner & Lucht, 2019, Tables 4 and 6).
#[derive(Debug, Clone)]
pub struct GTheoryDStudyRow {
    /// Proposed number of items `n_i'`.
    pub n_i_prime: usize,
    /// Proposed number of occasions `n_o'` (1 for the one-facet design,
    /// where it is unused).
    pub n_o_prime: usize,
    /// Relative error variance `sigma^2(delta)`.
    pub rel_error_var: f64,
    /// Absolute error variance `sigma^2(Delta)`.
    pub abs_error_var: f64,
    /// Generalizability coefficient `E-rho^2` (eq. 6). `NaN` when the
    /// denominator is <= 1e-12.
    pub generalizability: f64,
    /// Index of dependability `Phi` (eq. 7). `NaN` when the denominator is
    /// <= 1e-12.
    pub dependability: f64,
}

/// G-study + D-study output for the one-facet crossed `p x i` design.
#[derive(Debug, Clone)]
pub struct GTheoryPiResult {
    /// Degrees of freedom for (p, i, pi): `n_p-1`, `n_i-1`,
    /// `(n_p-1)(n_i-1)`.
    pub df: [f64; 3],
    /// Sums of squares for (p, i, pi).
    pub ss: [f64; 3],
    /// Mean squares for (p, i, pi).
    pub ms: [f64; 3],
    /// Raw ANOVA variance-component estimates for (p, i, pi); may be
    /// negative.
    pub var_raw: [f64; 3],
    /// Component-wise `max(., 0)` of `var_raw`; used for all D-study
    /// quantities (clamped-ANOVA policy, see module docs).
    pub var: [f64; 3],
    /// One row per requested `n_i'`.
    pub d_study: Vec<GTheoryDStudyRow>,
}

/// G-study + D-study output for the two-facet crossed `p x i x o` design.
/// Component order everywhere: (p, i, o, pi, po, io, pio).
#[derive(Debug, Clone)]
pub struct GTheoryPioResult {
    /// Degrees of freedom in component order.
    pub df: [f64; 7],
    /// Sums of squares in component order.
    pub ss: [f64; 7],
    /// Mean squares in component order.
    pub ms: [f64; 7],
    /// Raw ANOVA variance-component estimates; may be negative.
    pub var_raw: [f64; 7],
    /// Component-wise `max(., 0)` of `var_raw`; used for all D-study
    /// quantities (clamped-ANOVA policy, see module docs).
    pub var: [f64; 7],
    /// One row per requested `(n_i', n_o')` pair.
    pub d_study: Vec<GTheoryDStudyRow>,
}

const DENOM_EPS: f64 = 1e-12;

fn coef(var_p: f64, err: f64) -> f64 {
    let den = var_p + err;
    if den <= DENOM_EPS {
        f64::NAN
    } else {
        var_p / den
    }
}

fn validate(x: &[f64], expected_len: usize, what: &str) -> Result<(), String> {
    if x.len() != expected_len {
        return Err(format!(
            "{what}: expected {expected_len} scores, got {}",
            x.len()
        ));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(format!(
            "{what}: non-finite score; complete balanced data required"
        ));
    }
    Ok(())
}

/// Generalizability analysis for the one-facet crossed `p x i` design
/// (Huebner & Lucht, 2019, "One-facet p x i design" section).
///
/// `x` is row-major `n_p x n_i` (person-major); `n_i_prime` lists the
/// proposed D-study item counts.
pub fn gtheory_pi(
    x: &[f64],
    n_p: usize,
    n_i: usize,
    n_i_prime: &[usize],
) -> Result<GTheoryPiResult, String> {
    if n_p < 2 || n_i < 2 {
        return Err("gtheory_pi: need at least 2 persons and 2 items".to_string());
    }
    validate(x, crate::checked_mul_usize(n_p, n_i, "n_p * n_i overflows usize")?, "gtheory_pi")?;
    if n_i_prime.iter().any(|&n| n == 0) {
        return Err("gtheory_pi: n_i_prime entries must be >= 1".to_string());
    }

    let (fp, fi) = (n_p as f64, n_i as f64);
    let grand = x.iter().sum::<f64>() / (fp * fi);
    let mut pm = vec![0.0; n_p];
    let mut im = vec![0.0; n_i];
    for p in 0..n_p {
        for i in 0..n_i {
            pm[p] += x[p * n_i + i];
            im[i] += x[p * n_i + i];
        }
    }
    for m in pm.iter_mut() {
        *m /= fi;
    }
    for m in im.iter_mut() {
        *m /= fp;
    }

    let ss_p = fi * pm.iter().map(|m| (m - grand).powi(2)).sum::<f64>();
    let ss_i = fp * im.iter().map(|m| (m - grand).powi(2)).sum::<f64>();
    let mut ss_pi = 0.0;
    for p in 0..n_p {
        for i in 0..n_i {
            ss_pi += (x[p * n_i + i] - pm[p] - im[i] + grand).powi(2);
        }
    }

    let df = [fp - 1.0, fi - 1.0, (fp - 1.0) * (fi - 1.0)];
    let ss = [ss_p, ss_i, ss_pi];
    let ms = [ss[0] / df[0], ss[1] / df[1], ss[2] / df[2]];
    // Hand-derived EMS inversion (see module docs); verified against
    // Huebner & Lucht (2019) Table 3.
    let var_raw = [(ms[0] - ms[2]) / fi, (ms[1] - ms[2]) / fp, ms[2]];
    let var = var_raw.map(|v| v.max(0.0));

    let d_study = n_i_prime
        .iter()
        .map(|&n| {
            let fni = n as f64;
            // Table 1: components divided by n_i'; Table 2: delta = s(pI),
            // Delta = s(I) + s(pI).
            let rel = var[2] / fni;
            let abs = var[1] / fni + var[2] / fni;
            GTheoryDStudyRow {
                n_i_prime: n,
                n_o_prime: 1,
                rel_error_var: rel,
                abs_error_var: abs,
                generalizability: coef(var[0], rel),
                dependability: coef(var[0], abs),
            }
        })
        .collect();

    Ok(GTheoryPiResult {
        df,
        ss,
        ms,
        var_raw,
        var,
        d_study,
    })
}

/// Generalizability analysis for the two-facet crossed `p x i x o` design
/// (Huebner & Lucht, 2019, "Two-facet p x i x o design" section).
///
/// `x` is row-major `x[p*n_i*n_o + i*n_o + o]` (person-major, then item,
/// then occasion). `n_prime` lists the proposed `(n_i', n_o')` pairs.
pub fn gtheory_pio(
    x: &[f64],
    n_p: usize,
    n_i: usize,
    n_o: usize,
    n_prime: &[(usize, usize)],
) -> Result<GTheoryPioResult, String> {
    if n_p < 2 || n_i < 2 || n_o < 2 {
        return Err("gtheory_pio: need at least 2 levels per facet".to_string());
    }
    let n_pi = crate::checked_mul_usize(n_p, n_i, "n_p * n_i overflows usize")?;
    validate(x, crate::checked_mul_usize(n_pi, n_o, "n_p * n_i * n_o overflows usize")?, "gtheory_pio")?;
    if n_prime.iter().any(|&(a, b)| a == 0 || b == 0) {
        return Err("gtheory_pio: n_prime entries must be >= 1".to_string());
    }

    let (fp, fi, fo) = (n_p as f64, n_i as f64, n_o as f64);
    let total = fp * fi * fo;
    let at = |p: usize, i: usize, o: usize| x[p * n_i * n_o + i * n_o + o];
    let grand = x.iter().sum::<f64>() / total;

    let mut mp = vec![0.0; n_p];
    let mut mi = vec![0.0; n_i];
    let mut mo = vec![0.0; n_o];
    let mut mpi = vec![0.0; n_p * n_i];
    let mut mpo = vec![0.0; n_p * n_o];
    let mut mio = vec![0.0; n_i * n_o];
    for p in 0..n_p {
        for i in 0..n_i {
            for o in 0..n_o {
                let v = at(p, i, o);
                mp[p] += v;
                mi[i] += v;
                mo[o] += v;
                mpi[p * n_i + i] += v;
                mpo[p * n_o + o] += v;
                mio[i * n_o + o] += v;
            }
        }
    }
    for m in mp.iter_mut() {
        *m /= fi * fo;
    }
    for m in mi.iter_mut() {
        *m /= fp * fo;
    }
    for m in mo.iter_mut() {
        *m /= fp * fi;
    }
    for m in mpi.iter_mut() {
        *m /= fo;
    }
    for m in mpo.iter_mut() {
        *m /= fi;
    }
    for m in mio.iter_mut() {
        *m /= fp;
    }

    let ss_p = fi * fo * mp.iter().map(|m| (m - grand).powi(2)).sum::<f64>();
    let ss_i = fp * fo * mi.iter().map(|m| (m - grand).powi(2)).sum::<f64>();
    let ss_o = fp * fi * mo.iter().map(|m| (m - grand).powi(2)).sum::<f64>();
    let mut ss_pi = 0.0;
    for p in 0..n_p {
        for i in 0..n_i {
            ss_pi += (mpi[p * n_i + i] - mp[p] - mi[i] + grand).powi(2);
        }
    }
    ss_pi *= fo;
    let mut ss_po = 0.0;
    for p in 0..n_p {
        for o in 0..n_o {
            ss_po += (mpo[p * n_o + o] - mp[p] - mo[o] + grand).powi(2);
        }
    }
    ss_po *= fi;
    let mut ss_io = 0.0;
    for i in 0..n_i {
        for o in 0..n_o {
            ss_io += (mio[i * n_o + o] - mi[i] - mo[o] + grand).powi(2);
        }
    }
    ss_io *= fp;
    // Direct three-way residual sum (algebraically equal to
    // SS_total - all other SS, but immune to the catastrophic cancellation
    // the subtraction form shows for large-offset data).
    let mut ss_pio = 0.0;
    for p in 0..n_p {
        for i in 0..n_i {
            for o in 0..n_o {
                ss_pio += (at(p, i, o) - mpi[p * n_i + i] - mpo[p * n_o + o] - mio[i * n_o + o]
                    + mp[p]
                    + mi[i]
                    + mo[o]
                    - grand)
                    .powi(2);
            }
        }
    }

    let df = [
        fp - 1.0,
        fi - 1.0,
        fo - 1.0,
        (fp - 1.0) * (fi - 1.0),
        (fp - 1.0) * (fo - 1.0),
        (fi - 1.0) * (fo - 1.0),
        (fp - 1.0) * (fi - 1.0) * (fo - 1.0),
    ];
    let ss = [ss_p, ss_i, ss_o, ss_pi, ss_po, ss_io, ss_pio];
    let mut ms = [0.0; 7];
    for k in 0..7 {
        ms[k] = ss[k] / df[k];
    }
    // Hand-derived EMS inversion (see module docs); verified against
    // Huebner & Lucht (2019) Table 5.
    let var_raw = [
        (ms[0] - ms[3] - ms[4] + ms[6]) / (fi * fo),
        (ms[1] - ms[3] - ms[5] + ms[6]) / (fp * fo),
        (ms[2] - ms[4] - ms[5] + ms[6]) / (fp * fi),
        (ms[3] - ms[6]) / fo,
        (ms[4] - ms[6]) / fi,
        (ms[5] - ms[6]) / fp,
        ms[6],
    ];
    let var = var_raw.map(|v| v.max(0.0));

    let d_study = n_prime
        .iter()
        .map(|&(ni, no)| {
            let (fni, fno) = (ni as f64, no as f64);
            // Table 2: delta = s(pI) + s(pO) + s(pIO);
            // Delta = delta + s(I) + s(O) + s(IO).
            let rel = var[3] / fni + var[4] / fno + var[6] / (fni * fno);
            let abs = rel + var[1] / fni + var[2] / fno + var[5] / (fni * fno);
            GTheoryDStudyRow {
                n_i_prime: ni,
                n_o_prime: no,
                rel_error_var: rel,
                abs_error_var: abs,
                generalizability: coef(var[0], rel),
                dependability: coef(var[0], abs),
            }
        })
        .collect();

    Ok(GTheoryPioResult {
        df,
        ss,
        ms,
        var_raw,
        var,
        d_study,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/gtheory_tests.rs"]
mod tests;
