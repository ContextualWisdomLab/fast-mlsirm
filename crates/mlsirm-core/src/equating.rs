//! Observed-score equating (Kolen & Brennan, 2014, *Test Equating, Scaling, and
//! Linking*, 3rd ed.): the raw-score → raw-score complement to the parameter
//! linking in [`crate::linking`]. This module covers the equivalent-groups (EG)
//! design — mean, linear, and equipercentile equating — and the common-item
//! non-equivalent-groups (NEAT) design via chained equipercentile and
//! frequency-estimation (post-stratification) equipercentile.
//!
//! All continuization uses the Kolen-Brennan uniform-kernel convention (an
//! integer score `x` occupies the interval `[x-0.5, x+0.5)`; the discrete cdf is
//! interpolated linearly within it). Equated scores `e_Y(x)` are kept
//! real-valued (unrounded); producing an integer conversion table is left to the
//! caller.
//!
//! # References (APA 7th ed.)
//!
//! Kolen, M. J., & Brennan, R. L. (2014). *Test equating, scaling, and linking:
//!   Methods and practices* (3rd ed.). Springer.
//!   https://doi.org/10.1007/978-1-4939-0317-7
//!
//! Deferred to future work (each is a drop-in behind the density/table interface
//! here): Tucker/Levine linear NEAT (K&B §4.3–4.4), log-linear presmoothing
//! (K&B ch. 3), and Gaussian-kernel equating (von Davier, Holland & Thayer, 2004).

/// Result of an equating: the conversion table `y_equivalents[i] = e_Y(x_scores[i])`
/// (unrounded), the form moments, the moments of the equated scores under form
/// X's distribution, and — for the moment methods only — the linear
/// `slope`/`intercept` (`NaN` for equipercentile / NEAT). The `mu_x`/`sigma_x`/
/// `mu_y`/`sigma_y` fields are the raw form marginals for EG and chained equating,
/// but the *synthetic-population* moments for frequency estimation and for the
/// Tucker/Levine linear methods (which equate synthetic populations, not the raw
/// marginals) — so do not compare their moment fields against a chained or EG
/// result's field-for-field.
#[derive(Clone, Debug)]
pub struct EquateResult {
    pub x_scores: Vec<f64>,
    pub y_equivalents: Vec<f64>,
    pub mu_x: f64,
    pub sigma_x: f64,
    pub mu_y: f64,
    pub sigma_y: f64,
    pub mu_eq: f64,
    pub sigma_eq: f64,
    pub slope: f64,
    pub intercept: f64,
    pub n_x: usize,
    pub n_y: usize,
    /// Gaussian-kernel bandwidths actually used for form X / form Y; `NaN` for the
    /// uniform-kernel (equipercentile) and moment methods.
    pub h_x: f64,
    pub h_y: f64,
}

/// Equivalent-groups equating method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EquateMethod {
    Mean,
    Linear,
    Equipercentile,
}

impl EquateMethod {
    pub fn parse(name: &str) -> Option<EquateMethod> {
        match name.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "mean" | "m" => Some(EquateMethod::Mean),
            "linear" | "lin" | "l" => Some(EquateMethod::Linear),
            "equipercentile" | "equip" | "ep" => Some(EquateMethod::Equipercentile),
            _ => None,
        }
    }
}

/// NEAT (common-item non-equivalent groups) equating method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NeatMethod {
    ChainedEquipercentile,
    FrequencyEstimation,
}

impl NeatMethod {
    pub fn parse(name: &str) -> Option<NeatMethod> {
        match name.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "chained" | "chainedequipercentile" | "ce" => Some(NeatMethod::ChainedEquipercentile),
            "frequencyestimation" | "fe" | "poststratification" => {
                Some(NeatMethod::FrequencyEstimation)
            }
            _ => None,
        }
    }
}

// --- discrete-frequency utilities (none of this exists elsewhere in the crate) ---

/// Relative-frequency vector `g(0..=k)` from raw integer scores. Errors on empty
/// input or a score outside `0..=k`.
fn rel_freq(scores: &[f64], k: usize) -> Result<Vec<f64>, String> {
    if scores.is_empty() {
        return Err("score vector must be non-empty".into());
    }
    let mut freq = vec![0.0_f64; k + 1];
    for &s in scores {
        if !s.is_finite() {
            return Err("scores must be finite".into());
        }
        if s < -0.5 || s >= k as f64 + 0.5 {
            return Err(format!("score {s} outside 0..={k}"));
        }
        // bin to the category whose [c-0.5, c+0.5) interval contains s
        freq[(s + 0.5).floor() as usize] += 1.0;
    }
    let n = scores.len() as f64;
    for f in freq.iter_mut() {
        *f /= n;
    }
    Ok(freq)
}

/// Discrete cdf `F(x) = sum_{u<=x} g(u)`; the top cell is pinned to 1.0 to absorb
/// floating-point drift so the inverse always finds a bracketing score.
fn cdf(g: &[f64]) -> Vec<f64> {
    let mut f = vec![0.0_f64; g.len()];
    let mut acc = 0.0_f64;
    for (i, &gi) in g.iter().enumerate() {
        acc += gi;
        f[i] = acc;
    }
    if let Some(last) = f.last_mut() {
        *last = 1.0;
    }
    f
}

/// Population mean and standard deviation of a score distribution `g`.
fn moments(g: &[f64]) -> (f64, f64) {
    let mean: f64 = g.iter().enumerate().map(|(x, &p)| x as f64 * p).sum();
    let var: f64 = g.iter().enumerate().map(|(x, &p)| (x as f64 - mean).powi(2) * p).sum();
    (mean, var.max(0.0).sqrt())
}

/// Mean/SD of the equated scores `y_eq` weighted by form X's distribution `gx`.
fn weighted_moments(y_eq: &[f64], gx: &[f64]) -> (f64, f64) {
    let mean: f64 = y_eq.iter().zip(gx).map(|(&y, &w)| y * w).sum();
    let var: f64 = y_eq.iter().zip(gx).map(|(&y, &w)| (y - mean).powi(2) * w).sum();
    (mean, var.max(0.0).sqrt())
}

/// Percentile rank `P(x)` (K&B eq. 2.3), the uniform-kernel continuization of the
/// discrete cdf. `x` may be real (needed by chained equating). Returns a value in
/// `[0, 100]`.
fn perc_rank(g: &[f64], f: &[f64], k: usize, x: f64) -> f64 {
    if x < -0.5 {
        return 0.0;
    }
    if x >= k as f64 + 0.5 {
        return 100.0;
    }
    let xstar = x.round() as usize; // in 0..=k over the valid interval
    let f_lo = if xstar == 0 { 0.0 } else { f[xstar - 1] };
    100.0 * (f_lo + (x - (xstar as f64 - 0.5)) * g[xstar])
}

/// Inverse percentile rank `P^{-1}(p*)` (K&B eqs. 2.4/2.5), real-valued in
/// `[-0.5, k+0.5]`. The lower form is used throughout: it interpolates linearly
/// inside the bracketing score's interval — including at the low boundary
/// (`p* <= 100*F(0)` maps to `(p*/100)/g(0) - 0.5`, NOT a hard clamp to `-0.5`),
/// which is what makes the self-equating identity exact at `x = 0`.
fn perc_rank_inv(f: &[f64], k: usize, p_star: f64) -> f64 {
    if p_star <= 0.0 {
        return -0.5;
    }
    if p_star >= 100.0 {
        return k as f64 + 0.5;
    }
    let pp = p_star / 100.0;
    // smallest integer score x_u with F(x_u) > pp
    let mut x_u = k;
    for (i, &fi) in f.iter().enumerate() {
        if fi > pp {
            x_u = i;
            break;
        }
    }
    let f_lo = if x_u == 0 { 0.0 } else { f[x_u - 1] };
    let g_u = f[x_u] - f_lo;
    if g_u <= 0.0 {
        // unreachable when F(x_u) > pp >= F(x_u-1) (implies g_u > 0); defensive
        return x_u as f64 - 0.5;
    }
    (pp - f_lo) / g_u + (x_u as f64 - 0.5)
}

/// Equipercentile equivalents `e_Y(x) = P_Y^{-1}(P_X(x))` for `x = 0..=k_x`.
fn equipercentile(gx: &[f64], gy: &[f64], k_x: usize, k_y: usize) -> Vec<f64> {
    let fx = cdf(gx);
    let fy = cdf(gy);
    (0..=k_x)
        .map(|x| perc_rank_inv(&fy, k_y, perc_rank(gx, &fx, k_x, x as f64)))
        .collect()
}

/// Normalized bivariate density table `(k_s+1) x (k_v+1)` (row-major, score by
/// anchor) from paired total/anchor score vectors.
fn bivariate(total: &[f64], anchor: &[f64], k_s: usize, k_v: usize) -> Result<Vec<f64>, String> {
    if total.len() != anchor.len() {
        return Err("total and anchor vectors must have equal length".into());
    }
    if total.is_empty() {
        return Err("score vectors must be non-empty".into());
    }
    let mut tab = vec![0.0_f64; (k_s + 1) * (k_v + 1)];
    for (&s, &v) in total.iter().zip(anchor) {
        if !s.is_finite() || !v.is_finite() {
            return Err("scores must be finite".into());
        }
        if s < -0.5 || s >= k_s as f64 + 0.5 || v < -0.5 || v >= k_v as f64 + 0.5 {
            return Err("bivariate score out of range".into());
        }
        let si = (s + 0.5).floor() as usize;
        let vi = (v + 0.5).floor() as usize;
        tab[si * (k_v + 1) + vi] += 1.0;
    }
    let n = total.len() as f64;
    for t in tab.iter_mut() {
        *t /= n;
    }
    Ok(tab)
}

/// Equivalent-groups (or single-group) observed-score equating of form X onto
/// form Y. `x_scores`/`y_scores` are raw integer total scores; `k_x`/`k_y` are the
/// maximum possible scores (number of items). See the module docs for the method
/// definitions.
pub fn equate_eg(
    x_scores: &[f64],
    y_scores: &[f64],
    k_x: usize,
    k_y: usize,
    method: EquateMethod,
) -> Result<EquateResult, String> {
    if k_x == 0 || k_y == 0 {
        return Err("k_x and k_y must be positive".into());
    }
    let gx = rel_freq(x_scores, k_x)?;
    let gy = rel_freq(y_scores, k_y)?;
    let (mu_x, sigma_x) = moments(&gx);
    let (mu_y, sigma_y) = moments(&gy);

    let (y_eq, slope, intercept) = match method {
        EquateMethod::Mean => {
            let b = mu_y - mu_x;
            ((0..=k_x).map(|x| x as f64 + b).collect::<Vec<_>>(), 1.0, b)
        }
        EquateMethod::Linear => {
            if sigma_x <= 0.0 {
                return Err("linear equating needs a positive SD on form X".into());
            }
            let a = sigma_y / sigma_x;
            let b = mu_y - a * mu_x;
            ((0..=k_x).map(|x| a * x as f64 + b).collect::<Vec<_>>(), a, b)
        }
        EquateMethod::Equipercentile => (equipercentile(&gx, &gy, k_x, k_y), f64::NAN, f64::NAN),
    };
    let (mu_eq, sigma_eq) = weighted_moments(&y_eq, &gx);
    Ok(EquateResult {
        x_scores: (0..=k_x).map(|x| x as f64).collect(),
        y_equivalents: y_eq,
        mu_x,
        sigma_x,
        mu_y,
        sigma_y,
        mu_eq,
        sigma_eq,
        slope,
        intercept,
        n_x: x_scores.len(),
        n_y: y_scores.len(),
        h_x: f64::NAN,
        h_y: f64::NAN,
    })
}

/// NEAT (common-item non-equivalent groups) equating. Population 1 takes form X
/// plus the anchor V (`x_total`, `x_anchor`); population 2 takes form Y plus the
/// anchor V (`y_total`, `y_anchor`). `w1` is the synthetic-population weight for
/// population 1 (`w2 = 1 - w1`); it is used only by frequency estimation and
/// ignored by chained equating.
#[allow(clippy::too_many_arguments)]
pub fn equate_neat(
    x_total: &[f64],
    x_anchor: &[f64],
    y_total: &[f64],
    y_anchor: &[f64],
    k_x: usize,
    k_y: usize,
    k_v: usize,
    w1: f64,
    method: NeatMethod,
) -> Result<EquateResult, String> {
    if k_x == 0 || k_y == 0 || k_v == 0 {
        return Err("k_x, k_y, k_v must be positive".into());
    }
    if x_total.len() != x_anchor.len() || y_total.len() != y_anchor.len() {
        return Err("total and anchor vectors must have equal length within each group".into());
    }
    let gx = rel_freq(x_total, k_x)?;
    let gy = rel_freq(y_total, k_y)?;
    let (mu_x, sigma_x) = moments(&gx);
    let (mu_y, sigma_y) = moments(&gy);

    let y_eq = match method {
        NeatMethod::ChainedEquipercentile => {
            // X -> V in population 1, then V -> Y in population 2 (K&B §5.2). The
            // intermediate v is real, so the real-argument percentile rank on the
            // pop-2 anchor distribution is required.
            let fx = cdf(&gx);
            let gv1 = rel_freq(x_anchor, k_v)?;
            let fv1 = cdf(&gv1);
            let gv2 = rel_freq(y_anchor, k_v)?;
            let fv2 = cdf(&gv2);
            let fy = cdf(&gy);
            (0..=k_x)
                .map(|x| {
                    let v = perc_rank_inv(&fv1, k_v, perc_rank(&gx, &fx, k_x, x as f64));
                    perc_rank_inv(&fy, k_y, perc_rank(&gv2, &fv2, k_v, v))
                })
                .collect::<Vec<_>>()
        }
        NeatMethod::FrequencyEstimation => {
            if !(0.0..=1.0).contains(&w1) {
                return Err("w1 must be in [0, 1]".into());
            }
            let w2 = 1.0 - w1;
            let n1 = bivariate(x_total, x_anchor, k_x, k_v)?;
            let n2 = bivariate(y_total, y_anchor, k_y, k_v)?;
            let stride1 = k_v + 1;
            // anchor marginals and form marginals
            let mut h1 = vec![0.0_f64; k_v + 1];
            let mut h2 = vec![0.0_f64; k_v + 1];
            let mut f1 = vec![0.0_f64; k_x + 1];
            let mut g2 = vec![0.0_f64; k_y + 1];
            for x in 0..=k_x {
                for v in 0..=k_v {
                    let p = n1[x * stride1 + v];
                    f1[x] += p;
                    h1[v] += p;
                }
            }
            for y in 0..=k_y {
                for v in 0..=k_v {
                    let p = n2[y * stride1 + v];
                    g2[y] += p;
                    h2[v] += p;
                }
            }
            // synthetic-population densities (K&B §5.3), skipping anchor points a
            // group never observed (division guard)
            let mut f_s = vec![0.0_f64; k_x + 1];
            for x in 0..=k_x {
                let mut cross = 0.0_f64;
                for v in 0..=k_v {
                    if h1[v] > 0.0 {
                        cross += (n1[x * stride1 + v] / h1[v]) * h2[v];
                    }
                }
                f_s[x] = w1 * f1[x] + w2 * cross;
            }
            let mut g_s = vec![0.0_f64; k_y + 1];
            for y in 0..=k_y {
                let mut cross = 0.0_f64;
                for v in 0..=k_v {
                    if h2[v] > 0.0 {
                        cross += (n2[y * stride1 + v] / h2[v]) * h1[v];
                    }
                }
                g_s[y] = w1 * cross + w2 * g2[y];
            }
            // Frequency estimation is undefined when the two groups share no
            // anchor score: every cross term drops out and the synthetic density
            // collapses (to naive EG, or — at w1 in {0,1} — to an all-zero vector
            // that would silently yield a boundary-only conversion table). Refuse
            // rather than return a silently-wrong result.
            let overlap = (0..=k_v).any(|v| h1[v] > 0.0 && h2[v] > 0.0);
            if !overlap {
                return Err(
                    "frequency estimation needs overlapping anchor support between the two groups"
                        .into(),
                );
            }
            // Partial non-overlap only drops the un-estimable anchor points;
            // renormalize the surviving mass so the cdf/boundary logic stays valid
            // (FE then operates on the shared anchor support).
            renormalize(&mut f_s);
            renormalize(&mut g_s);
            let y_eq = equipercentile(&f_s, &g_s, k_x, k_y);
            // report the synthetic moments actually equated
            let (msx, ssx) = moments(&f_s);
            let (msy, ssy) = moments(&g_s);
            let (mu_eq, sigma_eq) = weighted_moments(&y_eq, &f_s);
            return Ok(EquateResult {
                x_scores: (0..=k_x).map(|x| x as f64).collect(),
                y_equivalents: y_eq,
                mu_x: msx,
                sigma_x: ssx,
                mu_y: msy,
                sigma_y: ssy,
                mu_eq,
                sigma_eq,
                slope: f64::NAN,
                intercept: f64::NAN,
                n_x: x_total.len(),
                n_y: y_total.len(),
                h_x: f64::NAN,
                h_y: f64::NAN,
            });
        }
    };
    let (mu_eq, sigma_eq) = weighted_moments(&y_eq, &gx);
    Ok(EquateResult {
        x_scores: (0..=k_x).map(|x| x as f64).collect(),
        y_equivalents: y_eq,
        mu_x,
        sigma_x,
        mu_y,
        sigma_y,
        mu_eq,
        sigma_eq,
        slope: f64::NAN,
        intercept: f64::NAN,
        n_x: x_total.len(),
        n_y: y_total.len(),
        h_x: f64::NAN,
        h_y: f64::NAN,
    })
}

fn renormalize(v: &mut [f64]) {
    let s: f64 = v.iter().sum();
    if s > 0.0 {
        for x in v.iter_mut() {
            *x /= s;
        }
    }
}

// ===================== Tucker / Levine linear NEAT equating =====================

/// Linear observed-score NEAT equating method (the linear counterpart to the
/// chained/frequency-estimation equipercentile NEAT methods).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NeatLinearMethod {
    /// Tucker (equal total-on-anchor regression across populations).
    Tucker,
    /// Levine observed-score (classical-congeneric assumption).
    LevineObserved,
}

impl NeatLinearMethod {
    pub fn parse(name: &str) -> Option<NeatLinearMethod> {
        match name.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "tucker" | "t" => Some(NeatLinearMethod::Tucker),
            "levine" | "levineobserved" | "l" => Some(NeatLinearMethod::LevineObserved),
            _ => None,
        }
    }
}

/// Whether the anchor items count toward the total score (internal) or are a
/// separate section (external). Affects only the Levine gamma; Tucker is
/// anchor-kind-invariant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnchorKind {
    Internal,
    External,
}

impl AnchorKind {
    pub fn parse(name: &str) -> Option<AnchorKind> {
        match name.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "internal" | "int" => Some(AnchorKind::Internal),
            "external" | "ext" => Some(AnchorKind::External),
            _ => None,
        }
    }
}

/// Population moments of paired vectors `(mean_a, var_a, mean_b, var_b, cov)` with
/// the `N`-denominator convention (matching [`moments`]).
fn paired_moments(a: &[f64], b: &[f64]) -> (f64, f64, f64, f64, f64) {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let va = a.iter().map(|&x| (x - ma).powi(2)).sum::<f64>() / n;
    let vb = b.iter().map(|&x| (x - mb).powi(2)).sum::<f64>() / n;
    let cov = a.iter().zip(b).map(|(&x, &y)| (x - ma) * (y - mb)).sum::<f64>() / n;
    (ma, va, mb, vb, cov)
}

/// Tucker & Levine linear observed-score equating for the NEAT (common-item
/// non-equivalent groups) design (Kolen & Brennan, 2014, §4.3–4.4) — the linear
/// counterpart to [`equate_neat`]'s equipercentile methods. Population 1 takes
/// form X plus the anchor V (`x_total`, `x_anchor`); population 2 takes form Y plus
/// the anchor V (`y_total`, `y_anchor`). Each method forms synthetic-population
/// moments of X and Y (weighted by `w1`/`w2 = 1-w1`) using a group total-on-anchor
/// slope `gamma`, then equates linearly. Tucker uses the regression slope
/// `Cov(total, V)/Var(V)`; Levine uses the congeneric effective-length ratio,
/// which differs for an `Internal` anchor (`Var(total)/Cov`) versus an `External`
/// one (`(Var(total)+Cov)/(Var(V)+Cov)`). With equal anchor moments in the two
/// groups all variants collapse to the equivalent-groups linear equating.
///
/// # References (APA 7th ed.)
///
/// Kolen, M. J., & Brennan, R. L. (2014). *Test equating, scaling, and linking:
///   Methods and practices* (3rd ed.). Springer.
///   https://doi.org/10.1007/978-1-4939-0317-7
///
/// Brennan, R. L. (2006). *Chained linear equating* (CASMA Technical Report No. 3).
///   Center for Advanced Studies in Measurement and Assessment, University of Iowa.
#[allow(clippy::too_many_arguments)]
pub fn equate_neat_linear(
    x_total: &[f64],
    x_anchor: &[f64],
    y_total: &[f64],
    y_anchor: &[f64],
    k_x: usize,
    k_y: usize,
    w1: f64,
    method: NeatLinearMethod,
    anchor_kind: AnchorKind,
) -> Result<EquateResult, String> {
    if k_x == 0 || k_y == 0 {
        return Err("k_x and k_y must be positive".into());
    }
    if x_total.len() != x_anchor.len() || y_total.len() != y_anchor.len() {
        return Err("total and anchor vectors must have equal length within each group".into());
    }
    if x_total.is_empty() || y_total.is_empty() {
        return Err("score vectors must be non-empty".into());
    }
    if !(0.0..=1.0).contains(&w1) {
        return Err("w1 must be in [0, 1]".into());
    }
    if x_total.iter().chain(x_anchor).chain(y_total).chain(y_anchor).any(|v| !v.is_finite()) {
        return Err("scores must be finite".into());
    }
    let (m1x, v1x, m1v, v1v, cov1) = paired_moments(x_total, x_anchor);
    let (m2y, v2y, m2v, v2v, cov2) = paired_moments(y_total, y_anchor);
    if v1v <= 0.0 || v2v <= 0.0 {
        return Err("anchor variance must be positive in both groups".into());
    }
    let (g1, g2) = match method {
        NeatLinearMethod::Tucker => (cov1 / v1v, cov2 / v2v),
        NeatLinearMethod::LevineObserved => {
            if cov1 <= 0.0 || cov2 <= 0.0 {
                return Err("Levine equating needs a positive total-anchor covariance in both groups".into());
            }
            match anchor_kind {
                AnchorKind::Internal => (v1x / cov1, v2y / cov2),
                AnchorKind::External => {
                    ((v1x + cov1) / (v1v + cov1), (v2y + cov2) / (v2v + cov2))
                }
            }
        }
    };
    let w2 = 1.0 - w1;
    let dmu = m1v - m2v;
    let dv = v1v - v2v;
    let mu_sx = m1x - w2 * g1 * dmu;
    let mu_sy = m2y + w1 * g2 * dmu;
    let var_sx = v1x - w2 * g1 * g1 * dv + w1 * w2 * g1 * g1 * dmu * dmu;
    let var_sy = v2y + w1 * g2 * g2 * dv + w1 * w2 * g2 * g2 * dmu * dmu;
    if var_sx <= 0.0 || var_sy <= 0.0 {
        return Err("synthetic variance is non-positive (degenerate equating)".into());
    }
    let a = var_sy.sqrt() / var_sx.sqrt();
    let b = mu_sy - a * mu_sx;
    let y_eq: Vec<f64> = (0..=k_x).map(|x| a * x as f64 + b).collect();
    Ok(EquateResult {
        x_scores: (0..=k_x).map(|x| x as f64).collect(),
        y_equivalents: y_eq,
        mu_x: mu_sx,
        sigma_x: var_sx.sqrt(),
        mu_y: mu_sy,
        sigma_y: var_sy.sqrt(),
        // the linear conversion maps the synthetic X moments onto the synthetic Y
        // moments exactly, so the equated-score moments are (mu_sy, sigma_sy)
        mu_eq: mu_sy,
        sigma_eq: var_sy.sqrt(),
        slope: a,
        intercept: b,
        n_x: x_total.len(),
        n_y: y_total.len(),
        h_x: f64::NAN,
        h_y: f64::NAN,
    })
}

// ===================== standard errors of equating =====================

/// Per-score-point standard errors of equating ([`bootstrap_see`] /
/// [`analytic_see`]): the sampling error of the conversion `y_equivalents[x]`.
pub struct SeeResult {
    pub x_scores: Vec<f64>,
    /// Point estimate `e_Y(x)` from equating the full original sample.
    pub y_equivalents: Vec<f64>,
    /// Standard error of equating at each score point.
    pub se: Vec<f64>,
    pub ci_lo: Vec<f64>,
    pub ci_hi: Vec<f64>,
    /// Bootstrap replicate count (0 for the analytic route).
    pub n_boot: usize,
    pub ci_level: f64,
}

/// Type-7 (linear-interpolated, NumPy-default) quantile of a pre-sorted slice.
fn quantile_type7(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let h = (n as f64 - 1.0) * p;
    let lo = h.floor() as usize;
    let frac = h - lo as f64;
    if lo + 1 < n {
        sorted[lo] + frac * (sorted[lo + 1] - sorted[lo])
    } else {
        sorted[n - 1]
    }
}

/// Nonparametric bootstrap standard errors of equating for the equivalent-groups
/// design (Kolen & Brennan, 2014, ch. 7; Efron & Tibshirani, 1993). Resamples
/// examinees **with replacement, per group independently, at the observed sample
/// sizes** (the two forms are given to separate random samples, so their sampling
/// errors are independent), re-equates each of `n_boot` replicates via
/// [`equate_eg`] unchanged, and returns the per-score bootstrap SD (divisor
/// `n_boot - 1`) and a percentile confidence interval at `ci_level`. Works for all
/// three EG methods, including equipercentile (which has no simple analytic SEE).
///
/// # References (APA 7th ed.)
///
/// Kolen, M. J., & Brennan, R. L. (2014). *Test equating, scaling, and linking:
///   Methods and practices* (3rd ed.). Springer.
///
/// Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*.
///   Chapman & Hall.
#[allow(clippy::too_many_arguments)]
pub fn bootstrap_see(
    x_scores: &[f64],
    y_scores: &[f64],
    k_x: usize,
    k_y: usize,
    method: EquateMethod,
    n_boot: usize,
    ci_level: f64,
    seed: u64,
) -> Result<SeeResult, String> {
    if !(0.0 < ci_level && ci_level < 1.0) {
        return Err("ci_level must be in (0, 1)".into());
    }
    if n_boot < 2 {
        return Err("n_boot must be >= 2".into());
    }
    let point = equate_eg(x_scores, y_scores, k_x, k_y, method)?;
    let (nx, ny) = (x_scores.len(), y_scores.len());
    let ncol = k_x + 1;
    let mut reps = vec![0.0_f64; n_boot * ncol];
    let mut st = seed.max(1);
    let mut u = || {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut xb = vec![0.0_f64; nx];
    let mut yb = vec![0.0_f64; ny];
    for b in 0..n_boot {
        for v in xb.iter_mut() {
            *v = x_scores[((u() * nx as f64) as usize).min(nx - 1)];
        }
        for v in yb.iter_mut() {
            *v = y_scores[((u() * ny as f64) as usize).min(ny - 1)];
        }
        let r = equate_eg(&xb, &yb, k_x, k_y, method)?;
        reps[b * ncol..(b + 1) * ncol].copy_from_slice(&r.y_equivalents);
    }
    let alpha = 1.0 - ci_level;
    let mut se = vec![0.0_f64; ncol];
    let mut ci_lo = vec![0.0_f64; ncol];
    let mut ci_hi = vec![0.0_f64; ncol];
    let mut col = vec![0.0_f64; n_boot];
    for x in 0..ncol {
        for b in 0..n_boot {
            col[b] = reps[b * ncol + x];
        }
        let mean = col.iter().sum::<f64>() / n_boot as f64;
        let var = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n_boot as f64 - 1.0);
        se[x] = var.sqrt();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ci_lo[x] = quantile_type7(&col, alpha / 2.0);
        ci_hi[x] = quantile_type7(&col, 1.0 - alpha / 2.0);
    }
    Ok(SeeResult {
        x_scores: point.x_scores,
        y_equivalents: point.y_equivalents,
        se,
        ci_lo,
        ci_hi,
        n_boot,
        ci_level,
    })
}

/// Closed-form delta-method (normal-theory) standard errors of equating for the
/// Mean and Linear equivalent-groups methods (Kolen & Brennan, 2014, ch. 7;
/// Braun & Holland, 1982). With `z = (x - mu_x)/sigma_x`,
/// `Var[e_Y(x)] = sigma_x^2/n_x + sigma_y^2/n_y` (Mean, constant in `x`) or
/// `sigma_y^2 (1 + z^2/2)(1/n_x + 1/n_y)` (Linear); the interval is the
/// asymptotic-normal `point +/- z_c * SE`. Errors on equipercentile — bootstrap
/// that with [`bootstrap_see`]. Exact only for approximately normal score
/// distributions.
pub fn analytic_see(
    x_scores: &[f64],
    y_scores: &[f64],
    k_x: usize,
    k_y: usize,
    method: EquateMethod,
    ci_level: f64,
) -> Result<SeeResult, String> {
    if !(0.0 < ci_level && ci_level < 1.0) {
        return Err("ci_level must be in (0, 1)".into());
    }
    if method == EquateMethod::Equipercentile {
        return Err("analytic_see supports only Mean and Linear; use bootstrap_see for equipercentile".into());
    }
    let res = equate_eg(x_scores, y_scores, k_x, k_y, method)?;
    let (nx, ny) = (res.n_x as f64, res.n_y as f64);
    let (sx, sy) = (res.sigma_x, res.sigma_y);
    if sx <= 0.0 {
        return Err("analytic SEE needs a positive SD on form X".into());
    }
    let z_c = crate::nodes::inv_normal_cdf(1.0 - (1.0 - ci_level) / 2.0);
    let mut se = vec![0.0_f64; k_x + 1];
    let mut ci_lo = vec![0.0_f64; k_x + 1];
    let mut ci_hi = vec![0.0_f64; k_x + 1];
    for x in 0..=k_x {
        let var = match method {
            EquateMethod::Mean => sx * sx / nx + sy * sy / ny,
            EquateMethod::Linear => {
                let z = (x as f64 - res.mu_x) / sx;
                sy * sy * (1.0 + z * z / 2.0) * (1.0 / nx + 1.0 / ny)
            }
            EquateMethod::Equipercentile => unreachable!(),
        };
        se[x] = var.sqrt();
        ci_lo[x] = res.y_equivalents[x] - z_c * se[x];
        ci_hi[x] = res.y_equivalents[x] + z_c * se[x];
    }
    Ok(SeeResult {
        x_scores: res.x_scores,
        y_equivalents: res.y_equivalents,
        se,
        ci_lo,
        ci_hi,
        n_boot: 0,
        ci_level,
    })
}

// ===================== log-linear presmoothing =====================

/// Result of [`loglinear_smooth`].
pub struct LoglinearFit {
    /// Smoothed relative-frequency density (sums to 1).
    pub probs: Vec<f64>,
    /// Poisson log-likelihood (up to the `-sum ln(N_x!)` constant, so it is
    /// comparable across degrees on the *same* data, not across datasets).
    pub log_lik: f64,
    pub aic: f64,
    pub bic: f64,
    /// Fitted raw moments on the scaled score `u = x/k`, orders `1..=degree`;
    /// equal to the sample moments to numerical precision (the defining property).
    pub moments: Vec<f64>,
    pub converged: bool,
    pub iters: usize,
    /// Stopping rule that ended Newton iteration: `gradient_tolerance`,
    /// `log_likelihood_tolerance`, `line_search_stalled`, or `max_iter`.
    pub termination_reason: String,
    /// Maximum absolute Poisson score component at the returned coefficients.
    pub final_gradient_max: f64,
    /// Scale-adjusted score tolerance used for the convergence decision.
    pub gradient_tolerance: f64,
}

/// Orthonormal polynomial design matrix `B` of shape `(k+1) x (degree+1)` over the
/// scores `0..=k`: column `j` spans degree `j`, `B^T B = I`. Built by modified
/// Gram-Schmidt on the Vandermonde of a *centered/scaled* score `u = 2x/k - 1`
/// (raw-`x` powers are catastrophically ill-conditioned for `k` in the tens).
fn ortho_poly_design(k: usize, degree: usize) -> Vec<Vec<f64>> {
    let n = k + 1;
    let t = degree + 1;
    let u: Vec<f64> =
        (0..n).map(|x| if k == 0 { 0.0 } else { 2.0 * x as f64 / k as f64 - 1.0 }).collect();
    let mut cols: Vec<Vec<f64>> = (0..t).map(|j| u.iter().map(|&ui| ui.powi(j as i32)).collect()).collect();
    for j in 0..t {
        for i in 0..j {
            let dot: f64 = (0..n).map(|r| cols[j][r] * cols[i][r]).sum();
            for r in 0..n {
                cols[j][r] -= dot * cols[i][r];
            }
        }
        let norm: f64 = (0..n).map(|r| cols[j][r] * cols[j][r]).sum::<f64>().sqrt();
        if norm > 0.0 {
            for r in 0..n {
                cols[j][r] /= norm;
            }
        }
    }
    (0..n).map(|r| (0..t).map(|j| cols[j][r]).collect()).collect()
}

/// Univariate log-linear presmoothing of a score-frequency distribution (Holland &
/// Thayer, 2000; Kolen & Brennan, 2014, ch. 3): fits `log m_x = (B beta)_x` by
/// Poisson ML, so the smoothed density preserves the first `degree` sample moments
/// exactly while damping sampling noise. `counts` are raw frequencies over scores
/// `0..=k` (length `k+1`); `degree` is the number of moments preserved
/// (`degree = k` reproduces the raw relative frequencies).
///
/// # References (APA 7th ed.)
///
/// Holland, P. W., & Thayer, D. T. (2000). Univariate and bivariate loglinear
///   models for discrete test score distributions. *Journal of Educational and
///   Behavioral Statistics, 25*(2), 133–183. https://doi.org/10.3102/10769986025002133
pub fn loglinear_smooth(counts: &[f64], degree: usize) -> Result<LoglinearFit, String> {
    let n_cells = counts.len();
    if n_cells < 2 {
        return Err("counts must cover at least two scores".into());
    }
    let k = n_cells - 1;
    if degree < 1 || degree > k {
        return Err("degree must be in 1..=k".into());
    }
    if counts.iter().any(|&c| !c.is_finite() || c < 0.0) {
        return Err("counts must be finite and non-negative".into());
    }
    let total: f64 = counts.iter().sum();
    if total <= 0.0 {
        return Err("counts must sum to a positive total".into());
    }
    let t = degree + 1;
    let b = ortho_poly_design(k, degree);
    let eta_of = |beta: &[f64], x: usize| -> f64 { (0..t).map(|j| b[x][j] * beta[j]).sum() };
    let ll = |beta: &[f64]| -> f64 {
        (0..n_cells).map(|x| { let e = eta_of(beta, x); counts[x] * e - e.exp() }).sum()
    };
    let mut beta = vec![0.0_f64; t];
    let mut converged = false;
    let mut iters = 0usize;
    let mut termination_reason = "max_iter";
    let mut prev_ll = ll(&beta);
    // Scale-free tolerances: the gradient B^T(counts-m) is O(N) and the
    // log-likelihood O(N), so absolute floors would never be reached for large
    // samples (spuriously reporting non-convergence). Test relative to the total.
    let gtol = 1e-9 * total.max(1.0);
    const MAX_IT: usize = 50;
    for it in 0..MAX_IT {
        iters = it + 1;
        let m: Vec<f64> = (0..n_cells).map(|x| eta_of(&beta, x).exp()).collect();
        let grad: Vec<f64> =
            (0..t).map(|j| (0..n_cells).map(|x| b[x][j] * (counts[x] - m[x])).sum()).collect();
        let gmax = grad.iter().fold(0.0_f64, |a, &g| a.max(g.abs()));
        if gmax < gtol {
            converged = true;
            termination_reason = "gradient_tolerance";
            break;
        }
        let mut hess = vec![vec![0.0_f64; t]; t];
        for a in 0..t {
            for c in a..t {
                let v: f64 = (0..n_cells).map(|x| b[x][a] * m[x] * b[x][c]).sum();
                hess[a][c] = v;
                hess[c][a] = v;
            }
        }
        let step = crate::poly::solve_small(hess, grad);
        let ll_tol = 1e-12 * prev_ll.abs().max(1.0);
        let mut lambda = 1.0_f64;
        let mut accepted = false;
        for _ in 0..30 {
            let trial: Vec<f64> = (0..t).map(|j| beta[j] + lambda * step[j]).collect();
            let llt = ll(&trial);
            if llt >= prev_ll - ll_tol {
                beta = trial;
                // a step with negligible relative improvement means the fit has
                // plateaued at the optimum — treat as converged, not stalled
                if llt - prev_ll <= ll_tol {
                    converged = true;
                    termination_reason = "log_likelihood_tolerance";
                }
                prev_ll = llt;
                accepted = true;
                break;
            }
            lambda *= 0.5;
        }
        if !accepted {
            termination_reason = "line_search_stalled";
            break;
        }
        if converged {
            break;
        }
    }
    let m: Vec<f64> = (0..n_cells).map(|x| eta_of(&beta, x).exp()).collect();
    let final_gradient_max = (0..t)
        .map(|j| {
            (0..n_cells)
                .map(|x| b[x][j] * (counts[x] - m[x]))
                .sum::<f64>()
                .abs()
        })
        .fold(0.0_f64, f64::max);
    let msum: f64 = m.iter().sum();
    let probs: Vec<f64> = m.iter().map(|&mx| mx / msum).collect();
    let log_lik = ll(&beta);
    let p = t as f64;
    let aic = -2.0 * log_lik + 2.0 * p;
    let bic = -2.0 * log_lik + p * total.ln();
    let moments: Vec<f64> = (1..=degree)
        .map(|j| {
            (0..n_cells)
                .map(|x| { let u = if k == 0 { 0.0 } else { x as f64 / k as f64 }; u.powi(j as i32) * probs[x] })
                .sum()
        })
        .collect();
    Ok(LoglinearFit {
        probs,
        log_lik,
        aic,
        bic,
        moments,
        converged,
        iters,
        termination_reason: termination_reason.into(),
        final_gradient_max,
        gradient_tolerance: gtol,
    })
}

// ===================== Gaussian-kernel equating =====================

/// Continuization kernel for the equipercentile family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Continuization {
    /// Kolen-Brennan uniform kernel (linear cdf interpolation) — the default,
    /// identical to [`equate_eg`]'s equipercentile.
    Uniform,
    /// Gaussian kernel (von Davier, Holland & Thayer, 2004).
    Gaussian,
}

impl Continuization {
    pub fn parse(name: &str) -> Option<Continuization> {
        match name.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "uniform" | "kb" | "equipercentile" => Some(Continuization::Uniform),
            "gaussian" | "kernel" | "normal" => Some(Continuization::Gaussian),
            _ => None,
        }
    }
}

/// Options for [`equate_eg_ext`]: continuization kernel, optional per-form
/// log-linear presmoothing degree, and optional fixed Gaussian bandwidths (`None`
/// = penalty-selected).
pub struct EgSmoothOptions {
    pub continuization: Continuization,
    pub smooth_degree_x: Option<usize>,
    pub smooth_degree_y: Option<usize>,
    pub bandwidth_x: Option<f64>,
    pub bandwidth_y: Option<f64>,
}

fn norm_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}
fn norm_cdf(z: f64) -> f64 {
    0.5 * crate::fitstats::erfc(-z / std::f64::consts::SQRT_2)
}
fn kernel_a(sig2: f64, h: f64) -> f64 {
    (sig2 / (sig2 + h * h)).sqrt()
}
fn kernel_cdf(r: &[f64], mu: f64, sig2: f64, h: f64, x: f64) -> f64 {
    let a = kernel_a(sig2, h);
    let ah = a * h;
    r.iter().enumerate().map(|(j, &rj)| rj * norm_cdf((x - a * j as f64 - (1.0 - a) * mu) / ah)).sum()
}
fn kernel_pdf(r: &[f64], mu: f64, sig2: f64, h: f64, x: f64) -> f64 {
    let a = kernel_a(sig2, h);
    let ah = a * h;
    r.iter().enumerate().map(|(j, &rj)| rj * norm_pdf((x - a * j as f64 - (1.0 - a) * mu) / ah) / ah).sum()
}
fn kernel_dpdf(r: &[f64], mu: f64, sig2: f64, h: f64, x: f64) -> f64 {
    let a = kernel_a(sig2, h);
    let ah = a * h;
    r.iter()
        .enumerate()
        .map(|(j, &rj)| {
            let z = (x - a * j as f64 - (1.0 - a) * mu) / ah;
            rj * (-z) * norm_pdf(z) / (ah * ah)
        })
        .sum()
}
/// `G_h^{-1}(p)` by safeguarded Newton (bisection fallback); `F_h` is strictly
/// increasing with full support, so the root is unique.
fn kernel_inv(r: &[f64], mu: f64, sig2: f64, h: f64, p: f64, k: usize) -> f64 {
    let mut lo = -0.5_f64;
    let mut hi = k as f64 + 0.5;
    let mut guard = 0;
    while kernel_cdf(r, mu, sig2, h, lo) > p && guard < 200 {
        lo -= 1.0;
        guard += 1;
    }
    guard = 0;
    while kernel_cdf(r, mu, sig2, h, hi) < p && guard < 200 {
        hi += 1.0;
        guard += 1;
    }
    let mut x = 0.5 * (lo + hi);
    for _ in 0..100 {
        let fx = kernel_cdf(r, mu, sig2, h, x) - p;
        if fx.abs() < 1e-10 {
            break;
        }
        if fx > 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        let d = kernel_pdf(r, mu, sig2, h, x);
        let mut xn = if d > 1e-12 { x - fx / d } else { 0.5 * (lo + hi) };
        if !(xn > lo && xn < hi) {
            xn = 0.5 * (lo + hi);
        }
        x = xn;
    }
    x
}
fn kernel_equate(
    rx: &[f64], ry: &[f64], mu_x: f64, s2x: f64, mu_y: f64, s2y: f64, k_x: usize, k_y: usize,
    h_x: f64, h_y: f64,
) -> Vec<f64> {
    (0..=k_x)
        .map(|x| {
            let p = kernel_cdf(rx, mu_x, s2x, h_x, x as f64);
            kernel_inv(ry, mu_y, s2y, h_y, p, k_y)
        })
        .collect()
}
/// von Davier penalty: squared density mismatch at the score points plus a
/// unit penalty for each local density valley (an under-smoothing signature).
fn kernel_penalty(r: &[f64], mu: f64, sig2: f64, h: f64, k: usize) -> f64 {
    let delta = 1e-3;
    let mut pen = 0.0_f64;
    for j in 0..=k {
        let xj = j as f64;
        let d = r[j] - kernel_pdf(r, mu, sig2, h, xj);
        pen += d * d;
        let a_ind = kernel_dpdf(r, mu, sig2, h, xj - delta) < 0.0;
        let b_ind = kernel_dpdf(r, mu, sig2, h, xj + delta) > 0.0;
        if a_ind && b_ind {
            pen += 1.0;
        }
    }
    pen
}
/// Penalty-optimal bandwidth: coarse grid to bracket the (non-smooth) valley
/// indicator, then golden-section refinement to grid resolution (heuristic — any
/// `h` preserves the mean/variance, so this only tunes smoothing, not validity).
fn optimal_bandwidth(r: &[f64], mu: f64, sig2: f64, k: usize) -> f64 {
    let mut lo = 0.1_f64;
    let mut hi = 3.0_f64;
    let n_grid = 40usize;
    let mut best_h = lo;
    let mut best_p = f64::INFINITY;
    for i in 0..=n_grid {
        let h = lo + (hi - lo) * i as f64 / n_grid as f64;
        let p = kernel_penalty(r, mu, sig2, h, k);
        if p < best_p {
            best_p = p;
            best_h = h;
        }
    }
    if best_h <= lo + 1e-9 {
        lo = 0.02;
    }
    if best_h >= hi - 1e-9 {
        hi = 6.0;
    }
    let cell = (hi - lo) / n_grid as f64;
    let mut a = (best_h - cell).max(lo);
    let mut b = (best_h + cell).min(hi);
    let gr = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut c = b - gr * (b - a);
    let mut d = a + gr * (b - a);
    let mut fc = kernel_penalty(r, mu, sig2, c, k);
    let mut fd = kernel_penalty(r, mu, sig2, d, k);
    for _ in 0..30 {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - gr * (b - a);
            fc = kernel_penalty(r, mu, sig2, c, k);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + gr * (b - a);
            fd = kernel_penalty(r, mu, sig2, d, k);
        }
    }
    // The penalty is non-unimodal (a discontinuous valley indicator), so
    // golden-section can land in a worse cell than the grid already found; keep
    // whichever the penalty actually rates lower.
    let h_g = 0.5 * (a + b);
    if kernel_penalty(r, mu, sig2, h_g, k) <= best_p {
        h_g
    } else {
        best_h
    }
}

fn density(scores: &[f64], k: usize, smooth: Option<usize>) -> Result<Vec<f64>, String> {
    let g = rel_freq(scores, k)?;
    match smooth {
        None => Ok(g),
        Some(t) => {
            let n = scores.len() as f64;
            let counts: Vec<f64> = g.iter().map(|&p| p * n).collect();
            let fit = loglinear_smooth(&counts, t)?;
            if !fit.converged {
                return Err(format!(
                    "log-linear presmoothing did not converge: reason={}, iterations={}, max|score|={:.3e}, tolerance={:.3e}",
                    fit.termination_reason,
                    fit.iters,
                    fit.final_gradient_max,
                    fit.gradient_tolerance
                ));
            }
            if fit.probs.iter().any(|p| !p.is_finite()) {
                return Err("log-linear presmoothing returned non-finite probabilities".into());
            }
            Ok(fit.probs)
        }
    }
}

/// Equipercentile-family equivalent-groups equating with optional log-linear
/// presmoothing and a choice of continuization kernel (Kolen & Brennan, 2014; von
/// Davier, Holland & Thayer, 2004). With `Continuization::Uniform` and no
/// smoothing this is identical to [`equate_eg`]'s equipercentile method;
/// `Continuization::Gaussian` uses the Gaussian-kernel continuization, resolving
/// each form's bandwidth by the penalty method unless one is fixed in `opts`.
///
/// # References (APA 7th ed.)
///
/// von Davier, A. A., Holland, P. W., & Thayer, D. T. (2004). *The kernel method
///   of test equating*. Springer. https://doi.org/10.1007/b97446
pub fn equate_eg_ext(
    x_scores: &[f64],
    y_scores: &[f64],
    k_x: usize,
    k_y: usize,
    opts: EgSmoothOptions,
) -> Result<EquateResult, String> {
    if k_x == 0 || k_y == 0 {
        return Err("k_x and k_y must be positive".into());
    }
    let gx = density(x_scores, k_x, opts.smooth_degree_x)?;
    let gy = density(y_scores, k_y, opts.smooth_degree_y)?;
    let (mu_x, sigma_x) = moments(&gx);
    let (mu_y, sigma_y) = moments(&gy);

    let (y_eq, h_x, h_y) = match opts.continuization {
        // the uniform kernel ignores bandwidth entirely, so it is not validated here
        Continuization::Uniform => (equipercentile(&gx, &gy, k_x, k_y), f64::NAN, f64::NAN),
        Continuization::Gaussian => {
            for (h, nm) in [(opts.bandwidth_x, "bandwidth_x"), (opts.bandwidth_y, "bandwidth_y")] {
                if let Some(hv) = h {
                    if !hv.is_finite() || hv <= 0.0 {
                        return Err(format!("{nm} must be positive and finite"));
                    }
                }
            }
            let (s2x, s2y) = (sigma_x * sigma_x, sigma_y * sigma_y);
            if s2x <= 0.0 || s2y <= 0.0 {
                return Err("gaussian kernel equating needs a positive SD on both forms".into());
            }
            let hx = opts.bandwidth_x.unwrap_or_else(|| optimal_bandwidth(&gx, mu_x, s2x, k_x));
            let hy = opts.bandwidth_y.unwrap_or_else(|| optimal_bandwidth(&gy, mu_y, s2y, k_y));
            (kernel_equate(&gx, &gy, mu_x, s2x, mu_y, s2y, k_x, k_y, hx, hy), hx, hy)
        }
    };
    let (mu_eq, sigma_eq) = weighted_moments(&y_eq, &gx);
    Ok(EquateResult {
        x_scores: (0..=k_x).map(|x| x as f64).collect(),
        y_equivalents: y_eq,
        mu_x,
        sigma_x,
        mu_y,
        sigma_y,
        mu_eq,
        sigma_eq,
        slope: f64::NAN,
        intercept: f64::NAN,
        n_x: x_scores.len(),
        n_y: y_scores.len(),
        h_x,
        h_y,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Small LCG + Box-Muller for deterministic test data.
    fn lcg(seed: u64) -> impl FnMut() -> f64 {
        let mut st = seed.max(1);
        move || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((st >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }
    fn normal(u: &mut impl FnMut() -> f64) -> f64 {
        let u1 = u().max(1e-12);
        let u2 = u();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    // R1: equipercentile self-equating is the exact identity at every integer
    // score with positive frequency (the tightest correctness anchor).
    #[test]
    fn equate_self_is_identity() {
        let mut u = lcg(11);
        let k = 40usize;
        // a spread of scores covering the interior, all cells populated
        let scores: Vec<f64> =
            (0..4000).map(|_| (8.0 + 24.0 * normal(&mut u)).round().clamp(0.0, k as f64)).collect();
        let g = rel_freq(&scores, k).unwrap();
        let res = equate_eg(&scores, &scores, k, k, EquateMethod::Equipercentile).unwrap();
        let mut maxdev = 0.0_f64;
        for x in 0..=k {
            if g[x] > 0.0 {
                maxdev = maxdev.max((res.y_equivalents[x] - x as f64).abs());
            }
        }
        assert!(maxdev < 1e-9, "self-equate must be identity, maxdev={maxdev}");
        // includes x=0 whenever it has mass (the low-boundary interpolation)
        assert!(g[0] == 0.0 || (res.y_equivalents[0]).abs() < 1e-9);
    }

    // R2(a): closed-form moment methods recover the exact generating transform.
    #[test]
    fn equate_mean_linear_recover_transform() {
        let mut u = lcg(7);
        let k_x = 30usize;
        let x_scores: Vec<f64> =
            (0..5000).map(|_| (15.0 + 6.0 * normal(&mut u)).round().clamp(0.0, k_x as f64)).collect();
        // mean: Y = X + 5 exactly
        let c = 5.0;
        let y_mean: Vec<f64> = x_scores.iter().map(|&x| x + c).collect();
        let rm = equate_eg(&x_scores, &y_mean, k_x, k_x + 5, EquateMethod::Mean).unwrap();
        assert!((rm.intercept - c).abs() < 1e-9 && (rm.slope - 1.0).abs() < 1e-12);
        assert!(rm.y_equivalents.iter().enumerate().all(|(x, &y)| (y - (x as f64 + c)).abs() < 1e-9));
        // linear: Y = 2*X + 3 exactly (integer affine, positive slope)
        let (a, b) = (2.0_f64, 3.0_f64);
        let k_y = (a * k_x as f64 + b) as usize;
        let y_lin: Vec<f64> = x_scores.iter().map(|&x| a * x + b).collect();
        let rl = equate_eg(&x_scores, &y_lin, k_x, k_y, EquateMethod::Linear).unwrap();
        assert!((rl.slope - a).abs() < 1e-9, "slope {} != {a}", rl.slope);
        assert!((rl.intercept - b).abs() < 1e-9, "intercept {} != {b}", rl.intercept);
        assert!(rl.y_equivalents.iter().enumerate().all(|(x, &y)| (y - (a * x as f64 + b)).abs() < 1e-9));
    }

    // R3: with EQUAL anchor distributions (h_V1 = h_V2) and genuinely different X
    // vs Y forms, both NEAT methods collapse to EG equipercentile of X onto Y.
    // (Equal anchor marginals make the anchor cancel in chaining, and make the FE
    // synthetic density equal each group's own marginal.)
    #[test]
    fn neat_collapses_to_eg_under_equal_anchors() {
        let mut u = lcg(3);
        let n = 6000usize;
        let (k_x, k_y, k_v) = (30usize, 40usize, 15usize);
        // identical anchor score vector for both populations => h_V1 == h_V2 exactly
        let anchor: Vec<f64> =
            (0..n).map(|_| (7.0 + 3.0 * normal(&mut u)).round().clamp(0.0, k_v as f64)).collect();
        // different X and Y forms, correlated with the anchor but not equal to it
        let x_total: Vec<f64> = (0..n)
            .map(|i| (anchor[i] * 1.4 + 4.0 + 4.0 * normal(&mut u)).round().clamp(0.0, k_x as f64))
            .collect();
        let y_total: Vec<f64> = (0..n)
            .map(|i| (anchor[i] * 2.0 + 6.0 + 5.0 * normal(&mut u)).round().clamp(0.0, k_y as f64))
            .collect();

        let eg = equate_eg(&x_total, &y_total, k_x, k_y, EquateMethod::Equipercentile).unwrap();
        let ch = equate_neat(
            &x_total, &anchor, &y_total, &anchor, k_x, k_y, k_v, 0.5,
            NeatMethod::ChainedEquipercentile,
        )
        .unwrap();
        let fe = equate_neat(
            &x_total, &anchor, &y_total, &anchor, k_x, k_y, k_v, 0.5,
            NeatMethod::FrequencyEstimation,
        )
        .unwrap();
        let mut dmax_ch = 0.0_f64;
        let mut dmax_fe = 0.0_f64;
        for x in 0..=k_x {
            dmax_ch = dmax_ch.max((ch.y_equivalents[x] - eg.y_equivalents[x]).abs());
            dmax_fe = dmax_fe.max((fe.y_equivalents[x] - eg.y_equivalents[x]).abs());
        }
        assert!(dmax_ch < 1e-9, "chained must equal EG under equal anchors: {dmax_ch}");
        assert!(dmax_fe < 1e-9, "FE must equal EG under equal anchors: {dmax_fe}");
        // FE weight is inert here (h1==h2), so w1 in {0,1} agrees too
        for w1 in [0.0_f64, 1.0] {
            let fw = equate_neat(
                &x_total, &anchor, &y_total, &anchor, k_x, k_y, k_v, w1,
                NeatMethod::FrequencyEstimation,
            )
            .unwrap();
            let d = (0..=k_x).map(|x| (fw.y_equivalents[x] - eg.y_equivalents[x]).abs()).fold(0.0, f64::max);
            assert!(d < 1e-9, "FE(w1={w1}) must match EG under equal anchors: {d}");
        }
    }

    #[test]
    fn method_and_error_paths() {
        assert_eq!(EquateMethod::parse("EquiPercentile"), Some(EquateMethod::Equipercentile));
        assert_eq!(EquateMethod::parse("mean-mean"), None);
        assert_eq!(NeatMethod::parse("FE"), Some(NeatMethod::FrequencyEstimation));
        assert!(equate_eg(&[], &[1.0], 5, 5, EquateMethod::Mean).is_err());
        assert!(equate_eg(&[6.0], &[1.0], 5, 5, EquateMethod::Mean).is_err()); // out of range
        assert!(equate_neat(&[1.0, 2.0], &[1.0], &[1.0], &[1.0], 5, 5, 5, 0.5, NeatMethod::FrequencyEstimation).is_err());
        // out-of-range score (>= k+0.5) is now rejected (the old ±0.4 tolerance
        // on the already-rounded index silently binned it to a boundary cell)
        assert!(rel_freq(&[30.6], 30).is_err());
        assert!(rel_freq(&[-0.6], 30).is_err());
        // in-range fractional scores bin to the containing category interval:
        // 30.4 -> cat 30 ([29.5,30.5)), and -0.5 -> cat 0 ([-0.5,0.5))
        assert_eq!(rel_freq(&[30.4], 30).unwrap()[30], 1.0);
        assert_eq!(rel_freq(&[-0.5, 0.0, 1.0], 3).unwrap()[0], 2.0 / 3.0);
    }

    // FE requires the two groups to share anchor support; fully disjoint anchors
    // would otherwise silently collapse the synthetic density (finding: garbage
    // conversion table returned as Ok). Chained composition has no such
    // requirement and still returns a result.
    #[test]
    fn fe_rejects_disjoint_anchor_support() {
        let x_total = vec![1.0, 2.0, 3.0, 2.0, 1.0, 3.0];
        let x_anchor = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]; // support {0,1}
        let y_total = vec![2.0, 3.0, 1.0, 2.0, 3.0, 1.0];
        let y_anchor = vec![4.0, 5.0, 4.0, 5.0, 4.0, 5.0]; // support {4,5}
        assert!(equate_neat(
            &x_total, &x_anchor, &y_total, &y_anchor, 5, 5, 5, 0.5,
            NeatMethod::FrequencyEstimation,
        )
        .is_err());
        // also at the boundary weight w1=0 (the all-zero-density degenerate case)
        assert!(equate_neat(
            &x_total, &x_anchor, &y_total, &y_anchor, 5, 5, 5, 0.0,
            NeatMethod::FrequencyEstimation,
        )
        .is_err());
        assert!(equate_neat(
            &x_total, &x_anchor, &y_total, &y_anchor, 5, 5, 5, 0.5,
            NeatMethod::ChainedEquipercentile,
        )
        .is_ok());
    }

    // 2PL population number-correct density on a GH grid, via Lord-Wingersky.
    fn pop_density(a: &[f64], b: &[f64], nodes: &[f64], weights: &[f64]) -> Vec<f64> {
        let n_items = a.len();
        let n_nodes = nodes.len();
        let mut probs = vec![0.0_f64; n_items * n_nodes];
        for i in 0..n_items {
            for (t, &th) in nodes.iter().enumerate() {
                probs[i * n_nodes + t] = 1.0 / (1.0 + (-(a[i] * th + b[i])).exp());
            }
        }
        let f = crate::scoring::lord_wingersky(&probs, n_items, n_nodes);
        (0..=n_items)
            .map(|s| (0..n_nodes).map(|t| weights[t] * f[s * n_nodes + t]).sum())
            .collect()
    }

    fn interior_bias_rmse(
        a_x: &[f64], b_x: &[f64], a_y: &[f64], b_y: &[f64], n: usize, reps: usize, seed: u64,
    ) -> (f64, f64) {
        let (k_x, k_y) = (a_x.len(), a_y.len());
        let (nodes, weights) = crate::quadrature::gh_rule(41).unwrap();
        // deterministic population reference e_Y*(x)
        let gx_pop = pop_density(a_x, b_x, nodes, weights);
        let gy_pop = pop_density(a_y, b_y, nodes, weights);
        let e_ref = equipercentile(&gx_pop, &gy_pop, k_x, k_y);
        let mut u = lcg(seed);
        let mut sum = vec![0.0_f64; k_x + 1];
        let mut sum2 = vec![0.0_f64; k_x + 1];
        let sim = |u: &mut dyn FnMut() -> f64, a: &[f64], b: &[f64]| -> Vec<f64> {
            (0..n)
                .map(|_| {
                    let th = {
                        let u1 = u().max(1e-12);
                        let u2 = u();
                        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                    };
                    a.iter()
                        .zip(b)
                        .filter(|(&ai, &bi)| u() < 1.0 / (1.0 + (-(ai * th + bi)).exp()))
                        .count() as f64
                })
                .collect()
        };
        for _ in 0..reps {
            let xs = sim(&mut u, a_x, b_x);
            let ys = sim(&mut u, a_y, b_y);
            let est = equate_eg(&xs, &ys, k_x, k_y, EquateMethod::Equipercentile).unwrap();
            for x in 0..=k_x {
                let d = est.y_equivalents[x] - e_ref[x];
                sum[x] += d;
                sum2[x] += d * d;
            }
        }
        // trim the outer ~5% of the score range where zero-cell sampling dominates
        let lo = (k_x as f64 * 0.05).ceil() as usize;
        let hi = k_x - lo;
        let mut max_bias = 0.0_f64;
        let mut rmse_acc = 0.0_f64;
        let mut cnt = 0usize;
        for x in lo..=hi {
            max_bias = max_bias.max((sum[x] / reps as f64).abs());
            rmse_acc += sum2[x] / reps as f64;
            cnt += 1;
        }
        (max_bias, (rmse_acc / cnt as f64).sqrt())
    }

    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn equate_monte_carlo_500() {
        // distinct 2PL forms X (30 items) and Y (40 items)
        let k_x = 30usize;
        let k_y = 40usize;
        let a_x: Vec<f64> = (0..k_x).map(|i| 0.8 + 0.5 * ((i % 5) as f64 / 4.0)).collect();
        let b_x: Vec<f64> = (0..k_x).map(|i| 1.5 - 3.0 * i as f64 / (k_x - 1) as f64).collect();
        let a_y: Vec<f64> = (0..k_y).map(|i| 0.9 + 0.4 * ((i % 4) as f64 / 3.0)).collect();
        let b_y: Vec<f64> = (0..k_y).map(|i| 1.8 - 3.6 * i as f64 / (k_y - 1) as f64).collect();

        let reps = 500usize;
        let (bias1, rmse1) = interior_bias_rmse(&a_x, &b_x, &a_y, &b_y, 1000, reps, 4001);
        let (bias4, rmse4) = interior_bias_rmse(&a_x, &b_x, &a_y, &b_y, 4000, reps, 7001);
        let ratio = rmse1 / rmse4;
        println!(
            "[equate 500] N=1000: max|bias|={bias1:.4} RMSE={rmse1:.4}  \
             N=4000: max|bias|={bias4:.4} RMSE={rmse4:.4}  RMSE ratio={ratio:.3} (expect ~2)"
        );
        // the empirical equipercentile converges to the population equipercentile
        // of the same Lord-Wingersky densities (that population transform IS the
        // estimand; R1/R2/R3 supply the independent identification):
        assert!(bias1 < 0.15 && bias4 < 0.08, "bias should be small and shrink: {bias1}, {bias4}");
        assert!((1.6..=2.4).contains(&ratio), "RMSE should shrink ~1/sqrt(N): ratio={ratio}");
    }

    fn ext(cont: Continuization, sx: Option<usize>, sy: Option<usize>, hx: Option<f64>, hy: Option<f64>) -> EgSmoothOptions {
        EgSmoothOptions {
            continuization: cont,
            smooth_degree_x: sx,
            smooth_degree_y: sy,
            bandwidth_x: hx,
            bandwidth_y: hy,
        }
    }

    // Anchor 1: uniform-kernel ext == existing equipercentile, bit-exact.
    #[test]
    fn ext_uniform_matches_equipercentile() {
        let mut u = lcg(21);
        let (n, kx, ky) = (3000usize, 30usize, 30usize);
        let xs: Vec<f64> = (0..n).map(|_| (15.0 + 6.0 * normal(&mut u)).round().clamp(0.0, kx as f64)).collect();
        let ys: Vec<f64> = (0..n).map(|_| (14.0 + 7.0 * normal(&mut u)).round().clamp(0.0, ky as f64)).collect();
        let base = equate_eg(&xs, &ys, kx, ky, EquateMethod::Equipercentile).unwrap();
        let e = equate_eg_ext(&xs, &ys, kx, ky, ext(Continuization::Uniform, None, None, None, None)).unwrap();
        let d = (0..=kx).map(|x| (base.y_equivalents[x] - e.y_equivalents[x]).abs()).fold(0.0, f64::max);
        assert!(d < 1e-12, "uniform-kernel ext must equal equipercentile: {d}");
    }

    // Anchors 2 & 3: log-linear presmoothing preserves the first T sample moments
    // exactly (on the u=x/k scale) and, saturated at T=k, reproduces rel_freq.
    #[test]
    fn loglinear_preserves_moments_and_saturates() {
        let mut u = lcg(5);
        let k = 40usize;
        let scores: Vec<f64> = (0..5000).map(|_| (20.0 + 7.0 * normal(&mut u)).round().clamp(0.0, k as f64)).collect();
        let g = rel_freq(&scores, k).unwrap();
        let n = scores.len() as f64;
        let counts: Vec<f64> = g.iter().map(|&p| p * n).collect();
        let fit = loglinear_smooth(&counts, 4).unwrap();
        assert!(fit.converged);
        assert!((fit.probs.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(fit.probs.iter().all(|&p| p >= 0.0));
        for (j, &fm) in fit.moments.iter().enumerate() {
            let order = (j + 1) as i32;
            let sm: f64 = (0..=k).map(|x| (x as f64 / k as f64).powi(order) * g[x]).sum();
            assert!((fm - sm).abs() < 1e-8, "moment {order} not preserved: {fm} vs {sm}");
        }
        let sat = loglinear_smooth(&counts, k).unwrap();
        let d = (0..=k).map(|x| (sat.probs[x] - g[x]).abs()).fold(0.0, f64::max);
        assert!(d < 1e-9, "saturated loglinear must reproduce rel_freq: {d}");
    }

    #[test]
    fn equating_rejects_nonconverged_presmoothing() {
        let counts = [0usize, 1564, 426, 0, 1008, 0, 0];
        let scores: Vec<f64> = counts
            .iter()
            .enumerate()
            .flat_map(|(score, &count)| std::iter::repeat_n(score as f64, count))
            .collect();
        let fit = loglinear_smooth(
            &counts.iter().map(|&count| count as f64).collect::<Vec<_>>(),
            5,
        )
        .unwrap();
        assert!(!fit.converged, "fixture must exercise the non-converged path");
        assert_eq!(fit.termination_reason, "line_search_stalled");
        assert!(fit.final_gradient_max > fit.gradient_tolerance);

        let err = equate_eg_ext(
            &scores,
            &scores,
            6,
            6,
            ext(Continuization::Uniform, Some(5), Some(5), None, None),
        )
        .unwrap_err();
        assert!(err.contains("did not converge"), "unexpected error: {err}");
    }

    // Anchors 4 & 6: Gaussian-kernel self-equate is the identity (F_h == G_h), and
    // the continuized density preserves the discrete mean and variance.
    #[test]
    fn kernel_self_equate_and_mean_var() {
        let mut u = lcg(9);
        let k = 30usize;
        let xs: Vec<f64> = (0..4000).map(|_| (15.0 + 6.0 * normal(&mut u)).round().clamp(0.0, k as f64)).collect();
        let res = equate_eg_ext(&xs, &xs, k, k, ext(Continuization::Gaussian, None, None, Some(0.6), Some(0.6))).unwrap();
        let g = rel_freq(&xs, k).unwrap();
        let mut dmax = 0.0_f64;
        for x in 0..=k {
            if g[x] > 0.0 {
                dmax = dmax.max((res.y_equivalents[x] - x as f64).abs());
            }
        }
        // exact in exact arithmetic (F_h == G_h); the ~1e-8 residual is the
        // erfc approximation (|err| < 1.2e-7) through the numeric inverse
        assert!(dmax < 1e-6, "kernel self-equate must be identity: {dmax}");
        assert_eq!(res.h_x, 0.6);
        let (mu, sd) = moments(&g);
        let sig2 = sd * sd;
        let h = 0.8;
        let (lo, hi, steps) = (-6.0_f64, k as f64 + 6.0, 20000usize);
        let dx = (hi - lo) / steps as f64;
        let (mut m0, mut m1, mut m2) = (0.0_f64, 0.0, 0.0);
        for i in 0..steps {
            let x = lo + (i as f64 + 0.5) * dx;
            let fh = kernel_pdf(&g, mu, sig2, h, x);
            m0 += fh * dx;
            m1 += x * fh * dx;
            m2 += x * x * fh * dx;
        }
        let mean = m1 / m0;
        let var = m2 / m0 - mean * mean;
        assert!((mean - mu).abs() < 1e-3, "kernel mean {mean} != {mu}");
        assert!((var - sig2).abs() < 1e-2 * sig2.max(1.0), "kernel var {var} != {sig2}");
    }

    // Anchor 5: a very large bandwidth drives Gaussian-kernel equating to LINEAR.
    #[test]
    fn kernel_large_bandwidth_is_linear() {
        let mut u = lcg(13);
        let (kx, ky) = (30usize, 40usize);
        let xs: Vec<f64> = (0..4000).map(|_| (15.0 + 6.0 * normal(&mut u)).round().clamp(0.0, kx as f64)).collect();
        let ys: Vec<f64> = (0..4000).map(|_| (22.0 + 8.0 * normal(&mut u)).round().clamp(0.0, ky as f64)).collect();
        let lin = equate_eg(&xs, &ys, kx, ky, EquateMethod::Linear).unwrap();
        let ker = equate_eg_ext(&xs, &ys, kx, ky, ext(Continuization::Gaussian, None, None, Some(1e6), Some(1e6))).unwrap();
        let d = (0..=kx).map(|x| (lin.y_equivalents[x] - ker.y_equivalents[x]).abs()).fold(0.0, f64::max);
        assert!(d < 1e-4, "large-h kernel must match linear: {d}");
    }

    // Anchor 8: presmoothed self-equate is still the identity.
    #[test]
    fn presmoothed_self_equate_is_identity() {
        let mut u = lcg(17);
        let k = 40usize;
        let xs: Vec<f64> = (0..3000).map(|_| (20.0 + 7.0 * normal(&mut u)).round().clamp(0.0, k as f64)).collect();
        let res = equate_eg_ext(&xs, &xs, k, k, ext(Continuization::Uniform, Some(5), Some(5), None, None)).unwrap();
        let g = density(&xs, k, Some(5)).unwrap();
        let mut dmax = 0.0_f64;
        for x in 0..=k {
            if g[x] > 1e-12 {
                dmax = dmax.max((res.y_equivalents[x] - x as f64).abs());
            }
        }
        assert!(dmax < 1e-8, "presmoothed self-equate must be identity: {dmax}");
    }

    // Fix guard: on a non-unimodal penalty (bimodal density) the golden-section
    // refinement can land in a worse cell, so optimal_bandwidth must fall back to
    // the grid best rather than ship it.
    #[test]
    fn optimal_bandwidth_never_worse_than_grid() {
        let k = 40usize;
        let mut r = vec![0.0_f64; k + 1];
        for j in 0..=k {
            let d1 = (j as f64 - 8.0) / 2.0;
            let d2 = (j as f64 - 32.0) / 2.0;
            r[j] = (-0.5 * d1 * d1).exp() + (-0.5 * d2 * d2).exp();
        }
        let s: f64 = r.iter().sum();
        for v in r.iter_mut() {
            *v /= s;
        }
        let (mu, sd) = moments(&r);
        let sig2 = sd * sd;
        let h = optimal_bandwidth(&r, mu, sig2, k);
        assert!(h.is_finite() && h > 0.0);
        let pen_h = kernel_penalty(&r, mu, sig2, h, k);
        let grid_best = (0..=40)
            .map(|i| kernel_penalty(&r, mu, sig2, 0.1 + (3.0 - 0.1) * i as f64 / 40.0, k))
            .fold(f64::INFINITY, f64::min);
        assert!(pen_h <= grid_best + 1e-12, "optimal_bandwidth worse than grid: {pen_h} vs {grid_best}");
    }

    // Gaussian-kernel MC with a FIXED bandwidth shared by the population reference
    // and the per-rep estimator, so the assertion measures density-sampling error
    // alone (penalty-selected h would inject selection noise).
    fn kernel_bias_rmse(
        a_x: &[f64], b_x: &[f64], a_y: &[f64], b_y: &[f64], n: usize, reps: usize, seed: u64, h: f64,
    ) -> (f64, f64) {
        let (k_x, k_y) = (a_x.len(), a_y.len());
        let (nodes, weights) = crate::quadrature::gh_rule(41).unwrap();
        let gx_pop = pop_density(a_x, b_x, nodes, weights);
        let gy_pop = pop_density(a_y, b_y, nodes, weights);
        let (mux, sdx) = moments(&gx_pop);
        let (muy, sdy) = moments(&gy_pop);
        let e_ref = kernel_equate(&gx_pop, &gy_pop, mux, sdx * sdx, muy, sdy * sdy, k_x, k_y, h, h);
        let mut u = lcg(seed);
        let sim = |u: &mut dyn FnMut() -> f64, a: &[f64], b: &[f64]| -> Vec<f64> {
            (0..n)
                .map(|_| {
                    let th = {
                        let u1 = u().max(1e-12);
                        let u2 = u();
                        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                    };
                    a.iter().zip(b).filter(|(&ai, &bi)| u() < 1.0 / (1.0 + (-(ai * th + bi)).exp())).count() as f64
                })
                .collect()
        };
        let mut sum = vec![0.0_f64; k_x + 1];
        let mut sum2 = vec![0.0_f64; k_x + 1];
        for _ in 0..reps {
            let xs = sim(&mut u, a_x, b_x);
            let ys = sim(&mut u, a_y, b_y);
            let est = equate_eg_ext(&xs, &ys, k_x, k_y, ext(Continuization::Gaussian, None, None, Some(h), Some(h))).unwrap();
            for x in 0..=k_x {
                let d = est.y_equivalents[x] - e_ref[x];
                sum[x] += d;
                sum2[x] += d * d;
            }
        }
        let lo = (k_x as f64 * 0.05).ceil() as usize;
        let hi = k_x - lo;
        let mut max_bias = 0.0_f64;
        let mut rmse_acc = 0.0_f64;
        let mut cnt = 0usize;
        for x in lo..=hi {
            max_bias = max_bias.max((sum[x] / reps as f64).abs());
            rmse_acc += sum2[x] / reps as f64;
            cnt += 1;
        }
        (max_bias, (rmse_acc / cnt as f64).sqrt())
    }

    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn kernel_equate_monte_carlo_500() {
        let k_x = 30usize;
        let k_y = 40usize;
        let a_x: Vec<f64> = (0..k_x).map(|i| 0.8 + 0.5 * ((i % 5) as f64 / 4.0)).collect();
        let b_x: Vec<f64> = (0..k_x).map(|i| 1.5 - 3.0 * i as f64 / (k_x - 1) as f64).collect();
        let a_y: Vec<f64> = (0..k_y).map(|i| 0.9 + 0.4 * ((i % 4) as f64 / 3.0)).collect();
        let b_y: Vec<f64> = (0..k_y).map(|i| 1.8 - 3.6 * i as f64 / (k_y - 1) as f64).collect();
        let reps = 500usize;
        let h = 0.6_f64;
        let (bias1, rmse1) = kernel_bias_rmse(&a_x, &b_x, &a_y, &b_y, 1000, reps, 5001, h);
        let (bias4, rmse4) = kernel_bias_rmse(&a_x, &b_x, &a_y, &b_y, 4000, reps, 8001, h);
        let ratio = rmse1 / rmse4;
        println!(
            "[kernel equate 500] h={h} N=1000: max|bias|={bias1:.4} RMSE={rmse1:.4}  \
             N=4000: max|bias|={bias4:.4} RMSE={rmse4:.4}  RMSE ratio={ratio:.3} (expect ~2)"
        );
        assert!(bias1 < 0.15 && bias4 < 0.08, "bias should be small and shrink: {bias1}, {bias4}");
        assert!((1.6..=2.4).contains(&ratio), "RMSE should shrink ~1/sqrt(N): {ratio}");
    }

    // Primary anchor: with equal anchor moments (a shared anchor vector) every
    // Tucker/Levine variant collapses to EG linear equating of X onto Y, for any
    // w1 and anchor kind.
    #[test]
    fn neat_linear_collapses_to_eg_linear() {
        let (kx, ky) = (30usize, 40usize);
        let mut u = lcg(41);
        let n = 4000usize;
        // a shared anchor vector (equal anchor moments by construction) that is
        // genuinely correlated with both totals (so Levine's covariance is positive)
        let anchor: Vec<f64> = (0..n).map(|_| (7.0 + 3.0 * normal(&mut u)).round().clamp(0.0, 15.0)).collect();
        let x_total: Vec<f64> =
            anchor.iter().map(|&v| (1.5 * v + 4.0 + 3.0 * normal(&mut u)).round().clamp(0.0, kx as f64)).collect();
        let y_total: Vec<f64> =
            anchor.iter().map(|&v| (1.8 * v + 6.0 + 4.0 * normal(&mut u)).round().clamp(0.0, ky as f64)).collect();
        let eg = equate_eg(&x_total, &y_total, kx, ky, EquateMethod::Linear).unwrap();
        for m in [NeatLinearMethod::Tucker, NeatLinearMethod::LevineObserved] {
            for ak in [AnchorKind::Internal, AnchorKind::External] {
                for w1 in [0.0_f64, 0.5, 1.0] {
                    let r = equate_neat_linear(&x_total, &anchor, &y_total, &anchor, kx, ky, w1, m, ak).unwrap();
                    assert!(
                        (r.slope - eg.slope).abs() < 1e-9 && (r.intercept - eg.intercept).abs() < 1e-9,
                        "collapse {m:?}/{ak:?}/w1={w1}: slope {} vs {}, int {} vs {}",
                        r.slope, eg.slope, r.intercept, eg.intercept
                    );
                    let d = (0..=kx).map(|x| (r.y_equivalents[x] - eg.y_equivalents[x]).abs()).fold(0.0, f64::max);
                    assert!(d < 1e-9, "table mismatch: {d}");
                }
            }
        }
    }

    // Pins the internal-vs-external Levine gamma (the crux) against a NumPy oracle
    // (N-denominator moments): the three gamma branches give three distinct
    // slope/intercept pairs.
    #[test]
    fn neat_linear_gamma_hand_computed() {
        let x1 = [3.0, 5., 7., 9., 4., 6., 8., 2.];
        let v1 = [1.0, 2., 2., 3., 1., 2., 3., 1.];
        let y2 = [2.0, 5., 8., 11., 4., 7., 10., 1.];
        let v2 = [2.0, 4., 4., 6., 3., 5., 6., 2.];
        let (kx, ky, w1) = (11usize, 11usize, 0.5_f64);
        let tk = equate_neat_linear(&x1, &v1, &y2, &v2, kx, ky, w1, NeatLinearMethod::Tucker, AnchorKind::Internal).unwrap();
        assert!((tk.slope - 0.8006819908).abs() < 1e-8 && (tk.intercept + 3.0616870634).abs() < 1e-8, "tucker {} {}", tk.slope, tk.intercept);
        let li = equate_neat_linear(&x1, &v1, &y2, &v2, kx, ky, w1, NeatLinearMethod::LevineObserved, AnchorKind::Internal).unwrap();
        assert!((li.slope - 0.7403094687).abs() < 1e-8 && (li.intercept + 3.0252464118).abs() < 1e-8, "levine-int {} {}", li.slope, li.intercept);
        let le = equate_neat_linear(&x1, &v1, &y2, &v2, kx, ky, w1, NeatLinearMethod::LevineObserved, AnchorKind::External).unwrap();
        assert!((le.slope - 0.7550256824).abs() < 1e-8 && (le.intercept + 3.017543311).abs() < 1e-8, "levine-ext {} {}", le.slope, le.intercept);
        // Tucker ignores the anchor kind
        let tk2 = equate_neat_linear(&x1, &v1, &y2, &v2, kx, ky, w1, NeatLinearMethod::Tucker, AnchorKind::External).unwrap();
        assert_eq!(tk.slope, tk2.slope);
        assert_eq!(NeatLinearMethod::parse("levine"), Some(NeatLinearMethod::LevineObserved));
        assert_eq!(AnchorKind::parse("ext"), Some(AnchorKind::External));
        // error paths: bad w1, constant anchor (zero variance), Levine on a zero-cov anchor
        assert!(equate_neat_linear(&x1, &v1, &y2, &v2, kx, ky, 1.5, NeatLinearMethod::Tucker, AnchorKind::Internal).is_err());
        let const_v = [2.0_f64; 8];
        assert!(equate_neat_linear(&x1, &const_v, &y2, &v2, kx, ky, w1, NeatLinearMethod::Tucker, AnchorKind::Internal).is_err());
    }

    // Common-regression generative model (satisfies the Tucker assumption); the
    // estimator's equated table converges to the large-N reference at ~1/sqrt(N).
    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn neat_linear_monte_carlo_500() {
        let (kt_x, kt_y, kv) = (40usize, 45usize, 15usize);
        let (sdv, beta, tau) = (2.5_f64, 1.2_f64, 3.0_f64);
        let gen = |u: &mut dyn FnMut() -> f64, n: usize, muv: f64, alpha: f64, kt: usize| -> (Vec<f64>, Vec<f64>) {
            let nd = |u: &mut dyn FnMut() -> f64| {
                let u1 = u().max(1e-12);
                let u2 = u();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            };
            let mut tot = vec![0.0_f64; n];
            let mut anc = vec![0.0_f64; n];
            for i in 0..n {
                let v = muv + sdv * nd(u);
                let t = alpha + beta * v + tau * nd(u);
                anc[i] = v.round().clamp(0.0, kv as f64);
                tot[i] = t.round().clamp(0.0, kt as f64);
            }
            (tot, anc)
        };
        // reference from a large calibration draw through the same sampler+rounding
        let mut ur = lcg(9100);
        let (rx, rxa) = gen(&mut ur, 2_000_000, 6.0, 5.0, kt_x);
        let (ry, rya) = gen(&mut ur, 2_000_000, 9.0, 8.0, kt_y);
        let e_ref = equate_neat_linear(&rx, &rxa, &ry, &rya, kt_x, kt_y, 0.5, NeatLinearMethod::Tucker, AnchorKind::Internal).unwrap();
        let bias_rmse = |n: usize, seed: u64| -> (f64, f64) {
            let mut u = lcg(seed);
            let reps = 500usize;
            let mut sum = vec![0.0_f64; kt_x + 1];
            let mut sum2 = vec![0.0_f64; kt_x + 1];
            for _ in 0..reps {
                let (xt, xa) = gen(&mut u, n, 6.0, 5.0, kt_x);
                let (yt, ya) = gen(&mut u, n, 9.0, 8.0, kt_y);
                let est = equate_neat_linear(&xt, &xa, &yt, &ya, kt_x, kt_y, 0.5, NeatLinearMethod::Tucker, AnchorKind::Internal).unwrap();
                for x in 0..=kt_x {
                    let d = est.y_equivalents[x] - e_ref.y_equivalents[x];
                    sum[x] += d;
                    sum2[x] += d * d;
                }
            }
            let lo = (kt_x as f64 * 0.05).ceil() as usize;
            let hi = kt_x - lo;
            let (mut mb, mut ra, mut c) = (0.0_f64, 0.0_f64, 0usize);
            for x in lo..=hi {
                mb = mb.max((sum[x] / reps as f64).abs());
                ra += sum2[x] / reps as f64;
                c += 1;
            }
            (mb, (ra / c as f64).sqrt())
        };
        let (b1, r1) = bias_rmse(1000, 111);
        let (b4, r4) = bias_rmse(4000, 222);
        let ratio = r1 / r4;
        println!("[neat-linear 500] N=1000: max|bias|={b1:.4} RMSE={r1:.4}  N=4000: max|bias|={b4:.4} RMSE={r4:.4}  ratio={ratio:.3}");
        assert!(b1 < 0.20 && b4 < 0.10, "bias should be small and shrink: {b1}, {b4}");
        assert!((1.6..=2.4).contains(&ratio), "RMSE should shrink ~1/sqrt(N): {ratio}");
    }

    // helper: two near-normal EG samples of size n
    fn see_gen(u: &mut impl FnMut() -> f64, n: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
        let xs = (0..n).map(|_| (15.0 + 5.0 * normal(u)).round().clamp(0.0, k as f64)).collect();
        let ys = (0..n).map(|_| (16.0 + 5.0 * normal(u)).round().clamp(0.0, k as f64)).collect();
        (xs, ys)
    }

    // A1: delta-method Linear SEE agrees with the bootstrap Linear SEE.
    #[test]
    fn see_analytic_linear_matches_bootstrap() {
        let mut u = lcg(71);
        let (k, n) = (30usize, 3000usize);
        let (xs, ys) = see_gen(&mut u, n, k);
        let a = analytic_see(&xs, &ys, k, k, EquateMethod::Linear, 0.95).unwrap();
        let b = bootstrap_see(&xs, &ys, k, k, EquateMethod::Linear, 2000, 0.95, 12345).unwrap();
        let (lo, hi) = ((k as f64 * 0.1).ceil() as usize, k - (k as f64 * 0.1).ceil() as usize);
        let mut maxrel = 0.0_f64;
        for x in lo..=hi {
            if a.se[x] > 1e-6 {
                maxrel = maxrel.max((b.se[x] - a.se[x]).abs() / a.se[x]);
            }
        }
        assert!(maxrel < 0.15, "analytic vs bootstrap Linear SEE relative gap too large: {maxrel}");
    }

    // A2: Mean SEE is constant in x and equals the closed form.
    #[test]
    fn see_mean_is_constant() {
        let mut u = lcg(72);
        let (k, n) = (30usize, 2000usize);
        let (xs, ys) = see_gen(&mut u, n, k);
        let a = analytic_see(&xs, &ys, k, k, EquateMethod::Mean, 0.95).unwrap();
        let (_, sx) = moments(&rel_freq(&xs, k).unwrap());
        let (_, sy) = moments(&rel_freq(&ys, k).unwrap());
        let expected = (sx * sx / n as f64 + sy * sy / n as f64).sqrt();
        for x in 0..=k {
            assert!((a.se[x] - expected).abs() < 1e-9 && (a.se[x] - a.se[0]).abs() < 1e-12, "Mean SEE not constant");
        }
    }

    // A3/A4: bootstrap sanity (positive SE, CI brackets the estimate, ~1/sqrt(N)
    // shrink), determinism, and the input guards.
    #[test]
    fn see_bootstrap_sanity_and_guards() {
        let mut u = lcg(73);
        let k = 20usize;
        let (x1, y1) = see_gen(&mut u, 1000, k);
        let (x4, y4) = see_gen(&mut u, 4000, k);
        let b1 = bootstrap_see(&x1, &y1, k, k, EquateMethod::Equipercentile, 500, 0.95, 7).unwrap();
        let b4 = bootstrap_see(&x4, &y4, k, k, EquateMethod::Equipercentile, 500, 0.95, 7).unwrap();
        let (lo, hi) = ((k as f64 * 0.1).ceil() as usize, k - (k as f64 * 0.1).ceil() as usize);
        for x in lo..=hi {
            assert!(b1.se[x] > 0.0);
            assert!(b1.ci_lo[x] <= b1.y_equivalents[x] + 1e-9 && b1.y_equivalents[x] <= b1.ci_hi[x] + 1e-9);
        }
        let ratio: f64 = (lo..=hi).map(|x| b1.se[x] / b4.se[x].max(1e-9)).sum::<f64>() / (hi - lo + 1) as f64;
        assert!((1.5..=2.6).contains(&ratio), "SE should ~halve when N x4: {ratio}");
        // determinism
        let d1 = bootstrap_see(&x1, &y1, k, k, EquateMethod::Linear, 300, 0.95, 99).unwrap();
        let d2 = bootstrap_see(&x1, &y1, k, k, EquateMethod::Linear, 300, 0.95, 99).unwrap();
        assert_eq!(d1.se, d2.se);
        // guards
        assert!(bootstrap_see(&x1, &y1, k, k, EquateMethod::Mean, 1, 0.95, 1).is_err());
        assert!(bootstrap_see(&x1, &y1, k, k, EquateMethod::Mean, 100, 1.5, 1).is_err());
        assert!(analytic_see(&x1, &y1, k, k, EquateMethod::Equipercentile, 0.95).is_err());
    }

    // The bootstrap SE approximates the TRUE sampling SD of e_Y(x) (from an outer
    // Monte-Carlo that redraws fresh 2PL samples) within Monte-Carlo tolerance.
    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn see_bootstrap_monte_carlo_500() {
        let (k_x, k_y, n) = (30usize, 40usize, 2000usize);
        let a_x: Vec<f64> = (0..k_x).map(|i| 0.8 + 0.5 * ((i % 5) as f64 / 4.0)).collect();
        let b_x: Vec<f64> = (0..k_x).map(|i| 1.5 - 3.0 * i as f64 / (k_x - 1) as f64).collect();
        let a_y: Vec<f64> = (0..k_y).map(|i| 0.9 + 0.4 * ((i % 4) as f64 / 3.0)).collect();
        let b_y: Vec<f64> = (0..k_y).map(|i| 1.8 - 3.6 * i as f64 / (k_y - 1) as f64).collect();
        let sim = |u: &mut dyn FnMut() -> f64, a: &[f64], b: &[f64]| -> Vec<f64> {
            (0..n)
                .map(|_| {
                    let th = {
                        let u1 = u().max(1e-12);
                        let u2 = u();
                        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                    };
                    a.iter().zip(b).filter(|(&ai, &bi)| u() < 1.0 / (1.0 + (-(ai * th + bi)).exp())).count() as f64
                })
                .collect()
        };
        let run = |method: EquateMethod, label: &str| {
            // outer MC: true SD of e_Y(x) over R fresh samples
            let r_out = 500usize;
            let mut uo = lcg(3300);
            let mut vals = vec![0.0_f64; r_out * (k_x + 1)];
            for r in 0..r_out {
                let xs = sim(&mut uo, &a_x, &b_x);
                let ys = sim(&mut uo, &a_y, &b_y);
                let e = equate_eg(&xs, &ys, k_x, k_y, method).unwrap();
                vals[r * (k_x + 1)..(r + 1) * (k_x + 1)].copy_from_slice(&e.y_equivalents);
            }
            let true_sd: Vec<f64> = (0..=k_x)
                .map(|x| {
                    let col: Vec<f64> = (0..r_out).map(|r| vals[r * (k_x + 1) + x]).collect();
                    let m = col.iter().sum::<f64>() / r_out as f64;
                    (col.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / (r_out as f64 - 1.0)).sqrt()
                })
                .collect();
            // mean bootstrap SE over n_samp fresh samples
            let n_samp = 40usize;
            let mut ub = lcg(9900);
            let mut sum_se = vec![0.0_f64; k_x + 1];
            for s_i in 0..n_samp {
                let xs = sim(&mut ub, &a_x, &b_x);
                let ys = sim(&mut ub, &a_y, &b_y);
                let s = bootstrap_see(&xs, &ys, k_x, k_y, method, 300, 0.95, 41_000 + s_i as u64).unwrap();
                for x in 0..=k_x {
                    sum_se[x] += s.se[x];
                }
            }
            let (lo, hi) = ((k_x as f64 * 0.05).ceil() as usize, k_x - (k_x as f64 * 0.05).ceil() as usize);
            let (mut rmin, mut rmax) = (f64::INFINITY, f64::NEG_INFINITY);
            for x in lo..=hi {
                let ratio = (sum_se[x] / n_samp as f64) / true_sd[x].max(1e-9);
                rmin = rmin.min(ratio);
                rmax = rmax.max(ratio);
            }
            println!("[see 500] {label}: interior boot/true SD ratio in [{rmin:.3}, {rmax:.3}]");
            assert!(rmin > 0.80 && rmax < 1.20, "{label} bootstrap SEE off true SD: [{rmin}, {rmax}]");
        };
        run(EquateMethod::Linear, "linear");
        run(EquateMethod::Equipercentile, "equipercentile");
    }
}
