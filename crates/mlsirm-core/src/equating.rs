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

pub const MAX_EQUATING_SCORE_POINTS: usize = 10_000;
pub const MAX_EQUATING_BIVARIATE_CELLS: usize = 1_000_000;

/// Relative-frequency vector `g(0..=k)` from raw integer scores. Errors on empty
/// input or a score outside `0..=k`.
fn rel_freq(scores: &[f64], k: usize) -> Result<Vec<f64>, String> {
    if scores.is_empty() {
        return Err("score vector must be non-empty".into());
    }
    if k > MAX_EQUATING_SCORE_POINTS {
        return Err(format!(
            "score ceiling must be <= {MAX_EQUATING_SCORE_POINTS}"
        ));
    }
    let n_points = k
        .checked_add(1)
        .ok_or("score ceiling exceeds the frequency buffer size")?;
    let mut freq = vec![0.0_f64; n_points];
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
    let var: f64 = g
        .iter()
        .enumerate()
        .map(|(x, &p)| (x as f64 - mean).powi(2) * p)
        .sum();
    (mean, var.max(0.0).sqrt())
}

/// Mean/SD of the equated scores `y_eq` weighted by form X's distribution `gx`.
fn weighted_moments(y_eq: &[f64], gx: &[f64]) -> (f64, f64) {
    let mean: f64 = y_eq.iter().zip(gx).map(|(&y, &w)| y * w).sum();
    let var: f64 = y_eq
        .iter()
        .zip(gx)
        .map(|(&y, &w)| (y - mean).powi(2) * w)
        .sum();
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
    // The selected cell satisfies F(x_u) > pp >= F(x_u-1), hence g_u > 0.
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
    if k_s > MAX_EQUATING_SCORE_POINTS || k_v > MAX_EQUATING_SCORE_POINTS {
        return Err(format!(
            "score ceiling must be <= {MAX_EQUATING_SCORE_POINTS}"
        ));
    }
    let n_s = k_s
        .checked_add(1)
        .ok_or("score ceiling exceeds the bivariate buffer size")?;
    let n_v = k_v
        .checked_add(1)
        .ok_or("anchor ceiling exceeds the bivariate buffer size")?;
    let n_cells = crate::checked_mul_usize(n_s, n_v, "bivariate score table exceeds buffer size")?;
    if n_cells > MAX_EQUATING_BIVARIATE_CELLS {
        return Err(format!(
            "bivariate score table must be <= {MAX_EQUATING_BIVARIATE_CELLS} cells"
        ));
    }
    let mut tab = vec![0.0_f64; n_cells];
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
            (
                (0..=k_x).map(|x| a * x as f64 + b).collect::<Vec<_>>(),
                a,
                b,
            )
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
    let cov = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x - ma) * (y - mb))
        .sum::<f64>()
        / n;
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
    if k_x > MAX_EQUATING_SCORE_POINTS || k_y > MAX_EQUATING_SCORE_POINTS {
        return Err(format!(
            "score ceiling must be <= {MAX_EQUATING_SCORE_POINTS}"
        ));
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
    if x_total
        .iter()
        .chain(x_anchor)
        .chain(y_total)
        .chain(y_anchor)
        .any(|v| !v.is_finite())
    {
        return Err("scores must be finite".into());
    }
    if x_total.iter().any(|&s| s < 0.0 || s > k_x as f64) {
        return Err("x_total score out of range".into());
    }
    if y_total.iter().any(|&s| s < 0.0 || s > k_y as f64) {
        return Err("y_total score out of range".into());
    }
    let (m1x, v1x, m1v, v1v, cov1) = paired_moments(x_total, x_anchor);
    let (m2y, v2y, m2v, v2v, cov2) = paired_moments(y_total, y_anchor);
    if [m1x, v1x, m1v, v1v, cov1, m2y, v2y, m2v, v2v, cov2]
        .iter()
        .any(|v| !v.is_finite())
    {
        return Err("derived NEAT linear moments must be finite".into());
    }
    if v1v <= 0.0 || v2v <= 0.0 {
        return Err("anchor variance must be positive in both groups".into());
    }
    let (g1, g2) = match method {
        NeatLinearMethod::Tucker => (cov1 / v1v, cov2 / v2v),
        NeatLinearMethod::LevineObserved => {
            if cov1 <= 0.0 || cov2 <= 0.0 {
                return Err(
                    "Levine equating needs a positive total-anchor covariance in both groups"
                        .into(),
                );
            }
            match anchor_kind {
                AnchorKind::Internal => (v1x / cov1, v2y / cov2),
                AnchorKind::External => ((v1x + cov1) / (v1v + cov1), (v2y + cov2) / (v2v + cov2)),
            }
        }
    };
    if !g1.is_finite() || !g2.is_finite() {
        return Err("NEAT linear regression coefficients must be finite".into());
    }
    let w2 = 1.0 - w1;
    let dmu = m1v - m2v;
    let dv = v1v - v2v;
    let mu_sx = m1x - w2 * g1 * dmu;
    let mu_sy = m2y + w1 * g2 * dmu;
    let var_sx = v1x - w2 * g1 * g1 * dv + w1 * w2 * g1 * g1 * dmu * dmu;
    let var_sy = v2y + w1 * g2 * g2 * dv + w1 * w2 * g2 * g2 * dmu * dmu;
    if [dmu, dv, mu_sx, mu_sy, var_sx, var_sy]
        .iter()
        .any(|v| !v.is_finite())
    {
        return Err("synthetic NEAT linear moments must be finite".into());
    }
    if var_sx <= 0.0 || var_sy <= 0.0 {
        return Err("synthetic variance is non-positive (degenerate equating)".into());
    }
    let a = var_sy.sqrt() / var_sx.sqrt();
    let b = mu_sy - a * mu_sx;
    if !a.is_finite() || !b.is_finite() {
        return Err("NEAT linear conversion coefficients must be finite".into());
    }
    let n_x = k_x
        .checked_add(1)
        .ok_or("k_x + 1 exceeds the equating buffer size")?;
    let mut x_scores = Vec::with_capacity(n_x);
    let mut y_eq = Vec::with_capacity(n_x);
    for x in 0..=k_x {
        x_scores.push(x as f64);
        y_eq.push(a * x as f64 + b);
    }
    if y_eq.iter().any(|v| !v.is_finite()) {
        return Err("NEAT linear conversion table must be finite".into());
    }
    Ok(EquateResult {
        x_scores,
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

pub const MAX_EQUATING_BOOTSTRAP_REPLICATES: usize = 10_000;
pub const MAX_EQUATING_BOOTSTRAP_CELLS: usize = 1_000_000;

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
///   https://doi.org/10.1007/978-1-4939-0317-7
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
    if n_boot > MAX_EQUATING_BOOTSTRAP_REPLICATES {
        return Err(format!(
            "n_boot must be <= {MAX_EQUATING_BOOTSTRAP_REPLICATES}"
        ));
    }
    let ncol = k_x
        .checked_add(1)
        .ok_or("k_x + 1 exceeds the bootstrap buffer size")?;
    let rep_cells = crate::checked_mul_usize(
        n_boot,
        ncol,
        "n_boot * (k_x + 1) exceeds the bootstrap buffer size",
    )?;
    if rep_cells > MAX_EQUATING_BOOTSTRAP_CELLS {
        return Err(format!(
            "n_boot * (k_x + 1) must be <= {MAX_EQUATING_BOOTSTRAP_CELLS}"
        ));
    }
    let point = equate_eg(x_scores, y_scores, k_x, k_y, method)?;
    let (nx, ny) = (x_scores.len(), y_scores.len());
    let mut reps = vec![0.0_f64; rep_cells];
    let mut st = seed.max(1);
    let mut u = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut xb = vec![0.0_f64; nx];
    let mut yb = vec![0.0_f64; ny];
    let max_attempts = n_boot.saturating_mul(100);
    let mut b = 0usize;
    let mut attempts = 0usize;
    while b < n_boot {
        attempts += 1;
        if attempts > max_attempts {
            return Err(
                "bootstrap SEE could not draw enough non-degenerate linear resamples".into(),
            );
        }
        for v in xb.iter_mut() {
            *v = x_scores[((u() * nx as f64) as usize).min(nx - 1)];
        }
        for v in yb.iter_mut() {
            *v = y_scores[((u() * ny as f64) as usize).min(ny - 1)];
        }
        if method == EquateMethod::Linear {
            let first = (xb[0] + 0.5).floor();
            if xb.iter().all(|&x| (x + 0.5).floor() == first) {
                continue;
            }
        }
        let r = equate_eg(&xb, &yb, k_x, k_y, method)?;
        reps[b * ncol..(b + 1) * ncol].copy_from_slice(&r.y_equivalents);
        b += 1;
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
        return Err(
            "analytic_see supports only Mean and Linear; use bootstrap_see for equipercentile"
                .into(),
        );
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
        let var = if method == EquateMethod::Mean {
            sx * sx / nx + sy * sy / ny
        } else {
            // Equipercentile was rejected above, so the remaining method is Linear.
            let z = (x as f64 - res.mu_x) / sx;
            sy * sy * (1.0 + z * z / 2.0) * (1.0 / nx + 1.0 / ny)
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
    let u: Vec<f64> = (0..n)
        .map(|x| {
            if k == 0 {
                0.0
            } else {
                2.0 * x as f64 / k as f64 - 1.0
            }
        })
        .collect();
    let mut cols: Vec<Vec<f64>> = (0..t)
        .map(|j| u.iter().map(|&ui| ui.powi(j as i32)).collect())
        .collect();
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
    (0..n)
        .map(|r| (0..t).map(|j| cols[j][r]).collect())
        .collect()
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
        (0..n_cells)
            .map(|x| {
                let e = eta_of(beta, x);
                counts[x] * e - e.exp()
            })
            .sum()
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
        let grad: Vec<f64> = (0..t)
            .map(|j| (0..n_cells).map(|x| b[x][j] * (counts[x] - m[x])).sum())
            .collect();
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
                .map(|x| {
                    let u = if k == 0 { 0.0 } else { x as f64 / k as f64 };
                    u.powi(j as i32) * probs[x]
                })
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
    r.iter()
        .enumerate()
        .map(|(j, &rj)| rj * norm_cdf((x - a * j as f64 - (1.0 - a) * mu) / ah))
        .sum()
}
fn kernel_pdf(r: &[f64], mu: f64, sig2: f64, h: f64, x: f64) -> f64 {
    let a = kernel_a(sig2, h);
    let ah = a * h;
    r.iter()
        .enumerate()
        .map(|(j, &rj)| rj * norm_pdf((x - a * j as f64 - (1.0 - a) * mu) / ah) / ah)
        .sum()
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
        let xn = if d > 1e-12 { x - fx / d } else { f64::NAN };
        x = if xn > lo && xn < hi {
            xn
        } else {
            0.5 * (lo + hi)
        };
    }
    x
}
fn kernel_equate(
    rx: &[f64],
    ry: &[f64],
    mu_x: f64,
    s2x: f64,
    mu_y: f64,
    s2y: f64,
    k_x: usize,
    k_y: usize,
    h_x: f64,
    h_y: f64,
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
fn expanded_upper_bandwidth(best_h: f64, upper: f64) -> f64 {
    if best_h >= upper - 1e-9 {
        2.0 * upper
    } else {
        upper
    }
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
    hi = expanded_upper_bandwidth(best_h, hi);
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
            // A finite-input fit can only be marked converged after a finite
            // gradient or log-likelihood stopping check; its normalized
            // probabilities are therefore finite here.
            Ok(fit.probs)
        }
    }
}

fn validate_optional_bandwidth(value: Option<f64>, name: &str) -> Result<(), String> {
    match value {
        Some(value) if !value.is_finite() || value <= 0.0 => {
            Err(format!("{name} must be positive and finite"))
        }
        _ => Ok(()),
    }
}

fn bandwidth_or_optimal(value: Option<f64>, r: &[f64], mu: f64, sig2: f64, k: usize) -> f64 {
    match value {
        Some(value) => value,
        None => optimal_bandwidth(r, mu, sig2, k),
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
            for (h, nm) in [
                (opts.bandwidth_x, "bandwidth_x"),
                (opts.bandwidth_y, "bandwidth_y"),
            ] {
                validate_optional_bandwidth(h, nm)?;
            }
            let (s2x, s2y) = (sigma_x * sigma_x, sigma_y * sigma_y);
            if s2x <= 0.0 || s2y <= 0.0 {
                return Err("gaussian kernel equating needs a positive SD on both forms".into());
            }
            let hx = bandwidth_or_optimal(opts.bandwidth_x, &gx, mu_x, s2x, k_x);
            let hy = bandwidth_or_optimal(opts.bandwidth_y, &gy, mu_y, s2y, k_y);
            (
                kernel_equate(&gx, &gy, mu_x, s2x, mu_y, s2y, k_x, k_y, hx, hy),
                hx,
                hy,
            )
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
#[path = "../../../tests/unit/equating_tests.rs"]
mod tests;
