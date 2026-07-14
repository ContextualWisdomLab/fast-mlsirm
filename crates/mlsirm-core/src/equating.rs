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
/// but the *synthetic-population* moments for frequency estimation (which equates
/// the post-stratified densities, not the raw marginals) — so do not compare a
/// chained result's moments against a frequency-estimation result's field-for-field.
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
}
