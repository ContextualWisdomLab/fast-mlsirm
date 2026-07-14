//! Item- and person-fit statistics on the Rust core (the compute path; the
//! NumPy implementations in `python/fast_mlsirm/fitstats.py` are the parity
//! reference and fallback).
//!
//! - S-X² (Orlando & Thissen 2000) with the Lord-Wingersky recursion on the
//!   joint `(theta, xi)` node set, per trait dimension, with score-group
//!   collapsing and — because the statistic is over-powered at large `N` — a
//!   practical-significance effect size: the `N_s`-weighted RMS of the
//!   observed-minus-expected proportions (cf. Sinharay & Haberman 2014,
//!   "How often is the misfit of item response theory models practically
//!   significant?").
//! - `l_z` (Drasgow, Levine & Williams 1985) and `l_z*` (Snijders 2001, MAP
//!   `r_0 = -(theta - prior_mean)` correction) at EAP estimates with the
//!   latent-space position fixed at its EAP.
//! - Infit/outfit mean squares at the EAP estimates.
//! - Chi-square upper tail via the regularized upper incomplete gamma
//!   (no external dependencies).

use crate::scoring::{lord_wingersky, ItemBank, PriorSpec};
use crate::nodes::{build_xi_nodes, XiRule};
use crate::quadrature::gh_rule;
use crate::model_exec_flags;

/// Regularized upper incomplete gamma `Q(a, x)` (Numerical Recipes 6.2).
fn gammainc_upper_reg(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        let mut ap = a;
        let mut total = 1.0 / a;
        let mut delta = total;
        for _ in 0..500 {
            ap += 1.0;
            delta *= x / ap;
            total += delta;
            if delta.abs() < total.abs() * 1e-15 {
                break;
            }
        }
        let p = total * (-x + a * x.ln() - ln_gamma(a)).exp();
        (1.0 - p).clamp(0.0, 1.0)
    } else {
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..500 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-15 {
                break;
            }
        }
        (h * (-x + a * x.ln() - ln_gamma(a)).exp()).clamp(0.0, 1.0)
    }
}

/// Lanczos log-gamma (g = 7, n = 9), |error| < 1e-13 on the positive axis.
fn ln_gamma(x: f64) -> f64 {
    const COEF: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        // reflection
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut acc = COEF[0];
    for (i, &c) in COEF.iter().enumerate().skip(1) {
        acc += c / (x + i as f64);
    }
    let t = x + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + acc.ln()
}

/// `P(Chi2_df >= x)`.
pub fn chi2_sf(x: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return f64::NAN;
    }
    gammainc_upper_reg(df / 2.0, x.max(0.0) / 2.0)
}

/// Benjamini-Hochberg step-up rejection mask at FDR level `q` (NaNs skipped).
pub fn benjamini_hochberg(p_values: &[f64], q: f64) -> Vec<bool> {
    let mut idx: Vec<usize> = (0..p_values.len()).filter(|&i| p_values[i].is_finite()).collect();
    let m = idx.len();
    let mut reject = vec![false; p_values.len()];
    if m == 0 {
        return reject;
    }
    idx.sort_by(|&a, &b| p_values[a].partial_cmp(&p_values[b]).unwrap());
    let mut k_max: Option<usize> = None;
    for (rank, &i) in idx.iter().enumerate() {
        if p_values[i] <= q * ((rank + 1) as f64) / (m as f64) {
            k_max = Some(rank);
        }
    }
    if let Some(k) = k_max {
        for &i in &idx[..=k] {
            reject[i] = true;
        }
    }
    reject
}

pub struct SX2Result {
    pub statistic: Vec<f64>,
    pub df: Vec<f64>,
    pub p_value: Vec<f64>,
    /// `N_s`-weighted RMS of `(O_s - E_s)` — the practical-significance
    /// effect size guarding against over-powered flags at large N.
    pub rms_residual: Vec<f64>,
    pub flagged_bh: Vec<bool>,
    pub n_score_groups: Vec<usize>,
}

#[derive(Clone, Copy)]
pub struct SX2Config {
    pub q_theta: usize,
    pub xi_rule: XiRule,
    pub min_expected: f64,
    pub fdr_q: f64,
    /// Flag only when BH-significant AND `rms_residual >= min_effect`.
    pub min_effect: f64,
}

impl Default for SX2Config {
    fn default() -> Self {
        Self {
            q_theta: 21,
            xi_rule: XiRule::GaussHermite { q_xi: 11 },
            min_expected: 1.0,
            fdr_q: 0.05,
            min_effect: 0.0,
        }
    }
}

/// Item success probabilities on the joint node set, plus node weights and
/// theta values, for one prior.
#[allow(clippy::type_complexity)]
fn icc_nodes(
    bank: &ItemBank<'_>,
    prior: &PriorSpec,
    q_theta: usize,
    xi_rule: XiRule,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, usize), String> {
    let (free_alpha, uses_space) = model_exec_flags(bank.model_type);
    let n_items = bank.b.len();
    let (t_nodes, t_weights) =
        gh_rule(q_theta).ok_or_else(|| format!("unsupported quadrature size {q_theta}"))?;
    let (x_grid, x_logw) = if uses_space {
        let nodes = build_xi_nodes(xi_rule, bank.latent_dim)?;
        (nodes.grid, nodes.logw)
    } else {
        (vec![0.0; bank.latent_dim], vec![0.0_f64])
    };
    let n_x = x_logw.len();
    let cell = q_theta * n_x;
    let gamma = if uses_space { bank.tau.exp() } else { 0.0 };
    let mut probs = vec![0.0_f64; n_items * cell];
    let mut weights = vec![0.0_f64; cell];
    let mut theta_by_dim = vec![0.0_f64; bank.n_dims * cell];
    for (t, &node_t) in t_nodes.iter().enumerate() {
        for x in 0..n_x {
            let c = t * n_x + x;
            weights[c] = (t_weights[t].ln() + x_logw[x]).exp();
            for d in 0..bank.n_dims {
                theta_by_dim[d * cell + c] = prior.mean[d] + prior.sd[d] * node_t;
            }
        }
    }
    for i in 0..n_items {
        let d = bank.factor_id[i];
        let a = if free_alpha { bank.alpha[i].exp() } else { 1.0 };
        for (t, _) in t_nodes.iter().enumerate() {
            for x in 0..n_x {
                let c = t * n_x + x;
                let mut eta = a * theta_by_dim[d * cell + c] + bank.b[i];
                if uses_space {
                    let mut dist2 = bank.eps_distance;
                    for k in 0..bank.latent_dim {
                        let diff = x_grid[x * bank.latent_dim + k]
                            - bank.zeta[i * bank.latent_dim + k];
                        dist2 += diff * diff;
                    }
                    eta -= gamma * dist2.sqrt();
                }
                probs[i * cell + c] = 1.0 / (1.0 + (-eta).exp());
            }
        }
    }
    Ok((probs, weights, theta_by_dim, cell))
}

/// Orlando-Thissen S-X² per item (summed scores within each trait dimension).
/// Persons with missing responses inside a dimension are excluded from that
/// dimension's observed table; `person_weight` (0/1) can screen aberrant
/// respondents out of the flagging statistics.
#[allow(clippy::too_many_arguments)]
pub fn s_x2(
    bank: &ItemBank<'_>,
    y: &[f64],
    observed: &[bool],
    n_persons: usize,
    prior: &PriorSpec,
    cfg: &SX2Config,
    person_weight: Option<&[f64]>,
) -> Result<SX2Result, String> {
    let n_items = bank.b.len();
    if y.len() != n_persons * n_items || observed.len() != y.len() {
        return Err("y and observed must both have length n_persons * n_items".into());
    }
    if let Some(w) = person_weight {
        if w.len() != n_persons {
            return Err("person_weight length must match n_persons".into());
        }
    }
    let (probs, weights, _theta, cell) = icc_nodes(bank, prior, cfg.q_theta, cfg.xi_rule)?;
    let n_free_base = if matches!(
        bank.model_type,
        crate::ModelType::Mlsrm | crate::ModelType::Ulsrm
    ) {
        1
    } else {
        2
    };
    let n_free = n_free_base
        + if matches!(bank.model_type, crate::ModelType::Mirt) { 0 } else { bank.latent_dim };

    let mut out = SX2Result {
        statistic: vec![f64::NAN; n_items],
        df: vec![f64::NAN; n_items],
        p_value: vec![f64::NAN; n_items],
        rms_residual: vec![f64::NAN; n_items],
        flagged_bh: vec![false; n_items],
        n_score_groups: vec![0; n_items],
    };

    for d in 0..bank.n_dims {
        let items: Vec<usize> = (0..n_items).filter(|&i| bank.factor_id[i] == d).collect();
        let n_d = items.len();
        if n_d < 2 {
            continue;
        }
        // persons complete on this dimension (and not screened out)
        let mut persons: Vec<usize> = Vec::new();
        for p in 0..n_persons {
            let w_ok = person_weight.map(|w| w[p] > 0.0).unwrap_or(true);
            if w_ok && items.iter().all(|&i| observed[p * n_items + i]) {
                persons.push(p);
            }
        }
        if persons.is_empty() {
            continue;
        }
        // observed counts by summed score
        let mut obs_n = vec![0.0_f64; n_d + 1];
        let mut obs_r = vec![vec![0.0_f64; n_d + 1]; n_d];
        for &p in &persons {
            let score: usize =
                items.iter().map(|&i| y[p * n_items + i] as usize).sum();
            obs_n[score] += 1.0;
            for (li, &i) in items.iter().enumerate() {
                obs_r[li][score] += y[p * n_items + i];
            }
        }
        // node-level probabilities for the dimension's items
        let mut p_flat = vec![0.0_f64; n_d * cell];
        for (row, &i) in items.iter().enumerate() {
            p_flat[row * cell..(row + 1) * cell]
                .copy_from_slice(&probs[i * cell..(i + 1) * cell]);
        }
        let s_all = lord_wingersky(&p_flat, n_d, cell);
        let denom: Vec<f64> = (0..=n_d)
            .map(|s| (0..cell).map(|c| s_all[s * cell + c] * weights[c]).sum())
            .collect();
        for (li, &i) in items.iter().enumerate() {
            // leave-one-out score distribution
            let mut rest = vec![0.0_f64; (n_d - 1) * cell];
            let mut row = 0;
            for (lj, &_j) in items.iter().enumerate() {
                if lj != li {
                    rest[row * cell..(row + 1) * cell]
                        .copy_from_slice(&p_flat[lj * cell..(lj + 1) * cell]);
                    row += 1;
                }
            }
            let s_rest = lord_wingersky(&rest, n_d - 1, cell);
            let mut e = vec![f64::NAN; n_d + 1];
            for s in 1..n_d {
                let num: f64 = (0..cell)
                    .map(|c| p_flat[li * cell + c] * s_rest[(s - 1) * cell + c] * weights[c])
                    .sum();
                if denom[s] > 0.0 {
                    e[s] = num / denom[s];
                }
            }
            // collapse adjacent score groups to the minimum expected count
            let mut groups: Vec<(f64, f64, f64)> = Vec::new();
            let (mut acc_n, mut acc_r, mut acc_e) = (0.0_f64, 0.0_f64, 0.0_f64);
            for s in 1..n_d {
                if !e[s].is_finite() {
                    continue;
                }
                acc_n += obs_n[s];
                acc_r += obs_r[li][s];
                acc_e += obs_n[s] * e[s];
                if acc_n > 0.0
                    && acc_e >= cfg.min_expected
                    && (acc_n - acc_e) >= cfg.min_expected
                {
                    groups.push((acc_n, acc_r, acc_e));
                    acc_n = 0.0;
                    acc_r = 0.0;
                    acc_e = 0.0;
                }
            }
            if acc_n > 0.0 {
                if let Some(last) = groups.last_mut() {
                    last.0 += acc_n;
                    last.1 += acc_r;
                    last.2 += acc_e;
                } else {
                    groups.push((acc_n, acc_r, acc_e));
                }
            }
            let (mut x2, mut n_grp) = (0.0_f64, 0usize);
            let (mut rss, mut n_tot) = (0.0_f64, 0.0_f64);
            for &(gn, gr, ge) in &groups {
                if gn <= 0.0 {
                    continue;
                }
                let e_prop = ge / gn;
                if e_prop <= 0.0 || e_prop >= 1.0 {
                    continue;
                }
                let o_prop = gr / gn;
                x2 += gn * (o_prop - e_prop) * (o_prop - e_prop) / (e_prop * (1.0 - e_prop));
                rss += gn * (o_prop - e_prop) * (o_prop - e_prop);
                n_tot += gn;
                n_grp += 1;
            }
            out.statistic[i] = x2;
            out.n_score_groups[i] = n_grp;
            out.rms_residual[i] = if n_tot > 0.0 { (rss / n_tot).sqrt() } else { f64::NAN };
            let df = n_grp as f64 - n_free as f64;
            if df >= 1.0 {
                out.df[i] = df;
                out.p_value[i] = chi2_sf(x2, df);
            }
        }
    }
    let bh = benjamini_hochberg(&out.p_value, cfg.fdr_q);
    for i in 0..n_items {
        out.flagged_bh[i] =
            bh[i] && out.rms_residual[i].is_finite() && out.rms_residual[i] >= cfg.min_effect;
    }
    Ok(out)
}

pub struct PersonFitResult {
    /// Row-major `n_persons x n_dims`.
    pub lz: Vec<f64>,
    pub lz_star: Vec<f64>,
    pub flagged: Vec<bool>,
}

/// `l_z` / `l_z*` per person and trait dimension at the EAP estimates
/// (`theta` row-major `n_persons x n_dims`, `xi` row-major
/// `n_persons x latent_dim`); `prior_mean` per (person, dim) or empty for 0.
#[allow(clippy::too_many_arguments)]
pub fn person_fit(
    bank: &ItemBank<'_>,
    y: &[f64],
    observed: &[bool],
    n_persons: usize,
    theta: &[f64],
    xi: &[f64],
    prior_mean: &[f64],
    flag_threshold: f64,
) -> Result<PersonFitResult, String> {
    let (free_alpha, uses_space) = model_exec_flags(bank.model_type);
    let n_items = bank.b.len();
    let (n_dims, latent_dim) = (bank.n_dims, bank.latent_dim);
    if y.len() != n_persons * n_items || observed.len() != y.len() {
        return Err("y and observed must both have length n_persons * n_items".into());
    }
    if theta.len() != n_persons * n_dims || xi.len() != n_persons * latent_dim {
        return Err("theta/xi shapes must match n_persons".into());
    }
    if !prior_mean.is_empty() && prior_mean.len() != n_persons * n_dims {
        return Err("prior_mean must be empty or n_persons x n_dims".into());
    }
    let gamma = if uses_space { bank.tau.exp() } else { 0.0 };
    let mut lz = vec![f64::NAN; n_persons * n_dims];
    let mut lz_star = vec![f64::NAN; n_persons * n_dims];
    let mut flagged = vec![false; n_persons];

    for p in 0..n_persons {
        for d in 0..n_dims {
            let (mut w_stat, mut var_l) = (0.0_f64, 0.0_f64);
            let (mut num_c, mut den_c) = (0.0_f64, 0.0_f64);
            let mut items_pd: Vec<(f64, f64, f64)> = Vec::new(); // (w_i, a_i, pv)
            let mut n_obs = 0usize;
            for i in 0..n_items {
                if bank.factor_id[i] != d || !observed[p * n_items + i] {
                    continue;
                }
                let a = if free_alpha { bank.alpha[i].exp() } else { 1.0 };
                let mut eta = a * theta[p * n_dims + d] + bank.b[i];
                if uses_space {
                    let mut dist2 = bank.eps_distance;
                    for k in 0..latent_dim {
                        let diff =
                            xi[p * latent_dim + k] - bank.zeta[i * latent_dim + k];
                        dist2 += diff * diff;
                    }
                    eta -= gamma * dist2.sqrt();
                }
                let prob = (1.0 / (1.0 + (-eta).exp())).clamp(1e-12, 1.0 - 1e-12);
                let w_i = (prob / (1.0 - prob)).ln();
                let pv = prob * (1.0 - prob);
                let yy = y[p * n_items + i];
                w_stat += (yy - prob) * w_i;
                var_l += pv * w_i * w_i;
                num_c += a * pv * w_i;
                den_c += a * pv * a;
                items_pd.push((w_i, a, pv));
                n_obs += 1;
            }
            if n_obs < 2 {
                continue;
            }
            if var_l > 0.0 {
                lz[p * n_dims + d] = w_stat / var_l.sqrt();
            }
            let c = if den_c > 0.0 { num_c / den_c } else { 0.0 };
            let mut tau2 = 0.0_f64;
            for &(w_i, a, pv) in &items_pd {
                let w_tilde = w_i - c * a;
                tau2 += w_tilde * w_tilde * pv;
            }
            tau2 /= n_obs as f64;
            let pm = if prior_mean.is_empty() { 0.0 } else { prior_mean[p * n_dims + d] };
            let r0 = -(theta[p * n_dims + d] - pm);
            if tau2 > 0.0 {
                lz_star[p * n_dims + d] =
                    (w_stat + c * r0) / ((n_obs as f64).sqrt() * tau2.sqrt());
            }
        }
        let min_star = (0..n_dims)
            .map(|d| lz_star[p * n_dims + d])
            .filter(|v| v.is_finite())
            .fold(f64::INFINITY, f64::min);
        flagged[p] = min_star < flag_threshold;
    }
    Ok(PersonFitResult { lz, lz_star, flagged })
}

pub struct InfitOutfit {
    pub infit: Vec<f64>,
    pub outfit: Vec<f64>,
}

/// Per-item infit/outfit mean squares at the EAP estimates.
pub fn infit_outfit(
    bank: &ItemBank<'_>,
    y: &[f64],
    observed: &[bool],
    n_persons: usize,
    theta: &[f64],
    xi: &[f64],
) -> Result<InfitOutfit, String> {
    let (free_alpha, uses_space) = model_exec_flags(bank.model_type);
    let n_items = bank.b.len();
    if y.len() != n_persons * n_items || observed.len() != y.len() {
        return Err("y and observed must both have length n_persons * n_items".into());
    }
    let gamma = if uses_space { bank.tau.exp() } else { 0.0 };
    let mut resid2_sum = vec![0.0_f64; n_items];
    let mut z2_sum = vec![0.0_f64; n_items];
    let mut var_sum = vec![0.0_f64; n_items];
    let mut counts = vec![0.0_f64; n_items];
    for p in 0..n_persons {
        for i in 0..n_items {
            if !observed[p * n_items + i] {
                continue;
            }
            let d = bank.factor_id[i];
            let a = if free_alpha { bank.alpha[i].exp() } else { 1.0 };
            let mut eta = a * theta[p * bank.n_dims + d] + bank.b[i];
            if uses_space {
                let mut dist2 = bank.eps_distance;
                for k in 0..bank.latent_dim {
                    let diff =
                        xi[p * bank.latent_dim + k] - bank.zeta[i * bank.latent_dim + k];
                    dist2 += diff * diff;
                }
                eta -= gamma * dist2.sqrt();
            }
            let prob = (1.0 / (1.0 + (-eta).exp())).clamp(1e-12, 1.0 - 1e-12);
            let v = prob * (1.0 - prob);
            let r2 = (y[p * n_items + i] - prob) * (y[p * n_items + i] - prob);
            resid2_sum[i] += r2;
            z2_sum[i] += r2 / v;
            var_sum[i] += v;
            counts[i] += 1.0;
        }
    }
    let infit = (0..n_items)
        .map(|i| if var_sum[i] > 0.0 { resid2_sum[i] / var_sum[i] } else { f64::NAN })
        .collect();
    let outfit = (0..n_items)
        .map(|i| if counts[i] > 0.0 { z2_sum[i] / counts[i] } else { f64::NAN })
        .collect();
    Ok(InfitOutfit { infit, outfit })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ModelType;

    #[test]
    fn chi2_sf_reference_values() {
        assert!((chi2_sf(3.841, 1.0) - 0.05).abs() < 1e-3);
        assert!((chi2_sf(18.307, 10.0) - 0.05).abs() < 1e-3);
        assert!((chi2_sf(0.0, 5.0) - 1.0).abs() < 1e-12);
        assert!(chi2_sf(1e6, 2.0) < 1e-12);
    }

    #[test]
    fn bh_step_up_known_case() {
        let p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216];
        let r = benjamini_hochberg(&p, 0.05);
        assert_eq!(r.iter().filter(|&&v| v).count(), 2);
        assert!(r[0] && r[1]);
    }

    fn toy_bank_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>, Vec<f64>, Vec<bool>, Vec<f64>, Vec<f64>) {
        // 1 dim, 20 items, 2000 persons simulated from a plain 1PL (MIRT
        // flags); person-fit asymptotics are in the item count, and the S-X2
        // effect size needs enough persons per score group to separate
        // sampling noise from systematic misfit.
        let n_items = 20usize;
        let n_persons = 2000usize;
        let alpha = vec![0.0; n_items];
        let b: Vec<f64> = (0..n_items).map(|i| -1.2 + 0.12 * i as f64).collect();
        let zeta = vec![0.0; n_items];
        let fid = vec![0usize; n_items];
        let mut state = 777u64;
        let mut unif = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut theta = vec![0.0_f64; n_persons];
        let mut y = vec![0.0_f64; n_persons * n_items];
        for p in 0..n_persons {
            let u1: f64 = unif().max(1e-12);
            let u2: f64 = unif();
            theta[p] = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            for i in 0..n_items {
                let eta: f64 = theta[p] + b[i];
                let prob = 1.0 / (1.0 + (-eta).exp());
                y[p * n_items + i] = if unif() < prob { 1.0 } else { 0.0 };
            }
        }
        let observed = vec![true; n_persons * n_items];
        let xi = vec![0.0_f64; n_persons];
        (alpha, b, zeta, fid, y, observed, theta, xi)
    }

    #[test]
    fn sx2_runs_and_effect_size_is_small_for_true_model() {
        let (alpha, b, zeta, fid, y, observed, _, _) = toy_bank_data();
        let bank = ItemBank {
            alpha: &alpha,
            b: &b,
            zeta: &zeta,
            tau: -30.0,
            factor_id: &fid,
            model_type: ModelType::Mirt,
            n_dims: 1,
            latent_dim: 1,
            eps_distance: 1e-8,
        };
        let res = s_x2(
            &bank,
            &y,
            &observed,
            2000,
            &PriorSpec::standard(1),
            &SX2Config { q_theta: 21, ..Default::default() },
            None,
        )
        .unwrap();
        let finite = res.statistic.iter().filter(|v| v.is_finite()).count();
        assert!(finite >= 15);
        // data simulated from the scoring model: typical effect sizes stay low
        // (the residual RMS at this N is dominated by ~sqrt(p(1-p)/N_s) noise)
        let mean_effect: f64 = res
            .rms_residual
            .iter()
            .filter(|v| v.is_finite())
            .sum::<f64>()
            / finite as f64;
        assert!(mean_effect < 0.05, "effect size too large for a true model: {mean_effect}");
    }

    #[test]
    fn person_fit_and_msq_finite_for_true_model() {
        let (alpha, b, zeta, fid, y, observed, _theta_true, _xi_true) = toy_bank_data();
        let bank = ItemBank {
            alpha: &alpha,
            b: &b,
            zeta: &zeta,
            tau: -30.0,
            factor_id: &fid,
            model_type: ModelType::Mirt,
            n_dims: 1,
            latent_dim: 1,
            eps_distance: 1e-8,
        };
        // designed usage: the Snijders correction applies to ESTIMATED scores
        let eap = crate::scoring::score_eap(
            &bank,
            &y,
            &observed,
            2000,
            &PriorSpec::standard(1),
            21,
            XiRule::GaussHermite { q_xi: 7 },
        )
        .unwrap();
        let pf = person_fit(
            &bank, &y, &observed, 2000, &eap.theta_eap, &eap.xi_eap, &[], -1.645,
        )
        .unwrap();
        let finite = pf.lz_star.iter().filter(|v| v.is_finite()).count();
        assert!(finite > 1800);
        let flag_rate =
            pf.flagged.iter().filter(|&&f| f).count() as f64 / 2000.0;
        assert!(flag_rate < 0.12, "flag rate should approach the nominal 5%: {flag_rate}");
        let msq = infit_outfit(&bank, &y, &observed, 2000, &eap.theta_eap, &eap.xi_eap)
            .unwrap();
        let mean_infit: f64 = msq.infit.iter().sum::<f64>() / 20.0;
        assert!((mean_infit - 1.0).abs() < 0.25, "infit should center near 1: {mean_infit}");
    }
}
