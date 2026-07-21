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

use crate::model_exec_flags;
use crate::nodes::{build_xi_nodes, XiRule};
use crate::scoring::{
    lord_wingersky, validate_bank, validate_dichotomous_responses, validate_prior, ItemBank,
    PriorSpec,
};

fn at_least_tiny(value: f64, tiny: f64) -> f64 {
    if value.abs() < tiny {
        tiny
    } else {
        value
    }
}

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
            d = at_least_tiny(an * d + b, tiny);
            c = at_least_tiny(b + an / c, tiny);
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
    let mut idx: Vec<usize> = (0..p_values.len())
        .filter(|&i| p_values[i].is_finite())
        .collect();
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
    let kind = crate::interaction_kind(bank.model_type);
    let n_items = bank.b.len();
    let (t_nodes, t_weights) = crate::quadrature::require_gh_rule(q_theta, "quadrature size")?;
    let (x_grid, x_logw) = if uses_space {
        let nodes = build_xi_nodes(xi_rule, bank.latent_dim)?;
        (nodes.grid, nodes.logw)
    } else {
        (vec![0.0; bank.latent_dim], vec![0.0_f64])
    };
    let n_x = x_logw.len();
    let cell = q_theta * n_x;
    let gamma = if kind == crate::InteractionKind::Distance {
        bank.tau.exp()
    } else {
        0.0
    };
    let _ = uses_space;
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
                match kind {
                    crate::InteractionKind::None => {}
                    crate::InteractionKind::Distance => {
                        let mut dist2 = bank.eps_distance;
                        for k in 0..bank.latent_dim {
                            let diff = x_grid[x * bank.latent_dim + k]
                                - bank.zeta[i * bank.latent_dim + k];
                            dist2 += diff * diff;
                        }
                        eta -= gamma * dist2.sqrt();
                    }
                    crate::InteractionKind::Inner => {
                        for k in 0..bank.latent_dim {
                            eta += bank.zeta[i * bank.latent_dim + k]
                                * x_grid[x * bank.latent_dim + k];
                        }
                    }
                }
                let bounded_eta = eta.clamp(-700.0, 700.0);
                probs[i * cell + c] = 1.0 / (1.0 + (-bounded_eta).exp());
            }
        }
    }
    Ok((probs, weights, theta_by_dim, cell))
}

/// Orlando-Thissen S-X² per item (summed scores within each trait dimension).
/// Persons with missing responses inside a dimension are excluded from that
/// dimension's observed table; `person_weight` (0/1) can screen aberrant
/// respondents out of the flagging statistics.
///
/// # References
///
/// Orlando, M., & Thissen, D. (2000). Likelihood-based item-fit indices for
/// dichotomous item response theory models. *Applied Psychological Measurement,
/// 24*(1), 50–64. <https://doi.org/10.1177/01466216000241003>
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
    let n_items = validate_bank(bank)?;
    validate_prior(prior, bank.n_dims)?;
    if !cfg.min_expected.is_finite() || cfg.min_expected <= 0.0 {
        return Err("min_expected must be finite and positive".into());
    }
    if !cfg.fdr_q.is_finite() || cfg.fdr_q <= 0.0 || cfg.fdr_q > 1.0 {
        return Err("fdr_q must be finite and in (0, 1]".into());
    }
    if !cfg.min_effect.is_finite() || cfg.min_effect < 0.0 {
        return Err("min_effect must be finite and non-negative".into());
    }
    let n_cells =
        crate::checked_mul_usize(n_persons, n_items, "n_persons * n_items overflows usize")?;
    if y.len() != n_cells || observed.len() != y.len() {
        return Err("y and observed must both have length n_persons * n_items".into());
    }
    // The summed-score table is indexed by `sum(y as usize)` and sized n_d+1, so a
    // non-dichotomous observed value would index out of bounds (panic). S-X2 is a
    // dichotomous-item statistic; reject anything but 0/1 on observed cells.
    if y.iter()
        .zip(observed)
        .any(|(&v, &o)| o && v != 0.0 && v != 1.0)
    {
        return Err("s_x2 requires dichotomous (0/1) observed responses".into());
    }
    if let Some(w) = person_weight {
        if w.len() != n_persons {
            return Err("person_weight length must match n_persons".into());
        }
        if w.iter()
            .any(|&value| !value.is_finite() || (value != 0.0 && value != 1.0))
        {
            return Err("person_weight must contain only finite 0/1 values".into());
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
        + if matches!(bank.model_type, crate::ModelType::Mirt) {
            0
        } else {
            bank.latent_dim
        };

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
            let score: usize = items.iter().map(|&i| y[p * n_items + i] as usize).sum();
            obs_n[score] += 1.0;
            for (li, &i) in items.iter().enumerate() {
                obs_r[li][score] += y[p * n_items + i];
            }
        }
        // node-level probabilities for the dimension's items
        let mut p_flat = vec![0.0_f64; n_d * cell];
        for (row, &i) in items.iter().enumerate() {
            p_flat[row * cell..(row + 1) * cell].copy_from_slice(&probs[i * cell..(i + 1) * cell]);
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
                if acc_n > 0.0 && acc_e >= cfg.min_expected && (acc_n - acc_e) >= cfg.min_expected {
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
            out.rms_residual[i] = if n_tot > 0.0 {
                (rss / n_tot).sqrt()
            } else {
                f64::NAN
            };
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
///
/// # References
///
/// Drasgow, F., Levine, M. V., & Williams, E. A. (1985). Appropriateness
/// measurement with polychotomous item response models and standardized indices.
/// *British Journal of Mathematical and Statistical Psychology, 38*(1), 67–86.
/// <https://doi.org/10.1111/j.2044-8317.1985.tb00817.x>
///
/// Snijders, T. A. B. (2001). Asymptotic null distribution of person fit
/// statistics with estimated person parameter. *Psychometrika, 66*(3), 331–342.
/// <https://doi.org/10.1007/BF02294437>
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
    let n_items = validate_bank(bank)?;
    let (n_dims, latent_dim) = (bank.n_dims, bank.latent_dim);
    validate_dichotomous_responses(y, observed, n_persons, n_items)?;
    if theta.len() != n_persons * n_dims || xi.len() != n_persons * latent_dim {
        return Err("theta/xi shapes must match n_persons".into());
    }
    if !prior_mean.is_empty() && prior_mean.len() != n_persons * n_dims {
        return Err("prior_mean must be empty or n_persons x n_dims".into());
    }
    let kind = crate::interaction_kind(bank.model_type);
    let gamma = if kind == crate::InteractionKind::Distance {
        bank.tau.exp()
    } else {
        0.0
    };
    let _ = uses_space;
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
                match kind {
                    crate::InteractionKind::None => {}
                    crate::InteractionKind::Distance => {
                        let mut dist2 = bank.eps_distance;
                        for k in 0..latent_dim {
                            let diff = xi[p * latent_dim + k] - bank.zeta[i * latent_dim + k];
                            dist2 += diff * diff;
                        }
                        eta -= gamma * dist2.sqrt();
                    }
                    crate::InteractionKind::Inner => {
                        for k in 0..latent_dim {
                            eta += bank.zeta[i * latent_dim + k] * xi[p * latent_dim + k];
                        }
                    }
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
            let pm = if prior_mean.is_empty() {
                0.0
            } else {
                prior_mean[p * n_dims + d]
            };
            let r0 = -(theta[p * n_dims + d] - pm);
            if tau2 > 0.0 {
                lz_star[p * n_dims + d] = (w_stat + c * r0) / ((n_obs as f64).sqrt() * tau2.sqrt());
            }
        }
        let min_star = (0..n_dims)
            .map(|d| lz_star[p * n_dims + d])
            .filter(|v| v.is_finite())
            .fold(f64::INFINITY, f64::min);
        flagged[p] = min_star < flag_threshold;
    }
    Ok(PersonFitResult {
        lz,
        lz_star,
        flagged,
    })
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
    let n_items = validate_bank(bank)?;
    validate_dichotomous_responses(y, observed, n_persons, n_items)?;
    if theta.len() != n_persons * bank.n_dims || xi.len() != n_persons * bank.latent_dim {
        return Err(
            "theta/xi must have lengths n_persons * n_dims / n_persons * latent_dim".into(),
        );
    }
    let kind = crate::interaction_kind(bank.model_type);
    let gamma = if kind == crate::InteractionKind::Distance {
        bank.tau.exp()
    } else {
        0.0
    };
    let _ = uses_space;
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
            match kind {
                crate::InteractionKind::None => {}
                crate::InteractionKind::Distance => {
                    let mut dist2 = bank.eps_distance;
                    for k in 0..bank.latent_dim {
                        let diff = xi[p * bank.latent_dim + k] - bank.zeta[i * bank.latent_dim + k];
                        dist2 += diff * diff;
                    }
                    eta -= gamma * dist2.sqrt();
                }
                crate::InteractionKind::Inner => {
                    for k in 0..bank.latent_dim {
                        eta += bank.zeta[i * bank.latent_dim + k] * xi[p * bank.latent_dim + k];
                    }
                }
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
        .map(|i| {
            if var_sum[i] > 0.0 {
                resid2_sum[i] / var_sum[i]
            } else {
                f64::NAN
            }
        })
        .collect();
    let outfit = (0..n_items)
        .map(|i| {
            if counts[i] > 0.0 {
                z2_sum[i] / counts[i]
            } else {
                f64::NAN
            }
        })
        .collect();
    Ok(InfitOutfit { infit, outfit })
}

#[cfg(test)]
#[path = "../../../tests/unit/fitstats_tests.rs"]
mod tests;

/// Information criteria for marginal (MML) fits — the standard indices whose
/// comparative behavior for IRT model selection is studied in Kang, Cohen &
/// Sung (2009): AIC, BIC (favored in their comparisons for dichotomous-kernel
/// models), corrected AIC, sample-size-adjusted BIC, and consistent AIC.
/// `n` is the number of persons (the marginal-likelihood sampling unit).
#[derive(Clone, Copy, Debug)]
pub struct InformationCriteria {
    pub loglik: f64,
    pub n_parameters: usize,
    pub n: usize,
    pub aic: f64,
    pub bic: f64,
    pub aicc: f64,
    pub sabic: f64,
    pub caic: f64,
}

pub fn information_criteria(loglik: f64, n_parameters: usize, n: usize) -> InformationCriteria {
    let k = n_parameters as f64;
    let nf = n as f64;
    let dev = -2.0 * loglik;
    let aic = dev + 2.0 * k;
    InformationCriteria {
        loglik,
        n_parameters,
        n,
        aic,
        bic: dev + k * nf.ln(),
        aicc: if nf - k - 1.0 > 0.0 {
            aic + 2.0 * k * (k + 1.0) / (nf - k - 1.0)
        } else {
            f64::NAN
        },
        sabic: dev + k * ((nf + 2.0) / 24.0).ln(),
        caic: dev + k * (nf.ln() + 1.0),
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/fitstats_ic_tests.rs"]
mod ic_tests;

/// Vuong (1989) test for non-nested model comparison from casewise marginal
/// log-likelihoods (Schneider et al., 2020): with
/// `m_i = l_i^A - l_i^B`, `omega^2 = Var(m)`,
/// `z = (sum m_i - correction) / (sqrt(n) * omega)`; the Schwarz correction
/// `(k_A - k_B)/2 * ln n` yields the BIC-adjusted variant. Positive z favors
/// model A. The pre-test of distinguishability (`omega^2 = 0`, weighted
/// chi-square tail) is not implemented here — inspect `omega` directly.
///
/// # References
///
/// Schneider, L., Chalmers, R. P., Debelak, R., & Merkle, E. C. (2020). Model
/// selection of nested and non-nested item response models using Vuong tests.
/// *Multivariate Behavioral Research, 55*(5), 664–684.
/// <https://doi.org/10.1080/00273171.2019.1664280>
///
/// Vuong, Q. H. (1989). Likelihood ratio tests for model selection and
/// non-nested hypotheses. *Econometrica, 57*(2), 307–333.
/// <https://doi.org/10.2307/1912557>
#[derive(Clone, Copy, Debug)]
pub struct VuongResult {
    pub z: f64,
    pub p_two_sided: f64,
    pub omega: f64,
    pub mean_diff: f64,
}

pub fn vuong_nonnested(
    loglik_a: &[f64],
    loglik_b: &[f64],
    k_a: usize,
    k_b: usize,
    bic_correction: bool,
) -> Result<VuongResult, String> {
    if loglik_a.len() != loglik_b.len() || loglik_a.len() < 2 {
        return Err("casewise log-likelihood vectors must be equal-length with n >= 2".into());
    }
    if loglik_a
        .iter()
        .chain(loglik_b)
        .any(|value| !value.is_finite())
    {
        return Err("casewise log-likelihoods must be finite".into());
    }
    let n = loglik_a.len() as f64;
    let m: Vec<f64> = loglik_a
        .iter()
        .zip(loglik_b)
        .map(|(&a, &b)| a - b)
        .collect();
    if m.iter().any(|value| !value.is_finite()) {
        return Err("casewise log-likelihood differences must be finite".into());
    }
    let mean = m.iter().sum::<f64>() / n;
    let var = m.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n;
    if !mean.is_finite() || !var.is_finite() {
        return Err("casewise log-likelihood moments must be finite".into());
    }
    if var <= 0.0 {
        return Err("models are indistinguishable on this sample (omega^2 = 0)".into());
    }
    let omega = var.sqrt();
    let correction = if bic_correction {
        (k_a as f64 - k_b as f64) / 2.0 * n.ln()
    } else {
        0.0
    };
    let z = (m.iter().sum::<f64>() - correction) / (n.sqrt() * omega);
    if !z.is_finite() {
        return Err("Vuong z statistic is non-finite for these inputs".into());
    }
    // two-sided normal tail via the complementary error function relation:
    // p = 2 * (1 - Phi(|z|)) = erfc(|z| / sqrt(2))
    let p = erfc(z.abs() / std::f64::consts::SQRT_2);
    Ok(VuongResult {
        z,
        p_two_sided: p,
        omega,
        mean_diff: mean,
    })
}

/// Complementary error function (Numerical Recipes rational approximation;
/// |error| < 1.2e-7 — adequate for p-value reporting).
pub(crate) fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let ans = t
        * (-z * z - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
            .exp();
    if x >= 0.0 {
        ans
    } else {
        2.0 - ans
    }
}

/// Yen's Q3 residual correlations and a repository-specific mean absolute
/// residual cross-product. `resid` is the row-major `n_persons x n_items`
/// matrix `y - P_hat` at EAP estimates, with NaN for missing cells.
///
/// The cross-product summary is not Levy and Svetina's (2011) GDDM, which is
/// defined from model-based covariance in a posterior-predictive framework.
///
/// # References
///
/// Yen, W. M. (1984). Effects of local item dependence on the fit and equating
/// performance of the three-parameter logistic model. *Applied Psychological
/// Measurement, 8*(2), 125–145. https://doi.org/10.1177/014662168400800201
///
/// Levy, R., & Svetina, D. (2011). A generalized dimensionality discrepancy
/// measure for dimensionality assessment in multidimensional item response
/// theory. *British Journal of Mathematical and Statistical Psychology, 64*(2),
/// 208–232. https://doi.org/10.1348/000711010X500483
#[derive(Clone, Debug)]
pub struct DimResidResult {
    /// Off-diagonal Q3 values (upper triangle, row-major pair order).
    pub q3: Vec<f64>,
    pub q3_max_abs: f64,
    pub q3_mean_abs: f64,
    /// Mean of `abs(mean(e_i * e_j))` over pairs with at least three cases.
    pub mean_abs_residual_cross_product: f64,
    /// Backward-compatible alias for `mean_abs_residual_cross_product`.
    /// This field is not the published Levy-Svetina GDDM.
    pub gddm: f64,
}

pub fn dimensionality_residuals(
    resid: &[f64],
    n_persons: usize,
    n_items: usize,
) -> Result<DimResidResult, String> {
    if n_persons == 0 || n_items == 0 {
        return Err("n_persons and n_items must be positive".into());
    }
    let n_cells = n_persons
        .checked_mul(n_items)
        .ok_or_else(|| "n_persons * n_items overflows usize".to_string())?;
    if resid.len() != n_cells {
        return Err("resid must be n_persons x n_items".into());
    }
    if resid.iter().any(|value| value.is_infinite()) {
        return Err("resid entries must be finite or NaN for missing cells".into());
    }
    let mut q3 = Vec::with_capacity(n_items * (n_items - 1) / 2);
    let (mut max_abs, mut sum_abs) = (0.0_f64, 0.0_f64);
    let mut gddm_sum = 0.0_f64;
    let mut gddm_cnt = 0.0_f64;
    for i in 0..n_items {
        for j in (i + 1)..n_items {
            let (mut sxy, mut sxx, mut syy, mut sx, mut sy, mut n) =
                (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
            for p in 0..n_persons {
                let a = resid[p * n_items + i];
                let b = resid[p * n_items + j];
                if a.is_nan() || b.is_nan() {
                    continue;
                }
                sxy += a * b;
                sxx += a * a;
                syy += b * b;
                sx += a;
                sy += b;
                n += 1.0;
            }
            if n < 3.0 {
                q3.push(f64::NAN);
                continue;
            }
            let cov = sxy / n - (sx / n) * (sy / n);
            let vx = sxx / n - (sx / n) * (sx / n);
            let vy = syy / n - (sy / n) * (sy / n);
            let r = if vx > 0.0 && vy > 0.0 {
                cov / (vx * vy).sqrt()
            } else {
                f64::NAN
            };
            q3.push(r);
            if r.is_finite() {
                sum_abs += r.abs();
                if r.abs() > max_abs {
                    max_abs = r.abs();
                }
            }
            // Repository-specific descriptive residual cross-product.
            gddm_sum += (sxy / n).abs();
            gddm_cnt += 1.0;
        }
    }
    let n_finite = q3.iter().filter(|v| v.is_finite()).count();
    let mean_abs_residual_cross_product = if gddm_cnt > 0.0 {
        gddm_sum / gddm_cnt
    } else {
        f64::NAN
    };
    Ok(DimResidResult {
        q3_max_abs: if n_finite > 0 { max_abs } else { f64::NAN },
        q3_mean_abs: if n_finite > 0 {
            sum_abs / n_finite as f64
        } else {
            f64::NAN
        },
        mean_abs_residual_cross_product,
        gddm: mean_abs_residual_cross_product,
        q3,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/fitstats_vuong_tests.rs"]
mod vuong_tests;

/// Residual-based item fit (Haberman, Sinharay & Chon 2013): bin persons by
/// EAP score on the item's dimension, compare observed proportions against
/// the model ICC at the bin's mean estimate, and standardize:
/// `z_bin = (obs - exp) / sqrt(exp (1 - exp) / n_bin)`. Reported per item as
/// the maximum |z| over bins and its Bonferroni-adjusted normal p-value.
/// Designed for LONG tests (the source's operational setting): with short
/// tests EAP shrinkage biases the extreme bins and inflates the statistic —
/// prefer S-X2 below ~25 items.
pub struct ResidualFitResult {
    pub max_abs_z: Vec<f64>,
    pub p_value: Vec<f64>,
    pub n_bins: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn residual_item_fit(
    bank: &ItemBank<'_>,
    y: &[f64],
    observed: &[bool],
    n_persons: usize,
    theta: &[f64],
    xi: &[f64],
    n_bins: usize,
) -> Result<ResidualFitResult, String> {
    let (free_alpha, uses_space) = crate::model_exec_flags(bank.model_type);
    let n_items = bank.b.len();
    if y.len() != n_persons * n_items || observed.len() != y.len() {
        return Err("y and observed must both have length n_persons * n_items".into());
    }
    if theta.len() != n_persons * bank.n_dims || xi.len() != n_persons * bank.latent_dim {
        return Err("theta/xi shapes must match n_persons".into());
    }
    if n_bins < 2 {
        return Err("n_bins must be >= 2".into());
    }
    let kind = crate::interaction_kind(bank.model_type);
    let gamma = if kind == crate::InteractionKind::Distance {
        bank.tau.exp()
    } else {
        0.0
    };
    let _ = uses_space;
    let mut max_abs_z = vec![f64::NAN; n_items];
    let mut p_value = vec![f64::NAN; n_items];
    for i in 0..n_items {
        let d = bank.factor_id[i];
        // persons observed on item i, sorted by their EAP on dim d
        let mut idx: Vec<usize> = (0..n_persons)
            .filter(|&p| observed[p * n_items + i])
            .collect();
        if idx.len() < n_bins * 5 {
            continue;
        }
        idx.sort_by(|&a, &b| {
            theta[a * bank.n_dims + d]
                .partial_cmp(&theta[b * bank.n_dims + d])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let a = if free_alpha { bank.alpha[i].exp() } else { 1.0 };
        let mut worst = 0.0_f64;
        let bin_size = idx.len() / n_bins;
        for bin in 0..n_bins {
            let lo = bin * bin_size;
            let hi = if bin == n_bins - 1 {
                idx.len()
            } else {
                (bin + 1) * bin_size
            };
            let members = &idx[lo..hi];
            let (mut obs_sum, mut exp_sum) = (0.0_f64, 0.0_f64);
            for &p in members {
                obs_sum += y[p * n_items + i];
                let mut eta = a * theta[p * bank.n_dims + d] + bank.b[i];
                match kind {
                    crate::InteractionKind::None => {}
                    crate::InteractionKind::Distance => {
                        let mut dist2 = bank.eps_distance;
                        for k in 0..bank.latent_dim {
                            let diff =
                                xi[p * bank.latent_dim + k] - bank.zeta[i * bank.latent_dim + k];
                            dist2 += diff * diff;
                        }
                        eta -= gamma * dist2.sqrt();
                    }
                    crate::InteractionKind::Inner => {
                        for k in 0..bank.latent_dim {
                            eta += bank.zeta[i * bank.latent_dim + k] * xi[p * bank.latent_dim + k];
                        }
                    }
                }
                exp_sum += 1.0 / (1.0 + (-eta).exp());
            }
            let n_bin = members.len() as f64;
            let e = (exp_sum / n_bin).clamp(1e-9, 1.0 - 1e-9);
            let z = (obs_sum / n_bin - e) / (e * (1.0 - e) / n_bin).sqrt();
            if z.abs() > worst {
                worst = z.abs();
            }
        }
        max_abs_z[i] = worst;
        // Bonferroni over bins on the two-sided normal tail
        let p_one = erfc(worst / std::f64::consts::SQRT_2);
        p_value[i] = (p_one * n_bins as f64).min(1.0);
    }
    Ok(ResidualFitResult {
        max_abs_z,
        p_value,
        n_bins,
    })
}

/// Adjusted chi-square-to-df ratios for item pairs (Drasgow tradition;
/// Tay & Drasgow 2012, "Adjusting the adjusted chi2/df ratio statistic for
/// dichotomous IRT analyses"): the pairwise 2x2 table chi-square against the
/// model-implied joint probabilities, rescaled to a reference sample size of
/// 3000: `adj = ((chi2 - df) * 3000 / N + df) / df`. Values above ~3 flag
/// pairwise misfit / local dependence.
pub struct AdjustedChi2Result {
    /// Upper-triangle pair values, row-major pair order.
    pub ratio: Vec<f64>,
    pub mean_ratio: f64,
    pub max_ratio: f64,
}

#[allow(clippy::too_many_arguments)]
pub fn adjusted_chi2_pairs(
    bank: &ItemBank<'_>,
    y: &[f64],
    observed: &[bool],
    n_persons: usize,
    prior: &PriorSpec,
    q_theta: usize,
    xi_rule: XiRule,
) -> Result<AdjustedChi2Result, String> {
    let n_items = bank.b.len();
    if y.len() != n_persons * n_items || observed.len() != y.len() {
        return Err("y and observed must both have length n_persons * n_items".into());
    }
    let (probs, weights, _theta, cell) = icc_nodes(bank, prior, q_theta, xi_rule)?;
    let mut ratio = Vec::with_capacity(n_items * (n_items - 1) / 2);
    let (mut sum, mut max, mut count) = (0.0_f64, 0.0_f64, 0usize);
    for i in 0..n_items {
        for j in (i + 1)..n_items {
            // model-implied joint cell probabilities (marginal over the grid)
            let (mut p11, mut p10, mut p01) = (0.0_f64, 0.0_f64, 0.0_f64);
            for c in 0..cell {
                let pi = probs[i * cell + c];
                let pj = probs[j * cell + c];
                p11 += weights[c] * pi * pj;
                p10 += weights[c] * pi * (1.0 - pj);
                p01 += weights[c] * (1.0 - pi) * pj;
            }
            let p00 = (1.0 - p11 - p10 - p01).max(1e-12);
            // observed joint counts over persons observed on both items
            let (mut o11, mut o10, mut o01, mut o00, mut n) =
                (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
            for p in 0..n_persons {
                if !observed[p * n_items + i] || !observed[p * n_items + j] {
                    continue;
                }
                let (yi, yj) = (y[p * n_items + i], y[p * n_items + j]);
                n += 1.0;
                if yi == 1.0 && yj == 1.0 {
                    o11 += 1.0;
                } else if yi == 1.0 {
                    o10 += 1.0;
                } else if yj == 1.0 {
                    o01 += 1.0;
                } else {
                    o00 += 1.0;
                }
            }
            if n < 20.0 {
                ratio.push(f64::NAN);
                continue;
            }
            let mut chi2 = 0.0_f64;
            for (o, e) in [(o11, p11), (o10, p10), (o01, p01), (o00, p00)] {
                let expc = (e * n).max(1e-9);
                chi2 += (o - expc) * (o - expc) / expc;
            }
            let df = 3.0;
            let adj = ((chi2 - df) * 3000.0 / n + df) / df;
            ratio.push(adj);
            sum += adj;
            if adj > max {
                max = adj;
            }
            count += 1;
        }
    }
    Ok(AdjustedChi2Result {
        ratio,
        mean_ratio: if count > 0 {
            sum / count as f64
        } else {
            f64::NAN
        },
        max_ratio: max,
    })
}

/// Parametric-bootstrap person fit (Sinharay 2016, "Assessment of person fit
/// using resampling-based approaches"): for each person, simulate replicate
/// response vectors from the fitted model AT the person's EAP estimates,
/// compute `l_z*` for each replicate, and report the empirical p-value
/// `P(l_z*_rep <= l_z*_obs)` — small values flag aberrance without relying
/// on the asymptotic N(0,1) reference (which degrades for short/sparse
/// tests).
#[allow(clippy::too_many_arguments)]
pub fn person_fit_resampling(
    bank: &ItemBank<'_>,
    y: &[f64],
    observed: &[bool],
    n_persons: usize,
    theta: &[f64],
    xi: &[f64],
    prior_mean: &[f64],
    n_replicates: usize,
    seed: u64,
) -> Result<Vec<f64>, String> {
    let (free_alpha, uses_space) = crate::model_exec_flags(bank.model_type);
    let n_items = bank.b.len();
    const MAX_REPLICATES: usize = 10_000;
    const MAX_WORK_CELLS: usize = 200_000_000;
    if !(1..=MAX_REPLICATES).contains(&n_replicates) {
        return Err(format!("n_replicates must be in 1..={MAX_REPLICATES}"));
    }
    let work_cells = n_persons
        .checked_mul(n_items)
        .and_then(|cells| cells.checked_mul(n_replicates))
        .ok_or("person-fit resampling work size overflows usize")?;
    if work_cells > MAX_WORK_CELLS {
        return Err(format!(
            "person-fit resampling exceeds the {MAX_WORK_CELLS}-cell work limit"
        ));
    }
    let base = person_fit(bank, y, observed, n_persons, theta, xi, prior_mean, -1.645)?;
    let kind = crate::interaction_kind(bank.model_type);
    let gamma = if kind == crate::InteractionKind::Distance {
        bank.tau.exp()
    } else {
        0.0
    };
    let _ = uses_space;
    let mut state = seed.max(1);
    let mut unif = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut p_values = vec![f64::NAN; n_persons];
    let mut y_rep = vec![0.0_f64; n_items];
    let mut obs_rep = vec![false; n_items];
    for p in 0..n_persons {
        // observed lz*: the minimum across dimensions (matches the flag rule)
        let obs_stat = (0..bank.n_dims)
            .map(|d| base.lz_star[p * bank.n_dims + d])
            .filter(|v| v.is_finite())
            .fold(f64::INFINITY, f64::min);
        if !obs_stat.is_finite() {
            continue;
        }
        let mut count_leq = 0usize;
        let mut count_valid = 0usize;
        for _ in 0..n_replicates {
            for i in 0..n_items {
                obs_rep[i] = observed[p * n_items + i];
                if !obs_rep[i] {
                    y_rep[i] = 0.0;
                    continue;
                }
                let d = bank.factor_id[i];
                let a = if free_alpha { bank.alpha[i].exp() } else { 1.0 };
                let mut eta = a * theta[p * bank.n_dims + d] + bank.b[i];
                match kind {
                    crate::InteractionKind::None => {}
                    crate::InteractionKind::Distance => {
                        let mut dist2 = bank.eps_distance;
                        for k in 0..bank.latent_dim {
                            let diff =
                                xi[p * bank.latent_dim + k] - bank.zeta[i * bank.latent_dim + k];
                            dist2 += diff * diff;
                        }
                        eta -= gamma * dist2.sqrt();
                    }
                    crate::InteractionKind::Inner => {
                        for k in 0..bank.latent_dim {
                            eta += bank.zeta[i * bank.latent_dim + k] * xi[p * bank.latent_dim + k];
                        }
                    }
                }
                let prob = 1.0 / (1.0 + (-eta).exp());
                y_rep[i] = if unif() < prob { 1.0 } else { 0.0 };
            }
            let pm: Vec<f64> = if prior_mean.is_empty() {
                Vec::new()
            } else {
                prior_mean[p * bank.n_dims..(p + 1) * bank.n_dims].to_vec()
            };
            let rep = person_fit(
                bank,
                &y_rep,
                &obs_rep,
                1,
                &theta[p * bank.n_dims..(p + 1) * bank.n_dims],
                &xi[p * bank.latent_dim..(p + 1) * bank.latent_dim],
                &pm,
                -1.645,
            )
            .expect("replicated data preserve the already-validated person-fit shapes");
            let rep_stat = (0..bank.n_dims)
                .map(|d| rep.lz_star[d])
                .filter(|v| v.is_finite())
                .fold(f64::INFINITY, f64::min);
            let valid = rep_stat.is_finite();
            count_valid += usize::from(valid);
            count_leq += usize::from(valid && rep_stat <= obs_stat);
        }
        if count_valid > 0 {
            // add-one smoothing keeps p in (0, 1]
            p_values[p] = (count_leq as f64 + 1.0) / (count_valid as f64 + 1.0);
        }
    }
    Ok(p_values)
}

/// Stepwise test-characteristic-curve drift detection (Guo, Zheng & Chang
/// 2015): given two calibrations of a common item set on the SAME scale
/// (e.g. FIPC-linked), compute the weighted area between the two TCCs over
/// the prior grid, and step-wise remove the item with the largest
/// contribution until the remaining area falls below `threshold` — the
/// removed items are the drift suspects.
pub struct TccDriftResult {
    /// Items flagged as drifted, in removal order.
    pub drifted: Vec<usize>,
    /// Weighted TCC area per removal round (before each removal).
    pub area_trace: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
pub fn tcc_drift(
    bank_old: &ItemBank<'_>,
    bank_new: &ItemBank<'_>,
    prior: &PriorSpec,
    q_theta: usize,
    xi_rule: XiRule,
    threshold: f64,
) -> Result<TccDriftResult, String> {
    let n_items = bank_old.b.len();
    if bank_new.b.len() != n_items {
        return Err("both calibrations must cover the same item set".into());
    }
    let (p_old, weights, _t, cell) = icc_nodes(bank_old, prior, q_theta, xi_rule)?;
    let (p_new, _w2, _t2, cell2) = icc_nodes(bank_new, prior, q_theta, xi_rule)?;
    if cell != cell2 {
        return Err("calibrations must share the quadrature configuration".into());
    }
    let mut active = vec![true; n_items];
    let mut drifted = Vec::new();
    let mut area_trace = Vec::new();
    loop {
        // weighted area between TCCs over active items
        let mut area = 0.0_f64;
        let mut per_item = vec![0.0_f64; n_items];
        for c in 0..cell {
            let mut diff_sum = 0.0_f64;
            for i in 0..n_items {
                if active[i] {
                    diff_sum += p_new[i * cell + c] - p_old[i * cell + c];
                }
            }
            area += weights[c] * diff_sum.abs();
            for i in 0..n_items {
                if active[i] {
                    per_item[i] += weights[c] * (p_new[i * cell + c] - p_old[i * cell + c]).abs();
                }
            }
        }
        area_trace.push(area);
        if area <= threshold || active.iter().filter(|&&a| a).count() <= 2 {
            break;
        }
        let worst = (0..n_items)
            .filter(|&i| active[i])
            .max_by(|&a, &b| {
                per_item[a]
                    .partial_cmp(&per_item[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        active[worst] = false;
        drifted.push(worst);
    }
    Ok(TccDriftResult {
        drifted,
        area_trace,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/fitstats_batch3_tests.rs"]
mod batch3_tests;

/// Chen & Thissen (1997) local-dependence indices for item pairs: the
/// standardized (signed) LD X2 — the pairwise 2x2 chi-square against the
/// model-implied joint probabilities, given the sign of the observed-vs-
/// expected association, plus the G2 variant. Values with |standardized|
/// above ~10 (the X2 scale) or repeated same-sign clusters indicate local
/// dependence the latent structure does not absorb.
///
/// # References (APA 7th ed.)
///
/// Chen, W.-H., & Thissen, D. (1997). Local dependence indexes for item pairs
/// using item response theory. *Journal of Educational and Behavioral
/// Statistics, 22*(3), 265–289. https://doi.org/10.3102/10769986022003265
pub struct LdIndexResult {
    /// Upper-triangle signed X2 per pair (row-major pair order).
    pub x2_signed: Vec<f64>,
    /// Upper-triangle signed G2 per pair.
    pub g2_signed: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
pub fn ld_indices(
    bank: &ItemBank<'_>,
    y: &[f64],
    observed: &[bool],
    n_persons: usize,
    prior: &PriorSpec,
    q_theta: usize,
    xi_rule: XiRule,
) -> Result<LdIndexResult, String> {
    let n_items = validate_bank(bank)?;
    validate_prior(prior, bank.n_dims)?;
    if n_items < 2 {
        return Err("local-dependence indices need at least 2 items".into());
    }
    let n_cells =
        crate::checked_mul_usize(n_persons, n_items, "n_persons * n_items overflows usize")?;
    if y.len() != n_cells || observed.len() != y.len() {
        return Err("y and observed must both have length n_persons * n_items".into());
    }
    if y.iter().zip(observed).any(|(&value, &is_observed)| {
        is_observed && (!value.is_finite() || (value != 0.0 && value != 1.0))
    }) {
        return Err("observed responses must be 0 or 1".into());
    }
    let (probs, weights, _theta, cell) = icc_nodes(bank, prior, q_theta, xi_rule)?;
    let n_pairs = n_items * (n_items - 1) / 2;
    let mut x2_signed = Vec::with_capacity(n_pairs);
    let mut g2_signed = Vec::with_capacity(n_pairs);
    for i in 0..n_items {
        for j in (i + 1)..n_items {
            let (mut p11, mut p10, mut p01) = (0.0_f64, 0.0_f64, 0.0_f64);
            for c in 0..cell {
                let pi = probs[i * cell + c];
                let pj = probs[j * cell + c];
                p11 += weights[c] * pi * pj;
                p10 += weights[c] * pi * (1.0 - pj);
                p01 += weights[c] * (1.0 - pi) * pj;
            }
            let p00 = (1.0 - p11 - p10 - p01).max(1e-12);
            let (mut o11, mut o10, mut o01, mut o00, mut n) =
                (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
            for p in 0..n_persons {
                if !observed[p * n_items + i] || !observed[p * n_items + j] {
                    continue;
                }
                let (yi, yj) = (y[p * n_items + i], y[p * n_items + j]);
                n += 1.0;
                if yi == 1.0 && yj == 1.0 {
                    o11 += 1.0;
                } else if yi == 1.0 {
                    o10 += 1.0;
                } else if yj == 1.0 {
                    o01 += 1.0;
                } else {
                    o00 += 1.0;
                }
            }
            if n < 20.0 {
                x2_signed.push(f64::NAN);
                g2_signed.push(f64::NAN);
                continue;
            }
            let (mut x2, mut g2) = (0.0_f64, 0.0_f64);
            for (o, e) in [(o11, p11), (o10, p10), (o01, p01), (o00, p00)] {
                let expc = (e * n).max(1e-9);
                x2 += (o - expc) * (o - expc) / expc;
                if o > 0.0 {
                    g2 += 2.0 * o * (o / expc).ln();
                }
            }
            // sign: direction of the observed-vs-expected association
            // (positive when the pair covaries beyond the model)
            let sign = if (o11 / n - p11) >= 0.0 { 1.0 } else { -1.0 };
            x2_signed.push(sign * x2);
            g2_signed.push(sign * g2);
        }
    }
    Ok(LdIndexResult {
        x2_signed,
        g2_signed,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/fitstats_ld_tests.rs"]
mod ld_tests;

// ---------------------------------------------------------------------------
// M2 limited-information goodness-of-fit (Maydeu-Olivares & Joe 2005, 2006;
// Cai & Hansen 2013 for the hierarchical/bifactor factorization) with the
// RMSEA2 approximate-fit index, its noncentral-chi-square confidence interval,
// and the standardized root-mean-square residual (SRMSR; Maydeu-Olivares 2013)
// over the bivariate margins.
//
// The residual vector stacks the univariate and bivariate model-vs-observed
// margins. Both the residuals and their multinomial covariance Xi_2 are exact
// under local independence, because every model-implied joint margin factors
// over the quadrature nodes: pi_S = sum_c w_c * prod_{i in S} P_i(c). The
// derivative matrix Delta_2 = d pi / d beta is taken by central differences of
// the same node moments. The quadratic form
//   M2 = N * e' [ Xi^-1 - Xi^-1 D (D' Xi^-1 D)^-1 D' Xi^-1 ] e
// is evaluated through one Cholesky factorization of Xi (never an explicit
// inverse): u = Xi^-1 e, W = Xi^-1 D, A = D'W, g = W'e, solve A z = g, then
//   M2 = N ( e'u - g'z ).
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct M2Result {
    pub m2: f64,
    pub df: f64,
    pub p_value: f64,
    pub rmsea2: f64,
    pub rmsea2_ci_lower: f64,
    pub rmsea2_ci_upper: f64,
    pub srmsr: f64,
    /// M2 for the fitted complete-independence (zero-factor) baseline model.
    pub null_m2: f64,
    /// Degrees of freedom for the complete-independence baseline model.
    pub null_df: f64,
    /// Comparative fit index based on target and null-model M2 statistics.
    pub cfi: f64,
    /// Tucker-Lewis index for IRT (TLIRT) based on target and null-model M2.
    pub tli: f64,
    pub n_moments: usize,
    pub n_parameters: usize,
    pub n_complete: usize,
}

/// One free item parameter, addressed for the finite-difference Delta.
#[derive(Clone, Copy)]
enum M2Param {
    B(usize),
    Alpha(usize),
    Zeta(usize, usize),
    Tau,
}

fn m2_parameters(
    n_items: usize,
    free_alpha: bool,
    uses_space: bool,
    latent_dim: usize,
    tau_free: bool,
) -> Vec<M2Param> {
    let mut params = Vec::new();
    for i in 0..n_items {
        params.push(M2Param::B(i));
        if free_alpha {
            params.push(M2Param::Alpha(i));
        }
        if uses_space {
            for k in 0..latent_dim {
                params.push(M2Param::Zeta(i, k));
            }
        }
    }
    if tau_free {
        params.push(M2Param::Tau);
    }
    params
}

fn m2_param_value(
    param: M2Param,
    alpha: &[f64],
    b: &[f64],
    zeta: &[f64],
    tau: f64,
    latent_dim: usize,
) -> f64 {
    match param {
        M2Param::B(i) => b[i],
        M2Param::Alpha(i) => alpha[i],
        M2Param::Zeta(i, k) => zeta[i * latent_dim + k],
        M2Param::Tau => tau,
    }
}

fn set_m2_param(
    param: M2Param,
    value: f64,
    alpha: &mut [f64],
    b: &mut [f64],
    zeta: &mut [f64],
    tau: &mut f64,
    latent_dim: usize,
) {
    match param {
        M2Param::B(i) => b[i] = value,
        M2Param::Alpha(i) => alpha[i] = value,
        M2Param::Zeta(i, k) => zeta[i * latent_dim + k] = value,
        M2Param::Tau => *tau = value,
    }
}

fn srmsr_from_sum(sum_squared: f64, count: usize) -> f64 {
    if count > 0 {
        (sum_squared / count as f64).sqrt()
    } else {
        f64::NAN
    }
}

fn comparative_fit_metrics(m2: f64, df: f64, null_m2: f64, null_df: f64) -> (f64, f64) {
    let cfi_denom = null_m2 - null_df;
    let cfi = if null_m2 > m2 && cfi_denom > 0.0 {
        (1.0 - (m2 - df) / cfi_denom).clamp(0.0, 1.0)
    } else {
        f64::NAN
    };
    let null_ratio = null_m2 / null_df;
    let tli_denom = null_ratio - 1.0;
    let tli = if null_m2 > m2 && tli_denom.abs() > 1e-12 {
        (null_ratio - m2 / df) / tli_denom
    } else {
        f64::NAN
    };
    (cfi, tli)
}

/// In-place lower-triangular Cholesky with an adaptive ridge; leaves the factor
/// in the lower triangle of `a` (row-major n x n) and zeros the upper triangle.
fn cholesky_lower(a: &mut [f64], n: usize) -> Result<(), String> {
    let diag_mean = (0..n).map(|i| a[i * n + i]).sum::<f64>() / n.max(1) as f64;
    let base = diag_mean.abs().max(1e-12) * 1e-10;
    let orig = a.to_vec();
    for attempt in 0..8 {
        if attempt > 0 {
            a.copy_from_slice(&orig);
            let ridge = base * (10.0_f64).powi(attempt as i32);
            for i in 0..n {
                a[i * n + i] += ridge;
            }
        }
        let mut ok = true;
        for j in 0..n {
            let mut d = a[j * n + j];
            for k in 0..j {
                d -= a[j * n + k] * a[j * n + k];
            }
            if d <= 0.0 {
                ok = false;
                break;
            }
            let ljj = d.sqrt();
            a[j * n + j] = ljj;
            for i in (j + 1)..n {
                let mut sdot = a[i * n + j];
                for k in 0..j {
                    sdot -= a[i * n + k] * a[j * n + k];
                }
                a[i * n + j] = sdot / ljj;
            }
        }
        if ok {
            for j in 0..n {
                for i in 0..j {
                    a[i * n + j] = 0.0;
                }
            }
            return Ok(());
        }
    }
    Err("matrix is not positive definite (degenerate margins?)".into())
}

/// Solve `L L' x = b` for the lower factor `l` (row-major n x n).
fn chol_solve(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut sdot = b[i];
        for k in 0..i {
            sdot -= l[i * n + k] * y[k];
        }
        y[i] = sdot / l[i * n + i];
    }
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sdot = y[i];
        for k in (i + 1)..n {
            sdot -= l[k * n + i] * x[k];
        }
        x[i] = sdot / l[i * n + i];
    }
    x
}

/// Evaluate the M2 projected quadratic form without forming either inverse.
fn projected_m2(
    e: &[f64],
    delta: &[f64],
    mut xi: Vec<f64>,
    n_moments: usize,
    n_parameters: usize,
    n: f64,
) -> Result<f64, String> {
    let s = n_moments;
    let p = n_parameters;
    cholesky_lower(&mut xi, s)?;
    let u = chol_solve(&xi, s, e); // Xi^-1 e
    let mut w = vec![0.0_f64; s * p]; // Xi^-1 Delta
    let mut col_b = vec![0.0_f64; s];
    for col in 0..p {
        for row in 0..s {
            col_b[row] = delta[row * p + col];
        }
        let wc = chol_solve(&xi, s, &col_b);
        for row in 0..s {
            w[row * p + col] = wc[row];
        }
    }
    let mut amat = vec![0.0_f64; p * p]; // Delta' Xi^-1 Delta
    let mut g = vec![0.0_f64; p]; // Delta' Xi^-1 e
    for r in 0..p {
        for c in 0..p {
            let mut acc = 0.0;
            for row in 0..s {
                acc += delta[row * p + r] * w[row * p + c];
            }
            amat[r * p + c] = acc;
        }
        let mut gg = 0.0;
        for row in 0..s {
            gg += w[row * p + r] * e[row];
        }
        g[r] = gg;
    }
    cholesky_lower(&mut amat, p)?;
    let z = chol_solve(&amat, p, &g);
    let quad: f64 = (0..s).map(|a| e[a] * u[a]).sum();
    let adj: f64 = (0..p).map(|r| g[r] * z[r]).sum();
    Ok((n * (quad - adj)).max(0.0))
}

/// Central chi-square CDF via the survival function.
#[inline]
fn chi2_cdf(x: f64, df: f64) -> f64 {
    1.0 - chi2_sf(x, df)
}

fn finish_ncchi2_mixture(weighted: f64, normalizer: f64, converged: bool) -> f64 {
    if converged {
        (weighted / normalizer).clamp(0.0, 1.0)
    } else {
        f64::NAN
    }
}

/// Noncentral chi-square CDF from a mode-centered Poisson mixture. Centering the
/// recurrence avoids underflow of the `exp(-lam / 2)` starting weight for large
/// noncentralities (Benton & Krishnamoorthy, 2003).
///
/// Benton, D., & Krishnamoorthy, K. (2003). Computing discrete mixtures of
/// continuous distributions: Noncentral chi-square, noncentral *t* and the
/// distribution of the square of the sample multiple correlation coefficient.
/// *Computational Statistics & Data Analysis, 43*(2), 249-267.
/// https://doi.org/10.1016/S0167-9473(02)00283-9
fn ncchi2_cdf(x: f64, df: f64, lam: f64) -> f64 {
    if lam <= 0.0 {
        return chi2_cdf(x, df);
    }
    if !(x.is_finite() && df.is_finite() && lam.is_finite()) {
        return f64::NAN;
    }
    let half = 0.5 * lam;
    let mode = half.floor() as usize;
    let mut weighted = chi2_cdf(x, df + 2.0 * mode as f64);
    let mut normalizer = 1.0_f64;

    let mut weight = 1.0_f64;
    let mut j = mode;
    while j > 0 {
        weight *= j as f64 / half;
        j -= 1;
        normalizer += weight;
        weighted += weight * chi2_cdf(x, df + 2.0 * j as f64);
        if weight <= 1e-15 * normalizer {
            break;
        }
    }

    weight = 1.0;
    j = mode;
    let mut converged = false;
    for _ in 0..100_000 {
        j += 1;
        weight *= half / j as f64;
        normalizer += weight;
        weighted += weight * chi2_cdf(x, df + 2.0 * j as f64);
        if weight <= 1e-15 * normalizer {
            converged = true;
            break;
        }
    }
    finish_ncchi2_mixture(weighted, normalizer, converged)
}

/// Smallest noncentrality `lam` with `ncchi2_cdf(x, df, lam) = target` (the CDF
/// is monotone decreasing in `lam`); returns 0 if already unattainable.
fn nc_lambda_for(x: f64, df: f64, target: f64) -> f64 {
    if chi2_cdf(x, df) <= target {
        return 0.0;
    }
    let mut hi = 1.0_f64;
    while ncchi2_cdf(x, df, hi) > target && hi < 1e8 {
        hi *= 2.0;
    }
    solve_decreasing_root(0.0, hi, target, &|lambda| ncchi2_cdf(x, df, lambda))
}

fn solve_decreasing_root(
    mut lo: f64,
    mut hi: f64,
    target: f64,
    evaluate: &dyn Fn(f64) -> f64,
) -> f64 {
    if evaluate(hi) > target {
        return f64::NAN;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if evaluate(mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= 1e-12 * (1.0 + mid) {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// Integrate simple-structure item-set margins over independent trait
/// dimensions conditional on the common latent-space node. `icc_nodes` stores
/// one trait axis because each item loads on one factor; products spanning
/// distinct factors must therefore be integrated separately, not evaluated at
/// the same trait node.
fn factorized_trait_moments(
    probs: &[f64],
    weights: &[f64],
    q_theta: usize,
    factor_id: &[usize],
    n_dims: usize,
    item_sets: &[Vec<usize>],
) -> Vec<f64> {
    let n_x = weights.len() / q_theta;
    let cell = weights.len();
    let mut trait_weights = vec![0.0_f64; q_theta];
    let mut space_weights = vec![0.0_f64; n_x];
    for t in 0..q_theta {
        for x in 0..n_x {
            let weight = weights[t * n_x + x];
            trait_weights[t] += weight;
            space_weights[x] += weight;
        }
    }
    item_sets
        .iter()
        .map(|set| {
            let mut total = 0.0_f64;
            for x in 0..n_x {
                let mut conditional = 1.0_f64;
                for d in 0..n_dims {
                    if !set.iter().any(|&item| factor_id[item] == d) {
                        continue;
                    }
                    let mut margin = 0.0_f64;
                    for t in 0..q_theta {
                        let mut product = 1.0_f64;
                        for &item in set {
                            if factor_id[item] == d {
                                product *= probs[item * cell + t * n_x + x];
                            }
                        }
                        margin += trait_weights[t] * product;
                    }
                    conditional *= margin;
                }
                total += space_weights[x] * conditional;
            }
            total
        })
        .collect()
}

/// M2 statistic (order-2 residuals), df, p-value, RMSEA2 (+ 90% CI), and the
/// bivariate SRMSR for a fitted dichotomous item bank on the `(theta, xi)`
/// node set. Complete cases only (M2 assumes a single sample size N).
///
/// ponytail: Xi is s x s (s = n + n(n-1)/2) so the build is O(s^2 * nodes) and
/// the Cholesky O(s^3); this is a one-shot diagnostic, not a hot path. For very
/// large banks prefer S-X2 (already streaming) and read M2 as an overall check.
pub fn m2_rmsea2(
    bank: &ItemBank<'_>,
    y: &[f64],
    observed: &[bool],
    n_persons: usize,
    prior: &PriorSpec,
    q_theta: usize,
    xi_rule: XiRule,
) -> Result<M2Result, String> {
    let n_items = bank.b.len();
    if n_items < 3 {
        return Err("M2 needs at least 3 items".into());
    }
    if y.len() != n_persons * n_items || observed.len() != y.len() {
        return Err("y and observed must both have length n_persons * n_items".into());
    }
    let (free_alpha, uses_space) = model_exec_flags(bank.model_type);
    let kind = crate::interaction_kind(bank.model_type);

    // moment layout: [0..n) univariate, then bivariate pairs (i < j)
    let mut moment_items: Vec<Vec<usize>> = (0..n_items).map(|i| vec![i]).collect();
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..n_items {
        for j in (i + 1)..n_items {
            pairs.push((i, j));
            moment_items.push(vec![i, j]);
        }
    }
    let s = moment_items.len();

    // free item parameters (Delta columns), matching the estimator's count
    let tau_free = kind == crate::InteractionKind::Distance && uses_space;
    let params = m2_parameters(n_items, free_alpha, uses_space, bank.latent_dim, tau_free);
    let p = params.len();
    if s <= p {
        return Err(format!(
            "M2 df non-positive: {s} moments <= {p} parameters (need more items)"
        ));
    }

    // observed margins on complete cases
    let mut complete: Vec<usize> = Vec::with_capacity(n_persons);
    for pp in 0..n_persons {
        if (0..n_items).all(|i| observed[pp * n_items + i]) {
            complete.push(pp);
        }
    }
    let n_c = complete.len();
    if n_c < p + 2 {
        return Err(format!("too few complete cases for M2: {n_c}"));
    }
    let n_f = n_c as f64;
    let mut p_obs = vec![0.0_f64; s];
    for &pp in &complete {
        for i in 0..n_items {
            if y[pp * n_items + i] != 0.0 {
                p_obs[i] += 1.0;
            }
        }
        for (idx, &(i, j)) in pairs.iter().enumerate() {
            if y[pp * n_items + i] != 0.0 && y[pp * n_items + j] != 0.0 {
                p_obs[n_items + idx] += 1.0;
            }
        }
    }
    for v in p_obs.iter_mut() {
        *v /= n_f;
    }

    // node probabilities at the fitted parameters + node weights
    let (probs0, weights, _theta, _cell) = icc_nodes(bank, prior, q_theta, xi_rule)?;
    let model_moments = |probs: &[f64]| -> Vec<f64> {
        factorized_trait_moments(
            probs,
            &weights,
            q_theta,
            bank.factor_id,
            bank.n_dims,
            &moment_items,
        )
    };
    let pi_set = |probs: &[f64], set: &[usize]| -> f64 {
        factorized_trait_moments(
            probs,
            &weights,
            q_theta,
            bank.factor_id,
            bank.n_dims,
            &[set.to_vec()],
        )[0]
    };
    let mom0 = model_moments(&probs0);
    let e: Vec<f64> = (0..s).map(|a| p_obs[a] - mom0[a]).collect();

    // Delta_2 (s x p, row-major) by central differences of the node moments
    let alpha0 = bank.alpha.to_vec();
    let b0 = bank.b.to_vec();
    let zeta0 = bank.zeta.to_vec();
    let tau0 = bank.tau;
    let probs_for =
        |alpha: &[f64], b: &[f64], zeta: &[f64], tau: f64| -> Result<Vec<f64>, String> {
            let tb = ItemBank {
                alpha,
                b,
                zeta,
                tau,
                factor_id: bank.factor_id,
                model_type: bank.model_type,
                n_dims: bank.n_dims,
                latent_dim: bank.latent_dim,
                eps_distance: bank.eps_distance,
            };
            let (pr, _w, _t, _c) = icc_nodes(&tb, prior, q_theta, xi_rule)?;
            Ok(pr)
        };
    let mut delta = vec![0.0_f64; s * p];
    let ld = bank.latent_dim;
    for (col, param) in params.iter().enumerate() {
        let base = m2_param_value(*param, &alpha0, &b0, &zeta0, tau0, ld);
        let h = 1e-4 * (1.0 + base.abs());
        let mut a = alpha0.clone();
        let mut b = b0.clone();
        let mut z = zeta0.clone();
        let mut t = tau0;
        set_m2_param(*param, base + h, &mut a, &mut b, &mut z, &mut t, ld);
        let mom_plus = model_moments(&probs_for(&a, &b, &z, t)?);
        set_m2_param(*param, base - h, &mut a, &mut b, &mut z, &mut t, ld);
        let mom_minus = model_moments(&probs_for(&a, &b, &z, t)?);
        let inv = 0.5 / h;
        for row in 0..s {
            delta[row * p + col] = (mom_plus[row] - mom_minus[row]) * inv;
        }
    }

    // Xi_2: multinomial covariance of the stacked margins (union margins exact
    // via the local-independence factorization)
    let mut xi = vec![0.0_f64; s * s];
    for a in 0..s {
        for b in a..s {
            let mut u = moment_items[a].clone();
            for &m in &moment_items[b] {
                if !u.contains(&m) {
                    u.push(m);
                }
            }
            let cov = pi_set(&probs0, &u) - mom0[a] * mom0[b];
            xi[a * s + b] = cov;
            xi[b * s + a] = cov;
        }
    }

    // M2 = N ( e'Xi^-1 e - (D'Xi^-1 e)'(D'Xi^-1 D)^-1 (D'Xi^-1 e) )
    let m2 = projected_m2(&e, &delta, xi, s, p, n_f)?;
    let df = (s - p) as f64;
    let p_value = chi2_sf(m2, df);
    let denom = df * (n_f - 1.0);
    let rmsea2 = ((m2 - df).max(0.0) / denom).sqrt();
    let rmsea2_ci_lower = (nc_lambda_for(m2, df, 0.95) / denom).sqrt();
    let rmsea2_ci_upper = (nc_lambda_for(m2, df, 0.05) / denom).sqrt();

    // bivariate SRMSR over residual phi-correlations
    let mut ssum = 0.0_f64;
    let mut cnt = 0usize;
    for (idx, &(i, j)) in pairs.iter().enumerate() {
        let (pi, pj, pij) = (p_obs[i], p_obs[j], p_obs[n_items + idx]);
        let (mi, mj, mij) = (mom0[i], mom0[j], mom0[n_items + idx]);
        let dobs = pi * (1.0 - pi) * pj * (1.0 - pj);
        let dmod = mi * (1.0 - mi) * mj * (1.0 - mj);
        if dobs > 1e-12 && dmod > 1e-12 {
            let robs = (pij - pi * pj) / dobs.sqrt();
            let rmod = (mij - mi * mj) / dmod.sqrt();
            ssum += (robs - rmod) * (robs - rmod);
            cnt += 1;
        }
    }
    let srmsr = srmsr_from_sum(ssum, cnt);

    // Fit a zero-factor / complete-independence baseline to the same complete
    // cases. Its free item margins reproduce the observed univariate margins;
    // all higher-order moments are products of those margins. The analytic
    // Jacobian below uses the margins themselves as the parameterization (the
    // projected statistic is invariant to a full-rank reparameterization).
    let null_p = n_items;
    let null_mom: Vec<f64> = moment_items
        .iter()
        .map(|set| set.iter().map(|&i| p_obs[i]).product())
        .collect();
    let null_e: Vec<f64> = (0..s).map(|a| p_obs[a] - null_mom[a]).collect();
    let mut null_delta = vec![0.0_f64; s * null_p];
    for (row, set) in moment_items.iter().enumerate() {
        for &col in set {
            let derivative: f64 = set
                .iter()
                .filter(|&&i| i != col)
                .map(|&i| p_obs[i])
                .product();
            null_delta[row * null_p + col] = derivative;
        }
    }
    let mut null_xi = vec![0.0_f64; s * s];
    for a in 0..s {
        for b in a..s {
            let mut union = moment_items[a].clone();
            for &item in &moment_items[b] {
                if !union.contains(&item) {
                    union.push(item);
                }
            }
            let union_moment: f64 = union.iter().map(|&i| p_obs[i]).product();
            let cov = union_moment - null_mom[a] * null_mom[b];
            null_xi[a * s + b] = cov;
            null_xi[b * s + a] = cov;
        }
    }
    let null_m2 = projected_m2(&null_e, &null_delta, null_xi, s, null_p, n_f)?;
    let null_df = (s - null_p) as f64;
    let (cfi, tli) = comparative_fit_metrics(m2, df, null_m2, null_df);

    Ok(M2Result {
        m2,
        df,
        p_value,
        rmsea2,
        rmsea2_ci_lower,
        rmsea2_ci_upper,
        srmsr,
        null_m2,
        null_df,
        cfi,
        tli,
        n_moments: s,
        n_parameters: p,
        n_complete: n_c,
    })
}

/// Per-item-pair local-dependence diagnostics (Chen & Thissen 1997).
#[derive(Clone, Debug)]
pub struct PolyLdResult {
    /// Upper-triangle pair order `(i, j)`, `i < j`.
    pub pairs: Vec<(usize, usize)>,
    /// Pearson LD chi-square per pair.
    pub x2: Vec<f64>,
    /// Likelihood-ratio (G²) LD statistic per pair.
    pub g2: Vec<f64>,
    /// Reference degrees of freedom `(K-1)²` shared by all pairs.
    pub df: f64,
    /// Upper-tail p-value of `x2` on `χ²(df)`.
    pub p_value: Vec<f64>,
    /// Cramér's V effect size `sqrt(X² / (N(K-1)))` per pair.
    pub cramers_v: Vec<f64>,
    /// Largest standardized cell residual `|O-E|/sqrt(E)` per pair.
    pub max_abs_std_resid: Vec<f64>,
    /// Pairwise-complete sample size per pair.
    pub n_pair: Vec<usize>,
}

fn validate_optional_observed_length(
    observed: Option<&[bool]>,
    expected: usize,
) -> Result<(), String> {
    if observed.is_some_and(|values| values.len() != expected) {
        Err("observed must have length n_persons * n_items".into())
    } else {
        Ok(())
    }
}

/// Local-dependence diagnostics for every item pair of a fitted unidimensional
/// GRM/GPCM (Chen & Thissen, 1997), the ordered-category generalization of the
/// binary pairwise chi-square in [`adjusted_chi2_pairs`]. For each pair `(i,j)`
/// it compares the observed `K×K` contingency table against the model-implied
/// joint under local independence,
/// `E_ab = N · Σ_t w_t P_i(a|θ_t) P_j(b|θ_t)`, reporting the Pearson `X²` and
/// likelihood-ratio `G²` with `df = (K-1)²`, the χ² p-value, Cramér's V, and the
/// largest standardized cell residual. Pairwise-complete cases (a person
/// observed on both items of a pair) enter that pair's table; a pair with too
/// few cases yields `NaN`. A significant `X²` beyond the fitted model flags
/// residual association (a violated conditional-independence assumption).
///
/// # References (APA 7th ed.)
///
/// Chen, W.-H., & Thissen, D. (1997). Local dependence indexes for item pairs
///   using item response theory. *Journal of Educational and Behavioral
///   Statistics, 22*(3), 265–289. https://doi.org/10.3102/10769986022003265
///
/// Liu, Y., & Maydeu-Olivares, A. (2013). Local dependence diagnostics in IRT
///   modeling of binary data. *Educational and Psychological Measurement,
///   73*(2), 254–274. https://doi.org/10.1177/0013164412453841
#[allow(clippy::too_many_arguments)]
pub fn poly_local_dependence(
    y: &[usize],
    observed: Option<&[bool]>,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    slope: &[f64],
    cat_params: &[f64],
    model: crate::poly::PolyModel,
    q_theta: usize,
) -> Result<PolyLdResult, String> {
    use crate::poly::{gpcm_logprobs, grm_logprobs, PolyModel};
    if n_items < 2 {
        return Err("local dependence needs at least 2 items".into());
    }
    if n_cat < 2 {
        return Err("n_cat must be >= 2".into());
    }
    if y.len() != n_persons * n_items {
        return Err("y must have length n_persons * n_items".into());
    }
    validate_optional_observed_length(observed, y.len())?;
    if slope.len() != n_items {
        return Err("slope must have length n_items".into());
    }
    if cat_params.len() != n_items * (n_cat - 1) {
        return Err("cat_params must have length n_items*(n_cat-1)".into());
    }
    if y.iter().any(|&v| v >= n_cat) {
        return Err("response categories must be < n_cat".into());
    }
    let z = n_cat - 1;

    // per-item, per-node category probabilities P_i(a | theta_t)
    let (nodes, weights) = crate::quadrature::require_gh_rule(q_theta, "quadrature size")?;
    let qn = nodes.len();
    let mut probs = vec![0.0_f64; n_items * qn * n_cat];
    for i in 0..n_items {
        let a = slope[i];
        let cp = &cat_params[i * z..(i + 1) * z];
        for (t, &theta) in nodes.iter().enumerate() {
            let base = a * theta;
            let lp = match model {
                PolyModel::Gpcm => {
                    let scores: Vec<f64> = (0..n_cat).map(|c| c as f64).collect();
                    let mut intercepts = vec![0.0_f64; n_cat];
                    intercepts[1..].copy_from_slice(cp);
                    gpcm_logprobs(base, &scores, &intercepts)
                }
                PolyModel::Grm => grm_logprobs(base, cp),
            };
            let off = (i * qn + t) * n_cat;
            for a2 in 0..n_cat {
                probs[off + a2] = lp[a2].exp();
            }
        }
    }

    let is_obs = |pp: usize, i: usize| observed.map_or(true, |o| o[pp * n_items + i]);
    // Chen & Thissen (1997) / mirt reference: the two-way independence df,
    // df = cells - independence-model params = K² - (2K-1) = (K-1)² (binary -> 1).
    // With FITTED item parameters the marginal MLE absorbs the univariate margins,
    // leaving the (K-1)² association dof; the reference is heuristic/conservative
    // (both papers note the null is stochastically smaller than χ²), so read it as
    // a diagnostic screen for residual association, not an exact test.
    let df = (z * z) as f64;
    let mut out = PolyLdResult {
        pairs: Vec::new(),
        x2: Vec::new(),
        g2: Vec::new(),
        df,
        p_value: Vec::new(),
        cramers_v: Vec::new(),
        max_abs_std_resid: Vec::new(),
        n_pair: Vec::new(),
    };
    let min_n = 2 * n_cat * n_cat; // enough for a non-degenerate K x K table

    for i in 0..n_items {
        for j in (i + 1)..n_items {
            // model-implied joint P(Y_i=a, Y_j=b) marginalized over theta
            let mut pj = vec![0.0_f64; n_cat * n_cat];
            for t in 0..qn {
                let wt = weights[t];
                let io = (i * qn + t) * n_cat;
                let jo = (j * qn + t) * n_cat;
                for a in 0..n_cat {
                    let pia = wt * probs[io + a];
                    for b in 0..n_cat {
                        pj[a * n_cat + b] += pia * probs[jo + b];
                    }
                }
            }
            // observed K x K counts (pairwise-complete)
            let mut o = vec![0.0_f64; n_cat * n_cat];
            let mut n_ij = 0usize;
            for pp in 0..n_persons {
                if is_obs(pp, i) && is_obs(pp, j) {
                    o[y[pp * n_items + i] * n_cat + y[pp * n_items + j]] += 1.0;
                    n_ij += 1;
                }
            }
            out.pairs.push((i, j));
            out.n_pair.push(n_ij);
            if n_ij < min_n {
                out.x2.push(f64::NAN);
                out.g2.push(f64::NAN);
                out.p_value.push(f64::NAN);
                out.cramers_v.push(f64::NAN);
                out.max_abs_std_resid.push(f64::NAN);
                continue;
            }
            let nf = n_ij as f64;
            let (mut x2, mut g2, mut maxr) = (0.0_f64, 0.0_f64, 0.0_f64);
            for cell in 0..n_cat * n_cat {
                let e = nf * pj[cell];
                if e > 1e-12 {
                    let d = o[cell] - e;
                    x2 += d * d / e;
                    let sr = (d / e.sqrt()).abs();
                    if sr > maxr {
                        maxr = sr;
                    }
                    if o[cell] > 0.0 {
                        g2 += 2.0 * o[cell] * (o[cell] / e).ln();
                    }
                }
            }
            out.x2.push(x2);
            out.g2.push(g2);
            out.p_value.push(chi2_sf(x2, df));
            out.cramers_v.push((x2 / (nf * z as f64)).sqrt());
            out.max_abs_std_resid.push(maxr);
        }
    }
    Ok(out)
}

/// Polytomous M2 / RMSEA2 limited-information goodness of fit for a fitted
/// unidimensional GRM or GPCM, the ordered-category generalization of
/// [`m2_rmsea2`]. Uses the CUMULATIVE marginal form: univariate
/// `m_i(c) = P(Y_i >= c)` for `c = 1..K-1` and bivariate
/// `m_ij(c,d) = P(Y_i >= c, Y_j >= d)` for `i < j`, `c,d = 1..K-1` — provably the
/// same M2 statistic as Maydeu-Olivares & Joe's category-equality form (the two
/// moment vectors differ by a fixed invertible block map `T` under which
/// `M2 = N e'[Ξ⁻¹ − Ξ⁻¹Δ(Δ'Ξ⁻¹Δ)⁻¹Δ'Ξ⁻¹]e` is invariant), and it reduces
/// EXACTLY to [`m2_rmsea2`] at `K = 2`. Model moments factor over the
/// `q_theta`-node `N(0,1)` grid by local independence
/// (`m_ij = Σ_t w_t S_i(c|t) S_j(d|t)`, `S_i(c|t) = P(Y_i >= c | θ_t)`); `Δ` is a
/// central-difference Jacobian; the multinomial covariance `Ξ` uses the nesting
/// rule `1{Y_i>=c}·1{Y_i>=c'} = 1{Y_i>=max(c,c')}` (max-threshold collapse). The
/// statistic reuses the same one-Cholesky solve as the binary path. Complete
/// cases only. `df = Q − P` with `Q = n(K-1) + C(n,2)(K-1)²`, `P = n·K`.
///
/// RMSEA2 uses the denominator `df·(N−1)` (as [`m2_rmsea2`] and the `mirt`
/// package); Maydeu-Olivares & Joe (2014, Eq. 14) instead scale by `N·df` — the
/// two differ negligibly and only in RMSEA2 and its interval, not in M2, df, or
/// the p-value.
///
/// # References (APA 7th ed.)
///
/// Maydeu-Olivares, A., & Joe, H. (2014). Assessing approximate fit in
///   categorical data analysis. *Multivariate Behavioral Research, 49*(4),
///   305–328. https://doi.org/10.1080/00273171.2014.911075
///
/// Maydeu-Olivares, A. (2013). Goodness-of-fit assessment of item response
///   theory models. *Measurement: Interdisciplinary Research and Perspectives,
///   11*(3), 71–101. https://doi.org/10.1080/15366367.2013.831680
#[allow(clippy::too_many_arguments)]
pub fn poly_m2(
    y: &[usize],
    observed: Option<&[bool]>,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    slope: &[f64],
    cat_params: &[f64],
    model: crate::poly::PolyModel,
    q_theta: usize,
) -> Result<M2Result, String> {
    use crate::poly::{gpcm_logprobs, grm_logprobs, PolyModel};
    if n_items < 3 {
        return Err("M2 needs at least 3 items".into());
    }
    if n_cat < 2 {
        return Err("n_cat must be >= 2".into());
    }
    if y.len() != n_persons * n_items {
        return Err("y must have length n_persons * n_items".into());
    }
    validate_optional_observed_length(observed, y.len())?;
    if slope.len() != n_items {
        return Err("slope must have length n_items".into());
    }
    if cat_params.len() != n_items * (n_cat - 1) {
        return Err("cat_params must have length n_items*(n_cat-1)".into());
    }
    if y.iter().any(|&v| v >= n_cat) {
        return Err("response categories must be < n_cat".into());
    }

    let z = n_cat - 1; // highest threshold index
                       // moment layout: item-major univariate (i,c), then bivariate pairs (i<j)x(c,d)
    let mut moment_cons: Vec<Vec<(usize, usize)>> = Vec::new();
    for i in 0..n_items {
        for c in 1..=z {
            moment_cons.push(vec![(i, c)]);
        }
    }
    let base_biv = moment_cons.len(); // = n_items * z
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..n_items {
        for j in (i + 1)..n_items {
            pairs.push((i, j));
            for c in 1..=z {
                for d in 1..=z {
                    moment_cons.push(vec![(i, c), (j, d)]);
                }
            }
        }
    }
    let s = moment_cons.len(); // Q
    let p = n_items * n_cat; // slope + (K-1) cat params per item
    if s <= p {
        return Err(format!(
            "M2 df non-positive: {s} moments <= {p} parameters (need more items)"
        ));
    }

    // complete cases (M2 assumes a single sample size N)
    let is_obs = |pp: usize, i: usize| observed.map_or(true, |o| o[pp * n_items + i]);
    let mut complete: Vec<usize> = Vec::with_capacity(n_persons);
    for pp in 0..n_persons {
        if (0..n_items).all(|i| is_obs(pp, i)) {
            complete.push(pp);
        }
    }
    let n_c = complete.len();
    if n_c < p + 2 {
        return Err(format!("too few complete cases for M2: {n_c}"));
    }
    let n_f = n_c as f64;

    // observed cumulative margins
    let mut p_hat = vec![0.0_f64; s];
    for &pp in &complete {
        for (a, cons) in moment_cons.iter().enumerate() {
            if cons.iter().all(|&(i, c)| y[pp * n_items + i] >= c) {
                p_hat[a] += 1.0;
            }
        }
    }
    for v in p_hat.iter_mut() {
        *v /= n_f;
    }

    // cumulative-probability tensor S[(i*qn+t)*z + (c-1)] = P(Y_i >= c | theta_t)
    let (nodes, weights) = crate::quadrature::require_gh_rule(q_theta, "quadrature size")?;
    let qn = nodes.len();
    let build_cum = |slope: &[f64], cat_params: &[f64]| -> Vec<f64> {
        let mut sc = vec![0.0_f64; n_items * qn * z];
        for i in 0..n_items {
            let a = slope[i];
            let cp = &cat_params[i * z..(i + 1) * z];
            for (t, &theta) in nodes.iter().enumerate() {
                let base = a * theta;
                let lp = match model {
                    PolyModel::Gpcm => {
                        let scores: Vec<f64> = (0..n_cat).map(|c| c as f64).collect();
                        let mut intercepts = vec![0.0_f64; n_cat];
                        intercepts[1..].copy_from_slice(cp);
                        gpcm_logprobs(base, &scores, &intercepts)
                    }
                    PolyModel::Grm => grm_logprobs(base, cp),
                };
                // P(Y>=c) = sum_{k>=c} P(Y=k), accumulated from the top category down
                let off = (i * qn + t) * z;
                let mut acc = 0.0_f64;
                for c in (1..=z).rev() {
                    acc += lp[c].exp();
                    sc[off + (c - 1)] = acc;
                }
            }
        }
        sc
    };
    // model marginal over a distinct-item constraint list (local independence)
    let cum_joint = |sc: &[f64], cons: &[(usize, usize)]| -> f64 {
        (0..qn)
            .map(|t| {
                let mut pr = weights[t];
                for &(i, c) in cons {
                    pr *= sc[(i * qn + t) * z + (c - 1)];
                }
                pr
            })
            .sum()
    };
    let model_moments =
        |sc: &[f64]| -> Vec<f64> { moment_cons.iter().map(|cons| cum_joint(sc, cons)).collect() };

    let s0 = build_cum(slope, cat_params);
    let mom0 = model_moments(&s0);
    let e: Vec<f64> = (0..s).map(|a| p_hat[a] - mom0[a]).collect();

    // guard degenerate moments (empty/boundary category => Xi singular, df invalid)
    for (a, &m) in mom0.iter().enumerate() {
        if m * (1.0 - m) < 1e-10 {
            return Err(format!(
                "degenerate moment {a} (empty/boundary category); M2 df invalid"
            ));
        }
    }

    // Delta (s x p) by central differences; columns per item: slope then z cat params
    let mut params: Vec<(usize, isize)> = Vec::new(); // (item, -1 = slope else cat index)
    for i in 0..n_items {
        params.push((i, -1));
        for m in 0..z as isize {
            params.push((i, m));
        }
    }
    let mut delta = vec![0.0_f64; s * p];
    for (col, &(pi, which)) in params.iter().enumerate() {
        let mut sl = slope.to_vec();
        let mut cp = cat_params.to_vec();
        let base = if which < 0 {
            sl[pi]
        } else {
            cp[pi * z + which as usize]
        };
        let h = 1e-4 * (1.0 + base.abs());
        if which < 0 {
            sl[pi] = base + h;
        } else {
            cp[pi * z + which as usize] = base + h;
        }
        let mom_plus = model_moments(&build_cum(&sl, &cp));
        if which < 0 {
            sl[pi] = base - h;
        } else {
            cp[pi * z + which as usize] = base - h;
        }
        let mom_minus = model_moments(&build_cum(&sl, &cp));
        let inv = 0.5 / h;
        for row in 0..s {
            delta[row * p + col] = (mom_plus[row] - mom_minus[row]) * inv;
        }
    }

    // Xi: multinomial covariance of the cumulative margins. Cumulative indicators
    // nest within an item (1{Y_i>=c}1{Y_i>=c'} = 1{Y_i>=max}), so merge the two
    // constraint lists keeping the LARGER threshold per shared item.
    let mut xi = vec![0.0_f64; s * s];
    for a in 0..s {
        for b in a..s {
            let mut merged = moment_cons[a].clone();
            for &(j, thr) in &moment_cons[b] {
                if let Some(slot) = merged.iter_mut().find(|(i, _)| *i == j) {
                    slot.1 = slot.1.max(thr);
                } else {
                    merged.push((j, thr));
                }
            }
            let cov = cum_joint(&s0, &merged) - mom0[a] * mom0[b];
            xi[a * s + b] = cov;
            xi[b * s + a] = cov;
        }
    }

    // M2 = N ( e'Xi^-1 e - (D'Xi^-1 e)'(D'Xi^-1 D)^-1 (D'Xi^-1 e) )
    let m2 = projected_m2(&e, &delta, xi, s, p, n_f)?;
    let df = (s - p) as f64;
    let p_value = chi2_sf(m2, df);
    let denom = df * (n_f - 1.0);
    let rmsea2 = ((m2 - df).max(0.0) / denom).sqrt();
    let rmsea2_ci_lower = (nc_lambda_for(m2, df, 0.95) / denom).sqrt();
    let rmsea2_ci_upper = (nc_lambda_for(m2, df, 0.05) / denom).sqrt();

    // first-order (c=d=1) bivariate SRMSR
    let uni1 = |i: usize| i * z; // (i, c=1)
    let biv11 = |idx: usize| base_biv + idx * z * z; // (idx, c=1, d=1)
    let (mut ssum, mut cnt) = (0.0_f64, 0usize);
    for (idx, &(i, j)) in pairs.iter().enumerate() {
        let (pi, pj, pij) = (p_hat[uni1(i)], p_hat[uni1(j)], p_hat[biv11(idx)]);
        let (mi, mj, mij) = (mom0[uni1(i)], mom0[uni1(j)], mom0[biv11(idx)]);
        let dobs = pi * (1.0 - pi) * pj * (1.0 - pj);
        let dmod = mi * (1.0 - mi) * mj * (1.0 - mj);
        if dobs > 1e-12 && dmod > 1e-12 {
            let robs = (pij - pi * pj) / dobs.sqrt();
            let rmod = (mij - mi * mj) / dmod.sqrt();
            ssum += (robs - rmod) * (robs - rmod);
            cnt += 1;
        }
    }
    let srmsr = srmsr_from_sum(ssum, cnt);

    // Complete-independence baseline. Each item's K-1 cumulative margins are
    // free and reproduce the observed univariate cumulative margins; joint
    // cumulative moments factor across items. This is the polytomous analogue
    // of the zero-factor baseline used by the dichotomous M2 path.
    let null_p = n_items * z;
    let null_mom: Vec<f64> = moment_cons
        .iter()
        .map(|cons| cons.iter().map(|&(i, c)| p_hat[i * z + (c - 1)]).product())
        .collect();
    let null_e: Vec<f64> = (0..s).map(|a| p_hat[a] - null_mom[a]).collect();
    let mut null_delta = vec![0.0_f64; s * null_p];
    for (row, cons) in moment_cons.iter().enumerate() {
        for &(item, threshold) in cons {
            let derivative: f64 = cons
                .iter()
                .filter(|&&(i, c)| i != item || c != threshold)
                .map(|&(i, c)| p_hat[i * z + (c - 1)])
                .product();
            null_delta[row * null_p + item * z + (threshold - 1)] = derivative;
        }
    }
    let mut null_xi = vec![0.0_f64; s * s];
    for a in 0..s {
        for b in a..s {
            let mut merged = moment_cons[a].clone();
            for &(item, threshold) in &moment_cons[b] {
                if let Some(slot) = merged.iter_mut().find(|(i, _)| *i == item) {
                    slot.1 = slot.1.max(threshold);
                } else {
                    merged.push((item, threshold));
                }
            }
            let union_moment: f64 = merged
                .iter()
                .map(|&(i, c)| p_hat[i * z + (c - 1)])
                .product();
            let cov = union_moment - null_mom[a] * null_mom[b];
            null_xi[a * s + b] = cov;
            null_xi[b * s + a] = cov;
        }
    }
    let null_m2 = projected_m2(&null_e, &null_delta, null_xi, s, null_p, n_f)?;
    let null_df = (s - null_p) as f64;
    let (cfi, tli) = comparative_fit_metrics(m2, df, null_m2, null_df);

    Ok(M2Result {
        m2,
        df,
        p_value,
        rmsea2,
        rmsea2_ci_lower,
        rmsea2_ci_upper,
        srmsr,
        null_m2,
        null_df,
        cfi,
        tli,
        n_moments: s,
        n_parameters: p,
        n_complete: n_c,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/fitstats_m2_branch_tests.rs"]
mod m2_branch_tests;
