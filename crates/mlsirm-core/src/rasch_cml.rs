//! Conditional maximum likelihood (CML) estimation of the Rasch model, with Andersen's LR test.
//!
//! The dichotomous Rasch model `P(X_vi = 1) = exp(theta_v - beta_i) / (1 + exp(theta_v - beta_i))` has
//! the raw score `r_v = sum_i x_vi` as the sufficient statistic for the person parameter `theta_v`.
//! Conditioning each response pattern on its raw score ELIMINATES the person parameters entirely, so the
//! item difficulties `beta` are estimated without any assumption on the ability distribution (Rasch's
//! specific objectivity) and consistently at fixed test length as `N -> infinity` — unlike joint ML
//! (inconsistent) or marginal ML (which must posit a `theta` distribution).
//!
//! With `eps_i = exp(-beta_i)`, `s_i = sum_v x_vi` (item total-correct over the retained persons),
//! `n_r` = number of persons with raw score `r`, and `gamma_r(eps)` the ELEMENTARY SYMMETRIC FUNCTION
//! of order `r` of `{eps_1, .., eps_k}` (the sum over all size-`r` subsets of products of `eps`), the
//! conditional log-likelihood is
//!
//! ```text
//!   ln L_c(beta) = -sum_i s_i beta_i - sum_{r=1}^{k-1} n_r ln gamma_r(eps).
//! ```
//!
//! Persons with raw score `0` or `k` carry no conditional information (their total contribution to
//! `ln L_c` is identically `0`, and their conditional expected item score is `0`/`1` and cancels in the
//! score equation), so they are dropped. The score equation is `observed s_i = expected`, with
//! `E[s_i | r] = eps_i gamma_{r-1}^{(i)} / gamma_r` (`gamma^{(i)}` = ESF over the items excluding `i`).
//! `beta` is identified up to an additive constant, reported centered to `sum_i beta_i = 0`.
//!
//! The ESF and its per-item / per-pair derivatives use the SUMMATION algorithm (a fresh forward
//! recursion `gamma_r += eps_j gamma_{r-1}` over the relevant item subset), which is numerically stable;
//! the subtractive "difference" recursion `gamma_r^{(i)} = gamma_r - eps_i gamma_{r-1}^{(i)}` is avoided
//! because it cancels catastrophically for very easy items (large `eps_i`) (Verhelst, Glas & van der
//! Sluis, 1984; Fischer & Molenaar, 1995).
//!
//! Andersen's (1973) conditional likelihood-ratio test of Rasch fit partitions the persons into `G`
//! subgroups, estimates `beta` within each and over the pooled sample, and refers
//! `LR = 2 [sum_g ln L_c^{(g)}(beta_hat_g) - ln L_c(beta_hat)]` to `chi^2((G - 1)(k - 1))`; a large `LR`
//! rejects the invariance of the item difficulties across the split.
//!
//! # References (APA 7th ed.)
//!
//! Andersen, E. B. (1970). Asymptotic properties of conditional maximum-likelihood estimators. *Journal
//!     of the Royal Statistical Society: Series B, 32*(2), 283-301.
//!     https://doi.org/10.1111/j.2517-6161.1970.tb00842.x
//! Andersen, E. B. (1972). The numerical solution of a set of conditional estimation equations.
//!     *Journal of the Royal Statistical Society: Series B, 34*(1), 42-54.
//!     https://doi.org/10.1111/j.2517-6161.1972.tb00887.x
//! Andersen, E. B. (1973). A goodness of fit test for the Rasch model. *Psychometrika, 38*(1), 123-140.
//!     https://doi.org/10.1007/BF02291180
//! Rasch, G. (1960). *Probabilistic models for some intelligence and attainment tests*. Danish
//!     Institute for Educational Research.
//! Verhelst, N. D., Glas, C. A. W., & van der Sluis, A. (1984). Estimation problems in the Rasch model:
//!     The basic symmetric functions. *Computational Statistics Quarterly, 1*(3), 245-262.

use crate::fitstats::chi2_sf;
use crate::poly::solve_small;
use crate::twopl::sym_inv_logdet;

/// Maximum number of items (bounds the `O(k^4)` per-iteration Hessian and keeps the plain-value ESF in
/// range; a log-domain ESF would be needed above this).
pub const CML_MAX_ITEMS: usize = 100;

/// Elementary symmetric functions `gamma_0..gamma_k` of `eps` by the summation algorithm.
fn esf(eps: &[f64]) -> Vec<f64> {
    let k = eps.len();
    let mut g = vec![0.0f64; k + 1];
    g[0] = 1.0;
    for (j, &e) in eps.iter().enumerate() {
        for r in (1..=(j + 1).min(k)).rev() {
            g[r] += e * g[r - 1];
        }
    }
    g
}

/// ESF `gamma_0..gamma_{k-1}` of the items EXCLUDING `omit`, by a fresh summation pass (stable; no
/// subtractive cancellation).
fn esf_omit(eps: &[f64], omit: usize) -> Vec<f64> {
    let k = eps.len();
    let mut g = vec![0.0f64; k]; // orders 0..=k-1
    g[0] = 1.0;
    let mut cnt = 0usize;
    for (j, &e) in eps.iter().enumerate() {
        if j == omit {
            continue;
        }
        cnt += 1;
        for r in (1..=cnt.min(k - 1)).rev() {
            g[r] += e * g[r - 1];
        }
    }
    g
}

/// ESF `gamma_0..gamma_{k-2}` of the items EXCLUDING both `a` and `b` (`a != b`), by a fresh pass.
fn esf_omit2(eps: &[f64], a: usize, b: usize) -> Vec<f64> {
    let k = eps.len();
    let mut g = vec![0.0f64; k - 1]; // orders 0..=k-2
    g[0] = 1.0;
    let mut cnt = 0usize;
    for (j, &e) in eps.iter().enumerate() {
        if j == a || j == b {
            continue;
        }
        cnt += 1;
        for r in (1..=cnt.min(k - 2)).rev() {
            g[r] += e * g[r - 1];
        }
    }
    g
}

/// Conditional log-likelihood, gradient, and Hessian at `beta` given item totals `s` and score
/// frequencies `nr` (`nr[r]` = retained persons with raw score `r`; `r = 0` and `r = k` are ignored).
fn cml_eval(beta: &[f64], s: &[f64], nr: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
    let k = beta.len();
    let eps: Vec<f64> = beta.iter().map(|b| (-b).exp()).collect();
    let g = esf(&eps);
    let gi: Vec<Vec<f64>> = (0..k).map(|i| esf_omit(&eps, i)).collect();

    let mut ll = 0.0;
    for i in 0..k {
        ll -= s[i] * beta[i];
    }
    for r in 1..k {
        if nr[r] != 0.0 {
            ll -= nr[r] * g[r].ln();
        }
    }

    let mut grad = vec![0.0f64; k];
    let mut hess = vec![0.0f64; k * k];
    // conditional expected item score E[s_i] and the diagonal (variance) term.
    for i in 0..k {
        grad[i] = -s[i];
        for r in 1..k {
            if nr[r] == 0.0 {
                continue;
            }
            let eir = eps[i] * gi[i][r - 1] / g[r];
            grad[i] += nr[r] * eir;
            hess[i * k + i] += nr[r] * (eir * eir - eir); // = -nr E_ir(1 - E_ir)
        }
    }
    // off-diagonal: H_ij = sum_r n_r [E_ir E_jr - eps_i eps_j gamma_{r-2}^{(ij)} / gamma_r]
    for i in 0..k {
        for j in (i + 1)..k {
            let gij = esf_omit2(&eps, i, j);
            let mut hij = 0.0;
            for r in 1..k {
                if nr[r] == 0.0 {
                    continue;
                }
                let eir = eps[i] * gi[i][r - 1] / g[r];
                let ejr = eps[j] * gi[j][r - 1] / g[r];
                let joint = if r >= 2 {
                    eps[i] * eps[j] * gij[r - 2] / g[r]
                } else {
                    0.0
                };
                hij += nr[r] * (eir * ejr - joint);
            }
            hess[i * k + j] = hij;
            hess[j * k + i] = hij;
        }
    }
    (ll, grad, hess)
}

/// A fitted Rasch CML result.
pub struct CmlFit {
    /// Item difficulties, centered to `sum_i beta_i = 0`.
    pub beta: Vec<f64>,
    /// Standard errors (sum-zero metric; `NaN` if the information is non-PD).
    pub se: Vec<f64>,
    /// Conditional log-likelihood at `beta`.
    pub loglik: f64,
    pub n_iter: usize,
    pub converged: bool,
    /// Persons retained (raw score in `1..k`).
    pub n_used: usize,
}

/// Reduce a complete `0/1` matrix to item totals `s` and score frequencies `nr` over the persons with
/// raw score in `1..k` (dropping the uninformative `0` and `k` patterns).
fn reduce(y: &[u8], n_persons: usize, n_items: usize) -> (Vec<f64>, Vec<f64>, usize) {
    let mut s = vec![0.0f64; n_items];
    let mut nr = vec![0.0f64; n_items + 1];
    let mut used = 0usize;
    for p in 0..n_persons {
        let row = &y[p * n_items..(p + 1) * n_items];
        let r: usize = row.iter().map(|&v| v as usize).sum();
        if r == 0 || r == n_items {
            continue;
        }
        used += 1;
        nr[r] += 1.0;
        for i in 0..n_items {
            s[i] += row[i] as f64;
        }
    }
    (s, nr, used)
}

fn center(beta: &mut [f64]) {
    let m = beta.iter().sum::<f64>() / beta.len() as f64;
    for b in beta.iter_mut() {
        *b -= m;
    }
}

/// Newton CML fit from precomputed sufficient statistics.
fn fit_from_stats(
    s: &[f64],
    nr: &[f64],
    n_used: usize,
    max_iter: usize,
    tol: f64,
) -> Result<CmlFit, String> {
    let k = s.len();
    let mut beta = vec![0.0f64; k];
    let (mut ll, mut grad, mut hess) = cml_eval(&beta, s, nr);
    let mut converged = false;
    let mut iter = 0;
    while iter < max_iter {
        iter += 1;
        // reduced Newton system: drop the last coordinate (pin its update to 0), re-center after.
        let m = k - 1;
        let mut hr: Vec<Vec<f64>> = (0..m)
            .map(|a| (0..m).map(|b| hess[a * k + b]).collect())
            .collect();
        // tiny ridge for a well-posed solve near the optimum
        for a in 0..m {
            hr[a][a] -= 1e-10;
        }
        let gr: Vec<f64> = grad[..m].to_vec();
        // Newton maximization step: beta -= H^{-1} grad.
        let step = solve_small(hr, gr);
        // backtracking to guarantee ascent of the concave conditional likelihood.
        let mut scale = 1.0f64;
        let mut accepted = false;
        for _ in 0..20 {
            let mut cand = beta.clone();
            for a in 0..m {
                cand[a] -= scale * step[a];
            }
            center(&mut cand);
            let (ll_c, g_c, h_c) = cml_eval(&cand, s, nr);
            if ll_c.is_finite() && ll_c >= ll - 1e-12 {
                beta = cand;
                ll = ll_c;
                grad = g_c;
                hess = h_c;
                accepted = true;
                break;
            }
            scale *= 0.5;
        }
        if !accepted {
            break;
        }
        if grad.iter().fold(0.0f64, |m, &v| m.max(v.abs())) < tol {
            converged = true;
            break;
        }
    }
    // SE from the pseudoinverse of the conditional information I_c = -H (rank k-1, null space = ones):
    // I_c^+ = (I_c + (1/k) J)^{-1} - (1/k) J, with J the all-ones matrix; SE_i = sqrt(I_c^+_{ii}).
    let mut se = vec![f64::NAN; k];
    let mut m = vec![0.0f64; k * k];
    let inv_k = 1.0 / k as f64;
    for a in 0..k {
        for b in 0..k {
            m[a * k + b] = -hess[a * k + b] + inv_k;
        }
    }
    if let Some((minv, _)) = sym_inv_logdet(&m, k) {
        for i in 0..k {
            let v = minv[i * k + i] - inv_k;
            se[i] = if v > 0.0 { v.sqrt() } else { f64::NAN };
        }
    }
    Ok(CmlFit {
        beta,
        se,
        loglik: ll,
        n_iter: iter,
        converged,
        n_used,
    })
}

/// Fit the dichotomous Rasch model by conditional maximum likelihood (Andersen, 1970, 1972).
///
/// `y` is a row-major `n_persons * n_items` complete `0/1` array (CML requires complete data — a person
/// with missing items has a different conditioning score set; that extension is out of scope). Persons
/// scoring `0` or `k` are dropped. Returns the sum-zero item difficulties and their standard errors.
pub fn fit_rasch_cml(
    y: &[u8],
    n_persons: usize,
    n_items: usize,
    max_iter: usize,
    tol: f64,
) -> Result<CmlFit, String> {
    validate(y, n_persons, n_items, max_iter, tol)?;
    let (s, nr, used) = reduce(y, n_persons, n_items);
    if used == 0 {
        return Err("no persons with an informative raw score (all scored 0 or k)".into());
    }
    fit_from_stats(&s, &nr, used, max_iter, tol)
}

/// One item's group-split difficulties and the Andersen LR test.
pub struct AndersenLr {
    /// Conditional LR statistic `2[sum_g llc_g(beta_g) - llc(beta_pooled)]`.
    pub lr: f64,
    /// Degrees of freedom `(G - 1)(k - 1)`.
    pub df: usize,
    /// Upper-tail `p`-value of `lr` under `chi^2(df)`.
    pub p_value: f64,
    /// Per-group retained-person counts.
    pub n_used: Vec<usize>,
    /// `true` only if the pooled AND every per-group CML fit converged. When `false` the `lr`/`p_value`
    /// are untrustworthy: a stalled group fit can drive the pre-clamp statistic negative (it is clamped
    /// to `0`), so do not interpret a non-converged result as a clean non-rejection.
    pub converged: bool,
}

/// Andersen's (1973) conditional likelihood-ratio test of Rasch fit across a person partition.
///
/// `group` is length `n_persons` with labels `0..n_groups`. Fits CML within each group and over the
/// pooled sample; `LR = 2[sum_g llc_g(beta_hat_g) - llc(beta_hat_pooled)]` is referred to
/// `chi^2((n_groups - 1)(n_items - 1))`. A significant `LR` rejects invariance of the item difficulties
/// across the split (Rasch misfit).
pub fn andersen_lr_test(
    y: &[u8],
    group: &[u8],
    n_groups: usize,
    n_persons: usize,
    n_items: usize,
    max_iter: usize,
    tol: f64,
) -> Result<AndersenLr, String> {
    validate(y, n_persons, n_items, max_iter, tol)?;
    if group.len() != n_persons {
        return Err(format!(
            "group has {} entries; expected {n_persons}",
            group.len()
        ));
    }
    if n_groups < 2 {
        return Err("the Andersen LR test needs at least 2 groups".into());
    }
    if group.iter().any(|&g| g as usize >= n_groups) {
        return Err("group labels must be in 0..n_groups".into());
    }
    // pooled fit
    let pooled = fit_rasch_cml(y, n_persons, n_items, max_iter, tol)?;
    let mut all_converged = pooled.converged;
    // per-group fits + their conditional loglik at the pooled beta
    let mut ll_groups = 0.0f64;
    let mut ll_pooled_on_groups = 0.0f64;
    let mut n_used = vec![0usize; n_groups];
    for gg in 0..n_groups {
        let rows: Vec<u8> = (0..n_persons)
            .filter(|&p| group[p] as usize == gg)
            .flat_map(|p| y[p * n_items..(p + 1) * n_items].iter().copied())
            .collect();
        let ng = rows.len() / n_items;
        if ng == 0 {
            return Err(format!("group {gg} has no persons"));
        }
        let (sg, nrg, usedg) = reduce(&rows, ng, n_items);
        if usedg == 0 {
            return Err(format!(
                "group {gg} has no informative persons (all scored 0 or {n_items})"
            ));
        }
        n_used[gg] = usedg;
        let fit_g = fit_from_stats(&sg, &nrg, usedg, max_iter, tol)?;
        all_converged &= fit_g.converged;
        ll_groups += fit_g.loglik;
        // pooled beta evaluated on group g's sufficient statistics
        ll_pooled_on_groups += cml_eval(&pooled.beta, &sg, &nrg).0;
    }
    // Each group term llc_g(beta_g) - llc_g(beta_pooled) is >= 0 only when beta_g maximizes llc_g; a
    // stalled fit can make it negative, so clamp rounding noise but flag non-convergence rather than
    // silently reporting a clamped lr = 0 as a clean non-rejection.
    let lr = (2.0 * (ll_groups - ll_pooled_on_groups)).max(0.0);
    let df = (n_groups - 1) * (n_items - 1);
    Ok(AndersenLr {
        lr,
        df,
        p_value: chi2_sf(lr, df as f64),
        n_used,
        converged: all_converged,
    })
}

fn validate(y: &[u8], n_persons: usize, n_items: usize, max_iter: usize, tol: f64) -> Result<(), String> {
    if n_persons < 1 || n_items < 2 {
        return Err("need n_persons >= 1 and n_items >= 2".into());
    }
    if n_items > CML_MAX_ITEMS {
        return Err(format!("n_items {n_items} exceeds the cap {CML_MAX_ITEMS}"));
    }
    let cells = n_persons
        .checked_mul(n_items)
        .ok_or("n_persons * n_items overflow")?;
    if y.len() != cells {
        return Err(format!("y has {} entries; expected {cells}", y.len()));
    }
    if y.iter().any(|&v| v > 1) {
        return Err("responses must be 0 or 1".into());
    }
    if max_iter == 0 {
        return Err("max_iter must be >= 1".into());
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err("tol must be finite and positive".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Brute-force elementary symmetric function of order `r` (sum over all size-`r` subsets).
    fn esf_brute(eps: &[f64], r: usize) -> f64 {
        let k = eps.len();
        let mut total = 0.0;
        for mask in 0u64..(1u64 << k) {
            if (mask.count_ones() as usize) == r {
                let mut prod = 1.0;
                for i in 0..k {
                    if mask & (1 << i) != 0 {
                        prod *= eps[i];
                    }
                }
                total += prod;
            }
        }
        total
    }

    /// The summation-algorithm ESF (and the leave-one-out / leave-two-out passes) match the brute-force
    /// subset sums exactly.
    #[test]
    fn esf_matches_brute_force() {
        let eps = [0.4, 1.1, 2.3, 0.7, 1.6];
        let k = eps.len();
        let g = esf(&eps);
        for r in 0..=k {
            assert!((g[r] - esf_brute(&eps, r)).abs() < 1e-10, "gamma_{r}");
        }
        // leave-one-out
        for omit in 0..k {
            let gi = esf_omit(&eps, omit);
            let sub: Vec<f64> = (0..k).filter(|&j| j != omit).map(|j| eps[j]).collect();
            for r in 0..k {
                assert!((gi[r] - esf_brute(&sub, r)).abs() < 1e-10, "gamma^({omit})_{r}");
            }
        }
        // leave-two-out
        let gij = esf_omit2(&eps, 1, 3);
        let sub: Vec<f64> = (0..k).filter(|&j| j != 1 && j != 3).map(|j| eps[j]).collect();
        for r in 0..k - 1 {
            assert!((gij[r] - esf_brute(&sub, r)).abs() < 1e-10, "gamma^(1,3)_{r}");
        }
    }

    /// Deterministic anchor: the analytic CML gradient and Hessian match finite differences of the
    /// conditional log-likelihood (pins the sign of `d eps/d beta = -eps` and the ESF derivative
    /// recursions — a sign error would flip the whole Newton direction).
    #[test]
    fn cml_gradient_hessian_match_finite_difference() {
        let beta = [-0.8, 0.3, 1.1, -0.2, 0.6];
        let k = beta.len();
        let s = [40.0, 55.0, 62.0, 48.0, 58.0];
        let nr = [0.0, 20.0, 30.0, 25.0, 15.0, 0.0]; // r = 0..=5, r=0,5 uninformative
        let (_ll, grad, hess) = cml_eval(&beta, &s, &nr);
        let eps = 1e-6;
        for i in 0..k {
            let mut bp = beta;
            bp[i] += eps;
            let mut bm = beta;
            bm[i] -= eps;
            let fd = (cml_eval(&bp, &s, &nr).0 - cml_eval(&bm, &s, &nr).0) / (2.0 * eps);
            assert!((grad[i] - fd).abs() < 1e-4, "grad[{i}] {} vs FD {fd}", grad[i]);
        }
        let hh = 1e-4;
        for a in 0..k {
            for b in 0..k {
                let mut pp = beta;
                pp[a] += hh;
                pp[b] += hh;
                let mut pm = beta;
                pm[a] += hh;
                pm[b] -= hh;
                let mut mp = beta;
                mp[a] -= hh;
                mp[b] += hh;
                let mut mm = beta;
                mm[a] -= hh;
                mm[b] -= hh;
                let d2 = (cml_eval(&pp, &s, &nr).0 - cml_eval(&pm, &s, &nr).0 - cml_eval(&mp, &s, &nr).0
                    + cml_eval(&mm, &s, &nr).0)
                    / (4.0 * hh * hh);
                assert!(
                    (hess[a * k + b] - d2).abs() < 1e-2,
                    "hess[{a}][{b}] {} vs FD {d2}",
                    hess[a * k + b]
                );
            }
        }
    }

    fn simulate(beta: &[f64], theta: &[f64], rng: &mut Lcg) -> Vec<u8> {
        let k = beta.len();
        let n = theta.len();
        let mut y = vec![0u8; n * k];
        for p in 0..n {
            for i in 0..k {
                let pr = 1.0 / (1.0 + (-(theta[p] - beta[i])).exp());
                y[p * k + i] = if rng.next_f64() < pr { 1 } else { 0 };
            }
        }
        y
    }

    fn rmse(a: &[f64], b: &[f64]) -> f64 {
        (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
    }

    /// THE DEFINING CML PROPERTY (person-distribution-free): the same beta_hat (up to the sum-zero
    /// constant) is recovered whether the simulating theta is N(0,1) or strongly right-skewed. A plain
    /// value-recovery test is INSUFFICIENT — JML also recovers beta at large k — so the discriminating
    /// assertion is the AGREEMENT between the two distributions' estimates, not merely closeness to
    /// truth.
    #[test]
    fn cml_is_person_distribution_free() {
        let mut beta = vec![-1.6, -0.9, -0.3, 0.2, 0.7, 1.2, 1.7, 0.0];
        center(&mut beta);
        let k = beta.len();
        let n = 4000usize;
        let mut rng = Lcg(918273);
        // (a) theta ~ N(0,1)
        let th_norm: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        // (b) theta strongly right-skew (standardized Exp - shifted), a very different distribution
        let th_skew: Vec<f64> = (0..n).map(|_| 1.5 * (-(rng.next_f64().max(1e-12)).ln()) - 1.0).collect();
        let ya = simulate(&beta, &th_norm, &mut rng);
        let yb = simulate(&beta, &th_skew, &mut rng);
        let fa = fit_rasch_cml(&ya, n, k, 100, 1e-9).unwrap();
        let fb = fit_rasch_cml(&yb, n, k, 100, 1e-9).unwrap();
        assert!(fa.converged && fb.converged);
        // both recover the truth within MC tolerance
        assert!(rmse(&fa.beta, &beta) < 0.15, "N(0,1) beta RMSE {}", rmse(&fa.beta, &beta));
        assert!(rmse(&fb.beta, &beta) < 0.15, "skew beta RMSE {}", rmse(&fb.beta, &beta));
        // and — the CML signature — the two estimates AGREE despite the very different ability
        // distributions (a distribution-DEPENDENT estimator would diverge here)
        assert!(
            rmse(&fa.beta, &fb.beta) < 0.15,
            "distribution-free property violated: N(0,1) vs skew beta RMSE {}",
            rmse(&fa.beta, &fb.beta)
        );
        // SEs finite and positive on-support
        assert!(fa.se.iter().all(|s| s.is_finite() && *s > 0.0));
    }

    /// Andersen (1973) LR: on Rasch-generated data an arbitrary (ability-independent) group split does
    /// NOT reject (statistic near its df), while data with a group-specific difficulty shift (Rasch
    /// misfit / DIF) is rejected with a large statistic. Pins the df and the upper-tail direction.
    #[test]
    fn andersen_lr_detects_group_difficulty_shift() {
        let mut beta = vec![-1.2, -0.6, 0.0, 0.6, 1.2, -0.3, 0.3, 0.9];
        center(&mut beta);
        let k = beta.len();
        let n = 3000usize;
        let mut rng = Lcg(0xA9D5);
        let theta: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let group: Vec<u8> = (0..n).map(|p| (p % 2) as u8).collect();
        // (1) true Rasch, split by an ARBITRARY label (independent of ability): should NOT reject
        let y_fit = simulate(&beta, &theta, &mut rng);
        let t1 = andersen_lr_test(&y_fit, &group, 2, n, k, 100, 1e-9).unwrap();
        assert_eq!(t1.df, (2 - 1) * (k - 1));
        assert!(t1.lr / (t1.df as f64) < 3.0, "Rasch data over-rejected: LR {} df {}", t1.lr, t1.df);
        assert!(t1.p_value > 0.01, "Rasch data p too small: {}", t1.p_value);
        // (2) group 1 gets a difficulty shift on item 0 (violates Rasch invariance): should reject
        let mut y_dif = vec![0u8; n * k];
        for p in 0..n {
            for i in 0..k {
                let mut bi = beta[i];
                if i == 0 && group[p] == 1 {
                    bi += 1.5;
                }
                let pr = 1.0 / (1.0 + (-(theta[p] - bi)).exp());
                y_dif[p * k + i] = if rng.next_f64() < pr { 1 } else { 0 };
            }
        }
        let t2 = andersen_lr_test(&y_dif, &group, 2, n, k, 100, 1e-9).unwrap();
        assert!(t2.lr > t1.lr + 15.0, "DIF not detected: LR {} vs baseline {}", t2.lr, t1.lr);
        assert!(t2.p_value < 0.01, "DIF p not significant: {}", t2.p_value);
        assert!(t1.converged && t2.converged, "converged flag not set on a converging fit");
        // a starved max_iter surfaces non-convergence rather than a silently clamped lr=0
        let t_bad = andersen_lr_test(&y_dif, &group, 2, n, k, 1, 1e-9).unwrap();
        assert!(!t_bad.converged, "non-convergence must be surfaced, not masked");
    }

    /// Validation guards.
    #[test]
    fn cml_validates() {
        let y = vec![0u8, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1]; // 3 persons x 4 items
        assert!(fit_rasch_cml(&y, 3, 4, 100, 1e-9).is_ok());
        assert!(fit_rasch_cml(&y, 3, 4, 0, 1e-9).is_err()); // max_iter 0
        let mut ybad = y.clone();
        ybad[0] = 2;
        assert!(fit_rasch_cml(&ybad, 3, 4, 100, 1e-9).is_err()); // non-binary
        assert!(fit_rasch_cml(&y, 3, 1, 100, 1e-9).is_err()); // n_items < 2 (length also wrong)
        // all-perfect / all-zero -> no informative persons
        let yflat = vec![1u8; 3 * 4];
        assert!(fit_rasch_cml(&yflat, 3, 4, 100, 1e-9).is_err());
    }
}
