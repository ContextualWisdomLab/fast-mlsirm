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

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
}

/// Smoke test: unidimensional 2PL recovery. MH-RM at `D = 1` should recover the loadings and
/// intercepts within Monte-Carlo tolerance (a fixed-seed anchor, NOT exact equality).
#[test]
fn mhrm_recovers_unidimensional_2pl() {
    let (n, n_items) = (1500usize, 12usize);
    let pattern = vec![1u8; n_items]; // D = 1, every item pure
    let mut rng = Lcg(20100507);
    let true_a: Vec<f64> = (0..n_items).map(|i| 0.8 + 0.1 * (i % 5) as f64).collect();
    let true_b: Vec<f64> = (0..n_items).map(|i| -0.8 + 0.15 * i as f64).collect();
    let mut theta = vec![0.0f64; n];
    for v in theta.iter_mut() {
        *v = rng.normal();
    }
    let mut y = vec![0usize; n * n_items];
    for p in 0..n {
        for i in 0..n_items {
            let base = true_a[i] * theta[p] + true_b[i];
            let prob = 1.0 / (1.0 + (-base).exp());
            y[p * n_items + i] = if rng.next_f64() < prob { 1 } else { 0 };
        }
    }
    let cfg = MhrmConfig {
        max_cycles: 1200,
        burn_in: 150,
        mh_steps: 8,
        seed: 424242,
        ..MhrmConfig::default()
    };
    let res = fit_mhrm(&y, None, &pattern, n, n_items, 1, &cfg).unwrap();
    assert_eq!(res.n_dims, 1);
    assert_eq!(res.loading.len(), n_items);
    assert_eq!(res.n_parameters, n_items + n_items);
    // reflection canonical: largest pure anchor positive
    assert!(res.loading.iter().cloned().fold(f64::MIN, f64::max) > 0.0);
    // acceptance in a sane band after tuning
    assert!(
        res.acceptance_rate > 0.1 && res.acceptance_rate < 0.7,
        "acceptance {}",
        res.acceptance_rate
    );
    // recover loadings and intercepts within MC tolerance
    assert!(
        rmse(&res.loading, &true_a) < 0.2,
        "loading RMSE {} loadings {:?}",
        rmse(&res.loading, &true_a),
        res.loading
    );
    assert!(
        rmse(&res.intercept, &true_b) < 0.2,
        "intercept RMSE {}",
        rmse(&res.intercept, &true_b)
    );
    // trait EAP correlates with the truth
    let th: Vec<f64> = (0..n).map(|p| res.theta[p]).collect();
    let mt = th.iter().sum::<f64>() / n as f64;
    let mtt = theta.iter().sum::<f64>() / n as f64;
    let cov: f64 = (0..n).map(|p| (th[p] - mt) * (theta[p] - mtt)).sum();
    let vt: f64 = th.iter().map(|x| (x - mt).powi(2)).sum();
    let vtt: f64 = theta.iter().map(|x| (x - mtt).powi(2)).sum();
    assert!(
        cov / (vt * vtt).sqrt() > 0.8,
        "theta corr {}",
        cov / (vt * vtt).sqrt()
    );
    // Louis SEs finite and positive
    assert!(res.se_loading.iter().all(|s| s.is_finite() && *s > 0.0));
}

fn corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let (da, db) = (a[i] - ma, b[i] - mb);
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    sab / (saa * sbb).sqrt()
}

fn item_loglik(
    params: &[f64],
    dims: &[usize],
    theta: &[f64],
    y: &[usize],
    np: usize,
    nd: usize,
) -> f64 {
    let li = dims.len();
    let mut ll = 0.0;
    for p in 0..np {
        let mut base = params[li];
        for (t, &d) in dims.iter().enumerate() {
            base += params[t] * theta[p * nd + d];
        }
        let pp = 1.0 / (1.0 + (-base).exp());
        ll += if y[p] == 1 { pp.ln() } else { (1.0 - pp).ln() };
    }
    ll
}

/// Deterministic anchor: the per-item score and information returned by `item_score_info` are
/// pinned against finite differences of the complete-data logistic log-likelihood, on ONE D=2
/// CROSS-loader item with ASYMMETRIC params (a NEGATIVE loading) at fixed asymmetric traits. A
/// sign flip in the residual, a transposed information layout, or a dropped dims-map entry all
/// fail here — none of which a centered/symmetric value-recovery test would catch.
#[test]
fn mhrm_score_and_info_match_finite_difference() {
    let nd = 2usize;
    let dims = vec![0usize, 1usize];
    let params = vec![0.8f64, -0.5, 0.3]; // [a0, a1, b] — a1 negative
    let theta = vec![0.5, -1.0, -0.7, 0.4, 1.2, 0.9]; // 3 persons x 2 dims (asymmetric)
    let y = vec![1usize, 0, 1];
    let np = 3usize;
    let pi = 3usize;
    let (s, h, hobs) = item_score_info(
        MhrmModel::TwoPl,
        &params,
        &dims,
        &theta,
        &y,
        None,
        0,
        np,
        1,
        nd,
    );
    // score[t] = d loglik / d params[t]
    let eps = 1e-6;
    for t in 0..pi {
        let mut pp = params.clone();
        pp[t] += eps;
        let mut pm = params.clone();
        pm[t] -= eps;
        let fd = (item_loglik(&pp, &dims, &theta, &y, np, nd)
            - item_loglik(&pm, &dims, &theta, &y, np, nd))
            / (2.0 * eps);
        assert!((s[t] - fd).abs() < 1e-4, "score[{t}] {} vs FD {}", s[t], fd);
    }
    // info[a][b] = -d^2 loglik / d params[a] d params[b] = sum_p w_p x_a x_b (symmetric, PD)
    let hh = 1e-3;
    for a in 0..pi {
        for b in 0..pi {
            let mut fpp = params.clone();
            fpp[a] += hh;
            fpp[b] += hh;
            let mut fpm = params.clone();
            fpm[a] += hh;
            fpm[b] -= hh;
            let mut fmp = params.clone();
            fmp[a] -= hh;
            fmp[b] += hh;
            let mut fmm = params.clone();
            fmm[a] -= hh;
            fmm[b] -= hh;
            let d2 = (item_loglik(&fpp, &dims, &theta, &y, np, nd)
                - item_loglik(&fpm, &dims, &theta, &y, np, nd)
                - item_loglik(&fmp, &dims, &theta, &y, np, nd)
                + item_loglik(&fmm, &dims, &theta, &y, np, nd))
                / (4.0 * hh * hh);
            assert!(
                (h[a * pi + b] - (-d2)).abs() < 1e-2,
                "info[{a}][{b}] {} vs -FDhess {}",
                h[a * pi + b],
                -d2
            );
            assert!(
                (h[a * pi + b] - h[b * pi + a]).abs() < 1e-12,
                "info symmetric"
            );
        }
    }
    // non-trivial layout: the cross term is genuinely nonzero (asymmetric traits)
    assert!(h[1].abs() > 0.05, "off-diag info nonzero: {}", h[1]);
    // Louis missing-information term: hobs = sum_p (w_p - r_p^2) X X' = H - sum_p r_p^2 X X'.
    // Pin the SIGN of the r^2 subtraction (the mutant `w + r^2` inverts it) by an INDEPENDENT
    // re-sum of the per-person score outer product r_p^2 X_p X_p'.
    let mut r2_outer = vec![0.0f64; pi * pi];
    for p in 0..np {
        let mut base = params[dims.len()];
        for (t, &d) in dims.iter().enumerate() {
            base += params[t] * theta[p * nd + d];
        }
        let pp = 1.0 / (1.0 + (-base).exp());
        let r2 = (y[p] as f64 - pp).powi(2);
        let x = [theta[p * nd], theta[p * nd + 1], 1.0];
        for a in 0..pi {
            for b in 0..pi {
                r2_outer[a * pi + b] += r2 * x[a] * x[b];
            }
        }
    }
    for idx in 0..pi * pi {
        assert!(
            (hobs[idx] - (h[idx] - r2_outer[idx])).abs() < 1e-9,
            "louis missing-info sign: hobs[{idx}] {} vs H-r2 {}",
            hobs[idx],
            h[idx] - r2_outer[idx]
        );
    }
}

/// White-box anchor on the Robbins-Monro gain schedule: constant `burn_in_gain` through burn-in,
/// then `1/(k - burn_in)^alpha` (an off-by-one at the boundary is a classic bug the recovery
/// tests would not localize).
#[test]
fn mhrm_gain_schedule() {
    let (b, g0) = (10usize, 0.8f64);
    assert_eq!(gain_at(1, b, g0, 1.0), g0);
    assert_eq!(gain_at(b, b, g0, 1.0), g0); // last burn-in cycle is still constant gain
    assert_eq!(gain_at(b + 1, b, g0, 1.0), 1.0); // first convergence-stage cycle: 1/1
    assert_eq!(gain_at(b + 4, b, g0, 1.0), 0.25); // 1/4
    assert!((gain_at(b + 4, b, g0, 0.5) - 0.5).abs() < 1e-12); // 1/4^0.5 = 0.5
}

/// Reduction anchor: at `D = 1`, MH-RM agrees with the established deterministic unidimensional
/// MMLE (`mmle::fit_mmle_2pl`) within Monte-Carlo tolerance (NOT bit-exact — MH-RM is stochastic).
#[test]
fn mhrm_reduces_to_mmle_2pl_at_d1() {
    use crate::mmle::{fit_mmle_2pl, MmleConfig};
    let (n, n_items) = (1200usize, 10usize);
    let pattern = vec![1u8; n_items];
    let mut rng = Lcg(77);
    let a_t: Vec<f64> = (0..n_items).map(|i| 0.9 + 0.08 * (i % 4) as f64).collect();
    let b_t: Vec<f64> = (0..n_items).map(|i| -0.6 + 0.13 * i as f64).collect();
    let mut th = vec![0.0f64; n];
    for v in th.iter_mut() {
        *v = rng.normal();
    }
    let mut y = vec![0usize; n * n_items];
    for p in 0..n {
        for i in 0..n_items {
            let pr = 1.0 / (1.0 + (-(a_t[i] * th[p] + b_t[i])).exp());
            y[p * n_items + i] = if rng.next_f64() < pr { 1 } else { 0 };
        }
    }
    let cfg = MhrmConfig {
        max_cycles: 1200,
        burn_in: 150,
        mh_steps: 8,
        seed: 9,
        ..MhrmConfig::default()
    };
    let res = fit_mhrm(&y, None, &pattern, n, n_items, 1, &cfg).unwrap();
    let yf: Vec<f64> = y.iter().map(|&v| v as f64).collect();
    let obs = vec![true; n * n_items];
    let m = fit_mmle_2pl(&yf, &obs, n, n_items, &MmleConfig::default());
    assert!(
        rmse(&res.loading, &m.a) < 0.12,
        "MH-RM vs MMLE loading RMSE {}",
        rmse(&res.loading, &m.a)
    );
    assert!(
        rmse(&res.intercept, &m.b) < 0.12,
        "MH-RM vs MMLE intercept RMSE {}",
        rmse(&res.intercept, &m.b)
    );
}

/// Headline capability: `D = 6` confirmatory 2PL. The `q^D` Gauss-Hermite grid (`21^6 ~ 8.6e7`)
/// and even the QMC E-step are infeasible at this dimensionality; MH-RM's stochastic imputation
/// is `D`-agnostic. Simple structure (3 pure anchors per dimension) plus two cross-loaders, one
/// genuinely NEGATIVE — recovered with the correct sign.
#[test]
fn mhrm_recovers_high_dim_d6() {
    let (n_dims, n) = (6usize, 2500usize);
    let n_items = 20usize;
    let mut pattern = vec![0u8; n_items * n_dims];
    for i in 0..18 {
        pattern[i * n_dims + i / 3] = 1; // items 0..17: 3 pure anchors per dimension
    }
    pattern[18 * n_dims] = 1;
    pattern[18 * n_dims + 3] = 1; // item18 cross-loads dims 0 and 3
    pattern[19 * n_dims + 1] = 1;
    pattern[19 * n_dims + 4] = 1; // item19 cross-loads dims 1 and 4
    let mut a_t = vec![0.0f64; n_items * n_dims];
    for i in 0..18 {
        a_t[i * n_dims + i / 3] = 0.9 + 0.1 * (i % 3) as f64;
    }
    a_t[18 * n_dims] = 1.0;
    a_t[18 * n_dims + 3] = -0.7; // NEGATIVE cross-loader
    a_t[19 * n_dims + 1] = 0.8;
    a_t[19 * n_dims + 4] = 0.6;
    let b_t: Vec<f64> = (0..n_items).map(|i| -0.5 + 0.1 * (i % 7) as f64).collect();
    let mut rng = Lcg(60606);
    let mut th = vec![0.0f64; n * n_dims];
    for v in th.iter_mut() {
        *v = rng.normal();
    }
    let mut y = vec![0usize; n * n_items];
    for p in 0..n {
        for i in 0..n_items {
            let mut base = b_t[i];
            for d in 0..n_dims {
                base += a_t[i * n_dims + d] * th[p * n_dims + d];
            }
            let pr = 1.0 / (1.0 + (-base).exp());
            y[p * n_items + i] = if rng.next_f64() < pr { 1 } else { 0 };
        }
    }
    let cfg = MhrmConfig {
        max_cycles: 1000,
        burn_in: 200,
        mh_steps: 6,
        seed: 13,
        ..MhrmConfig::default()
    };
    let res = fit_mhrm(&y, None, &pattern, n, n_items, n_dims, &cfg).unwrap();
    assert_eq!(res.n_dims, 6);
    for i in 0..n_items {
        for d in 0..n_dims {
            if pattern[i * n_dims + d] == 0 {
                assert_eq!(res.loading[i * n_dims + d], 0.0);
            }
        }
    }
    let (mut se2, mut cnt) = (0.0, 0usize);
    for idx in 0..n_items * n_dims {
        if pattern[idx] == 1 {
            se2 += (res.loading[idx] - a_t[idx]).powi(2);
            cnt += 1;
        }
    }
    let load_rmse = (se2 / cnt as f64).sqrt();
    assert!(load_rmse < 0.22, "D=6 on-pattern loading RMSE {load_rmse}");
    assert!(
        res.loading[18 * n_dims + 3] < -0.3,
        "negative cross-loader {}",
        res.loading[18 * n_dims + 3]
    );
    for d in 0..n_dims {
        let est: Vec<f64> = (0..n).map(|p| res.theta[p * n_dims + d]).collect();
        let tru: Vec<f64> = (0..n).map(|p| th[p * n_dims + d]).collect();
        assert!(
            corr(&est, &tru) > 0.5,
            "dim {d} theta corr {}",
            corr(&est, &tru)
        );
    }
}

/// The reflection canonicalization FIRES and is WITNESSED. dim0 has a WEAK reverse-keyed SOLE
/// pure anchor (item0, true `-0.7`) and a STRONG positively-keyed cross-loader (item1, dim0
/// `+1.7`) that dominates the axis orientation, so raw MH-RM lands the anchor NEGATIVE and
/// canonicalization must flip dim0: the anchor ends positive, the co-loader negative, and theta_0
/// correlates NEGATIVELY with the truth. Disabling the flip (in-loop + final) fails all three.
#[test]
fn mhrm_reflection_fires_on_negative_anchor() {
    let (n_dims, n) = (2usize, 5000usize);
    let n_items = 4usize;
    // item0 pure d0 (sole d0 anchor), item1 cross d0/d1, item2/3 pure d1
    let pattern = vec![1u8, 0, 1, 1, 0, 1, 0, 1];
    let mut a_t = vec![0.0f64; n_items * n_dims];
    a_t[0] = -0.7; // weak reverse-keyed pure d0 anchor
    a_t[1 * n_dims] = 1.7; // strong positive cross-loader on d0 (sets the axis)
    a_t[1 * n_dims + 1] = 0.6;
    a_t[2 * n_dims + 1] = 1.2;
    a_t[3 * n_dims + 1] = 1.0;
    let b_t = vec![0.2f64, -0.1, 0.3, -0.2];
    let mut rng = Lcg(1357);
    let mut th = vec![0.0f64; n * n_dims];
    for v in th.iter_mut() {
        *v = rng.normal();
    }
    let mut y = vec![0usize; n * n_items];
    for p in 0..n {
        for i in 0..n_items {
            let mut base = b_t[i];
            for d in 0..n_dims {
                base += a_t[i * n_dims + d] * th[p * n_dims + d];
            }
            let pr = 1.0 / (1.0 + (-base).exp());
            y[p * n_items + i] = if rng.next_f64() < pr { 1 } else { 0 };
        }
    }
    let cfg = MhrmConfig {
        max_cycles: 1000,
        burn_in: 200,
        mh_steps: 8,
        seed: 24,
        ..MhrmConfig::default()
    };
    let res = fit_mhrm(&y, None, &pattern, n, n_items, n_dims, &cfg).unwrap();
    assert!(
        res.loading[0] > 0.3,
        "reflected anchor positive: {}",
        res.loading[0]
    );
    assert!(
        res.loading[1 * n_dims] < -0.5,
        "co-loader flipped negative: {}",
        res.loading[1 * n_dims]
    );
    let th0: Vec<f64> = (0..n).map(|p| res.theta[p * n_dims]).collect();
    let tt0: Vec<f64> = (0..n).map(|p| th[p * n_dims]).collect();
    let th1: Vec<f64> = (0..n).map(|p| res.theta[p * n_dims + 1]).collect();
    let tt1: Vec<f64> = (0..n).map(|p| th[p * n_dims + 1]).collect();
    assert!(
        corr(&th0, &tt0) < -0.4,
        "flipped-dim theta corr negative: {}",
        corr(&th0, &tt0)
    );
    assert!(
        corr(&th1, &tt1) > 0.4,
        "unflipped-dim theta corr positive: {}",
        corr(&th1, &tt1)
    );
}

/// Correlated-Sigma MH-RM (Cai, 2010b): with `estimate_corr` the free latent correlation matrix
/// `Phi` is recovered from `theta ~ MVN(0, Phi)`. Covers a POSITIVE, a near-PD-boundary (D=3,
/// rho=0.5), and a NEGATIVE correlation (sign correctness); confirms `Phi` stays a valid PD
/// correlation matrix (unit diagonal) and `n_parameters` counts the `D(D-1)/2` correlations.
#[test]
fn mhrm_correlated_recovers_known_phi() {
    for &(n_dims, rho, n) in &[
        (2usize, 0.4f64, 3000usize),
        (3usize, 0.5f64, 3500usize),
        (2usize, -0.5f64, 3000usize),
    ] {
        // exchangeable Phi
        let mut phi = vec![rho; n_dims * n_dims];
        for a in 0..n_dims {
            phi[a * n_dims + a] = 1.0;
        }
        let l = chol_lower(&phi, n_dims).expect("Phi PD");
        let per = 4usize;
        let n_items = per * n_dims;
        let mut pattern = vec![0u8; n_items * n_dims];
        let mut a_t = vec![0.0f64; n_items * n_dims];
        for d in 0..n_dims {
            for a in 0..per {
                let i = d * per + a;
                pattern[i * n_dims + d] = 1;
                a_t[i * n_dims + d] = 1.0 + 0.1 * a as f64;
            }
        }
        let b_t: Vec<f64> = (0..n_items).map(|i| -0.4 + 0.1 * (i % 5) as f64).collect();
        let mut rng = Lcg(0x00C0FFEE ^ ((n_dims as u64) << 8) ^ ((rho < 0.0) as u64));
        // theta_p = L z_p ~ MVN(0, Phi)
        let mut th = vec![0.0f64; n * n_dims];
        for p in 0..n {
            let z: Vec<f64> = (0..n_dims).map(|_| rng.normal()).collect();
            for a in 0..n_dims {
                let mut v = 0.0;
                for b in 0..=a {
                    v += l[a * n_dims + b] * z[b];
                }
                th[p * n_dims + a] = v;
            }
        }
        let mut y = vec![0usize; n * n_items];
        for p in 0..n {
            for i in 0..n_items {
                let mut base = b_t[i];
                for d in 0..n_dims {
                    base += a_t[i * n_dims + d] * th[p * n_dims + d];
                }
                let pr = 1.0 / (1.0 + (-base).exp());
                y[p * n_items + i] = if rng.next_f64() < pr { 1 } else { 0 };
            }
        }
        let cfg = MhrmConfig {
            max_cycles: 1600,
            burn_in: 350,
            mh_steps: 8,
            estimate_corr: true,
            seed: 42,
            ..MhrmConfig::default()
        };
        let res = fit_mhrm(&y, None, &pattern, n, n_items, n_dims, &cfg).unwrap();
        assert_eq!(res.corr.len(), n_dims * n_dims);
        // valid correlation matrix: unit diagonal, symmetric, PD
        for a in 0..n_dims {
            assert!(
                (res.corr[a * n_dims + a] - 1.0).abs() < 1e-9,
                "unit diagonal"
            );
            for b in 0..n_dims {
                assert!((res.corr[a * n_dims + b] - res.corr[b * n_dims + a]).abs() < 1e-12);
            }
        }
        assert!(chol_lower(&res.corr, n_dims).is_some(), "recovered Phi PD");
        // recover the off-diagonals (sign + magnitude) within MC tolerance
        for a in 0..n_dims {
            for b in a + 1..n_dims {
                let est = res.corr[a * n_dims + b];
                assert!(
                    (est - rho).abs() < 0.12,
                    "D={n_dims} rho={rho} corr[{a}][{b}]={est}"
                );
            }
        }
        assert_eq!(
            res.n_parameters,
            n_items + n_items + n_dims * (n_dims - 1) / 2
        );
    }
}

/// Validation guards constructed non-vacuously (each input trips the INTENDED guard, not an
/// earlier one).
#[test]
fn mhrm_validates_and_structural_invariants() {
    let (n, n_items, n_dims) = (60usize, 4usize, 2usize);
    let pattern = vec![1u8, 0, 1, 0, 0, 1, 0, 1]; // pure anchors on both dims
    let mut y = vec![0usize; n * n_items];
    for p in 0..n {
        for i in 0..n_items {
            y[p * n_items + i] = (p + i) % 2; // non-degenerate mixed responses
        }
    }
    let short = MhrmConfig {
        max_cycles: 30,
        burn_in: 5,
        ..MhrmConfig::default()
    };
    let res = fit_mhrm(&y, None, &pattern, n, n_items, n_dims, &short).unwrap();
    assert_eq!(res.n_parameters, 4 + 4); // 4 loadings + 4 intercepts (no correlations)
    assert_eq!(res.se_loading.len(), n_items * n_dims);
    // estimate_corr=false -> Phi is EXACTLY the identity (orthogonal factors)
    assert_eq!(res.corr, vec![1.0, 0.0, 0.0, 1.0]);
    let mut mask = vec![true; n * n_items];
    mask[0] = false;
    assert!(fit_mhrm(&y, Some(&mask), &pattern, n, n_items, n_dims, &short).is_ok());
    let without_se = fit_mhrm(
        &y,
        None,
        &pattern,
        n,
        n_items,
        n_dims,
        &MhrmConfig {
            estimate_se: false,
            ..short
        },
    )
    .unwrap();
    assert!(without_se.se_loading.is_empty());
    assert!(without_se.se_intercept.is_empty());
    // no pure anchor on any dimension (every item loads both dims)
    let all_both = vec![1u8; n_items * n_dims];
    assert!(fit_mhrm(&y, None, &all_both, n, n_items, n_dims, &short).is_err());
    // non-binary response where observed
    let mut ybad = y.clone();
    ybad[0] = 2;
    assert!(fit_mhrm(&ybad, None, &pattern, n, n_items, n_dims, &short).is_err());
    // burn_in >= max_cycles
    let bad = MhrmConfig {
        max_cycles: 10,
        burn_in: 10,
        ..MhrmConfig::default()
    };
    assert!(fit_mhrm(&y, None, &pattern, n, n_items, n_dims, &bad).is_err());
    // gain_exponent out of (0.5, 1] Robbins-Monro band
    let badgain = MhrmConfig {
        gain_exponent: 0.3,
        ..short
    };
    assert!(fit_mhrm(&y, None, &pattern, n, n_items, n_dims, &badgain).is_err());
    // n_dims exceeds MHRM_MAX_DIMS (=64) — the n_dims guard is checked before pattern length
    let big_pat = vec![1u8; n_items * 65];
    assert!(fit_mhrm(&y, None, &big_pat, n, n_items, 65, &short).is_err());
    // y length mismatch (cells != y.len())
    let y_short = vec![0usize; n * n_items - 1];
    assert!(fit_mhrm(&y_short, None, &pattern, n, n_items, n_dims, &short).is_err());
    // loading_pattern entry other than 0/1 (correct length, so the >1 guard is the sole trip)
    let mut pat_bad = pattern.clone();
    pat_bad[0] = 2;
    assert!(fit_mhrm(&y, None, &pat_bad, n, n_items, n_dims, &short).is_err());

    assert!(validate(&[], None, &[], 0, 1, 1, &short).is_err());
    assert!(validate(&[], None, &[1], MHRM_MAX_CELLS + 1, 1, 1, &short,).is_err());
    assert!(validate(&y, Some(&[]), &pattern, n, n_items, n_dims, &short).is_err());
    assert!(validate(
        &y,
        None,
        &pattern[..pattern.len() - 1],
        n,
        n_items,
        n_dims,
        &short
    )
    .is_err());
    let mut zero_row = pattern.clone();
    zero_row[0] = 0;
    assert!(validate(&y, None, &zero_row, n, n_items, n_dims, &short).is_err());

    for invalid in [
        MhrmConfig {
            mh_steps: 0,
            ..short
        },
        MhrmConfig {
            proposal_sd: 0.0,
            ..short
        },
        MhrmConfig {
            target_accept: 2.0,
            ..short
        },
        MhrmConfig {
            burn_in_gain: 0.0,
            ..short
        },
        MhrmConfig { window: 0, ..short },
        MhrmConfig { tol: 0.0, ..short },
        MhrmConfig {
            ridge: 0.0,
            ..short
        },
    ] {
        assert!(validate(&y, None, &pattern, n, n_items, n_dims, &invalid).is_err());
    }

    let gpcm_y = [0usize, 0, 1, 1, 2, 2];
    let gpcm_pattern = [1u8, 1];
    let gpcm_observed = [false, true, false, true, false, true];
    let gpcm_cfg = MhrmConfig {
        max_cycles: 30,
        burn_in: 5,
        model: MhrmModel::Gpcm { n_cat: 3 },
        ..MhrmConfig::default()
    };
    assert!(validate(
        &gpcm_y,
        Some(&gpcm_observed),
        &gpcm_pattern,
        3,
        2,
        1,
        &gpcm_cfg,
    )
    .is_err());

    for model in [MhrmModel::TwoPl, MhrmModel::Gpcm { n_cat: 3 }] {
        let params = if model == MhrmModel::TwoPl {
            vec![1.0, 0.0]
        } else {
            vec![1.0, 0.0, 0.0]
        };
        let (score, information, observed_information) = item_score_info(
            model,
            &params,
            &[0],
            &[0.0],
            &[0],
            Some(&[false]),
            0,
            1,
            1,
            1,
        );
        assert!(score.iter().all(|&value| value == 0.0));
        assert!(information.iter().all(|&value| value == 0.0));
        assert!(observed_information.iter().all(|&value| value == 0.0));
    }
}

#[test]
fn mhrm_private_numeric_helpers_cover_reflection_backtracking_and_se_fallbacks() {
    let immediate = backtracked_corr_step(&[0.0], 1.0, &[0.1], 2);
    assert_eq!(immediate, vec![0.1]);
    let stepped = backtracked_corr_step(&[0.0, 0.0, 0.0], 1.0, &[1.0, 1.0, -1.0], 3);
    assert_eq!(stepped, vec![0.25, 0.25, -0.25]);
    let unchanged = backtracked_corr_step(&[0.0, 0.0, 0.0], 1.0, &[1e9, 1e9, -1e9], 3);
    assert_eq!(unchanged, vec![0.0; 3]);

    let dims_of = vec![vec![0], vec![0, 1]];
    let mut loading = vec![-2.0, 0.0, 1.0, 3.0];
    let mut theta = vec![1.0, 2.0, 3.0, 4.0];
    let mut offdiag = vec![0.4];
    canonicalize_final_dimension(
        0,
        2,
        2,
        2,
        &dims_of,
        &mut loading,
        &mut theta,
        &mut offdiag,
        true,
    );
    assert_eq!(loading, vec![2.0, 0.0, -1.0, 3.0]);
    assert_eq!(theta, vec![-1.0, 2.0, -3.0, 4.0]);
    assert_eq!(offdiag, vec![-0.4]);
    canonicalize_final_dimension(
        1,
        2,
        2,
        2,
        &dims_of,
        &mut loading,
        &mut theta,
        &mut offdiag,
        false,
    );
    assert_eq!(standard_error_from_variance(4.0), 2.0);
    assert!(standard_error_from_variance(-1.0).is_nan());
}

// ================================ GPCM MH-RM (Muraki, 1992) ================================

/// GPCM category probabilities at a scalar `base = sum_d a_d theta_d`: `psi_k = k*base + step_k`
/// (`step_0 = 0`), `P_k = softmax_k(psi)`.
fn gpcm_probs(base: f64, steps: &[f64], n_cat: usize) -> Vec<f64> {
    let mut psi = vec![0.0f64; n_cat];
    let mut m = f64::NEG_INFINITY;
    for k in 0..n_cat {
        psi[k] = (k as f64) * base + if k == 0 { 0.0 } else { steps[k - 1] };
        if psi[k] > m {
            m = psi[k];
        }
    }
    let mut z = 0.0;
    for p in psi.iter_mut() {
        *p = (*p - m).exp();
        z += *p;
    }
    for p in psi.iter_mut() {
        *p /= z;
    }
    psi
}

/// Inverse-CDF category draw from a probability vector and a uniform `u`.
fn gpcm_sample(probs: &[f64], u: f64) -> usize {
    let mut acc = 0.0;
    for (k, &p) in probs.iter().enumerate() {
        acc += p;
        if u < acc {
            return k;
        }
    }
    probs.len() - 1
}

/// Complete-data GPCM item log-likelihood at fixed traits (the FD target for the score/Hessian
/// anchor). `params = [a_d for d in dims, step_1..step_{K-1}]`.
fn gpcm_item_loglik(
    params: &[f64],
    dims: &[usize],
    theta: &[f64],
    y: &[usize],
    np: usize,
    nd: usize,
    n_cat: usize,
) -> f64 {
    let li = dims.len();
    let mut ll = 0.0;
    for p in 0..np {
        let mut base = 0.0;
        for (t, &d) in dims.iter().enumerate() {
            base += params[t] * theta[p * nd + d];
        }
        let mut m = f64::NEG_INFINITY;
        let mut psi = vec![0.0f64; n_cat];
        for k in 0..n_cat {
            psi[k] = (k as f64) * base + if k == 0 { 0.0 } else { params[li + k - 1] };
            if psi[k] > m {
                m = psi[k];
            }
        }
        let mut z = 0.0;
        for k in 0..n_cat {
            z += (psi[k] - m).exp();
        }
        ll += psi[y[p]] - (m + z.ln());
    }
    ll
}

/// Deterministic anchor for the GPCM (Muraki, 1992) per-item score and the CLOSED-FORM multinomial
/// information, on ONE `D = 2` CROSS-loader item with an ASYMMETRIC NEGATIVE loading and
/// NON-MONOTONE (unordered) steps at fixed asymmetric traits, `K = 3`. The score is pinned against
/// finite differences of the complete-data GPCM log-likelihood, and the information block against
/// the NEGATIVE FD Hessian — which equals the exact multinomial Hessian since it is
/// data-independent given `theta` (the mutant that uses the BHHH score cross-product as the
/// information fails here, and would make the Louis SE degenerate). A sign flip in the residual, a
/// transposed/dropped design-matrix slot, or an over-collapsed step block all fail here — none of
/// which a centered/symmetric value-recovery test would localize. The Louis block is pinned to
/// `H - sum_p s_p s_p'` by an INDEPENDENT per-person score outer-product re-sum (the mutant
/// `H + sum s s'` inverts the sign of the missing-information subtraction).
#[test]
fn gpcm_mhrm_score_and_info_match_finite_difference() {
    let nd = 2usize;
    let n_cat = 3usize;
    let dims = vec![0usize, 1usize];
    // [a0, a1, step_1, step_2]; a1 NEGATIVE, steps non-monotone (0.9 then -0.4 -> not increasing)
    let params = vec![0.9f64, -0.6, 0.9, -0.4];
    let pi = dims.len() + (n_cat - 1); // 4
                                       // 4 persons, asymmetric traits, responses spanning all 3 categories
    let theta = vec![0.5, -1.0, -0.7, 0.4, 1.2, 0.9, -0.3, -1.1];
    let y = vec![2usize, 0, 1, 2];
    let np = 4usize;
    let (s, h, hobs) = item_score_info(
        MhrmModel::Gpcm { n_cat },
        &params,
        &dims,
        &theta,
        &y,
        None,
        0,
        np,
        1,
        nd,
    );
    assert_eq!(s.len(), pi);
    // score[t] = d loglik / d params[t]
    let eps = 1e-6;
    for t in 0..pi {
        let mut pp = params.clone();
        pp[t] += eps;
        let mut pm = params.clone();
        pm[t] -= eps;
        let fd = (gpcm_item_loglik(&pp, &dims, &theta, &y, np, nd, n_cat)
            - gpcm_item_loglik(&pm, &dims, &theta, &y, np, nd, n_cat))
            / (2.0 * eps);
        assert!(
            (s[t] - fd).abs() < 1e-4,
            "gpcm score[{t}] {} vs FD {}",
            s[t],
            fd
        );
    }
    // info[a][b] = -d^2 loglik / d params[a] d params[b] (exact multinomial Hessian; symmetric, PD)
    let hh = 1e-3;
    for a in 0..pi {
        for b in 0..pi {
            let mut fpp = params.clone();
            fpp[a] += hh;
            fpp[b] += hh;
            let mut fpm = params.clone();
            fpm[a] += hh;
            fpm[b] -= hh;
            let mut fmp = params.clone();
            fmp[a] -= hh;
            fmp[b] += hh;
            let mut fmm = params.clone();
            fmm[a] -= hh;
            fmm[b] -= hh;
            let d2 = (gpcm_item_loglik(&fpp, &dims, &theta, &y, np, nd, n_cat)
                - gpcm_item_loglik(&fpm, &dims, &theta, &y, np, nd, n_cat)
                - gpcm_item_loglik(&fmp, &dims, &theta, &y, np, nd, n_cat)
                + gpcm_item_loglik(&fmm, &dims, &theta, &y, np, nd, n_cat))
                / (4.0 * hh * hh);
            assert!(
                (h[a * pi + b] - (-d2)).abs() < 1e-2,
                "gpcm info[{a}][{b}] {} vs -FDhess {}",
                h[a * pi + b],
                -d2
            );
            assert!(
                (h[a * pi + b] - h[b * pi + a]).abs() < 1e-12,
                "info symmetric"
            );
        }
    }
    // non-trivial layout: the a0-a1 cross term AND an a0-step1 cross term are genuinely nonzero
    assert!(h[1].abs() > 0.05, "a0-a1 cross-info nonzero: {}", h[1]);
    assert!(h[2].abs() > 0.02, "a0-step1 cross-info nonzero: {}", h[2]);
    // Louis: hobs = H - sum_p s_p s_p'. Re-sum the per-person score outer product INDEPENDENTLY
    // (design J[k][t<li] = k*theta_d, J[k][li+k-1] = 1; resid_k = [k==y] - P_k).
    let mut ss = vec![0.0f64; pi * pi];
    for p in 0..np {
        let mut base = 0.0;
        for (t, &d) in dims.iter().enumerate() {
            base += params[t] * theta[p * nd + d];
        }
        let pr = gpcm_probs(base, &params[dims.len()..], n_cat);
        let mut sp = vec![0.0f64; pi];
        for k in 0..n_cat {
            let resid = (if k == y[p] { 1.0 } else { 0.0 }) - pr[k];
            for (t, &d) in dims.iter().enumerate() {
                sp[t] += (k as f64) * theta[p * nd + d] * resid;
            }
            if k >= 1 {
                sp[dims.len() + k - 1] += resid;
            }
        }
        for a in 0..pi {
            for b in 0..pi {
                ss[a * pi + b] += sp[a] * sp[b];
            }
        }
    }
    for idx in 0..pi * pi {
        assert!(
            (hobs[idx] - (h[idx] - ss[idx])).abs() < 1e-9,
            "gpcm louis missing-info sign: hobs[{idx}] {} vs H-ss {}",
            hobs[idx],
            h[idx] - ss[idx]
        );
    }
}

/// Reduction anchor: at `D = 1`, GPCM MH-RM agrees with the deterministic unidimensional GPCM MMLE
/// (`poly::fit_poly_unidim(PolyModel::Gpcm)`, Bock-Aitkin quadrature) within Monte-Carlo tolerance.
/// NOT bit-exact — MH-RM is stochastic and uses an unconstrained slope (vs `fit_poly_unidim`'s
/// `log_a > 0`), so it is up to reflection (both land positive here on all-positive truth).
#[test]
fn gpcm_mhrm_reduces_to_poly_unidim_at_d1() {
    use crate::poly::{fit_poly_unidim, PolyModel};
    let (n, n_items, n_cat) = (1600usize, 8usize, 3usize);
    let pattern = vec![1u8; n_items];
    let a_t: Vec<f64> = (0..n_items).map(|i| 0.9 + 0.08 * (i % 4) as f64).collect();
    // non-monotone (unordered) steps per item
    let step_t: Vec<[f64; 2]> = (0..n_items)
        .map(|i| [0.6 - 0.1 * (i % 3) as f64, -0.5 + 0.12 * (i % 4) as f64])
        .collect();
    let mut rng = Lcg(2718281);
    let mut th = vec![0.0f64; n];
    for v in th.iter_mut() {
        *v = rng.normal();
    }
    let mut y = vec![0usize; n * n_items];
    for p in 0..n {
        for i in 0..n_items {
            let probs = gpcm_probs(a_t[i] * th[p], &step_t[i], n_cat);
            y[p * n_items + i] = gpcm_sample(&probs, rng.next_f64());
        }
    }
    let cfg = MhrmConfig {
        max_cycles: 1200,
        burn_in: 180,
        mh_steps: 8,
        model: MhrmModel::Gpcm { n_cat },
        seed: 31,
        ..MhrmConfig::default()
    };
    let res = fit_mhrm(&y, None, &pattern, n, n_items, 1, &cfg).unwrap();
    assert_eq!(res.n_cat, n_cat);
    assert!(res.intercept.is_empty());
    assert_eq!(res.step.len(), n_items * (n_cat - 1));
    assert_eq!(res.n_parameters, n_items + n_items * (n_cat - 1));
    // slopes land positive after canonicalization
    assert!(res.loading.iter().all(|&a| a > 0.0));
    let det = fit_poly_unidim(&y, None, n, n_items, n_cat, PolyModel::Gpcm, 41, 200, 1e-6).unwrap();
    assert!(
        rmse(&res.loading, &det.slope) < 0.15,
        "GPCM MH-RM vs MMLE slope RMSE {}",
        rmse(&res.loading, &det.slope)
    );
    let det_steps: Vec<f64> = det
        .cat_params
        .iter()
        .flat_map(|c| c.iter().copied())
        .collect();
    assert_eq!(det_steps.len(), res.step.len());
    assert!(
        rmse(&res.step, &det_steps) < 0.2,
        "GPCM MH-RM vs MMLE step RMSE {}",
        rmse(&res.step, &det_steps)
    );
}

/// Headline GPCM capability: `D = 5` confirmatory GPCM. The `q^D` Gauss-Hermite grid (`21^5`) and
/// the QMC E-step are infeasible; MH-RM's stochastic imputation is `D`-agnostic. Simple structure
/// (3 pure anchors per dimension) plus one genuinely NEGATIVE cross-loader, non-monotone steps,
/// `K = 3` — loadings and steps recovered with the correct sign.
#[test]
fn gpcm_mhrm_recovers_high_dim_d5() {
    let (n_dims, n, n_cat) = (5usize, 2200usize, 3usize);
    let n_items = 16usize;
    let mut pattern = vec![0u8; n_items * n_dims];
    for i in 0..15 {
        pattern[i * n_dims + i / 3] = 1; // items 0..14: 3 pure anchors per dimension
    }
    pattern[15 * n_dims] = 1;
    pattern[15 * n_dims + 2] = 1; // item15 cross-loads dims 0 and 2
    let mut a_t = vec![0.0f64; n_items * n_dims];
    for i in 0..15 {
        a_t[i * n_dims + i / 3] = 0.9 + 0.1 * (i % 3) as f64;
    }
    a_t[15 * n_dims] = 1.0;
    a_t[15 * n_dims + 2] = -0.7; // NEGATIVE cross-loader
    let step_t: Vec<[f64; 2]> = (0..n_items)
        .map(|i| [0.7 - 0.12 * (i % 3) as f64, -0.4 + 0.1 * (i % 4) as f64])
        .collect();
    let mut rng = Lcg(50505);
    let mut th = vec![0.0f64; n * n_dims];
    for v in th.iter_mut() {
        *v = rng.normal();
    }
    let mut y = vec![0usize; n * n_items];
    for p in 0..n {
        for i in 0..n_items {
            let mut base = 0.0;
            for d in 0..n_dims {
                base += a_t[i * n_dims + d] * th[p * n_dims + d];
            }
            let probs = gpcm_probs(base, &step_t[i], n_cat);
            y[p * n_items + i] = gpcm_sample(&probs, rng.next_f64());
        }
    }
    let cfg = MhrmConfig {
        max_cycles: 1000,
        burn_in: 200,
        mh_steps: 6,
        model: MhrmModel::Gpcm { n_cat },
        seed: 17,
        ..MhrmConfig::default()
    };
    let res = fit_mhrm(&y, None, &pattern, n, n_items, n_dims, &cfg).unwrap();
    assert_eq!(res.n_dims, 5);
    assert_eq!(res.n_cat, n_cat);
    for i in 0..n_items {
        for d in 0..n_dims {
            if pattern[i * n_dims + d] == 0 {
                assert_eq!(res.loading[i * n_dims + d], 0.0);
            }
        }
    }
    let (mut se2, mut cnt) = (0.0, 0usize);
    for idx in 0..n_items * n_dims {
        if pattern[idx] == 1 {
            se2 += (res.loading[idx] - a_t[idx]).powi(2);
            cnt += 1;
        }
    }
    let load_rmse = (se2 / cnt as f64).sqrt();
    assert!(
        load_rmse < 0.25,
        "D=5 GPCM on-pattern loading RMSE {load_rmse}"
    );
    assert!(
        res.loading[15 * n_dims + 2] < -0.25,
        "negative cross-loader {}",
        res.loading[15 * n_dims + 2]
    );
    let true_steps: Vec<f64> = (0..n_items).flat_map(|i| step_t[i]).collect();
    assert!(
        rmse(&res.step, &true_steps) < 0.25,
        "GPCM step RMSE {}",
        rmse(&res.step, &true_steps)
    );
    for d in 0..n_dims {
        let est: Vec<f64> = (0..n).map(|p| res.theta[p * n_dims + d]).collect();
        let tru: Vec<f64> = (0..n).map(|p| th[p * n_dims + d]).collect();
        assert!(
            corr(&est, &tru) > 0.5,
            "dim {d} theta corr {}",
            corr(&est, &tru)
        );
    }
}

/// The reflection canonicalization FIRES for GPCM and is WITNESSED, with the UNORDERED steps left
/// INVARIANT: `base = k*sum a_d theta_d` flips jointly with `(a, theta)`, so canonicalization
/// touches only the slope column and the trait chain — never the step intercepts. dim0 has a WEAK
/// reverse-keyed sole pure anchor (item0, true `-0.7`) and a STRONG positive cross-loader (item1,
/// dim0 `+1.7`) that sets the axis; raw MH-RM lands the anchor NEGATIVE, so canon must flip dim0.
/// A mutant that ALSO negated the flipped dimension's items' steps would push item0's step_1 to the
/// wrong sign — the final assertion catches it.
#[test]
fn gpcm_mhrm_reflection_fires_on_negative_anchor() {
    let (n_dims, n, n_cat) = (2usize, 5000usize, 3usize);
    let n_items = 4usize;
    // item0 pure d0 (sole d0 anchor), item1 cross d0/d1, item2/3 pure d1
    let pattern = vec![1u8, 0, 1, 1, 0, 1, 0, 1];
    let mut a_t = vec![0.0f64; n_items * n_dims];
    a_t[0] = -0.7; // weak reverse-keyed pure d0 anchor
    a_t[1 * n_dims] = 1.7; // strong positive cross-loader on d0 (sets the axis)
    a_t[1 * n_dims + 1] = 0.6;
    a_t[2 * n_dims + 1] = 1.2;
    a_t[3 * n_dims + 1] = 1.0;
    // item0's steps are positive-then-negative; if reflection wrongly swept them, step_1 -> ~-0.5
    let step_t = [[0.5f64, -0.3], [0.4, -0.5], [0.6, -0.2], [0.3, -0.4]];
    let mut rng = Lcg(97531);
    let mut th = vec![0.0f64; n * n_dims];
    for v in th.iter_mut() {
        *v = rng.normal();
    }
    let mut y = vec![0usize; n * n_items];
    for p in 0..n {
        for i in 0..n_items {
            let mut base = 0.0;
            for d in 0..n_dims {
                base += a_t[i * n_dims + d] * th[p * n_dims + d];
            }
            let probs = gpcm_probs(base, &step_t[i], n_cat);
            y[p * n_items + i] = gpcm_sample(&probs, rng.next_f64());
        }
    }
    let cfg = MhrmConfig {
        max_cycles: 1000,
        burn_in: 200,
        mh_steps: 8,
        model: MhrmModel::Gpcm { n_cat },
        seed: 24,
        ..MhrmConfig::default()
    };
    let res = fit_mhrm(&y, None, &pattern, n, n_items, n_dims, &cfg).unwrap();
    assert!(
        res.loading[0] > 0.3,
        "reflected anchor positive: {}",
        res.loading[0]
    );
    assert!(
        res.loading[1 * n_dims] < -0.5,
        "co-loader flipped negative: {}",
        res.loading[1 * n_dims]
    );
    let th0: Vec<f64> = (0..n).map(|p| res.theta[p * n_dims]).collect();
    let tt0: Vec<f64> = (0..n).map(|p| th[p * n_dims]).collect();
    assert!(
        corr(&th0, &tt0) < -0.4,
        "flipped-dim theta corr negative: {}",
        corr(&th0, &tt0)
    );
    // steps INVARIANT under reflection: item0's step_1 stays near its (un-flipped) truth +0.5, well
    // away from the mutant's -0.5.
    assert!(
        (res.step[0] - step_t[0][0]).abs() < 0.35,
        "GPCM step not swept by reflection: step_1 {} vs truth {}",
        res.step[0],
        step_t[0][0]
    );
}

/// GPCM validation guards constructed non-vacuously: the SAME well-formed GPCM dataset fits (and
/// exposes the `step`/`n_cat` result shape), then each defect trips its INTENDED guard — an
/// out-of-range response, and a declared category never observed for an item (an unidentified step,
/// Muraki, 1992).
#[test]
fn gpcm_mhrm_validates_and_structure() {
    let (n, n_items, n_dims, n_cat) = (60usize, 4usize, 2usize, 3usize);
    let pattern = vec![1u8, 0, 1, 0, 0, 1, 0, 1]; // pure anchors on both dims
                                                  // y = (p + i) % 3 -> every item sees all 3 categories across persons
    let mut y = vec![0usize; n * n_items];
    for p in 0..n {
        for i in 0..n_items {
            y[p * n_items + i] = (p + i) % n_cat;
        }
    }
    let cfg = MhrmConfig {
        max_cycles: 30,
        burn_in: 5,
        model: MhrmModel::Gpcm { n_cat },
        ..MhrmConfig::default()
    };
    let res = fit_mhrm(&y, None, &pattern, n, n_items, n_dims, &cfg).unwrap();
    assert_eq!(res.n_cat, n_cat);
    assert!(res.intercept.is_empty());
    assert_eq!(res.step.len(), n_items * (n_cat - 1));
    assert_eq!(res.se_step.len(), n_items * (n_cat - 1));
    assert!(res.se_intercept.is_empty());
    assert_eq!(res.n_parameters, n_items + n_items * (n_cat - 1));
    // (a) response out of 0..n_cat where observed
    let mut ybad = y.clone();
    ybad[0] = n_cat; // == 3, out of range
    assert!(fit_mhrm(&ybad, None, &pattern, n, n_items, n_dims, &cfg).is_err());
    // (b) item0's category-1 responses remapped to 0 -> category 1 never observed for item0
    // (still in range), tripping the coverage guard (the binary 2PL does NOT enforce this).
    let mut ycov = y.clone();
    for p in 0..n {
        if ycov[p * n_items] == 1 {
            ycov[p * n_items] = 0;
        }
    }
    assert!(fit_mhrm(&ycov, None, &pattern, n, n_items, n_dims, &cfg).is_err());
    // (c) n_cat above the MHRM_MAX_CAT cap is rejected (the cap guard fires before the
    // O(n_cat) coverage allocation) -- makes the MHRM_MAX_CAT constant live.
    let cfg_big = MhrmConfig {
        model: MhrmModel::Gpcm {
            n_cat: MHRM_MAX_CAT + 1,
        },
        ..cfg
    };
    assert!(fit_mhrm(&y, None, &pattern, n, n_items, n_dims, &cfg_big).is_err());
    // (d) GPCM with n_cat == 2 (also n_free_cat == 1, colliding with the 2PL) routes its single
    // step to `step`/`se_step` -- NOT the 2PL `intercept`/`se_intercept` -- honoring the
    // family-based contract. y2 = (p + i) % 2 sees both categories per item.
    let mut y2 = vec![0usize; n * n_items];
    for p in 0..n {
        for i in 0..n_items {
            y2[p * n_items + i] = (p + i) % 2;
        }
    }
    let cfg2 = MhrmConfig {
        model: MhrmModel::Gpcm { n_cat: 2 },
        ..cfg
    };
    let res2 = fit_mhrm(&y2, None, &pattern, n, n_items, n_dims, &cfg2).unwrap();
    assert_eq!(res2.n_cat, 2);
    assert!(
        res2.intercept.is_empty(),
        "GPCM n_cat=2 must not populate 2PL intercept"
    );
    assert!(res2.se_intercept.is_empty());
    assert_eq!(res2.step.len(), n_items); // J * (2 - 1)
    assert_eq!(res2.se_step.len(), n_items);
    assert!(res2.step.iter().all(|s| s.is_finite()));
    assert_eq!(res2.n_parameters, n_items + n_items); // 4 free loadings + 4 single steps
}

/// Literature-grade GPCM Monte-Carlo recovery (>=500 reps), normal + right-skew traits. Run with:
/// `cargo test -p mlsirm-core --release mc_gpcm_mhrm_recovery_500 -- --ignored --nocapture`.
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps)"]
fn mc_gpcm_mhrm_recovery_500() {
    let reps = 500usize;
    let n_cat = 3usize;
    // D=5 is the regime GH/QMC cannot reach for a polytomous item factor model.
    for &(n_dims, n) in &[(2usize, 2000usize), (5usize, 2500usize)] {
        for &skew in &[false, true] {
            let n_items = if n_dims == 2 { 8 } else { 15 };
            let mut pattern = vec![0u8; n_items * n_dims];
            let mut a_t = vec![0.0f64; n_items * n_dims];
            let per = n_items / n_dims;
            for i in 0..per * n_dims {
                let d = i / per;
                pattern[i * n_dims + d] = 1;
                a_t[i * n_dims + d] = 0.9 + 0.1 * (i % 3) as f64;
            }
            // last item cross-loads dims 0 and 1 (dim0 negative)
            let xi = n_items - 1;
            pattern[xi * n_dims] = 1;
            pattern[xi * n_dims + 1] = 1;
            a_t[xi * n_dims] = -0.8;
            a_t[xi * n_dims + 1] = 0.7;
            let step_t: Vec<[f64; 2]> = (0..n_items)
                .map(|i| [0.7 - 0.12 * (i % 3) as f64, -0.4 + 0.1 * (i % 4) as f64])
                .collect();
            let n_free: usize = pattern.iter().filter(|&&v| v == 1).count();

            let (mut conv, mut lse2, mut lbias, mut lcnt) = (0usize, 0.0, 0.0, 0usize);
            let (mut sse2, mut sbias) = (0.0, 0.0);
            let mut corr_sum = 0.0;
            for rep in 0..reps {
                let mut rng = Lcg(0x6CBC_u64
                    .wrapping_mul((rep as u64) + 1)
                    .wrapping_add(n_dims as u64));
                let mut th = vec![0.0f64; n * n_dims];
                for v in th.iter_mut() {
                    *v = if skew {
                        // standardized right-skew (Exp(1) - 1): mean 0, var 1
                        -(rng.next_f64().max(1e-12)).ln() - 1.0
                    } else {
                        rng.normal()
                    };
                }
                let mut y = vec![0usize; n * n_items];
                for p in 0..n {
                    for i in 0..n_items {
                        let mut base = 0.0;
                        for d in 0..n_dims {
                            base += a_t[i * n_dims + d] * th[p * n_dims + d];
                        }
                        let probs = gpcm_probs(base, &step_t[i], n_cat);
                        y[p * n_items + i] = gpcm_sample(&probs, rng.next_f64());
                    }
                }
                let cfg = MhrmConfig {
                    max_cycles: 900,
                    burn_in: 180,
                    mh_steps: 6,
                    model: MhrmModel::Gpcm { n_cat },
                    seed: 0xC0DE_u64.wrapping_add(rep as u64),
                    estimate_se: false,
                    ..MhrmConfig::default()
                };
                let res = fit_mhrm(&y, None, &pattern, n, n_items, n_dims, &cfg).unwrap();
                if res.converged {
                    conv += 1;
                }
                for idx in 0..n_items * n_dims {
                    if pattern[idx] == 1 {
                        let e = res.loading[idx] - a_t[idx];
                        lse2 += e * e;
                        lbias += e;
                        lcnt += 1;
                    }
                }
                for i in 0..n_items {
                    for j in 0..n_cat - 1 {
                        let e = res.step[i * (n_cat - 1) + j] - step_t[i][j];
                        sse2 += e * e;
                        sbias += e;
                    }
                }
                let est: Vec<f64> = (0..n).map(|p| res.theta[p * n_dims]).collect();
                let tru: Vec<f64> = (0..n).map(|p| th[p * n_dims]).collect();
                corr_sum += corr(&est, &tru);
            }
            let scnt = (reps * n_items * (n_cat - 1)) as f64;
            let load_rmse = (lse2 / lcnt as f64).sqrt();
            let step_rmse = (sse2 / scnt).sqrt();
            println!(
                "[gpcm MC D={n_dims} N={n} n_free={n_free} K={n_cat} skew={skew}] reps={reps} conv={:.3} loadRMSE={:.4} loadBias={:.4} stepRMSE={:.4} stepBias={:.4} thetaCorr={:.3}",
                conv as f64 / reps as f64,
                load_rmse,
                lbias / lcnt as f64,
                step_rmse,
                sbias / scnt,
                corr_sum / reps as f64
            );
            assert!(conv as f64 / reps as f64 > 0.9, "GPCM convergence rate");
            if !skew {
                assert!(load_rmse < 0.22, "GPCM normal loading RMSE {load_rmse}");
                assert!(step_rmse < 0.25, "GPCM normal step RMSE {step_rmse}");
            }
        }
    }
    println!("=== gpcm done ===");
}

/// Literature-grade Monte-Carlo recovery (>=500 reps). Run with:
/// `cargo test -p mlsirm-core --release mc_mhrm_recovery_500 -- --ignored --nocapture`.
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps)"]
fn mc_mhrm_recovery_500() {
    let reps = 500usize;
    // (n_dims, N) conditions; D=6 is the regime GH/QMC cannot reach.
    for &(n_dims, n) in &[(2usize, 2000usize), (6usize, 2500usize)] {
        for &skew in &[false, true] {
            let n_items = if n_dims == 2 { 8 } else { 20 };
            // confirmatory pattern: pure anchors per dim + one negative cross-loader
            let mut pattern = vec![0u8; n_items * n_dims];
            let mut a_t = vec![0.0f64; n_items * n_dims];
            let per = n_items / n_dims;
            for i in 0..per * n_dims {
                let d = i / per;
                pattern[i * n_dims + d] = 1;
                a_t[i * n_dims + d] = 0.9 + 0.1 * (i % 3) as f64;
            }
            // last item cross-loads dims 0 and 1 (dim0 negative)
            let xi = n_items - 1;
            pattern[xi * n_dims] = 1;
            pattern[xi * n_dims + 1] = 1;
            a_t[xi * n_dims] = -0.8;
            a_t[xi * n_dims + 1] = 0.7;
            let b_t: Vec<f64> = (0..n_items).map(|i| -0.4 + 0.12 * (i % 5) as f64).collect();
            let n_free: usize = pattern.iter().filter(|&&v| v == 1).count();

            let (mut conv, mut se2, mut sbias, mut cnt) = (0usize, 0.0, 0.0, 0usize);
            let mut corr_sum = 0.0;
            for rep in 0..reps {
                let mut rng = Lcg(0x51ED_u64
                    .wrapping_mul((rep as u64) + 1)
                    .wrapping_add(n_dims as u64));
                let mut th = vec![0.0f64; n * n_dims];
                for v in th.iter_mut() {
                    *v = if skew {
                        // standardized right-skew (Exp(1) - 1): mean 0, var 1
                        -(rng.next_f64().max(1e-12)).ln() - 1.0
                    } else {
                        rng.normal()
                    };
                }
                let mut y = vec![0usize; n * n_items];
                for p in 0..n {
                    for i in 0..n_items {
                        let mut base = b_t[i];
                        for d in 0..n_dims {
                            base += a_t[i * n_dims + d] * th[p * n_dims + d];
                        }
                        let pr = 1.0 / (1.0 + (-base).exp());
                        y[p * n_items + i] = if rng.next_f64() < pr { 1 } else { 0 };
                    }
                }
                let cfg = MhrmConfig {
                    max_cycles: 900,
                    burn_in: 180,
                    mh_steps: 6,
                    seed: 0xABCD_u64.wrapping_add(rep as u64),
                    estimate_se: false,
                    ..MhrmConfig::default()
                };
                let res = fit_mhrm(&y, None, &pattern, n, n_items, n_dims, &cfg).unwrap();
                if res.converged {
                    conv += 1;
                }
                for idx in 0..n_items * n_dims {
                    if pattern[idx] == 1 {
                        let e = res.loading[idx] - a_t[idx];
                        se2 += e * e;
                        sbias += e;
                        cnt += 1;
                    }
                }
                let est: Vec<f64> = (0..n).map(|p| res.theta[p * n_dims]).collect();
                let tru: Vec<f64> = (0..n).map(|p| th[p * n_dims]).collect();
                corr_sum += corr(&est, &tru);
            }
            let load_rmse = (se2 / cnt as f64).sqrt();
            let load_bias = sbias / cnt as f64;
            println!(
                "[mhrm MC D={n_dims} N={n} n_free={n_free} skew={skew}] reps={reps} conv={:.3} loadRMSE={:.4} loadBias={:.4} thetaCorr={:.3}",
                conv as f64 / reps as f64,
                load_rmse,
                load_bias,
                corr_sum / reps as f64
            );
            assert!(conv as f64 / reps as f64 > 0.9, "convergence rate");
            if !skew {
                assert!(load_rmse < 0.2, "normal loading RMSE {load_rmse}");
            }
        }
    }

    // correlated-Sigma condition (Cai 2010b): recover an exchangeable Phi at the near-PD-boundary
    // rho = 0.5, D = 3 (so a persistent PD-backtracking stall would surface over 500 reps).
    {
        let (n_dims, n, rho) = (3usize, 3000usize, 0.5f64);
        let per = 4usize;
        let n_items = per * n_dims;
        let mut pattern = vec![0u8; n_items * n_dims];
        let mut a_t = vec![0.0f64; n_items * n_dims];
        for d in 0..n_dims {
            for a in 0..per {
                let i = d * per + a;
                pattern[i * n_dims + d] = 1;
                a_t[i * n_dims + d] = 0.9 + 0.1 * a as f64;
            }
        }
        let b_t: Vec<f64> = (0..n_items).map(|i| -0.4 + 0.1 * (i % 5) as f64).collect();
        let mut phi = vec![rho; n_dims * n_dims];
        for a in 0..n_dims {
            phi[a * n_dims + a] = 1.0;
        }
        let l = chol_lower(&phi, n_dims).unwrap();
        let n_off = n_dims * (n_dims - 1) / 2;
        let (mut conv, mut se2, mut sbias) = (0usize, 0.0f64, 0.0f64);
        for rep in 0..reps {
            let mut rng = Lcg(0x5EED_u64.wrapping_mul((rep as u64) + 1));
            let mut th = vec![0.0f64; n * n_dims];
            for p in 0..n {
                let z: Vec<f64> = (0..n_dims).map(|_| rng.normal()).collect();
                for a in 0..n_dims {
                    let mut v = 0.0;
                    for b in 0..=a {
                        v += l[a * n_dims + b] * z[b];
                    }
                    th[p * n_dims + a] = v;
                }
            }
            let mut y = vec![0usize; n * n_items];
            for p in 0..n {
                for i in 0..n_items {
                    let mut base = b_t[i];
                    for d in 0..n_dims {
                        base += a_t[i * n_dims + d] * th[p * n_dims + d];
                    }
                    let pr = 1.0 / (1.0 + (-base).exp());
                    y[p * n_items + i] = if rng.next_f64() < pr { 1 } else { 0 };
                }
            }
            let cfg = MhrmConfig {
                max_cycles: 1200,
                burn_in: 300,
                mh_steps: 6,
                estimate_corr: true,
                estimate_se: false,
                seed: 0xBEEF_u64.wrapping_add(rep as u64),
                ..MhrmConfig::default()
            };
            let res = fit_mhrm(&y, None, &pattern, n, n_items, n_dims, &cfg).unwrap();
            if res.converged {
                conv += 1;
            }
            for a in 0..n_dims {
                for b in a + 1..n_dims {
                    let e = res.corr[a * n_dims + b] - rho;
                    se2 += e * e;
                    sbias += e;
                }
            }
        }
        let m = (reps * n_off) as f64;
        println!(
            "[mhrm MC correlated D={n_dims} N={n} rho={rho}] reps={reps} conv={:.3} corrRMSE={:.4} corrBias={:.4}",
            conv as f64 / reps as f64,
            (se2 / m).sqrt(),
            sbias / m
        );
        assert!(conv as f64 / reps as f64 > 0.9, "correlated convergence");
        assert!((se2 / m).sqrt() < 0.1, "correlated corr RMSE");
    }
    println!("=== done ===");
}
