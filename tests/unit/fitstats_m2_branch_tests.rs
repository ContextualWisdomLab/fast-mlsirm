use super::*;
use crate::quadrature::gh_rule;
use crate::scoring::{ItemBank, PriorSpec};

fn bank<'a>(alpha: &'a [f64], b: &'a [f64], zeta: &'a [f64], fid: &'a [usize]) -> ItemBank<'a> {
    ItemBank {
        alpha,
        b,
        zeta,
        tau: -30.0,
        factor_id: fid,
        model_type: crate::ModelType::Mirt,
        n_dims: 1,
        latent_dim: 1,
        eps_distance: 1e-8,
    }
}

#[test]
fn m2_factorizes_independent_trait_dimensions() {
    let probs = vec![0.2, 0.8, 0.3, 0.7];
    let weights = vec![0.5, 0.5];
    let sets = vec![vec![0, 1]];
    let moments = factorized_trait_moments(&probs, &weights, 2, &[0, 1], 2, &sets);
    assert!((moments[0] - 0.25).abs() < 1e-14);
    assert!(
        (moments[0] - 0.31).abs() > 1e-3,
        "must not share one trait node"
    );
    let missing_dimension = factorized_trait_moments(&probs, &weights, 2, &[0, 1], 3, &[vec![0]]);
    assert!((missing_dimension[0] - 0.5).abs() < 1e-14);
}

#[test]
fn m2_numeric_helpers_cover_every_parameter_and_metric_branch() {
    assert_eq!(finish_ncchi2_mixture(1.0, 2.0, true), 0.5);
    assert!(finish_ncchi2_mixture(1.0, 2.0, false).is_nan());
    assert!(solve_decreasing_root(0.0, 1.0, 0.5, &|_| 1.0).is_nan());
    let root = solve_decreasing_root(0.0, 1.0, 0.5, &|value| 1.0 - value);
    assert!((root - 0.5).abs() < 1e-10);

    let params = m2_parameters(1, true, true, 2, true);
    assert_eq!(params.len(), 5);
    let mut alpha = vec![3.0];
    let mut b = vec![2.0];
    let mut zeta = vec![4.0, 5.0];
    let mut tau = 6.0;
    for (index, param) in params.iter().copied().enumerate() {
        assert_eq!(
            m2_param_value(param, &alpha, &b, &zeta, tau, 2),
            index as f64 + 2.0
        );
        set_m2_param(
            param,
            index as f64 + 12.0,
            &mut alpha,
            &mut b,
            &mut zeta,
            &mut tau,
            2,
        );
        assert_eq!(
            m2_param_value(param, &alpha, &b, &zeta, tau, 2),
            index as f64 + 12.0
        );
    }
    assert_eq!(m2_parameters(1, false, false, 0, false).len(), 1);

    assert_eq!(srmsr_from_sum(4.0, 1), 2.0);
    assert!(srmsr_from_sum(0.0, 0).is_nan());
    let finite = comparative_fit_metrics(5.0, 3.0, 20.0, 5.0);
    assert!(finite.0.is_finite() && finite.1.is_finite());
    let unavailable = comparative_fit_metrics(5.0, 3.0, 2.0, 5.0);
    assert!(unavailable.0.is_nan() && unavailable.1.is_nan());
}

#[test]
fn ncchi2_large_noncentrality_matches_reference_values() {
    // Independently evaluated with scipy.stats.ncx2 and scipy.optimize.brentq.
    let cases = [
        (2_000.0, 50.0, 0.05, 2_099.928_758_291_509_4),
        (10_000.0, 50.0, 0.05, 10_282.274_417_418_035),
        (10_000.0, 50.0, 0.95, 9_625.139_462_181_574),
    ];
    for (statistic, df, target, expected) in cases {
        let got = nc_lambda_for(statistic, df, target);
        assert!((got - expected).abs() <= 1e-10 * expected);
        assert!((ncchi2_cdf(statistic, df, got) - target).abs() <= 1e-10);
    }
}

#[test]
fn m2_rejects_too_few_items() {
    let (alpha, b, zeta, fid) = (vec![0.0; 2], vec![0.0; 2], vec![0.0; 2], vec![0usize; 2]);
    let bk = bank(&alpha, &b, &zeta, &fid);
    let y = vec![0.0; 4];
    let obs = vec![true; 4];
    assert!(m2_rmsea2(
        &bk,
        &y,
        &obs,
        2,
        &PriorSpec::standard(1),
        11,
        XiRule::GaussHermite { q_xi: 7 }
    )
    .is_err());
}

#[test]
fn m2_rejects_length_mismatch() {
    let (alpha, b, zeta, fid) = (vec![0.0; 4], vec![0.0; 4], vec![0.0; 4], vec![0usize; 4]);
    let bk = bank(&alpha, &b, &zeta, &fid);
    let y = vec![0.0; 8]; // wrong length for n_persons=3
    let obs = vec![true; 8];
    assert!(m2_rmsea2(
        &bk,
        &y,
        &obs,
        3,
        &PriorSpec::standard(1),
        11,
        XiRule::GaussHermite { q_xi: 7 }
    )
    .is_err());
}

#[test]
fn m2_rejects_nonpositive_df() {
    // 3 MIRT items: s = 3 + 3 = 6 moments, p = 2*3 = 6 params -> df <= 0
    let (alpha, b, zeta, fid) = (vec![0.0; 3], vec![0.0; 3], vec![0.0; 3], vec![0usize; 3]);
    let bk = bank(&alpha, &b, &zeta, &fid);
    let n = 50usize;
    let y = vec![1.0; n * 3];
    let obs = vec![true; n * 3];
    assert!(m2_rmsea2(
        &bk,
        &y,
        &obs,
        n,
        &PriorSpec::standard(1),
        11,
        XiRule::GaussHermite { q_xi: 7 }
    )
    .is_err());
}

#[test]
fn m2_rejects_too_few_complete_cases() {
    // 8 items, but every row has a missing entry -> no complete cases
    let (alpha, b, zeta, fid) = (vec![0.0; 8], vec![0.0; 8], vec![0.0; 8], vec![0usize; 8]);
    let bk = bank(&alpha, &b, &zeta, &fid);
    let n = 40usize;
    let y = vec![0.0; n * 8];
    let mut obs = vec![true; n * 8];
    for p in 0..n {
        obs[p * 8] = false; // first item missing for everyone
    }
    assert!(m2_rmsea2(
        &bk,
        &y,
        &obs,
        n,
        &PriorSpec::standard(1),
        11,
        XiRule::GaussHermite { q_xi: 7 }
    )
    .is_err());
}

#[test]
fn m2_runs_on_small_hand_built_bank() {
    // exercises the full body (Cholesky, Delta, Xi, CI, SRMSR) under the lib
    // tests, not only the integration recovery test
    let n_items = 8usize;
    let n = 400usize;
    let alpha = vec![0.0; n_items];
    let b: Vec<f64> = (0..n_items).map(|i| -0.8 + 0.2 * i as f64).collect();
    let zeta = vec![0.0; n_items];
    let fid = vec![0usize; n_items];
    let mut state = 4242u64;
    let mut unif = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut y = vec![0.0; n * n_items];
    for p in 0..n {
        let u1 = unif().max(1e-12);
        let u2 = unif();
        let th = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        for i in 0..n_items {
            let prob = 1.0 / (1.0 + (-(th + b[i])).exp());
            y[p * n_items + i] = if unif() < prob { 1.0 } else { 0.0 };
        }
    }
    let obs = vec![true; n * n_items];
    let bk = bank(&alpha, &b, &zeta, &fid);
    let res = m2_rmsea2(
        &bk,
        &y,
        &obs,
        n,
        &PriorSpec::standard(1),
        21,
        XiRule::GaussHermite { q_xi: 7 },
    )
    .expect("m2 should run");
    assert_eq!(res.n_moments, 36);
    assert!(res.m2.is_finite() && res.df == 20.0);
    assert!(res.rmsea2_ci_lower <= res.rmsea2_ci_upper + 1e-9);
    assert!(res.srmsr.is_finite());
}

#[test]
fn poly_m2_reduces_to_binary_m2() {
    // At K=2 the polytomous M2 must equal the trusted binary m2_rmsea2 at the
    // same parameters (both GRM and GPCM cells reduce to the 2PL). This
    // anchors the cumulative-moment machinery, the merge-max Xi, and the
    // Delta/Cholesky solve against already-validated code.
    use crate::poly::PolyModel;
    let (n_persons, n_items) = (1500usize, 6usize);
    let mut st = 24680u64;
    let mut u = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let a_true: Vec<f64> = (0..n_items).map(|i| 0.9 + 0.1 * i as f64).collect();
    let b_true: Vec<f64> = (0..n_items).map(|i| -0.5 + 0.2 * i as f64).collect();
    let mut yf = vec![0.0_f64; n_persons * n_items];
    let mut yi = vec![0usize; n_persons * n_items];
    for pp in 0..n_persons {
        let u1 = u().max(1e-12);
        let u2 = u();
        let th = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        for i in 0..n_items {
            let pr = 1.0 / (1.0 + (-(a_true[i] * th + b_true[i])).exp());
            let v = if u() < pr { 1.0 } else { 0.0 };
            yf[pp * n_items + i] = v;
            yi[pp * n_items + i] = v as usize;
        }
    }
    let obs = vec![true; n_persons * n_items];
    let alpha: Vec<f64> = a_true.iter().map(|a| a.ln()).collect();
    let zeta = vec![0.0_f64; n_items];
    let fid = vec![0usize; n_items];
    let bk = bank(&alpha, &b_true, &zeta, &fid);
    let r_bin = m2_rmsea2(
        &bk,
        &yf,
        &obs,
        n_persons,
        &PriorSpec::standard(1),
        41,
        XiRule::GaussHermite { q_xi: 1 },
    )
    .unwrap();
    for model in [PolyModel::Gpcm, PolyModel::Grm] {
        let r_poly = poly_m2(
            &yi,
            Some(&obs),
            n_persons,
            n_items,
            2,
            &a_true,
            &b_true,
            model,
            41,
        )
        .unwrap();
        assert_eq!(r_poly.n_moments, r_bin.n_moments, "{model:?} n_moments");
        assert_eq!(
            r_poly.n_parameters, r_bin.n_parameters,
            "{model:?} n_parameters"
        );
        assert_eq!(r_poly.df, r_bin.df, "{model:?} df");
        assert!(
            (r_poly.m2 - r_bin.m2).abs() < 1e-4,
            "{model:?} M2: poly {} vs binary {}",
            r_poly.m2,
            r_bin.m2
        );
        assert!(
            (r_poly.p_value - r_bin.p_value).abs() < 1e-4,
            "{model:?} p_value"
        );
        assert!(
            (r_poly.rmsea2 - r_bin.rmsea2).abs() < 1e-4,
            "{model:?} rmsea2"
        );
    }
}

// GPCM Monte-Carlo for M2 calibration: returns (mean M2/df, rejection rate at
// .05, df) over `reps` datasets simulated at fixed true parameters. Under a
// NORMAL theta (matching the N(0,1) quadrature) the model is correctly
// specified, so M2 -> chi^2(df) even at the true parameters (the residual
// projector removes P dimensions); under a right-SKEWED theta the N(0,1)
// quadrature is a population misspecification the statistic should detect.
fn mc_poly_m2(reps: usize, n_persons: usize, skew: bool) -> (f64, f64, f64) {
    use crate::poly::{gpcm_logprobs, PolyModel};
    let (n_items, k) = (5usize, 3usize);
    let z = k - 1;
    let a_true: Vec<f64> = (0..n_items).map(|i| 0.9 + 0.12 * i as f64).collect();
    let cat_true: Vec<f64> = (0..n_items)
        .flat_map(|i| vec![0.8 - 0.1 * i as f64, -0.8 + 0.1 * i as f64])
        .collect();
    let (mut ratio_sum, mut n_reject, mut df_val) = (0.0_f64, 0usize, 0.0_f64);
    for rep in 0..reps {
        let mut st = 909_090u64 + rep as u64 * 131 + if skew { 5 } else { 0 };
        let mut u = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((st >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut yi = vec![0usize; n_persons * n_items];
        for pp in 0..n_persons {
            let theta = if skew {
                -(u().max(1e-12)).ln() - 1.0
            } else {
                let u1 = u().max(1e-12);
                let u2 = u();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            };
            for i in 0..n_items {
                let base = a_true[i] * theta;
                let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
                let mut ic = vec![0.0_f64; k];
                ic[1..].copy_from_slice(&cat_true[i * z..(i + 1) * z]);
                let lp = gpcm_logprobs(base, &scores, &ic);
                let draw = u();
                let (mut acc, mut cat) = (0.0_f64, k - 1);
                for (c, l) in lp.iter().enumerate() {
                    acc += l.exp();
                    if draw <= acc {
                        cat = c;
                        break;
                    }
                }
                yi[pp * n_items + i] = cat;
            }
        }
        let r = poly_m2(
            &yi,
            None,
            n_persons,
            n_items,
            k,
            &a_true,
            &cat_true,
            PolyModel::Gpcm,
            21,
        )
        .unwrap();
        ratio_sum += r.m2 / r.df;
        if r.p_value < 0.05 {
            n_reject += 1;
        }
        df_val = r.df;
    }
    (
        ratio_sum / reps as f64,
        n_reject as f64 / reps as f64,
        df_val,
    )
}

#[test]
fn poly_m2_calibration_null_and_skew_power() {
    // Fast CI guard. The authoritative >=500-replication study is
    // poly_m2_monte_carlo_500 (ignored). See mc_poly_m2 for the design.
    let (reps, n) = (20usize, 1500usize);
    let (mn, rej_n, df) = mc_poly_m2(reps, n, false);
    let (ms, rej_s, _) = mc_poly_m2(reps, n, true);
    println!(
        "[poly M2] df={df}  normal: mean(M2)/df={mn:.3} reject={rej_n:.3}  \
         skew: mean(M2)/df={ms:.3} reject={rej_s:.3}"
    );
    // matched N(0,1) prior => calibrated (mean ~ df, few false rejections)
    assert!((0.75..=1.35).contains(&mn), "normal M2/df off: {mn}");
    assert!(rej_n < 0.25, "normal rejection too high: {rej_n}");
    // skewed population is a misspecification M2 detects => inflated vs normal
    assert!(ms > mn, "skew must inflate M2 vs normal: {ms} vs {mn}");
}
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn poly_m2_monte_carlo_500() {
    let (reps, n) = (500usize, 2000usize);
    let (mn, rej_n, df) = mc_poly_m2(reps, n, false);
    let (ms, rej_s, _) = mc_poly_m2(reps, n, true);
    println!(
        "[poly M2 500] df={df}  normal: mean(M2)/df={mn:.4} reject={rej_n:.4}  \
         skew: mean(M2)/df={ms:.4} reject={rej_s:.4}"
    );
    assert!((0.9..=1.1).contains(&mn), "normal M2/df off: {mn}");
    assert!(rej_n < 0.12, "normal Type I too high: {rej_n}");
    assert!(
        ms > mn + 0.1 && rej_s > rej_n,
        "skew misfit not detected: {ms} vs {mn}"
    );
}

#[test]
fn poly_ld_matches_direct_2x2_at_k2() {
    // Deterministic anchor: at K=2 the polytomous LD X² for each pair must
    // equal a from-scratch 2x2 Pearson chi-square of observed counts vs the
    // model-implied joint on the same quadrature — validating the table
    // assembly, the local-independence marginalization, and the chi-square.
    use crate::poly::{gpcm_logprobs, PolyModel};
    let (n_persons, n_items) = (600usize, 3usize);
    let mut st = 13131u64;
    let mut u = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let a = vec![1.1_f64, 0.9, 1.3];
    let b = vec![0.3_f64, -0.4, 0.1]; // K=2 GPCM intercept per item
    let mut yi = vec![0usize; n_persons * n_items];
    for pp in 0..n_persons {
        let u1 = u().max(1e-12);
        let u2 = u();
        let th = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        for i in 0..n_items {
            let pr = 1.0 / (1.0 + (-(a[i] * th + b[i])).exp());
            yi[pp * n_items + i] = if u() < pr { 1 } else { 0 };
        }
    }
    let observed = vec![true; yi.len()];
    let r = poly_local_dependence(
        &yi,
        Some(&observed),
        n_persons,
        n_items,
        2,
        &a,
        &b,
        PolyModel::Gpcm,
        41,
    )
    .unwrap();
    assert_eq!(r.df, 1.0);
    let (nodes, weights) = gh_rule(41).unwrap();
    let pcat = |i: usize, t: usize| -> [f64; 2] {
        let lp = gpcm_logprobs(a[i] * nodes[t], &[0.0, 1.0], &[0.0, b[i]]);
        [lp[0].exp(), lp[1].exp()]
    };
    for (idx, &(i, j)) in r.pairs.iter().enumerate() {
        let mut pj = [[0.0_f64; 2]; 2];
        for t in 0..nodes.len() {
            let (pi, pjj) = (pcat(i, t), pcat(j, t));
            for aa in 0..2 {
                for bb in 0..2 {
                    pj[aa][bb] += weights[t] * pi[aa] * pjj[bb];
                }
            }
        }
        let mut o = [[0.0_f64; 2]; 2];
        for pp in 0..n_persons {
            o[yi[pp * n_items + i]][yi[pp * n_items + j]] += 1.0;
        }
        let nf = n_persons as f64;
        let mut x2ref = 0.0_f64;
        for aa in 0..2 {
            for bb in 0..2 {
                let e = nf * pj[aa][bb];
                if e > 1e-12 {
                    let d = o[aa][bb] - e;
                    x2ref += d * d / e;
                }
            }
        }
        assert!(
            (r.x2[idx] - x2ref).abs() < 1e-8,
            "pair ({i},{j}): poly {} vs direct 2x2 {}",
            r.x2[idx],
            x2ref
        );
    }
}

// GPCM Monte-Carlo for the LD X²: returns (mean X²/df over locally-INDEPENDENT
// pairs, their rejection rate, X²/df for the injected/target pair (0,1), its
// rejection rate, df). With `inject_ld` a shared specific factor couples items
// 0 and 1 (a testlet), which the LD X² for that pair should detect while the
// other pairs stay calibrated. A skewed ability is a population
// misspecification that inflates all pairs.
fn mc_poly_ld(
    reps: usize,
    n_persons: usize,
    skew: bool,
    inject_ld: bool,
) -> (f64, f64, f64, f64, f64) {
    use crate::poly::{fit_poly_unidim, gpcm_logprobs, PolyModel};
    let (n_items, k) = (5usize, 3usize);
    let z = k - 1;
    let a_true: Vec<f64> = (0..n_items).map(|i| 1.0 + 0.1 * i as f64).collect();
    let cat_true: Vec<f64> = (0..n_items)
        .flat_map(|i| vec![0.7 - 0.08 * i as f64, -0.7 + 0.08 * i as f64])
        .collect();
    let (mut ind_ratio, mut ind_rej, mut ind_cnt) = (0.0_f64, 0usize, 0usize);
    let (mut ld_ratio, mut ld_rej) = (0.0_f64, 0usize);
    let mut df_val = 0.0_f64;
    for rep in 0..reps {
        let mut st = 5150u64 + rep as u64 * 131 + (skew as u64) * 7 + (inject_ld as u64) * 101;
        let mut u = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((st >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut yi = vec![0usize; n_persons * n_items];
        for pp in 0..n_persons {
            let theta = if skew {
                -(u().max(1e-12)).ln() - 1.0
            } else {
                let u1 = u().max(1e-12);
                let u2 = u();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            };
            // shared specific factor coupling items 0 and 1 (testlet LD)
            let uij = if inject_ld {
                let u1 = u().max(1e-12);
                let u2 = u();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            } else {
                0.0
            };
            for i in 0..n_items {
                let extra = if inject_ld && (i == 0 || i == 1) {
                    uij
                } else {
                    0.0
                };
                let base = a_true[i] * theta + extra;
                let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
                let mut ic = vec![0.0_f64; k];
                ic[1..].copy_from_slice(&cat_true[i * z..(i + 1) * z]);
                let lp = gpcm_logprobs(base, &scores, &ic);
                let draw = u();
                let (mut acc, mut cat) = (0.0_f64, k - 1);
                for (c, l) in lp.iter().enumerate() {
                    acc += l.exp();
                    if draw <= acc {
                        cat = c;
                        break;
                    }
                }
                yi[pp * n_items + i] = cat;
            }
        }
        // LD is evaluated at the FITTED parameters (the operational case): the
        // marginal MLE absorbs the univariate margins, leaving the (K-1)²
        // residual-association dof the statistic references.
        let fit = fit_poly_unidim(
            &yi,
            None,
            n_persons,
            n_items,
            k,
            PolyModel::Gpcm,
            21,
            80,
            1e-6,
        )
        .unwrap();
        assert!(
            fit.converged,
            "polytomous LD replicate {rep} did not converge: reason={}, \
             n_iter={}/{}, delta={:.6e}, tolerance={:.6e}",
            fit.termination_reason, fit.n_iter, 80, fit.final_delta, fit.stopping_tolerance
        );
        let cp_flat: Vec<f64> = fit.cat_params.iter().flatten().copied().collect();
        let r = poly_local_dependence(
            &yi,
            None,
            n_persons,
            n_items,
            k,
            &fit.slope,
            &cp_flat,
            PolyModel::Gpcm,
            21,
        )
        .unwrap();
        df_val = r.df;
        for (idx, &(i, j)) in r.pairs.iter().enumerate() {
            let ratio = r.x2[idx] / r.df;
            let rej = r.p_value[idx] < 0.05;
            if (i, j) == (0, 1) {
                ld_ratio += ratio;
                ld_rej += rej as usize;
            } else if i >= 2 && j >= 2 {
                // pairs among the untouched items 2..; testlet-touching pairs excluded
                ind_ratio += ratio;
                ind_rej += rej as usize;
                ind_cnt += 1;
            }
        }
    }
    (
        ind_ratio / ind_cnt as f64,
        ind_rej as f64 / ind_cnt as f64,
        ld_ratio / reps as f64,
        ld_rej as f64 / reps as f64,
        df_val,
    )
}

#[test]
fn poly_ld_calibration_and_power() {
    // Fast CI guard (fits each dataset). Authoritative >=500-rep study is
    // poly_ld_monte_carlo_500 (ignored). "clean" = pairs among the untouched
    // items 2.. ; "pair01" = the item pair carrying the injected testlet.
    let (reps, n) = (20usize, 1500usize);
    let (c0, r0, t0, _, df) = mc_poly_ld(reps, n, false, false); // null, normal ability
    let (cl, rl, tl, tlrej, _) = mc_poly_ld(reps, n, false, true); // testlet on (0,1)
    let (cs, rs, _, _, _) = mc_poly_ld(reps, n, true, false); // skewed ability
    println!(
        "[poly LD] df={df}  null: clean X2/df={c0:.3} reject={r0:.3} pair01={t0:.3}  \
         LD: clean={cl:.3} reject={rl:.3} pair01 X2/df={tl:.3} reject={tlrej:.3}  \
         skew: clean={cs:.3} reject={rs:.3}"
    );
    // null: clean pairs calibrated (the Chen-Thissen reference is conservative)
    assert!((0.45..=1.35).contains(&c0), "null clean X2/df off: {c0}");
    assert!(r0 < 0.15, "null rejection too high: {r0}");
    // power: the testlet pair (0,1) is flagged; clean pairs stay calibrated
    assert!(
        tl > 3.0 && tlrej > 0.6,
        "LD pair not detected: X2/df={tl}, reject={tlrej}"
    );
    assert!(
        cl < 1.6 && rl < 0.20,
        "clean pairs inflated under LD: {cl}, {rl}"
    );
    // a skewed ability that the N(0,1)-quadrature model cannot match inflates
    // the pairwise residual association (a detectable distribution misfit)
    assert!(cs > 2.0, "skew misspecification should inflate LD: {cs}");
}
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn poly_ld_monte_carlo_500() {
    let (reps, n) = (500usize, 2000usize);
    let (c0, r0, _, _, df) = mc_poly_ld(reps, n, false, false);
    let (cl, rl, tl, tlrej, _) = mc_poly_ld(reps, n, false, true);
    println!(
        "[poly LD 500] df={df}  null: clean X2/df={c0:.4} reject={r0:.4}  \
         LD: clean X2/df={cl:.4} reject={rl:.4} pair01 X2/df={tl:.4} reject={tlrej:.4}"
    );
    assert!((0.6..=1.15).contains(&c0), "null clean X2/df off: {c0}");
    assert!(r0 < 0.09, "null Type I not conservative: {r0}");
    // a 2-item testlet biases the whole unidimensional fit, so clean pairs are
    // mildly elevated, but the LD pair is localized far above them
    assert!(cl < 1.6, "clean pairs too inflated under LD: {cl}");
    assert!(
        tl > 6.0 && tlrej > 0.95 && tl > 4.0 * cl,
        "LD pair power/separation too low: pair01={tl} clean={cl} reject={tlrej}"
    );
}
