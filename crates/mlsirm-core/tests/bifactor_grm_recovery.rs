//! True-parameter recovery for the single-group polytomous bifactor GRM
//! (stage 1 of #1912).
//!
//! Data are generated from the MODEL DEFINITION
//! (`P(Y >= k) = logistic(a_G * theta_G + a_S * theta_S + d_k)`, Gibbons et al.,
//! 2007; Gibbons & Hedeker, 1992; Samejima, 1969) with an independent
//! simulation RNG — never from crate internals — so the test fails if the
//! estimator drifts from the published model. Design: 14 items, 3 specifics
//! (4/5/5 items), at least 3 negative general slopes, 4 ordered categories,
//! `n = 2,000` and `n = 1,020` (the reanalysis sample size in #1912).
//!
//! Tolerance basis (measured, not tuned): fitting the same DGP at five
//! simulation seeds (11/22/33/44/55) with the production config
//! (`q_general = 21`, `q_specific = 15`, `tol = 1e-5`) gave worst-over-seeds
//! absolute errors of `a_G = 0.215`, `a_S = 0.249`, `d = 0.196` at
//! `n = 2,000` and `a_G = 0.185`, `a_S = 0.398`, `d = 0.349` at `n = 1,020`
//! (all converged in 11-12 EM iterations; EAP(`theta_G`) correlation ~0.89).
//! The asserted tolerances add margin on top of that spread, and the
//! `n = 1,020` bands additionally cover the expected `sqrt(2000/1020) ~= 1.40`
//! sampling-noise ratio. One of the five `n = 2,000` seeds promoted a
//! negative slope to largest-magnitude, exercising the alignment path.
//!
//! # References (APA 7th ed.)
//!
//! Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E., Bhaumik,
//! D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover, A. (2007).
//! Full-information item bifactor analysis of graded response data. *Applied
//! Psychological Measurement, 31*(1), 4-19.
//! https://doi.org/10.1177/0146621606289485
//!
//! Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor
//! analysis. *Psychometrika, 57*(3), 423-436.
//! https://doi.org/10.1007/BF02295430
//!
//! Samejima, F. (1969). Estimation of latent ability using a response pattern
//! of graded scores. *Psychometrika, 34*(S1), 1-97.
//! https://doi.org/10.1007/BF03372160

use mlsirm_core::bifactor_grm::{fit_bifactor_grm, BifactorGrmConfig};

const N_ITEMS: usize = 14;
const N_SPECIFIC: usize = 3;
const N_CAT: usize = 4;
/// Items 0-3 -> specific 0, items 4-8 -> specific 1, items 9-13 -> specific 2.
const SPECIFIC_MAP: [i32; N_ITEMS] = [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2];

/// Three negative general slopes (items 1, 4, 9); the largest magnitude
/// (item 0, 1.60) is positive so the crate's reflection canonicalization
/// (largest-magnitude slope positive per dimension) is a no-op and recovered
/// signs are directly comparable to truth.
const TRUE_A_GENERAL: [f64; N_ITEMS] = [
    1.60, -1.20, 1.10, 0.90, -1.40, 1.30, 1.00, 0.80, 1.20, -1.10, 1.40, 1.00, 0.90, 1.20,
];
const TRUE_A_SPECIFIC: [f64; N_ITEMS] = [
    1.10, 0.90, 1.20, 0.80, 1.00, 1.30, 0.70, 0.90, 1.10, 0.80, 1.20, 1.00, 0.90, 0.70,
];
/// Strictly decreasing boundary intercepts per item (row-major).
const TRUE_D: [[f64; 3]; N_ITEMS] = [
    [1.30, 0.10, -1.10],
    [1.10, -0.10, -1.30],
    [1.40, 0.20, -1.00],
    [1.00, 0.00, -1.20],
    [1.20, 0.10, -1.10],
    [1.50, 0.30, -0.90],
    [0.90, -0.20, -1.40],
    [1.10, 0.00, -1.20],
    [1.30, 0.20, -1.00],
    [1.00, -0.10, -1.30],
    [1.40, 0.10, -1.10],
    [1.20, 0.00, -1.20],
    [1.00, -0.20, -1.30],
    [1.10, 0.10, -1.10],
];

// --- tolerances (see module docs for the measurement basis) ---
const MAX_A_GENERAL_ERROR_N2000: f64 = 0.35;
const MAX_A_SPECIFIC_ERROR_N2000: f64 = 0.40;
const MAX_D_ERROR_N2000: f64 = 0.30;
const MAX_A_GENERAL_ERROR_N1020: f64 = 0.50;
const MAX_A_SPECIFIC_ERROR_N1020: f64 = 0.55;
const MAX_D_ERROR_N1020: f64 = 0.42;
const MIN_THETA_CORRELATION: f64 = 0.85;

struct Lcg(u64);

impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (((self.0 >> 11) as f64) + 0.5) / ((1u64 << 53) as f64)
    }

    fn standard_normal(&mut self) -> f64 {
        let u1 = self.uniform().clamp(1e-12, 1.0 - 1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Category probabilities written straight from the model definition:
/// `P(Y >= k | tG, tS) = logistic(aG * tG + aS * tS + d_k)` with
/// `P(Y >= 0) = 1`, `P(Y >= K) = 0`; cells are adjacent differences.
fn category_probs(a_g: f64, a_s: f64, d: &[f64; 3], t_g: f64, t_s: f64) -> [f64; 4] {
    let base = a_g * t_g + a_s * t_s;
    let mut ge = [1.0f64; 5];
    for k in 1..4 {
        ge[k] = sigmoid(base + d[k - 1]);
    }
    ge[4] = 0.0;
    [ge[0] - ge[1], ge[1] - ge[2], ge[2] - ge[3], ge[3] - ge[4]]
}

fn simulate(n_persons: usize, seed: u64) -> (Vec<usize>, Vec<f64>) {
    let mut rng = Lcg(seed);
    let mut y = vec![0usize; n_persons * N_ITEMS];
    let mut true_theta_g = vec![0.0f64; n_persons];
    for p in 0..n_persons {
        let t_g = rng.standard_normal();
        true_theta_g[p] = t_g;
        let mut t_s = [0.0f64; N_SPECIFIC];
        for slot in t_s.iter_mut() {
            *slot = rng.standard_normal();
        }
        for i in 0..N_ITEMS {
            let s = SPECIFIC_MAP[i] as usize;
            let probs = category_probs(
                TRUE_A_GENERAL[i],
                TRUE_A_SPECIFIC[i],
                &TRUE_D[i],
                t_g,
                t_s[s],
            );
            let mut draw = rng.uniform();
            let mut cat = N_CAT - 1;
            for (k, pr) in probs.iter().enumerate() {
                if draw < *pr {
                    cat = k;
                    break;
                }
                draw -= *pr;
            }
            y[p * N_ITEMS + i] = cat;
        }
    }
    (y, true_theta_g)
}

fn fit_config() -> BifactorGrmConfig {
    BifactorGrmConfig {
        q_general: 21,
        q_specific: 15,
        max_iter: 500,
        tol: 1e-5,
        n_starts: 1,
        seed: 0x9E37_79B9_7F4A_7C15,
        newton_iter: 10,
        ridge: 1e-8,
    }
}

#[allow(clippy::too_many_arguments)]
fn assert_recovery(
    n_persons: usize,
    seed: u64,
    max_a_general_error: f64,
    max_a_specific_error: f64,
    max_d_error: f64,
) -> (f64, f64, f64) {
    let (y, true_theta_g) = simulate(n_persons, seed);
    let fit = fit_bifactor_grm(
        &y,
        None,
        &SPECIFIC_MAP,
        n_persons,
        N_ITEMS,
        N_SPECIFIC,
        N_CAT,
        &fit_config(),
    )
    .expect("bifactor GRM recovery fit on well-conditioned simulated data must succeed");
    assert!(
        fit.converged,
        "recovery fit must converge (termination: {})",
        fit.termination_reason
    );

    // Reflection canonicalization check: the fit must pin each dimension so
    // its own largest-magnitude slope is positive (the crate rule). The truth
    // is canonical by construction, but sampling noise can promote a
    // different item to largest-magnitude in the ESTIMATE (observed on one of
    // five spread-measurement seeds, where a -1.40 true slope estimated at
    // -1.65 outranked the +1.60 anchor). Reflection is unidentified, so the
    // comparison below aligns the estimate to truth first and the errors are
    // measured after that alignment; what is asserted here is that the
    // estimator applied its own rule.
    let anchor_g = fit
        .a_general
        .iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.abs().total_cmp(&y.abs()))
        .map(|(i, _)| i)
        .expect("one general slope per item");
    assert!(
        fit.a_general[anchor_g] > 0.0,
        "general anchor must be canonicalized positive: {:?}",
        fit.a_general
    );
    for s in 0..N_SPECIFIC {
        let anchor = (0..N_ITEMS)
            .filter(|&i| SPECIFIC_MAP[i] as usize == s)
            .max_by(|&i, &j| fit.a_specific[i].abs().total_cmp(&fit.a_specific[j].abs()))
            .expect("each specific block is non-empty");
        assert!(
            fit.a_specific[anchor] > 0.0,
            "specific-{s} anchor must be canonicalized positive: {:?}",
            fit.a_specific
        );
    }

    let mut worst_a_g = 0.0f64;
    let mut worst_a_s = 0.0f64;
    let mut worst_d = 0.0f64;
    // Align the unidentified reflection to truth (general slopes + general
    // EAPs jointly; specifics block by block) before measuring errors.
    let mut a_g = fit.a_general.clone();
    let mut a_s = fit.a_specific.clone();
    let mut theta_eap = fit.theta_g_eap.clone();
    if (a_g[anchor_g] > 0.0) != (TRUE_A_GENERAL[anchor_g] > 0.0) {
        for v in a_g.iter_mut().chain(theta_eap.iter_mut()) {
            *v = -*v;
        }
    }
    for s in 0..N_SPECIFIC {
        let items: Vec<usize> = (0..N_ITEMS)
            .filter(|&i| SPECIFIC_MAP[i] as usize == s)
            .collect();
        let est_anchor = *items
            .iter()
            .max_by(|&&i, &&j| a_s[i].abs().total_cmp(&a_s[j].abs()))
            .expect("each specific block is non-empty");
        if (a_s[est_anchor] > 0.0) != (TRUE_A_SPECIFIC[est_anchor] > 0.0) {
            for &i in &items {
                a_s[i] = -a_s[i];
            }
        }
    }
    for i in 0..N_ITEMS {
        worst_a_g = worst_a_g.max((a_g[i] - TRUE_A_GENERAL[i]).abs());
        worst_a_s = worst_a_s.max((a_s[i] - TRUE_A_SPECIFIC[i]).abs());
        for (k, truth) in TRUE_D[i].iter().enumerate() {
            worst_d = worst_d.max((fit.threshold[i * (N_CAT - 1) + k] - truth).abs());
        }
    }
    assert!(
        worst_a_g <= max_a_general_error,
        "worst |a_G| error {worst_a_g:.4} exceeds {max_a_general_error} (n={n_persons}, seed={seed}); \
         estimated {:?}",
        fit.a_general
    );
    assert!(
        worst_a_s <= max_a_specific_error,
        "worst |a_S| error {worst_a_s:.4} exceeds {max_a_specific_error} (n={n_persons}, seed={seed}); \
         estimated {:?}",
        fit.a_specific
    );
    assert!(
        worst_d <= max_d_error,
        "worst |d| error {worst_d:.4} exceeds {max_d_error} (n={n_persons}, seed={seed})"
    );

    // Sign pattern of the general slopes AFTER alignment (reverse-keyed
    // items stay reversed relative to each other within the aligned frame).
    for i in 0..N_ITEMS {
        for j in (i + 1)..N_ITEMS {
            assert_eq!(
                a_g[i] * a_g[j] > 0.0,
                TRUE_A_GENERAL[i] * TRUE_A_GENERAL[j] > 0.0,
                "general-slope sign pattern must match truth for items {i}/{j} \
                 (true {:?}, estimated {:?})",
                TRUE_A_GENERAL,
                a_g
            );
        }
    }

    // General-factor EAP recovery (correlation is supplementary
    // order-preservation evidence; the slope/intercept errors above are the
    // parameter-recovery evidence). Uses the reflection-aligned EAPs.
    let mean_est: f64 = theta_eap.iter().sum::<f64>() / n_persons as f64;
    let mean_true: f64 = true_theta_g.iter().sum::<f64>() / n_persons as f64;
    let (mut cov, mut ve, mut vt) = (0.0f64, 0.0f64, 0.0f64);
    for p in 0..n_persons {
        cov += (theta_eap[p] - mean_est) * (true_theta_g[p] - mean_true);
        ve += (theta_eap[p] - mean_est).powi(2);
        vt += (true_theta_g[p] - mean_true).powi(2);
    }
    let corr = cov / (ve * vt).sqrt();
    assert!(
        corr >= MIN_THETA_CORRELATION,
        "general-factor EAP correlation {corr:.4} below {MIN_THETA_CORRELATION} \
         (n={n_persons}, seed={seed})"
    );
    (worst_a_g, worst_a_s, worst_d)
}

#[test]
fn bifactor_grm_recovers_true_parameters_at_n2000() {
    // Primary recovery condition. Five-seed spread measured during stage-1
    // development is reported in the PR; tolerances above add margin over the
    // worst observed per-parameter spread.
    assert_recovery(
        2_000,
        20_260_916,
        MAX_A_GENERAL_ERROR_N2000,
        MAX_A_SPECIFIC_ERROR_N2000,
        MAX_D_ERROR_N2000,
    );
}

#[test]
fn bifactor_grm_recovers_true_parameters_at_n1020() {
    // Reanalysis sample size (#1912: N = 1,020). Wider tolerances reflect the
    // sqrt(2000/1020) ~= 1.40 sampling-noise ratio on top of the n = 2,000
    // spread.
    assert_recovery(
        1_020,
        20_260_917,
        MAX_A_GENERAL_ERROR_N1020,
        MAX_A_SPECIFIC_ERROR_N1020,
        MAX_D_ERROR_N1020,
    );
}

/// #1929 regression: a fit at 121 nodes/dimension and a fit at 241
/// nodes/dimension (the maintainer's minimum-precision standard on this
/// fixture's DGP; see quadrature.rs module docs for the Golub & Welsch, 1969
/// generation method) must agree closely, since the marginal-likelihood
/// integral has converged well before either node count. This is a numerical
/// agreement check (both node counts vs each other), not a truth-recovery
/// check (`assert_recovery` above already covers that at the production
/// `q_general=21, q_specific=15`). n=500 keeps `--ignored` runtime bounded;
/// node count, not sample size, is what this test exercises.
/// Run with `cargo test --release -- --ignored --nocapture`.
#[test]
#[ignore = "slow (121/241 node grids); run with: cargo test --release -- --ignored --nocapture"]
fn bifactor_grm_121_vs_241_nodes_agree() {
    let n_persons = 500;
    let (y, _true_theta_g) = simulate(n_persons, 20_260_916);

    let mut loglik = [0.0f64; 2];
    let mut a_general = [Vec::new(), Vec::new()];
    let mut a_specific = [Vec::new(), Vec::new()];
    let mut threshold = [Vec::new(), Vec::new()];
    let mut elapsed_secs = [0.0f64; 2];

    for (idx, &q) in [121usize, 241usize].iter().enumerate() {
        let cfg = BifactorGrmConfig {
            q_general: q,
            q_specific: q,
            ..fit_config()
        };
        let start = std::time::Instant::now();
        let fit = fit_bifactor_grm(
            &y,
            None,
            &SPECIFIC_MAP,
            n_persons,
            N_ITEMS,
            N_SPECIFIC,
            N_CAT,
            &cfg,
        )
        .unwrap_or_else(|e| panic!("q={q} fit must succeed: {e}"));
        elapsed_secs[idx] = start.elapsed().as_secs_f64();
        assert!(
            fit.converged,
            "q={q} recovery fit must converge (termination: {})",
            fit.termination_reason
        );
        loglik[idx] = *fit.loglik_trace.last().expect("non-empty trace");
        a_general[idx] = fit.a_general.clone();
        a_specific[idx] = fit.a_specific.clone();
        threshold[idx] = fit.threshold.clone();
        eprintln!(
            "q={q}: n_iter={}, final_loglik={:.6}, elapsed={:.2}s",
            fit.n_iter, loglik[idx], elapsed_secs[idx]
        );
    }

    let loglik_diff = (loglik[0] - loglik[1]).abs();
    eprintln!(
        "121 vs 241 nodes: |loglik diff|={loglik_diff:.6}, \
         121 took {:.2}s, 241 took {:.2}s",
        elapsed_secs[0], elapsed_secs[1]
    );
    // Documented tolerance: EM stops at |delta loglik| < tol = 1e-5, so two
    // well-converged fits at different (already-stabilized, per the
    // maintainer's literature summary on #1929) node counts should agree to
    // a small multiple of that EM stopping tolerance, not to float epsilon
    // (independent EM runs land at slightly different points on a flat
    // likelihood ridge). 5e-3 is ~500x the EM tolerance, well inside that
    // band and far tighter than the truth-recovery tolerances above.
    assert!(
        loglik_diff < 5e-3,
        "loglik must agree closely between 121 and 241 nodes: {loglik_diff:.6}"
    );
    for i in 0..N_ITEMS {
        assert!(
            (a_general[0][i] - a_general[1][i]).abs() < 5e-3,
            "a_general[{i}] disagrees: 121={}, 241={}",
            a_general[0][i],
            a_general[1][i]
        );
        assert!(
            (a_specific[0][i] - a_specific[1][i]).abs() < 5e-3,
            "a_specific[{i}] disagrees: 121={}, 241={}",
            a_specific[0][i],
            a_specific[1][i]
        );
    }
    for i in 0..threshold[0].len() {
        assert!(
            (threshold[0][i] - threshold[1][i]).abs() < 5e-3,
            "threshold[{i}] disagrees: 121={}, 241={}",
            threshold[0][i],
            threshold[1][i]
        );
    }
}
