//! Fixed-item parameter calibration (FIPC) for the polytomous bifactor GRM:
//! RED tests for `bifactor_grm::fit_bifactor_grm_fipc`.
//!
//! Design (Kim, 2006, MWU-MEM, eqs. 14-15, pp. 361-362; Paek & Young, 2005):
//! a focal group responds to anchor items (general/specific slopes and
//! boundary intercepts fixed at reference-calibration values) plus new items.
//! The fitter estimates the new-item parameters AND the focal latent
//! distribution — general-factor mean/variance and, where identified,
//! specific-factor variances — by MML-EM with the prior updated after every
//! M-step. No rescaling of the latent points after an EM cycle (Kim, 2006,
//! p. 362) and no reflection canonicalization: the fixed anchors pin the
//! orientation, including reverse-keyed anchors.
//!
//! Data are generated from the model definition
//! (`P(Y >= k) = logistic(a_G*theta_G + a_S*theta_S + d_k)`, Gibbons et al.,
//! 2007; Gibbons & Hedeker, 1992; Samejima, 1969) with an independent
//! simulation RNG — never from crate internals.
//!
//! Tests:
//! - `bifactor_fipc_recovers_shifted_general`: focal general `N(0.5, 1.2^2)`
//!   (Kim's middle shift), specifics `N(0, 1)`; asserts the general
//!   mean/variance, free-item, and loglik recovery.
//! - `bifactor_fipc_matches_concurrent_at_true_anchors`: anchors fixed at
//!   TRUE values; FIPC must agree with the stage-2 concurrent calibration
//!   (`fit_bifactor_grm_multigroup`, reference + focal rows, all items
//!   common) the way two MLEs of the same population quantities agree.
//! - `bifactor_fipc_reverse_keyed_anchors`: anchors include negative general
//!   AND negative specific slopes; fixed signs survive bit-exact.
//! - `bifactor_fipc_specific_variances_where_identified`: with
//!   `estimate_specific_vars`, a shifted specific variance is recovered.
//! - `bifactor_fipc_study_n1020` (`#[ignore]`, statistical-studies):
//!   study-representative run at the reanalysis sample size, >= 121 nodes
//!   per dimension (the maintainer's minimum-precision standard, #1929).
//!
//! Tolerance basis (measured, not tuned): recovery bands carry margin over
//! the observed estimation spread at these sample sizes (bifactor GRM
//! slopes ~0.1-0.2 SE at N = 500-1,000); the specific-variance band (0.25)
//! covers a measured 5-seed spread of 0.85-1.20 around truth 1.0. A
//! disagreement beyond tolerance is REPORTED (assertions print both
//! values), never tuned away. The concurrent-equivalence band
//! (0.30 params / 0.20 moments) additionally covers the structural difference
//! that concurrent pools reference+focal rows for the free items while FIPC
//! uses focal rows only; measured 3-seed gaps (seeds 303-305) peak at 0.222
//! (params), 0.141 (mean), 0.073 (SD).
//!
//! # References (APA 7th ed.)
//!
//! Kim, S. (2006). A comparative study of IRT fixed parameter calibration
//! methods. *Journal of Educational Measurement, 43*(4), 355-381.
//! https://doi.org/10.1111/j.1745-3984.2006.00021.x
//!
//! Paek, I., & Young, M. J. (2005). Investigation of student growth recovery
//! in a fixed-item linking procedure with a fixed-person prior distribution
//! for mixed-format test data. *Applied Measurement in Education, 18*(2),
//! 199-215. https://doi.org/10.1207/s15324818ame1802_4
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

use mlsirm_core::bifactor_grm::{
    fit_bifactor_grm, fit_bifactor_grm_fipc, fit_bifactor_grm_multigroup, BifactorFipcConfig,
    BifactorGrmConfig, BifactorMultigroupConfig,
};

const N_ITEMS: usize = 12;
const N_SPECIFIC: usize = 2;
const N_CAT: usize = 3;
/// Items 0-5 -> specific 0, items 6-11 -> specific 1; anchors are 0-7.
const SPECIFIC_MAP: [i32; N_ITEMS] = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
const N_ANCHOR: usize = 8;
const N_REF: usize = 800;
const N_FOC: usize = 500;
/// Larger focal group for the concurrent-equivalence test so the band
/// reflects estimator agreement rather than focal sampling noise.
const N_FOC_CONC: usize = 1000;
const FOCAL_MEAN: f64 = 0.5;
const FOCAL_SD: f64 = 1.2;

const TRUE_A_G: [f64; N_ITEMS] = [1.5, 1.2, 1.0, 0.9, 1.4, 1.1, 1.0, 0.8, 1.3, 1.1, 0.9, 1.2];
const TRUE_A_S: [f64; N_ITEMS] = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.0, 1.1, 0.8, 0.9, 1.2, 0.7];
const TRUE_D1: [f64; N_ITEMS] = [1.3, 1.1, 1.4, 1.0, 1.2, 1.5, 0.9, 1.1, 1.3, 1.0, 1.2, 0.8];
const TRUE_D2: [f64; N_ITEMS] = [-1.1, -1.3, -1.0, -1.2, -1.1, -0.9, -1.4, -1.2, -1.0, -1.1, -0.9, -1.3];

/// Reverse-keyed variant: anchors 1 (general) and 6 (general + specific)
/// keyed against the construct.
const RK_A_G: [f64; N_ITEMS] = [1.5, -1.2, 1.0, 0.9, 1.4, 1.1, -1.0, 0.8, 1.3, 1.1, 0.9, 1.2];
const RK_A_S: [f64; N_ITEMS] = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, -1.0, 1.1, 0.8, 0.9, 1.2, 0.7];

const MAX_MU_ERROR: f64 = 0.15;
const MAX_SD_ERROR: f64 = 0.15;
const MAX_FREE_A_ERROR: f64 = 0.45;
const MAX_FREE_D_ERROR: f64 = 0.45;
const MAX_CONCURRENT_PARAM_GAP: f64 = 0.30;
const MAX_CONCURRENT_MU_GAP: f64 = 0.20;

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

#[allow(clippy::too_many_arguments)]
fn simulate(
    a_g: &[f64; N_ITEMS],
    a_s: &[f64; N_ITEMS],
    n_persons: usize,
    general_mean: f64,
    general_sd: f64,
    specific_sd: &[f64; 2],
    seed: u64,
) -> Vec<usize> {
    let mut rng = Lcg(seed);
    let mut y = vec![0usize; n_persons * N_ITEMS];
    for p in 0..n_persons {
        let t_g = general_mean + general_sd * rng.standard_normal();
        let t_s = [
            specific_sd[0] * rng.standard_normal(),
            specific_sd[1] * rng.standard_normal(),
        ];
        for i in 0..N_ITEMS {
            let s = SPECIFIC_MAP[i] as usize;
            let base = a_g[i] * t_g + a_s[i] * t_s[s];
            let p1 = sigmoid(base + TRUE_D1[i]);
            let p2 = sigmoid(base + TRUE_D2[i]);
            let u = rng.uniform();
            // Inversion of the cumulative model with one uniform: Y >= 1
            // w.p. p1 (u < p1), Y >= 2 w.p. p2 (u < p2).
            y[p * N_ITEMS + i] = usize::from(u < p1) + usize::from(u < p2);
        }
    }
    y
}

fn ref_config() -> BifactorGrmConfig {
    BifactorGrmConfig {
        q_general: 21,
        q_specific: 11,
        max_iter: 500,
        tol: 1e-5,
        n_starts: 1,
        seed: 0x9E37_79B9_7F4A_7C15,
        newton_iter: 10,
        ridge: 1e-8,
        device: mlsirm_core::Device::Cpu,
    }
}

fn fipc_config() -> BifactorFipcConfig {
    BifactorFipcConfig {
        q_general: 21,
        q_specific: 11,
        max_iter: 500,
        tol: 1e-5,
        newton_iter: 10,
        ridge: 1e-8,
        estimate_specific_vars: false,
    }
}

fn anchor_flags() -> Vec<bool> {
    (0..N_ITEMS).map(|i| i < N_ANCHOR).collect()
}

#[test]
fn bifactor_fipc_recovers_shifted_general() {
    let y_ref = simulate(&TRUE_A_G, &TRUE_A_S, N_REF, 0.0, 1.0, &[1.0, 1.0], 101);
    let reference = fit_bifactor_grm(
        &y_ref, None, &SPECIFIC_MAP, N_REF, N_ITEMS, N_SPECIFIC, N_CAT, &ref_config(),
    )
    .expect("reference bifactor fit must succeed");
    assert!(reference.converged);

    let y_foc = simulate(&TRUE_A_G, &TRUE_A_S, N_FOC, FOCAL_MEAN, FOCAL_SD, &[1.0, 1.0], 202);
    let fit = fit_bifactor_grm_fipc(
        &y_foc,
        None,
        &SPECIFIC_MAP,
        N_FOC,
        N_ITEMS,
        N_SPECIFIC,
        N_CAT,
        &anchor_flags(),
        &reference.a_general,
        &reference.a_specific,
        &reference.threshold,
        &fipc_config(),
    )
    .expect("bifactor FIPC fit must succeed");
    assert!(fit.converged, "FIPC must converge: {}", fit.termination_reason);

    // Anchors bit-exact (fixed means fixed, both slopes and thresholds).
    for i in 0..N_ANCHOR {
        assert!((fit.a_general[i] - reference.a_general[i]).abs() == 0.0, "anchor a_G {i}");
        assert!((fit.a_specific[i] - reference.a_specific[i]).abs() == 0.0, "anchor a_S {i}");
        for k in 0..N_CAT - 1 {
            assert!(
                (fit.threshold[i * (N_CAT - 1) + k] - reference.threshold[i * (N_CAT - 1) + k]).abs()
                    == 0.0,
                "anchor d {i}[{k}]"
            );
        }
    }
    assert!((fit.general_mean - FOCAL_MEAN).abs() < MAX_MU_ERROR, "mean = {}", fit.general_mean);
    assert!((fit.general_sd - FOCAL_SD).abs() < MAX_SD_ERROR, "sd = {}", fit.general_sd);
    for s in &fit.specific_sd {
        assert!((*s - 1.0).abs() == 0.0, "specific SDs stay pinned at 1");
    }
    for i in N_ANCHOR..N_ITEMS {
        assert!((fit.a_general[i] - TRUE_A_G[i]).abs() < MAX_FREE_A_ERROR, "free a_G {i}");
        assert!((fit.a_specific[i] - TRUE_A_S[i]).abs() < MAX_FREE_A_ERROR, "free a_S {i}");
        assert!((fit.threshold[i * 2] - TRUE_D1[i]).abs() < MAX_FREE_D_ERROR, "free d1 {i}");
        assert!((fit.threshold[i * 2 + 1] - TRUE_D2[i]).abs() < MAX_FREE_D_ERROR, "free d2 {i}");
    }
    assert!(fit.loglik_trace.iter().all(|v| v.is_finite()));
}

#[test]
fn bifactor_fipc_matches_concurrent_at_true_anchors() {
    // Anchors fixed at TRUE values; concurrent = reference + focal rows with
    // every item common (stage-2 compact model).
    let fixed_a_g: Vec<f64> = TRUE_A_G.to_vec();
    let fixed_a_s: Vec<f64> = TRUE_A_S.to_vec();
    let fixed_d: Vec<f64> = (0..N_ITEMS).flat_map(|i| [TRUE_D1[i], TRUE_D2[i]]).collect();

    let y_foc = simulate(&TRUE_A_G, &TRUE_A_S, N_FOC_CONC, FOCAL_MEAN, FOCAL_SD, &[1.0, 1.0], 303);
    let fipc = fit_bifactor_grm_fipc(
        &y_foc,
        None,
        &SPECIFIC_MAP,
        N_FOC_CONC,
        N_ITEMS,
        N_SPECIFIC,
        N_CAT,
        &anchor_flags(),
        &fixed_a_g,
        &fixed_a_s,
        &fixed_d,
        &fipc_config(),
    )
    .expect("FIPC at true anchors must succeed");
    assert!(fipc.converged);

    let y_ref = simulate(&TRUE_A_G, &TRUE_A_S, N_REF, 0.0, 1.0, &[1.0, 1.0], 404);
    let mut y_both = y_ref.clone();
    y_both.extend_from_slice(&y_foc);
    let mut group_id = vec![0usize; N_REF];
    group_id.extend(vec![1usize; N_FOC_CONC]);
    let anchor_all = vec![true; N_ITEMS];
    let conc = fit_bifactor_grm_multigroup(
        &y_both,
        None,
        &group_id,
        2,
        &SPECIFIC_MAP,
        N_REF + N_FOC_CONC,
        N_ITEMS,
        N_SPECIFIC,
        N_CAT,
        Some(&anchor_all),
        &BifactorMultigroupConfig {
            q_general: 21,
            q_specific: 11,
            max_iter: 500,
            tol: 1e-5,
            n_starts: 1,
            seed: 0x9E37_79B9_7F4A_7C15,
            newton_iter: 10,
            ridge: 1e-8,
            estimate_specific_vars: false,
            device: mlsirm_core::Device::Cpu,
        },
    )
    .expect("concurrent multigroup fit must succeed");
    assert!(conc.converged, "concurrent must converge: {}", conc.termination_reason);

    for i in N_ANCHOR..N_ITEMS {
        let (f, c) = (fipc.a_general[i], conc.a_general[1][i]);
        assert!((f - c).abs() < MAX_CONCURRENT_PARAM_GAP, "free a_G {i}: {f} vs {c}");
        let (f, c) = (fipc.a_specific[i], conc.a_specific[1][i]);
        assert!((f - c).abs() < MAX_CONCURRENT_PARAM_GAP, "free a_S {i}: {f} vs {c}");
        for k in 0..N_CAT - 1 {
            let (f, c) = (
                fipc.threshold[i * (N_CAT - 1) + k],
                conc.threshold[1][i * (N_CAT - 1) + k],
            );
            assert!((f - c).abs() < MAX_CONCURRENT_PARAM_GAP, "free d {i}[{k}]: {f} vs {c}");
        }
    }
    assert!(
        (fipc.general_mean - conc.general_mean[1]).abs() < MAX_CONCURRENT_MU_GAP,
        "mean: {} vs {}",
        fipc.general_mean,
        conc.general_mean[1]
    );
    assert!(
        (fipc.general_sd - conc.general_sd[1]).abs() < MAX_CONCURRENT_MU_GAP,
        "sd: {} vs {}",
        fipc.general_sd,
        conc.general_sd[1]
    );
}

#[test]
fn bifactor_fipc_reverse_keyed_anchors() {
    let y_ref = simulate(&RK_A_G, &RK_A_S, N_REF, 0.0, 1.0, &[1.0, 1.0], 505);
    let reference = fit_bifactor_grm(
        &y_ref, None, &SPECIFIC_MAP, N_REF, N_ITEMS, N_SPECIFIC, N_CAT, &ref_config(),
    )
    .expect("reference fit must succeed");
    assert!(reference.converged);

    let y_foc = simulate(&RK_A_G, &RK_A_S, N_FOC, FOCAL_MEAN, FOCAL_SD, &[1.0, 1.0], 606);
    let fit = fit_bifactor_grm_fipc(
        &y_foc,
        None,
        &SPECIFIC_MAP,
        N_FOC,
        N_ITEMS,
        N_SPECIFIC,
        N_CAT,
        &anchor_flags(),
        &reference.a_general,
        &reference.a_specific,
        &reference.threshold,
        &fipc_config(),
    )
    .expect("FIPC with reverse-keyed anchors must succeed");
    assert!(fit.converged);
    // Fixed signs survive bit-exact — anchors 1 and 6 stay negative on both
    // their general and (item 6) specific slopes.
    for i in 0..N_ANCHOR {
        assert!((fit.a_general[i] - reference.a_general[i]).abs() == 0.0, "anchor a_G {i}");
        assert!((fit.a_specific[i] - reference.a_specific[i]).abs() == 0.0, "anchor a_S {i}");
    }
    assert!(fit.a_general[1] < 0.0, "reverse-keyed general anchor keeps its sign");
    assert!(fit.a_general[6] < 0.0, "reverse-keyed general anchor keeps its sign");
    assert!(fit.a_specific[6] < 0.0, "reverse-keyed specific anchor keeps its sign");
    assert!((fit.general_mean - FOCAL_MEAN).abs() < MAX_MU_ERROR, "mean = {}", fit.general_mean);
    assert!((fit.general_sd - FOCAL_SD).abs() < MAX_SD_ERROR, "sd = {}", fit.general_sd);
}

#[test]
fn bifactor_fipc_specific_variances_where_identified() {
    // Focal specific-0 variance shifted to 1.44: with estimate_specific_vars
    // the FIPC M-step recovers it; specific 1 stays at 1.
    let y_ref = simulate(&TRUE_A_G, &TRUE_A_S, N_REF, 0.0, 1.0, &[1.0, 1.0], 707);
    let reference = fit_bifactor_grm(
        &y_ref, None, &SPECIFIC_MAP, N_REF, N_ITEMS, N_SPECIFIC, N_CAT, &ref_config(),
    )
    .expect("reference fit must succeed");

    let y_foc = simulate(&TRUE_A_G, &TRUE_A_S, N_FOC, FOCAL_MEAN, FOCAL_SD, &[1.2, 1.0], 808);
    let cfg = BifactorFipcConfig { estimate_specific_vars: true, ..fipc_config() };
    let fit = fit_bifactor_grm_fipc(
        &y_foc,
        None,
        &SPECIFIC_MAP,
        N_FOC,
        N_ITEMS,
        N_SPECIFIC,
        N_CAT,
        &anchor_flags(),
        &reference.a_general,
        &reference.a_specific,
        &reference.threshold,
        &cfg,
    )
    .expect("FIPC with specific-variance estimation must succeed");
    assert!(fit.converged);
    assert!((fit.general_mean - FOCAL_MEAN).abs() < MAX_MU_ERROR, "mean = {}", fit.general_mean);
    assert!((fit.general_sd - FOCAL_SD).abs() < MAX_SD_ERROR, "sd = {}", fit.general_sd);
    // Band from a measured 5-seed spread (same reference anchors):
    // specific-1 SDs 1.20/1.07/1.05/0.85/0.90 around truth 1.0, so 0.25
    // covers the spread with margin.
    assert!((fit.specific_sd[0] - 1.2).abs() < 0.25, "specific-0 sd = {}", fit.specific_sd[0]);
    assert!((fit.specific_sd[1] - 1.0).abs() < 0.25, "specific-1 sd = {}", fit.specific_sd[1]);
}

#[test]
#[ignore = "slow (121-node grids); run with: cargo test --release -- --ignored --nocapture"]
fn bifactor_fipc_study_n1020() {
    // Study-representative: reanalysis sample size at the maintainer's
    // minimum-precision standard, >= 121 nodes per dimension (#1929 removed
    // the fixed quadrature-table cap that used to bound this test at 41; see
    // `bifactor_grm_recovery.rs::bifactor_grm_121_vs_241_nodes_agree` for the
    // node-count numerical-agreement evidence this floor is based on).
    let n_foc = 1020usize;
    let y_ref = simulate(&TRUE_A_G, &TRUE_A_S, 3000, 0.0, 1.0, &[1.0, 1.0], 909);
    let reference = fit_bifactor_grm(
        &y_ref,
        None,
        &SPECIFIC_MAP,
        3000,
        N_ITEMS,
        N_SPECIFIC,
        N_CAT,
        &BifactorGrmConfig { q_general: 121, q_specific: 121, tol: 1e-6, max_iter: 1000, ..ref_config() },
    )
    .expect("study reference fit must succeed");
    assert!(reference.converged);
    let y_foc = simulate(&TRUE_A_G, &TRUE_A_S, n_foc, FOCAL_MEAN, FOCAL_SD, &[1.0, 1.0], 910);
    let fit = fit_bifactor_grm_fipc(
        &y_foc,
        None,
        &SPECIFIC_MAP,
        n_foc,
        N_ITEMS,
        N_SPECIFIC,
        N_CAT,
        &anchor_flags(),
        &reference.a_general,
        &reference.a_specific,
        &reference.threshold,
        &BifactorFipcConfig {
            q_general: 121,
            q_specific: 121,
            tol: 1e-6,
            max_iter: 1000,
            ..fipc_config()
        },
    )
    .expect("study FIPC fit must succeed");
    assert!(fit.converged, "study fit must converge: {}", fit.termination_reason);
    assert!((fit.general_mean - FOCAL_MEAN).abs() < 0.10, "mean = {}", fit.general_mean);
    assert!((fit.general_sd - FOCAL_SD).abs() < 0.10, "sd = {}", fit.general_sd);
    for i in N_ANCHOR..N_ITEMS {
        assert!((fit.a_general[i] - TRUE_A_G[i]).abs() < 0.30, "free a_G {i}");
        assert!((fit.a_specific[i] - TRUE_A_S[i]).abs() < 0.30, "free a_S {i}");
    }
}
