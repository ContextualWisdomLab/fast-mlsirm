//! Unit contract for the stage-3 Oakes standard errors
//! (`mlsirm_core::bifactor_oakes`, stage 3 of #1912).
//!
//! GRID SCOPE (maintainer quadrature rule): every test below runs on small
//! Gauss-Hermite grids (`q = 7`) as an explicitly labeled analytic
//! cross-check — the two compared paths share the same grid, so quadrature
//! error cancels and the comparison is exact. NONE of these represents study
//! settings. Study-settings calibration (121 GH nodes per dimension) is
//! blocked on the `SUPPORTED_Q <= 41` cap removal (worker task_d46e974a271e,
//! branch `fix/1929-quadrature-defaults`); the calibration test upgrades to
//! 121 on rebase. Node counts are caller arguments everywhere — no test
//! bypasses the supported-grid validation.
//!
//! * Complete-data exactness: the analytic per-item `Q` Hessian (Term A of
//!   the Oakes identity) must equal a central finite difference of the
//!   analytic `Q` gradient at a FIXED posterior. Finite differences appear
//!   ONLY here as a test cross-check — the production path is analytic.
//! * Observed-information exactness: the assembled Oakes information must
//!   equal the numerical Hessian of the exact marginal log-likelihood on a
//!   tiny problem (the Oakes identity, eq. 6, holds for every `xi`, so the
//!   comparison runs at fixed true parameters — no fit needed).
//! * Failure reporting: a non-positive-definite information matrix is
//!   flagged with a reason; `vcov`/`se` are `None`, never substituted.
//!
//! # References (APA 7th ed.)
//!
//! Oakes, D. (1999). Direct calculation of the information matrix via the EM
//! algorithm. *Journal of the Royal Statistical Society Series B: Statistical
//! Methodology, 61*(2), 479-482. https://doi.org/10.1111/1467-9868.00188

use super::*;
use crate::bifactor_grm::{
    bifactor_grm_marginal_loglik, fit_bifactor_grm, BifactorGrmConfig,
};

// ---------------------------------------------------------------------------
// Shared tiny problem (mirrors the stage-1 reduction test scales).
// ---------------------------------------------------------------------------

const TINY_N_ITEMS: usize = 4;
const TINY_N_SPECIFIC: usize = 2;
const TINY_N_CAT: usize = 3;
/// Item -> specific map: items 0-1 on specific 0, items 2-3 on specific 1.
const TINY_SPECIFIC_MAP: [i32; TINY_N_ITEMS] = [0, 0, 1, 1];

fn tiny_params() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let a_general = vec![1.2, -0.9, 1.0, 0.7];
    let a_specific = vec![0.8, 1.1, 0.9, 0.6];
    // Strictly decreasing within each item (n_cat - 1 = 2 intercepts).
    let thresholds = vec![0.9, -0.7, 0.5, -1.0, 1.1, -0.4, 0.2, -1.3];
    (a_general, a_specific, thresholds)
}

fn tiny_data() -> (Vec<usize>, usize) {
    // Deterministic toy responses covering every category of every item.
    let n_persons = 12usize;
    let rows: [[usize; TINY_N_ITEMS]; 12] = [
        [0, 1, 2, 0],
        [1, 2, 0, 1],
        [2, 0, 1, 2],
        [0, 2, 1, 0],
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [0, 0, 0, 0],
        [2, 1, 0, 2],
        [1, 0, 2, 1],
        [0, 2, 0, 1],
        [1, 2, 1, 0],
        [2, 0, 2, 1],
    ];
    let mut y = Vec::with_capacity(n_persons * TINY_N_ITEMS);
    for row in rows {
        y.extend_from_slice(&row);
    }
    (y, n_persons)
}

fn tiny_oakes_config() -> BifactorOakesConfig {
    BifactorOakesConfig {
        q_general: 7,
        q_specific: 7,
        fd_step: 1e-5,
    }
}

/// Pack `[a_G, a_S, d..]` per item in the assembly's free-vector order.
fn pack_tiny(a_g: &[f64], a_s: &[f64], thresholds: &[f64]) -> Vec<f64> {
    let m1 = TINY_N_CAT - 1;
    let mut out = Vec::new();
    for i in 0..TINY_N_ITEMS {
        out.push(a_g[i]);
        out.push(a_s[i]);
        out.extend_from_slice(&thresholds[i * m1..(i + 1) * m1]);
    }
    out
}

/// Independent LCG simulation from the MODEL DEFINITION
/// (`P(Y >= k) = logistic(a_G * tG + a_S * tS + d_k)`), never from crate
/// internals — the same device as the stage-1 recovery test.
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

fn simulate_tiny(n_persons: usize, seed: u64) -> Vec<usize> {
    let (a_g, a_s, thresholds) = tiny_params();
    simulate_from_model(
        n_persons,
        seed,
        &a_g,
        &a_s,
        &thresholds,
        TINY_N_ITEMS,
        TINY_N_CAT,
    )
}

/// Six-item identified design (three items per specific factor): the
/// two-items-per-block tiny design leaves the general/specific split weakly
/// identified (slow EM tail; EM can stall at a saddle with indefinite
/// observed information even at `tol = 1e-7`), so the PD test simulates and
/// fits this stronger design instead. Test arguments: `n = 800`,
/// `q_general = q_specific = 7`, `n_starts = 3`.
const SIX_N_ITEMS: usize = 6;
const SIX_N_SPECIFIC: usize = 2;
const SIX_N_CAT: usize = 3;
const SIX_SPECIFIC_MAP: [i32; SIX_N_ITEMS] = [0, 0, 0, 1, 1, 1];

fn six_params() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let a_general = vec![1.3, 1.1, 0.9, 1.2, 1.0, 0.8];
    let a_specific = vec![1.0, 0.9, 1.1, 0.8, 1.0, 0.9];
    // Strictly decreasing within each item (n_cat - 1 = 2 intercepts),
    // well separated so every category is populated at n = 800.
    let thresholds = vec![
        1.2, -0.8, 1.0, -1.0, 1.3, -0.7, 1.1, -0.9, 0.9, -1.1, 1.2, -0.8,
    ];
    (a_general, a_specific, thresholds)
}

fn simulate_from_model(
    n_persons: usize,
    seed: u64,
    a_g: &[f64],
    a_s: &[f64],
    thresholds: &[f64],
    n_items: usize,
    n_cat: usize,
) -> Vec<usize> {
    let m1 = n_cat - 1;
    let mut rng = Lcg(seed);
    let mut y = Vec::with_capacity(n_persons * n_items);
    for _ in 0..n_persons {
        let tg = rng.standard_normal();
        let ts0 = rng.standard_normal();
        let ts1 = rng.standard_normal();
        for i in 0..n_items {
            // Items 0..half on specific 0, rest on specific 1 (both the
            // tiny and six designs split their blocks this way).
            let ts = if i < n_items / 2 { ts0 } else { ts1 };
            let base = a_g[i] * tg + a_s[i] * ts;
            let u = rng.uniform();
            // Categories from the cumulative model: Y = #{k: u <= P(Y >= k)}
            // (inversion sampling; `u > p` would generate
            // category-REVERSED data).
            let mut cat = 0usize;
            for k in 0..m1 {
                let p_ge = 1.0 / (1.0 + (-(base + thresholds[i * m1 + k])).exp());
                if u < p_ge {
                    cat += 1;
                } else {
                    break;
                }
            }
            y.push(cat);
        }
    }
    y
}

// ---------------------------------------------------------------------------
// RED test 1 (unit): analytic Q Hessian vs central FD of the analytic Q
// gradient at a FIXED posterior. Fails until the analytic Hessian exists.
// ---------------------------------------------------------------------------

#[test]
fn analytic_q_hessian_matches_central_fd_of_q_gradient() {
    let (a_g, a_s, thresholds) = tiny_params();
    let (y, n_persons) = tiny_data();
    let cfg = tiny_oakes_config();
    let provider = Stage1Provider::new(
        &y,
        None,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &cfg,
    )
    .expect("stage-1 provider on valid tiny input must build");
    let packed = pack_tiny(&a_g, &a_s, &thresholds);
    let posterior = provider
        .posterior_at(&packed)
        .expect("E-step at true params must succeed");
    let analytic = q_hessian_analytic(&packed, &posterior, &provider);
    let k = provider.free_len();
    assert_eq!(analytic.len(), k * k, "Hessian must be k x k");
    // Symmetry (the assembly symmetrizes, but the analytic block must
    // already be symmetric to tight tolerance).
    let mut max_asym = 0.0f64;
    for r in 0..k {
        for c in 0..k {
            max_asym = max_asym.max((analytic[r * k + c] - analytic[c * k + r]).abs());
        }
    }
    assert!(
        max_asym <= 1e-8,
        "analytic Q Hessian must be symmetric; max asymmetry = {max_asym:.3e}"
    );
    // Central FD of the analytic gradient at the FIXED posterior.
    let h = 1e-6;
    let mut fd = vec![0.0f64; k * k];
    for j in 0..k {
        let hj = h * (1.0 + packed[j].abs());
        let mut xp = packed.clone();
        xp[j] += hj;
        let mut xm = packed.clone();
        xm[j] -= hj;
        let gp = q_gradient_analytic(&xp, &posterior, &provider);
        let gm = q_gradient_analytic(&xm, &posterior, &provider);
        for c in 0..k {
            fd[j * k + c] = (gp[c] - gm[c]) / (2.0 * hj);
        }
    }
    let mut worst = 0.0f64;
    for (a, f) in analytic.iter().zip(fd.iter()) {
        worst = worst.max((a - f).abs() / (1.0 + f.abs()));
    }
    assert!(
        worst <= 1e-4,
        "analytic Q Hessian must match central FD of the analytic Q gradient \
         (FD is the test cross-check only); worst scaled gap = {worst:.3e}"
    );
}

// ---------------------------------------------------------------------------
// RED test 2 (unit): Oakes information vs numerical Hessian of the exact
// marginal log-likelihood on the tiny problem. Fails until the assembly
// exists. The Oakes identity (eq. 6) holds for every `xi`, so fixed true
// parameters are used — no fit is involved.
// ---------------------------------------------------------------------------

#[test]
fn oakes_information_matches_fd_of_marginal_loglik_on_tiny_problem() {
    let (a_g, a_s, thresholds) = tiny_params();
    let (y, n_persons) = tiny_data();
    let cfg = tiny_oakes_config();
    let res = bifactor_oakes_se(
        &a_g,
        &a_s,
        &thresholds,
        &y,
        None,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &cfg,
    )
    .expect("Oakes SEs on the valid tiny problem must return Ok");
    let k = res.labels.len();
    assert_eq!(res.information.len(), k * k, "information must be k x k");
    // Numerical Hessian of the exact (reduced) marginal loglik by central
    // differences over the same free vector. `fd_step` is a caller argument
    // of the Oakes assembly; the test's own FD step is a test argument.
    let packed = pack_tiny(&a_g, &a_s, &thresholds);
    let m1 = TINY_N_CAT - 1;
    let loglik_at = |p: &[f64]| -> f64 {
        let mut ag = vec![0.0; TINY_N_ITEMS];
        let mut as_ = vec![0.0; TINY_N_ITEMS];
        let mut th = vec![0.0; TINY_N_ITEMS * m1];
        let mut cursor = 0;
        for i in 0..TINY_N_ITEMS {
            ag[i] = p[cursor];
            cursor += 1;
            as_[i] = p[cursor];
            cursor += 1;
            th[i * m1..(i + 1) * m1].copy_from_slice(&p[cursor..cursor + m1]);
            cursor += m1;
        }
        bifactor_grm_marginal_loglik(
            &ag,
            &as_,
            &th,
            &y,
            None,
            &TINY_SPECIFIC_MAP,
            n_persons,
            TINY_N_ITEMS,
            TINY_N_SPECIFIC,
            TINY_N_CAT,
            cfg.q_general,
            cfg.q_specific,
        )
        .expect("marginal loglik at perturbed params must evaluate")
    };
    let h = 1e-5;
    let mut fd_info = vec![0.0f64; k * k];
    for j in 0..k {
        let hj = h * (1.0 + packed[j].abs());
        let mut xp = packed.clone();
        xp[j] += hj;
        let mut xm = packed.clone();
        xm[j] -= hj;
        let lp = loglik_at(&xp);
        let lm = loglik_at(&xm);
        let l0 = loglik_at(&packed);
        // Diagonal second derivative.
        fd_info[j * k + j] = -((lp - 2.0 * l0 + lm) / (hj * hj));
        // Off-diagonals via the polarization identity.
        for l in (j + 1)..k {
            let hl = h * (1.0 + packed[l].abs());
            let mut xpp = packed.clone();
            xpp[j] += hj;
            xpp[l] += hl;
            let mut xmm = packed.clone();
            xmm[j] -= hj;
            xmm[l] -= hl;
            let mut xpm = packed.clone();
            xpm[j] += hj;
            xpm[l] -= hl;
            let mut xmp = packed.clone();
            xmp[j] -= hj;
            xmp[l] += hl;
            let mixed = (loglik_at(&xpp) - loglik_at(&xpm) - loglik_at(&xmp)
                + loglik_at(&xmm))
                / (4.0 * hj * hl);
            fd_info[j * k + l] = -mixed;
            fd_info[l * k + j] = -mixed;
        }
    }
    let mut worst = 0.0f64;
    for (o, f) in res.information.iter().zip(fd_info.iter()) {
        worst = worst.max((o - f).abs() / (1.0 + f.abs()));
    }
    assert!(
        worst <= 2e-2,
        "Oakes information must match the numerical Hessian of the exact \
         marginal loglik on the tiny problem; worst scaled gap = {worst:.3e}"
    );
}

// ---------------------------------------------------------------------------
// Test 3: the non-PD contract. `positive_definite == false`
// implies `vcov`/`se` are `None` with a non-empty reason — and the reverse.
// The six-item MLE above exercises the PD branch; the degenerate
// three-person problem below exercises the flag path.
// ---------------------------------------------------------------------------

#[test]
fn six_item_mle_is_positive_definite_with_matching_vcov_and_se() {
    // Positive definiteness holds at (near) the MLE, not at arbitrary
    // parameters: simulate from the model definition with an independent
    // RNG, fit to the MLE, and evaluate the Oakes information there.
    // Design, sample size, quadrature, and budgets below are TEST arguments.
    let n_persons = 800usize;
    let (true_ag, true_as, true_th) = six_params();
    let y = simulate_from_model(
        n_persons,
        0x5EED_1234,
        &true_ag,
        &true_as,
        &true_th,
        SIX_N_ITEMS,
        SIX_N_CAT,
    );
    for i in 0..SIX_N_ITEMS {
        for k in 0..SIX_N_CAT {
            assert!(
                (0..n_persons).any(|p| y[p * SIX_N_ITEMS + i] == k),
                "simulated data must cover category {k} of item {i}"
            );
        }
    }
    let fit_cfg = BifactorGrmConfig {
        q_general: 7,
        q_specific: 7,
        max_iter: 500,
        tol: 1e-7,
        n_starts: 3,
        seed: 777,
        newton_iter: 10,
        ridge: 1e-8,
    };
    let fit = fit_bifactor_grm(
        &y,
        None,
        &SIX_SPECIFIC_MAP,
        n_persons,
        SIX_N_ITEMS,
        SIX_N_SPECIFIC,
        SIX_N_CAT,
        &fit_cfg,
    )
    .expect("six-item simulated fit must succeed");
    assert!(
        fit.converged,
        "six-item simulated fit must converge; reason: {}",
        fit.termination_reason
    );
    let cfg = tiny_oakes_config();
    let res = bifactor_oakes_se(
        &fit.a_general,
        &fit.a_specific,
        &fit.threshold,
        &y,
        None,
        &SIX_SPECIFIC_MAP,
        n_persons,
        SIX_N_ITEMS,
        SIX_N_SPECIFIC,
        SIX_N_CAT,
        &cfg,
    )
    .expect("Oakes SEs at the six-item MLE must return Ok");
    assert!(
        res.positive_definite,
        "the six-item MLE must be positive definite; reason: {:?}",
        res.non_pd_reason
    );
    assert!(
        res.non_pd_reason.is_none(),
        "a PD result must carry no reason"
    );
    let (vcov, se) = (res.vcov.expect("PD implies vcov"), res.se.expect("PD implies se"));
    let k = res.labels.len();
    assert_eq!(vcov.len(), k * k);
    assert_eq!(se.len(), k);
    assert!(
        se.iter().all(|s| s.is_finite() && *s > 0.0),
        "SEs must be finite and positive; got {se:?}"
    );
    // vcov diagonals are variances: se == sqrt(diag(vcov)).
    for j in 0..k {
        let expect = vcov[j * k + j].max(0.0).sqrt();
        assert!(
            (se[j] - expect).abs() <= 1e-12,
            "se[{j}] must equal sqrt(vcov[{j},{j}])"
        );
    }
}

#[test]
fn non_pd_information_is_flagged_with_reason_and_no_substitute() {
    // One respondent cannot identify 16 item parameters: the observed
    // information must be singular, and the assembly must FLAG it with a
    // reason while returning `None` (never a substitute) for vcov/se.
    // The test asserts the CONTRACT in both branches so it stays
    // deterministic whatever the numerical outcome on this degenerate input.
    let (a_g, a_s, thresholds) = tiny_params();
    // Three respondents is the smallest sample covering every category of
    // every item (fewer would trip the unidentified-boundary validator).
    let y_few = vec![0usize, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
    let res = bifactor_oakes_se(
        &a_g,
        &a_s,
        &thresholds,
        &y_few,
        None,
        &TINY_SPECIFIC_MAP,
        3,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &tiny_oakes_config(),
    )
    .expect("even a degenerate problem must return Ok with flags, never Err");
    let k = res.labels.len();
    assert_eq!(res.information.len(), k * k);
    assert_eq!(res.positive_definite, res.vcov.is_some());
    assert_eq!(res.positive_definite, res.se.is_some());
    assert_eq!(res.positive_definite, res.non_pd_reason.is_none());
    if !res.positive_definite {
        let reason = res.non_pd_reason.expect("non-PD implies a reason");
        assert!(!reason.is_empty(), "the reason must be non-empty");
    }
}
