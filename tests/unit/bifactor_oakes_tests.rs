//! Unit contract for the stage-3 Oakes standard errors
//! (`mlsirm_core::bifactor_oakes`, stage 3 of #1912).
//!
//! GRID SCOPE (maintainer quadrature rule): every test below runs on small
//! Gauss-Hermite grids (`q = 7`) as an explicitly labeled analytic
//! cross-check — the two compared paths share the same grid, so quadrature
//! error cancels and the comparison is exact. NONE of these represents study
//! settings. Study-settings calibration (>= 121 GH nodes per dimension) is
//! `bifactor_oakes_calibration.rs::study_settings_se_converges_at_121_vs_241_nodes`.
//! Node counts are caller arguments everywhere — no test bypasses the
//! supported-grid validation.
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
use crate::bifactor_grm::{bifactor_grm_marginal_loglik, fit_bifactor_grm, BifactorGrmConfig};

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
            let mixed = (loglik_at(&xpp) - loglik_at(&xpm) - loglik_at(&xmp) + loglik_at(&xmm))
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
    // Design, sample size, quadrature, budgets, and BOTH seeds below are
    // TEST arguments (two seed pairs so a benign RNG change cannot flip
    // the PD outcome unnoticed).
    for (sim_seed, fit_seed) in [(0x5EED_1234u64, 777u64), (0x00C0_FFEEu64, 4242u64)] {
        let n_persons = 800usize;
        let (true_ag, true_as, true_th) = six_params();
        let y = simulate_from_model(
            n_persons,
            sim_seed,
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
            seed: fit_seed,
            newton_iter: 10,
            ridge: 1e-8,
            device: crate::Device::Cpu,
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
            "the six-item MLE must be positive definite (seeds {sim_seed:#X}/{fit_seed}); \
             reason: {:?}",
            res.non_pd_reason
        );
        assert!(
            res.non_pd_reason.is_none(),
            "a PD result must carry no reason"
        );
        let (vcov, se) = (
            res.vcov.expect("PD implies vcov"),
            res.se.expect("PD implies se"),
        );
        let k = res.labels.len();
        assert_eq!(vcov.len(), k * k);
        assert_eq!(se.len(), k);
        assert!(
            se.iter().all(|s| s.is_finite() && *s > 0.0),
            "SEs must be finite and positive; got {se:?}"
        );
        // vcov diagonals are variances: se == sqrt(diag(vcov)) exactly (the
        // assembly never clips; Cholesky success certifies PD).
        for j in 0..k {
            let expect = vcov[j * k + j].sqrt();
            assert!(
                (se[j] - expect).abs() <= 1e-12,
                "se[{j}] must equal sqrt(vcov[{j},{j}])"
            );
        }
    }
}

#[test]
fn non_pd_information_is_flagged_with_reason_and_no_substitute() {
    // Three respondents cannot identify 16 item parameters: the observed
    // information is singular, and the assembly must FLAG it with a reason
    // while returning `None` (never a substitute) for vcov/se. Three
    // respondents is the smallest sample covering every category of every
    // item (fewer would trip the unidentified-boundary validator instead).
    let (a_g, a_s, thresholds) = tiny_params();
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
    assert!(
        !res.positive_definite,
        "three respondents cannot identify 16 parameters; the flag must trip"
    );
    assert!(res.vcov.is_none(), "non-PD implies no vcov substitute");
    assert!(res.se.is_none(), "non-PD implies no SE substitute");
    let reason = res.non_pd_reason.expect("non-PD implies a reason");
    assert!(!reason.is_empty(), "the reason must be non-empty");
}

// ---------------------------------------------------------------------------
// General-only items (`specific_map == -1`): no `a_S` slot exists, the
// per-item node grid is general-only, and the Oakes information must still
// match the numerical marginal Hessian (same FD cross-check as the
// all-block tiny problem).
// ---------------------------------------------------------------------------

/// Item -> specific map with general-only items 0 and 2.
const MIXED_SPECIFIC_MAP: [i32; TINY_N_ITEMS] = [-1, 0, -1, 0];

fn mixed_params() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // `a_specific` must be exactly 0.0 on general-only items.
    let (a_g, _, thresholds) = tiny_params();
    let a_specific = vec![0.0, 1.1, 0.0, 0.6];
    (a_g, a_specific, thresholds)
}

#[test]
fn general_only_items_have_no_specific_slot_and_match_fd() {
    let (a_g, a_s, thresholds) = mixed_params();
    let (y, n_persons) = tiny_data();
    let cfg = tiny_oakes_config();
    let provider = Stage1Provider::new(
        &y,
        None,
        &MIXED_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        1,
        TINY_N_CAT,
        &cfg,
    )
    .expect("stage-1 provider with general-only items must build");
    // Free order: item 0 [a_G, d, d], item 1 [a_G, a_S, d, d], ...
    assert_eq!(
        provider.labels(),
        vec![
            "a_general:0",
            "d:0:0",
            "d:0:1",
            "a_general:1",
            "a_specific:1",
            "d:1:0",
            "d:1:1",
            "a_general:2",
            "d:2:0",
            "d:2:1",
            "a_general:3",
            "a_specific:3",
            "d:3:0",
            "d:3:1",
        ]
    );
    let k = provider.free_len();
    assert_eq!(k, 3 + 4 + 3 + 4);
    let res = bifactor_oakes_se(
        &a_g,
        &a_s,
        &thresholds,
        &y,
        None,
        &MIXED_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        1,
        TINY_N_CAT,
        &cfg,
    )
    .expect("Oakes SEs with general-only items must return Ok");
    assert_eq!(res.labels.len(), k);
    assert_eq!(res.information.len(), k * k);
    // Same marginal-Hessian cross-check as the all-block tiny problem,
    // over the mixed free vector (general-only items contribute 3 slots).
    let m1 = TINY_N_CAT - 1;
    let loglik_at = |p: &[f64], spec: &[ItemSpec]| -> f64 {
        let mut ag = vec![0.0; TINY_N_ITEMS];
        let mut as_ = vec![0.0; TINY_N_ITEMS];
        let mut th = vec![0.0; TINY_N_ITEMS * m1];
        for (i, s) in spec.iter().enumerate() {
            ag[i] = p[s.slots[0]];
            let mut off = 1;
            if s.has_specific {
                as_[i] = p[s.slots[1]];
                off = 2;
            }
            for j in 0..m1 {
                th[i * m1 + j] = p[s.slots[off + j]];
            }
        }
        bifactor_grm_marginal_loglik(
            &ag,
            &as_,
            &th,
            &y,
            None,
            &MIXED_SPECIFIC_MAP,
            n_persons,
            TINY_N_ITEMS,
            1,
            TINY_N_CAT,
            cfg.q_general,
            cfg.q_specific,
        )
        .expect("marginal loglik at perturbed params must evaluate")
    };
    let specs = provider.item_specs();
    let mut packed = vec![0.0f64; k];
    for (i, s) in specs.iter().enumerate() {
        packed[s.slots[0]] = a_g[i];
        let mut off = 1;
        if s.has_specific {
            packed[s.slots[1]] = a_s[i];
            off = 2;
        }
        for j in 0..m1 {
            packed[s.slots[off + j]] = thresholds[i * m1 + j];
        }
    }
    let h = 1e-5;
    let mut worst = 0.0f64;
    for j in 0..k {
        let hj = h * (1.0 + packed[j].abs());
        let mut xp = packed.clone();
        xp[j] += hj;
        let mut xm = packed.clone();
        xm[j] -= hj;
        let lp = loglik_at(&xp, &specs);
        let lm = loglik_at(&xm, &specs);
        let l0 = loglik_at(&packed, &specs);
        let fd_diag = -((lp - 2.0 * l0 + lm) / (hj * hj));
        let o = res.information[j * k + j];
        worst = worst.max((o - fd_diag).abs() / (1.0 + fd_diag.abs()));
    }
    assert!(
        worst <= 2e-2,
        "Oakes diagonal must match the marginal-Hessian diagonal with \
         general-only items present; worst scaled gap = {worst:.3e}"
    );
}

// ---------------------------------------------------------------------------
// Binary (`M = 1`) GRM cell Hessian: collapses to the logistic Hessian
// `-(r_0 + r_1) * s * (1 - s)`, and matches central FD of the cell gradient.
// ---------------------------------------------------------------------------

#[test]
fn binary_cell_hessian_matches_logistic_form_and_fd() {
    use crate::poly::{grm_node_gradient, grm_node_hessian};
    let (base, thr, counts) = (0.7f64, vec![0.2f64], vec![1.5f64, 2.5f64]);
    let (grand, row_sums, mat) = grm_node_hessian(base, &thr, &counts);
    let s = 1.0 / (1.0 + (-(base + thr[0])).exp());
    let expect = -(counts[0] + counts[1]) * s * (1.0 - s);
    assert!(
        (grand - expect).abs() <= 1e-12 * (1.0 + expect.abs()),
        "M=1 grand sum must equal the logistic Hessian; got {grand}, expect {expect}"
    );
    assert_eq!(row_sums.len(), 1);
    assert!((row_sums[0] - expect).abs() <= 1e-12 * (1.0 + expect.abs()));
    assert!((mat[0][0] - expect).abs() <= 1e-12 * (1.0 + expect.abs()));
    // FD cross-check of the cell gradient.
    let h = 1e-7;
    let f = |b: f64, d0: f64| grm_node_gradient(b, &[d0], &counts);
    let (gp, gtp) = f(base + h, thr[0]);
    let (gm, gtm) = f(base - h, thr[0]);
    let (gdp, _) = f(base, thr[0] + h);
    let (gdm, _) = f(base, thr[0] - h);
    let fd_bb = (gp - gm) / (2.0 * h);
    let fd_bd = (gdp - gdm) / (2.0 * h);
    let fd_db = (gtp[0] - gtm[0]) / (2.0 * h);
    let (_, gdtp) = f(base, thr[0] + h);
    let (_, gdtm) = f(base, thr[0] - h);
    let fd_dd = (gdtp[0] - gdtm[0]) / (2.0 * h);
    for (got, want, name) in [
        (grand, fd_bb, "d_base d_base"),
        (fd_bd, fd_db, "symmetry d_base d_thr"),
        (mat[0][0], fd_dd, "d_thr d_thr"),
    ] {
        assert!(
            (got - want).abs() <= 1e-5 * (1.0 + want.abs()),
            "binary cell Hessian {name}: got {got}, want {want}"
        );
    }
}

// ---------------------------------------------------------------------------
// `fd_step` validation: non-finite or non-positive steps are loud errors.
// ---------------------------------------------------------------------------

#[test]
fn rejects_non_positive_fd_steps() {
    let (y, n_persons) = tiny_data();
    for fd_step in [0.0, -1e-5, f64::NAN, f64::INFINITY] {
        let cfg = BifactorOakesConfig {
            q_general: 7,
            q_specific: 7,
            fd_step,
        };
        let err = Stage1Provider::new(
            &y,
            None,
            &TINY_SPECIFIC_MAP,
            n_persons,
            TINY_N_ITEMS,
            TINY_N_SPECIFIC,
            TINY_N_CAT,
            &cfg,
        )
        .expect_err("non-positive fd_step must fail loudly, never clamp");
        assert!(
            err.contains("fd_step"),
            "the error must name fd_step; got: {err}"
        );
    }
}
