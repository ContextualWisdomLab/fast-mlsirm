//! True-parameter recovery for the single-group polytomous two-tier GRM
//! (stage 4 of #1912), including the primary-factor correlation.
//!
//! Data are generated from the MODEL DEFINITION
//! (`P(Y >= k) = logistic(sum_p a_ip * theta_p + a_S * theta_S + d_k)` with
//! `theta_P ~ MVN(0, Phi)`, orthogonal `N(0, 1)` specifics; Cai, 2010; the
//! graded cumulative form of Cai, Yang, & Hansen, 2011, eq. 6-7) with an
//! independent simulation RNG — never from crate internals — so the test
//! fails if the estimator drifts from the published model. Design: 10 items,
//! `P = 2` correlated primaries (`rho = 0.40`) in simple structure (items
//! 0-4 on primary 0, items 5-9 on primary 1), `S = 2` specifics
//! cross-cutting the primary split (method-factor layout: S0 on items
//! 0, 1, 5, 6; S1 on items 2, 3, 4, 7, 8, 9), 4 ordered categories.
//!
//! Tolerance basis (measured, not tuned): fitting the same DGP at five
//! simulation seeds (11/22/33/44/55) with the production config
//! (`q_primary = 15`, `q_specific = 11`, `tol = 1e-5`) gave worst-over-seeds
//! absolute errors of `a_P = 0.314`, `a_S = 0.256`, `d = 0.207` and a worst
//! primary-correlation error of `0.071` (truth `rho = 0.40`; estimates
//! 0.329-0.394), all converged in 10-12 EM iterations. Primary-factor EAP
//! correlations on the committed seed measured 0.729/0.732 (the DGP carries
//! only five items per primary against strong specific slopes, so the
//! primary EAP reliability is genuinely lower than the 14-item stage-1
//! design's ~0.89); the asserted 0.70 floor encodes that with margin. The
//! asserted tolerances add margin on top of that spread. The primary-correlation
//! estimates sit slightly below truth at every seed (attenuation
//! consistent with fixed-grid quadrature at 15 nodes per primary
//! dimension); the 121-vs-241 convergence check after the quadrature
//! migration (below) will adjudicate whether the gap is numerical or
//! sampling noise.
//!
//! QUADRATURE (maintainer rule, 2026-09-16): node counts govern numerical
//! precision with a 121-per-dimension floor, chosen by precision convergence
//! (121 vs 241 vs ...). Node counts are required caller arguments with no
//! defaults (Project rule, #1929). The `q_primary = 15` / `q_specific = 11`
//! below predate the removal of the fixed `SUPPORTED_Q` table (#1929/#1945,
//! which replaced it with an arbitrary-`n` Golub & Welsch, 1969 rule); this
//! recovery test's tolerance bands are measured at those counts and are left
//! as-is, while `two_tier_grm_node_agreement.rs` adds the >= 121-node
//! numerical-agreement regression this comment used to defer. The counts
//! here are caller arguments (single knob in `fit_config`), not model constants.
//!
//! # References (APA 7th ed.)
//!
//! Cai, L. (2010). A two-tier full-information item factor analysis model
//! with applications. *Psychometrika, 75*(4), 581-612.
//! https://doi.org/10.1007/s11336-010-9178-0 (abstract read; full text not
//! accessible — no equation locator is drawn from it)
//!
//! Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information
//! item bifactor analysis. *Psychological Methods, 16*(3), 221-248.
//! https://doi.org/10.1037/a0023350 (full text read: eq. 6-7)

use mlsirm_core::two_tier_grm::{fit_two_tier_grm, TwoTierGrmConfig};

const N_ITEMS: usize = 10;
const N_PRIMARY: usize = 2;
const N_SPECIFIC: usize = 2;
const N_CAT: usize = 4;
const TRUE_RHO: f64 = 0.40;

/// Row-major `n_items x n_primary` free-slope pattern: simple structure.
const PRIMARY_MAP: [bool; N_ITEMS * N_PRIMARY] = [
    true, false, //
    true, false, //
    true, false, //
    true, false, //
    true, false, //
    false, true, //
    false, true, //
    false, true, //
    false, true, //
    false, true,
];
/// Specifics cross-cut the primary split (method-factor layout).
const SPECIFIC_MAP: [i32; N_ITEMS] = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1];

/// Row-major `n_items x n_primary` true primary slopes (exact 0 off-pattern).
const TRUE_A_PRIMARY: [[f64; N_PRIMARY]; N_ITEMS] = [
    [1.50, 0.0],
    [1.20, 0.0],
    [1.00, 0.0],
    [0.90, 0.0],
    [1.30, 0.0],
    [0.0, 1.40],
    [0.0, 1.10],
    [0.0, 1.00],
    [0.0, 0.80],
    [0.0, 1.20],
];
const TRUE_A_SPECIFIC: [f64; N_ITEMS] =
    [1.00, 0.90, 1.10, 0.80, 0.70, 1.00, 1.20, 0.90, 0.70, 0.80];
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
];

// Bands from the five-seed spread (see module docs) with margin.
const MAX_A_PRIMARY_ERROR: f64 = 0.45;
const MAX_A_SPECIFIC_ERROR: f64 = 0.40;
const MAX_D_ERROR: f64 = 0.30;
const MAX_RHO_ERROR: f64 = 0.12;
const MIN_THETA_CORRELATION: f64 = 0.70;

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

fn simulate(n_persons: usize, seed: u64) -> (Vec<usize>, Vec<f64>) {
    let mut rng = Lcg(seed);
    let rho = TRUE_RHO;
    let w = (1.0 - rho * rho).sqrt();
    let mut y = vec![0usize; n_persons * N_ITEMS];
    let mut true_theta = vec![0.0f64; n_persons * N_PRIMARY];
    for p in 0..n_persons {
        // Correlated primaries via the Cholesky factor of Phi (unit-variance
        // primaries with correlation rho; mirt `bfactor` COV convention).
        let z0 = rng.standard_normal();
        let z1 = rng.standard_normal();
        let t0 = z0;
        let t1 = rho * z0 + w * z1;
        true_theta[p * N_PRIMARY] = t0;
        true_theta[p * N_PRIMARY + 1] = t1;
        let t = [t0, t1];
        let mut t_s = [0.0f64; N_SPECIFIC];
        for slot in t_s.iter_mut() {
            *slot = rng.standard_normal();
        }
        for i in 0..N_ITEMS {
            let mut base = 0.0f64;
            for d in 0..N_PRIMARY {
                base += TRUE_A_PRIMARY[i][d] * t[d];
            }
            base += TRUE_A_SPECIFIC[i] * t_s[SPECIFIC_MAP[i] as usize];
            let mut ge = [1.0f64; 5];
            for k in 1..N_CAT {
                ge[k] = 1.0 / (1.0 + (-(base + TRUE_D[i][k - 1])).exp());
            }
            ge[N_CAT] = 0.0;
            let mut draw = rng.uniform();
            let mut cat = N_CAT - 1;
            for k in 0..N_CAT {
                let pr = ge[k] - ge[k + 1];
                if draw < pr {
                    cat = k;
                    break;
                }
                draw -= pr;
            }
            y[p * N_ITEMS + i] = cat;
        }
    }
    (y, true_theta)
}

fn fit_config() -> TwoTierGrmConfig {
    TwoTierGrmConfig {
        q_primary: 15,
        q_specific: 11,
        max_iter: 1000,
        tol: 1e-5,
        n_starts: 1,
        seed: 0x9E37_79B9_7F4A_7C15,
        newton_iter: 10,
        ridge: 1e-8,
        e_step_n_chunks: 1,
        e_step_n_threads: 1,
    }
}

fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma: f64 = a.iter().sum::<f64>() / n;
    let mb: f64 = b.iter().sum::<f64>() / n;
    let (mut cov, mut va, mut vb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        cov += (x - ma) * (y - mb);
        va += (x - ma).powi(2);
        vb += (y - mb).powi(2);
    }
    cov / (va * vb).sqrt()
}

#[test]
fn two_tier_grm_recovers_true_parameters_including_primary_correlation() {
    let n_persons = 2_000usize;
    let (y, true_theta) = simulate(n_persons, 24_191_204);
    let fit = fit_two_tier_grm(
        &y,
        None,
        &PRIMARY_MAP,
        &SPECIFIC_MAP,
        n_persons,
        N_ITEMS,
        N_PRIMARY,
        N_SPECIFIC,
        N_CAT,
        &fit_config(),
    )
    .expect("two-tier GRM recovery fit on well-conditioned simulated data must succeed");
    assert!(
        fit.converged,
        "recovery fit must converge (termination: {})",
        fit.termination_reason
    );

    // Reflection canonicalization check: each primary dimension's own
    // largest-magnitude slope must be positive (the crate rule, shared with
    // `poly::canonicalize_slope_reflection` and stage-1).
    for d in 0..N_PRIMARY {
        let anchor = (0..N_ITEMS)
            .filter(|&i| PRIMARY_MAP[i * N_PRIMARY + d])
            .max_by(|&i, &j| {
                fit.a_primary[i * N_PRIMARY + d]
                    .abs()
                    .total_cmp(&fit.a_primary[j * N_PRIMARY + d].abs())
            })
            .expect("each primary loads at least one item");
        assert!(
            fit.a_primary[anchor * N_PRIMARY + d] > 0.0,
            "primary-{d} anchor must be canonicalized positive: {:?}",
            fit.a_primary
        );
    }

    // Align the unidentified reflections to truth before measuring errors.
    let mut a_p = fit.a_primary.clone();
    let mut a_s = fit.a_specific.clone();
    for d in 0..N_PRIMARY {
        let items: Vec<usize> = (0..N_ITEMS)
            .filter(|&i| PRIMARY_MAP[i * N_PRIMARY + d])
            .collect();
        let est_anchor = *items
            .iter()
            .max_by(|&&i, &&j| {
                a_p[i * N_PRIMARY + d]
                    .abs()
                    .total_cmp(&a_p[j * N_PRIMARY + d].abs())
            })
            .expect("each primary loads at least one item");
        if (a_p[est_anchor * N_PRIMARY + d] > 0.0) != (TRUE_A_PRIMARY[est_anchor][d] > 0.0) {
            for &i in &items {
                a_p[i * N_PRIMARY + d] = -a_p[i * N_PRIMARY + d];
            }
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

    let mut worst_a_p = 0.0f64;
    let mut worst_a_s = 0.0f64;
    let mut worst_d = 0.0f64;
    for i in 0..N_ITEMS {
        for d in 0..N_PRIMARY {
            if PRIMARY_MAP[i * N_PRIMARY + d] {
                worst_a_p = worst_a_p.max((a_p[i * N_PRIMARY + d] - TRUE_A_PRIMARY[i][d]).abs());
            }
        }
        worst_a_s = worst_a_s.max((a_s[i] - TRUE_A_SPECIFIC[i]).abs());
        for (k, truth) in TRUE_D[i].iter().enumerate() {
            worst_d = worst_d.max((fit.threshold[i * (N_CAT - 1) + k] - truth).abs());
        }
    }
    assert!(
        worst_a_p <= MAX_A_PRIMARY_ERROR,
        "worst |a_primary| error {worst_a_p:.4} exceeds {MAX_A_PRIMARY_ERROR}; estimated {:?}",
        fit.a_primary
    );
    assert!(
        worst_a_s <= MAX_A_SPECIFIC_ERROR,
        "worst |a_specific| error {worst_a_s:.4} exceeds {MAX_A_SPECIFIC_ERROR}; estimated {:?}",
        fit.a_specific
    );
    assert!(
        worst_d <= MAX_D_ERROR,
        "worst |d| error {worst_d:.4} exceeds {MAX_D_ERROR}"
    );

    // The primary correlation itself must recover (reflection-invariant:
    // joint flips of a primary dimension leave Phi's off-diagonal sign
    // consistent with the aligned slopes, so compare after alignment by
    // re-signing phi rows/cols for flipped dimensions — tracked implicitly:
    // phi is estimated jointly with the slopes, hence in the ESTIMATED frame;
    // map it to the truth frame via the same flips applied above).
    let mut rho_hat = fit.phi[1];
    for d in 0..N_PRIMARY {
        let items: Vec<usize> = (0..N_ITEMS)
            .filter(|&i| PRIMARY_MAP[i * N_PRIMARY + d])
            .collect();
        let est_anchor = *items
            .iter()
            .max_by(|&&i, &&j| {
                fit.a_primary[i * N_PRIMARY + d]
                    .abs()
                    .total_cmp(&fit.a_primary[j * N_PRIMARY + d].abs())
            })
            .expect("each primary loads at least one item");
        let flipped = (fit.a_primary[est_anchor * N_PRIMARY + d] > 0.0)
            != (TRUE_A_PRIMARY[est_anchor][d] > 0.0);
        if flipped {
            rho_hat = -rho_hat;
        }
    }
    let rho_error = (rho_hat - TRUE_RHO).abs();
    assert!(
        rho_error <= MAX_RHO_ERROR,
        "primary correlation error {rho_error:.4} exceeds {MAX_RHO_ERROR} \
         (truth {TRUE_RHO}, estimated-frame phi {:?})",
        fit.phi
    );

    // Primary-factor EAP recovery (supplementary; aligned frame per dim).
    let mut eap0: Vec<f64> = (0..n_persons)
        .map(|p| fit.theta_p_eap[p * N_PRIMARY])
        .collect();
    let mut eap1: Vec<f64> = (0..n_persons)
        .map(|p| fit.theta_p_eap[p * N_PRIMARY + 1])
        .collect();
    let tru0: Vec<f64> = (0..n_persons).map(|p| true_theta[p * N_PRIMARY]).collect();
    let tru1: Vec<f64> = (0..n_persons)
        .map(|p| true_theta[p * N_PRIMARY + 1])
        .collect();
    // Sign-align EAPs to truth (same unidentified reflection).
    for (eap, tru) in [(&mut eap0, &tru0), (&mut eap1, &tru1)] {
        if correlation(eap, tru) < 0.0 {
            for v in eap.iter_mut() {
                *v = -*v;
            }
        }
    }
    let c0 = correlation(&eap0, &tru0);
    let c1 = correlation(&eap1, &tru1);
    assert!(
        c0 >= MIN_THETA_CORRELATION && c1 >= MIN_THETA_CORRELATION,
        "primary EAP correlations ({c0:.4}, {c1:.4}) below {MIN_THETA_CORRELATION}"
    );
}
