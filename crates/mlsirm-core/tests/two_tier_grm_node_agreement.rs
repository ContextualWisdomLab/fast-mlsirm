//! #1929 regression: a two-tier GRM fit at 121 nodes/dimension and a fit at
//! 241 nodes/dimension (the maintainer's minimum-precision standard, chosen
//! by precision convergence; see `quadrature.rs` module docs for the
//! Golub & Welsch, 1969 arbitrary-`n` generation method that made these
//! counts reachable) must agree closely, since the marginal-likelihood
//! integral has converged well before either node count. This mirrors
//! `bifactor_grm_recovery.rs::bifactor_grm_121_vs_241_nodes_agree`.
//!
//! `two_tier_grm`'s primary grid is a PRODUCT grid of size
//! `q_primary^n_primary` (see the module docs on `TwoTierGrmConfig`), so a
//! two-primary design at `q = 241` (`241^2 = 58,081` primary nodes, times
//! `q_specific` per specific-block item) is computationally infeasible for a
//! local, single-threaded `--ignored` run. This test therefore uses a
//! single-primary, single-specific design (`n_primary = n_specific = 1`, so
//! the grid is `q_primary * q_specific` rather than `q_primary^2 *
//! q_specific`), which still exercises the SAME arbitrary-`n` Gauss-Hermite
//! path (`quadrature::gh_rule`) the production `n_primary = 2` config in
//! `two_tier_grm_recovery.rs` uses; node-count convergence is a property of
//! the quadrature rule, not of `n_primary`. `n_persons = 300` keeps
//! `--ignored` runtime bounded; node count, not sample size or dimension
//! count, is what this test exercises.
//! Run with `cargo test --release -- --ignored --nocapture`.

use mlsirm_core::two_tier_grm::{fit_two_tier_grm, TwoTierGrmConfig};

const N_ITEMS: usize = 4;
const N_PRIMARY: usize = 1;
const N_SPECIFIC: usize = 1;
const N_CAT: usize = 3;
const PRIMARY_MAP: [bool; N_ITEMS * N_PRIMARY] = [true, true, true, true];
const SPECIFIC_MAP: [i32; N_ITEMS] = [0, 0, 0, 0];
const TRUE_A_PRIMARY: [f64; N_ITEMS] = [1.20, 1.00, 0.90, 1.10];
const TRUE_A_SPECIFIC: [f64; N_ITEMS] = [0.80, 0.70, 0.90, 0.60];
const TRUE_D: [[f64; 2]; N_ITEMS] = [
    [1.00, -1.00],
    [0.80, -1.20],
    [1.10, -0.90],
    [0.90, -1.10],
];

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

fn simulate(n_persons: usize, seed: u64) -> Vec<usize> {
    let mut rng = Lcg(seed);
    let mut y = vec![0usize; n_persons * N_ITEMS];
    for pp in 0..n_persons {
        let t0 = rng.standard_normal();
        let t_s = rng.standard_normal();
        for i in 0..N_ITEMS {
            let base = TRUE_A_PRIMARY[i] * t0 + TRUE_A_SPECIFIC[i] * t_s;
            let mut ge = [1.0f64; 4];
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
            y[pp * N_ITEMS + i] = cat;
        }
    }
    y
}

fn fit_config() -> TwoTierGrmConfig {
    TwoTierGrmConfig {
        q_primary: 15,
        q_specific: 15,
        max_iter: 1000,
        tol: 1e-5,
        n_starts: 1,
        seed: 0x9E37_79B9_7F4A_7C15,
        newton_iter: 10,
        ridge: 1e-8,
    }
}

#[test]
#[ignore = "slow (121/241 node grids); run with: cargo test --release -- --ignored --nocapture"]
fn two_tier_grm_121_vs_241_nodes_agree() {
    let n_persons = 300;
    let y = simulate(n_persons, 20_260_917);

    let mut loglik = [0.0f64; 2];
    let mut a_primary = [Vec::new(), Vec::new()];
    let mut a_specific = [Vec::new(), Vec::new()];
    let mut threshold = [Vec::new(), Vec::new()];
    let mut elapsed_secs = [0.0f64; 2];

    for (idx, &q) in [121usize, 241usize].iter().enumerate() {
        let cfg = TwoTierGrmConfig {
            q_primary: q,
            q_specific: q,
            ..fit_config()
        };
        let start = std::time::Instant::now();
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
            &cfg,
        )
        .unwrap_or_else(|e| panic!("q={q} fit must succeed: {e}"));
        elapsed_secs[idx] = start.elapsed().as_secs_f64();
        assert!(
            fit.converged,
            "q={q} fit must converge (termination: {})",
            fit.termination_reason
        );
        loglik[idx] = *fit.loglik_trace.last().expect("non-empty trace");
        a_primary[idx] = fit.a_primary.clone();
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
    // Same tolerance basis as the bifactor 121-vs-241 regression: the EM
    // stopping tolerance is 1e-5, so two well-converged fits at different
    // (already-stabilized) node counts should agree to a small multiple of
    // that, not to float epsilon (independent EM runs land at slightly
    // different points on a flat likelihood ridge).
    assert!(
        loglik_diff < 5e-3,
        "loglik must agree closely between 121 and 241 nodes: {loglik_diff:.6}"
    );
    for i in 0..N_ITEMS {
        assert!(
            (a_primary[0][i] - a_primary[1][i]).abs() < 5e-3,
            "a_primary[{i}] disagrees: 121={}, 241={}",
            a_primary[0][i],
            a_primary[1][i]
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
