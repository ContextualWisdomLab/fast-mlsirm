//! Anchored vs free items for the multiple-group bifactor GRM (stage 2).
//!
//! One item is simulated with group-specific thresholds (uniform DIF) while
//! the rest share parameters. Fitting with that item flagged free must keep
//! anchored rows identical across groups, return finite per-group parameters
//! for the free item, and recover the threshold shift direction; fitting with
//! all items free must fail loudly (no linking item — Cai, Yang, & Hansen,
//! 2011, at least one common item links the scales).
//!
//! # References (APA 7th ed.)
//!
//! Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information
//! item bifactor analysis. *Psychological Methods, 16*(3), 221-248.
//! https://doi.org/10.1037/a0023350

use mlsirm_core::bifactor_grm::{fit_bifactor_grm_multigroup, BifactorMultigroupConfig};

const N_ITEMS: usize = 6;
const N_SPECIFIC: usize = 2;
const N_CAT: usize = 4;
const SPECIFIC_MAP: [i32; N_ITEMS] = [0, 0, 0, 1, 1, 1];
const N_PER_GROUP: usize = 800;
const FREE_ITEM: usize = 5;
const SHIFT: f64 = 0.60;

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

fn cfg() -> BifactorMultigroupConfig {
    BifactorMultigroupConfig {
        q_general: 11,
        q_specific: 11,
        max_iter: 300,
        tol: 1e-5,
        n_starts: 1,
        seed: 20_260_918,
        newton_iter: 10,
        ridge: 1e-8,
        slope_prior: mlsirm_core::bifactor_grm::SlopePrior::None,
        device: mlsirm_core::Device::Cpu,
        estimate_specific_vars: false,
    }
}

fn simulate() -> (Vec<usize>, Vec<usize>) {
    let a_g = [1.30, 1.10, 1.00, 1.20, 0.90, 1.10];
    let a_s = [1.00, 0.90, 1.10, 1.00, 0.80, 0.90];
    let d_base = [
        [1.20, 0.00, -1.20],
        [1.00, -0.10, -1.30],
        [1.30, 0.20, -1.00],
        [1.10, 0.10, -1.10],
        [0.90, -0.20, -1.40],
        [1.20, 0.00, -1.20],
    ];
    let n_persons = 2 * N_PER_GROUP;
    let mut rng = Lcg(20_260_919);
    let mut y = vec![0usize; n_persons * N_ITEMS];
    let mut group_id = vec![0usize; n_persons];
    for p in 0..n_persons {
        let g = usize::from(p >= N_PER_GROUP);
        group_id[p] = g;
        let t_g = rng.standard_normal() + if g == 1 { 0.40 } else { 0.0 };
        let t_s0 = rng.standard_normal();
        let t_s1 = rng.standard_normal();
        for i in 0..N_ITEMS {
            let s = if SPECIFIC_MAP[i] == 0 { t_s0 } else { t_s1 };
            let mut d = d_base[i];
            if i == FREE_ITEM && g == 1 {
                for dk in d.iter_mut() {
                    *dk += SHIFT;
                }
            }
            let base = a_g[i] * t_g + a_s[i] * s;
            let mut cum = [0.0f64; 3];
            for k in 0..3 {
                cum[k] = sigmoid(base + d[k]);
            }
            let probs = [1.0 - cum[0], cum[0] - cum[1], cum[1] - cum[2], cum[2]];
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
    (y, group_id)
}

#[test]
fn free_item_keeps_anchored_rows_equal_and_recovers_shift() {
    let (y, group_id) = simulate();
    let n_persons = 2 * N_PER_GROUP;
    let mut anchor = vec![true; N_ITEMS];
    anchor[FREE_ITEM] = false;
    let fit = fit_bifactor_grm_multigroup(
        &y,
        None,
        &group_id,
        2,
        &SPECIFIC_MAP,
        n_persons,
        N_ITEMS,
        N_SPECIFIC,
        N_CAT,
        Some(&anchor),
        &cfg(),
    )
    .expect("free-anchor fit must succeed");
    assert!(
        fit.converged,
        "fit must converge: {}",
        fit.termination_reason
    );
    // Anchored rows identical; free rows finite.
    for i in 0..N_ITEMS {
        if anchor[i] {
            assert!(
                (fit.a_general[0][i] - fit.a_general[1][i]).abs() == 0.0,
                "anchored item {i} must match across groups"
            );
            for k in 0..N_CAT - 1 {
                assert!(
                    (fit.threshold[0][i * (N_CAT - 1) + k] - fit.threshold[1][i * (N_CAT - 1) + k])
                        .abs()
                        == 0.0
                );
            }
        }
    }
    for g in 0..2 {
        assert!(fit.a_general[g][FREE_ITEM].is_finite());
        for k in 0..N_CAT - 1 {
            assert!(fit.threshold[g][FREE_ITEM * (N_CAT - 1) + k].is_finite());
        }
    }
    // The simulated shift raises every boundary in group 1; the estimates
    // must preserve that direction on average.
    let mean0: f64 = (0..N_CAT - 1)
        .map(|k| fit.threshold[0][FREE_ITEM * (N_CAT - 1) + k])
        .sum::<f64>()
        / (N_CAT - 1) as f64;
    let mean1: f64 = (0..N_CAT - 1)
        .map(|k| fit.threshold[1][FREE_ITEM * (N_CAT - 1) + k])
        .sum::<f64>()
        / (N_CAT - 1) as f64;
    assert!(
        mean1 > mean0,
        "free thresholds must preserve the simulated upward shift: {mean0:.3} vs {mean1:.3}"
    );
}

#[test]
fn all_free_items_fail_loudly_without_a_link() {
    let (y, group_id) = simulate();
    let n_persons = 2 * N_PER_GROUP;
    let anchor = vec![false; N_ITEMS];
    let err = fit_bifactor_grm_multigroup(
        &y,
        None,
        &group_id,
        2,
        &SPECIFIC_MAP,
        n_persons,
        N_ITEMS,
        N_SPECIFIC,
        N_CAT,
        Some(&anchor),
        &cfg(),
    )
    .expect_err("all-free fit must fail loudly for lack of a linking item");
    assert!(
        err.contains("anchored"),
        "error must name the linking requirement: {err}"
    );
}
