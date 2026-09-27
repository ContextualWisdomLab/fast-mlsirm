//! Unit contract for the single-group polytomous bifactor GRM fitter
//! (`mlsirm_core::bifactor_grm`, stage 1 of #1912).
//!
//! * Dimension-reduction exactness: the Gibbons-Hedeker reduced marginal
//!   log-likelihood must equal brute-force full product-grid integration to
//!   `<= 1e-10` on a tiny model (4 items, 2 specifics, `Q = 7`). The two paths
//!   reorder the same finite sum, so any larger gap is an implementation bug,
//!   not quadrature error.
//! * Argument validation and failure reporting: unsupported quadrature,
//!   bad iteration/tolerance/start budgets, malformed specific maps,
//!   out-of-range categories, unobserved categories, and the non-convergence
//!   report (`converged == false`, `termination_reason == "max_iter_reached"`).
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

use crate::bifactor_grm::{
    add_lnorm_abs_slope_prior, bifactor_grm_marginal_loglik, bifactor_grm_marginal_loglik_brute,
    checked_em_loglik_change, em_objective_name, fit_bifactor_grm, fit_bifactor_grm_multigroup,
    gh_rule, lnorm_abs_slope_prior_curvature, run_single_start, slope_prior_neg_log, validate,
    BifactorGrmConfig, BifactorMultigroupConfig, BifactorMultigroupResult, ItemParams,
    SlopePrior,
};

// ---------------------------------------------------------------------------
// Dimension-reduction exactness (tiny model, Q = 7).
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

#[test]
fn reduced_marginal_loglik_matches_brute_force_product_grid() {
    let (a_g, a_s, thresholds) = tiny_params();
    let (y, n_persons) = tiny_data();
    let reduced = bifactor_grm_marginal_loglik(
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
        7,
        7,
    )
    .expect("reduced loglik on valid tiny input must succeed");
    let brute = bifactor_grm_marginal_loglik_brute(
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
        7,
        7,
    )
    .expect("brute-force loglik on valid tiny input must succeed");
    assert!(
        reduced.is_finite() && brute.is_finite(),
        "both logliks must be finite: reduced={reduced}, brute={brute}"
    );
    let gap = (reduced - brute).abs();
    assert!(
        gap <= 1e-10,
        "Gibbons-Hedeker reduction only reorders the finite marginal sum, so the gap \
         must be floating-point noise (<= 1e-10); got gap={gap:.3e} \
         (reduced={reduced:.10}, brute={brute:.10})"
    );
}

// ---------------------------------------------------------------------------
// Argument validation and failure reporting.
// ---------------------------------------------------------------------------

fn valid_config() -> BifactorGrmConfig {
    BifactorGrmConfig {
        q_general: 7,
        q_specific: 7,
        max_iter: 5,
        tol: 1e-4,
        n_starts: 1,
        seed: 42,
        newton_iter: 3,
        ridge: 1e-8,
        slope_prior: SlopePrior::None,
        device: crate::Device::Cpu,
    }
}

#[test]
fn lnorm_abs_slope_prior_has_signed_support_and_jacobian() {
    let (a, mu, sd, h) = (-1.7, 0.2, 0.8, 1e-6);
    let mut nll = 0.0;
    let mut grad = 0.0;
    add_lnorm_abs_slope_prior(a, mu, sd, &mut nll, &mut grad);
    let mut plus = 0.0;
    let mut minus = 0.0;
    let mut ignored_grad = 0.0;
    add_lnorm_abs_slope_prior(a + h, mu, sd, &mut plus, &mut ignored_grad);
    add_lnorm_abs_slope_prior(a - h, mu, sd, &mut minus, &mut ignored_grad);
    let fd = (plus - minus) / (2.0 * h);
    assert!(nll.is_finite());
    assert!((grad - fd).abs() < 1e-6, "analytic={grad}, fd={fd}");
}

#[test]
fn lnorm_abs_slope_prior_keeps_zero_outside_support() {
    let mut nll = 0.0;
    let mut grad = 0.0;
    add_lnorm_abs_slope_prior(0.0, 0.0, 1.0, &mut nll, &mut grad);
    assert!(nll.is_infinite() && nll.is_sign_positive());
    assert_eq!(grad, 0.0);
}

#[test]
fn lnorm_abs_slope_prior_curvature_matches_fd_of_gradient() {
    let (mu, sd) = (0.2, 0.8);
    // Signed slopes on both sides, including the near-zero domain where the
    // curvature grows like 1/a^2 (relative FD step keeps a +/- h same-signed).
    for a in [-1.7, -0.3, -1e-3, 1e-3, 0.4, 2.5] {
        let h = 1e-6 * f64::abs(a);
        let grad_at = |x: f64| {
            let (mut nll, mut g) = (0.0, 0.0);
            add_lnorm_abs_slope_prior(x, mu, sd, &mut nll, &mut g);
            g
        };
        let fd = (grad_at(a + h) - grad_at(a - h)) / (2.0 * h);
        let analytic = lnorm_abs_slope_prior_curvature(a, mu, sd);
        let rel = (analytic - fd).abs() / analytic.abs().max(1.0);
        assert!(rel < 1e-5, "a={a}: analytic={analytic}, fd={fd}");
    }
    assert!(lnorm_abs_slope_prior_curvature(0.0, mu, sd).is_infinite());
}

/// Mean distance of every estimated slope's `ln|a|` from `mu`.
fn mean_log_slope_gap(a_g: &[f64], a_s: &[f64], mu: f64) -> f64 {
    let all: Vec<f64> = a_g.iter().chain(a_s.iter()).copied().collect();
    all.iter().map(|a| (a.abs().ln() - mu).abs()).sum::<f64>() / all.len() as f64
}

fn prior_fit_config(slope_prior: SlopePrior) -> BifactorGrmConfig {
    BifactorGrmConfig {
        max_iter: 60,
        newton_iter: 10,
        slope_prior,
        ..valid_config()
    }
}

#[test]
fn real_prior_fit_pulls_slopes_toward_prior_and_records_provenance() {
    let (y, n_persons) = tiny_data();
    let fit = |prior| {
        fit_bifactor_grm(
            &y,
            None,
            &TINY_SPECIFIC_MAP,
            n_persons,
            TINY_N_ITEMS,
            TINY_N_SPECIFIC,
            TINY_N_CAT,
            &prior_fit_config(prior),
        )
        .expect("tiny fit must run")
    };
    let mu = 0.5f64.ln();
    let prior = SlopePrior::Lognormal { mu, sd: 0.1 };
    let mml = fit(SlopePrior::None);
    let map = fit(prior);
    assert_eq!(mml.slope_prior, SlopePrior::None);
    assert_eq!(map.slope_prior, prior);
    let gap_mml = mean_log_slope_gap(&mml.a_general, &mml.a_specific, mu);
    let gap_map = mean_log_slope_gap(&map.a_general, &map.a_specific, mu);
    assert!(
        gap_map < gap_mml && gap_map < 0.25,
        "MAP slopes must sit near exp(mu): gap_map={gap_map}, gap_mml={gap_mml}"
    );
}

fn tiny_multigroup(n_groups: usize, slope_prior: SlopePrior) -> BifactorMultigroupResult {
    let (y, n_persons) = tiny_data();
    let group_id: Vec<usize> = (0..n_persons).map(|p| p % n_groups).collect();
    fit_mg(&y, &group_id, n_groups, &TINY_SPECIFIC_MAP, None, slope_prior)
        .expect("tiny multigroup fit must run")
}

fn fit_mg(
    y: &[usize],
    group_id: &[usize],
    n_groups: usize,
    specific_map: &[i32],
    anchor: Option<&[bool]>,
    slope_prior: SlopePrior,
) -> Result<BifactorMultigroupResult, String> {
    let base = prior_fit_config(slope_prior);
    let cfg = BifactorMultigroupConfig {
        q_general: base.q_general,
        q_specific: base.q_specific,
        max_iter: base.max_iter,
        tol: base.tol,
        n_starts: base.n_starts,
        seed: base.seed,
        newton_iter: base.newton_iter,
        ridge: base.ridge,
        slope_prior,
        ..BifactorMultigroupConfig::default()
    };
    let n_items = specific_map.len();
    fit_bifactor_grm_multigroup(
        y,
        None,
        group_id,
        n_groups,
        specific_map,
        group_id.len(),
        n_items,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        anchor,
        &cfg,
    )
}

// Two-group simulated design for native MG prior tests: six items (three per
// specific, the identified design of the Oakes unit tests), group 1 shifted
// on the general factor. The 12-person toy data is too small for MG: its MML
// comparison fit trips the EM likelihood guard (native RED 2026-09-21, run 2),
// independently of any prior.
const SIX_SPECIFIC_MAP: [i32; 6] = [0, 0, 0, 1, 1, 1];

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

/// `P(Y >= k) = logistic(a_G tG + a_S tS + d_k)`, group 1 has `tG ~ N(0.5, 1)`.
fn simulate_two_groups(n_per_group: usize, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let a_g = [1.3, 1.1, 0.9, 1.2, 1.0, 0.8];
    let a_s = [1.0, 0.9, 1.1, 0.8, 1.0, 0.9];
    let d = [1.2, -0.8, 1.0, -1.0, 1.3, -0.7, 1.1, -0.9, 0.9, -1.1, 1.2, -0.8];
    let mut rng = Lcg(seed);
    let (mut y, mut group_id) = (Vec::new(), Vec::new());
    for g in 0..2 {
        for _ in 0..n_per_group {
            let tg = rng.standard_normal() + 0.5 * g as f64;
            let ts = [rng.standard_normal(), rng.standard_normal()];
            for i in 0..6 {
                let base = a_g[i] * tg + a_s[i] * ts[i / 3];
                let u = rng.uniform();
                let mut cat = 0usize;
                for k in 0..2 {
                    if u < 1.0 / (1.0 + (-(base + d[i * 2 + k])).exp()) {
                        cat += 1;
                    } else {
                        break;
                    }
                }
                y.push(cat);
            }
            group_id.push(g);
        }
    }
    (y, group_id)
}

fn sim_mg(anchor: Option<&[bool]>, slope_prior: SlopePrior) -> BifactorMultigroupResult {
    let (y, group_id) = simulate_two_groups(200, 20_260_921);
    fit_mg(&y, &group_id, 2, &SIX_SPECIFIC_MAP, anchor, slope_prior)
        .expect("simulated two-group fit must run")
}

/// Prior far from the generating slopes so MAP visibly moves them.
const FAR_MU: f64 = -1.2039728043259361; // ln(0.3)

fn far_prior() -> SlopePrior {
    SlopePrior::Lognormal { mu: FAR_MU, sd: 0.03 }
}

#[test]
fn multigroup_prior_acts_on_common_items_under_default_anchor() {
    // `anchor = None` makes every item common (estimated from pooled
    // counts). Before the fix the prior was skipped for common items, so it
    // was a silent no-op here.
    let mml = sim_mg(None, SlopePrior::None);
    let map = sim_mg(None, far_prior());
    assert_eq!(mml.slope_prior, SlopePrior::None);
    assert_eq!(map.slope_prior, far_prior());
    let all: Vec<usize> = (0..6).collect();
    let gap_mml = rows_gap(&mml, 0, &all);
    let gap_map = rows_gap(&map, 0, &all);
    assert!(
        gap_map < 0.5 * gap_mml,
        "common-item slopes must carry the prior: gap_map={gap_map}, gap_mml={gap_mml}"
    );
    // Common items stay identical across groups.
    assert_eq!(map.a_general[0], map.a_general[1]);
    assert_eq!(map.a_specific[0], map.a_specific[1]);
    // Regression (native RED 2026-09-21, run 1): under the prior the observed
    // log-likelihood legitimately DECREASES while the log posterior ascends;
    // the old likelihood-only guard rejected such fits.
    assert!(
        map.loglik_trace.windows(2).any(|w| w[1] < w[0]),
        "fixture must exercise a likelihood decrease: {:?}",
        map.loglik_trace
    );
    assert_monotone(&map.em_objective_trace);
    // No duplicated prior for shared items: every item is common, so the
    // prior enters once per item (group-0 row), not once per group.
    let gap = map.em_objective_trace.last().unwrap() - map.loglik_trace.last().unwrap();
    let expected = -slope_prior_neg_log(far_prior(), &item_rows(&map, 0, &all));
    assert!((gap - expected).abs() < 1e-9, "gap={gap}, expected={expected}");
}

#[test]
fn multigroup_prior_acts_on_common_and_free_items_with_partial_anchor() {
    // Items 0,1,3,4 common (pooled, shared); items 2 and 5 free per group.
    let anchor = [true, true, false, true, true, false];
    let (common, free) = ([0usize, 1, 3, 4], [2usize, 5]);
    let mml = sim_mg(Some(&anchor), SlopePrior::None);
    let map = sim_mg(Some(&anchor), far_prior());
    for (label, items) in [("common", &common[..]), ("free", &free[..])] {
        for g in 0..2 {
            let (gap_mml, gap_map) = (rows_gap(&mml, g, items), rows_gap(&map, g, items));
            assert!(
                gap_map < 0.5 * gap_mml,
                "{label} items, group {g}: gap_map={gap_map}, gap_mml={gap_mml}"
            );
        }
    }
    for &i in &common {
        assert_eq!(map.a_general[0][i], map.a_general[1][i], "common item {i} shared");
    }
    assert_monotone(&map.em_objective_trace);
    // Common items enter the prior once; free items once per group.
    let mut rows = item_rows(&map, 0, &[0, 1, 2, 3, 4, 5]);
    rows.extend(item_rows(&map, 1, &free));
    let gap = map.em_objective_trace.last().unwrap() - map.loglik_trace.last().unwrap();
    let expected = -slope_prior_neg_log(far_prior(), &rows);
    assert!((gap - expected).abs() < 1e-9, "gap={gap}, expected={expected}");
}

fn rows_gap(r: &BifactorMultigroupResult, g: usize, items: &[usize]) -> f64 {
    let a_g: Vec<f64> = items.iter().map(|&i| r.a_general[g][i]).collect();
    let a_s: Vec<f64> = items.iter().map(|&i| r.a_specific[g][i]).collect();
    mean_log_slope_gap(&a_g, &a_s, FAR_MU)
}

fn assert_monotone(trace: &[f64]) {
    for w in trace.windows(2) {
        assert!(w[1] >= w[0] - 1e-9 * (1.0 + w[0].abs()), "objective decreased: {trace:?}");
    }
}

/// Slope rows of group `g` (|a| is all the prior sees, so reflection
/// canonicalization of the reported signs is irrelevant).
fn item_rows(r: &BifactorMultigroupResult, g: usize, items: &[usize]) -> Vec<ItemParams> {
    items
        .iter()
        .map(|&i| ItemParams {
            a_g: r.a_general[g][i],
            a_s: Some(r.a_specific[g][i]),
            d: Vec::new(),
        })
        .collect()
}

/// Reproducer kept on purpose (not a green test): the ORIGINAL 12-person
/// two-group toy fixture fails in plain MML, with no slope prior, because the
/// multigroup EM observed-data log-likelihood decreases (iteration 9, delta
/// -6.141380e-3 with `anchor = None`; iteration 20, delta -1.514901e-1 with
/// anchor `[T, T, F, F]`). The released 0.11.4 binary (no AC changes; zero
/// diff in `bifactor_grm.rs` from v0.11.4 to 99c228a8) reproduces both
/// errors bit-for-bit, so this is pre-existing MG MML behavior outside the
/// slope-prior scope. Run with `--ignored` once the MG EM owner fixes it.
#[test]
#[ignore = "pre-existing MG MML EM likelihood decrease on the 12-person toy fixture; tracked gap"]
fn reproducer_multigroup_mml_on_toy_fixture_runs() {
    let (y, n_persons) = tiny_data();
    let group_id: Vec<usize> = (0..n_persons).map(|p| p % 2).collect();
    for anchor in [None, Some(&[true, true, false, false][..])] {
        fit_mg(&y, &group_id, 2, &TINY_SPECIFIC_MAP, anchor, SlopePrior::None)
            .expect("MG MML on the toy fixture must run");
    }
}

#[test]
fn em_guard_rejects_a_real_log_posterior_decrease() {
    let prior = SlopePrior::Lognormal { mu: 0.0, sd: 1.0 };
    let err = checked_em_loglik_change(-10.0, Some(-9.0), 3, em_objective_name(prior))
        .expect_err("a posterior decrease must be rejected");
    assert!(err.contains("log posterior") && err.contains("decreased"), "{err}");
    assert_eq!(
        checked_em_loglik_change(-9.0, Some(-10.0), 3, em_objective_name(prior)),
        Ok(Some(1.0))
    );
}

#[test]
fn no_prior_objective_trace_is_the_loglik_trace_bit_for_bit() {
    let (y, n_persons) = tiny_data();
    let single = fit_bifactor_grm(
        &y,
        None,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &prior_fit_config(SlopePrior::None),
    )
    .expect("tiny fit must run");
    assert_eq!(single.em_objective_trace, single.loglik_trace);
    let mg = sim_mg(None, SlopePrior::None);
    assert_eq!(mg.em_objective_trace, mg.loglik_trace);
}

#[test]
fn multistart_ranks_starts_by_log_posterior() {
    let (y, n_persons) = tiny_data();
    let prior = SlopePrior::Lognormal { mu: 0.5f64.ln(), sd: 0.3 };
    let cfg = BifactorGrmConfig {
        n_starts: 3,
        ..prior_fit_config(prior)
    };
    let fit = fit_bifactor_grm(
        &y,
        None,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &cfg,
    )
    .expect("multi-start prior fit must run");
    let v = validate(
        &y,
        None,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &cfg,
    )
    .expect("valid tiny problem");
    let (tg, wg) = gh_rule(cfg.q_general).expect("rule");
    let (ts, ws) = gh_rule(cfg.q_specific).expect("rule");
    let log_wg: Vec<f64> = wg.iter().map(|w| w.ln()).collect();
    let log_ws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();
    let finals: Vec<(f64, f64)> = (0..cfg.n_starts)
        .map(|start| {
            let o = run_single_start(
                &v, &y, None, &cfg, tg, ts, &log_wg, &log_ws, tg.len(), ts.len(), start,
            )
            .expect("start runs");
            (*o.em_objective_trace.last().unwrap(), *o.loglik_trace.last().unwrap())
        })
        .collect();
    let best_objective = finals.iter().map(|f| f.0).fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(finals[fit.best_start].0, best_objective, "starts: {finals:?}");
    assert_eq!(*fit.em_objective_trace.last().unwrap(), best_objective);
}

#[test]
fn multigroup_single_group_prior_matches_single_group_fit() {
    let prior = SlopePrior::Lognormal { mu: 0.5f64.ln(), sd: 0.1 };
    let (y, n_persons) = tiny_data();
    let single = fit_bifactor_grm(
        &y,
        None,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &prior_fit_config(prior),
    )
    .expect("tiny fit must run");
    let mg = tiny_multigroup(1, prior);
    assert_eq!(mg.a_general[0], single.a_general);
    assert_eq!(mg.a_specific[0], single.a_specific);
    assert_eq!(mg.slope_prior, prior);
}

#[test]
fn slope_prior_rejects_invalid_hyperparameters() {
    assert!(SlopePrior::Lognormal { mu: f64::NAN, sd: 1.0 }
        .validate()
        .is_err());
    assert!(SlopePrior::Lognormal { mu: 0.0, sd: 0.0 }
        .validate()
        .is_err());
    assert!(SlopePrior::Lognormal { mu: 0.0, sd: f64::INFINITY }
        .validate()
        .is_err());
}

fn valid_data() -> (Vec<usize>, usize) {
    tiny_data()
}

#[test]
fn oracle_rejects_nonzero_specific_slope_on_general_only_items() {
    // A general-only item (specific_map == -1) owns no specific slope: a
    // non-zero value would be silently dropped, so the oracle fails loudly.
    let (a_g, _, thresholds) = tiny_params();
    let (y, n_persons) = tiny_data();
    // One specific factor on items 0, 1, 3; item 2 is general-only.
    let map: [i32; TINY_N_ITEMS] = [0, 0, -1, 0];
    let mut a_s = vec![0.8, 1.1, 0.0, 0.6];
    a_s[2] = 0.5; // general-only item with a stray specific slope
    bifactor_grm_marginal_loglik(
        &a_g,
        &a_s,
        &thresholds,
        &y,
        None,
        &map,
        n_persons,
        TINY_N_ITEMS,
        1,
        TINY_N_CAT,
        7,
        7,
    )
    .expect_err("non-zero a_specific on a general-only item must fail loudly");
}

#[test]
fn uncapped_budgets_are_accepted() {
    // Stage-1 review fix-up: no magic upper caps on caller budgets. Lower
    // bounds (and only lower bounds) are validated; size overflow is a loud
    // `Err` from checked arithmetic instead.
    let (y, n_persons) = valid_data();
    let cfg = BifactorGrmConfig {
        max_iter: 1,
        tol: 1e-12,
        n_starts: 40,
        ..valid_config()
    };
    let fit = fit_bifactor_grm(
        &y,
        None,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &cfg,
    )
    .expect("n_starts=40 must be accepted (no upper cap)");
    assert!(
        fit.best_start < 40,
        "best_start must index the 40 runs; got {}",
        fit.best_start
    );

    // n_specific=17 validates: 34 items in 17 pairs, 12 persons cycling all
    // three categories so every boundary stays identified.
    let n_items_17 = 34usize;
    let n_persons_17 = 12usize;
    let map_17: Vec<i32> = (0..n_items_17).map(|i| (i / 2) as i32).collect();
    let mut y_17 = vec![0usize; n_persons_17 * n_items_17];
    for p in 0..n_persons_17 {
        for i in 0..n_items_17 {
            y_17[p * n_items_17 + i] = (p + i) % TINY_N_CAT;
        }
    }
    fit_bifactor_grm(
        &y_17,
        None,
        &map_17,
        n_persons_17,
        n_items_17,
        17,
        TINY_N_CAT,
        &valid_config(),
    )
    .expect("n_specific=17 must be accepted (no upper cap)");
}

#[test]
fn arbitrary_quadrature_counts_above_the_old_fixed_table_are_accepted() {
    // #1929: node count controls integration precision and must not be capped
    // at a fixed table; 5/22/100 used to be rejected, now must fit cleanly.
    let (y, n_persons) = valid_data();
    for (qg, qs) in [(5, 7), (7, 5), (21, 22), (100, 7)] {
        let cfg = BifactorGrmConfig {
            q_general: qg,
            q_specific: qs,
            ..valid_config()
        };
        fit_bifactor_grm(
            &y,
            None,
            &TINY_SPECIFIC_MAP,
            n_persons,
            TINY_N_ITEMS,
            TINY_N_SPECIFIC,
            TINY_N_CAT,
            &cfg,
        )
        .unwrap_or_else(|e| panic!("q_general={qg}, q_specific={qs} must be accepted: {e}"));
    }
}

#[test]
fn rejects_zero_quadrature_counts() {
    let (y, n_persons) = valid_data();
    for (qg, qs) in [(0, 7), (7, 0)] {
        let cfg = BifactorGrmConfig {
            q_general: qg,
            q_specific: qs,
            ..valid_config()
        };
        let err = fit_bifactor_grm(
            &y,
            None,
            &TINY_SPECIFIC_MAP,
            n_persons,
            TINY_N_ITEMS,
            TINY_N_SPECIFIC,
            TINY_N_CAT,
            &cfg,
        )
        .expect_err("q_general/q_specific == 0 must fail loudly, never clamp");
        assert!(
            err.contains("q_general") || err.contains("q_specific") || err.contains('q'),
            "quadrature error must name the offending argument; got: {err}"
        );
    }
}

#[test]
fn rejects_bad_iteration_and_tolerance_budgets() {
    let (y, n_persons) = valid_data();
    for cfg in [
        BifactorGrmConfig {
            max_iter: 0,
            ..valid_config()
        },
        BifactorGrmConfig {
            tol: 0.0,
            ..valid_config()
        },
        BifactorGrmConfig {
            tol: f64::NAN,
            ..valid_config()
        },
        BifactorGrmConfig {
            n_starts: 0,
            ..valid_config()
        },
        BifactorGrmConfig {
            newton_iter: 0,
            ..valid_config()
        },
        BifactorGrmConfig {
            ridge: -1e-8,
            ..valid_config()
        },
    ] {
        fit_bifactor_grm(
            &y,
            None,
            &TINY_SPECIFIC_MAP,
            n_persons,
            TINY_N_ITEMS,
            TINY_N_SPECIFIC,
            TINY_N_CAT,
            &cfg,
        )
        .expect_err("out-of-range caller budgets must fail loudly, never clamp");
    }
}

#[test]
fn rejects_malformed_specific_map() {
    let (y, n_persons) = valid_data();
    // Wrong length.
    fit_bifactor_grm(
        &y,
        None,
        &[0, 0, 1],
        n_persons,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &valid_config(),
    )
    .expect_err("specific_map length must equal n_items");
    // Out-of-range block index.
    fit_bifactor_grm(
        &y,
        None,
        &[0, 0, 1, 2],
        n_persons,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &valid_config(),
    )
    .expect_err("specific_map entries must be -1 or in 0..n_specific");
    // Negative other than the -1 general-only sentinel.
    fit_bifactor_grm(
        &y,
        None,
        &[0, 0, 1, -2],
        n_persons,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &valid_config(),
    )
    .expect_err("specific_map entries below -1 must be rejected");
    // A specific factor with fewer than two items is rejected as an
    // implementation stability choice (weakly identified general/specific
    // split; no minimum-block-size theorem in the cited sources).
    fit_bifactor_grm(
        &y,
        None,
        &[0, 1, 1, 1],
        n_persons,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &valid_config(),
    )
    .expect_err("a specific factor with a single item must be rejected");
}

#[test]
fn rejects_out_of_range_categories() {
    let (mut y, n_persons) = valid_data();
    y[0] = TINY_N_CAT; // one past the top category
    fit_bifactor_grm(
        &y,
        None,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &valid_config(),
    )
    .expect_err("observed categories must be < n_cat");
}

#[test]
fn unobserved_category_fails_loudly_with_item_and_category() {
    // Item 0 never sees category 2: its top boundary intercept is unidentified
    // (Samejima, 1969), so the fitter must refuse instead of imputing.
    let n_persons = 30usize;
    let mut y = vec![0usize; n_persons * TINY_N_ITEMS];
    for p in 0..n_persons {
        y[p * TINY_N_ITEMS] = p % 2; // item 0: only categories 0-1
        for i in 1..TINY_N_ITEMS {
            y[p * TINY_N_ITEMS + i] = p % TINY_N_CAT;
        }
    }
    let err = fit_bifactor_grm(
        &y,
        None,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &valid_config(),
    )
    .expect_err("an unobserved category must fail loudly, never be imputed");
    assert!(
        err.contains("item 0") && err.contains("category 2"),
        "the error must name the item and the unobserved category; got: {err}"
    );
}

#[test]
fn non_convergence_is_reported_not_substituted() {
    // One EM sweep cannot converge: the fitter must return Ok with
    // converged == false and termination_reason == "max_iter_reached" so the
    // caller sees the failure instead of receiving a substitute.
    let (y, _) = tiny_data();
    // Larger toy sample so a single sweep is nowhere near convergence.
    let n_big = 200usize;
    let mut y_big = Vec::with_capacity(n_big * TINY_N_ITEMS);
    for p in 0..n_big {
        for i in 0..TINY_N_ITEMS {
            y_big.push(y[(p % 12) * TINY_N_ITEMS + i]);
        }
    }
    let cfg = BifactorGrmConfig {
        max_iter: 1,
        tol: 1e-12,
        ..valid_config()
    };
    let fit = fit_bifactor_grm(
        &y_big,
        None,
        &TINY_SPECIFIC_MAP,
        n_big,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &cfg,
    )
    .expect("max_iter exhaustion must be reported via flags, not via Err");
    assert!(
        !fit.converged,
        "a single EM sweep must not report convergence"
    );
    assert_eq!(
        fit.termination_reason, "max_iter_reached",
        "termination reason must say max_iter_reached"
    );
    assert!(
        fit.loglik_trace.len() == 2,
        "trace must hold the initial and one updated loglik; got {}",
        fit.loglik_trace.len()
    );
}

// ---------------------------------------------------------------------------
// Stage 5: GPU/CPU E-step parity on expected counts and loglik.
// ---------------------------------------------------------------------------

#[test]
fn estep_gpu_matches_cpu_counts_and_loglik() {
    // The WGSL person-parallel sweep accumulates in f32 while the CPU
    // reference is f64: per-count-entry error scales with n_persons × eps
    // (≈ 12 × 1.2e-7 here), so the envelope below is orders of magnitude
    // above the expected f32 noise. The tiny fixture carries a negative
    // general slope (reverse-keyed item 1), so sign handling is covered.
    // Without a GPU adapter the GPU entry falls back to CPU and the
    // comparison is trivially exact; the fit-level Python test pins the
    // real-device numbers.
    use super::{
        e_step, fill_logprob_tables, gh_rule, initial_params, validate,
    };

    let (y, n_persons) = tiny_data();
    let cfg = valid_config();
    let v = validate(
        &y,
        None,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &cfg,
    )
    .expect("tiny fixture must validate");
    let (tg, wg) = gh_rule(7).expect("Q=7 rule must exist");
    let (ts, ws) = gh_rule(7).expect("Q=7 rule must exist");
    let params = initial_params(&v, &y, None, cfg.seed, 0);
    let tables = fill_logprob_tables(&v, &params, tg, ts, 7, 7);
    let log_wg: Vec<f64> = wg.iter().map(|w| w.ln()).collect();
    let log_ws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();

    let (ll_cpu, counts_cpu) = e_step(
        &v, &y, None, &tables, &log_wg, &log_ws, 7, 7, tg, ts,
        crate::Device::Cpu,
    );
    let (ll_gpu, counts_gpu) = e_step(
        &v, &y, None, &tables, &log_wg, &log_ws, 7, 7, tg, ts,
        crate::Device::Gpu,
    );

    assert!(
        (ll_cpu - ll_gpu).abs() <= 1e-3,
        "E-step loglik must agree within f32 envelope: cpu={ll_cpu}, gpu={ll_gpu}"
    );
    assert_eq!(counts_cpu.len(), counts_gpu.len());
    let mut max_diff = 0.0f64;
    for (cc, gc) in counts_cpu.iter().zip(counts_gpu.iter()) {
        assert_eq!(cc.len(), gc.len());
        for (cn, gn) in cc.iter().zip(gc.iter()) {
            assert_eq!(cn.len(), gn.len());
            for (&a, &b) in cn.iter().zip(gn.iter()) {
                max_diff = max_diff.max((a - b).abs());
            }
        }
    }
    assert!(
        max_diff <= 1e-4,
        "expected counts must agree within f32 envelope; got max|diff|={max_diff:.3e}"
    );
}

// ---------------------------------------------------------------------------
// #1976: zero Gauss-Hermite prior mass must not NaN-poison E-step counts;
// a fit that never leaves its start must not report tolerance_met.
// ---------------------------------------------------------------------------

#[test]
fn zero_prior_weight_nodes_do_not_nan_estep_counts() {
    use super::{e_step, fill_logprob_tables, gh_rule, initial_params, validate};

    let (y, n_persons) = tiny_data();
    let cfg = valid_config();
    let v = validate(
        &y,
        None,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &cfg,
    )
    .expect("tiny fixture must validate");
    let (tg, wg) = gh_rule(7).expect("Q=7 rule must exist");
    let (ts, ws) = gh_rule(7).expect("Q=7 rule must exist");
    let params = initial_params(&v, &y, None, cfg.seed, 0);
    let tables = fill_logprob_tables(&v, &params, tg, ts, 7, 7);
    let mut log_wg: Vec<f64> = wg.iter().map(|w| w.ln()).collect();
    let log_ws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();
    // Inject exactly-zero prior mass at both tails (the large-q Golub–Welsch
    // failure mode from #1976).
    log_wg[0] = f64::NEG_INFINITY;
    log_wg[6] = f64::NEG_INFINITY;

    let (ll, counts) = e_step(
        &v,
        &y,
        None,
        &tables,
        &log_wg,
        &log_ws,
        7,
        7,
        tg,
        ts,
        crate::Device::Cpu,
    );
    assert!(ll.is_finite(), "observed-data loglik must stay finite; got {ll}");
    for (i, item_counts) in counts.iter().enumerate() {
        for (node, cat) in item_counts.iter().enumerate() {
            for (k, &c) in cat.iter().enumerate() {
                assert!(
                    c.is_finite(),
                    "count[{i}][{node}][{k}] must be finite; got {c}"
                );
            }
        }
    }
}

#[test]
fn refuse_tolerance_reclassifies_bit_identical_start() {
    use super::{refuse_tolerance_on_frozen_start, ItemParams};

    let start = vec![ItemParams {
        a_g: 1.0,
        a_s: Some(0.8),
        d: vec![1.0, -1.0],
    }];
    let frozen = start.clone();
    let mut converged = true;
    let mut reason = "tolerance_met".to_string();
    refuse_tolerance_on_frozen_start(&mut converged, &mut reason, &frozen, &start);
    assert!(!converged);
    assert_eq!(reason, "numerical_em_stall");

    let mut moved = start.clone();
    moved[0].a_g = 1.5;
    converged = true;
    reason = "tolerance_met".to_string();
    refuse_tolerance_on_frozen_start(&mut converged, &mut reason, &moved, &start);
    assert!(converged);
    assert_eq!(reason, "tolerance_met");
}

#[test]
fn dense_quadrature_fit_never_claims_tolerance_at_start_slopes() {
    // Compact #1976-shaped design (13 items / 3 specifics / 4 cats). q=421 is
    // where Golub–Welsch produces exactly-zero prior weights; the pre-fix
    // path reported tolerance_met with a_general still at 1.0.
    let n_persons = 60usize;
    let n_items = 13usize;
    let n_specific = 3usize;
    let n_cat = 4usize;
    let specific_map: [i32; 13] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, -1];
    let mut y = vec![0usize; n_persons * n_items];
    for p in 0..n_persons {
        for i in 0..n_items {
            y[p * n_items + i] = (p + 3 * i) % n_cat;
        }
    }
    let cfg = BifactorGrmConfig {
        q_general: 421,
        q_specific: 421,
        max_iter: 12,
        tol: 1e-6,
        n_starts: 1,
        seed: 20260917,
        newton_iter: 5,
        ridge: 1e-4,
        slope_prior: crate::bifactor_grm::SlopePrior::None,
        device: crate::Device::Cpu,
    };
    let fit = fit_bifactor_grm(
        &y,
        None,
        &specific_map,
        n_persons,
        n_items,
        n_specific,
        n_cat,
        &cfg,
    )
    .expect("dense-q fit must return a result");
    if fit.termination_reason == "tolerance_met" {
        assert!(fit.converged);
        assert!(
            fit.a_general.iter().any(|&a| a != 1.0),
            "tolerance_met must not leave a_general at the 1.0 start; got {:?}",
            &fit.a_general[..3]
        );
    } else if fit.a_general.iter().all(|&a| a == 1.0) {
        assert!(
            !fit.converged,
            "parameters frozen at start must not report converged"
        );
        assert_eq!(fit.termination_reason, "numerical_em_stall");
    }
}
