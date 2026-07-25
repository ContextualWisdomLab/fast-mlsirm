use super::*;
use crate::poly::{fit_poly_unidim, PolyModel};

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
    (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / a.len() as f64).sqrt()
}
fn corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx) * (a - mx);
        syy += (b - my) * (b - my);
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

/// Simulate multidimensional GPCM responses: base = sum_d slope[i,d]*theta_d, then
/// softmax_k(k*base + step_ik).
fn simulate(
    slope: &[f64],
    step: &[f64],
    theta: &[f64],
    n: usize,
    n_items: usize,
    n_dims: usize,
    n_cat: usize,
    rng: &mut Lcg,
) -> Vec<usize> {
    let m1 = n_cat - 1;
    let scores: Vec<f64> = (0..n_cat).map(|c| c as f64).collect();
    let mut y = vec![0usize; n * n_items];
    for p in 0..n {
        for i in 0..n_items {
            let mut base = 0.0f64;
            for d in 0..n_dims {
                base += slope[i * n_dims + d] * theta[p * n_dims + d];
            }
            let mut intercepts = vec![0.0f64; n_cat];
            intercepts[1..].copy_from_slice(&step[i * m1..(i + 1) * m1]);
            let lp = gpcm_logprobs(base, &scores, &intercepts);
            let u = rng.next_f64();
            let mut acc = 0.0;
            let mut cat = n_cat - 1;
            for (k, l) in lp.iter().enumerate() {
                acc += l.exp();
                if u < acc {
                    cat = k;
                    break;
                }
            }
            y[p * n_items + i] = cat;
        }
    }
    y
}

/// D = 1 WITHIN-TOL reduction to fit_poly_unidim(GPCM). All-POSITIVE true slopes (fit_poly_unidim
/// forces a>0 via log_a); both reach the same MLE up to optimizer tolerance and the positive
/// reflection. NOT bit-exact.
#[test]
fn gpcm_reduces_to_poly_gpcm_at_d1() {
    let (n, n_items, n_cat) = (2000usize, 6usize, 4usize);
    let m1 = n_cat - 1;
    let mut rng = Lcg(717717);
    let mut slope = vec![0.0f64; n_items];
    let mut step = vec![0.0f64; n_items * m1];
    for i in 0..n_items {
        slope[i] = 0.8 + 0.2 * i as f64; // POSITIVE
                                         // UNORDERED steps (GPCM has no ordering constraint)
        step[i * m1] = 0.6 - 0.1 * i as f64;
        step[i * m1 + 1] = -0.4 + 0.05 * i as f64;
        step[i * m1 + 2] = 0.3 - 0.08 * i as f64;
    }
    let theta: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let y = simulate(&slope, &step, &theta, n, n_items, 1, n_cat, &mut rng);
    let pattern = vec![1u8; n_items];
    let cfg = GpcmConfig {
        q: 21,
        ..GpcmConfig::default()
    };
    let mm = fit_gpcm(&y, None, &pattern, n, n_items, 1, n_cat, &cfg).unwrap();
    let pf = fit_poly_unidim(&y, None, n, n_items, n_cat, PolyModel::Gpcm, 21, 500, 1e-6).unwrap();
    for i in 0..n_items {
        assert!(
            (mm.slope[i] - pf.slope[i]).abs() < 0.05,
            "slope[{i}] {} vs {}",
            mm.slope[i],
            pf.slope[i]
        );
        for j in 0..m1 {
            let d = (mm.step[i * m1 + j] - pf.cat_params[i][j]).abs();
            assert!(d < 0.06, "step[{i}][{j}] diff {d}");
        }
    }
    assert!(
        (*mm.loglik_trace.last().unwrap() - pf.loglik).abs() < 0.5,
        "loglik"
    );
    assert_eq!(mm.n_parameters, n_items * (1 + m1));
}

/// Deterministic FD GRADIENT anchor at D=2 (GH) AND D=4 (Halton, NON-IDENTITY dims [0,2,3]) with
/// M=4 categories, NON-MONOTONE steps (locks in that the GPCM softmax is finite for any steps —
/// no accidental ordering guard), and distinct random per-category counts (so a slope<->step slot
/// transposition is detected). The M-step uses an FD Hessian, so pin the GRADIENT.
#[test]
fn gpcm_gradient_matches_finite_difference() {
    let n_cat = 4usize;
    for &(n_dims, ref dims) in [(2usize, vec![0usize, 1]), (4usize, vec![0usize, 2, 3])].iter() {
        let l = dims.len();
        let (nodes, n_nodes) = if n_dims == 2 {
            let xn = build_xi_nodes(XiRule::GaussHermite { q_xi: 15 }, n_dims).unwrap();
            (xn.grid, xn.logw.len())
        } else {
            let xn = build_xi_nodes(
                XiRule::Halton {
                    n: 200,
                    shift_seed: 0,
                },
                n_dims,
            )
            .unwrap();
            (xn.grid, xn.logw.len())
        };
        let mut rng = Lcg(1414 + n_dims as u64);
        let counts: Vec<Vec<f64>> = (0..n_nodes)
            .map(|_| (0..n_cat).map(|_| 0.1 + rng.next_f64() * 3.0).collect())
            .collect();
        let mut params = vec![0.0f64; l + (n_cat - 1)];
        for t in 0..l {
            params[t] = 0.4 + 0.3 * t as f64 - if t == 1 { 0.9 } else { 0.0 };
        }
        // NON-MONOTONE steps
        let steps = [0.8f64, -0.3, 1.1];
        for j in 0..(n_cat - 1) {
            params[l + j] = steps[j];
        }
        let (_f0, grad) = gpcm_item_neg_ll_grad(&params, dims, &nodes, n_dims, &counts, n_cat);
        let eps = 1e-6;
        for j in 0..params.len() {
            let mut pp = params.clone();
            pp[j] += eps;
            let (fp, _) = gpcm_item_neg_ll_grad(&pp, dims, &nodes, n_dims, &counts, n_cat);
            let mut pm = params.clone();
            pm[j] -= eps;
            let (fm, _) = gpcm_item_neg_ll_grad(&pm, dims, &nodes, n_dims, &counts, n_cat);
            let fd = (fp - fm) / (2.0 * eps);
            assert!(
                (grad[j] - fd).abs() < 1e-4,
                "grad[{j}] {} vs fd {fd} (D={n_dims})",
                grad[j]
            );
        }
    }
}

/// Deterministic OBJECTIVE-VALUE dims-map pin at D=4 (Halton, dims=[0,2,3]). Computes base with the
/// CORRECT dim map and gpcm_logprobs with LITERAL integer scores [0,1,2,3] and a literal 0.0
/// baseline step, then matches the estimator's internal neg-loglik to < 1e-9. The FD anchor is
/// map-invariant AND scores-invariant; this is the only guard against a wrong-node-column, a
/// wrong-scores (e.g. [1,2,3,4]), or a dropped-baseline-step mutation on the QMC path.
#[test]
fn gpcm_objective_dims_map_pinned_at_d4() {
    let n_dims = 4usize;
    let dims = vec![0usize, 2, 3];
    let n_cat = 4usize;
    let l = dims.len();
    let xn = build_xi_nodes(
        XiRule::Halton {
            n: 64,
            shift_seed: 0,
        },
        n_dims,
    )
    .unwrap();
    let nodes = xn.grid;
    let n_nodes = xn.logw.len();
    let mut rng = Lcg(27182);
    let counts: Vec<Vec<f64>> = (0..n_nodes)
        .map(|_| (0..n_cat).map(|_| 0.1 + rng.next_f64() * 2.0).collect())
        .collect();
    let a = [0.9f64, -0.6, 0.7];
    let step = [0.5f64, -0.8, 0.2]; // non-monotone
    let mut params = vec![0.0f64; l + (n_cat - 1)];
    params[..l].copy_from_slice(&a);
    params[l..].copy_from_slice(&step);
    let (neg_ll, _g) = gpcm_item_neg_ll_grad(&params, &dims, &nodes, n_dims, &counts, n_cat);
    let mut hand = 0.0f64;
    for (nd, cnt) in counts.iter().enumerate() {
        let base = a[0] * nodes[nd * n_dims + 0]
            + a[1] * nodes[nd * n_dims + 2]
            + a[2] * nodes[nd * n_dims + 3];
        let lp = gpcm_logprobs(
            base,
            &[0.0, 1.0, 2.0, 3.0],
            &[0.0, step[0], step[1], step[2]],
        );
        hand += cnt.iter().zip(&lp).map(|(r, l2)| r * l2).sum::<f64>();
    }
    assert!(
        (neg_ll - (-hand)).abs() < 1e-9,
        "objective dims/scores map mismatch: {neg_ll} vs {}",
        -hand
    );
}

#[test]
fn gpcm_validation_sampling_rules_and_missing_paths() {
    let base = GpcmConfig {
        q: 7,
        max_iter: 1,
        newton_iter: 1,
        ..GpcmConfig::default()
    };
    let y = [0usize, 0, 1, 1, 0, 1, 1, 0];
    let observed = [true, false, true, true, true, true, true, true];
    let pattern = [1u8, 1];

    assert!(validate(&y, None, &pattern, 0, 2, 1, 2, &base).is_err());
    assert!(validate(&y, None, &pattern, 4, 2, 1, 1, &base).is_err());
    assert!(validate(
        &y,
        None,
        &pattern,
        4,
        2,
        1,
        2,
        &GpcmConfig {
            max_iter: 0,
            ..base
        }
    )
    .is_err());
    assert!(validate(
        &y,
        None,
        &pattern,
        4,
        2,
        1,
        2,
        &GpcmConfig {
            tol: f64::NAN,
            ..base
        }
    )
    .is_err());
    assert!(validate(
        &y,
        None,
        &pattern,
        4,
        2,
        1,
        2,
        &GpcmConfig { ridge: 0.0, ..base }
    )
    .is_err());
    assert!(validate(&y, None, &[], 4, 2, 0, 2, &base).is_err());
    assert!(validate(&y, None, &[1; 8], 4, 2, 4, 2, &base).is_err());
    assert!(validate(&y, None, &pattern, 4, 2, 1, 2, &GpcmConfig { q: 3, ..base }).is_err());
    let halton = GpcmConfig {
        xi_rule: XiRuleKind::Halton,
        xi_points: 4,
        ..base
    };
    assert!(validate(&y, None, &[], 4, 2, 0, 2, &halton).is_err());
    assert!(validate(&y, None, &[1; 14], 4, 2, 7, 2, &halton).is_err());
    assert!(validate(
        &y,
        None,
        &pattern,
        4,
        2,
        1,
        2,
        &GpcmConfig {
            xi_points: 0,
            ..halton
        },
    )
    .is_err());
    assert!(validate(&y[..7], None, &pattern, 4, 2, 1, 2, &base).is_err());
    assert!(validate(&y, Some(&[true]), &pattern, 4, 2, 1, 2, &base).is_err());
    assert!(validate(&y, None, &[1], 4, 2, 1, 2, &base).is_err());
    assert!(validate(&y, None, &[2, 1], 4, 2, 1, 2, &base).is_err());
    let bad_y = [2usize, 0, 1, 1, 0, 1, 1, 0];
    assert!(validate(&bad_y, None, &pattern, 4, 2, 1, 2, &base).is_err());
    assert!(validate(&y, None, &[0, 1], 4, 2, 1, 2, &base).is_err());
    assert!(validate(&y, Some(&[false; 8]), &pattern, 4, 2, 1, 2, &base).is_err());
    assert!(validate(&[0; 8], None, &pattern, 4, 2, 1, 2, &base).is_err());
    let cross = [1u8, 1, 1, 1];
    assert!(validate(&y, None, &cross, 4, 2, 2, 2, &base).is_err());
    assert!(validate(
        &[],
        None,
        &[],
        1,
        GP_MAX_NODES,
        1,
        2,
        &GpcmConfig {
            xi_rule: XiRuleKind::Halton,
            xi_points: GP_MAX_NODES,
            ..base
        },
    )
    .unwrap_err()
    .contains("count table"));
    assert!(validate(
        &[],
        None,
        &[],
        1,
        usize::MAX,
        1,
        2,
        &GpcmConfig {
            xi_rule: XiRuleKind::Halton,
            xi_points: GP_MAX_NODES,
            ..base
        },
    )
    .unwrap_err()
    .contains("overflows usize"));
    assert!(validate(&[], None, &[], usize::MAX, 2, 1, 2, &base,)
        .unwrap_err()
        .contains("n_persons * n_items"));

    for xi_rule in [XiRuleKind::Halton, XiRuleKind::MonteCarlo] {
        let result = fit_gpcm(
            &y,
            Some(&observed),
            &pattern,
            4,
            2,
            1,
            2,
            &GpcmConfig {
                xi_rule,
                xi_points: 16,
                xi_seed: 0,
                ..base
            },
        )
        .unwrap();
        assert_eq!(result.n_iter, 1);
        assert_eq!(result.termination_reason, "max_iter_reached");
        assert!(result.loglik_trace.iter().all(|value| value.is_finite()));
        assert!(result.theta.iter().all(|value| value.is_finite()));
    }
}

#[test]
fn gpcm_optimizer_and_em_diagnostics_cover_defensive_paths() {
    let dims = [0usize];
    let nodes = [-2.0, 0.0, 2.0];
    let zero_counts = vec![vec![0.0; 3]; 3];
    let initial = vec![1.0, 0.0, 0.0];
    assert_eq!(
        gpcm_m_step(initial.clone(), &dims, &nodes, 1, &zero_counts, 3, 0.1, 2),
        initial
    );

    let separated_counts = vec![
        vec![1000.0, 0.0, 0.0],
        vec![0.0, 1000.0, 0.0],
        vec![0.0, 0.0, 1000.0],
    ];
    let updated = gpcm_m_step(
        vec![0.0, 0.0, 0.0],
        &dims,
        &nodes,
        1,
        &separated_counts,
        3,
        -1.0e6,
        2,
    );
    assert!(updated.iter().all(|value| value.is_finite()));

    assert_eq!(checked_em_loglik_change(-10.0, None, 0).unwrap(), None);
    assert_eq!(
        checked_em_loglik_change(-9.5, Some(-10.0), 1).unwrap(),
        Some(0.5)
    );
    assert!(checked_em_loglik_change(f64::NAN, None, 2)
        .unwrap_err()
        .contains("non-finite"));
    assert!(checked_em_loglik_change(-10.5, Some(-10.0), 3)
        .unwrap_err()
        .contains("decreased"));
}

fn design_d2(n_cat: usize) -> (Vec<u8>, usize, Vec<f64>, Vec<f64>) {
    let n_dims = 2usize;
    let m1 = n_cat - 1;
    let pattern: Vec<u8> = vec![1, 0, 1, 0, 0, 1, 0, 1, 1, 1];
    let n_items = 5usize;
    let mut slope = vec![0.0f64; n_items * n_dims];
    slope[0 * n_dims + 0] = 1.4;
    slope[1 * n_dims + 0] = 1.0;
    slope[2 * n_dims + 1] = 1.2;
    slope[3 * n_dims + 1] = 1.1;
    slope[4 * n_dims + 0] = -1.0; // negative cross-loader (dim0 anchor item 0 positive)
    slope[4 * n_dims + 1] = 0.9;
    let mut step = vec![0.0f64; n_items * m1];
    for i in 0..n_items {
        step[i * m1] = 0.5 + 0.05 * i as f64; // non-monotone across k
        if m1 > 1 {
            step[i * m1 + 1] = -0.4 + 0.03 * i as f64;
        }
    }
    (pattern, n_items, slope, step)
}

/// D = 2 recovery on GH nodes: pure anchors + a NEGATIVE cross-loader on dim0 (positively
/// anchored). Asserts slope recovery, STEP recovery (numeric — GPCM steps are unordered, no
/// ordering canary), per-dim EAP, finite steps, EM monotone.
#[test]
fn gpcm_recovers_d2_with_negative_cross_loader() {
    let (n_dims, n_cat) = (2usize, 3usize);
    let (pattern, n_items, slope, step) = design_d2(n_cat);
    let n = 6000usize;
    let mut rng = Lcg(3535);
    let mut theta = vec![0.0f64; n * n_dims];
    for v in theta.iter_mut() {
        *v = rng.normal();
    }
    let y = simulate(&slope, &step, &theta, n, n_items, n_dims, n_cat, &mut rng);
    let cfg = GpcmConfig {
        q: 21,
        ..GpcmConfig::default()
    };
    let res = fit_gpcm(&y, None, &pattern, n, n_items, n_dims, n_cat, &cfg).unwrap();
    assert!(res.converged);
    for i in 0..n_items {
        for d in 0..n_dims {
            if pattern[i * n_dims + d] == 0 {
                assert_eq!(res.slope[i * n_dims + d], 0.0, "off-pattern zero");
            }
        }
    }
    assert!(res.step.iter().all(|v| v.is_finite()), "finite steps");
    assert!(res.slope[0 * n_dims + 0] > 0.5, "anchor0 positive");
    assert!(res.slope[2 * n_dims + 1] > 0.5, "anchor2 positive");
    assert!(
        res.slope[4 * n_dims + 0] < -0.4,
        "neg cross-loader: {}",
        res.slope[4 * n_dims + 0]
    );
    assert!(
        rmse(&res.slope, &slope) < 0.16,
        "slope RMSE {}",
        rmse(&res.slope, &slope)
    );
    assert!(
        rmse(&res.step, &step) < 0.16,
        "step RMSE {}",
        rmse(&res.step, &step)
    );
    for d in 0..n_dims {
        let th: Vec<f64> = (0..n).map(|j| res.theta[j * n_dims + d]).collect();
        let tt: Vec<f64> = (0..n).map(|j| theta[j * n_dims + d]).collect();
        assert!(corr(&th, &tt) > 0.6, "theta{d} corr {}", corr(&th, &tt));
    }
    for w in res.loglik_trace.windows(2) {
        assert!(w[1] >= w[0] - 1e-9, "EM monotone");
    }
}

/// The reflection canonicalization FIRES — and is WITNESSED by the raw EM mode landing on the
/// wrong side, so dropping the flip flips every assertion below (verified by mutation: disabling
/// the canonicalization block makes this test fail on all three sign checks).
///
/// The witness depends on which mirror mode raw EM converges to. Init is `+1.0` on each item's
/// first loaded dim (see `fit_gpcm`), so the dim0 axis is oriented by its STRONGEST-|slope|
/// loader. Here that is a positively-keyed CROSS-loader (`item1`, true `+1.7`), NOT the pure
/// anchor: raw EM therefore orients theta_0 to the +item1 axis (its true orientation), and the
/// WEAK reverse-keyed pure anchor (`item0`, true `-0.7`) converges NATIVELY NEGATIVE. Because the
/// pure anchor is the sole pure dim0 item, canonicalization must FLIP dim0 to make it positive —
/// negating item0 to `+0.7`, item1's dim0 slope to `-1.7`, and theta_0 to `-theta_0`. If the flip
/// is removed, item0 stays `-0.7` (anchor check fails), item1 stays `+1.7` (co-loader check
/// fails), and theta_0 stays positively correlated with truth (theta check fails). The STEPS are
/// invariant under the joint (slope, theta) flip (GPCM steps are unordered — no ordering canary —
/// so a reflection bug that also negated the steps could only be caught by this value check).
#[test]
fn gpcm_reflection_fires_on_negative_anchor() {
    let (n_dims, n_cat) = (2usize, 3usize);
    let m1 = n_cat - 1;
    // item0: WEAK reverse-keyed SOLE pure anchor on dim0 -> converges raw-NEGATIVE.
    // item1: STRONG positively-keyed cross-loader on dim0 -> dominates the dim0 orientation, so
    //        raw EM does NOT land the anchor in the canonical (positive) mode on its own.
    let pattern: Vec<u8> = vec![1, 0, 1, 1, 0, 1, 0, 1];
    let n_items = 4usize;
    let mut slope = vec![0.0f64; n_items * n_dims];
    slope[0 * n_dims + 0] = -0.7; // weak reverse-keyed SOLE pure anchor on dim0
    slope[1 * n_dims + 0] = 1.7; // strong cross-loader, positively keyed on dim0 (sets the axis)
    slope[1 * n_dims + 1] = 0.6;
    slope[2 * n_dims + 1] = 1.2; // pure anchor on dim1 (positively keyed -> dim1 not flipped)
    slope[3 * n_dims + 1] = 1.0;
    // non-monotone steps (unordered) so a step-negating reflection bug is caught by the RMSE check
    let mut step = vec![0.0f64; n_items * m1];
    for i in 0..n_items {
        step[i * m1] = 0.6;
        step[i * m1 + 1] = -0.5;
    }
    let n = 6000usize;
    let mut rng = Lcg(6262);
    let mut theta = vec![0.0f64; n * n_dims];
    for v in theta.iter_mut() {
        *v = rng.normal();
    }
    let y = simulate(&slope, &step, &theta, n, n_items, n_dims, n_cat, &mut rng);
    let cfg = GpcmConfig {
        q: 21,
        ..GpcmConfig::default()
    };
    let res = fit_gpcm(&y, None, &pattern, n, n_items, n_dims, n_cat, &cfg).unwrap();
    // canon FIRED: anchor flipped +, strong co-loader flipped -, theta_0 flipped (all three would
    // fail with the flip removed, because raw EM lands the anchor negative / co-loader positive).
    assert!(
        res.slope[0 * n_dims + 0] > 0.3,
        "reflected anchor positive: {}",
        res.slope[0 * n_dims + 0]
    );
    assert!(
        res.slope[1 * n_dims + 0] < -0.5,
        "co-loader flipped negative: {}",
        res.slope[1 * n_dims + 0]
    );
    // steps UNCHANGED by the reflection (recovered close to truth) — the unordered-step analogue
    // of the GRM's ordering canary: a step-negating reflection bug would blow this up.
    assert!(
        rmse(&res.step, &step) < 0.15,
        "steps preserved: RMSE {}",
        rmse(&res.step, &step)
    );
    // flipped dim0: EAP theta_0 correlates NEGATIVELY with truth; unflipped dim1 positive.
    let th0: Vec<f64> = (0..n).map(|j| res.theta[j * n_dims + 0]).collect();
    let tt0: Vec<f64> = (0..n).map(|j| theta[j * n_dims + 0]).collect();
    let th1: Vec<f64> = (0..n).map(|j| res.theta[j * n_dims + 1]).collect();
    let tt1: Vec<f64> = (0..n).map(|j| theta[j * n_dims + 1]).collect();
    assert!(
        corr(&th0, &tt0) < -0.5,
        "flipped-dim theta corr negative: {}",
        corr(&th0, &tt0)
    );
    assert!(
        corr(&th1, &tt1) > 0.5,
        "unflipped-dim theta corr positive: {}",
        corr(&th1, &tt1)
    );
}

/// Structural invariants + validation guards (constructed non-vacuously — the intended guard is
/// the failing branch).
#[test]
fn gpcm_validates_and_structural_invariants() {
    let (n_dims, n_cat) = (2usize, 3usize);
    let (pattern, n_items, slope, step) = design_d2(n_cat);
    let n = 500usize;
    let mut rng = Lcg(88);
    let mut theta = vec![0.0f64; n * n_dims];
    for v in theta.iter_mut() {
        *v = rng.normal();
    }
    let y = simulate(&slope, &step, &theta, n, n_items, n_dims, n_cat, &mut rng);
    let cfg = GpcmConfig {
        q: 15,
        max_iter: 25,
        ..GpcmConfig::default()
    };
    let res = fit_gpcm(&y, None, &pattern, n, n_items, n_dims, n_cat, &cfg).unwrap();
    assert_eq!(res.n_parameters, 4 * (1 + 2) + (2 + 2));
    let lp = gpcm_logprobs(0.4, &[0.0, 1.0, 2.0], &[0.0, 0.6, -0.4]);
    let s: f64 = lp.iter().map(|l| l.exp()).sum();
    assert!((s - 1.0).abs() < 1e-12);
    // GH D=4 rejected (y4 observes every category so the D-bound is the sole reason)
    let gh4 = GpcmConfig::default();
    let pat4: Vec<u8> = (0..4)
        .flat_map(|d| (0..4).map(move |k| (k == d) as u8))
        .collect();
    let y4: Vec<usize> = (0..n * 4).map(|idx| idx % n_cat).collect();
    assert!(
        fit_gpcm(&y4, None, &pat4, n, 4, 4, n_cat, &gh4).is_err(),
        "GH D=4 rejected"
    );
    // no pure anchor (3-item all-both pattern with the full 3-item y so the anchor guard fires)
    let no_anchor: Vec<u8> = vec![1, 1, 1, 1, 1, 1];
    assert!(
        fit_gpcm(&y, None, &no_anchor, n, n_items, n_dims, n_cat, &cfg).is_err(),
        "no pure anchor rejected"
    );
    let mut ybad = y.clone();
    ybad[0] = n_cat;
    assert!(
        fit_gpcm(&ybad, None, &pattern, n, n_items, n_dims, n_cat, &cfg).is_err(),
        "bad category rejected"
    );
    let mut ygap = y.clone();
    for p in 0..n {
        if ygap[p * n_items + 0] == 1 {
            ygap[p * n_items + 0] = 0;
        }
    }
    assert!(
        fit_gpcm(&ygap, None, &pattern, n, n_items, n_dims, n_cat, &cfg).is_err(),
        "unobserved category rejected"
    );
}

/// Literature-grade Monte-Carlo (>=500 reps): recover the multidimensional GPCM at D=2 and D=3
/// under normal AND per-dim-standardized right-skew traits. Per-rep monotone-EM + STEP finiteness
/// canaries (a diverging step is GPCM's characteristic failure mode).
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn mc_gpcm_recovery_500() {
    let reps = 500usize;
    let n_cat = 3usize;
    let m1 = n_cat - 1;
    for &(n_dims, q, n) in [(2usize, 15usize, 2500usize), (3usize, 11usize, 2000usize)].iter() {
        let mut pattern: Vec<u8> = Vec::new();
        for d in 0..n_dims {
            for _ in 0..2 {
                let mut r = vec![0u8; n_dims];
                r[d] = 1;
                pattern.extend_from_slice(&r);
            }
        }
        for d in 0..n_dims {
            let mut r = vec![0u8; n_dims];
            r[d] = 1;
            r[(d + 1) % n_dims] = 1;
            pattern.extend_from_slice(&r);
        }
        let n_items = 2 * n_dims + n_dims;
        let mut slope = vec![0.0f64; n_items * n_dims];
        for d in 0..n_dims {
            slope[(2 * d) * n_dims + d] = 1.3;
            slope[(2 * d + 1) * n_dims + d] = 1.0;
        }
        for d in 0..n_dims {
            let ci = 2 * n_dims + d;
            slope[ci * n_dims + d] = 1.0;
            slope[ci * n_dims + (d + 1) % n_dims] = if d % 2 == 0 { 0.7 } else { -0.7 };
        }
        let mut step = vec![0.0f64; n_items * m1];
        for i in 0..n_items {
            step[i * m1] = 0.6 + 0.03 * i as f64;
            step[i * m1 + 1] = -0.5 + 0.02 * i as f64;
        }
        for &skew in [false, true].iter() {
            let (mut lnum, mut lden, mut lbias) = (0.0f64, 0.0f64, 0.0f64);
            let (mut snum, mut sden) = (0.0f64, 0.0f64);
            let (mut csum, mut ccnt) = (0.0f64, 0.0f64);
            let mut nconv = 0usize;
            for rep in 0..reps {
                let mut rng = Lcg(0x9E3779B97F4A7C15u64
                    .wrapping_mul(rep as u64 + 1)
                    .wrapping_add((skew as u64 + 1) * 0xD1B54A32D192ED03)
                    .wrapping_add(n_dims as u64 * 0x100000001B3));
                let mut theta = vec![0.0f64; n * n_dims];
                for d in 0..n_dims {
                    let col: Vec<f64> = (0..n)
                        .map(|_| {
                            if skew {
                                let mut cc = 0.0;
                                for _ in 0..3 {
                                    let z = rng.normal();
                                    cc += z * z;
                                }
                                (cc - 3.0) / 6f64.sqrt()
                            } else {
                                rng.normal()
                            }
                        })
                        .collect();
                    let m = col.iter().sum::<f64>() / n as f64;
                    let v = col.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n as f64;
                    let sd = v.sqrt();
                    for j in 0..n {
                        theta[j * n_dims + d] = (col[j] - m) / sd;
                    }
                }
                let y = simulate(&slope, &step, &theta, n, n_items, n_dims, n_cat, &mut rng);
                let cfg = GpcmConfig {
                    q,
                    ..GpcmConfig::default()
                };
                let res = fit_gpcm(&y, None, &pattern, n, n_items, n_dims, n_cat, &cfg).unwrap();
                if res.converged {
                    nconv += 1;
                }
                for w in res.loglik_trace.windows(2) {
                    assert!(w[1] >= w[0] - 1e-9, "monotone (rep {rep})");
                }
                assert!(
                    res.slope.iter().all(|v| v.is_finite()),
                    "finite slope (rep {rep})"
                );
                assert!(
                    res.step.iter().all(|v| v.is_finite()),
                    "finite step (rep {rep})"
                );
                for i in 0..n_items {
                    for d in 0..n_dims {
                        if pattern[i * n_dims + d] != 0 {
                            let e = res.slope[i * n_dims + d] - slope[i * n_dims + d];
                            lnum += e * e;
                            lden += 1.0;
                            lbias += e;
                        }
                    }
                }
                for i in 0..n_items {
                    for j in 0..m1 {
                        let e = res.step[i * m1 + j] - step[i * m1 + j];
                        snum += e * e;
                        sden += 1.0;
                    }
                }
                for d in 0..n_dims {
                    let th: Vec<f64> = (0..n).map(|j| res.theta[j * n_dims + d]).collect();
                    let tt: Vec<f64> = (0..n).map(|j| theta[j * n_dims + d]).collect();
                    csum += corr(&th, &tt);
                    ccnt += 1.0;
                }
            }
            let lrmse = (lnum / lden).sqrt();
            let srmse = (snum / sden).sqrt();
            let (lb, tc, conv) = (lbias / lden, csum / ccnt, nconv as f64 / reps as f64);
            println!(
                "[gpcm-mirt MC D={n_dims} q={q} N={n} skew={skew}] reps={reps} conv={conv:.3} \
                 loadRMSE={lrmse:.4} loadBias={lb:.4} stepRMSE={srmse:.4} thetaCorr={tc:.3}"
            );
            assert!(conv > 0.90, "convergence {conv} (D={n_dims} skew={skew})");
            if skew {
                assert!(lrmse < 0.24, "skew load RMSE {lrmse} (D={n_dims})");
                assert!(tc > 0.55, "skew theta corr {tc} (D={n_dims})");
            } else {
                assert!(lb.abs() < 0.06, "load bias {lb} (D={n_dims})");
                assert!(lrmse < 0.16, "load RMSE {lrmse} (D={n_dims})");
                assert!(srmse < 0.16, "step RMSE {srmse} (D={n_dims})");
                assert!(tc > 0.6, "theta corr {tc} (D={n_dims})");
            }
        }
    }
}
