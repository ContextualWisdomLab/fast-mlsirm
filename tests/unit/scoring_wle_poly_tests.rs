use super::{poly_wle_terms, score_wle, score_wle_poly};
use crate::poly::{gpcm_logprobs, grm_logprobs, poly_item_information, PolyModel};

/// `I'(theta)` obtained by central-differencing the SHIPPED [`poly_item_information`], which is a
/// different code path from [`poly_wle_terms`]: the former sums `(v_k - v_{k+1})^2 / P_k` (GRM) or
/// `a^2 Var(k)` (GPCM), the latter accumulates `sum_k P_k r1_k^2` from the division-free ratios.
/// Both sides of the A1 comparison are therefore crate values, neither recomputed in the test.
fn info_prime_fd(theta: f64, a: f64, cat: &[f64], model: PolyModel) -> f64 {
    let h = 1e-5;
    (poly_item_information(theta + h, a, cat, model) - poly_item_information(theta - h, a, cat, model))
        / (2.0 * h)
}

/// A1. `J == I'` for both shipped families, at asymmetric non-centred parameters and off-centre theta.
///
/// This identity is a PROVED property of these two families (see the verification note on
/// `score_wle_poly`): `J - I' = -E[l' l'']`, which vanishes because the GPCM's `l''` is
/// category-free and the GRM's sum telescopes through `v_0 = v_K = 0`. It is used here as an
/// ORACLE ONLY -- the implementation computes `J` directly and never substitutes `I'`.
///
/// NOTE ON WHAT THIS CANNOT SEE: because `J = I'` is EXACT for both shipped families, an
/// implementation that replaced `J` by a numerical derivative of `I` would be behaviour-preserving
/// here and no polytomous test can detect it. The discriminating anchors for that substitution are in
/// the dichotomous suite, where a lower asymptote breaks the identity —
/// `wle_estimating_equation_root` (3PL) and `wle_information_derivative_identity`. A NEW family added
/// later must re-derive `J` rather than assume it.
///
/// The reference magnitudes are PINNED rather than narrated, which also subsumes a non-vacuity check
/// (a zeroed `jterm` fails them) and freezes the fixture against silent parameter edits. Values
/// measured on these exact fixtures at theta = +0.9: GRM K=5 `J = -0.068761`, GPCM K=4
/// `J = -0.451118`.
///
/// kills: sign flip on the `jterm` accumulator; a dropped category term; a `v`/`s` index off-by-one;
/// a silent edit to the fixture parameters; a symmetric fixture that passes because both sides are 0.
#[test]
fn poly_wle_j_equals_info_derivative() {
    let cases: [(PolyModel, f64, Vec<f64>, f64); 2] = [
        (PolyModel::Grm, 2.1, vec![2.3, 0.4, -0.7, -2.9], -0.068761),
        (PolyModel::Gpcm, 1.3, vec![0.4, -0.3, -1.9], -0.451118),
    ];
    for (model, a, cat, j_at_0_9) in cases {
        for theta in [-1.6, -0.3, 0.9, 2.4] {
            let (info, jterm) = poly_wle_terms(theta, a, &cat, model);
            let iprime = info_prime_fd(theta, a, &cat, model);
            let scale = jterm.abs().max(iprime.abs()).max(1e-8);
            assert!(
                (jterm - iprime).abs() / scale < 1e-6,
                "{model:?} theta={theta}: J={jterm} but I'={iprime}"
            );
            // the shipped information must also agree with the accumulator's own I
            let info_ref = poly_item_information(theta, a, &cat, model);
            assert!(
                (info - info_ref).abs() / info_ref.max(1e-12) < 1e-12,
                "{model:?} theta={theta}: I={info} vs poly_item_information={info_ref}"
            );
        }
        // the reference magnitude is pinned, not merely asserted to be "not tiny"
        let (_, j_mid) = poly_wle_terms(0.9, a, &cat, model);
        assert!(
            (j_mid - j_at_0_9).abs() < 1e-5,
            "{model:?}: J at theta=0.9 is {j_mid}, expected {j_at_0_9}"
        );
    }
}

/// A2. A LEMMA ABOUT THE ORACLE, NOT A TEST OF THE CODE — and the distinction is the whole point.
///
/// It would be natural to claim this "kills an implementation that sets `J := I'`". IT DOES NOT, AND
/// NOTHING HERE CAN: `J = I'` is an exact identity for both shipped families, so substituting one for
/// the other is BEHAVIOUR-PRESERVING for GRM and GPCM and no polytomous fixture can distinguish them.
/// The discriminating anchor for that substitution lives in the DICHOTOMOUS suite, where `c > 0`
/// breaks the identity: `wle_estimating_equation_root` (3PL, `c = 0.2`) and
/// `wle_information_derivative_identity` both fail under it. An earlier version of this comment
/// asserted the kill anyway, which was false.
///
/// What this DOES establish is that A1's oracle is non-trivial — `J = I'` is a property of these two
/// families and not an arithmetic tautology — by exhibiting two response functions where it fails:
/// a graded model with PER-BOUNDARY slopes (the `a^3` cannot be factored out of the telescoping sum)
/// and the 3PL (a lower asymptote). Both are computed here from finite differences because the crate
/// has no such family; that is exactly why they cannot test crate code.
///
/// The final block IS wired to the crate: it asserts the shipped GRM satisfies the identity on the
/// same `theta` where the per-boundary variant violates it, so the contrast is between a crate value
/// and a local one rather than between two local ones.
#[test]
fn poly_wle_identity_is_family_specific_not_universal() {
    // (i) graded response with per-boundary slopes a_j -- J and I' must DISAGREE
    let a = [1.3f64, 0.7, 1.9];
    let beta = [1.6f64, 0.1, -1.2];
    let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
    // P_k(theta) for the 4-category per-boundary-slope graded model
    let probs = |t: f64| -> Vec<f64> {
        let s: Vec<f64> = (0..3).map(|j| sig(a[j] * t + beta[j])).collect();
        vec![1.0 - s[0], s[0] - s[1], s[1] - s[2], s[2]]
    };
    let h = 1e-4;
    let terms = |t: f64| -> (f64, f64) {
        let (p0, pm, pp) = (probs(t), probs(t - h), probs(t + h));
        let (mut info, mut jterm) = (0.0, 0.0);
        for k in 0..4 {
            let d1 = (pp[k] - pm[k]) / (2.0 * h);
            let d2 = (pp[k] - 2.0 * p0[k] + pm[k]) / (h * h);
            info += d1 * d1 / p0[k];
            jterm += d1 * d2 / p0[k];
        }
        (info, jterm)
    };
    // measured relative gaps: 0.920 at theta = -1.6, 1.171 at theta = +0.9
    for theta in [-1.6f64, 0.9] {
        let (_, j) = terms(theta);
        let iprime = (terms(theta + h).0 - terms(theta - h).0) / (2.0 * h);
        let scale = j.abs().max(iprime.abs()).max(1e-8);
        assert!(
            (j - iprime).abs() / scale > 0.5,
            "per-boundary slopes must BREAK J == I' (theta={theta}): J={j} I'={iprime}"
        );
    }

    // (ii) the shipped 3PL path: c > 0 breaks the identity too
    let (a3, b3, c3) = (1.5f64, 0.2, 0.25);
    let p3 = |t: f64| c3 + (1.0 - c3) * sig(a3 * (t - b3));
    let terms3 = |t: f64| -> (f64, f64) {
        let (p0, pm, pp) = (p3(t), p3(t - h), p3(t + h));
        let d1 = (pp - pm) / (2.0 * h);
        let d2 = (pp - 2.0 * p0 + pm) / (h * h);
        let pq = p0 * (1.0 - p0);
        (d1 * d1 / pq, d1 * d2 / pq)
    };
    let theta = -1.6;
    let (_, j3) = terms3(theta);
    let ip3 = (terms3(theta + h).0 - terms3(theta - h).0) / (2.0 * h);
    // measured relative gap 0.474 at c = 0.25, theta = -1.6
    assert!(
        (j3 / ip3 - 1.0).abs() > 0.3,
        "a lower asymptote must break J == I': J={j3} I'={ip3}"
    );

    // ...and on the SAME theta the SHIPPED graded family does satisfy it. This is the only assertion
    // in this test that reads crate output, and it is what makes the contrast meaningful rather than
    // a statement about the test's own arithmetic: the identity holds for what we ship and fails for
    // the two variants above.
    let (_, j_shipped) = poly_wle_terms(-1.6, 1.3, &beta, PolyModel::Grm);
    let ip_shipped = info_prime_fd(-1.6, 1.3, &beta, PolyModel::Grm);
    let scale = j_shipped.abs().max(ip_shipped.abs()).max(1e-8);
    assert!(
        (j_shipped - ip_shipped).abs() / scale < 1e-6,
        "the shipped GRM must satisfy J == I' where the per-boundary variant does not: \
         J={j_shipped} I'={ip_shipped}"
    );
}

/// A3. CROSS-PATH PIN AT K = 2. At two categories both polytomous families ARE the 2PL, so
/// `score_wle_poly` must reproduce the shipped dichotomous `score_wle`.
///
/// Non-discriminating as a DESIGN argument -- it says nothing about which `J` formula is right,
/// because at K = 2 every candidate agrees. It is a PLUMBING pin, and it is the only anchor that
/// catches the whole layout class.
///
/// kills: a dropped factor of 1/2; a category-index off-by-one; an item/person layout transpose; a
/// missing chain-rule `a`; a `cat_params` block-stride bug.
#[test]
fn poly_wle_reduces_to_dichotomous_wle_at_two_categories() {
    // unequal slopes and asymmetric difficulties: a dropped `a` would be invisible at a = 1
    let a = [1.4f64, 0.8, 2.1, 1.1, 0.6];
    let b = [-0.9f64, 0.3, 1.4, -1.7, 0.8];
    let n_items = a.len();
    let n_persons = 4usize;
    // off-centre patterns, not all-same
    let y_bin: Vec<f64> = vec![
        1.0, 1.0, 0.0, 1.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, 0.0, //
        1.0, 1.0, 1.0, 1.0, 0.0, //
        1.0, 0.0, 0.0, 1.0, 1.0,
    ];
    let y_cat: Vec<usize> = y_bin.iter().map(|v| *v as usize).collect();
    let c = vec![0.0; n_items];
    let d = vec![1.0; n_items];
    let observed = vec![true; n_persons * n_items];
    let dich = score_wle(&a, &b, &c, &d, &y_bin, &observed, n_persons, 6.0, 1e-10).unwrap();

    // GPCM at K=2: psi = [0, a*theta + c_1] with c_1 = -a*b  =>  logit = a(theta - b)
    let gpcm_cat: Vec<f64> = (0..n_items).map(|i| -a[i] * b[i]).collect();
    // GRM at K=2: single threshold beta_0 with logit = a*theta + beta_0  =>  beta_0 = -a*b
    let grm_cat: Vec<f64> = gpcm_cat.clone();
    for (model, cat) in [(PolyModel::Gpcm, &gpcm_cat), (PolyModel::Grm, &grm_cat)] {
        let poly = score_wle_poly(
            &y_cat, None, n_persons, n_items, 2, &a, cat, model, 6.0, 1e-10,
        )
        .unwrap();
        for p in 0..n_persons {
            assert!(
                (poly.theta[p] - dich.theta[p]).abs() < 1e-6,
                "{model:?} person {p}: poly theta {} != dichotomous {}",
                poly.theta[p],
                dich.theta[p]
            );
            assert!(
                (poly.se[p] - dich.se[p]).abs() < 1e-6,
                "{model:?} person {p}: poly se {} != dichotomous {}",
                poly.se[p],
                dich.se[p]
            );
        }
    }
}

/// A4. GLOBAL-MODE FIXTURE, GPCM. Both polytomous log-likelihoods are log-concave, but the Warm
/// WEIGHT is not, so `Phi` is genuinely multimodal. On this bank an independent high-precision scan
/// found stationary points at `+0.098789` (`Phi = -4.720843`), `+0.377438` (a minimum) and
/// `+1.331420` (`Phi = -3.878708`, the GLOBAL maximum, `dPhi = 0.842135`); `max lnL'' = -5.55e-5`,
/// confirming the second mode comes entirely from the weight.
///
/// kills: replacing the grid scan with a single bracketed root, which converges to `+0.0988` and
/// errs by 1.23 logits. A residual-only check cannot catch this -- `g` vanishes at ALL THREE roots.
#[test]
fn poly_wle_gpcm_takes_the_global_mode_not_the_first_root() {
    let slope = [2.42f64, 1.09, 1.53];
    let cat_params = [
        -3.78f64, 0.50, 3.70, -1.91, //
        3.11, -2.12, -2.68, -1.68, //
        -0.35, -0.09, -3.78, 2.17,
    ];
    let y = [3usize, 2, 4];
    let out = score_wle_poly(
        &y, None, 1, 3, 5, &slope, &cat_params, PolyModel::Gpcm, 8.0, 1e-10,
    )
    .unwrap();
    assert!(
        (out.theta[0] - 1.331420).abs() < 1e-3,
        "expected the global mode 1.331420, got {}",
        out.theta[0]
    );
    assert!(
        (out.theta[0] - 0.098789).abs() > 0.1,
        "returned the leftmost stationary point 0.098789, i.e. a bracketed-root solver"
    );
    assert!(!out.boundary[0]);
}

/// A5. GLOBAL-MODE FIXTURE, GRM. Same failure mode for the other family, where the error is larger
/// (2.36 logits) and so would be missed by a GPCM-only fixture. High-precision scan: stationary
/// points at `+3.340794` (`Phi = -9.846605`), `+3.749233` (minimum) and `+5.701184`
/// (`Phi = -9.294478`, GLOBAL, `dPhi = 0.552127`).
///
/// kills: the same bracketed-root substitution, in the GRM branch specifically.
#[test]
fn poly_wle_grm_takes_the_global_mode_not_the_first_root() {
    let slope = [1.164f64, 1.568];
    let cat_params = [
        0.603f64, 0.230, -2.244, //
        -3.301, -3.719, -9.479,
    ];
    let y = [1usize, 3];
    let out = score_wle_poly(
        &y, None, 1, 2, 4, &slope, &cat_params, PolyModel::Grm, 8.0, 1e-10,
    )
    .unwrap();
    assert!(
        (out.theta[0] - 5.701184).abs() < 1e-3,
        "expected the global mode 5.701184, got {}",
        out.theta[0]
    );
    assert!(
        (out.theta[0] - 3.340794).abs() > 0.1,
        "returned the leftmost stationary point 3.340794, i.e. a bracketed-root solver"
    );
}

/// A6. ESTIMATING-EQUATION RESIDUAL, from FINITE DIFFERENCES of the log-probability routines ONLY.
/// The closed forms the implementation ships (`r1`, `r2`) are never used here, so an analytic sign
/// error in the SCORE term -- which A1 cannot see, because A1 only pins `I` and `J` -- shows up as a
/// non-zero residual at the returned theta.
///
/// Cannot catch a wrong-MODE error (`g` vanishes at every stationary point); that is A4/A5.
///
/// kills: a sign error or missing chain-rule factor in `r1`; a wrong category index on the score.
#[test]
fn poly_wle_satisfies_the_estimating_equation_by_finite_difference() {
    let h = 1e-4;
    let cases: [(PolyModel, usize, Vec<f64>, Vec<f64>, Vec<usize>); 2] = [
        (
            PolyModel::Grm,
            4,
            vec![1.7, 0.9, 1.3],
            vec![1.6, 0.1, -1.2, 2.4, -0.5, -1.9, 0.8, -0.2, -2.6],
            vec![1, 3, 2],
        ),
        (
            PolyModel::Gpcm,
            4,
            vec![1.3, 2.0, 0.7],
            vec![0.4, -0.3, -1.9, 1.1, 0.2, -0.8, -0.6, 1.4, 0.3],
            vec![2, 0, 3],
        ),
    ];
    for (model, n_cat, slope, cat_params, y) in cases {
        let n_items = slope.len();
        let out = score_wle_poly(
            &y, None, 1, n_items, n_cat, &slope, &cat_params, model, 8.0, 1e-11,
        )
        .unwrap();
        let theta = out.theta[0];
        // rebuild g from numeric derivatives of the shipped log-probability routines
        let probs = |i: usize, t: f64| -> Vec<f64> {
            let pars = &cat_params[i * (n_cat - 1)..(i + 1) * (n_cat - 1)];
            let base = slope[i] * t;
            match model {
                PolyModel::Grm => grm_logprobs(base, pars).iter().map(|l| l.exp()).collect(),
                PolyModel::Gpcm => {
                    let scores: Vec<f64> = (0..n_cat).map(|c| c as f64).collect();
                    let mut ints = vec![0.0; n_cat];
                    ints[1..].copy_from_slice(pars);
                    gpcm_logprobs(base, &scores, &ints)
                        .iter()
                        .map(|l| l.exp())
                        .collect()
                }
            }
        };
        let (mut score, mut info, mut jterm) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..n_items {
            let (p0, pm, pp) = (probs(i, theta), probs(i, theta - h), probs(i, theta + h));
            for k in 0..n_cat {
                let d1 = (pp[k] - pm[k]) / (2.0 * h);
                let d2 = (pp[k] - 2.0 * p0[k] + pm[k]) / (h * h);
                info += d1 * d1 / p0[k];
                jterm += d1 * d2 / p0[k];
                if k == y[i] {
                    score += d1 / p0[k];
                }
            }
        }
        let g = score + jterm / (2.0 * info);
        assert!(
            g.abs() < 1e-4,
            "{model:?}: estimating function {g} != 0 at the returned theta {theta}"
        );
    }
}

/// A7. FINITENESS AND CORRECTION MAGNITUDE on an ASYMMETRIC bank. The maximum-likelihood estimate
/// diverges for the all-lowest and all-highest patterns; the WLE must stay finite and interior.
/// A symmetric bank would put both estimates at 0 and make the comparison vacuous, so every slope
/// differs and no threshold set is mirrored.
///
/// The "MLE diverges" claim is asserted in its checkable form: the UNWEIGHTED score, accumulated
/// independently here, keeps a constant sign across the whole grid, so it has no interior root.
///
/// kills: a zeroed or sign-flipped correction term; a boundary-flag inversion.
#[test]
fn poly_wle_is_finite_where_the_mle_diverges() {
    let cases: [(PolyModel, usize, Vec<f64>, Vec<f64>); 2] = [
        (
            PolyModel::Gpcm,
            4,
            vec![1.3, 0.9, 1.8, 0.7, 1.5],
            vec![
                0.4, -0.3, -1.9, 1.2, 0.1, -0.7, -0.5, 0.9, 1.6, 2.1, -1.1, 0.3, -0.8, 0.6, -2.2,
            ],
        ),
        (
            PolyModel::Grm,
            4,
            vec![1.3, 0.9, 1.8, 0.7, 1.5],
            vec![
                1.6, 0.1, -1.2, 2.4, -0.5, -1.9, 0.8, -0.2, -2.6, 1.1, 0.4, -0.9, 2.0, 0.7, -1.4,
            ],
        ),
    ];
    for (model, n_cat, slope, cat_params) in cases {
        let n_items = slope.len();
        for (pattern, label) in [(0usize, "all-lowest"), (n_cat - 1, "all-highest")] {
            let y = vec![pattern; n_items];
            let out = score_wle_poly(
                &y, None, 1, n_items, n_cat, &slope, &cat_params, model, 8.0, 1e-10,
            )
            .unwrap();
            assert!(
                out.theta[0].is_finite(),
                "{model:?} {label}: WLE must be finite where the MLE diverges"
            );
            assert!(
                !out.boundary[0] && out.theta[0].abs() <= 7.0,
                "{model:?} {label}: theta {} hit the bound",
                out.theta[0]
            );
            // the MLE really does diverge here: the unweighted score never changes sign
            let sign_at = |t: f64| -> f64 {
                let mut score = 0.0f64;
                for i in 0..n_items {
                    let pars = &cat_params[i * (n_cat - 1)..(i + 1) * (n_cat - 1)];
                    let (_, r1, _) = super::poly_wle_ratios(t, slope[i], pars, model);
                    score += r1[y[i]];
                }
                score
            };
            let first = sign_at(-7.0).signum();
            for step in 1..=28 {
                let t = -7.0 + 0.5 * step as f64;
                assert!(
                    sign_at(t).signum() == first,
                    "{model:?} {label}: the unweighted score changed sign at {t}, so the MLE does \
                     NOT diverge and this fixture proves nothing"
                );
            }
        }
    }
}

/// A8. The `(n_cat - 1)` grid factor is a derived worst-case margin whose ONLY observable
/// consequence is when the work limit is reached, so that is where it is pinned: a configuration
/// that crosses 65,536 intervals WITH the factor must be refused, while the same `(theta_bound,
/// max|a|)` at `n_cat = 2` must not be.
///
/// kills: silently dropping the `(n_cat - 1)` factor.
#[test]
fn poly_wle_grid_factor_scales_with_category_count() {
    // 2 * theta_bound * a * INTERVALS_PER_LOGIT * (K - 1) against MAX_GRID = 65536.
    // At theta_bound = 8, a = 300 the base is 19_200, so the limit falls BETWEEN K = 4 (57_600, must
    // be Ok) and K = 5 (76_800, must be Err). Straddling the boundary is what pins the multiplier as
    // (n_cat - 1): a merely-large K would also fail under any constant factor >= 4.
    let theta_bound = 8.0;
    let big_a = [300.0f64];
    let cat = |k: usize| -> Vec<f64> { (0..k - 1).map(|j| 1.0 - j as f64 * 0.5).collect() };
    assert!(
        score_wle_poly(
            &[2usize], None, 1, 1, 4, &big_a, &cat(4), PolyModel::Gpcm, theta_bound, 1e-10
        )
        .is_ok(),
        "K=4 needs 57600 intervals and must stay within the 65536 limit"
    );
    assert!(
        score_wle_poly(
            &[2usize], None, 1, 1, 5, &big_a, &cat(5), PolyModel::Gpcm, theta_bound, 1e-10
        )
        .is_err(),
        "K=5 needs 76800 intervals and must be refused rather than silently under-resolved"
    );
}

/// Validation is non-vacuous: each guard trips on its own, with `score_poly_eap`'s wording.
#[test]
fn poly_wle_validates() {
    let slope = [1.3f64, 0.9];
    let cat = [0.4f64, -0.3, 1.1, 0.2];
    let y = [1usize, 2, 0, 1];
    let ok = |n_cat: usize, tb: f64, tol: f64| {
        score_wle_poly(&y, None, 2, 2, n_cat, &slope, &cat, PolyModel::Gpcm, tb, tol)
    };
    assert!(ok(3, 6.0, 1e-8).is_ok());
    assert!(ok(1, 6.0, 1e-8).is_err(), "n_cat < 2");
    assert!(ok(3, 0.0, 1e-8).is_err(), "theta_bound must be positive");
    assert!(ok(3, 6.0, 0.0).is_err(), "tol must be positive");
    // response outside 0..n_cat
    let bad = [1usize, 9, 0, 1];
    assert!(
        score_wle_poly(&bad, None, 2, 2, 3, &slope, &cat, PolyModel::Gpcm, 6.0, 1e-8).is_err(),
        "observed category >= n_cat"
    );
    // ...but an UNOBSERVED out-of-range entry is ignored rather than rejected
    let mask = [true, false, true, true];
    assert!(
        score_wle_poly(&bad, Some(&mask), 2, 2, 3, &slope, &cat, PolyModel::Gpcm, 6.0, 1e-8).is_ok(),
        "an unobserved cell must not be validated as a response"
    );
    // zero discrimination everywhere
    assert!(
        score_wle_poly(&y, None, 2, 2, 3, &[0.0, 0.0], &cat, PolyModel::Gpcm, 6.0, 1e-8).is_err(),
        "at least one item must have nonzero discrimination"
    );
    // sizes
    assert!(
        score_wle_poly(&y, None, 2, 2, 3, &slope, &cat[..3], PolyModel::Gpcm, 6.0, 1e-8).is_err(),
        "cat_params size inconsistent"
    );
}

/// A person with no observed items has undefined ability: `NaN` with the boundary flag, never a
/// spurious `theta = 0` that would read as an average examinee.
#[test]
fn poly_wle_reports_nan_for_a_person_with_no_observed_items() {
    let slope = [1.3f64, 0.9];
    let cat = [0.4f64, -0.3, 1.1, 0.2];
    let y = [1usize, 2, 0, 1];
    let observed = [true, true, false, false];
    let out = score_wle_poly(
        &y,
        Some(&observed),
        2,
        2,
        3,
        &slope,
        &cat,
        PolyModel::Gpcm,
        6.0,
        1e-10,
    )
    .unwrap();
    assert!(out.theta[0].is_finite());
    assert!(out.theta[1].is_nan() && out.se[1].is_nan() && out.boundary[1]);

    // The OTHER two degenerate branches, which nothing else reaches: a bound so tight that the global
    // mode lies outside it must clamp AND flag; and a near-zero-information fit must report se = NaN
    // while still returning a finite theta.
    let extreme = [3usize, 3];
    let tight = score_wle_poly(
        &extreme, None, 1, 2, 3, &slope, &cat, PolyModel::Gpcm, 0.05, 1e-10,
    );
    // n_cat = 3 so category 3 is out of range; use an in-range extreme instead
    assert!(tight.is_err());
    let hi = [2usize, 2];
    let clamped =
        score_wle_poly(&hi, None, 1, 2, 3, &slope, &cat, PolyModel::Gpcm, 0.05, 1e-10).unwrap();
    assert!(
        clamped.boundary[0] && (clamped.theta[0].abs() - 0.05).abs() < 1e-12,
        "a mode beyond theta_bound must clamp to the bound and set boundary: theta={} bnd={}",
        clamped.theta[0],
        clamped.boundary[0]
    );
    let flat = score_wle_poly(
        &hi, None, 1, 2, 3, &[1e-7, 1e-7], &cat, PolyModel::Gpcm, 6.0, 1e-10,
    )
    .unwrap();
    assert!(
        flat.theta[0].is_finite() && flat.se[0].is_nan(),
        "a near-zero-information fit must return a finite theta with se = NaN: theta={} se={}",
        flat.theta[0],
        flat.se[0]
    );
}
