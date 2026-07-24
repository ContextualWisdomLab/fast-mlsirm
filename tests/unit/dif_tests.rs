use super::*;

/// Minimal LCG + Box-Muller normal (crate PRNG idiom) for the simulation anchors.
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

/// Build a stratified `(resp, group, matching)` sample from explicit per-stratum `(A,B,C,D)` cells
/// (ref-correct, ref-incorrect, focal-correct, focal-incorrect), with `matching[p]` = the stratum
/// index. Lets the deterministic anchor pin the arithmetic without engineering total scores.
fn build(cells: &[(usize, u64, u64, u64, u64)], n_levels: usize) -> (Vec<u8>, Vec<u8>, Vec<usize>) {
    let (mut resp, mut group, mut matching) = (Vec::new(), Vec::new(), Vec::new());
    let mut push = |g: u8, r: u8, m: usize, n: u64| {
        for _ in 0..n {
            resp.push(r);
            group.push(g);
            matching.push(m);
        }
    };
    for &(m, a, b, c, d) in cells {
        push(0, 1, m, a);
        push(0, 0, m, b);
        push(1, 1, m, c);
        push(1, 0, m, d);
    }
    assert!(n_levels > cells.iter().map(|c| c.0).max().unwrap());
    (resp, group, matching)
}

/// Deterministic anchor: two strata hand-computed off Holland & Thayer (1988), the RBG (1986)
/// variance, and the ETS delta/classification. Pins alpha_MH, the CONTINUITY-CORRECTED chi-square,
/// MH D-DIF, SE, STD-P-DIF (focal minus reference), and the C label. A dropped `-0.5`, a wrong
/// variance denominator, a sign flip, or a reference-minus-focal STD-P-DIF all fail here.
#[test]
fn mh_two_stratum_hand_anchor() {
    // Stratum 1: A=80 B=20 C=40 D=60; Stratum 2: A=60 B=40 C=30 D=70.
    let (resp, group, matching) = build(&[(1, 80, 20, 40, 60), (2, 60, 40, 30, 70)], 3);
    let st = mh_item_stats(&resp, &group, &matching, 3);
    // alpha = (80*60/200 + 60*70/200) / (20*40/200 + 40*30/200) = 45 / 10 = 4.5
    assert!((st.alpha_mh - 4.5).abs() < 1e-12, "alpha {}", st.alpha_mh);
    // D-DIF = -2.35 ln(4.5)
    assert!(
        (st.mh_d_dif - (-2.35 * 4.5_f64.ln())).abs() < 1e-10,
        "d_dif {}",
        st.mh_d_dif
    );
    // chi2 = (|140 - 105| - 0.5)^2 / 24.497487... = 34.5^2 / 24.497487 = 48.5865...
    assert!((st.chi2_mh - 48.58647).abs() < 1e-3, "chi2 {}", st.chi2_mh);
    // SE = 2.35 * sqrt(0.04762963) = 0.512869...
    assert!((st.se_d_dif - 0.512869).abs() < 1e-5, "se {}", st.se_d_dif);
    // STD-P-DIF = (100*(0.4-0.8) + 100*(0.3-0.6)) / 200 = -0.35  (focal - reference, negative)
    assert!(
        (st.std_p_dif - (-0.35)).abs() < 1e-12,
        "std_p {}",
        st.std_p_dif
    );
    assert!(st.p_value < 1e-6, "p {}", st.p_value);
    // |D-DIF|=3.53 >= 1.5 and 3.53 - 1.645*0.5129 = 2.69 > 1.0 and significant -> C
    assert_eq!(st.ets_class, EtsClass::C);
    // sign agreement: both effect sizes negative (against the focal group)
    assert!(st.mh_d_dif < 0.0 && st.std_p_dif < 0.0);
}

/// No-DIF symmetry: identical reference/focal conditional response rates within every stratum give
/// alpha_MH = 1, MH D-DIF = 0, STD-P-DIF = 0, and class A.
#[test]
fn mh_no_dif_symmetry() {
    // Each stratum: A/n_R == C/n_F exactly, so every 2x2 has odds ratio 1.
    let (resp, group, matching) = build(&[(1, 60, 40, 60, 40), (2, 30, 70, 30, 70)], 3);
    let st = mh_item_stats(&resp, &group, &matching, 3);
    assert!((st.alpha_mh - 1.0).abs() < 1e-12, "alpha {}", st.alpha_mh);
    assert!(st.mh_d_dif.abs() < 1e-10, "d_dif {}", st.mh_d_dif);
    assert!(st.std_p_dif.abs() < 1e-12, "std_p {}", st.std_p_dif);
    assert!(st.chi2_mh < 1e-9, "chi2 {}", st.chi2_mh);
    assert_eq!(st.ets_class, EtsClass::A);
}

/// Degenerate guard: a single-group stratum (focal absent) contributes nothing, and a perfectly
/// separated table (no informative stratum) yields NaN statistics and an Undefined class — NOT A.
#[test]
fn mh_degenerate_is_undefined_not_a() {
    // Only a reference group present at level 1 (no focal anywhere) -> no informative strata.
    let (resp, group, matching) = build(&[(1, 30, 20, 0, 0)], 2);
    let st = mh_item_stats(&resp, &group, &matching, 2);
    assert!(st.alpha_mh.is_nan(), "alpha {}", st.alpha_mh);
    assert!(st.mh_d_dif.is_nan(), "d_dif {}", st.mh_d_dif);
    assert!(st.se_d_dif.is_nan(), "se {}", st.se_d_dif);
    assert!(st.chi2_mh.is_nan() && st.p_value.is_nan());
    assert_eq!(st.ets_class, EtsClass::Undefined);

    // Perfect separation: reference always correct, focal always incorrect (sum B_m C_m = 0 ->
    // alpha_MH = +inf). Both groups present and both responses present across strata, so chi2 is
    // defined, but the delta metric is undefined.
    let (resp2, group2, matching2) = build(&[(1, 50, 0, 0, 50)], 2);
    let st2 = mh_item_stats(&resp2, &group2, &matching2, 2);
    assert!(st2.mh_d_dif.is_nan(), "sep d_dif {}", st2.mh_d_dif);
    assert_eq!(st2.ets_class, EtsClass::Undefined);
}

#[test]
fn dif_serialization_and_shared_validation_cover_every_boundary() {
    assert_eq!(EtsClass::A.as_str(), "A");
    assert_eq!(EtsClass::B.as_str(), "B");
    assert_eq!(EtsClass::C.as_str(), "C");
    assert_eq!(EtsClass::Undefined.as_str(), "U");

    let cfg = MhDifConfig::default();
    assert!(validate_dif_inputs(&[], &[], 0, 1, &cfg).is_err());
    assert!(validate_dif_inputs(&[], &[], MAX_CELLS + 1, 1, &cfg).is_err());
    assert!(validate_dif_inputs(&[0, 1], &[0, 1], 2, 2, &cfg).is_err());
    assert!(validate_dif_inputs(&[0, 1], &[0], 2, 1, &cfg).is_err());
}

/// Simulation anchor: a 2PL DGP with a uniform (b-shift) DIF planted on one item, no group impact.
/// MH flags the planted item as large (class B/C, BH-significant) with the delta sign matching the
/// shift (item harder for the focal group -> negative D-DIF, negative STD-P-DIF), and classifies the
/// clean items as A (negligible). The clean items are asserted by the ETS practical-significance
/// CLASS, not by the raw BH flag: MH chi-square is over-powered at large N and the DIF item's
/// presence in the number-correct total mildly contaminates the matching criterion, so a clean
/// item's chi-square can be BH-significant while its effect size stays negligible (the A/B/C
/// classification is exactly the guard against this; item purification is the standard remedy and is
/// out of scope here). The parametric IRT-LR DIF, which does not match on the observed total, is
/// checked on the planted item plus one clean item for cross-method agreement.
#[test]
fn mh_flags_planted_uniform_dif_and_agrees_with_irt_lr() {
    use crate::poly::{poly_dif_sweep, PolyModel};
    let (n, n_items) = (3000usize, 12usize);
    let a = vec![1.2f64; n_items];
    let mut b = vec![0.0f64; n_items];
    for (i, bi) in b.iter_mut().enumerate() {
        *bi = -0.8 + 0.14 * i as f64;
    }
    let dif_item = 6usize;
    let clean_item = 0usize;
    let b_focal_shift = 0.7; // item dif_item is HARDER for the focal group (uniform DIF)
    let mut rng = Lcg(0xD1F);
    let mut y = vec![0u8; n * n_items];
    let mut group = vec![0u8; n];
    for p in 0..n {
        let g = if p % 2 == 0 { 0u8 } else { 1u8 };
        group[p] = g;
        // equal ability distribution across groups (no impact) so DIF is isolated
        let theta = rng.normal();
        for i in 0..n_items {
            let mut bi = b[i];
            if i == dif_item && g == 1 {
                bi += b_focal_shift;
            }
            let pr = 1.0 / (1.0 + (-(a[i] * (theta - bi))).exp());
            y[p * n_items + i] = if rng.next_f64() < pr { 1 } else { 0 };
        }
    }
    let rows = mantel_haenszel_dif(&y, &group, n, n_items, &MhDifConfig::default()).unwrap();
    // the planted item is flagged and large, harder-for-focal (negative delta + std_p)
    let dr = &rows[dif_item];
    assert!(
        dr.flagged_bh,
        "planted item not BH-flagged (p={})",
        dr.p_value
    );
    assert!(
        dr.mh_d_dif < -0.8,
        "planted delta not large-negative: {}",
        dr.mh_d_dif
    );
    assert!(dr.std_p_dif < 0.0, "planted std_p sign: {}", dr.std_p_dif);
    assert!(
        matches!(dr.ets_class, EtsClass::B | EtsClass::C),
        "planted class {:?}",
        dr.ets_class
    );
    // clean items are class A (negligible) by the practical-significance classification
    for (i, r) in rows.iter().enumerate() {
        if i != dif_item {
            assert_eq!(
                r.ets_class,
                EtsClass::A,
                "clean item {i} class {:?}",
                r.ets_class
            );
            assert!(
                r.mh_d_dif.abs() < 1.0,
                "clean item {i} |delta| {}",
                r.mh_d_dif
            );
        }
    }
    // agreement with the parametric IRT-LR DIF (uniform DIF, which MH is designed to catch): both
    // flag the planted item and leave a clean item unflagged. Scoped to two studied items to keep
    // the (per-item multigroup EM) cost bounded.
    let yl: Vec<usize> = y.iter().map(|&v| v as usize).collect();
    let gl: Vec<usize> = group.iter().map(|&v| v as usize).collect();
    let studied = [dif_item, clean_item];
    let lr = poly_dif_sweep(
        &yl,
        None,
        &gl,
        2,
        n,
        n_items,
        2,
        PolyModel::Gpcm,
        Some(&studied),
        21,
        200,
        1e-5,
        0.05,
    )
    .unwrap();
    let lr_dif = lr.iter().find(|r| r.item == dif_item).unwrap();
    let lr_clean = lr.iter().find(|r| r.item == clean_item).unwrap();
    assert!(
        lr_dif.flagged_bh,
        "IRT-LR missed the planted item (p={})",
        lr_dif.p_value
    );
    assert!(
        !lr_clean.flagged_bh,
        "IRT-LR spuriously flagged the clean item"
    );
}

/// Validation guards trip non-vacuously.
#[test]
fn mh_validates() {
    let n = 20usize;
    let n_items = 4usize;
    let y = vec![1u8; n * n_items];
    let mut group = vec![0u8; n];
    for p in 0..n {
        group[p] = (p % 2) as u8;
    }
    let cfg = MhDifConfig::default();
    // ok baseline (degenerate everywhere but valid input -> Undefined rows, not an error)
    assert!(mantel_haenszel_dif(&y, &group, n, n_items, &cfg).is_ok());
    // response > 1
    let mut ybad = y.clone();
    ybad[0] = 2;
    assert!(mantel_haenszel_dif(&ybad, &group, n, n_items, &cfg).is_err());
    // group label > 1
    let mut gbad = group.clone();
    gbad[0] = 2;
    assert!(mantel_haenszel_dif(&y, &gbad, n, n_items, &cfg).is_err());
    // only one group present
    let gone = vec![0u8; n];
    assert!(mantel_haenszel_dif(&y, &gone, n, n_items, &cfg).is_err());
    // y length mismatch
    assert!(mantel_haenszel_dif(&y[..n * n_items - 1], &group, n, n_items, &cfg).is_err());
    // fdr_q out of range
    let badq = MhDifConfig { fdr_q: 0.0, ..cfg };
    assert!(mantel_haenszel_dif(&y, &group, n, n_items, &badq).is_err());
}

/// Rest-score matching (`exclude_studied_item=true`) puts persons in different strata than the
/// item-included total, so the studied item's MH statistics differ between the two modes and the
/// rest-score path runs without an out-of-bounds level. A mutation dropping the `- y_i` (leaving the
/// rest score equal to the total) would make the two modes identical.
#[test]
fn mh_rest_score_matching_differs_from_item_included() {
    let (n, n_items) = (1200usize, 6usize);
    let a = 1.2f64;
    let b = [-0.6, -0.3, 0.0, 0.3, 0.6, 0.9];
    let dif_item = 2usize;
    let mut rng = Lcg(0x5E5);
    let mut y = vec![0u8; n * n_items];
    let mut group = vec![0u8; n];
    for p in 0..n {
        let g = (p % 2) as u8;
        group[p] = g;
        let theta = rng.normal();
        for i in 0..n_items {
            let mut bi = b[i];
            if i == dif_item && g == 1 {
                bi += 1.0;
            }
            let pr = 1.0 / (1.0 + (-(a * (theta - bi))).exp());
            y[p * n_items + i] = if rng.next_f64() < pr { 1 } else { 0 };
        }
    }
    let incl = mantel_haenszel_dif(
        &y,
        &group,
        n,
        n_items,
        &MhDifConfig {
            exclude_studied_item: false,
            fdr_q: 0.05,
        },
    )
    .unwrap();
    let excl = mantel_haenszel_dif(
        &y,
        &group,
        n,
        n_items,
        &MhDifConfig {
            exclude_studied_item: true,
            fdr_q: 0.05,
        },
    )
    .unwrap();
    // rest-score path completes (n_levels correct) and still flags the planted item
    assert!(incl[dif_item].flagged_bh && excl[dif_item].flagged_bh);
    // the studied item's strata genuinely change between the two matching schemes
    assert!(
        (incl[dif_item].chi2_mh - excl[dif_item].chi2_mh).abs() > 1e-6,
        "rest-score identical to item-included: {} vs {}",
        incl[dif_item].chi2_mh,
        excl[dif_item].chi2_mh
    );
}

/// ETS A/B/C/Undefined boundaries pinned directly, including the ONE-SIDED 1.645 critical value for
/// the C rule: at `|D|=1.5, SE=0.28` the `|D| - 1.645 SE = 1.039 > 1.0` test passes (C) but the
/// `1.96` mutant (`0.951`) would fail (B).
#[test]
fn mh_classify_boundaries() {
    assert_eq!(classify(f64::NAN, 0.3, 0.001), EtsClass::Undefined); // undefined delta
    assert_eq!(classify(-3.0, 0.4, 0.20), EtsClass::A); // not significant -> A
    assert_eq!(classify(-0.8, 0.2, 0.001), EtsClass::A); // |D| < 1.0 -> A
    assert_eq!(classify(-1.3, 0.2, 0.001), EtsClass::B); // 1.0 <= |D| < 1.5 -> B
    assert_eq!(classify(-1.5, 0.28, 0.001), EtsClass::C); // C via the 1.645 test (1.96 -> B)
    assert_eq!(classify(-1.6, 1.0, 0.001), EtsClass::B); // |D|>=1.5 but not sig. above 1.0 -> B
}

/// STD-P-DIF uses the WIDER "both groups present" stratum gate, not the MH 4-marginal gate: an
/// all-correct stratum (`m0 = 0`, not MH-informative) still contributes focal weight to the
/// Dorans-Kulick standardization denominator. Under the stricter gate |STD-P-DIF| would inflate from
/// `40/150` to `40/100`.
#[test]
fn mh_std_p_dif_includes_all_correct_stratum_weight() {
    // Stratum 1 informative (DIF); stratum 2 both-groups all-correct (m0 = 0).
    let (resp, group, matching) = build(&[(1, 80, 20, 40, 60), (2, 50, 0, 50, 0)], 3);
    let st = mh_item_stats(&resp, &group, &matching, 3);
    // STD-P-DIF = (100*(0.4-0.8) + 50*(1.0-1.0)) / (100 + 50) = -40/150
    assert!(
        (st.std_p_dif - (-40.0 / 150.0)).abs() < 1e-12,
        "std_p {}",
        st.std_p_dif
    );
    // MH uses only the informative stratum 1: alpha = (80*60/200)/(20*40/200) = 6
    assert!((st.alpha_mh - 6.0).abs() < 1e-12, "alpha {}", st.alpha_mh);
}

// ---------------- Zumbo (1999) logistic regression DIF ----------------

/// Log-likelihood of `n` Bernoulli trials with `k` successes evaluated at the MLE `p = k/n`.
fn bin_ll(k: f64, n: f64) -> f64 {
    if n <= 0.0 {
        return 0.0;
    }
    let p = k / n;
    let a = if k > 0.0 { k * p.ln() } else { 0.0 };
    let b = if n - k > 0.0 {
        (n - k) * (1.0 - p).ln()
    } else {
        0.0
    };
    a + b
}

/// Expand per-cell `(score, group, n, k)` counts into person-level response/score/group vectors.
fn expand(cells: &[(f64, f64, usize, usize)]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut resp, mut score, mut group) = (Vec::new(), Vec::new(), Vec::new());
    for &(s, g, n, k) in cells {
        for j in 0..n {
            resp.push(if j < k { 1.0 } else { 0.0 });
            score.push(s);
            group.push(g);
        }
    }
    (resp, score, group)
}

/// SATURATED-DESIGN closed-form anchor. With a two-level matching score and a binary group,
/// `{1, S, G, S x G}` is saturated, so the M2 MLE fitted probabilities are exactly the four observed
/// cell proportions and `ll(M2)`, `ll(M0)` (pooled over group within score level) and the
/// intercept-only `ll_null` are all closed-form binomial log-likelihoods. This pins the IRLS, the
/// log-likelihood, the omnibus chi-square and the Nagelkerke effect size against independent
/// arithmetic — far stronger than a self-consistent finite-difference check. It also pins the exact
/// LR decomposition `chi2_uniform + chi2_nonuniform == chi2_total`, which fails if any nested fit
/// lands off its maximum (the `.max(0.0)` clamps would otherwise hide it).
#[test]
fn logistic_dif_saturated_design_closed_form() {
    // (S, G, n, k): a crossing pattern - focal below reference at S=0, above it at S=1.
    let cells = [
        (0.0, 0.0, 100usize, 30usize),
        (1.0, 0.0, 100, 70),
        (0.0, 1.0, 100, 20),
        (1.0, 1.0, 100, 80),
    ];
    let (resp, score, group) = expand(&cells);
    let n = resp.len();
    let st = logistic_item_stats(&resp, &score, &group, n, 100);
    assert!(st.converged, "saturated fit did not converge");

    // closed forms
    let ll2: f64 = cells
        .iter()
        .map(|&(_, _, nn, kk)| bin_ll(kk as f64, nn as f64))
        .sum();
    let ll0 = bin_ll(30.0 + 20.0, 200.0) + bin_ll(70.0 + 80.0, 200.0); // pooled within score level
    let ll_null = bin_ll(200.0, 400.0);
    let chi2_total = 2.0 * (ll2 - ll0);
    assert!(
        (st.chi2_total - chi2_total).abs() < 1e-6,
        "chi2_total {} vs closed form {chi2_total}",
        st.chi2_total
    );
    // Nagelkerke delta R^2 from the same closed forms
    let nn = n as f64;
    let denom = 1.0 - (2.0 * ll_null / nn).exp();
    let r2n = |ll: f64| (1.0 - (2.0 * (ll_null - ll) / nn).exp()) / denom;
    let d_r2 = r2n(ll2) - r2n(ll0);
    assert!(
        (st.delta_r2 - d_r2).abs() < 1e-6,
        "delta_r2 {} vs closed form {d_r2}",
        st.delta_r2
    );
    assert!(st.delta_r2 > 0.0 && st.delta_r2 <= 1.0);
    // exact nesting decomposition (also the monotonicity check at converged MLEs)
    assert!(
        (st.chi2_uniform + st.chi2_nonuniform - st.chi2_total).abs() < 1e-6,
        "decomposition {} + {} != {}",
        st.chi2_uniform,
        st.chi2_nonuniform,
        st.chi2_total
    );
}

/// THE DISCRIMINATING ANCHOR versus Mantel-Haenszel. A crossing (slope-difference) DIF item whose
/// ICCs intersect at the COMMON group ability mean produces essentially no net uniform effect, so
/// the MH common odds ratio is ~1 and MH classifies it NEGLIGIBLE (class A) — the known blind spot
/// of a stratified odds-ratio test. The logistic-regression procedure detects it through the
/// `S x G` interaction: `chi2_nonuniform` is significant while `chi2_uniform` is not. Also checks
/// that a plain uniform (b-shift) item is picked up by the uniform component and not the
/// interaction, and that clean items stay class A. Fixed seed, equal ability distributions.
#[test]
fn logistic_dif_detects_crossing_dif_that_mantel_haenszel_misses() {
    let (n, n_items) = (4000usize, 10usize);
    let cross_item = 4usize;
    let unif_item = 7usize;
    // A pronounced slope difference: strong enough that the TOTAL Nagelkerke effect clears the
    // Jodoin-Gierl moderate cut-off while the uniform-only component stays negligible, which is
    // what separates "classified from delta_r2" from "classified from delta_r2_uniform".
    let a_ref = 2.6f64;
    let a_foc = 0.15f64; // same difficulty, different slope -> ICCs cross at theta = 0
    let mut rng = Lcg(0x2117B0);
    let b: Vec<f64> = (0..n_items).map(|i| -0.9 + 0.2 * i as f64).collect();
    let mut y = vec![0u8; n * n_items];
    let mut group = vec![0u8; n];
    for p in 0..n {
        let g = (p % 2) as u8;
        group[p] = g;
        let theta = rng.normal(); // identical ability distribution in both groups
        for i in 0..n_items {
            let (mut ai, mut bi) = (1.0f64, b[i]);
            if i == cross_item {
                // crossing centered at the common ability mean (b = 0)
                ai = if g == 0 { a_ref } else { a_foc };
                bi = 0.0;
            } else if i == unif_item && g == 1 {
                bi += 0.8; // pure uniform DIF
            }
            let pr = 1.0 / (1.0 + (-(ai * (theta - bi))).exp());
            y[p * n_items + i] = if rng.next_f64() < pr { 1 } else { 0 };
        }
    }
    let lr = logistic_dif(&y, &group, n, n_items, &LogisticDifConfig::default()).unwrap();
    let mh = mantel_haenszel_dif(&y, &group, n, n_items, &MhDifConfig::default()).unwrap();

    // (1) crossing item: logistic flags the INTERACTION, not the group main effect
    let c = &lr[cross_item];
    assert!(c.converged);
    assert!(
        c.p_nonuniform < 0.01,
        "crossing p_nonuniform {}",
        c.p_nonuniform
    );
    assert!(
        c.p_uniform > 0.05,
        "crossing p_uniform should be n.s.: {}",
        c.p_uniform
    );
    assert!(
        c.flagged_bh,
        "crossing item not flagged by the omnibus test"
    );
    // the class must come from the TOTAL delta_r2, not the uniform-only one: a crossing item has a
    // substantial total effect but a near-zero uniform component, so classifying the latter would
    // wrongly report A here.
    assert!(
        c.delta_r2 > c.delta_r2_uniform,
        "total effect {} should exceed the uniform-only {}",
        c.delta_r2,
        c.delta_r2_uniform
    );
    assert_ne!(
        c.jg_class,
        EtsClass::A,
        "crossing item classified from the wrong delta_r2 (total {} vs uniform-only {})",
        c.delta_r2,
        c.delta_r2_uniform
    );
    assert!(
        c.delta_r2_uniform < JG_MODERATE,
        "uniform-only component should stay negligible: {}",
        c.delta_r2_uniform
    );
    // ... and Mantel-Haenszel calls the very same item negligible (its blind spot)
    assert_eq!(
        mh[cross_item].ets_class,
        EtsClass::A,
        "MH unexpectedly flagged the crossing item (delta {})",
        mh[cross_item].mh_d_dif
    );

    // (2) uniform item: the group main effect fires, the interaction does not
    let u = &lr[unif_item];
    assert!(u.p_uniform < 0.01, "uniform p_uniform {}", u.p_uniform);
    assert!(
        u.p_nonuniform > 0.05,
        "uniform p_nonuniform should be n.s.: {}",
        u.p_nonuniform
    );
    assert!(u.flagged_bh);
    // MH does see the uniform item (it is not blind to this kind)
    assert_ne!(mh[unif_item].ets_class, EtsClass::A);

    // (3) clean items: negligible class, and the exact LR decomposition holds everywhere
    for (i, r) in lr.iter().enumerate() {
        assert!(
            (r.chi2_uniform + r.chi2_nonuniform - r.chi2_total).abs() < 1e-6,
            "item {i} decomposition"
        );
        if i != cross_item && i != unif_item {
            assert_eq!(
                r.jg_class,
                EtsClass::A,
                "clean item {i} class {:?}",
                r.jg_class
            );
        }
    }
}

/// Jodoin & Gierl (2001) classification pinned directly at its boundaries. Without this, three
/// distinct mutations survive the simulation tests (whose clean items have `delta_r2 ~ 0` either
/// way): dropping the "not significant => A" rule, swapping the LARGE/MODERATE comparisons, and
/// classifying `delta_r2_uniform` instead of `delta_r2`.
#[test]
fn jg_classify_boundaries() {
    // undefined statistic -> Undefined, never a letter
    assert_eq!(jg_classify(f64::NAN, true), EtsClass::Undefined);
    assert_eq!(jg_classify(f64::NAN, false), EtsClass::Undefined);
    // NOT significant -> A regardless of magnitude (conditional classification)
    assert_eq!(jg_classify(0.50, false), EtsClass::A);
    assert_eq!(jg_classify(JG_LARGE + 0.1, false), EtsClass::A);
    // significant: the two boundaries, inclusive at the cut-points
    assert_eq!(jg_classify(JG_MODERATE - 1e-9, true), EtsClass::A);
    assert_eq!(jg_classify(JG_MODERATE, true), EtsClass::B);
    assert_eq!(jg_classify(JG_LARGE - 1e-9, true), EtsClass::B);
    assert_eq!(jg_classify(JG_LARGE, true), EtsClass::C);
    assert_eq!(jg_classify(0.5, true), EtsClass::C);
    // the ordering itself (a swapped comparison would break this)
    assert_ne!(jg_classify(0.04, true), jg_classify(0.20, true));
}

/// Degenerate items are reported as UNDEFINED, never as a clean non-DIF result: an item everyone
/// answers identically has `ll_null = 0`, which makes the Nagelkerke normalizer zero (a 0/0), and a
/// rank-deficient design cannot be fitted at all.
#[test]
fn logistic_dif_undefined_on_degenerate_item() {
    let (n, n_items) = (200usize, 4usize);
    let mut y = vec![0u8; n * n_items];
    let mut group = vec![0u8; n];
    for p in 0..n {
        group[p] = (p % 2) as u8;
        for i in 0..n_items {
            // item 0 is answered correctly by everyone; the rest vary
            y[p * n_items + i] = if i == 0 { 1 } else { ((p / (i + 1)) % 2) as u8 };
        }
    }
    let rows = logistic_dif(&y, &group, n, n_items, &LogisticDifConfig::default()).unwrap();
    let r0 = &rows[0];
    assert!(
        !r0.converged,
        "constant item should not report a converged fit"
    );
    assert!(r0.chi2_total.is_nan() && r0.delta_r2.is_nan());
    // The p-values must be NaN too, NOT 1.0: chi2_sf maps a NaN statistic to 1.0 (f64::max ignores
    // NaN), which would read as "definitively no DIF" and, being finite, would make
    // Benjamini-Hochberg count this unfittable item in `m` and dilute every other item's threshold.
    assert!(
        r0.p_total.is_nan() && r0.p_uniform.is_nan() && r0.p_nonuniform.is_nan(),
        "failed fit reported p_total {} (expected NaN)",
        r0.p_total
    );
    assert_eq!(r0.jg_class, EtsClass::Undefined);
    assert!(!r0.flagged_bh, "an undefined item must never be BH-flagged");
    // validation is shared with the MH path
    let cfg_bad = LogisticDifConfig {
        fdr_q: 0.0,
        ..LogisticDifConfig::default()
    };
    assert!(logistic_dif(&y, &group, n, n_items, &cfg_bad).is_err());
    let cfg_it = LogisticDifConfig {
        max_iter: 0,
        ..LogisticDifConfig::default()
    };
    assert!(logistic_dif(&y, &group, n, n_items, &cfg_it).is_err());
    assert!(logistic_dif(&y, &vec![0u8; n], n, n_items, &LogisticDifConfig::default()).is_err());
}

#[test]
fn logistic_private_failures_and_rest_score_path_are_explicit() {
    assert!(logit_fit(&[f64::NAN], &[1.0], 1, 1, &[1.0], 1).is_none());
    assert!(logit_fit(&[1.0, 1.0], &[0.0, 1.0], 2, 1, &[0.0], 0).is_none());

    let x = vec![1.0; 20];
    let y: Vec<f64> = (0..20).map(|index| (index % 2) as f64).collect();
    let bounded = logit_fit(&x, &y, 20, 1, &[29.9], 1);
    assert!(bounded.is_none() || bounded.unwrap().0[0].abs() <= LOGIT_COEF_BOUND);
    assert!(logit_fit(&[0.1; 20], &[1.0; 20], 20, 1, &[31.0], 1).is_none());

    assert!(!logistic_item_stats(&[0.0; 19], &[0.0; 19], &[0.0; 19], 19, 50).converged);
    let response: Vec<f64> = (0..20).map(|index| (index % 2) as f64).collect();
    assert!(!logistic_item_stats(&response, &[0.0; 20], &[0.0; 20], 20, 50).converged);
    let score = response.clone();
    let group: Vec<f64> = (0..20).map(|index| ((index / 2) % 2) as f64).collect();
    assert!(!logistic_item_stats(&response, &score, &group, 20, 50).converged);
    assert!(
        !logistic_item_stats(
            &response,
            &(0..20).map(|v| v as f64).collect::<Vec<_>>(),
            &group,
            20,
            0
        )
        .converged
    );

    let n = 40;
    let n_items = 2;
    let responses: Vec<u8> = (0..n * n_items)
        .map(|index| ((index / n_items + index % n_items) % 2) as u8)
        .collect();
    let groups: Vec<u8> = (0..n).map(|index| (index % 2) as u8).collect();
    let rows = logistic_dif(
        &responses,
        &groups,
        n,
        n_items,
        &LogisticDifConfig {
            exclude_studied_item: true,
            ..LogisticDifConfig::default()
        },
    )
    .unwrap();
    assert_eq!(rows.len(), n_items);

    let score: Vec<f64> = (0..40).map(|index| (index % 2) as f64).collect();
    let group: Vec<f64> = (0..40).map(|index| ((index / 2) % 2) as f64).collect();
    let group_separated = group.clone();
    assert!(!logistic_item_stats(&group_separated, &score, &group, 40, 50).converged);

    let interaction_separated: Vec<f64> = score
        .iter()
        .zip(&group)
        .map(|(score, group)| if score == group { 1.0 } else { 0.0 })
        .collect();
    assert!(!logistic_item_stats(&interaction_separated, &score, &group, 40, 50).converged);
}

// ---------------- iterative item purification ----------------

/// Build a seeded bank whose `dif_items` are shifted UNIDIRECTIONALLY against the focal group.
/// The direction matters: bidirectional shifts cancel in the number-correct total and produce no
/// criterion contamination at all, which would make the whole fixture vacuous.
fn purification_bank(
    n: usize,
    n_items: usize,
    dif_items: &[usize],
    shift: f64,
    seed: u64,
) -> (Vec<u8>, Vec<u8>) {
    let mut rng = Lcg(seed);
    let b: Vec<f64> = (0..n_items).map(|i| -0.9 + 0.16 * i as f64).collect();
    let mut y = vec![0u8; n * n_items];
    let mut group = vec![0u8; n];
    for p in 0..n {
        let g = (p % 2) as u8;
        group[p] = g;
        let theta = rng.normal(); // identical ability distributions in both groups
        for i in 0..n_items {
            let mut bi = b[i];
            if g == 1 && dif_items.contains(&i) {
                bi += shift; // every planted item is harder for the SAME group
            }
            let pr = 1.0 / (1.0 + (-(1.2 * (theta - bi))).exp());
            y[p * n_items + i] = if rng.next_f64() < pr { 1 } else { 0 };
        }
    }
    (y, group)
}

/// SEEDED REGRESSION FIXTURE (not a general property of purification): with several items shifted
/// against the focal group, the unpurified number-correct criterion is depressed for that group, so
/// CLEAN items pick up spurious DIF. Rebuilding the criterion from the unflagged anchor reduces
/// those false flags while the planted items stay flagged. The precondition is asserted first —
/// without it the test would pass trivially on a simulation that produced no contamination.
#[test]
fn purification_reduces_criterion_contamination_false_flags() {
    let (n, n_items) = (3000usize, 12usize);
    let dif_items = [2usize, 5, 8];
    let (y, group) = purification_bank(n, n_items, &dif_items, 1.2, 0x9F1E2);
    let cfg = MhDifConfig::default();
    let plain = mantel_haenszel_dif(&y, &group, n, n_items, &cfg).unwrap();
    let pur = mantel_haenszel_dif_purified(&y, &group, n, n_items, &cfg, &PurifyConfig::default())
        .unwrap();

    let clean: Vec<usize> = (0..n_items).filter(|i| !dif_items.contains(i)).collect();
    let false_before: Vec<usize> = clean
        .iter()
        .copied()
        .filter(|&j| plain[j].ets_class != EtsClass::A)
        .collect();
    // PRECONDITION: the fixture must actually exhibit contamination, else nothing is being tested.
    assert!(
        !false_before.is_empty(),
        "fixture precondition failed: no clean item false-flagged. classes={:?} deltas={:?}",
        plain.iter().map(|r| r.ets_class).collect::<Vec<_>>(),
        plain
            .iter()
            .map(|r| (r.mh_d_dif * 100.0).round() / 100.0)
            .collect::<Vec<_>>()
    );
    let false_after: Vec<usize> = clean
        .iter()
        .copied()
        .filter(|&j| pur.rows[j].ets_class != EtsClass::A)
        .collect();
    assert!(
        false_after.len() < false_before.len(),
        "purification did not reduce false flags: {false_before:?} -> {false_after:?}; \
         rounds={} n_anchor={} classes={:?} deltas={:?}",
        pur.rounds,
        pur.n_anchor,
        plain.iter().map(|r| r.ets_class).collect::<Vec<_>>(),
        plain
            .iter()
            .map(|r| (r.mh_d_dif * 100.0).round() / 100.0)
            .collect::<Vec<_>>()
    );
    // TRUE POSITIVES retained - otherwise "removes false flags" is satisfiable by flagging nothing.
    for &d in &dif_items {
        assert!(
            matches!(pur.rows[d].ets_class, EtsClass::B | EtsClass::C),
            "planted item {d} lost after purification: {:?}",
            pur.rows[d].ets_class
        );
        assert!(!pur.anchor[d], "planted item {d} left in the anchor");
    }
    // The criterion genuinely changed: at least one clean item's statistic moved. Without this an
    // implementation that simply returned the unpurified rows would pass everything above.
    assert!(
        clean
            .iter()
            .any(|&j| (pur.rows[j].chi2_mh - plain[j].chi2_mh).abs() > 1e-6),
        "purified statistics identical to the unpurified sweep"
    );
    assert!(pur.rounds >= 1 && pur.n_anchor == n_items - dif_items.len());
}

/// A clean bank needs no purification: nothing is flagged, so the anchor stays the whole test, no
/// rounds run, and round 0 reproduces the shipped unpurified sweep EXACTLY (pinning the refactor's
/// no-op path - `anchor = None` and an all-true anchor must agree bit for bit).
#[test]
fn purification_is_a_no_op_on_a_clean_bank() {
    let (n, n_items) = (2000usize, 10usize);
    let (y, group) = purification_bank(n, n_items, &[], 0.0, 0x5AFE);
    let cfg = MhDifConfig::default();
    let plain = mantel_haenszel_dif(&y, &group, n, n_items, &cfg).unwrap();
    let pur = mantel_haenszel_dif_purified(&y, &group, n, n_items, &cfg, &PurifyConfig::default())
        .unwrap();
    assert!(pur.converged && pur.rounds == 0);
    assert!(pur.anchor.iter().all(|&a| a) && pur.n_anchor == n_items);
    for i in 0..n_items {
        assert_eq!(
            pur.rows[i].chi2_mh, plain[i].chi2_mh,
            "round 0 must equal the shipped sweep exactly at item {i}"
        );
        assert_eq!(pur.rows[i].alpha_mh, plain[i].alpha_mh);
        assert_eq!(pur.rows[i].ets_class, plain[i].ets_class);
    }
}

/// The round cap is observable: with `max_rounds = 1` on a bank that is still changing, the loop
/// stops after one purification round and reports `converged = false` with the round-1 rows.
#[test]
fn purification_round_cap_reports_non_convergence() {
    let (n, n_items) = (3000usize, 12usize);
    let dif_items = [2usize, 5, 8];
    let (y, group) = purification_bank(n, n_items, &dif_items, 1.2, 0x9F1E2);
    let cfg = MhDifConfig::default();
    let capped = mantel_haenszel_dif_purified(
        &y,
        &group,
        n,
        n_items,
        &cfg,
        &PurifyConfig {
            max_rounds: 1,
            ..PurifyConfig::default()
        },
    )
    .unwrap();
    assert_eq!(
        capped.rounds, 1,
        "rounds={} converged={} n_anchor={} anchor={:?}",
        capped.rounds, capped.converged, capped.n_anchor, capped.anchor
    );
    assert!(
        !capped.converged,
        "hitting the round cap must report converged = false"
    );
    assert_eq!(capped.termination_reason, "max_rounds_reached");
    // max_rounds = 0 is rejected rather than silently meaning "no purification"
    assert!(mantel_haenszel_dif_purified(
        &y,
        &group,
        n,
        n_items,
        &cfg,
        &PurifyConfig {
            max_rounds: 0,
            ..PurifyConfig::default()
        }
    )
    .is_err());
}

/// The anchor guard fires BEFORE sweeping on a uselessly short criterion: with `min_anchor_items`
/// set above what the flagged set leaves, purification stops and returns the last usable rows with
/// `converged = false` rather than matching on a near-empty anchor. Also exercises the logistic
/// variant of the purified entry point.
#[test]
fn purification_stops_on_a_too_short_anchor() {
    let (n, n_items) = (3000usize, 12usize);
    let dif_items = [2usize, 5, 8];
    let (y, group) = purification_bank(n, n_items, &dif_items, 1.2, 0x9F1E2);
    let strict = PurifyConfig {
        max_rounds: 3,
        min_anchor_items: n_items,
    };
    let pur =
        mantel_haenszel_dif_purified(&y, &group, n, n_items, &MhDifConfig::default(), &strict)
            .unwrap();
    // the guard tripped immediately: no round ran, the anchor is still the full test
    assert_eq!(pur.rounds, 0);
    assert!(!pur.converged && pur.n_anchor == n_items);
    assert_eq!(pur.termination_reason, "insufficient_anchor_items");
    // the logistic purified entry point runs and removes the planted items from its anchor
    let lp = logistic_dif_purified(
        &y,
        &group,
        n,
        n_items,
        &LogisticDifConfig::default(),
        &PurifyConfig::default(),
    )
    .unwrap();
    assert!(lp.n_anchor <= n_items);
    for &d in &dif_items {
        assert!(
            !lp.anchor[d],
            "logistic purification left planted item {d} in the anchor"
        );
    }
}

/// STRUCTURAL ANCHOR for the returned rows: `rows` must be the sweep against the REPORTED `anchor`,
/// on every exit path. Returning an earlier round's rows while reporting the final anchor is the
/// highest-severity failure mode of a purification loop and is invisible to a "did it flag the right
/// items" test, because the intermediate rounds usually flag the same items. Swept over round caps,
/// anchor floors and BOTH matching conventions so the `exclude_studied_item = true` branch of
/// [`matching_for_item`] — untested by the fixtures above — is covered here.
#[test]
fn purified_rows_are_the_sweep_against_the_reported_anchor() {
    let (n, n_items) = (1200usize, 12usize);
    let (y, group) = purification_bank(n, n_items, &[2, 5, 8], 1.2, 0x51A7);
    let mut seen_purified_round = false;
    for exclude_studied_item in [false, true] {
        let cfg = MhDifConfig {
            exclude_studied_item,
            ..MhDifConfig::default()
        };
        for max_rounds in [1usize, 2, 5] {
            for min_anchor_items in [1usize, 4, 9] {
                let purify = PurifyConfig {
                    max_rounds,
                    min_anchor_items,
                };
                let res =
                    mantel_haenszel_dif_purified(&y, &group, n, n_items, &cfg, &purify).unwrap();
                seen_purified_round |= res.rounds > 0;
                // A fresh sweep against the reported anchor must reproduce the reported rows.
                let refr = mh_sweep(&y, &group, n, n_items, &cfg, Some(&res.anchor)).unwrap();
                for i in 0..n_items {
                    assert_eq!(
                        res.rows[i].chi2_mh, refr[i].chi2_mh,
                        "item {i}: rows do not match the reported anchor \
                         (exclude={exclude_studied_item} max_rounds={max_rounds} \
                          min_anchor={min_anchor_items} rounds={} n_anchor={})",
                        res.rounds, res.n_anchor
                    );
                    assert_eq!(res.rows[i].mh_d_dif, refr[i].mh_d_dif, "item {i} d-DIF");
                }
                assert_eq!(res.n_anchor, res.anchor.iter().filter(|&&a| a).count());
            }
        }
    }
    // Guard the guard: if no configuration ever purified, the assertions above are vacuous.
    assert!(
        seen_purified_round,
        "no configuration performed a purification round"
    );
}

/// VALUE ANCHOR for the criterion itself, against an independent reference rather than against the
/// implementation's own arithmetic. Purification matches every item on `anchor UNION {studied}`, so
/// the purified row for item `i` must equal the ORDINARY unpurified sweep run on a test consisting of
/// exactly those columns. Checked for both a non-anchor item (the add-back branch) and an anchor item
/// (no add-back), with a deliberately NON-CONTIGUOUS anchor so an index-map or layout error cannot
/// hide behind a prefix. This is what fails if the add-back is dropped, doubled, or applied to the
/// wrong branch — none of which the flag-counting fixtures can see.
#[test]
fn purified_item_is_matched_on_the_anchor_union_itself() {
    let (n, n_items) = (1500usize, 10usize);
    let (y, group) = purification_bank(n, n_items, &[3], 1.0, 0x2C4B);
    // scattered anchor: items 1, 4, 6, 9 are OUT
    let anchor: Vec<bool> = (0..n_items).map(|i| !matches!(i, 1 | 4 | 6 | 9)).collect();
    for exclude_studied_item in [false, true] {
        let cfg = MhDifConfig {
            exclude_studied_item,
            ..MhDifConfig::default()
        };
        let swept = mh_sweep(&y, &group, n, n_items, &cfg, Some(&anchor)).unwrap();
        for studied in [4usize, 5] {
            // columns of the reference test: anchor UNION {studied}, original order preserved
            let cols: Vec<usize> = (0..n_items)
                .filter(|&j| anchor[j] || j == studied)
                .collect();
            let pos = cols.iter().position(|&j| j == studied).unwrap();
            let mut reduced = vec![0u8; n * cols.len()];
            for p in 0..n {
                for (c, &j) in cols.iter().enumerate() {
                    reduced[p * cols.len() + c] = y[p * n_items + j];
                }
            }
            let refr = mantel_haenszel_dif(&reduced, &group, n, cols.len(), &cfg).unwrap();
            let (a, b) = (&swept[studied], &refr[pos]);
            assert_eq!(
                a.chi2_mh, b.chi2_mh,
                "item {studied} (in_anchor={}, exclude={exclude_studied_item}) is not matched on \
                 anchor UNION itself",
                anchor[studied]
            );
            assert_eq!(a.alpha_mh, b.alpha_mh, "item {studied} alpha_MH");
            assert_eq!(a.mh_d_dif, b.mh_d_dif, "item {studied} ETS delta");
            assert_eq!(a.std_p_dif, b.std_p_dif, "item {studied} STD P-DIF");
            assert_eq!(a.ets_class, b.ets_class, "item {studied} ETS class");
        }
    }
}

/// The anchor rule is PRACTICAL significance, not `class != A`. `Undefined` is also `!= A`, so the
/// lazier predicate would purge unfittable items — which carry no evidence of DIF — and shrink the
/// anchor for free. No simulated bank distinguishes the two (a clean 2PL never produces `Undefined`),
/// so the predicate is pinned directly.
#[test]
fn purify_flagged_is_practical_significance_not_just_non_a() {
    assert!(!purify_flagged(EtsClass::A));
    assert!(purify_flagged(EtsClass::B));
    assert!(purify_flagged(EtsClass::C));
    assert!(
        !purify_flagged(EtsClass::Undefined),
        "an unfittable item carries no evidence of DIF and must stay in the anchor"
    );
}

// ---------------- SIBTEST (uniform) ----------------

/// Build one item's per-level cells from `(level, J_R, sum_y_R, J_F, sum_y_F)`. Responses are 0/1, so
/// the raw second moment equals the first -- the core still computes its own variance from these raw
/// moments, which keeps the biased-vs-unbiased mutation live.
fn sib_cells(rows: &[(usize, u64, u64, u64, u64)]) -> Vec<SibCell> {
    rows.iter()
        .map(|&(level, j_r, sy_r, j_f, sy_f)| SibCell {
            level,
            j_r,
            sum_y_r: sy_r as f64,
            sum_y2_r: sy_r as f64,
            j_f,
            sum_y_f: sy_f as f64,
            sum_y2_f: sy_f as f64,
        })
        .collect()
}

/// CLOSED-FORM ACCEPTANCE ANCHOR. Every constant was derived from the estimator in exact rational
/// arithmetic and re-derived independently before this test was written; nothing here is a
/// record-what-it-printed baseline. With `Xbar_R = 2.4` and `Xbar_F = 1.8`, the level-2 slopes come out
/// `M_R = 2.0` and `M_F = 8.0` -- the focal group's smaller alpha compresses its true-score scale, so
/// the same rise in `Ybar` across the same observed span implies a four times steeper slope -- and the
/// corrected means are `0.44` and `0.54`, giving `beta = -1/10` and `sigma^2 = 23/1950`.
///
/// kills: correction deleted [+0.20]; correction sign flipped [+0.50]; subtracting the OBSERVED mean
/// instead of `V*_Gk` [+0.26]; the midpoint replaced by each group's own `V*` [+0.20]; Kelley written
/// as `(1 - alpha)` [-0.25]; a pooled alpha in the `M` denominator [+0.008]; biased cell variance
/// [se = 0.107238053]; the slope taken over retained levels only [NaN].
#[test]
fn sibtest_closed_form_single_stratum_anchor() {
    let cells = sib_cells(&[(1, 10, 1, 40, 2), (2, 40, 20, 40, 12), (3, 50, 45, 20, 17)]);
    let st = sibtest_stats(&cells, 0.8, 0.2, 4, 5);
    assert_eq!(st.n_strata_used, 1, "only level 2 is interior and well-populated");
    assert!((st.beta_uni - (-0.10)).abs() < 1e-12, "beta_uni = {}", st.beta_uni);
    assert!((st.se_beta - 0.1086041978694737).abs() < 1e-12, "se = {}", st.se_beta);
    assert!((st.b_uni - (-0.920774721067277)).abs() < 1e-12, "b_uni = {}", st.b_uni);
    assert!((st.b_uni * st.b_uni - 39.0 / 46.0).abs() < 1e-12, "X2 must be 39/46");
    assert!((st.p_value - 0.35716805550697844).abs() < 1e-12, "p = {}", st.p_value);
}

/// MULTI-LEVEL WEIGHTING ANCHOR. The single-stratum anchor above is structurally blind to every
/// weighting question, because one retained level always carries weight 1. Here level 0 fails `j_min`
/// and level 4 is the maximum, so the retained weights are `(1/4, 2/5, 7/20)` and must sum to exactly 1
/// after renormalization.
///
/// The UNCORRECTED beta is `+0.11` under BOTH weighting schemes, so the same assertion made on an
/// uncorrected statistic would prove nothing; it is asserted on the corrected value.
///
/// kills: focal-group weights instead of combined-sample [-0.039932]; weights left unrenormalized; the
/// `j_min` gate dropped; the min/max exclusion dropped; renormalizing `beta` but not `se` (this pins
/// `X^2`, which is invariant to renormalization only when BOTH are scaled together).
#[test]
fn sibtest_multi_level_weighting_anchor() {
    let cells = sib_cells(&[
        (0, 4, 0, 3, 0),
        (1, 10, 1, 40, 2),
        (2, 40, 20, 40, 12),
        (3, 50, 45, 20, 17),
        (4, 6, 6, 7, 7),
    ]);
    let st = sibtest_stats(&cells, 0.8, 0.2, 4, 5);
    assert_eq!(st.n_strata_used, 3);
    assert!(
        (st.beta_uni - (-16993.0 / 88000.0)).abs() < 1e-12,
        "beta_uni = {} (expected -16993/88000)",
        st.beta_uni
    );
    assert!((st.se_beta - 0.06029378704091735).abs() < 1e-12, "se = {}", st.se_beta);
    assert!(
        (st.b_uni * st.b_uni - 10.257219401952296).abs() < 1e-9,
        "X2 = {}",
        st.b_uni * st.b_uni
    );
}

/// NON-CONTIGUOUS LEVELS. The central difference spans the adjacent OBSERVED positions, whose levels
/// here differ by 4 rather than 2 because levels 2 and 3 have no examinees at all. A slope built from a
/// hardcoded `2 * alpha / n_valid` denominator is exactly twice as steep, which is asserted directly.
///
/// kills: a hardcoded `2 * alpha / n_valid` denominator; arithmetic `k +/- 1` indexing instead of
/// positional; a dense `0..=n_valid` level vector, which would fabricate the empty interior levels.
#[test]
fn sibtest_uses_observed_level_spacing_not_arithmetic_neighbours() {
    let cells = sib_cells(&[
        (0, 20, 2, 20, 3),
        (1, 30, 9, 30, 12),
        (4, 30, 24, 30, 21),
        (5, 20, 18, 20, 17),
    ]);
    let st = sibtest_stats(&cells, 0.75, 0.5, 6, 5);
    assert_eq!(st.n_strata_used, 2, "levels 1 and 4 are the interior positions");
    // Pinned to the IMPLEMENTATION's output, in exact rational arithmetic: beta = 1/128. Both mutants
    // named above return 1/64 -- exactly double, because they divide the slope by a span of 2 where
    // the observed span is 4. An assertion computed from test-local arithmetic instead of from `st`
    // would be an identity about `f64` and would pass for ANY implementation; this one cannot.
    assert!(
        (st.beta_uni - 1.0 / 128.0).abs() < 1e-12,
        "beta_uni = {} (expected 1/128; the contiguous-span mutants give 1/64)",
        st.beta_uni
    );
}

/// STRICT-INEQUALITY BOUNDARY on `j_min`, tested on BOTH sides of the conjunction: level 1 sits at
/// exactly `j_min` in the REFERENCE group, level 2 at exactly `j_min` in the FOCAL group, and level 3
/// at `j_min + 1` in both. Only level 3 may survive. A one-sided fixture would let the untested half of
/// the gate be weakened or deleted outright.
///
/// kills: `>=` written where `>` is meant, in EITHER group's count gate; the focal-group conjunct
/// dropped entirely.
#[test]
fn sibtest_j_min_is_strictly_exceeded_in_both_groups() {
    let cells = sib_cells(&[
        (0, 20, 2, 20, 3),
        (1, 5, 2, 20, 6),
        (2, 20, 8, 5, 2),
        (3, 6, 4, 6, 3),
        (4, 20, 18, 20, 17),
    ]);
    let st = sibtest_stats(&cells, 0.8, 0.8, 5, 5);
    assert_eq!(
        st.n_strata_used, 1,
        "levels 1 (J_R == j_min) and 2 (J_F == j_min) must be excluded; only level 3 (j_min + 1 in \
         both) may survive"
    );
}

/// ASYMMETRIC NEIGHBOUR GUARD, and a documented DIVERGENCE from the reference implementation: it
/// imputes an absent group-by-level cell's mean to 0.0 and feeds that fabricated zero into the
/// neighbouring central difference, producing a finite but meaningless slope. This implementation drops
/// the level. The divergence is asserted so it cannot silently drift back.
///
/// kills: inheriting the NaN-to-zero imputation, which would retain the level and report a number.
#[test]
fn sibtest_drops_levels_whose_neighbour_lacks_a_group() {
    let full = sib_cells(&[
        (0, 20, 2, 20, 3),
        (1, 30, 9, 30, 12),
        (2, 30, 24, 30, 21),
        (3, 20, 18, 20, 17),
    ]);
    assert_eq!(sibtest_stats(&full, 0.8, 0.8, 4, 5).n_strata_used, 2);

    // The guard is a four-way conjunction (lower/upper x reference/focal). Testing one corner would
    // leave the other three deletable, so every corner is holed in turn.
    let base = [(0usize, 20u64, 2u64, 20u64, 3u64), (1, 30, 9, 30, 12), (2, 30, 24, 30, 21), (3, 20, 18, 20, 17)];
    for (pos, which, dropped) in [
        (0usize, "reference", 1usize),
        (0, "focal", 1),
        (3, "reference", 2),
        (3, "focal", 2),
    ] {
        let mut rows = base;
        if which == "reference" {
            rows[pos].1 = 0;
            rows[pos].2 = 0;
        } else {
            rows[pos].3 = 0;
            rows[pos].4 = 0;
        }
        let st = sibtest_stats(&sib_cells(&rows), 0.8, 0.8, 4, 5);
        assert_eq!(
            st.n_strata_used, 1,
            "emptying the {which} group at position {pos} must drop interior level {dropped}"
        );
    }
}

/// ALPHA GATE. The correction divides by the reliability through the local slope, so a non-positive or
/// non-finite alpha makes the statistic meaningless and must yield a NaN row rather than a clamp.
///
/// kills: a gate written only against NaN, which misses `alpha <= 0`; `alpha != 0` in place of
/// `alpha > 0`.
#[test]
fn sibtest_rejects_degenerate_reliability() {
    let cells = sib_cells(&[
        (0, 20, 2, 20, 3),
        (1, 30, 9, 30, 12),
        (2, 30, 24, 30, 21),
        (3, 20, 18, 20, 17),
    ]);
    for (ar, af, why) in [
        (-0.3, 0.8, "negative reference alpha"),
        (0.8, -0.3, "negative focal alpha"),
        (0.0, 0.8, "zero alpha is degenerate, not merely small"),
        (f64::NAN, 0.8, "non-finite alpha"),
    ] {
        let st = sibtest_stats(&cells, ar, af, 4, 5);
        assert!(st.beta_uni.is_nan(), "{why}: beta must be NaN");
        assert!(st.p_value.is_nan(), "{why}: p must be NaN, never 1.0");
        assert_eq!(st.n_strata_used, 0, "{why}");
    }
}

/// The undefined contract is NaN, never a zero-initialized accumulator and never `p = 1.0`. A `0.0`
/// beta reads as an affirmative "no DIF" claim, and a FINITE `p = 1.0` would be counted by
/// Benjamini-Hochberg, shrinking every other item's threshold.
///
/// kills: `return 0.0` on the degenerate path -- an `.abs() < eps` assertion would pass that mutant, so
/// `is_nan` is asserted explicitly; a dropped `is_finite` guard before squaring `b_uni`.
#[test]
fn sibtest_undefined_is_nan_not_zero_or_one() {
    let st = sibtest_stats(&sib_cells(&[(0, 30, 3, 30, 4), (1, 30, 20, 30, 18)]), 0.8, 0.8, 4, 5);
    assert!(st.beta_uni.is_nan() && st.se_beta.is_nan() && st.b_uni.is_nan());
    assert!(st.p_value.is_nan(), "must not collapse to 1.0");
    assert_eq!(st.n_strata_used, 0);
}

/// Seeded 2PL bank with UNEQUAL group sizes and non-mirrored difficulties. Deliberately not 50/50 and
/// deliberately not symmetric: a balanced, mirrored fixture cancels sign errors.
fn sibtest_bank(
    n_ref: usize,
    n_focal: usize,
    n_items: usize,
    dif_items: &[usize],
    shift: f64,
    impact: f64,
    seed: u64,
) -> (Vec<u8>, Vec<u8>) {
    let mut rng = Lcg(seed);
    let n = n_ref + n_focal;
    let b: Vec<f64> = (0..n_items).map(|i| -1.1 + 0.23 * i as f64).collect();
    let mut y = vec![0u8; n * n_items];
    let mut group = vec![0u8; n];
    for p in 0..n {
        let g = if p < n_ref { 0u8 } else { 1u8 };
        group[p] = g;
        // `impact` is a genuine ability difference, not DIF: it shifts the focal ability distribution.
        let theta = rng.normal() - if g == 1 { impact } else { 0.0 };
        for i in 0..n_items {
            let mut bi = b[i];
            if g == 1 && dif_items.contains(&i) {
                bi += shift; // harder for the focal group
            }
            let pr = 1.0 / (1.0 + (-(1.3 * (theta - bi))).exp());
            y[p * n_items + i] = u8::from(rng.next_f64() < pr);
        }
    }
    (y, group)
}

/// STUDIED-RESPONSE LEAK. SIBTEST's valid subtest and studied subtest are DISJOINT by construction, so
/// item `i`'s own response must not enter item `i`'s matching criterion at all.
///
/// The exact invariant is NOT that item `i`'s row is unchanged -- flipping `y_i` to `1 - y_i` sends
/// every conditional mean `Ybar_Gk` to `1 - Ybar_Gk` and every slope `M_G` to `-M_G`, so the
/// transported mean becomes `1 - Ybar*_Gk` and the DIFFERENCE, hence `beta_uni`, is exactly NEGATED
/// while `se_beta` is invariant (a Bernoulli variance is symmetric under complement). That is what is
/// asserted, and it is strictly stronger than an integer stratum-count comparison: it pins the whole
/// real-valued statistic rather than a coarse proxy. If the studied item leaked into its own criterion
/// the strata themselves would move and neither identity would hold.
///
/// The second block -- some OTHER item must react -- is what stops a constant or saturated criterion
/// from passing vacuously, since a criterion that ignores the data is also flip-invariant.
///
/// kills: the item-included Mantel-Haenszel convention reused by mistake; the studied item added back
/// into its own valid subtest; a criterion that ignores the responses entirely.
#[test]
fn sibtest_criterion_excludes_the_studied_item() {
    let (y, group) = sibtest_bank(400, 250, 7, &[3], 0.9, 0.0, 0x5B1);
    let cfg = SibtestConfig::default();
    let base_rows = sibtest(&y, &group, 650, 7, &cfg).unwrap();
    for i in 0..7 {
        let mut flipped = y.clone();
        for p in 0..650 {
            flipped[p * 7 + i] = 1 - flipped[p * 7 + i];
        }
        let rows = sibtest(&flipped, &group, 650, 7, &cfg).unwrap();
        assert_eq!(
            rows[i].n_strata_used, base_rows[i].n_strata_used,
            "item {i}: flipping its own column changed its own stratification"
        );
        assert!(
            (rows[i].beta_uni + base_rows[i].beta_uni).abs() < 1e-12,
            "item {i}: complementing the studied response must NEGATE beta_uni exactly ({} vs {})",
            base_rows[i].beta_uni,
            rows[i].beta_uni
        );
        assert!(
            (rows[i].se_beta - base_rows[i].se_beta).abs() < 1e-12,
            "item {i}: se_beta must be invariant under complementing the studied response"
        );
        // ...but at least one OTHER item must react, or the criterion is not reading the data at all
        assert!(
            (0..7).any(|j| j != i && rows[j].beta_uni != base_rows[j].beta_uni),
            "item {i}: flipping it changed no other item, so the criterion is degenerate"
        );
    }
}

/// CROSS-MODULE SIGN ANCHOR, in both directions. `beta_uni` is reference-minus-focal while `mh_d_dif`
/// and `std_p_dif` are focal-oriented, so on an item that is harder for the focal group SIBTEST goes
/// POSITIVE where both Mantel-Haenszel statistics go NEGATIVE. Swapping the group labels must flip all
/// of them, and the product assertion is what would catch a future "harmonisation" that flips both
/// modules at once and so preserves every single-module assertion.
///
/// kills: a sign flip in either module; an `abs()` collapse (caught by the mirror block); a
/// simultaneous flip of both conventions (caught by the product).
#[test]
fn sibtest_sign_is_opposite_to_mantel_haenszel() {
    let (y, group) = sibtest_bank(420, 300, 8, &[4], 1.1, 0.0, 0x7C3);
    let sib = sibtest(&y, &group, 720, 8, &SibtestConfig::default()).unwrap();
    let mh = mantel_haenszel_dif(&y, &group, 720, 8, &MhDifConfig::default()).unwrap();
    assert!(sib[4].beta_uni > 0.0, "beta_uni = {} (must be > 0)", sib[4].beta_uni);
    assert!(mh[4].mh_d_dif < 0.0, "mh_d_dif = {}", mh[4].mh_d_dif);
    assert!(mh[4].std_p_dif < 0.0, "std_p_dif = {}", mh[4].std_p_dif);
    assert!(
        sib[4].beta_uni * mh[4].std_p_dif < 0.0,
        "the two modules must keep OPPOSITE orientations"
    );

    // Mirror: swapping the labels flips the sign and (to numerical noise) the magnitude is preserved.
    let swapped: Vec<u8> = group.iter().map(|&g| 1 - g).collect();
    let sib_sw = sibtest(&y, &swapped, 720, 8, &SibtestConfig::default()).unwrap();
    assert!(sib_sw[4].beta_uni < 0.0, "swapping labels must flip beta_uni");
    assert!(
        (sib_sw[4].beta_uni + sib[4].beta_uni).abs() < 1e-9,
        "beta_uni must be antisymmetric in the group labels: {} vs {}",
        sib[4].beta_uni,
        sib_sw[4].beta_uni
    );
}

/// COEFFICIENT ALPHA is computed PER GROUP and on the VALID subtest only (which is a different item set
/// for every studied item). Pinned against the direct KR-20 definition recomputed here from the raw
/// matrix, on a fixture whose groups have deliberately different reliabilities.
///
/// kills: a single POOLED alpha, which survives every fixture whose groups happen to be equally
/// reliable; `n_items` substituted for `n_valid` in the `k/(k-1)` factor; alpha computed over all items
/// including the studied one.
#[test]
fn sibtest_alpha_is_per_group_on_the_valid_subtest() {
    // the focal group is given a much larger ability spread, so its alpha differs materially
    let (y, group) = sibtest_bank(500, 400, 6, &[], 0.0, 1.4, 0x2E9);
    let (n, n_items) = (900usize, 6usize);
    let rows = sibtest(&y, &group, n, n_items, &SibtestConfig::default()).unwrap();

    for studied in [0usize, 3, 5] {
        for (g, reported) in [(0u8, rows[studied].alpha_ref), (1u8, rows[studied].alpha_focal)] {
            let idx: Vec<usize> = (0..n).filter(|&p| group[p] == g).collect();
            let ng = idx.len() as f64;
            let valid: Vec<usize> = (0..n_items).filter(|&j| j != studied).collect();
            let mut item_var_sum = 0.0;
            for &j in &valid {
                let m = idx.iter().map(|&p| y[p * n_items + j] as f64).sum::<f64>() / ng;
                item_var_sum += m - m * m; // 0/1 item: E[y^2] = E[y]
            }
            let totals: Vec<f64> = idx
                .iter()
                .map(|&p| valid.iter().map(|&j| y[p * n_items + j] as f64).sum::<f64>())
                .collect();
            let tm = totals.iter().sum::<f64>() / ng;
            let tv = totals.iter().map(|t| (t - tm) * (t - tm)).sum::<f64>() / ng;
            let k = valid.len() as f64;
            let expected = (k / (k - 1.0)) * (1.0 - item_var_sum / tv);
            assert!(
                (reported - expected).abs() < 1e-12,
                "item {studied} group {g}: alpha {reported} != direct KR-20 {expected}"
            );
        }
    }
    assert!(
        (rows[0].alpha_ref - rows[0].alpha_focal).abs() > 0.05,
        "fixture precondition: the two groups must differ in reliability, else a pooled alpha would \
         pass this test ({} vs {})",
        rows[0].alpha_ref,
        rows[0].alpha_focal
    );
}

/// A degenerate item must produce a NaN row that is NOT Benjamini-Hochberg flagged and, critically, is
/// NOT counted in BH's `m`: a finite `p = 1.0` would silently shrink every other item's threshold.
///
/// kills: zero-initialized accumulators reaching the row; an undefined row entering the flag path; a
/// NaN p-value collapsing to 1.0 through `chi2_sf`.
#[test]
fn sibtest_degenerate_item_is_nan_and_not_flagged() {
    let (mut y, group) = sibtest_bank(380, 260, 5, &[2], 1.2, 0.0, 0x11D);
    let n = 640usize;
    for p in 0..n {
        y[p * 5 + 4] = 1; // constant item: no within-level variance anywhere
    }
    let rows = sibtest(&y, &group, n, 5, &SibtestConfig::default()).unwrap();
    assert!(rows[4].beta_uni.is_nan(), "constant item must be NaN, not 0.0");
    assert!(rows[4].p_value.is_nan(), "constant item p must be NaN, not 1.0");
    assert!(!rows[4].flagged_bh, "an undefined row must never be flagged");
    assert_eq!(rows[4].n_strata_used, 0);
    // the planted DIF item is still detected alongside it, and IS flagged -- the positive twin of the
    // assertion above, without which the whole Benjamini-Hochberg block could be deleted and every
    // test would still pass
    assert!(rows[2].p_value.is_finite() && rows[2].p_value < 0.05, "p = {}", rows[2].p_value);
    assert!(rows[2].flagged_bh, "the planted DIF item must be BH-flagged");
    // `fdr_q` is actually plumbed through rather than ignored. Asserted as NESTING plus a strict
    // reduction, not as "a tiny q flags nothing": the planted item's p-value is around 1e-16, so it
    // legitimately survives an arbitrarily small level, and a test demanding otherwise would be
    // asserting a bug.
    let strict = sibtest(&y, &group, n, 5, &SibtestConfig { fdr_q: 1e-9, ..SibtestConfig::default() })
        .unwrap();
    let (lax_n, strict_n) = (
        rows.iter().filter(|r| r.flagged_bh).count(),
        strict.iter().filter(|r| r.flagged_bh).count(),
    );
    assert!(
        strict_n < lax_n,
        "tightening fdr_q must flag strictly fewer items ({strict_n} vs {lax_n}); the configured \
         level is being ignored"
    );
    assert!(
        strict.iter().zip(&rows).all(|(s, r)| !s.flagged_bh || r.flagged_bh),
        "the strict flag set must be nested inside the lax one"
    );
    assert!(strict[2].flagged_bh, "p ~ 1e-16 survives any usable level");
}

/// MONTE-CARLO TYPE I, 500 replications per cell, no DIF planted so every rejection is a false
/// positive. This exists because the module note now makes a quantitative Type I CLAIM, and a claim in
/// the docs that nothing checks is how documentation rots.
///
/// The finding is deliberately unflattering and is asserted as such: SIBTEST over-rejects, and by more
/// than Mantel-Haenszel under impact, because `se_beta` treats the estimated regression correction as
/// fixed. Asserted as loose bounds rather than point values so the test pins the DIRECTION of the
/// finding without becoming a seed-dependent tripwire.
#[test]
#[ignore = "500-replication Monte-Carlo; run explicitly"]
fn sibtest_type_i_error_exceeds_mantel_haenszel_under_impact() {
    const REPS: usize = 500;
    for (impact, n_ref, n_focal, n_items) in [(0.0f64, 1000usize, 1000usize, 5usize), (1.0, 1000, 1000, 5)] {
        let (mut mh_fp, mut mh_tot, mut sib_fp, mut sib_tot) = (0usize, 0usize, 0usize, 0usize);
        for rep in 0..REPS {
            let (y, group) =
                sibtest_bank(n_ref, n_focal, n_items, &[], 0.0, impact, 0xA000 + rep as u64);
            let n = n_ref + n_focal;
            let mh = mantel_haenszel_dif(&y, &group, n, n_items, &MhDifConfig::default()).unwrap();
            let sib = sibtest(&y, &group, n, n_items, &SibtestConfig::default()).unwrap();
            for r in &mh {
                if r.p_value.is_finite() {
                    mh_tot += 1;
                    mh_fp += usize::from(r.p_value < 0.05);
                }
            }
            for r in &sib {
                if r.p_value.is_finite() {
                    sib_tot += 1;
                    sib_fp += usize::from(r.p_value < 0.05);
                }
            }
        }
        let (mh_rate, sib_rate) = (mh_fp as f64 / mh_tot as f64, sib_fp as f64 / sib_tot as f64);
        println!("impact={impact}: MH type-I={mh_rate}  SIBTEST type-I={sib_rate}");
        assert!(
            (0.02..0.09).contains(&mh_rate),
            "Mantel-Haenszel Type I drifted out of its documented band: {mh_rate}"
        );
        assert!(
            sib_rate > 0.05,
            "SIBTEST is documented as over-rejecting; if this now holds nominal the docs are stale: \
             {sib_rate}"
        );
        if impact > 0.0 {
            assert!(
                sib_rate > mh_rate,
                "under impact SIBTEST is documented as over-rejecting MORE than Mantel-Haenszel: \
                 {sib_rate} vs {mh_rate}"
            );
        }
    }
}

/// Validation is non-vacuous: each guard trips on its own.
#[test]
fn sibtest_validates() {
    let (y, group) = sibtest_bank(60, 60, 4, &[], 0.0, 0.0, 0x99);
    let ok = SibtestConfig::default();
    assert!(sibtest(&y, &group, 120, 4, &ok).is_ok());
    // the valid subtest needs >= 2 items, so a 2-item test cannot be swept
    let (y2, g2) = sibtest_bank(60, 60, 2, &[], 0.0, 0.0, 0x99);
    assert!(sibtest(&y2, &g2, 120, 2, &ok).is_err());
    // a within-level variance needs two examinees
    assert!(sibtest(&y, &group, 120, 4, &SibtestConfig { j_min: 1, ..ok }).is_err());
    // shared boundary checks still apply
    assert!(sibtest(&y, &group, 121, 4, &ok).is_err());
    assert!(sibtest(&y, &group, 120, 4, &SibtestConfig { fdr_q: 0.0, ..ok }).is_err());
}
