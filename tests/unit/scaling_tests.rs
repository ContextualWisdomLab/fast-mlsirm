//! Tests for Thurstone Case V scaling. All pins from the EXECUTED 50-digit
//! mpmath oracle (session files/thurstone_oracle.py); tolerance 1e-6 covers
//! the shared approximate qnorm/erfc kernels (reviewer-measured fixture
//! error <= 2.2e-8).
//!
//! Mutation kills (all EXECUTED):
//! - MU1 colmeans->rowmeans: fixture A scale would become
//!   [1.3097083, 0.8134645, 0] -> thur_fixture_a scale pin FAILS.
//! - MU2 model sign flip pnorm(S_i - S_j): A model[0][2] would become
//!   0.0951472686 -> thur_fixture_a model pin FAILS.
//! - MU3 drop min-subtraction: A scale[0] would become -0.6019840261 ->
//!   thur_fixture_a scale pin FAILS.
//! - MU4 GF denominator sum(model^2): fixture C GF would become
//!   0.98619247878773876 (delta 4.4e-5 > 1e-6) -> thur_fixture_c GF pin
//!   FAILS. KNOWN LIMITATION: fixture A's MU4 delta (4.3e-7) is below the
//!   1e-6 tolerance, so A alone cannot kill MU4 — C is the designated
//!   MU4 anchor.
//! - MU5 qnorm->identity: A scale would become [0, 1/6, 13/30] ->
//!   thur_fixture_a scale pin FAILS.
//!
//! Fixture B has zero residual by construction (exactly Case-V-consistent)
//! and is therefore variance-mutation-blind; it anchors only the round-trip
//! identity, never GF mutations.

use super::*;

/// Minimal LCG + Box-Muller normal (crate PRNG idiom) for the MC anchors.
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

const TOL: f64 = 1e-6;

fn fixture_a() -> Vec<f64> {
    vec![0.5, 0.7, 0.9, 0.3, 0.5, 0.8, 0.1, 0.2, 0.5]
}

#[test]
fn thur_fixture_a_scale_gof_model() {
    let res = thurstone_case_v(&fixture_a(), 3).unwrap();
    // Oracle scale: [0, 0.49624378579592261, 1.3097082924567186].
    assert!(
        (res.scale[0] - 0.0).abs() < TOL,
        "scale[0]={}",
        res.scale[0]
    );
    assert!(
        (res.scale[1] - 0.49624378579592261).abs() < TOL,
        "scale[1]={}",
        res.scale[1]
    );
    assert!(
        (res.scale[2] - 1.3097082924567186).abs() < TOL,
        "scale[2]={}",
        res.scale[2]
    );
    // Oracle GF (FULL matrix incl. diagonal — pins CODE behavior over the
    // stale .Rd "lower off diagonal" prose; lower-only GF would be
    // 0.99868280449920059).
    assert!(
        (res.gof - 0.99986967677023893).abs() < TOL,
        "gof={}",
        res.gof
    );
    // Oracle model pins (kills MU2 sign flip: flipped model[0][2] would be
    // 0.095147268609398397).
    assert!(
        (res.model[2] - 0.9048527313906016).abs() < TOL,
        "model[0][2]={}",
        res.model[2]
    );
    assert!(
        (res.model[6] - 0.095147268609398397).abs() < TOL,
        "model[2][0]={}",
        res.model[6]
    );
    // Nonzero residual pin (variance-sensitive): model[0][1] - 0.7.
    assert!(
        (res.residual[1] - (-0.00986121100602988)).abs() < TOL,
        "residual[0][1]={}",
        res.residual[1]
    );
}

#[test]
fn thur_fixture_b_consistent_roundtrip() {
    // choice01 = pnorm(qnorm(0.75)) round trip -> zero residual, GF = 1.
    // Structural anchor only (variance-mutation-blind by construction).
    let choice = vec![0.5, 0.75, 0.25, 0.5];
    let res = thurstone_case_v(&choice, 2).unwrap();
    assert!(
        (res.scale[0] - 0.0).abs() < TOL,
        "scale[0]={}",
        res.scale[0]
    );
    assert!(
        (res.scale[1] - 0.67448975019608174).abs() < TOL,
        "scale[1]={}",
        res.scale[1]
    );
    assert!((res.gof - 1.0).abs() < TOL, "gof={}", res.gof);
    for (k, &r) in res.residual.iter().enumerate() {
        assert!(r.abs() < TOL, "residual[{k}]={r}");
    }
}

#[test]
fn thur_fixture_c_min_not_first_column() {
    // 4x4 with an intransitivity; min colmean is column 1, NOT column 0,
    // so the min-subtraction (MU3) and the GF denominator (MU4, delta
    // 4.4e-5) are both observable here.
    let choice = vec![
        0.5, 0.3, 0.8, 0.6, //
        0.7, 0.5, 0.9, 0.75, //
        0.2, 0.1, 0.5, 0.55, //
        0.4, 0.25, 0.45, 0.5,
    ];
    let res = thurstone_case_v(&choice, 4).unwrap();
    let want_scale = [
        0.47746850111201244,
        0.0,
        1.1194883201777909,
        0.88348500715891964,
    ];
    for (k, (&got, &want)) in res.scale.iter().zip(want_scale.iter()).enumerate() {
        assert!(
            (got - want).abs() < TOL,
            "scale[{k}]: got {got}, want {want}"
        );
    }
    // Designated MU4 killer: mutant GF would be 0.98619247878773876.
    assert!(
        (res.gof - 0.98623644601573367).abs() < TOL,
        "gof={}",
        res.gof
    );
    let want_row0 = [
        0.5,
        0.31651427279825857,
        0.739569842635635,
        0.65763476377667398,
    ];
    for (j, (&got, &want)) in res.model[0..4].iter().zip(want_row0.iter()).enumerate() {
        assert!(
            (got - want).abs() < TOL,
            "model[0][{j}]: got {got}, want {want}"
        );
    }
}

#[test]
fn thur_error_contract() {
    // n < 2.
    let err = thurstone_case_v(&[0.5], 1).unwrap_err();
    assert!(err.contains("at least 2"), "{err}");
    // Length mismatch.
    let err = thurstone_case_v(&[0.5, 0.5, 0.5], 2).unwrap_err();
    assert!(err.contains("row-major"), "{err}");
    // Non-finite entry.
    let err = thurstone_case_v(&[0.5, f64::NAN, 0.5, 0.5], 2).unwrap_err();
    assert!(err.contains("finite"), "{err}");
    // Boundary 0/1 rejected (deliberate safety divergence from psych's
    // direct-matrix path, which would produce infinities).
    for bad in [0.0, 1.0, -0.1, 1.5] {
        let err = thurstone_case_v(&[0.5, bad, 0.5, 0.5], 2).unwrap_err();
        assert!(err.contains("strictly in (0, 1)"), "{err}");
    }
}

#[test]
#[ignore]
fn thur_mc_500_consistent_recovery() {
    // 500 reps: build an exactly Case-V-consistent choice matrix from a
    // random bounded scale (values in [0, 2] so |S_j - S_i| <= 2 and
    // pnorm stays in [0.0228, 0.9772], safely inside strict (0, 1) —
    // generator explicitly bounded per spec review), then assert the
    // crate recovers the generating scale (shift-aligned) and near-unit
    // GF. Every assert reads crate outputs.
    let mut state = 0x51ab_cdef_1234_5678u64;
    let mut lcg = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    fn pnorm_gen(z: f64) -> f64 {
        // Abramowitz-Stegun 7.1.26-style approximation, |err| < 7.5e-8 —
        // independent of the crate's erfc path.
        let x = z / std::f64::consts::SQRT_2;
        let t = 1.0 / (1.0 + 0.3275911 * x.abs());
        let poly = t
            * (0.254829592
                + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        let erf_abs = 1.0 - poly * (-x * x).exp();
        let erf = if x >= 0.0 { erf_abs } else { -erf_abs };
        0.5 * (1.0 + erf)
    }
    for rep in 0..500 {
        let n = 3 + (rep % 4); // n in 3..=6
        let scale_true: Vec<f64> = (0..n).map(|_| 2.0 * lcg()).collect();
        let mut choice = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                choice[i * n + j] = pnorm_gen(scale_true[j] - scale_true[i]);
            }
        }
        let res = thurstone_case_v(&choice, n).unwrap();
        let min_true = scale_true.iter().cloned().fold(f64::INFINITY, f64::min);
        for k in 0..n {
            let want = scale_true[k] - min_true;
            assert!(
                (res.scale[k] - want).abs() < 1e-5,
                "rep {rep} scale[{k}]: got {}, want {want}",
                res.scale[k]
            );
        }
        assert!(res.gof > 0.999999, "rep {rep} gof={}", res.gof);
    }
}

// ---------------------------------------------------------------------------
// Bradley-Terry MM (choix 0.4.1 READ; Hunter 2004 / Bradley & Terry 1952
// NOT READ, as-cited). Pins from files/bt_oracle.py (mpmath 50 digits,
// EXECUTED; cross-checked against installed choix 0.4.1 to <=1.4e-12).
//
// Mutation kills (all deviations oracle- or reviewer-EXECUTED):
// - MU1 wins[j] += c (swap winner/loser): A param deviation 0.7577 -> killed
//   by bt_fixture_a pins.
// - MU2 drop denoms[loser] update: deviation 0.3789 -> killed by A pins.
// - MU3 drop centering: deviation 0.0362 -> killed by A pins.
// - MU4 exp_transform sum=1 instead of sum=n: INVISIBLE at alpha=0 (the
//   update map is scale-invariant there; A/B/C pins cannot kill it) ->
//   killed ONLY by bt_fixture_d_alpha_map (alpha=0.5, deviation 0.0108).
// - MU5 convergence tol instead of tol*n: same fixed point, NOT
//   value-killable -> killed by the iterations==18 pin at tol=1e-8 in
//   bt_iteration_count_tol_semantics (mutant needs 19; reviewer-executed).

const BT_TOL: f64 = 1e-9;

fn bt_fixture_a() -> Vec<f64> {
    vec![0.0, 3.0, 1.0, 2.0, 0.0, 4.0, 5.0, 1.0, 0.0]
}

#[test]
fn bt_fixture_a_params_weights() {
    // Kills MU1/MU2/MU3: params and weights pinned to the 50-digit oracle.
    // All asserts read the BradleyTerryResult returned by the crate.
    let r = bradley_terry_mm(&bt_fixture_a(), 3, 0.0, 100000, 1e-12).unwrap();
    let params_pin = [
        -0.378869072353494149,
        0.274223212389322699,
        0.104645859964171450,
    ];
    let weights_pin = [
        0.660321972720804678,
        1.268791102424604100,
        1.070886924854591222,
    ];
    for k in 0..3 {
        assert!(
            (r.params[k] - params_pin[k]).abs() < BT_TOL,
            "param[{k}] = {}",
            r.params[k]
        );
        assert!(
            (r.weights[k] - weights_pin[k]).abs() < BT_TOL,
            "weight[{k}] = {}",
            r.weights[k]
        );
    }
    assert!(r.iterations >= 2);
    // choix conventions read back from crate outputs: mean-0 params,
    // sum-n weights.
    assert!(r.params.iter().sum::<f64>().abs() < 1e-12);
    assert!((r.weights.iter().sum::<f64>() - 3.0).abs() < 1e-12);
}

#[test]
fn bt_fixture_b_exact_closed_form() {
    // 2-item MLE has the exact closed form p = 3/4 -> params +-ln(3)/2,
    // weights (3/2, 1/2) (verified symbolically in the oracle).
    let r = bradley_terry_mm(&[0.0, 3.0, 1.0, 0.0], 2, 0.0, 100000, 1e-12).unwrap();
    let half_ln3 = 3.0f64.ln() / 2.0;
    assert!((r.params[0] - half_ln3).abs() < BT_TOL, "{}", r.params[0]);
    assert!((r.params[1] + half_ln3).abs() < BT_TOL, "{}", r.params[1]);
    assert!((r.weights[0] - 1.5).abs() < BT_TOL);
    assert!((r.weights[1] - 0.5).abs() < BT_TOL);
}

#[test]
fn bt_fixture_c_zero_pair() {
    // 4 items with an all-zero off-diagonal pair (0,3); graph still
    // strongly connected. Oracle pins.
    let wins = vec![
        0.0, 2.0, 5.0, 0.0, //
        4.0, 0.0, 1.0, 2.0, //
        1.0, 3.0, 0.0, 6.0, //
        0.0, 3.0, 2.0, 0.0,
    ];
    let r = bradley_terry_mm(&wins, 4, 0.0, 100000, 1e-12).unwrap();
    let pins = [
        0.364305649211397136,
        -0.090796520655515820,
        0.144160716278834374,
        -0.417669844834715690,
    ];
    for k in 0..4 {
        assert!(
            (r.params[k] - pins[k]).abs() < BT_TOL,
            "param[{k}] = {}",
            r.params[k]
        );
    }
    assert!((r.weights[3] - 0.632281914456067132).abs() < BT_TOL);
}

#[test]
fn bt_fixture_d_alpha_map() {
    // MAP path (alpha = 0.5) on fixture A. This is the designated MU4
    // killer: the alpha=0 update map is scale-invariant, so only a
    // regularized fit can detect a wrong exp_transform normalization.
    let r = bradley_terry_mm(&bt_fixture_a(), 3, 0.5, 100000, 1e-12).unwrap();
    let pins = [
        -0.337946615223381393,
        0.240502957855231605,
        0.097443657368149788,
    ];
    for k in 0..3 {
        assert!(
            (r.params[k] - pins[k]).abs() < BT_TOL,
            "param[{k}] = {}",
            r.params[k]
        );
    }
}

#[test]
fn bt_iteration_count_tol_semantics() {
    // MU5 killer: choix's NormOfDifferenceTest threshold is tol * n. On
    // fixture A at tol=1e-8 the correct rule fires after exactly 18
    // updates; the mutant per-vector rule (tol, not tol*n) needs 19
    // (independently executed at spec-verify in float and mpmath).
    let r = bradley_terry_mm(&bt_fixture_a(), 3, 0.0, 10000, 1e-8).unwrap();
    assert_eq!(r.iterations, 18, "iterations = {}", r.iterations);
}

#[test]
fn bt_error_contract() {
    let a = bt_fixture_a();
    assert!(bradley_terry_mm(&[0.0], 1, 0.0, 100, 1e-8).is_err()); // n < 2
    assert!(bradley_terry_mm(&a[..6], 3, 0.0, 100, 1e-8).is_err()); // len
    let mut bad = a.clone();
    bad[0] = f64::NAN;
    assert!(bradley_terry_mm(&bad, 3, 0.0, 100, 1e-8).is_err()); // non-finite
    let mut neg = a.clone();
    neg[1] = -1.0;
    assert!(bradley_terry_mm(&neg, 3, 0.0, 100, 1e-8).is_err()); // negative
    let mut diag = a.clone();
    diag[4] = 2.0;
    assert!(bradley_terry_mm(&diag, 3, 0.0, 100, 1e-8).is_err()); // diagonal
    assert!(bradley_terry_mm(&[0.0; 9], 3, 0.0, 100, 1e-8).is_err()); // no data
    assert!(bradley_terry_mm(&a, 3, -0.5, 100, 1e-8).is_err()); // alpha < 0
    assert!(bradley_terry_mm(&a, 3, 0.0, 100, 0.0).is_err()); // tol <= 0
    assert!(bradley_terry_mm(&a, 3, 0.0, 0, 1e-8).is_err()); // max_iter 0
                                                             // Item 0 never loses: Ford condition violated, no finite MLE; the MM
                                                             // iteration must fail to converge (demonstrated numerically; Ford
                                                             // 1957 NOT READ).
    let unbeaten = vec![0.0, 2.0, 2.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let e = bradley_terry_mm(&unbeaten, 3, 0.0, 5000, 1e-8).unwrap_err();
    assert!(e.contains("did not converge"), "{e}");
    // Item with zero wins at alpha=0: log-worth is -inf on the first
    // update -> non-finite error, not a bogus result.
    let winless = vec![0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0];
    let e2 = bradley_terry_mm(&winless, 3, 0.0, 5000, 1e-8).unwrap_err();
    assert!(e2.contains("non-finite"), "{e2}");
    // Extreme-magnitude counts (impl-review probe): symmetric finite 1e300
    // counts are VALID input whose MLE is the zero vector -- accepted with
    // the correct fit, not rejected. Counts whose row sum overflows f64
    // (2e308 -> inf) trip the non-finite update guard instead of
    // returning a bogus result.
    let big = vec![0.0, 1e300, 1e300, 0.0];
    let rb = bradley_terry_mm(&big, 2, 0.0, 100, 1e-8).unwrap();
    assert!(rb.params.iter().all(|p| p.abs() < 1e-12), "{:?}", rb.params);
    let overflow = vec![0.0, 1e308, 1e308, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0];
    let e3 = bradley_terry_mm(&overflow, 3, 0.0, 100, 1e-8).unwrap_err();
    assert!(e3.contains("non-finite"), "{e3}");
}

#[test]
#[ignore] // MC-500: cargo test -p mlsirm-core --lib scaling -- --ignored
fn bt_mc_500_recovery() {
    // 500 replications: bounded true params, 400 comparisons per ordered
    // pair, refit with the crate, mean absolute param error < 0.15.
    // Every rep's assert reads the crate's returned params.
    let n = 4usize;
    let mut rng = Lcg(20260727);
    let mut worst = 0.0f64;
    for rep in 0..500 {
        let mut truth: Vec<f64> = (0..n).map(|_| 0.5 * rng.normal()).collect();
        let m = truth.iter().sum::<f64>() / n as f64;
        for t in truth.iter_mut() {
            *t = (*t - m).clamp(-0.7, 0.7);
        }
        let m2 = truth.iter().sum::<f64>() / n as f64;
        for t in truth.iter_mut() {
            *t -= m2;
        }
        let w: Vec<f64> = truth.iter().map(|t| t.exp()).collect();
        let mut wins = vec![0.0f64; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                let p = w[i] / (w[i] + w[j]);
                for _ in 0..400 {
                    if rng.next_f64() < p {
                        wins[i * n + j] += 1.0;
                    } else {
                        wins[j * n + i] += 1.0;
                    }
                }
            }
        }
        let r = bradley_terry_mm(&wins, n, 0.0, 100000, 1e-10).unwrap();
        let mae = truth
            .iter()
            .zip(r.params.iter())
            .map(|(t, p)| (t - p).abs())
            .sum::<f64>()
            / n as f64;
        worst = worst.max(mae);
        assert!(mae < 0.15, "rep {rep}: mae = {mae}");
    }
    assert!(worst > 0.0);
}

// ---------------------------------------------------------------------------
// LSR / I-LSR (choix 0.4.1 lsr.py; Maystre & Grossglauser, 2015)
//
// Oracle: exact-Fraction one-shot statdist + mpmath 50-digit I-LSR fixed
// points, cross-checked against pip choix 0.4.1 (<= 2.2e-13) and against
// the shipped bradley_terry_mm pins (<= 4e-19). Fixtures:
// A = [[0,3,1],[2,0,4],[5,1,0]], B = [[0,3],[1,0]],
// C = [[0,2,5,0],[4,0,1,2],[1,3,0,6],[0,3,2,0]].
// ---------------------------------------------------------------------------

/// One-shot LSR exact pins on fixture A. Asserts read
/// `lsr_pairwise(...)` crate outputs: params, weights, iterations.
/// Kills: MU1 chain transpose (weights dev 0.685), MU2 dropped diagonal
/// subtraction (exact statdist becomes [9/2, 3/4, -9/4]; the negative
/// entry trips the positivity guard -> Err, so the unwrap fails),
/// MU3 dropped centering (params[0] dev 0.0326), MU4 statdist sum n ->
/// sum 1 (params PROVEN invariant under centering; the weights pins are
/// the designated anchor: 12/17 -> 4/51, dev 0.47).
#[test]
fn lsr_fixture_a_one_shot() {
    let a = [0.0, 3.0, 1.0, 2.0, 0.0, 4.0, 5.0, 1.0, 0.0];
    let r = lsr_pairwise(&a, 3, 0.0).unwrap();
    assert_eq!(r.iterations, 1);
    // Exact rationals: statdist = [12/17, 45/34, 33/34].
    let exact_w = [12.0 / 17.0, 45.0 / 34.0, 33.0 / 34.0];
    let exact_p = [
        -0.31568746351363625118,
        0.31292119590873788656,
        0.0027662676048983646254,
    ];
    for k in 0..3 {
        assert!(
            (r.weights[k] - exact_w[k]).abs() < 1e-12,
            "weights[{k}] = {}",
            r.weights[k]
        );
        assert!(
            (r.params[k] - exact_p[k]).abs() < 1e-12,
            "params[{k}] = {}",
            r.params[k]
        );
    }
    // Fixture B: n = 2 one-shot LSR is exact (+/- ln(3)/2, the 2-item
    // Bradley-Terry closed form) and statdist = [3/2, 1/2].
    let b = [0.0, 3.0, 1.0, 0.0];
    let rb = lsr_pairwise(&b, 2, 0.0).unwrap();
    let half_ln3 = 3.0f64.ln() / 2.0;
    assert!((rb.params[0] - half_ln3).abs() < 1e-14);
    assert!((rb.params[1] + half_ln3).abs() < 1e-14);
    assert!((rb.weights[0] - 1.5).abs() < 1e-14);
    assert!((rb.weights[1] - 0.5).abs() < 1e-14);
}

/// One-shot LSR exact pins on fixture C (n = 4, has a zero pair) and on
/// fixture A with alpha = 1/2 (regularization enters the chain rates).
/// Asserts read `lsr_pairwise(...)` crate outputs.
#[test]
fn lsr_fixture_c_and_alpha() {
    let c = [
        0.0, 2.0, 5.0, 0.0, 4.0, 0.0, 1.0, 2.0, 1.0, 3.0, 0.0, 6.0, 0.0, 3.0, 2.0, 0.0,
    ];
    let r = lsr_pairwise(&c, 4, 0.0).unwrap();
    // Exact: statdist = [1256/899, 880/899, 904/899, 556/899].
    let exact_w = [1256.0 / 899.0, 880.0 / 899.0, 904.0 / 899.0, 556.0 / 899.0];
    let exact_p = [
        0.37488561974235523196,
        0.019120180186463402615,
        0.046027633106387743118,
        -0.44003343303520637769,
    ];
    for k in 0..4 {
        assert!((r.weights[k] - exact_w[k]).abs() < 1e-12);
        assert!((r.params[k] - exact_p[k]).abs() < 1e-12);
    }
    // A with alpha = 1/2: statdist = [96/125, 153/125, 126/125].
    let a = [0.0, 3.0, 1.0, 2.0, 0.0, 4.0, 5.0, 1.0, 0.0];
    let ra = lsr_pairwise(&a, 3, 0.5).unwrap();
    let exact_wa = [96.0 / 125.0, 153.0 / 125.0, 126.0 / 125.0];
    let exact_pa = [
        -0.2460078151360803278,
        0.22008191478851889676,
        0.025925900347561431035,
    ];
    for k in 0..3 {
        assert!((ra.weights[k] - exact_wa[k]).abs() < 1e-12);
        assert!((ra.params[k] - exact_pa[k]).abs() < 1e-12);
    }
}

/// I-LSR at alpha = 0 reproduces the Bradley-Terry MLE (cross-algorithm
/// anchor; mpmath fixed point agrees with the BT oracle to 4e-19).
/// Asserts read `ilsr_pairwise(...)` AND `bradley_terry_mm(...)` crate
/// outputs. atol 1e-7: the converged iterate at tol = 1e-8 sits within
/// ~5e-9 of the fixed point (same rationale as the BT pytest pins);
/// mutant deviations are >= 3e-2. Kills MU5 (denominator (w_i + w_j) ->
/// w_i): UNOBSERVABLE in one-shot LSR (uniform weights make it a global
/// rate scale; statdist invariant — documented limitation, the one-shot
/// tests above cannot see it), but from pass 2 the weights are
/// non-uniform and the I-LSR params deviate by 0.331 on A.
/// Iteration pin (15 at tol = 1e-8) anchors the convergence loop.
#[test]
fn ilsr_fixture_a_matches_bt() {
    let a = [0.0, 3.0, 1.0, 2.0, 0.0, 4.0, 5.0, 1.0, 0.0];
    let r = ilsr_pairwise(&a, 3, 0.0, 100, 1e-8).unwrap();
    let mle = [
        -0.37886907235349414916,
        0.2742232123893226994,
        0.10464585996417144976,
    ];
    for k in 0..3 {
        assert!(
            (r.params[k] - mle[k]).abs() < 1e-7,
            "params[{k}] = {}",
            r.params[k]
        );
    }
    assert_eq!(r.iterations, 15);
    let wsum: f64 = r.weights.iter().sum();
    assert!((wsum - 3.0).abs() < 1e-9, "weights sum = {wsum}");
    // Same MLE as the MM algorithm (different algorithm, same likelihood).
    let bt = bradley_terry_mm(&a, 3, 0.0, 100, 1e-10).unwrap();
    for k in 0..3 {
        assert!((r.params[k] - bt.params[k]).abs() < 1e-6);
    }
    // alpha = 0.5 I-LSR pins: LSR regularization is NOT MM Dirichlet-MAP
    // regularization; these deliberately differ from bradley_terry_mm at
    // the same alpha (both follow their sources).
    let ra = ilsr_pairwise(&a, 3, 0.5, 100, 1e-8).unwrap();
    let pa = [
        -0.27752663700895662385,
        0.19348504484501984521,
        0.08404159216393677864,
    ];
    for k in 0..3 {
        assert!((ra.params[k] - pa[k]).abs() < 1e-7);
    }
}

/// I-LSR fixture C pins + the tol-semantics iteration anchor.
/// Asserts read `ilsr_pairwise(...)` crate outputs. Kills MU6
/// (tol * n -> tol in the convergence test): C converges in 17 passes
/// with choix semantics but 18 with bare tol (measured; fixture A does
/// NOT separate the two — both 15 — which is why C carries this pin).
#[test]
fn ilsr_fixture_c_iterations() {
    let c = [
        0.0, 2.0, 5.0, 0.0, 4.0, 0.0, 1.0, 2.0, 1.0, 3.0, 0.0, 6.0, 0.0, 3.0, 2.0, 0.0,
    ];
    let r = ilsr_pairwise(&c, 4, 0.0, 100, 1e-8).unwrap();
    let mle = [
        0.36430564921139713618,
        -0.090796520655515820053,
        0.14416071627883437422,
        -0.41766984483471569034,
    ];
    for k in 0..4 {
        assert!((r.params[k] - mle[k]).abs() < 1e-7);
    }
    assert_eq!(r.iterations, 17);
}

/// Error contract for both entry points. Every assert reads a crate
/// Result. Includes the spec-review overflow mandates: finite-but-huge
/// counts and huge finite alpha must Err (not NaN or finite garbage).
#[test]
fn lsr_error_contract() {
    let a = [0.0, 3.0, 1.0, 2.0, 0.0, 4.0, 5.0, 1.0, 0.0];
    // n < 2, wrong length.
    assert!(lsr_pairwise(&a, 1, 0.0).is_err());
    assert!(lsr_pairwise(&a[..8], 3, 0.0).is_err());
    // Non-finite / negative / nonzero-diagonal / all-zero.
    let mut bad = a;
    bad[1] = f64::NAN;
    assert!(lsr_pairwise(&bad, 3, 0.0).is_err());
    bad[1] = -1.0;
    assert!(lsr_pairwise(&bad, 3, 0.0).is_err());
    bad[1] = 3.0;
    bad[0] = 1.0;
    assert!(lsr_pairwise(&bad, 3, 0.0).is_err());
    assert!(lsr_pairwise(&[0.0; 9], 3, 0.0).is_err());
    // alpha domain (including inf, which choix maps to NaN output).
    assert!(lsr_pairwise(&a, 3, -0.5).is_err());
    assert!(lsr_pairwise(&a, 3, f64::INFINITY).is_err());
    // ilsr knobs.
    assert!(ilsr_pairwise(&a, 3, 0.0, 0, 1e-8).is_err());
    assert!(ilsr_pairwise(&a, 3, 0.0, 100, 0.0).is_err());
    assert!(ilsr_pairwise(&a, 3, 0.0, 100, f64::NAN).is_err());
    // Non-convergence.
    assert!(ilsr_pairwise(&a, 3, 0.0, 1, 1e-8).is_err());
    // Disconnected comparison graph (choix raises ValueError too):
    // D = [[0,2,0,0],[1,0,0,0],[0,0,0,3],[0,0,1,0]].
    let d = [
        0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 1.0, 0.0,
    ];
    assert!(lsr_pairwise(&d, 4, 0.0).is_err());
    assert!(ilsr_pairwise(&d, 4, 0.0, 100, 1e-8).is_err());
    // alpha > 0 makes the chain irreducible: D becomes estimable.
    let rd = lsr_pairwise(&d, 4, 0.5).unwrap();
    assert!(rd.params.iter().all(|p| p.is_finite()));
    // Overflow mandates: counts / alpha that overflow the generator row
    // sums -> Err, never NaN output. n = 4 with all off-diagonals at
    // 1.7e308 makes each row sum 3 * 8.5e307 = 2.55e308 -> inf. (An
    // n = 3 all-1e308 matrix does NOT overflow -- row sums are 1e308,
    // finite -- and is covered by the scale-invariance block below.)
    let mut huge = [1.7e308f64; 16];
    for k in 0..4 {
        huge[k * 4 + k] = 0.0;
    }
    assert!(lsr_pairwise(&huge, 4, 0.0).is_err());
    assert!(lsr_pairwise(&a, 3, 1e308).is_err());
    // Scale invariance (impl-review regression): the stationary
    // distribution is invariant under global count rescaling, so huge
    // but non-overflowing counts must give the SAME result, not a false
    // "disconnected" rejection from the pivot threshold.
    let base = lsr_pairwise(&a, 3, 0.0).unwrap();
    let scaled: Vec<f64> = a.iter().map(|c| c * 1e20).collect();
    let big = lsr_pairwise(&scaled, 3, 0.0).unwrap();
    for k in 0..3 {
        assert!(
            (big.params[k] - base.params[k]).abs() < 1e-12,
            "scaled params[{k}] = {}",
            big.params[k]
        );
        assert!((big.weights[k] - base.weights[k]).abs() < 1e-12);
    }
    let asym = [0.0, 1e150, 2e150, 3e149, 0.0, 4e149, 5e149, 6e149, 0.0];
    let ra = ilsr_pairwise(&asym, 3, 0.0, 100, 1e-8).unwrap();
    assert!(ra.params.iter().all(|p| p.is_finite()));
}

/// 500-rep Monte Carlo recovery (n = 4, bounded truth, 400 comparisons
/// per pair). Asserts read `ilsr_pairwise(...)` crate outputs.
#[test]
#[ignore = "500-rep Monte Carlo; run with -- --ignored"]
fn ilsr_mc_500_recovery() {
    let mut worst = 0.0f64;
    for rep in 0..500 {
        let mut rng = Lcg(0x1_57AB + rep as u64 * 7919);
        let n = 4usize;
        let truth: Vec<f64> = {
            let raw: Vec<f64> = (0..n).map(|_| rng.normal() * 0.8).collect();
            let m = raw.iter().sum::<f64>() / n as f64;
            raw.iter().map(|x| x - m).collect()
        };
        let mut wins = vec![0.0f64; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                let p = 1.0 / (1.0 + (truth[j] - truth[i]).exp());
                for _ in 0..400 {
                    if rng.next_f64() < p {
                        wins[i * n + j] += 1.0;
                    } else {
                        wins[j * n + i] += 1.0;
                    }
                }
            }
        }
        let r = ilsr_pairwise(&wins, n, 0.0, 100000, 1e-10).unwrap();
        let mae = truth
            .iter()
            .zip(r.params.iter())
            .map(|(t, p)| (t - p).abs())
            .sum::<f64>()
            / n as f64;
        worst = worst.max(mae);
        // 0.2 bound: with 400 comparisons/pair the worst-of-500 MAE lands
        // near 0.155 (measured); mutant deviations are >= 0.33.
        assert!(mae < 0.2, "rep {rep}: mae = {mae}");
    }
    assert!(worst > 0.0);
}
