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

// ===================== Rank Centrality (choix rank_centrality) ==========
//
// All pins from the EXECUTED exact-Fraction oracle
// (session files/rank_centrality_oracle.py; output cross-checked against
// pip choix to <= 2e-16). Mutation kills (all EXECUTED):
// - MU1 transpose (chain[i][j] built from wins[i][j] instead of the
//   loser->winner orientation): asymmetric fixture A pins FAIL.
// - MU2 skip ratio transform (use raw counts): reproduces lsr_pairwise,
//   whose A statdist is [12/17, 45/34, 33/34] != [138/181, 237/181,
//   168/181] -> rc_fixture_a pins FAIL.
// - MU3 ratio denominator c only (ratio == 1 wherever c > 0): A pins
//   FAIL (uniform weights). KNOWN LIMITATION: fixture E is blind to MU3
//   for its one-sided pair (true ratio there IS 1); A is the MU3 anchor.
// - MU4 statdist sum -> 1 instead of n: weights pins FAIL (params are
//   invariant to the scaling, as in LSR MU4 -- weights are the anchor).
// - MU5 drop log-centering: params pins FAIL.
// - MU6 half-updated ratio denominator (in-place transform reading
//   already-transformed symmetric entries): A weights would become
//   ~[1.1279, 1.0980, 0.7741] (spec-review probe) -> rc_fixture_a FAILS.

/// Fixture A exact pins (statdist [138/181, 237/181, 168/181]) plus
/// centered-log params. Asserts read `rank_centrality(...)` outputs.
#[test]
fn rc_fixture_a_exact() {
    let a = [0.0, 3.0, 1.0, 2.0, 0.0, 4.0, 5.0, 1.0, 0.0];
    let r = rank_centrality(&a, 3, 0.0).unwrap();
    assert_eq!(r.iterations, 1);
    let wexp = [138.0 / 181.0, 237.0 / 181.0, 168.0 / 181.0];
    let pexp = [
        -0.2458389167413269090,
        0.2949675392365995849,
        -0.0491286224952726759,
    ];
    for k in 0..3 {
        assert!(
            (r.weights[k] - wexp[k]).abs() < 1e-15,
            "w[{k}] = {}",
            r.weights[k]
        );
        assert!(
            (r.params[k] - pexp[k]).abs() < 1e-15,
            "p[{k}] = {}",
            r.params[k]
        );
    }
    assert!((r.params.iter().sum::<f64>()).abs() < 1e-15);
}

/// Fixture C (4x4) exact pins + fixture A at alpha = 1/2
/// (statdist [207/265, 333/265, 255/265]). Asserts read crate outputs.
#[test]
fn rc_fixture_c_and_alpha() {
    let c = [
        0.0, 2.0, 1.0, 3.0, 1.0, 0.0, 2.0, 1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 2.0, 4.0, 0.0,
    ];
    let r = rank_centrality(&c, 4, 0.0).unwrap();
    let wexp = [
        28436.0 / 22269.0,
        13720.0 / 22269.0,
        21100.0 / 22269.0,
        25820.0 / 22269.0,
    ];
    let pexp = [
        0.2809226990008633964,
        -0.4478786267686341056,
        -0.0174602085843524600,
        0.1844161363521231693,
    ];
    for k in 0..4 {
        assert!((r.weights[k] - wexp[k]).abs() < 1e-15);
        assert!((r.params[k] - pexp[k]).abs() < 1e-15);
    }
    let a = [0.0, 3.0, 1.0, 2.0, 0.0, 4.0, 5.0, 1.0, 0.0];
    let ra = rank_centrality(&a, 3, 0.5).unwrap();
    let wexp5 = [207.0 / 265.0, 333.0 / 265.0, 255.0 / 265.0];
    let pexp5 = [
        -0.2279894828693772754,
        0.2474342138456974782,
        -0.0194447309763202028,
    ];
    for k in 0..3 {
        assert!((ra.weights[k] - wexp5[k]).abs() < 1e-15);
        assert!((ra.params[k] - pexp5[k]).abs() < 1e-15);
    }
}

/// One-sided pair (item 0 always beats 1; ratio exactly 1 on that edge;
/// statdist [6/5, 3/5, 6/5]) and alpha = 0 exact scale invariance
/// (global k * wins leaves every ratio unchanged). Asserts read crate
/// outputs.
#[test]
fn rc_one_sided_and_scale() {
    let e = [0.0, 4.0, 1.0, 0.0, 0.0, 2.0, 3.0, 1.0, 0.0];
    let r = rank_centrality(&e, 3, 0.0).unwrap();
    let wexp = [1.2, 0.6, 1.2];
    let pexp = [
        0.2310490601866484365,
        -0.4620981203732968729,
        0.2310490601866484365,
    ];
    for k in 0..3 {
        assert!((r.weights[k] - wexp[k]).abs() < 1e-15);
        assert!((r.params[k] - pexp[k]).abs() < 1e-15);
    }
    let a = [0.0, 3.0, 1.0, 2.0, 0.0, 4.0, 5.0, 1.0, 0.0];
    let base = rank_centrality(&a, 3, 0.0).unwrap();
    let scaled: Vec<f64> = a.iter().map(|x| x * 1e150).collect();
    let big = rank_centrality(&scaled, 3, 0.0).unwrap();
    for k in 0..3 {
        assert!((big.params[k] - base.params[k]).abs() < 1e-15);
        assert!((big.weights[k] - base.weights[k]).abs() < 1e-15);
    }
}

/// Error contract. Every assert reads a crate Result. Includes the
/// denominator-overflow mandate (alpha = 1e308 -> the ratio denominator
/// 2 * alpha overflows -> Err, never NaN or silent zeros) and the
/// disconnected graph D (Err at alpha = 0; exact pins at alpha = 1/2:
/// statdist [9/8, 7/8, 6/5, 4/5]).
#[test]
fn rc_error_contract() {
    let a = [0.0, 3.0, 1.0, 2.0, 0.0, 4.0, 5.0, 1.0, 0.0];
    assert!(rank_centrality(&a, 1, 0.0).is_err());
    assert!(rank_centrality(&a[..8], 3, 0.0).is_err());
    let mut bad = a;
    bad[1] = f64::NAN;
    assert!(rank_centrality(&bad, 3, 0.0).is_err());
    bad[1] = -1.0;
    assert!(rank_centrality(&bad, 3, 0.0).is_err());
    bad[1] = 3.0;
    bad[0] = 1.0;
    assert!(rank_centrality(&bad, 3, 0.0).is_err());
    assert!(rank_centrality(&[0.0; 9], 3, 0.0).is_err());
    assert!(rank_centrality(&a, 3, -0.5).is_err());
    assert!(rank_centrality(&a, 3, f64::INFINITY).is_err());
    assert!(rank_centrality(&a, 3, 1e308).is_err());
    let mut huge = a;
    huge[1] = 1e308;
    huge[3] = 9e307;
    assert!(rank_centrality(&huge, 3, 0.0).is_err());
    let d = [
        0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 1.0, 0.0,
    ];
    assert!(rank_centrality(&d, 4, 0.0).is_err());
    let rd = rank_centrality(&d, 4, 0.5).unwrap();
    let wexp = [1.125, 0.875, 1.2, 0.8];
    for k in 0..4 {
        assert!((rd.weights[k] - wexp[k]).abs() < 1e-15);
    }
}

/// 500-rep Monte Carlo recovery (n = 4, bounded BT truth, 400
/// comparisons per pair). Rank centrality is consistent for BT data but
/// is not the MLE, so the bound is looser than the I-LSR one. Asserts
/// read `rank_centrality(...)` crate outputs.
#[test]
#[ignore = "500-rep Monte Carlo; run with -- --ignored"]
fn rc_mc_500_recovery() {
    let mut worst = 0.0f64;
    for rep in 0..500 {
        let mut rng = Lcg(0x2_C4A1 + rep as u64 * 7919);
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
        let r = rank_centrality(&wins, n, 0.0).unwrap();
        let mae = truth
            .iter()
            .zip(r.params.iter())
            .map(|(t, p)| (t - p).abs())
            .sum::<f64>()
            / n as f64;
        worst = worst.max(mae);
        // 0.15 bound: worst-of-500 MAE measured at 0.0912 on this seed
        // stream. Mutant deviations (e.g. MU3 uniform weights) are >= 0.3.
        assert!(mae < 0.15, "rep {rep}: mae = {mae}");
    }
    assert!(worst > 0.0);
}

// ---------------------------------------------------------------------------
// Plackett-Luce rankings LSR / I-LSR (choix 0.4.1 lsr_rankings /
// ilsr_rankings). Pins from the EXECUTED exact-Fraction/mpmath oracle
// (session files pl_rankings_oracle.py), cross-checked against pip
// choix 0.4.1 (<= 2.3e-13).
//
// Mutation kills (all EXECUTED):
// - MU1 drop `s -= w[winner]` (stale placement denominator): killed by
//   pl_fixture_ra_exact (oracle-verified separation).
// - MU2 transpose chain update (winner->loser): killed by
//   pl_fixture_ra_exact (probe maxdiff 1.67; RA's item-0/2 symmetry
//   blinds only item-swap mistakes, not transposition) and
//   pl_fixture_rb_partial.
// - MU3 loser loop over the whole ranking instead of the suffix: killed
//   by pl_fixture_ra_exact / pl_fixture_rb_partial (adds reverse edges).
// - MU4 subset sum replaced by all-items sum: killed by
//   pl_fixture_rb_partial ONLY — PROVEN invisible on full-ranking
//   fixtures (oracle MU-separation probe), which is why a
//   partial-rankings anchor is mandatory.
// - MU5 I-LSR never propagates params into the pass worths: killed by
//   pl_ilsr_fixed_points (one-shot RA param -0.4176 vs fixed point
//   -0.3915) and its iterations pins.
// ---------------------------------------------------------------------------

/// Fixture RA: n = 3, four full rankings. Oracle weights
/// [6/11, 21/11, 6/11]; alpha = 1/2 weights [3/4, 3/2, 3/4].
/// Dataset duplication is exactly invariant at alpha = 0, NOT at
/// alpha = 1/2 (both crate-output pins).
#[test]
fn pl_fixture_ra_exact() {
    let ra: Vec<usize> = vec![1, 0, 2, 2, 1, 0, 0, 1, 2, 1, 2, 0];
    let st: Vec<usize> = vec![0, 3, 6, 9, 12];
    let r = lsr_rankings(&ra, &st, 3, 0.0).unwrap();
    assert_eq!(r.iterations, 1);
    let expw = [6.0 / 11.0, 21.0 / 11.0, 6.0 / 11.0];
    for (a, b) in r.weights.iter().zip(expw.iter()) {
        assert!((a - b).abs() < 1e-14, "weights {:?}", r.weights);
    }
    let expp = [
        -0.41758765616512266523,
        0.83517531233024533046,
        -0.41758765616512266523,
    ];
    for (a, b) in r.params.iter().zip(expp.iter()) {
        assert!((a - b).abs() < 1e-14, "params {:?}", r.params);
    }
    let r5 = lsr_rankings(&ra, &st, 3, 0.5).unwrap();
    let expw5 = [0.75, 1.5, 0.75];
    for (a, b) in r5.weights.iter().zip(expw5.iter()) {
        assert!((a - b).abs() < 1e-14, "alpha=1/2 weights {:?}", r5.weights);
    }
    // Duplication: exactly invariant at alpha = 0 ...
    let mut ra2 = ra.clone();
    ra2.extend_from_slice(&ra);
    let st2: Vec<usize> = vec![0, 3, 6, 9, 12, 15, 18, 21, 24];
    let rd = lsr_rankings(&ra2, &st2, 3, 0.0).unwrap();
    for (a, b) in rd.weights.iter().zip(r.weights.iter()) {
        assert!((a - b).abs() < 1e-15, "duplication must be invariant");
    }
    // ... but NOT at alpha = 1/2.
    let rd5 = lsr_rankings(&ra2, &st2, 3, 0.5).unwrap();
    let diff: f64 = rd5
        .weights
        .iter()
        .zip(r5.weights.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-3,
        "alpha > 0 duplication must differ, diff = {diff}"
    );
}

/// Fixture RB: n = 4, PARTIAL rankings (mixed lengths 2 and 3). Oracle
/// weights [172/175, 12/7, 16/35, 148/175] — the discriminating anchor
/// for the all-items-sum mutant (MU4), which full rankings cannot see.
#[test]
fn pl_fixture_rb_partial() {
    let rb: Vec<usize> = vec![0, 1, 2, 3, 2, 1, 3, 2, 0, 3, 3, 1, 0];
    let st: Vec<usize> = vec![0, 3, 5, 7, 10, 13];
    let r = lsr_rankings(&rb, &st, 4, 0.0).unwrap();
    let expw = [172.0 / 175.0, 12.0 / 7.0, 16.0 / 35.0, 148.0 / 175.0];
    for (a, b) in r.weights.iter().zip(expw.iter()) {
        assert!((a - b).abs() < 1e-14, "weights {:?}", r.weights);
    }
    let expp = [
        0.089865511836540348005,
        0.64615350967928836513,
        -0.67560233030303108203,
        -0.0604166912127976311,
    ];
    for (a, b) in r.params.iter().zip(expp.iter()) {
        assert!((a - b).abs() < 1e-14, "params {:?}", r.params);
    }
}

/// I-LSR fixed points (mpmath dps=50 oracle; atol 1e-7 is an
/// ORACLE-MEASURED margin — converged tol=1e-8 iterates sit 8.6e-11 (RA)
/// / 1.7e-9 (RB) from the fixed point; mutant deviations >= 1e-2).
/// Also pins iteration counts at tol=1e-8 (RA 8, RB 11; spec-review
/// choix-equivalent probes) and the returned-weights invariant
/// weights == exp_transform(params) (kills stale/uniform-weights
/// mutants that params-only pins miss).
#[test]
fn pl_ilsr_fixed_points() {
    let ra: Vec<usize> = vec![1, 0, 2, 2, 1, 0, 0, 1, 2, 1, 2, 0];
    let sta: Vec<usize> = vec![0, 3, 6, 9, 12];
    let r = ilsr_rankings(&ra, &sta, 3, 0.0, 100, 1e-8).unwrap();
    let expa = [
        -0.39145300187318291897,
        0.78290600374636583794,
        -0.39145300187318291897,
    ];
    for (a, b) in r.params.iter().zip(expa.iter()) {
        assert!((a - b).abs() < 1e-7, "RA ilsr params {:?}", r.params);
    }
    assert_eq!(r.iterations, 8, "RA pass count at tol=1e-8");
    let rb: Vec<usize> = vec![0, 1, 2, 3, 2, 1, 3, 2, 0, 3, 3, 1, 0];
    let stb: Vec<usize> = vec![0, 3, 5, 7, 10, 13];
    let rr = ilsr_rankings(&rb, &stb, 4, 0.0, 100, 1e-8).unwrap();
    let expb = [
        0.11603240603370623172,
        0.56148183703958190534,
        -0.58273127354584793185,
        -0.094782969527440205212,
    ];
    for (a, b) in rr.params.iter().zip(expb.iter()) {
        assert!((a - b).abs() < 1e-7, "RB ilsr params {:?}", rr.params);
    }
    assert_eq!(rr.iterations, 11, "RB pass count at tol=1e-8");
    // Returned-weights invariant: weights == exp_transform(params),
    // positive, sum n (reads BOTH crate outputs).
    for res in [&r, &rr] {
        let n = res.params.len() as f64;
        let mean = res.params.iter().sum::<f64>() / n;
        let mut w: Vec<f64> = res.params.iter().map(|p| (p - mean).exp()).collect();
        let s: f64 = w.iter().sum();
        for x in w.iter_mut() {
            *x *= n / s;
        }
        for (a, b) in res.weights.iter().zip(w.iter()) {
            assert!(*a > 0.0);
            assert!((a - b).abs() < 1e-12, "weights/params invariant");
        }
        let sw: f64 = res.weights.iter().sum();
        assert!((sw - n).abs() < 1e-9);
    }
}

/// Length-2 rankings are EXACTLY the pairwise chain (uniform worths:
/// both add 1/2 per comparison on the loser->winner edge), so
/// lsr_rankings must bit-match lsr_pairwise on the induced win matrix
/// (spec-review probe maxdiff 0.0).
#[test]
fn pl_length2_equivalence() {
    let rk: Vec<usize> = vec![0, 1, 1, 2, 2, 0, 0, 2];
    let st: Vec<usize> = vec![0, 2, 4, 6, 8];
    let r = lsr_rankings(&rk, &st, 3, 0.0).unwrap();
    // Induced wins: 0>1, 1>2, 2>0, 0>2.
    let mut wins = vec![0.0f64; 9];
    wins[0 * 3 + 1] = 1.0;
    wins[1 * 3 + 2] = 1.0;
    wins[2 * 3 + 0] = 1.0;
    wins[0 * 3 + 2] = 1.0;
    let rp = lsr_pairwise(&wins, 3, 0.0).unwrap();
    assert_eq!(r.params, rp.params, "length-2 rankings == pairwise");
    assert_eq!(r.weights, rp.weights);
}

/// Error contract + disconnected graph + overflow (all Err paths read
/// crate outputs).
#[test]
fn pl_error_contract() {
    let ra: Vec<usize> = vec![1, 0, 2, 2, 1, 0, 0, 1, 2, 1, 2, 0];
    let st: Vec<usize> = vec![0, 3, 6, 9, 12];
    assert!(lsr_rankings(&ra, &st, 1, 0.0).is_err(), "n < 2");
    // Allocation cap: dense O(n^2) chain must reject huge n with an Err,
    // never abort the process (impl-review finding 1 regression).
    assert!(
        lsr_rankings(&[0, 1], &[0, 2], 1_000_000, 0.0).is_err(),
        "n over dense-chain cap"
    );
    assert!(lsr_rankings(&ra, &[], 3, 0.0).is_err(), "empty starts");
    assert!(lsr_rankings(&ra, &[0], 3, 0.0).is_err(), "single start");
    assert!(
        lsr_rankings(&ra, &[1, 12], 3, 0.0).is_err(),
        "starts[0] != 0"
    );
    assert!(
        lsr_rankings(&ra, &[0, 3, 6, 9], 3, 0.0).is_err(),
        "bad tail"
    );
    assert!(
        lsr_rankings(&ra, &[0, 6, 3, 12], 3, 0.0).is_err(),
        "non-monotone starts"
    );
    assert!(
        lsr_rankings(&[0, 1, 2, 2], &[0, 3, 4], 3, 0.0).is_err(),
        "length-1 ranking (documented divergence: choix no-ops it)"
    );
    assert!(
        lsr_rankings(&[0, 1, 5], &[0, 3], 3, 0.0).is_err(),
        "item out of range"
    );
    assert!(
        lsr_rankings(&[0, 1, 0], &[0, 3], 3, 0.0).is_err(),
        "duplicate item (documented divergence: choix accepts if connected)"
    );
    // Same item in DIFFERENT rankings is fine.
    assert!(lsr_rankings(&[0, 1, 1, 2, 2, 0], &[0, 2, 4, 6], 3, 0.0).is_ok());
    assert!(lsr_rankings(&ra, &st, 3, -1.0).is_err(), "negative alpha");
    assert!(lsr_rankings(&ra, &st, 3, f64::NAN).is_err(), "NaN alpha");
    // Overflow: alpha = 1e308 makes off-diagonal row sums (2 * 1e308)
    // non-finite -> explicit Err, never NaN output.
    assert!(lsr_rankings(&ra, &st, 3, 1e308).is_err(), "alpha overflow");
    assert!(
        ilsr_rankings(&ra, &st, 3, 0.0, 0, 1e-8).is_err(),
        "max_iter = 0"
    );
    assert!(
        ilsr_rankings(&ra, &st, 3, 0.0, 100, 0.0).is_err(),
        "tol = 0"
    );
    assert!(
        ilsr_rankings(&ra, &st, 3, 0.0, 1, 1e-8).is_err(),
        "non-convergence in 1 pass (first check never fires)"
    );
    // Disconnected at alpha = 0: {0,1} vs {2,3}.
    let rd: Vec<usize> = vec![0, 1, 1, 0, 2, 3, 3, 2];
    let sd: Vec<usize> = vec![0, 2, 4, 6, 8];
    assert!(lsr_rankings(&rd, &sd, 4, 0.0).is_err(), "disconnected");
    assert!(ilsr_rankings(&rd, &sd, 4, 0.0, 100, 1e-8).is_err());
    // ... estimable at alpha = 1/2, exactly uniform (oracle pin).
    let rd5 = lsr_rankings(&rd, &sd, 4, 0.5).unwrap();
    for w in rd5.weights.iter() {
        assert!((w - 1.0).abs() < 1e-14, "RD alpha=1/2 uniform");
    }
}

/// 500-replication Monte-Carlo Plackett-Luce recovery (ignored: slow).
/// True params centered before MAE per spec review.
#[test]
#[ignore]
fn pl_mc_500_recovery() {
    let n = 6usize;
    let mut rng = Lcg(0x9e3779b97f4a7c15);
    let mut worst = 0.0f64;
    for rep in 0..500 {
        let truth: Vec<f64> = {
            let raw: Vec<f64> = (0..n).map(|_| 0.8 * rng.normal()).collect();
            let m = raw.iter().sum::<f64>() / n as f64;
            raw.iter().map(|x| x - m).collect()
        };
        let worths: Vec<f64> = truth.iter().map(|t| t.exp()).collect();
        // 300 full rankings via sequential Luce sampling: categorical
        // draw u = next_f64() * remaining_sum with fallthrough (spec
        // review change 6).
        let mut rankings: Vec<usize> = Vec::with_capacity(300 * n);
        let mut starts: Vec<usize> = Vec::with_capacity(301);
        starts.push(0);
        for _ in 0..300 {
            let mut remaining: Vec<usize> = (0..n).collect();
            while !remaining.is_empty() {
                let s: f64 = remaining.iter().map(|&i| worths[i]).sum();
                let u = rng.next_f64() * s;
                let mut acc = 0.0;
                let mut pick = remaining.len() - 1;
                for (k, &i) in remaining.iter().enumerate() {
                    acc += worths[i];
                    if u < acc {
                        pick = k;
                        break;
                    }
                }
                rankings.push(remaining.remove(pick));
            }
            starts.push(rankings.len());
        }
        let r = ilsr_rankings(&rankings, &starts, n, 0.0, 200, 1e-8).unwrap();
        let mae = truth
            .iter()
            .zip(r.params.iter())
            .map(|(t, p)| (t - p).abs())
            .sum::<f64>()
            / n as f64;
        worst = worst.max(mae);
        // 0.2 bound: worst-of-500 MAE MEASURED at 0.1440 on this seed
        // stream; mutant deviations (MU4/MU5) exceed 0.3.
        assert!(mae < 0.2, "rep {rep}: mae = {mae}");
    }
    assert!(worst > 0.0);
}

// ---------------------------------------------------------------------------
// Plackett-Luce top-1 LSR / I-LSR (choix 0.4.1 lsr_top1 / ilsr_top1).
// Pins from the EXECUTED exact-Fraction/mpmath oracle (session files
// top1_oracle.py), cross-checked against pip choix 0.4.1 (<= 1.2e-16).
//
// Mutation kills (all EXECUTED):
// - MU1 transpose chain update (winner->loser edge): killed by
//   t1_fixture_ta_exact (item-1 dominance flips).
// - MU2 winner-EXCLUDED denominator (val = 1/sum(w[losers])): PROVEN
//   INVISIBLE on TA — full equal-size choice sets make it a constant
//   off-diagonal rescale, statdist-invariant (oracle separation probe:
//   TA False, TB True). Killed ONLY by t1_fixture_tb_partial.
// - MU3 all-items denominator (val = 1/sum(all w)): same TA blindness
//   (oracle-proven); killed by t1_fixture_tb_partial.
// - MU4 I-LSR feeds uniform worths each pass: INVISIBLE on TA (the
//   one-shot IS the fixed point there); killed by t1_ilsr_fixed_points
//   via the TB params pin and the TB iterations == 12 pin.
// - MU5 diagonal/alpha regularization mistakes: killed by the
//   t1_fixture_ta_exact alpha = 1/2 pins [15/19, 27/19, 15/19].
// ---------------------------------------------------------------------------

/// Fixture TA: n = 3, five FULL choice sets
/// [(1,[0,2]), (1,[0,2]), (0,[1,2]), (2,[0,1]), (1,[0,2])].
/// Oracle weights exactly [3/5, 9/5, 3/5]; alpha = 1/2 weights
/// [15/19, 27/19, 15/19]. Dataset duplication is exactly invariant at
/// alpha = 0, NOT at alpha = 1/2 (both crate-output pins).
#[test]
fn t1_fixture_ta_exact() {
    let wn: Vec<usize> = vec![1, 1, 0, 2, 1];
    let ls: Vec<usize> = vec![0, 2, 0, 2, 1, 2, 0, 1, 0, 2];
    let st: Vec<usize> = vec![0, 2, 4, 6, 8, 10];
    let r = lsr_top1(&wn, &ls, &st, 3, 0.0).unwrap();
    assert_eq!(r.iterations, 1);
    let expw = [0.6, 1.8, 0.6];
    for (a, b) in r.weights.iter().zip(expw.iter()) {
        assert!((a - b).abs() < 1e-14, "weights {:?}", r.weights);
    }
    let expp = [
        -0.36620409622270323047,
        0.73240819244540646093,
        -0.36620409622270323047,
    ];
    for (a, b) in r.params.iter().zip(expp.iter()) {
        assert!((a - b).abs() < 1e-14, "params {:?}", r.params);
    }
    let r5 = lsr_top1(&wn, &ls, &st, 3, 0.5).unwrap();
    let expw5 = [15.0 / 19.0, 27.0 / 19.0, 15.0 / 19.0];
    for (a, b) in r5.weights.iter().zip(expw5.iter()) {
        assert!((a - b).abs() < 1e-14, "alpha=1/2 weights {:?}", r5.weights);
    }
    // Duplication: exactly invariant at alpha = 0 ...
    let mut wn2 = wn.clone();
    wn2.extend_from_slice(&wn);
    let mut ls2 = ls.clone();
    ls2.extend_from_slice(&ls);
    let st2: Vec<usize> = vec![0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20];
    let rd = lsr_top1(&wn2, &ls2, &st2, 3, 0.0).unwrap();
    for (a, b) in rd.weights.iter().zip(r.weights.iter()) {
        assert!((a - b).abs() < 1e-15, "duplication must be invariant");
    }
    // ... but NOT at alpha = 1/2.
    let rd5 = lsr_top1(&wn2, &ls2, &st2, 3, 0.5).unwrap();
    let diff: f64 = rd5
        .weights
        .iter()
        .zip(r5.weights.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-3,
        "alpha > 0 duplication must differ, diff = {diff}"
    );
}

/// Fixture TB: n = 4, six PARTIAL choice sets (sizes 3/2/2/3/3/2)
/// [(0,[1,2]), (3,[2]), (1,[3]), (2,[0,3]), (3,[1,0]), (1,[2])].
/// Oracle weights exactly [38/41, 54/41, 22/41, 50/41] — the
/// discriminating anchor for the winner-excluded (MU2) and all-items
/// (MU3) denominator mutants, both PROVEN invisible on TA.
#[test]
fn t1_fixture_tb_partial() {
    let wn: Vec<usize> = vec![0, 3, 1, 2, 3, 1];
    let ls: Vec<usize> = vec![1, 2, 2, 3, 0, 3, 1, 0, 2];
    let st: Vec<usize> = vec![0, 2, 3, 4, 6, 8, 9];
    let r = lsr_top1(&wn, &ls, &st, 4, 0.0).unwrap();
    let expw = [38.0 / 41.0, 54.0 / 41.0, 22.0 / 41.0, 50.0 / 41.0];
    for (a, b) in r.weights.iter().zip(expw.iter()) {
        assert!((a - b).abs() < 1e-14, "weights {:?}", r.weights);
    }
    let expp = [
        -0.019822756542894746856,
        0.33157513029499386732,
        -0.5663664629109646628,
        0.25461408915886554234,
    ];
    for (a, b) in r.params.iter().zip(expp.iter()) {
        assert!((a - b).abs() < 1e-14, "params {:?}", r.params);
    }
}

/// I-LSR fixed points (mpmath dps=50 oracle). TA converges in 2 passes
/// with params equal to the one-shot — TA IS an exact fixed point
/// (tolerance 1e-14, NOT bit equality per spec review). TB converges in
/// 12 passes; atol 1e-7 is the ORACLE-MEASURED margin (converged
/// tol=1e-8 iterate sits ~1e-9 from the fixed point; the MU4
/// uniform-feed mutant deviates >= 1e-2 and would report 2 passes).
/// Also pins the returned-weights invariant
/// weights == exp_transform(params) (kills stale/uniform-weights
/// mutants that params-only pins miss).
#[test]
fn t1_ilsr_fixed_points() {
    let wna: Vec<usize> = vec![1, 1, 0, 2, 1];
    let lsa: Vec<usize> = vec![0, 2, 0, 2, 1, 2, 0, 1, 0, 2];
    let sta: Vec<usize> = vec![0, 2, 4, 6, 8, 10];
    let one = lsr_top1(&wna, &lsa, &sta, 3, 0.0).unwrap();
    let r = ilsr_top1(&wna, &lsa, &sta, 3, 0.0, 100, 1e-8).unwrap();
    assert_eq!(r.iterations, 2, "TA pass count at tol=1e-8");
    for (a, b) in r.params.iter().zip(one.params.iter()) {
        assert!((a - b).abs() < 1e-14, "TA ilsr == one-shot fixed point");
    }
    let wnb: Vec<usize> = vec![0, 3, 1, 2, 3, 1];
    let lsb: Vec<usize> = vec![1, 2, 2, 3, 0, 3, 1, 0, 2];
    let stb: Vec<usize> = vec![0, 2, 3, 4, 6, 8, 9];
    let rr = ilsr_top1(&wnb, &lsb, &stb, 4, 0.0, 100, 1e-8).unwrap();
    let expb = [
        0.038399971408422690899,
        0.26272439694352421265,
        -0.56384876453083402307,
        0.26272439617888711952,
    ];
    for (a, b) in rr.params.iter().zip(expb.iter()) {
        assert!((a - b).abs() < 1e-7, "TB ilsr params {:?}", rr.params);
    }
    assert_eq!(rr.iterations, 12, "TB pass count at tol=1e-8");
    // Returned-weights invariant: weights == exp_transform(params),
    // positive, sum n (reads BOTH crate outputs).
    for res in [&r, &rr] {
        let n = res.params.len() as f64;
        let mean = res.params.iter().sum::<f64>() / n;
        let mut w: Vec<f64> = res.params.iter().map(|p| (p - mean).exp()).collect();
        let s: f64 = w.iter().sum();
        for x in w.iter_mut() {
            *x *= n / s;
        }
        for (a, b) in res.weights.iter().zip(w.iter()) {
            assert!(*a > 0.0);
            assert!((a - b).abs() < 1e-12, "weights/params invariant");
        }
        let sw: f64 = res.weights.iter().sum();
        assert!((sw - n).abs() < 1e-9);
    }
}

/// Single-loser observations are EXACTLY pairwise comparisons (uniform
/// worths: val = 1/2 on the loser->winner edge, by construction), so
/// lsr_top1 must bit-match lsr_pairwise on the induced win matrix
/// (oracle probe maxdiff 0.0).
#[test]
fn t1_pairwise_equivalence() {
    // 0>1, 1>2, 2>0, 0>2 as top-1 observations with one loser each.
    let wn: Vec<usize> = vec![0, 1, 2, 0];
    let ls: Vec<usize> = vec![1, 2, 0, 2];
    let st: Vec<usize> = vec![0, 1, 2, 3, 4];
    let r = lsr_top1(&wn, &ls, &st, 3, 0.0).unwrap();
    let mut wins = vec![0.0f64; 9];
    wins[0 * 3 + 1] = 1.0;
    wins[1 * 3 + 2] = 1.0;
    wins[2 * 3 + 0] = 1.0;
    wins[0 * 3 + 2] = 1.0;
    let rp = lsr_pairwise(&wins, 3, 0.0).unwrap();
    assert_eq!(r.params, rp.params, "single-loser top-1 == pairwise");
    assert_eq!(r.weights, rp.weights);
}

/// Error contract + disconnected graph + overflow (all Err paths read
/// crate outputs).
#[test]
fn t1_error_contract() {
    let wn: Vec<usize> = vec![1, 1, 0, 2, 1];
    let ls: Vec<usize> = vec![0, 2, 0, 2, 1, 2, 0, 1, 0, 2];
    let st: Vec<usize> = vec![0, 2, 4, 6, 8, 10];
    assert!(lsr_top1(&wn, &ls, &st, 1, 0.0).is_err(), "n < 2");
    // Allocation cap: dense O(n^2) chain must reject huge n with an Err,
    // never abort the process.
    assert!(
        lsr_top1(&[0], &[1], &[0, 1], 1_000_000, 0.0).is_err(),
        "n over dense-chain cap"
    );
    assert!(lsr_top1(&wn, &ls, &[], 3, 0.0).is_err(), "empty starts");
    assert!(lsr_top1(&[], &[], &[0], 3, 0.0).is_err(), "single start");
    assert!(
        lsr_top1(&[0], &ls, &st, 3, 0.0).is_err(),
        "winners/starts length mismatch"
    );
    assert!(
        lsr_top1(&wn, &ls, &[1, 2, 4, 6, 8, 10], 3, 0.0).is_err(),
        "starts[0] != 0"
    );
    assert!(
        lsr_top1(&wn, &ls, &[0, 2, 4, 6, 8, 9], 3, 0.0).is_err(),
        "bad tail"
    );
    assert!(
        lsr_top1(&wn, &ls, &[0, 4, 2, 6, 8, 10], 3, 0.0).is_err(),
        "non-monotone starts"
    );
    assert!(
        lsr_top1(&[0, 1], &[1], &[0, 1, 1], 3, 0.0).is_err(),
        "empty loser set (documented divergence: choix no-ops it)"
    );
    assert!(
        lsr_top1(&[0], &[1, 5], &[0, 2], 3, 0.0).is_err(),
        "loser out of range"
    );
    assert!(
        lsr_top1(&[5], &[0, 1], &[0, 2], 3, 0.0).is_err(),
        "winner out of range"
    );
    assert!(
        lsr_top1(&[0], &[1, 0], &[0, 2], 3, 0.0).is_err(),
        "winner in its own loser set (documented divergence)"
    );
    assert!(
        lsr_top1(&[0], &[1, 1], &[0, 2], 3, 0.0).is_err(),
        "duplicate loser (documented divergence)"
    );
    // The same loser in DIFFERENT observations is fine.
    assert!(lsr_top1(&[0, 1, 2], &[1, 2, 0], &[0, 1, 2, 3], 3, 0.0).is_ok());
    assert!(lsr_top1(&wn, &ls, &st, 3, -1.0).is_err(), "negative alpha");
    assert!(lsr_top1(&wn, &ls, &st, 3, f64::NAN).is_err(), "NaN alpha");
    // Overflow: alpha = 1e308 makes off-diagonal row sums (2 * 1e308)
    // non-finite -> explicit Err, never NaN output.
    assert!(lsr_top1(&wn, &ls, &st, 3, 1e308).is_err(), "alpha overflow");
    assert!(
        ilsr_top1(&wn, &ls, &st, 3, 0.0, 0, 1e-8).is_err(),
        "max_iter = 0"
    );
    assert!(
        ilsr_top1(&wn, &ls, &st, 3, 0.0, 100, 0.0).is_err(),
        "tol = 0"
    );
    assert!(
        ilsr_top1(&wn, &ls, &st, 3, 0.0, 1, 1e-8).is_err(),
        "non-convergence in 1 pass (first check never fires)"
    );
    // Disconnected at alpha = 0: {0,1} vs {2,3} (fixture TD).
    let wd: Vec<usize> = vec![0, 1, 2, 3];
    let ld: Vec<usize> = vec![1, 0, 3, 2];
    let sd: Vec<usize> = vec![0, 1, 2, 3, 4];
    assert!(lsr_top1(&wd, &ld, &sd, 4, 0.0).is_err(), "disconnected");
    assert!(ilsr_top1(&wd, &ld, &sd, 4, 0.0, 100, 1e-8).is_err());
    // ... estimable at alpha = 1/2, exactly uniform (oracle pin).
    let rd5 = lsr_top1(&wd, &ld, &sd, 4, 0.5).unwrap();
    for w in rd5.weights.iter() {
        assert!((w - 1.0).abs() < 1e-14, "TD alpha=1/2 uniform");
    }
}

/// 500-replication Monte-Carlo top-1 recovery (ignored: slow).
/// Fully specified per spec review: seed 0x51a3c2b7d4e8f901 (crate Lcg),
/// n = 6, true params 0.8 * N(0,1) centered, 800 observations per rep,
/// each with a choice set of size 3..=6 (uniform via next_f64, sampled
/// without replacement) and the winner drawn Luce-categorically within
/// the set; estimator ilsr_top1(alpha=0, max_iter=200, tol=1e-8);
/// metric MAE(params, truth).
#[test]
#[ignore]
fn t1_mc_500_recovery() {
    let n = 6usize;
    let mut rng = Lcg(0x51a3c2b7d4e8f901);
    let mut worst = 0.0f64;
    for rep in 0..500 {
        let truth: Vec<f64> = {
            let raw: Vec<f64> = (0..n).map(|_| 0.8 * rng.normal()).collect();
            let m = raw.iter().sum::<f64>() / n as f64;
            raw.iter().map(|x| x - m).collect()
        };
        let worths: Vec<f64> = truth.iter().map(|t| t.exp()).collect();
        let mut winners: Vec<usize> = Vec::with_capacity(800);
        let mut losers: Vec<usize> = Vec::with_capacity(800 * n);
        let mut starts: Vec<usize> = Vec::with_capacity(801);
        starts.push(0);
        for _ in 0..800 {
            // Choice-set size uniform on 3..=n, sampled without
            // replacement from 0..n.
            let k = 3 + (rng.next_f64() * ((n - 2) as f64)) as usize;
            let k = k.min(n);
            let mut pool: Vec<usize> = (0..n).collect();
            let mut set: Vec<usize> = Vec::with_capacity(k);
            for _ in 0..k {
                let idx = (rng.next_f64() * (pool.len() as f64)) as usize;
                set.push(pool.remove(idx.min(pool.len() - 1)));
            }
            // Luce-categorical winner draw within the set.
            let s: f64 = set.iter().map(|&i| worths[i]).sum();
            let u = rng.next_f64() * s;
            let mut acc = 0.0;
            let mut pick = set.len() - 1;
            for (j, &i) in set.iter().enumerate() {
                acc += worths[i];
                if u < acc {
                    pick = j;
                    break;
                }
            }
            winners.push(set[pick]);
            for (j, &i) in set.iter().enumerate() {
                if j != pick {
                    losers.push(i);
                }
            }
            starts.push(losers.len());
        }
        let r = ilsr_top1(&winners, &losers, &starts, n, 0.0, 200, 1e-8).unwrap();
        let mae = truth
            .iter()
            .zip(r.params.iter())
            .map(|(t, p)| (t - p).abs())
            .sum::<f64>()
            / n as f64;
        worst = worst.max(mae);
        // 0.3 bound: worst-of-500 MAE MEASURED at 0.2580 on this seed
        // stream (800 obs/rep); mutant deviations (MU4 uniform feed)
        // exceed 0.4 on partial choice sets.
        assert!(mae < 0.3, "rep {rep}: mae = {mae}");
    }
    assert!(worst > 0.0);
}
// ---------------------------------------------------------------------
// Kendall & Babington Smith (1940) circular triads / agreement (kd_).
//
// Oracle: exact-Fraction recomputation of eba circular()/kendall.u()
// (session artifact kendall_oracle.py, EXECUTED). Every assert below
// reads crate outputs (CircularResult / KendallUResult fields).
//
// Mutation kills (all EXECUTED against these tests):
// - MU1 T pairing dropped (sum C(d,2) -> sum d): kd_dog_exact (T pin 5).
// - MU2 T_max parity swap (odd<->even formulas): kd_dog_exact
//   (t_max 8 -> 8.75), kd_transitive_two_sided (n=5).
// - MU3 two-sided opposite-tail accumulation dropped (p = p1 only):
//   kd_transitive_two_sided (9/64 vs own tail 15/128).
// - MU4 chi2 continuity-correction sign flip: kd_chi2_n12
//   (less 121/8 -> 129/8).
// - MU5 kendall_u correction dropped from Sigma (Sigma - corr ->
//   Sigma): kd_u_fixture_a (chi2 11 -> 13 with correct=true).
// ---------------------------------------------------------------------

/// Dog food example, Kendall & Babington Smith (1940, p. 326) via eba
/// man/circular.Rd. Reads: t, t_max, t_exp, zeta, p_value, chi2, df,
/// exact. Kills MU1 (T pin), MU2 (t_max pin).
#[test]
fn kd_dog_exact() {
    #[rustfmt::skip]
    let dog = [
        0., 1., 1., 0., 1., 1.,
        0., 0., 0., 1., 1., 0.,
        0., 1., 0., 1., 1., 1.,
        1., 0., 0., 0., 0., 0.,
        0., 0., 0., 1., 0., 1.,
        0., 1., 0., 1., 0., 0.,
    ];
    let r = super::circular_triads(&dog, 6, "less", true).unwrap();
    assert_eq!(r.t, 5.0);
    assert_eq!(r.t_max, 8.0);
    assert_eq!(r.t_exp, 5.0);
    assert_eq!(r.zeta, 3.0 / 8.0); // 0.375 exact
    assert!(r.exact);
    // Exact dyadic tails (oracle Fractions; exactly representable).
    assert_eq!(r.p_value, 1043.0 / 2048.0);
    assert!(r.chi2.is_nan() && r.df.is_nan());
    let rg = super::circular_triads(&dog, 6, "greater", true).unwrap();
    assert_eq!(rg.p_value, 1233.0 / 2048.0);
    let rt = super::circular_triads(&dog, 6, "two.sided", true).unwrap();
    assert_eq!(rt.p_value, 1.0);
}

/// Fully transitive n=5 tournament: T=0, zeta=1. The two-sided p
/// (9/64) exceeds the own lower tail (15/128) by the accumulated
/// far-tail atom 3/128 ? pins the opposite-tail accumulation (MU3)
/// and the parity branch of T_max (MU2, n odd).
#[test]
fn kd_transitive_two_sided() {
    let n = 5;
    let mut mat = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            mat[i * n + j] = 1.0;
        }
    }
    let r = super::circular_triads(&mat, n, "less", true).unwrap();
    assert_eq!(r.t, 0.0);
    assert_eq!(r.t_max, 5.0);
    assert_eq!(r.zeta, 1.0);
    assert_eq!(r.p_value, 15.0 / 128.0);
    let rt = super::circular_triads(&mat, n, "two.sided", true).unwrap();
    assert_eq!(rt.p_value, 9.0 / 64.0);
}

/// n=12 deterministic tournament ((i+j)%3 pattern): chi-square path.
/// Exact rational pins from the oracle; p-values vs scipy.stats.chi2
/// references. Kills MU4 (correction sign: less 121/8 vs 129/8).
#[test]
fn kd_chi2_n12() {
    let n = 12;
    let mut mat = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            if (i + j) % 3 != 0 {
                mat[i * n + j] = 1.0;
            } else {
                mat[j * n + i] = 1.0;
            }
        }
    }
    let r = super::circular_triads(&mat, n, "less", true).unwrap();
    assert!(!r.exact);
    assert_eq!(r.t, 60.0);
    assert_eq!(r.t_max, 70.0);
    assert_eq!(r.t_exp, 55.0);
    assert!((r.zeta - 1.0 / 7.0).abs() < 1e-15);
    assert_eq!(r.chi2, 121.0 / 8.0); // 15.125 exact
    assert_eq!(r.df, 165.0 / 8.0); // 20.625 exact
                                   // scipy.stats.chi2.sf(121/8, 165/8) = 0.7996995989775597
    assert!((r.p_value - 0.7996995989775597).abs() < 1e-9);
    let rg = super::circular_triads(&mat, n, "greater", true).unwrap();
    assert_eq!(rg.chi2, 129.0 / 8.0);
    // scipy chi2.cdf(129/8, 165/8) = 0.2568016649115205
    assert!((rg.p_value - 0.2568016649115205).abs() < 1e-9);
    let rt = super::circular_triads(&mat, n, "two.sided", true).unwrap();
    assert_eq!(rt.chi2, 129.0 / 8.0);
    // 2*min(sf, cdf) = 0.513603329823041
    assert!((rt.p_value - 0.513603329823041).abs() < 1e-9);
    let rnc = super::circular_triads(&mat, n, "less", false).unwrap();
    assert_eq!(rnc.chi2, 125.0 / 8.0); // no continuity correction
}

/// Integrity of the exact tables copied from eba circular.R: each row
/// n sums to 2^C(n,2) (all orientations of the complete graph K_n).
/// Reads the crate const CIRCULAR_EXACT directly.
#[test]
fn kd_table_integrity() {
    for (idx, row) in super::CIRCULAR_EXACT.iter().enumerate() {
        let n = idx as u32 + 2;
        let c_n2 = n * (n - 1) / 2;
        let total: u64 = row.iter().sum();
        assert_eq!(total, 1u64 << c_n2, "n = {n}");
        // Row length spans T = 0..=T_max.
        let t_max = if n % 2 == 1 {
            n as u64 * (n as u64 * n as u64 - 1) / 24
        } else {
            n as u64 * (n as u64 * n as u64 - 4) / 24
        };
        assert_eq!(row.len() as u64, t_max + 1, "n = {n}");
    }
}

/// Error contract for circular_triads. Every arm reads the crate Err.
#[test]
fn kd_circular_error_contract() {
    let ok3 = [0., 1., 1., 0., 0., 1., 0., 0., 0.];
    assert!(super::circular_triads(&ok3, 3, "two.sided", true).is_ok());
    // n = 2 rejected (documented divergence: eba yields zeta = NaN).
    assert!(super::circular_triads(&[0., 1., 0., 0.], 2, "less", true).is_err());
    // bad alternative string
    assert!(super::circular_triads(&ok3, 3, "both", true).is_err());
    // wrong length
    assert!(super::circular_triads(&ok3[..8], 3, "less", true).is_err());
    // nonzero diagonal rejected (eba silently zeroes)
    let mut bad = ok3;
    bad[0] = 1.0;
    assert!(super::circular_triads(&bad, 3, "less", true).is_err());
    // non-binary entry
    let mut bad = ok3;
    bad[1] = 0.5;
    assert!(super::circular_triads(&bad, 3, "less", true).is_err());
    // incomplete pair (both zero)
    let mut bad = ok3;
    bad[1] = 0.0;
    assert!(super::circular_triads(&bad, 3, "less", true).is_err());
    // both-one pair
    let mut bad = ok3;
    bad[3] = 1.0;
    assert!(super::circular_triads(&bad, 3, "less", true).is_err());
    // NaN entry
    let mut bad = ok3;
    bad[1] = f64::NAN;
    assert!(super::circular_triads(&bad, 3, "less", true).is_err());
    // n cap (rejected before allocating/reading n*n)
    assert!(super::circular_triads(&ok3, 10_001, "less", true).is_err());
}

/// Kendall u fixture A (m=4 judges, n=3): Sigma=11, u=2/9, min_u=-1/3,
/// chi2=11, df=9 (all exact); no-correction chi2=13. Kills MU5.
#[test]
fn kd_u_fixture_a() {
    let m = [0., 3., 4., 1., 0., 2., 0., 2., 0.];
    let r = super::kendall_u(&m, 3, true).unwrap();
    assert_eq!(r.sigma, 11.0);
    assert!((r.u - 2.0 / 9.0).abs() < 1e-15);
    assert!((r.min_u + 1.0 / 3.0).abs() < 1e-15);
    assert_eq!(r.chi2, 11.0);
    assert_eq!(r.df, 9.0);
    // scipy chi2.sf(11, 9) = 0.27570893677222197
    assert!((r.p_value - 0.27570893677222197).abs() < 1e-9);
    let rnc = super::kendall_u(&m, 3, false).unwrap();
    assert_eq!(rnc.chi2, 13.0);
}

/// Perfect agreement, m=5 (odd), n=4: u=1, min_u=-1/5, chi2=52,
/// df=40/3.
#[test]
fn kd_u_perfect() {
    let n = 4;
    let mut mat = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            mat[i * n + j] = 5.0;
        }
    }
    let r = super::kendall_u(&mat, n, true).unwrap();
    assert_eq!(r.sigma, 60.0);
    assert_eq!(r.u, 1.0);
    assert_eq!(r.min_u, -0.2);
    assert_eq!(r.chi2, 52.0);
    assert!((r.df - 40.0 / 3.0).abs() < 1e-12);
    // scipy chi2.sf(52, 40/3) = 1.7298363037807228e-06
    assert!((r.p_value - 1.7298363037807228e-06).abs() < 1e-12);
}

/// m=3 boundary ((m-2)^2 = 1 denominators): Sigma=5, u=1/9, chi2=16,
/// df=18; and the m-3 factor zeroes the centering term.
#[test]
fn kd_u_m3_boundary() {
    let m = [0., 2., 3., 1., 0., 1., 0., 2., 0.];
    let r = super::kendall_u(&m, 3, true).unwrap();
    assert_eq!(r.sigma, 5.0);
    assert!((r.u - 1.0 / 9.0).abs() < 1e-15);
    assert!((r.min_u + 1.0 / 3.0).abs() < 1e-15);
    assert_eq!(r.chi2, 16.0);
    assert_eq!(r.df, 18.0);
    // scipy chi2.sf(16, 18) = 0.5925473414375915
    assert!((r.p_value - 0.5925473414375915).abs() < 1e-9);
}

/// Strong disagreement (n=2, m=4 split 2-2): raw chi2 stays negative
/// (-1) while the p-value clamps to 1 (R pchisq semantics). Also pins
/// u == min_u at maximal disagreement for even m.
#[test]
fn kd_u_negative_chi2_raw() {
    let m = [0., 2., 2., 0.];
    let r = super::kendall_u(&m, 2, true).unwrap();
    assert_eq!(r.sigma, 2.0);
    assert!((r.u + 1.0 / 3.0).abs() < 1e-15);
    assert!((r.min_u + 1.0 / 3.0).abs() < 1e-15);
    assert_eq!(r.chi2, -1.0); // raw, NOT clamped
    assert_eq!(r.df, 3.0);
    assert_eq!(r.p_value, 1.0); // clamped inside chi2_sf
}

/// Error contract for kendall_u.
#[test]
fn kd_u_error_contract() {
    let ok = [0., 3., 4., 1., 0., 2., 0., 2., 0.];
    assert!(super::kendall_u(&ok, 3, true).is_ok());
    // n = 1: no pairs
    assert!(super::kendall_u(&[0.0], 1, true).is_err());
    // m = 2 (< 3 judges)
    assert!(super::kendall_u(&[0., 1., 1., 0.], 2, true).is_err());
    // unequal observations per pair (eba would silently use pair (0,1))
    let mut bad = ok;
    bad[2] = 5.0;
    assert!(super::kendall_u(&bad, 3, true).is_err());
    // non-integral entry
    let mut bad = ok;
    bad[1] = 2.5;
    assert!(super::kendall_u(&bad, 3, true).is_err());
    // negative entry
    let mut bad = ok;
    bad[1] = -1.0;
    assert!(super::kendall_u(&bad, 3, true).is_err());
    // nonzero diagonal
    let mut bad = ok;
    bad[0] = 1.0;
    assert!(super::kendall_u(&bad, 3, true).is_err());
    // NaN entry
    let mut bad = ok;
    bad[1] = f64::NAN;
    assert!(super::kendall_u(&bad, 3, true).is_err());
    // judge cap
    let mut bad = ok;
    bad[1] = 2_000_000.0;
    assert!(super::kendall_u(&bad, 3, true).is_err());
    // wrong length
    assert!(super::kendall_u(&ok[..8], 3, true).is_err());
    // n cap
    assert!(super::kendall_u(&ok, 10_001, true).is_err());
}

/// Seeded 500-rep invariant smoke test (fair-coin tournaments, n=6):
/// T integral in [0, T_max], zeta <= 1, exact p in [0, 1]; one
/// noiseless transitive rep pins T = 0. Fixed assertions, no
/// stochastic thresholds.
#[test]
#[ignore]
fn kd_mc_500_invariants() {
    let n = 6usize;
    let mut rng = Lcg(0x6b3a91c44f27e015);
    for rep in 0..500 {
        let mut mat = vec![0.0f64; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                if rng.next_f64() < 0.5 {
                    mat[i * n + j] = 1.0;
                } else {
                    mat[j * n + i] = 1.0;
                }
            }
        }
        let r = super::circular_triads(&mat, n, "two.sided", true).unwrap();
        assert!(r.exact, "rep {rep}");
        assert_eq!(r.t.fract(), 0.0, "rep {rep}: T must be integral");
        assert!(r.t >= 0.0 && r.t <= r.t_max, "rep {rep}");
        assert!(r.zeta <= 1.0, "rep {rep}");
        assert!((0.0..=1.0).contains(&r.p_value), "rep {rep}");
    }
    let mut mat = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            mat[i * n + j] = 1.0;
        }
    }
    let r = super::circular_triads(&mat, n, "less", true).unwrap();
    assert_eq!(r.t, 0.0);
    assert_eq!(r.zeta, 1.0);
}

// ---------------------------------------------------------------------------
// elo_rating tests (fixtures EA/EB/EC/ED and invariants from the executed
// exact-rational oracle; see the citation-governance header in scaling.rs).
// Every assertion below reads values returned by the crate's elo_rating.
// ---------------------------------------------------------------------------

/// EA: single period, exact-rational anchor. Kills MU2 (black actual-score
/// flip: ratings[1] would be 2016, not 1984), MU3 (gamma sign flip in the
/// white exponent: the gamma=400 draw moves E_w 10/11 -> 1/11, changing
/// ratings[0] and ratings[2]), and MU5 (divisor 400 -> 800).
#[test]
fn el_anchor_ea_exact() {
    let r = super::elo_rating(
        &[1, 1],
        &[0, 0],
        &[1, 2],
        &[1.0, 0.5],
        &[0.0, 400.0],
        3,
        2000.0,
        32.0,
    )
    .unwrap();
    // 22032/11 and 22144/11 are not dyadic; 1984 is exact in f64.
    assert!((r.ratings[0] - 22032.0 / 11.0).abs() < 1e-12);
    assert_eq!(r.ratings[1], 1984.0);
    assert!((r.ratings[2] - 22144.0 / 11.0).abs() < 1e-12);
    assert_eq!(r.games, vec![2, 1, 1]);
    assert_eq!(r.wins, vec![1, 0, 0]);
    assert_eq!(r.draws, vec![1, 0, 1]);
    assert_eq!(r.losses, vec![0, 1, 0]);
    assert_eq!(r.lag, vec![0, 0, 0]);
}

/// EB: two periods, kfac=400 — the batch-semantics proof. A sequential
/// per-game update (MU1) drifts ratings[0]/ratings[1] off these pins; a
/// dropped lag reset (MU4) turns lag into [2,2,2].
#[test]
fn el_anchor_eb_batch() {
    let r = super::elo_rating(
        &[1, 1, 2],
        &[0, 0, 0],
        &[1, 2, 1],
        &[1.0, 0.5, 1.0],
        &[0.0, 0.0, 0.0],
        3,
        2000.0,
        400.0,
    )
    .unwrap();
    assert!((r.ratings[0] - 24600.0 / 11.0).abs() < 1e-12);
    assert!((r.ratings[1] - 19400.0 / 11.0).abs() < 1e-12);
    assert_eq!(r.ratings[2], 2000.0); // player 2's dscore is exactly 0
    assert_eq!(r.games, vec![3, 2, 1]);
    assert_eq!(r.wins, vec![2, 0, 0]);
    assert_eq!(r.draws, vec![1, 0, 1]);
    assert_eq!(r.losses, vec![0, 2, 0]);
    assert_eq!(r.lag, vec![0, 0, 1]);
}

/// EC: float reference, 4 players / 3 periods / PlayerRatings defaults
/// (init 2200, kfac 27), generic non-multiple-of-400 diffs after period 1.
#[test]
fn el_anchor_ec_float() {
    let r = super::elo_rating(
        &[1, 1, 1, 2, 2, 3, 3],
        &[0, 2, 0, 1, 0, 2, 1],
        &[1, 3, 2, 3, 3, 0, 2],
        &[1.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0],
        &[0.0; 7],
        4,
        2200.0,
        27.0,
    )
    .unwrap();
    let exp = [
        2213.5,
        2187.528245191514,
        2226.49604864222,
        2172.475706166266,
    ];
    for p in 0..4 {
        assert!(
            (r.ratings[p] - exp[p]).abs() < 1e-9,
            "player {p}: {} vs {}",
            r.ratings[p],
            exp[p]
        );
    }
    assert_eq!(r.games, vec![4, 3, 4, 3]);
    assert_eq!(r.wins, vec![2, 1, 2, 0]);
    assert_eq!(r.draws, vec![1, 0, 2, 1]);
    assert_eq!(r.losses, vec![1, 2, 0, 2]);
    assert_eq!(r.lag, vec![0, 0, 0, 1]);
}

/// ED: closed-form nonzero-gamma single game. These pins differ from the
/// gamma=0 values (2016 / 1984), which is what kills a gamma-drop mutant.
/// NOTE: E_w + E_b = 1 identically (the exponents are exact negations even
/// with gamma != 0), so a mutant computing E_b = 1 - E_w is a behavioral
/// no-op and is documented as unkillable by design.
#[test]
fn el_anchor_ed_gamma() {
    let r = super::elo_rating(&[1], &[0], &[1], &[1.0], &[100.0], 2, 2000.0, 32.0).unwrap();
    assert!((r.ratings[0] - 2011.5179200063076).abs() < 1e-12);
    assert!((r.ratings[1] - 1988.4820799936924).abs() < 1e-12);
    // Discriminating check vs gamma=0: the same schedule without gamma
    // gives exactly 2016 / 1984.
    let r0 = super::elo_rating(&[1], &[0], &[1], &[1.0], &[0.0], 2, 2000.0, 32.0).unwrap();
    assert_eq!(r0.ratings[0], 2016.0);
    assert_eq!(r0.ratings[1], 1984.0);
    assert!((r.ratings[0] - r0.ratings[0]).abs() > 1.0);
}

/// kfac = 0: ratings stay exactly at init while all bookkeeping updates.
#[test]
fn el_kfac_zero() {
    let r = super::elo_rating(
        &[1, 2],
        &[0, 1],
        &[1, 0],
        &[1.0, 0.5],
        &[0.0, 0.0],
        3,
        2200.0,
        0.0,
    )
    .unwrap();
    assert_eq!(r.ratings, vec![2200.0, 2200.0, 2200.0]);
    assert_eq!(r.games, vec![2, 2, 0]);
    assert_eq!(r.wins, vec![1, 0, 0]);
    assert_eq!(r.draws, vec![1, 1, 0]);
    assert_eq!(r.losses, vec![0, 1, 0]);
    assert_eq!(r.lag, vec![0, 0, 0]);
}

/// Unsorted period labels must yield the same result as sorted input
/// (R split() groups by ascending factor level, not first appearance).
#[test]
fn el_unsorted_periods() {
    // EB with the period-2 game listed first.
    let r = super::elo_rating(
        &[2, 1, 1],
        &[0, 0, 0],
        &[1, 1, 2],
        &[1.0, 1.0, 0.5],
        &[0.0, 0.0, 0.0],
        3,
        2000.0,
        400.0,
    )
    .unwrap();
    let s = super::elo_rating(
        &[1, 1, 2],
        &[0, 0, 0],
        &[1, 2, 1],
        &[1.0, 0.5, 1.0],
        &[0.0, 0.0, 0.0],
        3,
        2000.0,
        400.0,
    )
    .unwrap();
    for p in 0..3 {
        assert_eq!(r.ratings[p], s.ratings[p], "player {p}");
    }
    assert_eq!(r.lag, s.lag);
    assert_eq!(r.games, s.games);
}

/// Fractional score (not exactly 0 / 0.5 / 1) counts a game but no W/D/L.
#[test]
fn el_fractional_score() {
    let r = super::elo_rating(&[1], &[0], &[1], &[0.25], &[0.0], 2, 2000.0, 32.0).unwrap();
    assert_eq!(r.games, vec![1, 1]);
    assert_eq!(r.wins, vec![0, 0]);
    assert_eq!(r.draws, vec![0, 0]);
    assert_eq!(r.losses, vec![0, 0]);
    // dscore still applies: d_w = 0.25 - 0.5 = -0.25 (dyadic, exact).
    assert_eq!(r.ratings[0], 1992.0);
    assert_eq!(r.ratings[1], 2008.0);
}

/// Extreme gamma saturates the expectation without panicking; here white
/// is a guaranteed favorite (E_w -> 1) who wins, so nothing moves.
#[test]
fn el_saturation_no_panic() {
    let r = super::elo_rating(&[1], &[0], &[1], &[1.0], &[1e6], 2, 2000.0, 32.0).unwrap();
    assert!(r.ratings[0].is_finite() && r.ratings[1].is_finite());
    assert_eq!(r.ratings[0], 2000.0); // s - E_w = 1 - 1 = 0 exactly
    assert_eq!(r.ratings[1], 2000.0); // (1-s) - E_b = 0 - 0 = 0 exactly
}

/// Error contract: every rejection path returns Err (never panics).
#[test]
fn el_error_contract() {
    let ok = (
        vec![1u64],
        vec![0usize],
        vec![1usize],
        vec![1.0f64],
        vec![0.0f64],
    );
    // empty games
    assert!(super::elo_rating(&[], &[], &[], &[], &[], 2, 2000.0, 32.0).is_err());
    // length mismatch
    assert!(super::elo_rating(&[1, 2], &ok.1, &ok.2, &ok.3, &ok.4, 2, 2000.0, 32.0).is_err());
    // n too small / too large
    assert!(super::elo_rating(&ok.0, &ok.1, &ok.2, &ok.3, &ok.4, 1, 2000.0, 32.0).is_err());
    assert!(super::elo_rating(&ok.0, &ok.1, &ok.2, &ok.3, &ok.4, 10_001, 2000.0, 32.0).is_err());
    // index out of range / self-play
    assert!(super::elo_rating(&ok.0, &[2], &ok.2, &ok.3, &ok.4, 2, 2000.0, 32.0).is_err());
    assert!(super::elo_rating(&ok.0, &[1], &[1], &ok.3, &ok.4, 2, 2000.0, 32.0).is_err());
    // score out of range / non-finite
    assert!(super::elo_rating(&ok.0, &ok.1, &ok.2, &[1.5], &ok.4, 2, 2000.0, 32.0).is_err());
    assert!(super::elo_rating(&ok.0, &ok.1, &ok.2, &[-0.1], &ok.4, 2, 2000.0, 32.0).is_err());
    assert!(super::elo_rating(&ok.0, &ok.1, &ok.2, &[f64::NAN], &ok.4, 2, 2000.0, 32.0).is_err());
    // gamma / init / kfac non-finite
    assert!(super::elo_rating(&ok.0, &ok.1, &ok.2, &ok.3, &[f64::NAN], 2, 2000.0, 32.0).is_err());
    assert!(super::elo_rating(&ok.0, &ok.1, &ok.2, &ok.3, &ok.4, 2, f64::INFINITY, 32.0).is_err());
    assert!(super::elo_rating(&ok.0, &ok.1, &ok.2, &ok.3, &ok.4, 2, 2000.0, f64::NAN).is_err());
}

/// MC-500: structural invariants on random schedules, including nonzero
/// gamma (the E_w + E_b = 1 identity makes rating-sum conservation hold
/// for ANY gamma). All assertions read crate outputs.
#[test]
#[ignore]
fn el_mc_500() {
    let mut rng = Lcg(0x5EED_E10u64);
    for rep in 0..500 {
        let n = 3 + (rng.next_f64() * 6.0) as usize; // 3..=8
        let g = 4 + (rng.next_f64() * 12.0) as usize; // 4..=15
        let mut periods = Vec::with_capacity(g);
        let mut white = Vec::with_capacity(g);
        let mut black = Vec::with_capacity(g);
        let mut score = Vec::with_capacity(g);
        let mut gamma = Vec::with_capacity(g);
        for _ in 0..g {
            periods.push(1 + (rng.next_f64() * 3.0) as u64); // 1..=3
            let w = (rng.next_f64() * n as f64) as usize % n;
            let mut b = (rng.next_f64() * n as f64) as usize % n;
            if b == w {
                b = (b + 1) % n;
            }
            white.push(w);
            black.push(b);
            score.push([0.0, 0.5, 1.0][(rng.next_f64() * 3.0) as usize % 3]);
            gamma.push((rng.next_f64() - 0.5) * 200.0); // nonzero gamma
        }
        let r =
            super::elo_rating(&periods, &white, &black, &score, &gamma, n, 2000.0, 32.0).unwrap();
        // (a) rating-sum conservation for ANY gamma.
        let sum: f64 = r.ratings.iter().sum();
        assert!(
            (sum - 2000.0 * n as f64).abs() < 1e-6,
            "rep {rep}: sum {sum}"
        );
        // (b) games = wins + draws + losses when all scores are in {0,.5,1}.
        for p in 0..n {
            assert_eq!(
                r.games[p],
                r.wins[p] + r.draws[p] + r.losses[p],
                "rep {rep} player {p}"
            );
        }
        // (c) every player appearing in the final period has lag 0.
        let last = *periods.iter().max().unwrap();
        for k in 0..g {
            if periods[k] == last {
                assert_eq!(r.lag[white[k]], 0, "rep {rep}");
                assert_eq!(r.lag[black[k]], 0, "rep {rep}");
            }
        }
        // (d) permuting game order within the input leaves output identical
        // (batch semantics + stable grouping by period value).
        let perm: Vec<usize> = (0..g).rev().collect();
        let r2 = super::elo_rating(
            &perm.iter().map(|&k| periods[k]).collect::<Vec<_>>(),
            &perm.iter().map(|&k| white[k]).collect::<Vec<_>>(),
            &perm.iter().map(|&k| black[k]).collect::<Vec<_>>(),
            &perm.iter().map(|&k| score[k]).collect::<Vec<_>>(),
            &perm.iter().map(|&k| gamma[k]).collect::<Vec<_>>(),
            n,
            2000.0,
            32.0,
        )
        .unwrap();
        for p in 0..n {
            assert!(
                (r.ratings[p] - r2.ratings[p]).abs() < 1e-12,
                "rep {rep} player {p}: order dependence"
            );
        }
    }
}
