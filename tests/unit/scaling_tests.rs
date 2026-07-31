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

// ---------------------------------------------------------------------------
// glicko_rating tests (fixtures GA-GF from the executed float64 oracle
// mirroring PlayerRatings glicko(); GA reproduces Glickman's worked
// example. See the citation-governance header in scaling.rs). Every
// assertion below reads values returned by the crate's glicko_rating.
// ---------------------------------------------------------------------------

/// GA: Glickman's worked example (heterogeneous RDs, cval = 0). Kills MU1
/// (opponent-g swap: uniform-RD fixtures are blind, GA's RDs 200/30/100/300
/// are not), MU4 (stale variance in the rating update: the paper's 131.9
/// multiplier is the NEW variance), MU5 (dval missing q^2), and MU6
/// (dscore sign flip: ratings[0] would rise above 1500 instead of
/// dropping to 1464.106...).
#[test]
fn gk_paper_anchor_ga() {
    let r = super::glicko_rating(
        &[1, 1, 1],
        &[0, 0, 0],
        &[1, 2, 3],
        &[1.0, 0.0, 0.0],
        &[0.0, 0.0, 0.0],
        &[1500.0, 1400.0, 1550.0, 1700.0],
        &[200.0, 30.0, 100.0, 300.0],
        0.0,
        350.0,
    )
    .unwrap();
    // Paper rounds these to r' = 1464, RD' = 151.4; pins are the exact
    // float64 oracle values.
    let er = [
        1464.1064627569112,
        1398.342512471733,
        1570.1876094547742,
        1784.3502813450064,
    ];
    let ed = [
        151.39890244796933,
        29.925091041592754,
        97.21172956677705,
        251.45899758288715,
    ];
    for p in 0..4 {
        assert!((r.ratings[p] - er[p]).abs() < 1e-12, "rating {p}");
        assert!((r.deviations[p] - ed[p]).abs() < 1e-12, "deviation {p}");
    }
    assert_eq!(r.games, vec![3, 1, 1, 1]);
    assert_eq!(r.wins, vec![1, 0, 1, 1]);
    assert_eq!(r.draws, vec![0, 0, 0, 0]);
    assert_eq!(r.losses, vec![2, 1, 0, 0]);
    assert_eq!(r.lag, vec![0, 0, 0, 0]);
}

/// GB: two periods with cval = 15 inflation and an idle player. Kills MU2
/// (inflation (lag+1)*c^2 -> lag*c^2: period-1 inflation vanishes; GA is
/// blind because cval = 0 there), MU8 (lag reset before increment: lag
/// would read [0,0,0]), and MU9 (all-player inflation: idle player 1's
/// period-2 deviation must stay 224.94066563436596 -- the FULL deviation
/// vector is pinned for exactly this kill). Also pins the rating sum
/// 6629.66... != 6600: Glicko does NOT conserve the rating sum (asymmetric
/// opponent-g weighting), so conservation must never be used as an
/// invariant.
#[test]
fn gk_two_period_gb() {
    let r = super::glicko_rating(
        &[1, 1, 2],
        &[0, 1, 2],
        &[1, 2, 0],
        &[1.0, 0.5, 1.0],
        &[0.0, 0.0, 0.0],
        &[2200.0, 2200.0, 2200.0],
        &[300.0, 300.0, 300.0],
        15.0,
        350.0,
    )
    .unwrap();
    let er = [2190.0061185685217, 2094.5895980175137, 2345.066036571678];
    let ed = [223.91939158372585, 224.94066563436596, 223.91939158372585];
    for p in 0..3 {
        assert!((r.ratings[p] - er[p]).abs() < 1e-12, "rating {p}");
        assert!((r.deviations[p] - ed[p]).abs() < 1e-12, "deviation {p}");
    }
    assert_eq!(r.games, vec![2, 2, 2]);
    assert_eq!(r.wins, vec![1, 0, 1]);
    assert_eq!(r.draws, vec![0, 1, 1]);
    assert_eq!(r.losses, vec![1, 1, 0]);
    assert_eq!(r.lag, vec![0, 1, 0]);
    // Documented non-identity: no rating-sum conservation.
    let sum: f64 = r.ratings.iter().sum();
    assert!((sum - 6600.0).abs() > 1e-3, "sum unexpectedly conserved");
    assert!((sum - 6629.661753157714).abs() < 1e-9);
}

/// GC: rdmax clamp active (cval = 400 > rdmax = 350). Kills MU3 (clamp
/// dropped: pre-game variance would be 300^2 + 400^2 = 250000 and the
/// post-game deviation would exceed 350; GA/GB are blind, their clamp is
/// inactive). A symmetric draw leaves both ratings exactly at init.
#[test]
fn gk_rdmax_clamp_gc() {
    let r = super::glicko_rating(
        &[1],
        &[0],
        &[1],
        &[0.5],
        &[0.0],
        &[2200.0, 2200.0],
        &[300.0, 300.0],
        400.0,
        350.0,
    )
    .unwrap();
    assert_eq!(r.ratings, vec![2200.0, 2200.0]);
    for p in 0..2 {
        assert!(
            (r.deviations[p] - 290.2305060910912).abs() < 1e-12,
            "dev {p}"
        );
        assert!(r.deviations[p] < 350.0);
    }
    assert_eq!(r.draws, vec![1, 1]);
}

/// GD: nonzero gamma (white advantage 30). The EXACT rating pins kill MU7
/// (gamma sign swap in the exponents); the structural white-gains-less
/// comparison alone would not, since both runs would swap together.
#[test]
fn gk_gamma_gd() {
    let r = super::glicko_rating(
        &[1],
        &[0],
        &[1],
        &[1.0],
        &[30.0],
        &[2000.0, 2000.0],
        &[200.0, 200.0],
        0.0,
        350.0,
    )
    .unwrap();
    assert!((r.ratings[0] - 2072.980891506514).abs() < 1e-12);
    assert!((r.ratings[1] - 1927.019108493486).abs() < 1e-12);
    assert!((r.deviations[0] - 179.97197655111717).abs() < 1e-12);
    assert!((r.deviations[1] - 179.97197655111717).abs() < 1e-12);
    // Structural check (reads crate outputs of both runs): with a white
    // advantage, E_w > 0.5 and white gains LESS than the gamma = 0 run.
    let r0 = super::glicko_rating(
        &[1],
        &[0],
        &[1],
        &[1.0],
        &[0.0],
        &[2000.0, 2000.0],
        &[200.0, 200.0],
        0.0,
        350.0,
    )
    .unwrap();
    assert!(r.ratings[0] < r0.ratings[0]);
}

/// GE: unsorted period labels produce exactly the GB outputs (stable
/// ascending grouping; both sides of every comparison are crate outputs).
#[test]
fn gk_unsorted_ge() {
    let sorted = super::glicko_rating(
        &[1, 1, 2],
        &[0, 1, 2],
        &[1, 2, 0],
        &[1.0, 0.5, 1.0],
        &[0.0, 0.0, 0.0],
        &[2200.0, 2200.0, 2200.0],
        &[300.0, 300.0, 300.0],
        15.0,
        350.0,
    )
    .unwrap();
    let unsorted = super::glicko_rating(
        &[2, 1, 1],
        &[2, 0, 1],
        &[0, 1, 2],
        &[1.0, 1.0, 0.5],
        &[0.0, 0.0, 0.0],
        &[2200.0, 2200.0, 2200.0],
        &[300.0, 300.0, 300.0],
        15.0,
        350.0,
    )
    .unwrap();
    assert_eq!(sorted.ratings, unsorted.ratings);
    assert_eq!(sorted.deviations, unsorted.deviations);
    assert_eq!(sorted.lag, unsorted.lag);
    assert_eq!(sorted.wins, unsorted.wins);
}

/// GF: fractional score 0.25 counts a game but no W/D/L (only scores
/// exactly 1 / 0.5 / 0 update the tallies); exact rating/deviation pins.
#[test]
fn gk_fractional_score_gf() {
    let r = super::glicko_rating(
        &[1, 1],
        &[0, 0],
        &[1, 2],
        &[0.25, 1.0],
        &[0.0, 0.0],
        &[2200.0, 2200.0, 2200.0],
        &[300.0, 300.0, 300.0],
        15.0,
        350.0,
    )
    .unwrap();
    let er = [2252.705200991243, 2267.5360775700997, 2064.9278448598];
    let ed = [224.94066563436596, 254.6297571494754, 254.6297571494754];
    for p in 0..3 {
        assert!((r.ratings[p] - er[p]).abs() < 1e-12, "rating {p}");
        assert!((r.deviations[p] - ed[p]).abs() < 1e-12, "deviation {p}");
    }
    assert_eq!(r.games, vec![2, 1, 1]);
    assert_eq!(r.wins, vec![1, 0, 0]);
    assert_eq!(r.draws, vec![0, 0, 0]);
    assert_eq!(r.losses, vec![0, 0, 1]);
}

/// Error contract: every rejection path returns Err (crate return value).
#[test]
fn gk_error_contract() {
    let ok = (
        vec![1u64],
        vec![0usize],
        vec![1usize],
        vec![1.0f64],
        vec![0.0f64],
        vec![2000.0f64, 2000.0],
        vec![200.0f64, 200.0],
    );
    let call = |p: &[u64],
                w: &[usize],
                b: &[usize],
                s: &[f64],
                g: &[f64],
                ir: &[f64],
                id: &[f64],
                c: f64,
                rm: f64| { super::glicko_rating(p, w, b, s, g, ir, id, c, rm) };
    // baseline sanity: the ok tuple actually passes
    assert!(call(&ok.0, &ok.1, &ok.2, &ok.3, &ok.4, &ok.5, &ok.6, 15.0, 350.0).is_ok());
    // empty games
    assert!(call(&[], &[], &[], &[], &[], &ok.5, &ok.6, 15.0, 350.0).is_err());
    // column length mismatch
    assert!(call(
        &ok.0,
        &[0, 1],
        &ok.2,
        &ok.3,
        &ok.4,
        &ok.5,
        &ok.6,
        15.0,
        350.0
    )
    .is_err());
    // n < 2
    assert!(call(
        &[1],
        &[0],
        &[0],
        &[1.0],
        &[0.0],
        &[2000.0],
        &[200.0],
        15.0,
        350.0
    )
    .is_err());
    // n > 10000
    let big_r = vec![2000.0; 10_001];
    let big_d = vec![200.0; 10_001];
    assert!(call(&ok.0, &ok.1, &ok.2, &ok.3, &ok.4, &big_r, &big_d, 15.0, 350.0).is_err());
    // init arrays length mismatch
    assert!(call(
        &ok.0,
        &ok.1,
        &ok.2,
        &ok.3,
        &ok.4,
        &ok.5,
        &[200.0],
        15.0,
        350.0
    )
    .is_err());
    // init_dev <= 0 / > rdmax / non-finite; init_rating non-finite
    assert!(call(
        &ok.0,
        &ok.1,
        &ok.2,
        &ok.3,
        &ok.4,
        &ok.5,
        &[0.0, 200.0],
        15.0,
        350.0
    )
    .is_err());
    assert!(call(
        &ok.0,
        &ok.1,
        &ok.2,
        &ok.3,
        &ok.4,
        &ok.5,
        &[400.0, 200.0],
        15.0,
        350.0
    )
    .is_err());
    assert!(call(
        &ok.0,
        &ok.1,
        &ok.2,
        &ok.3,
        &ok.4,
        &ok.5,
        &[f64::NAN, 200.0],
        15.0,
        350.0
    )
    .is_err());
    assert!(call(
        &ok.0,
        &ok.1,
        &ok.2,
        &ok.3,
        &ok.4,
        &[f64::INFINITY, 2000.0],
        &ok.6,
        15.0,
        350.0
    )
    .is_err());
    // rdmax <= 0 / non-finite; cval < 0 / non-finite
    assert!(call(&ok.0, &ok.1, &ok.2, &ok.3, &ok.4, &ok.5, &ok.6, 15.0, 0.0).is_err());
    assert!(call(
        &ok.0,
        &ok.1,
        &ok.2,
        &ok.3,
        &ok.4,
        &ok.5,
        &ok.6,
        15.0,
        f64::NAN
    )
    .is_err());
    assert!(call(&ok.0, &ok.1, &ok.2, &ok.3, &ok.4, &ok.5, &ok.6, -1.0, 350.0).is_err());
    assert!(call(
        &ok.0,
        &ok.1,
        &ok.2,
        &ok.3,
        &ok.4,
        &ok.5,
        &ok.6,
        f64::NAN,
        350.0
    )
    .is_err());
    // player index out of range; self-play
    assert!(call(&ok.0, &[2], &ok.2, &ok.3, &ok.4, &ok.5, &ok.6, 15.0, 350.0).is_err());
    assert!(call(&ok.0, &[1], &[1], &ok.3, &ok.4, &ok.5, &ok.6, 15.0, 350.0).is_err());
    // score outside [0,1] / non-finite; gamma non-finite
    assert!(call(
        &ok.0,
        &ok.1,
        &ok.2,
        &[1.5],
        &ok.4,
        &ok.5,
        &ok.6,
        15.0,
        350.0
    )
    .is_err());
    assert!(call(
        &ok.0,
        &ok.1,
        &ok.2,
        &[-0.1],
        &ok.4,
        &ok.5,
        &ok.6,
        15.0,
        350.0
    )
    .is_err());
    assert!(call(
        &ok.0,
        &ok.1,
        &ok.2,
        &[f64::NAN],
        &ok.4,
        &ok.5,
        &ok.6,
        15.0,
        350.0
    )
    .is_err());
    assert!(call(
        &ok.0,
        &ok.1,
        &ok.2,
        &ok.3,
        &[f64::NAN],
        &ok.5,
        &ok.6,
        15.0,
        350.0
    )
    .is_err());
}

/// MC-500: structural invariants on random schedules. All assertions read
/// crate outputs. Rating-sum conservation is deliberately NOT asserted
/// (Glicko does not conserve it; see gk_two_period_gb).
#[test]
#[ignore]
fn gk_mc_500() {
    let mut rng = Lcg(0x61c0_u64 ^ 0x9e3779b97f4a7c15);
    for rep in 0..500 {
        let n = 3 + (rng.next_f64() * 6.0) as usize; // 3..=8
        let g = 2 + (rng.next_f64() * 10.0) as usize; // 2..=11
        let mut periods = Vec::with_capacity(g);
        let mut white = Vec::with_capacity(g);
        let mut black = Vec::with_capacity(g);
        let mut score = Vec::with_capacity(g);
        let mut gamma = Vec::with_capacity(g);
        for _ in 0..g {
            periods.push(1 + (rng.next_f64() * 4.0) as u64);
            let w = (rng.next_f64() * n as f64) as usize % n;
            let mut b = (rng.next_f64() * n as f64) as usize % n;
            if b == w {
                b = (b + 1) % n;
            }
            white.push(w);
            black.push(b);
            score.push([0.0, 0.5, 1.0][(rng.next_f64() * 3.0) as usize % 3]);
            gamma.push((rng.next_f64() - 0.5) * 60.0);
        }
        let init_r = vec![2200.0; n];
        let init_d = vec![300.0; n];
        let r = super::glicko_rating(
            &periods, &white, &black, &score, &gamma, &init_r, &init_d, 15.0, 350.0,
        )
        .unwrap();
        for p in 0..n {
            // (a) all outputs finite; deviations in (0, rdmax].
            assert!(r.ratings[p].is_finite(), "rep {rep}");
            assert!(
                r.deviations[p] > 0.0 && r.deviations[p] <= 350.0,
                "rep {rep} dev {}",
                r.deviations[p]
            );
            // (b) games = wins + draws + losses for integer/half scores.
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
        // (d) permuting game order leaves output identical (batch
        // semantics + stable ascending grouping by period value).
        let perm: Vec<usize> = (0..g).rev().collect();
        let r2 = super::glicko_rating(
            &perm.iter().map(|&k| periods[k]).collect::<Vec<_>>(),
            &perm.iter().map(|&k| white[k]).collect::<Vec<_>>(),
            &perm.iter().map(|&k| black[k]).collect::<Vec<_>>(),
            &perm.iter().map(|&k| score[k]).collect::<Vec<_>>(),
            &perm.iter().map(|&k| gamma[k]).collect::<Vec<_>>(),
            &init_r,
            &init_d,
            15.0,
            350.0,
        )
        .unwrap();
        for p in 0..n {
            assert!(
                (r.ratings[p] - r2.ratings[p]).abs() < 1e-12,
                "rep {rep} player {p}: order dependence"
            );
        }
        // (e) cval = 0, every-period player: deviation never increases.
        // Run a two-period schedule where player 0 plays both periods and
        // compare crate deviations after period 1 vs after period 2.
        let one = super::glicko_rating(
            &[1],
            &[0],
            &[1],
            &[1.0],
            &[0.0],
            &init_r[..2],
            &init_d[..2],
            0.0,
            350.0,
        )
        .unwrap();
        let two = super::glicko_rating(
            &[1, 2],
            &[0, 0],
            &[1, 1],
            &[1.0, 1.0],
            &[0.0, 0.0],
            &init_r[..2],
            &init_d[..2],
            0.0,
            350.0,
        )
        .unwrap();
        assert!(two.deviations[0] <= one.deviations[0], "rep {rep}");
    }
}

// -------------------------------------------------------------------------
// glicko2_rating (Glickman 2022 note + PlayerRatings glicko2 semantics).
// All pins are crate outputs checked against the independently executed
// float64 oracle (files/glicko2_oracle.py); numeric tolerance 1e-7 abs
// per the spec's Illinois endpoint-A mandate (endpoint swap shifts values
// by ~2.5e-5, so 1e-7 pins the mandated algorithm while tolerating
// last-ulp reassociation).

fn g2_close(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

/// Glickman (2022) worked example, heterogeneous init.
/// Asserts read: r.ratings / r.deviations / r.volatilities from the crate.
/// Killed mutants: MU1 (own-g swap), MU4 (stale sigma), MU5 (old sigma in
/// phi*), MU6 (Illinois sign/bracket), MU8 (rating-before-deviation).
#[test]
fn g2_paper_anchor_g2a() {
    let r = crate::scaling::glicko2_rating(
        &[1, 1, 1],
        &[0, 0, 0],
        &[1, 2, 3],
        &[1.0, 0.0, 0.0],
        &[0.0; 3],
        &[1500.0, 1400.0, 1550.0, 1700.0],
        &[200.0, 30.0, 100.0, 300.0],
        &[0.06; 4],
        0.5,
        350.0,
    )
    .unwrap();
    // Note's printed digits: r' = 1464.06, RD' = 151.52, sigma' = 0.05999.
    assert!(g2_close(
        &r.ratings,
        &[
            1464.0506708196929,
            1398.143558212337,
            1570.3947406805573,
            1784.421789699685
        ],
        1e-7
    ));
    assert!(g2_close(
        &r.deviations,
        &[
            151.51652192592556,
            31.67021513485606,
            97.70916832182286,
            251.56556278667546
        ],
        1e-7
    ));
    assert!(g2_close(
        &r.volatilities,
        &[
            0.05999598428664987,
            0.05999912372888925,
            0.0599994194719928,
            0.059999011763705826
        ],
        1e-7
    ));
    assert_eq!(r.games, vec![3, 1, 1, 1]);
    assert_eq!(r.wins, vec![1, 0, 1, 1]);
    assert_eq!(r.losses, vec![2, 1, 0, 0]);
    assert_eq!(r.lag, vec![0, 0, 0, 0]);
}

/// Two periods with an idle player; defaults (2200, 300, 0.15), tau 1.2.
/// Asserts read crate ratings/deviations/volatilities/lag.
/// Killed mutants: MU2 (off-by-one inflation at lag 0); also pins the
/// NO-conservation identity (sum != 6600).
#[test]
fn g2_two_period_g2b() {
    let r = crate::scaling::glicko2_rating(
        &[1, 1, 2],
        &[0, 1, 0],
        &[1, 2, 2],
        &[1.0, 0.5, 0.0],
        &[0.0; 3],
        &[2200.0; 3],
        &[300.0; 3],
        &[0.15; 3],
        1.2,
        350.0,
    )
    .unwrap();
    assert!(g2_close(
        &r.ratings,
        &[2189.2376693682463, 2094.286091224066, 2346.354869237831],
        1e-7
    ));
    assert!(g2_close(
        &r.deviations,
        &[224.81517267835346, 225.1973735871327, 224.81375337135282],
        1e-7
    ));
    assert!(g2_close(
        &r.volatilities,
        &[0.15002063478252045, 0.1498718983333671, 0.14993884055293957],
        1e-7
    ));
    assert_eq!(r.lag, vec![0, 1, 0]);
    // Documented NON-identity: rating sum is not conserved.
    let sum: f64 = r.ratings.iter().sum();
    assert!((sum - 6600.0).abs() > 1e-3, "sum unexpectedly conserved");
    assert!((sum - 6629.878629830143).abs() <= 1e-6);
}

/// Long idle gap drives the deviation into the rdmax clamp.
/// Asserts read crate deviations/volatilities.
/// Killed mutants: MU3a (dropped phi^2 rdmax clamps).
#[test]
fn g2_rdmax_clamp_g2c() {
    let r = crate::scaling::glicko2_rating(
        &[1, 10],
        &[0, 0],
        &[1, 1],
        &[1.0, 0.5],
        &[0.0; 2],
        &[2200.0; 2],
        &[300.0; 2],
        &[1.0; 2],
        0.3,
        350.0,
    )
    .unwrap();
    assert!(g2_close(
        &r.ratings,
        &[2256.2545251058973, 2143.7454748941027],
        1e-7
    ));
    assert!(g2_close(
        &r.deviations,
        &[287.7492897078701, 287.7492897078701],
        1e-7
    ));
    assert!(g2_close(
        &r.volatilities,
        &[0.9985111330899122, 0.9985111330899122],
        1e-7
    ));
}

/// Single draw with white advantage gamma = 30: favored white DROPS.
/// Asserts read crate ratings. Killed mutants: MU7 (gamma sign flip).
#[test]
fn g2_gamma_g2d() {
    let r = crate::scaling::glicko2_rating(
        &[1],
        &[0],
        &[1],
        &[0.5],
        &[30.0],
        &[2200.0; 2],
        &[300.0; 2],
        &[0.15; 2],
        1.2,
        350.0,
    )
    .unwrap();
    assert!(g2_close(
        &r.ratings,
        &[2191.52226104287, 2208.47773895713],
        1e-7
    ));
    assert!(r.ratings[0] < r.ratings[1]);
    assert!(g2_close(
        &r.deviations,
        &[255.18592153180703, 255.18592153180703],
        1e-7
    ));
}

/// Unsorted period labels must equal the sorted G2B result exactly.
/// Asserts read crate outputs from both calls (exact equality).
#[test]
fn g2_unsorted_g2e() {
    let sorted = crate::scaling::glicko2_rating(
        &[1, 1, 2],
        &[0, 1, 0],
        &[1, 2, 2],
        &[1.0, 0.5, 0.0],
        &[0.0; 3],
        &[2200.0; 3],
        &[300.0; 3],
        &[0.15; 3],
        1.2,
        350.0,
    )
    .unwrap();
    let shuffled = crate::scaling::glicko2_rating(
        &[2, 1, 1],
        &[0, 0, 1],
        &[2, 1, 2],
        &[0.0, 1.0, 0.5],
        &[0.0; 3],
        &[2200.0; 3],
        &[300.0; 3],
        &[0.15; 3],
        1.2,
        350.0,
    )
    .unwrap();
    assert_eq!(sorted.ratings, shuffled.ratings);
    assert_eq!(sorted.deviations, shuffled.deviations);
    assert_eq!(sorted.volatilities, shuffled.volatilities);
    assert_eq!(sorted.lag, shuffled.lag);
}

/// Fractional score 0.25 earns no W/D/L tally; tau = 0 freezes sigma.
/// Asserts read crate tallies/volatilities/ratings.
/// Killed mutants: tally-on-fractional; volatility-update-despite-tau-0.
#[test]
fn g2_fractional_tau0_g2f() {
    let r = crate::scaling::glicko2_rating(
        &[1, 2],
        &[0, 1],
        &[1, 2],
        &[0.25, 1.0],
        &[0.0; 2],
        &[2200.0; 3],
        &[300.0; 3],
        &[0.15; 3],
        0.0,
        350.0,
    )
    .unwrap();
    assert_eq!(r.wins, vec![0, 1, 0]);
    assert_eq!(r.draws, vec![0, 0, 0]);
    assert_eq!(r.losses, vec![0, 0, 1]);
    assert_eq!(r.games, vec![1, 2, 1]);
    // tau = 0: volatilities EXACTLY unchanged (crate output).
    assert_eq!(r.volatilities, vec![0.15, 0.15, 0.15]);
    assert!(g2_close(
        &r.ratings,
        &[2132.2025470843405, 2359.81140705271, 2080.6221639984115],
        1e-7
    ));
    assert!(g2_close(
        &r.deviations,
        &[255.0462500503705, 226.607229056582, 250.63135994001263],
        1e-7
    ));
    assert_eq!(r.lag, vec![1, 0, 0]);
}

/// Return after a true idle period: P0 plays period 1, idles period 2,
/// returns period 3 with lag = 1, so inflation is 1 * sigma^2 (Glicko-2)
/// vs 2 * sigma^2 under the Glicko-1 (lag+1) regression.
/// Asserts read crate ratings/deviations/lag.
/// Killed mutants: MU2 (oracle-verified: mutant shifts devs by ~2.43).
#[test]
fn g2_idle_return_g2h() {
    let r = crate::scaling::glicko2_rating(
        &[1, 2, 3],
        &[0, 1, 0],
        &[1, 2, 1],
        &[1.0, 0.5, 0.5],
        &[0.0; 3],
        &[2200.0; 3],
        &[300.0; 3],
        &[0.15; 3],
        1.2,
        350.0,
    )
    .unwrap();
    assert!(g2_close(
        &r.ratings,
        &[2273.2760877220917, 2142.521696205541, 2157.7820411274447],
        1e-7
    ));
    assert!(g2_close(
        &r.deviations,
        &[228.9242656387728, 209.60776091326838, 253.1905234114569],
        1e-7
    ));
    assert!(g2_close(
        &r.volatilities,
        &[
            0.14988381084284416,
            0.14977477223682767,
            0.14988972503726555
        ],
        1e-7
    ));
    assert_eq!(r.lag, vec![0, 0, 1]);
}

/// Volatility ceiling active: huge favorite loses with large tau and
/// sigma near the cap, so the unclamped Illinois root exceeds
/// q * rdmax and must be clamped to EXACTLY that ceiling.
/// Asserts read crate volatilities/deviations/ratings.
/// Killed mutants: MU3b (dropped sigma clamp).
#[test]
fn g2_vol_clamp_g2i() {
    let r = crate::scaling::glicko2_rating(
        &[1],
        &[0],
        &[1],
        &[0.0],
        &[0.0],
        &[2400.0, 1200.0],
        &[50.0, 50.0],
        &[0.28, 0.28],
        2.0,
        50.0,
    )
    .unwrap();
    let vol_max = std::f64::consts::LN_10 / 400.0 * 50.0;
    assert_eq!(r.volatilities, vec![vol_max, vol_max]);
    assert_eq!(r.deviations, vec![50.0, 50.0]); // phi clamp also active
    assert!(g2_close(
        &r.ratings,
        &[2385.802146381853, 1214.197853618147],
        1e-7
    ));
}

/// Error contract: every rejection path returns Err (crate Result read).
#[test]
fn g2_error_contract() {
    let ok_p = [1u64];
    let ok_w = [0usize];
    let ok_b = [1usize];
    let ok_s = [1.0f64];
    let ok_g = [0.0f64];
    let r2 = [2200.0, 2200.0];
    let d2 = [300.0, 300.0];
    let v2 = [0.15, 0.15];
    let call = |p: &[u64],
                w: &[usize],
                b: &[usize],
                s: &[f64],
                gm: &[f64],
                ir: &[f64],
                id: &[f64],
                iv: &[f64],
                tau: f64,
                rdmax: f64| {
        crate::scaling::glicko2_rating(p, w, b, s, gm, ir, id, iv, tau, rdmax)
    };
    // no games / length mismatch
    assert!(call(&[], &[], &[], &[], &[], &r2, &d2, &v2, 1.2, 350.0).is_err());
    assert!(call(&ok_p, &ok_w, &ok_b, &ok_s, &[], &r2, &d2, &v2, 1.2, 350.0).is_err());
    // n < 2, init length mismatch
    assert!(call(
        &ok_p,
        &[0],
        &[0],
        &ok_s,
        &ok_g,
        &[2200.0],
        &[300.0],
        &[0.15],
        1.2,
        350.0
    )
    .is_err());
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &ok_g,
        &r2,
        &[300.0],
        &v2,
        1.2,
        350.0
    )
    .is_err());
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &ok_g,
        &r2,
        &d2,
        &[0.15],
        1.2,
        350.0
    )
    .is_err());
    // dev / vol domain
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &ok_g,
        &r2,
        &[300.0, 0.0],
        &v2,
        1.2,
        350.0
    )
    .is_err());
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &ok_g,
        &r2,
        &[300.0, 400.0],
        &v2,
        1.2,
        350.0
    )
    .is_err());
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &ok_g,
        &r2,
        &d2,
        &[0.15, 0.0],
        1.2,
        350.0
    )
    .is_err());
    // vol above ln(10)/400 * rdmax (= ~2.0148 at rdmax 350)
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &ok_g,
        &r2,
        &d2,
        &[0.15, 2.1],
        1.2,
        350.0
    )
    .is_err());
    // tau / rdmax domain
    assert!(call(&ok_p, &ok_w, &ok_b, &ok_s, &ok_g, &r2, &d2, &v2, -0.1, 350.0).is_err());
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &ok_g,
        &r2,
        &d2,
        &v2,
        f64::NAN,
        350.0
    )
    .is_err());
    assert!(call(&ok_p, &ok_w, &ok_b, &ok_s, &ok_g, &r2, &d2, &v2, 1.2, 0.0).is_err());
    // game rows
    assert!(call(&ok_p, &[2], &ok_b, &ok_s, &ok_g, &r2, &d2, &v2, 1.2, 350.0).is_err());
    assert!(call(&ok_p, &[1], &[1], &ok_s, &ok_g, &r2, &d2, &v2, 1.2, 350.0).is_err());
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &[1.5],
        &ok_g,
        &r2,
        &d2,
        &v2,
        1.2,
        350.0
    )
    .is_err());
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &[f64::NAN],
        &ok_g,
        &r2,
        &d2,
        &v2,
        1.2,
        350.0
    )
    .is_err());
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &[f64::INFINITY],
        &r2,
        &d2,
        &v2,
        1.2,
        350.0
    )
    .is_err());
    // non-finite init arrays
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &ok_g,
        &[2200.0, f64::NAN],
        &d2,
        &v2,
        1.2,
        350.0
    )
    .is_err());
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &ok_g,
        &r2,
        &[300.0, f64::INFINITY],
        &v2,
        1.2,
        350.0
    )
    .is_err());
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &ok_g,
        &r2,
        &d2,
        &[0.15, f64::NAN],
        1.2,
        350.0
    )
    .is_err());
    // rdmax non-finite / negative
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &ok_g,
        &r2,
        &d2,
        &v2,
        1.2,
        f64::NAN
    )
    .is_err());
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &ok_g,
        &r2,
        &d2,
        &v2,
        1.2,
        f64::INFINITY
    )
    .is_err());
    assert!(call(&ok_p, &ok_w, &ok_b, &ok_s, &ok_g, &r2, &d2, &v2, 1.2, -350.0).is_err());
    // negative score
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &[-0.5],
        &ok_g,
        &r2,
        &d2,
        &v2,
        1.2,
        350.0
    )
    .is_err());
    // finite init_vol exactly AT the ceiling is VALID; just above is not
    let vmax = std::f64::consts::LN_10 / 400.0 * 350.0;
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &ok_g,
        &r2,
        &d2,
        &[0.15, vmax],
        1.2,
        350.0
    )
    .is_ok());
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &ok_g,
        &r2,
        &d2,
        &[0.15, vmax * (1.0 + 1e-12)],
        1.2,
        350.0
    )
    .is_err());
    // tau = 0 is VALID (frozen volatility)
    assert!(call(&ok_p, &ok_w, &ok_b, &ok_s, &ok_g, &r2, &d2, &v2, 0.0, 350.0).is_ok());
}

/// 500-rep Monte Carlo: random schedules must satisfy structural
/// invariants read from crate outputs (finiteness, dev in (0, rdmax],
/// vol in (0, q*rdmax], tally consistency, unsorted == sorted).
#[test]
#[ignore = "500-rep Monte Carlo; run explicitly"]
fn g2_mc_500() {
    let mut rng = Lcg(0x61c0_2u64);
    for rep in 0..500 {
        let n = 3 + (rng.next_f64() * 6.0) as usize;
        let m = 4 + (rng.next_f64() * 20.0) as usize;
        let mut periods = Vec::with_capacity(m);
        let mut white = Vec::with_capacity(m);
        let mut black = Vec::with_capacity(m);
        let mut score = Vec::with_capacity(m);
        let mut gamma = Vec::with_capacity(m);
        for _ in 0..m {
            periods.push(1 + (rng.next_f64() * 5.0) as u64);
            let w = (rng.next_f64() * n as f64) as usize % n;
            let mut b = (rng.next_f64() * n as f64) as usize % n;
            if b == w {
                b = (b + 1) % n;
            }
            white.push(w);
            black.push(b);
            let u = rng.next_f64();
            score.push(if u < 0.4 {
                1.0
            } else if u < 0.8 {
                0.0
            } else if u < 0.9 {
                0.5
            } else {
                u
            });
            gamma.push((rng.next_f64() - 0.5) * 60.0);
        }
        let tau = rng.next_f64() * 1.2;
        let r = crate::scaling::glicko2_rating(
            &periods,
            &white,
            &black,
            &score,
            &gamma,
            &vec![2200.0; n],
            &vec![300.0; n],
            &vec![0.15; n],
            tau,
            350.0,
        )
        .unwrap();
        let vol_max = std::f64::consts::LN_10 / 400.0 * 350.0;
        for p in 0..n {
            assert!(r.ratings[p].is_finite(), "rep {} player {}", rep, p);
            assert!(
                r.deviations[p] > 0.0 && r.deviations[p] <= 350.0 + 1e-9,
                "rep {} dev {}",
                rep,
                r.deviations[p]
            );
            assert!(
                r.volatilities[p] > 0.0 && r.volatilities[p] <= vol_max + 1e-12,
                "rep {} vol {}",
                rep,
                r.volatilities[p]
            );
            assert!(r.wins[p] + r.draws[p] + r.losses[p] <= r.games[p]);
        }
        let total_games: u64 = r.games.iter().sum();
        assert_eq!(total_games, 2 * m as u64);
    }
}

// ---------------------------------------------------------------------------
// stephenson_rating tests. Oracle: faithful Python port of the READ
// PlayerRatings steph() R driver + stephenson_c C kernel, EXECUTED
// (session files/stephenson_oracle.py); anchors below are its pins.
// Every assert reads crate-returned StephensonResult fields.
// ---------------------------------------------------------------------------

fn st_close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs().max(1.0)
}

/// S1: single game, heterogeneous init deviations (gdevs[w] != gdevs[b],
/// so this kills the wrong-side-g mutant MU5). Asserts read
/// ratings/deviations/games/wins/losses from the crate result.
/// Killing mutants: MU4 (lag+1 -> lag: fresh players get NO inflation,
/// every pin moves), MU5 (own g instead of opponent's).
#[test]
fn st_anchor_s1() {
    let r = stephenson_rating(
        &[1],
        &[0],
        &[1],
        &[1.0],
        &[0.0],
        &[2200.0, 2300.0, 2100.0],
        &[300.0, 80.0, 150.0],
        &[0, 0, 0],
        &[0, 0, 0],
        10.0,
        10.0,
        0.0,
        2.0,
        350.0,
    )
    .unwrap();
    assert!(st_close(r.ratings[0], 2395.927345489556, 1e-9));
    assert!(st_close(r.deviations[0], 233.84458992614066, 1e-9));
    assert!(st_close(r.ratings[1], 2281.863290035741, 1e-9));
    assert!(st_close(r.deviations[1], 80.14766179107052, 1e-9));
    // untouched non-participant: exact state passthrough
    assert_eq!(r.ratings[2], 2100.0);
    assert_eq!(r.deviations[2], 150.0);
    assert_eq!(r.games, vec![1, 1, 0]);
    assert_eq!(r.wins, vec![1, 0, 0]);
    assert_eq!(r.losses, vec![0, 1, 0]);
    assert_eq!(r.draws, vec![0, 0, 0]);
    assert_eq!(r.lag, vec![0, 0, 0]);
}

/// S2: two periods with a draw and an idle player (p1 idle in period 2).
/// Killing mutants: MU3 (drop per-game hval^2), lag bookkeeping mutants.
#[test]
fn st_two_period_s2() {
    let r = stephenson_rating(
        &[1, 1, 2],
        &[0, 1, 0],
        &[1, 2, 2],
        &[1.0, 0.5, 0.0],
        &[0.0; 3],
        &[2200.0; 3],
        &[300.0; 3],
        &[0; 3],
        &[0; 3],
        10.0,
        10.0,
        0.0,
        2.0,
        350.0,
    )
    .unwrap();
    assert!(st_close(r.ratings[0], 2187.33261103678, 1e-9));
    assert!(st_close(r.deviations[0], 223.83153574055694, 1e-9));
    assert!(st_close(r.ratings[1], 2094.536265834189, 1e-9));
    assert!(st_close(r.deviations[1], 224.9604507965822, 1e-9));
    assert!(st_close(r.ratings[2], 2347.732126824051, 1e-9));
    assert_eq!(r.games, vec![2, 2, 2]);
    assert_eq!(r.wins, vec![1, 0, 1]);
    assert_eq!(r.draws, vec![0, 1, 1]);
    assert_eq!(r.losses, vec![1, 1, 0]);
    assert_eq!(r.lag, vec![0, 1, 0]);
}

/// Lambda drift is participant-only and separable: with lambda = 0 the S2
/// fixture's p0/p2 move but p1 (idle in period 2) is IDENTICAL across the
/// two crate calls. Killing mutants: MU2 (lambda sign flip: p0 pin moves
/// by 2 * 2.701 the other way), lambda-applied-to-all mutants.
#[test]
fn st_lambda_zero() {
    let base = stephenson_rating(
        &[1, 1, 2],
        &[0, 1, 0],
        &[1, 2, 2],
        &[1.0, 0.5, 0.0],
        &[0.0; 3],
        &[2200.0; 3],
        &[300.0; 3],
        &[0; 3],
        &[0; 3],
        10.0,
        10.0,
        0.0,
        2.0,
        350.0,
    )
    .unwrap();
    let nolam = stephenson_rating(
        &[1, 1, 2],
        &[0, 1, 0],
        &[1, 2, 2],
        &[1.0, 0.5, 0.0],
        &[0.0; 3],
        &[2200.0; 3],
        &[300.0; 3],
        &[0; 3],
        &[0; 3],
        10.0,
        10.0,
        0.0,
        0.0,
        350.0,
    )
    .unwrap();
    assert!(st_close(nolam.ratings[0], 2190.0339057939964, 1e-9));
    assert!(st_close(nolam.ratings[2], 2345.0308320668346, 1e-9));
    // p1's period-2 state is untouched by lambda (participant-only drift):
    // both crate calls must return bit-identical p1 ratings.
    assert_eq!(base.ratings[1], nolam.ratings[1]);
    // direction: lambda = 2 pulls p0 DOWN toward its lower-rated pool
    assert!(base.ratings[0] < nolam.ratings[0]);
}

/// S3: every knob nonzero at once (gamma = [30, 0], cval = 8, hval = 15,
/// bval = 5, lambda = 5). Killing mutants: MU1 (drop bval), MU3 (hval),
/// gamma-side mutants.
#[test]
fn st_full_knobs_s3() {
    let r = stephenson_rating(
        &[1, 2],
        &[0, 1],
        &[1, 2],
        &[0.5, 1.0],
        &[30.0, 0.0],
        &[2200.0; 3],
        &[300.0; 3],
        &[0; 3],
        &[0; 3],
        8.0,
        15.0,
        5.0,
        5.0,
        350.0,
    )
    .unwrap();
    assert!(st_close(r.ratings[0], 2205.081933917611, 1e-9));
    assert!(st_close(r.deviations[0], 254.8041343216954, 1e-9));
    assert!(st_close(r.ratings[1], 2332.590625814148, 1e-9));
    assert!(st_close(r.deviations[1], 225.48194679176544, 1e-9));
    assert!(st_close(r.ratings[2], 2082.6278256782653, 1e-9));
    assert!(st_close(r.deviations[2], 249.45302895967546, 1e-9));
    assert_eq!(r.lag, vec![1, 0, 0]);
    assert_eq!(r.games, vec![1, 2, 1]);
    assert_eq!(r.draws, vec![1, 1, 0]);
    assert_eq!(r.wins, vec![0, 1, 0]);
    assert_eq!(r.losses, vec![0, 0, 1]);
}

/// S4: rdmax clamp binds (init dev 340, cval = 60, continued lag 4 =>
/// unclamped variance 340^2 + 5 * 3600 > 350^2) and init_games/init_lag
/// continuation feeds both the inflation multiplier and the games output.
#[test]
fn st_rdmax_clamp_s4() {
    let r = stephenson_rating(
        &[1],
        &[0],
        &[1],
        &[1.0],
        &[0.0],
        &[2200.0; 2],
        &[340.0; 2],
        &[5, 5],
        &[4, 4],
        60.0,
        10.0,
        0.0,
        2.0,
        350.0,
    )
    .unwrap();
    assert!(st_close(r.ratings[0], 2362.303032952076, 1e-9));
    assert!(st_close(r.deviations[0], 290.31193063897507, 1e-9));
    assert!(st_close(r.ratings[1], 2037.6969670479239, 1e-9));
    assert_eq!(r.games, vec![6, 6]);
    // W/D/L are current-run tallies (init W/D/L is out of scope)
    assert_eq!(r.wins, vec![1, 0]);
    assert_eq!(r.losses, vec![0, 1]);
    assert_eq!(r.lag, vec![0, 0]);
}

/// S5: bval = 10 adds the bonus to BOTH actual scores; the loser's rating
/// RISES relative to the bval = 0 run (not a mirror). Killing mutant:
/// MU1 (asc = s, bonus dropped, reverts to the bval = 0 pins).
#[test]
fn st_bonus_s5() {
    let bonus = stephenson_rating(
        &[1],
        &[0],
        &[1],
        &[1.0],
        &[0.0],
        &[2200.0; 2],
        &[300.0; 2],
        &[0; 2],
        &[0; 2],
        10.0,
        10.0,
        10.0,
        2.0,
        350.0,
    )
    .unwrap();
    let plain = stephenson_rating(
        &[1],
        &[0],
        &[1],
        &[1.0],
        &[0.0],
        &[2200.0; 2],
        &[300.0; 2],
        &[0; 2],
        &[0; 2],
        10.0,
        10.0,
        0.0,
        2.0,
        350.0,
    )
    .unwrap();
    assert!(st_close(bonus.ratings[0], 2362.077685432997, 1e-9));
    assert!(st_close(bonus.ratings[1], 2091.9482097113355, 1e-9));
    assert!(st_close(plain.ratings[0], 2335.064737860831, 1e-9));
    assert!(st_close(plain.ratings[1], 2064.935262139169, 1e-9));
    // bonus lifts BOTH players (crate outputs compared across calls)
    assert!(bonus.ratings[0] > plain.ratings[0]);
    assert!(bonus.ratings[1] > plain.ratings[1]);
}

/// Period grouping sorts unique labels ascending regardless of row order:
/// the S2 fixture fed in reversed row order returns identical output.
#[test]
fn st_unsorted() {
    let sorted = stephenson_rating(
        &[1, 1, 2],
        &[0, 1, 0],
        &[1, 2, 2],
        &[1.0, 0.5, 0.0],
        &[0.0; 3],
        &[2200.0; 3],
        &[300.0; 3],
        &[0; 3],
        &[0; 3],
        10.0,
        10.0,
        0.0,
        2.0,
        350.0,
    )
    .unwrap();
    let rev = stephenson_rating(
        &[2, 1, 1],
        &[0, 1, 0],
        &[2, 2, 1],
        &[0.0, 0.5, 1.0],
        &[0.0; 3],
        &[2200.0; 3],
        &[300.0; 3],
        &[0; 3],
        &[0; 3],
        10.0,
        10.0,
        0.0,
        2.0,
        350.0,
    )
    .unwrap();
    for p in 0..3 {
        assert_eq!(sorted.ratings[p], rev.ratings[p]);
        assert_eq!(sorted.deviations[p], rev.deviations[p]);
        assert_eq!(sorted.lag[p], rev.lag[p]);
    }
}

/// Every rejection path returns Err; boundary accepts are pinned too.
#[test]
fn st_error_contract() {
    // u64 overflow guard: counters must reject rather than wrap/panic.
    assert!(stephenson_rating(
        &[1],
        &[0],
        &[1],
        &[1.0],
        &[0.0],
        &[2200.0, 2200.0],
        &[300.0, 300.0],
        &[u64::MAX, 0],
        &[0, 0],
        10.0,
        10.0,
        0.0,
        2.0,
        350.0
    )
    .is_err());
    assert!(stephenson_rating(
        &[1],
        &[0],
        &[1],
        &[1.0],
        &[0.0],
        &[2200.0, 2200.0],
        &[300.0, 300.0],
        &[0, 0],
        &[0, u64::MAX],
        10.0,
        10.0,
        0.0,
        2.0,
        350.0
    )
    .is_err());

    #[allow(clippy::too_many_arguments)]
    fn call(
        p: &[u64],
        w: &[usize],
        b: &[usize],
        s: &[f64],
        gm: &[f64],
        ir: &[f64],
        id: &[f64],
        ig: &[u64],
        il: &[u64],
        cval: f64,
        hval: f64,
        bval: f64,
        lambda: f64,
        rdmax: f64,
    ) -> Result<StephensonResult, String> {
        stephenson_rating(
            p, w, b, s, gm, ir, id, ig, il, cval, hval, bval, lambda, rdmax,
        )
    }
    let ok_p: Vec<u64> = vec![1];
    let ok_w: Vec<usize> = vec![0];
    let ok_b: Vec<usize> = vec![1];
    let ok_s = vec![1.0];
    let ok_g = vec![0.0];
    let r2 = vec![2200.0, 2200.0];
    let d2 = vec![300.0, 300.0];
    let z2: Vec<u64> = vec![0, 0];
    // empty games
    assert!(call(
        &[],
        &[],
        &[],
        &[],
        &[],
        &r2,
        &d2,
        &z2,
        &z2,
        10.0,
        10.0,
        0.0,
        2.0,
        350.0
    )
    .is_err());
    // length mismatches
    assert!(call(
        &ok_p,
        &[0, 1],
        &ok_b,
        &ok_s,
        &ok_g,
        &r2,
        &d2,
        &z2,
        &z2,
        10.0,
        10.0,
        0.0,
        2.0,
        350.0
    )
    .is_err());
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &[1.0, 0.0],
        &ok_g,
        &r2,
        &d2,
        &z2,
        &z2,
        10.0,
        10.0,
        0.0,
        2.0,
        350.0
    )
    .is_err());
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &[0.0, 0.0],
        &r2,
        &d2,
        &z2,
        &z2,
        10.0,
        10.0,
        0.0,
        2.0,
        350.0
    )
    .is_err());
    // n < 2 / n > 10000
    assert!(call(
        &ok_p,
        &[0],
        &[0],
        &ok_s,
        &ok_g,
        &[2200.0],
        &[300.0],
        &[0],
        &[0],
        10.0,
        10.0,
        0.0,
        2.0,
        350.0
    )
    .is_err());
    let big = vec![2200.0; 10_001];
    let bigd = vec![300.0; 10_001];
    let bigz = vec![0u64; 10_001];
    assert!(call(
        &ok_p, &ok_w, &ok_b, &ok_s, &ok_g, &big, &bigd, &bigz, &bigz, 10.0, 10.0, 0.0, 2.0, 350.0
    )
    .is_err());
    // init length mismatches: dev, games, lag each individually
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &ok_g,
        &r2,
        &[300.0],
        &z2,
        &z2,
        10.0,
        10.0,
        0.0,
        2.0,
        350.0
    )
    .is_err());
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &ok_g,
        &r2,
        &d2,
        &[0],
        &z2,
        10.0,
        10.0,
        0.0,
        2.0,
        350.0
    )
    .is_err());
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &ok_g,
        &r2,
        &d2,
        &z2,
        &[0],
        10.0,
        10.0,
        0.0,
        2.0,
        350.0
    )
    .is_err());
    // player index out of range / self-play
    assert!(call(
        &ok_p,
        &[2],
        &ok_b,
        &ok_s,
        &ok_g,
        &r2,
        &d2,
        &z2,
        &z2,
        10.0,
        10.0,
        0.0,
        2.0,
        350.0
    )
    .is_err());
    assert!(call(
        &ok_p,
        &[1],
        &[1],
        &ok_s,
        &ok_g,
        &r2,
        &d2,
        &z2,
        &z2,
        10.0,
        10.0,
        0.0,
        2.0,
        350.0
    )
    .is_err());
    // score / gamma
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &[1.5],
        &ok_g,
        &r2,
        &d2,
        &z2,
        &z2,
        10.0,
        10.0,
        0.0,
        2.0,
        350.0
    )
    .is_err());
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &[-0.5],
        &ok_g,
        &r2,
        &d2,
        &z2,
        &z2,
        10.0,
        10.0,
        0.0,
        2.0,
        350.0
    )
    .is_err());
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &[f64::NAN],
        &ok_g,
        &r2,
        &d2,
        &z2,
        &z2,
        10.0,
        10.0,
        0.0,
        2.0,
        350.0
    )
    .is_err());
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &[f64::INFINITY],
        &r2,
        &d2,
        &z2,
        &z2,
        10.0,
        10.0,
        0.0,
        2.0,
        350.0
    )
    .is_err());
    // init values
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &ok_g,
        &[f64::NAN, 2200.0],
        &d2,
        &z2,
        &z2,
        10.0,
        10.0,
        0.0,
        2.0,
        350.0
    )
    .is_err());
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &ok_g,
        &r2,
        &[0.0, 300.0],
        &z2,
        &z2,
        10.0,
        10.0,
        0.0,
        2.0,
        350.0
    )
    .is_err());
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &ok_g,
        &r2,
        &[400.0, 300.0],
        &z2,
        &z2,
        10.0,
        10.0,
        0.0,
        2.0,
        350.0
    )
    .is_err());
    // knobs
    assert!(call(
        &ok_p, &ok_w, &ok_b, &ok_s, &ok_g, &r2, &d2, &z2, &z2, -1.0, 10.0, 0.0, 2.0, 350.0
    )
    .is_err());
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &ok_g,
        &r2,
        &d2,
        &z2,
        &z2,
        f64::NAN,
        10.0,
        0.0,
        2.0,
        350.0
    )
    .is_err());
    assert!(call(
        &ok_p, &ok_w, &ok_b, &ok_s, &ok_g, &r2, &d2, &z2, &z2, 10.0, -1.0, 0.0, 2.0, 350.0
    )
    .is_err());
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &ok_g,
        &r2,
        &d2,
        &z2,
        &z2,
        10.0,
        f64::INFINITY,
        0.0,
        2.0,
        350.0
    )
    .is_err());
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &ok_g,
        &r2,
        &d2,
        &z2,
        &z2,
        10.0,
        10.0,
        f64::NAN,
        2.0,
        350.0
    )
    .is_err());
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &ok_g,
        &r2,
        &d2,
        &z2,
        &z2,
        10.0,
        10.0,
        0.0,
        f64::NAN,
        350.0
    )
    .is_err());
    assert!(
        call(&ok_p, &ok_w, &ok_b, &ok_s, &ok_g, &r2, &d2, &z2, &z2, 10.0, 10.0, 0.0, 2.0, 0.0)
            .is_err()
    );
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &ok_g,
        &r2,
        &d2,
        &z2,
        &z2,
        10.0,
        10.0,
        0.0,
        2.0,
        f64::NAN
    )
    .is_err());
    // negative bval / lambda are VALID (R does not restrict them);
    // cval = 0 and hval = 0 are VALID.
    assert!(call(
        &ok_p, &ok_w, &ok_b, &ok_s, &ok_g, &r2, &d2, &z2, &z2, 10.0, 10.0, -5.0, -2.0, 350.0
    )
    .is_ok());
    assert!(
        call(&ok_p, &ok_w, &ok_b, &ok_s, &ok_g, &r2, &d2, &z2, &z2, 0.0, 0.0, 0.0, 2.0, 350.0)
            .is_ok()
    );
    // init_dev exactly AT rdmax is VALID (R rejects only strictly greater)
    assert!(call(
        &ok_p,
        &ok_w,
        &ok_b,
        &ok_s,
        &ok_g,
        &r2,
        &[350.0, 300.0],
        &z2,
        &z2,
        10.0,
        10.0,
        0.0,
        2.0,
        350.0
    )
    .is_ok());
}

/// 500-rep Monte Carlo: random schedules with scores in {0, 0.5, 1} must
/// satisfy structural invariants read from crate outputs: finiteness,
/// deviations in (0, rdmax], per-player W + D + L == games (init_games
/// == 0 here; W/D/L are current-run tallies and every score is tallied),
/// lag == 0 for final-period participants, and reversed row order gives
/// identical output.
#[test]
#[ignore = "500-rep Monte Carlo; run explicitly"]
fn st_mc_500() {
    let mut rng = Lcg(20260131);
    for rep in 0..500 {
        let n = 3 + (rng.next_f64() * 6.0) as usize;
        let g = 2 + (rng.next_f64() * 18.0) as usize;
        let mut periods = Vec::with_capacity(g);
        let mut white = Vec::with_capacity(g);
        let mut black = Vec::with_capacity(g);
        let mut score = Vec::with_capacity(g);
        let mut gamma = Vec::with_capacity(g);
        for _ in 0..g {
            periods.push(1 + (rng.next_f64() * 4.0) as u64);
            let w = (rng.next_f64() * n as f64) as usize % n;
            let mut b = (rng.next_f64() * n as f64) as usize % n;
            if b == w {
                b = (b + 1) % n;
            }
            white.push(w);
            black.push(b);
            score.push([0.0, 0.5, 1.0][(rng.next_f64() * 3.0) as usize % 3]);
            gamma.push((rng.next_f64() - 0.5) * 60.0);
        }
        let ir = vec![2200.0; n];
        let id = vec![300.0; n];
        let zz = vec![0u64; n];
        let cval = rng.next_f64() * 20.0;
        let hval = rng.next_f64() * 20.0;
        let bval = (rng.next_f64() - 0.5) * 10.0;
        let lambda = (rng.next_f64() - 0.5) * 8.0;
        let r = stephenson_rating(
            &periods, &white, &black, &score, &gamma, &ir, &id, &zz, &zz, cval, hval, bval, lambda,
            350.0,
        )
        .unwrap_or_else(|e| panic!("rep {}: {}", rep, e));
        let last = *periods.iter().max().unwrap();
        for p in 0..n {
            assert!(r.ratings[p].is_finite(), "rep {}", rep);
            assert!(
                r.deviations[p] > 0.0 && r.deviations[p] <= 350.0 + 1e-9,
                "rep {}: dev {}",
                rep,
                r.deviations[p]
            );
            assert_eq!(
                r.wins[p] + r.draws[p] + r.losses[p],
                r.games[p],
                "rep {}: tally mismatch",
                rep
            );
            let in_last =
                (0..periods.len()).any(|k| periods[k] == last && (white[k] == p || black[k] == p));
            if in_last {
                assert_eq!(r.lag[p], 0, "rep {}: participant lag", rep);
            }
        }
        // order independence: periods permuted (descending blocks) with
        // within-period row order PRESERVED (float accumulation order
        // inside a period is part of the contract), identical output
        let mut uniq: Vec<u64> = periods.clone();
        uniq.sort_unstable();
        uniq.dedup();
        let mut rp = Vec::with_capacity(g);
        let mut rw = Vec::with_capacity(g);
        let mut rb = Vec::with_capacity(g);
        let mut rs = Vec::with_capacity(g);
        let mut rg = Vec::with_capacity(g);
        for &m in uniq.iter().rev() {
            for k in 0..g {
                if periods[k] == m {
                    rp.push(periods[k]);
                    rw.push(white[k]);
                    rb.push(black[k]);
                    rs.push(score[k]);
                    rg.push(gamma[k]);
                }
            }
        }
        let r2 = stephenson_rating(
            &rp, &rw, &rb, &rs, &rg, &ir, &id, &zz, &zz, cval, hval, bval, lambda, 350.0,
        )
        .unwrap();
        for p in 0..n {
            assert_eq!(r.ratings[p], r2.ratings[p], "rep {}: order dep", rep);
        }
    }
}

// ---------------------------------------------------------------------------
// elom_rating (PlayerRatings elom(); executed-oracle anchors E1-E9 from the
// Python port of the READ R driver + elom_c kernel + kriichi()).
// Every assert reads crate outputs (ElomResult fields).
// ---------------------------------------------------------------------------

/// E1: single 4-player event, all ratings equal (escore = 0), scalar
/// k = 0.5 -> ratings = 1500 + 0.5 * base. Exact.
#[test]
fn em_anchor_e1_scalar() {
    let r = elom_rating(
        &[1],
        &[0, 1, 2, 3],
        &[40.0, 20.0, -10.0, -50.0],
        &[30.0, 10.0, -10.0, -30.0],
        &[1500.0; 4],
        &[0; 4],
        &[0; 4],
        &[0; 16],
        ElomKFactor::Scalar(0.5),
        false,
    )
    .unwrap();
    assert_eq!(r.ratings, vec![1515.0, 1505.0, 1495.0, 1485.0]);
    assert_eq!(r.games, vec![1, 1, 1, 1]);
    assert_eq!(&r.places[0..4], &[1, 0, 0, 0]);
    assert_eq!(&r.places[12..16], &[0, 0, 0, 1]);
    assert_eq!(r.lag, vec![0, 0, 0, 0]);
}

/// E2: kriichi defaults (games = 0 -> k = 1 exactly), heterogeneous init
/// ratings -> escore = (r - 1500)/40 nonzero. Kills the /40-drop mutant
/// (MU1) and the post-period-games kriichi mutant (MU4: games 0 -> 1
/// after the period would give k = 0.998 != 1). Exact.
#[test]
fn em_anchor_e2_kriichi_het() {
    let r = elom_rating(
        &[1],
        &[0, 1, 2, 3],
        &[40.0, 20.0, -10.0, -50.0],
        &[30.0, 10.0, -10.0, -30.0],
        &[1600.0, 1500.0, 1400.0, 1500.0],
        &[0; 4],
        &[0; 4],
        &[0; 16],
        ElomKFactor::Kriichi { gv: 400.0, kv: 0.2 },
        false,
    )
    .unwrap();
    assert_eq!(r.ratings, vec![1627.5, 1510.0, 1392.5, 1470.0]);
}

/// E3: one empty seat among 5 indexed players; base shrinks once to
/// (30, 0, -30); avetab over the event's 3 participants only. Kills the
/// avetab-over-all-n mutant (MU2: including idle players 3 and 4 changes
/// avetab and escore). Exact; idle players untouched, lag stays 0 for
/// never-played players.
#[test]
fn em_anchor_e3_one_missing() {
    let r = elom_rating(
        &[1],
        &[0, 1, 2, -1],
        &[10.0, 5.0, 1.0, f64::NAN],
        &[30.0, 10.0, -10.0, -30.0],
        &[1500.0; 5],
        &[0; 5],
        &[0; 5],
        &[0; 20],
        ElomKFactor::Kriichi { gv: 400.0, kv: 0.2 },
        false,
    )
    .unwrap();
    assert_eq!(r.ratings, vec![1530.0, 1500.0, 1470.0, 1500.0, 1500.0]);
    assert_eq!(r.games, vec![1, 1, 1, 0, 0]);
    assert_eq!(r.lag, vec![0, 0, 0, 0, 0]);
    assert_eq!(&r.places[8..12], &[0, 0, 1, 0]);
    assert_eq!(&r.places[12..16], &[0, 0, 0, 0]);
}

/// E4: tied top scores use min ranks (1, 1, 3, 4) -> bases
/// (30, 30, -10, -30); rank 2 base (10) is skipped entirely. Kills the
/// ties.method = "average" mutant (MU5: average ranks 1.5 are not valid
/// min ranks and would change both bases and places). Exact.
#[test]
fn em_anchor_e4_ties_min() {
    let r = elom_rating(
        &[1],
        &[0, 1, 2, 3],
        &[10.0, 10.0, 5.0, 1.0],
        &[30.0, 10.0, -10.0, -30.0],
        &[1500.0; 4],
        &[0; 4],
        &[0; 4],
        &[0; 16],
        ElomKFactor::Scalar(0.25),
        false,
    )
    .unwrap();
    assert_eq!(r.ratings, vec![1507.5, 1507.5, 1497.5, 1492.5]);
    assert_eq!(&r.places[0..4], &[1, 0, 0, 0]);
    assert_eq!(&r.places[4..8], &[1, 0, 0, 0]);
    assert_eq!(&r.places[8..12], &[0, 0, 1, 0]);
}

/// E5: two periods with kriichi; player 3 idle in period 2 -> lag = 1.
/// Period 1 (games 0, k = 1): ratings become (1530, 1510, 1490, 1470).
/// Period 2 (3 seats, shrunk base (30, 0, -30), scores reversed): avetab
/// = 1510, dscore = (-30.5, 0, +30.5), k = 1 - 0.8/400. Expected values
/// are computed with the same f64 expression the oracle executed
/// (oracle hex pins 0x1.76e3e76c8b439p+10 / 0x1.7c1c189374bc7p+10).
#[test]
fn em_anchor_e5_lag_two_periods() {
    let r = elom_rating(
        &[1, 2],
        &[0, 1, 2, 3, 0, 1, 2, -1],
        &[4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, f64::NAN],
        &[30.0, 10.0, -10.0, -30.0],
        &[1500.0; 4],
        &[0; 4],
        &[0; 4],
        &[0; 16],
        ElomKFactor::Kriichi { gv: 400.0, kv: 0.2 },
        false,
    )
    .unwrap();
    let k2 = 1.0 - (1.0 - 0.2) * 1.0 / 400.0;
    assert_eq!(r.ratings[0], 1530.0 + k2 * (-30.5));
    assert_eq!(r.ratings[1], 1510.0);
    assert_eq!(r.ratings[2], 1490.0 + k2 * 30.5);
    assert_eq!(r.ratings[3], 1470.0);
    assert_eq!(r.games, vec![2, 2, 2, 1]);
    assert_eq!(r.lag, vec![0, 0, 0, 1]);
    assert_eq!(&r.places[0..4], &[1, 0, 1, 0]);
    assert_eq!(&r.places[4..8], &[0, 2, 0, 0]);
}

/// E6: placing = true (lower is better): placings 1..4 give ranks 1..4,
/// same outcome as E1. Kills a dropped-negation mutant (ranks would
/// reverse and places[0] would move to seat 3). Exact.
#[test]
fn em_anchor_e6_placing() {
    // Contrast: the same scores WITHOUT placing rank p3 best (reversed).
    let rev = elom_rating(
        &[1],
        &[0, 1, 2, 3],
        &[1.0, 2.0, 3.0, 4.0],
        &[30.0, 10.0, -10.0, -30.0],
        &[1500.0; 4],
        &[0; 4],
        &[0; 4],
        &[0; 16],
        ElomKFactor::Scalar(0.5),
        false,
    )
    .unwrap();
    assert_eq!(rev.ratings, vec![1485.0, 1495.0, 1505.0, 1515.0]);
    let r = elom_rating(
        &[1],
        &[0, 1, 2, 3],
        &[1.0, 2.0, 3.0, 4.0],
        &[30.0, 10.0, -10.0, -30.0],
        &[1500.0; 4],
        &[0; 4],
        &[0; 4],
        &[0; 16],
        ElomKFactor::Scalar(0.5),
        true,
    )
    .unwrap();
    assert_eq!(r.ratings, vec![1515.0, 1505.0, 1495.0, 1485.0]);
    assert_eq!(&r.places[0..4], &[1, 0, 0, 0]);
}

/// E7: continuation with prior games hitting the kriichi taper: p0 has
/// 400 prior games (k clamps to kv = 0.2 exactly), p1 has 200
/// (k = 1 - 0.8 * 200/400 = 0.6 exactly). All ratings equal -> dscore =
/// base. Prior lag values reset for all four participants. Exact.
#[test]
fn em_anchor_e7_kriichi_taper() {
    let r = elom_rating(
        &[1],
        &[0, 1, 2, 3],
        &[4.0, 3.0, 2.0, 1.0],
        &[30.0, 10.0, -10.0, -30.0],
        &[1500.0; 4],
        &[400, 200, 0, 0],
        &[3, 1, 0, 0],
        &[0; 16],
        ElomKFactor::Kriichi { gv: 400.0, kv: 0.2 },
        false,
    )
    .unwrap();
    assert_eq!(r.ratings, vec![1506.0, 1506.0, 1490.0, 1470.0]);
    assert_eq!(r.games, vec![401, 201, 1, 1]);
    assert_eq!(r.lag, vec![0, 0, 0, 0]);
}

/// E8: TWO empty seats (nan = 2, nn = 4). R tmpfun quirk (verified
/// verbatim: `sbase <- basev` is INSIDE the shrink loop): the base is
/// shrunk from the ORIGINAL exactly ONCE -> (30, 0, -30); ranks 1 and 2
/// index its front -> bases 30 and 0. Kills the cumulative-shrink mutant
/// (MU3: shrinking twice gives the odd-drop (30, -30) so p1 would get
/// -30, rating 1470). Exact.
#[test]
fn em_anchor_e8_shrink_quirk() {
    let r = elom_rating(
        &[1],
        &[0, 1, -1, -1],
        &[7.0, 3.0, f64::NAN, f64::NAN],
        &[30.0, 10.0, -10.0, -30.0],
        &[1500.0; 3],
        &[0; 3],
        &[0; 3],
        &[0; 12],
        ElomKFactor::Kriichi { gv: 400.0, kv: 0.2 },
        false,
    )
    .unwrap();
    assert_eq!(r.ratings, vec![1530.0, 1500.0, 1500.0]);
    assert_eq!(r.games, vec![1, 1, 0]);
}

/// E9: two events in the SAME period sharing players 0 and 3 with
/// heterogeneous ratings, scalar k = 0.5. ascore/escore accumulate
/// across both events and ratings update ONCE per period (elom_c
/// contract). Kills the update-after-each-event mutant (MU6: updating
/// after event 1 changes event 2's avetab and escore). Exact
/// (all-dyadic arithmetic; oracle E9).
#[test]
fn em_anchor_e9_aggregate_period() {
    let r = elom_rating(
        &[1, 1],
        &[0, 1, 2, 3, 0, 4, 5, 3],
        &[4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0],
        &[30.0, 10.0, -10.0, -30.0],
        &[1600.0, 1500.0, 1400.0, 1500.0, 1550.0, 1450.0],
        &[0; 6],
        &[0; 6],
        &[0; 24],
        ElomKFactor::Scalar(0.5),
        false,
    )
    .unwrap();
    assert_eq!(
        r.ratings,
        vec![1597.8125, 1505.0, 1396.25, 1500.3125, 1544.6875, 1455.9375]
    );
    assert_eq!(r.games, vec![2, 1, 1, 2, 1, 1]);
    assert_eq!(&r.places[0..4], &[1, 0, 0, 1]);
    assert_eq!(&r.places[12..16], &[1, 0, 0, 1]);
}

/// Error contract: every rejection path returns Err (never panics).
#[test]
fn em_error_contract() {
    let base = [30.0, 10.0, -10.0, -30.0];
    let ok_players = [0i64, 1, 2, 3];
    let ok_scores = [4.0, 3.0, 2.0, 1.0];
    let init = [1500.0; 4];
    let z4 = [0u64; 4];
    let z16 = [0u64; 16];
    let k = ElomKFactor::Scalar(0.5);
    // n bounds.
    assert!(elom_rating(
        &[1],
        &[0, -1, -1, -1],
        &[1.0, f64::NAN, f64::NAN, f64::NAN],
        &base,
        &[1500.0],
        &[0],
        &[0],
        &[0; 4],
        k,
        false
    )
    .is_err());
    // nn bounds.
    assert!(elom_rating(&[1], &[0], &[1.0], &[30.0], &init, &z4, &z4, &z4, k, false).is_err());
    // non-finite base.
    assert!(elom_rating(
        &[1],
        &ok_players,
        &ok_scores,
        &[30.0, f64::NAN, -10.0, -30.0],
        &init,
        &z4,
        &z4,
        &z16,
        k,
        false
    )
    .is_err());
    // empty schedule.
    assert!(elom_rating(&[], &[], &[], &base, &init, &z4, &z4, &z16, k, false).is_err());
    // players/scores length mismatch.
    assert!(elom_rating(
        &[1],
        &ok_players[..3],
        &ok_scores,
        &base,
        &init,
        &z4,
        &z4,
        &z16,
        k,
        false
    )
    .is_err());
    // non-finite init rating.
    assert!(elom_rating(
        &[1],
        &ok_players,
        &ok_scores,
        &base,
        &[f64::NAN, 1500.0, 1500.0, 1500.0],
        &z4,
        &z4,
        &z16,
        k,
        false
    )
    .is_err());
    // init_games/init_lag length.
    assert!(elom_rating(
        &[1],
        &ok_players,
        &ok_scores,
        &base,
        &init,
        &[0; 3],
        &z4,
        &z16,
        k,
        false
    )
    .is_err());
    // init_places length.
    assert!(elom_rating(
        &[1],
        &ok_players,
        &ok_scores,
        &base,
        &init,
        &z4,
        &z4,
        &[0; 15],
        k,
        false
    )
    .is_err());
    // decreasing periods.
    assert!(elom_rating(
        &[2, 1],
        &[0, 1, 2, 3, 0, 1, 2, 3],
        &[4.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0],
        &base,
        &init,
        &z4,
        &z4,
        &z16,
        k,
        false
    )
    .is_err());
    // u64 overflow guards.
    assert!(elom_rating(
        &[1],
        &ok_players,
        &ok_scores,
        &base,
        &init,
        &[u64::MAX, 0, 0, 0],
        &z4,
        &z16,
        k,
        false
    )
    .is_err());
    assert!(elom_rating(
        &[1],
        &ok_players,
        &ok_scores,
        &base,
        &init,
        &z4,
        &[0, u64::MAX, 0, 0],
        &z16,
        k,
        false
    )
    .is_err());
    let mut bad_places = [0u64; 16];
    bad_places[5] = u64::MAX;
    assert!(elom_rating(
        &[1],
        &ok_players,
        &ok_scores,
        &base,
        &init,
        &z4,
        &z4,
        &bad_places,
        k,
        false
    )
    .is_err());
    // K-factor validation.
    assert!(elom_rating(
        &[1],
        &ok_players,
        &ok_scores,
        &base,
        &init,
        &z4,
        &z4,
        &z16,
        ElomKFactor::Scalar(0.0),
        false
    )
    .is_err());
    assert!(elom_rating(
        &[1],
        &ok_players,
        &ok_scores,
        &base,
        &init,
        &z4,
        &z4,
        &z16,
        ElomKFactor::Scalar(f64::NAN),
        false
    )
    .is_err());
    assert!(elom_rating(
        &[1],
        &ok_players,
        &ok_scores,
        &base,
        &init,
        &z4,
        &z4,
        &z16,
        ElomKFactor::Kriichi { gv: 0.0, kv: 0.2 },
        false
    )
    .is_err());
    assert!(elom_rating(
        &[1],
        &ok_players,
        &ok_scores,
        &base,
        &init,
        &z4,
        &z4,
        &z16,
        ElomKFactor::Kriichi { gv: 400.0, kv: 1.5 },
        false
    )
    .is_err());
    assert!(elom_rating(
        &[1],
        &ok_players,
        &ok_scores,
        &base,
        &init,
        &z4,
        &z4,
        &z16,
        ElomKFactor::Kriichi { gv: 400.0, kv: 0.0 },
        false
    )
    .is_err());
    // Empty seat with finite score (missing-seat contract).
    assert!(elom_rating(
        &[1],
        &[0, 1, 2, -1],
        &[4.0, 3.0, 2.0, 1.0],
        &base,
        &init,
        &z4,
        &z4,
        &z16,
        k,
        false
    )
    .is_err());
    // Occupied seat with NaN score.
    assert!(elom_rating(
        &[1],
        &ok_players,
        &[4.0, 3.0, 2.0, f64::NAN],
        &base,
        &init,
        &z4,
        &z4,
        &z16,
        k,
        false
    )
    .is_err());
    // Player index out of range / below -1.
    assert!(elom_rating(
        &[1],
        &[0, 1, 2, 4],
        &ok_scores,
        &base,
        &init,
        &z4,
        &z4,
        &z16,
        k,
        false
    )
    .is_err());
    assert!(elom_rating(
        &[1],
        &[0, 1, 2, -2],
        &ok_scores,
        &base,
        &init,
        &z4,
        &z4,
        &z16,
        k,
        false
    )
    .is_err());
    // Duplicate player within one event.
    assert!(elom_rating(
        &[1],
        &[0, 1, 2, 0],
        &ok_scores,
        &base,
        &init,
        &z4,
        &z4,
        &z16,
        k,
        false
    )
    .is_err());
    // Too many empty seats (nan > nn - 2).
    assert!(elom_rating(
        &[1],
        &[0, -1, -1, -1],
        &[1.0, f64::NAN, f64::NAN, f64::NAN],
        &base,
        &init,
        &z4,
        &z4,
        &z16,
        k,
        false
    )
    .is_err());
    // nn = 2 complete events are ACCEPTED (rev-2 finding 2).
    let r2 = elom_rating(
        &[1],
        &[0, 1],
        &[2.0, 1.0],
        &[16.0, -16.0],
        &[1500.0, 1500.0],
        &[0; 2],
        &[0; 2],
        &[0; 4],
        ElomKFactor::Scalar(1.0),
        false,
    )
    .unwrap();
    assert_eq!(r2.ratings, vec![1516.0, 1484.0]);
}

/// Monte-Carlo structural invariants over 500 random schedules:
/// (a) per-player place counts sum to games gained this run,
/// (b) permuting event order WITHIN a period leaves ratings unchanged
///     up to floating-point summation order (1e-9 relative),
/// (c) all outputs finite, lag bounded by the period count.
#[test]
#[ignore]
fn em_mc_500() {
    let mut rng = Lcg(0xE10E_2024);
    let nn = 4usize;
    let base = [30.0, 10.0, -10.0, -30.0];
    for rep in 0..500 {
        let n = 4 + (rng.next_f64() * 5.0) as usize; // 4..=8
        let g = 1 + (rng.next_f64() * 5.0) as usize; // 1..=5
        let mut periods = Vec::with_capacity(g);
        let mut cur = 1u64;
        for _ in 0..g {
            if rng.next_f64() < 0.5 {
                cur += 1;
            }
            periods.push(cur);
        }
        let mut players = Vec::with_capacity(g * nn);
        let mut scores = Vec::with_capacity(g * nn);
        for _ in 0..g {
            // Random distinct participants; 0-2 empty tail seats.
            let nan = (rng.next_f64() * 3.0) as usize; // 0..=2
            let mut pool: Vec<i64> = (0..n as i64).collect();
            for j in 0..nn - nan {
                let pick = j + (rng.next_f64() * (pool.len() - j) as f64) as usize;
                pool.swap(j, pick);
            }
            for j in 0..nn {
                if j < nn - nan {
                    players.push(pool[j]);
                    scores.push((rng.next_f64() * 100.0).round() + j as f64 / 8.0);
                } else {
                    players.push(-1);
                    scores.push(f64::NAN);
                }
            }
        }
        let init: Vec<f64> = (0..n).map(|_| 1400.0 + rng.next_f64() * 200.0).collect();
        let kfac = if rep % 2 == 0 {
            ElomKFactor::Scalar(0.25 + rng.next_f64())
        } else {
            ElomKFactor::Kriichi { gv: 400.0, kv: 0.2 }
        };
        let r = elom_rating(
            &periods,
            &players,
            &scores,
            &base,
            &init,
            &vec![0u64; n],
            &vec![0u64; n],
            &vec![0u64; n * nn],
            kfac,
            false,
        )
        .unwrap();
        let n_periods = {
            let mut c = 1u64;
            for w in periods.windows(2) {
                if w[1] != w[0] {
                    c += 1;
                }
            }
            c
        };
        for p in 0..n {
            assert!(r.ratings[p].is_finite(), "rep {rep} nonfinite rating");
            let place_sum: u64 = r.places[p * nn..(p + 1) * nn].iter().sum();
            assert_eq!(place_sum, r.games[p], "rep {rep} places/games mismatch");
            assert!(r.lag[p] <= n_periods, "rep {rep} lag exceeds period count");
        }
        // (b) reverse event order within each period block.
        let mut order: Vec<usize> = (0..g).collect();
        let mut s = 0usize;
        while s < g {
            let mut e = s + 1;
            while e < g && periods[e] == periods[s] {
                e += 1;
            }
            order[s..e].reverse();
            s = e;
        }
        let players2: Vec<i64> = order
            .iter()
            .flat_map(|&e| players[e * nn..(e + 1) * nn].iter().copied())
            .collect();
        let scores2: Vec<f64> = order
            .iter()
            .flat_map(|&e| scores[e * nn..(e + 1) * nn].iter().copied())
            .collect();
        let r2 = elom_rating(
            &periods,
            &players2,
            &scores2,
            &base,
            &init,
            &vec![0u64; n],
            &vec![0u64; n],
            &vec![0u64; n * nn],
            kfac,
            false,
        )
        .unwrap();
        for p in 0..n {
            let diff = (r.ratings[p] - r2.ratings[p]).abs();
            assert!(
                diff <= 1e-9 * r.ratings[p].abs().max(1.0),
                "rep {rep} player {p}: within-period order changed rating by {diff}"
            );
            assert_eq!(r.games[p], r2.games[p], "rep {rep} games order-dependent");
            assert_eq!(r.lag[p], r2.lag[p], "rep {rep} lag order-dependent");
        }
    }
}

// ---------------------------------------------------------------------------
// metrics_rating (PlayerRatings metrics(), R/ratings.R 936-957) -- mr_ tests
//
// Every assert reads values returned by the crate fn `metrics_rating`.
// Anchor pins come from an exact-Fraction oracle executed against the R
// source semantics (files/metrics_oracle.py, session-external), with the
// M5 column pins additionally re-derived by hand (bdev = 1.5 ln 2,
// mse = sqrt(13/32), mae = 5/8).
//
// Mutation-kill map (all EXECUTED during development):
//   MU1 cap applied to mse/mae      -> mr_anchor_m2_cap_quirk (mutant mae 1.0)
//   MU2 bdev uses uncapped pred     -> mr_anchor_m2_cap_quirk (mutant bdev 0.1000500...)
//   MU3 stride pred[j*nr+i]         -> mr_anchor_m5_columns (both cols shift)
//   MU4 missing sqrt / wrong scale constant / missing x100 -> mr_anchor_m3_scaled
//   MU5 baseline uses pair-removal row set -> mr_anchor_m6_baseline_rows
// KNOWN-UNOBSERVABLE: algebraic rearrangements such as
// sqrt(a)/sqrt(b) == sqrt(a/b) cannot be distinguished by any value test
// (identical reals, sub-ulp float differences at most); the discriminating
// anchors above pin observable semantics only. The bdev baseline is the
// constant ln 2 for every finite act, so bdev cannot witness baseline
// ROW-SET mutations (mse/mae in mr_anchor_m6_baseline_rows do).

fn mr_rel_close(got: f64, want: f64, tol: f64) -> bool {
    (got - want).abs() <= tol * want.abs().max(1.0)
}

#[test]
fn mr_anchor_m1_unscaled() {
    let act = [1.0, 0.0, 1.0, 0.0];
    let pred = [0.75, 0.25, 0.5, 0.5];
    let out = metrics_rating(&act, &pred, 4, 1, (0.01, 0.99), false).unwrap();
    assert_eq!(out.len(), 3);
    assert!(
        mr_rel_close(out[0], 49.041462650586311, 1e-13),
        "bdev {}",
        out[0]
    );
    assert!(
        mr_rel_close(out[1], 39.528470752104745, 1e-13),
        "mse {}",
        out[1]
    );
    // mae is exact in dyadic arithmetic: mean(1/4,1/4,1/2,1/2)*100 = 37.5.
    assert_eq!(out[2], 37.5, "mae {}", out[2]);
}

#[test]
fn mr_anchor_m2_cap_quirk() {
    // pred outside the cap: bdev must use the CAPPED values (-100*ln 0.99)
    // while mse/mae use the RAW 0.999/0.001 (R:949, 951 reference pred[,i]).
    let act = [1.0, 0.0];
    let pred = [0.999, 0.001];
    let out = metrics_rating(&act, &pred, 2, 1, (0.01, 0.99), false).unwrap();
    assert!(
        mr_rel_close(out[0], 1.0050335853501451, 1e-13),
        "bdev {}",
        out[0]
    );
    // Mutant values: cap-on-mse/mae gives 1.0000000000000004 / 1.0;
    // uncapped bdev gives 0.10005003335835344.
    assert!(
        mr_rel_close(out[1], 0.10000000000000001, 1e-13),
        "mse {}",
        out[1]
    );
    assert!(
        mr_rel_close(out[2], 0.10000000000000001, 1e-13),
        "mae {}",
        out[2]
    );
}

#[test]
fn mr_anchor_m3_scaled() {
    // scale = true divides by the 0.5-baseline and multiplies by 100:
    // missing sqrt, a wrong bdev constant, or a dropped x100 all shift
    // these pins (MU4).
    let act = [1.0, 0.0, 1.0];
    let pred = [0.8, 0.4, 0.6];
    let out = metrics_rating(&act, &pred, 3, 1, (0.01, 0.99), true).unwrap();
    assert!(
        mr_rel_close(out[0], 59.86197610732583, 1e-13),
        "bdev {}",
        out[0]
    );
    // mse^2 ratio is exactly 12/25 -> 100*sqrt(12/25).
    assert!(
        mr_rel_close(out[1], 69.282032302755098, 1e-13),
        "mse {}",
        out[1]
    );
    // mae = 100*(1/5)/(3/10)... = 100*2/3 exactly in the rationals.
    assert!(
        mr_rel_close(out[2], 66.666666666666671, 1e-13),
        "mae {}",
        out[2]
    );
}

#[test]
fn mr_anchor_m4_nan() {
    // Numerators drop rows where EITHER act or pred is NaN (rows 1, 2);
    // scaled baselines drop only act-NaN rows (rows 0, 1, 3 remain).
    let act = [1.0, 0.0, f64::NAN, 1.0];
    let pred = [0.7, f64::NAN, 0.4, 0.6];
    let out = metrics_rating(&act, &pred, 4, 1, (0.01, 0.99), true).unwrap();
    assert!(
        mr_rel_close(out[0], 62.576938349798226, 1e-13),
        "bdev {}",
        out[0]
    );
    // mse^2 ratio exactly 1/2 -> 100/sqrt(2).
    assert!(
        mr_rel_close(out[1], 70.710678118654755, 1e-13),
        "mse {}",
        out[1]
    );
    // mae = 100*(7/20)/(1/2) = 70 exactly in the rationals.
    assert!(mr_rel_close(out[2], 70.0, 1e-13), "mae {}", out[2]);
}

#[test]
fn mr_anchor_m6_baseline_rows() {
    // MU5 killer: act = 0.2 in an act-only baseline row (row 3) plus a
    // pred-NaN row (row 1) makes pair-removal baselines diverge:
    // pair-removal mutant gives mse = 85.749292571254429,
    // mae = 87.5; the true act-row-set values are pinned below
    // (mae = 100*1050/130... = 1050/13 exactly, mse^2 ratio = 75/118).
    let act = [1.0, 0.0, f64::NAN, 0.2];
    let pred = [0.7, f64::NAN, 0.4, 0.6];
    let out = metrics_rating(&act, &pred, 4, 1, (0.01, 0.99), true).unwrap();
    assert!(
        mr_rel_close(out[0], 85.975438378644469, 1e-13),
        "bdev {}",
        out[0]
    );
    assert!(
        mr_rel_close(out[1], 79.724100517910088, 1e-13),
        "mse {}",
        out[1]
    );
    assert!(
        mr_rel_close(out[2], 80.769230769230774, 1e-13),
        "mae {}",
        out[2]
    );
}

#[test]
fn mr_anchor_m5_columns() {
    // Two-column row-major layout: pred[i * np + j]. A stride mutant
    // (pred[j * nr + i]) moves BOTH columns to (76.5067..., 53.0330..., 50).
    let act = [1.0, 0.0, 1.0, 0.0];
    #[rustfmt::skip]
    let pred = [
        0.75, 0.50,
        0.25, 0.50,
        0.50, 0.25,
        0.50, 0.75,
    ];
    let out = metrics_rating(&act, &pred, 4, 2, (0.01, 0.99), false).unwrap();
    assert_eq!(out.len(), 6);
    // Column 0 == M1.
    assert!(
        mr_rel_close(out[0], 49.041462650586311, 1e-13),
        "c0 bdev {}",
        out[0]
    );
    assert!(
        mr_rel_close(out[1], 39.528470752104745, 1e-13),
        "c0 mse {}",
        out[1]
    );
    assert_eq!(out[2], 37.5, "c0 mae {}", out[2]);
    // Column 1: bdev = 100*1.5*ln 2, mse = 100*sqrt(13/32), mae = 62.5 exact.
    assert!(
        mr_rel_close(out[3], 103.97207708399179, 1e-13),
        "c1 bdev {}",
        out[3]
    );
    assert!(
        mr_rel_close(out[4], 63.737743919909803, 1e-13),
        "c1 mse {}",
        out[4]
    );
    assert_eq!(out[5], 62.5, "c1 mae {}", out[5]);
}

#[test]
fn mr_error_contract() {
    let act = [1.0, 0.0];
    let pred = [0.6, 0.4];
    // Dimension bounds and length mismatches.
    assert!(metrics_rating(&act, &pred, 0, 1, (0.01, 0.99), false).is_err());
    assert!(metrics_rating(&act, &pred, 2, 0, (0.01, 0.99), false).is_err());
    assert!(metrics_rating(&act, &pred, 10_000_001, 1, (0.01, 0.99), false).is_err());
    assert!(metrics_rating(&act, &pred, 2, 10_001, (0.01, 0.99), false).is_err());
    assert!(metrics_rating(&act[..1], &pred, 2, 1, (0.01, 0.99), false).is_err());
    assert!(metrics_rating(&act, &pred[..1], 2, 1, (0.01, 0.99), false).is_err());
    // Inf rejected anywhere (NaN is the missing marker).
    assert!(metrics_rating(&[1.0, f64::INFINITY], &pred, 2, 1, (0.01, 0.99), false).is_err());
    assert!(metrics_rating(&act, &[0.6, f64::NEG_INFINITY], 2, 1, (0.01, 0.99), false).is_err());
    // Cap domain: 0 < lo <= hi < 1, finite.
    assert!(metrics_rating(&act, &pred, 2, 1, (0.0, 0.99), false).is_err());
    assert!(metrics_rating(&act, &pred, 2, 1, (0.01, 1.0), false).is_err());
    assert!(metrics_rating(&act, &pred, 2, 1, (0.6, 0.4), false).is_err());
    assert!(metrics_rating(&act, &pred, 2, 1, (f64::NAN, 0.99), false).is_err());
    // Empty per-column row set after NaN removal.
    assert!(metrics_rating(&act, &[f64::NAN, f64::NAN], 2, 1, (0.01, 0.99), false).is_err());
    assert!(metrics_rating(&[f64::NAN, f64::NAN], &pred, 2, 1, (0.01, 0.99), false).is_err());
    // scale = true with a degenerate all-0.5 act baseline (R yields Inf/NaN).
    assert!(metrics_rating(&[0.5, 0.5], &pred, 2, 1, (0.01, 0.99), true).is_err());
    // ... but the same inputs are fine unscaled.
    assert!(metrics_rating(&[0.5, 0.5], &pred, 2, 1, (0.01, 0.99), false).is_ok());
}

#[test]
#[ignore = "Monte-Carlo, 500 replications; run with -- --ignored"]
fn mr_mc_500() {
    // Preconditions: act in {0, 1} (baselines are then exactly 0.25 / 0.5,
    // finite and nonzero), pred in (0.01, 0.99) so capping is inactive and
    // every row is a valid pair.
    let mut rng = Lcg(0x5EED_AE7A_1C5F_0001);
    for rep in 0..500 {
        let nr = 5 + (rng.next_f64() * 40.0) as usize;
        let np = 1 + (rng.next_f64() * 4.0) as usize;
        let act: Vec<f64> = (0..nr)
            .map(|_| if rng.next_f64() < 0.5 { 0.0 } else { 1.0 })
            .collect();
        let pred: Vec<f64> = (0..nr * np).map(|_| 0.02 + 0.96 * rng.next_f64()).collect();
        let raw = metrics_rating(&act, &pred, nr, np, (0.01, 0.99), false).unwrap();
        let scaled = metrics_rating(&act, &pred, nr, np, (0.01, 0.99), true).unwrap();
        // Invariant 1: scaled = unscaled / column-constant baseline.
        // bdev ratio is exactly ln 2; mse/mae ratios are column-constant.
        for j in 0..np {
            let r_bdev = raw[3 * j] / scaled[3 * j];
            assert!(
                mr_rel_close(r_bdev, std::f64::consts::LN_2, 1e-12),
                "rep {rep} col {j}: bdev ratio {r_bdev}"
            );
            let r_mse = raw[3 * j + 1] / scaled[3 * j + 1];
            let r_mae = raw[3 * j + 2] / scaled[3 * j + 2];
            let r_mse0 = raw[1] / scaled[1];
            let r_mae0 = raw[2] / scaled[2];
            assert!(
                mr_rel_close(r_mse, r_mse0, 1e-12),
                "rep {rep} col {j}: mse ratio {r_mse} vs {r_mse0}"
            );
            assert!(
                mr_rel_close(r_mae, r_mae0, 1e-12),
                "rep {rep} col {j}: mae ratio {r_mae} vs {r_mae0}"
            );
        }
        // Invariant 2: permuting COLUMNS permutes the output rows exactly
        // (per-column row-accumulation order is unchanged by a column
        // permutation, so this is bitwise, not tolerance-based).
        if np > 1 {
            let mut perm_pred = vec![0.0f64; nr * np];
            for i in 0..nr {
                for j in 0..np {
                    // rotate columns left by 1
                    perm_pred[i * np + j] = pred[i * np + (j + 1) % np];
                }
            }
            let perm = metrics_rating(&act, &perm_pred, nr, np, (0.01, 0.99), false).unwrap();
            for j in 0..np {
                let src = (j + 1) % np;
                for k in 0..3 {
                    assert_eq!(
                        perm[3 * j + k],
                        raw[3 * src + k],
                        "rep {rep} col {j} metric {k}: column permutation not exact"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// fide_rating tests (fd_ prefix) — PlayerRatings fide() port.
//
// All asserts read crate outputs (FideResult fields returned by
// fide_rating). Oracle: independently executed transcription of the READ
// R source (fide() lines 125-272, kfide() 959-972), exact-Fraction anchor
// F1 plus float pins F2-F5, adversarially spec-reviewed.
//
// Mutation-kill map (all mutants EXECUTED before the PR):
// - MU1 K from post-period games          -> fd_k_switch_f2
// - MU2 elite recomputed (non-sticky)     -> fd_elite_sticky_f3
// - MU3 opponent from PRE-update ratings  -> fd_anchor_f1_exact, fd_opponent_avg_f4
// - MU4 opponent running-average weights swapped -> fd_opponent_weights_f5
// - MU5 kfide branch swap (kv.1 <-> kv.2) -> fd_anchor_f1_exact, fd_k_switch_f2
// Documented unkillable mutant: E_b = 1 - E_w refactor (identity
// E_w + E_b = 1, proved in the elo_rating header; no black-box test can
// distinguish it).
// ---------------------------------------------------------------------------

fn fd_rel_close(a: f64, b: f64, tol: f64) -> bool {
    if a == b {
        return true;
    }
    (a - b).abs() <= tol * a.abs().max(b.abs())
}

/// F1: n=2, one game, equal init 2200, gamma 0 -> E = 1/2 exactly; fresh
/// non-elite players must get K = kv.2 = 30 (kills MU5: a kv.1<->kv.2
/// swap gives 2207.5/2192.5). Opponent must be the POST-update opposite
/// rating (kills MU3: pre-update would give 2200/2200). All values dyadic
/// -> exact equality.
#[test]
fn fd_anchor_f1_exact() {
    let r = fide_rating(
        &[1],
        &[0],
        &[1],
        &[1.0],
        &[0.0],
        2,
        2200.0,
        (10.0, 15.0, 30.0),
    )
    .unwrap();
    assert_eq!(r.ratings, vec![2215.0, 2185.0]);
    assert_eq!(r.opponent, vec![2185.0, 2215.0]);
    assert_eq!(r.games, vec![1, 1]);
    assert_eq!(r.wins, vec![1, 0]);
    assert_eq!(r.draws, vec![0, 0]);
    assert_eq!(r.losses, vec![0, 1]);
    assert_eq!(r.lag, vec![0, 0]);
    assert_eq!(r.elite, vec![0, 0]);
}

/// F2: init 1500 (elite unreachable), p0 plays 29 games in period 1 then
/// 1 game in period 2. Period-2 K for p0 uses PERIOD-START games = 29
/// -> K = 30; a mutant reading post-update games sees 30 -> K = 15
/// (kills MU1). Pins from the executed oracle.
#[test]
fn fd_k_switch_f2() {
    let mut periods = vec![1u64; 29];
    periods.push(2);
    let mut white = vec![0usize; 30];
    let _ = &mut white; // all games have white = 0
    let mut black: Vec<usize> = (1..30).collect();
    black.push(1);
    let score = vec![1.0f64; 30];
    let gamma = vec![0.0f64; 30];
    let r = fide_rating(
        &periods,
        &white,
        &black,
        &score,
        &gamma,
        31,
        1500.0,
        (10.0, 15.0, 30.0),
    )
    .unwrap();
    assert!(fd_rel_close(r.ratings[0], 1937.0927486207672, 1e-15));
    assert!(fd_rel_close(r.ratings[1], 1482.9072513792328, 1e-15));
    assert_eq!(r.ratings[2], 1485.0); // dyadic: 1500 - 30/2
    assert_eq!(r.ratings[30], 1500.0); // never played
    assert!(fd_rel_close(r.opponent[0], 1484.9302417126412, 1e-15));
    assert!(fd_rel_close(r.opponent[1], 1936.0463743103837, 1e-15));
    assert_eq!(r.opponent[30], 0.0);
    assert_eq!(r.games[0], 30);
    assert_eq!(r.games[1], 2);
    assert_eq!(r.wins[0], 30);
    assert_eq!(r.losses[1], 2);
    assert_eq!(r.lag[2], 1); // played period 1 only
    assert_eq!(r.lag[30], 0); // never played
    assert_eq!(r.elite, vec![0u8; 31]);
}

/// F3: init 2395; p0 wins period 1 (rating 2410 >= 2400 -> elite), then
/// loses periods 2-3 (dropping below 2400), wins period 4. The sticky
/// elite flag keeps K = 10 for p0 in period 4 even though its rating is
/// then < 2400; a mutant recomputing elite each period gives K = 30 and
/// a different final rating (kills MU2). p1 also crosses 2400 by period
/// 3; both end elite. Pins from the executed oracle.
#[test]
fn fd_elite_sticky_f3() {
    let periods = [1u64, 2, 3, 4];
    let white = [0usize, 0, 0, 0];
    let black = [1usize, 1, 1, 1];
    let score = [1.0f64, 0.0, 0.0, 1.0];
    let gamma = [0.0f64; 4];
    let r = fide_rating(
        &periods,
        &white,
        &black,
        &score,
        &gamma,
        2,
        2395.0,
        (10.0, 15.0, 30.0),
    )
    .unwrap();
    assert!(fd_rel_close(r.ratings[0], 2404.6257234642635, 1e-15));
    assert!(fd_rel_close(r.ratings[1], 2406.4738023175405, 1e-15));
    assert_eq!(r.elite, vec![1, 1]);
    assert!(fd_rel_close(r.opponent[0], 2398.603771437728, 1e-15));
    assert!(fd_rel_close(r.opponent[1], 2404.661323913285, 1e-15));
    assert_eq!(r.games, vec![4, 4]);
    assert_eq!(r.wins, vec![2, 2]);
    assert_eq!(r.losses, vec![2, 2]);
    assert_eq!(r.lag, vec![0, 0]);
}

/// F4: n=3; period 1: p0 beats p1; period 2: p0 beats p2 and p1 beats
/// p2. Opponent values are means of POST-update ratings (kills MU3):
/// p0 = (2185 + 2170)/2 = 2177.5, p1 = (2215 + 2170)/2 = 2192.5,
/// p2 = (post_p0 + post_p1)/2 = 2215 exactly (E-sum identity makes p2's
/// dscore exactly -1). Pins from the executed oracle.
#[test]
fn fd_opponent_avg_f4() {
    let periods = [1u64, 2, 2];
    let white = [0usize, 0, 1];
    let black = [1usize, 2, 2];
    let score = [1.0f64, 1.0, 1.0];
    let gamma = [0.0f64; 3];
    let r = fide_rating(
        &periods,
        &white,
        &black,
        &score,
        &gamma,
        3,
        2200.0,
        (10.0, 15.0, 30.0),
    )
    .unwrap();
    assert!(fd_rel_close(r.ratings[0], 2229.3528000084657, 1e-15));
    assert!(fd_rel_close(r.ratings[1], 2200.6471999915343, 1e-15));
    assert_eq!(r.ratings[2], 2170.0); // dyadic via E-sum identity
    assert_eq!(r.opponent[0], 2177.5);
    assert_eq!(r.opponent[1], 2192.5);
    assert_eq!(r.opponent[2], 2215.0);
    assert_eq!(r.games, vec![2, 2, 2]);
}

/// F5: p0 plays one game in each of periods 1..3 (vs p1, p2, p3). At
/// period 3 p0 has prior games = 2, current = 1, prior opponent nonzero
/// -> correct old-value weight 2/3 vs 1/3 under a weight-swap mutant
/// (kills MU4). Pins from the executed oracle.
#[test]
fn fd_opponent_weights_f5() {
    let periods = [1u64, 2, 3];
    let white = [0usize, 0, 0];
    let black = [1usize, 2, 3];
    let score = [1.0f64, 1.0, 0.5];
    let gamma = [0.0f64; 3];
    let r = fide_rating(
        &periods,
        &white,
        &black,
        &score,
        &gamma,
        4,
        2200.0,
        (10.0, 15.0, 30.0),
    )
    .unwrap();
    assert!(fd_rel_close(r.ratings[0], 2228.088544238428, 1e-15));
    assert!(fd_rel_close(r.opponent[0], 2190.6371519205236, 1e-15));
    assert!(fd_rel_close(r.opponent[3], 2228.088544238428, 1e-15));
    assert_eq!(r.opponent[1], 2215.0); // p1's only opponent, post-P1
    assert_eq!(r.games, vec![3, 1, 1, 1]);
    assert_eq!(r.draws, vec![1, 0, 0, 1]);
    assert_eq!(r.lag, vec![0, 2, 1, 0]);
}

/// Error contract: every documented rejection returns Err.
#[test]
fn fd_error_contract() {
    let ok = (
        vec![1u64],
        vec![0usize],
        vec![1usize],
        vec![1.0f64],
        vec![0.0f64],
    );
    // empty
    assert!(fide_rating(&[], &[], &[], &[], &[], 2, 1500.0, (10.0, 15.0, 30.0)).is_err());
    // length mismatch
    assert!(fide_rating(
        &ok.0,
        &[0, 1],
        &ok.2,
        &ok.3,
        &ok.4,
        2,
        1500.0,
        (10.0, 15.0, 30.0)
    )
    .is_err());
    // n too small / too large
    assert!(fide_rating(
        &ok.0,
        &ok.1,
        &ok.2,
        &ok.3,
        &ok.4,
        1,
        1500.0,
        (10.0, 15.0, 30.0)
    )
    .is_err());
    assert!(fide_rating(
        &ok.0,
        &ok.1,
        &ok.2,
        &ok.3,
        &ok.4,
        10_001,
        1500.0,
        (10.0, 15.0, 30.0)
    )
    .is_err());
    // non-finite init / kv
    assert!(fide_rating(
        &ok.0,
        &ok.1,
        &ok.2,
        &ok.3,
        &ok.4,
        2,
        f64::NAN,
        (10.0, 15.0, 30.0)
    )
    .is_err());
    assert!(fide_rating(
        &ok.0,
        &ok.1,
        &ok.2,
        &ok.3,
        &ok.4,
        2,
        1500.0,
        (f64::INFINITY, 15.0, 30.0)
    )
    .is_err());
    assert!(fide_rating(
        &ok.0,
        &ok.1,
        &ok.2,
        &ok.3,
        &ok.4,
        2,
        1500.0,
        (10.0, f64::NAN, 30.0)
    )
    .is_err());
    assert!(fide_rating(
        &ok.0,
        &ok.1,
        &ok.2,
        &ok.3,
        &ok.4,
        2,
        1500.0,
        (10.0, 15.0, f64::NEG_INFINITY)
    )
    .is_err());
    // player index out of range / self-play
    assert!(fide_rating(
        &ok.0,
        &[2],
        &ok.2,
        &ok.3,
        &ok.4,
        2,
        1500.0,
        (10.0, 15.0, 30.0)
    )
    .is_err());
    assert!(fide_rating(
        &ok.0,
        &[1],
        &[1],
        &ok.3,
        &ok.4,
        2,
        1500.0,
        (10.0, 15.0, 30.0)
    )
    .is_err());
    // score out of range / non-finite
    assert!(fide_rating(
        &ok.0,
        &ok.1,
        &ok.2,
        &[1.5],
        &ok.4,
        2,
        1500.0,
        (10.0, 15.0, 30.0)
    )
    .is_err());
    assert!(fide_rating(
        &ok.0,
        &ok.1,
        &ok.2,
        &[-0.1],
        &ok.4,
        2,
        1500.0,
        (10.0, 15.0, 30.0)
    )
    .is_err());
    assert!(fide_rating(
        &ok.0,
        &ok.1,
        &ok.2,
        &[f64::NAN],
        &ok.4,
        2,
        1500.0,
        (10.0, 15.0, 30.0)
    )
    .is_err());
    // non-finite gamma
    assert!(fide_rating(
        &ok.0,
        &ok.1,
        &ok.2,
        &ok.3,
        &[f64::INFINITY],
        2,
        1500.0,
        (10.0, 15.0, 30.0)
    )
    .is_err());
}

/// MC-500: with kv = (k, k, k) the kfide schedule returns k for every
/// player regardless of elite/games, so ratings/games/wins/draws/losses/
/// lag must be BITWISE identical to elo_rating(kfac = k). Random
/// schedules keep ratings finite (scores in {0, 0.5, 1}, small gamma),
/// so plain equality is well-defined. Kills any divergence between the
/// shared plumbing of the two implementations.
#[test]
#[ignore]
fn fd_mc_500_elo_reduction() {
    let mut rng = Lcg(0x5EED_F1DE_0001_0001);
    for rep in 0..500 {
        let n = 3 + (rng.next_f64() * 8.0) as usize; // 3..=10
        let n_games = 4 + (rng.next_f64() * 28.0) as usize; // 4..=31
        let mut periods = Vec::with_capacity(n_games);
        let mut white = Vec::with_capacity(n_games);
        let mut black = Vec::with_capacity(n_games);
        let mut score = Vec::with_capacity(n_games);
        let mut gamma = Vec::with_capacity(n_games);
        for _ in 0..n_games {
            periods.push(1 + (rng.next_f64() * 4.0) as u64); // 1..=4
            let w = (rng.next_f64() * n as f64) as usize % n;
            let mut b = (rng.next_f64() * n as f64) as usize % n;
            if b == w {
                b = (b + 1) % n;
            }
            white.push(w);
            black.push(b);
            let s = match (rng.next_f64() * 3.0) as u32 {
                0 => 0.0,
                1 => 0.5,
                _ => 1.0,
            };
            score.push(s);
            gamma.push((rng.next_f64() - 0.5) * 20.0);
        }
        let k = 5.0 + rng.next_f64() * 40.0;
        let f = fide_rating(
            &periods,
            &white,
            &black,
            &score,
            &gamma,
            n,
            1500.0,
            (k, k, k),
        )
        .unwrap();
        let e = elo_rating(&periods, &white, &black, &score, &gamma, n, 1500.0, k).unwrap();
        assert_eq!(f.ratings, e.ratings, "rep {rep}: ratings diverge");
        assert_eq!(f.games, e.games, "rep {rep}");
        assert_eq!(f.wins, e.wins, "rep {rep}");
        assert_eq!(f.draws, e.draws, "rep {rep}");
        assert_eq!(f.losses, e.losses, "rep {rep}");
        assert_eq!(f.lag, e.lag, "rep {rep}");
    }
}

// ---------------------------------------------------------------------------
// predict_rating_two / predict_rating_multi (PlayerRatings predict.rating,
// R lines 1056-1133 READ). Pins from the executed oracle
// (files/predict_oracle.py / predict_oracle_output.txt).
// ---------------------------------------------------------------------------

/// P1+P2: Elo branch exact half + gamma-sign pin.
/// Asserts read `predict_rating_two` return values. Killing mutants:
/// gamma sign flip (MU1) changes the P2 pin; equal-rating case pins 1/2.
#[test]
fn pr_anchor_elo_gamma_sign() {
    let p1 = predict_rating_two(
        &[2200.0, 2200.0],
        None,
        &[20, 20],
        &[0],
        &[1],
        &[0.0],
        15,
        None,
        None,
    )
    .unwrap();
    assert_eq!(p1, vec![0.5]);
    let p2 = predict_rating_two(
        &[2200.0, 2000.0],
        None,
        &[20, 20],
        &[0],
        &[1],
        &[30.0],
        15,
        None,
        None,
    )
    .unwrap();
    assert!((p2[0] - 0.7898441797581306).abs() < 1e-14, "{}", p2[0]);
    // MU1 (gamma sign flip) would give ~0.7276; assert distance from it.
    let mu1 = 1.0 / (1.0 + 10f64.powf((2000.0 - 2200.0 + 30.0) / 400.0));
    assert!((p2[0] - mu1).abs() > 1e-3);
}

/// P3: deviation-branch shrink pin + dev=0 reduction to the Elo branch.
/// Asserts read `predict_rating_two` values from BOTH branches. Killing
/// mutants: qip3 factor 3->2 (MU2) changes the nonzero-dev pin; the dev=0
/// crate-vs-crate identity kills stray vec offsets.
#[test]
fn pr_deviation_shrink() {
    let p3 = predict_rating_two(
        &[2200.0, 2000.0],
        Some(&[50.0, 100.0]),
        &[20, 20],
        &[0],
        &[1],
        &[30.0],
        15,
        None,
        None,
    )
    .unwrap();
    assert!((p3[0] - 0.776912664201114).abs() < 1e-14, "{}", p3[0]);
    let p3z = predict_rating_two(
        &[2200.0, 2000.0],
        Some(&[0.0, 0.0]),
        &[20, 20],
        &[0],
        &[1],
        &[30.0],
        15,
        None,
        None,
    )
    .unwrap();
    let p2 = predict_rating_two(
        &[2200.0, 2000.0],
        None,
        &[20, 20],
        &[0],
        &[1],
        &[30.0],
        15,
        None,
        None,
    )
    .unwrap();
    assert_eq!(p3z[0], p2[0], "dev=0 must reduce to the Elo branch");
}

/// P4: tng boundary — games == tng is KEPT (strict <); games < tng is
/// replaced by trat or NaN. Asserts read `predict_rating_two` values.
/// Killing mutant: `<` -> `<=` (MU3) turns the kept pin into NaN.
#[test]
fn pr_tng_boundary() {
    let kept = predict_rating_two(
        &[2200.0, 2000.0],
        None,
        &[15, 15],
        &[0],
        &[1],
        &[0.0],
        15,
        None,
        None,
    )
    .unwrap();
    assert!((kept[0] - 0.7597469266479578).abs() < 1e-14, "{}", kept[0]);
    let dropped = predict_rating_two(
        &[2200.0, 2000.0],
        None,
        &[14, 15],
        &[0],
        &[1],
        &[0.0],
        15,
        None,
        None,
    )
    .unwrap();
    assert!(dropped[0].is_nan());
    let replaced = predict_rating_two(
        &[2200.0, 2000.0],
        None,
        &[14, 15],
        &[0],
        &[1],
        &[0.0],
        15,
        Some((2000.0, 0.0)),
        None,
    )
    .unwrap();
    assert_eq!(replaced[0], 0.5);
}

/// P5+P9: unmatched (-1) and matched-but-stored-NaN players — trat
/// replaces ALL extracted NAs; without trat they propagate. Asserts read
/// `predict_rating_two` values (incl. a crate-vs-crate deviation anchor).
#[test]
fn pr_unmatched_and_stored_na_trat() {
    let una = predict_rating_two(
        &[2200.0, 2000.0],
        None,
        &[20, 20],
        &[-1],
        &[1],
        &[0.0],
        15,
        None,
        None,
    )
    .unwrap();
    assert!(una[0].is_nan());
    let unb = predict_rating_two(
        &[2200.0, 2000.0],
        None,
        &[20, 20],
        &[-1],
        &[1],
        &[0.0],
        15,
        Some((2000.0, 0.0)),
        None,
    )
    .unwrap();
    assert_eq!(unb[0], 0.5);
    let p9a = predict_rating_two(
        &[f64::NAN, 2000.0],
        None,
        &[20, 20],
        &[0],
        &[1],
        &[0.0],
        15,
        Some((2000.0, 0.0)),
        None,
    )
    .unwrap();
    assert_eq!(p9a[0], 0.5);
    let p9b = predict_rating_two(
        &[f64::NAN, 2000.0],
        None,
        &[20, 20],
        &[0],
        &[1],
        &[0.0],
        15,
        None,
        None,
    )
    .unwrap();
    assert!(p9b[0].is_nan());
    // stored-NaN deviation replaced by trat.1 == crate value with real dev.
    let p9c = predict_rating_two(
        &[2200.0, 2000.0],
        Some(&[f64::NAN, 50.0]),
        &[20, 20],
        &[0],
        &[1],
        &[30.0],
        15,
        Some((2200.0, 50.0)),
        None,
    )
    .unwrap();
    let p9c_ref = predict_rating_two(
        &[2200.0, 2000.0],
        Some(&[50.0, 50.0]),
        &[20, 20],
        &[0],
        &[1],
        &[30.0],
        15,
        None,
        None,
    )
    .unwrap();
    assert_eq!(p9c[0], p9c_ref[0]);
    assert!((p9c[0] - 0.7844611342833985).abs() < 1e-14, "{}", p9c[0]);
}

/// P6: thresh uses >= (exact equality -> 1) and NaN preds stay NaN.
/// Asserts read `predict_rating_two` values. Killing mutant: `>=` -> `>`
/// (MU6) turns the exact-equality 1.0 into 0.0.
#[test]
fn pr_thresh_ge_and_nan() {
    let p = predict_rating_two(
        &[2200.0, 2200.0],
        None,
        &[20, 20],
        &[0, -1],
        &[1, 1],
        &[0.0, 0.0],
        15,
        None,
        Some(0.5),
    )
    .unwrap();
    assert_eq!(p[0], 1.0, "pred exactly == thresh must map to 1 (>=)");
    assert!(p[1].is_nan());
    let below = predict_rating_two(
        &[2000.0, 2200.0],
        None,
        &[20, 20],
        &[0],
        &[1],
        &[0.0],
        15,
        None,
        Some(0.5),
    )
    .unwrap();
    assert_eq!(below[0], 0.0);
}

/// P7: EloM (rat - rowmean)/40 pins, na.rm rowmean, all-NaN row.
/// Asserts read `predict_rating_multi` values. Killing mutant: divisor
/// 40 -> 400 (MU4) changes the exact 2.5/3.75 pins.
#[test]
fn pr_elom_rowmean() {
    let p7 = predict_rating_multi(
        &[2300.0, 2200.0, 2100.0, 2000.0],
        &[20, 20, 20, 20],
        &[0, 1, 2, 0, 3, -1],
        2,
        3,
        15,
        None,
        false,
    )
    .unwrap();
    assert_eq!(&p7[0..3], &[2.5, 0.0, -2.5]);
    assert_eq!(p7[3], 3.75);
    assert_eq!(p7[4], -3.75);
    assert!(p7[5].is_nan());
    let allna = predict_rating_multi(
        &[2300.0, 2200.0],
        &[20, 20],
        &[-1, -1],
        1,
        2,
        15,
        None,
        false,
    )
    .unwrap();
    assert!(allna.iter().all(|v| v.is_nan()));
    // P9d: stored-NaN seat replaced by trat.
    let p9d = predict_rating_multi(
        &[f64::NAN, 2200.0, 2100.0],
        &[20, 20, 20],
        &[0, 1, 2],
        1,
        3,
        15,
        Some(2300.0),
        false,
    )
    .unwrap();
    // tng boundary in the multi branch: games == tng KEPT (strict <).
    let boundary = predict_rating_multi(
        &[2300.0, 2200.0, 2100.0],
        &[15, 20, 20],
        &[0, 1, 2],
        1,
        3,
        15,
        None,
        false,
    )
    .unwrap();
    assert_eq!(&boundary[0..3], &[2.5, 0.0, -2.5]);
    let dropped = predict_rating_multi(
        &[2300.0, 2200.0, 2100.0],
        &[14, 20, 20],
        &[0, 1, 2],
        1,
        3,
        15,
        None,
        false,
    )
    .unwrap();
    assert!(dropped[0].is_nan());
    assert_eq!(dropped[1], 1.25);
    assert_eq!(dropped[2], -1.25);
}

/// P8: placing ranks — ties share the MINIMUM rank, NaN kept.
/// Asserts read `predict_rating_multi` values. Killing mutant: min ->
/// average/max tie handling (MU5) changes the (1,1,3) pattern.
#[test]
fn pr_placing_min_ties() {
    let p8 = predict_rating_multi(
        &[2300.0, 2300.0, 2100.0, 2000.0],
        &[20, 20, 20, 20],
        &[0, 1, 2, -1],
        1,
        4,
        15,
        None,
        true,
    )
    .unwrap();
    assert_eq!(p8[0], 1.0);
    assert_eq!(p8[1], 1.0);
    assert_eq!(p8[2], 3.0);
    assert!(p8[3].is_nan());
}

/// Error contract for both functions. Asserts read Err values.
#[test]
fn pr_error_contract() {
    let r = |x: Result<Vec<f64>, String>| x.unwrap_err();
    // player count bounds
    assert!(r(predict_rating_two(
        &[1500.0],
        None,
        &[0],
        &[0],
        &[0],
        &[0.0],
        15,
        None,
        None
    ))
    .contains("2..=10000"));
    // games length mismatch
    assert!(r(predict_rating_two(
        &[1500.0, 1500.0],
        None,
        &[0],
        &[0],
        &[1],
        &[0.0],
        15,
        None,
        None
    ))
    .contains("games length"));
    // empty game rows
    assert!(r(predict_rating_two(
        &[1500.0, 1500.0],
        None,
        &[0, 0],
        &[],
        &[],
        &[],
        15,
        None,
        None
    ))
    .contains("at least one game"));
    // infinite rating rejected (NaN allowed elsewhere)
    assert!(r(predict_rating_two(
        &[f64::INFINITY, 1500.0],
        None,
        &[0, 0],
        &[0],
        &[1],
        &[0.0],
        15,
        None,
        None
    ))
    .contains("infinite"));
    // non-finite gamma
    assert!(r(predict_rating_two(
        &[1500.0, 1500.0],
        None,
        &[0, 0],
        &[0],
        &[1],
        &[f64::NAN],
        15,
        None,
        None
    ))
    .contains("gamma"));
    // non-finite trat
    assert!(r(predict_rating_two(
        &[1500.0, 1500.0],
        None,
        &[0, 0],
        &[0],
        &[1],
        &[0.0],
        15,
        Some((f64::NAN, 0.0)),
        None
    ))
    .contains("trat"));
    // trat deviation checked only when deviations supplied
    assert!(predict_rating_two(
        &[1500.0, 1500.0],
        None,
        &[0, 0],
        &[0],
        &[1],
        &[0.0],
        15,
        Some((1500.0, f64::NAN)),
        None
    )
    .is_ok());
    assert!(r(predict_rating_two(
        &[1500.0, 1500.0],
        Some(&[50.0, 50.0]),
        &[0, 0],
        &[0],
        &[1],
        &[0.0],
        15,
        Some((1500.0, f64::NAN)),
        None
    ))
    .contains("trat deviation"));
    // non-finite thresh
    assert!(r(predict_rating_two(
        &[1500.0, 1500.0],
        None,
        &[0, 0],
        &[0],
        &[1],
        &[0.0],
        15,
        None,
        Some(f64::NAN)
    ))
    .contains("thresh"));
    // out-of-range index (only -1 sentinel allowed)
    assert!(r(predict_rating_two(
        &[1500.0, 1500.0],
        None,
        &[0, 0],
        &[-2],
        &[1],
        &[0.0],
        15,
        None,
        None
    ))
    .contains("out of range"));
    assert!(r(predict_rating_two(
        &[1500.0, 1500.0],
        None,
        &[0, 0],
        &[0],
        &[2],
        &[0.0],
        15,
        None,
        None
    ))
    .contains("out of range"));
    // self-play
    assert!(r(predict_rating_two(
        &[1500.0, 1500.0],
        None,
        &[0, 0],
        &[1],
        &[1],
        &[0.0],
        15,
        None,
        None
    ))
    .contains("self-play"));
    // deviations length mismatch
    assert!(r(predict_rating_two(
        &[1500.0, 1500.0],
        Some(&[50.0]),
        &[0, 0],
        &[0],
        &[1],
        &[0.0],
        15,
        None,
        None
    ))
    .contains("deviations length"));
    // multi: seat bounds, players length, nr=0, index range, trat
    assert!(r(predict_rating_multi(
        &[1500.0, 1500.0],
        &[0, 0],
        &[0],
        1,
        1,
        15,
        None,
        false
    ))
    .contains("2..=1000"));
    assert!(r(predict_rating_multi(
        &[1500.0, 1500.0],
        &[0, 0],
        &[0, 1, 0],
        1,
        2,
        15,
        None,
        false
    ))
    .contains("players length"));
    // nr*np usize overflow must be a checked error, not a wrap + panic
    // (kills: replacing checked_mul with wrapping `nr * np`, which lets an
    // empty players slice pass the length check and then index OOB).
    assert!(r(predict_rating_multi(
        &[1500.0, 1500.0],
        &[0, 0],
        &[],
        1usize << 61,
        8,
        15,
        None,
        false
    ))
    .contains("overflows"));
    assert!(r(predict_rating_multi(
        &[1500.0, 1500.0],
        &[0, 0],
        &[],
        0,
        2,
        15,
        None,
        false
    ))
    .contains("at least one event"));
    assert!(r(predict_rating_multi(
        &[1500.0, 1500.0],
        &[0, 0],
        &[0, 5],
        1,
        2,
        15,
        None,
        false
    ))
    .contains("out of range"));
    assert!(r(predict_rating_multi(
        &[1500.0, 1500.0],
        &[0, 0],
        &[0, 1],
        1,
        2,
        15,
        Some(f64::INFINITY),
        false
    ))
    .contains("trat"));
}

/// MC-500: structural invariants over random inputs. Every assert reads
/// crate return values: complement symmetry pred(w,b)+pred(b,w) ~= 1 at
/// gamma=0 (both branches), preds in (0,1), thresh output consistent with
/// the crate's own unthresholded preds, placing ranks are a valid
/// min-tie ranking of the crate's own EloM preds.
#[test]
#[ignore]
fn pr_mc_500_invariants() {
    let mut rng = Lcg(0x5eed_cafe_1234_0001);
    for rep in 0..500 {
        let n = 3 + (rng.next_f64() * 8.0) as usize;
        let ratings: Vec<f64> = (0..n).map(|_| 1200.0 + 1600.0 * rng.next_f64()).collect();
        let devs: Vec<f64> = (0..n).map(|_| 30.0 + 300.0 * rng.next_f64()).collect();
        let games: Vec<u64> = (0..n)
            .map(|_| 15 + (rng.next_f64() * 40.0) as u64)
            .collect();
        let w = (rng.next_f64() * n as f64) as i64;
        let mut b = (rng.next_f64() * n as f64) as i64;
        if b == w {
            b = (b + 1) % n as i64;
        }
        for dev_opt in [None, Some(devs.as_slice())] {
            let fwd = predict_rating_two(
                &ratings,
                dev_opt,
                &games,
                &[w],
                &[b],
                &[0.0],
                15,
                None,
                None,
            )
            .unwrap();
            let bwd = predict_rating_two(
                &ratings,
                dev_opt,
                &games,
                &[b],
                &[w],
                &[0.0],
                15,
                None,
                None,
            )
            .unwrap();
            assert!(fwd[0] > 0.0 && fwd[0] < 1.0, "rep {rep}");
            assert!(
                (fwd[0] + bwd[0] - 1.0).abs() < 1e-12,
                "rep {rep}: complement symmetry"
            );
            let th = predict_rating_two(
                &ratings,
                dev_opt,
                &games,
                &[w],
                &[b],
                &[0.0],
                15,
                None,
                Some(0.5),
            )
            .unwrap();
            let expect = if fwd[0] >= 0.5 { 1.0 } else { 0.0 };
            assert_eq!(th[0], expect, "rep {rep}: thresh vs crate pred");
        }
        // EloM: placing must be the min-tie ranking of the crate's preds.
        let np = 3.min(n);
        let seats: Vec<i64> = (0..np as i64).collect();
        let preds = predict_rating_multi(&ratings, &games, &seats, 1, np, 15, None, false).unwrap();
        let ranks = predict_rating_multi(&ratings, &games, &seats, 1, np, 15, None, true).unwrap();
        for s in 0..np {
            let expected = 1.0
                + preds
                    .iter()
                    .filter(|u| !u.is_nan() && **u > preds[s])
                    .count() as f64;
            assert_eq!(ranks[s], expected, "rep {rep} seat {s}");
        }
        // Row mean of preds is 0 (crate values; na.rm mean identity).
        let sum: f64 = preds.iter().sum();
        assert!(sum.abs() < 1e-9, "rep {rep}: pred row sum {sum}");
    }
}
