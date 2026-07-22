use super::{finite_wle_value, item_information_4pl, refine_wle_root, score_wle};

fn sig(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// The Warm estimating function `g = score + J/(2I)` recomputed INDEPENDENTLY from FINITE-DIFFERENCE
/// derivatives of `P` (no analytic `P'`/`P''`), so a sign error in the implementation's `J = P' P''`
/// term is not shared. Returns `g` at `theta`.
fn g_fd(a: &[f64], b: &[f64], c: &[f64], d: &[f64], y: &[f64], theta: f64) -> f64 {
    let h = 1e-4;
    let pf = |i: usize, t: f64| c[i] + (d[i] - c[i]) * sig(a[i] * (t - b[i]));
    let (mut score, mut info, mut jterm) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let p0 = pf(i, theta);
        let p1 = (pf(i, theta + h) - pf(i, theta - h)) / (2.0 * h); // P' by FD
        let p2 = (pf(i, theta + h) - 2.0 * p0 + pf(i, theta - h)) / (h * h); // P'' by FD
        let pq = p0 * (1.0 - p0);
        score += (y[i] - p0) * p1 / pq;
        info += p1 * p1 / pq;
        jterm += p1 * p2 / pq;
    }
    score + jterm / (2.0 * info)
}

/// Root anchor across {2PL, 3PL, Rasch}: the returned `theta_hat` satisfies the Warm estimating
/// equation, verified by the FD-derivative recomputation (independent of the analytic derivatives).
#[test]
fn wle_estimating_equation_root() {
    let j = 10usize;
    let a2: Vec<f64> = (0..j).map(|i| 0.8 + 0.09 * i as f64).collect();
    let b: Vec<f64> = (0..j).map(|i| -2.0 + 0.4 * i as f64).collect();
    let y: Vec<f64> = (0..j).map(|i| (i % 2) as f64).collect(); // mixed -> interior root
    let obs = vec![true; j];
    let one = vec![1.0f64; j];
    let zero = vec![0.0f64; j];
    let d = vec![1.0f64; j];
    let c02 = vec![0.2f64; j];
    for (label, a, c) in [
        ("2PL", &a2, &zero),
        ("3PL", &a2, &c02),
        ("Rasch", &one, &zero),
    ] {
        let res = score_wle(a, &b, c, &d, &y, &obs, 1, 20.0, 1e-9).unwrap();
        assert!(!res.boundary[0], "{label}: unexpected boundary");
        let g = g_fd(a, &b, c, &d, &y, res.theta[0]);
        assert!(
            g.abs() < 1e-4,
            "{label}: WLE root residual {g} at theta {}",
            res.theta[0]
        );
        // SE matches 1/sqrt(I) recomputed from item_information_4pl at the estimate
        let info: f64 = (0..j)
            .map(|i| {
                let p = c[i] + (d[i] - c[i]) * sig(a[i] * (res.theta[0] - b[i]));
                item_information_4pl(a[i], p, c[i], d[i])
            })
            .sum();
        assert!(
            (res.se[0] - (1.0 / info).sqrt()).abs() < 1e-9,
            "{label}: SE"
        );
    }
}

/// Finiteness (scoped to the 2PL, `c=0, d=1`): the all-correct and all-incorrect patterns — where
/// the MLE is `+/-infinity` — return FINITE, interior WLE estimates, with correct > incorrect.
#[test]
fn wle_finite_at_perfect_score_2pl() {
    let j = 6usize;
    let a: Vec<f64> = (0..j).map(|i| 1.0 + 0.1 * i as f64).collect();
    let b: Vec<f64> = (0..j).map(|i| -1.5 + 0.6 * i as f64).collect();
    let c = vec![0.0f64; j];
    let d = vec![1.0f64; j];
    let obs = vec![true; j];
    let all1 = vec![1.0f64; j];
    let all0 = vec![0.0f64; j];
    let hi = score_wle(&a, &b, &c, &d, &all1, &obs, 1, 20.0, 1e-9).unwrap();
    let lo = score_wle(&a, &b, &c, &d, &all0, &obs, 1, 20.0, 1e-9).unwrap();
    assert!(
        hi.theta[0].is_finite() && !hi.boundary[0],
        "all-correct theta {}",
        hi.theta[0]
    );
    assert!(
        lo.theta[0].is_finite() && !lo.boundary[0],
        "all-incorrect theta {}",
        lo.theta[0]
    );
    assert!(
        hi.theta[0] > lo.theta[0],
        "correct {} !> incorrect {}",
        hi.theta[0],
        lo.theta[0]
    );
    // the FD estimating equation is also ~0 at these finite roots
    assert!(g_fd(&a, &b, &c, &d, &all1, hi.theta[0]).abs() < 1e-4);
    assert!(g_fd(&a, &b, &c, &d, &all0, lo.theta[0]).abs() < 1e-4);
}

/// Monotonicity: for a fixed Rasch item set the WLE is nondecreasing in the number-correct score.
#[test]
fn wle_monotone_in_raw_score() {
    let j = 8usize;
    let a = vec![1.0f64; j];
    let b: Vec<f64> = (0..j).map(|i| -2.0 + 0.5 * i as f64).collect();
    let c = vec![0.0f64; j];
    let d = vec![1.0f64; j];
    let obs = vec![true; j];
    let mut prev = f64::NEG_INFINITY;
    for k in 0..=j {
        let y: Vec<f64> = (0..j).map(|i| if i < k { 1.0 } else { 0.0 }).collect();
        let res = score_wle(&a, &b, &c, &d, &y, &obs, 1, 20.0, 1e-9).unwrap();
        assert!(
            res.theta[0] >= prev - 1e-9,
            "raw score {k}: theta {} < previous {prev}",
            res.theta[0]
        );
        prev = res.theta[0];
    }
}

/// Validation guards trip non-vacuously.
#[test]
fn wle_validates() {
    let a = vec![1.0, 1.2];
    let b = vec![0.0, 0.5];
    let c = vec![0.0, 0.0];
    let d = vec![1.0, 1.0];
    let y = vec![1.0, 0.0];
    let obs = vec![true, true];
    assert!(score_wle(&a, &b, &c, &d, &y, &obs, 1, 20.0, 1e-9).is_ok());
    // length mismatch
    assert!(score_wle(&a, &b[..1], &c, &d, &y, &obs, 1, 20.0, 1e-9).is_err());
    // c >= d
    let cbad = vec![1.0, 0.0];
    assert!(score_wle(&a, &b, &cbad, &d, &y, &obs, 1, 20.0, 1e-9).is_err());
    // response not 0/1
    let ybad = vec![2.0, 0.0];
    assert!(score_wle(&a, &b, &c, &d, &ybad, &obs, 1, 20.0, 1e-9).is_err());
    // theta_bound non-positive
    assert!(score_wle(&a, &b, &c, &d, &y, &obs, 1, 0.0, 1e-9).is_err());
    // no information, and controls whose required adaptive grid would be intractable
    assert!(score_wle(&[0.0, 0.0], &b, &c, &d, &y, &obs, 1, 20.0, 1e-9).is_err());
    assert!(score_wle(&a, &b, &c, &d, &y, &obs, 1, 1e308, 1e-9).is_err());
}

/// The 3PL weighted likelihood is multimodal here; the WLE must return the GLOBAL mode, not merely
/// a root of the estimating equation. Adversarial-review worst case: a single bracketed bisection
/// returns `theta ~ +1.70`, but the dominant weighted-likelihood mode is `theta ~ -4.13` (~10x more
/// probable). Pins the global-mode selection.
#[test]
fn wle_selects_global_mode_3pl_multimodal() {
    let a = [0.59, 1.38, 2.16, 3.45, 1.53, 2.58, 1.13, 1.02, 2.9, 2.07];
    let b = [
        -3.5, -3.78, -0.06, 2.82, 2.51, 2.73, -2.84, 3.48, 1.77, 0.07,
    ];
    let c = [0.37, 0.23, 0.26, 0.45, 0.28, 0.3, 0.4, 0.22, 0.22, 0.21];
    let d = [1.0f64; 10];
    let y = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let obs = [true; 10];
    let res = score_wle(&a, &b, &c, &d, &y, &obs, 1, 20.0, 1e-9).unwrap();
    assert!(
        res.theta[0] < -3.0,
        "did not select the global mode: theta {} (expected ~ -4.13, not the +1.70 root)",
        res.theta[0]
    );
}

/// A fixed 512-node theta grid misses the narrow dominant mode created by the third item's high
/// discrimination and returns the lower weighted-likelihood mode near -3.37. A 0.001-step
/// independent numerical integral of `g` places the global maximum near -2.74.
#[test]
fn wle_resolves_narrow_global_mode_4pl() {
    let a = [
        3.329447657883643,
        0.27232757528116147,
        84.38646237902715,
        4.507142332708399,
        0.216076032654272,
        1.152868526694496,
        0.5026701543207452,
        3.594020470848568,
    ];
    let b = [
        -2.2559085720992726,
        4.784793518100594,
        -2.7313173853279284,
        3.16639784715872,
        2.45483432935667,
        3.577399138394002,
        -0.541499889021253,
        -3.1606254220709538,
    ];
    let c = [
        0.4293154638946107,
        0.03968316086976924,
        0.2117187277379179,
        0.4041453105751009,
        0.14842532496042327,
        0.2781240730868334,
        0.07100800469041686,
        0.16882942315223948,
    ];
    let d = [
        0.9271440266982822,
        0.8326920519773708,
        0.7052699247299387,
        0.7321429393598535,
        0.7250331916969143,
        0.8800003001396377,
        0.7964931220169523,
        0.8078636510671307,
    ];
    let y = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
    let obs = [true; 8];
    let res = score_wle(&a, &b, &c, &d, &y, &obs, 1, 20.0, 1e-10).unwrap();
    assert!(!res.boundary[0]);
    assert!(
        (res.theta[0] + 2.74).abs() < 0.02,
        "selected theta {} instead of the narrow global mode near -2.74",
        res.theta[0]
    );
    assert!(g_fd(&a, &b, &c, &d, &y, res.theta[0]).abs() < 1e-3);
}

/// A person with no observed items has undefined ability: `NaN` estimate and SE, flagged — not a
/// spurious `theta = 0` (which the `g == 0` bisection shortcut would otherwise return).
#[test]
fn wle_all_missing_is_nan() {
    let a = [1.0, 1.2, 0.9];
    let b = [-0.5, 0.0, 0.7];
    let c = [0.0f64; 3];
    let d = [1.0f64; 3];
    let y = [0.0, 0.0, 0.0];
    let obs = [false, false, false];
    let res = score_wle(&a, &b, &c, &d, &y, &obs, 1, 20.0, 1e-9).unwrap();
    assert!(res.theta[0].is_nan() && res.se[0].is_nan() && res.boundary[0]);
}

#[test]
fn wle_numeric_helpers_and_representational_boundaries() {
    assert_eq!(finite_wle_value(1.0, "unused".into()).unwrap(), 1.0);
    assert_eq!(
        finite_wle_value(f64::NAN, "non-finite".into()).unwrap_err(),
        "non-finite"
    );
    let mut exact_root = |x| x;
    assert_eq!(
        refine_wle_root(-1.0, 1.0, 1e-12, &mut exact_root).unwrap(),
        0.0
    );
    let mut unbracketed = |x| x * x + 1.0;
    assert!(refine_wle_root(-1.0, 1.0, 1e-12, &mut unbracketed).is_err());
    let mut discontinuous = |x| {
        if x <= 1.0 / 3.0 {
            1.0
        } else {
            -1.0
        }
    };
    assert!(refine_wle_root(0.0, 1.0, f64::MIN_POSITIVE, &mut discontinuous).is_err());

    let c = [0.0, 0.0];
    let d = [1.0, 1.0];
    let masked = score_wle(
        &[1.0, 0.8],
        &[0.0, 0.5],
        &c,
        &d,
        &[1.0, 0.0],
        &[true, false],
        1,
        6.0,
        1e-9,
    )
    .unwrap();
    assert!(masked.theta[0].is_finite());

    let boundary = score_wle(
        &[1.0],
        &[10.0],
        &[0.0],
        &[1.0],
        &[1.0],
        &[true],
        1,
        0.1,
        1e-9,
    )
    .unwrap();
    assert!(boundary.boundary[0]);
    assert_eq!(boundary.theta[0], 0.1);

    let negligible_information = score_wle(
        &[1e-8],
        &[10.0],
        &[0.0],
        &[1.0],
        &[1.0],
        &[true],
        1,
        0.1,
        1e-9,
    )
    .unwrap();
    assert!(negligible_information.se[0].is_nan());

    assert!(score_wle(
        &[1e308],
        &[1e308],
        &[0.0],
        &[1.0],
        &[1.0],
        &[true],
        1,
        1e-308,
        1e-12,
    )
    .is_err());

    assert!(score_wle(
        &[1.0, 1.2],
        &[0.0, 0.5],
        &c,
        &d,
        &[1.0, 0.0],
        &[true, true],
        1,
        6.0,
        f64::MIN_POSITIVE,
    )
    .is_err());
}

/// Literature-grade bias comparison (>=500 reps): Warm's WLE has smaller mean bias than the MLE,
/// especially at extreme abilities where perfect/near-perfect patterns bias the (boundary-clamped)
/// MLE. Run with: `cargo test -p mlsirm-core --release wle_reduces_mle_bias_500 -- --ignored`.
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps)"]
fn wle_reduces_mle_bias_500() {
    let reps = 500usize;
    let j = 15usize;
    let a: Vec<f64> = (0..j).map(|i| 0.9 + 0.05 * (i % 5) as f64).collect();
    let b: Vec<f64> = (0..j)
        .map(|i| -2.0 + 4.0 * i as f64 / (j as f64 - 1.0))
        .collect();
    let c = vec![0.0f64; j];
    let d = vec![1.0f64; j];
    let obs = vec![true; j];
    // MLE by bisection on the score (clamped to +/-6 for separable patterns).
    let mle = |y: &[f64]| -> f64 {
        let score = |t: f64| -> f64 {
            (0..j)
                .map(|i| {
                    let p = sig(a[i] * (t - b[i]));
                    a[i] * (y[i] - p)
                })
                .sum::<f64>()
        };
        let (mut loi, mut hii) = (-6.0f64, 6.0f64);
        let (glo, ghi) = (score(loi), score(hii));
        if glo * ghi > 0.0 {
            return if glo > 0.0 { hii } else { loi };
        }
        for _ in 0..100 {
            let mid = 0.5 * (loi + hii);
            if score(mid) > 0.0 {
                loi = mid;
            } else {
                hii = mid;
            }
        }
        0.5 * (loi + hii)
    };
    let grid = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let (mut wle_abs, mut mle_abs) = (0.0f64, 0.0f64);
    for &theta in &grid {
        let (mut wsum, mut msum, mut n) = (0.0f64, 0.0f64, 0usize);
        for rep in 0..reps {
            let mut rng = Lcg(0x9E1E_u64
                .wrapping_mul(rep as u64 + 1)
                .wrapping_add((theta as i64 as u64).wrapping_mul(97)));
            let y: Vec<f64> = (0..j)
                .map(|i| {
                    let p = sig(a[i] * (theta - b[i]));
                    if rng.next_f64() < p {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            let w = score_wle(&a, &b, &c, &d, &y, &obs, 1, 20.0, 1e-9)
                .unwrap()
                .theta[0];
            wsum += w - theta;
            msum += mle(&y) - theta;
            n += 1;
        }
        let (wb, mb) = (wsum / n as f64, msum / n as f64);
        println!("[wle bias theta={theta}] WLE={wb:.4} MLE={mb:.4}");
        wle_abs += wb.abs();
        mle_abs += mb.abs();
    }
    println!("[wle] sum|bias| WLE={wle_abs:.4} MLE={mle_abs:.4}");
    assert!(
        wle_abs < mle_abs,
        "WLE did not reduce aggregate bias: {wle_abs} vs {mle_abs}"
    );
}

/// Pins the identity `I' = 2J - T` with `T = sum_i P_i'^3 (1 - 2 P_i) / (P_i Q_i)^2`, stated in the
/// [`score_wle`] doc block. Both `J` and `T` are formed from the SHIPPED `item_information_4pl` and
/// the same closed forms the estimator uses, while `I'` comes from a central difference of
/// `item_information_4pl` — a different code path — so neither side is a private restatement.
///
/// This exists because that sentence has been wrong twice: it first claimed `J` coincides with `I'/2`
/// for the 2PL, and the correction of that claim dropped the `(1 - 2 P)` factor from `T`. A formula
/// asserted in prose and checked by nothing is how that happens.
///
/// kills: `T` written without the `(1 - 2 P)` factor (off by ~5x at the 2PL point below); a claim that
/// `T = J` for the 3PL, where the two genuinely differ.
#[test]
fn wle_information_derivative_identity() {
    let jt = |a: f64, b: f64, c: f64, d: f64, t: f64| -> (f64, f64) {
        let s = sig(a * (t - b));
        let dc = d - c;
        let p = c + dc * s;
        let pq = p * (1.0 - p);
        let p1 = a * dc * s * (1.0 - s);
        let p2 = a * a * dc * s * (1.0 - s) * (1.0 - 2.0 * s);
        (p1 * p2 / pq, p1 * p1 * p1 * (1.0 - 2.0 * p) / (pq * pq))
    };
    let h = 1e-5;
    for (a, b, c, d, t, two_pl) in [
        (1.5f64, 0.2, 0.0, 1.0, 0.7, true),
        (1.5, 0.2, 0.25, 1.0, -1.6, false),
        (1.2, -0.4, 0.0, 0.95, 0.3, false),
    ] {
        let (j, tt) = jt(a, b, c, d, t);
        // `item_information_4pl` takes the PROBABILITY, so the finite difference is over theta via P.
        let pf = |x: f64| c + (d - c) * sig(a * (x - b));
        let iprime = (item_information_4pl(a, pf(t + h), c, d)
            - item_information_4pl(a, pf(t - h), c, d))
            / (2.0 * h);
        assert!(
            (2.0 * j - tt - iprime).abs() / iprime.abs().max(1e-8) < 1e-5,
            "I' = 2J - T failed at a={a} b={b} c={c} d={d} theta={t}: 2J-T={} I'={iprime}",
            2.0 * j - tt
        );
        // T = J exactly for the 2PL/Rasch (hence J = I' and the weight is sqrt(I)); NOT otherwise.
        let same = (tt - j).abs() / j.abs().max(1e-8) < 1e-9;
        assert_eq!(
            same, two_pl,
            "T == J must hold for the 2PL only: a={a} c={c} d={d} gave T={tt} J={j}"
        );
    }
}
