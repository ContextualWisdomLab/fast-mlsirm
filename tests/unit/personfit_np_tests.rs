//! Tests for the PerFit nonparametric person-fit port.
//!
//! Every assertion reads values RETURNED BY `person_fit_np` (crate
//! outputs). Pinned oracle: line-by-line Python transcription of the
//! PerFit R sources (complete-data specialization) at numpy
//! default_rng(2033), N=12, I=8 — see files/perfit_spec.md. Mutation
//! kills executed and logged in the spec.

use super::person_fit_np;

const TOL: f64 = 1e-12;

/// MAIN fixture (numpy default_rng(2033), N=12, I=8; row 9 planted
/// reversed, row 10 all-0s, row 11 all-1s). pi HAS ties (cols {3,4,7}
/// and {5,6,8}, 1-based), so this fixture also exercises the
/// ascending-index tie-break.
fn main_fixture() -> Vec<Vec<f64>> {
    let rows = [
        "11000000", "01010110", "11010000", "11111011", "11001100", "11111010", "11101101",
        "11100001", "01000101", "00001111", "00000000", "11111111",
    ];
    rows.iter()
        .map(|r| r.chars().map(|c| (c as u8 - b'0') as f64).collect())
        .collect()
}

fn assert_close(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() < TOL,
        "{}: got {}, want {}",
        what,
        actual,
        expected
    );
}

// Kills: MU1 (ascending column order: g[9] 10->8), MU2 (denominator
// NC*(I-NC+1): gnormed[1] .625->.5), MU3 (1-Gnormed: nci[3] flips sign).
#[test]
fn main_fixture_g_gnormed_nci_pinned() {
    let r = person_fit_np(&main_fixture()).unwrap();
    let g_exp = [0.0, 10.0, 4.0, 4.0, 0.0, 6.0, 0.0, 4.0, 4.0, 10.0, 0.0, 0.0];
    for (p, &e) in g_exp.iter().enumerate() {
        assert_close(r.g[p], e, &format!("g[{}]", p));
    }
    assert_close(r.gnormed[1], 0.625, "gnormed[1]");
    assert_close(r.gnormed[2], 0.266_666_666_666_666_66, "gnormed[2]");
    assert_close(r.gnormed[3], 0.571_428_571_428_571_4, "gnormed[3]");
    assert_close(r.nci[1], -0.25, "nci[1]");
    assert_close(r.nci[3], -0.142_857_142_857_142_8, "nci[3]");
    assert_close(r.nci[7], 0.5, "nci[7]");
}

// Kills: MU4 (numerator slast instead of sfirst: u3[1] -> -0.5875...).
#[test]
fn main_fixture_u3_pinned() {
    let r = person_fit_np(&main_fixture()).unwrap();
    assert_close(r.u3[0], 0.0, "u3[0]");
    assert_close(r.u3[1], 0.412_467_062_253_308_6, "u3[1]");
    assert_close(r.u3[2], 0.101_591_803_582_623_5, "u3[2]");
    assert_close(r.u3[9], 0.796_816_392_834_753_1, "u3[9]");
}

// Kills: MU5 (missing sqrt on variance: zu3[9] -> 10.44...).
#[test]
fn main_fixture_zu3_pinned() {
    let r = person_fit_np(&main_fixture()).unwrap();
    assert_close(r.zu3[0], -1.472_357_950_198_406_5, "zu3[0]");
    assert_close(r.zu3[9], 2.315_959_529_860_312, "zu3[9]");
}

// Kills: MU6 (non-Guttman denominator vector: c_sato all NaN).
#[test]
fn main_fixture_c_sato_pinned() {
    let r = person_fit_np(&main_fixture()).unwrap();
    assert_close(r.c_sato[1], 0.888_888_888_888_888_7, "c_sato[1]");
    assert_close(r.c_sato[3], 0.727_272_727_272_727_3, "c_sato[3]");
    assert_close(r.c_sato[9], 1.555_555_555_555_555_6, "c_sato[9]");
}

// Kills: MU7 (denominator sfp+slp: cstar[9] -> 0.137...).
#[test]
fn main_fixture_cstar_pinned() {
    let r = person_fit_np(&main_fixture()).unwrap();
    assert_close(r.cstar[1], 0.444_444_444_444_444_03, "cstar[1]");
    assert_close(r.cstar[3], 0.199_999_999_999_999_37, "cstar[3]");
    assert_close(r.cstar[9], 0.777_777_777_777_777_7, "cstar[9]");
}

/// Perfect rows, statistic by statistic (spec review finding 1/9):
/// G=0, Gnormed=0, NCI=0 (NCI.R transforms BEFORE NaN->0), U3/ZU3/C/C*
/// = NaN. Reads crate outputs for rows 10 (all-0s) and 11 (all-1s).
#[test]
fn perfect_rows_contract() {
    let r = person_fit_np(&main_fixture()).unwrap();
    for &p in &[10usize, 11] {
        assert_close(r.g[p], 0.0, &format!("g[{}]", p));
        assert_close(r.gnormed[p], 0.0, &format!("gnormed[{}]", p));
        assert_close(r.nci[p], 0.0, &format!("nci[{}]", p));
        assert!(r.u3[p].is_nan(), "u3[{}] must be NaN", p);
        assert!(r.zu3[p].is_nan(), "zu3[{}] must be NaN", p);
        assert!(r.c_sato[p].is_nan(), "c_sato[{}] must be NaN", p);
        assert!(r.cstar[p].is_nan(), "cstar[{}] must be NaN", p);
    }
}

/// NCI == 1 - 2*Gnormed for NON-perfect rows only (identity between two
/// crate outputs; the pinned values above anchor each side separately).
#[test]
fn nci_gnormed_identity_nonperfect() {
    let r = person_fit_np(&main_fixture()).unwrap();
    for p in 0..10 {
        assert_close(
            r.nci[p],
            1.0 - 2.0 * r.gnormed[p],
            &format!("nci[{}] vs 1-2*gnormed", p),
        );
    }
}

/// Tie-order fixture: pi = [.75, .5, .25, .5]; columns 1 and 3 (0-based)
/// tie at .5 and R keeps ascending original index, so the ordered layout
/// is (col0, col1, col3, col2). Row 3 = [1,0,0,1] becomes [1,0,1,0] ->
/// g=1; under the opposite tie-break it would become [1,1,0,0] -> g=0.
/// Pins the ascending-index tie-break (crate outputs).
#[test]
fn tie_order_fixture_pinned() {
    let x = vec![
        vec![1.0, 1.0, 0.0, 0.0],
        vec![1.0, 0.0, 1.0, 0.0],
        vec![0.0, 1.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
    ];
    let r = person_fit_np(&x).unwrap();
    let g_exp = [0.0, 2.0, 2.0, 1.0];
    for (p, &e) in g_exp.iter().enumerate() {
        assert_close(r.g[p], e, &format!("tie g[{}]", p));
    }
    assert_close(r.nci[3], 0.5, "tie nci[3]");
    assert_close(r.u3[1], 0.5, "tie u3[1]");
    assert_close(r.zu3[1], 0.816_496_580_927_725_9, "tie zu3[1]");
    assert_close(r.c_sato[1], 1.0, "tie c_sato[1]");
    assert_close(r.cstar[2], 0.5, "tie cstar[2]");
}

/// No-tie fixture (pi = 5/6, 4/6, 3/6, 2/6, 1/6 all distinct):
/// column-permutation invariance holds because the pi-descending order
/// restores the canonical layout. Scoped per spec review finding 2 —
/// NOT asserted on the main fixture (which has ties).
#[test]
fn column_permutation_invariance_no_ties() {
    let x = vec![
        vec![1.0, 1.0, 1.0, 0.0, 0.0],
        vec![1.0, 1.0, 0.0, 1.0, 0.0],
        vec![1.0, 0.0, 1.0, 0.0, 0.0],
        vec![1.0, 1.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 1.0],
        vec![1.0, 1.0, 1.0, 1.0, 0.0],
    ];
    let perm = [3usize, 0, 4, 1, 2];
    let xp: Vec<Vec<f64>> = x
        .iter()
        .map(|row| perm.iter().map(|&j| row[j]).collect())
        .collect();
    let r0 = person_fit_np(&x).unwrap();
    let r1 = person_fit_np(&xp).unwrap();
    for p in 0..x.len() {
        for (a, b, name) in [
            (r0.g[p], r1.g[p], "g"),
            (r0.gnormed[p], r1.gnormed[p], "gnormed"),
            (r0.nci[p], r1.nci[p], "nci"),
            (r0.u3[p], r1.u3[p], "u3"),
            (r0.zu3[p], r1.zu3[p], "zu3"),
            (r0.c_sato[p], r1.c_sato[p], "c_sato"),
            (r0.cstar[p], r1.cstar[p], "cstar"),
        ] {
            if a.is_nan() {
                assert!(b.is_nan(), "perm {}[{}]", name, p);
            } else {
                assert_close(b, a, &format!("perm {}[{}]", name, p));
            }
        }
    }
    // Pin one no-tie value so this test also anchors the fixture itself.
    assert_close(r0.u3[4], 1.0, "notie u3[4]");
    assert_close(r0.zu3[4], 3.016_144_715_531_305, "notie zu3[4]");
}

/// Reversed row 9 strictly exceeds every Guttman-CONSISTENT non-perfect
/// row (0, 4, 6) on g, gnormed, u3, cstar (crate outputs both sides).
/// Wording narrowed per spec review finding 15 (row 1 also has g=10).
#[test]
fn reversed_row_exceeds_conforming_rows() {
    let r = person_fit_np(&main_fixture()).unwrap();
    for &q in &[0usize, 4, 6] {
        assert!(r.g[9] > r.g[q], "g[9] <= g[{}]", q);
        assert!(r.gnormed[9] > r.gnormed[q], "gnormed[9] <= gnormed[{}]", q);
        assert!(r.u3[9] > r.u3[q], "u3[9] <= u3[{}]", q);
        assert!(r.cstar[9] > r.cstar[q], "cstar[9] <= cstar[{}]", q);
    }
}

/// Degenerate all-equal-pi data: log-odds and pi sums collapse, so
/// U3/ZU3/C*/C denominators are 0 -> NaN outputs (not Err, not Inf);
/// G is still well-defined. Reads crate outputs.
#[test]
fn degenerate_equal_pi_yields_nan_not_inf() {
    // Two columns each with pi = 0.5.
    let x = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let r = person_fit_np(&x).unwrap();
    for p in 0..2 {
        assert!(r.u3[p].is_nan(), "u3[{}] finite in degenerate data", p);
        assert!(r.zu3[p].is_nan(), "zu3[{}] finite", p);
        assert!(r.cstar[p].is_nan(), "cstar[{}] finite", p);
        assert!(r.c_sato[p].is_nan(), "c_sato[{}] finite", p);
    }
    // g[1] = 1: ordered layout (tie keeps col 0 first), row 1 = [0, 1].
    assert_close(r.g[1], 1.0, "degenerate g[1]");
    assert_close(r.g[0], 0.0, "degenerate g[0]");
}

#[test]
fn error_paths() {
    assert!(person_fit_np(&[]).is_err(), "empty must Err");
    assert!(person_fit_np(&[vec![1.0]]).is_err(), "single item must Err");
    assert!(
        person_fit_np(&[vec![1.0, 0.0], vec![1.0]]).is_err(),
        "ragged must Err"
    );
    assert!(
        person_fit_np(&[vec![1.0, 0.5]]).is_err(),
        "non-binary must Err"
    );
    assert!(
        person_fit_np(&[vec![1.0, f64::NAN]]).is_err(),
        "NaN input must Err (missing data out of scope)"
    );
    assert!(person_fit_np(&[vec![1.0, -1.0]]).is_err(), "-1 must Err");
}

/// MC-500: 2PL-conforming data plus one planted reversed respondent.
/// The reversed respondent's U3 (crate output) must exceed the max
/// conforming U3 in >= 95% of replications.
#[test]
#[ignore]
fn mc_500_reversed_respondent_flagged_by_u3() {
    // Local LCG (crate Lcg types are module-private): splitmix-style.
    struct Rng(u64);
    impl Rng {
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
    let n = 60usize;
    let ni = 20usize;
    let mut flagged = 0usize;
    let reps = 500usize;
    for rep in 0..reps {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ (rep as u64).wrapping_mul(0xd128_2e5b_8f2d_3f4d));
        let b: Vec<f64> = (0..ni)
            .map(|i| -2.0 + 4.0 * (i as f64) / ((ni - 1) as f64))
            .collect();
        let mut x: Vec<Vec<f64>> = Vec::with_capacity(n);
        for _ in 0..n - 1 {
            let th = rng.normal();
            x.push(
                b.iter()
                    .map(|&bi| {
                        let pr = 1.0 / (1.0 + (-(th - bi)).exp());
                        if rng.next_f64() < pr {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect(),
            );
        }
        // Planted reversed respondent: answers the HARDEST half correct.
        let mut rev = vec![0.0; ni];
        for i in ni / 2..ni {
            rev[i] = 1.0;
        }
        x.push(rev);
        let r = person_fit_np(&x).unwrap();
        let max_conforming = r.u3[..n - 1]
            .iter()
            .filter(|v| !v.is_nan())
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        if r.u3[n - 1].is_nan() {
            continue;
        }
        if r.u3[n - 1] > max_conforming {
            flagged += 1;
        }
    }
    let rate = flagged as f64 / reps as f64;
    assert!(
        rate >= 0.95,
        "reversed respondent flagged in only {:.1}% of {} reps",
        rate * 100.0,
        reps
    );
}
