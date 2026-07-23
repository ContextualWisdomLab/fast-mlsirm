//! Tests for the generalizability-theory module (`gtheory.rs`).
//!
//! Every assertion reads values RETURNED BY THE CRATE (`gtheory_pi` /
//! `gtheory_pio` outputs); no assert recomputes the formula locally.
//!
//! Fixture provenance:
//! - `pi_dat` and `pio_cross_dat`: the data appendix of Huebner & Lucht
//!   (2019), with expected values pinned from an independent NumPy
//!   implementation of the same ANOVA/EMS pipeline, which reproduces the
//!   paper's published Tables 3-6 to their printed rounding.
//! - Independent fixtures: generated with numpy `default_rng(20260723)`;
//!   expected literals pinned from the same independent NumPy pipeline.
//!
//! Mutation kills. M1, M3, M5, M6 were each ACTUALLY applied to
//! src/gtheory.rs, observed to FAIL the named tests, and reverted
//! (2026-07-23). M2 and M4 are documented expectations only (same code
//! sites as M1/M3, killed by the same fixtures; not separately executed):
//! - M1 (EXECUTED) swap person/item EMS divisors in `gtheory_pi`
//!   (`/fi` <-> `/fp`): `gt_pi_paper_fixture`, `gt_pi_independent_fixture`,
//!   and `gt_pi_negative_variance_clamped` FAIL.
//! - M2 (documented) drop the `- ms_pi` subtraction in `gtheory_pi` var_p:
//!   var_p becomes 2.225 -> `gt_pi_paper_fixture` fails.
//! - M3 (EXECUTED) flip the `+ ms[6]` sign in the `gtheory_pio` var_p
//!   inversion: `gt_pio_paper_fixture` and `gt_pio_independent_fixture` FAIL.
//! - M4 (documented) drop the `s_io/(n_i' n_o')` term from the absolute
//!   error: Delta(1,1) becomes 5.3319 vs 5.3542 -> `gt_pio_paper_fixture`
//!   fails.
//! - M5 (EXECUTED) drop the clamp (`var = var_raw`) in `gtheory_pi`:
//!   `gt_pi_negative_variance_clamped` FAILS (var == var_raw identity and
//!   sign-sensitive D-study outputs).
//! - M6 (EXECUTED) transpose the pio layout (swap `i`/`o` strides): the
//!   5x4x3 independent fixture has n_i=4 != n_o=3 ->
//!   `gt_pio_independent_fixture` and `gt_pio_paper_fixture` FAIL.
//!
//! Weak identity checks (disclosed, NOT counted as kills): `n' = 1` D-study
//! columns (divisor identity point), DF values, and sigma^2(p) invariance
//! across n'.

use super::{gtheory_pi, gtheory_pio};

const TOL: f64 = 1e-9;

fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
    assert!((got - want).abs() <= tol, "{what}: got {got}, want {want}");
}

/// Huebner & Lucht (2019) appendix `pi_dat` (6 persons x 4 items).
fn pi_dat() -> Vec<f64> {
    vec![
        9., 9., 7., 4., 9., 8., 4., 6., 8., 8., 6., 2., 9., 8., 6., 3., 10., 9., 8., 7., 6., 4.,
        5., 1.,
    ]
}

/// Huebner & Lucht (2019) appendix `pio_cross_dat` (6 x 4 x 2), re-laid out
/// to the crate layout x[p*n_i*n_o + i*n_o + o].
fn pio_cross_dat() -> Vec<f64> {
    vec![
        9., 9., 9., 8., 7., 5., 4., 5., 9., 6., 8., 5., 4., 3., 6., 3., 8., 8., 8., 7., 6., 3., 2.,
        2., 9., 9., 8., 6., 6., 6., 3., 2., 10., 8., 9., 8., 8., 9., 7., 7., 6., 3., 4., 2., 5.,
        3., 1., 2.,
    ]
}

/// Paper fixture: reproduces Tables 3 and 4 of Huebner & Lucht (2019).
/// Asserts read: ss, ms, var_raw, var, d_study rows from the crate result.
/// Kills M1, M2 (see header).
#[test]
fn gt_pi_paper_fixture() {
    let r = gtheory_pi(&pi_dat(), 6, 4, &[4, 10]).unwrap();
    // ANOVA table (paper Table 3, pinned at full precision).
    assert_eq!(r.df, [5.0, 3.0, 15.0]);
    assert_close(r.ss[0], 44.5, TOL, "ss_p");
    assert_close(r.ss[1], 76.333333333333, 1e-9, "ss_i");
    assert_close(r.ss[2], 19.166666666667, 1e-9, "ss_pi");
    assert_close(r.ms[0], 8.9, TOL, "ms_p");
    assert_close(r.ms[1], 25.444444444444, 1e-9, "ms_i");
    assert_close(r.ms[2], 1.277777777778, 1e-9, "ms_pi");
    // Variance components (paper: 1.91 / 4.03 / 1.28).
    assert_close(r.var_raw[0], 1.905555555556, 1e-9, "var_p");
    assert_close(r.var_raw[1], 4.027777777778, 1e-9, "var_i");
    assert_close(r.var_raw[2], 1.277777777778, 1e-9, "var_pi");
    // All components positive here: clamped == raw.
    assert_eq!(r.var, r.var_raw);
    // D study (paper Table 4 columns n_i' = 4 and 10).
    let d4 = &r.d_study[0];
    assert_eq!((d4.n_i_prime, d4.n_o_prime), (4, 1));
    assert_close(d4.rel_error_var, 0.319444444444, 1e-9, "delta(4)");
    assert_close(d4.abs_error_var, 1.326388888889, 1e-9, "Delta(4)");
    assert_close(d4.generalizability, 0.856429463171, 1e-9, "Erho2(4)");
    assert_close(d4.dependability, 0.589600343790, 1e-9, "Phi(4)");
    let d10 = &r.d_study[1];
    assert_close(d10.rel_error_var, 0.127777777778, 1e-9, "delta(10)");
    assert_close(d10.abs_error_var, 0.530555555556, 1e-9, "Delta(10)");
    assert_close(d10.generalizability, 0.937158469945, 1e-9, "Erho2(10)");
    assert_close(d10.dependability, 0.782212086659, 1e-9, "Phi(10)");
}

/// Paper fixture: reproduces Tables 5 and 6 of Huebner & Lucht (2019).
/// Asserts read: ss, ms, var_raw, d_study rows from the crate result.
/// Kills M3, M4 (see header).
#[test]
fn gt_pio_paper_fixture() {
    let r = gtheory_pio(&pio_cross_dat(), 6, 4, 2, &[(4, 2), (1, 1), (5, 3)]).unwrap();
    assert_eq!(r.df, [5.0, 3.0, 1.0, 15.0, 5.0, 3.0, 15.0]);
    let want_ss = [
        112.9375,
        117.895833333333,
        15.1875,
        35.479166666667,
        5.9375,
        2.895833333333,
        12.479166666667,
    ];
    let want_ms = [
        22.5875,
        39.298611111111,
        15.1875,
        2.365277777778,
        1.1875,
        0.965277777778,
        0.831944444444,
    ];
    // Paper Table 5: 2.48 / 3.07 / 0.58 / 0.77 / 0.09 / 0.02 / 0.83.
    let want_var = [
        2.483333333333,
        3.066666666667,
        0.577777777778,
        0.766666666667,
        0.088888888889,
        0.022222222222,
        0.831944444444,
    ];
    for k in 0..7 {
        assert_close(r.ss[k], want_ss[k], 1e-9, "pio ss");
        assert_close(r.ms[k], want_ms[k], 1e-9, "pio ms");
        assert_close(r.var_raw[k], want_var[k], 1e-9, "pio var");
    }
    assert_eq!(r.var, r.var_raw);
    // Paper Table 6 columns (n_i', n_o') = (4,2), (1,1), (5,3).
    let want_d = [
        // rel, abs, Erho2, Phi
        [0.340104166667, 1.3984375, 0.879542519830, 0.639742385617],
        [1.6875, 5.354166666667, 0.595404595405, 0.316852737905],
        [
            0.238425925926,
            1.045833333333,
            0.912400068039,
            0.703659976387,
        ],
    ];
    for (row, want) in r.d_study.iter().zip(want_d.iter()) {
        assert_close(row.rel_error_var, want[0], 1e-9, "pio delta");
        assert_close(row.abs_error_var, want[1], 1e-9, "pio Delta");
        assert_close(row.generalizability, want[2], 1e-9, "pio Erho2");
        assert_close(row.dependability, want[3], 1e-9, "pio Phi");
    }
    assert_eq!((r.d_study[2].n_i_prime, r.d_study[2].n_o_prime), (5, 3));
}

/// Independent 7x5 fixture (numpy default_rng(20260723); literals pinned
/// from an independent NumPy pipeline). n_p=7 != n_i=5 keeps the divisor
/// asymmetry live at a size not used by the paper fixture.
#[test]
fn gt_pi_independent_fixture() {
    #[rustfmt::skip]
    let x = vec![
        50.1, 39.0, 41.1, 56.9, 52.3,
        52.4, 52.5, 41.8, 46.3, 37.6,
        59.9, 43.2, 52.7, 47.0, 54.5,
        34.1, 52.4, 45.1, 54.1, 32.9,
        59.7, 46.7, 57.7, 57.9, 45.5,
        54.5, 47.6, 61.2, 55.2, 45.6,
        49.8, 32.8, 69.3, 60.1, 57.7,
    ];
    let r = gtheory_pi(&x, 7, 5, &[8]).unwrap();
    assert_close(r.ss[0], 484.0, 1e-8, "ss_p");
    assert_close(r.ss[1], 439.2845714286, 1e-8, "ss_i");
    assert_close(r.ss[2], 1623.3514285714, 1e-8, "ss_pi");
    assert_close(r.ms[0], 80.6666666667, 1e-8, "ms_p");
    assert_close(r.ms[1], 109.8211428571, 1e-8, "ms_i");
    assert_close(r.ms[2], 67.6396428571, 1e-8, "ms_pi");
    assert_close(r.var_raw[0], 2.6054047619, 1e-8, "var_p");
    assert_close(r.var_raw[1], 6.0259285714, 1e-8, "var_i");
    let d = &r.d_study[0];
    assert_close(d.rel_error_var, 8.4549553571, 1e-8, "delta(8)");
    assert_close(d.abs_error_var, 9.2081964286, 1e-8, "Delta(8)");
    assert_close(d.generalizability, 0.2355623808, 1e-8, "Erho2(8)");
    assert_close(d.dependability, 0.2205428065, 1e-8, "Phi(8)");
}

/// Negative-variance anchor for the clamped-ANOVA policy (module docs):
/// item means are nearly equal but the interaction is large, so the raw
/// sigma^2_i and sigma^2_p are negative. Asserts read var_raw, var, and the
/// D-study row from the crate result. Kills M5.
#[test]
fn gt_pi_negative_variance_clamped() {
    let x = vec![5., 1., 4., 2., 6., 3., 7., 3., 8., 1., 5., 2.];
    let r = gtheory_pi(&x, 4, 3, &[3]).unwrap();
    assert_close(r.var_raw[0], -0.083333333333, 1e-9, "raw var_p < 0");
    assert_close(r.var_raw[1], -1.555555555556, 1e-9, "raw var_i < 0");
    assert!(r.var_raw[0] < 0.0 && r.var_raw[1] < 0.0);
    assert_eq!(r.var[0], 0.0);
    assert_eq!(r.var[1], 0.0);
    assert_close(r.var[2], 6.555555555556, 1e-9, "var_pi");
    let d = &r.d_study[0];
    // Clamped policy: delta = Delta = var_pi/3 (clamped var_i contributes 0).
    // Clamped var_p = 0 with a POSITIVE error variance gives Erho2 = Phi = 0
    // (zero universe-score variance), not NaN — the NaN rule only fires when
    // the whole denominator is <= 1e-12.
    assert_close(d.rel_error_var, 6.555555555556 / 3.0, 1e-9, "delta");
    assert_close(d.abs_error_var, 6.555555555556 / 3.0, 1e-9, "Delta");
    assert_eq!(d.generalizability, 0.0, "Erho2 = 0 at zero var_p");
    assert_eq!(d.dependability, 0.0, "Phi = 0 at zero var_p");
}

/// Independent 5x4x3 fixture (numpy default_rng(20260723); literals pinned
/// from an independent NumPy pipeline). All facet sizes distinct
/// (n_p=5, n_i=4, n_o=3) so any axis/stride mix-up shifts every output.
/// Raw sigma^2_pi is negative here, so this doubles as the pio clamp
/// anchor. Kills M5, M6.
#[test]
fn gt_pio_independent_fixture() {
    #[rustfmt::skip]
    let x = vec![
        19.3, 17.2, 16.0,  15.9, 18.0, 17.8,  21.4, 22.4, 12.0,  18.1, 17.1, 14.3,
        19.8, 19.6, 18.5,  21.4, 21.4, 17.8,  23.3, 26.0, 17.8,  22.2, 22.2, 21.1,
        15.8, 17.7, 14.5,  15.4, 19.6, 19.7,  21.8, 14.8, 20.3,  18.9, 10.8, 18.6,
        12.0, 16.1, 17.3,  25.4, 15.3, 16.7,  25.5, 25.5, 20.1,  21.3, 21.7, 23.2,
        16.5,  6.9, 16.1,  16.2, 15.7, 17.6,  22.8, 15.3, 16.4,  17.7, 14.4, 14.9,
    ];
    let r = gtheory_pio(&x, 5, 4, 3, &[(6, 2)]).unwrap();
    let want_ss = [
        208.164,
        128.8205,
        45.6333333333,
        82.792,
        95.475,
        56.448,
        213.077,
    ];
    let want_ms = [
        52.041,
        42.9401666667,
        22.8166666667,
        6.8993333333,
        11.934375,
        9.408,
        8.8782083333,
    ];
    let want_var = [
        3.507125,
        2.3674027778,
        0.517625,
        -0.659625,
        0.7640416667,
        0.1059583333,
        8.8782083333,
    ];
    for k in 0..7 {
        assert_close(r.ss[k], want_ss[k], 1e-8, "ind pio ss");
        assert_close(r.ms[k], want_ms[k], 1e-8, "ind pio ms");
        assert_close(r.var_raw[k], want_var[k], 1e-8, "ind pio var_raw");
    }
    // pi component is clamped; the rest pass through.
    assert!(r.var_raw[3] < 0.0);
    assert_eq!(r.var[3], 0.0);
    assert_close(r.var[0], 3.507125, 1e-8, "clamped var_p");
    // D study at (6, 2), computed from CLAMPED components.
    let d = &r.d_study[0];
    assert_close(d.rel_error_var, 1.1218715278, 1e-8, "ind delta");
    assert_close(d.abs_error_var, 1.7840810185, 1e-8, "ind Delta");
    assert_close(d.generalizability, 0.7576426076, 1e-8, "ind Erho2");
    assert_close(d.dependability, 0.6628214792, 1e-8, "ind Phi");
}

#[test]
fn gt_error_paths() {
    // Too few levels.
    assert!(gtheory_pi(&[1., 2.], 1, 2, &[1]).is_err());
    assert!(gtheory_pi(&[1., 2.], 2, 1, &[1]).is_err());
    assert!(gtheory_pio(&[0.0; 4], 1, 2, 2, &[(1, 1)]).is_err());
    // Length mismatch.
    assert!(gtheory_pi(&[1., 2., 3.], 2, 2, &[1]).is_err());
    assert!(gtheory_pio(&[0.0; 7], 2, 2, 2, &[(1, 1)]).is_err());
    // Non-finite input.
    assert!(gtheory_pi(&[1., 2., f64::NAN, 4.], 2, 2, &[1]).is_err());
    // Zero proposed size.
    assert!(gtheory_pi(&[1., 2., 3., 5.], 2, 2, &[0]).is_err());
    assert!(gtheory_pio(&[0.0; 8], 2, 2, 2, &[(1, 0)]).is_err());
}

/// 500-rep Monte Carlo: simulate the fully random p x i model with known
/// components and check that the mean raw ANOVA estimates are close to the
/// truth (the estimators are unbiased for this model). All quantities under
/// test come from `gtheory_pi` outputs.
#[test]
#[ignore]
fn gt_pi_mc_recovery_500() {
    // xorshift-style LCG + Box-Muller, matching the crate's test convention
    // of dependency-free generators.
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next_u = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut next_normal = || {
        let (u1, u2): (f64, f64) = (next_u().max(1e-12), next_u());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let (n_p, n_i) = (60usize, 12usize);
    let (sp, si, spi) = (1.5f64, 0.8f64, 0.6f64); // true variances
    let reps = 500;
    let mut sums = [0.0f64; 3];
    for _ in 0..reps {
        let mut x = vec![0.0; n_p * n_i];
        let pe: Vec<f64> = (0..n_p).map(|_| next_normal() * sp.sqrt()).collect();
        let ie: Vec<f64> = (0..n_i).map(|_| next_normal() * si.sqrt()).collect();
        for p in 0..n_p {
            for i in 0..n_i {
                x[p * n_i + i] = 10.0 + pe[p] + ie[i] + next_normal() * spi.sqrt();
            }
        }
        let r = gtheory_pi(&x, n_p, n_i, &[1]).unwrap();
        for k in 0..3 {
            sums[k] += r.var_raw[k];
        }
    }
    let means = sums.map(|s| s / reps as f64);
    // Unbiasedness: generous MC tolerances at 500 reps.
    assert!(
        (means[0] - sp).abs() < 0.10,
        "var_p bias: {}",
        means[0] - sp
    );
    assert!(
        (means[1] - si).abs() < 0.12,
        "var_i bias: {}",
        means[1] - si
    );
    assert!(
        (means[2] - spi).abs() < 0.02,
        "var_pi bias: {}",
        means[2] - spi
    );
}

/// Regression: impl-review found the old `ss_pio = ss_total - ...`
/// subtraction form returns a NEGATIVE residual SS from catastrophic
/// cancellation on large-offset additive data (`x = 1e12 + p + i + o`,
/// true pio interaction = 0; observed ss[6] = -0.037 before the fix).
/// Asserts read the crate ss/var outputs; kills reintroduction of the
/// subtraction form.
#[test]
fn gt_pio_large_offset_no_cancellation() {
    let (n_p, n_i, n_o) = (5usize, 4usize, 3usize);
    let mut x = vec![0.0; n_p * n_i * n_o];
    for p in 0..n_p {
        for i in 0..n_i {
            for o in 0..n_o {
                x[p * n_i * n_o + i * n_o + o] = 1e12 + p as f64 + i as f64 + o as f64;
            }
        }
    }
    let r = gtheory_pio(&x, n_p, n_i, n_o, &[(4, 3)]).unwrap();
    assert!(
        r.ss[6] >= 0.0 && r.ss[6] < 1e-3,
        "ss_pio must be ~0 and non-negative, got {}",
        r.ss[6]
    );
    // Interaction components are exactly-zero in truth; clamped var stays 0.
    for k in 3..7 {
        assert!(
            r.var[k].abs() < 1e-3,
            "interaction var[{k}] should be ~0, got {}",
            r.var[k]
        );
    }
}
