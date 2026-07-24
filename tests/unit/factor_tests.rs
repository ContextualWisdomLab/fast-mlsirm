//! Tests for minres factor analysis + omega_total_1f (`factor.rs`).
//!
//! Oracle: independent scipy L-BFGS-B transcription of psych fa.R
//! (session `files/oracle_minres.py`), fixtures pinned at 12 decimals.
//! Same optimizer family as R's `optim(method="L-BFGS-B")`; NOT claimed
//! bit-identical to any R run.
//!
//! Mutation-kill audit (every assert reads crate outputs):
//! - M1 EXECUTED: flip the sign convention in `minres_fa_corr`
//!   (`colsum < 0.0` -> `colsum > 0.0`) => oracle-parity loadings FAIL.
//! - M2 EXECUTED: change the objective from strictly-lower-triangle to all
//!   off-diagonal residuals (x2 scale, same argmin) => this is UNKILLABLE
//!   by any argmin-based assert (disclosed limitation); it is killed ONLY
//!   by the absolute-objective anchor in `fa_nf1_absolute_objective_anchor`
//!   (0.7076 vs 1.4153).
//! - M3 EXECUTED: sort eigenvalues ascending instead of descending in
//!   `symmetric_eigen_desc` => parity + rank-1 recovery FAIL.
//! - M4 EXECUTED: drop `sum psi` from the omega denominator => omega -> 1,
//!   `fa_omega_6var_oracle_parity` FAILs.
//! - Documented-only: swapping the FD-fallback pass order (pass 0 vs 1)
//!   changes only the search path, not the fixed point asserted on; it is
//!   principle-unkillable by value asserts and is instead pinned by the
//!   KKT-violation assert (any non-convergent path fails `converged`).

use super::*;

const S9: [f64; 81] = [
    1.0, 0.56, 0.48, 0.40, 0.32, 0.0, 0.0, 0.0, 0.0, //
    0.56, 1.0, 0.42, 0.35, 0.28, 0.0, 0.0, 0.0, 0.0, //
    0.48, 0.42, 1.0, 0.30, 0.24, 0.0, 0.0, 0.0, 0.0, //
    0.40, 0.35, 0.30, 1.0, 0.20, 0.0, 0.0, 0.0, 0.0, //
    0.32, 0.28, 0.24, 0.20, 1.0, 0.21, 0.18, 0.15, 0.135, //
    0.0, 0.0, 0.0, 0.0, 0.21, 1.0, 0.42, 0.35, 0.315, //
    0.0, 0.0, 0.0, 0.0, 0.18, 0.42, 1.0, 0.30, 0.27, //
    0.0, 0.0, 0.0, 0.0, 0.15, 0.35, 0.30, 1.0, 0.225, //
    0.0, 0.0, 0.0, 0.0, 0.135, 0.315, 0.27, 0.225, 1.0,
];

/// Oracle parity, 9 variables / 2 factors (exact-structure matrix).
/// Asserts read: crate `loadings`, `uniquenesses`, `objective`,
/// `kkt_violation`, `converged`. Kills M1, M3, optimizer breakage,
/// eigenvalue-clamp removal that yields NaN loadings.
#[test]
fn fa_oracle_parity_9x2() {
    let r = minres_fa_corr(&S9, 9, 2).unwrap();
    assert!(r.converged, "kkt_violation = {}", r.kkt_violation);
    assert!(r.kkt_violation < 1e-6);
    assert!(r.objective < 1e-9, "objective = {}", r.objective);
    #[rustfmt::skip]
    let want_l = [
        0.780565885133, -0.175262374782,
        0.682995149541, -0.153354577950,
        0.585424412028, -0.131446780509,
        0.487853678110, -0.109538984208,
        0.456006332771, 0.205081019484,
        0.153354577763, 0.682995148989,
        0.131446781080, 0.585424414300,
        0.109538983915, 0.487853676587,
        0.098585085703, 0.439068310057,
    ];
    let want_psi = [
        0.359999998175,
        0.509999998034,
        0.640000004172,
        0.749999998560,
        0.749999999543,
        0.509999999905,
        0.639999997852,
        0.750000003439,
        0.797499999930,
    ];
    for j in 0..18 {
        assert!(
            (r.loadings[j] - want_l[j]).abs() < 5e-5,
            "loading[{j}] = {} vs oracle {}",
            r.loadings[j],
            want_l[j]
        );
    }
    for j in 0..9 {
        assert!(
            (r.uniquenesses[j] - want_psi[j]).abs() < 5e-5,
            "psi[{j}] = {} vs oracle {}",
            r.uniquenesses[j],
            want_psi[j]
        );
    }
    // Structure invariants read from crate outputs: unrotated columns are
    // orthogonal, communalities are the row sums of squared loadings, and
    // both sign-convention column sums are nonnegative.
    let cross: f64 = (0..9)
        .map(|j| r.loadings[j * 2] * r.loadings[j * 2 + 1])
        .sum();
    assert!(cross.abs() < 1e-6, "columns not orthogonal: {cross}");
    for j in 0..9 {
        let h2 = r.loadings[j * 2].powi(2) + r.loadings[j * 2 + 1].powi(2);
        assert!((r.communalities[j] - h2).abs() < 1e-12);
    }
    for k in 0..2 {
        let cs: f64 = (0..9).map(|j| r.loadings[j * 2 + k]).sum();
        assert!(cs >= 0.0, "column {k} sum negative: {cs}");
    }
}

/// Absolute-objective anchor: 1-factor fit of the 2-factor matrix has a
/// NONZERO minimum whose value pins the objective's scale and triangle
/// restriction. Oracle (scipy L-BFGS-B): 0.7076498390951. This is the ONLY
/// assert that kills M2 (lower-tri -> all-off-diag doubles it to 1.41530).
/// Asserts read: crate `objective`, `loadings`, `uniquenesses`.
#[test]
fn fa_nf1_absolute_objective_anchor() {
    let r = minres_fa_corr(&S9, 9, 1).unwrap();
    assert!(r.converged, "kkt_violation = {}", r.kkt_violation);
    assert!(
        (r.objective - 0.7076498390951).abs() < 1e-6,
        "objective = {}",
        r.objective
    );
    let want_l = [
        0.784393051702,
        0.690451800142,
        0.593544036321,
        0.495443115065,
        0.431236098560,
        0.090215272048,
        0.081738564688,
        0.071559166778,
        0.065837065736,
    ];
    let want_psi = [
        0.384727568730,
        0.523276332788,
        0.647705509431,
        0.754536227830,
        0.814035399097,
        0.991860082805,
        0.993318442088,
        0.994880557142,
        0.995668949643,
    ];
    for j in 0..9 {
        assert!(
            (r.loadings[j] - want_l[j]).abs() < 1e-3,
            "loading[{j}] = {} vs oracle {}",
            r.loadings[j],
            want_l[j]
        );
        assert!(
            (r.uniquenesses[j] - want_psi[j]).abs() < 1e-3,
            "psi[{j}] = {} vs oracle {}",
            r.uniquenesses[j],
            want_psi[j]
        );
    }
}

/// omega_total_1f oracle parity on a sampled-data correlation matrix
/// (numpy default_rng(20260723), n = 400, 6 variables, 1 factor).
/// Asserts read: crate `omega_total`, `fa.loadings`, `fa.uniquenesses`.
/// Kills M4 (omega denominator) and any loadings/psi drift.
#[test]
fn fa_omega_6var_oracle_parity() {
    #[rustfmt::skip]
    let r6: [f64; 36] = [
        1.000000000000, 0.498568868921, 0.625966200012, 0.370348287807, 0.540999349418, 0.424034177377,
        0.498568868921, 1.000000000000, 0.511016876110, 0.334681533406, 0.380581491599, 0.347762451610,
        0.625966200012, 0.511016876110, 1.000000000000, 0.345954222399, 0.593952865689, 0.454414314062,
        0.370348287807, 0.334681533406, 0.345954222399, 1.000000000000, 0.327886204696, 0.311070632680,
        0.540999349418, 0.380581491599, 0.593952865689, 0.327886204696, 1.000000000000, 0.415636442046,
        0.424034177377, 0.347762451610, 0.454414314062, 0.311070632680, 0.415636442046, 1.000000000000,
    ];
    let o = omega_total_1f_corr(&r6, 6).unwrap();
    assert!(o.fa.converged, "kkt_violation = {}", o.fa.kkt_violation);
    assert!(
        (o.omega_total - 0.825058734426).abs() < 1e-5,
        "omega_total = {}",
        o.omega_total
    );
    let want_l = [
        0.774100663868,
        0.619619195261,
        0.809271424864,
        0.481544336034,
        0.696500756198,
        0.573550085378,
    ];
    let want_psi = [
        0.400768161859,
        0.616072053550,
        0.345079760758,
        0.768115052738,
        0.514886696139,
        0.671040299233,
    ];
    for j in 0..6 {
        assert!(
            (o.fa.loadings[j] - want_l[j]).abs() < 5e-5,
            "loading[{j}] = {} vs oracle {}",
            o.fa.loadings[j],
            want_l[j]
        );
        assert!(
            (o.fa.uniquenesses[j] - want_psi[j]).abs() < 5e-5,
            "psi[{j}] = {} vs oracle {}",
            o.fa.uniquenesses[j],
            want_psi[j]
        );
    }
}

/// Rank-1 exact recovery: S = ll' + diag(1 - l^2) with asymmetric
/// loadings. Asserts read: crate `loadings`, `objective`, `uniquenesses`.
/// Kills eigenvector-scaling and clamp-order mutations (M3 included).
#[test]
fn fa_rank1_exact_recovery() {
    let l = [0.9, 0.8, 0.7, 0.6, 0.5];
    let p = 5;
    let mut s = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            s[i * p + j] = if i == j { 1.0 } else { l[i] * l[j] };
        }
    }
    let r = minres_fa_corr(&s, p, 1).unwrap();
    assert!(r.converged, "kkt_violation = {}", r.kkt_violation);
    assert!(r.objective < 1e-10, "objective = {}", r.objective);
    for j in 0..p {
        assert!(
            (r.loadings[j] - l[j]).abs() < 1e-4,
            "loading[{j}] = {} vs true {}",
            r.loadings[j],
            l[j]
        );
        assert!(
            (r.uniquenesses[j] - (1.0 - l[j] * l[j])).abs() < 1e-4,
            "psi[{j}] = {}",
            r.uniquenesses[j]
        );
    }
}

/// Error contract. Asserts read: crate `Err` messages.
#[test]
fn fa_error_paths() {
    // p < 3
    assert!(minres_fa_corr(&[1.0, 0.2, 0.2, 1.0], 2, 1).is_err());
    // bad nf
    assert!(minres_fa_corr(&S9, 9, 0).is_err());
    assert!(minres_fa_corr(&S9, 9, 9).is_err());
    // asymmetric
    let mut bad = S9.to_vec();
    bad[1] += 0.01;
    assert!(minres_fa_corr(&bad, 9, 1)
        .unwrap_err()
        .contains("symmetric"));
    // non-unit diagonal
    let mut bad = S9.to_vec();
    bad[0] = 2.0;
    assert!(minres_fa_corr(&bad, 9, 1).unwrap_err().contains("diagonal"));
    // non-finite
    let mut bad = S9.to_vec();
    bad[3] = f64::NAN;
    bad[27] = f64::NAN;
    assert!(minres_fa_corr(&bad, 9, 1).unwrap_err().contains("finite"));
    // wrong length
    assert!(minres_fa_corr(&S9[..80], 9, 1).is_err());
    // singular matrix -> smc start values error
    let p = 4;
    let mut sing = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            sing[i * p + j] = if i == j { 1.0 } else { 1.0 };
        }
    }
    assert!(minres_fa_corr(&sing, p, 1).is_err());
    // data entry points
    assert!(minres_fa_data(&[0.0; 10], 2, 5, 1).is_err()); // n < 3
    assert!(minres_fa_data(&[0.0; 11], 3, 4, 1).is_err()); // length mismatch
    assert!(omega_total_1f_data(&[0.0; 11], 3, 4).is_err());
    let mut d = vec![0.0; 12];
    d[5] = f64::INFINITY;
    assert!(minres_fa_data(&d, 4, 3, 1).unwrap_err().contains("finite"));
    // impl-review regression: dimension products must not overflow/panic
    // (checked_mul guards). Asserts read crate Err values.
    assert!(minres_fa_corr(&[], usize::MAX, 1)
        .unwrap_err()
        .contains("overflow"));
    assert!(minres_fa_data(&[], usize::MAX, usize::MAX, 1)
        .unwrap_err()
        .contains("overflow"));
    assert!(omega_total_1f_data(&[], usize::MAX, usize::MAX)
        .unwrap_err()
        .contains("overflow"));
}

/// Monte Carlo recovery (>= 500 reps): 1-factor data, n = 300, p = 6;
/// average estimated loadings and omega_total from `minres_fa_data` /
/// `omega_total_1f_data` recover the population values. Asserts read:
/// crate loadings and omega across replications.
#[test]
#[ignore]
fn fa_mc_recovery_500() {
    let lam = [0.75, 0.65, 0.8, 0.55, 0.7, 0.6];
    let p = 6;
    let n = 300;
    let lsum: f64 = lam.iter().sum();
    let psum: f64 = lam.iter().map(|l| 1.0 - l * l).sum();
    let omega_pop = lsum * lsum / (lsum * lsum + psum);
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
    let reps = 500;
    let mut lsums = [0.0f64; 6];
    let mut osum = 0.0f64;
    for _ in 0..reps {
        let mut x = vec![0.0; n * p];
        for r in 0..n {
            let f = next_normal();
            for (j, &l) in lam.iter().enumerate() {
                x[r * p + j] = l * f + (1.0 - l * l).sqrt() * next_normal();
            }
        }
        let fit = omega_total_1f_data(&x, n, p).unwrap();
        for j in 0..p {
            lsums[j] += fit.fa.loadings[j];
        }
        osum += fit.omega_total;
    }
    for j in 0..p {
        let mean = lsums[j] / reps as f64;
        assert!(
            (mean - lam[j]).abs() < 0.02,
            "mean loading[{j}] = {mean} vs true {}",
            lam[j]
        );
    }
    let omean = osum / reps as f64;
    assert!(
        (omean - omega_pop).abs() < 0.01,
        "mean omega = {omean} vs population {omega_pop}"
    );
}

// ---------------------------------------------------------------------------
// glb_fa (psych glbs.R glb.fa transcription; oracle: oracle_glbfa.py, an
// independent scipy L-BFGS-B reimplementation of the same psych path)
//
// Mutation-kill audit (all asserts read crate outputs):
// - M1 EXECUTED: skip the diagonal substitution in the glb ratio (use
//   sum(R)/sum(R)) => S9 glb returns 1.0, fixture FAILs.
// - M2 EXECUTED: use the 1-factor communalities instead of the nf-factor
//   refit => S9 (nf = 2) glb becomes ~0.665251972925, fixture FAILs.
// - M3 EXECUTED: drop the df < 0 single decrement => Rm3 fits nf = 4
//   instead of 3, glb and nf asserts FAIL. (S9 has nf = 2, df = 19 and r6
//   has nf = 3, df = 0 — neither triggers the decrement, hence Rm3.)

/// Pinned oracle: S9 population 2-factor matrix. glb 0.730905233399,
/// detected nf = 2 (df = 19, no decrement). Asserts read crate glb, nf,
/// communalities.
#[test]
fn glbfa_oracle_s9() {
    let g = glb_fa_corr(&S9, 9).unwrap();
    assert_eq!(g.nf, 2);
    assert!((g.glb - 0.730905233399).abs() < 1e-5, "glb = {}", g.glb);
    let want_h2 = [
        0.640000001825,
        0.490000001966,
        0.359999995828,
        0.250000001440,
        0.250000000457,
        0.490000000095,
        0.360000002148,
        0.249999996561,
        0.202500000070,
    ];
    for (j, w) in want_h2.iter().enumerate() {
        assert!(
            (g.communalities[j] - w).abs() < 1e-4,
            "h2[{j}] = {} want {w}",
            g.communalities[j]
        );
    }
}

/// Pinned oracle: r6 sampled 1-factor corr. Sampling noise makes the
/// eigenvalue count nf = 3 (df = 0, no decrement); oracle glb
/// 0.876394018918. df = 0 means the 3-factor minres fit is SATURATED, the
/// objective is flat at zero and communalities are not unique, so
/// BB+Armijo (crate) and L-BFGS-B (oracle) legitimately land on nearby
/// but distinct solutions (observed crate glb 0.873934, |delta| 2.5e-3);
/// the band below reflects that indeterminacy, not a defect. Documents
/// that glb_fa on 1-factor DATA is NOT omega_total_1f (nf > 1 is
/// detected), so no identity shortcut exists. Asserts read crate outputs.
#[test]
fn glbfa_oracle_r6() {
    #[rustfmt::skip]
    let r6: [f64; 36] = [
        1.000000000000, 0.498568868921, 0.625966200012, 0.370348287807, 0.540999349418, 0.424034177377,
        0.498568868921, 1.000000000000, 0.511016876110, 0.334681533406, 0.380581491599, 0.347762451610,
        0.625966200012, 0.511016876110, 1.000000000000, 0.345954222399, 0.593952865689, 0.454414314062,
        0.370348287807, 0.334681533406, 0.345954222399, 1.000000000000, 0.327886204696, 0.311070632680,
        0.540999349418, 0.380581491599, 0.593952865689, 0.327886204696, 1.000000000000, 0.415636442046,
        0.424034177377, 0.347762451610, 0.454414314062, 0.311070632680, 0.415636442046, 1.000000000000,
    ];
    let g = glb_fa_corr(&r6, 6).unwrap();
    assert_eq!(g.nf, 3);
    assert!((g.glb - 0.876394018918).abs() < 5e-3, "glb = {}", g.glb);
}

/// df-adjustment fixture: eigenvalue count detects nf = 4, df = 15 - 24 +
/// 6 = -3 < 0, so psych decrements to nf = 3; oracle glb 0.739790363509.
/// This is the only fixture where the decrement fires: the M3 mutant
/// (decrement dropped) fits nf = 4 and the exact `nf == 3` assert FAILs —
/// that assert is the M3 kill. The nf = 3 fit here is again df = 0
/// (saturated, flat objective, non-unique communalities with several at
/// the 0.995 bound), so the glb band is wide (observed crate glb
/// 0.752597, |delta| 1.3e-2 vs oracle) and serves as a sanity anchor
/// only. Asserts read crate nf and glb.
#[test]
fn glbfa_df_adjustment_rm3() {
    #[rustfmt::skip]
    let rm3: [f64; 36] = [
        1.000000000000, 0.318781496354, -0.085229976062, 0.335304209671, 0.344321815016, 0.328050481937,
        0.318781496354, 1.000000000000, -0.639587262249, -0.268114491438, -0.263043397262, 0.260168422634,
        -0.085229976062, -0.639587262249, 1.000000000000, 0.604841164303, -0.314993496385, 0.156083866739,
        0.335304209671, -0.268114491438, 0.604841164303, 1.000000000000, -0.178619109663, -0.278011941380,
        0.344321815016, -0.263043397262, -0.314993496385, -0.178619109663, 1.000000000000, -0.309262040181,
        0.328050481937, 0.260168422634, 0.156083866739, -0.278011941380, -0.309262040181, 1.000000000000,
    ];
    let g = glb_fa_corr(&rm3, 6).unwrap();
    assert_eq!(g.nf, 3);
    assert!((g.glb - 0.739790363509).abs() < 3e-2, "glb = {}", g.glb);
}

/// Error contract. Asserts read crate Err values.
#[test]
fn glbfa_error_paths() {
    assert!(glb_fa_corr(&[1.0; 4], 2).is_err()); // p < 3
    assert!(glb_fa_corr(&S9, 8).is_err()); // length mismatch
    assert!(glb_fa_data(&[0.0; 12], usize::MAX, usize::MAX).is_err()); // overflow
    assert!(glb_fa_data(&[f64::NAN; 12], 4, 3).is_err());
    assert!(glb_fa_data(&[0.5; 6], 2, 3).is_err()); // n < 3
    // Regression (impl-review): impossible off-diagonals must Err, not
    // panic in eigen. Asserts read crate Err.
    let mut bad = [1.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                bad[i * 3 + j] = 1e308;
            }
        }
    }
    assert!(glb_fa_corr(&bad, 3).is_err());
    // Regression (review-thread): glb denominator sum(R) can be zero for
    // valid (possibly indefinite) symmetric unit-diagonal inputs; this must
    // return an Err, not inf/NaN.
    #[rustfmt::skip]
    let zero_sum: [f64; 9] = [
        1.0, -0.15885683833831, -0.4821664994140733,
        -0.15885683833831, 1.0, -0.8589766622476167,
        -0.4821664994140733, -0.8589766622476167, 1.0,
    ];
    let err = glb_fa_corr(&zero_sum, 3).unwrap_err();
    assert!(err.contains("sum(R) too close to zero"), "err = {err}");
}

/// 500-rep MC: 1-factor model, loadings [.7,.6,.8,.5,.65,.75], n = 1000
/// per rep. Population omega_total = 0.830090791180 (spec-verify derived).
/// glb_fa's strict eigenvalue count typically detects nf > 1 on sampled
/// matrices, so glb is biased slightly UPWARD relative to omega; tolerance
/// set from an observed run (see test body comment). Asserts read crate
/// glb across reps.
#[test]
#[ignore]
fn glbfa_mc_recovery_500() {
    let mut state: u64 = 0xC0FFEE123456789;
    let mut next_u = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut next_normal = move || {
        let (u1, u2): (f64, f64) = (next_u().max(1e-12), next_u());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let lam = [0.7, 0.6, 0.8, 0.5, 0.65, 0.75];
    let (n, p, reps) = (1000usize, 6usize, 500usize);
    let mut sum_glb = 0.0;
    for _ in 0..reps {
        let mut data = vec![0.0; n * p];
        for row in data.chunks_mut(p) {
            let f = next_normal();
            for (j, x) in row.iter_mut().enumerate() {
                *x = lam[j] * f + (1.0 - lam[j] * lam[j]).sqrt() * next_normal();
            }
        }
        sum_glb += glb_fa_data(&data, n, p).unwrap().glb;
    }
    let mean_glb = sum_glb / reps as f64;
    let pop_omega = 0.830090791180;
    eprintln!("glbfa_mc_recovery_500: mean glb = {mean_glb}");
    // Observed mean over 500 reps: 0.86314 (deterministic LCG seed above),
    // i.e. +0.033 above population omega — the expected upward bias from
    // nf > 1 detection on sampled matrices; band is asymmetric accordingly.
    assert!(
        mean_glb >= pop_omega - 0.005 && mean_glb <= pop_omega + 0.045,
        "mean glb = {mean_glb} vs population omega {pop_omega}"
    );
}
