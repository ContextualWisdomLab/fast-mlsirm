//! Tests for selection utility analysis (`utility.rs`).
//!
//! Mutation-kill audit (all asserts read crate return values; oracle values
//! pinned from scipy via `tests/oracles/oracle_utility.py` — runnable, and
//! regenerates every fixture below — verified adversarially during spec
//! review):
//!
//! - M1 `pux = rxy * ux` -> `pux = ux` (drop rxy): killed by
//!   `utility_pux_fixture` (mutant 1.158975380667 vs good 0.579487690333).
//! - M2 Q integrand `Phi((rho*x - k)/s)` -> `Phi((rho*x + k)/s)`: killed by
//!   `taylor_russell_oracle_fixtures` (br != .5 rows; e.g. success
//!   0.942953792170 -> 0.384157434057 at rho=.5, sr=.3, br=.8).
//! - M3 `ux = phi(xc)/sr` -> `phi(xc)/(1-sr)`: killed by `utility_ux_fixture`
//!   at sr=.3 (mutant/good = 3/7).
//! - M4 role swap sr<->br inside taylor_russell (cutoffs AND divisor): killed
//!   by any sr != br fixture (0.94295 -> 0.35361 at rho=.5, sr=.3, br=.8).
//!
//! IDENTITY LIMITATION (documented; cannot be killed in principle): the mutant
//! `Q(h,k,rho) -> Q(k,h,rho)` alone is output-identical everywhere by exchange
//! symmetry of the standard bivariate normal, including at the success-ratio
//! level (the divisor sr is unchanged). No test claims to kill it; the
//! discriminating anchor for cutoff-role bugs is the M4 role-swap mutant above.

use super::*;

fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
    // Tolerances of 1e-7 on cutoff-derived quantities reflect the Acklam
    // inverse-normal-CDF precision (~1.15e-9 relative) propagated through
    // phi(xc)/sr, NOT implementation slack; all planned mutants miss by
    // >= 6 orders of magnitude more.
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got}, want {want} (tol {tol})"
    );
}

/// M3 kill: ux(sr) = phi(Phi^-1(1-sr))/sr. Oracle scipy.stats.norm.
#[test]
fn utility_ux_fixture() {
    // asymmetric sr grid (0.5 is excluded from kills: xc=0 makes some
    // mutants of xc invisible there; kept as an extra anchor only)
    let r = selection_utility(1.0, 1.0, 0.5, 0.05, 0.0, 1.0).unwrap();
    assert_close(r.ux, 2.062712807507427, 1e-7, "ux(sr=.05)");
    assert_close(r.xc, 1.644853626951472, 1e-7, "xc(sr=.05)");
    let r = selection_utility(1.0, 1.0, 0.5, 0.3, 0.0, 1.0).unwrap();
    assert_close(r.ux, 1.158975380666913, 1e-7, "ux(sr=.3)");
    let r = selection_utility(1.0, 1.0, 0.5, 0.9, 0.0, 1.0).unwrap();
    assert_close(r.ux, 0.194998146591652, 1e-7, "ux(sr=.9)");
    assert_close(r.xc, -1.281551565544600, 1e-7, "xc(sr=.9)");
}

/// M1 kill: pux = rxy * ux, NOT ux. Oracle 0.5 * 1.158975380666913.
#[test]
fn utility_pux_fixture() {
    let r = selection_utility(1.0, 1.0, 0.5, 0.3, 0.0, 1.0).unwrap();
    assert_close(r.pux, 0.579487690333456, 1e-7, "pux(rxy=.5, sr=.3)");
    // negative-validity asymmetry: pux flips sign with rxy
    let r = selection_utility(1.0, 1.0, -0.5, 0.3, 0.0, 1.0).unwrap();
    assert_close(r.pux, -0.579487690333456, 1e-7, "pux(rxy=-.5)");
}

/// BCG utility gain: n*period*sdy*pux - cost_total (iopsych utilityBcg).
#[test]
fn utility_bcg_fixtures() {
    let r = selection_utility(1.0, 10000.0, 0.5, 0.3, 0.0, 1.0).unwrap();
    assert_close(r.utility_gain, 5794.8769033346, 1e-4, "bcg base");
    // cost>0, period>1, n>1: exact algebra vs oracle (kills cost*n mutants
    // and period misplacement)
    let r = selection_utility(50.0, 8000.0, 0.4, 0.2, 25000.0, 3.0).unwrap();
    assert_close(r.utility_gain, 646908.6089787399, 1e-3, "bcg full");
}

/// M2 + M4 kills: Taylor-Russell success ratio vs scipy mvn oracle.
/// Rows are deliberately sr != br and include negative rho.
#[test]
fn taylor_russell_oracle_fixtures() {
    let cases: [(f64, f64, f64, f64, f64); 5] = [
        // (rxy, sr, br, success, q) from tests/oracles/oracle_utility.py
        (0.5, 0.5, 0.6, 0.760872752614590, 0.380436376307295),
        (0.5, 0.3, 0.2, 0.384157434057053, 0.115247230217116),
        (0.3, 0.05, 0.8, 0.935885600599995, 0.046794280030000),
        (-0.6, 0.3, 0.5, 0.209092126479301, 0.062727637943790),
        (0.8, 0.3, 0.2, 0.533433021447364, 0.160029906434209),
    ];
    for (rxy, sr, br, success, q) in cases {
        let r = taylor_russell(rxy, sr, br).unwrap();
        assert_close(
            r.success_ratio,
            success,
            1e-7,
            &format!("success(rxy={rxy}, sr={sr}, br={br})"),
        );
        assert_close(
            r.q_joint,
            q,
            1e-7,
            &format!("q(rxy={rxy}, sr={sr}, br={br})"),
        );
        assert_close(r.base_rate, br, 0.0, "base_rate echo");
    }
}

/// Analytic anchor rho=0: success == br exactly (kills BVN wiring bugs);
/// plus strict monotonicity in rxy at fixed (sr, br).
#[test]
fn taylor_russell_structure_invariants() {
    let r = taylor_russell(0.0, 0.3, 0.7).unwrap();
    assert_close(r.success_ratio, 0.7, 1e-6, "rho=0 success == br");
    assert_close(r.q_joint, 0.21, 1e-7, "rho=0 q == sr*br");
    // strict increase in rxy (structure invariant; not an identity because
    // it compares two distinct crate outputs at asymmetric inputs)
    let lo = taylor_russell(0.2, 0.3, 0.6).unwrap().success_ratio;
    let hi = taylor_russell(0.6, 0.3, 0.6).unwrap().success_ratio;
    assert!(
        hi > lo + 1e-4,
        "success ratio must increase in rxy: {lo} !< {hi}"
    );
    // sr -> 1 limit approaches br
    let near = taylor_russell(0.7, 0.9999, 0.37).unwrap().success_ratio;
    assert_close(near, 0.370037002745, 1e-6, "sr->1 limit");
}

/// Input validation: reject out-of-domain arguments with Err (not panic).
#[test]
fn utility_error_paths() {
    assert!(taylor_russell(1.0, 0.3, 0.5).is_err(), "rxy=1 rejected");
    assert!(taylor_russell(-1.0, 0.3, 0.5).is_err(), "rxy=-1 rejected");
    assert!(taylor_russell(f64::NAN, 0.3, 0.5).is_err(), "rxy NaN");
    assert!(taylor_russell(0.5, 0.0, 0.5).is_err(), "sr=0 rejected");
    assert!(taylor_russell(0.5, 1.0, 0.5).is_err(), "sr=1 rejected");
    assert!(taylor_russell(0.5, 0.3, 0.0).is_err(), "br=0 rejected");
    assert!(taylor_russell(0.5, 0.3, 1.0).is_err(), "br=1 rejected");
    assert!(
        selection_utility(0.5, 1.0, 0.5, 0.3, 0.0, 1.0).is_err(),
        "n < 1 rejected"
    );
    assert!(
        selection_utility(1.0, -1.0, 0.5, 0.3, 0.0, 1.0).is_err(),
        "sdy < 0 rejected"
    );
    assert!(
        selection_utility(1.0, 1.0, 0.5, 0.3, f64::INFINITY, 1.0).is_err(),
        "infinite cost rejected"
    );
    assert!(
        selection_utility(1.0, 1.0, 0.5, 0.3, 0.0, 0.5).is_err(),
        "period < 1 rejected"
    );
}

/// Monte-Carlo recovery (>= 500 reps): empirical selected-group success ratio
/// and mean criterion vs crate outputs. Run with:
/// `cargo test --lib utility_mc_recovery_500 -- --ignored --nocapture`
#[test]
#[ignore]
fn utility_mc_recovery_500() {
    let (rxy, sr, br) = (0.5, 0.3, 0.6);
    let expect = taylor_russell(rxy, sr, br).unwrap();
    let pux = selection_utility(1.0, 1.0, rxy, sr, 0.0, 1.0).unwrap().pux;
    let reps = 500usize;
    let n = 20_000usize;
    let mut state: u64 = 0x1657_0016_D00D_F00Du64;
    let mut succ_sum = 0.0;
    let mut ybar_sum = 0.0;
    for _ in 0..reps {
        // draw BVN via Y = rho X + sqrt(1-rho^2) Z, top-down select on X
        let mut xs: Vec<(f64, f64)> = (0..n)
            .map(|_| {
                let x = mc_normal(&mut state);
                let z = mc_normal(&mut state);
                (x, rxy * x + (1.0 - rxy * rxy).sqrt() * z)
            })
            .collect();
        xs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let n_sel = (sr * n as f64).round() as usize;
        let yc = crate::nodes::inv_normal_cdf(1.0 - br);
        let sel = &xs[..n_sel];
        let succ = sel.iter().filter(|p| p.1 > yc).count() as f64 / n_sel as f64;
        let ybar = sel.iter().map(|p| p.1).sum::<f64>() / n_sel as f64;
        succ_sum += succ;
        ybar_sum += ybar;
    }
    let succ_mc = succ_sum / reps as f64;
    let ybar_mc = ybar_sum / reps as f64;
    eprintln!(
        "MC(500x20000): success {succ_mc:.6} vs crate {:.6}; pux {ybar_mc:.6} vs crate {pux:.6}",
        expect.success_ratio
    );
    // SE(success) ~ sqrt(p(1-p)/(reps*n_sel)) ~ 2.8e-4; band 3*SE rounded up
    assert_close(succ_mc, expect.success_ratio, 1.5e-3, "MC success ratio");
    // SE(ybar) ~ sd(Y|sel)/sqrt(reps*n_sel) ~ 5e-4; band widened for
    // top-down (order-statistic) vs threshold selection discrepancy at finite n
    assert_close(ybar_mc, pux, 4e-3, "MC mean criterion of selected");
}

/// Adversarial impl-review regressions.
///
/// HIGH: sr/br so small that `1.0 - v` rounds to 1.0 previously produced
/// NaN or silent-zero Ok results; must be Err.
#[test]
fn utility_subulp_ratio_rejected() {
    assert!(taylor_russell(0.5, 1e-17, 0.5).is_err(), "sr=1e-17 Err");
    assert!(taylor_russell(0.5, 0.5, 1e-17).is_err(), "br=1e-17 Err");
    assert!(
        selection_utility(1.0, 1.0, 0.5, 1e-17, 0.0, 1.0).is_err(),
        "selection sr=1e-17 Err"
    );
    // all finite outputs at a small-but-representable ratio
    let r = taylor_russell(0.5, 1e-12, 0.5).unwrap();
    assert!(
        r.success_ratio.is_finite() && r.q_joint.is_finite(),
        "finite outputs at sr=1e-12: {r:?}"
    );
}

/// MEDIUM: near-degenerate rho (transition width sqrt(1-rho^2) << old fixed
/// 0.25 panels) was off by ~1e-3. Oracle: scipy.integrate.quad on the same
/// conditional integral, epsrel 1e-12 (`q_quad` in
/// tests/oracles/oracle_utility.py — regenerates both fixtures below).
#[test]
fn taylor_russell_near_degenerate_rho() {
    let r = taylor_russell(-0.999999, 0.9, 0.9).unwrap();
    assert_close(
        r.success_ratio,
        0.8888888888911659,
        1e-6,
        "success(rho=-0.999999)",
    );
    let r = taylor_russell(0.999999, 1e-12, 1e-12).unwrap();
    assert_close(
        r.success_ratio,
        0.9959319518472849,
        1e-6,
        "success(rho=0.999999, tiny tails)",
    );
    // |rho| beyond the quadrature's resolvable band must Err, not silently
    // return an inaccurate value
    assert!(
        taylor_russell(1.0 - 1e-12, 0.3, 0.5).is_err(),
        "rho ~ 1-1e-12 rejected"
    );
}

/// LOW: q_joint must respect the probability bound Q <= min(sr, br); the
/// old code returned q_joint slightly above sr and clamped only success.
#[test]
fn taylor_russell_q_bound_invariant() {
    let r = taylor_russell(0.95, 0.02, 0.9).unwrap();
    assert!(
        r.q_joint <= 0.02 && r.success_ratio <= 1.0,
        "Q <= sr bound: {r:?}"
    );
}

fn mc_normal(state: &mut u64) -> f64 {
    // Box-Muller on a SplitMix64-ish stream (test-local; independent of impl)
    let next = |s: &mut u64| {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let u1 = next(state).max(1e-15);
    let u2 = next(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
