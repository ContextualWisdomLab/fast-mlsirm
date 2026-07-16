//! Latent-space integration node sets shared by the marginal estimator and
//! the scoring module.
//!
//! Three constructions for the `xi in R^K` integral (the trait margins stay
//! on 1-D Gauss-Hermite):
//!
//! * `gh_tensor` — tensor-product Gauss-Hermite grid (`q^K` nodes). Exact for
//!   near-polynomial integrands, exponential in `K`; the default for `K <= 3`.
//! * `halton` — Quasi-Monte Carlo: Halton low-discrepancy points mapped
//!   through the inverse normal CDF, weights `1/N`. Error `O(N^-1 (log N)^K)`
//!   vs `O(N^-1/2)` for plain MC (Jank 2005, QMC-EM). An optional Cranley-
//!   Patterson random shift gives randomized QMC.
//! * `mc` — plain Monte Carlo EM draws (Wei & Tanner 1990; Meng & Schilling
//!   1996 for item factor analysis): seeded, reproducible standard-normal
//!   points, weights `1/N`.
//!
//! All node sets are deterministic given their parameters — the Rust<->NumPy
//! parity contract extends to the QMC/MC constructions (same Halton radical
//! inverse, same inverse-CDF coefficients, same generator).

use crate::quadrature::gh_rule;

/// A weighted node set for the latent-space integral: `grid` is row-major
/// `n x latent_dim`, `logw` the log integration weights (summing to ~1).
pub struct XiNodes {
    pub grid: Vec<f64>,
    pub logw: Vec<f64>,
}

/// Repository resource limits mirrored by Python's `FitConfig` validation.
pub const MAX_XI_POINTS: usize = 1_000_000;
pub const MAX_XI_LATENT_DIM: usize = 8;

/// How to build the latent-space node set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum XiRule {
    /// Tensor Gauss-Hermite with `q_xi` nodes per axis.
    GaussHermite { q_xi: usize },
    /// Halton QMC with `n` points; `shift_seed` != 0 applies a Cranley-
    /// Patterson random shift (randomized QMC).
    Halton { n: usize, shift_seed: u64 },
    /// Seeded Monte Carlo with `n` standard-normal points.
    MonteCarlo { n: usize, seed: u64 },
}

pub fn build_xi_nodes(rule: XiRule, latent_dim: usize) -> Result<XiNodes, String> {
    if !(1..=MAX_XI_LATENT_DIM).contains(&latent_dim) {
        return Err(format!(
            "latent_dim must be in 1..={MAX_XI_LATENT_DIM} for latent-space nodes"
        ));
    }
    match rule {
        XiRule::GaussHermite { q_xi } => {
            let (nodes, weights) =
                gh_rule(q_xi).ok_or_else(|| format!("unsupported quadrature size {q_xi}"))?;
            if latent_dim > 3 {
                return Err(
                    "tensor Gauss-Hermite supports latent_dim <= 3; use Halton/MonteCarlo"
                        .into(),
                );
            }
            let n = q_xi
                .checked_pow(latent_dim as u32)
                .ok_or("Gauss-Hermite node count overflows usize")?;
            let grid_len = n
                .checked_mul(latent_dim)
                .ok_or("Gauss-Hermite grid length overflows usize")?;
            let mut grid = vec![0.0_f64; grid_len];
            let mut logw = vec![0.0_f64; n];
            for j in 0..n {
                let mut rem = j;
                for k in 0..latent_dim {
                    let idx = rem % q_xi;
                    rem /= q_xi;
                    grid[j * latent_dim + k] = nodes[idx];
                    logw[j] += weights[idx].ln();
                }
            }
            Ok(XiNodes { grid, logw })
        }
        XiRule::Halton { n, shift_seed } => {
            let grid_len = checked_stochastic_grid_len("Halton", n, latent_dim)?;
            if latent_dim > HALTON_PRIMES.len() {
                return Err(format!(
                    "Halton rule supports latent_dim <= {}",
                    HALTON_PRIMES.len()
                ));
            }
            let mut shift = vec![0.0_f64; latent_dim];
            if shift_seed != 0 {
                let mut state = shift_seed;
                for s in shift.iter_mut() {
                    *s = lcg_uniform(&mut state);
                }
            }
            let mut grid = vec![0.0_f64; grid_len];
            for j in 0..n {
                for k in 0..latent_dim {
                    // skip the first point (index j+1) — Halton index 0 is 0.
                    let mut u = radical_inverse(j as u64 + 1, HALTON_PRIMES[k]) + shift[k];
                    if u >= 1.0 {
                        u -= 1.0;
                    }
                    grid[j * latent_dim + k] = inv_normal_cdf(u.clamp(1e-12, 1.0 - 1e-12));
                }
            }
            Ok(XiNodes { grid, logw: vec![-(n as f64).ln(); n] })
        }
        XiRule::MonteCarlo { n, seed } => {
            let grid_len = checked_stochastic_grid_len("MonteCarlo", n, latent_dim)?;
            let mut state = seed.max(1);
            let mut grid = vec![0.0_f64; grid_len];
            for v in grid.iter_mut() {
                // Box-Muller on LCG uniforms (deterministic, mirrored in NumPy).
                *v = normal_draw(&mut state);
            }
            Ok(XiNodes { grid, logw: vec![-(n as f64).ln(); n] })
        }
    }
}

fn checked_stochastic_grid_len(rule: &str, n: usize, latent_dim: usize) -> Result<usize, String> {
    if n == 0 {
        return Err(format!("{rule} rule needs n >= 1"));
    }
    if n > MAX_XI_POINTS {
        return Err(format!(
            "{rule} rule supports at most {MAX_XI_POINTS} points; got {n}"
        ));
    }
    n.checked_mul(latent_dim)
        .ok_or_else(|| format!("{rule} grid length overflows usize"))
}

const HALTON_PRIMES: [u64; 6] = [2, 3, 5, 7, 11, 13];

/// Van der Corput radical inverse of `i` in base `b`.
fn radical_inverse(mut i: u64, b: u64) -> f64 {
    let mut inv = 0.0_f64;
    let mut f = 1.0 / b as f64;
    while i > 0 {
        inv += (i % b) as f64 * f;
        i /= b;
        f /= b as f64;
    }
    inv
}

#[inline]
fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state
}

#[inline]
fn lcg_uniform(state: &mut u64) -> f64 {
    (lcg_next(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn normal_draw(state: &mut u64) -> f64 {
    let u1 = lcg_uniform(state).max(1e-12);
    let u2 = lcg_uniform(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Acklam's rational approximation to the standard-normal inverse CDF
/// (relative error < 1.15e-9; the same coefficients are used by the NumPy
/// reference for parity).
pub fn inv_normal_cdf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gh_tensor_matches_marginal_grid_convention() {
        let nodes = build_xi_nodes(XiRule::GaussHermite { q_xi: 7 }, 2).unwrap();
        assert_eq!(nodes.grid.len(), 49 * 2);
        let total: f64 = nodes.logw.iter().map(|w| w.exp()).sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn halton_points_have_moments_of_standard_normal() {
        let nodes = build_xi_nodes(XiRule::Halton { n: 4096, shift_seed: 0 }, 2).unwrap();
        for k in 0..2 {
            let vals: Vec<f64> = (0..4096).map(|j| nodes.grid[j * 2 + k]).collect();
            let mean = vals.iter().sum::<f64>() / 4096.0;
            let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / 4096.0;
            assert!(mean.abs() < 0.02, "halton mean off: {mean}");
            assert!((var - 1.0).abs() < 0.05, "halton var off: {var}");
        }
    }

    #[test]
    fn mc_points_are_reproducible_and_gaussian() {
        let a = build_xi_nodes(XiRule::MonteCarlo { n: 2048, seed: 42 }, 3).unwrap();
        let b = build_xi_nodes(XiRule::MonteCarlo { n: 2048, seed: 42 }, 3).unwrap();
        assert_eq!(a.grid, b.grid);
        let mean = a.grid.iter().sum::<f64>() / a.grid.len() as f64;
        assert!(mean.abs() < 0.05);
    }

    #[test]
    fn inv_normal_cdf_reference_values() {
        assert!((inv_normal_cdf(0.5)).abs() < 1e-12);
        assert!((inv_normal_cdf(0.975) - 1.959963984540054).abs() < 1e-8);
        assert!((inv_normal_cdf(0.025) + 1.959963984540054).abs() < 1e-8);
        assert!((inv_normal_cdf(1e-6) + 4.753424308822899).abs() < 1e-6);
    }

    #[test]
    fn rqmc_shift_changes_points_but_not_moments() {
        let a = build_xi_nodes(XiRule::Halton { n: 1024, shift_seed: 7 }, 2).unwrap();
        let b = build_xi_nodes(XiRule::Halton { n: 1024, shift_seed: 0 }, 2).unwrap();
        assert_ne!(a.grid, b.grid);
        let mean = a.grid.iter().sum::<f64>() / a.grid.len() as f64;
        assert!(mean.abs() < 0.05);
    }

    #[test]
    fn invalid_rules_rejected() {
        assert!(build_xi_nodes(XiRule::GaussHermite { q_xi: 12 }, 2).is_err());
        assert!(build_xi_nodes(XiRule::GaussHermite { q_xi: 7 }, 4).is_err());
        assert!(build_xi_nodes(XiRule::Halton { n: 0, shift_seed: 0 }, 2).is_err());
        assert!(build_xi_nodes(XiRule::MonteCarlo { n: 0, seed: 1 }, 2).is_err());
    }

    #[test]
    fn node_rules_reject_overflow_without_panicking() {
        for rule in [
            XiRule::Halton {
                n: usize::MAX,
                shift_seed: 0,
            },
            XiRule::MonteCarlo {
                n: usize::MAX,
                seed: 1,
            },
        ] {
            let result = std::panic::catch_unwind(|| build_xi_nodes(rule, 2));
            assert!(
                result.is_ok(),
                "node-size overflow must return Err, not panic"
            );
            assert!(result.unwrap().is_err());
        }
    }

    #[test]
    fn node_rules_reject_oversized_point_counts() {
        assert!(build_xi_nodes(
            XiRule::Halton {
                n: MAX_XI_POINTS + 1,
                shift_seed: 0,
            },
            1,
        )
        .is_err());
        assert!(build_xi_nodes(
            XiRule::MonteCarlo {
                n: MAX_XI_POINTS + 1,
                seed: 1,
            },
            1,
        )
        .is_err());
    }

    #[test]
    fn node_rules_reject_unsafe_latent_dimensions() {
        assert!(build_xi_nodes(
            XiRule::Halton {
                n: 1,
                shift_seed: 0,
            },
            0,
        )
        .is_err());
        assert!(build_xi_nodes(
            XiRule::MonteCarlo { n: 1, seed: 1 },
            MAX_XI_LATENT_DIM + 1,
        )
        .is_err());
    }
}


#[cfg(test)]
mod coverage_branch_tests {
    use super::*;

    #[test]
    fn gh_rule_none_for_unsupported_size() {
        // build_xi_nodes surfaces the gh_rule None branch as an error
        assert!(build_xi_nodes(XiRule::GaussHermite { q_xi: 999 }, 1).is_err());
        assert!(crate::quadrature::gh_rule(999).is_none());
        assert!(crate::quadrature::gh_rule(21).is_some());
    }

    #[test]
    fn halton_rejects_high_latent_dim() {
        assert!(build_xi_nodes(XiRule::Halton { n: 8, shift_seed: 0 }, 7).is_err());
        // a valid Halton grid with a nonzero shift seed exercises the shift path
        let nodes = build_xi_nodes(XiRule::Halton { n: 16, shift_seed: 42 }, 2).unwrap();
        assert_eq!(nodes.grid.len(), 16 * 2);
    }
}
