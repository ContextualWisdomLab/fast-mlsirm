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
    let nodes = build_xi_nodes(
        XiRule::Halton {
            n: 4096,
            shift_seed: 0,
        },
        2,
    )
    .unwrap();
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
    let a = build_xi_nodes(
        XiRule::Halton {
            n: 1024,
            shift_seed: 7,
        },
        2,
    )
    .unwrap();
    let b = build_xi_nodes(
        XiRule::Halton {
            n: 1024,
            shift_seed: 0,
        },
        2,
    )
    .unwrap();
    assert_ne!(a.grid, b.grid);
    let mean = a.grid.iter().sum::<f64>() / a.grid.len() as f64;
    assert!(mean.abs() < 0.05);
}

#[test]
fn invalid_rules_rejected() {
    assert!(build_xi_nodes(XiRule::GaussHermite { q_xi: 12 }, 2).is_err());
    assert!(build_xi_nodes(XiRule::GaussHermite { q_xi: 7 }, 4).is_err());
    assert!(build_xi_nodes(
        XiRule::Halton {
            n: 0,
            shift_seed: 0
        },
        2
    )
    .is_err());
    assert!(build_xi_nodes(XiRule::MonteCarlo { n: 0, seed: 1 }, 2).is_err());
}

#[test]
fn node_rules_reject_overflow_without_panicking() {
    assert!(checked_stochastic_grid_len("test", 2, usize::MAX).is_err());
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
    assert!(build_xi_nodes(XiRule::MonteCarlo { n: 1, seed: 1 }, MAX_XI_LATENT_DIM + 1,).is_err());
}

/// Deterministic LAYOUT pin for the Halton grid at D=4. A finite-difference gradient anchor
/// (used downstream in the MIRT QMC tests) reads the SAME grid for both the analytic and the
/// numeric derivative, so a transposed grid, a wrong prime-to-axis assignment, a dropped `+1`
/// index skip, or a mis-ordered row-major write is fed CONSISTENTLY to both and stays
/// invisible to that check. This pins each cell against an INDEPENDENT recomputation of the
/// exact construction, so any of those layout bugs fails here.
#[test]
fn halton_grid_layout_is_prime_per_axis_row_major() {
    let (n, d) = (37usize, 4usize);
    let nodes = build_xi_nodes(XiRule::Halton { n, shift_seed: 0 }, d).unwrap();
    assert_eq!(nodes.grid.len(), n * d);
    for j in 0..n {
        for k in 0..d {
            // axis k must use the k-th prime; point j must use radical index j+1 (skip 0).
            let expect = inv_normal_cdf(
                radical_inverse(j as u64 + 1, HALTON_PRIMES[k]).clamp(1e-12, 1.0 - 1e-12),
            );
            assert_eq!(
                nodes.grid[j * d + k],
                expect,
                "halton grid[{j}*{d}+{k}] layout mismatch (prime {})",
                HALTON_PRIMES[k]
            );
        }
    }
}

/// The QMC weights are equal `-ln(n)` (a uniform average over the prior-sampled nodes). Because
/// this constant cancels in the self-normalized posterior and in every posterior moment, a
/// wrong weight (e.g. `0` or a missing `1/n`) is invisible to every fit-level test and surfaces
/// only as a constant shift in the reported marginal loglik — a direct assertion is the ONLY
/// possible guard.
#[test]
fn qmc_weights_are_uniform_log_of_n() {
    for (grid, expect) in [
        (
            build_xi_nodes(
                XiRule::Halton {
                    n: 500,
                    shift_seed: 0,
                },
                3,
            )
            .unwrap(),
            -(500f64).ln(),
        ),
        (
            build_xi_nodes(XiRule::MonteCarlo { n: 750, seed: 5 }, 4).unwrap(),
            -(750f64).ln(),
        ),
    ] {
        assert!(
            grid.logw.iter().all(|&w| w == expect),
            "QMC logw not uniform -ln(n)"
        );
        let total: f64 = grid.logw.iter().map(|w| w.exp()).sum();
        assert!((total - 1.0).abs() < 1e-12, "sum exp(logw) != 1: {total}");
    }
}

#[test]
fn inverse_normal_rejects_probabilities_outside_the_unit_interval() {
    assert!(inv_normal_cdf(-f64::EPSILON).is_nan());
    assert!(inv_normal_cdf(1.0 + f64::EPSILON).is_nan());
    assert!(inv_normal_cdf(f64::NAN).is_nan());
}
