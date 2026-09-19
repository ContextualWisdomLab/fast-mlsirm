use super::*;

#[test]
fn required_rule_covers_success_and_error_contracts() {
    let (nodes, weights) = require_gh_rule(7, "quadrature size").unwrap();
    assert_eq!(nodes.len(), 7);
    assert_eq!(weights.len(), 7);
    // #1929: no node-count cap. 8 used to be "unsupported"; it must now
    // succeed like any other n >= 1.
    let (nodes8, weights8) = require_gh_rule(8, "quadrature size").unwrap();
    assert_eq!(nodes8.len(), 8);
    assert_eq!(weights8.len(), 8);
    // Only q == 0 remains an error.
    assert_eq!(
        require_gh_rule(0, "quadrature size").unwrap_err(),
        "quadrature size 0: quadrature node count must be >= 1"
    );
    assert!(crate::quadrature::gh_rule(0).is_none());
    assert!(crate::quadrature::gh_rule(61).is_some());
}

#[test]
fn dense_unidimensional_rule_is_normalized_and_symmetric() {
    for q in [61, 81, 121] {
        let (nodes, weights) = crate::quadrature::gh_rule_unidim(q).unwrap();
        assert_eq!(nodes.len(), q);
        assert_eq!(weights.len(), q);
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-14);
        let second = nodes
            .iter()
            .zip(weights)
            .map(|(node, weight)| weight * node.powi(2))
            .sum::<f64>();
        let fourth = nodes
            .iter()
            .zip(weights)
            .map(|(node, weight)| weight * node.powi(4))
            .sum::<f64>();
        assert!((second - 1.0).abs() < 1e-14);
        assert!((fourth - 3.0).abs() < 1e-12);
        for i in 0..q {
            assert!((nodes[i] + nodes[q - 1 - i]).abs() < 1e-14);
            assert!((weights[i] - weights[q - 1 - i]).abs() < 1e-14);
        }
    }
}

/// #1929: node count must no longer be capped at 41. Any n >= 1 is
/// generated on demand via Golub & Welsch (1969) (see module-level comment
/// in quadrature.rs for full citations).
#[test]
fn arbitrary_node_counts_above_the_old_cap_are_accepted() {
    for n in [42, 61, 81, 100, 121, 200, 241, 481] {
        let (nodes, weights) = require_gh_rule(n, "q").unwrap();
        assert_eq!(nodes.len(), n);
        assert_eq!(weights.len(), n);
    }
}

#[test]
fn zero_nodes_is_rejected_with_a_named_error() {
    let err = require_gh_rule(0, "q_theta").unwrap_err();
    assert!(err.contains("q_theta"));
    assert!(err.contains(">= 1"));
}

#[test]
fn computed_rules_agree_with_embedded_tables() {
    // The embedded tables (7..=41) were generated independently via
    // numpy.polynomial.hermite_e.hermegauss; recomputing them with the
    // Golub-Welsch eigensolve must reproduce the same nodes/weights to
    // float tolerance, which cross-validates the new solver against a
    // trusted reference implementation.
    for &n in &[7usize, 11, 15, 21, 31, 41] {
        let (table_nodes, table_weights) = gh_rule(n).unwrap();
        let (computed_nodes, computed_weights) = gauss_hermite_probabilists(n).unwrap();
        for i in 0..n {
            assert!(
                (table_nodes[i] - computed_nodes[i]).abs() < 1e-9,
                "node {i} mismatch for n={n}: table={} computed={}",
                table_nodes[i],
                computed_nodes[i]
            );
            assert!(
                (table_weights[i] - computed_weights[i]).abs() < 1e-9,
                "weight {i} mismatch for n={n}: table={} computed={}",
                table_weights[i],
                computed_weights[i]
            );
        }
    }
}

#[test]
fn computed_rules_are_symmetric_and_weights_sum_to_one() {
    for &n in &[7usize, 41, 121, 200, 241, 481] {
        let (nodes, weights) = require_gh_rule(n, "q").unwrap();
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "n={n} weight sum={sum}");
        for i in 0..n {
            assert!(
                (nodes[i] + nodes[n - 1 - i]).abs() < 1e-6,
                "n={n} asymmetric nodes at {i}"
            );
            assert!(
                (weights[i] - weights[n - 1 - i]).abs() < 1e-9,
                "n={n} asymmetric weights at {i}"
            );
        }
        if n % 2 == 1 {
            assert!(nodes[n / 2].abs() < 1e-9, "n={n} center node not 0");
        }
    }
}

/// Exact integration of standard-normal moments E[X^k] for k <= 2n - 1
/// (Golub & Welsch, 1969, Theorem, p. 222). E[X^k] = 0 for odd k, and
/// (k-1)!! for even k.
#[test]
fn computed_rules_integrate_standard_normal_moments_exactly() {
    fn double_factorial_moment(k: usize) -> f64 {
        if k % 2 == 1 {
            return 0.0;
        }
        let mut prod = 1.0_f64;
        let mut m = k;
        while m >= 2 {
            prod *= (m - 1) as f64;
            m -= 2;
        }
        prod
    }

    for &n in &[7usize, 41, 121, 241, 481] {
        let (nodes, weights) = require_gh_rule(n, "q").unwrap();
        let max_degree = 2 * n - 1;
        for k in 0..=max_degree.min(20) {
            let approx: f64 = nodes
                .iter()
                .zip(weights.iter())
                .map(|(&x, &w)| w * x.powi(k as i32))
                .sum();
            let exact = double_factorial_moment(k);
            let tol = 1e-6 * (1.0 + exact.abs());
            assert!(
                (approx - exact).abs() < tol,
                "n={n} k={k}: approx={approx} exact={exact}"
            );
        }
    }
}
