use super::*;

#[test]
fn required_rule_covers_success_and_error_contracts() {
    let (nodes, weights) = require_gh_rule(7, "quadrature size").unwrap();
    assert_eq!(nodes.len(), 7);
    assert_eq!(weights.len(), 7);
    assert_eq!(
        require_gh_rule(8, "quadrature size").unwrap_err(),
        "unsupported quadrature size 8"
    );
    assert!(crate::quadrature::gh_rule(61).is_none());
}

#[test]
fn dense_unidimensional_rule_is_normalized_and_symmetric() {
    let (nodes, weights) = crate::quadrature::gh_rule_unidim(61).unwrap();
    assert_eq!(nodes.len(), 61);
    assert_eq!(weights.len(), 61);
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
    for i in 0..61 {
        assert!((nodes[i] + nodes[60 - i]).abs() < 1e-14);
        assert!((weights[i] - weights[60 - i]).abs() < 1e-14);
    }
}
