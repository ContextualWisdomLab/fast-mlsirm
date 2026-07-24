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
}
