use super::*;

#[test]
fn gh_rule_none_for_unsupported_size() {
    // #1929: no node-count cap; build_xi_nodes surfaces the gh_rule None
    // branch (q == 0) as an error, and q=999 is now a perfectly valid rule.
    assert!(build_xi_nodes(XiRule::GaussHermite { q_xi: 0 }, 1).is_err());
    assert!(crate::quadrature::gh_rule(0).is_none());
    assert!(crate::quadrature::gh_rule(21).is_some());
    assert!(crate::quadrature::gh_rule(999).is_some());
}

#[test]
fn halton_rejects_high_latent_dim() {
    assert!(build_xi_nodes(
        XiRule::Halton {
            n: 8,
            shift_seed: 0
        },
        7
    )
    .is_err());
    // a valid Halton grid with a nonzero shift seed exercises the shift path
    let nodes = build_xi_nodes(
        XiRule::Halton {
            n: 16,
            shift_seed: 42,
        },
        2,
    )
    .unwrap();
    assert_eq!(nodes.grid.len(), 16 * 2);
}
