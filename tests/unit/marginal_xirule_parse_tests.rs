use super::XiRuleKind;

#[test]
fn parse_covers_all_arms() {
    assert_eq!(XiRuleKind::parse("gh"), Some(XiRuleKind::GaussHermite));
    assert_eq!(
        XiRuleKind::parse("gauss-hermite"),
        Some(XiRuleKind::GaussHermite)
    );
    assert_eq!(XiRuleKind::parse("qmc"), Some(XiRuleKind::Halton));
    assert_eq!(XiRuleKind::parse("halton"), Some(XiRuleKind::Halton));
    assert_eq!(XiRuleKind::parse("mc"), Some(XiRuleKind::MonteCarlo));
    assert_eq!(
        XiRuleKind::parse("monte-carlo"),
        Some(XiRuleKind::MonteCarlo)
    );
    assert_eq!(XiRuleKind::parse("nope"), None);
}
