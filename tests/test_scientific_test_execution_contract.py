from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PERSONFIT_TESTS = ROOT / "tests" / "unit" / "personfit_np_tests.rs"


def test_personfit_monte_carlo_acceptance_is_not_ignored() -> None:
    """Keep deterministic person-fit Monte Carlo acceptance in the normal Rust suite."""
    source = PERSONFIT_TESTS.read_text(encoding="utf-8")
    ignored = (
        "#[test]\n#[ignore]\n"
        "fn mc_500_reversed_respondent_flagged_by_u3()"
    )
    active = "#[test]\nfn mc_500_reversed_respondent_flagged_by_u3()"

    assert ignored not in source, (
        "the deterministic 500-rep person-fit Monte Carlo acceptance is skipped; "
        "scientific acceptance must run rather than rely on #[ignore]"
    )
    assert active in source, (
        "the deterministic person-fit Monte Carlo acceptance must remain an active Rust test"
    )
