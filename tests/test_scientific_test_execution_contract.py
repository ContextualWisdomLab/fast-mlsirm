from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PERSONFIT_TESTS = ROOT / "tests" / "unit" / "personfit_np_tests.rs"
TARGET_TEST = "mc_500_reversed_respondent_flagged_by_u3"


def _personfit_acceptance_is_ignored(source: str) -> bool:
    """Detect the currently guarded ignored-test spelling."""
    return f"#[test]\n#[ignore]\nfn {TARGET_TEST}()" in source


def test_personfit_monte_carlo_acceptance_is_not_ignored() -> None:
    """Keep deterministic person-fit Monte Carlo acceptance in the normal Rust suite."""
    source = PERSONFIT_TESTS.read_text(encoding="utf-8")
    active = f"#[test]\nfn {TARGET_TEST}()"

    assert not _personfit_acceptance_is_ignored(source), (
        "the deterministic 500-rep person-fit Monte Carlo acceptance is skipped; "
        "scientific acceptance must run rather than rely on #[ignore]"
    )
    assert active in source, (
        "the deterministic person-fit Monte Carlo acceptance must remain an active Rust test"
    )


def test_ignore_guard_is_attribute_order_independent() -> None:
    """Reject an ignore attribute even when it precedes the test attribute."""
    reordered = f"#[ignore]\n#[test]\nfn {TARGET_TEST}() {{}}\n"

    assert _personfit_acceptance_is_ignored(reordered)
