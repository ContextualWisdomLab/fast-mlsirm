from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PERSONFIT_TESTS = ROOT / "tests" / "unit" / "personfit_np_tests.rs"
TARGET_TEST = "mc_500_reversed_respondent_flagged_by_u3"


def _personfit_acceptance_attributes(source: str) -> tuple[str, ...]:
    """Return Rust attributes attached to the guarded Monte Carlo test."""
    lines = source.splitlines()
    target = f"fn {TARGET_TEST}() {{"
    try:
        target_index = next(
            index for index, line in enumerate(lines) if line.strip() == target
        )
    except StopIteration as exc:
        raise AssertionError(
            "the deterministic person-fit Monte Carlo acceptance test is missing"
        ) from exc

    attributes: list[str] = []
    index = target_index - 1
    seen_attribute = False
    while index >= 0:
        stripped = lines[index].strip()
        if stripped.startswith("#[") and stripped.endswith("]"):
            attributes.append(stripped)
            seen_attribute = True
        elif seen_attribute and (not stripped or stripped.startswith("///")):
            pass
        else:
            break
        index -= 1

    return tuple(reversed(attributes))


def _personfit_acceptance_is_ignored(source: str) -> bool:
    """Report whether the guarded acceptance carries the direct ignore marker."""
    return "#[ignore]" in _personfit_acceptance_attributes(source)


def test_personfit_monte_carlo_acceptance_is_not_ignored() -> None:
    """Keep deterministic person-fit Monte Carlo acceptance in the normal Rust suite."""
    source = PERSONFIT_TESTS.read_text(encoding="utf-8")
    attributes = _personfit_acceptance_attributes(source)

    assert "#[test]" in attributes, (
        "the deterministic person-fit Monte Carlo acceptance must remain a Rust test"
    )
    assert not _personfit_acceptance_is_ignored(source), (
        "the deterministic 500-rep person-fit Monte Carlo acceptance is skipped; "
        "scientific acceptance must run rather than rely on #[ignore]"
    )


def test_ignore_guard_is_attribute_order_independent() -> None:
    """Reject an ignore attribute even when it precedes the test attribute."""
    reordered = f"#[ignore]\n#[test]\nfn {TARGET_TEST}() {{}}\n"

    assert _personfit_acceptance_is_ignored(reordered)


def test_ignore_guard_rejects_reason_form() -> None:
    """Reject Rust's name-value ignore syntax as non-execution evidence."""
    reasoned = f'#[test]\n#[ignore = "slow"]\nfn {TARGET_TEST}() {{}}\n'

    assert _personfit_acceptance_is_ignored(reasoned)
