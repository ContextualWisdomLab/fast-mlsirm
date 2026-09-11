"""Contracts for monotonic Architecture Decision Record identities."""

from collections import defaultdict
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
ADR_DIR = ROOT / "docs" / "adr"
ADR_INDEX = ADR_DIR / "README.md"
NUMBERED_ADR_RE = re.compile(r"^(\d{4})-.+\.md$")


def test_protected_tree_has_one_material_adr_per_number() -> None:
    """A four-digit ADR identity cannot name two material decisions in one tree."""
    by_number: dict[str, list[str]] = defaultdict(list)
    for path in ADR_DIR.glob("*.md"):
        match = NUMBERED_ADR_RE.fullmatch(path.name)
        if match is None or match.group(1) == "0000":
            continue
        by_number[match.group(1)].append(path.name)

    duplicates = {
        number: sorted(paths)
        for number, paths in by_number.items()
        if len(paths) > 1
    }
    assert duplicates == {}


def test_adr_index_requires_live_reservation_inventory_before_allocation() -> None:
    """Keep the concurrent-proposal reservation rule discoverable and fail-closed."""
    index = ADR_INDEX.read_text(encoding="utf-8")
    for requirement in (
        "active PR reservation",
        "Looking only at the highest number on protected main is insufficient",
        "earliest-created still-valid proposal",
        "later colliders the next free numbers in PR creation order",
        "The free-number search includes protected ADRs and all live reservations",
        "ordinary-forward repair",
        "Do not force-rewrite history",
        "branch-only proposal Accepted",
    ):
        assert requirement in index
