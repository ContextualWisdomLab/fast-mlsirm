"""Supply-chain metadata contracts for Dependabot update policy."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_dependabot_ecosystem_has_explicit_cooldown() -> None:
    text = (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    update_blocks = re.findall(
        r"(?ms)^  - package-ecosystem:.*?(?=^  - package-ecosystem:|\Z)", text
    )

    assert update_blocks
    for block in update_blocks:
        assert re.search(r"(?m)^    cooldown:\n      default-days: 7$", block)
