"""Contracts for Dependabot coverage of independently locked Cargo roots."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
DEPENDABOT = ROOT / ".github" / "dependabot.yml"
STANDALONE_CARGO_ROOTS = {
    ROOT / "Cargo.lock": "/",
    ROOT / "fuzz" / "Cargo.lock": "/fuzz",
    ROOT / "crates" / "fast-mlsirm-py" / "Cargo.lock": "/crates/fast-mlsirm-py",
}


def _cargo_update_directories(config: str) -> set[str]:
    """Return directories configured as Cargo Dependabot update roots."""
    directories: set[str] = set()
    block_pattern = re.compile(
        r'^  - package-ecosystem: "cargo"\n(?P<body>(?:    .*\n)*)',
        re.MULTILINE,
    )
    directory_pattern = re.compile(r'^    directory: "([^"]+)"$', re.MULTILINE)
    for match in block_pattern.finditer(config):
        directory = directory_pattern.search(match.group("body"))
        assert directory is not None, "each Cargo Dependabot entry needs a directory"
        directories.add(directory.group(1))
    return directories


def test_dependabot_covers_every_repository_owned_cargo_lock_root() -> None:
    """Every independently resolved Cargo lock must receive dependency updates."""
    config = DEPENDABOT.read_text(encoding="utf-8")
    cargo_directories = _cargo_update_directories(config)

    for lockfile, directory in STANDALONE_CARGO_ROOTS.items():
        assert lockfile.is_file(), f"expected repository-owned lockfile: {lockfile}"
        assert directory in cargo_directories, (
            f"{lockfile.relative_to(ROOT)} is independently locked but Dependabot "
            f"does not monitor Cargo directory {directory!r}"
        )
