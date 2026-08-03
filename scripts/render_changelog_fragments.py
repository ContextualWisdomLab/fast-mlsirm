#!/usr/bin/env python3
"""Render authoritative unreleased changelog fragments deterministically."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
FRAGMENT_DIR = ROOT / "docs" / "changelog.d"
_ALLOWED_SECTIONS = (
    "Added",
    "Changed",
    "Deprecated",
    "Removed",
    "Fixed",
    "Security",
)


def fragment_paths(directory: Path = FRAGMENT_DIR) -> tuple[Path, ...]:
    """Return every authoritative Markdown fragment in stable path order."""
    return tuple(
        path
        for path in sorted(directory.glob("*.md"))
        if path.name.casefold() != "readme.md"
    )


def parse_fragment(path: Path) -> tuple[str, dict[str, tuple[str, ...]]]:
    """Parse one fragment into a title and validated release-note sections."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError(f"{path}: first line must be a level-one title")
    title = lines[0][2:].strip()
    if not title:
        raise ValueError(f"{path}: title must not be empty")

    sections: dict[str, list[str]] = defaultdict(list)
    active: str | None = None
    for line in lines[1:]:
        if line.startswith("## "):
            active = line[3:].strip()
            if active not in _ALLOWED_SECTIONS:
                raise ValueError(f"{path}: unsupported section {active!r}")
            continue
        if line.startswith("#"):
            raise ValueError(f"{path}: unsupported heading depth")
        if line.strip():
            if active is None:
                raise ValueError(f"{path}: content must follow a release section")
            sections[active].append(line.rstrip())

    if not sections:
        raise ValueError(f"{path}: at least one release section is required")
    return title, {name: tuple(values) for name, values in sections.items()}


def render_unreleased(paths: Iterable[Path] | None = None) -> str:
    """Render all fragments as one deterministic ``Unreleased`` Markdown block."""
    selected = fragment_paths() if paths is None else tuple(paths)
    if not selected:
        raise ValueError("at least one changelog fragment is required")

    grouped: dict[str, list[tuple[str, tuple[str, ...]]]] = defaultdict(list)
    for path in selected:
        title, sections = parse_fragment(path)
        for section, lines in sections.items():
            grouped[section].append((title, lines))

    output = ["## Unreleased", ""]
    for section in _ALLOWED_SECTIONS:
        entries = grouped.get(section)
        if not entries:
            continue
        output.extend((f"### {section}", ""))
        for title, lines in entries:
            output.extend((f"#### {title}", "", *lines, ""))
    return "\n".join(output).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    """Render fragments to stdout or an explicitly selected output file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write rendered Markdown here")
    args = parser.parse_args(argv)
    rendered = render_unreleased()
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
