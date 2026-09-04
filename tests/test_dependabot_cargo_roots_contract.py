"""Contracts for Dependabot coverage of standalone Cargo lock roots."""

from __future__ import annotations

from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def _cargo_dependabot_blocks() -> list[str]:
    """Return every Cargo update block from the Dependabot configuration."""
    text = (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    return re.findall(
        r'(?ms)^  - package-ecosystem: "cargo".*?(?=^  - package-ecosystem:|\Z)',
        text,
    )


def _cargo_directories(block: str) -> set[str]:
    """Return single- or multi-directory Cargo roots from one update block."""
    single = re.findall(r'(?m)^    directory: "([^"]+)"$', block)
    directories = re.search(
        r'(?ms)^    directories:\n(?P<body>(?:      - "/[^"]*"\n?)+)',
        block,
    )
    multiple = (
        []
        if directories is None
        else re.findall(r'(?m)^      - "([^"]+)"$', directories.group("body"))
    )
    return set(single + multiple)


def _required_cargo_roots() -> set[str]:
    """Derive independently locked Cargo roots used by package and fuzz builds."""
    with (ROOT / "Cargo.toml").open("rb") as fh:
        workspace = tomllib.load(fh)["workspace"]
    with (ROOT / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)
    with (ROOT / "fuzz" / "Cargo.toml").open("rb") as fh:
        fuzz_manifest = tomllib.load(fh)

    maturin_manifest = Path(pyproject["tool"]["maturin"]["manifest-path"])
    assert (ROOT / maturin_manifest).is_file()
    maturin_root = maturin_manifest.parent
    excluded_roots = {Path(path) for path in workspace.get("exclude", [])}
    assert maturin_root in excluded_roots
    assert fuzz_manifest["package"]["metadata"]["cargo-fuzz"] is True
    assert "workspace" in fuzz_manifest

    standalone_roots = {maturin_root, Path("fuzz")}
    for root in standalone_roots:
        assert (ROOT / root / "Cargo.toml").is_file()
        assert (ROOT / root / "Cargo.lock").is_file()

    return {"/"} | {f"/{root.as_posix()}" for root in standalone_roots}


def test_standalone_cargo_lock_roots_are_dependabot_managed() -> None:
    """Require every independently locked Cargo root used by CI/package builds."""
    blocks = _cargo_dependabot_blocks()
    assert blocks
    configured = set().union(*(_cargo_directories(block) for block in blocks))
    assert _required_cargo_roots() <= configured


def test_cargo_dependency_updates_are_grouped_across_lock_roots() -> None:
    """Require one dependency-name-grouped Cargo lane across every lock root."""
    blocks = _cargo_dependabot_blocks()
    assert len(blocks) == 1
    block = blocks[0]
    assert _cargo_directories(block) == _required_cargo_roots()
    assert re.search(r"(?m)^    directories:$", block)
    assert re.search(
        r"(?ms)^    groups:\n"
        r"      cargo-lock-roots:\n"
        r"        group-by: dependency-name$",
        block,
    )


def test_cargo_security_updates_are_grouped_across_lock_roots() -> None:
    """Require repository-owned Cargo security updates to share one root lane."""
    blocks = _cargo_dependabot_blocks()
    assert len(blocks) == 1
    block = blocks[0]
    assert _cargo_directories(block) == _required_cargo_roots()
    assert re.search(
        r"(?ms)^      cargo-security-lock-roots:\n"
        r"        applies-to: security-updates\n"
        r"        patterns:\n"
        r'          - "\*"$',
        block,
    )
