"""Contracts for Dependabot coverage of standalone Cargo lock roots."""

from __future__ import annotations

from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def _cargo_dependabot_directories() -> set[str]:
    """Return Cargo directories explicitly managed by Dependabot."""
    text = (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    return set(
        re.findall(
            r'- package-ecosystem:\s*"cargo"\s*\n\s+directory:\s*"([^"]+)"',
            text,
        )
    )


def test_standalone_cargo_lock_roots_are_dependabot_managed() -> None:
    """Require every independently locked Cargo root used by CI/package builds."""
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

    required = {"/"} | {f"/{root.as_posix()}" for root in standalone_roots}
    assert required <= _cargo_dependabot_directories()
