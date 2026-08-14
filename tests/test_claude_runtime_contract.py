"""Prevent canonical agent guidance from drifting from shipped runtime policy."""

from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_claude_python_floor_matches_package_metadata() -> None:
    """Agent guidance must advertise the exact supported Python floor."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requires_python = project["project"]["requires-python"]
    guidance = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")

    assert f'`requires-python = "{requires_python}"`' in guidance
    assert "`>=3.10`" not in guidance


def test_claude_auto_backend_matches_fail_closed_runtime() -> None:
    """Canonical guidance must not advertise a silent NumPy production fallback."""
    guidance = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    normalized = " ".join(guidance.split())

    assert "`auto` fails closed when the compiled Rust core is unavailable" in normalized
    assert "otherwise the numerically-identical NumPy reference" not in normalized
