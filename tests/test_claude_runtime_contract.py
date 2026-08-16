"""Prevent canonical agent guidance from drifting from shipped runtime policy."""

from __future__ import annotations

from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_CONTRACT_START = "<!-- BEGIN fast-mlsirm-runtime-contract -->"
_RUNTIME_CONTRACT_END = "<!-- END fast-mlsirm-runtime-contract -->"


def _runtime_contract(guidance: str) -> dict[str, str]:
    """Return the single machine-readable runtime contract from agent guidance."""
    assert guidance.count(_RUNTIME_CONTRACT_START) == 1
    assert guidance.count(_RUNTIME_CONTRACT_END) == 1
    payload = guidance.split(_RUNTIME_CONTRACT_START, 1)[1].split(
        _RUNTIME_CONTRACT_END, 1
    )[0]
    match = re.fullmatch(r"\s*```toml\s*(.*?)\s*```\s*", payload, re.DOTALL)
    assert match is not None
    contract = tomllib.loads(match.group(1))["runtime_contract"]
    assert all(type(value) is str for value in contract.values())
    return contract


def test_claude_machine_runtime_contract_matches_shipped_policy() -> None:
    """Canonical machine guidance must pin package and backend ownership policy."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    guidance = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")

    assert _runtime_contract(guidance) == {
        "requires_python": project["project"]["requires-python"],
        "auto_backend": "rust_required",
        "numpy_role": "reference_parity_only",
    }


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
