"""Prevent canonical guidance from drifting from shipped runtime policy."""

from __future__ import annotations

from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_CONTRACT_START = "<!-- BEGIN fast-mlsirm-runtime-contract -->"
_RUNTIME_CONTRACT_END = "<!-- END fast-mlsirm-runtime-contract -->"
_STALE_AUTO_NUMPY_FALLBACK = (
    "transparently falls back to the NumPy reference implementation"
)
_STALE_PARITY_AND_FALLBACK = "parity testing and fallback"
_STALE_NUMPY_DEFAULT = "NumPy reference backend as the default runtime path"


def _runtime_contract(guidance: str) -> dict[str, str]:
    """Return the single machine-readable runtime contract from agent guidance."""
    if guidance.count(_RUNTIME_CONTRACT_START) != 1:
        raise AssertionError("CLAUDE.md must contain exactly one runtime-contract start marker")
    if guidance.count(_RUNTIME_CONTRACT_END) != 1:
        raise AssertionError("CLAUDE.md must contain exactly one runtime-contract end marker")
    payload = guidance.split(_RUNTIME_CONTRACT_START, 1)[1].split(
        _RUNTIME_CONTRACT_END, 1
    )[0]
    match = re.fullmatch(r"\s*```toml\s*(.*?)\s*```\s*", payload, re.DOTALL)
    if match is None:
        raise AssertionError("runtime-contract markers must wrap exactly one toml fence")
    loaded = tomllib.loads(match.group(1))
    if "runtime_contract" not in loaded:
        raise AssertionError("toml fence must define a [runtime_contract] table")
    contract = loaded["runtime_contract"]
    if not all(type(value) is str for value in contract.values()):
        raise AssertionError("runtime_contract values must be strings")
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


def test_readme_auto_backend_matches_fail_closed_runtime() -> None:
    """Buyer-facing install guidance must match shipped auto-backend ownership."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())

    assert _STALE_AUTO_NUMPY_FALLBACK not in normalized
    assert _STALE_PARITY_AND_FALLBACK not in normalized
    assert "fails closed when that extension is unavailable" in normalized
    assert "Automatic resolution never silently selects NumPy" in normalized


def test_commercial_readiness_names_rust_as_default_auto_path() -> None:
    """Sales-review copy must not name NumPy as the default runtime path."""
    readiness = (ROOT / "docs" / "commercial_readiness.md").read_text(encoding="utf-8")

    assert _STALE_NUMPY_DEFAULT not in readiness
    assert "Rust/PyO3 backend as the default `auto` runtime path" in readiness
    assert "Explicit NumPy reference backend for parity testing only" in readiness


def test_prd_current_capabilities_keep_numpy_as_explicit_parity() -> None:
    """Protected-main PRD capabilities must not advertise auto-to-NumPy fallback."""
    prd = (ROOT / "docs" / "PRD.md").read_text(encoding="utf-8")

    assert "NumPy reference/fallback paths and parity tests" not in prd
    assert "explicit NumPy reference/parity path" in prd
    assert "`auto` fails closed without the compiled Rust core" in prd


def test_architecture_summaries_do_not_call_auto_a_transparent_fallback() -> None:
    """Architecture summaries must not describe auto resolution as transparent fallback."""
    trd = (ROOT / "docs" / "TRD.md").read_text(encoding="utf-8")
    summary = (ROOT / "docs" / "prd_trd_summary.md").read_text(encoding="utf-8")
    adr = (ROOT / "docs" / "adr" / "0002-rust-first-numerical-ownership.md").read_text(
        encoding="utf-8"
    )

    assert "transparent reference/fallback paths" not in trd
    assert "governed reference/parity paths" in trd
    assert "transparent governed reference/fallback paths" not in summary
    assert "fails closed otherwise; it never silently selects NumPy" in adr
