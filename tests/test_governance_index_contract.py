"""Require the living governance index and its doctoring note."""

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_INDEX = _ROOT / "docs" / "GOVERNANCE_INDEX.md"
_DOC = _ROOT / "docs" / "doctoring" / "governance_index.md"


def test_governance_index_has_required_sections() -> None:
    """Buyers must find ADR, threat model, test strategy, and multilevel links."""
    text = _INDEX.read_text(encoding="utf-8")
    required = [
        "# Governance index",
        "ADR index",
        "Threat model",
        "Test strategy",
        "Operability",
        "Traceability",
        "multilevel",
        "APA 7th",
        "ARCHITECTURE.md",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"missing sections: {missing}"


def test_governance_doctoring_cites_multilevel_literature() -> None:
    """Doctoring note must cite multilevel / LSIRM literature."""
    note = _DOC.read_text(encoding="utf-8")
    assert "Fox" in note and "Jeon" in note and "Kang" in note
    assert "https://doi.org/" in note
