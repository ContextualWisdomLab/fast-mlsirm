"""Require the living governance index and its doctoring note."""

import re
from pathlib import Path
from urllib.parse import urlsplit

_ROOT = Path(__file__).resolve().parents[1]
_INDEX = _ROOT / "docs" / "GOVERNANCE_INDEX.md"
_DOC = _ROOT / "docs" / "doctoring" / "governance_index.md"


def _has_doi_link(note: str) -> bool:
    for match in re.findall(r"https?://[^\s<>()]+", note):
        url = urlsplit(match)
        if url.scheme == "https" and url.netloc == "doi.org" and url.path.strip("/"):
            return True
    return False


def test_doi_link_check_rejects_lookalikes() -> None:
    assert not _has_doi_link("No DOI link here")
    assert not _has_doi_link("https://doi.org.evil.example/10.1000/example")
    assert not _has_doi_link("http://doi.org/10.1000/example")
    assert not _has_doi_link("https://evil.example/https://doi.org/10.1000/example")
    assert not _has_doi_link("https://doi.org/")
    assert _has_doi_link("https://doi.org/10.1000/example")


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
    assert _has_doi_link(note)
