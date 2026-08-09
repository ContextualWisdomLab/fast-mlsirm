"""Contracts for the published-standard and research-watch registry."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WATCH = ROOT / "docs/standards_watch.md"


def _watch_text() -> str:
    """Return the standards-watch document as UTF-8 text."""
    return WATCH.read_text(encoding="utf-8")


def test_standards_watch_exists_and_separates_normative_from_watch_items() -> None:
    """Keep published sources distinct from drafts and revision projects."""
    assert WATCH.is_file()
    text = _watch_text()
    assert "## Governing published references" in text
    assert "## Normative-versus-watch policy" in text
    assert "## Active watch items" in text
    assert "watch item, not a normative requirement" in text
    assert "before every release" in text


def test_standards_watch_covers_required_governance_surfaces() -> None:
    """Retain the architecture, quality, AI-risk, testing, and accessibility set."""
    text = _watch_text()
    for reference in (
        "ISO/IEC/IEEE 29148:2018",
        "ISO/IEC/IEEE 42010:2022",
        "ISO/IEC 25010:2023",
        "ISO/IEC 42001:2023",
        "ISO/IEC 42005:2025",
        "ISO/IEC 23894:2023",
        "NIST AI RMF 1.0",
        "NIST AI 600-1",
        "Standards for Educational and Psychological Testing",
        "WCAG 2.2",
    ):
        assert reference in text


def test_standards_watch_disclaims_unsupported_conformance_claims() -> None:
    """Citing a source must not become a certification or validity claim."""
    text = _watch_text()
    assert "does not claim certification or conformance" in text
    assert "A citation does not establish implementation" in text
    assert "without independent evidence" in text
