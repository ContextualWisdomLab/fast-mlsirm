"""Contracts for the repository's canonical architecture documentation set."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DOCUMENTS = (
    "ARCHITECTURE.md",
    "docs/PRD.md",
    "docs/TRD.md",
    "docs/adr/README.md",
    "docs/uml/README.md",
    "docs/uml/component.puml",
    "docs/uml/scoring-sequence.puml",
    "docs/uml/model-selection-sequence.puml",
    "docs/uml/item-lifecycle.puml",
    "docs/uml/deployment.puml",
    "docs/erd/domain-model.puml",
    "docs/traceability/requirements-matrix.md",
    "docs/traceability/research-basis.md",
)

ADR_STATUS_RE = re.compile(r"^Status: \*\*(Accepted|Proposed|Deprecated|Superseded)\*\*$", re.MULTILINE)


def _read(path: str) -> str:
    """Return repository UTF-8 text for a documentation contract path."""
    return (ROOT / path).read_text(encoding="utf-8")


def test_canonical_architecture_documentation_files_exist() -> None:
    """Keep requirements, decisions, diagrams, ERD, and traceability discoverable."""
    missing = [path for path in REQUIRED_DOCUMENTS if not (ROOT / path).is_file()]
    assert missing == []


def test_every_indexed_adr_exists_and_declares_supported_status() -> None:
    """Prevent the ADR index from pointing at missing or statusless decisions."""
    index = _read("docs/adr/README.md")
    linked = re.findall(r"\]\((\d{4}[^)]+\.md)\)", index)
    assert linked
    for relative_path in linked:
        adr_path = ROOT / "docs" / "adr" / relative_path
        assert adr_path.is_file(), relative_path
        assert ADR_STATUS_RE.search(adr_path.read_text(encoding="utf-8")), relative_path


def test_canonical_documents_state_the_hosted_product_boundary() -> None:
    """Do not let core-library documentation drift into hosted-runtime ownership."""
    architecture = _read("ARCHITECTURE.md")
    prd = _read("docs/PRD.md")
    trd = _read("docs/TRD.md")

    for text in (architecture, prd, trd):
        assert "psychometrics-commons" in text.lower()
        assert "independ" in text.lower()
    assert "never the reverse" in architecture
    assert "hosted HTTP/admin APIs" in trd


def test_legacy_prd_trd_summary_is_explicitly_deprecated() -> None:
    """Historical MVP notes must not compete with the canonical PRD and TRD."""
    summary = _read("docs/prd_trd_summary.md")
    assert "Deprecated as an authoritative requirements source" in summary
    assert "[Product Requirements Document](PRD.md)" in summary
    assert "[Technical Requirements Document](TRD.md)" in summary


def test_root_architecture_links_to_canonical_views() -> None:
    """Keep the root navigation graph connected to its diagram and decision sources."""
    architecture = _read("ARCHITECTURE.md")
    required_links = (
        "docs/uml/component.puml",
        "docs/uml/scoring-sequence.puml",
        "docs/uml/model-selection-sequence.puml",
        "docs/uml/item-bank-state.puml",
        "docs/uml/deployment.puml",
        "docs/erd/domain-model.puml",
        "docs/adr/README.md",
    )
    for target in required_links:
        assert target in architecture
        assert (ROOT / target).is_file(), target


def test_requirements_traceability_names_core_contract_sources() -> None:
    """Pin executable source anchors for reusable scoring and rubric contracts."""
    trace = _read("docs/traceability/requirements-matrix.md")
    assert "python/fast_mlsirm/scoring/contracts.py" in trace
    assert "python/fast_mlsirm/rubric/__init__.py" in trace
    assert "crates/mlsirm-core/" in trace
    assert "crates/fast-mlsirm-py/" in trace
