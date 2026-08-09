"""Regression contract for the authoritative architecture documentation set."""

from __future__ import annotations

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_AUTHORITATIVE_DOCS = (
    "ARCHITECTURE.md",
    "docs/README.md",
    "docs/PRD.md",
    "docs/TRD.md",
    "docs/adr/README.md",
    "docs/adr/0001-reusable-measurement-core-boundary.md",
    "docs/adr/0002-rust-first-numerical-authority.md",
    "docs/adr/0003-governed-assessment-rubric-scoring-lifecycle.md",
    "docs/adr/0004-structural-model-selection-and-context.md",
    "docs/adr/0005-privacy-purpose-limitation-and-audit.md",
    "docs/UML.md",
    "docs/ERD.md",
    "docs/traceability.md",
)


def _read(relative_path: str) -> str:
    """Read one repository-relative UTF-8 document."""
    return (_REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_authoritative_architecture_documentation_exists_and_is_nonempty() -> None:
    """Every canonical architecture artifact must exist with substantive content."""
    for relative_path in _AUTHORITATIVE_DOCS:
        path = _REPO_ROOT / relative_path
        assert path.is_file(), relative_path
        assert len(path.read_text(encoding="utf-8").strip()) >= 200, relative_path


def test_compatibility_summary_points_to_authoritative_documents() -> None:
    """The historical PRD/TRD path must redirect readers to the current set."""
    summary = _read("docs/prd_trd_summary.md")
    for expected_target in (
        "../ARCHITECTURE.md",
        "PRD.md",
        "TRD.md",
        "adr/README.md",
        "UML.md",
        "ERD.md",
        "traceability.md",
    ):
        assert expected_target in summary
    assert "no longer authoritative" in summary


def test_architecture_pins_product_boundary_and_rust_authority() -> None:
    """Architecture must retain the highest-risk bounded-context decisions."""
    architecture = _read("ARCHITECTURE.md")
    assert "not the hosted Psychometrics Commons application" in architecture
    assert "Rust owns production numerical arithmetic" in architecture
    assert "LLM judges are fallible raters" in architecture


def test_adr_index_pins_privacy_and_governance_boundary() -> None:
    """The ADR index must keep the purpose-limited privacy decision discoverable."""
    adr_index = _read("docs/adr/README.md")
    assert "0005-privacy-purpose-limitation-and-audit.md" in adr_index
    privacy_adr = _read("docs/adr/0005-privacy-purpose-limitation-and-audit.md")
    assert "blanket PII masking" in privacy_adr
    assert "does not claim certification" not in privacy_adr  # keep wording factual
    assert "do not claim certification" in privacy_adr or "does not claim certification" in privacy_adr or "falsely claiming certification" in privacy_adr


def test_traceability_uses_explicit_status_taxonomy() -> None:
    """Traceability must distinguish integrated work from PRs and plans."""
    traceability = _read("docs/traceability.md")
    for status in (
        "implemented_on_main",
        "open_pr",
        "planned",
        "research_only",
        "out_of_scope",
    ):
        assert status in traceability
    assert "PR #566" in traceability
