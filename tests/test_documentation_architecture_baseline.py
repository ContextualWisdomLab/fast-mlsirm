"""Contracts for the repository's authoritative architecture documentation."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]


REQUIRED_DOCUMENTS = (
    "ARCHITECTURE.md",
    "docs/PRD.md",
    "docs/TRD.md",
    "docs/UML.md",
    "docs/ERD.md",
    "docs/documentation_coverage_matrix.md",
    "docs/adr/ADR-0001-product-boundaries-and-scientific-governance.md",
    "docs/prd_trd_summary.md",
)


def _text(relative_path: str) -> str:
    """Return UTF-8 repository documentation for one required path."""

    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_authoritative_architecture_document_set_exists() -> None:
    """PRD, TRD, architecture, UML, ERD, ADR and coverage must all be present."""

    for relative_path in REQUIRED_DOCUMENTS:
        path = ROOT / relative_path
        assert path.is_file(), relative_path
        assert path.read_text(encoding="utf-8").strip(), relative_path


def test_prd_trd_summary_is_an_index_not_the_stale_numpy_first_contract() -> None:
    """The legacy summary cannot reintroduce the obsolete early-MVP architecture."""

    summary = _text("docs/prd_trd_summary.md")

    assert "This file is an index" in summary
    assert "Rust-first" in summary
    assert "NumPy remains the default runtime backend" not in summary
    assert "Ordinal graded response models" not in summary
    assert "GUI dashboards" not in summary


def test_architecture_pins_repository_and_numerical_ownership_boundaries() -> None:
    """The baseline keeps hosted-product and numerical ownership unambiguous."""

    architecture = _text("ARCHITECTURE.md")

    assert "psychometrics-commons" in architecture
    assert "Rust-first" in architecture
    assert "does **not** own hosted participant/session/consent lifecycle" in architecture
    assert "Finite multi-start" in architecture
    assert "blanket masking" in architecture


def test_trd_requires_multilevel_temporal_and_recovery_evidence() -> None:
    """The technical baseline prevents atomistic and correlation-only shortcuts."""

    trd = _text("docs/TRD.md")

    for phrase in (
        "cross-classified",
        "multiple-membership",
        "continuous-time",
        "bias, MAE, RMSE",
        "formal distinguishability",
        "NVIDIA_NIM_API_KEY",
        "COPILOT_GITHUB_TOKEN",
    ):
        assert phrase in trd


def test_uml_and_erd_contain_machine_renderable_mermaid_contracts() -> None:
    """Architecture diagrams remain version-controlled rather than prose-only."""

    uml = _text("docs/UML.md")
    erd = _text("docs/ERD.md")

    assert "classDiagram" in uml
    assert "sequenceDiagram" in uml
    assert "flowchart" in uml
    assert "erDiagram" in erd
    assert "score_observation" in erd
    assert "calibration_run" in erd
    assert "release_bundle" in erd
    assert "score_observation ||--o{ contextual_membership" in erd
    assert "contextual_membership ||--o{ score_observation" not in erd
    assert "generated_item_version ||--o{ item_bank_entry" in erd
    assert "generated_item_version ||--o{ item_bank_version" not in erd
    assert "calibration_run ||--o{ item_bank_entry" in erd


def test_adr_records_current_standards_and_scientific_boundaries() -> None:
    """The cross-cutting ADR must remain tied to standards and research evidence."""

    adr = _text("docs/adr/ADR-0001-product-boundaries-and-scientific-governance.md")

    for phrase in (
        "ISO/IEC 25010:2023",
        "ISO/IEC 42001:2023",
        "NIST AI 100-1",
        "NIST AI 600-1",
        "Bifactor fit does not authorize score interpretation",
        "No universal rotation criterion",
        "PII is protected by architecture, not blanket masking",
    ):
        assert phrase in adr
