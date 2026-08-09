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
    "docs/architecture/capability_maturity.md",
    "docs/documentation_coverage_matrix.md",
    "docs/adr/README.md",
    "docs/adr/0000-template.md",
    "docs/adr/ADR-0001-product-boundaries-and-scientific-governance.md",
    "docs/adr/ADR-0002-rust-numerical-source-of-truth.md",
    "docs/adr/ADR-0003-canonical-contracts-and-provenance.md",
    "docs/adr/ADR-0004-relation-safe-model-selection.md",
    "docs/adr/ADR-0005-rater-aware-ai-evaluation.md",
    "docs/adr/ADR-0006-multilevel-membership-and-time.md",
    "docs/adr/ADR-0007-adaptive-rotation-selection.md",
    "docs/adr/ADR-0008-statistical-evidence-and-release-gates.md",
    "docs/adr/ADR-0009-governed-rubric-item-bank-lifecycle.md",
    "docs/prd_trd_summary.md",
)


def _text(relative_path: str) -> str:
    """Return UTF-8 repository documentation for one required path."""

    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_authoritative_architecture_document_set_exists() -> None:
    """PRD, TRD, architecture, diagrams, ADRs and coverage must all be present."""

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
    """The baseline keeps hosted-product, maturity and numerical ownership unambiguous."""

    architecture = _text("ARCHITECTURE.md")

    assert "psychometrics-commons" in architecture
    assert "Rust-first" in architecture
    assert "does **not** own hosted participant/session/consent lifecycle" in architecture
    assert "Finite multi-start" in architecture
    assert "blanket masking" in architecture
    assert "capability_maturity.md" in architecture
    assert "does not promote it to shipped status" in architecture


def test_capability_maturity_prevents_design_docs_from_claiming_shipped_features() -> None:
    """Target architecture must stay distinct from protected-main implementation maturity."""

    maturity = _text("docs/architecture/capability_maturity.md")

    for status in (
        "IMPLEMENTED",
        "PARTIAL",
        "DESIGN-REQUIREMENT",
        "OWNED-BY-OTHER-REPO",
        "NOT-APPLICABLE",
    ):
        assert status in maturity
    assert "Continuous-time psychometric transition model | DESIGN-REQUIREMENT" in maturity
    assert "Physical application database / ORM | NOT-APPLICABLE" in maturity
    assert "Psychometrics Commons" in maturity
    assert "Documentation, open pull requests and plans cannot promote" in maturity


def test_trd_requires_multilevel_temporal_recovery_and_current_governance_evidence() -> None:
    """The technical baseline prevents atomistic, correlation-only and stale-standard shortcuts."""

    trd = _text("docs/TRD.md")

    for phrase in (
        "cross-classified",
        "multiple-membership",
        "continuous-time",
        "bias, MAE, RMSE",
        "formal distinguishability",
        "NVIDIA_NIM_API_KEY",
        "COPILOT_GITHUB_TOKEN",
        "ISO/IEC 23894:2023",
        "ISO/IEC 42005:2025",
        "ISO/IEC 40500:2025",
        "AI RMF 1.0 is being revised",
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
    assert "does not own an application database or ORM" in erd
    assert "score_observation ||--o{ contextual_membership" in erd
    assert "contextual_membership ||--o{ score_observation" not in erd
    assert "generated_item_version ||--o{ item_bank_entry" in erd
    assert "generated_item_version ||--o{ item_bank_version" not in erd
    assert "calibration_run ||--o{ item_bank_entry" in erd


def test_adr_index_and_template_require_durable_decision_mechanics() -> None:
    """The ADR set must be reviewable, decomposed and supersedable rather than one monolith."""

    index = _text("docs/adr/README.md")
    template = _text("docs/adr/0000-template.md")

    for number in range(1, 10):
        assert f"ADR-{number:04d}" in index
    for phrase in (
        "implementation status separately from decision status",
        "fail-closed/degraded/recovery",
        "migration, compatibility window, rollback",
        "objective supersession/reversal conditions",
    ):
        assert phrase in index
    for heading in (
        "## Context and decision drivers",
        "## Contract and interface consequences",
        "## Numerical and scientific consequences",
        "## Failure, degraded and recovery behavior",
        "## Security, privacy and compliance consequences",
        "## Migration, compatibility and rollback",
        "## Verification and acceptance evidence",
        "## Supersession / reversal criteria",
    ):
        assert heading in template


def test_adr_records_current_standards_and_scientific_boundaries() -> None:
    """The cross-cutting ADR must remain tied to current standards and research evidence."""

    adr = _text("docs/adr/ADR-0001-product-boundaries-and-scientific-governance.md")

    for phrase in (
        "ISO/IEC 25010:2023",
        "ISO/IEC 42001:2023",
        "ISO/IEC 23894:2023",
        "ISO/IEC 42005:2025",
        "ISO/IEC 40500:2025",
        "NIST AI 100-1",
        "NIST AI 600-1",
        "Bifactor fit does not authorize score interpretation",
        "No universal rotation criterion",
        "PII is protected by architecture, not blanket masking",
    ):
        assert phrase in adr
