"""Contracts for the repository's authoritative architecture documentation."""

from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).parents[1]


def _text(path: str) -> str:
    """Return one required repository document as UTF-8 text."""

    document = _ROOT / path
    assert document.is_file(), f"missing authoritative document: {path}"
    return document.read_text(encoding="utf-8")


def test_authoritative_documentation_set_exists_and_is_indexed() -> None:
    """Keep PRD, TRD, architecture, ADR, diagram, and traceability sources durable."""

    required = (
        "ARCHITECTURE.md",
        "docs/README.md",
        "docs/product_requirements.md",
        "docs/technical_requirements.md",
        "docs/prd_trd_summary.md",
        "docs/adr/README.md",
        "docs/adr/0000-template.md",
        "docs/architecture/diagrams.md",
        "docs/traceability_matrix.md",
        "docs/doctoring/conversation_architecture_baseline.md",
    )
    for path in required:
        assert (_ROOT / path).is_file(), path

    index = _text("docs/README.md")
    for path in (
        "../ARCHITECTURE.md",
        "product_requirements.md",
        "technical_requirements.md",
        "adr/README.md",
        "traceability_matrix.md",
        "architecture/diagrams.md",
    ):
        assert path in index


def test_stale_mvp_summary_is_only_a_compatibility_pointer() -> None:
    """The legacy summary cannot reassert the obsolete Python-first MVP boundary."""

    summary = _text("docs/prd_trd_summary.md")
    assert "Compatibility pointer" in summary
    assert "product_requirements.md" in summary
    assert "technical_requirements.md" in summary
    assert "early Python-first MLS2PLM MVP" in summary
    assert "NumPy is the default backend" not in summary


def test_architecture_preserves_reusable_core_and_rust_authority() -> None:
    """Hosted-product and numerical-ownership decisions stay explicit."""

    architecture = _text("ARCHITECTURE.md")
    assert "psychometrics-commons" in architecture
    assert "not the hosted assessment product" in architecture
    assert "Rust is numerical authority" in architecture
    assert "No hidden hosted persistence" in architecture
    assert "Humans and LLM judges are fallible raters" in architecture
    assert "No atomistic default" in architecture


def test_adr_index_contains_material_decisions_and_template() -> None:
    """Material decisions remain discoverable rather than scattered across plans."""

    index = _text("docs/adr/README.md")
    for adr in (
        "0001-reusable-core-product-boundary.md",
        "0002-rust-numerical-authority.md",
        "0003-versioned-contracts-provenance.md",
        "0004-generated-item-trust-boundary.md",
        "0005-fallible-raters-model-selection.md",
        "0006-multilevel-temporal-measurement.md",
        "0007-factor-rotation-selection.md",
        "0008-scientific-ci-release-evidence.md",
        "0000-template.md",
    ):
        assert adr in index
        assert (_ROOT / "docs" / "adr" / adr).is_file()


def test_diagram_bundle_covers_component_sequence_state_and_logical_erd() -> None:
    """The architecture has executable text diagrams for the required viewpoints."""

    diagrams = _text("docs/architecture/diagrams.md")
    assert diagrams.count("```mermaid") >= 7
    assert "sequenceDiagram" in diagrams
    assert "stateDiagram-v2" in diagrams
    assert "erDiagram" in diagrams
    assert "RubricSpecification" in diagrams
    assert "GeneratedItemCandidate" in diagrams
    assert "Governed item-bank state machine" in diagrams
    assert "logical artifact" in diagrams.lower()


def test_traceability_marks_unmerged_or_incomplete_work_without_release_claims() -> None:
    """The matrix must expose active/planned gaps instead of documenting them as done."""

    matrix = _text("docs/traceability_matrix.md")
    assert "Active, not released" in matrix
    assert "Planned/partial" in matrix
    assert "canonical RAG observation adapter" in matrix
    assert "Governed item-bank closed loop" in matrix
    assert "Multilevel and temporal release integration" in matrix


def test_prd_and_trd_preserve_scientific_nonclaims_and_evidence_policy() -> None:
    """Product docs must not turn fit, judge agreement, or correlation into validity."""

    prd = _text("docs/product_requirements.md")
    trd = _text("docs/technical_requirements.md")
    assert "LLM judge outputs remain calibratable rater observations" in prd
    assert "Correlation alone is insufficient" in prd
    assert "bias and MAE/RMSE" in trd
    assert "relation=unknown" in trd
    assert "No hosted database" in trd


def test_doctoring_records_primary_method_and_standards_baseline() -> None:
    """The consolidation remains grounded in primary methods and standards."""

    doctoring = _text("docs/doctoring/conversation_architecture_baseline.md")
    for reference in (
        "Standards for educational and psychological testing",
        "Artificial Intelligence Risk Management Framework",
        "Model selection of nested and non-nested item response models",
        "Evaluating bifactor models",
        "A two-tier full-information item factor analysis model",
        "JSON",
        "WCAG",
    ):
        assert reference in doctoring
