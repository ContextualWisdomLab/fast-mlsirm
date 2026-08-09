"""Contracts for the repository-wide architecture documentation baseline."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    """Return UTF-8 repository documentation for one required relative path."""
    path = ROOT / relative_path
    assert path.is_file(), f"required architecture document is missing: {relative_path}"
    return path.read_text(encoding="utf-8")


def test_authoritative_architecture_document_set_exists() -> None:
    """PRD/TRD/ADR/UML/ERD/traceability authorities remain repository-local."""
    required = (
        "ARCHITECTURE.md",
        "docs/PRD.md",
        "docs/TRD.md",
        "docs/adr/README.md",
        "docs/architecture/README.md",
        "docs/architecture/uml.md",
        "docs/architecture/logical-data-model.md",
        "docs/requirements-traceability.md",
    )

    for relative_path in required:
        assert (ROOT / relative_path).is_file(), relative_path


def test_architecture_preserves_hosted_product_boundary() -> None:
    """The reusable core never reclaims Psychometrics Commons product ownership."""
    architecture = _read("ARCHITECTURE.md")
    agents = _read("AGENTS.md")
    claude = _read("CLAUDE.md")

    for document in (architecture, agents, claude):
        assert "psychometrics-commons" in document
        assert "services/assessment_runtime" in document

    assert "must not\nbe recreated" in architecture or "must not be recreated" in architecture
    assert "must not be recreated" in agents
    assert "must not be recreated" in claude


def test_prd_and_trd_pin_scientific_product_invariants() -> None:
    """Core research requirements cannot disappear from the component baseline."""
    prd = _read("docs/PRD.md").lower()
    trd = _read("docs/TRD.md").lower()

    for phrase in (
        "true-parameter",
        "relation-safe model selection",
        "rater-aware ai evaluation",
        "context and time",
        "governed item lifecycle",
    ):
        assert phrase in prd

    for phrase in (
        "rust numerical requirements",
        "model relation before selection",
        "bifactor",
        "rotation",
        "multilevel/multiple membership/time",
        "automated scoring and llm-as-a-judge",
        "rubric and governed item bank",
    ):
        assert phrase in trd


def test_adr_index_covers_material_conversation_decisions() -> None:
    """Material scientific and architecture decisions remain independently reviewable."""
    index = _read("docs/adr/README.md")
    required_adrs = (
        "0001-reusable-core-hosted-product-boundary.md",
        "0002-rust-numerical-source-of-truth.md",
        "0003-canonical-contracts-and-provenance.md",
        "0004-relation-safe-model-selection.md",
        "0005-rater-aware-ai-evaluation.md",
        "0006-multilevel-membership-and-time.md",
        "0007-adaptive-rotation-selection.md",
        "0008-statistical-evidence-and-release-gates.md",
        "0009-governed-rubric-item-bank-lifecycle.md",
        "0010-canonical-pyo3-export-registry.md",
    )

    for filename in required_adrs:
        assert filename in index
        assert (ROOT / "docs" / "adr" / filename).is_file()


def test_diagrams_cover_component_sequence_state_and_logical_erd() -> None:
    """Reviewable text diagrams cover system, workflow, lifecycle and relationships."""
    uml = _read("docs/architecture/uml.md")
    data_model = _read("docs/architecture/logical-data-model.md")

    assert "flowchart" in uml
    assert "sequenceDiagram" in uml
    assert "stateDiagram-v2" in uml
    assert "erDiagram" in data_model
    assert "fast-mlsirm" in uml
    assert "psychometrics-commons" in uml
    assert "This is **not** a hosted-product database schema" in data_model


def test_traceability_distinguishes_implemented_active_planned_and_downstream() -> None:
    """Conversation coverage cannot make unmerged or downstream work look shipped."""
    traceability = _read("docs/requirements-traceability.md")

    for state in ("Implemented", "Active", "Planned", "Downstream"):
        assert state.lower() in traceability.lower()

    for requirement in (
        "True-parameter recovery",
        "Relation-safe model selection",
        "Adaptive factor rotation",
        "Governed item-bank lifecycle",
        "Multilevel / cross-classified / multiple membership",
        "NVIDIA NIM/OpenCode",
        "PRD/TRD/ADR/Architecture/UML/ERD completeness",
    ):
        assert requirement in traceability
