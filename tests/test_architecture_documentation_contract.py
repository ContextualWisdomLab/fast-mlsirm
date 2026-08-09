"""Regression contracts for the canonical product and architecture documentation."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DOCUMENTS = (
    "ARCHITECTURE.md",
    "docs/README.md",
    "docs/PRD.md",
    "docs/TRD.md",
    "docs/architecture/UML.md",
    "docs/architecture/ERD.md",
    "docs/adr/README.md",
    "docs/requirements_traceability.md",
    "docs/documentation_coverage.md",
)
ADR_FILES = tuple(
    f"docs/adr/ADR-{index:03d}-{slug}.md"
    for index, slug in (
        (1, "domain-boundary"),
        (2, "rust-first-numerics"),
        (3, "content-addressed-contracts"),
        (4, "governed-item-bank"),
        (5, "relation-safe-model-selection"),
        (6, "multilevel-temporal-first-class"),
        (7, "fallible-raters-and-llm-orchestration"),
        (8, "logical-persistence-boundary"),
    )
)


def _read(relative_path: str) -> str:
    """Return one repository documentation file as UTF-8 text."""
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_canonical_architecture_documents_exist_without_placeholders() -> None:
    """Keep the architecture spine present and free from unfinished placeholders."""
    for relative_path in (*CANONICAL_DOCUMENTS, *ADR_FILES):
        path = ROOT / relative_path
        assert path.is_file(), relative_path
        text = path.read_text(encoding="utf-8")
        assert text.strip(), relative_path
        assert "TODO" not in text
        assert "TBD" not in text


def test_prd_covers_all_governing_product_requirements() -> None:
    """Pin the cross-feature requirements derived from the product research program."""
    prd = _read("docs/PRD.md")
    required_ids = (
        "PRD-CONTRACT-001",
        "PRD-RUBRIC-001",
        "PRD-RUBRIC-002",
        "PRD-BANK-001",
        "PRD-SCORING-001",
        "PRD-SCORING-002",
        "PRD-RAG-001",
        "PRD-PSY-001",
        "PRD-MODEL-001",
        "PRD-MODEL-002",
        "PRD-ROT-001",
        "PRD-MULTI-001",
        "PRD-TIME-001",
        "PRD-RECOVERY-001",
        "PRD-REPORT-001",
        "PRD-RELEASE-001",
    )
    for requirement_id in required_ids:
        assert requirement_id in prd


def test_trd_preserves_numerical_scientific_and_security_boundaries() -> None:
    """Prevent future documentation from erasing critical implementation invariants."""
    trd = _read("docs/TRD.md")
    required_contracts = (
        "TRD-NUM-001",
        "TRD-NUM-003",
        "TRD-NUM-005",
        "TRD-PSY-002",
        "TRD-PSY-005",
        "TRD-PSY-006",
        "TRD-RUBRIC-002",
        "TRD-SCORE-003",
        "TRD-SEC-002",
        "TRD-QUAL-001",
        "NVIDIA_NIM_API_KEY",
        "COPILOT_GITHUB_TOKEN",
        "bias and RMSE",
    )
    for contract in required_contracts:
        assert contract in trd


def test_architecture_keeps_hosted_product_outside_the_core() -> None:
    """Keep standalone/MSA ownership explicit instead of drifting toward a monolith."""
    architecture = _read("ARCHITECTURE.md")
    assert "domain-neutral measurement and psychometric computation layer" in architecture
    assert "fast-mlsirm **does not own**" in architecture
    assert "Psychometrics Commons" in architecture
    assert "no mandatory server deployment topology" in architecture
    assert "Production mathematical/statistical/psychometric arithmetic is **Rust-first**" in architecture


def test_uml_suite_covers_structure_behavior_state_and_deployment() -> None:
    """Require more than one informal diagram for the canonical UML companion."""
    uml = _read("docs/architecture/UML.md")
    assert uml.count("```mermaid") >= 7
    for notation in (
        "flowchart",
        "classDiagram",
        "sequenceDiagram",
        "stateDiagram-v2",
    ):
        assert notation in uml
    assert "Model-selection activity view" in uml
    assert "Multilevel and temporal contract view" in uml


def test_erd_is_logical_and_does_not_claim_core_database_ownership() -> None:
    """Keep the ERD useful for interoperability without introducing ORM ownership."""
    erd = _read("docs/architecture/ERD.md")
    assert erd.count("erDiagram") >= 2
    assert "logical information model" in erd
    assert "not a physical database schema" in erd
    assert "user/account tables" in erd
    assert "context_membership" in erd
    assert "longitudinal_state_spec" in erd


def test_adr_index_and_traceability_cover_every_accepted_baseline_decision() -> None:
    """Keep accepted ADRs discoverable and linked to the traceability model."""
    adr_index = _read("docs/adr/README.md")
    traceability = _read("docs/requirements_traceability.md")
    for index, relative_path in enumerate(ADR_FILES, start=1):
        adr_id = f"ADR-{index:03d}"
        assert adr_id in adr_index
        assert (ROOT / relative_path).is_file()
        assert adr_id in traceability or index == 8
    assert "Implemented" in traceability
    assert "Partial" in traceability
    assert "Planned" in traceability


def test_documentation_audit_distinguishes_baseline_from_feature_completion() -> None:
    """Prevent the architecture baseline from overstating roadmap implementation state."""
    coverage = _read("docs/documentation_coverage.md")
    assert "not sufficient as a canonical architecture package" in coverage
    assert "does **not** claim that every roadmap feature is implemented" in coverage
    for priority in ("P0", "P1", "P2"):
        assert priority in coverage
    assert "protected-main functionality" in coverage
