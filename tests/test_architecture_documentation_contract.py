"""Contracts for the repository's canonical architecture documentation set."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DOCUMENTS = (
    "ARCHITECTURE.md",
    "docs/README.md",
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
    "docs/security/threat-model.md",
    "docs/documentation_coverage.md",
)

ADR_STATUS_RE = re.compile(
    r"^Status: \*\*(Accepted|Proposed|Deprecated|Superseded)\*\*$", re.MULTILINE
)


def _read(path: str) -> str:
    """Return repository UTF-8 text for a documentation contract path."""
    return (ROOT / path).read_text(encoding="utf-8")


def test_canonical_architecture_documentation_files_exist() -> None:
    """Keep requirements, decisions, diagrams, ERD, security, and traceability discoverable."""
    missing = [path for path in REQUIRED_DOCUMENTS if not (ROOT / path).is_file()]
    assert missing == []


def test_every_indexed_adr_exists_and_declares_supported_status() -> None:
    """Prevent the ADR index from pointing at missing or statusless decisions."""
    index = _read("docs/adr/README.md")
    linked = re.findall(r"\]\((\d{4}[^)]+\.md)\)", index)
    assert linked
    assert "0011-canonical-pyo3-public-export-registry.md" in linked
    assert "0012-purpose-limited-sensitive-data.md" in linked
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


def test_documentation_index_and_completeness_matrix_cover_security_and_gaps() -> None:
    """Canonical navigation must expose threat, completeness, and current-vs-planned state."""
    index = _read("docs/README.md")
    coverage = _read("docs/documentation_coverage.md")
    for target in (
        "../ARCHITECTURE.md",
        "PRD.md",
        "TRD.md",
        "adr/README.md",
        "security/threat-model.md",
        "documentation_coverage.md",
        "traceability/requirements-matrix.md",
    ):
        assert target in index
    for state in ("IMPLEMENTED", "ACTIVE PR", "PLANNED", "DOWNSTREAM"):
        assert state in coverage
    assert "P0 documentation gaps" in coverage
    assert "Canonical PyO3/public-export registry" in coverage


def test_requirements_traceability_names_core_contract_sources_and_interpretation_rules() -> None:
    """Pin executable sources and conversation-wide scientific interpretation boundaries."""
    trace = _read("docs/traceability/requirements-matrix.md")
    assert "python/fast_mlsirm/scoring/contracts.py" in trace
    assert "python/fast_mlsirm/rubric/__init__.py" in trace
    assert "crates/mlsirm-core/" in trace
    assert "crates/fast-mlsirm-py/" in trace
    assert "LLM and human judges are fallible raters" in trace
    assert "Correlation is not parameter recovery or absolute agreement" in trace
    assert "Latent space follows substantive diagnosis" in trace
    assert "Psychometric discrimination is not business/safety criticality" in trace
    assert "Reference-free is not truth-free" in trace


def test_reusable_threat_model_covers_core_trust_and_misuse_boundaries() -> None:
    """Keep security, privacy, resource, and scientific-integrity threats explicit."""
    threat = _read("docs/security/threat-model.md")
    for concept in (
        "Untrusted JSON/member ambiguity",
        "Provider replay/provenance substitution",
        "PyO3/native shape/type confusion",
        "Numeric overflow/non-finite output",
        "CPU oversubscription/resource exhaustion",
        "GPU evidence spoofing",
        "Scientific model misuse",
        "Credential cross-contamination",
        "Benchmark contamination / double dipping",
        "Blanket masking destroys scientific design",
        "Self-modifying CI / source laundering",
        "Scientific-integrity recovery failure",
    ):
        assert concept in threat
    assert "hosted-product threats" in threat
    assert "does not claim certification" in threat


def test_pyo3_and_sensitive_data_adrs_preserve_future_integration_boundaries() -> None:
    """Prevent feature PRs from recreating native-export and privacy architecture drift."""
    pyo3 = _read("docs/adr/0011-canonical-pyo3-public-export-registry.md")
    privacy = _read("docs/adr/0012-purpose-limited-sensitive-data.md")
    assert "one canonical PyO3/public-export registry" in pyo3
    assert "runtime source rewriting" in pyo3
    assert "Proposed" in pyo3
    assert "does **not** use blanket PII masking" in privacy
    assert "purpose limitation" in privacy
    assert "Psychometrics Commons" in privacy


def test_research_basis_and_llm_credential_adr_keep_primary_boundaries() -> None:
    """Architecture evidence must preserve primary-source and model-credential authority."""
    research = _read("docs/traceability/research-basis.md")
    llm = _read("docs/adr/0010-llm-orchestration-and-credentials.md")
    assert "APA" in research
    assert "NVIDIA" in llm
    assert "COPILOT_GITHUB_TOKEN" in llm


def test_documentation_contract_does_not_promote_proposed_work_to_main() -> None:
    """Active multilevel/rotation/item-bank work must remain visibly non-released."""
    trace = _read("docs/traceability/requirements-matrix.md")
    coverage = _read("docs/documentation_coverage.md")
    assert "Proposed/partial / active PR" in trace
    assert "Proposed / active work" in trace
    assert "ACTIVE PR" in coverage
    assert "PLANNED/partial" in coverage
