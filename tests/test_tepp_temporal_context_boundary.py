"""Architecture fitness tests for the TEPP temporal/event Anti-Corruption Layer."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADR_PATH = "docs/adr/0028-tepp-temporal-event-composition-boundary.md"
ADR_INDEX_PATH = "docs/adr/README.md"
ARCHITECTURE_PATH = "ARCHITECTURE.md"
PRD_PATH = "docs/PRD.md"
TRD_PATH = "docs/TRD.md"
REQUIREMENTS_MATRIX_PATH = "docs/traceability/requirements-matrix.md"
AGENTS_PATH = "AGENTS.md"
CLAUDE_PATH = "CLAUDE.md"
RUST_MANIFEST_PATHS = (
    "Cargo.toml",
    "crates/mlsirm-core/Cargo.toml",
    "crates/fast-mlsirm-py/Cargo.toml",
    "fuzz/Cargo.toml",
)


def _read(path: str) -> str:
    """Return UTF-8 repository text for one governed boundary artifact."""
    return (ROOT / path).read_text(encoding="utf-8")


def test_temporal_event_semantics_have_one_foreign_owner() -> None:
    """Keep event composition in TEPP while retaining fast numerical kernels."""
    adr = _read(ADR_PATH)
    index = _read(ADR_INDEX_PATH)

    assert "Status: Accepted" in adr
    assert "**TEPP owns temporal/event composition and semantics.**" in adr
    assert "**fast-mlsirm owns reusable time-indexed psychometric numerical kernels.**" in adr
    assert (
        "Cross-context integration uses an Anti-Corruption Layer. A TEPP-originated "
        "occasion/temporal design is admitted only through an explicit versioned, immutable contract."
        in adr
    )
    assert (
        "An adapter is permitted only when it implements that contract and exposes a compatibility "
        "identity that binds the contract family, contract version, and immutable provenance/content "
        "identity; an adapter is not an unversioned escape hatch."
        in adr
    )
    assert (
        "No cross-service SQL, direct TEPP database access, or hidden TEPP runtime dependency is permitted."
        in adr
    )
    assert (
        "| [0028](0028-tepp-temporal-event-composition-boundary.md) | Accepted | TEPP owns "
        "temporal/event composition and semantics; fast-mlsirm retains reusable time-indexed "
        "psychometric numerical kernels behind an explicit Anti-Corruption Layer. |"
        in index
    )


def test_boundary_qualifies_prior_longitudinal_model_proposals() -> None:
    """The owner ADR and index must qualify earlier numerical proposals without erasing them."""
    adr = _read(ADR_PATH)
    index = _read(ADR_INDEX_PATH)

    for historical_adr in ("ADR-0007", "ADR-0019", "ADR-0020"):
        assert historical_adr in adr
    assert "remain useful numerical/model-design records" in adr
    assert "interpreted through this ownership boundary" in adr
    assert (
        "| [0007](0007-multilevel-multiple-membership-temporal.md) | Proposed | Multilevel, "
        "cross-classified, multiple-membership and time-indexed psychometric structure are first-class; "
        "Rust estimators require recovery evidence, while TEPP temporal/event ownership is governed by ADR-0028. |"
        in index
    )
    assert (
        "| [0019](0019-rust-longitudinal-state-engine.md) | Proposed | Rust numerical state-layer proposal; "
        "event ontology, temporal validity and composition remain TEPP-owned under ADR-0028. |"
        in index
    )
    assert (
        "| [0020](0020-joint-hierarchical-ctar-rasch.md) | Proposed | Joint MAP hierarchical continuous-time "
        "AR(1) Rasch numerical kernel; temporal/event composition remains TEPP-owned under ADR-0028. |"
        in index
    )


def test_adr_0007_delegates_temporal_event_semantics_to_tepp() -> None:
    """Prevent the historical umbrella ADR from becoming a second temporal owner."""
    adr = _read("docs/adr/0007-multilevel-multiple-membership-temporal.md")

    assert (
        "ADR-0028 governs bounded-context ownership for every temporal statement in this Proposed record."
        in adr
    )
    assert "TEPP owns temporal/event composition and semantics" in adr
    assert (
        "fast-mlsirm owns only reusable psychometric numerical kernels over explicit supplied "
        "occasion/time carriers"
        in adr
    )
    assert (
        "Event ontology, temporal validity, event ordering, changing-membership history, and "
        "longitudinal leakage policy remain outside this ADR"
        in adr
    )


def test_root_architecture_preserves_tepp_temporal_semantics_boundary() -> None:
    """Keep the authoritative population view from absorbing TEPP event semantics."""
    architecture = _read(ARCHITECTURE_PATH)

    assert (
        "Time-indexed psychometric kernels consume explicit person, occasion, and elapsed-time carriers; "
        "they do not define event ontology, temporal validity, event ordering, changing-membership history, "
        "or longitudinal leakage policy."
        in architecture
    )
    assert (
        "TEPP owns those temporal/event semantics and supplies them only through the versioned, immutable "
        "Anti-Corruption Layer governed by ADR-0028."
        in architecture
    )
    assert (
        "A measurement-occasion facet used by a psychometric model is not a TEPP temporal event model."
        in architecture
    )


def test_prd_and_trd_preserve_tepp_ownership_boundary() -> None:
    """Keep requirements and technical design aligned with the accepted temporal owner decision."""
    prd = _read(PRD_PATH)
    trd = _read(TRD_PATH)

    assert (
        "**PRD-FR-064** TEPP owns temporal/event semantics, event ordering, temporal validity, "
        "changing-membership history, and longitudinal leakage policy; fast-mlsirm owns reusable "
        "time-indexed psychometric numerical kernels over explicit supplied carriers."
        in prd
    )
    assert (
        "**PRD-FR-065** TEPP-originated temporal designs shall enter fast-mlsirm only through the "
        "versioned, immutable Anti-Corruption Layer governed by ADR-0028; direct TEPP database access, "
        "cross-service SQL, and hidden TEPP runtime dependencies are prohibited."
        in prd
    )
    assert (
        "**TRD-MLT-007** Measurement-occasion carriers and elapsed-time inputs are numerical model inputs, "
        "not a local event ontology. TEPP retains event semantics, temporal validity, event ordering, "
        "changing-membership history, and longitudinal leakage policy."
        in trd
    )
    assert (
        "**TRD-MLT-008** A TEPP-originated temporal design shall be admitted only through a versioned, "
        "immutable ACL contract with compatibility identity and provenance; adapters shall not bypass "
        "that contract or introduce cross-service SQL, direct TEPP database access, or a hidden TEPP runtime dependency."
        in trd
    )


def test_requirements_matrix_traces_temporal_owner_boundary() -> None:
    """Map the accepted temporal owner decision to its exact product and technical requirements."""
    matrix = _read(REQUIREMENTS_MATRIX_PATH)

    assert (
        "| Temporal event ownership boundary | PRD-FR-064/065, TRD-MLT-007/008 | ADR-0028 | "
        "`docs/adr/0028-tepp-temporal-event-composition-boundary.md`, `tests/test_tepp_temporal_context_boundary.py`; "
        "existing Rust CT-AR Rasch kernel remains fast-mlsirm-owned while TEPP owns event semantics/composition | "
        "Accepted ownership boundary / active PR |"
        in matrix
    )


def test_agent_guidance_preserves_temporal_and_context_fabric_boundaries() -> None:
    """Prevent coding agents from rebuilding TEPP or unreleased Context Fabric authority locally."""
    agents = _read(AGENTS_PATH)
    claude = _read(CLAUDE_PATH)

    for document in (agents.replace("\n", " "), claude.replace("\n", " ")):
        assert "TEPP owns temporal/event composition and semantics" in document
        assert "fast-mlsirm owns reusable time-indexed psychometric numerical kernels" in document
        assert "cross-service SQL" in document
        assert "immutable released" in document
        assert "context-graph-contracts" in document
        assert "enterprise-architecture-core" in document
        assert "Estimator values, latent scores, DIF/fit diagnostics" in document


def test_existing_ctar_rasch_remains_a_rust_numerical_kernel() -> None:
    """Boundary repair must not erase or provider-couple the protected-main CT-AR estimand."""
    rust = _read("crates/mlsirm-core/src/longitudinal_irt.rs")
    assert 'const ESTIMAND_SCOPE: &str = "joint_map_hierarchical_ctar_rasch"' in rust
    assert 'const TRANSITION_KIND: &str = "continuous_time_ar1_ou"' in rust
    assert "pub fn ctar_phi" in rust
    assert "TEPP" not in rust
    assert "contextual_orchestrator" not in rust


def test_rust_manifests_do_not_declare_foreign_temporal_or_llm_integration() -> None:
    """Keep TEPP and contextual-orchestrator out of Rust dependency/build declarations."""
    manifests = "\n".join(_read(path) for path in RUST_MANIFEST_PATHS).lower()

    assert "tepp" not in manifests
    assert "contextual-orchestrator" not in manifests
    assert "contextual_orchestrator" not in manifests


def test_ea_projection_requires_released_context_graph_contract() -> None:
    """Architecture facts must fail closed rather than couple to an unreleased sibling head."""
    adr = _read(ADR_PATH)
    assert (
        "Architecture/package/backend/toolchain/consumer-lifecycle facts may be projected to "
        "`ContextualWisdomLab/enterprise-architecture-core` only through an immutable released "
        "`ContextualWisdomLab/context-graph-contracts` versioned Context Assertion / CloudEvent / "
        "conformance contract with provenance."
        in adr
    )
    assert (
        "Estimator values, latent scores, DIF/fit diagnostics, and scientific-validity evidence are not "
        "authoritative EA facts and must not be duplicated into the architecture decision plane."
        in adr
    )
    assert (
        "Therefore production EA projection fails closed rather than pinning an unreleased sibling PR head."
        in adr
    )
