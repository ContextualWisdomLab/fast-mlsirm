"""Architecture fitness tests for the TEPP temporal/event Anti-Corruption Layer."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADR_PATH = "docs/adr/0028-tepp-temporal-event-composition-boundary.md"
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
    index = _read("docs/adr/README.md")

    assert "Status: Accepted" in adr
    assert "**TEPP owns temporal/event composition and semantics.**" in adr
    assert "**fast-mlsirm owns reusable time-indexed psychometric numerical kernels.**" in adr
    assert "event ontology and graph construction" in adr
    assert (
        "A TEPP-originated occasion/temporal design is admitted only through an explicit "
        "versioned, immutable contract with a defined compatibility identity."
    ) in adr
    assert (
        "No cross-service SQL, TEPP database access, or hidden runtime dependency is permitted."
    ) in adr
    assert (
        "| [0028](0028-tepp-temporal-event-composition-boundary.md) | Accepted | "
        "TEPP owns temporal/event composition and semantics; fast-mlsirm retains reusable "
        "time-indexed psychometric numerical kernels behind an explicit Anti-Corruption Layer. |"
    ) in index


def test_boundary_qualifies_prior_longitudinal_model_proposals() -> None:
    """The owner ADR and index must qualify earlier numerical proposals without erasing them."""
    adr = _read(ADR_PATH)
    index = _read("docs/adr/README.md")

    for historical_adr in ("ADR-0007", "ADR-0019", "ADR-0020"):
        assert historical_adr in adr
    assert "remain useful numerical/model-design records" in adr
    assert "interpreted through this ownership boundary" in adr
    assert "[0007](0007-multilevel-multiple-membership-temporal.md)" in index
    assert "TEPP temporal/event ownership is governed by ADR-0028" in index
    assert "[0019](0019-rust-longitudinal-state-engine.md)" in index
    assert "event ontology, temporal validity and composition remain TEPP-owned under ADR-0028" in index
    assert "[0020](0020-joint-hierarchical-ctar-rasch.md)" in index
    assert "temporal/event composition remains TEPP-owned under ADR-0028" in index


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
    """Architecture facts must not couple to an unreleased sibling repository head."""
    adr = _read(ADR_PATH)
    assert (
        "Architecture/package/backend/toolchain/consumer-lifecycle facts may be projected to "
        "`ContextualWisdomLab/enterprise-architecture-core` only through an immutable released "
        "`ContextualWisdomLab/context-graph-contracts` versioned Context Assertion / CloudEvent / "
        "conformance contract with provenance."
    ) in adr
    assert (
        "Estimator values, latent scores, DIF/fit diagnostics, and scientific-validity evidence "
        "are not authoritative EA facts and must not be duplicated into the architecture decision plane."
    ) in adr
    assert (
        "Therefore production EA projection fails closed rather than pinning an unreleased sibling PR head."
    ) in adr
