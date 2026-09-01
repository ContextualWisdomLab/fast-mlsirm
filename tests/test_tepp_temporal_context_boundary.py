"""Architecture fitness tests for the TEPP temporal/event Anti-Corruption Layer."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADR_PATH = "docs/adr/0028-tepp-temporal-event-composition-boundary.md"


def _read(path: str) -> str:
    """Return UTF-8 repository text for one governed boundary artifact."""
    return (ROOT / path).read_text(encoding="utf-8")


def test_temporal_event_semantics_have_one_foreign_owner() -> None:
    """Keep event composition in TEPP while retaining fast numerical kernels."""
    adr = _read(ADR_PATH)
    index = _read("docs/adr/README.md")

    assert "Status: Accepted" in adr
    assert "TEPP owns temporal/event composition and semantics" in adr
    assert "fast-mlsirm owns reusable time-indexed psychometric numerical kernels" in adr
    assert "event ontology and graph construction" in adr
    assert "Anti-Corruption Layer" in adr
    assert "cross-service SQL" in adr
    assert "0028-tepp-temporal-event-composition-boundary.md" in index
    assert "TEPP owns temporal/event composition and semantics" in index


def test_boundary_qualifies_prior_longitudinal_model_proposals() -> None:
    """The owner ADR must qualify earlier numerical proposals without erasing them."""
    adr = _read(ADR_PATH)

    for historical_adr in ("ADR-0007", "ADR-0019", "ADR-0020"):
        assert historical_adr in adr
    assert "remain useful numerical/model-design records" in adr
    assert "interpreted through this ownership boundary" in adr


def test_existing_ctar_rasch_remains_a_rust_numerical_kernel() -> None:
    """Boundary repair must not erase or provider-couple the protected-main CT-AR estimand."""
    rust = _read("crates/mlsirm-core/src/longitudinal_irt.rs")
    assert 'const ESTIMAND_SCOPE: &str = "joint_map_hierarchical_ctar_rasch"' in rust
    assert 'const TRANSITION_KIND: &str = "continuous_time_ar1_ou"' in rust
    assert "pub fn ctar_phi" in rust
    assert "TEPP" not in rust
    assert "contextual_orchestrator" not in rust


def test_ea_projection_requires_released_context_graph_contract() -> None:
    """Architecture facts must not couple to an unreleased sibling repository head."""
    adr = _read(ADR_PATH)
    assert "immutable released" in adr
    assert "context-graph-contracts" in adr
    assert "enterprise-architecture-core" in adr
    assert "Estimator values, latent scores, DIF/fit diagnostics" in adr
    assert "unreleased sibling PR head" in adr
