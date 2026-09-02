"""Architecture fitness tests for the fast-mlsirm bounded-context map."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTEXT_MAP = ROOT / "docs" / "context-map.md"


def test_context_map_declares_internal_bounded_contexts() -> None:
    """Keep numerical ownership and public binding responsibilities explicit."""
    context_map = CONTEXT_MAP.read_text(encoding="utf-8")

    for bounded_context in (
        "Model Specification",
        "Estimation",
        "Scoring",
        "Diagnostics",
        "Simulation-Recovery",
        "Compute Backend",
        "Public Binding",
    ):
        assert f"`{bounded_context}`" in context_map

    assert "Production psychometric arithmetic is Rust-owned" in context_map
    assert "Python is limited to validation, immutable marshalling, reporting, and binding ergonomics" in context_map


def test_context_map_declares_foreign_owner_relationships() -> None:
    """Prevent temporal, product, and architecture authorities from leaking inward."""
    context_map = CONTEXT_MAP.read_text(encoding="utf-8")

    assert "TEPP owns temporal/event composition and semantics" in context_map
    assert "psychometrics-commons is a downstream hosted-product consumer" in context_map
    assert "context-graph-contracts is the contract-only Shared Kernel" in context_map
    assert "enterprise-architecture-core is the authoritative EA Decision Plane" in context_map
    assert "cross-service SQL is prohibited" in context_map
    assert "immutable released context-graph-contracts contract" in context_map
    assert "Estimator values, latent scores, DIF/fit diagnostics, and scientific-validity evidence are not EA-authoritative facts" in context_map
