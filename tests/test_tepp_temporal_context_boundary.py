"""Architecture fitness tests for the TEPP temporal/event Anti-Corruption Layer."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    """Return UTF-8 repository text for one governed boundary artifact."""
    return (ROOT / path).read_text(encoding="utf-8")


def test_temporal_event_semantics_have_one_foreign_owner() -> None:
    """Keep event composition in TEPP while retaining fast numerical kernels."""
    architecture = _read("ARCHITECTURE.md")
    prd = _read("docs/PRD.md")
    trd = _read("docs/TRD.md")

    required_boundary = "TEPP owns temporal/event composition and semantics"
    for document in (architecture, prd, trd):
        assert required_boundary in document
        assert "fast-mlsirm owns reusable time-indexed psychometric numerical kernels" in document


def test_proposed_longitudinal_adrs_do_not_claim_event_ontology_ownership() -> None:
    """Proposed numerical-model ADRs must name the foreign temporal authority."""
    for path in (
        "docs/adr/0007-multilevel-multiple-membership-temporal.md",
        "docs/adr/0019-rust-longitudinal-state-engine.md",
        "docs/adr/0020-joint-hierarchical-ctar-rasch.md",
    ):
        text = _read(path)
        assert "TEPP owns temporal/event composition and semantics" in text
        assert "event ontology" in text.lower()
        assert "numerical" in text.lower()


def test_existing_ctar_rasch_remains_a_rust_numerical_kernel() -> None:
    """Boundary repair must not erase the protected-main CT-AR estimand."""
    rust = _read("crates/mlsirm-core/src/longitudinal_irt.rs")
    assert 'const ESTIMAND_SCOPE: &str = "joint_map_hierarchical_ctar_rasch"' in rust
    assert 'const TRANSITION_KIND: &str = "continuous_time_ar1_ou"' in rust
    assert "pub fn ctar_phi" in rust
    assert "TEPP" not in rust
