"""Research-traceability fitness tests for temporal psychometric ownership."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACEABILITY = ROOT / "docs" / "traceability" / "temporal-research-ownership.md"
CONTEXT_MAP = ROOT / "docs" / "context-map.md"


def test_temporal_research_record_separates_science_from_owner_authority() -> None:
    """Keep valid longitudinal research distinct from TEPP semantic ownership."""
    record = TRACEABILITY.read_text(encoding="utf-8")

    assert "Scientific status: Proposed / evolving implementation" in record
    assert "Ownership status: Accepted via ADR-0028" in record
    assert "TEPP owns temporal/event composition and semantics" in record
    assert "fast-mlsirm owns reusable time-indexed psychometric numerical kernels" in record
    assert "do not transfer temporal/event semantic authority" in record
    assert "Fox, J.-P., & Glas, C. A. W. (2001)" in record
    assert "Browne, W. J., Goldstein, H., & Rasbash, J. (2001)" in record
    assert "Jeon, M., & Rabe-Hesketh, S. (2016)" in record
    assert "Oravecz, Z., Tuerlinckx, F., & Vandekerckhove, J. (2011)" in record


def test_temporal_research_record_fails_closed_on_ea_projection() -> None:
    """Scientific recovery evidence must not become architecture authority."""
    record = TRACEABILITY.read_text(encoding="utf-8")

    assert "immutable released context-graph-contracts contract" in record
    assert "Estimator values, latent scores, DIF/fit diagnostics, bias, RMSE, coverage, and scientific-validity evidence are not EA-authoritative facts" in record
    assert "cross-service SQL is prohibited" in record


def test_context_map_links_temporal_research_traceability() -> None:
    """Make the ownership-specific research record discoverable from the DDD map."""
    context_map = CONTEXT_MAP.read_text(encoding="utf-8")

    assert "docs/traceability/temporal-research-ownership.md" in context_map
