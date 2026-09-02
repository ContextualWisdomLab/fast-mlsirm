"""Regression tests for historical longitudinal ADR ownership qualification."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    """Return one historical ADR as UTF-8 repository text."""
    return (ROOT / path).read_text(encoding="utf-8")


def test_adr_0019_keeps_temporal_event_semantics_in_tepp() -> None:
    """Keep the narrow OLS/AR numerical layer from becoming a temporal owner."""
    adr = _read("docs/adr/0019-rust-longitudinal-state-engine.md")

    assert (
        "ADR-0028 governs the temporal/event ownership boundary for this Proposed numerical layer."
        in adr
    )
    assert (
        "TEPP owns event ontology, temporal validity, event ordering, changing-membership history, "
        "longitudinal leakage policy, and temporal/event composition."
        in adr
    )
    assert (
        "fast-mlsirm owns only the OLS and discrete-AR psychometric arithmetic over explicit supplied "
        "occasion/time carriers described here."
        in adr
    )


def test_adr_0020_keeps_temporal_event_semantics_in_tepp() -> None:
    """Keep the CT-AR Rasch numerical kernel from becoming a temporal owner."""
    adr = _read("docs/adr/0020-joint-hierarchical-ctar-rasch.md")

    assert (
        "ADR-0028 governs temporal/event composition for this Proposed CT-AR Rasch numerical kernel."
        in adr
    )
    assert (
        "TEPP owns event ontology, temporal validity, event ordering, changing-membership history, "
        "longitudinal leakage policy, and temporal/event composition."
        in adr
    )
    assert (
        "fast-mlsirm owns the joint MAP likelihood, elapsed-time transition arithmetic, optimization, "
        "and uncertainty calculations over explicit supplied occasion/time carriers."
        in adr
    )
