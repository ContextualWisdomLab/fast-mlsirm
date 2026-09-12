"""Metering validator diagnostic-formatting resource regressions."""

from __future__ import annotations

from types import SimpleNamespace

import fast_mlsirm.metering as metering
from fast_mlsirm.metering import CanonicalComputeUsageSink


def test_validator_diagnostic_rendering_is_bounded_per_entry(monkeypatch) -> None:
    """Package formatting cannot amplify accepted validator diagnostics unboundedly."""
    queued: list[dict[str, object]] = []

    def builder(**_: object) -> dict[str, object]:
        return {"event_contract_version": 1}

    def validator(_: object) -> tuple[str, ...]:
        return ("abcdefgh", "ij", "klmnop", "unused")

    monkeypatch.setattr(
        metering,
        "_MAX_FORMATTED_VALIDATOR_DIAGNOSTIC_CHARS",
        4,
        raising=False,
    )
    sink = CanonicalComputeUsageSink(
        event_builder=builder,
        event_validator=validator,
        enqueue=queued.append,
        identity={},
    )

    try:
        sink.emit_fit(
            SimpleNamespace(model="MLS2PLM", backend="rust"),
            run_reference="run",
            artifact_reference="artifact",
            configuration_reference="config",
            seed_reference="seed",
            occurred_at="2026-08-30T00:00:00Z",
            response_rows=1,
            response_items=1,
        )
    except ValueError as error:
        assert str(error) == (
            "event_builder output violates canonical usage-event v1 contract: "
            "abcd; ij; klmn"
        )
    else:
        raise AssertionError("schema-invalid event was accepted")

    assert queued == []
