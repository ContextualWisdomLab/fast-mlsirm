"""Fail-first regression for transient validator mutation of canonical evidence."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fast_mlsirm.metering import CanonicalComputeUsageSink


def test_validator_cannot_mutate_then_restore_before_enqueue() -> None:
    """Validator success must apply to the exact event selected for durable enqueue."""
    queued: list[dict[str, object]] = []

    def validator(event: object) -> tuple[str, ...]:
        assert type(event) is dict
        original = event["source_event_key"]
        event["source_event_key"] = "temporarily-valid"
        # A buggy validator may validate the temporary state here, then restore
        # the original evidence before returning success.
        event["source_event_key"] = original
        return ()

    sink = CanonicalComputeUsageSink(
        event_builder=lambda **_: {
            "event_contract_version": 1,
            "source_event_key": "producer-owned",
        },
        event_validator=validator,
        enqueue=queued.append,
        identity={},
    )

    with pytest.raises((TypeError, ValueError)):
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

    assert queued == []
