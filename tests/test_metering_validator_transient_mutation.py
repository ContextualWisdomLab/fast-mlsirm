"""Fail-first regression for transient validator mutation of canonical evidence."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import pytest

from fast_mlsirm.metering import CanonicalComputeUsageSink


def test_validator_cannot_mutate_then_restore_before_enqueue() -> None:
    """A read-only validator view makes mutate-then-restore impossible to begin."""
    queued: list[dict[str, object]] = []

    def validator(event: Mapping[str, object]) -> tuple[str, ...]:
        original = event["source_event_key"]
        event["source_event_key"] = "temporarily-valid"  # type: ignore[index]
        # Unreachable by contract: the first mutation must fail before a buggy
        # validator could validate the temporary state and restore the original.
        event["source_event_key"] = original  # type: ignore[index]
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

    with pytest.raises(TypeError):
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
