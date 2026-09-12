"""Fail-first regression for transient validator mutation of canonical evidence."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fast_mlsirm.metering import CanonicalComputeUsageSink


def test_validator_mutate_then_restore_is_rejected_before_enqueue() -> None:
    """Mutation evidence survives restoration of the validator's JSON tree."""
    queued: list[dict[str, object]] = []
    callback_completed = False

    def validator(event: dict[str, object]) -> tuple[str, ...]:
        nonlocal callback_completed
        original = event["source_event_key"]
        event["source_event_key"] = "temporarily-valid"
        # A buggy validator could validate the temporary state and then restore
        # the producer value before returning. The sink must still remember that
        # mutation occurred during the callback.
        event["source_event_key"] = original
        callback_completed = True
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

    with pytest.raises(ValueError, match="event_validator must not mutate event"):
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

    assert callback_completed
    assert queued == []
