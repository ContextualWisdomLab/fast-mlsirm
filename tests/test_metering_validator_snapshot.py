"""Metering validator/enqueue trust-boundary regressions."""

from __future__ import annotations

from types import SimpleNamespace

from fast_mlsirm.metering import CanonicalComputeUsageSink


def test_validator_mutation_cannot_change_enqueued_event() -> None:
    """Validator-local top-level mutation cannot rewrite durable event evidence."""
    queued: list[dict[str, object]] = []

    def builder(**_: object) -> dict[str, object]:
        return {
            "event_contract_version": 1,
            "source_event_key": "producer-owned",
        }

    def validator(event: object) -> tuple[str, ...]:
        assert type(event) is dict
        event["event_contract_version"] = 2
        event["source_event_key"] = "validator-mutated"
        return ()

    sink = CanonicalComputeUsageSink(
        event_builder=builder,
        event_validator=validator,
        enqueue=queued.append,
        identity={},
    )
    sink.emit_fit(
        SimpleNamespace(model="MLS2PLM", backend="rust"),
        run_reference="run",
        artifact_reference="artifact",
        configuration_reference="config",
        seed_reference="seed",
        occurred_at="2026-08-29T00:00:00Z",
        response_rows=1,
        response_items=1,
    )

    assert queued == [
        {
            "event_contract_version": 1,
            "source_event_key": "producer-owned",
        }
    ]
