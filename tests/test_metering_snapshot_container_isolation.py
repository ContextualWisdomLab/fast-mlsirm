"""Mutable-container isolation regressions for metering event snapshots."""

from __future__ import annotations

from types import SimpleNamespace

import fast_mlsirm.metering as metering
from fast_mlsirm.metering import CanonicalComputeUsageSink


def _emit_one_fit(sink: CanonicalComputeUsageSink) -> None:
    """Exercise the fit emission boundary with inert package-owned metadata."""
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


def _sink_for_event(
    event: dict[str, object], queued: list[dict[str, object]]
) -> CanonicalComputeUsageSink:
    """Build a permissive-validator sink around one retained producer event."""

    def builder(**_: object) -> dict[str, object]:
        return event

    def permissive_validator(_: object) -> tuple[str, ...]:
        return ()

    return CanonicalComputeUsageSink(
        event_builder=builder,
        event_validator=permissive_validator,
        enqueue=queued.append,
        identity={},
    )


def test_root_dict_is_frozen_before_recursive_snapshot_descent(monkeypatch) -> None:
    """Later root members retain the state present when root admission starts."""
    measurements: list[object] = [{"quantity": 1}]
    producer_event: dict[str, object] = {
        "event_contract_version": 1,
        "measurements": measurements,
        "source_reference": "source-old",
    }
    queued: list[dict[str, object]] = []
    original_snapshot = metering._snapshot_exact_json
    mutated = False

    def mutating_snapshot(value: object, **kwargs: object) -> object:
        nonlocal mutated
        if value is measurements and not mutated:
            producer_event["event_contract_version"] = 2
            producer_event["source_reference"] = "source-new"
            mutated = True
        return original_snapshot(value, **kwargs)

    monkeypatch.setattr(metering, "_snapshot_exact_json", mutating_snapshot)
    _emit_one_fit(_sink_for_event(producer_event, queued))

    assert producer_event["event_contract_version"] == 2
    assert producer_event["source_reference"] == "source-new"
    assert len(queued) == 1
    assert queued[0]["event_contract_version"] == 1
    assert queued[0]["source_reference"] == "source-old"


def test_nested_list_is_frozen_before_recursive_snapshot_descent(monkeypatch) -> None:
    """Later list elements retain the state present when list admission starts."""
    trigger: dict[str, object] = {"quantity": 1}
    measurements: list[object] = [trigger, "member-old"]
    producer_event: dict[str, object] = {
        "event_contract_version": 1,
        "measurements": measurements,
    }
    queued: list[dict[str, object]] = []
    original_snapshot = metering._snapshot_exact_json
    mutated = False

    def mutating_snapshot(value: object, **kwargs: object) -> object:
        nonlocal mutated
        if value is trigger and not mutated:
            measurements[1] = "member-new"
            mutated = True
        return original_snapshot(value, **kwargs)

    monkeypatch.setattr(metering, "_snapshot_exact_json", mutating_snapshot)
    _emit_one_fit(_sink_for_event(producer_event, queued))

    assert measurements[1] == "member-new"
    assert len(queued) == 1
    assert queued[0]["measurements"] == [{"quantity": 1}, "member-old"]
