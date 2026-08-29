"""Producer-result snapshot ordering regressions for metering export."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import fast_mlsirm.metering as metering
from fast_mlsirm.metering import CanonicalComputeUsageSink


def _emit_one_fit(sink: CanonicalComputeUsageSink) -> None:
    """Exercise the fit emission boundary with inert package-owned evidence."""
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


def test_version_is_validated_from_package_snapshot(monkeypatch) -> None:
    """A producer mutation at snapshot time cannot bypass the explicit v1 guard."""
    producer_event: dict[str, object] = {"event_contract_version": 1}
    validator_calls = 0
    queued: list[dict[str, object]] = []
    snapshot_calls = 0
    original_snapshot = metering._snapshot_exact_json

    def builder(**_: object) -> dict[str, object]:
        return producer_event

    def mutating_snapshot(value: object, **kwargs: object) -> object:
        nonlocal snapshot_calls
        if value is producer_event and snapshot_calls == 0:
            producer_event["event_contract_version"] = 2
        snapshot_calls += 1
        return original_snapshot(value, **kwargs)

    def permissive_validator(_: object) -> tuple[str, ...]:
        nonlocal validator_calls
        validator_calls += 1
        return ()

    monkeypatch.setattr(metering, "_snapshot_exact_json", mutating_snapshot)
    sink = CanonicalComputeUsageSink(
        event_builder=builder,
        event_validator=permissive_validator,
        enqueue=queued.append,
        identity={},
    )

    with pytest.raises(
        ValueError,
        match="event_builder must return event_contract_version=1",
    ):
        _emit_one_fit(sink)

    assert producer_event["event_contract_version"] == 2
    assert validator_calls == 0
    assert queued == []
