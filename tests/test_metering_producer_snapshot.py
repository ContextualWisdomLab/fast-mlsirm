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


def test_version_is_validated_from_package_root_snapshot(monkeypatch) -> None:
    """Mutation after bounded root capture cannot rewrite admitted v1 evidence."""
    producer_event: dict[str, object] = {"event_contract_version": 1}
    validator_calls = 0
    queued: list[dict[str, object]] = []
    original_items = metering._bounded_exact_dict_items
    mutated = False

    def builder(**_: object) -> dict[str, object]:
        return producer_event

    def mutating_items(
        value: dict[object, object],
        *,
        max_items: int,
        too_large_message: str,
        changed_message: str,
    ) -> tuple[tuple[object, object], ...]:
        nonlocal mutated
        items = original_items(
            value,
            max_items=max_items,
            too_large_message=too_large_message,
            changed_message=changed_message,
        )
        if value is producer_event and not mutated:
            producer_event["event_contract_version"] = 2
            mutated = True
        return items

    def permissive_validator(_: object) -> tuple[str, ...]:
        nonlocal validator_calls
        validator_calls += 1
        return ()

    monkeypatch.setattr(metering, "_bounded_exact_dict_items", mutating_items)
    sink = CanonicalComputeUsageSink(
        event_builder=builder,
        event_validator=permissive_validator,
        enqueue=queued.append,
        identity={},
    )

    _emit_one_fit(sink)

    assert mutated
    assert producer_event["event_contract_version"] == 2
    assert validator_calls == 1
    assert queued == [{"event_contract_version": 1}]


def test_wrong_exact_version_fails_before_recursive_snapshot_values() -> None:
    """A package-owned wrong v1 marker outranks descendant JSON traversal."""
    validator_calls = 0
    queued: list[dict[str, object]] = []

    def builder(**_: object) -> dict[str, object]:
        return {
            "event_contract_version": 2,
            "measurements": [{"quantity": float("nan")}],
        }

    def permissive_validator(_: object) -> tuple[str, ...]:
        nonlocal validator_calls
        validator_calls += 1
        return ()

    sink = CanonicalComputeUsageSink(
        event_builder=builder,
        event_validator=permissive_validator,
        enqueue=queued.append,
        identity={},
    )

    with pytest.raises(
        ValueError,
        match="^event_builder must return event_contract_version=1$",
    ):
        _emit_one_fit(sink)

    assert validator_calls == 0
    assert queued == []
