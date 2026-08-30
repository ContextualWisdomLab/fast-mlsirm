"""Validator immutability regressions for canonical metering export."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace

import pytest

from fast_mlsirm.metering import CanonicalComputeUsageSink


def _emit_one_fit(sink: CanonicalComputeUsageSink) -> None:
    """Exercise one fit emission with inert package-owned input metadata."""
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


def test_top_level_validator_mutation_is_rejected_before_enqueue() -> None:
    """The validator cannot mutate the read-only root selected for validation."""
    queued: list[dict[str, object]] = []

    def validator(event: Mapping[str, object]) -> tuple[str, ...]:
        event["source_event_key"] = "validator-mutated"  # type: ignore[index]
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
        _emit_one_fit(sink)

    assert queued == []


def test_nested_validator_mutation_is_rejected_before_enqueue() -> None:
    """Nested mapping evidence is recursively read-only during validation."""
    queued: list[dict[str, object]] = []

    def validator(event: Mapping[str, object]) -> tuple[str, ...]:
        measurements = event["measurements"]
        assert isinstance(measurements, Sequence)
        measurement = measurements[0]
        assert isinstance(measurement, Mapping)
        measurement["quantity"] = "999"  # type: ignore[index]
        return ()

    sink = CanonicalComputeUsageSink(
        event_builder=lambda **_: {
            "event_contract_version": 1,
            "measurements": [
                {
                    "meter_code": "response_rows",
                    "quantity": "1",
                    "unit_code": "count",
                    "quality_code": "deterministically_derived",
                }
            ],
        },
        event_validator=validator,
        enqueue=queued.append,
        identity={},
    )

    with pytest.raises(TypeError):
        _emit_one_fit(sink)

    assert queued == []


def test_validator_json_type_mutation_is_rejected_before_equality_can_hide_it() -> None:
    """Integer-to-Boolean mutation cannot begin on the immutable validator view."""
    queued: list[dict[str, object]] = []

    def validator(event: Mapping[str, object]) -> tuple[str, ...]:
        event["event_contract_version"] = True  # type: ignore[index]
        return ()

    sink = CanonicalComputeUsageSink(
        event_builder=lambda **_: {"event_contract_version": 1},
        event_validator=validator,
        enqueue=queued.append,
        identity={},
    )

    with pytest.raises(TypeError):
        _emit_one_fit(sink)

    assert queued == []
