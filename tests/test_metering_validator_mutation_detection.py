"""Fail-first validator mutation detection for canonical metering export."""

from __future__ import annotations

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
    """A successful validator cannot validate one tree while another is queued."""
    queued: list[dict[str, object]] = []

    def validator(event: object) -> tuple[str, ...]:
        assert type(event) is dict
        event["source_event_key"] = "validator-mutated"
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

    with pytest.raises(ValueError, match="validator mutated"):
        _emit_one_fit(sink)

    assert queued == []


def test_nested_validator_mutation_is_rejected_before_enqueue() -> None:
    """Nested measurement mutation invalidates the validator's success evidence."""
    queued: list[dict[str, object]] = []

    def validator(event: object) -> tuple[str, ...]:
        assert type(event) is dict
        measurements = event["measurements"]
        assert type(measurements) is list
        measurement = measurements[0]
        assert type(measurement) is dict
        measurement["quantity"] = "999"
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

    with pytest.raises(ValueError, match="validator mutated"):
        _emit_one_fit(sink)

    assert queued == []


def test_validator_json_type_mutation_is_not_hidden_by_python_equality() -> None:
    """Exact JSON type identity distinguishes integer 1 from Boolean true."""
    queued: list[dict[str, object]] = []

    def validator(event: object) -> tuple[str, ...]:
        assert type(event) is dict
        event["event_contract_version"] = True
        return ()

    sink = CanonicalComputeUsageSink(
        event_builder=lambda **_: {"event_contract_version": 1},
        event_validator=validator,
        enqueue=queued.append,
        identity={},
    )

    with pytest.raises(ValueError, match="validator mutated"):
        _emit_one_fit(sink)

    assert queued == []
