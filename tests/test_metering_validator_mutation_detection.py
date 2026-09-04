"""Validator mutation regressions for canonical metering export."""

from __future__ import annotations

import json
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


def test_top_level_validator_mutation_is_rejected_after_callback_before_enqueue() -> None:
    """A validator may inspect ordinary JSON but any top-level write fails closed."""
    queued: list[dict[str, object]] = []
    callback_completed = False

    def validator(event: dict[str, object]) -> tuple[str, ...]:
        nonlocal callback_completed
        assert isinstance(event, dict)
        event["source_event_key"] = "validator-mutated"
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
        _emit_one_fit(sink)

    assert callback_completed
    assert queued == []


def test_base_dict_mutator_cannot_bypass_validator_mutation_detection() -> None:
    """Calling the built-in dict mutator directly must not bypass the guard."""
    queued: list[dict[str, object]] = []
    callback_completed = False

    def validator(event: dict[str, object]) -> tuple[str, ...]:
        nonlocal callback_completed
        # ``dict.__setitem__`` bypasses a dict subclass' Python override. The
        # sink must still reject the resulting validator tree before enqueue.
        dict.__setitem__(event, "source_event_key", "validator-mutated")
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
        _emit_one_fit(sink)

    assert callback_completed
    assert queued == []


def test_nested_validator_mutation_is_rejected_after_callback_before_enqueue() -> None:
    """Mutation tracking covers nested ordinary list/dict JSON carriers."""
    queued: list[dict[str, object]] = []
    callback_completed = False

    def validator(event: dict[str, object]) -> tuple[str, ...]:
        nonlocal callback_completed
        measurements = event["measurements"]
        assert isinstance(measurements, list)
        measurement = measurements[0]
        assert isinstance(measurement, dict)
        measurement["quantity"] = "999"
        callback_completed = True
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

    with pytest.raises(ValueError, match="event_validator must not mutate event"):
        _emit_one_fit(sink)

    assert callback_completed
    assert queued == []


def test_validator_json_type_mutation_is_rejected_even_when_values_compare_equal() -> None:
    """Integer-to-Boolean writes are mutation even though ``1 == True``."""
    queued: list[dict[str, object]] = []

    def validator(event: dict[str, object]) -> tuple[str, ...]:
        event["event_contract_version"] = True
        return ()

    sink = CanonicalComputeUsageSink(
        event_builder=lambda **_: {"event_contract_version": 1},
        event_validator=validator,
        enqueue=queued.append,
        identity={},
    )

    with pytest.raises(ValueError, match="event_validator must not mutate event"):
        _emit_one_fit(sink)

    assert queued == []


def test_non_mutating_validator_receives_json_compatible_dicts_and_lists() -> None:
    """Mutation protection must not replace ordinary JSON carriers with proxies/tuples."""
    queued: list[dict[str, object]] = []
    serialized: list[str] = []
    event = {
        "event_contract_version": 1,
        "source_event_key": "producer-owned",
        "measurements": [
            {
                "meter_code": "response_rows",
                "quantity": "1",
                "unit_code": "count",
                "quality_code": "deterministically_derived",
            }
        ],
    }

    def validator(candidate: dict[str, object]) -> tuple[str, ...]:
        assert isinstance(candidate, dict)
        measurements = candidate["measurements"]
        assert isinstance(measurements, list)
        assert isinstance(measurements[0], dict)
        serialized.append(json.dumps(candidate, sort_keys=True))
        return ()

    sink = CanonicalComputeUsageSink(
        event_builder=lambda **_: event,
        event_validator=validator,
        enqueue=queued.append,
        identity={},
    )

    _emit_one_fit(sink)

    assert serialized == [json.dumps(event, sort_keys=True)]
    assert queued == [event]
