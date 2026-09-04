"""Metering validator/enqueue trust-boundary regressions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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


def test_validator_mutation_cannot_change_enqueued_event() -> None:
    """Top-level mutation is detected after callback and before durable enqueue."""
    queued: list[dict[str, object]] = []
    callback_completed = False

    def builder(**_: object) -> dict[str, object]:
        return {
            "event_contract_version": 1,
            "source_event_key": "producer-owned",
        }

    def validator(event: Mapping[str, object]) -> tuple[str, ...]:
        nonlocal callback_completed
        event["event_contract_version"] = 2  # type: ignore[index]
        event["source_event_key"] = "validator-mutated"  # type: ignore[index]
        callback_completed = True
        return ()

    sink = CanonicalComputeUsageSink(
        event_builder=builder,
        event_validator=validator,
        enqueue=queued.append,
        identity={},
    )

    with pytest.raises(ValueError, match="event_validator must not mutate event"):
        _emit_one_fit(sink)

    assert callback_completed
    assert queued == []


def test_validator_nested_mutation_cannot_change_enqueued_event() -> None:
    """Nested mutation is detected after callback and before durable enqueue."""
    queued: list[dict[str, object]] = []
    callback_completed = False

    def builder(**_: object) -> dict[str, object]:
        return {
            "event_contract_version": 1,
            "measurements": [
                {
                    "meter_code": "response_rows",
                    "quantity": "1",
                    "unit_code": "count",
                    "quality_code": "deterministically_derived",
                }
            ],
        }

    def validator(event: Mapping[str, object]) -> tuple[str, ...]:
        nonlocal callback_completed
        measurements = event["measurements"]
        assert isinstance(measurements, Sequence)
        measurement = measurements[0]
        assert isinstance(measurement, Mapping)
        measurement["quantity"] = "999"  # type: ignore[index]
        callback_completed = True
        return ()

    sink = CanonicalComputeUsageSink(
        event_builder=builder,
        event_validator=validator,
        enqueue=queued.append,
        identity={},
    )

    with pytest.raises(ValueError, match="event_validator must not mutate event"):
        _emit_one_fit(sink)

    assert callback_completed
    assert queued == []


def test_nested_callback_carrier_is_rejected_before_validator() -> None:
    """Protocol-bearing nested carriers cannot cross package admission."""
    callbacks = 0
    validator_calls = 0
    queued: list[dict[str, object]] = []

    class HostileMeasurements(list[object]):
        def __iter__(self):  # type: ignore[no-untyped-def]
            nonlocal callbacks
            callbacks += 1
            raise AssertionError("nested iteration callback executed")

        def __len__(self) -> int:
            nonlocal callbacks
            callbacks += 1
            raise AssertionError("nested length callback executed")

        def __getitem__(self, index):  # type: ignore[no-untyped-def]
            nonlocal callbacks
            callbacks += 1
            raise AssertionError(f"nested item callback executed for {index}")

    def builder(**_: object) -> dict[str, object]:
        return {
            "event_contract_version": 1,
            "measurements": HostileMeasurements(),
        }

    def validator(_: object) -> tuple[str, ...]:
        nonlocal validator_calls
        validator_calls += 1
        return ()

    sink = CanonicalComputeUsageSink(
        event_builder=builder,
        event_validator=validator,
        enqueue=queued.append,
        identity={},
    )
    try:
        _emit_one_fit(sink)
    except ValueError as error:
        assert "exact JSON" in str(error)
    else:
        raise AssertionError("callback-bearing nested carrier was accepted")

    assert callbacks == 0
    assert validator_calls == 0
    assert queued == []


def test_oversized_top_level_event_preflights_before_key_scan(monkeypatch) -> None:
    """Impossible top-level cardinality wins before result-key inspection."""
    validator_calls = 0
    queued: list[dict[str, object]] = []

    class NonExactKey(str):
        pass

    def builder(**_: object) -> dict[str, object]:
        return {
            "event_contract_version": 1,
            "source_event_key": "producer-owned",
            NonExactKey("third_key"): "unreachable",
        }

    def validator(_: object) -> tuple[str, ...]:
        nonlocal validator_calls
        validator_calls += 1
        return ()

    monkeypatch.setattr(metering, "_MAX_EVENT_SNAPSHOT_NODES", 3)
    sink = CanonicalComputeUsageSink(
        event_builder=builder,
        event_validator=validator,
        enqueue=queued.append,
        identity={},
    )

    try:
        _emit_one_fit(sink)
    except ValueError as error:
        assert str(error) == "event_builder result exact JSON tree is too large"
    else:
        raise AssertionError("oversized top-level event was accepted")

    assert validator_calls == 0
    assert queued == []


def test_oversized_nested_dict_preflights_before_nested_key_scan(monkeypatch) -> None:
    """Impossible nested cardinality wins before nested result-key inspection."""
    validator_calls = 0
    queued: list[dict[str, object]] = []

    class NonExactKey(str):
        pass

    def builder(**_: object) -> dict[str, object]:
        return {
            "event_contract_version": 1,
            "metadata": {
                "first_key": "value",
                NonExactKey("second_key"): "unreachable",
            },
        }

    def validator(_: object) -> tuple[str, ...]:
        nonlocal validator_calls
        validator_calls += 1
        return ()

    monkeypatch.setattr(metering, "_MAX_EVENT_SNAPSHOT_NODES", 4)
    sink = CanonicalComputeUsageSink(
        event_builder=builder,
        event_validator=validator,
        enqueue=queued.append,
        identity={},
    )

    try:
        _emit_one_fit(sink)
    except ValueError as error:
        assert str(error) == "event_builder result exact JSON tree is too large"
    else:
        raise AssertionError("oversized nested event was accepted")

    assert validator_calls == 0
    assert queued == []


def test_oversized_validator_diagnostics_preflight_before_entry_scan(monkeypatch) -> None:
    """Impossible validator-result cardinality wins before entry inspection."""
    queued: list[dict[str, object]] = []

    class NonExactDiagnostic(str):
        pass

    def builder(**_: object) -> dict[str, object]:
        return {"event_contract_version": 1}

    def validator(_: object) -> tuple[str, ...]:
        return ("first", "second", NonExactDiagnostic("third"))

    monkeypatch.setattr(metering, "_MAX_VALIDATOR_DIAGNOSTICS", 2, raising=False)
    sink = CanonicalComputeUsageSink(
        event_builder=builder,
        event_validator=validator,
        enqueue=queued.append,
        identity={},
    )

    try:
        _emit_one_fit(sink)
    except ValueError as error:
        assert str(error) == "event_validator returned too many diagnostics"
    else:
        raise AssertionError("oversized validator diagnostics were accepted")

    assert queued == []
