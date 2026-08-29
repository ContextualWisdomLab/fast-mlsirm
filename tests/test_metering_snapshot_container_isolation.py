"""Mutable-container isolation regressions for metering event snapshots."""

from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import pytest

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


def _profile_snapshot_copy_calls(value: object) -> list[str]:
    """Return built-in copy calls made directly by exact-JSON snapshot admission."""
    copy_calls: list[str] = []
    snapshot_code = metering._snapshot_exact_json.__code__

    def profiler(frame, event: str, arg: object) -> None:
        if event != "c_call" or frame.f_code is not snapshot_code:
            return
        qualname = getattr(arg, "__qualname__", "")
        if qualname in {"list.copy", "dict.copy"}:
            copy_calls.append(qualname)

    sys.setprofile(profiler)
    try:
        with pytest.raises(
            ValueError, match="event_builder result exact JSON tree is too large"
        ):
            metering._snapshot_exact_json(value)
    finally:
        sys.setprofile(None)
    return copy_calls


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


def test_list_growth_after_preflight_uses_only_a_bounded_snapshot(monkeypatch) -> None:
    """A stale list length cannot authorize an unbounded full-container copy."""
    values: list[object] = [0]
    real_len = builtins.len
    raced = False

    def racing_len(value: object) -> int:
        nonlocal raced
        observed = real_len(value)
        if value is values and not raced:
            values.extend(range(1, 10))
            raced = True
        return observed

    monkeypatch.setattr(metering, "_MAX_EVENT_SNAPSHOT_NODES", 8)
    monkeypatch.setattr(metering, "len", racing_len, raising=False)

    copy_calls = _profile_snapshot_copy_calls(values)

    assert raced
    assert real_len(values) == 10
    assert "list.copy" not in copy_calls


def test_dict_growth_after_preflight_uses_only_a_bounded_snapshot(monkeypatch) -> None:
    """A stale dict length cannot authorize an unbounded full-container copy."""
    values: dict[str, object] = {"k0": 0}
    real_len = builtins.len
    raced = False

    def racing_len(value: object) -> int:
        nonlocal raced
        observed = real_len(value)
        if value is values and not raced:
            values.update({f"k{index}": index for index in range(1, 10)})
            raced = True
        return observed

    monkeypatch.setattr(metering, "_MAX_EVENT_SNAPSHOT_NODES", 8)
    monkeypatch.setattr(metering, "len", racing_len, raising=False)

    copy_calls = _profile_snapshot_copy_calls(values)

    assert raced
    assert real_len(values) == 10
    assert "dict.copy" not in copy_calls
