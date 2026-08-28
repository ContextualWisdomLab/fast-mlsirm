"""Fail-first admission tests for the count-only metering producer boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fast_mlsirm.metering import CanonicalComputeUsageSink


def _sink_that_must_not_build() -> CanonicalComputeUsageSink:
    """Return a sink whose producer callback proves preflight ordering."""

    def forbidden_builder(**_: object) -> dict[str, object]:
        raise AssertionError("producer builder ran before input admission")

    return CanonicalComputeUsageSink(
        event_builder=forbidden_builder,
        event_validator=lambda _: (),
        enqueue=lambda _: None,
        identity={},
    )


def _valid_fit_payload() -> dict[str, object]:
    """Return one otherwise-valid fit emission payload."""
    return {
        "run_reference": "urn:cwl:run:fit-7",
        "artifact_reference": "urn:cwl:artifact:fit-7",
        "configuration_reference": "urn:cwl:config:fit-7",
        "seed_reference": "urn:cwl:seed:7",
        "occurred_at": "2026-08-28T00:00:00Z",
        "response_rows": 3,
        "response_items": 2,
    }


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("run_reference", object()),
        ("artifact_reference", object()),
        ("configuration_reference", object()),
        ("seed_reference", object()),
        ("occurred_at", object()),
        ("project_reference", object()),
    ],
)
def test_fit_rejects_non_string_metadata_before_producer(
    field: str, bad_value: object
) -> None:
    """Callback-bearing metadata objects never cross the producer boundary."""
    payload = _valid_fit_payload()
    payload[field] = bad_value

    with pytest.raises(ValueError, match=field):
        _sink_that_must_not_build().emit_fit(
            SimpleNamespace(model="MLS2PLM", backend="rust"),
            **payload,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("response_rows", -1),
        ("response_rows", True),
        ("response_items", -1),
        ("response_items", False),
        ("artifact_bytes", -1),
        ("artifact_bytes", True),
        ("artifact_bytes", 1.0),
    ],
)
def test_fit_rejects_invalid_counts_before_producer(
    field: str, bad_value: object
) -> None:
    """Usage quantities are exact non-negative integers before producer work."""
    payload = _valid_fit_payload()
    payload[field] = bad_value

    with pytest.raises(ValueError, match=field):
        _sink_that_must_not_build().emit_fit(
            SimpleNamespace(model="MLS2PLM", backend="rust"),
            **payload,  # type: ignore[arg-type]
        )


def test_builder_result_rejects_boolean_contract_version_before_validator() -> None:
    """Boolean equality cannot impersonate the exact usage-event version integer."""
    validator_calls = 0
    queued: list[object] = []

    def builder(**_: object) -> dict[str, object]:
        return {"event_contract_version": True}

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

    with pytest.raises(ValueError, match="event_contract_version=1"):
        sink.emit_fit(
            SimpleNamespace(model="MLS2PLM", backend="rust"),
            **_valid_fit_payload(),  # type: ignore[arg-type]
        )

    assert validator_calls == 0
    assert queued == []


def test_builder_result_rejects_callback_bearing_contract_version_without_comparison() -> None:
    """Version admission uses type identity before equality can execute callbacks."""
    callbacks = 0
    validator_calls = 0
    queued: list[object] = []

    class HostileVersion(int):
        def __eq__(self, other: object) -> bool:
            nonlocal callbacks
            callbacks += 1
            raise AssertionError(f"version equality callback executed for {other!r}")

        def __ne__(self, other: object) -> bool:
            nonlocal callbacks
            callbacks += 1
            raise AssertionError(f"version inequality callback executed for {other!r}")

    version = HostileVersion(1)

    def builder(**_: object) -> dict[str, object]:
        return {"event_contract_version": version}

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

    with pytest.raises(ValueError, match="event_contract_version=1"):
        sink.emit_fit(
            SimpleNamespace(model="MLS2PLM", backend="rust"),
            **_valid_fit_payload(),  # type: ignore[arg-type]
        )

    assert callbacks == 0
    assert validator_calls == 0
    assert queued == []


def test_builder_result_rejects_callback_bearing_key_before_version_lookup() -> None:
    """Exact-dict event keys are inert before package-owned version lookup."""
    callbacks = 0
    validator_calls = 0
    queued: list[object] = []

    class HostileKey(str):
        def __hash__(self) -> int:
            nonlocal callbacks
            callbacks += 1
            return str.__hash__(self)

        def __eq__(self, other: object) -> bool:
            nonlocal callbacks
            callbacks += 1
            raise AssertionError(f"event-key comparison executed for {other!r}")

    key = HostileKey("event_contract_version")
    event = {key: 1}
    callbacks = 0

    def builder(**_: object) -> dict[str, object]:
        return event  # type: ignore[return-value]

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

    with pytest.raises(ValueError, match="event_builder result keys must be exact strings"):
        sink.emit_fit(
            SimpleNamespace(model="MLS2PLM", backend="rust"),
            **_valid_fit_payload(),  # type: ignore[arg-type]
        )

    assert callbacks == 0
    assert validator_calls == 0
    assert queued == []


def test_validator_result_rejects_callback_bearing_carrier_without_observation() -> None:
    """Validator result carriers are sealed after the callback returns."""
    callbacks = 0
    queued: list[object] = []

    class HostileErrors:
        def __bool__(self) -> bool:
            nonlocal callbacks
            callbacks += 1
            raise AssertionError("validator-result truthiness callback executed")

        def __getitem__(self, key: object) -> object:
            nonlocal callbacks
            callbacks += 1
            raise AssertionError(f"validator-result item callback executed for {key!r}")

        def __iter__(self):
            nonlocal callbacks
            callbacks += 1
            raise AssertionError("validator-result iteration callback executed")

    def builder(**_: object) -> dict[str, object]:
        return {"event_contract_version": 1}

    def validator(_: object) -> object:
        return HostileErrors()

    sink = CanonicalComputeUsageSink(
        event_builder=builder,
        event_validator=validator,  # type: ignore[arg-type]
        enqueue=queued.append,
        identity={},
    )

    with pytest.raises(ValueError, match="event_validator must return an exact tuple"):
        sink.emit_fit(
            SimpleNamespace(model="MLS2PLM", backend="rust"),
            **_valid_fit_payload(),  # type: ignore[arg-type]
        )

    assert callbacks == 0
    assert queued == []


def test_validator_result_rejects_callback_bearing_error_without_conversion() -> None:
    """Validator error entries are exact strings before diagnostic formatting."""
    callbacks = 0
    queued: list[object] = []

    class HostileError(str):
        def __str__(self) -> str:
            nonlocal callbacks
            callbacks += 1
            raise AssertionError("validator-error string callback executed")

    def builder(**_: object) -> dict[str, object]:
        return {"event_contract_version": 1}

    def validator(_: object) -> tuple[object, ...]:
        return (HostileError("bad event"),)

    sink = CanonicalComputeUsageSink(
        event_builder=builder,
        event_validator=validator,  # type: ignore[arg-type]
        enqueue=queued.append,
        identity={},
    )

    with pytest.raises(ValueError, match="event_validator errors must be exact strings"):
        sink.emit_fit(
            SimpleNamespace(model="MLS2PLM", backend="rust"),
            **_valid_fit_payload(),  # type: ignore[arg-type]
        )

    assert callbacks == 0
    assert queued == []
