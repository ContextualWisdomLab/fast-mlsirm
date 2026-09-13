"""Fail-first source-event identity admission for canonical metering export."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fast_mlsirm.metering import CanonicalComputeUsageSink


_CANONICAL_SOURCE_EVENT_KEY_MAX_CHARS = 256


def _sink_that_must_not_build() -> CanonicalComputeUsageSink:
    """Return a sink whose producer callback proves preflight ordering."""

    def forbidden_builder(**_: object) -> dict[str, object]:
        raise AssertionError("producer builder ran before run-reference admission")

    return CanonicalComputeUsageSink(
        event_builder=forbidden_builder,
        event_validator=lambda _: (),
        enqueue=lambda _: None,
        identity={},
    )


def _fit_payload(run_reference: str) -> dict[str, object]:
    return {
        "run_reference": run_reference,
        "artifact_reference": "artifact",
        "configuration_reference": "config",
        "seed_reference": "seed",
        "occurred_at": "2026-08-30T00:00:00Z",
        "response_rows": 1,
        "response_items": 1,
    }


@pytest.mark.parametrize(
    "run_reference",
    ["", "r" * (_CANONICAL_SOURCE_EVENT_KEY_MAX_CHARS + 1)],
)
def test_fit_rejects_unrepresentable_run_reference_before_producer(
    run_reference: str,
) -> None:
    """Canonical source-event identity bounds apply before fit producer work."""
    with pytest.raises(ValueError, match="run_reference"):
        _sink_that_must_not_build().emit_fit(
            SimpleNamespace(model="MLS2PLM", backend="rust"),
            **_fit_payload(run_reference),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "run_reference",
    ["", "r" * (_CANONICAL_SOURCE_EVENT_KEY_MAX_CHARS + 1)],
)
def test_simulation_rejects_unrepresentable_run_reference_before_producer(
    run_reference: str,
) -> None:
    """Canonical source-event identity bounds apply before simulation producer work."""
    with pytest.raises(ValueError, match="run_reference"):
        _sink_that_must_not_build().emit_simulation(
            SimpleNamespace(Y=SimpleNamespace(shape=(1, 1))),
            run_reference=run_reference,
            artifact_reference="artifact",
            configuration_reference="config",
            seed_reference="seed",
            occurred_at="2026-08-30T00:00:00Z",
        )


def test_exact_source_event_key_width_boundary_reaches_producer() -> None:
    """Exactly 256 characters remain representable by usage-event/v1."""
    built_payloads: list[dict[str, object]] = []
    queued: list[dict[str, object]] = []

    def builder(**payload: object) -> dict[str, object]:
        built_payloads.append(dict(payload))
        return {"event_contract_version": 1}

    sink = CanonicalComputeUsageSink(
        event_builder=builder,
        event_validator=lambda _: (),
        enqueue=queued.append,
        identity={},
    )
    run_reference = "r" * _CANONICAL_SOURCE_EVENT_KEY_MAX_CHARS
    sink.emit_fit(
        SimpleNamespace(model="MLS2PLM", backend="rust"),
        **_fit_payload(run_reference),  # type: ignore[arg-type]
    )

    assert built_payloads[0]["run_reference"] == run_reference
    assert queued == [{"event_contract_version": 1}]
