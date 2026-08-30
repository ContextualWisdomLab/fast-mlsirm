"""Fail-first artifact-byte quantity admission tests for metering."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fast_mlsirm.metering import CanonicalComputeUsageSink


_CANONICAL_V1_QUANTITY_DIGITS = 39
_MAX_CANONICAL_V1_QUANTITY = 10**_CANONICAL_V1_QUANTITY_DIGITS - 1


def _sink_that_must_not_build() -> CanonicalComputeUsageSink:
    """Return a sink whose producer callback proves artifact-size preflight."""

    def forbidden_builder(**_: object) -> dict[str, object]:
        raise AssertionError("producer builder ran before artifact-byte admission")

    return CanonicalComputeUsageSink(
        event_builder=forbidden_builder,
        event_validator=lambda _: (),
        enqueue=lambda _: None,
        identity={},
    )


def _common_payload(artifact_bytes: int) -> dict[str, object]:
    """Return otherwise-valid emission metadata with one artifact byte count."""
    return {
        "run_reference": "urn:cwl:run:artifact-quantity",
        "artifact_reference": "urn:cwl:artifact:artifact-quantity",
        "configuration_reference": "urn:cwl:config:artifact-quantity",
        "seed_reference": "urn:cwl:seed:artifact-quantity",
        "occurred_at": "2026-08-30T00:00:00Z",
        "artifact_bytes": artifact_bytes,
    }


def _emit_fit(sink: CanonicalComputeUsageSink, artifact_bytes: int) -> None:
    """Emit one otherwise-valid fit observation."""
    sink.emit_fit(
        SimpleNamespace(model="MLS2PLM", backend="rust"),
        response_rows=3,
        response_items=2,
        **_common_payload(artifact_bytes),  # type: ignore[arg-type]
    )


def _emit_simulation(sink: CanonicalComputeUsageSink, artifact_bytes: int) -> None:
    """Emit one otherwise-valid simulation observation."""
    sink.emit_simulation(
        SimpleNamespace(Y=SimpleNamespace(shape=(3, 2))),
        **_common_payload(artifact_bytes),  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("emit", [_emit_fit, _emit_simulation])
def test_artifact_bytes_over_canonical_quantity_width_fails_before_producer(
    emit,
) -> None:
    """A 40-digit artifact count cannot reach canonical-v1 producer work."""
    with pytest.raises(ValueError, match="artifact_bytes"):
        emit(_sink_that_must_not_build(), 10**_CANONICAL_V1_QUANTITY_DIGITS)


@pytest.mark.parametrize("emit", [_emit_fit, _emit_simulation])
def test_artifact_bytes_exact_canonical_quantity_boundary_is_admitted(emit) -> None:
    """The exact 39-digit canonical-v1 quantity boundary remains admissible."""
    observed_payloads: list[dict[str, object]] = []
    queued: list[object] = []

    def builder(**payload: object) -> dict[str, object]:
        observed_payloads.append(dict(payload))
        return {"event_contract_version": 1}

    sink = CanonicalComputeUsageSink(
        event_builder=builder,
        event_validator=lambda _: (),
        enqueue=queued.append,
        identity={},
    )

    emit(sink, _MAX_CANONICAL_V1_QUANTITY)

    assert len(observed_payloads) == 1
    assert observed_payloads[0]["artifact_bytes"] == _MAX_CANONICAL_V1_QUANTITY
    assert queued == [{"event_contract_version": 1}]
