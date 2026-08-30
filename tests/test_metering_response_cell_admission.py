"""Fail-first response-cell resource replay for compute metering."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fast_mlsirm.irt_contract import MAX_IRT_RESPONSE_CELLS
from fast_mlsirm.metering import CanonicalComputeUsageSink


def _sink(calls: dict[str, int]) -> CanonicalComputeUsageSink:
    def builder(**_: object) -> dict[str, object]:
        calls["producer"] += 1
        return {"event_contract_version": 1}

    def validator(_: object) -> tuple[str, ...]:
        calls["validator"] += 1
        return ()

    def enqueue(_: object) -> None:
        calls["enqueue"] += 1

    return CanonicalComputeUsageSink(
        event_builder=builder,
        event_validator=validator,
        enqueue=enqueue,
        identity={},
    )


@pytest.mark.parametrize("kind", ("fit", "simulation"))
def test_metering_rejects_response_cell_overflow_before_producer(kind: str) -> None:
    """Metering must replay the package response-cell ceiling before producer work."""
    calls = {"producer": 0, "validator": 0, "enqueue": 0}
    sink = _sink(calls)
    rows = MAX_IRT_RESPONSE_CELLS + 1

    with pytest.raises(ValueError, match="response cell count exceeds package limit"):
        if kind == "fit":
            sink.emit_fit(
                SimpleNamespace(model="MLS2PLM", backend="rust"),
                run_reference="run",
                artifact_reference="artifact",
                configuration_reference="config",
                seed_reference="seed",
                occurred_at="2026-08-30T00:00:00Z",
                response_rows=rows,
                response_items=1,
            )
        else:
            sink.emit_simulation(
                SimpleNamespace(Y=SimpleNamespace(shape=(rows, 1))),
                run_reference="run",
                artifact_reference="artifact",
                configuration_reference="config",
                seed_reference="seed",
                occurred_at="2026-08-30T00:00:00Z",
            )

    assert calls == {"producer": 0, "validator": 0, "enqueue": 0}


@pytest.mark.parametrize("kind", ("fit", "simulation"))
def test_metering_admits_exact_response_cell_boundary(kind: str) -> None:
    """The existing package response-cell boundary remains admissible."""
    calls = {"producer": 0, "validator": 0, "enqueue": 0}
    sink = _sink(calls)

    if kind == "fit":
        sink.emit_fit(
            SimpleNamespace(model="MLS2PLM", backend="rust"),
            run_reference="run",
            artifact_reference="artifact",
            configuration_reference="config",
            seed_reference="seed",
            occurred_at="2026-08-30T00:00:00Z",
            response_rows=MAX_IRT_RESPONSE_CELLS,
            response_items=1,
        )
    else:
        sink.emit_simulation(
            SimpleNamespace(
                Y=SimpleNamespace(shape=(MAX_IRT_RESPONSE_CELLS, 1))
            ),
            run_reference="run",
            artifact_reference="artifact",
            configuration_reference="config",
            seed_reference="seed",
            occurred_at="2026-08-30T00:00:00Z",
        )

    assert calls == {"producer": 1, "validator": 1, "enqueue": 1}
