"""Fail-first fit model/backend admission regressions for compute metering."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fast_mlsirm.metering import CanonicalComputeUsageSink


@pytest.mark.parametrize(
    ("model", "backend", "message"),
    (
        ("unsupported_model", "rust", "result.model"),
        ("MLS2PLM", "cuda", "result.backend"),
    ),
)
def test_fit_export_rejects_unsupported_codes_before_producer(
    model: str,
    backend: str,
    message: str,
) -> None:
    """Package-owned fit vocabularies must fail before producer observation."""
    producer_calls = 0
    validator_calls = 0
    enqueue_calls = 0

    def builder(**_: object) -> dict[str, object]:
        nonlocal producer_calls
        producer_calls += 1
        return {"event_contract_version": 1}

    def validator(_: object) -> tuple[str, ...]:
        nonlocal validator_calls
        validator_calls += 1
        return ()

    def enqueue(_: object) -> None:
        nonlocal enqueue_calls
        enqueue_calls += 1

    sink = CanonicalComputeUsageSink(
        event_builder=builder,
        event_validator=validator,
        enqueue=enqueue,
        identity={},
    )

    with pytest.raises(ValueError, match=message):
        sink.emit_fit(
            SimpleNamespace(model=model, backend=backend),
            run_reference="run",
            artifact_reference="artifact",
            configuration_reference="config",
            seed_reference="seed",
            occurred_at="2026-08-30T00:00:00Z",
            response_rows=1,
            response_items=1,
        )

    assert producer_calls == 0
    assert validator_calls == 0
    assert enqueue_calls == 0
