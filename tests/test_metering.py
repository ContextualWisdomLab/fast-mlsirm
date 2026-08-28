"""Canonical compute usage export tests."""

from __future__ import annotations

from types import SimpleNamespace

from fast_mlsirm.config import MLS2PLMConfig
from fast_mlsirm.metering import CanonicalComputeUsageSink
from fast_mlsirm.simulation import simulate


def _sink(queued: list[dict[str, object]]) -> CanonicalComputeUsageSink:
    """Build a sink with anonymized identity references for tests."""

    def builder(**payload: object) -> dict[str, object]:
        return {"event_contract_version": 1, **payload}

    return CanonicalComputeUsageSink(
        event_builder=builder,
        enqueue=queued.append,
        identity={
            "tenant_reference": "urn:cwl:tenant:test",
            "billing_account_reference": "urn:cwl:tenant:test:account:1",
            "billing_principal_reference": "urn:cwl:tenant:test:principal:1",
        },
    )


def test_simulation_export_uses_real_response_shape_without_content() -> None:
    """A real simulation emits only bounded shape/provenance metadata."""
    data = simulate(MLS2PLMConfig(n_persons=3, n_dims=1, items_per_dim=2, seed=7))
    queued: list[dict[str, object]] = []
    _sink(queued).emit_simulation(
        data,
        run_reference="urn:cwl:run:simulation-7",
        artifact_reference="urn:cwl:artifact:simulation-7",
        configuration_reference="urn:cwl:config:simulation-7",
        seed_reference="urn:cwl:seed:7",
        occurred_at="2026-08-28T00:00:00Z",
    )

    event = queued[0]
    assert event["response_rows"] == 3
    assert event["response_items"] == 2
    assert event["seed_reference"] == "urn:cwl:seed:7"
    assert "responses" not in event


def test_fit_export_uses_result_backend_and_explicit_shape() -> None:
    """Fit export keeps backend identity and caller-provided input shape."""
    queued: list[dict[str, object]] = []
    result = SimpleNamespace(model="MLS2PLM", backend="rust")
    _sink(queued).emit_fit(
        result,
        run_reference="urn:cwl:run:fit-7",
        artifact_reference="urn:cwl:artifact:fit-7",
        configuration_reference="urn:cwl:config:fit-7",
        seed_reference="urn:cwl:seed:7",
        occurred_at="2026-08-28T00:00:00Z",
        response_rows=3,
        response_items=2,
        artifact_bytes=128,
    )

    assert queued[0]["model_code"] == "mls2plm"
    assert queued[0]["backend_code"] == "rust"
    assert queued[0]["artifact_bytes"] == 128


def test_identity_cannot_override_versioned_event_fields() -> None:
    """Reserved event fields stay under the sink's explicit authority."""
    try:
        CanonicalComputeUsageSink(
            event_builder=lambda **payload: {"event_contract_version": 1, **payload},
            enqueue=lambda _: None,
            identity={"run_reference": "must-not-override"},
        )
    except ValueError as error:
        assert "run_reference" in str(error)
    else:
        raise AssertionError("reserved identity field was accepted")


def test_sink_rejects_non_v1_builder_output() -> None:
    """A producer cannot enqueue an event from a different contract version."""
    queued: list[dict[str, object]] = []
    sink = CanonicalComputeUsageSink(
        event_builder=lambda **payload: {"event_contract_version": 2, **payload},
        enqueue=queued.append,
        identity={},
    )
    fake_data = SimpleNamespace(Y=SimpleNamespace(shape=(1, 1)))
    try:
        sink.emit_simulation(
            fake_data,
            run_reference="run",
            artifact_reference="artifact",
            configuration_reference="config",
            seed_reference="seed",
            occurred_at="2026-08-28T00:00:00Z",
        )
    except ValueError as error:
        assert "event_contract_version=1" in str(error)
    else:
        raise AssertionError("non-v1 event was accepted")
    try:
        sink.emit_fit(
            SimpleNamespace(model="MLS2PLM", backend="rust"),
            run_reference="run",
            artifact_reference="artifact",
            configuration_reference="config",
            seed_reference="seed",
            occurred_at="2026-08-28T00:00:00Z",
            response_rows=1,
            response_items=1,
        )
    except ValueError as error:
        assert "event_contract_version=1" in str(error)
    else:
        raise AssertionError("non-v1 fit event was accepted")
    assert queued == []
