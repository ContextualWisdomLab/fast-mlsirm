"""Canonical compute usage export tests."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import SimpleNamespace

from fast_mlsirm.config import MLS2PLMConfig
from fast_mlsirm.metering import CanonicalComputeUsageSink
from fast_mlsirm.simulation import simulate


_REQUIRED_USAGE_EVENT_FIELDS = frozenset(
    {
        "event_id",
        "event_contract_version",
        "source_event_key",
        "source_payload_hash",
        "tenant_reference",
        "billing_account_reference",
        "billing_principal_reference",
        "product_code",
        "occurred_at",
        "measurements",
    }
)
_OPTIONAL_USAGE_EVENT_FIELDS = frozenset(
    {
        "credential_reference",
        "cost_center_reference",
        "project_reference",
        "operation_code",
        "recorded_at",
    }
)


def _validate_usage_event_v1(event: object) -> tuple[str, ...]:
    """Test-only replay of the released closed v1 usage-event envelope."""
    if type(event) is not dict:
        return ("$: usage event must be an object",)
    errors: list[str] = []
    keys = set(event)
    missing = _REQUIRED_USAGE_EVENT_FIELDS - keys
    unexpected = keys - (_REQUIRED_USAGE_EVENT_FIELDS | _OPTIONAL_USAGE_EVENT_FIELDS)
    if missing:
        errors.append(f"$: missing required fields: {', '.join(sorted(missing))}")
    if unexpected:
        errors.append(f"$: unexpected fields: {', '.join(sorted(unexpected))}")
    if event.get("event_contract_version") != 1:
        errors.append("$: event_contract_version must be 1")
    measurements = event.get("measurements")
    if not isinstance(measurements, list) or not measurements:
        errors.append("$: measurements must be a non-empty array")
    return tuple(errors)


def _canonical_builder(
    captured: dict[str, object] | None = None,
    *,
    contract_version: int = 1,
):
    """Return a deterministic stand-in for the released producer boundary."""

    def builder(**payload: object) -> dict[str, object]:
        if captured is not None:
            captured.update(payload)
        measurements = [
            {
                "meter_code": "response_rows",
                "quantity": str(payload["response_rows"]),
                "unit_code": "count",
                "quality_code": "deterministically_derived",
            },
            {
                "meter_code": "response_items",
                "quantity": str(payload["response_items"]),
                "unit_code": "count",
                "quality_code": "deterministically_derived",
            },
        ]
        event: dict[str, object] = {
            "event_id": "00000000-0000-4000-8000-000000000001",
            "event_contract_version": contract_version,
            "source_event_key": str(payload["run_reference"]),
            "source_payload_hash": f"sha256:{'0' * 64}",
            "tenant_reference": str(
                payload.get("tenant_reference", "urn:cwl:tenant:test")
            ),
            "billing_account_reference": str(
                payload.get(
                    "billing_account_reference",
                    "urn:cwl:tenant:test:account:1",
                )
            ),
            "billing_principal_reference": str(
                payload.get(
                    "billing_principal_reference",
                    "urn:cwl:tenant:test:principal:1",
                )
            ),
            "product_code": "fast_mlsirm",
            "operation_code": "compute_run",
            "occurred_at": payload["occurred_at"],
            "measurements": measurements,
        }
        project_reference = payload.get("project_reference")
        if project_reference is not None:
            event["project_reference"] = project_reference
        return event

    return builder


def _sink(
    queued: list[dict[str, object]],
    captured: dict[str, object] | None = None,
) -> CanonicalComputeUsageSink:
    """Build a sink with anonymized identity references for tests."""
    return CanonicalComputeUsageSink(
        event_builder=_canonical_builder(captured),
        event_validator=_validate_usage_event_v1,
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
    captured: dict[str, object] = {}
    _sink(queued, captured).emit_simulation(
        data,
        run_reference="urn:cwl:run:simulation-7",
        artifact_reference="urn:cwl:artifact:simulation-7",
        configuration_reference="urn:cwl:config:simulation-7",
        seed_reference="urn:cwl:seed:7",
        project_reference="urn:cwl:project:measurement",
        occurred_at="2026-08-28T00:00:00Z",
    )

    assert captured["response_rows"] == 3
    assert captured["response_items"] == 2
    assert captured["seed_reference"] == "urn:cwl:seed:7"
    assert captured["project_reference"] == "urn:cwl:project:measurement"
    assert "responses" not in captured
    assert _validate_usage_event_v1(queued[0]) == ()
    assert "responses" not in queued[0]


def test_fit_export_uses_result_backend_and_explicit_shape() -> None:
    """Fit export keeps backend identity and caller-provided input shape."""
    queued: list[dict[str, object]] = []
    captured: dict[str, object] = {}
    result = SimpleNamespace(model="MLS2PLM", backend="rust")
    _sink(queued, captured).emit_fit(
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

    assert captured["model_code"] == "mls2plm"
    assert captured["backend_code"] == "rust"
    assert captured["artifact_bytes"] == 128
    assert _validate_usage_event_v1(queued[0]) == ()


def test_identity_cannot_override_versioned_event_fields() -> None:
    """Reserved event fields stay under the sink's explicit authority."""
    try:
        CanonicalComputeUsageSink(
            event_builder=_canonical_builder(),
            event_validator=_validate_usage_event_v1,
            enqueue=lambda _: None,
            identity={"run_reference": "must-not-override"},
        )
    except ValueError as error:
        assert "run_reference" in str(error)
    else:
        raise AssertionError("reserved identity field was accepted")


def test_identity_rejects_callback_bearing_mapping_without_observation() -> None:
    """Identity carrier callbacks cannot execute before package admission."""
    callbacks = 0
    producer_calls = 0

    class HostileIdentity(Mapping[str, str]):
        def __getitem__(self, key: str) -> str:
            nonlocal callbacks
            callbacks += 1
            raise AssertionError(f"identity callback executed for {key}")

        def __iter__(self) -> Iterator[str]:
            nonlocal callbacks
            callbacks += 1
            raise AssertionError("identity iteration callback executed")

        def __len__(self) -> int:
            nonlocal callbacks
            callbacks += 1
            raise AssertionError("identity length callback executed")

    def builder(**payload: object) -> dict[str, object]:
        nonlocal producer_calls
        producer_calls += 1
        return _canonical_builder()(**payload)

    try:
        CanonicalComputeUsageSink(
            event_builder=builder,
            event_validator=_validate_usage_event_v1,
            enqueue=lambda _: None,
            identity=HostileIdentity(),
        )
    except ValueError as error:
        assert "identity must be an exact dict" in str(error)
    else:
        raise AssertionError("callback-bearing identity mapping was accepted")
    assert callbacks == 0
    assert producer_calls == 0


def test_identity_rejects_callback_bearing_key_without_comparison() -> None:
    """An exact dict cannot smuggle a protocol-bearing key into allowlist checks."""
    callbacks = 0

    class HostileKey(str):
        def __hash__(self) -> int:
            nonlocal callbacks
            callbacks += 1
            return str.__hash__(self)

        def __eq__(self, other: object) -> bool:
            nonlocal callbacks
            callbacks += 1
            return str.__eq__(self, other)

    key = HostileKey("tenant_reference")
    identity = {key: "urn:cwl:tenant:test"}
    callbacks = 0

    try:
        CanonicalComputeUsageSink(
            event_builder=_canonical_builder(),
            event_validator=_validate_usage_event_v1,
            enqueue=lambda _: None,
            identity=identity,  # type: ignore[arg-type]
        )
    except ValueError as error:
        assert "identity keys must be exact strings" in str(error)
    else:
        raise AssertionError("callback-bearing identity key was accepted")
    assert callbacks == 0


def test_identity_rejects_noncanonical_fields_before_producer_boundary() -> None:
    """Private content cannot enter the count-only producer payload as identity."""
    captured: dict[str, object] = {}
    try:
        CanonicalComputeUsageSink(
            event_builder=_canonical_builder(captured),
            event_validator=_validate_usage_event_v1,
            enqueue=lambda _: None,
            identity={"response_text": "must-not-reach-producer"},
        )
    except ValueError as error:
        assert "response_text" in str(error)
    else:
        raise AssertionError("noncanonical identity field was accepted")
    assert captured == {}


def test_identity_rejects_non_string_reference_before_producer_boundary() -> None:
    """Allowed keys cannot smuggle arbitrary content objects to the producer."""
    captured: dict[str, object] = {}
    try:
        CanonicalComputeUsageSink(
            event_builder=_canonical_builder(captured),
            event_validator=_validate_usage_event_v1,
            enqueue=lambda _: None,
            identity={"tenant_reference": object()},  # type: ignore[dict-item]
        )
    except ValueError as error:
        assert "tenant_reference" in str(error)
    else:
        raise AssertionError("non-string identity reference was accepted")
    assert captured == {}


def test_none_identity_reference_is_omitted_from_producer_payload() -> None:
    """Unset optional identity references do not cross the producer boundary."""
    captured: dict[str, object] = {}
    sink = CanonicalComputeUsageSink(
        event_builder=_canonical_builder(captured),
        event_validator=_validate_usage_event_v1,
        enqueue=lambda _: None,
        identity={"credential_reference": None},
    )
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

    assert "credential_reference" not in captured


def test_optional_project_reference_is_omitted_when_unset() -> None:
    """Older builders do not receive omitted optional fields."""
    captured: dict[str, object] = {}
    sink = CanonicalComputeUsageSink(
        event_builder=_canonical_builder(captured),
        event_validator=_validate_usage_event_v1,
        enqueue=lambda _: None,
        identity={},
    )
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

    assert "project_reference" not in captured
    assert "artifact_bytes" not in captured


def test_sink_rejects_non_v1_builder_output() -> None:
    """A producer cannot enqueue an event from a different contract version."""
    queued: list[dict[str, object]] = []
    sink = CanonicalComputeUsageSink(
        event_builder=_canonical_builder(contract_version=2),
        event_validator=_validate_usage_event_v1,
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


def test_sink_rejects_same_version_event_that_fails_canonical_validation() -> None:
    """A version marker alone cannot bypass the canonical producer schema."""
    queued: list[dict[str, object]] = []
    canonical_builder = _canonical_builder()

    def invalid_builder(**payload: object) -> dict[str, object]:
        event = canonical_builder(**payload)
        event["response_rows"] = payload["response_rows"]
        return event

    sink = CanonicalComputeUsageSink(
        event_builder=invalid_builder,
        event_validator=_validate_usage_event_v1,
        enqueue=queued.append,
        identity={},
    )
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
        assert "canonical usage-event v1 contract" in str(error)
    else:
        raise AssertionError("schema-invalid v1 event was accepted")
    assert queued == []
