"""Provider-neutral usage export helpers for simulation and fitting results."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from itertools import islice
from math import isfinite
from typing import Any, Protocol

from .types import FitResult, SimulationData


class ComputeUsageEventBuilder(Protocol):
    """Version-one builder boundary supplied by the metering producer SDK."""

    def __call__(self, **payload: Any) -> dict[str, Any]: ...


class ComputeUsageEventValidator(Protocol):
    """Canonical schema validator supplied by the metering producer contract."""

    def __call__(self, event: Any) -> tuple[str, ...]: ...


_CANONICAL_IDENTITY_FIELDS = frozenset(
    {
        "tenant_reference",
        "billing_account_reference",
        "billing_principal_reference",
        "credential_reference",
        "cost_center_reference",
    }
)
_MAX_EVENT_SNAPSHOT_DEPTH = 16
_MAX_EVENT_SNAPSHOT_NODES = 4096
_MAX_VALIDATOR_DIAGNOSTICS = 256


def _exact_string(name: str, value: object, *, optional: bool = False) -> str | None:
    """Admit one callback-free string reference before producer code can observe it."""
    if value is None and optional:
        return None
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    return value


def _nonnegative_int(
    name: str, value: object, *, optional: bool = False
) -> int | None:
    """Admit one exact non-negative count without numeric coercion callbacks."""
    if value is None and optional:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be an exact non-negative integer")
    return value


def _exact_2d_shape(value: object) -> tuple[int, int]:
    """Admit one inert two-dimensional response shape as count metadata."""
    if type(value) is not tuple or len(value) != 2:
        raise ValueError("response shape must be an exact 2-D shape")
    response_rows = _nonnegative_int("response_rows", value[0])
    response_items = _nonnegative_int("response_items", value[1])
    assert response_rows is not None and response_items is not None
    return response_rows, response_items


def _bounded_exact_dict_items(
    value: dict[Any, Any],
    *,
    max_items: int,
    too_large_message: str,
    changed_message: str,
) -> tuple[tuple[Any, Any], ...]:
    """Freeze at most ``max_items + 1`` exact-dict entries without a full copy."""
    try:
        items_snapshot = tuple(islice(value.items(), max_items + 1))
    except RuntimeError as exc:
        raise ValueError(changed_message) from exc
    if len(items_snapshot) > max_items:
        raise ValueError(too_large_message)
    return items_snapshot


def _snapshot_exact_json(
    value: object,
    *,
    depth: int = 0,
    remaining_nodes: list[int] | None = None,
) -> Any:
    """Copy one bounded exact-JSON tree without invoking caller protocols."""
    if depth > _MAX_EVENT_SNAPSHOT_DEPTH:
        raise ValueError("event_builder result exact JSON nesting is too deep")
    if remaining_nodes is None:
        remaining_nodes = [_MAX_EVENT_SNAPSHOT_NODES]
    remaining_nodes[0] -= 1
    if remaining_nodes[0] < 0:
        raise ValueError("event_builder result exact JSON tree is too large")

    value_type = type(value)
    if value_type is float:
        if not isfinite(value):
            raise ValueError("event_builder result exact JSON floats must be finite")
        return value
    if value is None or value_type in (str, int, bool):
        return value
    if value_type is list:
        if len(value) > remaining_nodes[0]:
            raise ValueError("event_builder result exact JSON tree is too large")
        value_snapshot = value[: remaining_nodes[0] + 1]
        if len(value_snapshot) > remaining_nodes[0]:
            raise ValueError("event_builder result exact JSON tree is too large")
        return [
            _snapshot_exact_json(
                item,
                depth=depth + 1,
                remaining_nodes=remaining_nodes,
            )
            for item in value_snapshot
        ]
    if value_type is dict:
        if len(value) > remaining_nodes[0]:
            raise ValueError("event_builder result exact JSON tree is too large")
        item_snapshot = _bounded_exact_dict_items(
            value,
            max_items=remaining_nodes[0],
            too_large_message="event_builder result exact JSON tree is too large",
            changed_message="event_builder result changed during exact JSON snapshot",
        )
        if any(type(key) is not str for key, _ in item_snapshot):
            raise ValueError("event_builder result exact JSON keys must be exact strings")
        return {
            key: _snapshot_exact_json(
                item,
                depth=depth + 1,
                remaining_nodes=remaining_nodes,
            )
            for key, item in item_snapshot
        }
    raise ValueError("event_builder result must use exact JSON carriers and scalars")


class CanonicalComputeUsageSink:
    """Build, validate, and enqueue count-only events for real compute results."""

    def __init__(
        self,
        *,
        event_builder: ComputeUsageEventBuilder,
        event_validator: ComputeUsageEventValidator,
        enqueue: Callable[[Mapping[str, Any]], None],
        identity: dict[str, str | None],
    ) -> None:
        """Store producer-owned build/validation boundaries and durable enqueue."""
        if type(identity) is not dict:
            raise ValueError("identity must be an exact dict")
        if len(identity) > len(_CANONICAL_IDENTITY_FIELDS):
            raise ValueError("identity contains too many fields")
        identity_items = _bounded_exact_dict_items(
            identity,
            max_items=len(_CANONICAL_IDENTITY_FIELDS),
            too_large_message="identity contains too many fields",
            changed_message="identity changed during snapshot",
        )
        if any(type(key) is not str for key, _ in identity_items):
            raise ValueError("identity keys must be exact strings")
        identity_snapshot = {key: value for key, value in identity_items}
        unexpected = set(identity_snapshot).difference(_CANONICAL_IDENTITY_FIELDS)
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise ValueError(f"identity contains noncanonical fields: {names}")
        invalid_values = sorted(
            key
            for key, value in identity_snapshot.items()
            if value is not None and type(value) is not str
        )
        if invalid_values:
            names = ", ".join(invalid_values)
            raise ValueError(f"identity references must be exact strings or None: {names}")
        self._event_builder = event_builder
        self._event_validator = event_validator
        self._enqueue = enqueue
        self._identity = {
            key: value for key, value in identity_snapshot.items() if value is not None
        }

    def _validate_and_enqueue(self, event: object) -> None:
        """Fail closed unless the producer output is a valid canonical v1 event."""
        if type(event) is not dict:
            raise ValueError("event_builder must return an exact dict")
        if len(event) >= _MAX_EVENT_SNAPSHOT_NODES:
            raise ValueError("event_builder result exact JSON tree is too large")
        producer_items = _bounded_exact_dict_items(
            event,
            max_items=_MAX_EVENT_SNAPSHOT_NODES - 1,
            too_large_message="event_builder result exact JSON tree is too large",
            changed_message="event_builder result changed during exact JSON snapshot",
        )
        if any(type(key) is not str for key, _ in producer_items):
            raise ValueError("event_builder result keys must be exact strings")
        producer_event = {key: value for key, value in producer_items}
        producer_contract_version = producer_event.get("event_contract_version")
        if type(producer_contract_version) is not int:
            raise ValueError("event_builder must return event_contract_version=1")
        validation_event = _snapshot_exact_json(producer_event)
        contract_version = validation_event.get("event_contract_version")
        if type(contract_version) is not int or contract_version != 1:
            raise ValueError("event_builder must return event_contract_version=1")
        enqueue_event = _snapshot_exact_json(validation_event)
        validation_errors = self._event_validator(validation_event)
        if type(validation_errors) is not tuple:
            raise ValueError("event_validator must return an exact tuple")
        if len(validation_errors) > _MAX_VALIDATOR_DIAGNOSTICS:
            raise ValueError("event_validator returned too many diagnostics")
        if any(type(error) is not str for error in validation_errors):
            raise ValueError("event_validator errors must be exact strings")
        if validation_errors:
            detail = "; ".join(validation_errors[:3])
            raise ValueError(
                "event_builder output violates canonical usage-event v1 contract"
                + (f": {detail}" if detail else "")
            )
        self._enqueue(enqueue_event)

    def emit_simulation(
        self,
        data: SimulationData,
        *,
        run_reference: str,
        artifact_reference: str,
        configuration_reference: str,
        seed_reference: str,
        occurred_at: str,
        project_reference: str | None = None,
        artifact_bytes: int | None = None,
    ) -> None:
        """Export one simulation's response-cell and artifact counts."""
        admitted_run_reference = _exact_string("run_reference", run_reference)
        admitted_artifact_reference = _exact_string(
            "artifact_reference", artifact_reference
        )
        admitted_configuration_reference = _exact_string(
            "configuration_reference", configuration_reference
        )
        admitted_seed_reference = _exact_string("seed_reference", seed_reference)
        admitted_occurred_at = _exact_string("occurred_at", occurred_at)
        admitted_project_reference = _exact_string(
            "project_reference", project_reference, optional=True
        )
        admitted_artifact_bytes = _nonnegative_int(
            "artifact_bytes", artifact_bytes, optional=True
        )
        response_shape = data.Y.shape
        response_rows, response_items = _exact_2d_shape(response_shape)
        payload: dict[str, Any] = {
            **self._identity,
            "run_reference": admitted_run_reference,
            "artifact_reference": admitted_artifact_reference,
            "configuration_reference": admitted_configuration_reference,
            "seed_reference": admitted_seed_reference,
            "model_code": "mls2plm",
            "backend_code": "numpy",
            "occurred_at": admitted_occurred_at,
            "response_rows": response_rows,
            "response_items": response_items,
        }
        if admitted_project_reference is not None:
            payload["project_reference"] = admitted_project_reference
        if admitted_artifact_bytes is not None:
            payload["artifact_bytes"] = admitted_artifact_bytes
        self._validate_and_enqueue(self._event_builder(**payload))

    def emit_fit(
        self,
        result: FitResult,
        *,
        run_reference: str,
        artifact_reference: str,
        configuration_reference: str,
        seed_reference: str,
        occurred_at: str,
        response_rows: int,
        response_items: int,
        project_reference: str | None = None,
        artifact_bytes: int | None = None,
    ) -> None:
        """Export one fit's response-cell and artifact counts."""
        admitted_run_reference = _exact_string("run_reference", run_reference)
        admitted_artifact_reference = _exact_string(
            "artifact_reference", artifact_reference
        )
        admitted_configuration_reference = _exact_string(
            "configuration_reference", configuration_reference
        )
        admitted_seed_reference = _exact_string("seed_reference", seed_reference)
        admitted_occurred_at = _exact_string("occurred_at", occurred_at)
        admitted_project_reference = _exact_string(
            "project_reference", project_reference, optional=True
        )
        admitted_response_rows = _nonnegative_int("response_rows", response_rows)
        admitted_response_items = _nonnegative_int("response_items", response_items)
        admitted_artifact_bytes = _nonnegative_int(
            "artifact_bytes", artifact_bytes, optional=True
        )
        model = result.model
        backend = result.backend
        if type(model) is not str:
            raise ValueError("result.model must be an exact string")
        if type(backend) is not str:
            raise ValueError("result.backend must be an exact string")
        payload: dict[str, Any] = {
            **self._identity,
            "run_reference": admitted_run_reference,
            "artifact_reference": admitted_artifact_reference,
            "configuration_reference": admitted_configuration_reference,
            "seed_reference": admitted_seed_reference,
            "model_code": model.lower(),
            "backend_code": backend.lower(),
            "occurred_at": admitted_occurred_at,
            "response_rows": admitted_response_rows,
            "response_items": admitted_response_items,
        }
        if admitted_project_reference is not None:
            payload["project_reference"] = admitted_project_reference
        if admitted_artifact_bytes is not None:
            payload["artifact_bytes"] = admitted_artifact_bytes
        self._validate_and_enqueue(self._event_builder(**payload))


__all__ = ["CanonicalComputeUsageSink"]