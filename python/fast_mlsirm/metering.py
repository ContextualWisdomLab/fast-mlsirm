"""Provider-neutral usage export helpers for simulation and fitting results."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from .types import FitResult, SimulationData


class ComputeUsageEventBuilder(Protocol):
    """Version-one builder boundary supplied by the metering SDK."""

    def __call__(self, **payload: Any) -> Mapping[str, Any]: ...


_RESERVED_EVENT_FIELDS = frozenset(
    {
        "run_reference",
        "artifact_reference",
        "configuration_reference",
        "seed_reference",
        "model_code",
        "backend_code",
        "occurred_at",
        "response_rows",
        "response_items",
        "artifact_bytes",
        "project_reference",
    }
)


class CanonicalComputeUsageSink:
    """Build and enqueue count-only events for real compute results."""

    def __init__(
        self,
        *,
        event_builder: ComputeUsageEventBuilder,
        enqueue: Callable[[Mapping[str, Any]], None],
        identity: Mapping[str, str | None],
    ) -> None:
        """Store the versioned builder, durable enqueue callback, and identity."""
        reserved = _RESERVED_EVENT_FIELDS.intersection(identity)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(f"identity contains reserved event fields: {names}")
        self._event_builder = event_builder
        self._enqueue = enqueue
        self._identity = dict(identity)

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
        payload: dict[str, Any] = {
            **self._identity,
            "run_reference": run_reference,
            "artifact_reference": artifact_reference,
            "configuration_reference": configuration_reference,
            "seed_reference": seed_reference,
            "model_code": "mls2plm",
            "backend_code": "numpy",
            "occurred_at": occurred_at,
            "response_rows": int(data.Y.shape[0]),
            "response_items": int(data.Y.shape[1]),
            "artifact_bytes": artifact_bytes,
        }
        if project_reference is not None:
            payload["project_reference"] = project_reference
        event = self._event_builder(**payload)
        if event.get("event_contract_version") != 1:
            raise ValueError("event_builder must return event_contract_version=1")
        self._enqueue(event)

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
        payload: dict[str, Any] = {
            **self._identity,
            "run_reference": run_reference,
            "artifact_reference": artifact_reference,
            "configuration_reference": configuration_reference,
            "seed_reference": seed_reference,
            "model_code": result.model.lower(),
            "backend_code": result.backend.lower(),
            "occurred_at": occurred_at,
            "response_rows": response_rows,
            "response_items": response_items,
            "artifact_bytes": artifact_bytes,
        }
        if project_reference is not None:
            payload["project_reference"] = project_reference
        event = self._event_builder(**payload)
        if event.get("event_contract_version") != 1:
            raise ValueError("event_builder must return event_contract_version=1")
        self._enqueue(event)


__all__ = ["CanonicalComputeUsageSink"]
