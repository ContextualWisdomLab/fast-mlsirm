"""Provider-neutral usage export helpers for simulation and fitting results."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from .types import FitResult, SimulationData


class ComputeUsageEventBuilder(Protocol):
    """Version-one builder boundary supplied by the metering producer SDK."""

    def __call__(self, **payload: Any) -> Mapping[str, Any]: ...


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


class CanonicalComputeUsageSink:
    """Build, validate, and enqueue count-only events for real compute results."""

    def __init__(
        self,
        *,
        event_builder: ComputeUsageEventBuilder,
        event_validator: ComputeUsageEventValidator,
        enqueue: Callable[[Mapping[str, Any]], None],
        identity: Mapping[str, str | None],
    ) -> None:
        """Store producer-owned build/validation boundaries and durable enqueue."""
        try:
            identity_snapshot = dict(identity)
        except Exception:
            raise ValueError("identity could not be read safely") from None
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

    def _validate_and_enqueue(self, event: Mapping[str, Any]) -> None:
        """Fail closed unless the producer output is a valid canonical v1 event."""
        if event.get("event_contract_version") != 1:
            raise ValueError("event_builder must return event_contract_version=1")
        validation_errors = self._event_validator(event)
        if validation_errors:
            detail = "; ".join(str(error) for error in validation_errors[:3])
            raise ValueError(
                "event_builder output violates canonical usage-event v1 contract"
                + (f": {detail}" if detail else "")
            )
        self._enqueue(event)

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
        }
        if project_reference is not None:
            payload["project_reference"] = project_reference
        if artifact_bytes is not None:
            payload["artifact_bytes"] = artifact_bytes
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
        }
        if project_reference is not None:
            payload["project_reference"] = project_reference
        if artifact_bytes is not None:
            payload["artifact_bytes"] = artifact_bytes
        self._validate_and_enqueue(self._event_builder(**payload))


__all__ = ["CanonicalComputeUsageSink"]