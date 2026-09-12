"""Fail-first tests for compute-result metadata snapshotting in metering."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fast_mlsirm.metering import CanonicalComputeUsageSink


def _capturing_sink(captured: dict[str, object]) -> CanonicalComputeUsageSink:
    """Return a sink that captures private producer metadata."""

    def builder(**payload: object) -> dict[str, object]:
        captured.update(payload)
        return {"event_contract_version": 1}

    return CanonicalComputeUsageSink(
        event_builder=builder,
        event_validator=lambda _: (),
        enqueue=lambda _: None,
        identity={},
    )


def _forbidden_sink() -> CanonicalComputeUsageSink:
    """Return a sink whose producer proves admission happened first."""

    def forbidden_builder(**_: object) -> dict[str, object]:
        raise AssertionError("producer builder ran before result metadata admission")

    return CanonicalComputeUsageSink(
        event_builder=forbidden_builder,
        event_validator=lambda _: (),
        enqueue=lambda _: None,
        identity={},
    )


def _simulation_kwargs() -> dict[str, str]:
    """Return ordinary simulation metering provenance."""
    return {
        "run_reference": "run",
        "artifact_reference": "artifact",
        "configuration_reference": "config",
        "seed_reference": "seed",
        "occurred_at": "2026-08-29T00:00:00Z",
    }


def _fit_kwargs() -> dict[str, object]:
    """Return ordinary fit metering provenance and counts."""
    return {
        **_simulation_kwargs(),
        "response_rows": 3,
        "response_items": 2,
    }


def test_simulation_shape_is_read_once_as_one_coherent_snapshot() -> None:
    """Row/item counts must come from the same observed response shape."""
    shape_reads = 0
    captured: dict[str, object] = {}

    class ChangingResponses:
        @property
        def shape(self) -> tuple[int, int]:
            nonlocal shape_reads
            shape_reads += 1
            if shape_reads == 1:
                return (3, 2)
            return (97, 89)

    data = SimpleNamespace(Y=ChangingResponses())
    _capturing_sink(captured).emit_simulation(data, **_simulation_kwargs())  # type: ignore[arg-type]

    assert shape_reads == 1
    assert captured["response_rows"] == 3
    assert captured["response_items"] == 2


def test_simulation_rejects_protocol_bearing_shape_before_indexing() -> None:
    """A shape subclass cannot execute item callbacks during count admission."""
    callbacks = 0

    class HostileShape(tuple):
        def __getitem__(self, key: object) -> object:
            nonlocal callbacks
            callbacks += 1
            raise AssertionError(f"shape item callback executed for {key!r}")

    data = SimpleNamespace(Y=SimpleNamespace(shape=HostileShape((3, 2))))

    with pytest.raises(ValueError, match="response shape must be an exact 2-D shape"):
        _forbidden_sink().emit_simulation(data, **_simulation_kwargs())  # type: ignore[arg-type]

    assert callbacks == 0


def test_simulation_rejects_wrong_length_exact_shape_before_producer() -> None:
    """Malformed exact tuple cardinality fails with a package-owned diagnostic."""
    data = SimpleNamespace(Y=SimpleNamespace(shape=(3,)))

    with pytest.raises(ValueError, match="response shape must be an exact 2-D shape"):
        _forbidden_sink().emit_simulation(data, **_simulation_kwargs())  # type: ignore[arg-type]


def test_fit_model_and_backend_are_each_read_once_before_normalization() -> None:
    """Validated fit metadata snapshots alone may cross the producer boundary."""
    model_reads = 0
    backend_reads = 0
    captured: dict[str, object] = {}

    class ChangingFitResult:
        @property
        def model(self) -> str:
            nonlocal model_reads
            model_reads += 1
            if model_reads == 1:
                return "MLS2PLM"
            return "MIRT"

        @property
        def backend(self) -> str:
            nonlocal backend_reads
            backend_reads += 1
            if backend_reads == 1:
                return "rust"
            return "numpy"

    _capturing_sink(captured).emit_fit(ChangingFitResult(), **_fit_kwargs())  # type: ignore[arg-type]

    assert model_reads == 1
    assert backend_reads == 1
    assert captured["model_code"] == "mls2plm"
    assert captured["backend_code"] == "rust"
