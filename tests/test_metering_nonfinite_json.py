"""Exact-JSON numeric admission regressions for metering producer output."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fast_mlsirm.metering import CanonicalComputeUsageSink


def _emit_one_fit(sink: CanonicalComputeUsageSink) -> None:
    """Exercise the fit emission boundary with inert package-owned evidence."""
    sink.emit_fit(
        SimpleNamespace(model="MLS2PLM", backend="rust"),
        run_reference="run",
        artifact_reference="artifact",
        configuration_reference="config",
        seed_reference="seed",
        occurred_at="2026-08-29T00:00:00Z",
        response_rows=1,
        response_items=1,
    )


@pytest.mark.parametrize(
    "nonfinite_value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_nonfinite_exact_json_float_is_rejected_before_validator(
    nonfinite_value: float,
) -> None:
    """Non-finite Python floats cannot become canonical JSON event evidence."""
    validator_calls = 0
    queued: list[dict[str, object]] = []

    def builder(**_: object) -> dict[str, object]:
        return {
            "event_contract_version": 1,
            "metadata": {"numeric_evidence": nonfinite_value},
        }

    def validator(_: object) -> tuple[str, ...]:
        nonlocal validator_calls
        validator_calls += 1
        return ()

    sink = CanonicalComputeUsageSink(
        event_builder=builder,
        event_validator=validator,
        enqueue=queued.append,
        identity={},
    )

    with pytest.raises(
        ValueError,
        match="event_builder result exact JSON floats must be finite",
    ):
        _emit_one_fit(sink)

    assert validator_calls == 0
    assert queued == []


def test_finite_exact_json_float_remains_admissible() -> None:
    """Finite exact floats remain generic JSON-number carriers."""
    queued: list[dict[str, object]] = []

    def builder(**_: object) -> dict[str, object]:
        return {
            "event_contract_version": 1,
            "metadata": {"numeric_evidence": 1.25},
        }

    def validator(_: object) -> tuple[str, ...]:
        return ()

    sink = CanonicalComputeUsageSink(
        event_builder=builder,
        event_validator=validator,
        enqueue=queued.append,
        identity={},
    )
    _emit_one_fit(sink)

    assert queued == [
        {
            "event_contract_version": 1,
            "metadata": {"numeric_evidence": 1.25},
        }
    ]
