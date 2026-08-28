"""Fail-first admission tests for the count-only metering producer boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fast_mlsirm.metering import CanonicalComputeUsageSink


def _sink_that_must_not_build() -> CanonicalComputeUsageSink:
    """Return a sink whose producer callback proves preflight ordering."""

    def forbidden_builder(**_: object) -> dict[str, object]:
        raise AssertionError("producer builder ran before input admission")

    return CanonicalComputeUsageSink(
        event_builder=forbidden_builder,
        event_validator=lambda _: (),
        enqueue=lambda _: None,
        identity={},
    )


def _valid_fit_payload() -> dict[str, object]:
    """Return one otherwise-valid fit emission payload."""
    return {
        "run_reference": "urn:cwl:run:fit-7",
        "artifact_reference": "urn:cwl:artifact:fit-7",
        "configuration_reference": "urn:cwl:config:fit-7",
        "seed_reference": "urn:cwl:seed:7",
        "occurred_at": "2026-08-28T00:00:00Z",
        "response_rows": 3,
        "response_items": 2,
    }


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("run_reference", object()),
        ("artifact_reference", object()),
        ("configuration_reference", object()),
        ("seed_reference", object()),
        ("occurred_at", object()),
        ("project_reference", object()),
    ],
)
def test_fit_rejects_non_string_metadata_before_producer(
    field: str, bad_value: object
) -> None:
    """Callback-bearing metadata objects never cross the producer boundary."""
    payload = _valid_fit_payload()
    payload[field] = bad_value

    with pytest.raises(ValueError, match=field):
        _sink_that_must_not_build().emit_fit(
            SimpleNamespace(model="MLS2PLM", backend="rust"),
            **payload,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("response_rows", -1),
        ("response_rows", True),
        ("response_items", -1),
        ("response_items", False),
        ("artifact_bytes", -1),
        ("artifact_bytes", True),
        ("artifact_bytes", 1.0),
    ],
)
def test_fit_rejects_invalid_counts_before_producer(
    field: str, bad_value: object
) -> None:
    """Usage quantities are exact non-negative integers before producer work."""
    payload = _valid_fit_payload()
    payload[field] = bad_value

    with pytest.raises(ValueError, match=field):
        _sink_that_must_not_build().emit_fit(
            SimpleNamespace(model="MLS2PLM", backend="rust"),
            **payload,  # type: ignore[arg-type]
        )
