from __future__ import annotations

import pytest

from fast_mlsirm import metering
from fast_mlsirm.metering import CanonicalComputeUsageSink


def _construct_sink(identity: dict[str, str | None]) -> CanonicalComputeUsageSink:
    return CanonicalComputeUsageSink(
        event_builder=lambda **payload: payload,
        event_validator=lambda event: (),
        enqueue=lambda event: None,
        identity=identity,
    )


def test_noncanonical_identity_key_diagnostic_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        metering,
        "_MAX_FORMATTED_IDENTITY_FIELD_CHARS",
        4,
        raising=False,
    )

    with pytest.raises(ValueError) as exc_info:
        _construct_sink({"abcdefgh": "opaque-reference"})

    assert str(exc_info.value) == "identity contains noncanonical fields: abcd"


def test_short_noncanonical_identity_key_diagnostic_is_preserved() -> None:
    with pytest.raises(ValueError) as exc_info:
        _construct_sink({"bad": "opaque-reference"})

    assert str(exc_info.value) == "identity contains noncanonical fields: bad"
