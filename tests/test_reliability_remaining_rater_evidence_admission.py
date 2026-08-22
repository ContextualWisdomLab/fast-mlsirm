"""Regression tests for remaining reliability rater-evidence trust boundaries."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import reliability


class _HostileArrayProvider:
    """Array provider that must not execute during package admission."""

    def __init__(self) -> None:
        self.calls = 0

    def __array__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("caller-controlled __array__ callback executed")


class _HostileText(str):
    """Text subclass whose comparison/string callbacks must never run."""

    def __new__(cls, value: str):
        instance = super().__new__(cls, value)
        instance.calls = 0
        return instance

    def __eq__(self, other):
        self.calls += 1
        raise AssertionError("caller-controlled text equality callback executed")

    def __hash__(self):
        self.calls += 1
        raise AssertionError("caller-controlled text hash callback executed")

    def __str__(self):
        self.calls += 1
        raise AssertionError("caller-controlled text conversion callback executed")


def _native_discovery_must_not_run():
    raise AssertionError("compiled-core discovery ran before evidence admission")


@pytest.mark.parametrize(
    "invoke",
    [
        lambda value: reliability.kripp_alpha(value),
        lambda value: reliability.finn_coefficient(value, 5),
        lambda value: reliability.maxwell_re(value),
        lambda value: reliability.robinson_a(value),
    ],
)
def test_remaining_rater_apis_reject_array_provider_before_native(
    monkeypatch, invoke
):
    """Caller array protocols must not synthesize scientific ratings evidence."""
    hostile = _HostileArrayProvider()
    monkeypatch.setattr(fitstats, "_core_module", _native_discovery_must_not_run)

    with pytest.raises(ValueError, match="ratings must be real numeric evidence"):
        invoke(hostile)

    assert hostile.calls == 0


@pytest.mark.parametrize(
    ("invoke", "message"),
    [
        (
            lambda ratings, control: reliability.kripp_alpha(
                ratings, method=control
            ),
            "method must be one of nominal, ordinal, interval, ratio",
        ),
        (
            lambda ratings, control: reliability.finn_coefficient(
                ratings, 5, model=control
            ),
            "model must be one of oneway, twoway",
        ),
    ],
)
def test_remaining_rater_text_controls_fail_before_data_or_native(
    monkeypatch, invoke, message
):
    """Semantic text controls must be exact built-ins before caller data work."""
    ratings = _HostileArrayProvider()
    control = _HostileText("nominal")
    monkeypatch.setattr(fitstats, "_core_module", _native_discovery_must_not_run)

    with pytest.raises(ValueError, match=message):
        invoke(ratings, control)

    assert ratings.calls == 0
    assert control.calls == 0
