"""Fail-before-materialization regressions for Rasch CML semantic controls."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.rasch_cml import andersen_lr_test, fit_rasch_cml


class _HostileResponses:
    """Array provider that must remain untouched for rejected semantic controls."""

    calls = 0

    def __array__(self, dtype=None, copy=None):
        type(self).calls += 1
        raise AssertionError("caller-owned response materialization executed")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iter": 0}, "max_iter"),
        ({"tol": 0.0}, "tol"),
    ],
)
def test_fit_rasch_cml_rejects_controls_before_response_materialization(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Invalid controls fail before caller response conversion can execute."""

    _HostileResponses.calls = 0

    with pytest.raises(ValueError, match=message):
        fit_rasch_cml(_HostileResponses(), **kwargs)

    assert _HostileResponses.calls == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iter": 0}, "max_iter"),
        ({"tol": 0.0}, "tol"),
    ],
)
def test_andersen_rejects_controls_before_response_materialization(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Andersen controls fail before caller response conversion can execute."""

    _HostileResponses.calls = 0

    with pytest.raises(ValueError, match=message):
        andersen_lr_test(_HostileResponses(), np.array([0, 1]), **kwargs)

    assert _HostileResponses.calls == 0
