"""Fail-first privacy contract for exposure-control integer validation."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import exposure


class _LeakyIntegerControl:
    """Invalid control value whose representation must never reach package errors."""

    def __repr__(self) -> str:
        return "EXPOSURE_CONTROL_SECRET_SENTINEL"


def test_sympson_hetter_invalid_integer_control_does_not_reflect_value() -> None:
    """Public validation identifies the field without reflecting hostile content."""
    with pytest.raises(ValueError) as captured:
        exposure.sympson_hetter(
            np.ones(4, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            test_length=_LeakyIntegerControl(),
            n_simulees=4,
            max_iter=1,
            q_theta=5,
        )

    message = str(captured.value)
    assert "test_length must be an integer" in message
    assert "EXPOSURE_CONTROL_SECRET_SENTINEL" not in message
