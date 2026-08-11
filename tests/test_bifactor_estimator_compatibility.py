"""Fail-first contracts for model × estimator compatibility preflight."""

from __future__ import annotations

import pytest

from fast_mlsirm.config import FitConfig


def test_bifactor_jmle_is_rejected_during_configuration_validation() -> None:
    """BIFAC2PLM must not advertise an unimplemented JMLE fitting path."""
    with pytest.raises(ValueError, match=r"BIFAC2PLM.*mmle"):
        FitConfig(model="BIFAC2PLM", estimator="jmle").validate()


def test_bifactor_mmle_remains_a_valid_configuration() -> None:
    """The implemented bifactor MMLE path remains accepted by preflight."""
    FitConfig(model="BIFAC2PLM", estimator="mmle").validate()
