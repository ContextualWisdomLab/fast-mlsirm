"""Regression tests for Boolean sequence diagnostics at reliability boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import reliability


def _core_discovery_must_not_run() -> SimpleNamespace:
    """Fail if rejected Boolean ratings reach native capability discovery."""
    raise AssertionError("compiled-core discovery ran before Boolean evidence rejection")


@pytest.mark.parametrize(
    "invoke",
    [
        lambda ratings: reliability.icc(ratings),
        lambda ratings: reliability.mean_pairwise_cor(ratings),
        lambda ratings: reliability.mean_pairwise_rho(ratings),
    ],
)
def test_boolean_rating_sequences_preserve_historical_diagnostic(monkeypatch, invoke):
    """Exact Boolean sequences should use the established ratings diagnostic."""
    monkeypatch.setattr(fitstats, "_core_module", _core_discovery_must_not_run)

    with pytest.raises(ValueError, match="ratings must be numeric, not boolean"):
        invoke([[True, False], [False, True]])
