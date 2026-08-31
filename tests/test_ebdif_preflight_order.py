"""Admission-order regressions for Empirical Bayes DIF evidence."""

from __future__ import annotations

import pytest

import fast_mlsirm.ebdif as ebdif
import fast_mlsirm.fitstats as fitstats


def _unexpected_core() -> object:
    """Fail if compiled-core discovery happens before bounded admission."""

    raise AssertionError("compiled core discovered before EBDIF preflight completed")


def test_second_vector_resource_overflow_wins_before_first_builtin_scalar_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inert resource impossibility must win before built-in leaf traversal."""

    monkeypatch.setattr(ebdif, "_MAX_EBDIF_ITEMS", 2, raising=False)
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    with pytest.raises(ValueError, match="se exceeds the 2-item resource limit"):
        ebdif.eb_mh_dif([object(), 0.0], [0.3, 0.4, 0.5])


def test_aggregate_result_budget_wins_before_input_leaf_traversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deterministic output footprint must be bounded from carrier metadata."""

    monkeypatch.setattr(ebdif, "_MAX_EBDIF_RESULT_VALUES", 16, raising=False)
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    with pytest.raises(ValueError, match="EBDIF result exceeds the 16-value resource limit"):
        ebdif.eb_mh_dif([object(), 0.0, 0.0], [0.3, 0.4, 0.5])
