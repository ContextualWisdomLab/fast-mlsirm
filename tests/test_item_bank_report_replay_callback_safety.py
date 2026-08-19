"""Callback-safety regressions for lifecycle-report identity replay."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from fast_mlsirm.rubric.item_bank_report import (
    ItemBankReportError,
    build_item_bank_report,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_item_bank_report.py"))
)
_lifecycle = _FIXTURES["_lifecycle"]


class _HostileApprovedUses(tuple):
    """Caller container whose iteration is executable code."""

    callback_count = 0

    def __iter__(self):
        type(self).callback_count += 1
        raise AssertionError("caller approved-use iterator executed")


def test_report_replay_rejects_mutated_containers_without_callbacks() -> None:
    """Fingerprint replay must fail before iterating a mutated record field."""
    records = _lifecycle()
    _HostileApprovedUses.callback_count = 0
    object.__setattr__(
        records[-1],
        "approved_use_ids",
        _HostileApprovedUses(("forged_use",)),
    )

    with pytest.raises(ItemBankReportError, match="creation-time identity"):
        build_item_bank_report(records)

    assert _HostileApprovedUses.callback_count == 0
