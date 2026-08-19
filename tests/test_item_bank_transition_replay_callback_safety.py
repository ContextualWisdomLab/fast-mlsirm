"""Callback-safety regressions for governed item-bank transition replay."""

from __future__ import annotations

from pathlib import Path
import runpy
from typing import Callable

import pytest

from fast_mlsirm.rubric.item_bank import (
    ItemBankEvidenceKind,
    ItemBankLifecycleError,
    ItemBankLifecycleState,
    transition_item_bank_record,
)

_LIFECYCLE_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_item_bank_lifecycle.py"))
)


def _approval_evidence() -> tuple[object, ...]:
    """Return one ordinary approval reference from the lifecycle fixtures."""
    return (
        _LIFECYCLE_FIXTURES["_evidence"](
            ItemBankEvidenceKind.APPROVAL,
            "approval",
            fingerprint_character="e",
        ),
    )


def _approve(record: object) -> object:
    """Attempt the next governed transition from one calibrated record."""
    return transition_item_bank_record(
        record,
        ItemBankLifecycleState.APPROVED,
        evidence_references=_approval_evidence(),
        transition_reason_id="governance_approval",
        approved_use_ids=("production_scoring",),
    )


def _hostile_callback(calls: list[str], label: str) -> Callable[[], object]:
    """Return a callback that records any caller-controlled replay dispatch."""

    def callback() -> object:
        calls.append(label)
        raise AssertionError("caller callback executed during lifecycle replay")

    return callback


def test_transition_replay_rejects_shadowed_content_dict_without_callback() -> None:
    """A shadowed record serializer cannot execute before transition authority."""
    record = _LIFECYCLE_FIXTURES["_calibrated_record"]()
    calls: list[str] = []
    object.__setattr__(record, "_content_dict", _hostile_callback(calls, "record"))

    with pytest.raises(ItemBankLifecycleError) as caught:
        _approve(record)

    assert caught.value.code == "lifecycle_record_replay_mismatch"
    assert caught.value.path == "$.current_record"
    assert calls == []


def test_transition_replay_rejects_shadowed_evidence_to_dict_without_callback() -> None:
    """A shadowed evidence serializer cannot execute during record replay."""
    record = _LIFECYCLE_FIXTURES["_calibrated_record"]()
    calls: list[str] = []
    reference = record.evidence_references[0]
    object.__setattr__(reference, "to_dict", _hostile_callback(calls, "evidence"))

    with pytest.raises(ItemBankLifecycleError) as caught:
        _approve(record)

    assert caught.value.code == "lifecycle_record_replay_mismatch"
    assert caught.value.path == "$.current_record"
    assert calls == []
