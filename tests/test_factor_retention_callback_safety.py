"""Callback-safety regressions for governed factor-retention evidence."""

from __future__ import annotations

import pytest

from fast_mlsirm.factor_retention import (
    FactorRetentionEvidence,
    FactorRetentionMethod,
    govern_factor_retention,
)


def test_candidate_count_subclass_is_rejected_without_comparison_callback() -> None:
    """An integer subclass cannot execute callbacks during count admission."""
    callbacks: list[str] = []

    class HostileInt(int):
        def __le__(self, other: object) -> bool:
            callbacks.append("le")
            raise AssertionError("candidate_count comparison callback executed")

        def __gt__(self, other: object) -> bool:
            callbacks.append("gt")
            raise AssertionError("candidate_count comparison callback executed")

    with pytest.raises(ValueError, match="positive integer"):
        FactorRetentionEvidence(FactorRetentionMethod.PARALLEL_ANALYSIS, HostileInt(2))

    assert callbacks == []


def test_evidence_subclass_is_rejected_without_record_attribute_callback() -> None:
    """A record subclass cannot execute callbacks before evidence admission."""
    callbacks: list[str] = []

    class HostileEvidence(FactorRetentionEvidence):
        def __getattribute__(self, name: str):
            if name in {"method", "candidate_count"}:
                callbacks.append(name)
                raise AssertionError("factor-retention record callback executed")
            return super().__getattribute__(name)

    record = object.__new__(HostileEvidence)
    object.__setattr__(record, "method", FactorRetentionMethod.PARALLEL_ANALYSIS)
    object.__setattr__(record, "candidate_count", 2)

    with pytest.raises(TypeError, match="FactorRetentionEvidence"):
        govern_factor_retention((record,))

    assert callbacks == []
