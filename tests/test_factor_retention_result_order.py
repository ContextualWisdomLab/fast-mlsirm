"""Regression coverage for canonical factor-retention result ordering."""

from __future__ import annotations


def test_direct_result_constructor_canonicalizes_valid_unsorted_evidence() -> None:
    """Evidence is a governed set, so valid direct construction is order-insensitive."""
    from fast_mlsirm.factor_retention import (
        FactorRetentionDecision,
        FactorRetentionEvidence,
        FactorRetentionMethod,
        FactorRetentionResult,
    )

    evidence = (
        FactorRetentionEvidence(FactorRetentionMethod.VELICER_MAP, 3),
        FactorRetentionEvidence(FactorRetentionMethod.PARALLEL_ANALYSIS, 3),
    )

    result = FactorRetentionResult(
        decision=FactorRetentionDecision.CONSENSUS,
        retained_count=3,
        candidate_range=(3, 3),
        evidence=evidence,
    )

    assert tuple(record.method for record in result.evidence) == (
        FactorRetentionMethod.PARALLEL_ANALYSIS,
        FactorRetentionMethod.VELICER_MAP,
    )
