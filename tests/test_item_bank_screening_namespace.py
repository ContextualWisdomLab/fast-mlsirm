"""Public scoring-namespace exposure for governed item-bank screening contracts."""

from __future__ import annotations

import fast_mlsirm.scoring as scoring
from fast_mlsirm.scoring.item_screening import (
    CandidateScreeningResult,
    ItemScreeningFinding,
    ScreeningDimension,
    ScreeningStatus,
    build_candidate_screening_result,
    build_item_screening_finding,
)


def test_screening_contracts_are_explicit_scoring_attributes_without_star_widening() -> None:
    """Expose the screening API explicitly while preserving the pinned ``__all__`` surface."""
    expected = {
        "CandidateScreeningResult": CandidateScreeningResult,
        "ItemScreeningFinding": ItemScreeningFinding,
        "ScreeningDimension": ScreeningDimension,
        "ScreeningStatus": ScreeningStatus,
        "build_candidate_screening_result": build_candidate_screening_result,
        "build_item_screening_finding": build_item_screening_finding,
    }
    for name, value in expected.items():
        assert getattr(scoring, name) is value
        assert name not in scoring.__all__
