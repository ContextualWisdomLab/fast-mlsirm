"""Regression tests for generalized-mixed membership-axis identity."""

import pytest

from fast_mlsirm.model_specification import (
    MembershipClassification,
    MembershipMultiplicity,
    MembershipStructure,
    MembershipWeightAuthority,
)


def test_cross_classified_membership_requires_distinct_axes() -> None:
    """A repeated label cannot masquerade as two classification axes."""
    with pytest.raises(
        ValueError,
        match="cross-classified membership requires at least two distinct axes",
    ):
        MembershipStructure(
            classification=MembershipClassification.CROSS_CLASSIFIED,
            multiplicity=MembershipMultiplicity.SINGLE,
            weight_authority=MembershipWeightAuthority.NOT_APPLICABLE,
            classification_axes=("organization", "organization"),
        )
