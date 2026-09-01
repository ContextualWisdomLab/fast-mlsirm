"""Focused topology regressions for generalized mixed-model membership."""

from __future__ import annotations

import pytest

from fast_mlsirm.model_specification import (
    MembershipClassification,
    MembershipMultiplicity,
    MembershipStructure,
    MembershipWeightAuthority,
)


def test_cross_classified_membership_requires_two_distinct_axes() -> None:
    """A repeated label cannot impersonate two independent classification axes."""
    with pytest.raises(
        ValueError,
        match="cross-classified membership requires at least two distinct axes",
    ):
        MembershipStructure(
            classification=MembershipClassification.CROSS_CLASSIFIED,
            multiplicity=MembershipMultiplicity.SINGLE,
            weight_authority=MembershipWeightAuthority.NOT_APPLICABLE,
            classification_axes=("school", "school"),
        )


def test_cross_classified_membership_preserves_distinct_axes() -> None:
    """Two distinct admitted axes remain a valid cross-classified structure."""
    membership = MembershipStructure(
        classification=MembershipClassification.CROSS_CLASSIFIED,
        multiplicity=MembershipMultiplicity.SINGLE,
        weight_authority=MembershipWeightAuthority.NOT_APPLICABLE,
        classification_axes=("school", "neighbourhood"),
    )

    assert membership.classification_axes == ("school", "neighbourhood")
