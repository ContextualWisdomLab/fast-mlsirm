"""End-to-end fail-closed coverage for contextual random-effect values."""

from __future__ import annotations

import math

import pytest

from fast_mlsirm.multilevel import (
    build_context_membership,
    build_context_membership_design,
    weighted_contextual_effect,
)


def _single_context_design():
    edge = build_context_membership(
        observation_id="person_one",
        context_dimension_id="team_membership",
        context_id="team_alpha",
        membership_weight=1.0,
        membership_revision_fingerprint="a".rjust(64, "0"),
    )
    return build_context_membership_design([edge])


@pytest.mark.parametrize("effect", [math.nan, math.inf, -math.inf])
def test_public_predictor_rejects_non_finite_context_effects(effect: float) -> None:
    design = _single_context_design()

    with pytest.raises(ValueError, match="effects must be finite"):
        weighted_contextual_effect(
            design,
            {("team_membership", "team_alpha"): effect},
        )
