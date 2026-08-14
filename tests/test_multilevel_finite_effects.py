"""End-to-end fail-closed coverage for contextual random-effect values."""

from __future__ import annotations

import math
import sys

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


def test_public_predictor_rejects_finite_inputs_that_overflow_weighted_sum() -> None:
    design = build_context_membership_design(
        [
            build_context_membership(
                observation_id="person_one",
                context_dimension_id="team_membership",
                context_id="team_alpha",
                membership_weight=1.0,
                membership_revision_fingerprint="b".rjust(64, "0"),
            ),
            build_context_membership(
                observation_id="person_one",
                context_dimension_id="site_membership",
                context_id="site_alpha",
                membership_weight=1.0,
                membership_revision_fingerprint="c".rjust(64, "0"),
            ),
        ]
    )

    with pytest.raises(ValueError, match="weighted contextual effects must be finite"):
        weighted_contextual_effect(
            design,
            {
                ("team_membership", "team_alpha"): sys.float_info.max,
                ("site_membership", "site_alpha"): sys.float_info.max,
            },
        )
