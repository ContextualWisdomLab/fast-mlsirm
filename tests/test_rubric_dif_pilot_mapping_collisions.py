"""Adversarial normalization coverage for DIF respondent-group mappings."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from fast_mlsirm.rubric import PilotObservationError, build_dif_pilot_design

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_pilot_observations.py"))
)
_pilot = _FIXTURES["_pilot"]
_observation = _FIXTURES["_observation"]


def test_builder_rejects_distinct_keys_that_normalize_to_one_respondent():
    """Whitespace normalization cannot silently replace an earlier assignment."""
    item = _pilot("generated_item_alpha")
    records = (
        _observation(item, respondent_id="respondent_alpha", category=1),
        _observation(item, respondent_id="respondent_beta", category=0),
    )
    respondent_groups = {
        "respondent_alpha": "reference_group_alpha",
        " respondent_alpha ": "focal_group_alpha",
        "respondent_beta": "focal_group_alpha",
    }

    with pytest.raises(PilotObservationError) as rejection:
        build_dif_pilot_design(
            records,
            respondent_groups=respondent_groups,
            reference_group_id="reference_group_alpha",
            focal_group_id="focal_group_alpha",
        )

    assert rejection.value.code == "dif_duplicate_group_assignment"
    assert rejection.value.path == "$.respondent_groups"
    assert "unique after normalization" in rejection.value.message
