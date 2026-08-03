"""Direct invariant coverage for the factory-sealed DIF pilot artifact."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

import fast_mlsirm.rubric.dif_pilot as dif_pilot
from fast_mlsirm.rubric import DifPilotDesign, build_dif_pilot_design

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_pilot_observations.py"))
)
_pilot = _FIXTURES["_pilot"]
_observation = _FIXTURES["_observation"]


def _valid_design():
    """Return the smallest complete two-group binary DIF pilot design."""
    item = _pilot("generated_item_alpha")
    records = (
        _observation(item, respondent_id="respondent_alpha", category=1),
        _observation(item, respondent_id="respondent_beta", category=0),
    )
    return build_dif_pilot_design(
        records,
        respondent_groups={
            "respondent_alpha": "reference_group_alpha",
            "respondent_beta": "focal_group_alpha",
        },
        reference_group_id="reference_group_alpha",
        focal_group_id="focal_group_alpha",
    )


def _direct(binary_design, groups, *, reference="reference_group_alpha", focal="focal_group_alpha"):
    """Call the sealed constructor with the private test-only token."""
    return DifPilotDesign(
        binary_design=binary_design,
        reference_group_id=reference,
        focal_group_id=focal,
        respondent_group_ids=groups,
        _design_token=dif_pilot._DIF_DESIGN_TOKEN,
    )


def test_direct_constructor_rechecks_group_structure_after_factory_validation():
    """Mutated or forged group layouts cannot bypass dataclass invariants."""
    design = _valid_design()
    binary = design.binary_design

    with pytest.raises(ValueError, match="must differ"):
        _direct(
            binary,
            ("shared_group_alpha", "shared_group_alpha"),
            reference="shared_group_alpha",
            focal="shared_group_alpha",
        )
    with pytest.raises(ValueError, match="only the declared"):
        _direct(
            binary,
            ("reference_group_alpha", "comparison_group_alpha"),
        )
    with pytest.raises(ValueError, match="reference group"):
        _direct(
            binary,
            ("focal_group_alpha", "focal_group_alpha"),
        )
    with pytest.raises(ValueError, match="focal group"):
        _direct(
            binary,
            ("reference_group_alpha", "reference_group_alpha"),
        )
    with pytest.raises(ValueError, match=r"respondent_group_ids\[1\]"):
        _direct(
            binary,
            ("reference_group_alpha", "focal"),
        )


def test_direct_constructor_rechecks_schema_binding():
    """A mutated nested schema cannot be wrapped under a stale outer schema."""
    design = _valid_design()
    object.__setattr__(design.binary_design, "schema_version", "different_schema")
    with pytest.raises(ValueError, match="schema_version"):
        _direct(design.binary_design, design.respondent_group_ids)
