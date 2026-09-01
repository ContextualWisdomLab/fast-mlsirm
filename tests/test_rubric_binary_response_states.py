"""Binary pilot response-state contracts preserve measurement evidence semantics."""

from __future__ import annotations

from pathlib import Path
import runpy

import numpy as np
import pytest

from fast_mlsirm.rubric import PilotObservationError, PilotResponseState, build_mirt_pilot_design

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_pilot_observations.py"))
)
_observation = _FIXTURES["_observation"]
_pilot = _FIXTURES["_pilot"]


def test_binary_nonresponse_states_remain_distinct_from_zero_and_one() -> None:
    """Nonresponses retain their exact state and never acquire a numeric category."""
    states = (
        PilotResponseState.MISSING,
        PilotResponseState.NOT_OBSERVED,
        PilotResponseState.ABSTAINED,
        PilotResponseState.INVALID,
        PilotResponseState.OMITTED,
        PilotResponseState.NOT_APPLICABLE,
        PilotResponseState.INSUFFICIENT_EVIDENCE,
    )

    assert len({state.value for state in states}) == len(states)
    for state in states:
        record = _observation(response_state=state, category=None)
        assert record.response_state is state
        assert record.category is None
        assert record.to_dict()["response_state"] == state.value


def test_adjudicated_binary_response_keeps_value_and_state_as_separate_evidence() -> None:
    """An adjudicated 0/1 remains scoreable while its provenance state stays explicit."""
    record = _observation(response_state=PilotResponseState.ADJUDICATED, category=1)

    assert record.category == 1
    assert record.response_state is PilotResponseState.ADJUDICATED
    assert record.to_dict()["response_state"] == "adjudicated"

    with pytest.raises(ValueError, match="category must be an integer"):
        _observation(response_state=PilotResponseState.ADJUDICATED, category=None)
    with pytest.raises(ValueError, match="category must be None"):
        _observation(response_state=PilotResponseState.ABSTAINED, category=0)


def test_mirt_handoff_preserves_adjudicated_and_abstained_cells_without_dichotomizing() -> None:
    """The binary handoff scores only 0/1 values and retains nonresponse state evidence."""
    item_alpha = _pilot("generated_item_alpha")
    item_beta = _pilot("generated_item_beta", query_testlet_id="query_testlet_beta")
    records = (
        _observation(
            item_alpha,
            respondent_id="respondent_alpha",
            response_state=PilotResponseState.ADJUDICATED,
            category=1,
        ),
        _observation(
            item_beta,
            respondent_id="respondent_alpha",
            category=0,
        ),
        _observation(
            item_alpha,
            respondent_id="respondent_beta",
            category=0,
        ),
        _observation(
            item_beta,
            respondent_id="respondent_beta",
            response_state=PilotResponseState.ABSTAINED,
            category=None,
        ),
    )

    design = build_mirt_pilot_design(records)

    assert design.responses == ((1, 0), (0, None))
    assert design.response_states == (
        (PilotResponseState.ADJUDICATED, PilotResponseState.OBSERVED),
        (PilotResponseState.OBSERVED, PilotResponseState.ABSTAINED),
    )
    numeric = design.responses_array()
    assert numeric[0, 0] == 1.0
    assert numeric[0, 1] == 0.0
    assert np.isnan(numeric[1, 1])


def test_mirt_rejects_polytomous_adjudicated_values_instead_of_thresholding() -> None:
    """Adjudication cannot make an ordinal category admissible to the binary MIRT path."""
    item_alpha = _pilot("generated_item_alpha")
    item_beta = _pilot("generated_item_beta", query_testlet_id="query_testlet_beta")
    records = (
        _observation(
            item_alpha,
            response_state=PilotResponseState.ADJUDICATED,
            category=2,
        ),
        _observation(item_beta, category=0),
    )

    with pytest.raises(PilotObservationError) as caught:
        build_mirt_pilot_design(records)

    assert caught.value.code == "non_binary_observed_category"
