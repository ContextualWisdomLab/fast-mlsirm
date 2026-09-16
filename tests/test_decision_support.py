"""Decision-support contract and Rust-arithmetic boundary tests."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.decision_support as decision_support


def _table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a two-state intervention table with an explicit no-action row."""
    return (
        np.array([0.75, 0.25], dtype=np.float64),
        np.array([[0.0, 0.0], [10.0, -10.0]], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
    )


def test_expected_net_value_evpi_and_evsi_are_rust_backed() -> None:
    """A coherent joint signal table produces the decision-theory identities."""
    probabilities, utilities, costs = _table()
    result = decision_support.evaluate_decision_support(
        probabilities,
        utilities,
        costs,
        signal_joint_probabilities=np.array(
            [[0.75, 0.0], [0.0, 0.25]],
            dtype=np.float64,
        ),
        information_cost=0.5,
    )

    assert np.allclose(result.action_expected_net_values, [0.0, 4.0])
    assert result.selected_action == 1
    assert result.expected_net_intervention_value == pytest.approx(4.0)
    assert result.expected_value_perfect_information == pytest.approx(2.75)
    assert result.expected_value_sample_information == pytest.approx(2.75)
    assert result.net_expected_value_sample_information == pytest.approx(2.25)


def test_no_sample_information_returns_none_values() -> None:
    """Omitting a joint signal distribution does not invent an EVSI."""
    probabilities, utilities, costs = _table()
    result = decision_support.evaluate_decision_support(probabilities, utilities, costs)

    assert result.expected_value_sample_information is None
    assert result.net_expected_value_sample_information is None


def test_public_adapter_only_marshals_to_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Python adapter preserves input rows and does not compute results."""
    calls: list[tuple[object, ...]] = []

    class Core:
        """Capture the Rust-shaped call and return a minimal result envelope."""

        def evaluate_decision_support(self, *args: object) -> dict[str, object]:
            """Record the marshaled arguments without performing arithmetic."""
            calls.append(args)
            return {
                "action_expected_net_values": [0.0, 4.0],
                "selected_action": 1,
                "expected_net_intervention_value": 4.0,
                "expected_value_perfect_information": 2.75,
                "expected_value_sample_information": None,
                "net_expected_value_sample_information": None,
            }

    monkeypatch.setattr(decision_support, "_core_module", lambda: Core())
    probabilities, utilities, costs = _table()
    result = decision_support.evaluate_decision_support(probabilities, utilities, costs)

    assert len(calls) == 1
    assert calls[0][3] == 0
    assert calls[0][4] is None
    assert result.selected_action == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        (
            {"state_probabilities": [0.5], "action_utilities": [[0.0]], "intervention_costs": [0.0]},
            "sum to one",
        ),
        (
            {"state_probabilities": [1.0], "action_utilities": [[0.0]], "intervention_costs": [0.0], "information_cost": -1.0},
            "information_cost",
        ),
        (
            {"state_probabilities": [1.0], "action_utilities": [[0.0]], "intervention_costs": [0.0], "no_action_index": True},
            "no_action_index",
        ),
        (
            {"state_probabilities": [1.0], "action_utilities": [[0.0, 1.0]], "intervention_costs": [0.0]},
            "columns",
        ),
        (
            {
                "state_probabilities": [1.0],
                "action_utilities": [[0.0]],
                "intervention_costs": [0.0],
                "signal_joint_probabilities": [[0.5]],
            },
            "sum to one",
        ),
    ),
)
def test_invalid_decision_inputs_fail_closed(
    kwargs: dict[str, object], message: str
) -> None:
    """Malformed distributions and controls are rejected before a result exists."""
    with pytest.raises(ValueError, match=message):
        decision_support.evaluate_decision_support(**kwargs)


def test_mismatched_signal_marginal_fails_closed() -> None:
    """A signal table with the wrong state marginal cannot masquerade as EVSI."""
    probabilities, utilities, costs = _table()
    with pytest.raises(ValueError, match="state marginal"):
        decision_support.evaluate_decision_support(
            probabilities,
            utilities,
            costs,
            signal_joint_probabilities=np.array([[1.0, 0.0]], dtype=np.float64),
        )


def test_boolean_array_is_not_a_decision_distribution() -> None:
    """Boolean arrays are not accepted as numeric probability evidence."""
    with pytest.raises(ValueError, match="booleans"):
        decision_support.evaluate_decision_support(
            np.array([True, False]),
            np.zeros((1, 2), dtype=np.float64),
            np.array([0.0], dtype=np.float64),
        )
