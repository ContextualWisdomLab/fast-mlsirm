"""Decision-support dimension preflight ordering regressions."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.decision_support as decision_support


def _guard_dense_marshalling(
    monkeypatch: pytest.MonkeyPatch,
    *,
    forbidden_name: str,
) -> None:
    """Fail if the structurally impossible target reaches value marshalling."""
    original = decision_support._trusted_real_array

    def guarded(value: object, name: str):  # type: ignore[no-untyped-def]
        if name == forbidden_name:
            raise AssertionError(f"dense marshalling reached for {name}")
        return original(value, name)

    monkeypatch.setattr(decision_support, "_trusted_real_array", guarded)


def test_state_limit_preflights_before_dense_marshalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An impossible state count fails from shape metadata before value work."""
    _guard_dense_marshalling(monkeypatch, forbidden_name="state_probabilities")
    probabilities = np.broadcast_to(
        np.array([1.0], dtype=np.float64),
        (decision_support.MAX_DECISION_STATES + 1,),
    )

    with pytest.raises(ValueError, match="exceeds 4096 states"):
        decision_support.evaluate_decision_support(
            probabilities,
            [[0.0]],
            [0.0],
        )


def test_action_limit_preflights_before_dense_marshalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An impossible action count fails from shape metadata before value work."""
    _guard_dense_marshalling(monkeypatch, forbidden_name="action_utilities")
    utilities = np.broadcast_to(
        np.array([[0.0]], dtype=np.float64),
        (decision_support.MAX_DECISION_ACTIONS + 1, 1),
    )

    with pytest.raises(ValueError, match="exceeds 1024 actions"):
        decision_support.evaluate_decision_support(
            [1.0],
            utilities,
            [0.0],
        )


def test_signal_limit_preflights_before_dense_marshalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An impossible signal count fails from shape metadata before value work."""
    _guard_dense_marshalling(
        monkeypatch,
        forbidden_name="signal_joint_probabilities",
    )
    signals = np.broadcast_to(
        np.array([[1.0]], dtype=np.float64),
        (decision_support.MAX_DECISION_SIGNALS + 1, 1),
    )

    with pytest.raises(ValueError, match="exceeds 1024 signals"):
        decision_support.evaluate_decision_support(
            [1.0],
            [[0.0]],
            [0.0],
            signal_joint_probabilities=signals,
        )


def test_generic_cell_limit_keeps_precedence_over_action_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generic evidence-cell envelope still wins when both limits fail."""
    _guard_dense_marshalling(monkeypatch, forbidden_name="action_utilities")
    utilities = np.broadcast_to(
        np.array([[0.0]], dtype=np.float64),
        (decision_support.MAX_DECISION_ACTIONS + 1, 1000),
    )

    with pytest.raises(ValueError, match="exceeds 1000000 cells"):
        decision_support.evaluate_decision_support(
            [1.0],
            utilities,
            [0.0],
        )


def test_intervention_cost_length_preflights_before_dense_marshalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cost/action mismatch fails from shape metadata before cost value work."""
    _guard_dense_marshalling(monkeypatch, forbidden_name="intervention_costs")

    with pytest.raises(
        ValueError,
        match="intervention_costs length must match action_utilities rows",
    ):
        decision_support.evaluate_decision_support(
            [1.0],
            [[0.0], [1.0]],
            [0.0, 1.0, 2.0],
        )


def test_action_state_mismatch_preflights_before_dense_marshalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A utility/state mismatch fails from shape metadata before utility values."""
    _guard_dense_marshalling(monkeypatch, forbidden_name="action_utilities")

    with pytest.raises(
        ValueError,
        match="action_utilities columns must match state_probabilities",
    ):
        decision_support.evaluate_decision_support(
            [0.5, 0.5],
            [[0.0, 1.0, 2.0]],
            [0.0],
        )


def test_signal_state_mismatch_preflights_before_dense_marshalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A signal/state mismatch fails from shape metadata before signal values."""
    _guard_dense_marshalling(
        monkeypatch,
        forbidden_name="signal_joint_probabilities",
    )

    with pytest.raises(
        ValueError,
        match="signal_joint_probabilities columns must match state_probabilities",
    ):
        decision_support.evaluate_decision_support(
            [1.0],
            [[0.0]],
            [0.0],
            signal_joint_probabilities=[[0.5, 0.5]],
        )


def test_builtin_state_limit_wins_before_element_traversal() -> None:
    """Exact built-in state cardinality is known before scalar inspection."""
    probabilities = [object()] * (decision_support.MAX_DECISION_STATES + 1)

    with pytest.raises(ValueError, match="exceeds 4096 states"):
        decision_support.evaluate_decision_support(
            probabilities,
            [[0.0]],
            [0.0],
        )


def test_builtin_action_limit_wins_before_row_element_traversal() -> None:
    """Exact built-in action-row cardinality is known before row contents."""
    utilities = [[object()]] * (decision_support.MAX_DECISION_ACTIONS + 1)

    with pytest.raises(ValueError, match="exceeds 1024 actions"):
        decision_support.evaluate_decision_support(
            [1.0],
            utilities,
            [0.0],
        )


def test_builtin_signal_limit_wins_before_row_element_traversal() -> None:
    """Exact built-in signal-row cardinality is known before row contents."""
    signals = [[object()]] * (decision_support.MAX_DECISION_SIGNALS + 1)

    with pytest.raises(ValueError, match="exceeds 1024 signals"):
        decision_support.evaluate_decision_support(
            [1.0],
            [[0.0]],
            [0.0],
            signal_joint_probabilities=signals,
        )


def test_builtin_cost_mismatch_wins_before_element_traversal() -> None:
    """Exact built-in cost cardinality is known before scalar inspection."""
    costs = [object(), 1.0, 2.0]

    with pytest.raises(
        ValueError,
        match="intervention_costs length must match action_utilities rows",
    ):
        decision_support.evaluate_decision_support(
            [1.0],
            [[0.0], [1.0]],
            costs,
        )
