"""Fail-first trust-boundary tests for decision-support Rust results."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.decision_support as decision_support


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the smallest valid decision table."""
    return (
        np.array([1.0], dtype=np.float64),
        np.array([[0.0]], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
    )


def test_native_result_mapping_subclass_is_rejected_without_index_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A foreign mapping cannot execute result-side lookup callbacks."""
    callbacks = 0

    class HostileResult(dict[str, object]):
        def __getitem__(self, key: str) -> object:
            nonlocal callbacks
            callbacks += 1
            raise AssertionError(f"native result callback executed for {key}")

    class Core:
        def evaluate_decision_support(self, *args: object) -> object:
            return HostileResult()

    monkeypatch.setattr(decision_support, "_core_module", lambda: Core())
    probabilities, utilities, costs = _inputs()

    with pytest.raises(RuntimeError, match="invalid decision-support Rust result"):
        decision_support.evaluate_decision_support(probabilities, utilities, costs)
    assert callbacks == 0


def test_native_result_cardinality_and_selected_action_are_replayed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale extension cannot publish a contradictory action identity."""

    class Core:
        def evaluate_decision_support(self, *args: object) -> dict[str, object]:
            return {
                "action_expected_net_values": [0.0, 1.0],
                "selected_action": 1,
                "expected_net_intervention_value": 1.0,
                "expected_value_perfect_information": 0.0,
                "expected_value_sample_information": None,
                "net_expected_value_sample_information": None,
            }

    monkeypatch.setattr(decision_support, "_core_module", lambda: Core())
    probabilities, utilities, costs = _inputs()

    with pytest.raises(RuntimeError, match="invalid decision-support Rust result"):
        decision_support.evaluate_decision_support(probabilities, utilities, costs)
