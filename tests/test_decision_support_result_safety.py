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


def _payload() -> dict[object, object]:
    """Return one concrete PyO3-shaped no-information result."""
    return {
        "action_expected_net_values": [0.0],
        "selected_action": 0,
        "expected_net_intervention_value": 0.0,
        "expected_value_perfect_information": 0.0,
        "expected_value_sample_information": None,
        "net_expected_value_sample_information": None,
    }


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


def _invalid_payloads() -> list[tuple[object, int, bool]]:
    """Build structurally invalid native payloads for branch-complete replay."""
    payloads: list[tuple[object, int, bool]] = [([], 1, False)]

    extra = _payload()
    extra["unexpected"] = 1.0
    payloads.append((extra, 1, False))

    non_string_key = _payload()
    non_string_key[1] = 1.0
    payloads.append((non_string_key, 1, False))

    for replacement in ((0.0,), [0.0, 1.0], [0], [float("nan")]):
        payload = _payload()
        payload["action_expected_net_values"] = replacement
        payloads.append((payload, 1, False))

    for replacement in (True, 1):
        payload = _payload()
        payload["selected_action"] = replacement
        payloads.append((payload, 1, False))

    for replacement in (0, float("nan")):
        payload = _payload()
        payload["expected_net_intervention_value"] = replacement
        payloads.append((payload, 1, False))

    for replacement in (0, float("inf")):
        payload = _payload()
        payload["expected_value_perfect_information"] = replacement
        payloads.append((payload, 1, False))

    net_mismatch = _payload()
    net_mismatch["expected_net_intervention_value"] = 1.0
    payloads.append((net_mismatch, 1, False))

    wrong_argmax = _payload()
    wrong_argmax["action_expected_net_values"] = [0.0, 1.0]
    wrong_argmax["selected_action"] = 0
    payloads.append((wrong_argmax, 2, False))

    missing_evsi = _payload()
    payloads.append((missing_evsi, 1, True))

    missing_net_evsi = _payload()
    missing_net_evsi["expected_value_sample_information"] = 0.0
    payloads.append((missing_net_evsi, 1, True))

    unexpected_evsi = _payload()
    unexpected_evsi["expected_value_sample_information"] = 0.0
    payloads.append((unexpected_evsi, 1, False))

    unexpected_net_evsi = _payload()
    unexpected_net_evsi["net_expected_value_sample_information"] = 0.0
    payloads.append((unexpected_net_evsi, 1, False))
    return payloads


@pytest.mark.parametrize(("payload", "action_count", "has_sample"), _invalid_payloads())
def test_native_result_contract_rejects_malformed_payloads(
    payload: object,
    action_count: int,
    has_sample: bool,
) -> None:
    """Every concrete result invariant fails with one package-owned error."""
    with pytest.raises(RuntimeError, match="invalid decision-support Rust result"):
        decision_support._validated_rust_result(
            payload,
            action_count=action_count,
            has_sample_information=has_sample,
        )
