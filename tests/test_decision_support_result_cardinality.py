"""Fail-first resource-ordering evidence for decision-support Rust results."""

from __future__ import annotations

import pytest

import fast_mlsirm.decision_support as decision_support


def _over_cardinality_payload() -> dict[object, object]:
    """Return an exact native-result dict that cannot match the six-key contract."""
    return {
        "action_expected_net_values": [0.0],
        "selected_action": 0,
        "expected_net_intervention_value": 0.0,
        "expected_value_perfect_information": 0.0,
        "expected_value_sample_information": None,
        "net_expected_value_sample_information": None,
        "unexpected": 0.0,
    }


def test_over_cardinality_native_result_fails_before_key_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Impossible exact-dict cardinality must fail before copying result keys."""

    def forbidden_list(*args: object, **kwargs: object) -> object:
        raise AssertionError("native result keys were materialized before cardinality admission")

    monkeypatch.setattr(decision_support, "list", forbidden_list, raising=False)

    with pytest.raises(RuntimeError, match="invalid decision-support Rust result payload"):
        decision_support._validated_rust_result(
            _over_cardinality_payload(),
            action_count=1,
            has_sample_information=False,
        )
