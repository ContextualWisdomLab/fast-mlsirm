"""Regression contracts for heuristic-free Mokken AISP controls."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import fitstats, mokken


class _Core:
    """Minimal inert compiled-core double for admission-boundary tests."""

    @staticmethod
    def mokken_coef_h(x, n_persons, n_items):  # noqa: ANN001, ARG004
        return {
            "hij": [float("nan"), 0.5, 0.5, float("nan")],
            "hi": [0.5, 0.5],
            "h": 0.5,
            "zij": [float("nan"), 2.0, 2.0, float("nan")],
            "zi": [2.0, 2.0],
            "z": 2.0,
        }

    @staticmethod
    def mokken_aisp(x, n_persons, n_items, lower_bound, alpha):  # noqa: ANN001, ARG004
        return [1, 1]


def test_aisp_decision_controls_have_no_rule_of_thumb_defaults(monkeypatch) -> None:
    """Valid response evidence cannot silently receive c=.3 or alpha=.05."""
    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())
    responses = np.array([[0, 1], [1, 1], [1, 0]], dtype=np.int8)

    with pytest.raises(ValueError, match="lower_bound must be explicitly provided"):
        mokken.mokken_analysis(responses)
    with pytest.raises(ValueError, match="alpha must be explicitly provided"):
        mokken.mokken_analysis(responses, lower_bound=0.3)


def test_explicit_aisp_controls_remain_supported(monkeypatch) -> None:
    """Caller-governed controls remain a transparent input to the Rust AISP."""
    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())
    responses = np.array([[0, 1], [1, 1], [1, 0]], dtype=np.int8)

    result = mokken.mokken_analysis(responses, lower_bound=0.3, alpha=0.05)
    assert result.scale.tolist() == [1, 1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"lower_bound": 0.3},
        {"alpha": 0.05},
    ],
)
def test_omitted_aisp_controls_keep_response_admission_precedence(
    kwargs: dict[str, float],
) -> None:
    """Missing-control errors never mask malformed response evidence."""
    with pytest.raises(ValueError, match="responses must be a numeric array"):
        mokken.mokken_analysis(object(), **kwargs)
