"""Decision-support control-ordering regressions."""

from __future__ import annotations

import pytest

import fast_mlsirm.decision_support as decision_support


@pytest.mark.parametrize(
    ("control_kwargs", "message"),
    (
        ({"no_action_index": True}, "no_action_index must be a non-negative integer"),
        ({"information_cost": -1.0}, "information_cost must be non-negative"),
    ),
)
def test_invalid_independent_controls_fail_before_scientific_evidence_preflight(
    monkeypatch: pytest.MonkeyPatch,
    control_kwargs: dict[str, object],
    message: str,
) -> None:
    """Independent invalid controls must not trigger scientific-evidence traversal."""

    def forbidden_preflight(*args: object, **kwargs: object) -> object:
        raise AssertionError("scientific evidence preflight ran before control admission")

    monkeypatch.setattr(
        decision_support,
        "_preflight_real_evidence",
        forbidden_preflight,
    )

    with pytest.raises(ValueError, match=message):
        decision_support.evaluate_decision_support(
            [1.0],
            [[0.0]],
            [0.0],
            **control_kwargs,
        )
