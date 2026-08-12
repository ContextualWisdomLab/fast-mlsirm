"""Regression for exact built-in judge criterion descriptions."""

import pytest

from fast_mlsirm.llm_judge import JudgeCriterion


def test_criterion_description_rejects_runtime_string_subclass_before_hooks() -> None:
    """Descriptions reject string subclasses before invoking caller hooks."""

    class _HookedString(str):
        invoked = False

        def strip(self, *args, **kwargs):
            type(self).invoked = True
            return super().strip(*args, **kwargs)

    with pytest.raises(ValueError, match="criterion description must be a string"):
        JudgeCriterion("task_alignment", _HookedString("observable evidence"))
    assert _HookedString.invoked is False
