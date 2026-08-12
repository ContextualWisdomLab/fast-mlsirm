"""Regression tests for fail-closed rubric iterable boundaries."""

from __future__ import annotations

import pytest

from fast_mlsirm.rubric import ResponseFormat, RubricLevel, RubricSpecification


class _ExplodingIterable:
    """Yield a valid prefix before raising a caller-controlled runtime error."""

    def __init__(self, values: tuple[object, ...]) -> None:
        """Store the bounded prefix used before the hostile iterator failure."""
        self._values = values

    def __iter__(self):
        """Yield valid values and then raise an error containing source-like text."""
        yield from self._values
        raise RuntimeError("sensitive_source_text_should_not_escape")


def _level(score: int) -> RubricLevel:
    """Return one minimal valid rubric level."""
    return RubricLevel(
        score=score,
        label=f"Level {score}",
        descriptor=f"Observed performance at score {score}.",
        observable_indicators=(f"indicator {score}",),
    )


def test_rubric_levels_iterator_failure_is_redacted_value_error() -> None:
    """Caller iterator exceptions must not cross the public rubric boundary raw."""
    hostile_levels = _ExplodingIterable((_level(0), _level(1)))

    with pytest.raises(ValueError) as caught:
        RubricSpecification(
            rubric_id="writing_rubric",
            construct_id="argument_quality",
            construct_definition="Quality of evidence-grounded argumentation.",
            response_format=ResponseFormat.CONSTRUCTED_RESPONSE,
            levels=hostile_levels,  # type: ignore[arg-type]
            task_families=("essay_task",),
            evidence_requirements=("Cites relevant evidence.",),
        )

    assert "levels" in str(caught.value)
    assert "sensitive_source_text_should_not_escape" not in str(caught.value)
